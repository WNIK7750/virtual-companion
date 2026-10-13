// Main-process caller supplies the existing authenticated loopback transport.
import { approvedScopeMatches } from './approved-display.mjs';

export class StreamDeliveryClient {
  constructor({ call, presenter, onReceived }) {
    this.call = call; this.presenter = presenter; this.pendingAcks = new Map();
    this.onReceived = onReceived;
    this.receivedSent = new Set(); this.visibleReceipts = [];
    this.receiptEvidence = new Map();
    this.receivedTimelines = new Map();
  }
  async stop() { return this.call('streamStop', {}); }
  async acknowledge(event, receipt) {
    const pending = this.pendingAcks.get(event.fingerprint);
    if (pending && (pending.sequence !== event.sequence || pending.host_receipt !== receipt)) {
      throw new Error('Conflicting pending display acknowledgement');
    }
    const acknowledgement = pending || { sequence: event.sequence, fingerprint: event.fingerprint,
      host_receipt: receipt, basis: structuredClone(event.basis) };
    // A server commit can succeed even when its HTTP response is lost. Preserve
    // this exact device receipt until an idempotent ACK request returns; the
    // unknown-only server reconciliation list cannot recover that response.
    this.pendingAcks.set(event.fingerprint, acknowledgement);
    return this.sendAcknowledgement(acknowledgement);
  }
  async sendAcknowledgement(acknowledgement) {
    const { sequence, fingerprint, host_receipt } = acknowledgement;
    await this.call('streamAck', { sequence, fingerprint, host_receipt });
    this.receiptEvidence.set(fingerprint, structuredClone(acknowledgement));
    this.pendingAcks.delete(fingerprint);
    return structuredClone(acknowledgement);
  }
  takeReceipts() {
    const receipts = [...this.receiptEvidence.values()]; this.receiptEvidence.clear(); return receipts;
  }
  async retryAcknowledgements() {
    const receipts = [];
    for (const acknowledgement of this.pendingAcks.values()) {
      try { receipts.push(await this.sendAcknowledgement(acknowledgement)); }
      catch (error) {
        if (!receipts.length) throw error;
        return { receipts, error: 'reconciliation_incomplete' };
      }
    }
    return { receipts };
  }
  async pollOnce(options) {
    const waitMs = options?.waitMs || 0;
    const { event } = await this.call('streamPoll', waitMs ? { wait_ms: waitMs } : {});
    if (!event) return { status: 'idle' };
    if (options) {
      const { scope } = options;
      if (options.getState) {
        const current = await options.getState();
        if (!approvedScopeMatches(scope, current, current.session_id)) return { status: 'stale' };
      }
      if (!['session_id', 'input_id', 'attempt_id'].every(key => event.basis?.[key] === scope[key])) {
        return { status: 'stale' };
      }
    }
    const receipt = await this.presenter.present(event, options);
    if (!receipt) return { status: 'unknown' };
    const acknowledged = await this.acknowledge(event, receipt);
    return { status: 'confirmed', receipts: [acknowledged] };
  }
  pulse(getState, { waitMs = 0 } = {}) {
    if (this.recovering) return Promise.resolve({ status: 'recovering' });
    if (!this.inFlight) this.inFlight = this.deliverCurrent(getState, waitMs).finally(() => { this.inFlight = null; });
    return this.inFlight;
  }
  async resume(fingerprint, getState) {
    if (this.recovering) throw new Error('Display recovery already in progress');
    this.recovering = true;
    const cancellation = new AbortController(); this.recoveryCancellation = cancellation;
    try {
      // Let an existing normal delivery settle before checking the idle state.
      await this.inFlight;
      const state = await getState();
      if (state.busy || state.delivery_protocol !== 'approved-prefix.v1') throw new Error('Turn still running');
      const { reconcile } = await this.call('streamState', {});
      const event = reconcile.find(value => value.fingerprint === fingerprint);
      if (!event) throw new Error('No matching unknown display');
      if (cancellation.signal.aborted) return { status: 'unknown' };
      const receipt = await this.presenter.resume(event);
      if (!receipt) return { status: 'unknown' };
      return { status: 'confirmed', receipts: [await this.acknowledge(event, receipt)] };
    } finally { this.recovering = false; this.recoveryCancellation = null; }
  }
  cancelRecovery() {
    this.recoveryCancellation?.abort(); this.presenter.cancelRecovery();
  }
  async deliverCurrent(getState, waitMs = 0) {
    const retried = await this.retryAcknowledgements();
    if (retried.receipts.length) return {
      status: retried.error ? 'partial' : 'reconciled', receipts: retried.receipts,
      ...(retried.error ? { error: retried.error } : {}),
    };
    let state = await getState();
    if (state.delivery_protocol !== 'approved-prefix.v1') return { status: 'idle' };
    const recovered = await this.reconcile();
    // Deliver successful ACK evidence before another poll/state failure can
    // discard it. Remaining offers are processed by the next ordinary pulse.
    if (recovered.receipts.length) return {
      status: recovered.error ? 'partial' : 'reconciled', receipts: recovered.receipts,
      ...(recovered.error ? { error: recovered.error } : {}),
    };
    state = await getState();
    if (state.reliable_text_receive === true) return {
      ...await this.receiveCurrent(state, waitMs), recoverable: recovered.recoverable,
    };
    const scope = state.delivery_scope;
    if (!approvedScopeMatches(scope, state, state.session_id)) {
      return { status: 'stale', recoverable: recovered.recoverable };
    }
    const result = await this.pollOnce({ scope, ...(waitMs ? { waitMs, getState } : {}) });
    const receipts = [...recovered.receipts, ...(result.receipts || [])];
    return { ...result, recoverable: recovered.recoverable, ...(receipts.length ? { receipts } : {}) };
  }
  async reconcile() {
    const { reconcile, received = [] } = await this.call('streamState', {});
    const receipts = [], recoverable = [];
    for (const event of [...reconcile, ...received]) {
      const known = this.presenter.reconcile(event);
      // Reconciliation never renders, including when this device has no intent.
      if (known.status === 'confirmed') {
        try { receipts.push(await this.acknowledge(event, known.receipt)); }
        catch (error) {
          if (!receipts.length) throw error;
          return { confirmed: receipts.length, receipts, error: 'reconciliation_incomplete' };
        }
      } else if (!received.some(row => row.fingerprint === event.fingerprint)) {
        recoverable.push({ fingerprint: event.fingerprint });
      }
    }
    return { confirmed: receipts.length, receipts, recoverable };
  }
  async receiveCurrent(state, waitMs) {
    // Retry durable receive evidence after host restart or a lost HTTP response.
    for (const row of this.presenter.pendingReceived()) {
      await this.sendReceived(row);
    }
    const scope = state.delivery_scope;
    let status = 'idle';
    if (approvedScopeMatches(scope, state, state.session_id)) {
      const { event } = await this.call('streamPoll', waitMs ? { wait_ms: waitMs } : {});
      if (event) {
        const row = this.presenter.receive(event, scope);
        await this.sendReceived(row);
        // Native reminders cannot block receiving or manufacture visibility.
        try { Promise.resolve(this.onReceived?.(row.scope)).catch(() => {}); }
        catch { /* Optional host capability. */ }
        status = 'received';
      }
    }
    // Display proceeds independently. Hidden/offscreen text cannot hold the
    // receive pump, cancel generation, or become a visible acknowledgement.
    if (!this.displayInFlight) {
      const row = this.presenter.pendingReceived(state.session_id)[0];
      if (row) this.displayInFlight = this.presenter.presentReceived(row, {
        bubbleOrders: this.receivedTimelines.get(row.event.fingerprint),
      }).then(async receipt => {
        if (receipt) this.visibleReceipts.push(await this.acknowledge(row.event, receipt));
      }).catch(error => { this.presentationError = error.message; }).finally(() => { this.displayInFlight = null; });
    }
    const receipts = this.visibleReceipts.splice(0);
    const error = this.presentationError; this.presentationError = null;
    return { status, ...(receipts.length ? { receipts } : {}), ...(error ? { error } : {}) };
  }
  async sendReceived(row) {
    if (this.receivedSent.has(row.event.fingerprint)) return;
    const result = await this.call('streamReceive', { sequence: row.event.sequence,
      fingerprint: row.event.fingerprint, host_receipt: row.receipt });
    if (result.bubble_orders) this.receivedTimelines.set(row.event.fingerprint, result.bubble_orders);
    this.receivedSent.add(row.event.fingerprint);
  }
}
