// Renderer-local diagnostics, never presentation authority. Reopening starts a
// new clock domain; history/receipts alone cannot reconstruct a missing t0.
export class DeliveryTiming {
  constructor(clock = () => performance.now(),
    epochClock = () => Math.round((performance.timeOrigin + performance.now()) * 1000)) {
    this.clock = clock; this.epochClock = epochClock; this.rows = []; this.nextId = 0;
  }
  begin(sessionId, kind = 'send') {
    const row = { sample_id: ++this.nextId, session_id: sessionId, kind,
      t0_ms: this.clock(), t0_epoch_us: this.epochClock(),
      turn_id: null, text_attempt_id: null, outcome: 'pending', terminal_ms: null, units: [] };
    this.rows.push(row);
    if (this.rows.length > 100) this.rows.shift();
    return row.sample_id;
  }
  bind(id, attempt) {
    const row = this.rows.find(item => item.sample_id === id);
    if (!row || row.text_attempt_id || !attempt?.attempt_id || !attempt?.turn_id) return;
    row.text_attempt_id = attempt.attempt_id; row.turn_id = attempt.turn_id;
  }
  reject(id) {
    const row = this.rows.find(item => item.sample_id === id);
    if (row && !row.text_attempt_id) { row.outcome = 'send_rejected'; row.terminal_ms = this.clock(); }
  }
  visible(payload) {
    const row = this.rows.find(item => item.text_attempt_id === payload.scope?.text_attempt_id
      && item.turn_id === payload.scope?.turn_id && item.session_id === payload.scope?.session_id);
    const now = this.clock();
    if (!row || row.outcome !== 'pending' || now < row.t0_ms
        || !payload.display_text?.trim() || !payload.scope.input_id || !payload.scope.attempt_id
        || !payload.fingerprint || !Number.isSafeInteger(payload.sequence) || payload.sequence < 1
        || row.units.some(unit => unit.fingerprint === payload.fingerprint)) return;
    row.units.push({ input_id: payload.scope.input_id, attempt_id: payload.scope.attempt_id,
      sequence: payload.sequence, fingerprint: payload.fingerprint, visible_ms: now,
      // Epoch is only for ordering renderer visibility against backend wall time;
      // latency deltas always use the monotonic renderer clock above.
      visible_epoch_us: this.epochClock(), ack_observed_ms: null,
      ack_observed_epoch_us: null, host_receipt: null });
  }
  acknowledged(receipt) {
    for (const row of this.rows) {
      if (row.session_id !== receipt.basis?.session_id) continue;
      const unit = row.units.find(item => item.fingerprint === receipt.fingerprint
        && item.sequence === receipt.sequence && item.input_id === receipt.basis?.input_id
        && item.attempt_id === receipt.basis?.attempt_id);
      const now = this.clock();
      if (unit && unit.ack_observed_ms === null && now >= unit.visible_ms
          && typeof receipt.host_receipt === 'string' && receipt.host_receipt.length) {
        unit.ack_observed_ms = now; unit.ack_observed_epoch_us = this.epochClock();
        unit.host_receipt = receipt.host_receipt;
      }
    }
  }
  observe(state) {
    const row = this.rows.find(item => item.session_id === state.session_id
      && item.text_attempt_id && item.text_attempt_id === state.attempt?.attempt_id
      && item.turn_id === state.attempt?.turn_id);
    const now = this.clock();
    if (row && row.outcome === 'pending' && !state.busy && now >= row.t0_ms
        && ['completed', 'failed', 'cancelled', 'interrupted'].includes(state.attempt?.status)) {
      row.outcome = state.attempt.status; row.terminal_ms = now;
      row.generation_terminal_us = Number.isSafeInteger(state.generation_terminal_us)
        ? state.generation_terminal_us : null;
    }
  }
  snapshot() { return structuredClone(this.rows); }
}

export const deliveryTiming = new DeliveryTiming();
