// Optional host effect after reliable receipt. No message text, ACK or model state
// belongs here. Restored receipt scopes prevent late/repeated native reminders.
const key = scope => JSON.stringify([scope.session_id, scope.turn_id]);
export class ReplyNotifications {
  constructor({ supported, create, context, activate, receivedScopes = [] }) {
    Object.assign(this, { supported, create, context, activate });
    this.seen = new Set(receivedScopes.map(key)); this.active = new Set(); this.status = 'idle';
  }
  receive(scope) {
    const identity = key(scope);
    if (this.seen.has(identity)) return;
    this.seen.add(identity);
    try {
      const state = this.context();
      if (state.quiet || state.focused && state.session_id === scope.session_id) return;
      if (!this.supported()) { this.status = 'unsupported'; return; }
      const target = { session_id: scope.session_id, turn_id: scope.turn_id };
      const notification = this.create({ title: '文字伴侣', body: '有一条新回复', silent: false });
      this.active.add(notification);
      notification.once('close', () => this.active.delete(notification));
      notification.once('failed', () => { this.active.delete(notification); this.status = 'unavailable'; });
      notification.once('show', () => { this.status = 'shown'; });
      notification.once('click', () => {
        this.active.delete(notification);
        Promise.resolve().then(() => this.activate(target)).catch(() => { this.status = 'activation_failed'; });
      });
      this.status = 'requested'; notification.show();
    } catch { this.status = 'unavailable'; }
  }
  dismiss() {
    for (const notification of this.active) {
      try { notification.close(); } catch { /* Optional host effect only. */ }
    }
    this.active.clear();
  }
}

// A notification click is explicit navigation, never a cancellation. If another
// session is generating, retain the destination until ordinary completion.
export class NotificationNavigation {
  constructor({ getState, selectSession, showTarget, focus }) {
    Object.assign(this, { getState, selectSession, showTarget, focus });
  }
  async open(target) {
    this.pending = target; this.focus(); await this.flush();
  }
  flush() {
    if (!this.inFlight) this.inFlight = this.deliver().finally(() => { this.inFlight = null; });
    return this.inFlight;
  }
  async deliver() {
    const target = this.pending;
    if (!target) return;
    const state = await this.getState();
    if (state.session_id !== target.session_id) {
      if (state.busy) return;
      await this.selectSession(target.session_id);
    }
    if (this.pending !== target) return;
    this.showTarget(target); this.pending = null;
  }
}
