// Trusted main-process bridge. Renderer ACKs are window/frame/request bound.
import { randomUUID } from 'node:crypto';
import { displayText } from './stream-receipts.mjs';

export function createStreamPresenter({ window, ipcMain, journal, allowedUrl, timeoutMs = 5000 }) {
  if (!Number.isInteger(timeoutMs) || timeoutMs < 1 || timeoutMs > 10000) throw new Error('Bounded render deadline required');
  const pending = new Map(); let disposed = false;
  const visible = () => !window.isDestroyed() && window.isVisible() && !window.isMinimized()
    && !window.webContents.isDestroyed() && window.webContents.getURL() === allowedUrl;
  const receive = (event, payload) => {
    if (!visible() || event.sender !== window.webContents || event.senderFrame !== window.webContents.mainFrame
        || event.senderFrame.url !== allowedUrl || !payload || payload.visible !== true) return;
    const item = pending.get(payload.request_id);
    if (item && Date.now() < item.deadline && item.fingerprint === payload.fingerprint) item.finish(true);
  };
  ipcMain.on('companion:stream-rendered', receive);
  function render(event, scope, recovery = false, received = false, bubbleOrders) {
    if (disposed || !visible() || pending.size) return Promise.resolve(false);
    return new Promise((resolve, reject) => {
      const requestId = randomUUID();
      const duration = recovery ? 120000 : timeoutMs;
      const deadline = Date.now() + duration;
      const timer = setTimeout(() => finish(false), duration);
      const finish = value => { clearTimeout(timer); pending.delete(requestId); resolve(value); };
      pending.set(requestId, { fingerprint: event.fingerprint, finish, recovery, deadline });
      try {
        window.webContents.send('companion:stream-unit', { request_id: requestId,
          fingerprint: event.fingerprint, text: event.text,
          ...(scope || recovery ? { scope, sequence: event.sequence,
            ...(received ? { received: true } : {}),
            ...(recovery ? { recovery: true } : {}),
            deadline_ms: deadline,
            ...(event.bubbles ? { bubble_ranges: event.bubbles } : {}),
            ...(bubbleOrders ? { bubble_orders: bubbleOrders } : {}),
            display_text: displayText(event.presentation.candidate) } : {}) });
      } catch (error) { clearTimeout(timer); pending.delete(requestId); reject(error); }
    });
  }
  return {
    receive: (event, scope) => journal.receive(event, scope),
    pendingReceived: sessionId => journal.pendingReceived(sessionId),
    presentReceived: (row, { bubbleOrders } = {}) => {
      if (disposed || !visible() || pending.size) return Promise.resolve(null);
      return journal.presentReceived(row, event => render(event, row.scope, false, true, bubbleOrders));
    },
    present: (event, options) => {
      if (disposed) return Promise.reject(new Error('Stream presenter disposed'));
      // Refusing locally before any IPC is not an uncertain display attempt.
      // This check, the journal's synchronous intent write and render's send
      // run without yielding. Keep render's second check: a native-window
      // change during filesystem IO still fails conservatively after intent.
      if (!visible() || pending.size) return Promise.resolve(false);
      return journal.present(event, value => render(value, options?.scope), options);
    },
    reconcile: event => journal.lookup(event),
    resume: event => {
      if (disposed) return Promise.reject(new Error('Stream presenter disposed'));
      return journal.resume(event, value => render(value, null, true));
    },
    cancelRecovery() {
      for (const item of pending.values()) if (item.recovery) item.finish(false);
    },
    dispose() {
      disposed = true; ipcMain.removeListener('companion:stream-rendered', receive);
      for (const item of pending.values()) item.finish(false);
    },
  };
}
