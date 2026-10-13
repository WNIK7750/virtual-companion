import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { createStreamPresenter } from '../src/stream-presenter.mjs';
import { event as streamEvent } from './stream-fixture.mjs';
import { digest, validateStreamEvent } from '../src/stream-receipts.mjs';

test('history readset is included in the immutable approved manifest fingerprint', () => {
  const event = streamEvent();
  event.readset = { protocol: 'history-readset.v1', companion: event.basis.companion_id,
    input: event.basis.input_id, state_version: 1, context_sha256: 'a'.repeat(64),
    facts: [], exchanges: [], history: [] };
  event.fingerprint = digest({ protocol: event.protocol, basis: event.basis, sequence: event.sequence,
    presentation: event.presentation, readset: event.readset });
  assert.equal(validateStreamEvent(event).fingerprint, event.fingerprint);
  event.readset.context_sha256 = 'b'.repeat(64);
  assert.throws(() => validateStreamEvent(event), /fingerprint/);
});

test('hidden or disposed windows cannot attest text visibility', async () => {
  // Event validation is exercised through the real journal suite and Electron pilot.
  const ipcMain = new EventEmitter(); const frame = { url: 'companion://app/index.html' };
  const webContents = { mainFrame: frame, getURL: () => frame.url, isDestroyed: () => false };
  const window = { webContents, isDestroyed: () => false, isVisible: () => false, isMinimized: () => false };
  let journalCalls = 0;
  const presenter = createStreamPresenter({ window, ipcMain, allowedUrl: frame.url,
    journal: { present: async (event, callback) => { journalCalls++; return callback(event); }, lookup: () => ({ status: 'unknown' }) } });
  assert.equal(await presenter.present({}), false);
  assert.equal(journalCalls, 0);
  assert.equal(presenter.reconcile({}).status, 'unknown');
  presenter.dispose();
  assert.equal(ipcMain.listenerCount('companion:stream-rendered'), 0);
  await assert.rejects(presenter.present({}));
});

test('wrong window, child frame and stale fingerprint cannot confirm a display', async () => {
  const ipcMain = new EventEmitter(); const frame = { url: 'companion://app/index.html' }; let sent;
  const webContents = { mainFrame: frame, getURL: () => frame.url, isDestroyed: () => false,
    send: (_channel, payload) => { sent = payload; } };
  const window = { webContents, isDestroyed: () => false, isVisible: () => true, isMinimized: () => false };
  const presenter = createStreamPresenter({ window, ipcMain, allowedUrl: frame.url,
    journal: { present: (event, callback) => callback(event), lookup: () => ({ status: 'unknown' }) } });
  try {
    let resolved = false;
    const receipt = presenter.present(streamEvent()).then(value => { resolved = true; return value; });
    const payload = { ...sent, visible: true };
    ipcMain.emit('companion:stream-rendered', { sender: {}, senderFrame: frame }, payload);
    ipcMain.emit('companion:stream-rendered', { sender: webContents, senderFrame: { ...frame } }, payload);
    ipcMain.emit('companion:stream-rendered', { sender: webContents, senderFrame: frame }, { ...payload, fingerprint: '0'.repeat(64) });
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(resolved, false);
    ipcMain.emit('companion:stream-rendered', { sender: webContents, senderFrame: frame }, payload);
    assert.equal(await receipt, true);
  } finally { presenter.dispose(); }
});

test('default presenter carries UI scope outside the approved immutable envelope', async () => {
  const ipcMain = new EventEmitter(); const frame = { url: 'companion://app/index.html' }; let sent;
  const webContents = { mainFrame: frame, getURL: () => frame.url, isDestroyed: () => false,
    send: (_channel, payload) => { sent = payload; } };
  const window = { webContents, isDestroyed: () => false, isVisible: () => true, isMinimized: () => false };
  const event = streamEvent(); const original = structuredClone(event);
  const presenter = createStreamPresenter({ window, ipcMain, allowedUrl: frame.url,
    journal: { present: (value, callback) => callback(value) } });
  const scope = { session_id: 'session', input_id: 'input', attempt_id: 'attempt', turn_id: 'turn', text_attempt_id: 'ui' };
  try {
    const shown = presenter.present(event, { scope });
    assert.deepEqual(sent.scope, scope); assert.equal(sent.display_text, event.text); assert.equal(sent.sequence, 1);
    assert.ok(sent.deadline_ms > Date.now() && sent.deadline_ms <= Date.now() + 5000);
    assert.deepEqual(event, original);
    ipcMain.emit('companion:stream-rendered', { sender: webContents, senderFrame: frame }, { ...sent, visible: true });
    assert.equal(await shown, true);
  } finally { presenter.dispose(); }
});


test('explicit recovery has a separate bounded challenge and closed callbacks cannot confirm a later view', async () => {
  const ipcMain = new EventEmitter(); const frame = { url: 'companion://app/index.html' }; const sent = [];
  const webContents = { mainFrame: frame, getURL: () => frame.url, isDestroyed: () => false,
    send: (_channel, payload) => sent.push(payload) };
  const window = { webContents, isDestroyed: () => false, isVisible: () => true, isMinimized: () => false };
  const presenter = createStreamPresenter({ window, ipcMain, allowedUrl: frame.url,
    journal: { resume: (value, callback) => callback(value) } });
  const sender = { sender: webContents, senderFrame: frame };
  try {
    const first = presenter.resume(streamEvent());
    assert.equal(sent[0].recovery, true); assert.equal(sent[0].scope, null);
    assert.ok(sent[0].deadline_ms > Date.now() + 100000);
    presenter.cancelRecovery(); assert.equal(await first, false);
    let confirmed = false;
    const second = presenter.resume(streamEvent()).then(value => { confirmed = value; return value; });
    assert.notEqual(sent[0].request_id, sent[1].request_id);
    ipcMain.emit('companion:stream-rendered', sender, { ...sent[0], visible: true });
    await new Promise(resolve => setImmediate(resolve)); assert.equal(confirmed, false);
    ipcMain.emit('companion:stream-rendered', sender, { ...sent[1], visible: true });
    assert.equal(await second, true);
  } finally { presenter.dispose(); }
});


test('an expired challenge cannot confirm even before its timeout callback is scheduled', async () => {
  const ipcMain = new EventEmitter(); const frame = { url: 'companion://app/index.html' }; let sent;
  const webContents = { mainFrame: frame, getURL: () => frame.url, isDestroyed: () => false,
    send: (_channel, payload) => { sent = payload; } };
  const window = { webContents, isDestroyed: () => false, isVisible: () => true, isMinimized: () => false };
  const presenter = createStreamPresenter({ window, ipcMain, allowedUrl: frame.url,
    journal: { resume: (value, callback) => callback(value) } });
  const originalNow = Date.now;
  try {
    let resolved = false;
    const shown = presenter.resume(streamEvent()).then(value => { resolved = value; return value; });
    Date.now = () => sent.deadline_ms;
    ipcMain.emit('companion:stream-rendered', { sender: webContents, senderFrame: frame }, { ...sent, visible: true });
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(resolved, false); presenter.cancelRecovery(); assert.equal(await shown, false);
  } finally { Date.now = originalNow; presenter.dispose(); }
});
