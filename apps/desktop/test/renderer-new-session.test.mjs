import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
import { DeliveryTiming } from '../src/delivery-timing.mjs';
import { approvedScopeMatches, canRetry } from '../src/approved-display.mjs';

const source = await readFile(new URL('../src/renderer.mjs', import.meta.url), 'utf8');
const deferred = () => {
  let resolve; const promise = new Promise(done => { resolve = done; }); return { promise, resolve };
};
function harness() {
  const elements = new Map(), tags = [{ value: 'synthetic', checked: false }];
  class Element {
    constructor() { this.children = []; this.hidden = false; this.disabled = false; this.open = false;
      this.value = ''; this.textContent = ''; this.attributes = {}; this.classList = { remove() {} }; }
    set id(value) { this._id = value; elements.set(value, this); }
    get id() { return this._id; }
    append(...values) { this.children.push(...values); }
    before() {} after() {} focus() { this.focused = true; }
    setAttribute(key, value) { this.attributes[key] = value; }
    replaceChildren(...values) { this.children = values; }
    addEventListener() {}
    showModal() { this.open = true; }
    close() { this.open = false; }
  }
  const element = id => {
    if (!elements.has(id)) { const item = new Element(); item.id = id; } return elements.get(id);
  };
  const document = { body: new Element(), visibilityState: 'visible',
    getElementById: element, createElement: () => new Element(),
    querySelectorAll: () => tags };
  let state = { state: 'ready', mode: 'fake', session_id: 'old-session', busy: false,
    memory_management: true, delivery_protocol: 'approved-prefix.v1',
    attempt: { attempt_id: 'old-attempt', turn_id: 'old-turn', status: 'completed' } };
  const historyCalls = [], messages = [{ id: 'old-user', turn_id: 'old-turn', role: 'user', content: 'old' }];
  let list, navigations = 0, createCalls = 0;
  const timing = new DeliveryTiming(() => 100, () => 100000);
  timing.bind(timing.begin('old-session'), state.attempt);
  const api = { state: async () => ({ ...state }),
    newSession: async () => { createCalls++; state = { ...state, session_id: 'new-session', attempt: null }; },
    history: async payload => { historyCalls.push(payload); return { session_id: payload.session_id,
      messages: payload.session_id === 'old-session' ? messages : [],
      next_before: payload.session_id === 'old-session' ? 'old-cursor' : null }; },
    sessions: async () => ({ sessions: [{ id: state.session_id, title: state.session_id }], next_offset: 50 }),
    presentApproved: async () => ({ receipts: [], recoverable: [] }), onApproved() {},
    cancelRecovery: async () => ({ status: 'closed' }),
    feedback: async () => ({ events: [] }),
    send: async () => ({ attempt_id: 'new-attempt', turn_id: 'new-turn' }) };
  class List {
    constructor(_element, onFeedback) { list = this; this.onFeedback = onFeedback; this.approved = new Map(); }
    update(values, _state, session) { if (this.session !== session) this.approved.clear();
      this.session = session; this.messages = values; }
  }
  class Panel { setAvailable() {} setState() {} }
  const context = vm.createContext({ document, window: { companion: api },
    icon: () => new Element(), populateIcons() {}, initializePanels() {},
    MessageList: List, MemoryPanel: Panel, ProviderPanel: Panel,
    approvedScopeMatches, canRetry, isReadingNavigation: () => false, waitForApprovedText: async () => false,
    deliveryTiming: timing, crypto: { randomUUID: () => 'synthetic-uuid' },
    location: { reload: () => { navigations++; } }, setTimeout() {}, console });
  // Execute the actual renderer control flow, replacing only imported leaf UI
  // adapters. No Electron, HTTP listener, credential, or product DB is used.
  vm.runInContext(source.replace(/^import .*;\r?\n/gmu, '') + '\n' +
    "globalThis.rendererTest = { refresh, openFeedback," +
    "seedLocal() { olderSessions=[{id:'stale-older',title:'stale'}]; olderLoaded=true;" +
    "nextOffset=100; feedbackTarget={id:'old-assistant',turn_id:'old-turn'};" +
    "feedbackEvent={feedback_id:'old-feedback'}; previewId='old-preview';" +
    "recoveryFingerprint='old-fingerprint'; recoverable=[{fingerprint:'old-fingerprint'}]; }," +
    "local() { return {sessionId,messages,nextMessageBefore,olderSessions,olderLoaded,nextOffset," +
    "feedbackTarget,feedbackEvent,previewId,recoveryFingerprint,recoverable,sending}; }};", context);
  return { api, element, tags, timing, historyCalls, get list() { return list; },
    get navigations() { return navigations; }, get createCalls() { return createCalls; },
    renderer: context.rendererTest };
}

test('new-session real renderer handler refreshes the new scope without page navigation', async () => {
  const h = harness(); await h.renderer.refresh();
  h.renderer.seedLocal(); h.list.approved.set('old-turn', { content: 'old approved' });
  h.element('message').value = 'old draft'; h.element('feedback-dialog').open = true;
  h.element('feedback-text').value = 'old feedback'; h.element('export-panel').hidden = false;
  h.element('export-confirmation').checked = true; h.tags[0].checked = true;
  h.element('recovery-dialog').open = true;
  await h.element('new-session').onclick();
  const local = h.renderer.local();
  assert.equal(h.navigations, 0);
  assert.equal(h.createCalls, 1); assert.equal(local.sessionId, 'new-session');
  assert.equal(local.messages.length, 0); assert.equal(local.nextMessageBefore, null);
  assert.equal(local.olderSessions.length, 0); assert.equal(local.olderLoaded, false);
  assert.equal(local.feedbackTarget, undefined); assert.equal(local.feedbackEvent, undefined);
  assert.equal(local.previewId, undefined); assert.equal(local.recoveryFingerprint, null);
  assert.equal(h.element('feedback-dialog').open, false);
  assert.equal(h.element('feedback-text').value, ''); assert.equal(h.tags[0].checked, false);
  assert.equal(h.element('export-panel').hidden, true); assert.equal(h.element('export-confirmation').checked, false);
  assert.equal(h.element('recovery-dialog').open, false); assert.equal(h.list.approved.size, 0);
  assert.equal(h.element('message').value, ''); assert.equal(h.element('send').disabled, false);
  assert.equal(h.element('new-session').disabled, false);
  assert.equal(h.historyCalls.at(-1).session_id, 'new-session');
  assert.equal('after' in h.historyCalls.at(-1), false);
  assert.equal(h.timing.snapshot()[0].session_id, 'old-session', 'Same document keeps its real clock and old measurement rows');
  h.element('message').value = 'new message';
  await h.element('composer').onsubmit({ preventDefault() {} });
  await new Promise(done => setImmediate(done));
  assert.equal(h.timing.snapshot().at(-1).session_id, 'new-session');
  assert.equal(h.timing.snapshot().at(-1).text_attempt_id, 'new-attempt');
});

test('failed new-session preserves old view and restores input availability', async () => {
  const h = harness(); await h.renderer.refresh();
  h.element('message').value = 'retained draft';
  h.api.newSession = async () => { throw Error('synthetic create failure'); };
  await h.element('new-session').onclick();
  assert.equal(h.navigations, 0); assert.equal(h.renderer.local().sessionId, 'old-session');
  assert.equal(h.element('message').value, 'retained draft');
  assert.equal(h.element('send').disabled, false); assert.equal(h.element('new-session').disabled, false);
  assert.equal(h.element('error').textContent, 'synthetic create failure');
});

test('new-session yields a held old refresh and discards its late response', async () => {
  const h = harness(); await h.renderer.refresh();
  const gate = deferred(), base = h.api.sessions; let held = false;
  h.api.sessions = async offset => { if (!held) { held = true; await gate.promise; } return base(offset); };
  const old = h.renderer.refresh();
  await new Promise(done => setImmediate(done));
  const next = h.element('new-session').onclick();
  await new Promise(done => setImmediate(done));
  try { assert.equal(h.renderer.local().sessionId, 'new-session');
    assert.equal(h.renderer.local().sending, false); } finally { gate.resolve(); }
  await Promise.all([old, next]);
  assert.equal(h.navigations, 0); assert.equal(h.renderer.local().sessionId, 'new-session');
  assert.equal(h.historyCalls.filter(row => row.session_id === 'new-session').length, 1);
});

test('late older-session page cannot refill the cache after new-session', async () => {
  const h = harness(); await h.renderer.refresh();
  const gate = deferred(), base = h.api.sessions;
  h.api.sessions = offset => offset === undefined ? base() : gate.promise;
  const older = h.element('history-more').onclick();
  await h.element('new-session').onclick();
  gate.resolve({ sessions: [{ id: 'late-old', title: 'old' }], next_offset: 100 }); await older;
  assert.equal(h.renderer.local().sessionId, 'new-session');
  assert.equal(h.renderer.local().olderSessions.length, 0);
});

test('late old feedback list cannot reopen a dialog in the new session', async () => {
  const h = harness(); await h.renderer.refresh();
  const gate = deferred(); h.api.feedback = () => gate.promise;
  const feedback = h.renderer.openFeedback({ id: 'old-assistant', turn_id: 'old-turn' });
  await h.element('new-session').onclick();
  gate.resolve({ events: [{ assistant_message_id: 'old-assistant', feedback_id: 'old-feedback', quick_tags: [] }] });
  await feedback;
  assert.equal(h.element('feedback-dialog').open, false);
  assert.equal(h.renderer.local().feedbackTarget, undefined);
  assert.equal(h.renderer.local().feedbackEvent, undefined);
});
