import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';

const source = await readFile(new URL('../src/renderer.mjs', import.meta.url), 'utf8');

function harness() {
  const elements = new Map(), tags = [{ value: 'too_mechanical', checked: false }];
  class Element {
    constructor() {
      this.children = []; this.hidden = false; this.disabled = false; this.open = false;
      this.value = ''; this.textContent = ''; this.attributes = {}; this.scrollTop = 0;
      this.scrollHeight = 0; this.clientHeight = 0;
      this.classList = { add() {}, remove() {}, contains: () => false };
    }
    set id(value) { this._id = value; elements.set(value, this); }
    get id() { return this._id; }
    append(...values) { this.children.push(...values); }
    before() {} after() {} focus() {}
    setAttribute(key, value) { this.attributes[key] = value; }
    replaceChildren(...values) { this.children = values; }
    addEventListener() {}
    showModal() { this.open = true; }
    close() { this.open = false; }
  }
  const element = id => {
    if (!elements.has(id)) { const item = new Element(); item.id = id; }
    return elements.get(id);
  };
  const document = {
    body: new Element(), visibilityState: 'visible', getElementById: element,
    createElement: () => new Element(),
    querySelectorAll: selector => selector.includes(':checked') ? tags.filter(item => item.checked)
      : selector.includes('quick_tags') ? tags : [],
  };
  const apiCalls = [];
  let state = { state: 'ready', mode: 'fake', session_id: 'session', busy: false,
    memory_management: false, attempt: null, delivery_protocol: 'none',
    receiving_preferences: { quiet: false } };
  let revision = 1;
  let currentPayload = { schema_version: 1, feedback: [{ freeform_reaction: 'private reaction' }],
    context: [{ selected_text: 'private answer' }] };
  const api = {
    state: async () => ({ ...state }),
    history: async payload => ({ session_id: payload.session_id, messages: [], next_before: null }),
    sessions: async () => ({ sessions: [{ id: 'session', title: 'synthetic' }], next_offset: null }),
    presentApproved: async () => ({ receipts: [], recoverable: [] }),
    resumeApproved: async () => ({ status: 'confirmed' }), cancelRecovery: async () => ({}),
    onApproved() {},
    feedback: async payload => {
      apiCalls.push(structuredClone(payload));
      if (payload.operation === 'list') return { events: [] };
      if (payload.operation === 'submit') return {
        feedback_id: 'feedback-1', feedback_revision: 1, session_id: payload.session_id,
        turn_id: payload.turn_id, assistant_message_id: payload.assistant_message_id || null,
        quick_tags: payload.quick_tags, freeform_reaction: payload.freeform_reaction,
        target_binding: payload.target?.kind === 'attempt' ? { kind: 'attempt', status: 'attempt-only' }
          : { kind: 'bubble', status: 'visible' },
      };
      if (payload.operation === 'preview' && payload.feedback_ids) {
        revision = 1;
        return { preview_id: 'preview-1', preview_revision: revision,
          payload_sha256: 'a'.repeat(64), payload: structuredClone(currentPayload) };
      }
      if (payload.operation === 'preview') {
        revision++;
        currentPayload = structuredClone(payload.redacted_payload);
        return { preview_id: payload.preview_id, preview_revision: revision,
          payload_sha256: String.fromCharCode(96 + revision).repeat(64), payload: structuredClone(currentPayload) };
      }
      if (payload.operation === 'confirm') return { saved_locally: true };
      throw new Error('unexpected feedback operation');
    },
  };
  class MessageList {
    constructor(_container, onFeedback) { this.onFeedback = onFeedback; this.approved = new Map(); }
    update(_messages, _state, session) { this.sessionId = session; }
  }
  class Panel { setAvailable() {} setState() {} }
  const timing = { observe() {}, begin: () => 'timing', bind() {}, reject() {},
    acknowledged() {}, visible() {} };
  const context = vm.createContext({ document, window: { companion: api },
    icon: () => new Element(), populateIcons() {}, initializePanels() {},
    MessageList, MemoryPanel: Panel, ProviderPanel: Panel,
    approvedScopeMatches: () => false, canRetry: () => false, isReadingNavigation: () => false,
    waitForApprovedText: async () => false, deliveryTiming: timing,
    crypto: { randomUUID: () => 'synthetic-key' }, TextEncoder, setTimeout() {}, console });
  vm.runInContext(source.replace(/^import .*;\r?\n/gmu, '') +
    '\nglobalThis.rendererTest = { refresh, openFeedback };', context);
  return { api, apiCalls, element, tags, renderer: context.rendererTest };
}

test('renderer submits the exact selected target and confirms only the applied preview hash/revision', async () => {
  const h = harness();
  await h.renderer.refresh();
  const message = { id: 'reply-1', turn_id: 'turn-1', role: 'assistant', content: '完整回复' };
  const target = { kind: 'bubble', start: 1, end: 3 };
  await h.renderer.openFeedback(message, target);
  h.element('feedback-text').value = '我的感受';
  await h.element('feedback-form').onsubmit({ preventDefault() {} });
  const submit = h.apiCalls.find(call => call.operation === 'submit');
  assert.equal(submit.assistant_message_id, message.id);
  assert.deepEqual(submit.target, target);

  await h.element('feedback-export').onclick();
  h.element('export-preview').value = JSON.stringify({ schema_version: 1,
    feedback: [{ freeform_reaction: 'reaction' }], context: [{ selected_text: '[已脱敏]' }] });
  await h.element('export-apply').onclick();
  let update = h.apiCalls.filter(call => call.operation === 'preview').at(-1);
  assert.equal(update.expected_revision, 1);
  assert.equal(update.expected_sha256, 'a'.repeat(64));
  assert.equal(update.redacted_payload.context[0].selected_text, '[已脱敏]');
  assert.match(h.element('export-preview-meta').textContent, /预览修订 2.*SHA-256/);

  h.element('export-confirmation').checked = true;
  h.element('export-confirmation').onchange();
  assert.equal(h.element('export-save').disabled, false);
  h.element('export-preview').value = JSON.stringify(currentPayloadForTest());
  h.element('export-preview').oninput();
  assert.equal(h.element('export-confirmation').checked, false);
  assert.equal(h.element('export-save').disabled, true);
  await h.element('export-apply').onclick();
  h.element('export-confirmation').checked = true;
  h.element('export-confirmation').onchange();
  await h.element('export-save').onclick();
  const confirm = h.apiCalls.find(call => call.operation === 'confirm');
  assert.equal(confirm.preview_revision, 3);
  assert.equal(confirm.payload_sha256, 'c'.repeat(64));
  assert.equal(confirm.redaction_confirmed, true);
  assert.equal(confirm.external_processing_confirmed, true);

  await h.renderer.openFeedback({ id: 'user-1', turn_id: 'turn-2', role: 'user', content: '问题' },
    { kind: 'attempt', attempt_id: 'attempt-2' });
  h.element('feedback-text').value = '这次失败';
  await h.element('feedback-form').onsubmit({ preventDefault() {} });
  const attempt = h.apiCalls.filter(call => call.operation === 'submit').at(-1);
  assert.deepEqual(attempt.target, { kind: 'attempt', attempt_id: 'attempt-2' });
  assert.equal(Object.hasOwn(attempt, 'assistant_message_id'), false);
  assert.match(h.element('feedback-title').textContent, /尝试/);
});

function currentPayloadForTest() {
  return { schema_version: 1, feedback: [{ freeform_reaction: 'reaction' }],
    context: [{ selected_text: '[已脱敏]' }] };
}
