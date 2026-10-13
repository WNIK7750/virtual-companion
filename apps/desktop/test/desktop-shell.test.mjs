import test from 'node:test';
import assert from 'node:assert/strict';
import { createShellHandlers, windowOptions } from '../src/shell.mjs';
import { attentionAllowsSelection, memoryStatsText, mergeCurrentMemoryPage } from '../src/memory-panel.mjs';

test('shell rejects foreign frames and undeclared actions before service invocation', async () => {
  let called = 0;
  const frame = { url: 'companion://app/index.html' };
  const contents = { mainFrame: frame };
  const handle = createShellHandlers(contents, { state: () => { called++; return { state: 'stopped' }; } });
  assert.throws(() => handle({ sender: {}, senderFrame: frame }, 'state', {}), /sender/);
  assert.throws(() => handle({ sender: contents, senderFrame: { url: frame.url } }, 'state', {}), /sender/);
  assert.throws(() => handle({ sender: contents, senderFrame: frame }, 'execute', {}), /action/);
  assert.throws(() => handle({ sender: contents, senderFrame: frame }, 'state', { path: 'x' }), /payload/);
  assert.equal(called, 0);
  assert.deepEqual(await handle({ sender: contents, senderFrame: frame }, 'state', {}), { state: 'stopped' });
});

test('shell rejects oversized or empty messages, including inherited input', () => {
  const frame = { url: 'companion://app/index.html' };
  const contents = { mainFrame: frame };
  const handle = createShellHandlers(contents, {});
  const event = { sender: contents, senderFrame: frame };
  for (const payload of [{ text: '' }, { text: 'a'.repeat(8001) }, Object.create({ text: 'hi' })]) {
    assert.throws(() => handle(event, 'send', payload), /payload/);
  }
});

test('renderer send accepts text only and rejects caller-supplied receiving context', () => {
  const frame = { url: 'companion://app/index.html' }, contents = { mainFrame: frame };
  const sent = [];
  const handle = createShellHandlers(contents, { send: payload => { sent.push(payload); return payload; } });
  const event = { sender: contents, senderFrame: frame };
  assert.deepEqual(handle(event, 'send', { text: 'hello' }), { text: 'hello' });
  assert.throws(() => handle(event, 'send', {
    text: 'hello', receiving_context: {
      protocol: 'receiving-context.v1', source: 'receiving-client', scope: 'this_client',
      observation: 'at_input_submission', quiet: true,
    },
  }), /payload/);
  assert.deepEqual(sent, [{ text: 'hello' }]);
});

test('feedback shell preserves exact selectors and binds preview edits to revision and hash', () => {
  const frame = { url: 'companion://app/index.html' }, contents = { mainFrame: frame };
  const calls = [];
  const handle = createShellHandlers(contents, { feedback: payload => { calls.push(payload); return payload; } });
  const event = { sender: contents, senderFrame: frame };
  const submit = {
    operation: 'submit', session_id: 'session', turn_id: 'turn', assistant_message_id: 'message',
    target: { kind: 'bubble', start: 3, end: 8 }, quick_tags: [],
    freeform_reaction: '真实感受', idempotency_key: 'key',
  };
  assert.deepEqual(handle(event, 'feedback', submit), submit);
  const attempt = { ...submit, target: { kind: 'attempt', attempt_id: 'attempt-1' } };
  delete attempt.assistant_message_id;
  assert.deepEqual(handle(event, 'feedback', attempt), attempt);
  for (const target of [
    { kind: 'attempt', attempt_id: 'a'.repeat(129) },
    { kind: 'bubble', start: 0, end: 2, text: 'not allowed' },
    { kind: 'bubble', start: 1.5, end: 2 },
    { kind: 'bubble', start: 3, end: 3 },
  ]) assert.throws(() => handle(event, 'feedback', { ...submit, target }), /payload/);
  const previewUpdate = { operation: 'preview', preview_id: 'preview-1',
    expected_revision: 1, expected_sha256: 'a'.repeat(64),
    redacted_payload: { schema_version: 1, feedback: [], context: [] } };
  assert.deepEqual(handle(event, 'feedback', previewUpdate), previewUpdate);
  const confirm = { operation: 'confirm', preview_id: 'preview-1', preview_revision: 2,
    payload_sha256: 'b'.repeat(64), redaction_confirmed: true, external_processing_confirmed: true };
  assert.deepEqual(handle(event, 'feedback', confirm), confirm);
  for (const payload of [
    { ...previewUpdate, expected_sha256: 'A'.repeat(64) },
    { ...previewUpdate, expected_revision: 0 },
    { ...previewUpdate, redacted_payload: { schema_version: 1, feedback: [],
      context: [Object.assign(Object.create({ inherited: 'hidden' }), { selected_text: null })] } },
    { ...confirm, preview_revision: 1.5 },
    { ...confirm, payload_sha256: 'bad' },
  ]) assert.throws(() => handle(event, 'feedback', payload), /payload/);
  assert.equal(calls.length, 4);
});

test('window always isolates renderer and has usable minimum dimensions', () => {
  const options = windowOptions('/trusted/preload.cjs');
  assert.equal(options.show, false, 'main window must remain hidden until configured and painted');
  assert.equal(options.webPreferences.sandbox, true);
  assert.equal(options.webPreferences.contextIsolation, true);
  assert.equal(options.webPreferences.nodeIntegration, false);
  assert.equal(options.webPreferences.webSecurity, true);
  assert.ok(options.minWidth >= 480);
});

test('renderer can request delivery but cannot choose its receipt or scope', () => {
  const frame = { url: 'companion://app/index.html' }, contents = { mainFrame: frame };
  const handle = createShellHandlers(contents, { presentApproved: () => 'delivery' });
  const event = { sender: contents, senderFrame: frame };
  assert.equal(handle(event, 'presentApproved', {}), 'delivery');
  for (const payload of [{ host_receipt: 'fake' }, { scope: {} }, { fingerprint: 'fake' }]) {
    assert.throws(() => handle(event, 'presentApproved', payload), /payload/);
  }
  assert.throws(() => handle(event, 'streamAck', {}), /action/);
});

test('history accepts one bounded cursor and rejects unexpected fields', () => {
  const frame = { url: 'companion://app/index.html' }, contents = { mainFrame: frame };
  const handle = createShellHandlers(contents, { history: payload => payload });
  const event = { sender: contents, senderFrame: frame };
  assert.deepEqual(handle(event, 'history', { before: 'message-id', session_id: 'session-id' }), { before: 'message-id', session_id: 'session-id' });
  for (const payload of [{ before: 'a', after: 'b' }, { before: -1 }, { after: '' }, { session_id: 'a'.repeat(129) }, { limit: 99999 }]) {
    assert.throws(() => handle(event, 'history', payload), /payload/);
  }
});

test('memory list and actions accept only bounded typed payloads', () => {
  const frame = { url: 'companion://app/index.html' }, contents = { mainFrame: frame };
  const service = {
    memories: payload => payload, memoryDetails: payload => payload,
    memoryAction: payload => payload, memoryAbandon: payload => payload,
  };
  const handle = createShellHandlers(contents, service);
  const event = { sender: contents, senderFrame: frame };
  assert.deepEqual(handle(event, 'memories', { offset: 50, limit: 50 }), { offset: 50, limit: 50 });
  assert.deepEqual(handle(event, 'memoryDetails', { record_id: 'memory-1' }), { record_id: 'memory-1' });
  const action = { operation_id: 'operation-1', action: 'forget', record_id: 'memory-1', expected_version: 2, replacement: '' };
  assert.deepEqual(handle(event, 'memoryAction', action), action);
  assert.deepEqual(handle(event, 'memoryAbandon', { job: 'job-1', operation_id: 'abandon-1' }), { job: 'job-1', operation_id: 'abandon-1' });
  for (const payload of [
    { offset: -1 }, { limit: 101 }, { offset: '0' }, { unexpected: 1 },
  ]) assert.throws(() => handle(event, 'memories', payload), /payload/);
  for (const payload of [
    { ...action, action: 'delete' }, { ...action, expected_version: 0 },
    { ...action, replacement: 'not-empty' }, { ...action, extra: true },
  ]) assert.throws(() => handle(event, 'memoryAction', payload), /payload/);
});

test('stale memory actions keep the refreshed active record visible', () => {
  const old = { record_id: 'memory-1', version: 1, status: 'active' };
  const current = { record_id: 'memory-1', version: 2, status: 'active' };
  assert.deepEqual(
    mergeCurrentMemoryPage([{ record_id: 'memory-2', version: 1, status: 'active' }], current),
    [current, { record_id: 'memory-2', version: 1, status: 'active' }],
  );
  assert.deepEqual(mergeCurrentMemoryPage([old], current), [current]);
  assert.deepEqual(mergeCurrentMemoryPage([old], { ...current, status: 'forgotten' }), [old]);
});

test('mixed memory attention never offers a partial record selection', () => {
  assert.equal(attentionAllowsSelection({ resolution_mode: 'record_selection' }), true);
  assert.equal(attentionAllowsSelection({ resolution_mode: 'whole_source' }), false);
  assert.equal(attentionAllowsSelection({}), false);
});

test('memory overview distinguishes effective records, revisions, and unknown event time', () => {
  assert.equal(memoryStatsText({
    effective_count: 2, current_record_count: 3, history_version_count: 5,
    unknown_event_time_count: 2,
  }), '当前有效 2 条 · 当前记录 3 条 · 历史版本 5 条 · 发生时间不详 2 条');
  assert.equal(memoryStatsText(null), '记忆概况待确认');
});

test('provider selection is an explicit bounded action', () => {
  const frame = { url: 'companion://app/index.html' }, contents = { mainFrame: frame };
  const handle = createShellHandlers(contents, { selectProvider: payload => payload });
  const event = { sender: contents, senderFrame: frame };
  assert.deepEqual(handle(event, 'selectProvider', { provider_id: 'deepseek-api' }), { provider_id: 'deepseek-api' });
  for (const payload of [{ provider_id: '' }, { provider_id: 'a'.repeat(65) }, { provider_id: 3 }, { provider_id: 'deepseek-api', extra: true }]) {
    assert.throws(() => handle(event, 'selectProvider', payload), /payload/);
  }
});


test('recovery renderer may select only a fingerprint, never content, scope or a receipt', () => {
  const frame = { url: 'companion://app/index.html' }, contents = { mainFrame: frame };
  const handle = createShellHandlers(contents, { resumeApproved: payload => payload, cancelRecovery: () => 'closed' });
  const event = { sender: contents, senderFrame: frame };
  const fingerprint = 'a'.repeat(64);
  assert.deepEqual(handle(event, 'resumeApproved', { fingerprint }), { fingerprint });
  for (const payload of [{}, { fingerprint: 'wrong' }, { fingerprint, text: 'fake' },
    { fingerprint, host_receipt: 'fake' }, { fingerprint, scope: {} }]) {
    assert.throws(() => handle(event, 'resumeApproved', payload), /payload/);
  }
  assert.equal(handle(event, 'cancelRecovery', {}), 'closed');
  assert.throws(() => handle(event, 'cancelRecovery', { fingerprint }), /payload/);
});
