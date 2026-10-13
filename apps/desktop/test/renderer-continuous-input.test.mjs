import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
import { DeliveryTiming } from '../src/delivery-timing.mjs';
import { approvedScopeMatches, canRetry } from '../src/approved-display.mjs';

const source = await readFile(process.env.VC_CONTINUOUS_RENDERER_SOURCE
  || new URL('../src/renderer.mjs', import.meta.url), 'utf8');
const flush = () => new Promise(resolve => setImmediate(resolve));
function deferred() {
  let resolve, reject;
  const promise = new Promise((done, fail) => { resolve = done; reject = fail; });
  return { promise, resolve, reject };
}

function harness({ busy = false, stateName = 'ready' } = {}) {
  const elements = new Map();
  class Element {
    constructor() {
      this.children = []; this.value = ''; this.hidden = false; this.disabled = false;
      this.open = false; this.textContent = ''; this.scrollTop = 0; this.scrollHeight = 100; this.clientHeight = 100; this.classList = { remove() {} };
    }
    set id(value) { this._id = value; elements.set(value, this); }
    get id() { return this._id; }
    append(...values) { this.children.push(...values); }
    before() {} after() {} focus() {} setAttribute() {} addEventListener() {}
    replaceChildren(...values) { this.children = values; }
    close() { this.open = false; }
    showModal() { this.open = true; }
  }
  function element(id) {
    if (!elements.has(id)) { const value = new Element(); value.id = id; }
    return elements.get(id);
  }
  const document = { body: new Element(), visibilityState: 'visible', getElementById: element,
    createElement: () => new Element(), querySelectorAll: () => [] };
  let state = { state: stateName, mode: 'fake', busy, session_id: 'same-session',
    delivery_protocol: 'approved-prefix.v1', memory_management: true,
    attempt: { attempt_id: 'current-text-attempt', turn_id: 'current-turn', status: 'streaming' },
    delivery_scope: { text_attempt_id: 'stale-delivery-attempt', turn_id: 'stale-turn' } };
  const calls = [], timers = []; let approvedCallback; let updates = 0; let presentations = 0;
  let resumeInputsCalls = 0, stopCalls = 0;
  let coverage = async () => false;
  const api = {
    state: async () => ({ ...state }),
    sessions: async () => ({ sessions: [{ id: state.session_id, title: 'synthetic' }], next_offset: null }),
    history: async () => ({ session_id: state.session_id, messages: [], next_before: null }),
    presentApproved: async () => ({ status: 'idle', receipts: [], recoverable: [] }),
    onApproved(callback) { approvedCallback = callback; },
    send: async text => { calls.push(text); return { attempt_id: 'next-attempt', turn_id: 'next-turn' }; },
    stop: async () => { stopCalls++; return { busy: false }; },
    resumeInputs: async () => { resumeInputsCalls++; return { status: 'resumed' }; },
    cancelRecovery: async () => ({}), feedback: async () => ({ events: [] }),
  };
  class List { update() { updates++; } presentApproved() { presentations++; return new Element(); } }
  class Panel { setAvailable() {} setState() {} }
  const timing = new DeliveryTiming(() => 100, () => 100000);
  const context = vm.createContext({ document, window: { companion: api },
    icon: () => new Element(), populateIcons() {}, initializePanels() {}, MessageList: List,
    MemoryPanel: Panel, ProviderPanel: Panel, approvedScopeMatches, canRetry,
    isReadingNavigation: () => false, waitForApprovedText: options => coverage(options), deliveryTiming: timing,
    crypto: { randomUUID: () => 'synthetic-only' }, location: {}, setTimeout(callback, delay) { timers.push({ callback, delay }); }, console });
  // Actual renderer handler and refresh; imported leaf DOM panels are CPU stubs.
  vm.runInContext(source.replace(/^import .*;\r?\n/gmu, '') + '\n'
    + `globalThis.continuousTest = {refresh, pollDelivery: typeof pollDelivery === 'function' ? pollDelivery : poll, pollApproved: typeof pollApproved === 'function' ? pollApproved : poll,
     local:()=>({sending,busy,cancelledDeliveryAttempt,sessionId,manualReading})};`, context);
  return { element, api, calls, timing, timers, renderer: context.continuousTest,
    approved: payload => approvedCallback(payload), setCoverage: value => { coverage = value; },
    get resumeInputsCalls() { return resumeInputsCalls; }, get stopCalls() { return stopCalls; },
    get updates() { return updates; }, get presentations() { return presentations; },
    setState: next => { state = { ...state, ...next }; } };
}

async function submit(h) {
  h.element('composer').onsubmit({ preventDefault() {} });
  await flush();
}

test('unconfigured API state blocks sending and shows settings instead of local startup', async () => {
  const h = harness();
  await h.renderer.refresh(); await flush();
  h.setState({ mode: 'api', state: 'provider_unavailable', error: '合成受管配置缺失' });
  await h.renderer.refresh(); await flush();
  assert.match(h.element('status').textContent, /DeepSeek Flash 未就绪/u);
  assert.equal(h.element('send').disabled, true);
  assert.equal(h.element('start').hidden, true);
  assert.equal(h.element('error').textContent, '合成受管配置缺失');
  h.element('message').value = '合成未发送';
  await submit(h);
  assert.deepEqual(h.calls, []);
});

test('successful API binding clears its prior configuration error and keeps local start hidden', async () => {
  const h = harness();
  await h.renderer.refresh(); await flush();
  h.setState({ mode: 'api', state: 'provider_unavailable', error: '合成受管配置缺失' });
  await h.renderer.refresh(); await flush();
  assert.equal(h.element('error').textContent, '合成受管配置缺失');
  h.setState({ mode: 'api', state: 'ready', error: '' });
  await h.renderer.refresh(); await flush();
  assert.equal(h.element('status').textContent, 'DeepSeek Flash 已就绪');
  assert.equal(h.element('send').disabled, false);
  assert.equal(h.element('start').hidden, true);
  assert.equal(h.element('error').hidden, true);
});

test('busy ready renderer submits once and keeps current delivery active despite stale delivery scope', async () => {
  const h = harness({ busy: true }); await h.renderer.refresh(); await flush();
  assert.equal(h.element('send').disabled, false);
  h.element('message').value = '嗯，继续';
  await submit(h); await flush();
  assert.deepEqual(h.calls, ['嗯，继续']);
  assert.equal(h.renderer.local().cancelledDeliveryAttempt, undefined);
  assert.equal(h.timing.snapshot().at(-1).text_attempt_id, 'next-attempt');
});

test('busy unpaused input queue reports received count without changing current cancellation', async () => {
  const h = harness({ busy: true }); await h.renderer.refresh(); await flush();
  await h.element('stop').onclick();
  const cancelledAttempt = h.renderer.local().cancelledDeliveryAttempt;
  assert.equal(cancelledAttempt, 'current-text-attempt');
  h.setState({ busy: true, input_queue_paused: false, queued_inputs: [
    { turn_id: 'queued-turn-one', attempt_id: 'queued-attempt-one' },
    { turn_id: 'queued-turn-two', attempt_id: 'queued-attempt-two' },
  ] });
  await h.renderer.refresh(); await flush();
  assert.equal(h.element('input-queue-status').hidden, false);
  assert.match(h.element('input-queue-status').textContent, /2 条消息已收到/u);
  assert.equal(h.element('resume-inputs').hidden, true);
  assert.equal(h.renderer.local().cancelledDeliveryAttempt, cancelledAttempt);
});

test('paused queued inputs expose one resume action and hide it after unpaused refresh', async () => {
  const h = harness(); await h.renderer.refresh(); await flush();
  h.setState({ busy: false, input_queue_paused: true, queued_inputs: [
    { turn_id: 'queued-turn-one', attempt_id: 'queued-attempt-one' },
    { turn_id: 'queued-turn-two', attempt_id: 'queued-attempt-two' },
  ] });
  await h.renderer.refresh(); await flush();
  assert.equal(h.element('input-queue-status').hidden, false);
  assert.match(h.element('input-queue-status').textContent, /等待继续处理/u);
  assert.equal(h.element('resume-inputs').hidden, false);
  await h.element('resume-inputs').onclick(); await flush();
  assert.equal(h.resumeInputsCalls, 1);
  h.setState({ input_queue_paused: false });
  await h.renderer.refresh(); await flush();
  assert.equal(h.element('resume-inputs').hidden, true);
  assert.equal(h.stopCalls, 0);
  assert.deepEqual(h.calls, []);
});

test('submitted text is frozen, duplicate submits stay blocked, and editing while awaiting is retained', async () => {
  const h = harness(); await h.renderer.refresh(); await flush();
  const gate = deferred();
  h.api.send = text => { h.calls.push(text); return gate.promise; };
  h.element('message').value = '提交的原文';
  await submit(h);
  assert.equal(h.renderer.local().sending, true);
  h.element('message').value = '等待期间的新编辑';
  await submit(h);
  assert.deepEqual(h.calls, ['提交的原文']);
  gate.resolve({ attempt_id: 'next-attempt', turn_id: 'next-turn' });
  await flush(); await flush();
  assert.equal(h.element('message').value, '等待期间的新编辑');
  assert.equal(h.element('send').disabled, false);
});

test('successful unchanged submitted snapshot clears once', async () => {
  const h = harness(); await h.renderer.refresh(); await flush();
  h.element('message').value = '同一份提交';
  await submit(h); await flush();
  assert.equal(h.element('message').value, '');
  assert.deepEqual(h.calls, ['同一份提交']);
});

for (const reason of ['Cancellation not yet confirmed', 'Unknown presentation receipt']) {
  test(`failed admission keeps user text and exposes ${reason}`, async () => {
    const h = harness({ busy: true }); await h.renderer.refresh(); await flush();
    h.api.send = async text => { h.calls.push(text); throw Error(reason); };
    h.element('message').value = '保留的新输入';
    await submit(h); await flush();
    assert.deepEqual(h.calls, ['保留的新输入']);
    assert.equal(h.element('message').value, '保留的新输入');
    assert.equal(h.element('error').textContent, reason);
    assert.equal(h.renderer.local().sending, false);
  });
}

test('explicit stop marks actual text attempt rather than old delivery scope', async () => {
  const h = harness({ busy: true }); await h.renderer.refresh(); await flush();
  await h.element('stop').onclick();
  assert.equal(h.renderer.local().cancelledDeliveryAttempt, 'current-text-attempt');
});

test('model-not-ready still blocks submission', async () => {
  const h = harness({ busy: true, stateName: 'loading_model' }); await h.renderer.refresh(); await flush();
  h.element('message').value = '还不能发送';
  await submit(h);
  assert.deepEqual(h.calls, []);
  assert.equal(h.element('message').value, '还不能发送');
});


const activeScope = { session_id: 'same-session', input_id: 'input', attempt_id: 'attempt',
  turn_id: 'current-turn', text_attempt_id: 'current-text-attempt' };
const approvedPayload = () => ({ scope: activeScope, fingerprint: 'synthetic-approved',
  display_text: '已批准全文🌿', sequence: 1, deadline_ms: Date.now() + 5000 });

for (const metadata of ['history', 'sessions']) {
  test(`approved presenter runs while unrelated ${metadata} is held`, async () => {
    const h = harness({ busy: true }); await h.renderer.refresh(); await flush();
    h.setState({ delivery_scope: activeScope,
      attempt: { attempt_id: 'current-text-attempt', turn_id: 'current-turn', status: 'approved' } });
    const gate = deferred(), base = h.api[metadata]; let held = false;
    h.api[metadata] = async (...args) => { held = true; await gate.promise; return base(...args); };
    const refresh = h.renderer.refresh(); await flush(); assert.equal(held, true);
    let dispatches = 0;
    h.api.presentApproved = async () => { dispatches++; await h.approved(approvedPayload());
      return { status: 'unknown', recoverable: [] }; };
    const presentation = h.renderer.pollApproved(); await flush();
    try { assert.equal(dispatches, 1); assert.equal(h.presentations, 1); }
    finally { gate.resolve(); await Promise.all([refresh, presentation]); }
  });
}

test('metadata cannot replace the frozen body until actual coverage finishes', async () => {
  const h = harness({ busy: true }); await h.renderer.refresh(); await flush(); h.setState({ delivery_scope: activeScope });
  const gate = deferred(); h.setCoverage(() => gate.promise);
  const visible = h.approved(approvedPayload()); await flush();
  const before = h.updates; await h.renderer.refresh(); await flush();
  try { assert.equal(h.updates, before); assert.equal(h.timing.snapshot().length, 0); }
  finally { gate.resolve(false); await visible; }
  await h.renderer.refresh(); await flush(); assert.ok(h.updates > before);
});

test('stop and changed scope remain observable during held native coverage', async () => {
  const h = harness({ busy: true }); await h.renderer.refresh(); await flush(); h.setState({ delivery_scope: activeScope });
  const gate = deferred(); let observe;
  h.setCoverage(options => { observe = options; return gate.promise; });
  const visible = h.approved(approvedPayload()); await flush();
  await h.element('stop').onclick(); assert.equal(observe.cancelled(), true);
  gate.resolve(false); assert.equal(await visible, false);
});


test('metadata held during continuous submission cannot hold composer admission', async () => {
  const h = harness({ busy: true }); await h.renderer.refresh(); await flush();
  const gate = deferred(), base = h.api.sessions;
  h.api.sessions = async () => { await gate.promise; return base(); };
  const old = h.renderer.refresh(); await flush();
  h.element('message').value = '新的更正'; await submit(h); await flush();
  try {
    assert.deepEqual(h.calls, ['新的更正']);
    assert.equal(h.renderer.local().sending, false);
    assert.equal(h.element('send').disabled, false);
  } finally { gate.resolve(); await old; }
});

test('delivery pulse remains single flight across two callers while coverage is held', async () => {
  const h = harness({ busy: true }); await h.renderer.refresh(); await flush(); h.setState({ delivery_scope: activeScope });
  const gate = deferred(); let calls = 0;
  h.api.presentApproved = async () => { calls++; await gate.promise; return { status: 'idle', waited: true }; };
  const one = h.renderer.pollApproved(), two = h.renderer.pollApproved(); await flush();
  assert.equal(calls, 1); gate.resolve(); await Promise.all([one, two]);
});

test('metadata refresh cannot schedule another delivery after a held pulse settles', async () => {
  const h = harness({ busy: true }); await h.renderer.refresh(); await flush();
  const gate = deferred(); let calls = 0;
  h.api.presentApproved = async () => { calls++; await gate.promise; return { status: 'idle' }; };
  const pulse = h.renderer.pollDelivery(); await flush();
  await h.renderer.refresh(); await h.renderer.refresh();
  assert.equal(calls, 1);
  gate.resolve(); await pulse; await flush();
  assert.equal(calls, 1);
  assert.equal(h.timers.at(-1).delay, 180);
});

test('only actual painted coverage signs timing and later metadata resumes', async () => {
  const h = harness({ busy: true }); await h.renderer.refresh(); await flush(); h.setState({ delivery_scope: activeScope });
  const gate = deferred(); h.setCoverage(() => gate.promise);
  const id = h.timing.begin('same-session'); h.timing.bind(id, { attempt_id: 'current-text-attempt', turn_id: 'current-turn' });
  const visible = h.approved(approvedPayload()); await flush();
  assert.equal(h.timing.snapshot().at(-1).units.length, 0);
  gate.resolve(true); assert.equal(await visible, true);
  assert.equal(h.timing.snapshot().at(-1).units.length, 1);
  assert.equal(h.timing.snapshot().at(-1).units[0].ack_observed_ms, null);
  const before = h.updates; await h.renderer.refresh(); await flush(); assert.ok(h.updates > before);
});


test('presentation failure releases the one flight and a later pulse can recover', async () => {
  const h = harness({ busy: true }); await h.renderer.refresh(); await flush(); h.setState({ delivery_scope: activeScope });
  let calls = 0; h.api.presentApproved = async () => {
    if (++calls === 1) throw Error('synthetic transport failure');
    return { status: 'idle', waited: true };
  };
  await assert.rejects(h.renderer.pollApproved(), /synthetic transport failure/);
  assert.equal((await h.renderer.pollApproved()).status, 'idle'); assert.equal(calls, 2);
});

test('native coverage failure and user reading yield do not become timing or renewed follow', async () => {
  const h = harness({ busy: true }); await h.renderer.refresh(); await flush(); h.setState({ delivery_scope: activeScope });
  const options = [];
  h.setCoverage(value => { options.push(value); value.onUserNavigation(); return false; });
  assert.equal(await h.approved(approvedPayload()), false);
  assert.equal(options[0].presentUnseen, true); assert.equal(h.renderer.local().manualReading, true);
  assert.equal(await h.approved({ ...approvedPayload(), sequence: 2 }), false);
  assert.equal(options[1].presentUnseen, false); assert.equal(h.timing.snapshot().length, 0);
  h.element('message').value = '明确的新输入'; await submit(h); await flush();
  assert.equal(h.renderer.local().manualReading, false);
});


test('an immediate empty bounded poll backs off instead of spinning without progress', async () => {
  const h = harness({ busy: true }); await h.renderer.refresh(); await flush();
  h.api.presentApproved = async () => ({ status: 'idle' });
  await h.renderer.pollDelivery();
  assert.equal(h.timers.at(-1).delay, 180);
});
