import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { StreamReceiptJournal } from '../src/stream-receipts.mjs';
import { StreamDeliveryClient } from '../src/stream-delivery-client.mjs';
import { event as streamEvent } from './stream-fixture.mjs';

const scope = { session_id: 'session', input_id: 'input', attempt_id: 'attempt', turn_id: 'turn', text_attempt_id: 'ui-attempt' };
const state = () => ({ delivery_protocol: 'approved-prefix.v1', delivery_scope: scope, busy: true,
  session_id: 'session', attempt: { attempt_id: 'ui-attempt', turn_id: 'turn' } });

test('default delivery is single flight and carries only the exact current scope', async () => {
  let release; let displays = 0; const calls = [];
  const client = new StreamDeliveryClient({ call: async (action, payload) => {
    calls.push(action);
    if (action === 'streamState') return { reconcile: [] };
    if (action === 'streamPoll') return { event: streamEvent() };
    assert.equal(payload.host_receipt, 'receipt'); return {};
  }, presenter: { present: async (_event, options) => {
    displays++; assert.deepEqual(options.scope, scope);
    await new Promise(resolve => { release = resolve; }); return 'receipt';
  } } });
  const first = client.pulse(state); const second = client.pulse(state);
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(displays, 1); release(); const [result] = await Promise.all([first, second]);
  assert.equal(result.receipts[0].fingerprint, streamEvent().fingerprint);
  assert.equal(result.receipts[0].host_receipt, 'receipt');
  assert.deepEqual(calls, ['streamState', 'streamPoll', 'streamAck']);
});

test('stale session or attempt dispatch never reaches the device journal', async () => {
  const client = new StreamDeliveryClient({ call: async action => action === 'streamState'
    ? { reconcile: [] } : { event: streamEvent() },
  presenter: { present: () => assert.fail('stale dispatch must not create intent') } });
  for (const field of ['session_id', 'input_id', 'attempt_id']) {
    const value = state(); value.delivery_scope = { ...scope, [field]: 'other' };
    assert.deepEqual(await client.pulse(() => value), { status: 'stale', recoverable: [] });
  }
});

test('lost ACK is reconciled without rendering a second time', async () => {
  let known = false; let renders = 0; let ackCalls = 0;
  const event = streamEvent();
  const client = new StreamDeliveryClient({ call: async action => {
    if (action === 'streamState') return { reconcile: known ? [event] : [] };
    if (action === 'streamPoll') return { event: ackCalls > 1 ? null : event };
    ackCalls++; if (ackCalls === 1) throw new Error('lost ACK'); return {};
  }, presenter: { present: async () => { known = true; renders++; return 'receipt'; },
    reconcile: () => ({ status: 'confirmed', receipt: 'receipt' }) } });
  await assert.rejects(client.pulse(state), /lost ACK/);
  const result = await client.pulse(state);
  assert.equal(result.receipts[0].fingerprint, event.fingerprint);
  assert.equal(renders, 1); assert.equal(ackCalls, 2);
});

test('ACK committed before its response is lost is retried with the same receipt without rendering', async () => {
  const event = streamEvent(); let committed = false; let renders = 0;
  const sent = [];
  const client = new StreamDeliveryClient({ call: async (action, payload) => {
    if (action === 'streamState') return { reconcile: [] }; // Confirmed rows are not returned here.
    if (action === 'streamPoll') return { event: committed ? null : event };
    sent.push(structuredClone(payload)); committed = true;
    if (sent.length === 1) throw new Error('ACK response lost after commit');
    return { status: 'confirmed' };
  }, presenter: { present: async () => { renders++; return 'same-durable-receipt'; } } });
  await assert.rejects(client.pulse(state), /ACK response lost after commit/);
  assert.equal(committed, true);
  const recovered = await client.pulse(() => assert.fail('pending ACK must precede unrelated state reads'));
  assert.deepEqual(recovered.receipts, [{ sequence: event.sequence, fingerprint: event.fingerprint,
    host_receipt: 'same-durable-receipt', basis: event.basis }]);
  assert.deepEqual(sent[1], sent[0]);
  assert.equal(renders, 1);
  const completed = { ...state(), busy: false };
  assert.equal((await client.pulse(() => completed)).status, 'stale');
  assert.equal(sent.length, 2, 'successful retry must clear pending transport evidence');
});

test('repeated ACK response loss remains pending and completed state cannot fabricate success', async () => {
  const event = streamEvent(); let calls = 0;
  const client = new StreamDeliveryClient({ call: async action => {
    if (action === 'streamState') return { reconcile: [event] };
    if (action === 'streamAck') { calls++; throw new Error('ACK response unavailable'); }
    assert.fail('unknown confirmation must not render again');
  }, presenter: { reconcile: () => ({ status: 'confirmed', receipt: 'durable' }) } });
  await assert.rejects(client.pulse(state), /ACK response unavailable/);
  await assert.rejects(client.pulse(() => assert.fail('retry must not infer ACK from completed state')),
    /ACK response unavailable/);
  assert.equal(calls, 2);
});

test('reconciliation acknowledges only durable receipts and never renders unknown or unseen units', async () => {
  const acknowledged = [];
  const events = ['confirmed', 'unknown', 'unseen'].map((status, i) => ({
    sequence: i + 1, fingerprint: String(i), status,
  }));
  const client = new StreamDeliveryClient({
    call: async (action, payload) => {
      if (action === 'streamState') return { reconcile: events };
      assert.equal(action, 'streamAck'); acknowledged.push(payload); return { status: 'confirmed' };
    },
    presenter: {
      present: () => assert.fail('reconciliation must never display'),
      reconcile: event => ({ status: event.status, receipt: 'durable-host-receipt' }),
    },
  });
  assert.deepEqual(await client.reconcile(), { confirmed: 1, receipts: [{
    sequence: 1, fingerprint: '0', host_receipt: 'durable-host-receipt', basis: undefined,
  }], recoverable: [{ fingerprint: '1' }, { fingerprint: '2' }] });
  assert.deepEqual(acknowledged, [{ sequence: 1, fingerprint: '0', host_receipt: 'durable-host-receipt' }]);
});

test('poll with no offer or unknown device display never acknowledges', async () => {
  let offered = null;
  const client = new StreamDeliveryClient({
    call: async action => { assert.equal(action, 'streamPoll'); return { event: offered }; },
    presenter: { present: async () => null },
  });
  assert.deepEqual(await client.pollOnce(), { status: 'idle' });
  offered = { sequence: 1 };
  assert.deepEqual(await client.pollOnce(), { status: 'unknown' });
});

test('a recovered ACK is returned before a later failed poll can discard its timing evidence', async () => {
  const event = streamEvent(); let acked = false; let polls = 0;
  const client = new StreamDeliveryClient({
    call: async action => {
      if (action === 'streamState') return { reconcile: acked ? [] : [event] };
      if (action === 'streamAck') { acked = true; return { status: 'confirmed' }; }
      polls++; throw new Error('poll transport failure after durable ACK');
    },
    presenter: { reconcile: () => ({ status: 'confirmed', receipt: 'durable' }) },
  });
  const result = await client.pulse(state);
  assert.equal(result.receipts[0].host_receipt, 'durable');
  assert.equal(polls, 0);
});

test('partial reconciliation reports committed receipts even if a later ACK fails', async () => {
  const first = streamEvent(), second = { ...streamEvent(), sequence: 2, fingerprint: 'other' };
  const client = new StreamDeliveryClient({
    call: async (action, payload) => {
      if (action === 'streamState') return { reconcile: [first, second] };
      if (payload.sequence === 2) throw new Error('second ACK unavailable');
      return { status: 'confirmed' };
    },
    presenter: { reconcile: () => ({ status: 'confirmed', receipt: 'durable' }) },
  });
  const result = await client.pulse(state);
  assert.equal(result.status, 'partial');
  assert.equal(result.receipts.length, 1);
  assert.equal(result.receipts[0].fingerprint, first.fingerprint);
  assert.equal(result.error, 'reconciliation_incomplete');
});


test('explicit recovery fetches original server event, never accepts renderer supplied text', async () => {
  const event = streamEvent(); const calls = [];
  const client = new StreamDeliveryClient({ call: async (action, payload) => {
    calls.push(action);
    if (action === 'streamState') return { reconcile: [event] };
    assert.equal(payload.fingerprint, event.fingerprint); return { status: 'confirmed' };
  }, presenter: { resume: async value => { assert.deepEqual(value, event); return 'original-receipt'; } } });
  const result = await client.resume(event.fingerprint, () => ({ ...state(), busy: false }));
  assert.equal(result.status, 'confirmed'); assert.deepEqual(calls, ['streamState', 'streamAck']);
  await assert.rejects(client.resume(event.fingerprint, state), /running/);
  await assert.rejects(client.resume('0'.repeat(64), () => ({ ...state(), busy: false })), /unknown/);
});

test('an active recovery blocks duplicate recovery and ordinary redispatch', async () => {
  const event = streamEvent(); let release;
  const client = new StreamDeliveryClient({ call: async action => action === 'streamState' ? { reconcile: [event] } : {},
    presenter: { resume: () => new Promise(resolve => { release = resolve; }) } });
  const idle = () => ({ ...state(), busy: false });
  const first = client.resume(event.fingerprint, idle);
  await new Promise(resolve => setImmediate(resolve));
  await assert.rejects(client.resume(event.fingerprint, idle), /progress/);
  assert.equal((await client.pulse(idle)).status, 'recovering');
  release(null); assert.equal((await first).status, 'unknown');
});


test('closing recovery during its state read cannot open a late modal', async () => {
  const event = streamEvent(); let release;
  const client = new StreamDeliveryClient({ call: async () => ({ reconcile: [event] }),
    presenter: { resume: () => assert.fail('closed request must not open'), cancelRecovery() {} } });
  const recovering = client.resume(event.fingerprint, () => new Promise(resolve => { release = resolve; }));
  await new Promise(resolve => setImmediate(resolve)); client.cancelRecovery();
  release({ ...state(), busy: false });
  assert.equal((await recovering).status, 'unknown');
});


test('bounded current poll keeps one flight and waits before presenter/ACK', async () => {
  let release; const calls = []; let displays = 0;
  const client = new StreamDeliveryClient({ call: async (action, payload) => {
    calls.push({ action, payload });
    if (action === 'streamState') return { reconcile: [] };
    if (action === 'streamPoll') { await new Promise(resolve => { release = resolve; }); return { event: streamEvent() }; }
    return { status: 'confirmed' };
  }, presenter: { present: async () => { displays++; return 'visible-receipt'; } } });
  const first = client.pulse(state, { waitMs: 1000 });
  const duplicate = client.pulse(state, { waitMs: 1000 });
  await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(calls.at(-1), { action: 'streamPoll', payload: { wait_ms: 1000 } });
  assert.equal(displays, 0); assert.equal(calls.some(row => row.action === 'streamAck'), false);
  release(); const [result] = await Promise.all([first, duplicate]);
  assert.equal(result.status, 'confirmed'); assert.equal(displays, 1);
  assert.equal(calls.filter(row => row.action === 'streamAck').length, 1);
});

for (const change of ['cancelled', 'session', 'attempt']) {
  test(`bounded offer rereads actual state before intent after ${change}`, async () => {
    let current = state(), release;
    const client = new StreamDeliveryClient({ call: async action => {
      if (action === 'streamState') return { reconcile: [] };
      await new Promise(resolve => { release = resolve; }); return { event: streamEvent() };
    }, presenter: { present: () => assert.fail('stale offer must not dispatch or ACK') } });
    const pulse = client.pulse(() => current, { waitMs: 1000 });
    await new Promise(resolve => setImmediate(resolve));
    current = change === 'cancelled' ? { ...current, busy: false }
      : change === 'session' ? { ...current, session_id: 'other' }
        : { ...current, attempt: { attempt_id: 'new', turn_id: 'new' } };
    release(); assert.equal((await pulse).status, 'stale');
  });
}

test('legacy pulse keeps the empty immediate poll payload', async () => {
  const client = new StreamDeliveryClient({ call: async (action, payload) => {
    if (action === 'streamState') return { reconcile: [] };
    assert.deepEqual(payload, {}); return { event: null };
  }, presenter: {} });
  assert.equal((await client.pulse(state)).status, 'idle');
});


test('a repeated failed offer uses the same intent and cannot re-render or ACK', async t => {
  const dir = mkdtempSync(join(tmpdir(), 'delivery-scheduling-'));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  const journal = new StreamReceiptJournal(dir), event = streamEvent();
  let renders = 0, acks = 0;
  const client = new StreamDeliveryClient({ call: async action => {
    if (action === 'streamState') return { reconcile: [event] };
    if (action === 'streamPoll') return { event };
    acks++; return {};
  }, presenter: {
    reconcile: value => journal.lookup(value),
    present: value => journal.present(value, async () => { renders++; return false; }),
  } });
  assert.equal((await client.pulse(state, { waitMs: 1000 })).status, 'unknown');
  assert.equal((await client.pulse(state, { waitMs: 1000 })).status, 'unknown');
  assert.equal(renders, 1); assert.equal(acks, 0); assert.equal(journal.lookup(event).status, 'unknown');
});
