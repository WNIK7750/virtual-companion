import test from 'node:test';
import assert from 'node:assert/strict';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { StreamReceiptJournal } from '../src/stream-receipts.mjs';
import { StreamDeliveryClient } from '../src/stream-delivery-client.mjs';
import { event } from './stream-fixture.mjs';
import { mkdtempForTest } from './fixture-lifecycle.mjs';

const scope = { session_id: 'session', input_id: 'input', attempt_id: 'attempt',
  turn_id: 'turn', text_attempt_id: 'ui-attempt' };

test('reliable receipt persists the approved text before any rendering and reopens unseen', async context => {
  const directory = await mkdtempForTest(context, join(tmpdir(), 'vc-received-'));
  const journal = new StreamReceiptJournal(directory);
  const first = journal.receive(event(), scope);
  assert.match(first.receipt, /^stream-received-/u);
  assert.equal(journal.lookup(event()).status, 'unseen');
  const reopened = new StreamReceiptJournal(directory);
  assert.deepEqual(reopened.pendingReceived('session').map(row => row.event), [event()]);
  assert.equal(reopened.receive(event(), scope).receipt, first.receipt);
  await reopened.presentReceived(reopened.pendingReceived('session')[0], async () => false);
  assert.equal(reopened.lookup(event()).status, 'unknown');
  const visible = await reopened.presentReceived(reopened.pendingReceived('session')[0], async () => true);
  assert.match(visible, /^stream-visible-/u);
  assert.deepEqual(reopened.pendingReceived('session'), []);
});

test('receiving never waits for an offscreen presentation and sends no visible ACK', { timeout: 2000 }, async () => {
  const calls = [];
  let release;
  const client = new StreamDeliveryClient({ call: async (action, payload) => {
    calls.push(action);
    if (action === 'streamPoll') return { event: event() };
    if (action === 'streamState') return { reconcile: [] };
    assert.equal(action, 'streamReceive');
    assert.equal(payload.host_receipt, 'stream-received-synthetic');
    return { status: 'received' };
  }, presenter: {
    receive: () => ({ receipt: 'stream-received-synthetic', event: event(), scope }),
    pendingReceived: () => [],
    reconcile: () => ({ status: 'unseen' }),
    present: () => new Promise(resolve => { release = resolve; }),
  } });
  const result = await client.pulse(async () => ({ reliable_text_receive: true,
    delivery_protocol: 'approved-prefix.v1', delivery_scope: scope, busy: true,
    session_id: 'session', attempt: { attempt_id: 'ui-attempt', turn_id: 'turn' } }));
  assert.equal(result.status, 'received');
  assert.ok(calls.includes('streamReceive'));
  assert.ok(!calls.includes('streamAck'));
  assert.equal(release, undefined);
});

test('optional reminder observes only confirmed receiving and its failure cannot hold the pump', async () => {
  const calls = [];
  const client = new StreamDeliveryClient({ call: async action => {
    calls.push(action);
    if (action === 'streamPoll') return { event: event() };
    if (action === 'streamState') return { reconcile: [] };
    if (action === 'streamReceive') return { status: 'received' };
    throw new Error('Unexpected visible acknowledgement');
  }, presenter: { receive: () => ({ receipt: 'stream-received-synthetic', event: event(), scope }),
    pendingReceived: () => [], reconcile: () => ({ status: 'unseen' }) },
  onReceived: observed => { assert.deepEqual(observed, scope); calls.push('notification'); return Promise.reject(new Error('denied')); } });
  const state = { reliable_text_receive: true, delivery_protocol: 'approved-prefix.v1',
    delivery_scope: scope, busy: true, session_id: 'session',
    attempt: { attempt_id: 'ui-attempt', turn_id: 'turn' } };
  assert.equal((await client.pulse(async () => state)).status, 'received');
  assert.ok(calls.indexOf('notification') > calls.indexOf('streamReceive'));
  assert.ok(!calls.includes('streamAck'));
});
