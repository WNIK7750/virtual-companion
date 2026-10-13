import test from 'node:test';
import assert from 'node:assert/strict';
import { DeliveryTiming } from '../src/delivery-timing.mjs';

const scope = { session_id: 's', input_id: 'i', attempt_id: 'a', turn_id: 't', text_attempt_id: 'ui-a' };
const unit = { scope, sequence: 1, fingerprint: 'f', display_text: '已批准的正文。' };
const ack = { sequence: 1, fingerprint: 'f', host_receipt: 'durable-receipt',
  basis: { session_id: 's', input_id: 'i', attempt_id: 'a' } };
function setup() {
  let now = 10;
  const timer = new DeliveryTiming(() => now);
  const id = timer.begin('s');
  timer.bind(id, { attempt_id: 'ui-a', turn_id: 't' });
  return { timer, id, at: n => { now = n; } };
}

test('renderer send t0, visible unit, durable ACK and terminal have separate times', () => {
  const { timer, at } = setup();
  at(60); timer.visible(unit);
  at(90); timer.acknowledged(ack);
  at(110); timer.observe({ session_id: 's', busy: false,
    attempt: { attempt_id: 'ui-a', turn_id: 't', status: 'completed' } });
  const [row] = timer.snapshot();
  assert.equal(row.t0_ms, 10);
  assert.equal(row.units[0].visible_ms, 60);
  assert.equal(row.units[0].ack_observed_ms, 90);
  assert.equal(row.terminal_ms, 110);
  assert.equal(row.outcome, 'completed');
  assert.equal(row.units[0].host_receipt, 'durable-receipt');
  assert.equal(JSON.stringify(row).includes(unit.display_text), false);
});

test('renderer records an epoch ACK beside the backend generation terminal for true-stream proof', () => {
  let now = 10;
  let epochUs = 1_000_000;
  const timer = new DeliveryTiming(() => now, () => epochUs);
  const id = timer.begin('s');
  timer.bind(id, { attempt_id: 'ui-a', turn_id: 't' });
  now = 60; epochUs = 1_000_050; timer.visible(unit);
  now = 90; epochUs = 1_000_080; timer.acknowledged(ack);
  now = 110; epochUs = 1_000_100;
  timer.observe({ session_id: 's', busy: false, generation_terminal_us: 1_000_120,
    attempt: { attempt_id: 'ui-a', turn_id: 't', status: 'completed' } });

  const [row] = timer.snapshot();
  assert.equal(row.t0_epoch_us, 1_000_000);
  assert.equal(row.units[0].visible_epoch_us, 1_000_050);
  assert.equal(row.units[0].ack_observed_epoch_us, 1_000_080);
  assert.equal(row.generation_terminal_us, 1_000_120);
  assert.ok(row.units[0].ack_observed_epoch_us < row.generation_terminal_us);
});

test('foreign attempt, empty candidate and ACK-only records cannot create visible samples', () => {
  const { timer } = setup();
  timer.visible({ ...unit, scope: { ...scope, text_attempt_id: 'other' } });
  timer.visible({ ...unit, display_text: '  ' });
  timer.acknowledged(ack);
  assert.deepEqual(timer.snapshot()[0].units, []);
});

test('visible without durable ACK stays unconfirmed; later reconciliation retains original time', () => {
  const { timer, at } = setup();
  at(30); timer.visible(unit);
  at(50); timer.visible(unit);
  assert.equal(timer.snapshot()[0].units[0].ack_observed_ms, null);
  at(90); timer.acknowledged({ ...ack, basis: { ...ack.basis, input_id: 'other' } });
  assert.equal(timer.snapshot()[0].units[0].ack_observed_ms, null);
  at(120); timer.acknowledged(ack);
  at(140); timer.acknowledged(ack);
  assert.equal(timer.snapshot()[0].units[0].visible_ms, 30);
  assert.equal(timer.snapshot()[0].units[0].ack_observed_ms, 120);
});

test('reopen history and unbound or rejected requests are never synthesized as fast successes', () => {
  const { timer } = setup();
  const reopened = new DeliveryTiming(() => 0);
  reopened.acknowledged(ack); reopened.visible(unit);
  assert.deepEqual(reopened.snapshot(), []);
  const rejected = timer.begin('s'); timer.reject(rejected);
  const row = timer.snapshot()[1];
  assert.equal(row.outcome, 'send_rejected');
  assert.equal(row.units.length, 0);
});

test('cancelled turn retains confirmed prefix and retry has independent attempt and t0', () => {
  const { timer, at } = setup();
  at(30); timer.visible(unit); timer.acknowledged(ack);
  at(40); timer.observe({ session_id: 's', busy: false,
    attempt: { attempt_id: 'ui-a', turn_id: 't', status: 'cancelled' } });
  at(60); const retry = timer.begin('s', 'retry');
  timer.bind(retry, { attempt_id: 'ui-b', turn_id: 't' });
  timer.visible(unit);
  assert.equal(timer.snapshot()[0].outcome, 'cancelled');
  assert.equal(timer.snapshot()[0].units.length, 1);
  assert.equal(timer.snapshot()[1].t0_ms, 60);
  assert.equal(timer.snapshot()[1].units.length, 0);
});

test('bounded local diagnostics return detached data and reject reversed timestamps', () => {
  const { timer, at } = setup();
  at(9); timer.visible(unit);
  assert.equal(timer.snapshot()[0].units.length, 0);
  const rows = timer.snapshot(); rows[0].outcome = 'fake';
  assert.equal(timer.snapshot()[0].outcome, 'pending');
  for (let n = 0; n < 110; n++) timer.begin('s');
  assert.equal(timer.snapshot().length, 100);
});
