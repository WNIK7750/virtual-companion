import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { ReplyNotifications, NotificationNavigation } from '../src/reply-notifications.mjs';

const scope = { session_id: 'session', turn_id: 'turn' };
function setup(options = {}) {
  const shown = [], opened = [];
  const context = { quiet: false, focused: false, session_id: 'session' };
  const policy = new ReplyNotifications({
    supported: () => true, context: () => context,
    create: () => { const item = new EventEmitter(); item.show = () => shown.push(item);
      item.close = () => item.emit('close'); return item; },
    activate: async target => opened.push(target), ...options,
  });
  return { policy, shown, opened, context };
}

test('a logical reply notifies once across chunks, retries and journal restoration', async () => {
  const { policy, shown, opened } = setup();
  policy.receive(scope); policy.receive({ ...scope, sequence: 2 });
  assert.equal(shown.length, 1); assert.equal(opened.length, 0);
  shown[0].emit('click'); await Promise.resolve();
  assert.deepEqual(opened, [scope]);
  const restored = setup({ receivedScopes: [scope] });
  restored.policy.receive(scope); assert.equal(restored.shown.length, 0);
  policy.receive({ ...scope, turn_id: 'next' }); assert.equal(shown.length, 2);
});

test('quiet and focused receiving consume reminder eligibility without late replay', () => {
  const { policy, shown, context } = setup();
  context.quiet = true; policy.receive(scope); context.quiet = false; policy.receive(scope);
  context.focused = true; policy.receive({ ...scope, turn_id: 'focused' });
  context.focused = false; policy.receive({ ...scope, turn_id: 'focused' });
  assert.equal(shown.length, 0);
  context.focused = true;
  policy.receive({ ...scope, session_id: 'other' }); assert.equal(shown.length, 1);
  policy.dismiss(); assert.equal(policy.active.size, 0);
});

test('unsupported, native failure and activation failure never throw into receiving', async () => {
  const absent = setup({ supported: () => false });
  assert.doesNotThrow(() => absent.policy.receive(scope)); assert.equal(absent.shown.length, 0);
  const broken = setup({ create: () => { throw new Error('denied'); } });
  assert.doesNotThrow(() => broken.policy.receive(scope)); assert.equal(broken.policy.status, 'unavailable');
  const failed = setup({ activate: async () => { throw new Error('busy'); } });
  failed.policy.receive(scope); failed.shown[0].emit('failed', {}, 'denied');
  assert.equal(failed.policy.status, 'unavailable');
  failed.shown[0].emit('click'); await new Promise(resolve => setImmediate(resolve));
  assert.equal(failed.policy.status, 'activation_failed');
});

test('click routing waits for another session to finish and newer clicks replace the destination', async () => {
  const calls = [];
  const state = { session_id: 'current', busy: true };
  const navigation = new NotificationNavigation({ getState: async () => state,
    selectSession: async session_id => { calls.push(['select', session_id]); state.session_id = session_id; },
    showTarget: target => calls.push(['target', target]), focus: () => calls.push(['focus']) });
  await navigation.open(scope);
  assert.deepEqual(calls, [['focus']]);
  await navigation.open({ ...scope, turn_id: 'newer' });
  state.busy = false; await navigation.flush();
  assert.deepEqual(calls.slice(2), [['select', 'session'], ['target', { ...scope, turn_id: 'newer' }]]);
  assert.equal(navigation.pending, null);
  state.busy = true;
  await navigation.open(scope); // Same-session navigation can show the received prefix while busy.
  assert.deepEqual(calls.at(-1), ['target', scope]);
});
