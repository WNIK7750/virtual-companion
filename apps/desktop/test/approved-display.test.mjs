import test from 'node:test';
import assert from 'node:assert/strict';
import { approvedScopeMatches, visiblePartial, mergeApprovedMessages, canRetry } from '../src/approved-display.mjs';

const scope = { session_id: 's', input_id: 'i', attempt_id: 'a', turn_id: 't', text_attempt_id: 'ui-a' };
const state = { delivery_protocol: 'approved-prefix.v1', delivery_scope: scope, busy: true,
  session_id: 's', attempt: { attempt_id: 'ui-a', turn_id: 't' } };

test('unknown approved delivery hides retry while legacy retry remains available', () => {
  const failed = { busy: false, attempt: { status: 'failed' } };
  assert.equal(canRetry(failed), true);
  assert.equal(canRetry({ ...failed, retry_allowed: false }), false);
  assert.equal(canRetry({ ...failed, retry_allowed: true }), true);
  assert.equal(canRetry({ ...failed, busy: true }), false);
});

test('approved protocol never exposes raw partial even after failure', () => {
  assert.equal(visiblePartial({ ...state, partial: 'unreviewed', busy: false, attempt: { status: 'failed' } }), '');
});

test('renderer requires exact active scope and rejects cancelled or switched sessions', () => {
  assert.equal(approvedScopeMatches(scope, state, 's'), true);
  for (const key of Object.keys(scope)) assert.equal(approvedScopeMatches({ ...scope, [key]: 'other' }, state, 's'), false);
  assert.equal(approvedScopeMatches(scope, { ...state, busy: false }, 's'), false);
  assert.equal(approvedScopeMatches(scope, state, 'other'), false);
});

test('confirmed history replaces approved transient by turn identity without text matching', () => {
  const visible = { id: 'visible:a', turn_id: 't', role: 'assistant', content: 'answer' };
  assert.deepEqual(mergeApprovedMessages([], [visible]), [visible]);
  const history = [{ id: 'persisted', turn_id: 't', role: 'assistant', content: 'answer' }];
  assert.deepEqual(mergeApprovedMessages(history, [visible]), history);
  assert.equal(mergeApprovedMessages([{ id: 'other', turn_id: 'other', role: 'assistant', content: 'answer' }], [visible]).length, 2);
});
