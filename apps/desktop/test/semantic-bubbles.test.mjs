import test from 'node:test';
import assert from 'node:assert/strict';
import { digest, validateStreamEvent } from '../src/stream-receipts.mjs';
import { spanned } from './stream-fixture.mjs';
import { feedbackTargetFor, feedbackTargetFromSelection, MessageList, projectBubbles } from '../src/message-list.mjs';

function fingerprint(value) {
  const { fingerprint, status, host_receipt, text, ...manifest } = value;
  value.fingerprint = digest(manifest);
  return value;
}

test('bubble manifest binds exact source paragraphs in code points, including astral text', () => {
  const source = '第一段🌿。\n\n第二段。';
  const event = fingerprint({ ...spanned({ source, cuts: [Array.from(source).length] }),
    bubbles: [{ start: 0, end: 7 }, { start: 7, end: 11 }] });
  assert.equal(validateStreamEvent(event).fingerprint, event.fingerprint);
  for (const ranges of [
    [{ start: 0, end: 8 }, { start: 8, end: 11 }],
    [{ start: 0, end: 11 }],
    [{ start: 0, end: 7 }, { start: 6, end: 11 }],
  ]) {
    assert.throws(() => validateStreamEvent(fingerprint({ ...event, bubbles: ranges })), /bubble ranges/u);
  }
  assert.throws(() => validateStreamEvent({ ...event, bubbles: [{ start: 0, end: 11 }] }), /fingerprint/u);
  assert.doesNotThrow(() => validateStreamEvent(spanned()));
});

test('separate bubbles retain the source and interleave with later user messages', () => {
  const source = '先说这一段。\n\n后说这一段🌿。';
  const firstEnd = Array.from('先说这一段。\n\n').length;
  const messages = [
    { id: 'u1', role: 'user', content: '第一条输入', timeline_order: [1, 0] },
    { id: 'a1', turn_id: 'turn1', role: 'assistant', content: source,
      bubble_ranges: [{ start: 0, end: firstEnd }, { start: firstEnd, end: Array.from(source).length }],
      bubble_orders: [{ start: 0, order: [1, 1] }, { start: firstEnd, order: [3, 2] }] },
    { id: 'u2', role: 'user', content: '在两段之间插话', timeline_order: [3, 0] },
  ];
  const visual = projectBubbles(messages);
  assert.deepEqual(visual.map(item => item.role), ['user', 'assistant', 'user', 'assistant']);
  assert.equal(visual.filter(item => item.role === 'assistant').map(item => item.content).join(''), source);
  assert.equal(visual[1].original_message, messages[1]);
  assert.equal(visual[3].final_bubble, true);
  assert.deepEqual(feedbackTargetFor(visual[1]), { kind: 'bubble', start: 0, end: firstEnd });
  assert.deepEqual(feedbackTargetFor(visual[3]), {
    kind: 'bubble', start: firstEnd, end: Array.from(source).length,
  });
  assert.deepEqual(feedbackTargetFor(messages[0]), null);
});

test('attempt feedback target carries an exact ID and no hidden text range', () => {
  const user = { id: 'u', turn_id: 'turn', role: 'user', content: '原始输入' };
  assert.deepEqual(feedbackTargetFor(user, { attempt: {
    turn_id: 'turn', attempt_id: 'attempt-1', status: 'received',
  } }), { kind: 'attempt', attempt_id: 'attempt-1' });
  assert.equal(feedbackTargetFor(user, { attempt: {
    turn_id: 'turn', attempt_id: 'attempt-2', status: 'completed',
  } }), null);
});

test('selected bubble feedback uses Unicode code-point offsets in the original reply', () => {
  const original = { id: 'reply', role: 'assistant', content: 'prefixA🌿Btail' };
  const message = { ...original, id: 'reply:bubble:6', original_message: original,
    content: 'A🌿B', bubble_start: 6, bubble_end: 9 };
  const text = {};
  const body = { contains: node => node === text };
  const raw = 'A🌿B', start = 1, end = 3; // UTF-16 offsets select the single astral symbol.
  const range = {
    startContainer: text, startOffset: start, endContainer: text, endOffset: end,
    toString: () => raw.slice(start, end),
    cloneRange() {
      let prefixEnd = 0;
      return { selectNodeContents() {}, setEnd(_node, offset) { prefixEnd = offset; },
        toString: () => raw.slice(0, prefixEnd) };
    },
  };
  assert.deepEqual(feedbackTargetFromSelection(message, body, range), {
    message: original, target: { kind: 'bubble', start: 7, end: 8 },
  });
  assert.equal(feedbackTargetFromSelection(message, { contains: () => false }, range), null);
  assert.equal(feedbackTargetFromSelection({ ...message, approvedOnly: true }, body, range), null);
});

test('bubble feedback click retains a selected range when mouse focus clears selection', () => {
  const prior = globalThis.document;
  const text = {};
  const created = [];
  class Element {
    constructor(tag) { this.tag = tag; this.children = []; this.attributes = {}; created.push(this); }
    setAttribute(key, value) { this.attributes[key] = value; }
    append(...items) { this.children.push(...items); }
    get lastChild() { return this.children.at(-1); }
  }
  globalThis.document = { createElement: tag => new Element(tag), createTextNode: value => ({ textContent: value }),
    getSelection: () => selection };
  const raw = 'A🌿B';
  let currentRange = {
    startContainer: text, startOffset: 1, endContainer: text, endOffset: 3,
    toString: () => '🌿',
    cloneRange() {
      let end = 0;
      return { selectNodeContents() {}, setEnd(_node, value) { end = value; }, toString: () => raw.slice(0, end) };
    },
  };
  let selection = { rangeCount: 1, isCollapsed: false, getRangeAt: () => currentRange };
  const source = { id: 'reply', role: 'assistant', turn_id: 'turn', content: 'prefixA🌿Btail' };
  const message = { ...source, id: 'reply:bubble:6', original_message: source, content: raw,
    bubble_start: 6, bubble_end: 9, bubble_ranges: [{ start: 0, end: 3 }] };
  const body = { contains: node => node === text };
  const node = { message, body, element: { append() {} } };
  let received;
  try {
    const list = new MessageList({}, (actualMessage, target) => { received = { actualMessage, target }; });
    list.updateFeedback(node, message, {});
    node.feedback.onmousedown();
    currentRange = null;
    selection = { rangeCount: 0, isCollapsed: true };
    node.feedback.onclick();
    assert.deepEqual(received, { actualMessage: source, target: { kind: 'bubble', start: 7, end: 8 } });
    assert.match(node.feedback.title, /选中本条消息/u);
  } finally {
    if (prior === undefined) delete globalThis.document;
    else globalThis.document = prior;
  }
});
