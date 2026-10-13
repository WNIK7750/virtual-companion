import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { approvedTextIsVisible, isReadingNavigation, waitForApprovedText } from '../src/approved-visibility.mjs';

function setup({ top = 140, bottom = 170, visibility = 'visible' } = {}) {
  const node = { nodeType: 3, textContent: 'Old prefix.New approved text.' };
  const body = { textContent: node.textContent, firstChild: node };
  const calls = [];
  const document = { visibilityState: visibility, createRange: () => ({
    setStart: (target, offset) => calls.push(['start', target, offset]),
    setEnd: (target, offset) => calls.push(['end', target, offset]),
    getClientRects: () => [{ top, bottom, left: 80, right: 300 }],
  }) };
  const view = { innerWidth: 800, innerHeight: 600 };
  const container = { isConnected: true, clientTop: 2, clientLeft: 2,
    clientWidth: 500, clientHeight: 300,
    getBoundingClientRect: () => ({ top: 100, left: 50, bottom: 404, right: 554 }) };
  const element = { isConnected: true, getClientRects: () => [{ top: 120, bottom: 900 }],
    querySelector: () => body, checkVisibility: () => true };
  const payload = { display_text: node.textContent, text: 'New approved text.' };
  return { element, container, payload, document, view, body, node, calls };
}

test('approved new text must intersect the actual scrolling content viewport', () => {
  assert.equal(approvedTextIsVisible(setup({ top: 500, bottom: 540 })), false);
  assert.equal(approvedTextIsVisible(setup({ top: 20, bottom: 60 })), false);
  assert.equal(approvedTextIsVisible(setup()), true);
  assert.equal(approvedTextIsVisible(setup({ top: 390, bottom: 430 })), false);
  assert.equal(approvedTextIsVisible(setup({ top: 401, bottom: 450 })), false);
});

test('approved range may span exact text nodes in separate semantic bubbles', () => {
  const sample = setup();
  const first = { nodeType: 3, textContent: 'Old prefix.' };
  const second = { nodeType: 3, textContent: 'New approved text.' };
  sample.body.firstChild = { nodeType: 1, childNodes: [first] };
  sample.body.childNodes = [sample.body.firstChild, { nodeType: 1, childNodes: [second] }];
  assert.equal(approvedTextIsVisible(sample), true);
  assert.deepEqual(sample.calls.at(-1), ['end', second, second.textContent.length]);
  sample.body.textContent = 'changed';
  assert.equal(approvedTextIsVisible(sample), false);
});

test('offscreen text may be confirmed after user scroll, without moving scroll position', async () => {
  const sample = setup({ top: 500, bottom: 540 });
  const events = new EventTarget();
  const frames = new Map(); let frameId = 0;
  for (const target of [sample.document, sample.container, sample.view]) {
    target.addEventListener = events.addEventListener.bind(events);
    target.removeEventListener = events.removeEventListener.bind(events);
  }
  sample.view.requestAnimationFrame = callback => {
    const id = ++frameId;
    frames.set(id, setTimeout(callback, 0));
    return id;
  };
  sample.view.cancelAnimationFrame = id => clearTimeout(frames.get(id));
  sample.container.scrollTop = 50;
  const waiting = waitForApprovedText({ ...sample, deadlineMs: Date.now() + 1000 });
  await new Promise(resolve => setTimeout(resolve, 10));
  assert.equal(sample.container.scrollTop, 50);
  sample.document.createRange = setup().document.createRange;
  events.dispatchEvent(new Event('scroll'));
  assert.equal(await waiting, true);
  assert.equal(sample.container.scrollTop, 50);
});

test('hidden or never fully visible text times out even if animation frames stop', async () => {
  const sample = setup({ visibility: 'hidden' });
  for (const target of [sample.document, sample.container, sample.view]) {
    target.addEventListener = () => {};
    target.removeEventListener = () => {};
  }
  sample.view.requestAnimationFrame = () => 1;
  sample.view.cancelAnimationFrame = () => {};
  assert.equal(await waitForApprovedText({ ...sample, deadlineMs: Date.now() + 10 }), false);
});

test('layout outside the window and hidden document cannot acknowledge', () => {
  const sample = setup(); sample.view.innerHeight = 90;
  assert.equal(approvedTextIsVisible(sample), false);
  assert.equal(approvedTextIsVisible(setup({ visibility: 'hidden' })), false);
});

test('old visible prefix is not evidence that the appended unit was visible', () => {
  const sample = setup({ top: 500, bottom: 540 });
  assert.equal(approvedTextIsVisible(sample), false);
  assert.deepEqual(sample.calls, [
    ['start', sample.node, 'Old prefix.'.length],
    ['end', sample.node, sample.payload.display_text.length],
  ]);
});

test('stale, disconnected, CSS-hidden or whitespace-only content cannot acknowledge', () => {
  const stale = setup(); stale.body.textContent = 'some other content';
  assert.equal(approvedTextIsVisible(stale), false);
  const disconnected = setup(); disconnected.element.isConnected = false;
  assert.equal(approvedTextIsVisible(disconnected), false);
  const hidden = setup(); hidden.element.checkVisibility = () => false;
  assert.equal(approvedTextIsVisible(hidden), false);
  const blank = setup(); blank.payload.text = ' ';
  assert.equal(approvedTextIsVisible(blank), false);
  const hiddenBody = setup(); hiddenBody.body.checkVisibility = () => false;
  assert.equal(approvedTextIsVisible(hiddenBody), false);
  const covered = setup(); covered.document.elementFromPoint = () => ({ tagName: 'DIALOG' });
  assert.equal(approvedTextIsVisible(covered), false);
  const uncovered = setup(); uncovered.document.elementFromPoint = () => uncovered.body;
  assert.equal(approvedTextIsVisible(uncovered), true);
});

function scrollingText(text = 'A\nB\nC') {
  const sample = setup();
  sample.node.textContent = text;
  sample.body.textContent = text;
  sample.payload = { text, display_text: text };
  sample.container.clientTop = sample.container.clientLeft = 0;
  sample.container.clientWidth = 100; sample.container.clientHeight = 20;
  sample.container.scrollTop = 0;
  sample.container.getBoundingClientRect = () => ({ top: 0, left: 0 });
  sample.blocked = new Set();
  const points = [];
  let offset = 0, row = 0, column = 0;
  for (const char of text) {
    points.push({ char, start: offset, end: offset + char.length, row, column });
    offset += char.length;
    if (char === '\n') { row += 1; column = 0; } else column += 1;
  }
  const boundaries = new Set([0, ...points.map(point => point.end)]);
  sample.probes = [];
  sample.document.createRange = () => {
    let start = 0, end = 0;
    return {
      setStart(target, value) {
        assert.equal(target, sample.body.firstChild); assert.ok(boundaries.has(value)); start = value;
      },
      setEnd(target, value) {
        assert.equal(target, sample.body.firstChild); assert.ok(boundaries.has(value)); end = value;
      },
      getClientRects() {
        sample.probes.push([start, end]);
        return points.filter(point => point.start >= start && point.end <= end).map(point => ({
          left: point.column * 10, right: point.column * 10 + (/^[\n\u200d\u0301]$/u.test(point.char) ? 0 : 10),
          top: point.row * 20 - sample.container.scrollTop,
          bottom: point.row * 20 - sample.container.scrollTop + 20,
        }));
      },
    };
  };
  sample.document.elementFromPoint = (x, y) => {
    const point = points.find(item => item.column === Math.floor(x / 10)
      && item.row === Math.floor((y + sample.container.scrollTop) / 20));
    return sample.blocked.has(point?.start) ? { tagName: 'DIALOG' } : sample.body;
  };
  for (const target of [sample.document, sample.container, sample.view]) {
    const events = new EventTarget();
    target.addEventListener = events.addEventListener.bind(events);
    target.removeEventListener = events.removeEventListener.bind(events);
    target.dispatchEvent = events.dispatchEvent.bind(events);
  }
  const frames = new Map(); let frameId = 0;
  sample.view.requestAnimationFrame = callback => { frames.set(++frameId, callback); return frameId; };
  sample.view.cancelAnimationFrame = id => frames.delete(id);
  sample.paint = () => {
    for (let frame = 0; frame < 2; frame += 1) {
      const batch = [...frames.values()]; frames.clear();
      for (const callback of batch) callback();
    }
  };
  sample.scroll = row => {
    sample.container.scrollTop = row * 20;
    sample.container.dispatchEvent(new Event('scroll')); sample.paint();
  };
  return sample;
}

test('a long suffix is confirmed only after every character has been visible across user scrolls', async () => {
  const sample = scrollingText();
  assert.equal(approvedTextIsVisible(sample), false);
  const waiting = waitForApprovedText({ ...sample, deadlineMs: Date.now() + 1000 });
  sample.paint(); sample.scroll(1); sample.scroll(2);
  assert.equal(await waiting, true);
  assert.equal(sample.container.scrollTop, 40);
  assert.equal(approvedTextIsVisible(sample), false);
});

test('jumping past an unobserved middle line leaves the complete unit unknown', async () => {
  const sample = scrollingText();
  const waiting = waitForApprovedText({ ...sample, deadlineMs: Date.now() + 20 });
  sample.paint(); sample.scroll(2);
  assert.equal(await waiting, false);
});

test('hidden and covered middle characters do not accumulate coverage', async () => {
  for (const kind of ['document', 'element', 'body', 'overlay']) {
    const sample = scrollingText('A\nBC\nD');
    const waiting = waitForApprovedText({ ...sample, deadlineMs: Date.now() + 20 });
    sample.paint();
    if (kind === 'document') sample.document.visibilityState = 'hidden';
    if (kind === 'element') sample.element.checkVisibility = () => false;
    if (kind === 'body') sample.body.checkVisibility = () => false;
    if (kind === 'overlay') sample.blocked.add(3); // Only C is occluded, not the line midpoint.
    sample.scroll(1);
    sample.document.visibilityState = 'visible';
    sample.element.checkVisibility = sample.body.checkVisibility = () => true;
    sample.blocked.clear(); sample.scroll(2);
    assert.equal(await waiting, false, kind);
  }
});

test('an occluded character can be confirmed after it actually becomes visible', async () => {
  const sample = scrollingText(); sample.blocked.add(2);
  const waiting = waitForApprovedText({ ...sample, deadlineMs: Date.now() + 1000 });
  sample.paint(); sample.scroll(1); sample.scroll(2);
  sample.blocked.clear(); sample.scroll(1);
  assert.equal(await waiting, true);
});

test('changing the bound text or replacing its node cannot combine observations', async () => {
  for (const kind of ['text', 'node', 'payload', 'disconnect']) {
    const sample = scrollingText();
    const waiting = waitForApprovedText({ ...sample, deadlineMs: Date.now() + 20 });
    sample.paint();
    if (kind === 'text') sample.body.textContent = 'A\nX\nC';
    if (kind === 'node') sample.body.firstChild = { ...sample.node };
    if (kind === 'payload') sample.payload.text = 'C';
    if (kind === 'disconnect') sample.element.isConnected = false;
    sample.scroll(1);
    sample.body.textContent = sample.payload.text = 'A\nB\nC';
    sample.body.firstChild = sample.node; sample.element.isConnected = true;
    sample.scroll(1); sample.scroll(2);
    assert.equal(await waiting, false, kind);
  }
});

test('cancellation prevents later coverage and settles even without another scroll', async () => {
  const sample = scrollingText(); let cancelled = false;
  const waiting = waitForApprovedText({ ...sample, cancelled: () => cancelled, deadlineMs: Date.now() + 1000 });
  sample.paint(); const probes = sample.probes.length; cancelled = true;
  sample.scroll(1); sample.scroll(2);
  assert.equal(await waiting, false);
  assert.equal(sample.probes.length, probes);
  const idle = scrollingText();
  const waitingIdle = waitForApprovedText({ ...idle, cancelled: () => cancelled, deadlineMs: Date.now() + 1000 });
  assert.equal(await waitingIdle, false);
});

test('unicode code points and unpainted newlines stay whole while accumulating', async () => {
  const sample = scrollingText('\n甲🙂\n\n乙\n');
  const waiting = waitForApprovedText({ ...sample, deadlineMs: Date.now() + 1000 });
  sample.paint(); sample.scroll(1); sample.scroll(3);
  assert.equal(await waiting, true);
  assert.ok(sample.probes.some(([start, end]) => start === 2 && end === 4));
});

test('combined emoji and combining marks use their complete painted grapheme', async () => {
  const sample = scrollingText('A\n👩‍💻e\u0301\nB');
  const waiting = waitForApprovedText({ ...sample, deadlineMs: Date.now() + 1000 });
  sample.paint(); sample.scroll(1); sample.scroll(2);
  assert.equal(await waiting, true);
  assert.ok(sample.probes.some(([start, end]) => start === 2 && end === 7));
  assert.ok(sample.probes.some(([start, end]) => start === 7 && end === 9));
});

test('restoring changed text before the next paint still invalidates prior coverage', async () => {
  const sample = scrollingText(); const records = [];
  let disconnected = false;
  sample.view.MutationObserver = class {
    observe() {}
    disconnect() { disconnected = true; }
    takeRecords() { return records.splice(0); }
  };
  const waiting = waitForApprovedText({ ...sample, deadlineMs: Date.now() + 1000 });
  sample.paint(); records.push({ type: 'characterData' });
  sample.scroll(1); sample.scroll(2);
  assert.equal(await waiting, false);
  assert.equal(disconnected, true);
});

test('closing one observer cannot settle the next observation on the same text', async () => {
  const sample = scrollingText(); const observers = [];
  sample.view.MutationObserver = class {
    constructor(callback) { this.callback = callback; observers.push(this); }
    observe() {}
    disconnect() {}
    takeRecords() { return []; }
  };
  let cancelled = false;
  const first = waitForApprovedText({ ...sample, cancelled: () => cancelled, deadlineMs: Date.now() + 1000 });
  sample.paint(); cancelled = true; sample.scroll(1);
  assert.equal(await first, false);
  const second = waitForApprovedText({ ...sample, deadlineMs: Date.now() + 1000 });
  observers[0].callback(); sample.paint(); sample.scroll(0); sample.scroll(2);
  assert.equal(await second, true);
});

test('only explicit recovery can wait past ten seconds and it remains bounded', async t => {
  t.mock.timers.enable({ apis: ['setTimeout', 'setInterval', 'Date'] });
  const normal = scrollingText();
  const defaultWaiting = waitForApprovedText({ ...normal, deadlineMs: Date.now() + 200000 });
  normal.paint(); t.mock.timers.tick(10000);
  assert.equal(await defaultWaiting, false);
  const recovery = scrollingText(); let result;
  const recoveryWaiting = waitForApprovedText({ ...recovery, recovery: true, deadlineMs: Date.now() + 200000 });
  recoveryWaiting.then(value => { result = value; }); recovery.paint();
  t.mock.timers.tick(119999); await Promise.resolve();
  assert.equal(result, undefined);
  t.mock.timers.tick(1); assert.equal(await recoveryWaiting, false);
});

test('idle recovery cancellation does not depend on animation frames or scroll events', async t => {
  t.mock.timers.enable({ apis: ['setTimeout', 'setInterval', 'Date'] });
  const sample = scrollingText(); let cancelled = false;
  const waiting = waitForApprovedText({ ...sample, recovery: true,
    cancelled: () => cancelled, deadlineMs: Date.now() + 120000 });
  sample.paint(); const probes = sample.probes.length;
  cancelled = true; t.mock.timers.tick(100);
  assert.equal(await waiting, false);
  assert.equal(sample.probes.length, probes);
});

test('large visible units yield between bounded layout batches instead of one unbounded scan', async () => {
  const sample = scrollingText('X'.repeat(2300));
  sample.container.clientWidth = sample.view.innerWidth = 24000;
  const waiting = waitForApprovedText({ ...sample, deadlineMs: Date.now() + 1000 });
  sample.paint(); assert.ok(sample.probes.length <= 4096);
  sample.paint(); assert.equal(await waiting, true);
});

test('ancestor scrolling triggers a fresh observation without scrolling the text itself', async () => {
  const sample = scrollingText();
  const waiting = waitForApprovedText({ ...sample, deadlineMs: Date.now() + 1000 });
  sample.paint();
  for (const row of [1, 2]) {
    sample.container.scrollTop = row * 20;
    sample.document.dispatchEvent(new Event('scroll')); sample.paint();
  }
  assert.equal(await waiting, true);
});

test('partially ancestor-clipped glyph cannot be confirmed by an exposed center point', async () => {
  const sample = scrollingText('A\nB'); sample.container.clientHeight = 40;
  sample.container.parentElement = { clientTop: 0, clientLeft: 0, clientWidth: 100, clientHeight: 35,
    getBoundingClientRect: () => ({ top: 0, left: 0 }) };
  sample.view.getComputedStyle = () => ({ overflowY: 'hidden', overflowX: 'visible' });
  const waiting = waitForApprovedText({ ...sample, deadlineMs: Date.now() + 20 });
  sample.paint(); assert.equal(await waiting, false);
});

function automaticText(text = 'A\nB\nC\nD\nE') {
  const sample = scrollingText(text);
  const lines = text.split('\n').length;
  sample.container.scrollHeight = lines * 20 + 20;
  sample.container.scrollWidth = sample.container.clientWidth;
  let top = (lines - 1) * 20;
  sample.moves = [];
  Object.defineProperty(sample.container, 'scrollTop', {
    get: () => top,
    set: value => {
      sample.moves.push(value);
      if (!sample.unreachable) top = Math.max(0, Math.min(value,
        sample.container.scrollHeight - sample.container.clientHeight));
    },
  });
  sample.container.scrollLeft = 0;
  return sample;
}

test('product presents the complete approved suffix across real viewport samples', async () => {
  const sample = automaticText();
  const original = sample.body.textContent, node = sample.body.firstChild;
  const waiting = waitForApprovedText({ ...sample, presentUnseen: true, deadlineMs: Date.now() + 50 });
  for (let i = 0; i < 12; i += 1) sample.paint();
  assert.equal(await waiting, true);
  assert.equal(sample.body.textContent, original); assert.equal(sample.body.firstChild, node);
  assert.ok(sample.moves.length >= 2);
  assert.ok(sample.probes.some(([start, end]) => start === 0 && end === 1));
  assert.ok(sample.probes.some(([start, end]) => start === 8 && end === 9));
});

test('automatic presentation proves only the new suffix, never the old visible prefix', async () => {
  const sample = automaticText('Old\nA\nB\nC');
  sample.payload = { text: 'A\nB\nC', display_text: sample.body.textContent };
  const waiting = waitForApprovedText({ ...sample, presentUnseen: true, deadlineMs: Date.now() + 50 });
  for (let i = 0; i < 8; i += 1) sample.paint();
  assert.equal(await waiting, true);
  assert.ok(sample.probes.every(([start]) => start >= 'Old\n'.length));
  assert.equal(sample.body.textContent, 'Old\nA\nB\nC');
});

test('automatic presentation keeps emoji and combining graphemes whole', async () => {
  const sample = automaticText('甲\n👩‍💻e\u0301\n🙂\n乙');
  const waiting = waitForApprovedText({ ...sample, presentUnseen: true, deadlineMs: Date.now() + 50 });
  for (let i = 0; i < 12; i += 1) sample.paint();
  assert.equal(await waiting, true);
  assert.ok(sample.probes.some(([start, end]) => start === 2 && end === 7));
  assert.ok(sample.probes.some(([start, end]) => start === 7 && end === 9));
});

test('wheel, pointer and navigation keys yield without taking the reading position back', async () => {
  for (const type of ['wheel', 'pointerdown', 'keydown']) {
    const sample = automaticText(); let yielded = 0;
    const waiting = waitForApprovedText({ ...sample, presentUnseen: true,
      onUserNavigation: () => { yielded += 1; }, deadlineMs: Date.now() + 20 });
    const event = new Event(type); Object.defineProperty(event, 'key', { value: 'PageDown' });
    (type === 'keydown' ? sample.document : sample.container).dispatchEvent(event);
    const position = sample.container.scrollTop;
    for (let i = 0; i < 12; i += 1) sample.paint();
    assert.equal(await waiting, false, type);
    assert.equal(sample.container.scrollTop, position, type); assert.equal(sample.moves.length, 0, type);
    assert.equal(yielded, 1, type);
  }
});

test('manual coverage may still acknowledge after automatic following has yielded', async () => {
  const sample = automaticText(); let yielded = 0;
  const waiting = waitForApprovedText({ ...sample, presentUnseen: true,
    onUserNavigation: () => { yielded += 1; }, deadlineMs: Date.now() + 1000 });
  sample.container.dispatchEvent(new Event('wheel'));
  sample.paint();
  for (let i = 0; i < 5; i += 1) sample.scroll(i);
  assert.equal(await waiting, true); assert.equal(yielded, 1);
  assert.equal(sample.container.scrollTop, 80);
});

test('cancelling between automatic pages preserves prior coverage and stops movement', async () => {
  const sample = automaticText(); let cancelled = false;
  const waiting = waitForApprovedText({ ...sample, presentUnseen: true,
    cancelled: () => cancelled, deadlineMs: Date.now() + 1000 });
  sample.paint(); cancelled = true;
  const moves = sample.moves.length, probes = sample.probes.length;
  for (let i = 0; i < 12; i += 1) sample.paint();
  assert.equal(await waiting, false);
  assert.equal(sample.moves.length, moves); assert.equal(sample.probes.length, probes);
});

test('hidden, occluded or unreachable ranges remain unknown without a frame probe loop', async () => {
  for (const kind of ['hidden', 'overlay', 'unreachable']) {
    const sample = automaticText();
    if (kind === 'hidden') sample.document.visibilityState = 'hidden';
    if (kind === 'overlay') sample.blocked.add(0);
    if (kind === 'unreachable') sample.unreachable = true;
    const waiting = waitForApprovedText({ ...sample, presentUnseen: true, deadlineMs: Date.now() + 20 });
    for (let i = 0; i < 12; i += 1) sample.paint();
    const probes = sample.probes.length;
    for (let i = 0; i < 12; i += 1) sample.paint();
    assert.equal(sample.probes.length, probes, kind);
    assert.equal(await waiting, false, kind);
    if (kind === 'hidden') assert.equal(sample.moves.length, 0);
    assert.ok(sample.moves.length <= 1, kind);
  }
});

test('automatic pages use the effective ancestor clip and cannot invent reachable space', async () => {
  const sample = automaticText('A\nB\nC'); sample.container.clientHeight = 40;
  sample.container.parentElement = { clientTop: 0, clientLeft: 0, clientWidth: 100, clientHeight: 25,
    getBoundingClientRect: () => ({ top: 0, left: 0 }) };
  sample.view.getComputedStyle = () => ({ overflowY: 'hidden', overflowX: 'visible' });
  const waiting = waitForApprovedText({ ...sample, presentUnseen: true, deadlineMs: Date.now() + 50 });
  for (let i = 0; i < 12; i += 1) sample.paint();
  assert.equal(await waiting, true);
  const cropped = automaticText('A\nB'); cropped.container.clientHeight = 40;
  cropped.container.scrollHeight = 40;
  cropped.container.parentElement = sample.container.parentElement;
  cropped.view.getComputedStyle = sample.view.getComputedStyle;
  cropped.unreachable = true;
  const unknown = waitForApprovedText({ ...cropped, presentUnseen: true, deadlineMs: Date.now() + 20 });
  for (let i = 0; i < 12; i += 1) cropped.paint();
  assert.equal(await unknown, false);
});

test('mutating and restoring the frozen node during automatic presentation invalidates coverage', async () => {
  const sample = automaticText(); const records = [];
  sample.view.MutationObserver = class { observe() {} disconnect() {} takeRecords() { return records.splice(0); } };
  const waiting = waitForApprovedText({ ...sample, presentUnseen: true, deadlineMs: Date.now() + 1000 });
  sample.paint(); records.push({ type: 'characterData' });
  const moves = sample.moves.length;
  for (let i = 0; i < 12; i += 1) sample.paint();
  assert.equal(await waiting, false); assert.equal(sample.moves.length, moves);
});

test('automatic presentation keeps the original deadline even when paints stop', async t => {
  t.mock.timers.enable({ apis: ['setTimeout', 'setInterval', 'Date'] });
  const sample = automaticText();
  const waiting = waitForApprovedText({ ...sample, presentUnseen: true, deadlineMs: Date.now() + 5000 });
  sample.paint(); t.mock.timers.tick(5000);
  assert.equal(await waiting, false);
});

test('user navigation between automatic paints yields before the next move or latest-position restore', async () => {
  const sample = automaticText();
  const waiting = waitForApprovedText({ ...sample, presentUnseen: true, deadlineMs: Date.now() + 20 });
  sample.paint(); assert.ok(sample.moves.length > 0);
  sample.container.dispatchEvent(new Event('pointerdown')); sample.container.scrollTop = 40;
  const moves = sample.moves.length;
  for (let i = 0; i < 12; i += 1) sample.paint();
  assert.equal(await waiting, false);
  assert.equal(sample.container.scrollTop, 40); assert.equal(sample.moves.length, moves);
});

test('reading keys yield while composer input and ordinary keys retain automatic following', () => {
  assert.equal(isReadingNavigation({ type: 'keydown', key: 'PageDown' }), true);
  assert.equal(isReadingNavigation({ type: 'keydown', key: 'a' }), false);
  assert.equal(isReadingNavigation({ type: 'keydown', key: 'ArrowUp', target: { isContentEditable: true } }), false);
  assert.equal(isReadingNavigation({ type: 'keydown', key: ' ', target: { closest: () => ({}) } }), false);
});

test('the literal renderer honors navigation before approval and retains takeover across pending units', async () => {
  const source = readFileSync(new URL('../src/renderer.mjs', import.meta.url), 'utf8');
  const start = source.indexOf('api.onApproved(async payload => {'), end = source.indexOf('\n});', start) + 5;
  const navigationStart = source.indexOf('let manualReading = false;');
  const navigationEnd = source.indexOf('let recoveryFingerprint', navigationStart);
  const submitStart = source.indexOf("$('composer').onsubmit =");
  const submitEnd = source.indexOf("$('message').onkeydown", submitStart);
  assert.ok(start >= 0 && end > start);
  assert.ok(navigationStart >= 0 && navigationEnd > navigationStart);
  assert.ok(submitStart >= 0 && submitEnd > submitStart);
  // Execute the literal owned callback, with ports substituted only for this CPU
  // interface check. Native layout and ACK remain ROOT's recorded UI check.
  const instance = () => {
    let handler, pending; const options = [];
    const element = () => Object.assign(new EventTarget(), { after() {}, setAttribute() {} });
    const container = Object.assign(element(), { scrollTop: 80, scrollHeight: 100, clientHeight: 20 });
    const controls = { composer: {}, send: { disabled: false }, message: { value: 'explicit next input' },
      status: element(), messages: container };
    const document = Object.assign(new EventTarget(), {
      visibilityState: 'visible', createElement: () => element(),
    });
    document.body = document;
    const api = { onApproved: callback => { handler = callback; }, state: async () => ({}), send: async () => ({}) };
    new Function('api', 'showState', 'approvedScopeMatches', 'document', 'messageList', '$',
      'waitForApprovedText', 'deliveryTiming', 'window', 'isReadingNavigation', 'action', 'refresh',
      'let lastState, sessionId, cancelledDeliveryAttempt, busy=false, sending=false;'
        + 'let sessionEpoch=0,presentationActive=false; const messages=[];'
        + 'const adoptState=state=>{lastState=state;showState(state);};'
        + source.slice(navigationStart, navigationEnd) + source.slice(start, end) + source.slice(submitStart, submitEnd))(
      api, () => {}, () => true, document,
      { presentApproved: () => { container.scrollTop = 100; return {}; } }, id => controls[id] || container,
      async value => { options.push(value); return false; }, { begin() {}, bind() {} }, {}, isReadingNavigation,
      callback => { pending = callback(); }, async () => {});
    return { container, options, handler, submit: () => { controls.composer.onsubmit({ preventDefault() {} }); return pending; } };
  };
  const payload = { scope: { text_attempt_id: 'attempt-1' }, deadline_ms: Date.now() + 1000 };
  const latest = instance(); await latest.handler(payload);
  assert.equal(latest.options.at(-1).presentUnseen, true);
  latest.options.at(-1).onUserNavigation();
  await latest.handler({ ...payload, scope: { text_attempt_id: 'attempt-2' } });
  assert.equal(latest.options.at(-1).presentUnseen, false);
  const priorNavigation = instance(); priorNavigation.container.dispatchEvent(new Event('wheel'));
  await priorNavigation.handler(payload);
  assert.equal(priorNavigation.options.at(-1).presentUnseen, false);
  assert.equal(priorNavigation.container.scrollTop, 80);
  await priorNavigation.submit(); await priorNavigation.handler(payload);
  assert.equal(priorNavigation.options.at(-1).presentUnseen, true, 'An explicit next submit resumes following');
  const history = instance(); history.container.scrollTop = 0;
  await history.handler(payload);
  assert.equal(history.options.at(-1).presentUnseen, false); assert.equal(history.container.scrollTop, 0);
});
