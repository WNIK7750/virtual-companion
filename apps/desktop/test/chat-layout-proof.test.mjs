import test from 'node:test';
import assert from 'node:assert/strict';
import { assertChatLayout } from '../scripts/chat-layout-proof.mjs';

const rect = (left, top, width, height) => ({
  left, top, right: left + width, bottom: top + height, width, height,
});
const zeroRect = () => rect(0, 0, 0, 0);

function normalLayout({ receivedStatus = false } = {}) {
  const mainLeft = 220, mainWidth = 788, viewportHeight = 805, composerTop = 624;
  const conversationMessages = rect(mainLeft, 72, mainWidth, composerTop - 72);
  const messages = rect(mainLeft, 72, mainWidth, composerTop - 72);
  const status = receivedStatus
    ? rect(mainLeft + 270, conversationMessages.top + 180, 248, 42)
    : zeroRect();
  return {
    viewport: { width: 1280, height: viewportHeight },
    main: rect(mainLeft, 0, mainWidth, viewportHeight),
    header: rect(mainLeft, 0, mainWidth, 72),
    error: zeroRect(),
    queueStatus: zeroRect(),
    resumeInputs: zeroRect(),
    conversationMessages,
    messages,
    status,
    footer: rect(mainLeft, composerTop, mainWidth, viewportHeight - composerTop),
    composer: rect(mainLeft + 27, composerTop, mainWidth - 54, 131),
    leftSidebar: rect(0, 0, 220, viewportHeight),
    rightSidebar: rect(mainLeft + mainWidth, 0, 272, viewportHeight),
    messagesInlineStyle: null,
    documentScrollWidth: 1280,
  };
}

test('normal chat layout fills the window and keeps the composer inside the bottom footer', () => {
  const snapshot = normalLayout();
  assert.equal(assertChatLayout(snapshot), snapshot);
});

test('the old injected 120px screenshot fixture is rejected by style and geometry', () => {
  const snapshot = normalLayout();
  snapshot.messages = rect(220, 72, 788, 120);
  snapshot.messagesInlineStyle = 'height: 120px; flex: none; overflow: auto;';

  assert.throws(() => assertChatLayout(snapshot), /#messages must not use inline layout overrides/u);
  assert.throws(() => assertChatLayout({ ...snapshot, messagesInlineStyle: null }), /chat message viewport is unexpectedly short/u);
});

test('a received-status prompt overlays messages without moving messages or composer', () => {
  const normal = normalLayout();
  const prompted = normalLayout({ receivedStatus: true });
  assert.ok(prompted.status.height >= 32);
  assert.deepEqual(prompted.messages, normal.messages);
  assert.deepEqual(prompted.conversationMessages, normal.conversationMessages);
  assert.equal(assertChatLayout(prompted, { previous: normal }), prompted);
});

test('chat layout rejects horizontal overflow outside the viewport', () => {
  const snapshot = normalLayout();
  snapshot.documentScrollWidth = snapshot.viewport.width + 8;
  assert.throws(() => assertChatLayout(snapshot), /document horizontal overflow/u);
});
