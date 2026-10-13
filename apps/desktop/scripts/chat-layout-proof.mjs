/**
 * Measure the live chat layout from the page's DOM. This function is kept
 * self-contained because the E2E passes it directly to page.evaluate().
 */
export function measureChatLayout() {
  const rectFor = selector => {
    const element = document.querySelector(selector);
    if (!element) return null;
    const rect = element.getBoundingClientRect();
    return {
      left: rect.left,
      top: rect.top,
      right: rect.right,
      bottom: rect.bottom,
      width: rect.width,
      height: rect.height,
    };
  };
  const messages = document.querySelector('#messages');
  return {
    viewport: { width: window.innerWidth, height: window.innerHeight },
    main: rectFor('main'),
    header: rectFor('.conversation-header'),
    error: rectFor('#error'),
    queueStatus: rectFor('#input-queue-status'),
    resumeInputs: rectFor('#resume-inputs'),
    conversationMessages: rectFor('.conversation-messages'),
    messages: rectFor('#messages'),
    status: rectFor('#received-status'),
    footer: rectFor('.composer-footer'),
    composer: rectFor('#composer'),
    leftSidebar: rectFor('#history-sidebar'),
    rightSidebar: rectFor('#companion-panel'),
    messagesInlineStyle: messages?.getAttribute('style') ?? null,
    documentScrollWidth: document.documentElement.scrollWidth,
  };
}

const near = (actual, expected, tolerance = 2) =>
  Math.abs(actual - expected) <= tolerance;
const visible = rect => rect && rect.width > 0 && rect.height > 0;

/** Assert the chat's vertical and horizontal layout invariants. */
export function assertChatLayout(snapshot, { previous } = {}) {
  const { viewport, main, header, conversationMessages, messages, footer, composer } = snapshot ?? {};
  if (!viewport || !main || !header || !conversationMessages || !messages || !footer || !composer) {
    throw new TypeError('chat layout snapshot is missing required rectangles');
  }
  const { width, height } = viewport;
  if (!(width > 0 && height > 0)) throw new Error('viewport dimensions must be positive');
  if (!near(main.top, 0) || !near(main.bottom, height)) {
    throw new Error('main must fill the viewport height');
  }
  if (!near(footer.bottom, height)) {
    throw new Error('#composer-footer must stay aligned with the viewport bottom');
  }
  if (composer.top < footer.top - 1 || composer.bottom > footer.bottom + 1) {
    throw new Error('#composer must remain inside #composer-footer');
  }
  const composerBottomGap = footer.bottom - composer.bottom;
  if (composerBottomGap < -1 || composerBottomGap >= 110) {
    throw new Error('#composer must stay close to the bottom of #composer-footer');
  }
  if (conversationMessages.top < header.bottom - 1 ||
      conversationMessages.bottom > footer.top + 1) {
    throw new Error('.conversation-messages must fill the area between header and footer');
  }
  if (!near(conversationMessages.bottom, footer.top)) {
    throw new Error('.conversation-messages must end at the composer footer');
  }
  if (snapshot.messagesInlineStyle !== null) {
    throw new Error('#messages must not use inline layout overrides');
  }
  if (conversationMessages.height < height * 0.30 || messages.height < height * 0.30) {
    throw new Error('chat message viewport is unexpectedly short');
  }
  if (messages.top < conversationMessages.top - 1 ||
      messages.bottom > conversationMessages.bottom + 1 ||
      messages.height < conversationMessages.height * 0.75) {
    throw new Error('#messages must fill its conversation viewport');
  }
  if (snapshot.documentScrollWidth > width + 1) {
    throw new Error('document horizontal overflow exceeds the viewport');
  }
  for (const [name, rect] of [
    ['main', main],
    ['left sidebar', snapshot.leftSidebar],
    ['right sidebar', snapshot.rightSidebar],
  ]) {
    if (visible(rect) && (rect.left < -1 || rect.right > width + 1)) {
      throw new Error(`${name} extends outside the viewport`);
    }
  }
  if (visible(snapshot.status) && (
    snapshot.status.left < conversationMessages.left - 1 ||
    snapshot.status.right > conversationMessages.right + 1 ||
    snapshot.status.top < conversationMessages.top - 1 ||
    snapshot.status.bottom > conversationMessages.bottom + 1
  )) {
    throw new Error('#received-status must remain inside the conversation viewport');
  }
  if (previous && previous.viewport.width === width && previous.viewport.height === height) {
    for (const key of ['top', 'bottom', 'left', 'right']) {
      if (!near(composer[key], previous.composer[key], 1)) {
        throw new Error(`#composer ${key} moved while the viewport stayed the same`);
      }
      if (!near(messages[key], previous.messages[key], 1)) {
        throw new Error(`#messages ${key} moved while the viewport stayed the same`);
      }
    }
  }
  return snapshot;
}
