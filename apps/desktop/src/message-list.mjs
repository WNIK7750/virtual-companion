import { icon } from './icons.mjs';
import { mergeApprovedMessages, visiblePartial } from './approved-display.mjs';

export function projectBubbles(messages) {
  const visual = messages.flatMap((message, index) => {
    if (!message.bubble_ranges) return [{ ...message, visual_order: message.timeline_order || [index, 0] }];
    const points = Array.from(message.content); let cursor = 0;
    const bubbles = message.bubble_ranges.map((range, bubbleIndex) => {
      if (!Number.isSafeInteger(range.start) || !Number.isSafeInteger(range.end)
          || range.start !== cursor || range.end <= cursor || range.end > points.length) {
        throw new Error('Invalid message bubble range');
      }
      const content = points.slice(cursor, range.end).join(''); cursor = range.end;
      const order = message.bubble_orders?.find(item => item.start === range.start)?.order;
      return { ...message, id: `${message.id}:bubble:${range.start}`, logical_id: message.id,
        original_message: message, content, bubble_start: range.start,
        bubble_end: range.end,
        bubble_ranges: [{ start: 0, end: range.end - range.start }],
        final_bubble: bubbleIndex === message.bubble_ranges.length - 1,
        visual_order: order || message.timeline_order || [index, bubbleIndex] };
    });
    if (cursor !== points.length) throw new Error('Incomplete message bubble ranges');
    return bubbles;
  });
  visual.sort((a, b) => a.visual_order[0] - b.visual_order[0] || a.visual_order[1] - b.visual_order[1]);
  return visual;
}

export function feedbackTargetFor(message, state = {}) {
  if (message.role === 'user') {
    const attempt = state.attempt;
    if (attempt && attempt.turn_id === message.turn_id && typeof attempt.attempt_id === 'string'
        && ['failed', 'cancelled', 'interrupted', 'listened', 'received'].includes(attempt.status)) {
      return { kind: 'attempt', attempt_id: attempt.attempt_id };
    }
    return null;
  }
  if (message.role !== 'assistant' || message.approvedOnly || !message.id) return null;
  const start = message.bubble_start ?? 0;
  const end = message.bubble_end ?? start + Array.from(message.content).length;
  return message.bubble_ranges
    ? { kind: 'bubble', start, end }
    : { kind: 'reply', start: 0, end: Array.from(message.content).length };
}

export function feedbackTargetFromSelection(message, body, range) {
  if (!message || message.role !== 'assistant' || message.approvedOnly || !message.id ||
      !body || !range || typeof body.contains !== 'function' ||
      !body.contains(range.startContainer) || !body.contains(range.endContainer)) return null;
  const selected = range.toString();
  if (!selected) return null;
  const prefix = range.cloneRange();
  prefix.selectNodeContents(body);
  prefix.setEnd(range.startContainer, range.startOffset);
  const source = message.original_message || message;
  const start = (message.bubble_start ?? 0) + Array.from(prefix.toString()).length;
  const end = start + Array.from(selected).length;
  if (end > Array.from(source.content || '').length) return null;
  return {
    message: source,
    target: { kind: 'bubble', start, end },
  };
}

function renderBody(body, message) {
  const signature = JSON.stringify(message.bubble_ranges || null);
  if (body.textContent === message.content && body.bubbleSignature === signature) return;
  if (!message.bubble_ranges) body.textContent = message.content;
  else {
    const points = Array.from(message.content), children = []; let cursor = 0;
    if (!Array.isArray(message.bubble_ranges) || !message.bubble_ranges.length) throw new Error('Invalid message bubbles');
    for (const range of message.bubble_ranges) {
      if (!Number.isSafeInteger(range.start) || !Number.isSafeInteger(range.end)
          || range.start !== cursor || range.end <= cursor || range.end > points.length) {
        throw new Error('Invalid message bubble range');
      }
      const text = points.slice(cursor, range.end).join('');
      // Keep separators in the exact DOM text, outside the painted bubble box.
      const separator = text.match(/(?:\r?\n[ \t]*)+$/u)?.[0] || '';
      const bubble = document.createElement('div'); bubble.className = 'message-bubble';
      bubble.textContent = separator ? text.slice(0, -separator.length) : text;
      children.push(bubble);
      if (separator) children.push(document.createTextNode(separator));
      cursor = range.end;
    }
    if (cursor !== points.length) throw new Error('Incomplete message bubble ranges');
    body.replaceChildren(...children);
  }
  body.className = message.bubble_ranges ? 'message-body semantic-bubbles' : 'message-body';
  body.bubbleSignature = signature;
}

export class MessageList {
  constructor(container, onFeedback) {
    this.container = container;
    this.onFeedback = onFeedback;
    this.nodes = new Map();
    this.approved = new Map();
  }

  create(message) {
    const element = document.createElement('article');
    element.className = `message ${message.role}`;
    if (message.id) element.dataset.messageId = message.id;
    if (message.turn_id) element.dataset.turnId = message.turn_id;
    const label = document.createElement('small');
    label.textContent = message.role === 'user' ? '你' : '伴侣';
    const body = document.createElement('div'); renderBody(body, message);
    element.append(label, body);
    if (message.role === 'assistant') {
      const avatar = icon('flower'); avatar.className = 'message-avatar'; element.prepend(avatar);
    }
    return { element, body };
  }

  updateFeedback(node, message, state) {
    const target = feedbackTargetFor(message, state);
    const shown = target !== null;
    if (!shown) { node.feedback?.remove(); node.feedback = null; }
    else if (!node.feedback) {
      node.feedback = document.createElement('button'); node.feedback.className = 'message-feedback';
      node.feedback.append(icon('heart'), document.createTextNode('留下感受'));
      node.feedback.title = '选中本条消息中的文字后点此，可只反馈该范围';
      node.feedback.setAttribute?.('aria-label', node.feedback.title);
      node.feedback.onmousedown = () => {
        const selection = document.getSelection?.();
        node.selectedFeedback = this.feedbackTargetFromSelection(selection, node);
      };
      node.feedback.onclick = () => {
        const selection = document.getSelection?.();
        const selected = node.selectedFeedback || this.feedbackTargetFromSelection(selection, node);
        node.selectedFeedback = null;
        const messageValue = selected?.message || node.message.original_message || node.message;
        this.onFeedback(messageValue, selected?.target || node.feedbackTarget);
      };
      node.element.append(node.feedback);
    }
    if (shown) {
      node.feedbackTarget = target;
      if (target.kind === 'attempt') {
        node.feedback.title = '仅反馈这次尝试；未显示的文字不会进入导出';
        node.feedback.setAttribute?.('aria-label', node.feedback.title);
        if (node.feedback.lastChild?.textContent !== '反馈这次尝试') {
          node.feedback.lastChild.textContent = '反馈这次尝试';
        }
      } else {
        node.feedback.title = '选中本条消息中的文字后点此，可只反馈该范围';
        node.feedback.setAttribute?.('aria-label', node.feedback.title);
        if (node.feedback.lastChild?.textContent !== '留下感受') {
          node.feedback.lastChild.textContent = '留下感受';
        }
      }
    }
  }

  feedbackTargetFromSelection(selection, node) {
    if (!selection || selection.rangeCount !== 1 || selection.isCollapsed) return null;
    return feedbackTargetFromSelection(node.message, node.body, selection.getRangeAt(0));
  }

  update(messages, state, sessionId, { prepend = false } = {}) {
    const reset = this.sessionId !== sessionId;
    if (reset) this.approved.clear();
    for (const message of messages) {
      const approved = this.approved.get(message.turn_id);
      if (message.role === 'assistant' && approved
          && message.content.length >= approved.content.length) this.approved.delete(message.turn_id);
    }
    if (this.approved.size) {
      // A first ACK can already be mirrored while the next approved unit is
      // arriving. Extend that same bubble until the mirror catches the prefix.
      messages = messages.map(message => {
        const approved = this.approved.get(message.turn_id);
        return message.role === 'assistant' && approved?.content.startsWith(message.content)
          ? { ...message, content: approved.content, bubble_ranges: approved.bubble_ranges,
            bubble_orders: approved.bubble_orders } : message;
      });
      messages = mergeApprovedMessages(messages, [...this.approved.values()]);
    }
    const partial = visiblePartial(state);
    const partialKey = partial ? state.attempt?.attempt_id : null;
    const feedbackStateKey = JSON.stringify([state?.attempt?.turn_id,
      state?.attempt?.attempt_id, state?.attempt?.status]);
    const feedbackStateChanged = feedbackStateKey !== this.feedbackStateKey;
    if (!reset && messages === this.messages && partial === this.partialText
        && partialKey === this.partialKey && !feedbackStateChanged) return;
    const container = this.container;
    const oldTop = container.scrollTop, oldHeight = container.scrollHeight;
    const atBottom = oldHeight - oldTop - container.clientHeight < 48;
    if (reset) {
      container.replaceChildren(); this.nodes.clear(); this.partialNode = null; this.empty = null;
    }
    if (messages.length || partial) { this.empty?.remove(); this.empty = null; }
    if (reset || messages !== this.messages) {
      const visual = projectBubbles(messages);
      const ids = new Set(visual.map(message => message.id));
      for (const [id, node] of this.nodes) {
        if (!ids.has(id)) { node.element.remove(); this.nodes.delete(id); }
      }
      let reference = container.firstChild;
      let previous;
      for (const message of visual) {
        let node = this.nodes.get(message.id);
        if (!node) { node = this.create(message); this.nodes.set(message.id, node); }
        node.message = message;
        const continuation = message.logical_id && previous?.logical_id === message.logical_id;
        node.element.className = `message ${message.role}${message.logical_id ? ' semantic-message' : ''}${continuation ? ' continuation' : ''}`;
        renderBody(node.body, message);
        this.updateFeedback(node, message, state);
        if (node.element !== reference) container.insertBefore(node.element, reference);
        reference = node.element.nextSibling;
        previous = message;
      }
    } else if (feedbackStateChanged) {
      for (const node of this.nodes.values()) this.updateFeedback(node, node.message, state);
    }
    if (partialKey !== this.partialKey || !partial) {
      this.partialNode?.element.remove(); this.partialNode = null;
    }
    if (partial) {
      if (!this.partialNode) {
        this.partialNode = this.create({ role: 'assistant', content: partial });
        container.append(this.partialNode.element);
      } else if (partial !== this.partialText) this.partialNode.body.textContent = partial;
    }
    if (!messages.length && !partial && !this.empty) {
      this.empty = document.createElement('div'); this.empty.className = 'empty';
      const mark = icon('flower'); mark.className = 'empty-mark';
      const heading = document.createElement('h2'); heading.textContent = '留一点空白，给今天。';
      const hint = document.createElement('p'); hint.textContent = '想说什么，都可以从这里开始。';
      this.empty.append(mark, heading, hint); container.append(this.empty);
    }
    if (prepend) container.scrollTop = oldTop + container.scrollHeight - oldHeight;
    else if (this.followLatest !== false && (reset || atBottom)) container.scrollTop = container.scrollHeight;
    else container.scrollTop = oldTop;
    this.sessionId = sessionId; this.messages = messages;
    this.partialText = partial; this.partialKey = partialKey;
    this.feedbackStateKey = feedbackStateKey;
  }

  presentApproved(payload, messages, state, sessionId) {
    // Initialize/reset the selected session before adding its trusted transient.
    this.update(messages, state, sessionId);
    const confirmed = messages.find(message => message.role === 'assistant' && message.turn_id === payload.scope.turn_id);
    if (payload.received && confirmed?.content === payload.display_text) {
      return this.presentationFor(confirmed.id, payload);
    }
    if (confirmed && (!payload.display_text.startsWith(confirmed.content)
        || payload.display_text === confirmed.content)) return null;
    const previous = this.approved.get(payload.scope.turn_id);
    if (previous && payload.sequence <= previous.sequence) {
      return payload.received && payload.sequence === previous.sequence
        ? this.presentationFor(previous.id, payload) : null;
    }
    const message = { id: confirmed?.id || `visible:${payload.scope.text_attempt_id}`, role: 'assistant',
      turn_id: payload.scope.turn_id, content: payload.display_text, approvedOnly: true, sequence: payload.sequence,
      ...(payload.bubble_ranges ? { bubble_ranges: payload.bubble_ranges } : {}),
      ...(payload.bubble_orders ? { bubble_orders: payload.bubble_orders } : {}) };
    this.approved.set(message.turn_id, message);
    this.update(messages, state, sessionId);
    return this.presentationFor(message.id, payload);
  }

  presentationFor(id, payload) {
    const nodes = [...this.nodes.values()].filter(node => node.message.logical_id === id || node.message.id === id)
      .sort((a, b) => (a.message.bubble_start || 0) - (b.message.bubble_start || 0));
    this.presentationElements = nodes.map(node => node.element);
    const start = Array.from(payload.display_text.slice(0, payload.display_text.length - payload.text.length)).length;
    this.pendingElement = nodes.find(node => (node.message.bubble_start || 0)
      + Array.from(node.message.content).length > start)?.element;
    return nodes[0]?.element;
  }
}
