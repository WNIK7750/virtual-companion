import { icon, populateIcons } from './icons.mjs';
import { initializePanels } from './theme.mjs';
import { MessageList } from './message-list.mjs';
import { MemoryPanel } from './memory-panel.mjs';
import { ProviderPanel } from './provider-panel.mjs';
import { approvedScopeMatches, canRetry } from './approved-display.mjs';
import { isReadingNavigation, waitForApprovedText } from './approved-visibility.mjs';
import { deliveryTiming } from './delivery-timing.mjs';
populateIcons();
initializePanels();
const $ = id => document.getElementById(id);
const api = window.companion;
const memoryPanel = new MemoryPanel({ api, document, randomUUID: () => crypto.randomUUID() });
const providerPanel = new ProviderPanel({ api, document });
let busy = false;
let sending = false;
let messages = [];
let historyRevision = '';
let pendingNewer = false;
let nextMessageBefore = null;
let loadingMessages = false;
let lastState = {};
let stateError = '';
let refreshing;
let presentationPolling;
let presentationActive = false;
let sessionEpoch = 0;
const messageList = new MessageList($('messages'), (message, target) =>
  action(() => openFeedback(message, target)));
const moreMessages = document.createElement('button');
moreMessages.id = 'messages-more'; moreMessages.textContent = '加载更早的消息'; moreMessages.hidden = true;
$('messages').before(moreMessages);
let sessionId = '';
let feedbackTarget;
let feedbackEvent;
let previewId;
let previewRevision;
let previewHash;
let previewApplied = false;
let feedbackKey;
let previousSessions = '';
let olderSessions = [];
let olderLoaded = false;
let nextOffset = null;
let loadingOlder = false;
let cancelledDeliveryAttempt;
let manualReading = false;
let quiet = false;
let notificationTarget;
function revealNotificationTarget() {
  if (!notificationTarget || notificationTarget.session_id !== sessionId) return;
  const node = [...$('messages').querySelectorAll('article.assistant')]
    .find(element => element.dataset.turnId === notificationTarget.turn_id);
  if (node) { node.scrollIntoView({ block: 'start' }); notificationTarget = null; }
}
api.onNotification?.(target => { void action(async () => {
  const state = await api.state();
  if (state.session_id !== target.session_id) return;
  if (sessionId !== target.session_id) changeSession();
  manualReading = true; notificationTarget = target;
  adoptState(state); await refresh();
  // Old native reminders can outlive the latest history page. Load actual
  // earlier pages until the bound reply is found; never treat navigation as ACK.
  while (notificationTarget && nextMessageBefore && sessionId === target.session_id) {
    const before = nextMessageBefore;
    await moreMessages.onclick(); revealNotificationTarget();
    if (nextMessageBefore === before) break;
  }
  revealNotificationTarget();
}); });
const quietButton = document.createElement('button');
quietButton.id = 'quiet-receiving'; quietButton.type = 'button';
quietButton.title = '文字照常接收，关闭自动滚动、通知和声音。';
$('status').after(quietButton);
quietButton.onclick = () => action(async () => { await api.setQuiet(!quiet); adoptState(await api.state()); });
const receivedStatus = document.createElement('button');
receivedStatus.id = 'received-status'; receivedStatus.type = 'button'; receivedStatus.hidden = true;
receivedStatus.title = '跳到首条新回复'; $('messages').after(receivedStatus);
receivedStatus.onclick = () => {
  manualReading = false;
  messageList.pendingElement?.scrollIntoView({ block: 'start' });
};
const yieldReading = event => { if (isReadingNavigation(event)) {
  manualReading = true; notificationTarget = null;
} };
for (const [target, event, capture = false] of [[$('messages'), 'wheel'],
  [$('messages'), 'pointerdown'], [document.body, 'keydown', true]]) {
  target.addEventListener(event, yieldReading, { passive: true, capture });
}
let recoveryFingerprint = null;
let recoverable = [];
const queueStatus = document.createElement('p');
queueStatus.id = 'input-queue-status'; queueStatus.hidden = true;
queueStatus.setAttribute('role', 'status'); $('error').after(queueStatus);
const resumeInputs = document.createElement('button');
resumeInputs.id = 'resume-inputs'; resumeInputs.textContent = '继续处理已收到的消息';
resumeInputs.hidden = true; queueStatus.after(resumeInputs);
resumeInputs.onclick = () => action(async () => { await api.resumeInputs(); await refresh(); });
const recoveryButton = document.createElement('button');
recoveryButton.id = 'resume-approved'; recoveryButton.hidden = true;
recoveryButton.textContent = '继续查看未确认的回复';
$('error').after(recoveryButton);
const recoveryDialog = document.createElement('dialog');
recoveryDialog.id = 'recovery-dialog'; recoveryDialog.className = 'recovery-dialog';
recoveryDialog.setAttribute('aria-labelledby', 'recovery-title');
const recoveryTitle = document.createElement('h2'); recoveryTitle.id = 'recovery-title';
recoveryTitle.textContent = '上次未确认回复的原文';
const recoveryHint = document.createElement('p');
recoveryHint.textContent = '这是上次回复的原文，内容可能已过时。请滚动查看；完整显示后会保存确认。关闭后仍可继续查看。';
const recoveryBody = document.createElement('section'); recoveryBody.id = 'recovery-text';
recoveryBody.tabIndex = 0; recoveryBody.setAttribute('aria-label', '待确认的回复原文');
const recoveryClose = document.createElement('button'); recoveryClose.textContent = '稍后查看';
recoveryClose.id = 'recovery-close';
recoveryDialog.append(recoveryTitle, recoveryHint, recoveryBody, recoveryClose);
document.body.append(recoveryDialog);
const closeRecovery = () => { recoveryFingerprint = null; recoveryDialog.close(); void api.cancelRecovery(); };
recoveryClose.onclick = closeRecovery;
recoveryDialog.addEventListener('cancel', event => { event.preventDefault(); closeRecovery(); });
recoveryButton.onclick = () => action(async () => {
  if (busy || recoveryFingerprint || !recoverable.length) return;
  recoveryFingerprint = recoverable[0].fingerprint;
  recoveryButton.disabled = true;
  try {
    const result = await api.resumeApproved(recoveryFingerprint);
    // Recovery observations are intentionally excluded from normal t0 timing.
    error(result.status === 'confirmed' ? '' : '这段回复尚未完整显示，可以稍后继续查看。');
    if (result.status === 'confirmed') {
      messages = []; historyRevision = ''; pendingNewer = true;
    }
  } finally {
    recoveryFingerprint = null; recoveryDialog.close(); recoveryButton.disabled = false;
    await refresh();
  }
});
api.onApproved(async payload => {
  // The backend may have prepared the scope after this refresh began. This is
  // an independent state read, never a recursive refresh waiting on delivery.
  const epoch = sessionEpoch;
  const current = await api.state();
  if (epoch !== sessionEpoch) return false;
  adoptState(current);
  if (payload.recovery === true) {
    if (busy || recoveryFingerprint !== payload.fingerprint || !Number.isSafeInteger(payload.deadline_ms)
        || payload.deadline_ms <= Date.now()) return false;
    const element = document.createElement('article');
    const body = document.createElement('div'); body.textContent = payload.display_text;
    element.append(body); recoveryBody.replaceChildren(element);
    recoveryDialog.showModal(); recoveryBody.scrollTop = 0;
    return waitForApprovedText({ element, container: recoveryBody, payload, document, view: window,
      deadlineMs: payload.deadline_ms, recovery: true,
      cancelled: () => recoveryFingerprint !== payload.fingerprint || !recoveryDialog.open });
  }
  const receivedScope = payload.received === true && lastState.reliable_text_receive === true
    && payload.scope?.session_id === sessionId && sessionId === lastState.session_id;
  if ((!receivedScope && !approvedScopeMatches(payload.scope, lastState, sessionId))
      || (!receivedScope && cancelledDeliveryAttempt === payload.scope.text_attempt_id)
      || !Number.isSafeInteger(payload.deadline_ms) || Date.now() >= payload.deadline_ms
      || document.visibilityState !== 'visible') return false;
  const container = $('messages');
  const followingLatest = !quiet && !manualReading && container.scrollHeight - container.scrollTop - container.clientHeight < 48;
  const readingTop = container.scrollTop;
  presentationActive = true;
  try {
    const element = messageList.presentApproved(payload, messages, lastState, sessionId);
    if (!followingLatest) container.scrollTop = readingTop;
    revealNotificationTarget();
    if (!element) return false;
    // A visible result remains evidence even if cancellation arrived after display.
    const visible = await waitForApprovedText({ element, elements: messageList.presentationElements, container, payload,
      document, view: window, deadlineMs: payload.deadline_ms,
      presentUnseen: followingLatest,
      onUserNavigation: () => { manualReading = true; },
      cancelled: () => receivedScope ? payload.scope.session_id !== sessionId
        : cancelledDeliveryAttempt === payload.scope.text_attempt_id
          || !approvedScopeMatches(payload.scope, lastState, sessionId) });
    if (visible) deliveryTiming.visible(payload);
    return visible;
  } finally { presentationActive = false; }
});
const moreHistory = document.createElement('button');
moreHistory.id = 'history-more'; moreHistory.textContent = '加载更早的对话'; moreHistory.hidden = true;
$('sessions').after(moreHistory);
const labels = { stopped: '尚未启动', starting_backend: '正在启动', loading_model: '模型加载中', ready: '本地模型已就绪', failed: '需要重试', stopping: '正在停止', configuration_unavailable: '启动配置未就绪，请修复配置后重启', provider_unavailable: 'DeepSeek Flash 未就绪，请检查模型设置', provider_recovery_required: '请在模型设置中恢复路由' };
function error(message) { $('error').textContent = message; $('error').hidden = !message; }
function showState(value) {
  quiet = value.receiving_preferences?.quiet === true;
  messageList.followLatest = !quiet && !manualReading;
  quietButton.textContent = quiet ? '免打扰：已开启' : '免打扰：已关闭';
  quietButton.setAttribute('aria-pressed', String(quiet));
  receivedStatus.hidden = !(value.received_pending > 0);
  receivedStatus.textContent = `${value.received_replies || value.received_pending || 0} 条新回复 · 查看`;
  deliveryTiming.observe(value);
  busy = !!value.busy;
  $('status').textContent = value.mode === 'fake' ? '开发模式 · 未连接模型'
    : value.mode === 'api' && value.state === 'ready' ? 'DeepSeek Flash 已就绪'
    : (labels[value.state] || value.state);
  $('send').disabled = value.state !== 'ready' || value.startup?.can_send === false || sending;
  $('start').hidden = value.state === 'ready' || value.mode === 'api' || value.state === 'configuration_unavailable';
  $('start').disabled = ['starting_backend', 'loading_model', 'stopping'].includes(value.state);
  $('stop').hidden = !busy;
  $('retry').hidden = !canRetry(value);
  $('new-session').disabled = busy || sending;
  const queued = value.queued_inputs?.length || 0;
  queueStatus.hidden = !queued;
  queueStatus.textContent = value.input_queue_waiting || (value.input_queue_paused
    ? `${queued} 条消息已保存，等待继续处理。` : `${queued} 条消息已收到，等当前表达结束后接续。`);
  resumeInputs.hidden = !queued || !value.input_queue_paused || busy;
  memoryPanel.setAvailable(value.memory_management === true);
  providerPanel.setState(value.provider);
  if (value.error || $('error').textContent === stateError) error(value.error || '');
  stateError = value.error || '';
}
function adoptState(state) {
  if (state.session_id && sessionId !== state.session_id) {
    sessionId = state.session_id; messages = []; historyRevision = ''; pendingNewer = false;
    nextMessageBefore = null;
    if (!presentationActive) messageList.update(messages, state, sessionId);
  }
  lastState = state; showState(state);
}
function changeSession() {
  sessionEpoch++;
  refreshing = undefined;
  sessionId = ''; messages = []; historyRevision = ''; pendingNewer = false;
  nextMessageBefore = null;
}
async function refreshOnce() {
  const epoch = sessionEpoch;
  const state = await api.state();
  if (epoch !== sessionEpoch) return;
  adoptState(state);
  const revision = JSON.stringify([state.session_id, state.attempt?.attempt_id, state.attempt?.status,
    state.queued_inputs]);
  if (state.session_id && (revision !== historyRevision || pendingNewer)) {
    const changedSession = !historyRevision;
    // A confirmed stream appends to one assistant message ID. Query from that
    // turn's user message so an ACK refresh can also replace its cached prefix.
    const approvedCursor = pendingNewer && state.delivery_protocol === 'approved-prefix.v1'
      ? messages.findLast(message => message.role === 'user' && message.turn_id === state.attempt?.turn_id)?.id : null;
    const tail = changedSession ? null : approvedCursor || messages.at(-1)?.id;
    let history;
    try {
      history = await api.history({ session_id: state.session_id, ...(tail ? { after: tail } : {}) });
    } catch (reason) {
      if (epoch !== sessionEpoch || (await api.state()).session_id !== state.session_id) return;
      throw reason;
    }
    if (epoch !== sessionEpoch || sessionId !== state.session_id) return;
    if (changedSession || !tail) {
      messages = history.messages; nextMessageBefore = history.next_before;
    } else if (history.messages.length) messages = [...new Map([...messages, ...history.messages]
      .map(message => [message.id, message])).values()];
    sessionId = history.session_id; historyRevision = revision;
    pendingNewer = !!history.next_after;
  }
  const readingTop = $('messages').scrollTop;
  if (!presentationActive) messageList.update(messages, lastState, sessionId);
  if (manualReading || quiet) $('messages').scrollTop = readingTop;
  revealNotificationTarget();
  moreMessages.hidden = !nextMessageBefore;
  moreMessages.disabled = loadingMessages;
  const firstPage = await api.sessions();
  if (epoch !== sessionEpoch || sessionId !== state.session_id) return;
  if (!olderLoaded) nextOffset = firstPage.next_offset;
  const sessions = [...new Map([...firstPage.sessions, ...olderSessions].map(item => [item.id, item])).values()];
  moreHistory.hidden = nextOffset == null;
  moreHistory.disabled = busy || loadingOlder;
  $('conversation-title').textContent = sessions.find(item => item.id === sessionId)?.title || '新的对话';
  const signature = JSON.stringify([sessions, sessionId, busy]);
  if (signature !== previousSessions) {
    previousSessions = signature;
    $('sessions').replaceChildren(...sessions.map(item => {
      const button = document.createElement('button');
      const title = document.createElement('span'); title.className = 'session-title'; title.textContent = item.title;
      button.append(icon('chat'), title); button.title = item.title;
      button.setAttribute('aria-current', String(item.id === sessionId));
      button.disabled = busy || item.id === sessionId;
      button.onclick = () => action(async () => { manualReading = false;
        await api.selectSession(item.id); changeSession();
        document.body.classList.remove('history-open'); $('history-toggle').setAttribute('aria-expanded', 'false');
        adoptState(await api.state()); void action(refresh); });
      return button;
    }));
  }
}
function refresh() {
  if (!refreshing) {
    const current = refreshOnce().finally(() => { if (refreshing === current) refreshing = undefined; });
    refreshing = current;
  }
  return refreshing;
}
function pollApproved() {
  if (!presentationPolling) {
    const current = (async () => {
      const epoch = sessionEpoch;
      const state = await api.state();
      if (epoch !== sessionEpoch) return { status: 'stale' };
      adoptState(state);
      if (state.delivery_protocol !== 'approved-prefix.v1') return { status: 'idle' };
      const result = await api.presentApproved();
      // Durable receipts are real even if the user selected another session.
      for (const receipt of result.receipts || []) deliveryTiming.acknowledged(receipt);
      if (epoch !== sessionEpoch) return result;
      if (result.receipts?.length) pendingNewer = true;
      if (result.error) error('部分显示状态仍在恢复，会自动重试。');
      if (result.recoverable) recoverable = result.recoverable;
      recoveryButton.hidden = busy || !recoverable.length;
      recoveryButton.disabled = !!recoveryFingerprint;
      return result;
    })().finally(() => { if (presentationPolling === current) presentationPolling = undefined; });
    presentationPolling = current;
  }
  return presentationPolling;
}
moreMessages.onclick = () => action(async () => {
  if (loadingMessages || !nextMessageBefore) return;
  const targetSession = sessionId;
  loadingMessages = true; moreMessages.disabled = true;
  try {
    const page = await api.history({ before: nextMessageBefore, session_id: targetSession });
    if (targetSession !== sessionId) return;
    messages = [...new Map([...page.messages, ...messages].map(message => [message.id, message])).values()];
    nextMessageBefore = page.next_before;
    if (!presentationActive) messageList.update(messages, lastState, sessionId, { prepend: true });
    moreMessages.hidden = !nextMessageBefore;
  } finally { loadingMessages = false; moreMessages.disabled = false; }
});
async function action(callback) { try { error(''); await callback(); } catch (reason) { error(reason.message); } }
moreHistory.onclick = () => action(async () => {
  if (loadingOlder || nextOffset == null) return;
  const targetSession = sessionId;
  loadingOlder = true; moreHistory.disabled = true;
  try {
    const page = await api.sessions(nextOffset);
    if (targetSession !== sessionId) return;
    olderSessions.push(...page.sessions); olderLoaded = true; nextOffset = page.next_offset;
    await refresh();
  } finally { loadingOlder = false; moreHistory.disabled = busy; }
});
$('start').onclick = () => action(async () => { showState(await api.start()); });
$('new-session').onclick = () => action(async () => {
  if (busy || sending) return;
  sending = true; showState(lastState);
  try {
    manualReading = false;
    await api.newSession();
    // A held metadata request belongs to the old selection. Its response may
    // finish later, but cannot hold admission or refill this new scope.
    changeSession();
    olderSessions = []; olderLoaded = false;
    nextOffset = null; previousSessions = '';
    feedbackTarget = feedbackEvent = feedbackKey = previewId = previewRevision = previewHash = undefined;
    previewApplied = false;
    $('feedback-dialog').close(); $('feedback-text').value = '';
    for (const input of document.querySelectorAll('[name="quick_tags"]')) input.checked = false;
    $('export-panel').hidden = true; $('export-confirmation').checked = false;
    $('export-preview').value = ''; $('export-preview-meta').textContent = '';
    $('export-apply').disabled = false; $('export-save').disabled = true;
    $('feedback-status').textContent = '';
    if (recoveryFingerprint) await api.cancelRecovery();
    recoveryFingerprint = null; recoverable = []; cancelledDeliveryAttempt = undefined;
    recoveryDialog.close(); recoveryBody.replaceChildren(); recoveryButton.hidden = true;
    $('message').value = '';
    document.body.classList.remove('history-open');
    $('history-toggle').setAttribute('aria-expanded', 'false');
    // Same document, same renderer clock. DeliveryTiming retains old rows and
    // binds the next send to this newly selected session.
    adoptState(await api.state());
    void action(refresh); $('message').focus();
  } finally { sending = false; showState(lastState); }
});
$('composer').onsubmit = event => { event.preventDefault(); if ($('send').disabled) return; action(async () => {
  const submittedText = $('message').value;
  if (sending || !submittedText.trim()) return;
  manualReading = false;
  const timingId = deliveryTiming.begin(sessionId);
  sending = true; $('send').disabled = true;
  try {
    deliveryTiming.bind(timingId, await api.send(submittedText));
    if ($('message').value === submittedText) $('message').value = '';
    void action(refresh);
  } catch (reason) { deliveryTiming.reject(timingId); throw reason; }
  finally { sending = false; adoptState(await api.state()); }
}); };
$('message').onkeydown = event => { if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); $('composer').requestSubmit(); } };
$('stop').onclick = () => action(() => {
  cancelledDeliveryAttempt = lastState.attempt?.attempt_id;
  return api.stop();
});
$('retry').onclick = () => action(async () => {
  manualReading = false;
  const timingId = deliveryTiming.begin(sessionId, 'retry');
  try { deliveryTiming.bind(timingId, await api.retry()); }
  catch (reason) { deliveryTiming.reject(timingId); throw reason; }
});
async function openFeedback(message, target) {
  const targetSession = sessionId;
  feedbackTarget = { message, target }; feedbackKey = crypto.randomUUID();
  const events = (await api.feedback({ operation: 'list' })).events;
  if (targetSession !== sessionId) return;
  feedbackEvent = events.find(event => {
    if (event.session_id !== targetSession || event.turn_id !== message.turn_id) return false;
    const bound = event.target_binding;
    if (target.kind === 'attempt') {
      return bound?.kind === 'attempt' && bound.attempt?.text_attempt_id === target.attempt_id;
    }
    if (!bound) return target.kind === 'reply' && event.assistant_message_id === message.id;
    return event.assistant_message_id === message.id && bound.kind === target.kind
      && bound.range?.start === target.start && bound.range?.end === target.end;
  });
  $('feedback-text').value = feedbackEvent?.freeform_reaction || '';
  for (const input of document.querySelectorAll('[name="quick_tags"]')) input.checked = !!feedbackEvent?.quick_tags.includes(input.value);
  $('feedback-title').textContent = target.kind === 'attempt' ? '这次尝试的反馈'
    : target.kind === 'bubble' ? '这段回复的反馈' : '这条回复的反馈';
  const targetStatus = feedbackEvent?.target_binding?.status;
  const scopeStatus = targetStatus === 'unknown' ? '送达可见状态未知。'
    : targetStatus === 'attempt-only' ? '只记录尝试标识；未显示的文字不会进入导出。' : '';
  $('feedback-status').textContent = feedbackEvent
    ? `已保存 · 修订 ${feedbackEvent.feedback_revision}${scopeStatus ? ` · ${scopeStatus}` : ''}`
    : target.kind === 'attempt'
      ? '只记录这次尝试；未显示的文字不会进入反馈或导出。反馈不会进入下一轮对话或角色记忆。'
      : '反馈只在本机保存，不会进入下一轮对话或角色记忆。';
  for (const id of ['feedback-withdraw','feedback-delete','feedback-export']) $(id).hidden = !feedbackEvent;
  previewId = previewRevision = previewHash = undefined; previewApplied = false;
  $('export-panel').hidden = true; $('export-confirmation').checked = false;
  $('export-preview').value = ''; $('export-preview-meta').textContent = '';
  $('export-save').disabled = true;
  $('feedback-dialog').showModal();
}
async function feedbackAction(callback) { try { await callback(); } catch { $('feedback-status').textContent = '操作未完成，请检查内容或重新打开反馈后重试。'; } }
$('feedback-close').onclick = () => $('feedback-dialog').close();
$('feedback-form').onsubmit = event => { event.preventDefault(); return feedbackAction(async () => {
  const content = $('feedback-text').value;
  const tags = [...document.querySelectorAll('[name="quick_tags"]:checked')].map(input => input.value);
  if (!content.trim() && !tags.length) { $('feedback-status').textContent = '请选一个标签，或写下一点你的感受。'; return; }
  const payload = feedbackEvent ? { operation: 'revise', feedback_id: feedbackEvent.feedback_id, expected_revision: feedbackEvent.feedback_revision }
    : { operation: 'submit', session_id: sessionId, turn_id: feedbackTarget.message.turn_id,
      ...(feedbackTarget.target.kind === 'attempt'
        ? { target: feedbackTarget.target }
        : { assistant_message_id: feedbackTarget.message.id, target: feedbackTarget.target }),
      idempotency_key: feedbackKey };
  feedbackEvent = await api.feedback({ ...payload, quick_tags: tags, freeform_reaction: content });
  previewId = previewRevision = previewHash = undefined; previewApplied = false;
  $('export-panel').hidden = true; $('export-confirmation').checked = false; $('export-save').disabled = true;
  $('feedback-status').textContent = `已保存 · 修订 ${feedbackEvent.feedback_revision}`;
  for (const id of ['feedback-withdraw','feedback-delete','feedback-export']) $(id).hidden = false;
}); };
for (const operation of ['withdraw','delete']) $(`feedback-${operation}`).onclick = () => feedbackAction(async () => {
  const payload = { operation, feedback_id: feedbackEvent.feedback_id };
  if (operation === 'withdraw') payload.expected_revision = feedbackEvent.feedback_revision;
  await api.feedback(payload); feedbackEvent = undefined; feedbackKey = crypto.randomUUID();
  $('feedback-status').textContent = operation === 'withdraw' ? '已撤销' : '已删除';
  for (const id of ['feedback-withdraw','feedback-delete','feedback-export']) $(id).hidden = true;
  previewId = previewRevision = previewHash = undefined; previewApplied = false;
  $('export-preview').value = ''; $('export-preview-meta').textContent = '';
  $('export-confirmation').checked = false; $('export-save').disabled = true;
  $('export-panel').hidden = true;
});
$('feedback-export').onclick = () => feedbackAction(async () => {
  const preview = await api.feedback({ operation: 'preview', feedback_ids: [feedbackEvent.feedback_id] });
  previewId = preview.preview_id; previewRevision = preview.preview_revision;
  previewHash = preview.payload_sha256; previewApplied = false;
  $('export-preview').value = JSON.stringify(preview.payload, null, 2);
  $('export-preview-meta').textContent = `预览修订 ${previewRevision} · SHA-256 ${previewHash} · 先应用脱敏，再确认`;
  $('export-confirmation').checked = false; $('export-save').disabled = true;
  $('export-panel').hidden = false;
});
$('export-preview').oninput = () => {
  previewApplied = false; $('export-confirmation').checked = false; $('export-save').disabled = true;
  if (previewRevision && previewHash) {
    $('export-preview-meta').textContent = `预览修订 ${previewRevision} · SHA-256 ${previewHash} · 修改尚未应用`;
  }
};
$('export-apply').onclick = () => feedbackAction(async () => {
  if (!previewId || !previewRevision || !previewHash) throw new Error('Missing preview binding');
  const encoded = new TextEncoder().encode($('export-preview').value);
  if (encoded.byteLength > 38000) {
    $('feedback-status').textContent = '当前预览超过安全请求上限，请先删减更多内容后再应用。'; return;
  }
  let candidate;
  try { candidate = JSON.parse($('export-preview').value); }
  catch { $('feedback-status').textContent = '预览不是有效 JSON；修正后再应用。'; return; }
  const updated = await api.feedback({ operation: 'preview', preview_id: previewId,
    expected_revision: previewRevision, expected_sha256: previewHash, redacted_payload: candidate });
  previewRevision = updated.preview_revision; previewHash = updated.payload_sha256;
  previewApplied = true; $('export-preview').value = JSON.stringify(updated.payload, null, 2);
  $('export-confirmation').checked = false; $('export-save').disabled = true;
  $('export-preview-meta').textContent = `预览修订 ${previewRevision} · SHA-256 ${previewHash} · 已应用；请核对后勾选确认`;
  $('feedback-status').textContent = '脱敏已应用。只有当前显示的修订和哈希会被确认保存。';
});
$('export-confirmation').onchange = () => {
  $('export-save').disabled = !previewApplied || !$('export-confirmation').checked;
};
$('export-save').onclick = () => feedbackAction(async () => {
  if (!previewApplied || !$('export-confirmation').checked || !previewId || !previewRevision || !previewHash) {
    $('feedback-status').textContent = '请应用并检查当前预览，再确认处理范围。'; return;
  }
  await api.feedback({ operation: 'confirm', preview_id: previewId,
    preview_revision: previewRevision, payload_sha256: previewHash,
    redaction_confirmed: true, external_processing_confirmed: true });
  $('feedback-status').textContent = '已保存到本地 exports 目录，未上传。'; $('export-panel').hidden = true;
});
async function poll() { await action(refresh); setTimeout(poll, 180); }
async function pollDelivery() {
  let result;
  await action(async () => { result = await pollApproved(); });
  // Only a durable ACK advances directly to the next suffix. An empty poll
  // may have returned immediately after cancellation, so it still backs off.
  const ready = result?.status === 'confirmed' && result.receipts?.length;
  setTimeout(pollDelivery, ready ? 0 : 180);
}
pollDelivery();
poll();
