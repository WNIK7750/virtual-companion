import { _electron as electron } from 'playwright';
import assert from 'node:assert/strict';
import { mkdir, writeFile, mkdtemp, readFile, readdir } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { tmpdir } from 'node:os';
import { execFileSync } from 'node:child_process';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { setTimeout as delay } from 'node:timers/promises';
import { assertSyntheticBackendEnvironment, createSyntheticBackendEnvironment } from './synthetic-env.mjs';
import { assertChatLayout, measureChatLayout } from './chat-layout-proof.mjs';

const desktopRoot = fileURLToPath(new URL('..', import.meta.url));
const output = resolve(process.env.VC_QA_DIR || resolve(desktopRoot, '../../.local/e2e'));
await mkdir(output, { recursive: true });
const dataRoot = await mkdtemp(resolve(tmpdir(), 'vc-public-e2e-'));
const parentEnv = Object.fromEntries(Object.entries(process.env).filter(([key]) =>
  !/^VC_/i.test(key) && key.toUpperCase() !== 'ELECTRON_RUN_AS_NODE'));
const environment = createSyntheticBackendEnvironment({ dataRoot, parentEnv });
const syntheticReply = '这是开发模式的合成回复。真实模型尚未参与这段对话。';
const scriptSha256 = createHash('sha256').update(await readFile(fileURLToPath(import.meta.url))).digest('hex');
const readTranscript = page => page.locator('#messages .message').evaluateAll(nodes => nodes.map(node => ({
  role: node.classList.contains('assistant') ? 'assistant' : 'user',
  turn_id: node.dataset.turnId,
  content: node.querySelector('.message-body').textContent,
})));
async function waitForCondition(check, label) {
  const deadline = Date.now() + 20000;
  while (Date.now() < deadline) {
    const result = await check();
    if (result) return result;
    await delay(100);
  }
  throw new Error(`Timed out waiting for ${label}`);
}
const waitForReplies = (page, count) => waitForCondition(() => page.evaluate(async ({ count, reply }) => {
  const state = await window.companion.state();
  const history = await window.companion.history();
  const replies = history.messages.filter(message => message.role === 'assistant');
  return !state.busy && replies.length === count && replies.every(message => message.content === reply);
}, { count, reply: syntheticReply }), `${count} completed synthetic replies`);
const launch = () => {
  assertSyntheticBackendEnvironment(environment);
  return electron.launch({ args: ['.', '--fake'], cwd: desktopRoot, env: environment, timeout: 30000 });
};
let app;
const errors = [];
try {
  app = await launch();
  const page = await app.firstWindow();
  page.setDefaultTimeout(20000);
  assert.equal(await app.evaluate(({ app }) => app.getPath('userData')), environment.VC_USER_DATA_ROOT);
  page.on('pageerror', error => errors.push(error.message));
  await page.waitForSelector('#status');
  await page.waitForFunction(() => document.querySelector('#status').textContent.includes('开发模式'));
  assert.equal(await page.title(), '文字伴侣 · 文字空间');
  assert.equal(page.url(), 'companion://app/index.html');
  assert.equal(await page.evaluate(() => typeof window.require), 'undefined');
  const initialLayout = assertChatLayout(await page.evaluate(measureChatLayout));
  assert.equal(await page.locator('#voice-slot .component-state').textContent(), '未启用');
  const initialState = await page.evaluate(() => window.companion.state());
  assert.equal(initialState.mode, 'fake');
  assert.equal(initialState.reliable_text_receive, true);
  assert.equal(initialState.provider.active.provider_id, 'local');
  assert.equal(initialState.provider.active.status, 'active');
  await page.locator('#provider-open').click();
  const provider = initialState.provider;
  assert.equal(await page.locator('.provider-option').count(), provider.options.length);
  assert.ok((await page.locator('#provider-active').textContent()).includes(provider.active.label));
  const remoteOptions = provider.options.filter(option => option.provider_id !== 'local');
  assert.ok(remoteOptions.length > 0, 'the remote route remains visible but unavailable in the offline fixture');
  for (const option of remoteOptions) {
    assert.equal(option.selectable, false);
    assert.equal(option.network_enabled, false);
    const row = page.locator('.provider-option').filter({ hasText: `${option.label} · ${option.model_label}` });
    assert.equal(await row.count(), 1);
    assert.equal(await row.locator('button').isDisabled(), true);
    assert.ok((await row.textContent()).includes('没有发送请求'));
  }
  assert.equal(await page.locator('#provider-label').textContent(), `${provider.active.label} · 当前`);
  await page.locator('#provider-close').click();
  // Real content creates scrolling; never shrink the chat viewport to fake unread behavior.
  const firstMessage = '合成窗口测试\n' + Array.from({ length: 28 }, (_, index) =>
    `测试段落 ${index + 1}：仅用于验证正常窗口中的连续阅读位置。`).join('\n');
  await page.locator('#message').fill(firstMessage);
  await page.locator('#send').click();
  await waitForReplies(page, 1);
  await page.waitForFunction(reply => document.querySelector('#messages .assistant .message-body')?.textContent === reply, syntheticReply);
  assert.equal(await page.locator('#messages .message').count(), 2);
  assert.equal(await page.locator('#messages .assistant .message-bubble').count(), 1);
  assert.equal(await page.locator('#error').isVisible(), false);
  assertChatLayout(await page.evaluate(measureChatLayout), { previous: initialLayout });
  await page.locator('.message-feedback').first().click();
  await page.locator('#feedback-text').fill('合成反馈：这句很清楚。');
  await page.locator('#feedback-save').click();
  await page.waitForFunction(() => document.querySelector('#feedback-status').textContent.includes('修订 1'));
  await page.locator('#feedback-export').click();
  assert.equal(await page.locator('#export-save').isDisabled(), true);
  await page.locator('#export-apply').click();
  await page.waitForFunction(() => document.querySelector('#export-preview-meta').textContent.includes('已应用'));
  await page.locator('#export-confirmation').check();
  await page.locator('#export-save').click();
  await page.waitForFunction(() => document.querySelector('#feedback-status').textContent.includes('未上传'));
  assert.equal((await readdir(resolve(dataRoot, 'exports'))).filter(name => name.endsWith('.json')).length, 1);
  await page.locator('#feedback-text').fill('');
  await page.locator('[name="quick_tags"][value="too_verbose"]').check();
  await page.locator('#feedback-save').click();
  await page.waitForFunction(() => document.querySelector('#feedback-status').textContent.includes('修订 2'));
  await page.locator('#feedback-delete').click();
  await page.waitForFunction(() => document.querySelector('#feedback-status').textContent.includes('已删除'));
  await page.locator('#feedback-save').click();
  await page.waitForFunction(() => document.querySelector('#feedback-status').textContent.includes('修订 1'));
  await page.locator('#feedback-withdraw').click();
  await page.waitForFunction(() => document.querySelector('#feedback-status').textContent.includes('已撤销'));
  await page.locator('#feedback-close').click();
  // Observe native show calls without replacing their implementation. This is
  // a host-level suppression check, not proof of physical OS toast behavior.
  await app.evaluate(({ Notification }) => {
    globalThis.syntheticNotificationShows = 0;
    const show = Notification.prototype.show;
    Notification.prototype.show = function (...args) {
      globalThis.syntheticNotificationShows += 1;
      return show.apply(this, args);
    };
  });
  await page.locator('#message').fill('合成安静接收测试：这条回复应继续送达。');
  await page.locator('#send').click();
  await waitForCondition(() => page.evaluate(async () => (await window.companion.state()).busy), 'active generation');
  const legacyStopVisibleDuringGeneration = await page.locator('#stop').isVisible();
  await page.locator('#quiet-receiving').click();
  await page.waitForFunction(() => document.querySelector('#quiet-receiving').getAttribute('aria-pressed') === 'true');
  const readingTop = await page.locator('#messages').evaluate(element => {
    if (element.scrollHeight <= element.clientHeight) throw new Error('Synthetic conversation must be scrollable');
    element.scrollTop = 0;
    return element.scrollTop;
  });
  await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].hide());
  // Receiving unseen text is distinct from confirming it in visible history.
  const quietState = await waitForCondition(() => page.evaluate(async () => {
    const state = await window.companion.state();
    return !state.busy && state.received_pending > 0 ? state : null;
  }), 'durable unread reception');
  assert.equal(quietState.receiving_preferences.quiet, true);
  assert.equal(quietState.attempt.status, 'received');
  assert.equal(quietState.received_replies, 1);
  assert.ok(quietState.received_pending > 0);
  const receivedDirectory = resolve(environment.VC_USER_DATA_ROOT, 'stream-receipts');
  const receivedRows = await Promise.all((await readdir(receivedDirectory))
    .filter(name => name.endsWith('.received.json'))
    .map(async name => JSON.parse(await readFile(resolve(receivedDirectory, name), 'utf8'))));
  const currentReceived = receivedRows.filter(row => row.scope.turn_id === quietState.attempt.turn_id)
    .sort((left, right) => left.event.sequence - right.event.sequence);
  assert.equal(currentReceived.at(-1)?.event.display_text, syntheticReply);
  assert.equal(await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].isVisible()), false);
  assert.equal(await app.evaluate(() => globalThis.syntheticNotificationShows), 0);
  await app.evaluate(({ BrowserWindow }) => { const window = BrowserWindow.getAllWindows()[0]; window.show(); window.focus(); });
  await page.waitForFunction(() => document.querySelectorAll('#messages .assistant').length === 2 && !document.querySelector('#received-status').hidden);
  assert.equal(await page.locator('#messages').evaluate(element => element.scrollTop), readingTop);
  assert.equal(await page.locator('#received-status').textContent(), '1 条新回复 · 查看');
  const quietLayout = assertChatLayout(await page.evaluate(measureChatLayout), { previous: initialLayout });
  await page.screenshot({ path: resolve(output, 'quiet-unread.png') });
  await page.locator('#received-status').click();
  await waitForCondition(() => page.evaluate(async () => (await window.companion.state()).received_pending === 0), 'visible reply acknowledgement');
  await waitForReplies(page, 2);
  await page.waitForFunction(() => document.querySelector('#received-status').hidden);
  assert.ok(await page.locator('#messages').evaluate(element => element.scrollTop > 0));
  assert.equal(await page.locator('#messages .message').count(), 4);
  assert.equal(await page.locator('#messages .assistant .message-bubble').count(), 2);
  assert.equal(await page.locator('#retry').isVisible(), false);
  assert.equal(await page.locator('#error').isVisible(), false);
  const transcript = await readTranscript(page);
  await page.screenshot({ path: resolve(output, 'desktop.png') });
  await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0].setSize(560, 640));
  await page.waitForFunction(() => window.innerWidth <= 560 && window.innerHeight <= 640);
  assertChatLayout(await page.evaluate(measureChatLayout));
  await page.screenshot({ path: resolve(output, 'compact.png') });
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
  assert.deepEqual(errors, []);
  const preferences = await app.evaluate(({ BrowserWindow }) => {
    const p = BrowserWindow.getAllWindows()[0].webContents.getLastWebPreferences();
    return { sandbox: p.sandbox, contextIsolation: p.contextIsolation, nodeIntegration: p.nodeIntegration };
  });
  assert.deepEqual(preferences, { sandbox: true, contextIsolation: true, nodeIntegration: false });
  await app.close();
  app = await launch();
  const restored = await app.firstWindow();
  restored.setDefaultTimeout(20000);
  restored.on('pageerror', error => errors.push(error.message));
  await restored.waitForFunction(() => document.querySelectorAll('#messages .message').length === 4);
  assert.deepEqual(await readTranscript(restored), transcript);
  assert.equal(await restored.locator('#quiet-receiving').getAttribute('aria-pressed'), 'true');
  await restored.locator('#quiet-receiving').click();
  await restored.waitForFunction(() => document.querySelector('#quiet-receiving').getAttribute('aria-pressed') === 'false');
  assertChatLayout(await restored.evaluate(measureChatLayout));
  assert.deepEqual(errors, []);
  await writeFile(resolve(output, 'desktop-receipt.json'), JSON.stringify({
    tested_commit: execFileSync('git', ['-C', desktopRoot, 'rev-parse', 'HEAD'], { encoding: 'utf8', windowsHide: true }).trim(),
    script_sha256: scriptSha256, data_root: dataRoot, title: await restored.title(), preferences, errors,
    interaction: 'send -> actual fake bubble -> feedback -> apply export preview -> confirmed local export -> tags-only revision -> delete -> resubmit -> withdraw -> enable quiet during generation -> receive while hidden -> preserve reading position -> inspect unread reply -> restart -> exact transcript and quiet preference restored -> disable quiet',
    native_notification_show_calls_while_quiet: 0, legacy_stop_visible_during_generation: legacyStopVisibleDuringGeneration,
    limits: ['Synthetic model only; no dialogue-quality acceptance.', 'Native show-call suppression does not verify physical system notifications.', 'Legacy stop control remains present; this script does not change product behavior.'],
    layouts: { normal: initialLayout, quiet_unread: quietLayout }, window_sizes: ['1320x860', '560x640'],
  }, null, 2));
  console.log('Electron synthetic vertical E2E passed');
} catch (error) {
  const current = app?.windows()[0];
  await writeFile(resolve(output, 'desktop-failure.json'), JSON.stringify({
    status: 'failed', error: error.message, stack: error.stack, data_root: environment.VC_DATA_ROOT,
    title: current ? await current.title() : null,
    visible_text: current ? await current.locator('body').innerText() : null,
    errors,
  }, null, 2));
  if (current) {
    try {
      await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0]?.show());
      await current.screenshot({ path: resolve(output, 'failure.png'), timeout: 5000 });
    } catch { /* A diagnostic screenshot must not replace the original failure. */ }
  }
  throw error;
} finally { if (app) await app.close(); }
