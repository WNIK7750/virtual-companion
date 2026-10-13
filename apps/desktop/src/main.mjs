import { app, BrowserWindow, Notification, ipcMain, protocol, session, net } from 'electron';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, join, resolve } from 'node:path';
import { windowOptions, createShellHandlers } from './shell.mjs';
import { Backend } from './backend.mjs';
import { StreamReceiptJournal } from './stream-receipts.mjs';
import { createStreamPresenter } from './stream-presenter.mjs';
import { StreamDeliveryClient } from './stream-delivery-client.mjs';
import { createReceivingContextSendAdapter, ReceivingPreferences } from './receiving-preferences.mjs';
import { ReplyNotifications, NotificationNavigation } from './reply-notifications.mjs';

const root = dirname(fileURLToPath(import.meta.url));
protocol.registerSchemesAsPrivileged([{ scheme: 'companion', privileges: { standard: true, secure: true, supportFetchAPI: true } }]);
app.setPath('userData', process.env.VC_USER_DATA_ROOT ? resolve(process.env.VC_USER_DATA_ROOT) : join(app.getPath('appData'), 'VirtualCompanion-Dogfood'));
let win;
const fake = process.argv.includes('--fake');
const backend = new Backend(join(root, '../../..'), { fake });
let quitting = false;
let streamPresenter;
let streamDelivery;
let receivingTimer;
let receivingPreferences;
let replyNotifications;
let notificationNavigation;
let selectedSession;
const sendWithReceivingContext = createReceivingContextSendAdapter({
  getPreferences: () => receivingPreferences,
  call: (action, payload) => backend.call(action, payload),
});
async function currentState() {
  const state = await backend.state(); selectedSession = state.session_id; return state;
}
const service = {
  presentApproved: async () => {
    if (backend.status !== 'ready' || !streamDelivery) return { status: 'idle' };
    const result = await streamDelivery.pulse(currentState, { waitMs: 1000 });
    const receipts = streamDelivery.takeReceipts();
    return { ...result, receipts };
  },
  resumeApproved: ({ fingerprint }) => streamDelivery.resume(fingerprint, currentState),
  cancelRecovery: () => { streamDelivery?.cancelRecovery(); return { status: 'closed' }; },
  state: async () => {
    const state = await currentState();
    const pending = streamPresenter?.pendingReceived(state.session_id) || [];
    return { ...state, receiving_preferences: receivingPreferences?.snapshot(),
      received_pending: pending.length,
      received_replies: new Set(pending.map(row => row.scope.turn_id)).size };
  },
  setQuiet: ({ quiet }) => {
    const value = receivingPreferences.setQuiet(quiet);
    if (quiet) replyNotifications?.dismiss(); return value;
  },
  start: () => backend.start(),
  history: payload => backend.status === 'ready' ? backend.call('history', payload) : { messages: [] },
  newSession: () => backend.call('newSession'), send: sendWithReceivingContext,
  stop: () => backend.call('stop'), retry: () => backend.call('retry'),
  resumeInputs: () => backend.call('resumeInputs'),
  selectProvider: payload => backend.call('selectProvider', payload),
  feedback: payload => backend.call('feedback', payload),
  memories: payload => backend.call('memories', payload),
  memoryDetails: payload => backend.call('memoryDetails', payload),
  memoryAction: payload => backend.call('memoryAction', payload),
  memoryAbandon: payload => backend.call('memoryAbandon', payload),
  sessions: payload => backend.status === 'ready' ? backend.call('sessions', payload) : { sessions: [] },
  selectSession: payload => backend.call('selectSession', payload),
};

if (!app.requestSingleInstanceLock()) app.quit();
else {
  app.on('second-instance', () => { if (win?.isMinimized()) win.restore(); win?.focus(); });
  app.whenReady().then(async () => {
    if (process.platform === 'win32') app.setAppUserModelId('VirtualCompanion.Dogfood');
    receivingPreferences = new ReceivingPreferences(join(app.getPath('userData'), 'receiving-preferences.json'));
    const assets = new Set(['index.html', 'renderer.mjs', 'memory-panel.mjs', 'provider-panel.mjs', 'message-list.mjs', 'approved-display.mjs', 'icons.mjs', 'theme.mjs', 'style.css']);
    assets.add('delivery-timing.mjs');
    assets.add('approved-visibility.mjs');
    protocol.handle('companion', request => {
      const url = new URL(request.url);
      const name = url.pathname.slice(1);
      if (url.host !== 'app' || !assets.has(name) || request.method !== 'GET') return new Response('', { status: 404 });
      return net.fetch(pathToFileURL(join(root, name)).href);
    });
    session.defaultSession.setPermissionRequestHandler((_wc, _permission, done) => done(false));
    session.defaultSession.setPermissionCheckHandler(() => false);
    session.defaultSession.webRequest.onBeforeRequest((details, done) => {
      done({ cancel: !details.url.startsWith('companion://app/') && !details.url.startsWith('file://') });
    });
    win = new BrowserWindow(windowOptions(join(root, 'preload.cjs')));
    // loadURL follows the backend's configuration/authority initialization.
    // Native first-paint readiness is an additional gate before showing it.
    win.once('ready-to-show', () => { if (!quitting && !win.isDestroyed()) win.show(); });
    win.webContents.on('will-navigate', event => event.preventDefault());
    win.webContents.on('will-attach-webview', event => event.preventDefault());
    win.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));
    const journal = new StreamReceiptJournal(join(app.getPath('userData'), 'stream-receipts'));
    notificationNavigation = new NotificationNavigation({ getState: currentState,
      selectSession: session_id => backend.call('selectSession', { session_id }),
      showTarget: target => win.webContents.send('companion:notification-target', target),
      focus: () => { if (win.isMinimized()) win.restore(); win.show(); win.focus(); } });
    replyNotifications = new ReplyNotifications({ supported: () => Notification.isSupported(),
      create: options => new Notification(options),
      context: () => ({ quiet: receivingPreferences.snapshot().quiet,
        focused: win.isFocused(), session_id: selectedSession }),
      activate: target => notificationNavigation.open(target),
      receivedScopes: [...journal.receivedRecords().values()].map(row => row.scope) });
    streamPresenter = createStreamPresenter({ window: win, ipcMain,
      journal,
      allowedUrl: 'companion://app/index.html' });
    streamDelivery = new StreamDeliveryClient({ call: (action, payload) => backend.call(action, payload),
      presenter: streamPresenter, onReceived: scope => replyNotifications.receive(scope) });
    const handle = createShellHandlers(win.webContents, service);
    ipcMain.handle('companion:action', async (...args) => {
      try { return { ok: true, value: await handle(...args) }; }
      catch (error) { return { ok: false, error: error.message }; }
    });
    await backend.start(false);
    await win.loadURL('companion://app/index.html');
    // Receiving is a host responsibility, including while the native window
    // is hidden. Renderer polling remains a consumer of actual visible ACKs.
    receivingTimer = setInterval(() => {
      if (backend.status === 'ready') {
        streamDelivery.pulse(currentState).catch(error => { console.error('Text receipt pump:', error.message); });
        notificationNavigation.flush().catch(error => { console.error('Notification navigation:', error.message); });
      }
    }, 180);
  }).catch(error => { console.error(error.message); app.exit(1); });
  app.on('window-all-closed', () => app.quit());
  app.on('before-quit', event => {
    if (quitting) return;
    event.preventDefault(); quitting = true;
    clearInterval(receivingTimer);
    replyNotifications?.dismiss();
    streamPresenter?.dispose();
    backend.close().finally(() => app.quit());
  });
}
