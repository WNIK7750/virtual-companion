import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createStreamPresenter } from '../src/stream-presenter.mjs';
import { StreamReceiptJournal } from '../src/stream-receipts.mjs';
import { event as streamEvent } from './stream-fixture.mjs';

test('a window unavailable before IPC does not consume the one permitted presentation', async () => {
  const directory = mkdtempSync(join(tmpdir(), 'presenter-ready-'));
  const journal = new StreamReceiptJournal(directory);
  const ipcMain = new EventEmitter(), frame = { url: 'companion://app/index.html' };
  let visible = false, minimized = false, sent;
  const webContents = { mainFrame: frame, getURL: () => frame.url, isDestroyed: () => false,
    send: (_channel, payload) => { sent = payload; } };
  const window = { webContents, isDestroyed: () => false, isVisible: () => visible, isMinimized: () => minimized };
  const presenter = createStreamPresenter({ window, ipcMain, journal, allowedUrl: frame.url, timeoutMs: 50 });
  const event = streamEvent();
  try {
    assert.ok(!await presenter.present(event));
    assert.equal(sent, undefined);
    assert.equal(journal.lookup(event).status, 'unseen');
    visible = true; minimized = true;
    assert.ok(!await presenter.present(event));
    assert.equal(journal.lookup(event).status, 'unseen');
    minimized = false;
    const waiting = presenter.present(event);
    assert.equal(journal.lookup(event).status, 'unknown');
    assert.ok(sent);
    // Once IPC has been sent, a missed ACK must still forbid automatic replay.
    assert.equal(await waiting, null);
    sent = undefined;
    assert.equal(await presenter.present(event), null);
    assert.equal(sent, undefined);
    assert.equal(journal.lookup(event).status, 'unknown');
  } finally { presenter.dispose(); rmSync(directory, { recursive: true, force: true }); }
});
