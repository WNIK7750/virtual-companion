import test from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'node:fs';
import { mkdtempSync, rmSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { StreamReceiptJournal, digest } from '../src/stream-receipts.mjs';

import { event, spanned } from './stream-fixture.mjs';

function temporary(fn) {
  return async () => {
    const directory = mkdtempSync(join(tmpdir(), 'vc-stream-receipts-'));
    try { await fn(directory); } finally { rmSync(directory, { recursive: true }); }
  };
}

test('faithful spans are accepted and broken ranges are rejected', temporary(async directory => {
  const journal = new StreamReceiptJournal(directory);
  const faithful = spanned();
  assert.equal(faithful.presentation.candidate.source_text, '甲句。乙句。\n\n丙句。');
  assert.equal(faithful.text, '丙句。');
  await journal.present(faithful, async () => true);
  assert.equal(journal.lookup(faithful).status, 'confirmed');

  for (const breakIt of [
    value => { value.presentation.candidate.units[1].start = 4; },
    value => { value.presentation.candidate.units[1].text = '改写'; },
    value => { value.presentation.candidate.units.pop(); },
    value => { value.presentation.candidate.source_text = '别的文本'; },
  ]) {
    const changed = structuredClone(faithful);
    breakIt(changed);
    const { fingerprint, status, host_receipt, text, ...manifest } = changed;
    changed.fingerprint = digest(manifest);
    await assert.rejects(journal.present(changed, async () => assert.fail('must not render')));
  }
  const legacyWithSpans = event();
  legacyWithSpans.presentation.candidate.units[0].start = 0;
  legacyWithSpans.presentation.candidate.units[0].end = 3;
  legacyWithSpans.presentation.review.candidate_sha256 = digest(legacyWithSpans.presentation.candidate);
  const { fingerprint: _, status: __, host_receipt: ___, text: ____, ...legacyManifest } = legacyWithSpans;
  legacyWithSpans.fingerprint = digest(legacyManifest);
  await assert.rejects(journal.present(legacyWithSpans, async () => assert.fail('no spans in joined')),
    /Joined display/);
}));

test('durable visible receipt survives reopen and never renders twice', temporary(async directory => {
  const value = event(); let renders = 0;
  const journal = new StreamReceiptJournal(directory);
  const receipt = await journal.present(value, async () => { renders++; return true; });
  const reopened = new StreamReceiptJournal(directory);
  assert.equal(await reopened.present(value, async () => { renders++; return true; }), receipt);
  assert.equal(renders, 1);
  assert.equal(reopened.lookup(value).status, 'confirmed');
  for (const file of readdirSync(directory)) assert.ok(!readFileSync(join(directory, file), 'utf8').includes('你好'));
}));

test('render failure after durable intent stays unknown across reopen', temporary(async directory => {
  const value = event(); const journal = new StreamReceiptJournal(directory);
  await assert.rejects(journal.present(value, async () => { throw new Error('renderer exited'); }));
  const reopened = new StreamReceiptJournal(directory);
  assert.equal(reopened.lookup(value).status, 'unknown');
  assert.equal(await reopened.present(value, async () => { assert.fail('must not replay'); }), null);
}));

test('concurrent duplicate has one renderer and no inferred ACK', temporary(async directory => {
  const value = event(); const first = new StreamReceiptJournal(directory);
  const second = new StreamReceiptJournal(directory);
  let release; const wait = new Promise(resolve => { release = resolve; });
  const pending = first.present(value, async () => { await wait; return true; });
  assert.equal(await second.present(value, async () => assert.fail('duplicate render')), null);
  release(); await pending;
  assert.equal(second.lookup(value).status, 'confirmed');
}));

test('wrong content and same identity with different approval are rejected', temporary(async directory => {
  const value = event(); const journal = new StreamReceiptJournal(directory);
  await assert.rejects(journal.present({ ...value, text: '伪造' }, async () => true));
  await journal.present(value, async () => true);
  const changed = structuredClone(value);
  changed.presentation.review.review_id = 'other-review';
  const { fingerprint, status, host_receipt, text, ...manifest } = changed;
  changed.fingerprint = digest(manifest);
  await assert.rejects(journal.present(changed, async () => assert.fail('identity conflict')));
}));

test('receipt sync or rename failure after render remains unknown on restart', temporary(async directory => {
  for (const operation of ['fsyncSync', 'renameSync']) {
    const subdir = join(directory, operation); let syncs = 0; let renders = 0;
    const journal = new StreamReceiptJournal(subdir, { io: { ...fs,
      fsyncSync(fd) { if (operation === 'fsyncSync' && ++syncs === 2) throw new Error('disk fault'); fs.fsyncSync(fd); },
      renameSync(from, to) { if (operation === 'renameSync') throw new Error('rename fault'); fs.renameSync(from, to); },
    } });
    await assert.rejects(journal.present(event(), async () => { renders++; return true; }));
    const reopened = new StreamReceiptJournal(subdir);
    assert.equal(reopened.lookup(event()).status, 'unknown');
    assert.equal(await reopened.present(event(), async () => { renders++; return true; }), null);
    assert.equal(renders, 1);
  }
}));

test('corrupt intent is not silently reset and cancellation never invents visibility', temporary(async directory => {
  const journal = new StreamReceiptJournal(directory); const cancelled = new AbortController();
  cancelled.abort();
  assert.equal(await journal.present(event(), async () => assert.fail('cancelled'), { signal: cancelled.signal }), null);
  assert.equal(journal.lookup(event()).status, 'unseen');
  await journal.present(event(), async () => false);
  const intent = readdirSync(directory).find(name => name.endsWith('.intent.json'));
  writeFileSync(join(directory, intent), '{incomplete');
  await assert.rejects(journal.present(event(), async () => assert.fail('corrupt must not replay')));
}));


test('only explicit recovery may redisplay an unknown unit, keeping its original receipt identity', temporary(async directory => {
  const value = event(); const journal = new StreamReceiptJournal(directory);
  await journal.present(value, async () => false);
  const intentFile = readdirSync(directory).find(name => name.endsWith('.intent.json'));
  const original = JSON.parse(readFileSync(join(directory, intentFile), 'utf8'));
  const reopened = new StreamReceiptJournal(directory);
  assert.equal(await reopened.present(value, async () => assert.fail('automatic replay')), null);
  assert.equal(await reopened.resume(value, async () => false), null);
  assert.equal(reopened.lookup(value).status, 'unknown');
  const receipt = await reopened.resume(value, async () => true);
  assert.equal(receipt, original.receipt_id);
  assert.equal(await reopened.resume(value, async () => assert.fail('already visible')), receipt);
}));

test('recovery requires a fresh actual display after a pending receipt write failure', temporary(async directory => {
  const value = event();
  const broken = new StreamReceiptJournal(directory, { io: { ...fs, renameSync() { throw new Error('disk fault'); } } });
  await assert.rejects(broken.present(value, async () => true));
  const reopened = new StreamReceiptJournal(directory);
  assert.equal(reopened.lookup(value).status, 'unknown');
  let renders = 0;
  const receipt = await reopened.resume(value, async () => { renders++; return true; });
  assert.equal(renders, 1); assert.equal(reopened.lookup(value).receipt, receipt);
}));
