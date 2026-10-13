import { tmpdir } from 'node:os';
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { mkdtempForTest } from './fixture-lifecycle.mjs';
import { createReceivingContextSendAdapter, ReceivingPreferences } from '../src/receiving-preferences.mjs';

const fixtureRoot = tmpdir();

test('quiet receiving preference defaults off and persists both changes after reopen', async t => {
  const directory = await mkdtempForTest(t, resolve(fixtureRoot, 'receiving-preferences-'));
  const path = join(directory, 'preferences.json');
  const defaults = new ReceivingPreferences(path);
  assert.equal(defaults.quiet, false);

  assert.deepEqual(defaults.setQuiet(true), { protocol: 'receiving-preferences.v1', quiet: true });
  const enabled = new ReceivingPreferences(path);
  assert.equal(enabled.quiet, true);

  enabled.setQuiet(false);
  assert.equal(new ReceivingPreferences(path).quiet, false);
});

test('non-boolean quiet preference is rejected without changing stored bytes', async t => {
  const directory = await mkdtempForTest(t, resolve(fixtureRoot, 'receiving-preferences-'));
  const path = join(directory, 'preferences.json');
  const preferences = new ReceivingPreferences(path);
  preferences.setQuiet(true);
  const original = await readFile(path);

  assert.throws(() => preferences.setQuiet('true'), /Explicit receiving preference required/u);
  assert.deepEqual(await readFile(path), original);
});

test('corrupt and multi-field preferences are rejected without overwriting their files', async t => {
  const directory = await mkdtempForTest(t, resolve(fixtureRoot, 'receiving-preferences-'));
  const cases = [
    { name: 'corrupt.json', bytes: Buffer.from('{ not-json', 'utf8'), error: SyntaxError },
    { name: 'extra-field.json', bytes: Buffer.from(JSON.stringify({
      protocol: 'receiving-preferences.v1', quiet: false, notification: true,
    })), error: /Invalid receiving preference/u },
  ];

  for (const item of cases) {
    const path = join(directory, item.name);
    await writeFile(path, item.bytes, { flag: 'wx' });
    const original = await readFile(path);
    assert.throws(() => new ReceivingPreferences(path), item.error);
    assert.deepEqual(await readFile(path), original);
  }
});

test('send adapter freezes the active receiving owner snapshot before an asynchronous backend call', async t => {
  const directory = await mkdtempForTest(t, resolve(fixtureRoot, 'receiving-preferences-'));
  const firstOwner = new ReceivingPreferences(join(directory, 'first.json'));
  firstOwner.setQuiet(true);
  const nextOwner = new ReceivingPreferences(join(directory, 'next.json'));
  nextOwner.setQuiet(false);
  let activeOwner = firstOwner;
  const calls = [];
  let finishFirstCall;
  const firstCall = new Promise(resolveCall => { finishFirstCall = resolveCall; });
  const send = createReceivingContextSendAdapter({
    getPreferences: () => activeOwner,
    call: (action, payload) => {
      calls.push([action, payload]);
      return calls.length === 1 ? firstCall : Promise.resolve('second call');
    },
  });

  const pending = send({ text: 'hello' });
  activeOwner = nextOwner;
  firstOwner.setQuiet(false);

  const expectedContext = {
    protocol: 'receiving-context.v1',
    source: 'receiving-client',
    scope: 'this_client',
    observation: 'at_input_submission',
    quiet: true,
  };
  assert.deepEqual(calls[0], ['send', { text: 'hello', receiving_context: expectedContext }]);
  assert.equal(Object.isFrozen(calls[0][1].receiving_context), true);
  assert.equal(calls[0][1].receiving_context.quiet, true);

  finishFirstCall('first call');
  assert.equal(await pending, 'first call');
  assert.equal(await send({ text: 'next' }), 'second call');
  assert.equal(calls[1][1].receiving_context.quiet, false);
});
