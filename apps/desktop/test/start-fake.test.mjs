import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { mkdtempSync, mkdirSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { createFakeEnvironment, startFake } from '../scripts/start-fake.mjs';

function fixture(t) {
  const repository = mkdtempSync(join(tmpdir(), 'vc-public-fake-'));
  const desktopRoot = join(repository, 'apps', 'desktop');
  mkdirSync(desktopRoot, { recursive: true });
  t.after(() => rmSync(repository, { recursive: true, force: true }));
  return { repository, desktopRoot };
}

test('fake startup replaces every inherited VC setting without mutating the parent', t => {
  const { repository, desktopRoot } = fixture(t);
  const parentEnv = {
    PATH: 'synthetic-system-path',
    SystemRoot: 'synthetic-system-root',
    VC_DATA_ROOT: 'inherited-data',
    vc_memory_management_db: 'inherited-memory',
    VC_MEMORY_COMPANION_ID: 'inherited-identity',
    VC_PROVIDER: 'inherited-provider',
    Vc_Unexpected_Future_Flag: 'inherited-feature',
    ELECTRON_RUN_AS_NODE: '1',
    electron_run_as_node: '1',
  };
  const before = { ...parentEnv };
  const environment = createFakeEnvironment({ desktopRoot, parentEnv });
  assert.deepEqual(parentEnv, before);
  assert.equal(environment.PATH, parentEnv.PATH);
  assert.equal(environment.SystemRoot, parentEnv.SystemRoot);
  assert.deepEqual(Object.keys(environment).filter(key => /^VC_/i.test(key)).sort(), [
    'VC_DATA_ROOT', 'VC_MEMORY_COMPANION_ID', 'VC_MEMORY_MANAGEMENT_DB', 'VC_USER_DATA_ROOT',
  ]);
  assert.equal(Object.keys(environment).some(key => /^ELECTRON_RUN_AS_NODE$/i.test(key)), false);
  const demoRoot = resolve(repository, '.local', 'synthetic-demo');
  assert.equal(environment.VC_DATA_ROOT, demoRoot);
  assert.equal(environment.VC_USER_DATA_ROOT, join(demoRoot, 'electron-profile'));
  assert.equal(environment.VC_MEMORY_MANAGEMENT_DB, join(demoRoot, 'memory.sqlite3'));
});

test('reopening the demo retains a stable synthetic identity and local paths', t => {
  const { desktopRoot } = fixture(t);
  const first = createFakeEnvironment({ desktopRoot, parentEnv: {} });
  const second = createFakeEnvironment({ desktopRoot, parentEnv: { VC_MEMORY_COMPANION_ID: 'other' } });
  assert.deepEqual(second, first);
  assert.equal(first.VC_MEMORY_COMPANION_ID, '00000000-0000-4000-8000-000000000001');
});

test('launcher starts Electron in the desktop directory with fake mode and isolated environment', async t => {
  const { desktopRoot, repository } = fixture(t);
  const child = new EventEmitter();
  const calls = [];
  const pending = startFake({
    desktopRoot,
    parentEnv: { VC_DATA_ROOT: 'unrelated-data', ELECTRON_RUN_AS_NODE: '1' },
    electronPath: 'synthetic-electron-binary',
    spawn: (...args) => { calls.push(args); return child; },
  });
  assert.equal(calls.length, 1);
  const [executable, args, options] = calls[0];
  assert.equal(executable, 'synthetic-electron-binary');
  assert.deepEqual(args, ['.', '--fake']);
  assert.equal(options.cwd, desktopRoot);
  assert.equal(options.stdio, 'inherit');
  assert.equal(options.env.VC_DATA_ROOT, join(repository, '.local', 'synthetic-demo'));
  assert.equal('ELECTRON_RUN_AS_NODE' in options.env, false);
  child.emit('close', 0, null);
  assert.equal(await pending, 0);
});

test('launcher propagates unsuccessful Electron exit codes', async t => {
  const { desktopRoot } = fixture(t);
  const child = new EventEmitter();
  const pending = startFake({ desktopRoot, parentEnv: {}, electronPath: 'fake', spawn: () => child });
  child.emit('close', 37, null);
  assert.equal(await pending, 37);
});

test('launcher does not report a signal termination as success', async t => {
  const { desktopRoot } = fixture(t);
  const child = new EventEmitter();
  const pending = startFake({ desktopRoot, parentEnv: {}, electronPath: 'fake', spawn: () => child });
  child.emit('close', null, 'SIGTERM');
  assert.equal(await pending, 143);
});

test('launcher surfaces startup errors instead of hanging or claiming success', async t => {
  const { desktopRoot } = fixture(t);
  const child = new EventEmitter();
  const pending = startFake({ desktopRoot, parentEnv: {}, electronPath: 'fake', spawn: () => child });
  child.emit('error', new Error('synthetic startup failure'));
  await assert.rejects(pending, /synthetic startup failure/);
});

test('npm fake entry uses the isolation wrapper', () => {
  const manifest = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8'));
  assert.equal(manifest.scripts['start:fake'], 'node scripts/start-fake.mjs');
});
