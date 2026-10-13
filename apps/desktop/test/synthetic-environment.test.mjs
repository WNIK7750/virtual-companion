import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { mkdtemp, mkdir, readFile, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

test('parent memory overrides are replaced with a controlled synthetic root and identity', async () => {
  const { createSyntheticBackendEnvironment } = await import('../scripts/synthetic-env.mjs');
  const dataRoot = await mkdtemp(join(tmpdir(), 'vc-eval-isolation-root-'));
  const externalMarker = join(tmpdir(), `vc-nonexistent-external-memory-${process.pid}.sqlite3`);
  if (existsSync(externalMarker)) throw new Error('Synthetic external marker must not already exist');

  const environment = createSyntheticBackendEnvironment({
    parentEnv: {
      VC_MEMORY_MANAGEMENT_DB: externalMarker,
      VC_MEMORY_COMPANION_ID: '00000000-0000-0000-0000-000000000099',
      ELECTRON_RUN_AS_NODE: '1',
    },
    dataRoot,
  });

  assert.equal(environment.VC_DATA_ROOT, resolve(dataRoot));
  assert.equal(environment.VC_USER_DATA_ROOT, resolve(dataRoot, 'electron-profile'));
  assert.equal(environment.VC_MEMORY_MANAGEMENT_DB, resolve(dataRoot, 'memory.sqlite3'));
  assert.match(environment.VC_MEMORY_COMPANION_ID, /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i);
  assert.notEqual(environment.VC_MEMORY_COMPANION_ID, '00000000-0000-0000-0000-000000000099');
  assert.equal('ELECTRON_RUN_AS_NODE' in environment, false);
  assert.equal(existsSync(externalMarker), false, 'the inherited external marker must not be opened or created');
});

test('accidental external memory roots and identity changes are rejected', async () => {
  const { assertSyntheticBackendEnvironment, createSyntheticBackendEnvironment } = await import('../scripts/synthetic-env.mjs');
  const dataRoot = await mkdtemp(join(tmpdir(), 'vc-eval-isolation-reject-'));
  const externalMarker = join(tmpdir(), `vc-nonexistent-rejected-memory-${process.pid}.sqlite3`);
  if (existsSync(externalMarker)) throw new Error('Synthetic external marker must not already exist');

  assert.throws(() => createSyntheticBackendEnvironment({
    parentEnv: {},
    dataRoot,
    memoryDatabase: externalMarker,
  }), /outside/);

  const environment = createSyntheticBackendEnvironment({ parentEnv: {}, dataRoot });
  environment.VC_MEMORY_MANAGEMENT_DB = externalMarker;
  assert.throws(() => assertSyntheticBackendEnvironment(environment), /outside/);
  environment.VC_MEMORY_MANAGEMENT_DB = resolve(dataRoot, 'memory.sqlite3');
  environment.VC_MEMORY_COMPANION_ID = '00000000-0000-0000-0000-000000000099';
  assert.throws(() => assertSyntheticBackendEnvironment(environment), /identity/);
  assert.equal(existsSync(externalMarker), false);
});

test('validation rejects Windows case-variant environment aliases after construction', async () => {
  const { assertSyntheticBackendEnvironment, createSyntheticBackendEnvironment } = await import('../scripts/synthetic-env.mjs');
  const dataRoot = await mkdtemp(join(tmpdir(), 'vc-eval-isolation-alias-'));
  const externalMarker = join(tmpdir(), `vc-nonexistent-alias-memory-${process.pid}.sqlite3`);
  if (existsSync(externalMarker)) throw new Error('Synthetic external marker must not already exist');

  const environment = createSyntheticBackendEnvironment({
    parentEnv: {
      vC_mEmOrY_mAnAgEmEnT_dB: externalMarker,
      Vc_Memory_Companion_Id: '00000000-0000-0000-0000-000000000099',
      vc_data_root: externalMarker,
    },
    dataRoot,
  });
  const dbKeys = Object.keys(environment).filter(key => key.toUpperCase() === 'VC_MEMORY_MANAGEMENT_DB');
  assert.deepEqual(dbKeys, ['VC_MEMORY_MANAGEMENT_DB']);
  environment.vC_mEmOrY_mAnAgEmEnT_dB = externalMarker;
  assert.throws(() => assertSyntheticBackendEnvironment(environment), /alias|ambiguous|environment/i);
  assert.equal(existsSync(externalMarker), false);
});

test('companion UUID is canonicalized for the backend identity parser', async () => {
  const { assertSyntheticBackendEnvironment, createSyntheticBackendEnvironment } = await import('../scripts/synthetic-env.mjs');
  const dataRoot = await mkdtemp(join(tmpdir(), 'vc-eval-isolation-uuid-'));
  const environment = createSyntheticBackendEnvironment({
    parentEnv: {},
    dataRoot,
    companionId: 'AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE',
  });

  assert.equal(environment.VC_MEMORY_COMPANION_ID, 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee');
  assertSyntheticBackendEnvironment(environment);
});

test('real backend command preparation forwards only the parsed synthetic memory scope', async () => {
  const { createSyntheticBackendEnvironment } = await import('../scripts/synthetic-env.mjs');
  const repository = resolve(fileURLToPath(new URL('../../..', import.meta.url)));
  const dataRoot = await mkdtemp(join(tmpdir(), 'vc-eval-isolation-backend-'));
  const externalMarker = join(tmpdir(), `vc-nonexistent-backend-memory-${process.pid}.sqlite3`);
  if (existsSync(externalMarker)) throw new Error('Synthetic external marker must not already exist');
  const environment = createSyntheticBackendEnvironment({
    parentEnv: {
      vC_mEmOrY_mAnAgEmEnT_dB: externalMarker,
      Vc_Memory_Companion_Id: '00000000-0000-0000-0000-000000000099',
      vc_data_root: externalMarker,
    },
    dataRoot,
    companionId: 'AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE',
  });
  const python = join(repository, '.venv', 'Scripts', 'python.exe');

  const commandPreparationProbe = String.raw`
import { EventEmitter } from 'node:events';
import { PassThrough } from 'node:stream';
import childProcess from 'node:child_process';
import fs from 'node:fs';
import { syncBuiltinESMExports } from 'node:module';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const repository = process.argv[1];
const python = process.argv[2];
const realExecFileSync = childProcess.execFileSync;
const prepared = {};
childProcess.execFileSync = () => '0000000000000000000000000000000000000000';
childProcess.spawn = (command, args, options) => {
  prepared.command = command;
  prepared.args = args;
  prepared.env = options.env;
  const child = new EventEmitter();
  child.stdin = new PassThrough();
  child.stdout = new PassThrough();
  child.stderr = new PassThrough();
  setImmediate(() => child.stdout.write('{"event":"ready","job":{"id":"synthetic"},"pid":1}\n'));
  return child;
};
fs.existsSync = () => true;
syncBuiltinESMExports();

const { Backend } = await import(pathToFileURL(resolve(repository, 'apps/desktop/src/backend.mjs')).href);
const backend = new Backend(repository, { fake: true });
backend.state = async () => ({ status: backend.status });
const ready = await backend.start(false);
if (ready.status !== 'ready' || prepared.args?.[2] !== 'virtual_companion.text') {
  throw new Error('Backend command preparation did not reach the synthetic handshake');
}
const pythonDiagnostic = [
  "import json, os",
  "from pathlib import Path",
  "from virtual_companion.text.identity import _identity",
  "",
  "data_root = Path(os.environ['VC_DATA_ROOT']).resolve()",
  "memory_override = os.environ.get('VC_MEMORY_MANAGEMENT_DB')",
  "companion_override = os.environ.get('VC_MEMORY_COMPANION_ID')",
  "if (memory_override is None) != (companion_override is None):",
  "    raise ValueError('Memory management override requires an explicit database and companion')",
  "companion_id = _identity(companion_override)",
  "memory_path = Path(memory_override or data_root / 'memory.sqlite3').resolve()",
  "print(json.dumps({'data_root': str(data_root), 'memory_database': str(memory_path), 'companion_id': companion_id}))",
].join('\n');
const resolvedScope = realExecFileSync(python, ['-B', '-c', pythonDiagnostic], {
  cwd: repository,
  encoding: 'utf8',
  windowsHide: true,
  env: prepared.env,
});
process.stdout.write(JSON.stringify({ args: prepared.args, resolved: JSON.parse(resolvedScope.trim()) }));
`;
  const output = execFileSync(process.execPath, ['--input-type=module', '-e', commandPreparationProbe, repository, python], {
    encoding: 'utf8',
    windowsHide: true,
    env: environment,
  });
  const result = JSON.parse(output.trim());

  assert.deepEqual(result.args.slice(0, 3), ['-u', '-m', 'virtual_companion.text']);
  assert.ok(result.args.includes('--fake'));
  assert.deepEqual(result.resolved, {
    data_root: resolve(dataRoot),
    memory_database: resolve(dataRoot, 'memory.sqlite3'),
    companion_id: 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee',
  });
  assert.equal(existsSync(externalMarker), false);
});

test('an explicitly scoped external synthetic fixture is preserved', async () => {
  const { assertSyntheticBackendEnvironment, createSyntheticBackendEnvironment } = await import('../scripts/synthetic-env.mjs');
  const dataRoot = await mkdtemp(join(tmpdir(), 'vc-eval-isolation-data-'));
  const fixtureRoot = await mkdtemp(join(tmpdir(), 'vc-eval-isolation-fixture-'));
  const fixture = join(fixtureRoot, 'memory-v19.sqlite3');
  const fixtureBytes = 'synthetic fixture marker; not a real database';
  await mkdir(dataRoot, { recursive: true });
  await writeFile(fixture, fixtureBytes, { flag: 'wx' });

  const environment = createSyntheticBackendEnvironment({
    parentEnv: { VC_MEMORY_MANAGEMENT_DB: 'ignored-parent-path' },
    dataRoot,
    userDataRoot: join(fixtureRoot, 'electron'),
    memoryDatabase: fixture,
    companionId: '00000000-0000-0000-0000-000000000001',
    fixtureRoot,
  });
  assertSyntheticBackendEnvironment(environment);

  assert.equal(environment.VC_MEMORY_MANAGEMENT_DB, resolve(fixture));
  assert.equal(environment.VC_USER_DATA_ROOT, resolve(fixtureRoot, 'electron'));
  assert.equal(environment.VC_MEMORY_COMPANION_ID, '00000000-0000-0000-0000-000000000001');
  assert.equal(await readFile(fixture, 'utf8'), fixtureBytes);
});
