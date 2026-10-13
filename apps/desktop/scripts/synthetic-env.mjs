import { randomUUID } from 'node:crypto';
import { existsSync, realpathSync } from 'node:fs';
import { basename, dirname, isAbsolute, relative, resolve, sep } from 'node:path';

const approvedScopes = new WeakMap();
const MANAGED_ENV_KEYS = new Set([
  'VC_DATA_ROOT',
  'VC_USER_DATA_ROOT',
  'VC_MEMORY_MANAGEMENT_DB',
  'VC_MEMORY_COMPANION_ID',
  'ELECTRON_RUN_AS_NODE',
]);
const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

function canonicalCompanionId(value) {
  if (typeof value !== 'string' || !UUID_PATTERN.test(value)) {
    throw new Error('Synthetic backend requires a UUID companion identity');
  }
  return value.toLowerCase();
}

function actualPath(value) {
  let existing = resolve(value);
  const suffix = [];
  while (!existsSync(existing)) {
    const parent = dirname(existing);
    if (parent === existing) throw new Error('Synthetic backend path cannot be resolved');
    suffix.unshift(basename(existing));
    existing = parent;
  }
  return resolve(realpathSync(existing), ...suffix);
}

function isWithin(root, target) {
  const pathFromRoot = relative(root, target);
  return pathFromRoot === '' || (
    pathFromRoot !== '..' &&
    !pathFromRoot.startsWith(`..${sep}`) &&
    !isAbsolute(pathFromRoot)
  );
}

function setManagedEnvironmentValues(environment, values) {
  for (const key of Object.keys(environment)) {
    if (MANAGED_ENV_KEYS.has(key.toUpperCase())) delete environment[key];
  }
  Object.assign(environment, values);
  delete environment.ELECTRON_RUN_AS_NODE;
}

function assertUnambiguousManagedEnvironment(environment) {
  const keys = Object.keys(environment);
  for (const managedKey of MANAGED_ENV_KEYS) {
    const aliases = keys.filter(key => key.toUpperCase() === managedKey);
    if (managedKey === 'ELECTRON_RUN_AS_NODE') {
      if (aliases.length > 0) throw new Error('Synthetic backend environment contains an Electron alias');
    } else if (aliases.length !== 1 || aliases[0] !== managedKey) {
      throw new Error('Synthetic backend environment contains an ambiguous managed key');
    }
  }
}

export function createSyntheticBackendEnvironment({
  parentEnv = process.env,
  dataRoot,
  userDataRoot,
  memoryDatabase,
  companionId = randomUUID(),
  fixtureRoot,
}) {
  if (typeof dataRoot !== 'string' || dataRoot.length === 0) {
    throw new Error('Synthetic backend requires an explicit data root');
  }
  const canonicalId = canonicalCompanionId(companionId);

  const expected = {
    dataRoot: actualPath(dataRoot),
    userDataRoot: actualPath(userDataRoot ?? resolve(dataRoot, 'electron-profile')),
    memoryDatabase: actualPath(memoryDatabase ?? resolve(dataRoot, 'memory.sqlite3')),
    companionId: canonicalId,
    fixtureRoots: fixtureRoot === undefined ? [] : [actualPath(fixtureRoot)],
  };
  const environment = { ...parentEnv };
  setManagedEnvironmentValues(environment, {
    VC_DATA_ROOT: resolve(dataRoot),
    VC_USER_DATA_ROOT: resolve(userDataRoot ?? resolve(dataRoot, 'electron-profile')),
    VC_MEMORY_MANAGEMENT_DB: resolve(memoryDatabase ?? resolve(dataRoot, 'memory.sqlite3')),
    VC_MEMORY_COMPANION_ID: canonicalId,
  });
  approvedScopes.set(environment, expected);
  assertSyntheticBackendEnvironment(environment);
  return environment;
}

export function assertSyntheticBackendEnvironment(environment) {
  const expected = approvedScopes.get(environment);
  if (!expected) throw new Error('Synthetic backend environment was not created by the isolation helper');
  assertUnambiguousManagedEnvironment(environment);

  if (actualPath(environment.VC_DATA_ROOT) !== expected.dataRoot) {
    throw new Error('Synthetic backend data root changed after isolation');
  }
  const userDataRootIsControlled = isWithin(expected.dataRoot, expected.userDataRoot) ||
    expected.fixtureRoots.some(root => isWithin(root, expected.userDataRoot));
  if (actualPath(environment.VC_USER_DATA_ROOT) !== expected.userDataRoot || !userDataRootIsControlled) {
    throw new Error('Synthetic backend user data root is outside its controlled root');
  }
  if (!UUID_PATTERN.test(environment.VC_MEMORY_COMPANION_ID ?? '') ||
      environment.VC_MEMORY_COMPANION_ID !== expected.companionId) {
    throw new Error('Synthetic backend companion identity changed after isolation');
  }

  const memoryDatabase = actualPath(environment.VC_MEMORY_MANAGEMENT_DB ?? '');
  const memoryRootIsControlled = isWithin(expected.dataRoot, memoryDatabase);
  const memoryRootIsExplicitFixture = expected.fixtureRoots.some(root => isWithin(root, memoryDatabase));
  if (memoryDatabase !== expected.memoryDatabase || (!memoryRootIsControlled && !memoryRootIsExplicitFixture)) {
    throw new Error('Synthetic backend memory database resolves outside its approved roots');
  }
  return true;
}
