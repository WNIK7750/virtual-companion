import { spawn as spawnChild } from 'node:child_process';
import { createRequire } from 'node:module';
import { constants } from 'node:os';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createSyntheticBackendEnvironment } from './synthetic-env.mjs';

const require = createRequire(import.meta.url);
const defaultDesktopRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
// This identity belongs only to the persistent, synthetic public demo.
const demoCompanionId = '00000000-0000-4000-8000-000000000001';

export function createFakeEnvironment({ desktopRoot = defaultDesktopRoot, parentEnv = process.env } = {}) {
  const cleanParent = Object.fromEntries(Object.entries(parentEnv).filter(([key]) => {
    const normalized = key.toUpperCase();
    return !normalized.startsWith('VC_') && normalized !== 'ELECTRON_RUN_AS_NODE';
  }));
  const dataRoot = resolve(desktopRoot, '..', '..', '.local', 'synthetic-demo');
  return createSyntheticBackendEnvironment({
    parentEnv: cleanParent,
    dataRoot,
    userDataRoot: resolve(dataRoot, 'electron-profile'),
    memoryDatabase: resolve(dataRoot, 'memory.sqlite3'),
    companionId: demoCompanionId,
  });
}

export async function startFake({
  desktopRoot = defaultDesktopRoot,
  parentEnv = process.env,
  electronPath,
  spawn = spawnChild,
} = {}) {
  const environment = createFakeEnvironment({ desktopRoot, parentEnv });
  const executable = electronPath ?? require('electron');
  return new Promise((resolveExit, reject) => {
    const child = spawn(executable, ['.', '--fake'], {
      cwd: desktopRoot,
      env: environment,
      stdio: 'inherit',
    });
    child.once('error', reject);
    child.once('close', (code, signal) => {
      resolveExit(code ?? (signal && constants.signals[signal] ? 128 + constants.signals[signal] : 1));
    });
  });
}

if (process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url) {
  try {
    process.exitCode = await startFake();
  } catch {
    console.error('Unable to start the offline demo. Check the local Electron installation and demo data paths.');
    process.exitCode = 1;
  }
}
