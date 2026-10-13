import { mkdtemp, rm } from 'node:fs/promises';
import { resolve } from 'node:path';

// Set this to "1" to keep fixture directories even when their test passes.
export const KEEP_SYNTHETIC_TEST_FIXTURES = 'VC_KEEP_SYNTHETIC_TEST_FIXTURES';
const directoriesByTest = new WeakMap();

function registerLifecycle(testContext) {
  let directories = directoriesByTest.get(testContext);
  if (directories) return directories;

  directories = [];
  directoriesByTest.set(testContext, directories);
  testContext.after(async context => {
    const preserveAll = process.env[KEEP_SYNTHETIC_TEST_FIXTURES] === '1';
    const retained = preserveAll
      ? new Set(directories)
      : context.passed === true ? new Set() : new Set(directories.slice(-1));
    for (const directory of directories) {
      if (retained.has(directory)) {
        context.diagnostic(`Retained synthetic test fixture at ${directory}`);
      } else {
        await rm(directory, { recursive: true, force: false });
      }
    }
  });
  return directories;
}

export async function mkdtempForTest(testContext, template) {
  if (typeof testContext?.after !== 'function') {
    throw new TypeError('A node:test context is required to manage a synthetic fixture directory');
  }

  const directories = registerLifecycle(testContext);
  const directory = await mkdtemp(resolve(template));
  directories.push(directory);
  return directory;
}
