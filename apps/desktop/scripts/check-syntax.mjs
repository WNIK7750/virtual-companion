import { readdirSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
for (const directory of ['src', 'scripts', 'test']) {
  for (const entry of readdirSync(directory)) {
    if (/\.(?:cjs|mjs|js)$/.test(entry)) execFileSync(process.execPath, ['--check', `${directory}/${entry}`], { stdio: 'inherit' });
  }
}
console.log('Desktop source, scripts and tests syntax valid.');
