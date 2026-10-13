import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { PassThrough } from 'node:stream';
import { waitForBackendHandshake } from '../src/backend.mjs';

const child = () => Object.assign(new EventEmitter(), { stdout: new PassThrough() });
test('startup accepts only the completed backend handshake, including split chunks', async () => {
  const process = child(); const pending = waitForBackendHandshake(process, 100);
  process.stdout.write('{"event":"rea');
  process.stdout.write('dy","pid":42,"job":true}\n');
  assert.equal((await pending).pid, 42);
  assert.equal(process.stdout.listenerCount('data'), 0);
  assert.equal(process.listenerCount('exit'), 0);
});
test('configuration failure rejects immediately with a safe fixed category', async () => {
  const process = child(); const pending = waitForBackendHandshake(process, 100);
  process.stdout.write('{"event":"startup_failed","category":"configuration","error":"sensitive input"}\n');
  await assert.rejects(pending, error => error.code === 'STARTUP_CONFIGURATION' && !error.message.includes('sensitive'));
});
test('invalid, oversized and unexpected startup handshakes fail closed', async () => {
  for (const text of ['bad\n', '{"event":"ready","job":false}\n', '{"event":"other"}\n', 'a'.repeat(8193)]) {
    const process = child(); const pending = waitForBackendHandshake(process, 100);
    process.stdout.write(text); await assert.rejects(pending);
  }
});
test('startup exit and timeout release listeners and remain failures', async () => {
  const process = child(); const exited = waitForBackendHandshake(process, 100);
  process.emit('exit', 1); await assert.rejects(exited, /exited/);
  const hanging = child(); await assert.rejects(waitForBackendHandshake(hanging, 5), /timeout/);
  assert.equal(hanging.stdout.listenerCount('data'), 0);
});
