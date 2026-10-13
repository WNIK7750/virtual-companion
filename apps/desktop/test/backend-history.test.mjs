import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { Backend } from '../src/backend.mjs';

test('history transport preserves UTF-8 across network chunk boundaries', async () => {
  const server = createServer((_request, response) => {
    response.writeHead(200, { 'Content-Type': 'application/json' });
    const data = Buffer.from(JSON.stringify({ messages: [{ content: '中文🙂原样保留' }] }));
    const boundary = data.indexOf(Buffer.from('中')) + 1;
    response.write(data.subarray(0, boundary));
    setTimeout(() => response.end(data.subarray(boundary)), 20);
  });
  await new Promise((resolve, reject) => { server.once('error', reject); server.listen(18084, '127.0.0.1', resolve); });
  try {
    const backend = new Backend('unused'); backend.child = {}; backend.status = 'ready'; backend.token = 'synthetic';
    assert.equal((await backend.call('history')).messages[0].content, '中文🙂原样保留');
  } finally { await new Promise(resolve => server.close(resolve)); }
});
