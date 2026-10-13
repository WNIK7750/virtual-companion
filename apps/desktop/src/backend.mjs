import { spawn, execFileSync } from 'node:child_process';
import { randomBytes } from 'node:crypto';
import { request } from 'node:http';
import { existsSync } from 'node:fs';
import { join } from 'node:path';

export function waitForBackendHandshake(child, timeoutMs = 15000) {
  return new Promise((resolve, reject) => {
    let buffer = '';
    const finish = (error, receipt) => {
      clearTimeout(timer);
      child.removeListener('error', failed);
      child.removeListener('exit', exited);
      child.stdout.removeListener('data', receive);
      if (error) reject(error); else resolve(receipt);
    };
    const failed = error => finish(error);
    const exited = () => finish(new Error('Backend exited during startup'));
    const receive = chunk => {
      buffer += chunk;
      if (Buffer.byteLength(buffer) > 8192) return finish(new Error('Invalid backend handshake'));
      if (!buffer.includes('\n')) return;
      try {
        const receipt = JSON.parse(buffer.split('\n')[0]);
        if (receipt.event === 'startup_failed' && receipt.category === 'configuration') {
          const error = new Error('Startup configuration or runtime initialization failed');
          error.code = 'STARTUP_CONFIGURATION';
          return finish(error);
        }
        if (receipt.event !== 'ready' || !receipt.job || !Number.isSafeInteger(receipt.pid) || receipt.pid < 1) {
          return finish(new Error('Invalid backend handshake'));
        }
        finish(null, receipt);
      } catch { finish(new Error('Invalid backend handshake')); }
    };
    const timer = setTimeout(() => finish(new Error('Backend startup timeout')), timeoutMs);
    child.once('error', failed);
    child.once('exit', exited);
    child.stdout.on('data', receive);
  });
}

export class Backend {
  constructor(repository, options = {}) { this.repository = repository; this.options = options; this.status = 'stopped'; this.error = ''; }
  async start(loadModel = true) {
    if (this.child && this.status === 'ready') return loadModel ? this.call('start') : this.state();
    if (this.child || this.status === 'starting_backend') return this.state();
    this.status = 'starting_backend'; this.error = '';
    this.token = randomBytes(32).toString('hex');
    const python = join(this.repository, '.venv', 'Scripts', 'python.exe');
    if (!existsSync(python)) { this.status = 'failed'; this.error = '请先运行 scripts/check.ps1 准备 Python 环境。'; return this.state(); }
    const child = spawn(python, ['-u', '-m', 'virtual_companion.text', ...(this.options.fake ? ['--fake'] : [])], {
      cwd: this.repository, windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'],
      env: { ...process.env, PYTHONPATH: join(this.repository, 'apps/backend/src'), PYTHONUTF8: '1',
        VC_APP_COMMIT: execFileSync('git', ['-C', this.repository, 'rev-parse', 'HEAD'], { encoding: 'utf8', windowsHide: true }).trim(),
        VC_DATA_ROOT: process.env.VC_DATA_ROOT || join(this.repository, '.local', this.options.fake ? 'synthetic-dogfood' : 'dogfood'),
      },
    });
    this.child = child;
    child.stdin.on('error', () => {});
    child.stdin.write(this.token + '\n');
    child.stderr.on('data', () => {}); // Never forward raw provider/user data to renderer.
    child.once('exit', () => {
      if (this.child === child) this.child = undefined;
      if (this.status !== 'stopping' && this.status !== 'stopped') {
        this.status = 'failed'; this.error ||= '后端已退出。请检查端口或重新启动。';
      }
    });
    try {
      this.pid = (await waitForBackendHandshake(child)).pid;
      this.status = 'ready';
      if (loadModel) await this.call('start');
    } catch (error) {
      child.stdin.end(); this.status = 'failed';
      this.error = error.code === 'STARTUP_CONFIGURATION'
        ? '启动配置或运行环境无法校验，请修复所选配置后重启。'
        : '后端启动失败，请检查端口占用或 Python 环境。';
    }
    return this.state();
  }
  async state() {
    if (this.status === 'ready') {
      try { return await this.call('state', {}); }
      catch { this.status = 'failed'; this.error = '后端连接中断，请重新启动。'; this.child?.stdin.end(); }
    }
    return { state: this.status, error: this.error, mode: 'local' };
  }
  call(action, payload = {}) {
    if (!this.child || this.status !== 'ready') return Promise.reject(new Error('后端尚未就绪。'));
    return new Promise((resolve, reject) => {
      const data = JSON.stringify(payload);
      const req = request({ host: '127.0.0.1', port: 18084, path: `/v1/${action}`, method: 'POST',
        headers: { Authorization: `Bearer ${this.token}`, 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(data) }, timeout: 10000,
      }, response => {
        response.setEncoding('utf8');
        let body = '';
        response.on('data', chunk => { body += chunk; if (body.length > 2_000_000) req.destroy(new Error('Response too large')); });
        response.on('error', reject);
        response.on('end', () => { try { const value = JSON.parse(body); if (response.statusCode !== 200) reject(new Error(value.error)); else resolve(value); } catch { reject(new Error('后端响应无效。')); } });
      });
      req.on('error', reject); req.on('timeout', () => req.destroy(new Error('后端响应超时。'))); req.end(data);
    });
  }
  async close() {
    this.status = 'stopping';
    const child = this.child;
    if (child) { child.stdin.end(); await new Promise(resolve => { const timer = setTimeout(resolve, 5000); child.once('exit', () => { clearTimeout(timer); resolve(); }); }); }
    this.status = 'stopped';
  }
}
