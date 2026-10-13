// Host adapter for a platform-neutral receiving preference. It never stops a turn.
import * as fs from 'node:fs';
import { randomUUID } from 'node:crypto';

export class ReceivingPreferences {
  constructor(path) {
    this.path = path; this.quiet = false;
    if (fs.existsSync(path)) {
      if (fs.statSync(path).size > 1024) throw new Error('Receiving preference is too large');
      const value = JSON.parse(fs.readFileSync(path, 'utf8'));
      if (value.protocol !== 'receiving-preferences.v1' || typeof value.quiet !== 'boolean'
          || Object.keys(value).length !== 2) throw new Error('Invalid receiving preference');
      this.quiet = value.quiet;
    }
  }
  snapshot() { return { protocol: 'receiving-preferences.v1', quiet: this.quiet }; }
  setQuiet(quiet) {
    if (typeof quiet !== 'boolean') throw new Error('Explicit receiving preference required');
    const value = { protocol: 'receiving-preferences.v1', quiet };
    const temporary = `${this.path}.${randomUUID()}.pending`;
    const fd = fs.openSync(temporary, 'wx', 0o600);
    try { fs.writeFileSync(fd, JSON.stringify(value)); fs.fsyncSync(fd); }
    finally { fs.closeSync(fd); }
    fs.renameSync(temporary, this.path);
    this.quiet = quiet;
    return this.snapshot();
  }
}

export function createReceivingContextSendAdapter({ getPreferences, call }) {
  if (typeof getPreferences !== 'function' || typeof call !== 'function') {
    throw new TypeError('Receiving context send dependencies are required');
  }

  return payload => {
    const owner = getPreferences();
    const preference = owner?.snapshot?.();
    if (preference?.protocol !== 'receiving-preferences.v1' || typeof preference.quiet !== 'boolean') {
      throw new Error('Receiving preference snapshot is unavailable');
    }

    const receiving_context = Object.freeze({
      protocol: 'receiving-context.v1',
      source: 'receiving-client',
      scope: 'this_client',
      observation: 'at_input_submission',
      quiet: preference.quiet,
    });
    return call('send', { text: payload.text, receiving_context });
  };
}
