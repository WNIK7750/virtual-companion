// Main-process device receipts. An intent is durable BEFORE invoking the renderer.
// A crash between display and receipt publication remains unknown, never replayed.
import * as fs from 'node:fs';
import { createHash, randomUUID } from 'node:crypto';
import { join, resolve } from 'node:path';

function canonical(value) {
  if (value === null || typeof value === 'boolean') return JSON.stringify(value);
  if (typeof value === 'string' && value.isWellFormed()) return JSON.stringify(value);
  if (typeof value === 'number' && Number.isSafeInteger(value)) return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (value && Object.getPrototypeOf(value) === Object.prototype) {
    return `{${Object.keys(value).sort().map(key => `${canonical(key)}:${canonical(value[key])}`).join(',')}}`;
  }
  throw new Error('Invalid canonical stream value');
}

export function digest(value) { return createHash('sha256').update(canonical(value), 'utf8').digest('hex'); }

// Faithful units bind to code-point ranges of the immutable source text, so the
// reconstructed display text must equal that source exactly. A missing
// representation keeps the legacy blank-line join.
export function displayText(candidate) {
  if (candidate.representation !== 'spans-v1') {
    if (candidate.source_text !== undefined
        || candidate.units.some(unit => unit.start !== undefined || unit.end !== undefined)) {
      throw new Error('Joined display cannot carry source ranges');
    }
    return candidate.units.map(unit => unit.text).join('\n\n');
  }
  if (typeof candidate.source_text !== 'string' || !candidate.source_text) {
    throw new Error('Invalid faithful source text');
  }
  const points = Array.from(candidate.source_text);
  let cursor = 0;
  for (const unit of candidate.units) {
    if (!Number.isInteger(unit.start) || !Number.isInteger(unit.end) || unit.start !== cursor
        || unit.end <= unit.start || unit.end > points.length
        || points.slice(unit.start, unit.end).join('') !== unit.text) {
      throw new Error('Invalid faithful display range');
    }
    cursor = unit.end;
  }
  if (cursor !== points.length) throw new Error('Faithful units must cover the source text');
  return candidate.source_text;
}

export function validateStreamEvent(event) {
  if (!event || event.protocol !== 'approved-prefix.v1' || event.status !== 'unknown'
      || !/^[0-9a-f]{64}$/.test(event.fingerprint) || !Number.isInteger(event.sequence)
      || event.sequence < 1 || event.sequence > 32) throw new Error('Invalid stream envelope');
  const manifest = { protocol: event.protocol, basis: event.basis, sequence: event.sequence,
    presentation: event.presentation };
  if (Object.hasOwn(event, 'readset')) manifest.readset = event.readset;
  if (Object.hasOwn(event, 'bubbles')) manifest.bubbles = event.bubbles;
  if (Buffer.byteLength(canonical(manifest)) > 2_000_000 || digest(manifest) !== event.fingerprint) {
    throw new Error('Stream fingerprint mismatch');
  }
  const { candidate, policy, review } = event.presentation;
  const basis = event.basis;
  if (!candidate || !policy || !review || !basis || !Array.isArray(candidate.units)
      || candidate.units.length !== event.sequence || candidate.intent && candidate.intent !== 'speak'
      || candidate.scope.companion_id !== basis.companion_id || candidate.scope.session_id !== basis.session_id
      || candidate.input_id !== basis.input_id || candidate.attempt_id !== basis.attempt_id
      || candidate.units.some(unit => unit.kind !== 'text' || typeof unit.text !== 'string' || !unit.text.trim())
      || candidate.units.at(-1).text !== event.text
      || review.candidate_sha256 !== digest(candidate) || review.policy_sha256 !== digest(policy)
      || !Array.isArray(review.issues) || review.issues.length
      || !(review.model_approved === true || review.approval_source === 'host_policy'
        && review.model_approved === false && typeof review.policy_revision === 'string' && review.policy_revision)) {
    throw new Error('Invalid reviewed display binding');
  }
  if (Buffer.byteLength(displayText(candidate)) > 64000) {
    throw new Error('Display text exceeds the bounded surface');
  }
  if (Object.hasOwn(event, 'bubbles')) {
    const source = displayText(candidate), ranges = []; let start = 0;
    for (const separator of source.matchAll(/\r?\n[ \t]*(?:\r?\n)+/gu)) {
      const end = separator.index + separator[0].length;
      if (end < source.length && source.slice(start, end).trim()) {
        ranges.push({ start: Array.from(source.slice(0, start)).length,
          end: Array.from(source.slice(0, end)).length });
        start = end;
      }
    }
    ranges.push({ start: Array.from(source.slice(0, start)).length, end: Array.from(source).length });
    if (candidate.representation !== 'spans-v1' || canonical(event.bubbles) !== canonical(ranges)) {
      throw new Error('Invalid semantic bubble ranges');
    }
  }
  const identity = { companion: basis.companion_id, input: basis.input_id, attempt: basis.attempt_id,
    candidate: candidate.candidate_id, revision: candidate.revision, sequence: event.sequence };
  for (const key of ['companion', 'input', 'attempt', 'candidate']) {
    if (typeof identity[key] !== 'string' || !identity[key].trim() || [...identity[key]].length > 128) {
      throw new Error('Invalid display identity');
    }
  }
  if (!Number.isSafeInteger(identity.revision) || identity.revision < 1) throw new Error('Invalid revision');
  return { identity, key: digest(identity), fingerprint: event.fingerprint };
}

export class StreamReceiptJournal {
  constructor(directory, { io = fs } = {}) {
    this.directory = resolve(directory); this.io = io;
    io.mkdirSync(this.directory, { recursive: true });
  }
  paths(binding) {
    return { intent: join(this.directory, `${binding.key}.intent.json`),
      visible: join(this.directory, `${binding.key}.visible.json`),
      pending: join(this.directory, `${binding.key}.pending.json`) };
  }
  read(path) {
    if (this.io.statSync(path).size > 16384) throw new Error('Receipt record capacity exceeded');
    const value = JSON.parse(this.io.readFileSync(path, 'utf8'));
    if (value.version !== 1 || typeof value.receipt_id !== 'string'
        || !/^stream-visible-[0-9a-f-]{36}$/.test(value.receipt_id)) throw new Error('Invalid device receipt');
    return value;
  }
  lookup(event) {
    return this.#lookup(validateStreamEvent(event));
  }
  #lookup(binding) {
    const paths = this.paths(binding);
    if (!this.io.existsSync(paths.intent)) {
      if (this.io.existsSync(paths.visible)) throw new Error('Receipt has no intent');
      return { status: 'unseen', receipt: null };
    }
    const intent = this.read(paths.intent);
    if (intent.fingerprint !== binding.fingerprint || digest(intent.identity) !== binding.key) {
      throw new Error('Conflicting durable display identity');
    }
    if (!this.io.existsSync(paths.visible)) return { status: 'unknown', receipt: null };
    const visible = this.read(paths.visible);
    if (canonical(visible) !== canonical(intent)) throw new Error('Conflicting visible receipt');
    return { status: 'confirmed', receipt: visible.receipt_id };
  }
  writeExclusive(path, value) {
    const fd = this.io.openSync(path, 'wx', 0o600);
    try { this.io.writeFileSync(fd, canonical(value), 'utf8'); this.io.fsyncSync(fd); }
    finally { this.io.closeSync(fd); }
  }
  receivedRecords() {
    if (!this.received) {
      this.received = new Map();
      for (const name of this.io.readdirSync(this.directory)) {
        if (!/^[0-9a-f]{64}\.received\.json$/u.test(name)) continue;
        const path = join(this.directory, name);
        if (this.io.statSync(path).size > 2_100_000) throw new Error('Received text capacity exceeded');
        const row = JSON.parse(this.io.readFileSync(path, 'utf8'));
        const binding = validateStreamEvent(row.event);
        if (row.version !== 1 || `${binding.key}.received.json` !== name
            || !/^stream-received-[0-9a-f-]{36}$/u.test(row.receipt)
            || !Number.isSafeInteger(row.order) || row.order < 1) throw new Error('Invalid received text');
        this.received.set(binding.key, row);
      }
    }
    return this.received;
  }
  receive(event, scope) {
    const binding = validateStreamEvent(event);
    if (!scope || !['session_id', 'input_id', 'attempt_id'].every(key => scope[key] === event.basis[key])
        || !['turn_id', 'text_attempt_id'].every(key => typeof scope[key] === 'string' && scope[key])) {
      throw new Error('Reliable text must retain its original UI source');
    }
    const records = this.receivedRecords();
    const old = records.get(binding.key);
    if (old) {
      if (old.event.fingerprint !== event.fingerprint || canonical(old.scope) !== canonical(scope)) {
        throw new Error('Conflicting received text');
      }
      return structuredClone(old);
    }
    const order = [...records.values()].reduce((latest, row) => Math.max(latest, row.order), 0) + 1;
    const row = { version: 1, event: structuredClone(event), scope: structuredClone(scope),
      receipt: `stream-received-${randomUUID()}`, order };
    this.writeExclusive(join(this.directory, `${binding.key}.received.json`), row);
    records.set(binding.key, row);
    return structuredClone(row);
  }
  pendingReceived(sessionId) {
    return [...this.receivedRecords().values()].filter(row => (!sessionId || row.scope.session_id === sessionId)
      && this.lookup(row.event).status !== 'confirmed').sort((a, b) => a.order - b.order).map(row => structuredClone(row));
  }
  async presentReceived(row, render, options) {
    const binding = validateStreamEvent(row.event);
    const saved = this.receivedRecords().get(binding.key);
    if (!saved || canonical(saved) !== canonical(row)) throw new Error('Unreceived text cannot be projected');
    // Reliable text is an idempotent projection by original message identity.
    // This does not automatically replay legacy uncertain display intents.
    return this.resume(row.event, render, options);
  }
  async present(event, render, { signal } = {}) {
    if (typeof render !== 'function') throw new Error('Trusted renderer required');
    const binding = validateStreamEvent(event); const paths = this.paths(binding);
    const current = this.#lookup(binding);
    if (current.status !== 'unseen') return current.receipt;
    if (signal?.aborted) return null;
    const intent = { version: 1, identity: binding.identity, fingerprint: binding.fingerprint,
      receipt_id: `stream-visible-${randomUUID()}` };
    try { this.writeExclusive(paths.intent, intent); }
    catch (error) {
      if (error.code === 'EEXIST') return this.#lookup(binding).receipt;
      throw error;
    }
    if (signal?.aborted) return null;
    if (await render(structuredClone(event)) !== true) return null;
    // No cancellation check here: an actual visible result remains real evidence.
    this.writeExclusive(paths.pending, intent);
    this.io.renameSync(paths.pending, paths.visible);
    return intent.receipt_id;
  }

  async resume(event, render, { signal } = {}) {
    // Only an explicit user action may enter this path. Lookup/present still
    // never infer visibility or replay an unknown dispatch automatically.
    if (typeof render !== 'function') throw new Error('Trusted renderer required');
    const binding = validateStreamEvent(event); const paths = this.paths(binding);
    const current = this.#lookup(binding);
    if (current.status === 'confirmed') return current.receipt;
    if (current.status === 'unseen') return this.present(event, render, { signal });
    if (signal?.aborted) return null;
    const intent = this.read(paths.intent);
    if (await render(structuredClone(event)) !== true) return null;
    const latest = this.lookup(event);
    if (latest.status === 'confirmed') return latest.receipt;
    // A previous failed fsync/rename may have left the original pending file.
    // Preserve it as evidence; this fresh observation uses a new temporary name.
    const pending = `${paths.pending}.${randomUUID()}`;
    this.writeExclusive(pending, intent);
    this.io.renameSync(pending, paths.visible);
    return intent.receipt_id;
  }
}
