import { digest } from '../src/stream-receipts.mjs';

export function event() {
  const scope = { companion_id: '00000000-0000-0000-0000-000000000001', session_id: 'session' };
  const candidate = { scope, input_id: 'input', attempt_id: 'attempt', candidate_id: 'candidate',
    revision: 1, semantic_text: '你好🌿', units: [{ unit_id: 'unit1', kind: 'text', text: '你好🌿' }] };
  const policy = { max_exclamations: 10, max_unit_codepoints: 1000, max_units: 32, allow_actions: false };
  const manifest = { protocol: 'approved-prefix.v1', sequence: 1,
    basis: { companion_id: scope.companion_id, session_id: 'session', input_id: 'input',
      attempt_id: 'attempt', owner: { companion_id: scope.companion_id, owner_id: 'worker', epoch: 1 },
      state_version: 0, config_json: '{}' },
    presentation: { candidate, policy, review: { review_id: 'review',
      candidate_sha256: digest(candidate), policy_sha256: digest(policy), model_approved: false,
      approval_source: 'host_policy', policy_revision: 'synthetic-v1', issues: [] } } };
  return { ...manifest, fingerprint: digest(manifest), status: 'unknown', host_receipt: null, text: '你好🌿' };
}

export function spanned({ source = '甲句。乙句。\n\n丙句。', cuts = [3, 8, 11],
  units = null, text = null } = {}) {
  const scope = { companion_id: '00000000-0000-0000-0000-000000000001', session_id: 'session' };
  const points = Array.from(source);
  const ranges = units ?? (() => {
    let cursor = 0;
    return cuts.map((end, index) => {
      const unit = { unit_id: `unit-${index + 1}`, kind: 'text',
        text: points.slice(cursor, end).join(''), start: cursor, end };
      cursor = end;
      return unit;
    });
  })();
  const candidate = { scope, input_id: 'input', attempt_id: 'attempt', candidate_id: 'candidate',
    revision: 1, semantic_text: source, units: ranges, representation: 'spans-v1', source_text: source };
  const policy = { max_exclamations: 10, max_unit_codepoints: 1000, max_units: 32, allow_actions: false };
  const manifest = { protocol: 'approved-prefix.v1', sequence: ranges.length,
    basis: { companion_id: scope.companion_id, session_id: 'session', input_id: 'input',
      attempt_id: 'attempt', owner: { companion_id: scope.companion_id, owner_id: 'worker', epoch: 1 },
      state_version: 0, config_json: '{}' },
    presentation: { candidate, policy, review: { review_id: 'review',
      candidate_sha256: digest(candidate), policy_sha256: digest(policy), model_approved: false,
      approval_source: 'host_policy', policy_revision: 'synthetic-span-v1', issues: [] } } };
  return { ...manifest, fingerprint: digest(manifest), status: 'unknown', host_receipt: null,
    text: text ?? ranges.at(-1).text };
}
