export function windowOptions(preload) {
  return { show: false, width: 1320, height: 860, minWidth: 520, minHeight: 560,
    backgroundColor: '#f5f6f8', title: '文字伴侣 · 文字空间',
    webPreferences: { preload, sandbox: true, contextIsolation: true,
      nodeIntegration: false, webSecurity: true, spellcheck: false } };
}

export function createShellHandlers(contents, service) {
  const actions = new Set(['state', 'start', 'send', 'stop', 'retry', 'history', 'newSession', 'feedback', 'sessions', 'selectSession', 'selectProvider', 'memories', 'memoryDetails', 'memoryAction', 'memoryAbandon', 'presentApproved', 'resumeApproved', 'cancelRecovery', 'resumeInputs']);
  return (event, action, payload) => {
    if (event.sender !== contents || event.senderFrame !== contents.mainFrame ||
        event.senderFrame?.url !== 'companion://app/index.html') throw new Error('Invalid sender');
    if (!actions.has(action) && action !== 'setQuiet') throw new Error('Invalid action');
    if (!payload || Object.getPrototypeOf(payload) !== Object.prototype) throw new Error('Invalid payload');
    if (action === 'setQuiet') {
      if (Object.keys(payload).length !== 1 || typeof payload.quiet !== 'boolean') throw new Error('Invalid payload');
      return service.setQuiet(payload);
    }
    if (action === 'resumeApproved') {
      if (Object.keys(payload).length !== 1 || typeof payload.fingerprint !== 'string'
          || !/^[0-9a-f]{64}$/.test(payload.fingerprint)) throw new Error('Invalid payload');
      return service.resumeApproved(payload);
    }
    if (action === 'history') {
      if (Object.keys(payload).some(key => !['before', 'after', 'session_id'].includes(key)) ||
          (payload.before != null && payload.after != null) ||
          Object.values(payload).some(value => value != null && (typeof value !== 'string' || !value.length || value.length > 128))) throw new Error('Invalid payload');
      return service.history(payload);
    }
    if (action === 'feedback') {
      const exact = fields => Object.keys(payload).length === fields.length &&
        Object.keys(payload).every(key => fields.includes(key));
      const sha256 = value => typeof value === 'string' && /^[0-9a-f]{64}$/.test(value);
      const plainJson = (value, depth = 0) => {
        if (depth > 32) return false;
        if (value === null || typeof value === 'string' || typeof value === 'boolean') return true;
        if (typeof value === 'number') return Number.isFinite(value);
        if (Array.isArray(value)) return value.every(item => plainJson(item, depth + 1));
        return typeof value === 'object' && Object.getPrototypeOf(value) === Object.prototype &&
          Object.values(value).every(item => plainJson(item, depth + 1));
      };
      let valid = false, maxBytes = 16000;
      if (payload.operation === 'list') valid = exact(['operation']);
      else if (payload.operation === 'submit') {
        const base = ['operation','session_id','turn_id','quick_tags','freeform_reaction','idempotency_key'];
        if (!Object.hasOwn(payload, 'target')) valid = exact([...base,'assistant_message_id']);
        else if (payload.target && typeof payload.target === 'object' &&
            Object.getPrototypeOf(payload.target) === Object.prototype) {
          const target = payload.target;
          if (target.kind === 'attempt') {
            valid = exact([...base,'target']) && Object.keys(target).length === 2 &&
              Object.keys(target).every(key => ['kind','attempt_id'].includes(key)) &&
              typeof target.attempt_id === 'string' && target.attempt_id.length > 0 &&
              target.attempt_id.length <= 128;
          } else if (target.kind === 'reply' || target.kind === 'bubble') {
            valid = exact([...base,'assistant_message_id','target']) && Object.keys(target).length === 3 &&
              Object.keys(target).every(key => ['kind','start','end'].includes(key)) &&
              Number.isSafeInteger(target.start) && Number.isSafeInteger(target.end) &&
              target.start >= 0 && target.end > target.start;
          }
        }
      } else if (payload.operation === 'revise') valid = exact(['operation','feedback_id','expected_revision','quick_tags','freeform_reaction']);
      else if (payload.operation === 'withdraw') valid = exact(['operation','feedback_id','expected_revision']);
      else if (payload.operation === 'delete') valid = exact(['operation','feedback_id']);
      else if (payload.operation === 'preview' && Object.hasOwn(payload, 'feedback_ids')) {
        valid = exact(['operation','feedback_ids']) && Array.isArray(payload.feedback_ids) &&
          payload.feedback_ids.length > 0 && payload.feedback_ids.length <= 30 &&
          payload.feedback_ids.every(value => typeof value === 'string' && value.length > 0 && value.length <= 128);
      } else if (payload.operation === 'preview' && Object.hasOwn(payload, 'redacted_payload')) {
        const redacted = payload.redacted_payload;
        valid = exact(['operation','preview_id','expected_revision','expected_sha256','redacted_payload']) &&
          typeof payload.preview_id === 'string' && payload.preview_id.length > 0 && payload.preview_id.length <= 128 &&
          Number.isSafeInteger(payload.expected_revision) && payload.expected_revision >= 1 &&
          sha256(payload.expected_sha256) && plainJson(redacted) &&
          redacted && typeof redacted === 'object' &&
          !Array.isArray(redacted) && Object.keys(redacted).length === 3 &&
          Object.keys(redacted).every(key => ['schema_version','feedback','context'].includes(key)) &&
          Array.isArray(redacted.feedback) && redacted.feedback.length <= 30 &&
          Array.isArray(redacted.context) && redacted.context.length <= 30;
        maxBytes = 40000;
      } else if (payload.operation === 'confirm') {
        valid = exact(['operation','preview_id','preview_revision','payload_sha256',
          'redaction_confirmed','external_processing_confirmed']) &&
          typeof payload.preview_id === 'string' && payload.preview_id.length > 0 && payload.preview_id.length <= 128 &&
          Number.isSafeInteger(payload.preview_revision) && payload.preview_revision >= 1 &&
          sha256(payload.payload_sha256) && typeof payload.redaction_confirmed === 'boolean' &&
          typeof payload.external_processing_confirmed === 'boolean';
      }
      const serializedBytes = new TextEncoder().encode(JSON.stringify(payload)).byteLength;
      if (!valid || serializedBytes > maxBytes) throw new Error('Invalid payload');
      return service.feedback(payload);
    }
    if (action === 'memories') {
      if (Object.keys(payload).some(key => !['offset', 'limit'].includes(key)) ||
          Object.values(payload).some(value => !Number.isInteger(value) || value < 0) ||
          (payload.limit !== undefined && (payload.limit < 1 || payload.limit > 100)) ||
          (payload.offset !== undefined && payload.offset > 1e6)) throw new Error('Invalid payload');
      return service.memories(payload);
    }
    if (action === 'memoryDetails') {
      if (Object.keys(payload).length !== 1 || typeof payload.record_id !== 'string' ||
          !payload.record_id.length || payload.record_id.length > 128) throw new Error('Invalid payload');
      return service.memoryDetails(payload);
    }
    if (action === 'memoryAction') {
      const required = ['operation_id','action','record_id','expected_version','replacement'];
      const allowed = [...required, 'resolves_job'];
      if (required.some(key => !Object.hasOwn(payload, key)) ||
          Object.keys(payload).some(key => !allowed.includes(key)) ||
          !['correct','forget'].includes(payload.action) ||
          !Number.isInteger(payload.expected_version) || payload.expected_version < 1 ||
          [payload.operation_id,payload.record_id].some(value => typeof value !== 'string' || !value.length || value.length > 128) ||
          (payload.resolves_job !== undefined && (typeof payload.resolves_job !== 'string' || !payload.resolves_job.length || payload.resolves_job.length > 128)) ||
          typeof payload.replacement !== 'string' || payload.replacement.length > 16000 ||
          (payload.action === 'correct' && !payload.replacement.trim()) ||
          (payload.action === 'forget' && payload.replacement) || JSON.stringify(payload).length > 18000) throw new Error('Invalid payload');
      return service.memoryAction(payload);
    }
    if (action === 'memoryAbandon') {
      if (Object.keys(payload).length !== 2 ||
          ['job','operation_id'].some(key => typeof payload[key] !== 'string' || !payload[key].length || payload[key].length > 128)) throw new Error('Invalid payload');
      return service.memoryAbandon(payload);
    }
    const keys = action === 'send' ? ['text'] : action === 'selectSession' ? ['session_id'] : action === 'selectProvider' ? ('confirmed' in payload ? ['provider_id', 'confirmed'] : ['provider_id']) : action === 'sessions' ? ['offset'] : [];
    if (Object.keys(payload).some(key => !keys.includes(key))) throw new Error('Invalid payload');
    if (action === 'sessions' && payload.offset !== undefined && (!Number.isInteger(payload.offset) || payload.offset < 0 || payload.offset > 1e9)) throw new Error('Invalid payload');
    if (action === 'selectSession' && (typeof payload.session_id !== 'string' || payload.session_id.length > 128)) throw new Error('Invalid payload');
    if (action === 'selectProvider' && (typeof payload.provider_id !== 'string' || !payload.provider_id.length || payload.provider_id.length > 64)) throw new Error('Invalid payload');
    if (action === 'selectProvider' && 'confirmed' in payload && typeof payload.confirmed !== 'boolean') throw new Error('Invalid payload');
    if (action === 'send' && (typeof payload.text !== 'string' || !payload.text.trim() || payload.text.length > 8000)) {
      throw new Error('Invalid payload');
    }
    return service[action](payload);
  };
}
