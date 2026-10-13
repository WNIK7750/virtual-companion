const scopeFields = ['session_id', 'input_id', 'attempt_id', 'turn_id', 'text_attempt_id'];

export function approvedScopeMatches(scope, state, sessionId) {
  return state?.delivery_protocol === 'approved-prefix.v1' && state.busy === true
    && scopeFields.every(key => typeof scope?.[key] === 'string' && scope[key].length > 0
      && scope[key] === state.delivery_scope?.[key])
    && scope.session_id === sessionId && scope.session_id === state.session_id
    && scope.text_attempt_id === state.attempt?.attempt_id && scope.turn_id === state.attempt?.turn_id;
}

export function visiblePartial(state) {
  if (state.delivery_protocol === 'approved-prefix.v1') return '';
  return state.partial && state.attempt?.status !== 'completed'
    ? state.partial + (state.busy ? '' : '\n（未完成，未计入历史）') : '';
}

export function canRetry(state) {
  return !state.busy && state.retry_allowed !== false
    && ['cancelled', 'failed', 'interrupted'].includes(state.attempt?.status);
}

export function mergeApprovedMessages(messages, approved) {
  const confirmed = new Set(messages.filter(message => message.role === 'assistant').map(message => message.turn_id));
  return [...messages, ...approved.filter(message => !confirmed.has(message.turn_id))];
}
