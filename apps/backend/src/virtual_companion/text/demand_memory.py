"""Bounded JSON action carrier over the existing checked text request Port.

The caller owns intake, source delegation, refreshed admission and delivery.
This carrier never clears a source barrier, grants presentation approval, or
turns structured text into provider function-calling capabilities.
"""

import json
import re
from dataclasses import dataclass

from virtual_companion.boundary.memory_relevance import (
    MAX_RECALL_QUERY_CHARS,
    validate_recall_query,
)

from .evidence_recall import EvidenceSelection
from .ports import PreparedTextRequest

ANSWER_SCHEMA = {
    "type": "object", "properties": {
        "action": {"const": "answer"}, "text": {"type": "string", "minLength": 1},
    }, "required": ["action", "text"], "additionalProperties": False,
}
ACK_SCHEMA = {
    **ANSWER_SCHEMA, "properties": {
        "action": {"const": "ack"}, "text": {"type": "string", "minLength": 1},
    },
}
LISTEN_SCHEMA = {
    "type": "object", "properties": {"action": {"const": "listen"}},
    "required": ["action"], "additionalProperties": False,
}
RESPONSE_SCHEMA = {"type": "object", "oneOf": [ANSWER_SCHEMA, ACK_SCHEMA, LISTEN_SCHEMA]}
ACTION_SCHEMA = {"type": "object", "oneOf": [*RESPONSE_SCHEMA["oneOf"], {
    "type": "object", "properties": {
        "action": {"const": "memory"},
        "query": {"type": "string", "minLength": 1, "maxLength": MAX_RECALL_QUERY_CHARS},
    },
    "required": ["action"], "additionalProperties": False,
}]}


RECALL_SCHEMA = {
    "type": "object", "properties": {
        "action": {"const": "recall"},
        "query": {"type": "string", "minLength": 1, "maxLength": MAX_RECALL_QUERY_CHARS},
    }, "required": ["action", "query"], "additionalProperties": False,
}

CONSTRAINED_RECALL_VERSION = 'constrained-recall.v1'
CONSTRAINED_RECALL_SCHEMA = {
    **RECALL_SCHEMA, 'properties': {
        **RECALL_SCHEMA['properties'],
        'exact_terms': {'type': 'array', 'minItems': 1, 'maxItems': 6,
                        'items': {'type': 'string', 'minLength': 1, 'maxLength': 64}},
        'conditions': {'type': 'object', 'minProperties': 1,
                       'properties': {field: {'type': 'string', 'minLength': 1, 'maxLength': 128}
                                      for field in ('subject', 'predicate', 'scope')},
                       'additionalProperties': False},
    },
}


def recall_action(value, *, constrained=False):
    """Validate at the host boundary, including limits JSON Schema cannot encode."""
    from jsonschema import Draft202012Validator

    from virtual_companion.cognition.hybrid_retrieval import query_constraints

    schema = CONSTRAINED_RECALL_SCHEMA if constrained else RECALL_SCHEMA
    if not Draft202012Validator(schema).is_valid(value):
        raise ValueError('Exact bounded recall action required')
    query = validate_recall_query(value['query'])
    if query is None:
        raise ValueError('Nonempty recall query required')
    query_constraints(value.get('exact_terms', ()), value.get('conditions'))
    return {**value, 'query': query}


NAVIGATION_SCHEMAS = [{
    'type': 'object', 'properties': {
        'action': {'const': 'browse_memory'},
        'node_id': {'type': 'string', 'minLength': 1, 'maxLength': 128},
        'cursor': {'type': ['string', 'null'], 'maxLength': 512},
    }, 'required': ['action', 'node_id'], 'additionalProperties': False,
}, {
    'type': 'object', 'properties': {
        'action': {'const': 'read_memory'},
        'references': {'type': 'array', 'minItems': 1, 'maxItems': 8, 'items': {
            'type': 'object', 'properties': {
                'record_id': {'type': 'string', 'minLength': 1, 'maxLength': 128},
                'version': {'type': 'integer', 'minimum': 1},
            }, 'required': ['record_id', 'version'], 'additionalProperties': False}},
    }, 'required': ['action', 'references'], 'additionalProperties': False,
}]


def action_schema(*, allow_memory, allow_recall, allow_navigation=False,
                  allow_constrained_recall=False, allow_public_search=False,
                  allow_public_extract=False):
    """Explicit capability selection; the adopted schemas remain byte-compatible."""
    if any(type(value) is not bool for value in (
            allow_memory, allow_recall, allow_navigation,
            allow_constrained_recall, allow_public_search, allow_public_extract)):
        raise ValueError("Explicit action capabilities required")
    if allow_public_search or allow_public_extract:
        from virtual_companion.providers.external_information import EXTRACT_SCHEMA, SEARCH_SCHEMA

        base = action_schema(allow_memory=allow_memory, allow_recall=allow_recall,
                             allow_navigation=allow_navigation,
                             allow_constrained_recall=allow_constrained_recall)
        return {'type': 'object', 'oneOf': [*base['oneOf'],
                *([SEARCH_SCHEMA] if allow_public_search else []),
                *([EXTRACT_SCHEMA] if allow_public_extract else [])]}
    if not allow_recall:
        return ACTION_SCHEMA if allow_memory else RESPONSE_SCHEMA
    choices = [*RESPONSE_SCHEMA["oneOf"]]
    if allow_memory:
        choices.append(ACTION_SCHEMA["oneOf"][-1])
    recall = CONSTRAINED_RECALL_SCHEMA if allow_constrained_recall else RECALL_SCHEMA
    return {"type": "object", "oneOf": [*choices, recall,
                *(NAVIGATION_SCHEMAS if allow_navigation else [])]}


@dataclass(frozen=True)
class DemandTurnResult:
    claim: object
    text: str
    prepared: PreparedTextRequest
    generation_calls: int
    memory_calls: int
    response_type: str = "answer"
    delivery_status: str = "completed"
    recall_calls: int = 0
    public_calls: int = 0


@dataclass(frozen=True)
class TextCandidate:
    """Buffered text with the same action identity as a decoded live candidate."""

    response_type: str
    text: str

    def __iter__(self):
        yield self.text


@dataclass(frozen=True)
class MalformedAnswerEnvelope:
    """A complete decoded string whose original carrier failed JSON syntax."""

    raw: str
    text: str
    response_type: str
    error: str


def _checkpoint(cancelled):
    if cancelled.is_set():
        raise InterruptedError("Demand turn cancelled")


def _unique(pairs, *, strict=False):
    result = {}
    for key, value in pairs:
        if key in result and (
            strict or key != "text" or not isinstance(value, str) or value != result[key]
        ):
            raise ValueError("Duplicate demand action field")
        result[key] = value
    return result


def _reject_json_constant(value):
    raise ValueError(f"Non-standard demand JSON constant: {value}")


def _action(text, *, final, strict=False, allow_recall=False, allow_navigation=False,
            allow_constrained_recall=False, allow_public_search=False, allow_public_extract=False):
    value = json.loads(text, object_pairs_hook=lambda pairs: _unique(pairs, strict=strict),
                       parse_constant=_reject_json_constant)
    if not isinstance(value, dict):
        raise ValueError("Demand action must be one JSON object")
    if strict and set(value) != {"action", "text"}:
        raise ValueError("Carrier recovery requires exact action and text fields")
    action = value.get("action")
    if isinstance(action, str) and action in {"answer", "ack"}:
        if isinstance(value.get("text"), str) and value["text"].strip():
            return {"action": action, "text": value["text"]}
    if value == {"action": "listen"}:
        return value
    if (not final and value.get("action") == "memory"
            and set(value) in ({"action"}, {"action", "query"})):
        if "query" in value:
            if not isinstance(value["query"], str):
                raise ValueError("Plain-text memory query required")
            value["query"] = validate_recall_query(value["query"])
        return value
    if allow_recall and value.get('action') == 'recall':
        return recall_action(value, constrained=allow_constrained_recall)
    if ((allow_public_search and value.get('action') == 'search_public')
            or (allow_public_extract and value.get('action') == 'extract_public')):
        from virtual_companion.providers.external_information import parse_public_action

        return parse_public_action(value)
    if (allow_recall and allow_navigation and isinstance(action, str)
            and action in {'browse_memory', 'read_memory'}):
        from jsonschema import Draft202012Validator

        if Draft202012Validator({'oneOf': NAVIGATION_SCHEMAS}).is_valid(value):
            return value
    raise ValueError("Exact answer, ack, listen or permitted evidence action required")


_ANSWER_START = re.compile(r'\s*\{\s*"action"\s*:\s*"(answer|ack)"\s*,\s*"text"\s*:\s*"')


class _AnswerStream:
    """Decode only the stable answer string; complete JSON still owns validity.

    Escaped characters and surrogate pairs are held until they are complete.
    This is unapproved candidate text. Only the caller's reviewer may publish it.
    Other legal field orders keep the existing buffered action path.
    """

    def __init__(self, fragments, start, *, final, response_type="answer", recover=None):
        self.fragments, self.start, self.final = fragments, start, final
        self.text, self.action = "", None
        self._response_type = response_type
        self.recover = recover

    @property
    def response_type(self):
        return self._response_type

    def __iter__(self):
        raw, cursor, closed = "", self.start, False
        for fragment in self.fragments:
            raw += fragment
            decoded = []
            while not closed and cursor < len(raw):
                char = raw[cursor]
                if char == '"':
                    closed = True
                    break
                if char != "\\":
                    if ord(char) < 32 or 0xD800 <= ord(char) <= 0xDFFF:
                        raise ValueError("Invalid answer string character")
                    decoded.append(char)
                    cursor += 1
                    continue
                if cursor + 1 >= len(raw):
                    break
                size = 6 if raw[cursor + 1] == "u" else 2
                if cursor + size > len(raw):
                    break
                token = raw[cursor:cursor + size]
                value = json.loads('"' + token + '"')
                if len(value) == 1 and 0xD800 <= ord(value) <= 0xDBFF:
                    if cursor + 12 > len(raw):
                        break
                    token = raw[cursor:cursor + 12]
                    value = json.loads('"' + token + '"')
                    size = 12
                if any(0xD800 <= ord(item) <= 0xDFFF for item in value):
                    raise ValueError("Unpaired answer surrogate")
                decoded.append(value)
                cursor += size
            if decoded:
                delta = "".join(decoded)
                self.text += delta
                yield delta
        try:
            self.action = _action(raw, final=self.final)
        except json.JSONDecodeError as error:
            if not closed or not self.text.strip() or self.recover is None:
                raise
            # No decoded text is yielded again. The host may obtain one legal
            # carrier for exactly this complete string, before final review.
            self.action = self.recover(MalformedAnswerEnvelope(
                raw, self.text, self.response_type, f"{error.msg} at char {error.pos}"))
        if (self.action["action"] != self.response_type
                or self.action["text"] != self.text):
            raise ValueError("Answer terminal differs from decoded candidate")


def _generate(prepared, cancelled, *, final, on_answer=None, recover_answer=None, strict=False,
              allow_recall=False, allow_navigation=False, allow_constrained_recall=False,
              allow_public_search=False, allow_public_extract=False):
    _checkpoint(cancelled)
    prepared.check_context()
    _checkpoint(cancelled)
    stream = prepared.stream(cancelled)
    fragments, size = [], 0

    def receive():
        nonlocal size
        for fragment in stream:
            _checkpoint(cancelled)
            if not isinstance(fragment, str):
                raise ValueError("Text request returned a non-text fragment")
            size += len(fragment.encode("utf-8"))
            if size > 64000:
                raise ValueError("Demand action exceeds bounded response size")
            yield fragment
        _checkpoint(cancelled)

    incoming = receive()
    try:
        for fragment in incoming:
            fragments.append(fragment)
            prefix = "".join(fragments)
            match = _ANSWER_START.match(prefix) if on_answer is not None else None
            if match is not None:
                def remaining(prefix=prefix):
                    yield prefix
                    yield from incoming
                answer = _AnswerStream(remaining(), match.end(), final=final,
                    response_type=match.group(1), recover=(
                        (lambda failure: recover_answer(prepared, failure))
                        if recover_answer is not None else None))
                on_answer(prepared, answer)
                _checkpoint(cancelled)
                if answer.action is None:
                    raise ValueError("Answer consumer did not exhaust the checked stream")
                return answer.action
        _checkpoint(cancelled)
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            close()
    _checkpoint(cancelled)
    return _action("".join(fragments), final=final, strict=strict, allow_recall=allow_recall,
                   allow_navigation=allow_navigation,
                   allow_constrained_recall=allow_constrained_recall,
                   allow_public_search=allow_public_search,
                   allow_public_extract=allow_public_extract)


def run_demand_turn(*, claim, prepare, resolve_current, refresh, cancelled, on_answer=None,
                    recover_answer=None, recall_evidence=None, max_recall_calls=0,
                    memory_calls_used=0, recall_calls_used=0, evidence=None,
                    allow_navigation=False, allow_constrained_recall=False,
                    search_public=None, max_public_calls=0, public_calls_used=0,
                    extract_public=None, allow_public_extract=False):
    """One source delegation and optional bounded readonly actions under one host.

    The host owns owner/deadline/provider budgets and exact readset admission.
    Evidence is never source-processing completion or presentation permission.
    Existing callers omit readonly capability and retain the two-stage protocol.
    Repair continues the same memory/recall counters and evidence, without reset.
    """
    if (type(max_public_calls) is not int or max_public_calls not in (0, 1)
            or type(public_calls_used) is not int or not 0 <= public_calls_used <= max_public_calls
            or (max_public_calls and not callable(search_public))
            or type(allow_public_extract) is not bool
            or (allow_public_extract and (not max_public_calls or not callable(extract_public)))
            or type(allow_constrained_recall) is not bool
            or (allow_constrained_recall and not allow_navigation)
            or type(max_recall_calls) is not int
            or not 0 <= max_recall_calls <= (6 if allow_navigation else 2)
            or (max_recall_calls and not callable(recall_evidence))
            or type(memory_calls_used) is not int or memory_calls_used not in (0, 1)
            or type(recall_calls_used) is not int
            or not 0 <= recall_calls_used <= max_recall_calls):
        raise ValueError("Bounded same-turn evidence capabilities required")
    memory_calls, recall_calls = memory_calls_used, recall_calls_used
    recall_query, recovery_calls, generation_calls = None, 0, 0
    public_calls = public_calls_used

    def recover(prepared, failure):
        nonlocal recovery_calls
        recovery_calls += 1
        return recover_answer(prepared, failure)

    for _ in range(2 + max_recall_calls + max_public_calls):
        _checkpoint(cancelled)
        final = memory_calls == 1
        allowed = bool(max_recall_calls and recall_calls < max_recall_calls
                       and (evidence is None or evidence.can_recall))
        kwargs = {"final": final}
        if recall_query is not None:
            kwargs["recall_query"] = recall_query
        if max_recall_calls:
            kwargs.update(allow_recall=allowed, evidence=evidence)
        public_allowed = public_calls < max_public_calls
        if max_public_calls:
            kwargs['allow_public_search'] = public_allowed
        if allow_public_extract:
            kwargs['allow_public_extract'] = public_allowed
        prepared = prepare(claim, **kwargs)
        if max_recall_calls and hasattr(prepared, "readonly_evidence"):
            current = prepared.readonly_evidence
            if not isinstance(current, EvidenceSelection):
                raise ValueError("Prepared readonly evidence must be host-bound")
            evidence = current
            allowed = recall_calls < max_recall_calls and current.can_recall
        _checkpoint(cancelled)
        generation_calls += 1
        action = _generate(prepared, cancelled, final=final, on_answer=on_answer,
                           recover_answer=recover if recover_answer is not None else None,
                           allow_recall=allowed, allow_navigation=allow_navigation,
                           allow_constrained_recall=allow_constrained_recall,
                           allow_public_search=public_allowed,
                           allow_public_extract=allow_public_extract and public_allowed)
        _checkpoint(cancelled)
        if action['action'] in {'search_public', 'extract_public'}:
            public_calls += 1
            if action['action'] == 'search_public':
                search_public(claim, action['query'], cancelled=cancelled)
            else:
                extract_public(claim, action['url'], cancelled=cancelled)
            _checkpoint(cancelled)
            continue
        if action["action"] in {"recall", "browse_memory", "read_memory"}:
            recall_calls += 1
            request = (action['query'] if action['action'] == 'recall'
                       and set(action) == {'action', 'query'} else action)
            evidence = recall_evidence(claim, request, cancelled=cancelled)
            _checkpoint(cancelled)
            continue
        if action["action"] != "memory":
            return DemandTurnResult(claim, action.get("text", ""), prepared,
                                    generation_calls + recovery_calls, memory_calls,
                                    action["action"], recall_calls=recall_calls,
                                    public_calls=public_calls)
        # Close the inference stream before either capability can execute.
        resolve_current(claim)
        memory_calls += 1
        _checkpoint(cancelled)
        claim = refresh(claim)
        recall_query = action.get("query")
        _checkpoint(cancelled)
    raise AssertionError("Bounded action carrier did not produce a response")
