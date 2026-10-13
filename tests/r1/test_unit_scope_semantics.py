"""Compact protocol/identity recordings with declared CPU opinions, not semantic scores."""
import asyncio
import json
import os
from contextlib import asynccontextmanager
from copy import deepcopy
from pathlib import Path
from threading import Event, Thread

import pytest
from jsonschema import Draft202012Validator
from test_rebuilt_turns import A, claim, config
from test_stream_unit_review import protocol_packet
from virtual_companion.boundary.contracts import canonical
from virtual_companion.boundary.presentation import digest
from virtual_companion.boundary.sqlite_streaming import SQLiteStreamingStore
from virtual_companion.cognition import stream_unit_review as review_port
from virtual_companion.cognition.short_dialogue import ShortContext, ShortRequest
from virtual_companion.cognition.stream_unit_review import (
    BoundedUnitReviewer,
    CandidateRejected,
    StructuredUnitAssessor,
    candidate_segments,
    semantic_review_view,
)
from virtual_companion.providers import httpx_transport
from virtual_companion.providers.contracts import ProviderCapabilities, ProviderRoute
from virtual_companion.providers.httpx_transport import HttpxProviderTransport, ProviderFailure
from virtual_companion.text.provider_model import ProviderModel

COMPACT = 'candidate-unit-scope.compact.v5'
CHECK_KEYS = ('self_contained', 'no_open_dependency', 'consistent_with_displayed',
              'constraints_satisfied')
USER = '我那几张公交卡边角磨亮了。'
TEXT = '你提到卡的边角磨亮了。一般物件也能让人想到某次出门。'


def envelope(user=USER, text=TEXT):
    packet = protocol_packet(user_text=user, candidate=text)
    packet['payload']['audit_version'] = COMPACT
    packet['sha256'] = digest(packet['payload'])
    return packet


def supplied(payload, *, personal='none', personal_ids=(), fiction='none', fiction_ids=(),
             decision='approve', checks=(True, True, True, True)):
    return dict(audit_version=COMPACT, decision=decision, checks=list(checks), units=[
        [f'u{index + 1}', personal, list(personal_ids), fiction, list(fiction_ids)]
        for index, _ in enumerate(payload['review_segments'])])


def record(label, value):
    directory = os.environ.get('P103_COMPACT_REVIEW_RECORD_DIR')
    if directory:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / (label + '.json')).write_text(canonical(value) + '\n', encoding='utf-8')


def invoke(packet, opinion):
    calls = []

    def declared(stage, messages, schema, cancelled):
        assert not cancelled.is_set()
        calls.append(dict(stage=stage, messages=deepcopy(messages), schema=deepcopy(schema)))
        assert json.loads(messages[-1]['content']) == semantic_review_view(packet['payload'])
        Draft202012Validator(schema).validate(opinion)
        return deepcopy(opinion)

    bound = StructuredUnitAssessor(declared, Event(), audit_version=COMPACT)(packet)
    assert len(calls) == 1 and calls[0]['stage'] == 'stream_unit_review'
    record('declared-' + digest(opinion)[:12], dict(envelope=packet, calls=calls,
        supplied_opinion=opinion, bound_verdict=bound, semantic_effect_proven=False))
    return bound, calls


def test_compact_port_is_explicit_and_binds_full_original_ranges():
    packet = envelope()
    value = supplied(packet['payload'], personal='supported', personal_ids=['s1'])
    value['units'].reverse()
    bound, calls = invoke(packet, value)
    assert review_port.COMPACT_UNIT_SCOPE_AUDIT_VERSION == COMPACT
    assert review_port.SUPPORTED_AUDIT_VERSIONS == (
        'candidate-propositions.v3', 'candidate-unit-scope.v4', COMPACT,
        'candidate-unit-grounding.v6', 'candidate-unit-meaning.v7', 'candidate-unit-verdict.v8')
    assert bound['model_opinion'] == value and bound['audit_version'] == COMPACT
    assert bound['request_sha256'] == packet['sha256']
    assert ''.join(unit['text'] for unit in bound['unit_scopes']) == TEXT
    assert [unit['segment_id'] for unit in bound['unit_scopes']] == [
        unit['segment_id'] for unit in candidate_segments(TEXT)]
    assert all(unit['personal_source_ids'] == ['current:source-input']
               for unit in bound['unit_scopes'])
    assert calls[0]['schema']['properties']['units']['items']['items'] is False
    assert set(value) == {'audit_version', 'decision', 'checks', 'units'}


def test_compact_view_keeps_every_word_source_access_operation_and_shown_range():
    packet = envelope()
    payload = packet['payload']
    source = payload['personal_sources'][0]
    history = {**source, 'id': 'history:old-input', 'kind': 'history',
               'source_id': 'old-input', 'text': '我以前偏爱慢走。'}
    memory = {**source, 'id': 'memory:valid-fact', 'kind': 'memory', 'revision': 3,
              'source_id': 'corrected-input', 'text': '我现在更偏爱骑车。'}
    payload['personal_sources'] += [history, memory]
    context = dict(history=[dict(id='old-input', role='user', text=history['text']),
                            dict(id='assistant-input', role='assistant', text='已收到更正。')],
                   memories=[dict(record_id='valid-fact', version=3, status='active',
                                  content=memory['text'])],
                   memory_status='pending_source', memory_access='not_accessed',
                   source_processing=dict(protocol='current-source-processing.v1',
                       source_id='source-input', source_sha256='opaque-original-sha',
                       status='queued', error=None, receipt_sha256=None),
                   current_operation=dict(action='correct', status='committed'),
                   exchanges=[dict(content='仍待回答的问题', direction='assistant_response')])
    payload.update(context=canonical(context), displayed_prefix=TEXT[:5],
                   constraints=[dict(kind='max_chars', limit=160, span='160字以内')],
                   response_task_contract=dict(kind='exact_items', count=2, span='给两条'),
                   response_type='ack', forbidden_characters='!', unparsed_constraints=['保留条件'])
    before = deepcopy(payload)
    view = semantic_review_view(payload)
    assert payload == before and view['view_version'] == 'unit-scope-compact-semantics.v1'
    assert view['audit_version'] == COMPACT and view['user_source_id'] == 's1'
    assert [unit['id'] for unit in view['units']] == ['u1', 'u2']
    assert ''.join(unit['text'] for unit in view['units']) == TEXT
    assert view['sources'] == [dict(id=f's{index + 1}', kind=item['kind'], text=item['text'])
                              for index, item in enumerate(payload['personal_sources'])]
    assert view['context']['history'][0]['personal_source_id'] == 's2'
    assert view['context']['memories'][0]['personal_source_id'] == 's3'
    assert view['context']['history'][1] == context['history'][1]
    for key in ('memory_status', 'memory_access', 'source_processing',
                'current_operation', 'exchanges'):
        assert view['context'][key] == context[key]
    assert view['displayed_units'] == []
    assert view['displayed_partial']['unit_id'] == 'u1'
    assert view['displayed_partial']['text'] == payload['displayed_prefix']
    for key in ('final', 'constraints', 'response_task_contract', 'response_type',
                'forbidden_characters', 'unparsed_constraints'):
        assert view[key] == payload[key]
    record('full-view', dict(original_payload=payload, view=view, semantic_effect_proven=False))


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'foreign-unit', 'foreign-source',
                                    'missing-check', 'nonboolean', 'extra-column', 'old-version'])
def test_compact_missing_duplicate_foreign_or_malformed_relations_never_approve(mutation):
    packet = envelope()
    value = supplied(packet['payload'], personal='supported', personal_ids=['s1'])
    if mutation == 'missing':
        value['units'].pop()
    elif mutation == 'duplicate':
        value['units'].append(deepcopy(value['units'][0]))
    elif mutation == 'foreign-unit':
        value['units'][0][0] = 'u-from-other-packet'
    elif mutation == 'foreign-source':
        value['units'][0][2] = ['s-from-other-packet']
    elif mutation == 'missing-check':
        value['checks'].pop()
    elif mutation == 'nonboolean':
        value['checks'][0] = 1
    elif mutation == 'extra-column':
        value['units'][0].append('copied claim')
    else:
        value['audit_version'] = 'candidate-unit-scope.v4'
    with pytest.raises(ValueError):
        StructuredUnitAssessor(lambda *args: value, Event(), audit_version=COMPACT)(packet)


@pytest.mark.parametrize('kind,status', [('personal', 'unsupported'), ('personal', 'uncertain'),
                                      ('fiction', 'unauthorized'), ('fiction', 'uncertain')])
def test_compact_adverse_whole_unit_rejects_without_laundering_original_approve(kind, status):
    packet = envelope(text='卡片能留住一段记忆，你每天经过那条带顶棚的路。')
    value = supplied(packet['payload'], **{kind: status})
    bound, _ = invoke(packet, value)
    assert bound['decision'] == 'reject' and bound['constraints_satisfied'] is False
    assert bound['model_opinion'] == value and value['decision'] == 'approve'
    assert bound['unit_scopes'][0][kind + '_support'] == status


@pytest.mark.parametrize('which,decision', [(0, 'hold'), (1, 'hold'), (2, 'reject'), (3, 'reject')])
def test_compact_none_keeps_all_four_whole_candidate_checks(which, decision):
    packet = envelope()
    checks = [True] * 4
    checks[which] = False
    bound, _ = invoke(packet, supplied(packet['payload'], checks=checks))
    assert bound['decision'] == decision and bound[CHECK_KEYS[which]] is False


@pytest.mark.parametrize('current', [True, False])
def test_compact_fiction_authorization_is_only_this_current_source(current):
    packet = envelope(user='请写一个虚构雨天片段。', text='虚构中，小孩在雨里追一盏灯。')
    source = packet['payload']['personal_sources'][0]
    packet['payload']['personal_sources'].append({**source, 'id': 'history:old-permission',
        'kind': 'history', 'source_id': 'old-permission'})
    packet['sha256'] = digest(packet['payload'])
    value = supplied(
        packet['payload'], fiction='authorized', fiction_ids=['s1' if current else 's2'],
    )
    if current:
        assert invoke(packet, value)[0]['decision'] == 'approve'
    else:
        with pytest.raises(ValueError, match='current user authorization'):
            invoke(packet, value)


def test_compact_false_supported_or_none_remains_an_observable_model_counterexample():
    packet = envelope(user='我收好了几张旧卡片。', text='你每天都沿着那条有顶棚的路走。')
    for value in (supplied(packet['payload']), supplied(packet['payload'],
                  personal='supported', personal_ids=['s1'])):
        bound, _ = invoke(packet, value)
        assert bound['decision'] == 'approve' and bound['model_opinion'] == value
        # A supplied semantic lie is not corrected by a host word classifier.
        assert bound['unit_scopes'][0]['text'] == packet['payload']['candidate_prefix']
    record('false-semantic-counterexample', dict(envelope=packet, declared_false=value,
        bound_verdict=bound, semantic_effect_proven=False))


def test_compact_packet_and_verdict_cannot_cross_sealed_candidate_or_source_revision():
    packet = envelope()
    bound, _ = invoke(packet, supplied(packet['payload']))
    for field in ('candidate_prefix', 'source_revision'):
        foreign = deepcopy(packet)
        if field == 'candidate_prefix':
            foreign['payload']['candidate_prefix'] = '另一个候选。'
        else:
            foreign['payload']['personal_sources'][0]['revision'] += 1
        foreign['sha256'] = digest(foreign['payload'])
        with pytest.raises(ValueError):
            BoundedUnitReviewer._validate_opinion(bound, foreign['sha256'], foreign['payload'])
    rewritten = deepcopy(bound)
    rewritten['unit_scopes'][0]['text'] = '被改写的字'
    with pytest.raises(ValueError):
        BoundedUnitReviewer._validate_opinion(rewritten, packet['sha256'], packet['payload'])


def test_compact_early_and_final_keep_original_prefix_full_context_and_call_limits(tmp_path):
    store = SQLiteStreamingStore(tmp_path / 'compact-prefix.sqlite3')
    run = claim(store, store.accept(config(), 'compact-input', USER))
    request = ShortRequest(USER, context=ShortContext(A, run.input_id, run.state_version,
        canonical(dict(memory_status='current', history=[], memories=[], exchanges=[]))))
    packets = []

    def declared(packet):
        packets.append(deepcopy(packet))
        return StructuredUnitAssessor(lambda *args: supplied(packet['payload']), Event(),
                                      audit_version=COMPACT)(packet)

    reviewer = BoundedUnitReviewer(run, request, declared, audit_version=COMPACT)
    prefix, suffix = '一般物件也能让人想到出门。', '不必替它编造具体经历。'
    first = reviewer(prefix, False)
    final = reviewer(prefix + suffix, True)
    assert first.candidate.text == prefix and final.candidate.text == prefix + suffix
    assert final.candidate.candidate_id == first.candidate.candidate_id
    assert final.candidate.attempt_id == run.attempt_id and final.candidate.input_id == run.input_id
    assert [unit.text for unit in final.candidate.units] == [prefix, suffix]
    assert reviewer.early_calls == reviewer.final_calls == 1
    assert len(packets) == 2 and packets[1]['payload']['displayed_prefix'] == prefix
    assert packets[1]['payload']['candidate_prefix'] == prefix + suffix
    assert packets[1]['payload']['personal_sources'] == packets[0]['payload']['personal_sources']
    with pytest.raises(ValueError, match='revise'):
        reviewer('改掉已批准前缀。', True)
    with pytest.raises(ValueError, match='Final opinion already spent'):
        reviewer(prefix + suffix + '新的未审尾部。', True)


def test_compact_rejection_retains_feedback_and_does_not_spend_another_review(tmp_path):
    store = SQLiteStreamingStore(tmp_path / 'compact-rejected.sqlite3')
    run = claim(store, store.accept(config(), 'rejected-input', USER))
    request = ShortRequest(USER)
    calls = []

    def declared(packet):
        calls.append(deepcopy(packet))
        value = supplied(packet['payload'], personal='unsupported')
        return StructuredUnitAssessor(lambda *args: value, Event(), audit_version=COMPACT)(packet)

    reviewer = BoundedUnitReviewer(run, request, declared, audit_version=COMPACT)
    with pytest.raises(CandidateRejected) as first:
        reviewer('你每天经过那条有顶棚的路。', False)
    assert reviewer.approved is None and len(calls) == reviewer.early_calls == 1
    assert reviewer.final_calls == 0
    assert first.value.verdict['model_opinion']['decision'] == 'approve'
    assert first.value.verdict['decision'] == 'reject'
    with pytest.raises(CandidateRejected) as repeated:
        reviewer('你每天经过那条有顶棚的路。新的尾部。', True)
    assert repeated.value is first.value and len(calls) == 1


class RecordingCounter:
    def __init__(self):
        self.bodies = []

    def __call__(self, body):
        self.bodies.append(deepcopy(body))
        return 17

    def local_byte_bound(self, body):
        return len(canonical(body).encode())


class RecordingReservation:
    def __init__(self):
        self.rows, self.terminals = [], []

    def reserve(self, **values):
        self.rows.append(deepcopy(values))
        return 'cpu-reservation'

    def finish(self, receipt, **values):
        assert receipt == 'cpu-reservation'
        self.terminals.append(deepcopy(values))


class PublicSyntheticCredential:
    def get(self, reference, endpoint):
        assert reference == 'dpapi:' + 'a' * 32 and endpoint == 'https://api.deepseek.com/chat/completions'
        return 'PUBLIC-CPU-NONCREDENTIAL'


@pytest.mark.parametrize('terminal', ['valid', 'eof', 'length', 'missing-usage', 'cancel'])
def test_compact_actual_prepared_counter_send_waits_for_complete_stop_usage_and_binding(
    monkeypatch, tmp_path, terminal
):
    packet = envelope()
    opinion = supplied(packet['payload'])
    counted, ledger = RecordingCounter(), RecordingReservation()
    configured = ProviderRoute(
        'compact-recording', 1, 'api', 'https://api.deepseek.com/chat/completions',
        'deepseek-flash', 'bearer', 'dpapi:' + 'a' * 32,
        True, ('current', 'history', 'valid_memory', 'derived'),
        ProviderCapabilities(True, False, True, 1_048_576, 'max_tokens'),
        wire_profile='deepseek-flash.focused.v2')
    transport = HttpxProviderTransport(
        configured, secrets=PublicSyntheticCredential(), ledger=ledger,
    )
    model = ProviderModel(configured, counter=counted, transport=transport, timeout_seconds=3)
    prepared, sent, outcomes = [], [], []
    content_ready, release = Event(), Event()
    cancelled = Event()

    def sse(value):
        return b'data: ' + canonical(value).encode() + b'\n\n'

    @asynccontextmanager
    async def recorded_http(client, method, url, *, content, headers):
        assert method == 'POST' and url == configured.endpoint
        sent.append(dict(raw_request=content.decode(), body=json.loads(content)))

        class Response:
            status_code = 200
            headers = {'Content-Type': 'text/event-stream'}

            async def aiter_bytes(self):
                yield sse(dict(choices=[dict(index=0, delta=dict(content=canonical(opinion)),
                                             finish_reason=None)]))
                content_ready.set()
                while not release.is_set():
                    await asyncio.sleep(0.001)
                if terminal == 'eof':
                    return
                yield sse(dict(choices=[dict(index=0, delta={},
                    finish_reason='length' if terminal == 'length' else 'stop')]))
                if terminal != 'missing-usage':
                    yield sse(dict(choices=[], usage=dict(prompt_tokens=17, completion_tokens=5,
                                                         total_tokens=22)))
                yield b'data: [DONE]\n\n'

        yield Response()

    monkeypatch.setattr(httpx_transport.httpx.AsyncClient, 'stream', recorded_http)

    def invoker(stage, messages, schema, signal):
        request = model.prepare_stage_request(
            messages, stage=stage, schema=schema, output_tokens=1536,
        )
        request.check_context()
        prepared.append(deepcopy(request._body))
        return json.loads(''.join(request.stream(signal)))

    def assess():
        try:
            value = StructuredUnitAssessor(invoker, cancelled, audit_version=COMPACT)(packet)
            outcomes.append(dict(verdict=value))
        except (ProviderFailure, InterruptedError, ValueError) as error:
            outcomes.append(dict(error=type(error).__name__, message=str(error)))

    worker = Thread(target=assess)
    worker.start()
    try:
        assert content_ready.wait(1), 'Prepared request never reached the CPU HTTP seam'
        assert outcomes == [] and ledger.terminals == [], 'JSON content alone cannot approve'
        assert counted.bodies == prepared == [sent[0]['body']]
        assert digest(prepared[0]) == ledger.rows[0]['request_sha256']
        assert prepared[0]['max_tokens'] == 1536 and prepared[0]['temperature'] == 0
        assert prepared[0]['thinking'] == {'type': 'disabled'}
        assert json.loads(prepared[0]['messages'][-1]['content']) == semantic_review_view(
            packet['payload'],
        )
        if terminal == 'cancel':
            cancelled.set()
        else:
            release.set()
        worker.join(2)
        assert not worker.is_alive()
        assert len(ledger.rows) == len(ledger.terminals) == len(outcomes) == 1
        if terminal == 'valid':
            assert outcomes[0]['verdict']['model_opinion'] == opinion
            assert outcomes[0]['verdict']['request_sha256'] == packet['sha256']
            assert ledger.terminals[0]['status'] == 'completed'
            assert ledger.terminals[0]['usage'] == dict(prompt_tokens=17, completion_tokens=5,
                                                       total_tokens=22)
        else:
            assert 'verdict' not in outcomes[0]
            if terminal == 'missing-usage':
                # STOP/DONE are a completed transport, not a usable hosted count.
                assert ledger.terminals[0]['status'] == 'completed'
                assert 'hosted input count differs' in outcomes[0]['message']
                assert model.failed
            else:
                assert ledger.terminals[0]['status'] != 'completed'
            if terminal in {'eof', 'missing-usage', 'cancel'}:
                assert ledger.terminals[0]['usage'] is None
        record('prepared-' + terminal, dict(envelope=packet, prepared_body=prepared[0],
            raw_sent=sent[0], simulated_reservations=ledger.rows, terminals=ledger.terminals,
            outcomes=outcomes, real_api_requests=0, credential_body_reads=0,
            semantic_effect_proven=False))
    finally:
        release.set()
        cancelled.set()
        model.cancel()
        worker.join(2)



def test_v7_view_keeps_original_roles_modal_text_context_and_displayed_only_scope():
    from test_unit_scope_review_v4 import USE_AUDIT, sealed_v7, supplied_v7

    packet = sealed_v7("把唱片放回架子了。", "唱片好像也能把一段声音带回来。")
    payload = packet["payload"]
    context = {"history": [], "memories": [], "exchanges": [], "memory_status": "pending_source"}
    context["history"] = [{"id": "assistant-raw", "role": "assistant", "text": "旧助手推测。"}]
    payload["context"] = canonical(context)
    packet["sha256"] = digest(payload)
    view = semantic_review_view(payload)
    assert view["context"] == context and view["units"] == payload["review_segments"]
    assert view["sources"] == [
        {key: source[key] for key in ("id", "kind", "role", "text")}
        for source in payload["personal_sources"]
    ]
    current_source = next(
        source for source in view["sources"] if source["id"] == view["user_source_id"]
    )
    assert current_source["text"] == payload["user_text"]
    assert view["conversation_frame"]["candidate_speaker"] == "assistant"
    opinion = supplied_v7(payload)
    opinion["units"][0].update(
        meaning="助手以好像作一般比喻，未归成用户原话。", use="承接放回唱片的分享。"
    )
    bound = StructuredUnitAssessor(
        lambda *args: deepcopy(opinion), Event(), audit_version=USE_AUDIT
    )(packet)
    assert bound["model_opinion"] == opinion
    for field in ("meaning", "use", "response_fit"):
        assert bound["unit_scopes"][0][field] == opinion["units"][0][field]
    # The binder preserves a declared modal reading; it does not prove this reading.


def test_v8_same_context_task_operations_and_original_roles_survive_one_semantic_projection():
    from test_unit_scope_review_v4 import VERDICT_AUDIT, sealed_v8, supplied_v8

    packet = sealed_v8("把唱片放回架子了。", "唱片好像也能把一段声音带回来。")
    payload = packet["payload"]
    context = {"history": [{"id": "assistant-raw", "role": "assistant", "text": "助手旧推测。"}],
               "memories": [], "exchanges": [], "memory_status": "pending_source"}
    payload["context"] = canonical(context)
    payload["response_type"] = "ack"
    packet["sha256"] = digest(payload)
    view = semantic_review_view(payload)
    assert view["context"] == context and view["response_type"] == "ack"
    assert view["units"] == payload["review_segments"]
    assert view["conversation_frame"]["candidate_speaker"] == "assistant"
    assert all(row["role"] == "user" for row in view["sources"])
    current = next(row for row in view["sources"] if row["id"] == view["user_source_id"])
    assert current["text"] == payload["user_text"]
    opinion = supplied_v8(payload, meaning="助手用好像保留印象模态，未说成用户原话。")
    bound = StructuredUnitAssessor(lambda *args: deepcopy(opinion), Event(),
                                   audit_version=VERDICT_AUDIT)(packet)
    assert bound["model_opinion"] == opinion and bound["checks"] == opinion["checks"]
    assert bound["failure"] == opinion["failure"]
    for field in ("meaning", "use", "response_fit"):
        assert bound["unit_scopes"][0][field] == opinion["units"][0][field]
