"""Synthetic real-provider wire checks for the explicit v20 diagnostic candidate."""

import json
import threading
import time
from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from test_default_text_delivery import setup, wait_for
from test_model_provider_contract import event, route
from test_provider_loopback import Ledger
from test_text_provider_model import Counter
from unit_review_recording import bound_unit_packets as bound_unit_packets
from unit_review_recording import recorded_review_packet
from virtual_companion.cognition.short_dialogue import PROMPTS
from virtual_companion.cognition.stream_unit_review import semantic_review_view
from virtual_companion.providers.httpx_transport import HttpxProviderTransport
from virtual_companion.text.application import TextSession
from virtual_companion.text.default_runtime import (
    ACCESS_INITIAL,
    DEFAULT_PROMPT_REVISION,
    DIAGNOSTIC_PROMPT_REVISIONS,
    DefaultTextRuntime,
)
from virtual_companion.text.fake import synthetic_unit_opinion
from virtual_companion.text.provider_model import ConservativeReservation, ProviderModel

CANDIDATE = 'short-dialogue.v20'
# Captured through real Httpx on the unmodified pre-v20 runtime, not a serializer.
V12_GENERATION_SHA256 = 'a7c31f4a22fd2f5fd6f6f6df55657caf9c964f220dd516cc1a1cc66c32880a39'
V19_GENERATION_SHA256 = '90468fbfb26689449b3fbdad282c4be0501035cb2b939b5e441ff4eb67227240'
S05 = '我想去散步但在下雨，请给我一个简单安排。'
S06 = '我在想，为什么旧照片会让人同时开心又难过？'
S04 = '给我两条下雨天通勤的实用建议。'
PLAIN = '刚整理完一堆旧票根。'
TASKS = [
    (S05, dict(version='response-task-contract.v1', kind='activity_plan', source='user_text',
        span=S05, requires_complete_answer=True, activity='散步',
        waiting='condition_of_same_path', substitute_activities=False,
        numeric_claims='supported_or_explicit_plan_choice')),
    (S06, dict(version='response-task-contract.v1', kind='general_explanation', source='user_text',
        span=S06, requires_complete_answer=True, topic='旧照片', aspects=['开心', '难过'],
        attribution='general_not_user_personal')),
    (S04, dict(version='response-task-contract.v1', kind='exact_items', source='user_text',
        span=S04, requires_complete_answer=True, count=2,
        numeric_claims='supported_or_explicit_plan_choice')),
]
REPLIES = {
    S05: '雨小且无雷电、积水时，带伞沿平整路线散步再原路回来，否则等天气安全再出门。',
    S06: '旧照片能触发愉快联想，也能让人意识到当下与过去的距离，两种感受可能同时出现。',
    S04: '1. 出门前查看天气并选好路线。\n2. 步行时放慢脚步并避开积水。',
    PLAIN: '首句回应当前输入。后句补充具体信息。',
}


def generation_contracts(body):
    contracts = []
    for message in body['messages']:
        if message['role'] != 'system':
            continue
        try:
            data = json.loads(message['content'])
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and 'response_task_contract' in data:
            assert set(data) == {'response_task_contract'}
            contracts.append(data['response_task_contract'])
    return contracts


@pytest.fixture
def provider_host(tmp_path, record_property):
    @contextmanager
    def run(revision=CANDIDATE, *, ending='stop', hold_terminal_until_ack=False):
        wire, timeline, observations, errors, ack_states = [], [], [], [], []
        prepared_stages = []
        release, abort, prefix_received = (threading.Event() for _ in range(3))

        def mark(name, **fields):
            timeline.append(dict(event=name, monotonic_ns=time.monotonic_ns(), **fields))

        def content(value):
            return event({'choices': [{'index': 0, 'delta': {'content': value}}]})

        terminal = event({'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})
        terminal += event({'choices': [], 'usage': {'prompt_tokens': 17,
                           'completion_tokens': 5, 'total_tokens': 22}})
        terminal += b'data: [DONE]\n\n'

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):  # noqa: N802
                raw = self.rfile.read(int(self.headers['Content-Length']))
                body = json.loads(raw)
                review = '你只审查候选文字' in body['messages'][0]['content']
                kind = 'review' if review else 'generation'
                wire.append(dict(kind=kind, body=body, body_utf8=raw.decode('utf-8'),
                                 request_sha256=sha256(raw).hexdigest()))
                mark('server_request', kind=kind)
                if review:
                    # Recorded approval validates host wiring, never model semantics.
                    reply = json.dumps(synthetic_unit_opinion(
                        json.loads(body['messages'][-1]['content'])))
                    first, tail = content(reply), terminal
                elif ending != 'stop':
                    first = content('{"action":"answer","text":"合成先导首句。')
                    tail = content('尚未完成尾句。"}') + terminal
                else:
                    user_text = body['messages'][-1]['content']
                    first = content(json.dumps(dict(action='answer', text=REPLIES[user_text]),
                                               ensure_ascii=False))
                    tail = terminal
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Content-Length', str(len(first + tail)))
                self.end_headers()
                try:
                    self.wfile.write(first)
                    self.wfile.flush()
                    if not review and (ending != 'stop' or hold_terminal_until_ack):
                        while not release.wait(0.01):
                            if abort.is_set():
                                return
                        if ending == 'disconnect' or abort.is_set():
                            return  # Real incomplete HTTP response, never a fake client.
                    self.wfile.write(tail)
                    self.wfile.flush()
                    mark('server_response_finished', kind=kind)
                except OSError as error:
                    if not abort.is_set():
                        errors.append(f'{type(error).__name__}: {error}')

        service = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        service.daemon_threads = True
        worker = threading.Thread(target=lambda: service.serve_forever(poll_interval=0.01),
                                  name='synthetic-v20-provider-http', daemon=True)
        configured = route(route_id='synthetic-task-contract', explicit=True,
            model='synthetic-task-contract-model',
            endpoint=f'http://127.0.0.1:{service.server_port}/chat/completions',
            data_scope=('current', 'history', 'valid_memory', 'derived'))
        ledger = Ledger(limit=12)
        transport = HttpxProviderTransport(configured, secrets=None,
                                           ledger=ConservativeReservation(ledger))
        model = ProviderModel(configured, counter=Counter(), transport=transport,
                              output_tokens=192, timeout_seconds=4)
        runtime = session = memory = mirror = None
        worker.start()
        try:
            runtime, session, memory, mirror, _ = setup(tmp_path, model)
            if revision is not None:
                runtime = DefaultTextRuntime(memory, mirror, model, binding={'test': 'synthetic'},
                    delivery_timeout=0.8, prompt_revision=revision)
                session = TextSession(model, mirror, {}, prepare_turn=runtime.prepare_turn,
                                      before_start=runtime.before_start)
            prepare = runtime._prepare_turn_stage

            def record_prepare(*args, **kwargs):
                stage = prepare(*args, **kwargs)
                prepared_stages.append(stage)
                return stage

            runtime._prepare_turn_stage = record_prepare
            observe = runtime.observe_stream

            def observe_stream(item):
                observations.append(deepcopy(item))
                mark('host_' + item['event'], sequence=item.get('sequence'))
                observe(item)
                if item['event'] == 'delta':
                    prefix_received.set()

            runtime.observe_stream = observe_stream
            yield SimpleNamespace(runtime=runtime, session=session, memory=memory, mirror=mirror,
                wire=wire, ledger=ledger, timeline=timeline, ack_states=ack_states, mark=mark,
                prepared_stages=prepared_stages,
                release=release, prefix_received=prefix_received,
                hold_terminal_until_ack=hold_terminal_until_ack)
        finally:
            abort.set()
            if session is not None and session.snapshot()['busy']:
                runtime.stop()
                session.stop()
                wait_for(lambda: not session.snapshot()['busy'])
            model.close()
            service.shutdown()
            service.server_close()
            worker.join(1)
            replay = dict(revision=revision, ending=ending,
                hold_terminal_until_ack=hold_terminal_until_ack, wire=wire, timeline=timeline,
                observations=observations, ack_states=ack_states,
                prepared_stages=[dict(messages=stage.messages,
                    context=json.loads(stage.request.context.payload),
                    body_sha256=stage.wire._fingerprint) for stage in prepared_stages],
                ledger_calls=ledger.calls, ledger_finished=ledger.finished, errors=errors,
                loopback_only=True, real_model_calls=0, paid_api_calls=0, real_secret_reads=0)
            record_property('task_contract_provider_wire', json.dumps(replay, ensure_ascii=False))
            (tmp_path / 'task-contract-provider-wire.json').write_text(
                json.dumps(replay, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            if memory is not None:
                memory.close()
                mirror.close()
            assert not worker.is_alive() and not errors

    return run


def complete(state, user_text):
    state.session.send(user_text)
    until = time.monotonic() + 4
    events, seen = [], set()
    while state.session.snapshot()['busy'] and time.monotonic() < until:
        item = state.runtime.poll()['event']
        if item is not None and item['sequence'] not in seen:
            state.mark('host_poll', sequence=item['sequence'])
            seen.add(item['sequence'])
            events.append(deepcopy(item))
            assert state.runtime.ack(dict(sequence=item['sequence'],
                fingerprint=item['fingerprint'], host_receipt='synthetic-v20-visible-'
                + item['basis']['input_id'] + '-' + str(item['sequence']))) \
                == {'status': 'confirmed'}
            generation_terminal_us_at_ack = state.runtime.generation_terminal_us
            state.mark('host_ack_returned', sequence=item['sequence'],
                       generation_terminal_us_at_ack=generation_terminal_us_at_ack)
            durable = state.memory.store.stream_state(state.memory.companion,
                                                       state.runtime.claim.input_id)
            state.ack_states.append(dict(sequence=item['sequence'], durable=durable,
                generation_terminal_us_at_ack=generation_terminal_us_at_ack))
            if state.hold_terminal_until_ack and item['sequence'] == 1:
                state.mark('host_release_generation_terminal', sequence=1)
                state.release.set()
        time.sleep(0.005)
    assert not state.session.snapshot()['busy'], 'Real host delivery did not settle'
    assert state.session.snapshot()['attempt']['status'] == 'completed'
    assert events and state.runtime.poll()['event'] is None
    assert events[-1]['display_text'] == REPLIES[user_text]
    assert state.mirror.history()['messages'][-1]['content'] == REPLIES[user_text]
    return events


def test_v20_is_diagnostic_and_existing_prompt_bytes_stay_selectable():
    assert CANDIDATE in DIAGNOSTIC_PROMPT_REVISIONS
    assert DEFAULT_PROMPT_REVISION == 'short-dialogue.v12'
    assert PROMPTS['short-dialogue.v19'] == PROMPTS['short-dialogue.v18'].replace(
        'short-dialogue.v18', 'short-dialogue.v19')
    assert sha256(PROMPTS['short-dialogue.v12'].encode()).hexdigest() \
        == '485c8a2e3cddfa55d88bd9e107b6f19be9d1b39ba9a5ae4589d9502d797bd874'


@pytest.mark.parametrize('user_text,expected', TASKS, ids=['s05', 's06', 'exact-two'])
def test_v20_typed_task_reaches_real_generation_and_existing_review_wire(
    provider_host, user_text, expected,
):
    with provider_host() as state:
        events = complete(state, user_text)
        assert len(state.wire) == 2
        generation, review = state.wire
        semantic = json.loads(review['body']['messages'][-1]['content'])
        payload = recorded_review_packet(review['body']['messages'])
        assert semantic_review_view(payload) == semantic
        assert semantic['response_task_contract'] == payload['response_task_contract'] == expected
        assert generation_contracts(generation['body']) == [expected], \
            'The typed review task must reach the actual provider generation request'
        guidance = generation['body']['messages'][0]['content']
        for condition in ('substitute_activities=false', 'waiting=condition_of_same_path',
                          'attribution=general_not_user_personal'):
            assert condition in guidance
        assert generation['body']['messages'][-1] == {'role': 'user', 'content': user_text}
        assert payload['user_text'] == user_text
        assert set(payload['basis']) == {'companion_id', 'session_id', 'input_id',
                                         'attempt_id', 'state_version'}
        assert payload['basis'] == {key: events[0]['basis'][key] for key in payload['basis']}
        assert [row['stage'] for row in state.ledger.calls] \
            == ['demand_control', 'stream_unit_review']
        assert [row['request_sha256'] for row in state.wire] \
            == [row['request_sha256'] for row in state.ledger.calls]
        assert all(row['status'] == 'completed' for row in state.ledger.finished)


def test_v20_recomputes_contract_for_each_actual_input_and_omits_it_for_plain_share(provider_host):
    with provider_host() as state:
        inputs = []
        for user_text, expected in [TASKS[0], TASKS[1], (PLAIN, None)]:
            start = len(state.wire)
            events = complete(state, user_text)
            inputs.append(events[0]['basis']['input_id'])
            generation, *reviews = state.wire[start:]
            assert generation_contracts(generation['body']) == ([] if expected is None
                                                               else [expected])
            assert generation['body']['messages'][-1]['content'] == user_text
            for review in reviews:
                semantic = json.loads(review['body']['messages'][-1]['content'])
                payload = recorded_review_packet(review['body']['messages'])
                assert semantic_review_view(payload) == semantic
                assert semantic.get('response_task_contract') == expected
                assert payload['user_text'] == user_text
                assert payload['basis']['input_id'] == inputs[-1]
                assert payload.get('response_task_contract') == expected
        assert len(set(inputs)) == 3


def test_v20_plain_stream_keeps_review_poll_ack_and_confirmed_mirror_order(provider_host):
    # Hold the real HTTP STOP until the first actual durable ACK; immediate
    # provider completion cannot promise that ACK wins a thread scheduling race.
    with provider_host(hold_terminal_until_ack=True) as state:
        events = complete(state, PLAIN)
        assert [item['sequence'] for item in events] == [1, 2]
        assert len(state.wire) == 3  # one generation, existing early/final opinions
        assert generation_contracts(state.wire[0]['body']) == []
        assert state.ack_states[0]['durable']['confirmed_units'] == 1
        first_ack_ns = next(row['monotonic_ns'] for row in state.timeline
                            if row['event'] == 'host_ack_returned' and row['sequence'] == 1)
        terminal_ns = next(row['monotonic_ns'] for row in state.timeline
                           if row['event'] == 'host_model_terminal')
        assert first_ack_ns < terminal_ns
        assert state.ack_states[-1]['durable']['confirmed_units'] == 2
        assert state.ack_states[-1]['durable']['unknown_units'] == 0
        for item in events:
            moments = {row['event']: row['monotonic_ns'] for row in state.timeline
                       if row.get('sequence') == item['sequence']}
            assert moments['host_approved'] < moments['host_poll'] < moments['host_ack_returned']
        assistants = [row for row in state.mirror.history()['messages']
                      if row['role'] == 'assistant']
        assert len(assistants) == 1 and assistants[0]['content'] == REPLIES[PLAIN]


def assert_admission_generation_wire(state, historical_sha256):
    # The old full-body captures remain above. Admission metadata/access prose
    # intentionally changes the body, while generation policy/control stay fixed.
    from virtual_companion.boundary.contracts import canonical
    from virtual_companion.boundary.history_readset import history_readset

    stage, = state.prepared_stages
    generation, review = state.wire
    context = stage.request.context
    data = json.loads(context.payload)
    historical_prompts = {revision: prompt for revision, prompt in PROMPTS.items()
                          if revision not in {
                              'short-dialogue.v28', 'short-dialogue.v29', 'short-dialogue.v30',
                              'short-dialogue.v31', 'short-dialogue.v32', 'short-dialogue.v33',
                              'short-dialogue.v34', 'short-dialogue.v35', 'short-dialogue.v36'}}
    assert len(historical_prompts) == 23
    assert sha256(canonical(historical_prompts).encode()).hexdigest() == (
        '29e6e4de5a03ef467f71f21105fdca5f741e2d5e373b778e99bc70722bb4eddc')
    control_sha256 = {
        'short-dialogue.v12': '91819f92b04d66fc002eb703a9a7cffff282e6949164625e23ccf123152290a5',
        'short-dialogue.v19': '6eb2cc9f5d38393e09f8ad762a155af02b6299d484d6cc65478b0bcad95d117a',
    }
    assert sha256(state.runtime.control.encode()).hexdigest() == (
        control_sha256[state.runtime.prompt_revision])
    assert stage.access_status == ACCESS_INITIAL
    assert stage.messages == stage.request.model_request(
        192, state.runtime.control + ACCESS_INITIAL)['messages']
    assert generation['body']['messages'][0]['content'].startswith(state.runtime.control)
    assert generation['body']['messages'][-1] == {'role': 'user', 'content': S05}
    assert generation['body'] == stage.wire._body
    assert stage.wire.checked and stage.wire.used
    assert generation['request_sha256'] == stage.wire._fingerprint \
        == sha256(canonical(generation['body']).encode()).hexdigest()
    assert generation['request_sha256'] == state.ledger.calls[0]['request_sha256']
    assert generation['request_sha256'] != historical_sha256
    assert data['source_processing'] == {
        'protocol': 'current-source-processing.v1', 'source_id': context.input_id,
        'source_sha256': sha256(S05.encode()).hexdigest(), 'status': 'queued',
        'error': None, 'receipt_sha256': None,
    }
    assert data['memory_access'] == 'not_accessed' and data['memory_status'] == 'pending_source'
    assert data['history'] == data['memories'] == data['exchanges'] == []
    packet = recorded_review_packet(review['body']['messages'])
    assert packet['context'] == context.payload
    assert packet['basis']['input_id'] == context.input_id
    assert packet['basis']['state_version'] == context.state_version
    readset = history_readset(context.companion, context.input_id, context.state_version,
                              context.payload)
    assert readset['context_sha256'] == sha256(canonical(data).encode()).hexdigest()
    assert state.ack_states[-1]['durable']['confirmed_units'] == 1


def test_v12_default_real_generation_wire_omits_candidate_task_data(provider_host):
    with provider_host(None) as state:
        complete(state, S05)
        assert state.runtime.prompt_revision == DEFAULT_PROMPT_REVISION == 'short-dialogue.v12'
        assert generation_contracts(state.wire[0]['body']) == []
        assert 'response_task_contract' not in state.wire[0]['body']['messages'][0]['content']
        assert state.wire[0]['body']['messages'][-1]['content'] == S05
        assert len(state.wire) == 2
        assert_admission_generation_wire(state, V12_GENERATION_SHA256)


def test_v19_generation_stays_without_task_data_and_review_keeps_existing_contract(provider_host):
    with provider_host('short-dialogue.v19') as state:
        complete(state, S05)
        assert generation_contracts(state.wire[0]['body']) == []
        payload = json.loads(state.wire[1]['body']['messages'][-1]['content'])
        assert payload['response_task_contract'] == TASKS[0][1]
        assert len(state.wire) == 2
        assert_admission_generation_wire(state, V19_GENERATION_SHA256)


@pytest.mark.parametrize('ending', ['cancel', 'disconnect'])
def test_v20_cancel_or_real_http_failure_keeps_task_prefix_unapproved(provider_host, ending):
    with provider_host(ending=ending) as state:
        state.session.send(S05)
        assert state.prefix_received.wait(2), 'Real HTTP content never reached StreamDialogue'
        assert state.runtime.poll()['event'] is None
        assert state.runtime.generation_terminal_us is None
        if ending == 'cancel':
            state.runtime.stop()
            state.session.stop()
        else:
            state.release.set()
        wait_for(lambda: not state.session.snapshot()['busy'])
        expected = 'cancelled' if ending == 'cancel' else 'failed'
        durable = state.memory.store.stream_state(state.memory.companion,
                                                   state.runtime.claim.input_id)
        assert durable['outcome'] == state.session.snapshot()['attempt']['status'] == expected
        assert durable['approved_units'] == durable['confirmed_units'] == 0
        assert durable['confirmed_text'] == '' and state.runtime.poll()['event'] is None
        assert all(row['role'] == 'user' for row in state.mirror.history()['messages'])
        assert len(state.wire) == 1 and state.wire[0]['kind'] == 'generation'
