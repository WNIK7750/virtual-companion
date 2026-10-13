import asyncio
import json
import threading
import time
from copy import deepcopy
from hashlib import sha256

import httpx
import pytest
from unit_review_recording import bound_unit_packets as bound_unit_packets
from unit_review_recording import recorded_review_packet
from virtual_companion.text.application import TextSession
from virtual_companion.text.default_runtime import DefaultTextRuntime
from virtual_companion.text.fake import synthetic_unit_opinion
from virtual_companion.text.memory_runtime import DEFAULT_COMPANION_ID, TextMemoryRuntime
from virtual_companion.text.sqlite_store import SQLiteConversationStore


class Model:
    def __init__(self):
        self.calls = []

    def check_context(self, messages):
        pass

    def stream(self, messages, cancelled):
        self.calls.append(messages)
        if '你只审查候选文字' in messages[0]['content']:
            yield json.dumps(synthetic_unit_opinion(json.loads(messages[-1]['content'])))
        else:
            yield json.dumps({'action': 'answer', 'text': '这是合成答案。'}, ensure_ascii=False)

    def cancel(self):
        pass


def setup(tmp_path, model=None):
    model = model or Model()
    memory = TextMemoryRuntime(tmp_path / 'memory.db', DEFAULT_COMPANION_ID,
                               model=model, fake=True)
    mirror = SQLiteConversationStore(tmp_path / 'chat', companion_id=DEFAULT_COMPANION_ID)
    runtime = DefaultTextRuntime(memory, mirror, model, binding={'test': 'synthetic'},
                                 delivery_timeout=0.8)
    session = TextSession(model, mirror, {}, prepare_turn=runtime.prepare_turn,
                          before_start=runtime.before_start)
    return runtime, session, memory, mirror, model


def wait_for(predicate):
    until = time.monotonic() + 4
    while time.monotonic() < until:
        value = predicate()
        if value:
            return value
        time.sleep(0.01)
    raise AssertionError('Timed out')


def test_default_intake_precedes_model_and_no_raw_output_until_ack(tmp_path):
    runtime, text, memory, mirror, model = setup(tmp_path)
    attempt = text.send('合成输入')
    event = wait_for(lambda: runtime.poll()['event'])
    assert text.snapshot()['busy']
    assert text.snapshot()['partial'] == ''
    assert [m['role'] for m in mirror.history()['messages']] == ['user']
    assert runtime.snapshot()['delivery_scope']['turn_id'] == attempt['turn_id']
    assert len(model.calls) == 2  # one answer and one existing policy opinion
    assert memory.store.inspect(
        DEFAULT_COMPANION_ID, event['basis']['input_id'])['text'] == '合成输入'
    runtime.ack(dict(sequence=event['sequence'], fingerprint=event['fingerprint'],
                     host_receipt='synthetic-dom-receipt'))
    wait_for(lambda: not text.snapshot()['busy'])
    assert text.snapshot()['attempt']['status'] == 'completed'
    assert mirror.history()['messages'][-1]['content'] == '这是合成答案。'


@pytest.mark.parametrize('action_first', [True, False])
def test_default_answer_consumption_does_not_rebuild_or_invoke_short_model(
    tmp_path, monkeypatch, action_first,
):
    from virtual_companion.cognition.short_dialogue import ShortDialogue, ShortRequest

    class OrderedModel(Model):
        def stream(self, messages, cancelled):
            if '你只审查候选文字' in messages[0]['content']:
                yield from super().stream(messages, cancelled)
            else:
                self.calls.append(messages)
                fields = [('action', 'answer'), ('text', '这是合成答案。')]
                yield json.dumps(dict(fields if action_first else reversed(fields)),
                                 ensure_ascii=False)

    bodies, short_calls = [], []
    build = ShortRequest.model_request
    generate = ShortDialogue.generate

    def model_request(request, *args, **kwargs):
        bodies.append(build(request, *args, **kwargs))
        return bodies[-1]

    def short_generate(dialogue, request):
        short_calls.append(request)
        return generate(dialogue, request)

    monkeypatch.setattr(ShortRequest, 'model_request', model_request)
    monkeypatch.setattr(ShortDialogue, 'generate', short_generate)
    runtime, text, memory, mirror, model = setup(tmp_path, OrderedModel())
    try:
        text.send('合成输入')
        event = wait_for(lambda: runtime.poll()['event'])
        assert len(bodies) == 1
        assert not short_calls
        assert len(model.calls) == 2
        assert event['text'] == '这是合成答案。'
        assert text.snapshot()['partial'] == ''
        runtime.ack(dict(sequence=event['sequence'], fingerprint=event['fingerprint'],
                         host_receipt='synthetic-direct-answer-receipt'))
        wait_for(lambda: not text.snapshot()['busy'])
        assert text.snapshot()['attempt']['status'] == 'completed'
        assert mirror.history()['messages'][-1]['content'] == event['text']
    finally:
        runtime.stop()
        text.stop()
        memory.close()
        mirror.close()


def test_intake_failure_prevents_count_and_generation(tmp_path, monkeypatch):
    runtime, text, memory, mirror, model = setup(tmp_path)
    def fail(*args, **kwargs):
        raise RuntimeError('synthetic intake failure')
    monkeypatch.setattr(memory.store, 'accept', fail)
    text.send('合成输入')
    wait_for(lambda: not text.snapshot()['busy'])
    assert not model.calls
    assert text.snapshot()['attempt']['status'] == 'failed'


def test_unknown_delivery_not_replayed_on_retry_and_late_ack_recovers(tmp_path):
    runtime, text, memory, mirror, model = setup(tmp_path)
    text.send('合成输入')
    event = wait_for(lambda: runtime.poll()['event'])
    wait_for(lambda: not text.snapshot()['busy'])
    calls = len(model.calls)
    with pytest.raises(ValueError, match='呈现收据'):
        text.retry()
    assert len(model.calls) == calls
    assert runtime.poll()['event'] is None
    assert runtime.state()['reconcile'][0]['fingerprint'] == event['fingerprint']
    runtime.ack(dict(sequence=event['sequence'], fingerprint=event['fingerprint'],
                     host_receipt='synthetic-late-receipt'))
    assert mirror.history()['messages'][-1]['content'] == '这是合成答案。'
    assert text.snapshot()['attempt']['status'] != 'completed'
    with pytest.raises(ValueError):
        runtime.ack(dict(sequence=1, fingerprint='f' * 64, host_receipt='wrong'))


def test_cancel_before_generation_never_admits_source(tmp_path):
    runtime, text, memory, mirror, model = setup(tmp_path)
    attempt = mirror.begin('合成输入', {})
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(InterruptedError):
        runtime.prepare_turn(attempt, mirror.session_id, mirror.history()['messages'], cancelled)
    assert memory.store.background_sources(DEFAULT_COMPANION_ID) == []


def test_control_answer_cannot_leak_reasoning_tags_to_display(tmp_path):
    class TaggedModel(Model):
        def stream(self, messages, cancelled):
            self.calls.append(messages)
            yield json.dumps({'action': 'answer', 'text': '<think>内部推理</think>正文'})
    runtime, text, memory, mirror, model = setup(tmp_path, TaggedModel())
    text.send('合成输入')
    wait_for(lambda: not text.snapshot()['busy'])
    assert text.snapshot()['attempt']['status'] == 'failed'
    assert len(model.calls) == 1
    assert runtime.poll()['event'] is None
    assert all(row['role'] == 'user' for row in mirror.history()['messages'])


def test_turn_clock_cannot_rewind_between_stages(tmp_path):
    runtime, text, memory, mirror, model = setup(tmp_path)
    moment = [1_000_000]
    runtime.clock = lambda: moment[0]
    attempt = mirror.begin('合成输入', {})
    request = runtime.prepare_turn(attempt, mirror.session_id, mirror.history()['messages'],
                                   threading.Event())
    request.check_context()
    moment[0] -= 1
    with pytest.raises(Exception, match='clock moved backwards'):
        request.check_context()
    assert memory.store.stream_state(DEFAULT_COMPANION_ID, runtime.claim.input_id)['outcome'] \
        == 'failed'


class StreamingModel(Model):
    """A provider stays in generation until the actual host ACK is observed."""

    def __init__(self, *, ending='stop', early='approve'):
        super().__init__()
        self.ending, self.early = ending, early
        self.release, self.terminal = threading.Event(), threading.Event()
        self.reviewed = threading.Event()
        self.terminal_at = None

    def review_peer(self):
        owner = self

        class Reviewer(Model):
            def stream(self, messages, cancelled):
                owner.calls.append(messages)
                payload = json.loads(messages[-1]['content'])
                final = payload['final']
                owner.reviewed.set()
                if (final and owner.ending == 'review_error') or owner.early == 'transport':
                    raise OSError('synthetic review transport failure')
                if not final and owner.early == 'malformed':
                    yield '{"decision":"approve"}'
                    return
                decision = ('reject' if final and owner.ending == 'reject'
                            else 'approve' if final else owner.early)
                yield json.dumps(synthetic_unit_opinion(payload, decision=decision,
                    self_contained=decision == 'approve', no_open_dependency=decision == 'approve',
                    consistent_with_displayed=True, constraints_satisfied=decision == 'approve'))

        return Reviewer()

    def stream(self, messages, cancelled):
        self.calls.append(messages)
        try:
            yield '{"action":"answer","text":"首句回应当前输入。'
            while not self.release.wait(0.01):
                if cancelled.is_set():
                    raise InterruptedError('synthetic provider cancelled')
            if cancelled.is_set():
                raise InterruptedError('synthetic provider cancelled')
            if self.ending == 'error':
                raise OSError('synthetic provider terminal failure')
            yield '后句补充具体信息。"' + ('' if self.ending == 'json' else '}')
        finally:
            self.terminal_at = time.time_ns() // 1000
            self.terminal.set()

    def cancel(self):
        self.release.set()


def test_default_approved_unit_and_ack_precede_provider_terminal(tmp_path):
    runtime, text, memory, mirror, model = setup(tmp_path, StreamingModel())
    try:
        text.send('合成输入')
        first = wait_for(lambda: runtime.poll()['event'])
        assert first['text'] == '首句回应当前输入。'
        assert not model.terminal.is_set()
        assert runtime.snapshot()['generation_terminal_us'] is None
        assert text.snapshot()['partial'] == ''
        ack_at = time.time_ns() // 1000
        runtime.ack(dict(sequence=first['sequence'], fingerprint=first['fingerprint'],
                         host_receipt='synthetic-visible-first'))
        state = memory.store.stream_state(memory.companion, runtime.claim.input_id)
        assert state['confirmed_units'] == 1 and state['outcome'] == 'unsettled'
        first_message = mirror.history()['messages'][-1]
        assert first_message['content'] == first['text']
        assert not model.terminal.is_set()
        model.release.set()
        last = wait_for(lambda: (event if (event := runtime.poll()['event'])
                                and event['sequence'] == 2 else None))
        assert last['text'] == '后句补充具体信息。'
        assert last['display_text'] == first['text'] + last['text']
        runtime.ack(dict(sequence=last['sequence'], fingerprint=last['fingerprint'],
                         host_receipt='synthetic-visible-final'))
        wait_for(lambda: not text.snapshot()['busy'])
        state = memory.store.stream_state(memory.companion, runtime.claim.input_id)
        assert state['outcome'] == 'completed' and state['confirmed_units'] == 2
        assert ack_at < model.terminal_at <= runtime.snapshot()['generation_terminal_us']
        assistants = [message for message in mirror.history()['messages']
                      if message['role'] == 'assistant']
        assert len(assistants) == 1 and assistants[0]['id'] == first_message['id']
        assert assistants[0]['content'] == last['display_text']
    finally:
        model.release.set()
        memory.close()
        mirror.close()


def test_unknown_early_prefix_stops_generation_and_late_ack_never_replays(tmp_path):
    runtime, text, memory, mirror, model = setup(tmp_path, StreamingModel())
    try:
        text.send('合成输入')
        event = wait_for(lambda: runtime.poll()['event'])
        assert not model.terminal.is_set()
        wait_for(lambda: not text.snapshot()['busy'])
        state = memory.store.stream_state(memory.companion, runtime.claim.input_id)
        assert state['unknown_units'] == state['approved_units'] == 1
        assert state['outcome'] == text.snapshot()['attempt']['status'] == 'failed'
        assert state['confirmed_text'] == ''
        assert runtime.poll()['event'] is None
        runtime.ack(dict(sequence=event['sequence'], fingerprint=event['fingerprint'],
                         host_receipt='synthetic-recovered-early'))
        recovered = memory.store.stream_state(memory.companion, runtime.claim.input_id)
        assert recovered['outcome'] != 'completed'
        assert recovered['confirmed_text'] == event['text']
        assert mirror.history()['messages'][-1]['content'] == event['text']
        with pytest.raises(ValueError, match='呈现收据'):
            text.retry()
        assert sum('desktop-demand.v4' in call[0]['content']
                   and '你只审查候选文字' not in call[0]['content'] for call in model.calls) == 1
    finally:
        model.release.set()
        memory.close()
        mirror.close()


def test_confirmed_mirror_extends_same_identity_but_never_rewrites(tmp_path):
    runtime, text, memory, mirror, model = setup(tmp_path)
    try:
        attempt = mirror.begin('合成输入', {})
        first = mirror.reconcile_confirmed(attempt['turn_id'], '已确认首句。', completed=False)
        second = mirror.reconcile_confirmed(attempt['turn_id'], '已确认首句。后句。',
                                            completed=False)
        assert first['message_id'] == second['message_id']
        for changed_text in ['替换。', '已确认首句。']:
            with pytest.raises(ValueError, match='Conflicting'):
                mirror.reconcile_confirmed(attempt['turn_id'], changed_text, completed=False)
        with pytest.raises(ValueError, match='Conflicting'):
            mirror.finish(attempt['attempt_id'], 'completed', '已确认首句。后句。未经确认追加。')
        mirror.finish(attempt['attempt_id'], 'completed', '已确认首句。后句。')
    finally:
        memory.close()
        mirror.close()


@pytest.mark.parametrize('ending', ['error', 'json', 'reject', 'review_error', 'cancel'])
def test_stream_failure_or_cancel_keeps_only_confirmed_prefix(tmp_path, ending):
    runtime, text, memory, mirror, model = setup(tmp_path, StreamingModel(ending=ending))
    try:
        text.send('合成输入')
        first = wait_for(lambda: runtime.poll()['event'])
        runtime.ack(dict(sequence=first['sequence'], fingerprint=first['fingerprint'],
                         host_receipt='synthetic-known-prefix'))
        if ending == 'cancel':
            runtime.stop()
            text.stop()
        else:
            model.release.set()
        wait_for(lambda: not text.snapshot()['busy'])
        state = memory.store.stream_state(memory.companion, runtime.claim.input_id)
        assert state['outcome'] == ('cancelled' if ending == 'cancel' else 'failed')
        assert state['confirmed_units'] == state['approved_units'] == 1
        assert state['confirmed_text'] == first['text']
        assert runtime.cancelled.is_set() == (ending == 'cancel')
        assert mirror.history()['messages'][-1]['content'] == first['text']
        assert runtime.poll()['event'] is None
        assert sum('desktop-demand.v4' in call[0]['content']
                   and '你只审查候选文字' not in call[0]['content'] for call in model.calls) == (
                       2 if ending == 'json' else 1)
        assert sum('{"carrier_recovery":' in message['content']
                   for call in model.calls for message in call) == int(ending == 'json')
        assert runtime.snapshot()['retry_allowed'] is False
        assert '重试' not in text.snapshot()['error']
        with pytest.raises(ValueError, match='呈现收据'):
            text.retry()
    finally:
        model.release.set()
        memory.close()
        mirror.close()


def test_early_review_failure_aborts_provider_without_user_cancellation(tmp_path):
    runtime, text, memory, mirror, model = setup(tmp_path, StreamingModel(early='transport'))
    try:
        text.send('合成输入')
        wait_for(lambda: not text.snapshot()['busy'])
        state = memory.store.stream_state(memory.companion, runtime.claim.input_id)
        assert state['outcome'] == text.snapshot()['attempt']['status'] == 'failed'
        assert not runtime.cancelled.is_set()
        assert state['approved_units'] == 0 and model.terminal.is_set()
        assert runtime.poll()['event'] is None
    finally:
        model.release.set()
        memory.close()
        mirror.close()


def test_delivery_failure_does_not_turn_its_live_guard_into_user_cancellation(
    tmp_path, monkeypatch,
):
    """A scheduled observer can run immediately after the owner writes failure."""
    from virtual_companion.cognition.turn_guardian import TurnGuardian

    runtime, text, memory, mirror, model = setup(tmp_path, StreamingModel())
    active, guard_lock = [], threading.RLock()
    enter, exit_guard = TurnGuardian.__enter__, TurnGuardian.__exit__
    finish = runtime.store.finish_stream

    def enter_guard(guardian):
        result = enter(guardian)
        with guard_lock:
            active.append(guardian)
        return result

    def leave_guard(guardian, *error):
        try:
            return exit_guard(guardian, *error)
        finally:
            with guard_lock:
                active.remove(guardian)

    def finish_stream(*args, **kwargs):
        result = finish(*args, **kwargs)
        if kwargs['outcome'] == 'failed':
            with guard_lock:
                guardians = list(active)
            for guardian in guardians:
                try:
                    guardian.checkpoint()
                except Exception as error:
                    assert guardian.failure is error
        return result

    monkeypatch.setattr(TurnGuardian, '__enter__', enter_guard)
    monkeypatch.setattr(TurnGuardian, '__exit__', leave_guard)
    monkeypatch.setattr(runtime.store, 'finish_stream', finish_stream)
    try:
        text.send('合成输入')
        event = wait_for(lambda: runtime.poll()['event'])
        assert not model.terminal.is_set()
        wait_for(lambda: not text.snapshot()['busy'])
        state = memory.store.stream_state(memory.companion, runtime.claim.input_id)
        assert state['outcome'] == 'failed' and state['unknown_units'] == 1
        assert not runtime.cancelled.is_set()
        assert text.snapshot()['attempt']['status'] == 'failed'
        assert runtime.poll()['event'] is None
        assert all(message['role'] == 'user' for message in mirror.history()['messages'])
        assert event['sequence'] == 1
    finally:
        model.release.set()
        memory.close()
        mirror.close()


@pytest.mark.parametrize('early', ['hold', 'malformed'])
def test_valid_hold_continues_but_malformed_early_opinion_fails_closed(tmp_path, early):
    runtime, text, memory, mirror, model = setup(tmp_path, StreamingModel(early=early))
    try:
        text.send('合成输入')
        assert model.reviewed.wait(2)
        assert runtime.poll()['event'] is None
        assert not memory.store.segments(memory.companion, runtime.claim.input_id)
        assert all(message['role'] == 'user' for message in mirror.history()['messages'])
        model.release.set()
        if early == 'malformed':
            wait_for(lambda: not text.snapshot()['busy'])
            assert text.snapshot()['attempt']['status'] == 'failed'
            assert not runtime.cancelled.is_set()
            assert model.terminal.is_set() and len(model.calls) == 2
            assert runtime.poll()['event'] is None
            assert not memory.store.segments(memory.companion, runtime.claim.input_id)
            return
        event = wait_for(lambda: runtime.poll()['event'])
        assert event['sequence'] == 1
        assert event['text'] == '首句回应当前输入。后句补充具体信息。'
        runtime.ack(dict(sequence=1, fingerprint=event['fingerprint'],
                         host_receipt='synthetic-complete-only'))
        wait_for(lambda: not text.snapshot()['busy'])
        assert text.snapshot()['attempt']['status'] == 'completed'
    finally:
        model.release.set()
        memory.close()
        mirror.close()


@pytest.mark.parametrize('ending', ['read_error', 'cancel'])
def test_actual_provider_transport_failure_or_cancel_keeps_only_host_ack_prefix(
    tmp_path, monkeypatch, record_property, ending,
):
    """Real SSE/parser/provider/host path, fake AsyncClient; no socket or credential I/O."""
    from test_model_provider_contract import event, route
    from test_provider_loopback import Ledger
    from test_text_provider_model import Counter
    from virtual_companion.providers.httpx_transport import HttpxProviderTransport
    from virtual_companion.text.provider_model import ConservativeReservation, ProviderModel

    release, finish, tail_observed, transport_closed = (threading.Event() for _ in range(4))
    wire, emitted, observations, injected = [], [], [], []
    prefix, tail = '首句回应当前输入。', '这段后续尚未确认'
    ledger = Ledger(limit=4)
    configured = route(route_id='synthetic-deepseek', kind='api',
        endpoint='https://api.deepseek.com/chat/completions', model='deepseek-flash',
        auth='bearer', secret_ref='dpapi:' + 'a' * 32, explicit=True,
        data_scope=('current', 'history', 'valid_memory', 'derived'),
        capabilities=type(route().capabilities)(True, False, True, 1048576, 'max_tokens'),
        wire_profile='deepseek-flash.v1')

    class SyntheticSecrets:
        def get(self, reference, endpoint):
            assert reference == configured.secret_ref and endpoint == configured.endpoint
            return 'synthetic-in-memory-port-value'

    class Response:
        status_code = 200
        headers = {'Content-Type': 'text/event-stream'}

        def __init__(self, body):
            self.review = any('你只审查候选文字' in m['content'] for m in body['messages'])
            self.body = body

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            if not self.review:
                transport_closed.set()

        async def aiter_bytes(self):
            if self.review:
                opinion = json.dumps(synthetic_unit_opinion(
                    json.loads(self.body['messages'][-1]['content'])))
                yield event({'choices': [{'index': 0, 'delta': {'content': opinion}}]})
                yield event({'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})
                yield event({'choices': [], 'usage': {'prompt_tokens': 17,
                    'completion_tokens': 10, 'total_tokens': 27}})
                yield b'data: [DONE]\n\n'
                return
            first = '{"action":"answer","text":"' + prefix
            emitted.append(first)
            yield event({'choices': [{'index': 0, 'delta': {'content': first}}]})
            while not release.is_set():
                await asyncio.sleep(0.005)
            emitted.append(tail)
            yield event({'choices': [{'index': 0, 'delta': {'content': tail}}]})
            while not finish.is_set():
                await asyncio.sleep(0.005)
            injected.append('httpx.ReadError')
            raise httpx.ReadError('synthetic stream disconnected after confirmed prefix')

    class Client:
        def __init__(self, **kwargs):
            assert kwargs['trust_env'] is False and kwargs['follow_redirects'] is False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        def stream(self, method, endpoint, *, content, headers):
            assert method == 'POST' and endpoint == configured.endpoint
            # Headers are deliberately never stored or inspected by the observer.
            body = json.loads(content)
            wire.append(dict(body=body, request_sha256=sha256(content).hexdigest()))
            return Response(body)

    def fake_http_transport(**kwargs):
        assert kwargs == {'retries': 0, 'trust_env': False}
        return object()  # No real http transport or network implementation is instantiated.

    monkeypatch.setattr(httpx, 'AsyncClient', Client)
    monkeypatch.setattr(httpx, 'AsyncHTTPTransport', fake_http_transport)
    transport = HttpxProviderTransport(configured, secrets=SyntheticSecrets(),
                                       ledger=ConservativeReservation(ledger))
    model = ProviderModel(configured, counter=Counter(), transport=transport,
                          output_tokens=192, timeout_seconds=4)
    runtime, text, memory, mirror, _ = setup(tmp_path, model)
    observe = runtime.observe_stream

    def observe_stream(item):
        observations.append(deepcopy(item))
        observe(item)
        if item['event'] == 'delta' and item['sequence'] == 2:
            tail_observed.set()

    runtime.observe_stream = observe_stream
    try:
        text.send('合成输入')
        first = wait_for(lambda: runtime.poll()['event'])
        assert first['display_text'] == prefix
        assert not transport_closed.is_set() and runtime.generation_terminal_us is None
        assert text.snapshot()['partial'] == ''
        assert all(row['role'] == 'user' for row in mirror.history()['messages'])
        runtime.ack(dict(sequence=first['sequence'], fingerprint=first['fingerprint'],
                         host_receipt='synthetic-httpx-visible-prefix'))
        wait_for(lambda: runtime.poll()['event'] is None)
        release.set()
        assert tail_observed.wait(2), 'Raw suffix must reach the real host stream before failure'
        assert runtime.poll()['event'] is None
        assert tail not in mirror.history()['messages'][-1]['content']
        if ending == 'cancel':
            runtime.stop()
            text.stop()
        else:
            finish.set()
        wait_for(lambda: not text.snapshot()['busy'])
        wait_for(lambda: transport_closed.is_set() and len(ledger.finished) == 2)
        state = memory.store.stream_state(memory.companion, runtime.claim.input_id)
        expected = 'cancelled' if ending == 'cancel' else 'failed'
        assert state['outcome'] == text.snapshot()['attempt']['status'] == expected
        assert runtime.cancelled.is_set() == (ending == 'cancel')
        assert state['confirmed_text'] == mirror.history()['messages'][-1]['content'] == prefix
        assert state['approved_units'] == state['confirmed_units'] == 1
        assert state['unknown_units'] == 0 and runtime.poll()['event'] is None
        assert emitted == ['{"action":"answer","text":"' + prefix, tail]
        assert injected == ([] if ending == 'cancel' else ['httpx.ReadError'])
        assert [c['stage'] for c in ledger.calls] == ['demand_control', 'stream_unit_review']
        assert [c['request_sha256'] for c in ledger.calls] == [c['request_sha256'] for c in wire]
        generation_receipt = next(c for c in ledger.finished if c['receipt'] == '1')
        assert generation_receipt == dict(receipt='1', status=expected, usage=None,
                                         transmission='response_received')
        assert not model.failed  # A transport failure is not a tokenizer incompatibility.
        assert runtime.snapshot()['retry_allowed'] is False
        with pytest.raises(ValueError, match='呈现收据'):
            text.retry()
        assert len(wire) == 2
        record_property('host_transport_replay', json.dumps(dict(
            ending=ending, wire=wire, emitted=emitted, observations=observations,
            event=first, state=state, history=mirror.history()['messages'],
            ledger_calls=ledger.calls, ledger_finished=ledger.finished,
            network_calls=0, real_secret_reads=0), ensure_ascii=False))
    finally:
        release.set()
        finish.set()
        transport.cancel()
        memory.close()
        mirror.close()


def test_real_loopback_default_unit_is_durably_acked_before_server_terminal(
    tmp_path, record_property,
):
    """The producer holds its suffix/STOP until the real default host persists ACK."""
    import sqlite3
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from test_model_provider_contract import event, route
    from test_provider_loopback import Ledger
    from test_text_provider_model import Counter
    from virtual_companion.providers.httpx_transport import HttpxProviderTransport
    from virtual_companion.text.provider_model import ConservativeReservation, ProviderModel

    release, abort, terminal_written, generation_closed = (
        threading.Event() for _ in range(4)
    )
    timeline, wire, reviews, observations, server_errors, secret_calls = [], [], [], [], [], []
    prefix, suffix = '首句回应当前输入。', '后句补充具体信息。'
    first_delta = '{"action":"answer","text":"' + prefix
    last_delta = suffix + '"}'

    def mark(name, **fields):
        timeline.append(dict(event=name, monotonic_ns=time.monotonic_ns(), **fields))

    def content(value):
        return event({'choices': [{'index': 0, 'delta': {'content': value},
                                   'finish_reason': None}]})

    terminal = event({'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})
    terminal += event({'choices': [], 'usage': {'prompt_tokens': 17,
                       'completion_tokens': 5, 'total_tokens': 22}})
    terminal += b'data: [DONE]\n\n'
    first_wire, last_wire = content(first_delta), content(last_delta)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):  # noqa: N802
            raw = self.rfile.read(int(self.headers['Content-Length']))
            body = json.loads(raw)
            review = '你只审查候选文字' in body['messages'][0]['content']
            wire.append(dict(review=review, body=body, request_sha256=sha256(raw).hexdigest()))
            if review:
                packet = recorded_review_packet(body['messages'])
                reviews.append(packet)
                opinion = json.dumps(synthetic_unit_opinion(packet))
                payload = content(opinion) + terminal
            else:
                payload = first_wire + last_wire + terminal
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            try:
                if review:
                    self.wfile.write(payload)
                    self.wfile.flush()
                    mark('server_review_written', final=packet['final'])
                    return
                self.wfile.write(first_wire)
                self.wfile.flush()
                mark('server_prefix_write_finished')
                # Cleanup aborts the response without writing a terminal. Only
                # the test's durable-ACK checks may release further generation.
                while not release.wait(0.01):
                    if abort.is_set():
                        mark('server_generation_aborted')
                        return
                if abort.is_set():
                    return
                mark('server_generation_resumed')
                self.wfile.write(last_wire)
                self.wfile.flush()
                mark('server_suffix_write_finished')
                mark('server_terminal_write_attempt')
                self.wfile.write(terminal)
                self.wfile.flush()
                mark('server_terminal_write_finished')
                terminal_written.set()
            except Exception as error:
                server_errors.append(f'{type(error).__name__}: {error}')
            finally:
                if not review:
                    generation_closed.set()

    service = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    service.daemon_threads = True
    configured = route(route_id='synthetic-default-delayed-producer', explicit=True,
        endpoint=f'http://127.0.0.1:{service.server_port}/chat/completions',
        model='synthetic-sse-model', auth='bearer', secret_ref='dpapi:' + 'a' * 32,
        data_scope=('current', 'history', 'valid_memory', 'derived'))

    class SyntheticSecrets:
        def get(self, reference, endpoint):
            assert (reference, endpoint) == (configured.secret_ref, configured.endpoint)
            secret_calls.append('synthetic-in-memory')
            return 'synthetic-in-memory-port-value'

    ledger = Ledger(limit=4)
    transport = HttpxProviderTransport(configured, secrets=SyntheticSecrets(),
                                       ledger=ConservativeReservation(ledger))
    model = ProviderModel(configured, counter=Counter(), transport=transport,
                          output_tokens=192, timeout_seconds=4)
    runtime, text, memory, mirror, _ = setup(tmp_path, model)
    observe = runtime.observe_stream

    def observe_stream(item):
        observations.append(deepcopy(item))
        mark('host_stream_' + item['event'], sequence=item.get('sequence'))
        observe(item)

    runtime.observe_stream = observe_stream
    server_worker = threading.Thread(
        target=lambda: service.serve_forever(poll_interval=0.01),
        name='synthetic-default-provider-http', daemon=True,
    )
    server_worker.start()
    first = last = prefix_state = final_state = None
    durable_ack_ns = visible_ns = None
    try:
        text.send('合成输入')
        first = wait_for(lambda: runtime.poll()['event'])
        visible_ns = time.monotonic_ns()
        timeline.append(dict(event='host_poll_visible', monotonic_ns=visible_ns, sequence=1))
        assert first['sequence'] == 1 and first['text'] == first['display_text'] == prefix
        assert text.snapshot()['busy'] and text.snapshot()['partial'] == ''
        assert all(row['role'] == 'user' for row in mirror.history()['messages'])
        assert not release.is_set() and not terminal_written.is_set()
        assert not any(row['event'].startswith('server_terminal_') for row in timeline)
        assert transport.metrics['finished'] is None
        assert runtime.snapshot()['generation_terminal_us'] is None
        assert not any(row['receipt'] == '1' for row in ledger.finished)
        pending = memory.store.stream_state(memory.companion, runtime.claim.input_id)
        assert pending['approved_units'] == 1 and pending['confirmed_units'] == 0

        receipt = 'synthetic-real-httpx-visible-first'
        assert runtime.ack(dict(sequence=first['sequence'], fingerprint=first['fingerprint'],
                                host_receipt=receipt)) == {'status': 'confirmed'}
        durable_ack_ns = time.monotonic_ns()
        timeline.append(dict(event='host_durable_ack_returned', monotonic_ns=durable_ack_ns,
                             sequence=1))
        prefix_state = memory.store.stream_state(memory.companion, runtime.claim.input_id)
        assert prefix_state['confirmed_units'] == 1 and prefix_state['outcome'] == 'unsettled'
        assert prefix_state['confirmed_text'] == prefix
        # A separate read-only connection verifies commit visibility; it never
        # manufactures approval, dispatch or confirmation rows.
        with sqlite3.connect(memory.store.path.as_uri() + '?mode=ro', uri=True) as db:
            durable = db.execute(
                'SELECT sequence,status,host_receipt FROM stream_segments '
                'WHERE companion=? AND input=? ORDER BY sequence',
                (memory.companion, runtime.claim.input_id),
            ).fetchall()
        assert durable == [(1, 'confirmed', receipt)]
        first_message = mirror.history()['messages'][-1]
        assert first_message['role'] == 'assistant' and first_message['content'] == prefix
        assert runtime.snapshot()['generation_terminal_us'] is None
        assert not terminal_written.is_set() and transport.metrics['finished'] is None
        assert text.snapshot()['busy']
        mark('host_release_after_durable_prefix')
        release.set()

        last = wait_for(lambda: (item if (item := runtime.poll()['event'])
                                and item['sequence'] == 2 else None))
        mark('host_poll_visible', sequence=2)
        assert terminal_written.wait(1), 'The producer must finish after the first durable ACK'
        assert last['text'] == suffix and last['display_text'] == prefix + suffix
        before_last_ack = memory.store.stream_state(memory.companion, runtime.claim.input_id)
        assert before_last_ack['confirmed_units'] == 1
        assert before_last_ack['confirmed_text'] == prefix
        assert mirror.history()['messages'][-1] == first_message
        assert runtime.ack(dict(sequence=last['sequence'], fingerprint=last['fingerprint'],
                                host_receipt='synthetic-real-httpx-visible-final')) \
            == {'status': 'confirmed'}
        mark('host_durable_ack_returned', sequence=2)
        wait_for(lambda: not text.snapshot()['busy'])
        final_state = memory.store.stream_state(memory.companion, runtime.claim.input_id)
        assert text.snapshot()['attempt']['status'] == final_state['outcome'] == 'completed'
        assert final_state['approved_units'] == final_state['confirmed_units'] == 2
        assert final_state['unknown_units'] == 0
        assert final_state['confirmed_text'] == prefix + suffix
        assert runtime.poll()['event'] is None
        assistants = [row for row in mirror.history()['messages'] if row['role'] == 'assistant']
        assert len(assistants) == 1 and assistants[0]['id'] == first_message['id']
        assert assistants[0]['content'] == prefix + suffix
        segments = memory.store.segments(memory.companion, runtime.claim.input_id)
        assert [row['status'] for row in segments] == ['confirmed', 'confirmed']
        assert [row['sequence'] for row in segments] == [1, 2]
        moments = {row['event']: row['monotonic_ns'] for row in timeline
                   if row['event'].startswith('server_') and 'review' not in row['event']}
        assert moments['server_prefix_write_finished'] < visible_ns <= durable_ack_ns
        assert durable_ack_ns < moments['server_generation_resumed'] \
            < moments['server_suffix_write_finished'] \
            < moments['server_terminal_write_attempt'] \
            <= moments['server_terminal_write_finished']
        assert [row['final'] for row in reviews] == [False, True]
        assert [row['candidate_prefix'] for row in reviews] == [prefix, prefix + suffix]
        assert [row['stage'] for row in ledger.calls] \
            == ['demand_control', 'stream_unit_review', 'stream_unit_review']
        assert [row['request_sha256'] for row in ledger.calls] \
            == [row['request_sha256'] for row in wire]
        assert len(ledger.finished) == len(secret_calls) == len(wire) == 3
        assert all(row['status'] == 'completed' for row in ledger.finished)
        assert not server_errors and not model.failed and not model._streams
    finally:
        abort.set()
        if text.snapshot()['busy']:
            runtime.stop()
            text.stop()
            wait_for(lambda: not text.snapshot()['busy'])
        model.close()
        service.shutdown()
        service.server_close()
        server_worker.join(1)
        replay = dict(timeline=timeline, wire=wire, reviews=reviews,
            observations=observations, first_event=first, last_event=last,
            prefix_state=prefix_state, final_state=final_state,
            durable_ack_ns=durable_ack_ns, visible_ns=visible_ns,
            producer_released_after_ack=release.is_set(),
            producer_terminal_written=terminal_written.is_set(),
            generation_closed=generation_closed.is_set(), server_errors=server_errors,
            ledger_calls=ledger.calls, ledger_finished=ledger.finished,
            transport_metrics=transport.metrics,
            history=mirror.history()['messages'], loopback_only=True,
            real_secret_reads=0, paid_api_calls=0, real_renderer=False,
            remote_deepseek_timing=False, formal_s_acceptance=False)
        record_property('host_delayed_producer_replay', json.dumps(replay, ensure_ascii=False))
        (tmp_path / 'host-delayed-producer-replay.json').write_text(
            json.dumps(replay, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        memory.close()
        mirror.close()
        assert not server_worker.is_alive(), 'Loopback server worker leaked'
