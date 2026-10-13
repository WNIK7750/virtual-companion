"""Input-time client observations are immutable evidence, never operation results."""
import json
import threading
from dataclasses import replace

import pytest
from test_rebuilt_turns import A, claim, config
from test_segment_store import reviewed
from virtual_companion.boundary.contracts import VersionConflict, canonical
from virtual_companion.boundary.history_readset import history_readset
from virtual_companion.boundary.receiving_context import validate_receiving_context
from virtual_companion.boundary.sqlite_memory_management import SQLiteExplicitMemoryStore
from virtual_companion.cognition.short_context import ShortContextAdmission
from virtual_companion.cognition.short_dialogue import ShortRequest
from virtual_companion.text.transport import validate


def observation(quiet=True):
    return dict(protocol='receiving-context.v1', source='receiving-client', scope='this_client',
                observation='at_input_submission', quiet=quiet)


@pytest.mark.parametrize('value', [None, {}, {**observation(), 'quiet': 1},
                                  {**observation(), 'scope': 'all_devices'},
                                  {**observation(), 'success': True}])
def test_receiving_transport_refuses_malformed_observations(value):
    with pytest.raises(ValueError):
        validate('send', {'text': '接着聊', 'receiving_context': value})


def test_intake_freezes_observation_and_keeps_legacy_identity(tmp_path):
    store = SQLiteExplicitMemoryStore(tmp_path / 'memory.db')
    value = observation()
    assert validate_receiving_context(value) == value
    first = store.accept(config(), 'first', '今天路过旧书店。', receiving_context=value)
    value['quiet'] = False
    assert json.loads(first.config_json)['receiving_context'] == observation()
    assert store.accept(config(), 'first', '今天路过旧书店。',
                        receiving_context=observation()) == first
    with pytest.raises(ValueError, match='Conflicting'):
        store.accept(config(), 'first', '今天路过旧书店。', receiving_context=value)
    legacy = store.accept(config(), 'legacy', '没提供偏好')
    assert json.loads(legacy.config_json) == config().attempt_binding({})
    assert validate('send', {'text': '没提供偏好'}) == {'text': '没提供偏好'}
    accepted = validate('send', {'text': '接着聊', 'receiving_context': value})
    assert accepted['receiving_context'] == value


def admission(store, run, text):
    port = ShortContextAdmission(store, lambda *args: ShortRequest(args[-1]),
                                 evidence_clock=lambda: 101)
    return port, port(run, store.snapshot(A), text)


def test_generation_review_share_original_observation_without_personal_or_operation_claim(tmp_path):
    from virtual_companion.cognition.stream_unit_review import (
        BoundedUnitReviewer,
        StructuredUnitAssessor,
    )

    store = SQLiteExplicitMemoryStore(tmp_path / 'memory.db')
    run = claim(store, store.accept(config(), 'first', '今天路过旧书店。',
                                    receiving_context=observation()))
    port, request = admission(store, run, '今天路过旧书店。')
    data = json.loads(request.context.payload)
    assert data['receiving_context'] == observation()
    assert 'current_operation' not in data
    assert not any(s['text'] == canonical(observation()) for s in request.personal_sources(
        A, run.input_id, run.state_version))
    rendered = replace(request, context=replace(request.context, rendering='prose'))
    assert any('输入提交时' in m['content'] and '本接收端' in m['content']
               for m in rendered.model_request(192)['messages'])

    def capture(stage, messages, schema, cancelled):
        view = json.loads(messages[-1]['content'])
        assert view['context']['receiving_context'] == data['receiving_context']
        raise RuntimeError('captured same receiving observation')

    review = BoundedUnitReviewer(run, request, StructuredUnitAssessor(capture, threading.Event()))
    with pytest.raises(RuntimeError, match='captured same'):
        review('旧书店的关门告示确实让人停了一下。', True)
    port.validate(request.context)


@pytest.mark.parametrize('change', ['omit', 'replace'])
def test_observation_cannot_be_omitted_or_changed_for_generation_and_approval(tmp_path, change):
    store = SQLiteExplicitMemoryStore(tmp_path / 'memory.db')
    run = claim(store, store.accept(config(), 'first', '接着聊', receiving_context=observation()))
    port, request = admission(store, run, '接着聊')
    data = json.loads(request.context.payload)
    if change == 'omit':
        del data['receiving_context']
    else:
        data['receiving_context'] = observation(False)
    context = replace(request.context, payload=canonical(data))
    with pytest.raises(VersionConflict, match='receiving'):
        port.validate(context)
    readset = history_readset(A, run.input_id, run.state_version, context.payload)
    with pytest.raises(VersionConflict):
        store.approve_segment(run, reviewed(run), now_us=102, readset=readset)


def test_observation_coexists_with_received_and_character_readset(tmp_path):
    from test_character_context import character
    from test_character_context import config as role_config
    from test_received_delivery import setup_received

    store, first, event = setup_received(tmp_path)
    store.receive_segment(A, first.input_id, 1, fingerprint=event['fingerprint'],
                          host_receipt='received')
    store.finish_stream(first, outcome='received', now_us=100)
    run = claim(store, store.accept(role_config(character()), 'second', '继续',
        session_id=first.session_id, receiving_context=observation(False)))
    _, request = admission(store, run, '继续')
    readset = history_readset(A, run.input_id, run.state_version, request.context.payload)
    assert readset['protocol'] == 'history-readset.v5'
    assert readset['character_sha256'] == character().reference.sha256
    assert len(readset['received_output']) == 1
    assert len(readset['receiving_sha256']) == 64
    store.approve_segment(run, reviewed(run), now_us=102, readset=readset)


def test_default_runtime_retains_queued_observation_through_reopen(tmp_path):
    from test_default_text_delivery import setup, wait_for
    from virtual_companion.text.sqlite_store import SQLiteConversationStore

    runtime, session, memory, mirror, model = setup(tmp_path)
    queued = mirror.enqueue_input('现在只想聊聊。',
                                   {**session.binding, 'receiving_context': observation()})
    mirror.close()
    mirror = SQLiteConversationStore(tmp_path / 'chat', companion_id=A)
    runtime.mirror = session.store = mirror
    session.start_queued(mirror.queued_inputs()[0])
    event = wait_for(lambda: runtime.poll()['event'])
    original = memory.store.inspect(A, event['basis']['input_id'])
    assert json.loads(original['config'])['receiving_context'] == observation()
    runtime.ack(dict(sequence=event['sequence'], fingerprint=event['fingerprint'],
                     host_receipt='visible'))
    wait_for(lambda: not session.snapshot()['busy'])
    assert session.binding == {}
    # The next input's observation is independent of both the queue and global model binding.
    next_attempt = session.send('这次帮我想办法。', receiving_context=observation(False))
    next_event = wait_for(lambda: runtime.poll()['event'])
    assert next_attempt['binding']['receiving_context'] == observation(False)
    assert json.loads(memory.store.inspect(A, next_event['basis']['input_id'])['config'])[
        'receiving_context'] == observation(False)
    runtime.ack(dict(sequence=next_event['sequence'], fingerprint=next_event['fingerprint'],
                     host_receipt='visible2'))
    wait_for(lambda: not session.snapshot()['busy'])
    assert queued['turn_id'] != next_attempt['turn_id']
    mirror.close()
    memory.close()


def test_retry_keeps_input_observation_when_model_binding_changes(tmp_path):
    from test_default_text_delivery import Model, setup, wait_for

    class FailOnce(Model):
        failed = False

        def stream(self, messages, cancelled):
            if not self.failed:
                self.failed = True
                raise RuntimeError('synthetic generation failure before approval')
            yield from super().stream(messages, cancelled)

    runtime, session, memory, mirror, _ = setup(tmp_path, FailOnce())
    first = session.send('先聊一会儿', receiving_context=observation())
    wait_for(lambda: not session.snapshot()['busy'])
    assert session.snapshot()['attempt']['status'] == 'failed'
    session.binding = {'new_model_binding': 'synthetic'}
    retried = session.retry()
    assert retried['turn_id'] == first['turn_id']
    assert retried['binding']['receiving_context'] == observation()
    assert retried['binding']['new_model_binding'] == 'synthetic'
    event = wait_for(lambda: runtime.poll()['event'])
    assert json.loads(memory.store.inspect(A, event['basis']['input_id'])['config'])[
        'receiving_context'] == observation()
    runtime.ack(dict(sequence=event['sequence'], fingerprint=event['fingerprint'],
                     host_receipt='retry-visible'))
    wait_for(lambda: not session.snapshot()['busy'])
    mirror.close()
    memory.close()
