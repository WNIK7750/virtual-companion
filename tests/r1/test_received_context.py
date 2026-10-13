import json
from dataclasses import replace

import pytest
from test_rebuilt_turns import A, claim, config
from test_received_delivery import setup_received
from test_segment_store import reviewed
from virtual_companion.boundary.contracts import VersionConflict, canonical
from virtual_companion.boundary.history_readset import history_readset
from virtual_companion.cognition.short_context import ShortContextAdmission
from virtual_companion.cognition.short_dialogue import ShortRequest


def receive_then_continue(tmp_path):
    store, first, event = setup_received(tmp_path)
    store.receive_segment(A, first.input_id, 1, fingerprint=event['fingerprint'],
                          host_receipt='own-sent-not-visible')
    store.finish_stream(first, outcome='received', now_us=100)
    item = store.accept(config(), 'next', '接着刚才说的继续', session_id=first.session_id)
    run = claim(store, item, now=200)
    admission = ShortContextAdmission(store, lambda *args: ShortRequest(args[-1]),
                                      evidence_clock=lambda: 200)
    request = admission(run, store.snapshot(A), '接着刚才说的继续')
    return store, first, event, run, admission, request


def test_received_output_is_self_context_not_visible_history_or_user_fact(tmp_path):
    store, first, event, run, admission, request = receive_then_continue(tmp_path)
    data = json.loads(request.context.payload)
    assert all(item['role'] == 'user' for item in data['history'])
    assert data['received_output'] == [{
        'input_id': first.input_id, 'sequence': 1, 'fingerprint': event['fingerprint'],
        'text': '我在这里。', 'response_type': 'answer',
        'visibility': 'not_confirmed_at_admission',
    }]
    rendered = replace(request, context=replace(request.context, rendering='prose'))
    messages = rendered.model_request(192)['messages']
    assert any('我在这里。' in m['content'] and '不代表用户已看见' in m['content']
               for m in messages)
    assert not any(s['text'] == '我在这里。' for s in request.personal_sources(
        A, run.input_id, run.state_version))
    assert not [s for s in store.background_sources(A) if s['role'] == 'assistant']
    assert store.stream_state(A, first.input_id)['confirmed_units'] == 0
    admission.validate(request.context)
    readset = history_readset(A, run.input_id, run.state_version, request.context.payload)
    assert readset['protocol'] == 'history-readset.v4'
    store.approve_segment(run, reviewed(run), now_us=200, readset=readset)


def test_received_context_tamper_is_rejected_before_generation_and_approval(tmp_path):
    store, _, _, run, admission, request = receive_then_continue(tmp_path)
    data = json.loads(request.context.payload)
    data['received_output'][0]['text'] = '并没有发送的文本'
    tampered = replace(request.context, payload=canonical(data))
    with pytest.raises(VersionConflict, match='Received'):
        admission.validate(tampered)
    readset = history_readset(A, run.input_id, run.state_version, tampered.payload)
    with pytest.raises(VersionConflict):
        store.approve_segment(run, reviewed(run), now_us=200, readset=readset)


def test_late_visibility_preserves_admitted_reference_then_removes_duplicate_projection(tmp_path):
    store, first, event, run, admission, request = receive_then_continue(tmp_path)
    assert json.loads(request.context.payload)['received_output']
    store.confirm_segment(A, first.input_id, 1, fingerprint=event['fingerprint'],
                          host_receipt='now-visible', channel='text_visible')
    admission.validate(request.context)
    readset = history_readset(A, run.input_id, run.state_version, request.context.payload)
    store.approve_segment(run, reviewed(run), now_us=200, readset=readset)
    next_request = admission(run, store.snapshot(A), '接着刚才说的继续')
    data = json.loads(next_request.context.payload)
    assert not data.get('received_output')
    assert [i['text'] for i in data['history'] if i['role'] == 'assistant'] == ['我在这里。']


def test_visibility_between_history_and_received_queries_uses_one_snapshot(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    store, first, event, run, admission, _ = receive_then_continue(tmp_path)
    original = store._exchange_context
    started = Event()
    futures = []

    def acknowledge():
        started.set()
        return store.confirm_segment(A, first.input_id, 1, fingerprint=event['fingerprint'],
                                     host_receipt='racing-visible', channel='text_visible')

    def racing_ack(db, *args, **kwargs):
        result = original(db, *args, **kwargs)
        futures.append(pool.submit(acknowledge))
        assert started.wait(1)
        return result

    monkeypatch.setattr(store, '_exchange_context', racing_ack)
    with ThreadPoolExecutor(max_workers=1) as pool:
        request = admission(run, store.snapshot(A), '接着刚才说的继续')
        assert futures[0].result(timeout=2)['status'] == 'confirmed'
    data = json.loads(request.context.payload)
    texts = [i['text'] for i in data['history']] + [i['text'] for i in data['received_output']]
    assert texts.count('我在这里。') == 1
    admission.validate(request.context)


@pytest.mark.parametrize('change', ['correct', 'revoke', 'expiry'])
def test_invalid_memory_cannot_survive_through_two_unseen_answers(tmp_path, change):
    from datetime import UTC, datetime, timedelta

    from virtual_companion.boundary.contracts import Scope
    from virtual_companion.boundary.memory import FactKey, MemoryMutation
    from virtual_companion.text.chronology import utc_microseconds

    store, first, event, run, admission, request = receive_then_continue(tmp_path)
    readset = history_readset(A, run.input_id, run.state_version, request.context.payload)
    store.approve_segment(run, reviewed(run, ('沿着上一条继续。',)), now_us=200, readset=readset)
    second = store.dispatch_segment(run, 1, now_us=200)
    store.receive_segment(A, run.input_id, 1, fingerprint=second['fingerprint'],
                          host_receipt='second-not-visible')
    store.finish_stream(run, outcome='received', now_us=200)
    when = datetime(2026, 10, 10, tzinfo=UTC)
    seed = MemoryMutation(Scope(A, first.session_id), 'seed', 'assert_fact', 'home',
                          '临时住址', 'manual', first.input_id, 'synthetic', when,
                          FactKey('user', 'home', 'profile'),
                          expires_at=when + timedelta(minutes=1))
    owner = store.owner(A)
    store.mutate(seed, owner, expected_version=store.snapshot(A).write_version)
    if change != 'expiry':
        store.mutate(replace(seed, mutation_id='change', kind=change,
                             content='新住址' if change == 'correct' else '',
                             source_id='new-source', expected_record_version=1, fact_key=None),
                     owner, expected_version=store.snapshot(A).write_version)
    now = utc_microseconds(when + timedelta(minutes=2 if change == 'expiry' else 0))
    excluded = store.ineffective_history_sources(A, now_us=now)
    assert {'segment-' + event['fingerprint'], 'segment-' + second['fingerprint']} <= excluded
    item = store.accept(config(), 'third', '继续聊', session_id=first.session_id)
    third = claim(store, item, now=now)
    admission = ShortContextAdmission(store, lambda *args: ShortRequest(args[-1]),
                                      evidence_clock=lambda: now)
    request = admission(third, store.snapshot(A), '继续聊')
    assert not json.loads(request.context.payload).get('received_output')
    assert store.stream_state(A, first.input_id)['received_units'] == 1
    assert not [s for s in store.background_sources(A) if s['role'] == 'assistant']


def test_sent_context_is_session_scoped_and_old_store_cannot_authorize_it(tmp_path):
    from virtual_companion.boundary.sqlite_memory_management import SQLiteExplicitMemoryStore

    store, _, _, run, _, request = receive_then_continue(tmp_path)
    refs = history_readset(A, run.input_id, run.state_version,
                          request.context.payload)['received_output']
    with store.transaction(write=False) as db:
        assert not SQLiteExplicitMemoryStore._received_references_current(
            store, db, A, run.input_id, refs)
    store.cancel(A, run.input_id)
    other = store.accept(config(), 'other-session', '新会话')
    other_run = claim(store, other, now=300)
    assert other_run.session_id != run.session_id
    with pytest.raises(VersionConflict, match='Received'):
        store.validate_received_output_context(A, other_run.input_id,
            json.loads(request.context.payload)['received_output'], now_us=300)
    assert store.dialogue_context(other_run, now_us=300)[1] == []


def test_overlapping_validation_cannot_read_a_future_cache_timestamp(tmp_path, monkeypatch):
    from itertools import count

    store, _, _, _, admission, request = receive_then_continue(tmp_path)
    clock = count(201)
    admission.evidence_clock = lambda: next(clock)
    original = store.validate_received_output_context
    entered = False

    def overlapping(*args, **kwargs):
        nonlocal entered
        if not entered:
            entered = True
            # Another checkpoint finishes while this checkpoint is querying DB.
            admission.validate(request.context)
        return original(*args, **kwargs)

    monkeypatch.setattr(store, 'validate_received_output_context', overlapping)
    admission.validate(request.context)
