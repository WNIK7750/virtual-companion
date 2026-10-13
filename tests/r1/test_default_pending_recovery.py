"""Approved but never dispatched is distinct from unknown presentation."""

import threading

import pytest
from test_default_runtime_recovery import completed, configured
from test_default_text_delivery import wait_for
from virtual_companion.boundary.contracts import VersionConflict
from virtual_companion.text.application import TextSession
from virtual_companion.text.default_runtime import DefaultTextRuntime
from virtual_companion.text.sqlite_store import SQLiteConversationStore


@pytest.mark.parametrize('reopen', [False, True])
def test_failed_undispatched_unit_kept_without_blocking_new_input(tmp_path, monkeypatch, reopen):
    runtime, session, memory, mirror, model = configured(tmp_path)
    dispatch = memory.store.dispatch_segment

    def fail_before_dispatch(*args, **kwargs):
        raise OSError('Synthetic crash before any dispatch')

    monkeypatch.setattr(memory.store, 'dispatch_segment', fail_before_dispatch)
    first = session.send('未派发的合成输入')
    wait_for(lambda: not session.snapshot()['busy'])
    claim = runtime.claim
    old = memory.store.segments(memory.companion, claim.input_id)
    assert len(old) == 1 and old[0]['status'] == 'pending'
    assert memory.store.stream_state(memory.companion, claim.input_id)['outcome'] == 'failed'
    assert [m['role'] for m in mirror.history()['messages']] == ['user']
    monkeypatch.setattr(memory.store, 'dispatch_segment', dispatch)
    if reopen:
        mirror.close()
        mirror = SQLiteConversationStore(tmp_path / 'chat', companion_id=memory.companion)
        runtime = DefaultTextRuntime(memory, mirror, model, binding={'test': 'synthetic'},
                                     delivery_timeout=0.8)
        session = TextSession(model, mirror, {}, prepare_turn=runtime.prepare_turn,
                              before_start=runtime.before_start)
    with pytest.raises(ValueError):
        session.retry()  # Never turn approved history into whole-turn replay.
    second = session.send('独立的下一条普通交流')
    event = completed(runtime, session)
    assert second['turn_id'] != first['turn_id']
    assert event['basis']['input_id'] != claim.input_id
    assert memory.store.segments(memory.companion, claim.input_id) == old
    with pytest.raises(VersionConflict):
        dispatch(claim, 1, now_us=runtime.clock(), clock=runtime.clock)
    with pytest.raises(ValueError):
        runtime.ack(dict(sequence=1, fingerprint=old[0]['fingerprint'],
                         host_receipt='cannot-invent-visibility'))
    assert len([m for m in mirror.history()['messages'] if m['role'] == 'assistant']) == 1
    mirror.close()


def test_live_pending_still_blocks_and_cancel_does_not_dispatch_it(tmp_path, monkeypatch):
    runtime, session, memory, mirror, _ = configured(tmp_path)
    dispatch = memory.store.dispatch_segment
    entered, release = threading.Event(), threading.Event()

    def held_dispatch(*args, **kwargs):
        entered.set()
        assert release.wait(4)
        return dispatch(*args, **kwargs)

    monkeypatch.setattr(memory.store, 'dispatch_segment', held_dispatch)
    session.send('持有未派发批准单元')
    assert entered.wait(4)
    claim = runtime.claim
    try:
        with pytest.raises(ValueError):
            runtime.before_start(False)
        runtime.stop()
    finally:
        release.set()
    wait_for(lambda: not session.snapshot()['busy'])
    assert memory.store.segments(memory.companion, claim.input_id)[0]['status'] == 'pending'
    assert runtime.poll()['event'] is None
    assert [m['role'] for m in mirror.history()['messages']] == ['user']
    runtime.before_start(False)
    mirror.close()
