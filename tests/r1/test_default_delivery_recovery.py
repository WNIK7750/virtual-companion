"""Late trusted display receipts repair history, not failed generation outcomes.

All generation/review/semantic decisions are recordings. These tests deliberately
do not claim that a renderer or user actually observed the supplied receipts.
"""

from datetime import timedelta
from uuid import uuid4

import pytest
from test_default_cross_turn_memory import ORIGINAL, QUERY, UPDATED, Scenario, build, turn
from test_default_text_delivery import Model, setup, wait_for
from virtual_companion.text.application import TextSession
from virtual_companion.text.default_runtime import DefaultTextRuntime
from virtual_companion.text.memory_runtime import DEFAULT_COMPANION_ID, TextMemoryRuntime
from virtual_companion.text.sqlite_store import SQLiteConversationStore


def unknown_reply(runtime, session, text, outcome="failed"):
    attempt = session.send(text)
    event = wait_for(lambda: runtime.poll()["event"])
    if outcome == "cancelled":
        runtime.stop()
        session.stop()
    wait_for(lambda: not session.snapshot()["busy"])
    assert session.snapshot()["attempt"]["status"] == outcome
    assert runtime.poll()["event"] is None
    assert runtime.state()["reconcile"] == [event]
    with pytest.raises(ValueError, match="呈现收据"):
        runtime.before_start(False)
    return attempt, event


def receipt_for(event):
    return dict(sequence=event["sequence"], fingerprint=event["fingerprint"],
                host_receipt="recorded-recovery-" + str(uuid4()))


def assert_one_source(memory, event):
    source = "segment-" + event["fingerprint"]
    with memory.store.transaction(write=False) as db:
        for table in ("stream_history", "background_sources"):
            # Both table names are fixed internal test constants.
            count = db.execute(f"SELECT count(*) FROM {table} WHERE companion=? AND source=?",  # noqa: S608
                               (memory.companion, source)).fetchone()[0]
            assert count == 1


def close(session, memory, mirror):
    if session.snapshot()["busy"]:
        session.stop()
    memory.close()
    mirror.close()


@pytest.mark.parametrize("outcome", ["failed", "cancelled"])
def test_default_late_receipt_preserves_terminal_and_unblocks_next_send(tmp_path, outcome):
    runtime, session, memory, mirror, model = setup(tmp_path)
    try:
        attempt, event = unknown_reply(runtime, session, "合成原轮输入", outcome)
        terminal = memory.store.stream_state(memory.companion, event["basis"]["input_id"])[
            "settlement"]
        calls = len(model.calls)
        receipt = receipt_for(event)
        assert runtime.ack(receipt) == runtime.ack(receipt) == {"status": "confirmed"}
        assert len(model.calls) == calls  # Receipt recovery never reruns the model.
        assert runtime.state()["reconcile"] == []
        state = memory.store.stream_state(memory.companion, event["basis"]["input_id"])
        assert state["outcome"] == outcome and state["settlement"] == terminal
        assert state["confirmed_text"] == event["display_text"]
        assert state["confirmed_units"] == 1 and state["unknown_units"] == 0
        assert session.snapshot()["attempt"]["status"] == outcome
        mirrored = [m for m in mirror.history()["messages"] if m["role"] == "assistant"]
        assert len(mirrored) == 1
        assert mirrored[0]["turn_id"] == attempt["turn_id"]
        assert mirrored[0]["content"] == event["display_text"]
        assert_one_source(memory, event)

        next_attempt = session.send("合成独立下一轮")
        assert next_attempt["turn_id"] != attempt["turn_id"]
        next_event = wait_for(lambda: runtime.poll()["event"])
        runtime.ack(receipt_for(next_event))
        wait_for(lambda: not session.snapshot()["busy"])
        assert session.snapshot()["attempt"]["status"] == "completed"
        assert memory.store.stream_state(memory.companion, event["basis"]["input_id"])[
            "settlement"] == terminal
        assert_one_source(memory, event)
    finally:
        close(session, memory, mirror)


def open_shared(tmp_path, companion):
    model = Model()
    memory = TextMemoryRuntime(tmp_path / "shared.sqlite3", companion, model=model, fake=True)
    mirror = SQLiteConversationStore(tmp_path / companion, companion_id=companion)
    runtime = DefaultTextRuntime(memory, mirror, model, binding={"test": "recovery-isolation"},
                                 delivery_timeout=0.8)
    session = TextSession(model, mirror, {}, prepare_turn=runtime.prepare_turn,
                          before_start=runtime.before_start)
    return runtime, session, memory, mirror, model


def test_reopened_unknowns_and_late_ack_are_scoped_to_current_companion(tmp_path):
    other = str(uuid4())
    events = {}
    for companion in (DEFAULT_COMPANION_ID, other):
        runtime, session, memory, mirror, _ = open_shared(tmp_path, companion)
        try:
            _, events[companion] = unknown_reply(runtime, session, "合成待恢复输入")
        finally:
            close(session, memory, mirror)

    runtime, session, memory, mirror, model = open_shared(tmp_path, DEFAULT_COMPANION_ID)
    try:
        assert runtime.state()["reconcile"] == [events[DEFAULT_COMPANION_ID]]
        assert runtime.poll()["event"] is None  # Reopening does not automatically redispatch.
        with pytest.raises(ValueError, match="Unknown scoped display"):
            runtime.ack(receipt_for(events[other]))
        event = events[DEFAULT_COMPANION_ID]
        receipt = receipt_for(event)
        runtime.ack(receipt)
        runtime.ack(receipt)
        assert model.calls == []
        assert session.snapshot()["attempt"]["status"] == "failed"
        assert runtime.state()["reconcile"] == []
        assert_one_source(memory, event)
        assert memory.store.segments(other, events[other]["basis"]["input_id"])[0][
            "status"] == "unknown"
        runtime.before_start(False)
    finally:
        close(session, memory, mirror)

    runtime, session, memory, mirror, model = open_shared(tmp_path, DEFAULT_COMPANION_ID)
    try:
        assert runtime.state()["reconcile"] == []
        runtime.ack(receipt)
        assert model.calls == []
        assert len([m for m in mirror.history()["messages"] if m["role"] == "assistant"]) == 1
        assert_one_source(memory, event)
        assert session.snapshot()["attempt"]["status"] == "failed"
    finally:
        close(session, memory, mirror)


@pytest.mark.parametrize("change", ["correct", "forget", "expiry"])
def test_late_ack_after_fact_invalidation_keeps_audit_without_reviving_memory(tmp_path, change):
    model = Scenario(expires=change == "expiry")
    memory, mirror, runtime, session = build(tmp_path, model)
    try:
        turn(runtime, session, ORIGINAL)
        fact = memory.store.recall(memory.companion, now_us=model.clock())[0]
        attempt, event = unknown_reply(runtime, session, QUERY)
        assert event["display_text"] == "你喜欢安静的音乐。"
        assert event["readset"]["facts"]
        terminal = memory.store.stream_state(memory.companion, event["basis"]["input_id"])[
            "settlement"]
        if change == "expiry":
            model.now += timedelta(days=2)
        else:
            memory.service.clock = model.clock
            result = memory.apply_action(dict(operation_id="recovery-" + change,
                action=change, record_id=fact["record_id"], expected_version=fact["version"],
                replacement=UPDATED if change == "correct" else ""))
            assert result["status"] == "committed"
        before_ack = len(model.calls), len(model.worker_calls)
        receipt = receipt_for(event)
        runtime.ack(receipt)
        runtime.ack(receipt)
        assert (len(model.calls), len(model.worker_calls)) == before_ack
        assert memory.store.stream_state(memory.companion, event["basis"]["input_id"])[
            "settlement"] == terminal
        assert session.snapshot()["attempt"]["status"] == "failed"
        assert_one_source(memory, event)
        assert any(m["turn_id"] == attempt["turn_id"] and m["content"] == event["display_text"]
                   for m in mirror.history()["messages"])
        assert all(row["content"] != ORIGINAL for row in
                   memory.store.recall(memory.companion, now_us=model.clock()))
        assert memory.scheduler.drain("late-receipt-background", limit=16) > 0
        assert len(model.worker_calls) == before_ack[1]
        source = "segment-" + event["fingerprint"]
        invalidated = next(row for row in memory.store.background_sources(memory.companion)
                           if row["source"] == source)
        assert invalidated["status"] == "invalidated"
    finally:
        close(session, memory, mirror)

    memory, mirror, runtime, session = build(tmp_path, model)
    try:
        recalled = turn(runtime, session, QUERY)
        assert recalled["display_text"] == ("你喜欢摇滚音乐。" if change == "correct"
                                             else "没有有效的音乐偏好记录。")
        evidence = "\n".join(item["content"] for item in model.primary[-1][1:-1])
        assert ORIGINAL not in evidence and "你喜欢安静的音乐" not in evidence
        assert memory.store.segments(memory.companion, event["basis"]["input_id"])[0][
            "text"] == "你喜欢安静的音乐。"
    finally:
        close(session, memory, mirror)
