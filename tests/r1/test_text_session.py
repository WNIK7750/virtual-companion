from __future__ import annotations

import threading
import time

import pytest
from virtual_companion.text.application import TextSession
from virtual_companion.text.memory import MemoryConversationStore


class ControlledModel:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.requests = []

    def stream(self, messages, cancelled):
        self.requests.append(messages)
        self.started.set()
        yield "partial"
        self.release.wait(2)
        if not cancelled.is_set():
            yield " complete"

    def cancel(self):
        self.release.set()


def wait_done(session):
    deadline = time.monotonic() + 3
    while session.snapshot()["busy"] and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not session.snapshot()["busy"]


def test_cancelled_attempt_is_not_committed_and_retry_does_not_duplicate_user():
    model = ControlledModel()
    session = TextSession(model, MemoryConversationStore(), {"profile": "synthetic-v1"})
    first = session.send("hello")
    assert model.started.wait(1)
    session.stop()
    assert [m["role"] for m in session.history()["messages"]] == ["user"]
    assert session.snapshot()["attempt"]["status"] == "cancelled"
    second = session.retry()
    assert second["attempt_id"] != first["attempt_id"]
    wait_done(session)
    assert [m["role"] for m in session.history()["messages"]] == ["user", "assistant"]
    assert model.requests[-1] == [{"role": "user", "content": "hello"}]
    with pytest.raises(ValueError):
        session.retry()


def test_busy_session_rejects_send_and_switch_and_keeps_state_consistent():
    model = ControlledModel()
    session = TextSession(model, MemoryConversationStore(), {})
    session.send("one")
    assert model.started.wait(1)
    with pytest.raises(ValueError):
        session.send("two")
    with pytest.raises(ValueError):
        session.new_session()
    session.stop()
    assert len(session.history()["messages"]) == 1


def test_generator_failure_retains_user_and_makes_retry_visible():
    class FailingModel:
        def stream(self, messages, cancelled):
            yield "uncommitted"
            raise OSError("sensitive internals")

        def cancel(self):
            pass

    session = TextSession(FailingModel(), MemoryConversationStore(), {})
    session.send("hello")
    wait_done(session)
    state = session.snapshot()
    assert state["attempt"]["status"] == "failed"
    assert "sensitive" not in str(state)
    assert len(session.history()["messages"]) == 1


@pytest.mark.parametrize("invalid_delta", [False, True])
def test_session_closes_retained_stream_before_finishing_attempt(invalid_delta):
    closed = threading.Event()
    store = MemoryConversationStore()

    class Model:
        def stream(self, messages, cancelled):
            def generate():
                try:
                    if not invalid_delta:
                        cancelled.set()
                    yield object() if invalid_delta else "not published"
                finally:
                    closed.set()
            # Do not rely on reference counting to close this active resource.
            self.retained = generate()
            return self.retained

        def cancel(self):
            pass

    original_finish = store.finish
    closed_at_finish = []

    def finish(*args):
        closed_at_finish.append(closed.is_set())
        return original_finish(*args)

    store.finish = finish
    model = Model()
    session = TextSession(model, store, {})
    session.send("hello")
    wait_done(session)
    try:
        assert closed_at_finish == [True]
        assert session.snapshot()["attempt"]["status"] == (
            "failed" if invalid_delta else "cancelled"
        )
        assert [m["role"] for m in session.history()["messages"]] == ["user"]
    finally:
        model.retained.close()


def test_recording_store_switch_preserves_each_session_and_attempt():
    store = MemoryConversationStore()
    first_id = store.session_id
    first = store.begin("first", {})
    store.finish(first["attempt_id"], "completed", "reply")
    second_id = store.new_session()["session_id"]
    second = store.begin("second", {})
    store.finish(second["attempt_id"], "cancelled", "")
    assert store.select_session(first_id)["messages"][-1]["content"] == "reply"
    assert store.latest_attempt()["status"] == "completed"
    assert len(store.select_session(second_id)["messages"]) == 1
    assert store.latest_attempt()["status"] == "cancelled"
    with pytest.raises(ValueError):
        store.select_session("unknown")
    assert store.session_id == second_id


def test_history_read_failure_does_not_leave_unstarted_running_attempt(monkeypatch):
    store = MemoryConversationStore()
    session = TextSession(ControlledModel(), store, {})
    with monkeypatch.context() as scoped:
        scoped.setattr(store, "history", lambda: (_ for _ in ()).throw(OSError("disk read")))
        with pytest.raises(OSError):
            session.send("not accepted")
    assert store.latest_attempt() is None
    assert store.history()["messages"] == []


@pytest.mark.parametrize("failure", ["before_accept", "after_accept", "preflight"])
def test_bound_preparation_failure_retries_same_persisted_source(tmp_path, failure):
    from virtual_companion.text.memory_runtime import DEFAULT_COMPANION_ID, TextMemoryRuntime
    from virtual_companion.text.sqlite_store import SQLiteConversationStore

    store = SQLiteConversationStore(tmp_path / "conversation")
    memory = TextMemoryRuntime(tmp_path / "memory.sqlite3", DEFAULT_COMPANION_ID)
    calls, receipts = [], []
    fail = True

    class Prepared:
        def check_context(self):
            calls.append("count")
            if fail and failure == "preflight":
                raise OSError("synthetic count failure")

        def stream(self, cancelled):
            calls.append("stream")
            yield "synthetic approved elsewhere"

    def prepare(attempt, session_id, messages, cancelled):
        assert store.latest_attempt()["attempt_id"] == attempt["attempt_id"]
        current = [m for m in messages if m["turn_id"] == attempt["turn_id"]
                   and m["role"] == "user"]
        assert len(current) == 1
        if fail and failure == "before_accept":
            raise OSError("synthetic intake failure")
        receipts.append(memory.accept_user(attempt["turn_id"], current[0]["content"], session_id))
        if fail and failure == "after_accept":
            raise OSError("synthetic post-intake failure")
        return Prepared()

    session = TextSession(ControlledModel(), store, {}, prepare_turn=prepare)
    first = session.send("hello")
    wait_done(session)
    assert session.snapshot()["attempt"]["status"] == "failed"
    assert "stream" not in calls
    assert len(store.history()["messages"]) == 1
    store.close()
    store = SQLiteConversationStore(tmp_path / "conversation")
    fail = False
    session = TextSession(ControlledModel(), store, {}, prepare_turn=prepare)
    second = session.retry()
    wait_done(session)
    assert second["turn_id"] == first["turn_id"]
    assert second["attempt_id"] != first["attempt_id"]
    assert all(r.input_id == receipts[0].input_id for r in receipts)
    assert calls[-2:] == ["count", "stream"]
    assert len(memory.store.background_sources(DEFAULT_COMPANION_ID)) == 1
    store.close()


def test_bound_preparation_is_cancellable_without_session_lock_or_model_fallback():
    entered = threading.Event()

    def prepare(attempt, session_id, messages, cancelled):
        entered.set()
        assert cancelled.wait(2)
        raise InterruptedError("cancelled intake")

    model = ControlledModel()
    session = TextSession(model, MemoryConversationStore(), {}, prepare_turn=prepare)
    session.send("hello")
    assert entered.wait(1)
    assert session.snapshot()["busy"]
    session.stop()
    assert session.snapshot()["attempt"]["status"] == "cancelled"
    assert not model.requests


def test_thread_start_failure_retains_retryable_input(monkeypatch):
    store = MemoryConversationStore()
    model = ControlledModel()
    session = TextSession(model, store, {})
    with monkeypatch.context() as patch:
        patch.setattr(threading.Thread, "start", lambda self: (_ for _ in ()).throw(
            RuntimeError("synthetic thread launch failure")))
        with pytest.raises(RuntimeError, match="thread launch"):
            session.send("hello")
    failed = store.latest_attempt()
    assert failed["status"] == "failed"
    assert not session.snapshot()["busy"]
    model.release.set()
    retried = session.retry()
    wait_done(session)
    assert retried["turn_id"] == failed["turn_id"]
    assert len([m for m in store.history()["messages"] if m["role"] == "user"]) == 1


@pytest.mark.parametrize("already_accepted", [False, True])
def test_interrupted_intake_reopen_uses_original_identity(tmp_path, already_accepted):
    from virtual_companion.text.memory_runtime import DEFAULT_COMPANION_ID, TextMemoryRuntime
    from virtual_companion.text.sqlite_store import SQLiteConversationStore

    store = SQLiteConversationStore(tmp_path / "conversation")
    runtime = TextMemoryRuntime(tmp_path / "memory.sqlite3", DEFAULT_COMPANION_ID)
    first = store.begin("hello", {})
    original = (runtime.accept_user(first["turn_id"], "hello", store.session_id)
                if already_accepted else None)
    store.close()  # Persisted running attempt without finalization, not a host-crash test.
    store = SQLiteConversationStore(tmp_path / "conversation")
    assert store.latest_attempt()["status"] == "interrupted"
    receipts = []

    class Prepared:
        def check_context(self):
            pass

        def stream(self, cancelled):
            yield "hello"

    def prepare(attempt, session_id, messages, cancelled):
        current = next(m for m in messages if m["turn_id"] == attempt["turn_id"])
        receipts.append(runtime.accept_user(attempt["turn_id"], current["content"], session_id))
        return Prepared()

    session = TextSession(ControlledModel(), store, {}, prepare_turn=prepare)
    second = session.retry()
    wait_done(session)
    assert second["turn_id"] == first["turn_id"]
    if original is not None:
        assert receipts == [original]
    assert len(runtime.store.background_sources(DEFAULT_COMPANION_ID)) == 1
    store.close()


def test_bound_request_is_same_for_count_and_send_and_never_uses_display_history(tmp_path):
    from virtual_companion.text.memory_runtime import (
        DEFAULT_COMPANION_ID,
        MemoryAwareModel,
        TextMemoryRuntime,
    )

    store = MemoryConversationStore()
    old = store.begin("old display-only source", {})
    store.finish(old["attempt_id"], "completed", "old ungrounded display response")
    memory = TextMemoryRuntime(tmp_path / "memory.db", DEFAULT_COMPANION_ID)
    bodies = []

    class Model:
        def check_context(self, messages):
            bodies.append(messages)

        def stream(self, messages, cancelled):
            bodies.append(messages)
            yield "current-only answer"

        def cancel(self):
            pass

    model = Model()
    wrapped = MemoryAwareModel(model, memory, memory.generation_gate)

    def prepare(attempt, session_id, messages, cancelled):
        current = next(m for m in messages if m["turn_id"] == attempt["turn_id"])
        memory.accept_user(attempt["turn_id"], current["content"], session_id)
        # This test is the no-memory-needed path, not a recall bypass.
        return wrapped.prepare_request([{"role": "user", "content": current["content"]}])

    session = TextSession(model, store, {}, prepare_turn=prepare)
    session.send("hello")
    wait_done(session)
    assert bodies == [[{"role": "user", "content": "hello"}]] * 2
    assert len(store.history()["messages"]) == 4
    assert memory.store.memory_barrier(DEFAULT_COMPANION_ID)


def test_bound_preparation_cannot_mix_with_legacy_preflight():
    with pytest.raises(ValueError, match="owns preflight"):
        TextSession(ControlledModel(), MemoryConversationStore(), {},
                    prepare_turn=lambda *args: None, preflight=lambda messages: None)


@pytest.mark.parametrize("retry", [False, True])
def test_bound_preparation_reads_source_history_only_after_persisted_start(monkeypatch, retry):
    store = MemoryConversationStore()
    previous = store.begin("hello", {}) if retry else None
    if previous is not None:
        store.finish(previous["attempt_id"], "failed", "")
    read_states, sources = [], []
    history = store.history

    def read_history():
        attempt = store.latest_attempt()
        read_states.append(None if attempt is None else attempt["status"])
        return history()

    class Prepared:
        def check_context(self):
            pass

        def stream(self, cancelled):
            yield "prepared answer"

    def prepare(attempt, session_id, records, cancelled):
        assert session_id == store.session_id
        sources.extend(records)
        assert [(row["turn_id"], row["content"]) for row in records] == [
            (attempt["turn_id"], "hello")]
        return Prepared()

    monkeypatch.setattr(store, "history", read_history)
    model = ControlledModel()
    session = TextSession(model, store, {}, prepare_turn=prepare)
    attempt = session.retry() if retry else session.send("hello")
    wait_done(session)
    assert read_states == ["running"]
    assert session.snapshot()["attempt"]["status"] == "completed"
    assert sources[0]["turn_id"] == attempt["turn_id"]
    if previous is not None:
        assert attempt["turn_id"] == previous["turn_id"]
    assert not model.requests
