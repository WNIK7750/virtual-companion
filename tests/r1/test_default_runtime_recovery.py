"""Independent default-entry crash, cancellation and durable ACK recovery checks."""

import threading

import pytest
from test_default_text_delivery import setup, wait_for
from virtual_companion.text.application import TextSession
from virtual_companion.text.default_runtime import DefaultTextRuntime
from virtual_companion.text.memory_runtime import DEFAULT_COMPANION_ID
from virtual_companion.text.sqlite_store import SQLiteConversationStore


def configured(tmp_path):
    runtime, session, memory, mirror, model = setup(tmp_path)
    session.before_start = runtime.before_start
    return runtime, session, memory, mirror, model


def test_listening_terminal_reopens_without_empty_message_or_retry_and_accepts_next_input(tmp_path):
    from test_default_demand_integration import TypedTurnModel, demand_setup

    runtime, session, memory, mirror, model, source, _ = demand_setup(
        tmp_path, model=TypedTurnModel("listen"), source="我还没说完，你先听。", seed_memory=False)
    try:
        first = session.send(source)
        wait_for(lambda: not session.snapshot()["busy"])
        original = runtime.claim.input_id
        assert session.snapshot()["attempt"]["status"] == "listened"
        assert memory.scheduler.user.run_one(memory.companion, "synthetic-user", source=original)
        target = memory.store.pending_exchanges(memory.companion, now_us=runtime.clock())[0]
        session_id = mirror.session_id
        mirror.close()
        mirror = SQLiteConversationStore(tmp_path / "chat", companion_id=memory.companion)
        runtime = DefaultTextRuntime(memory, mirror, model, binding={"test": "synthetic"},
                                     delivery_timeout=0.8)
        session = TextSession(model, mirror, {}, prepare_turn=runtime.prepare_turn,
                              before_start=runtime.before_start)
        assert mirror.session_id == session_id
        assert mirror.latest_attempt()["status"] == "listened"
        assert not runtime.snapshot()["retry_allowed"] and runtime.state()["reconcile"] == []
        with pytest.raises(ValueError, match="retryable"):
            session.retry()
        assert mirror.history()["messages"][0]["turn_id"] == first["turn_id"]
        model.response_type = "answer"
        session.send("补充好了，现在可以回应。")
        event = completed(runtime, session)
        assert [m["role"] for m in mirror.history()["messages"]] == ["user", "user", "assistant"]
        assert event["basis"]["input_id"] != original
        assert memory.store.exchange_history(memory.companion, target["exchange_id"])[-1][
            "status"] == "pending"
        assert memory.store.stream_state(memory.companion, original)["outcome"] == "listened"
    finally:
        memory.close()
        mirror.close()


def test_reopen_restores_listening_when_canonical_terminal_precedes_mirror_finish(tmp_path):
    from test_default_demand_integration import TypedTurnModel

    runtime, _, memory, mirror, model = setup(tmp_path, TypedTurnModel("listen"))
    try:
        attempt = mirror.begin("我还有补充，先听。", {})
        cancelled = threading.Event()
        prepared = runtime.prepare_turn(attempt, mirror.session_id, mirror.history()["messages"],
                                        cancelled)
        prepared.check_context()
        assert list(prepared.stream(cancelled)) == []
        assert prepared.result.response_type == "listen"
        assert mirror.latest_attempt()["status"] == "running"
        original = runtime.claim.input_id
        mirror.close()
        mirror = SQLiteConversationStore(tmp_path / "chat", companion_id=memory.companion)
        assert mirror.latest_attempt()["status"] == "interrupted"
        reopened = DefaultTextRuntime(memory, mirror, model, binding={"test": "synthetic"})
        assert mirror.latest_attempt()["status"] == "listened"
        assert len(mirror.history()["messages"]) == 1 and reopened.state()["reconcile"] == []
        assert memory.store.stream_state(memory.companion, original)["settlement"][
            "response_type"] == "listen"
        assert len(model.calls) == 1
    finally:
        memory.close()
        mirror.close()


def test_prepare_turn_returns_only_explicit_orchestrator_capabilities(tmp_path):
    from virtual_companion.text.turn_orchestrator import TurnOrchestrator

    runtime, _, memory, mirror, model = configured(tmp_path)
    try:
        attempt = mirror.begin("合成接口输入", {})
        cancelled = threading.Event()
        prepared = runtime.prepare_turn(attempt, mirror.session_id,
                                        mirror.history()["messages"], cancelled)
        assert isinstance(prepared, TurnOrchestrator)
        assert not any(hasattr(prepared, name) for name in ("runtime", "store", "memory", "model"))
        for capability in ("prepare", "resolve_current", "refresh", "check",
                           "has_approved_prefix", "deliver", "settle_failure"):
            assert callable(getattr(prepared, capability))
        assert prepared.prepare.__self__ is runtime
        assert prepared.deliver.__self__ is runtime
        assert prepared.claim is runtime.claim and prepared.cancelled is cancelled
        cancelled.set()
        prepared.close()
        assert memory.store.stream_state(memory.companion, runtime.claim.input_id)[
            "outcome"] == "cancelled"
        assert not model.calls
        assert memory.store.segments(memory.companion, runtime.claim.input_id) == []
    finally:
        memory.close()
        mirror.close()


def ack(runtime, event, receipt="synthetic-visible"):
    return runtime.ack({"sequence": event["sequence"], "fingerprint": event["fingerprint"],
                        "host_receipt": receipt})


def completed(runtime, session):
    event = wait_for(lambda: runtime.poll()["event"])
    ack(runtime, event)
    wait_for(lambda: not session.snapshot()["busy"])
    assert session.snapshot()["attempt"]["status"] == "completed"
    return event


def test_cancelled_preflight_settles_original_claim_and_can_retry_same_input(tmp_path):
    runtime, session, memory, mirror, model = configured(tmp_path)
    attempt = mirror.begin("合成输入", {})
    cancelled = threading.Event()
    prepared = runtime.prepare_turn(attempt, mirror.session_id, mirror.history()["messages"],
                                    cancelled)
    input_id = runtime.claim.input_id
    cancelled.set()
    with pytest.raises(InterruptedError):
        prepared.check_context()
    assert memory.store.stream_state(DEFAULT_COMPANION_ID, input_id)["outcome"] == "cancelled"
    mirror.finish(attempt["attempt_id"], "cancelled", "")
    session.retry()
    event = completed(runtime, session)
    assert event["basis"]["input_id"] == input_id
    assert len(model.calls) == 2
    mirror.close()


def test_reopened_zero_unit_owner_keeps_retry_reachable_before_new_input(tmp_path):
    runtime, _, memory, mirror, model = configured(tmp_path)
    original = mirror.begin("合成旧输入", {})
    runtime.prepare_turn(original, mirror.session_id, mirror.history()["messages"],
                         threading.Event())
    old_claim = runtime.claim
    mirror.close()
    mirror = SQLiteConversationStore(tmp_path / "chat", companion_id=DEFAULT_COMPANION_ID)
    runtime = DefaultTextRuntime(memory, mirror, model, binding={"test": "synthetic"},
                                 delivery_timeout=0.8)
    session = TextSession(model, mirror, {}, before_start=runtime.before_start,
                          prepare_turn=runtime.prepare_turn)
    with pytest.raises(ValueError):
        session.send("不能盖住旧输入")
    assert mirror.latest_attempt()["turn_id"] == original["turn_id"]
    assert len(mirror.history()["messages"]) == 1 and not model.calls
    session.retry()
    event = completed(runtime, session)
    assert event["basis"]["input_id"] == old_claim.input_id
    assert event["basis"]["attempt_id"] != old_claim.attempt_id
    assert event["basis"]["owner"]["epoch"] > old_claim.owner.epoch
    mirror.close()


def test_legacy_memory_only_queue_is_explicitly_blocked_without_new_input_or_fake_ack(tmp_path):
    runtime, session, memory, mirror, model = configured(tmp_path)
    original = mirror.begin("合成旧记忆输入", {})
    mirror.finish(original["attempt_id"], "completed", "旧兼容回复")
    intake = memory.accept_user(original["turn_id"], "合成旧记忆输入", mirror.session_id)
    with pytest.raises(ValueError):
        session.send("尚未迁移的新输入")
    assert len(mirror.history()["messages"]) == 2
    assert mirror.latest_attempt()["attempt_id"] == original["attempt_id"]
    assert memory.store.inspect(DEFAULT_COMPANION_ID, intake.input_id)["run"] is None
    assert memory.store.segments(DEFAULT_COMPANION_ID, intake.input_id) == []
    assert not model.calls
    mirror.close()


def test_cancel_after_accept_before_claim_is_recoverable_without_duplicate_source(
    tmp_path, monkeypatch,
):
    runtime, session, memory, mirror, _ = configured(tmp_path)
    original_accept = memory.store.accept
    admitted = []

    def accept_then_cancel(*args, **kwargs):
        receipt = original_accept(*args, **kwargs)
        admitted.append(receipt)
        session.cancelled.set()
        return receipt

    monkeypatch.setattr(memory.store, "accept", accept_then_cancel)
    first = session.send("合成输入")
    wait_for(lambda: not session.snapshot()["busy"])
    assert session.snapshot()["attempt"]["status"] == "cancelled"
    monkeypatch.setattr(memory.store, "accept", original_accept)
    session.retry()
    event = completed(runtime, session)
    assert event["basis"]["input_id"] == admitted[0].input_id
    assert session.snapshot()["attempt"]["turn_id"] == first["turn_id"]
    with memory.store.transaction(write=False) as db:
        assert db.execute("SELECT count(*) FROM turn_inputs").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM background_sources WHERE role='user'") \
            .fetchone()[0] == 1
    mirror.close()


def test_cancel_between_prepare_return_and_preflight_still_closes_owner(tmp_path):
    runtime, session, memory, mirror, _ = configured(tmp_path)
    original = runtime.prepare_turn

    def cancelled_prepared(*args):
        prepared = original(*args)
        args[-1].set()
        return prepared

    session.prepare_turn = cancelled_prepared
    session.send("合成输入")
    wait_for(lambda: not session.snapshot()["busy"])
    state = memory.store.stream_state(DEFAULT_COMPANION_ID, runtime.claim.input_id)
    assert state["outcome"] == "cancelled"
    assert state["approved_units"] == 0
    mirror.close()


def test_close_after_canonical_completion_never_rewrites_success_as_failure(tmp_path):
    runtime, _, memory, mirror, _ = configured(tmp_path)
    attempt = mirror.begin("合成输入", {})
    cancelled = threading.Event()
    prepared = runtime.prepare_turn(attempt, mirror.session_id, mirror.history()["messages"],
                                    cancelled)
    prepared.check_context()
    errors = []

    def acknowledge():
        try:
            ack(runtime, wait_for(lambda: runtime.poll()["event"]))
        except Exception as error:
            errors.append(error)

    host = threading.Thread(target=acknowledge)
    host.start()
    stream = prepared.stream(cancelled)
    assert next(stream) == "这是合成答案。"
    cancelled.set()
    stream.close()
    host.join(2)
    assert not host.is_alive() and not errors
    state = memory.store.stream_state(DEFAULT_COMPANION_ID, runtime.claim.input_id)
    assert state["outcome"] == "completed" and state["confirmed_units"] == 1
    mirror.close()


def test_unknown_delivery_rejects_new_send_and_retry_before_mirror_write(tmp_path):
    runtime, session, _, mirror, model = configured(tmp_path)
    first = session.send("合成输入")
    event = wait_for(lambda: runtime.poll()["event"])
    wait_for(lambda: not session.snapshot()["busy"])
    attempts = mirror.turn_records()
    for operation in (lambda: session.send("不能跳过未知显示"), session.retry):
        with pytest.raises(ValueError):
            operation()
    assert mirror.turn_records() == attempts
    assert len(model.calls) == 2
    ack(runtime, event)
    assert mirror.latest_attempt()["turn_id"] == first["turn_id"]
    assert mirror.latest_attempt()["status"] == "failed"
    assert mirror.history()["messages"][-1]["content"] == "这是合成答案。"
    mirror.close()


def test_polling_state_uses_one_transaction_without_per_turn_reads(tmp_path, monkeypatch):
    runtime, _, memory, mirror, _ = configured(tmp_path)
    count = []
    original = memory.store.transaction

    def counted(*args, **kwargs):
        count.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(memory.store, "transaction", counted)
    assert runtime.state() == {"reconcile": []}
    assert len(count) == 1
    mirror.close()
