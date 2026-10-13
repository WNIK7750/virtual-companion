"""Durably ineffective ACK sources cannot block later source maintenance."""

import json
from dataclasses import replace

import pytest
from test_effective_history import NOW, complete, emit, seeded
from test_explicit_memory_management import list_command
from test_rebuilt_turns import A, config
from virtual_companion.boundary.contracts import VersionConflict
from virtual_companion.cognition.background_worker import (
    AssistantExchangeWorker,
    BackgroundScheduler,
)
from virtual_companion.text.memory_runtime import _fake_memory_result


def invalidate(store, seed, change):
    if change == "correct":
        store.mutate(replace(seed, mutation_id="fix", kind="correct", content="new preference",
                             source_id="correction", expected_record_version=1, fact_key=None),
                     store.owner(A), expected_version=store.snapshot(A).write_version)
    elif change == "forget":
        store.execute_memory_command(list_command(), now_us=NOW + 20)


@pytest.mark.parametrize("change", ["correct", "forget", "expiry"])
def test_invalidated_head_is_audited_without_model_and_fresh_successor_runs(tmp_path, change):
    store, _, _, seed = seeded(tmp_path, expires=change == "expiry")
    old = next(row for row in store.background_sources(A) if row["role"] == "assistant")
    invalidate(store, seed, change)
    fresh = store.accept(config(), "fresh", "synthetic unrelated topic")
    complete(store, fresh)
    emit(store, fresh, "An unrelated acknowledged answer.")
    calls = []

    def model(*args):
        calls.append(args)
        return _fake_memory_result(*args)

    scheduler = BackgroundScheduler(store, user_invoke=model, assistant_invoke=model,
                                    clock=lambda: NOW + 2_000_000,
                                    resolution_protocol=True, resolution_strategy="intent_first")
    assert scheduler.drain("worker") == 2
    assert calls
    receipt = store.memory_source_receipt(A, old["job"])
    assert receipt["disposition"] == "source_invalidated"
    assert "resolution" not in receipt
    assert store.next_background_source(now_us=NOW + 2_000_000) is None
    assert next(row for row in store.background_sources(A) if row["job"] == old["job"])[
        "status"] == "invalidated"


def test_pending_user_barrier_is_not_permanent_invalidation(tmp_path):
    store, _, _, _ = seeded(tmp_path)
    pending = store.accept(config(), "pending", "unprocessed new input")
    old = next(row for row in store.background_sources(A) if row["role"] == "assistant")
    worker = AssistantExchangeWorker(store, _fake_memory_result, clock=lambda: NOW + 20,
                                     resolution_protocol=True, resolution_strategy="intent_first")
    # This terminal ACK has an effective readset. Unrelated unprocessed user
    # input is queue state, not permanent invalidation of its actual source.
    assert store.memory_barrier(A)
    assert old["source"] not in store.ineffective_history_sources(A, now_us=NOW + 20)
    receipt = worker.run_one(A, "worker")
    assert receipt is not None
    persisted = store.memory_source_receipt(A, old["job"])
    assert persisted == json.loads(receipt)
    assert persisted.get("disposition") != "source_invalidated"
    assert "resolution" in persisted
    with store.transaction(write=False) as db:
        assert db.execute("SELECT status FROM background_sources WHERE companion=? AND job=?",
                          (A, old["job"])).fetchone()[0] == "completed"
    assert next(row for row in store.background_sources(A) if row["source"] == pending.input_id)[
        "status"] == "queued"
    assert store.memory_barrier(A)  # Processing an ACK does not resolve the user.
    complete(store, pending)
    store.cancel(A, pending.input_id)
    assert worker.run_one(A, "worker") is None  # The actual ACK source is not processed twice.


def test_scheduler_resolves_queued_user_barrier_before_older_assistant(tmp_path):
    store, _, _, _ = seeded(tmp_path)
    pending = store.accept(config(), "pending", "unprocessed new input")
    calls = []

    def model(*args):
        calls.append(args)
        return _fake_memory_result(*args)

    scheduler = BackgroundScheduler(store, user_invoke=model, assistant_invoke=model,
                                    clock=lambda: NOW + 20,
                                    resolution_protocol=True, resolution_strategy="intent_first")
    assert store.next_background_source(now_us=NOW + 20)["role"] == "user"
    assert scheduler.drain("worker") == 2
    assert calls and not store.memory_barrier(A)
    assert store.background_sources(A) == []
    assert store.inspect(A, pending.input_id)["text"] == "unprocessed new input"


@pytest.mark.parametrize("protection", ["clarification", "retry_delay"])
def test_unresolved_user_barrier_does_not_dispatch_old_assistant(tmp_path, protection):
    store, _, _, _ = seeded(tmp_path)
    store.accept(config(), "pending", "unprocessed new input")
    child = store.claim_background(A, "worker", role="user", now_us=NOW + 20, lease_us=100)
    if protection == "clarification":
        store.protect_background(
            child, error="needs_clarification.source_coverage", now_us=NOW + 21,
        )
    else:
        store.fail_background(child, error="model_error", now_us=NOW + 21)
    assert store.next_background_source(now_us=NOW + 22) is None
    assert store.memory_barrier(A)
    assert next(row for row in store.background_sources(A) if row["role"] == "assistant")[
        "status"] == "queued"
    if protection == "retry_delay":
        assert store.next_background_source(now_us=NOW + 1_000_022)["role"] == "user"


def test_stale_claim_cannot_dispose_source_after_new_input(tmp_path):
    store, _, _, seed = seeded(tmp_path)
    invalidate(store, seed, "correct")
    child = store.claim_background(A, "worker", role="assistant", now_us=NOW + 20,
                                   lease_us=100)
    store.accept(config(), "new", "concurrent new input")
    with pytest.raises(VersionConflict):
        store.invalidate_background_source(child, now_us=NOW + 21)
    assert store.memory_source_receipt(A, child.job) is None


@pytest.mark.parametrize("terminal", [True, False])
def test_only_terminal_invalidated_assistant_sources_release_queue_capacity(tmp_path, terminal):
    store, _, _, seed = seeded(tmp_path)
    invalidate(store, seed, "correct")
    child = store.claim_background(A, "worker", role="assistant", now_us=NOW + 20,
                                   lease_us=100)
    receipt = store.invalidate_background_source(child, now_us=NOW + 21)
    assert receipt
    # Capacity is a storage count, so replicate synthetic terminal audit rows;
    # real source disposal itself is exercised above, not assumed by the fixture.
    with store.transaction() as db:
        for index in range(store.max_pending_sources + 1):
            job = f"audit-{index}"
            db.execute("INSERT INTO background_sources VALUES(?,?,?,?,?,?,?,?,?)",
                       (A, child.session, job, "assistant", job, "a" * 64, 1, "failed", 0))
            db.execute("INSERT INTO background_work VALUES(?,?,?,?,?,?,?,?)",
                       (A, job, "worker", None, 1, 0,
                        "source_invalidated" if terminal else "model_error", receipt))
    if terminal:
        accepted = store.accept(config(), "next", "new accepted input")
        assert any(row["source"] == accepted.input_id for row in store.background_sources(A))
    else:
        with pytest.raises(ValueError, match="backpressure"):
            store.accept(config(), "next", "must remain blocked")


def test_source_expiring_during_inference_is_invalidated_without_semantic_completion(tmp_path):
    store, _, _, _ = seeded(tmp_path, expires=True)
    now = [NOW + 20]
    calls = []

    def model(*args):
        calls.append(args)
        result = _fake_memory_result(*args)
        now[0] = NOW + 2_000_000
        return result

    worker = AssistantExchangeWorker(store, model, clock=lambda: now[0],
                                     resolution_protocol=True, resolution_strategy="intent_first")
    receipt = worker.run_one(A, "worker")
    assert calls and '"disposition":"source_invalidated"' in receipt
    assert store.pending_exchanges(A, now_us=now[0]) == []


def test_unrelated_version_conflict_still_requeues_valid_source(tmp_path, monkeypatch):
    store, _, _, _ = seeded(tmp_path)

    def conflict(*args, **kwargs):
        raise VersionConflict("synthetic concurrent context change")

    monkeypatch.setattr(store, "background_exchange_context", conflict)
    worker = AssistantExchangeWorker(store, _fake_memory_result, clock=lambda: NOW + 20)
    assert worker.run_one(A, "worker") is None
    assert next(row for row in store.background_sources(A) if row["role"] == "assistant")[
        "status"] == "queued"


def test_child_lease_expiring_during_invalidation_rolls_back(tmp_path, monkeypatch):
    store, _, _, seed = seeded(tmp_path)
    invalidate(store, seed, "correct")
    child = store.claim_background(A, "worker", role="assistant", now_us=NOW + 20,
                                   lease_us=10)
    now = [NOW + 21]
    original = store._background_source_invalidated

    def delayed(*args, **kwargs):
        result = original(*args, **kwargs)
        now[0] = NOW + 31
        return result

    monkeypatch.setattr(store, "_background_source_invalidated", delayed)
    with pytest.raises(VersionConflict):
        store.invalidate_background_source(child, now_us=now[0], clock=lambda: now[0])
    assert store.memory_source_receipt(A, child.job) is None


def test_ordinary_failure_cannot_claim_terminal_invalidation(tmp_path):
    store, _, _, _ = seeded(tmp_path)
    child = store.claim_background(A, "worker", role="assistant", now_us=NOW + 20,
                                   lease_us=100)
    with pytest.raises(ValueError, match="[Ii]nvalidation"):
        store.fail_background(child, error="source_invalidated", now_us=NOW + 21)
    assert store.memory_source_receipt(A, child.job) is None
