"""Same-owner source work is optional, fenced, and invalidates old context."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event

import pytest
from test_memory_command_worker import command, seed
from test_rebuilt_turns import A, config
from virtual_companion.boundary.contracts import Scope, VersionConflict
from virtual_companion.boundary.memory_management import ExplicitMemoryCommand
from virtual_companion.boundary.sqlite_memory_management import SQLiteExplicitMemoryStore
from virtual_companion.cognition.background_worker import UserFactWorker
from virtual_companion.cognition.short_context import ShortContextAdmission
from virtual_companion.cognition.short_dialogue import ShortRequest
from virtual_companion.text.memory_runtime import _PriorityGenerationGate


def prepared(tmp_path, text="合成陈述"):
    store = SQLiteExplicitMemoryStore(tmp_path / "owner.sqlite3")
    item = store.accept(config(), "current", text)
    foreground = store.claim(A, item.input_id, "foreground", now_us=10, lease_us=100)
    return store, item, foreground


def no_effect_model(item, text="合成陈述"):
    def invoke(stage, messages, schema, cancelled):
        if stage in {"fact_extract", "exchange_extract"}:
            return {"operations": []}
        if stage == "exchange_intent":
            return {"acts": [{"segmentId": 0,
                              "evidenceQuote": text, "intent": "statement",
                              "targetId": None}], "coverage": "complete"}
        raise AssertionError(stage)
    return invoke


def worker(store, model, *, clock=lambda: 11):
    return UserFactWorker(store, model, clock=clock, resolution_protocol=True,
                          resolution_strategy="intent_first")


def delegated(store, foreground):
    return store.claim_background(A, "source-worker", now_us=11, lease_us=1000,
                                  role="user", source=foreground.input_id,
                                  foreground=foreground)


def test_same_owner_source_completes_and_requires_refreshed_claim(tmp_path):
    store, item, foreground = prepared(tmp_path)
    assert store.claim_background(A, "background", now_us=11, lease_us=50) is None
    old_snapshot = store.snapshot(A)
    assert worker(store, no_effect_model(item)).run_one(
        A, "source-worker", source=item.input_id, foreground=foreground,
    )
    assert not store.memory_barrier(A)
    assert store.snapshot(A).write_version > foreground.state_version
    admission = ShortContextAdmission(store, lambda *args: ShortRequest(args[-1]),
                                      evidence_clock=lambda: 12)
    with pytest.raises(VersionConflict):
        admission(foreground, old_snapshot, "合成陈述")
    fresh = store.refresh_source_owner(foreground, source=item.input_id, now_us=12)
    assert fresh.owner == foreground.owner and fresh.attempt_id == foreground.attempt_id
    assert fresh.state_version == store.snapshot(A).write_version
    admission(fresh, store.snapshot(A), "合成陈述")


def test_child_lease_is_capped_and_background_cannot_take_over(tmp_path):
    store, _, foreground = prepared(tmp_path)
    child = delegated(store, foreground)
    assert child.lease_until == foreground.lease_until_us
    store.background_claim_current(child, now_us=12)
    assert store.claim_background(A, "ordinary", now_us=12, lease_us=10) is None
    assert store.next_background_source(now_us=12) is None
    with pytest.raises(VersionConflict):
        store.background_claim_current(replace(child, foreground=None), now_us=12)


@pytest.mark.parametrize("fault", ["session", "input", "attempt", "owner", "basis"])
def test_delegation_rejects_wrong_parent_binding(tmp_path, fault):
    store, _, foreground = prepared(tmp_path)
    fields = {
        "session": {"session_id": "other"}, "input": {"input_id": "other"},
        "attempt": {"attempt_id": "other"},
        "owner": {"owner": replace(foreground.owner, epoch=foreground.owner.epoch + 1)},
        "basis": {"state_version": foreground.state_version + 1},
    }
    with pytest.raises(VersionConflict):
        delegated(store, replace(foreground, **fields[fault]))
    assert store.background_sources(A)[0]["status"] == "queued"


@pytest.mark.parametrize("fault", ["cancel", "expiry", "new_owner", "new_input"])
def test_lost_parent_or_preempted_generation_cannot_finish(tmp_path, fault):
    store, item, foreground = prepared(tmp_path)
    now = [11]
    invoked = []

    def invoke(stage, messages, schema, cancelled):
        invoked.append(stage)
        if fault == "cancel":
            store.cancel(A, item.input_id)
        elif fault == "expiry":
            now[0] = 111
        elif fault == "new_owner":
            store.claim(A, item.input_id, "replacement", now_us=111, lease_us=100)
        else:
            store.accept(config(), "later", "新合成输入", session_id=item.session_id)
        return {"operations": []}

    assert worker(store, invoke, clock=lambda: now[0]).run_one(
        A, "source-worker", source=item.input_id, foreground=foreground,
    ) is None
    assert invoked == ["fact_extract"]
    assert store.memory_barrier(A)
    assert all(row["status"] != "completed" for row in store.background_sources(A))
    with pytest.raises(VersionConflict):
        store.refresh_source_owner(foreground, source=item.input_id, now_us=now[0])


def test_external_cancel_requeues_source_without_semantic_failure(tmp_path):
    store, item, foreground = prepared(tmp_path)
    cancelled = Event()

    def invoke(*args):
        cancelled.set()
        return {"operations": []}

    assert worker(store, invoke).run_one(
        A, "source-worker", source=item.input_id, foreground=foreground, cancelled=cancelled,
    ) is None
    row = store.background_sources(A)[0]
    assert row["status"] == "queued" and row["attempts"] == 0
    assert store.memory_barrier(A)


def test_refresh_requires_this_source_commit_and_rejects_new_input(tmp_path):
    store, item, foreground = prepared(tmp_path)
    with pytest.raises(VersionConflict):
        store.refresh_source_owner(foreground, source=item.input_id, now_us=11)
    assert worker(store, no_effect_model(item)).run_one(
        A, "source-worker", source=item.input_id, foreground=foreground,
    )
    store.accept(config(), "later", "新的合成输入", session_id=item.session_id)
    with pytest.raises(VersionConflict):
        store.refresh_source_owner(foreground, source=item.input_id, now_us=12)


def test_explicit_memory_command_stays_with_foreground_owner(tmp_path):
    store = SQLiteExplicitMemoryStore(tmp_path / "owner.sqlite3")
    seed(store)
    source = "请忘记我喜欢安静音乐这件事。"
    item = store.accept(config(), "forget", source)
    foreground = store.claim(A, item.input_id, "foreground", now_us=10, lease_us=100)

    def invoke(stage, messages, schema, cancelled):
        if stage == "memory_command":
            return command(source)
        if stage == "memory_command_support":
            return {"meaning": "explicit_single_forget", "verdict": "supported"}
        raise AssertionError(stage)

    receipt = UserFactWorker(store, invoke, clock=lambda: 11, memory_commands=True).run_one(
        A, "source-worker", source=item.input_id, foreground=foreground,
    )
    assert receipt["status"] == "committed"
    fresh = store.refresh_source_owner(foreground, source=item.input_id, now_us=12)
    assert fresh.owner == foreground.owner
    assert fresh.state_version > foreground.state_version
    assert not store.memory_barrier(A)


def test_exchange_effect_uses_existing_owner_without_epoch_transfer(tmp_path):
    text = "能推荐电影吗？"
    store, item, foreground = prepared(tmp_path, text)

    def invoke(stage, messages, schema, cancelled):
        if stage == "fact_extract":
            return {"operations": []}
        if stage == "exchange_intent":
            return {"acts": [{"segmentId": 0,
                              "evidenceQuote": text, "intent": "question",
                              "targetId": None}], "coverage": "complete"}
        if stage == "exchange_extract":
            return {"operations": [{"operation": "add", "type": "response_obligation",
                                     "evidenceQuote": text,
                                    "supersedesAtomId": None, "expiresAt": None}]}
        if stage == "exchange_support":
            return {"intent": "question", "verdict": "supported"}
        raise AssertionError(stage)

    assert worker(store, invoke).run_one(
        A, "source-worker", source=item.input_id, foreground=foreground,
    )
    assert store.owner(A) == foreground.owner
    assert [row["content"] for row in store.pending_exchanges(A, now_us=12)] == [text]
    assert store.refresh_source_owner(
        foreground, source=item.input_id, now_us=12,
    ).owner == foreground.owner


def test_parent_can_renew_before_delegating_but_child_cannot_extend_it(tmp_path):
    store, _, foreground = prepared(tmp_path)
    renewed = store.renew(foreground, now_us=11, lease_us=200)
    child = delegated(store, renewed)
    assert child.lease_until == renewed.lease_until_us == 211
    store.background_claim_current(child, now_us=150)
    with pytest.raises(VersionConflict):
        store.background_claim_current(child, now_us=211)


def test_source_generation_remains_bound_under_same_owner(tmp_path):
    store, _, foreground = prepared(tmp_path)
    child = delegated(store, foreground)
    store.requeue_background(child, now_us=12)
    replacement = delegated(store, foreground)
    assert replacement.generation == child.generation + 1
    with pytest.raises(VersionConflict):
        store.background_claim_current(child, now_us=13)
    store.background_claim_current(replacement, now_us=13)


def test_delegation_runs_between_released_inference_gate_calls(tmp_path):
    store, item, foreground = prepared(tmp_path)
    gate = _PriorityGenerationGate()
    stages = []
    with gate.foreground():
        stages.append("foreground-before")
    model = no_effect_model(item)

    def invoke(stage, messages, schema, cancelled):
        with gate.background(lambda: None, cancelled):
            stages.append(stage)
            return model(stage, messages, schema, cancelled)

    assert worker(store, invoke).run_one(
        A, "source-worker", source=item.input_id, foreground=foreground,
    )
    fresh = store.refresh_source_owner(foreground, source=item.input_id, now_us=12)
    with gate.foreground():
        stages.append("foreground-after")
        assert fresh.owner == foreground.owner
    assert stages == ["foreground-before", "fact_extract", "exchange_intent",
                      "exchange_extract", "foreground-after"]


def test_concurrent_new_input_cancels_inflight_delegated_source(tmp_path):
    store, item, foreground = prepared(tmp_path)
    entered = Event()

    def invoke(stage, messages, schema, cancelled):
        entered.set()
        assert cancelled.wait(2), "Guardian must observe source preemption"
        return {"operations": []}

    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(worker(store, invoke).run_one, A, "source-worker",
                             source=item.input_id, foreground=foreground)
        assert entered.wait(2)
        store.accept(config(), "concurrent", "更正合成输入", session_id=item.session_id)
        assert result.result(timeout=3) is None
    assert store.memory_barrier(A)
    assert len(store.background_sources(A)) == 2


@pytest.mark.parametrize("fault", ["expiry", "cancel"])
def test_interruption_during_source_commit_rolls_back_completion_and_basis(
    tmp_path, monkeypatch, fault,
):
    store, item, foreground = prepared(tmp_path)
    now = [11]
    cancelled = Event()
    apply = store._apply_fact_commands

    def slow_apply(*args, **kwargs):
        result = apply(*args, **kwargs)
        if fault == "expiry":
            now[0] = 111
        else:
            cancelled.set()
        return result

    monkeypatch.setattr(store, "_apply_fact_commands", slow_apply)
    assert worker(store, no_effect_model(item), clock=lambda: now[0]).run_one(
        A, "source-worker", source=item.input_id, foreground=foreground, cancelled=cancelled,
    ) is None
    assert store.snapshot(A).write_version == foreground.state_version
    assert store.memory_barrier(A)


@pytest.mark.parametrize("forged_lease", [False, True])
@pytest.mark.parametrize("commit_end", [109, 111])
def test_parent_renewal_cannot_extend_or_forge_child_lease(
    tmp_path, monkeypatch, forged_lease, commit_end,
):
    store, item, foreground = prepared(tmp_path)
    now = [109]
    claim_background = store.claim_background
    apply = store._apply_fact_commands

    def claim_then_renew(*args, **kwargs):
        child = claim_background(*args, **kwargs)
        assert child.lease_until == 110
        assert store.renew(foreground, now_us=109, lease_us=100).lease_until_us == 209
        return replace(child, lease_until=1000) if forged_lease else child

    def finish_at_deadline(*args, **kwargs):
        result = apply(*args, **kwargs)
        now[0] = commit_end
        return result

    monkeypatch.setattr(store, "claim_background", claim_then_renew)
    monkeypatch.setattr(store, "_apply_fact_commands", finish_at_deadline)
    receipt = worker(store, no_effect_model(item), clock=lambda: now[0]).run_one(
        A, "source-worker", source=item.input_id, foreground=foreground,
    )
    if not forged_lease and commit_end == 109:
        assert receipt
        assert not store.memory_barrier(A)
        assert store.refresh_source_owner(
            foreground, source=item.input_id, now_us=109,
        ).lease_until_us == 209
    else:
        assert receipt is None
        assert store.memory_barrier(A)
        assert store.snapshot(A).write_version == foreground.state_version
        with pytest.raises(VersionConflict):
            store.refresh_source_owner(foreground, source=item.input_id, now_us=now[0])


@pytest.mark.parametrize("operation", ["requeue", "fail", "protect", "memory_command"])
def test_completed_claim_cannot_reopen_or_rewrite_source(tmp_path, monkeypatch, operation):
    store, item, foreground = prepared(tmp_path)
    claim_background = store.claim_background
    captured = []

    def record_claim(*args, **kwargs):
        child = claim_background(*args, **kwargs)
        captured.append(child)
        return child

    monkeypatch.setattr(store, "claim_background", record_claim)
    assert worker(store, no_effect_model(item)).run_one(
        A, "source-worker", source=item.input_id, foreground=foreground,
    )
    child = captured[0]
    receipt = store.memory_source_receipt(A, child.job)
    mutate = {
        "requeue": lambda: store.requeue_background(child, now_us=111),
        "fail": lambda: store.fail_background(child, now_us=111, error="synthetic"),
        "protect": lambda: store.protect_background(
            child, now_us=111, error="needs_clarification.source_coverage",
        ),
        "memory_command": lambda: store.execute_memory_command(
            ExplicitMemoryCommand(
                Scope(A, item.session_id), "late-command", "forget", "missing-record", 1,
                "conversation", child.source, child.source_hash, "single_record",
                "pure_command", "synthetic", "a" * 64,
                evidence_quote="合成陈述", quote_start=0, quote_end=4,
            ), now_us=111, claim=child,
        ),
    }
    with pytest.raises(VersionConflict):
        mutate[operation]()
    assert store.memory_source_receipt(A, child.job) == receipt
    assert not store.memory_barrier(A)
    assert store.claim_background(A, "ordinary", now_us=111, lease_us=100) is None
