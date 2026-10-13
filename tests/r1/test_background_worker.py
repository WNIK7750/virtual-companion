import json
from hashlib import sha256
from threading import Event

import pytest
from test_evidence_memory import operation
from test_rebuilt_turns import A, B, config
from virtual_companion.boundary.contracts import (
    Scope,
    SemanticConflict,
    SourceResolutionUnresolved,
    VersionConflict,
)
from virtual_companion.boundary.evidence import EvidenceTurn
from virtual_companion.boundary.sqlite_background_worker import SQLiteBackgroundWorkerStore
from virtual_companion.cognition.background_worker import BackgroundScheduler, UserFactWorker
from virtual_companion.cognition.fact_model import FactModel
from virtual_companion.cognition.fact_target_selector import FactTargetSelector


def test_worker_restarts_and_stale_generation_cannot_finish(tmp_path):
    store = SQLiteBackgroundWorkerStore(tmp_path / "work.sqlite3")
    store.accept(config(), "u1", "我喜欢茶")
    first = store.claim_background(A, "w1", now_us=10, lease_us=20)
    assert first is not None
    store.accept(config(), "u2", "更正，我不喜欢茶")
    with pytest.raises(VersionConflict):
        store.finish_background(first, (), evaluator="fact-model.v2", now_us=11)
    second = SQLiteBackgroundWorkerStore(store.path).claim_background(
        A, "w2", now_us=12, lease_us=20
    )
    assert second.job == first.job and second.generation == first.generation + 1
    assert store.finish_background(second, (), evaluator="fact-model.v2", now_us=13)
    assert store.memory_barrier(A)


def test_expired_lease_is_fenced_and_completion_is_idempotent(tmp_path):
    store = SQLiteBackgroundWorkerStore(tmp_path / "work.sqlite3")
    store.accept(config(), "u1", "今天下雨")
    first = store.claim_background(A, "w1", now_us=10, lease_us=20)
    second = store.claim_background(A, "w2", now_us=31, lease_us=20)
    assert second.job == first.job and second.generation > first.generation
    with pytest.raises(VersionConflict):
        store.finish_background(first, (), evaluator="fact-model.v2", now_us=32)
    receipt = store.finish_background(second, (), evaluator="fact-model.v2", now_us=32)
    assert receipt == store.finish_background(second, (), evaluator="fact-model.v2", now_us=33)
    assert not store.memory_barrier(A)


def test_failure_retains_barrier_and_bounded_retry(tmp_path):
    store = SQLiteBackgroundWorkerStore(tmp_path / "work.sqlite3")
    store.accept(config(), "u1", "请忘了旧事")
    claim = store.claim_background(A, "w1", now_us=10, lease_us=20)
    store.fail_background(claim, error="model_invalid", now_us=11)
    assert store.memory_barrier(A)
    assert store.claim_background(A, "w2", now_us=11, lease_us=20) is None
    assert store.claim_background(A, "w2", now_us=1_000_012, lease_us=20)


def test_foreground_preemption_does_not_consume_semantic_failure_budget(tmp_path):
    store = SQLiteBackgroundWorkerStore(tmp_path / "work.sqlite3")
    store.accept(config(), "u1", "我喜欢茶")
    claim = store.claim_background(A, "w1", now_us=10, lease_us=100)
    for index in (2, 3):
        store.accept(config(), f"u{index}", f"第{index}条新消息")
        claim = store.claim_background(A, "w1", now_us=10, lease_us=100)
        assert claim is not None
    store.fail_background(claim, error="model_invalid", now_us=11)
    assert store.background_sources(A)[0]["status"] == "retryable"
    assert store.memory_barrier(A)


def test_cooperative_yield_does_not_consume_retry_budget(tmp_path):
    store = SQLiteBackgroundWorkerStore(tmp_path / "work.sqlite3")
    store.accept(config(), "u1", "我喜欢茶")
    for _ in range(3):
        claim = store.claim_background(A, "w1", now_us=10, lease_us=100)
        assert claim is not None
        store.requeue_background(claim, now_us=11)
    claim = store.claim_background(A, "w1", now_us=12, lease_us=100)
    store.fail_background(claim, error="model_invalid", now_us=13)
    assert store.background_sources(A)[0]["status"] == "retryable"


@pytest.mark.parametrize(
    ("failure", "code"),
    [
        (SemanticConflict("Fact batch conflicts with current target"),
         "needs_clarification.fact_target_conflict"),
        (SourceResolutionUnresolved("Whole source coverage remains unresolved"),
         "needs_clarification.source_coverage"),
    ],
)
def test_worker_protects_semantic_source_failures_without_automatic_requeue(
        tmp_path, monkeypatch, failure, code):
    store = SQLiteBackgroundWorkerStore(tmp_path / "work.sqlite3")
    source = "更正，我说的是另一条记录"
    store.accept(config(), "u1", source)

    def model(stage, messages, schema, cancelled):
        if isinstance(failure, SourceResolutionUnresolved):
            raise failure
        return {"operations": []}

    if isinstance(failure, SemanticConflict):
        monkeypatch.setattr(store, "finish_background",
                            lambda *args, **kwargs: (_ for _ in ()).throw(failure))

    worker = UserFactWorker(store, model, clock=lambda: 10)
    if isinstance(failure, SourceResolutionUnresolved):
        with pytest.raises(SourceResolutionUnresolved):
            worker.run_one(A, "w1")
    else:
        assert worker.run_one(A, "w1") is None
    job = store.background_sources(A)[0]
    assert job["status"] == "failed"
    assert job["attempts"] == 1
    assert job["error"] == code
    assert source not in job["error"]
    assert store.memory_status(A) == "needs_clarification"
    assert store.memory_barrier(A)

    reopened = SQLiteBackgroundWorkerStore(store.path)
    persisted = reopened.background_sources(A)[0]
    assert persisted["attempts"] == 1 and persisted["error"] == code
    assert reopened.claim_background(A, "w2", now_us=10**9, lease_us=20) is None
    assert reopened.memory_status(A) == "needs_clarification"


def test_true_version_race_still_requeues_without_consuming_attempt(tmp_path, monkeypatch):
    store = SQLiteBackgroundWorkerStore(tmp_path / "work.sqlite3")
    store.accept(config(), "u1", "我喜欢茶")

    def model(stage, messages, schema, cancelled):
        return {"operations": []}

    monkeypatch.setattr(
        store, "finish_background",
        lambda *args, **kwargs: (_ for _ in ()).throw(VersionConflict("stale lease")),
    )
    assert UserFactWorker(store, model, clock=lambda: 10).run_one(A, "w1") is None
    job = store.background_sources(A)[0]
    assert job["status"] == "queued" and job["attempts"] == 0
    assert job["error"] is None
    assert store.memory_status(A) == "pending_source"


def test_source_hash_is_bound_to_claim(tmp_path):
    store = SQLiteBackgroundWorkerStore(tmp_path / "work.sqlite3")
    store.accept(config(), "u1", "你好")
    claim = store.claim_background(A, "w1", now_us=10, lease_us=20)
    assert claim.source_hash == sha256("你好".encode()).hexdigest()
    with store.transaction() as db:
        db.execute("UPDATE turn_inputs SET text='篡改' WHERE companion=?", (A,))
    with pytest.raises(VersionConflict):
        store.finish_background(claim, (), evaluator="fact-model.v2", now_us=11)
    assert store.memory_barrier(A)


def test_supported_fact_and_completion_are_one_transaction(tmp_path):
    store = SQLiteBackgroundWorkerStore(tmp_path / "work.sqlite3")
    item = store.accept(config(), "u1", "我喜欢茶")
    claim = store.claim_background(A, "w1", now_us=10, lease_us=100)

    def model(stage, messages, schema, cancelled):
        if stage == "fact_extract":
            return {"operations": [{**operation(content="我喜欢茶",
                                                  evidenceTurnId=item.input_id),
                                    "factKey": {"subject": "user", "predicate": "drink"}}]}
        return {"intent": "assertion", "verdict": "supported"}

    commands = FactModel(model, Event()).extract(
        scope=Scope(A, item.session_id), batch_id=item.input_id, now_us=10,
        turn=EvidenceTurn(Scope(A, item.session_id), item.input_id, "user", "我喜欢茶", 4),
        targets={},
    )
    receipt = store.finish_background(claim, commands, evaluator="fact-model.v2", now_us=11)
    assert store.finish_background(claim, commands, evaluator="fact-model.v2", now_us=12) == receipt
    assert [row["content"] for row in store.recall(A, now_us=12)] == ["我喜欢茶"]
    assert not store.memory_barrier(A)
    store.accept(config(), "u2", "更正，我不喜欢茶")
    assert store.recall(A, now_us=12) == []
    assert SQLiteBackgroundWorkerStore(store.path).recall(A, now_us=12) == []


def test_invalid_fact_cannot_clear_barrier_or_write_partial_batch(tmp_path):
    from datetime import UTC, datetime

    from virtual_companion.boundary.memory import FactKey, MemoryMutation

    store = SQLiteBackgroundWorkerStore(tmp_path / "work.sqlite3")
    item = store.accept(config(), "u1", "我不喜欢茶")
    claim = store.claim_background(A, "w1", now_us=10, lease_us=100)
    false_fact = MemoryMutation(Scope(A, item.session_id), "bad", "assert_fact", "bad-record",
                                "我喜欢茶", "input", item.input_id, "unsupported",
                                datetime.fromtimestamp(0, UTC), FactKey("user", "drink", "profile"))
    with pytest.raises(ValueError):
        store.finish_background(claim, (false_fact,), evaluator="fact-model.v2", now_us=11)
    assert store.memory_barrier(A)
    assert store.recall(A, now_us=11) == []


def test_worker_persists_the_failing_model_stage_without_source_text(tmp_path):
    store = SQLiteBackgroundWorkerStore(tmp_path / "work.sqlite3")
    text = "更正，我不喜欢茶"
    item = store.accept(config(), "u1", text)
    stages = []

    def model(stage, messages, schema, cancelled):
        stages.append(stage)
        if stage == "fact_extract":
            return {"operations": [{**operation(content=text, evidenceQuote=text,
                                                  evidenceTurnId=item.input_id),
                                    "factKey": {"subject": "user", "predicate": "drink"}}]}
        return {"intent": "correction", "verdict": "supported"}

    with pytest.raises(ValueError, match="Correction cannot be added"):
        UserFactWorker(store, model, clock=lambda: 10).run_one(A, "w1")
    with store.transaction(write=False) as db:
        row = db.execute("SELECT error FROM background_work WHERE companion=?", (A,)).fetchone()
        error = row[0]
    assert error == "semantic_processing_failed.fact_support"
    assert text not in error
    assert stages == ["fact_extract", "fact_support"]
    assert store.background_sources(A)[0]["status"] == "retryable"
    assert store.memory_barrier(A)
    assert store.recall(A, now_us=10) == []


def test_worker_uses_persisted_user_source_and_empty_semantic_result(tmp_path):
    store = SQLiteBackgroundWorkerStore(tmp_path / "work.sqlite3")
    store.accept(config(), "u1", "你好？")
    seen = []

    def model(stage, messages, schema, cancelled):
        data = json.loads(messages[-1]["content"])
        seen.append(data["source"] if stage == "fact_extract" else data["sources"][-1])
        return {"operations": []}

    worker = UserFactWorker(store, model, clock=lambda: 10)
    receipt = worker.run_one(A, "w1")
    assert receipt and seen[0]["text"] == "你好？"
    assert not store.memory_barrier(A)
    assert worker.run_one(A, "w1") is None


def test_user_request_creates_assistant_owes_from_persisted_source(tmp_path):
    store = SQLiteBackgroundWorkerStore(tmp_path / "work.sqlite3")
    store.accept(config(), "u1", "请给我一个建议")

    def model(stage, messages, schema, cancelled):
        data = json.loads(messages[-1]["content"])
        if stage == "fact_extract":
            return {"operations": []}
        if stage == "exchange_extract":
            source = data["sources"][-1]
            return {"operations": [{"operation": "add", "type": "response_obligation",
                                    "evidenceTurnId": source["id"],
                                    "evidenceQuote": source["text"],
                                    "supersedesAtomId": None, "expiresAt": None}]}
        return {"intent": "request", "verdict": "supported"}

    assert UserFactWorker(store, model, clock=lambda: 10).run_one(A, "w1")
    pending = store.pending_exchanges(A, now_us=11)
    assert len(pending) == 1 and pending[0]["direction"] == "assistant_response"


def test_scheduler_drains_ten_inputs_across_two_companions_in_source_order(tmp_path):
    from dataclasses import replace

    store = SQLiteBackgroundWorkerStore(tmp_path / "work.sqlite3")
    expected = []
    for index in range(10):
        who = A if index % 2 == 0 else B
        item = store.accept(replace(config(), companion_id=who), f"u{index}",
                            f"第{index}条")
        expected.append(item.input_id)
    seen = []

    def model(stage, messages, schema, cancelled):
        if stage == "fact_extract":
            seen.append(json.loads(messages[-1]["content"])["source"]["id"])
        return {"operations": []}

    scheduler = BackgroundScheduler(store, user_invoke=model, assistant_invoke=model,
                                    clock=lambda: 10)
    assert scheduler.drain("maintenance") == 10
    assert seen == expected
    assert not store.memory_barrier(A) and not store.memory_barrier(B)


def test_background_yields_when_foreground_claim_arrives_during_inference(tmp_path):
    store = SQLiteBackgroundWorkerStore(tmp_path / "work.sqlite3")
    item = store.accept(config(), "u1", "请给我一个建议")

    def model(stage, messages, schema, cancelled):
        if stage == "fact_extract":
            return {"operations": []}
        if stage == "exchange_extract":
            return {"operations": [{"operation": "add", "type": "response_obligation",
                                    "evidenceTurnId": item.input_id,
                                    "evidenceQuote": "请给我一个建议",
                                    "supersedesAtomId": None, "expiresAt": None}]}
        store.claim(A, item.input_id, "foreground", now_us=11, lease_us=100)
        return {"intent": "request", "verdict": "supported"}

    assert UserFactWorker(store, model, clock=lambda: 10).run_one(A, "background") is None
    jobs = store.background_sources(A)
    assert len(jobs) == 1 and jobs[0]["status"] == "queued"
    assert store.memory_barrier(A)
    assert store.pending_exchanges(A, now_us=12) == []


def test_new_input_interrupts_inflight_background_model_call(tmp_path):
    from threading import Event, Thread

    store = SQLiteBackgroundWorkerStore(tmp_path / "work.sqlite3")
    store.accept(config(), "u1", "我喜欢茶")
    entered, released, observed = Event(), Event(), Event()
    outcome = []

    def model(stage, messages, schema, cancelled):
        entered.set()
        if cancelled.wait(1):
            observed.set()
        else:
            released.wait(1)
        return {"operations": []}

    worker = UserFactWorker(store, model, clock=lambda: 10)
    thread = Thread(target=lambda: outcome.append(worker.run_one(A, "background")), daemon=True)
    thread.start()
    try:
        assert entered.wait(1)
        store.accept(config(), "u2", "更正，我不喜欢茶")
        thread.join(0.5)
        assert not thread.is_alive() and observed.is_set()
        assert outcome == [None]
        assert store.background_sources(A)[0]["status"] == "queued"
        assert store.memory_barrier(A)
    finally:
        released.set()
        thread.join(2)


def test_explicit_fact_target_selection_reaches_101st_record_without_truncation(tmp_path):
    from datetime import UTC, datetime

    from virtual_companion.boundary.memory import FactKey, MemoryMutation

    store = SQLiteBackgroundWorkerStore(tmp_path / "work.sqlite3")
    store.register(A)
    owner = store.claim_owner(A, "seed", expected_epoch=0)
    for index in range(101):
        command = MemoryMutation(
            Scope(A, "seed"), f"seed-{index}", "assert_fact", f"record-{index}",
            f"第{index}条事实", "manual", f"source-{index}", "seed",
            datetime.fromtimestamp(0, UTC), FactKey("user", f"attribute-{index}", "profile"),
        )
        assert store.mutate(command, owner, expected_version=index)["disposition"] == "applied"
    with pytest.raises(ValueError, match="capacity"):
        store.fact_targets(A, now_us=1)
    selected = store.fact_targets(A, now_us=1, candidates=(("record-100", 1),))
    assert list(selected) == ["record-100"]
    assert selected["record-100"]["content"] == "第100条事实"
    with pytest.raises(VersionConflict):
        store.fact_targets(A, now_us=1, candidates=(("record-100", 2),))


def test_background_correction_locates_101st_fact_and_rechecks_revision(tmp_path):
    from datetime import UTC, datetime

    from virtual_companion.boundary.memory import FactKey, MemoryMutation

    store = SQLiteBackgroundWorkerStore(tmp_path / "work.sqlite3")
    store.register(A)
    owner = store.claim_owner(A, "seed", expected_epoch=0)
    for index in range(101):
        store.mutate(MemoryMutation(
            Scope(A, "seed"), f"seed-{index}", "assert_fact", f"record-{index}",
            f"我喜欢饮品{index}", "manual", f"source-{index}", "seed",
            datetime.fromtimestamp(0, UTC), FactKey("user", f"drink-{index}", "profile"),
        ), owner, expected_version=index)
    item = store.accept(config(), "u1", "更正，我现在不喜欢饮品100")
    stages = []

    def model(stage, messages, schema, cancelled):
        stages.append(stage)
        data = json.loads(messages[-1]["content"])
        if stage == "fact_target_select":
            matches = [row["record_id"] for row in data["catalog"]
                       if row["record_id"] == "record-100"]
            return {"selected": matches, "uncertain": False, "intent": "correction"}
        if stage == "fact_extract":
            assert [row["record_id"] for row in data["targets"]] == ["record-100"]
            return {"operations": [{**operation(
                operation="supersede", evidenceTurnId=item.input_id,
                evidenceQuote="更正，我现在不喜欢饮品100",
                supersedesAtomId="record-100", content="更正，我现在不喜欢饮品100",
            ), "factKey": None}]}
        if stage == "fact_support":
            return {"intent": "correction", "verdict": "supported"}
        return {"operations": []}

    assert UserFactWorker(store, model, clock=lambda: 10).run_one(A, "background")
    assert "fact_target_select" in stages
    assert store.history(A, "record-100")[-1]["content"] == "更正，我现在不喜欢饮品100"
    assert not store.memory_barrier(A)


def test_background_repairs_single_misclassified_correction_before_commit(tmp_path):
    from datetime import UTC, datetime

    from virtual_companion.boundary.memory import FactKey, MemoryMutation

    store = SQLiteBackgroundWorkerStore(tmp_path / "work.sqlite3")
    store.register(A)
    owner = store.claim_owner(A, "seed", expected_epoch=0)
    store.mutate(MemoryMutation(
        Scope(A, "seed"), "seed", "assert_fact", "old-drink", "我喜欢茶",
        "manual", "seed-source", "seed", datetime.fromtimestamp(0, UTC),
        FactKey("user", "drink", "profile"),
    ), owner, expected_version=0)
    source = "更正，我现在不喜欢茶"
    item = store.accept(config(), "u1", source)
    stages = []

    def model(stage, messages, schema, cancelled):
        stages.append(stage)
        if stage == "fact_extract":
            return {"operations": [{**operation(
                content=source, evidenceQuote=source, evidenceTurnId=item.input_id,
            ), "factKey": {"subject": "user", "predicate": "drink"}}]}
        if stage == "fact_repair":
            return {"operations": [{**operation(
                operation="supersede", content=source, evidenceQuote=source,
                evidenceTurnId=item.input_id, supersedesAtomId="old-drink",
            ), "factKey": None}]}
        if stage == "fact_support":
            return {"intent": "correction", "verdict": "supported"}
        return {"operations": []}

    assert UserFactWorker(store, model, clock=lambda: 10).run_one(A, "background")
    assert stages == ["fact_extract", "fact_support", "fact_repair", "fact_support",
                      "exchange_extract"]
    assert [row["content"] for row in store.history(A, "old-drink")] == ["我喜欢茶", source]
    assert not store.memory_barrier(A)


def test_same_slot_conflict_gets_one_target_bound_repair_before_commit(tmp_path):
    from datetime import UTC, datetime

    from virtual_companion.boundary.memory import FactKey, MemoryMutation

    store = SQLiteBackgroundWorkerStore(tmp_path / "work.sqlite3")
    store.register(A)
    owner = store.claim_owner(A, "seed", expected_epoch=0)
    store.mutate(MemoryMutation(
        Scope(A, "seed"), "seed", "assert_fact", "old-drink", "我喜欢茶",
        "manual", "seed-source", "seed", datetime.fromtimestamp(0, UTC),
        FactKey("user", "drink", "profile"),
    ), owner, expected_version=0)
    source = "刚才说错了，我喜欢咖啡"
    item = store.accept(config(), "u1", source)
    stages = []
    supports = iter(({"intent": "assertion", "verdict": "supported"},
                     {"intent": "correction", "verdict": "supported"}))

    def model(stage, messages, schema, cancelled):
        stages.append(stage)
        if stage == "fact_extract":
            return {"operations": [{**operation(
                content=source, evidenceQuote=source, evidenceTurnId=item.input_id,
            ), "factKey": {"subject": "user", "predicate": "drink"}}]}
        if stage == "fact_repair":
            data = json.loads(messages[-1]["content"])
            assert data["targets"][0]["record_id"] == "old-drink"
            assert data["rejected_support"]["intent"] == "assertion"
            return {"operations": [{**operation(
                operation="supersede", content=source, evidenceQuote=source,
                evidenceTurnId=item.input_id, supersedesAtomId="old-drink",
            ), "factKey": None}]}
        if stage == "fact_support":
            return next(supports)
        return {"operations": []}

    assert UserFactWorker(store, model, clock=lambda: 10).run_one(A, "background")
    assert stages == ["fact_extract", "fact_support", "fact_repair", "fact_support",
                      "exchange_extract"]
    assert [row["content"] for row in store.history(A, "old-drink")] == [
        "我喜欢茶", source,
    ]
    assert not store.memory_barrier(A)


def test_conflict_repair_quota_survives_failure_and_reopen(tmp_path):
    from datetime import UTC, datetime

    from virtual_companion.boundary.memory import FactKey, MemoryMutation

    store = SQLiteBackgroundWorkerStore(tmp_path / "work.sqlite3")
    store.register(A)
    owner = store.claim_owner(A, "seed", expected_epoch=0)
    store.mutate(MemoryMutation(
        Scope(A, "seed"), "seed", "assert_fact", "old-drink", "我喜欢茶",
        "manual", "seed-source", "seed", datetime.fromtimestamp(0, UTC),
        FactKey("user", "drink", "profile"),
    ), owner, expected_version=0)
    source = "改成咖啡"
    item = store.accept(config(), "u1", source)
    stages = []

    def model(stage, messages, schema, cancelled):
        stages.append(stage)
        if stage == "fact_extract":
            return {"operations": [{**operation(
                content=source, evidenceQuote=source, evidenceTurnId=item.input_id,
            ), "factKey": {"subject": "user", "predicate": "drink"}}]}
        if stage == "fact_support":
            return {"intent": "assertion", "verdict": "supported"}
        if stage == "fact_repair":
            raise RuntimeError("injected transport failure")
        return {"operations": []}

    with pytest.raises(RuntimeError, match="transport"):
        UserFactWorker(store, model, clock=lambda: 10).run_one(A, "w1")
    first = store.background_sources(A)[0]
    assert first["status"] == "retryable" and first["attempts"] == 1

    reopened = SQLiteBackgroundWorkerStore(store.path)
    with pytest.raises(SourceResolutionUnresolved):
        UserFactWorker(reopened, model, clock=lambda: 1_000_011).run_one(A, "w2")
    protected = reopened.background_sources(A)[0]
    assert protected["status"] == "failed" and protected["attempts"] == 2
    assert stages.count("fact_repair") == 1


def test_opt_in_target_selection_binds_unique_correction_despite_slot_name_drift(tmp_path):
    from datetime import UTC, datetime

    from virtual_companion.boundary.memory import FactKey, MemoryMutation

    store = SQLiteBackgroundWorkerStore(tmp_path / "work.sqlite3")
    store.register(A)
    owner = store.claim_owner(A, "seed", expected_epoch=0)
    store.mutate(MemoryMutation(
        Scope(A, "seed"), "seed", "assert_fact", "old-drink", "我喜欢茶",
        "manual", "seed-source", "seed", datetime.fromtimestamp(0, UTC),
        FactKey("user", "drink", "profile"),
    ), owner, expected_version=0)
    source = "更正，我现在喜欢咖啡"
    item = store.accept(config(), "u1", source)
    stages = []

    def model(stage, messages, schema, cancelled):
        stages.append(stage)
        if stage == "fact_target_select":
            return {"selected": ["old-drink"], "uncertain": False,
                    "intent": "correction"}
        if stage == "fact_extract":
            return {"operations": [{**operation(
                content=source, evidenceQuote=source, evidenceTurnId=item.input_id,
            ), "factKey": {"subject": "user", "predicate": "beverage-preference"}}]}
        if stage == "fact_repair":
            return {"operations": [{**operation(
                operation="supersede", content=source, evidenceQuote=source,
                evidenceTurnId=item.input_id, supersedesAtomId="old-drink",
            ), "factKey": None}]}
        if stage == "fact_support":
            return {"intent": "correction", "verdict": "supported"}
        return {"operations": []}

    worker = UserFactWorker(store, model, clock=lambda: 10, target_selection=True)
    assert worker.run_one(A, "w1")
    assert stages == ["fact_target_select", "fact_extract", "fact_support", "fact_repair",
                      "fact_support", "exchange_extract"]
    assert store.history(A, "old-drink")[-1]["content"] == source
    assert not store.memory_barrier(A)


def test_opt_in_target_selection_protects_ambiguous_correction_before_extraction(tmp_path):
    from datetime import UTC, datetime

    from virtual_companion.boundary.memory import FactKey, MemoryMutation

    store = SQLiteBackgroundWorkerStore(tmp_path / "work.sqlite3")
    store.register(A)
    owner = store.claim_owner(A, "seed", expected_epoch=0)
    for index, predicate in enumerate(("drink", "morning-drink")):
        store.mutate(MemoryMutation(
            Scope(A, "seed"), f"seed-{index}", "assert_fact", f"record-{index}",
            f"饮品事实{index}", "manual", f"seed-source-{index}", "seed",
            datetime.fromtimestamp(0, UTC), FactKey("user", predicate, "profile"),
        ), owner, expected_version=index)
    store.accept(config(), "u1", "更正饮品偏好")
    stages = []

    def model(stage, messages, schema, cancelled):
        stages.append(stage)
        assert stage == "fact_target_select"
        return {"selected": ["record-0", "record-1"], "uncertain": False,
                "intent": "correction"}

    with pytest.raises(SourceResolutionUnresolved, match="unique"):
        UserFactWorker(store, model, clock=lambda: 10, target_selection=True).run_one(A, "w1")
    assert stages == ["fact_target_select"]
    source = store.background_sources(A)[0]
    assert source["status"] == "failed"
    assert source["error"] == "needs_clarification.source_coverage"


def test_opt_in_target_selection_requeues_when_selected_revision_moves(tmp_path):
    from datetime import UTC, datetime

    from virtual_companion.boundary.memory import FactKey, MemoryMutation

    store = SQLiteBackgroundWorkerStore(tmp_path / "work.sqlite3")
    store.register(A)
    owner = store.claim_owner(A, "seed", expected_epoch=0)
    store.mutate(MemoryMutation(
        Scope(A, "seed"), "seed", "assert_fact", "old-drink", "我喜欢茶",
        "manual", "seed-source", "seed", datetime.fromtimestamp(0, UTC),
        FactKey("user", "drink", "profile"),
    ), owner, expected_version=0)
    store.accept(config(), "u1", "更正为咖啡")
    stages = []

    def model(stage, messages, schema, cancelled):
        stages.append(stage)
        assert stage == "fact_target_select"
        store.mutate(MemoryMutation(
            Scope(A, "seed"), "concurrent", "correct", "old-drink", "并发改成水",
            "manual", "concurrent-source", "concurrent", datetime.fromtimestamp(0, UTC),
            expected_record_version=1,
        ), store.owner(A), expected_version=store.snapshot(A).write_version)
        return {"selected": ["old-drink"], "uncertain": False,
                "intent": "correction"}

    assert UserFactWorker(
        store, model, clock=lambda: 10, target_selection=True
    ).run_one(A, "w1") is None
    job = store.background_sources(A)[0]
    assert job["status"] == "queued" and job["attempts"] == 0
    assert stages == ["fact_target_select"]
    assert store.history(A, "old-drink")[-1]["content"] == "并发改成水"


def test_target_selection_rejects_uncertainty_and_unlisted_ids():
    catalog = tuple({"record_id": f"record-{i}", "version": 1,
                     "content": f"饮品{i}"} for i in range(101))

    def uncertain(stage, messages, schema, cancelled):
        assert stage == "fact_target_select"
        return {"selected": [], "uncertain": True, "intent": "correction"}

    with pytest.raises(ValueError, match="uncertain"):
        FactTargetSelector(uncertain, Event()).select("更正饮品100", catalog)

    def invented(stage, messages, schema, cancelled):
        return {"selected": ["record-outside-page"], "uncertain": False,
                "intent": "correction"}

    with pytest.raises(ValueError, match="schema"):
        FactTargetSelector(invented, Event()).select("更正饮品100", catalog)


def test_unknown_correction_target_cannot_clear_selection_barrier():
    catalog = tuple({"record_id": f"record-{i}", "version": 1,
                     "content": f"饮品{i}"} for i in range(101))

    def missing(stage, messages, schema, cancelled):
        return {"selected": [], "uncertain": False, "intent": "correction"}

    with pytest.raises(ValueError, match="Unresolved correction target"):
        FactTargetSelector(missing, Event()).select("更正，我不喜欢从前那款饮品", catalog)
