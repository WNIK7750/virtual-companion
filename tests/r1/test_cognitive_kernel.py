import json
from dataclasses import replace

import pytest
from test_cognition_preparation import basis
from test_rebuilt_turns import A
from virtual_companion.boundary.presentation import PresentationPolicy
from virtual_companion.boundary.runtime import KernelFailure
from virtual_companion.cognition.kernel import CognitiveKernel, SourceSelection
from virtual_companion.cognition.preparation import CallCancelled


class RecordedModel:
    def __init__(self, *, rejects=0):
        self.requests, self.checked = [], []
        self.rejects = rejects
        self.mutate = lambda stage, result: result

    def check_context(self, messages):
        self.checked.append(json.loads(messages[-1]["content"]))

    def stream(self, messages, cancelled):
        request = json.loads(messages[-1]["content"])
        self.requests.append(request)
        stage = request["stage"]
        result = {"context_sha256": request["context_sha256"]}
        if stage == "prepare":
            result.update(
                engagement={"mode": "respond", "reason": "answer_obligation"},
                appraisal={
                    "source_input_id": request["context"]["claim"]["input_id"],
                    "state_version": request["context"]["state"]["write_version"],
                    "observation": "synthetic neutral",
                },
                director={
                    "state_version": request["context"]["state"]["write_version"],
                    "instructions": "synthetic concise response",
                },
            )
        elif stage == "reply":
            result.update(
                semantic_text="合成回答",
                units=[{"unit_id": "one", "kind": "text", "text": "合成回答"}],
            )
        else:
            result["candidate_sha256"] = request["candidate_sha256"]
            passed = request["review_attempt"] > self.rejects
            result.update(
                causal_review={"passed": passed, "reason": "recorded"},
                expression_review={"passed": True, "reason": "recorded"},
            )
        yield json.dumps(self.mutate(stage, result), ensure_ascii=False)

    def cancel(self):
        pass


def kernel(store, persona, model, **kwargs):
    return CognitiveKernel(
        store=store,
        model=model,
        persona=lambda claim: persona,
        select=lambda claim: SourceSelection(),
        project=lambda context, appraisal: None,
        policy=PresentationPolicy(2, 200, 4, False),
        clock=lambda: 100,
        lease_us=50,
        deadline_us=1000,
        renewal_margin_us=10,
        **kwargs,
    )


def test_concrete_chain_commits_only_reviewed_surface_and_records_six_duties(tmp_path):
    store, item, run, persona = basis(tmp_path)
    model = RecordedModel()
    candidate = kernel(store, persona, model)
    result = candidate.generate(run, store.snapshot(A), "完整当前输入🙂")
    receipt = store.commit(result, now_us=100)
    assert receipt["status"] == "committed"
    assert result.text == "合成回答"
    assert [row["stage"] for row in model.requests] == ["prepare", "reply", "review"]
    assert model.requests == model.checked
    assert len(candidate.trace[-1]["duties"]) == 6
    assert model.requests[1]["preparation"]["appraisal"]["observation"] == "synthetic neutral"
    assert model.requests[-1]["candidate"]["units"][0]["text"] == result.text


@pytest.mark.parametrize("rejects", [1, 2, 3])
def test_review_rewrites_are_bounded_and_keep_rejection_feedback(tmp_path, rejects):
    store, _, run, persona = basis(tmp_path)
    model = RecordedModel(rejects=rejects)
    candidate = kernel(store, persona, model)
    if rejects == 3:
        with pytest.raises(KernelFailure, match="review_exhausted"):
            candidate.generate(run, store.snapshot(A), "input")
    else:
        result = candidate.generate(run, store.snapshot(A), "input")
        assert result.revision == rejects + 1
    replies = [r for r in model.requests if r["stage"] == "reply"]
    assert len(replies) == min(3, rejects + 1)
    assert replies[1]["feedback"]["causal_review"]["passed"] is False
    assert store.inspect(A, run.input_id)["commit"] is None


@pytest.mark.parametrize("stage", ["prepare", "reply", "review"])
def test_foreign_context_never_reaches_commit(tmp_path, stage):
    store, _, run, persona = basis(tmp_path)
    model = RecordedModel()
    model.mutate = lambda s, row: {**row, "context_sha256": "0" * 64} if s == stage else row
    with pytest.raises(KernelFailure, match="model_contract"):
        kernel(store, persona, model).generate(run, store.snapshot(A), "input")
    assert store.inspect(A, run.input_id)["commit"] is None


def test_capacity_rejection_happens_before_model_request(tmp_path):
    store, _, run, persona = basis(tmp_path)
    model = RecordedModel()

    def reject(messages):
        raise ValueError("capacity")

    model.check_context = reject
    with pytest.raises(ValueError, match="capacity"):
        kernel(store, persona, model).generate(run, store.snapshot(A), "input")
    assert not model.requests


def test_actual_surface_hard_failure_cannot_be_overridden_by_positive_reviews(tmp_path):
    store, _, run, persona = basis(tmp_path)
    model = RecordedModel()
    model.mutate = lambda stage, row: (
        {**row, "units": [{"unit_id": "one", "kind": "action", "text": "action"}]}
        if stage == "reply"
        else row
    )
    with pytest.raises(KernelFailure, match="review_exhausted"):
        kernel(store, persona, model).generate(run, store.snapshot(A), "input")
    assert len([r for r in model.requests if r["stage"] == "review"]) == 3


def test_durable_cancel_prevents_even_capacity_check(tmp_path):
    store, item, run, persona = basis(tmp_path)
    store.cancel(A, item.input_id)
    model = RecordedModel()
    with pytest.raises(CallCancelled):
        kernel(store, persona, model).generate(run, store.snapshot(A), "input")
    assert not model.checked


def test_stale_state_snapshot_rejected(tmp_path):
    store, _, run, persona = basis(tmp_path)
    model = RecordedModel()
    with pytest.raises(ValueError):
        kernel(store, persona, model).generate(
            run, replace(store.snapshot(A), write_version=1), "input"
        )
    assert not model.checked


def test_memory_reads_current_source_and_never_scans_without_explicit_refs(tmp_path):
    from test_rebuilt_memory import apply, fact, mutation

    store, _, run, persona = basis(tmp_path)
    fact(store)
    apply(
        store,
        mutation(
            "fix", kind="correct", record="home", content="住在上海", expected_record_version=1
        ),
    )
    run = replace(run, state_version=store.snapshot(A).write_version)
    model = RecordedModel()
    candidate = kernel(store, persona, model)
    candidate.select = lambda claim: SourceSelection(recall_refs=(("home", 1), ("home", 2)))
    candidate.generate(run, store.snapshot(A), "input")
    context = model.requests[0]["context"]
    assert [m["content"] for m in context["memories"]] == ["住在上海"]
    assert context["memories"][0]["kind"] == "fact"
    assert context["memories"][0]["fact_key"] == {
        "subject": "user",
        "predicate": "home-city",
        "scope": "profile",
    }
    assert "住在北京" not in json.dumps(context, ensure_ascii=False)
    candidate.select = lambda claim: SourceSelection()
    candidate.generate(run, store.snapshot(A), "input")
    assert model.requests[-3]["context"]["memories"] == []


def test_memory_ttl_uses_calendar_clock_not_process_lease_ticks(tmp_path):
    from datetime import timedelta

    from test_rebuilt_memory import WHEN, apply, mutation
    from virtual_companion.boundary.memory import FactKey
    from virtual_companion.text.chronology import utc_microseconds

    store, _, run, persona = basis(tmp_path)
    apply(
        store,
        mutation(
            kind="assert_fact",
            record="temporary",
            content="今天在杭州",
            fact_key=FactKey("user", "temporary-location", "profile"),
            expires_at=WHEN + timedelta(seconds=1),
        ),
    )
    run = replace(run, state_version=store.snapshot(A).write_version)
    model = RecordedModel()
    candidate = kernel(store, persona, model)
    candidate.evidence_clock = lambda: utc_microseconds(WHEN + timedelta(seconds=2))
    candidate.select = lambda _: SourceSelection(recall_refs=(("temporary", 1),))
    candidate.generate(run, store.snapshot(A), "input")
    context = model.requests[0]["context"]
    assert context["memories"] == []
    assert context["observed_at_us"] == candidate.evidence_clock()


@pytest.mark.parametrize(
    "fault", ["foreign_candidate", "string_boolean", "stale_director", "defer"]
)
def test_unsupported_or_invalid_model_decisions_are_explicit_failures(tmp_path, fault):
    store, _, run, persona = basis(tmp_path)
    model = RecordedModel()

    def corrupt(stage, row):
        if stage == "review":
            if fault == "foreign_candidate":
                row["candidate_sha256"] = "0" * 64
            if fault == "string_boolean":
                row["causal_review"]["passed"] = "false"
        if stage == "prepare":
            if fault == "stale_director":
                row["director"]["state_version"] += 1
            if fault == "defer":
                row["engagement"]["mode"] = "schedule_later"
        return row

    model.mutate = corrupt
    with pytest.raises(KernelFailure, match="model_contract"):
        kernel(store, persona, model).generate(run, store.snapshot(A), "input")
    assert store.inspect(A, run.input_id)["commit"] is None


def test_deliberate_silence_is_reviewed_and_does_not_call_reply(tmp_path):
    store, _, run, persona = basis(tmp_path)
    model = RecordedModel()

    def silence(stage, row):
        if stage == "prepare":
            row["engagement"]["mode"] = "silence"
        return row

    model.mutate = silence
    candidate = kernel(store, persona, model)
    result = candidate.generate(run, store.snapshot(A), "先别说话")
    assert result.text == ""
    assert [r["stage"] for r in model.requests] == ["prepare", "review"]
    assert store.commit(result, now_us=100)["status"] == "committed"


def test_projected_state_commits_atomically_with_reviewed_candidate(tmp_path):
    from datetime import datetime

    from virtual_companion.boundary.contracts import Scope, StateEvent

    store, _, run, persona = basis(tmp_path)
    candidate = kernel(store, persona, RecordedModel())
    when = datetime.fromisoformat("2026-09-09T00:00:00Z")
    candidate.project = lambda context, appraisal: StateEvent(
        Scope(A, run.session_id),
        "appraisal",
        "state",
        when,
        when,
        values_json='{"synthetic":true}',
        business_revision=1,
    )
    result = candidate.generate(run, store.snapshot(A), "input")
    assert store.snapshot(A).business_revision == 0
    store.commit(result, now_us=100)
    assert store.snapshot(A).business_revision == 1


def test_runtime_known_failure_explicit_retry_and_committed_recovery(tmp_path):
    from virtual_companion.boundary.runtime import CompanionRuntime
    from virtual_companion.text.identity import CompanionConfig

    store, item, _, persona = basis(tmp_path)
    config = CompanionConfig.from_mapping(json.loads(item.config_json)["companion_config"])
    runtime = CompanionRuntime(config, store)
    now = [200]
    model = RecordedModel(rejects=3)
    candidate = kernel(store, persona, model)
    candidate.clock = lambda: now[0]
    failed = runtime.execute(item.input_id, "worker", candidate, clock=candidate.clock, lease_us=50)
    assert failed["status"] == "failed"
    runtime.retry(item.input_id, failed["attempt"], "retry", config)
    model.rejects = 0
    committed = runtime.execute(
        item.input_id, "worker", candidate, clock=candidate.clock, lease_us=50
    )
    count = len(model.requests)
    now[0] = 2000
    assert (
        runtime.execute(item.input_id, "worker", candidate, clock=candidate.clock, lease_us=50)
        == committed
    )
    assert len(model.requests) == count
