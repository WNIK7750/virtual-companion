import json
from dataclasses import replace

import pytest
from test_cognition_preparation import basis
from test_cognitive_kernel import RecordedModel, kernel
from test_rebuilt_turns import A
from virtual_companion.boundary.sqlite_lifecycle import SQLiteLifecycleStore
from virtual_companion.cognition.pilot import PilotPreparation, appraise, direct, engagement
from virtual_companion.cognition.preparation import CallCancelled


def test_appraisal_failure_and_invalid_are_observable_noops():
    for raw in (None, {"event": "invalid"}, {"affect_delta": {"valence": float("nan")}}):
        result = appraise(raw, "stimulus")
        assert result["event"]["event_type"] == "routine"
        assert result["event"]["source_input_ids"] == ["stimulus"]
        assert result["fallback_reason"]


def test_appraisal_binds_evidence_and_preserves_effects():
    raw = {
        "event": {
            "event_type": "conflict",
            "source_input_ids": ["foreign"],
            "confidence": 0.8,
            "novelty": 0.8,
            "tension_impact": 0.2,
        },
        "affect_delta": {"valence": -0.1},
    }
    result = appraise(raw, "stimulus")
    assert result["event"]["source_input_ids"] == ["stimulus"]
    assert result["event"]["tension_impact"] == 0.2
    assert result["affect_delta"] == {"valence": -0.1}
    assert result["source_rebound"]


@pytest.mark.parametrize(
    "text,availability,action",
    [
        ("先别说话", "available", "respect_silence"),
        ("你在听吗", "sleeping", "respond_now"),
        ("请回答这个问题", "sleeping", "schedule_later"),
    ],
)
def test_engagement_preserves_ordered_obligation_rules(text, availability, action):
    assert engagement(text, availability, "input", 100)["action"] == action


def test_director_narrow_fast_path_does_not_swallow_risk_or_staleness():
    context = {"state_revision": 4, "relationship_revision": 2, "user_act": "greeting"}
    assert direct(context, None, routine=True)["fallback_reason"] == "deterministic_routine"
    context["tension"] = 0.3
    assert direct(context, None, routine=True) is None
    result = direct(context, {"state_revision": 3, "relationship_revision": 2})
    assert result["state_revision"] == 4
    assert result["fallback_reason"] == "invalid_or_stale_output"


class PreparationModel(RecordedModel):
    def stream(self, messages, cancelled):
        request = json.loads(messages[-1]["content"])
        if request["stage"] not in {"appraise", "direct"}:
            yield from super().stream(messages, cancelled)
            return
        self.requests.append(request)
        payload = (
            {"event": {"event_type": "routine"}}
            if request["stage"] == "appraise"
            else {
                "state_revision": request["director_context"]["state_revision"],
                "relationship_revision": request["director_context"]["relationship_revision"],
            }
        )
        yield json.dumps({"context_sha256": request["context_sha256"], "output": payload})


def test_deferred_obligation_is_atomic_durable_and_distinct_from_silence(tmp_path):
    store, _, run, persona = basis(tmp_path)
    candidate = kernel(store, persona, PreparationModel())
    candidate.preparation = PilotPreparation(availability="sleeping")
    result = candidate.generate(run, store.snapshot(A), "请回答这个问题")
    assert result.text == ""
    assert "a2_pilot" not in json.loads(store.snapshot(A).values_json)
    receipt = store.commit(result, now_us=100)
    assert receipt["status"] == "committed"
    reopened = SQLiteLifecycleStore(store.path)
    state = json.loads(reopened.snapshot(A).values_json)["a2_pilot"]
    assert state["engagement"]["action"] == "schedule_later"
    assert state["pending_obligations"][0]["source_input_id"] == run.input_id
    assert state["pending_obligations"][0]["status"] == "delayed"
    assert store.commit(result, now_us=100) == receipt


def test_preparation_cancel_never_falls_back_to_success(tmp_path):
    store, _, run, persona = basis(tmp_path)
    model = PreparationModel()

    def cancelled(messages, cancel):
        raise CallCancelled()
        yield ""

    model.stream = cancelled
    candidate = kernel(store, persona, model)
    candidate.preparation = PilotPreparation()
    with pytest.raises(CallCancelled):
        candidate.generate(run, store.snapshot(A), "input")
    assert store.inspect(A, run.input_id)["commit"] is None


def test_stale_project_cannot_overwrite_new_state(tmp_path):
    store, _, run, persona = basis(tmp_path)
    candidate = kernel(store, persona, PreparationModel())
    candidate.preparation = PilotPreparation()
    result = candidate.generate(run, store.snapshot(A), "input")
    newer = replace(result.assessment, event_id="newer", values_json='{"newer":true}')
    store.apply(newer, run.owner, expected_version=run.state_version)
    with pytest.raises(ValueError):
        store.commit(result, now_us=100)
    assert json.loads(store.snapshot(A).values_json) == {"newer": True}


def test_current_risk_prevents_greeting_fast_path_and_keeps_business_versions(tmp_path):
    from datetime import UTC, datetime

    from virtual_companion.boundary.contracts import Scope, StateEvent

    store, item, run, persona = basis(tmp_path)
    values = {"revision": 0, "relationship": {"tension": {"value": 0.8}}}
    when = datetime(1970, 1, 1, tzinfo=UTC)
    store.apply(
        StateEvent(
            Scope(A, run.session_id), "risk", "state", when, when, values_json=json.dumps(values)
        ),
        run.owner,
        expected_version=0,
    )
    run = store.claim(A, item.input_id, "worker", now_us=200, lease_us=100)
    model = PreparationModel()
    candidate = kernel(store, persona, model)
    candidate.clock = lambda: 200
    candidate.preparation = PilotPreparation(user_act="greeting")
    result = candidate.generate(run, store.snapshot(A), "你好")
    assert "direct" in [r["stage"] for r in model.requests]
    store.commit(result, now_us=200)
    state = store.snapshot(A)
    assert json.loads(state.values_json)["revision"] == state.business_revision == 1
    assert json.loads(state.values_json)["relationship"] == values["relationship"]


@pytest.mark.parametrize("count", [160, 200])
def test_complete_long_history_reaches_every_model_stage(tmp_path, count):
    from virtual_companion.boundary.contracts import Scope
    from virtual_companion.cognition.kernel import SourceSelection
    from virtual_companion.cognition.preparation import DisplayedTurn

    store, _, run, persona = basis(tmp_path)
    model = PreparationModel()
    candidate = kernel(store, persona, model)
    candidate.preparation = PilotPreparation()
    history = tuple(
        DisplayedTurn(
            Scope(A, "old"),
            f"turn-{n}",
            "user",
            "早期事实：青豆" if n == 0 else "完整历史",
            len("早期事实：青豆" if n == 0 else "完整历史"),
        )
        for n in range(count)
    )
    candidate.select = lambda claim: SourceSelection(history=history)
    candidate.generate(run, store.snapshot(A), "当前输入")
    assert [r["stage"] for r in model.requests] == ["appraise", "direct", "reply", "review"]
    for request in model.requests:
        assert len(request["context"]["history"]) == count
        assert request["context"]["history"][0]["text"] == "早期事实：青豆"
        assert request["context"]["current_text"] == "当前输入"


def test_foreign_preparation_is_not_treated_as_a_fallback(tmp_path):
    from virtual_companion.boundary.runtime import KernelFailure

    store, _, run, persona = basis(tmp_path)
    model = PreparationModel()

    def foreign(messages, cancelled):
        yield json.dumps(
            {"context_sha256": "0" * 64, "output": {"event": {"event_type": "routine"}}}
        )

    model.stream = foreign
    candidate = kernel(store, persona, model)
    candidate.preparation = PilotPreparation()
    with pytest.raises(KernelFailure, match="preparation_binding"):
        candidate.generate(run, store.snapshot(A), "input")
    assert store.inspect(A, run.input_id)["commit"] is None


def test_director_rejects_unknown_mode_and_bad_configuration():
    context = {"state_revision": 1, "relationship_revision": 1}
    assert (
        direct(context, {**context, "mode": "invalid"})["fallback_reason"]
        == "invalid_or_stale_output"
    )
    with pytest.raises(ValueError):
        PilotPreparation(availability="")


@pytest.mark.parametrize("act", ["share", "greeting"])
def test_duty_trace_identifies_actual_calls_and_deterministic_paths(tmp_path, act):
    store, _, run, persona = basis(tmp_path)
    candidate = kernel(store, persona, PreparationModel())
    candidate.preparation = PilotPreparation(user_act=act)
    candidate.generate(run, store.snapshot(A), "input")
    duties = {d["duty"]: d for d in candidate.trace[-1]["duties"]}
    calls = {r["call_id"]: r for r in candidate.trace if r["stage"] == "transport_call"}
    assert duties["engagement"]["call_id"] is None
    assert calls[duties["state_appraisal"]["call_id"]]["request_stage"] == "appraise"
    if act == "greeting":
        assert duties["expression_director"]["call_id"] is None
        assert duties["expression_director"]["reason"] == "deterministic_routine"
    else:
        assert calls[duties["expression_director"]["call_id"]]["request_stage"] == "direct"
    for d in ("main_reply", "causal_review", "expression_review"):
        assert calls[duties[d]["call_id"]]["status"] == "returned"
