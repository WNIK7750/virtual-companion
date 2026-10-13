"""Recorded default desktop failure: a repeated question cannot resolve itself.

Synthetic witness: stage1-delivery-memory-20260927/memory-tail-02/call-005.json.
Keep both the rejected operation and controlled replacement; no external fixture
or real model is needed to replay the counterexample.
"""

import json
import threading
from copy import deepcopy

import pytest
from test_rebuilt_turns import A, config
from virtual_companion.boundary.contracts import Scope
from virtual_companion.boundary.sqlite_background_worker import SQLiteResolvedBackgroundStore
from virtual_companion.cognition.background_worker import UserFactWorker
from virtual_companion.cognition.evidence_memory import EvidenceTurn
from virtual_companion.cognition.exchange_model import ExchangeModel

QUESTION = "我的饮食偏好是什么？"
BAD_PROPOSAL = {"operations": [{
    "operation": "resolve", "type": None,
    "evidenceQuote": QUESTION,
    "supersedesAtomId": "old-question", "expiresAt": None,
}]}


def inputs():
    scope = Scope(A, "session")
    return dict(
        scope=scope, batch_id="current", now_us=100,
        turns=(EvidenceTurn(scope, "old-source", "user", QUESTION, len(QUESTION)),
               EvidenceTurn(scope, "current", "user", QUESTION, len(QUESTION))),
        targets={"old-question": {
            "companion_id": A, "exchange_id": "old-question", "revision": 1,
            "direction": "assistant_response", "content": QUESTION,
            "status": "pending", "expires_us": None,
        }},
        eligible_source_ids=("current",), strategy="intent_first",
    )


class Recorded:
    def __init__(self, *, first=BAD_PROPOSAL, replacement=None, cancel_after_first=False):
        self.first = deepcopy(first)
        self.replacement = {"operations": []} if replacement is None else replacement
        self.cancel_after_first = cancel_after_first
        self.calls = []

    def __call__(self, stage, messages, schema, cancelled):
        data = json.loads(messages[-1]["content"])
        self.calls.append((stage, data, deepcopy(schema)))
        if stage == "exchange_intent":
            return {"coverage": "complete", "acts": [{
                "segmentId": 0,
                "evidenceQuote": QUESTION, "intent": "question", "targetId": None,
            }]}
        assert stage == "exchange_extract", "No illegal proposal may reach semantic support"
        if "repair" not in data:
            if self.cancel_after_first:
                cancelled.set()
            return deepcopy(self.first)
        return deepcopy(self.replacement)


@pytest.mark.parametrize("replacement", [
    {"operations": []},
    {"operations": [{**BAD_PROPOSAL["operations"][0], "operation": "add",
                     "type": "response_obligation", "supersedesAtomId": None}]},
])
def test_recorded_question_resolve_gets_one_bound_repair_without_resolving_target(replacement):
    model, params = Recorded(replacement=replacement), inputs()
    original_targets = deepcopy(params["targets"])
    proposals, resolution = ExchangeModel(model, threading.Event()).resolve_source(**params)
    assert proposals == ()
    assert resolution.no_exchange is True
    assert resolution.evaluator == "exchange-model.v22"
    assert params["targets"] == original_targets
    assert [call[0] for call in model.calls] == [
        "exchange_intent", "exchange_extract", "exchange_extract",
    ]
    _, original, schema = model.calls[1]
    _, repaired, repaired_schema = model.calls[2]
    assert repaired_schema == schema
    assert {key: value for key, value in repaired.items() if key != "repair"} == original
    assert repaired["repair"]["previous"] == BAD_PROPOSAL
    assert "question" in repaired["repair"]["error"]


def test_persisting_question_resolve_stays_rejected_after_one_repair():
    model = Recorded(replacement=BAD_PROPOSAL)
    with pytest.raises(ValueError, match="responding party"):
        ExchangeModel(model, threading.Event()).resolve_source(**inputs())
    assert [call[0] for call in model.calls].count("exchange_extract") == 2


@pytest.mark.parametrize("changed", [
    {"evidenceTurnId": "unknown-source"},
    {"evidenceQuote": "编造的原文"},
    {"supersedesAtomId": "unknown-target"},
    {"expiresAt": 1234},
    {"type": "response_obligation"},
])
def test_original_identity_or_shape_error_is_not_softened_into_repair(changed):
    first = {"operations": [{**BAD_PROPOSAL["operations"][0], **changed}]}
    model = Recorded(first=first)
    with pytest.raises(ValueError):
        ExchangeModel(model, threading.Event()).resolve_source(**inputs())
    assert [call[0] for call in model.calls].count("exchange_extract") == 1


@pytest.mark.parametrize("changed", [
    {"evidenceTurnId": "unknown-source"},
    {"evidenceQuote": "编造的原文"},
    {"supersedesAtomId": "unknown-target"},
])
def test_replacement_still_has_exact_source_and_target_checks(changed):
    model = Recorded(replacement={"operations": [{**BAD_PROPOSAL["operations"][0], **changed}]})
    with pytest.raises(ValueError, match="schema"):
        ExchangeModel(model, threading.Event()).resolve_source(**inputs())
    assert [call[0] for call in model.calls].count("exchange_extract") == 2


@pytest.mark.parametrize("change", ["other-content", "expired", "ambiguous", "wrong-direction"])
def test_target_must_be_one_exact_live_assistant_response(change):
    params = inputs()
    target = params["targets"]["old-question"]
    if change == "other-content":
        target["content"] = "你想聊什么？"
    elif change == "expired":
        target["expires_us"] = 99
    elif change == "ambiguous":
        params["targets"]["another-question"] = {**target, "exchange_id": "another-question"}
    else:
        target["direction"] = "user_question"
    model = Recorded()
    with pytest.raises(ValueError):
        ExchangeModel(model, threading.Event()).resolve_source(**params)
    assert [call[0] for call in model.calls].count("exchange_extract") == 1


def test_cancel_after_illegal_candidate_prevents_corrective_call():
    model = Recorded(cancel_after_first=True)
    with pytest.raises(InterruptedError):
        ExchangeModel(model, threading.Event()).resolve_source(**inputs())
    assert [call[0] for call in model.calls].count("exchange_extract") == 1


@pytest.mark.parametrize("repaired", [True, False])
def test_worker_commits_only_valid_no_change_and_keeps_pending_target(tmp_path, repaired):
    store = SQLiteResolvedBackgroundStore(tmp_path / "repeated-question.sqlite3")
    stages = []

    def invoke(stage, messages, schema, cancelled):
        data = json.loads(messages[-1]["content"])
        stages.append(stage)
        if stage == "fact_extract":
            return {"operations": []}
        if stage == "exchange_intent":
            return {"coverage": "complete", "acts": [{
                "segmentId": 0, "evidenceQuote": QUESTION,
                "intent": "question", "targetId": None,
            }]}
        if stage == "exchange_extract":
            if data["targets"]:
                if repaired and "repair" in data:
                    return {"operations": []}
                return {"operations": [{**BAD_PROPOSAL["operations"][0],

                    "supersedesAtomId": data["targets"][0]["exchange_id"]}]}
            return {"operations": [{**BAD_PROPOSAL["operations"][0], "operation": "add",
                "type": "response_obligation", "supersedesAtomId": None}]}
        assert stage == "exchange_support"
        return {"intent": "question", "verdict": "supported"}

    store.accept(config(), "first", QUESTION)
    worker = UserFactWorker(store, invoke, clock=lambda: 10,
                           resolution_protocol=True, resolution_strategy="intent_first")
    assert worker.run_one(A, "worker")
    before = store.pending_exchanges(A, now_us=10)
    assert len(before) == 1
    item = store.accept(config(), "repeat", QUESTION)
    stages.clear()
    if repaired:
        assert worker.run_one(A, "worker")
    else:
        with pytest.raises(ValueError, match="responding party"):
            worker.run_one(A, "worker")
    with store.transaction(write=False) as db:
        status = db.execute("SELECT status FROM background_sources WHERE source=?",
                            (item.input_id,)).fetchone()["status"]
    assert status == ("completed" if repaired else "retryable")
    assert store.memory_barrier(A) is (not repaired)
    assert store.pending_exchanges(A, now_us=10) == before
    assert stages == ["fact_extract", "exchange_intent", "exchange_extract", "exchange_extract"]
