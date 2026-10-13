"""Impossible speech-act target bindings are repaired before domain support."""

import json

import pytest
from test_joint_source_proposal import question_store
from test_rebuilt_turns import A, config
from virtual_companion.boundary.contracts import SourceResolutionUnresolved
from virtual_companion.cognition.background_worker import UserFactWorker


@pytest.mark.parametrize("fixed", [True, False])
def test_question_with_transition_target_gets_one_bounded_repair(tmp_path, fixed):
    store, session = question_store(tmp_path)
    text = "你还记得我的音乐偏好吗？"
    store.accept(config(), "question", text, session_id=session)
    old = store.pending_exchanges(A, now_us=13)[0]["exchange_id"]
    stages = []

    def model(stage, messages, schema, cancelled):
        stages.append(stage)
        if stage in {"source_propose", "source_repair"}:
            if stage == "source_repair":
                assert json.loads(messages[-1]["content"])["reason"] == "intent_target_conflict"
            return {
                "facts": {"operations": []},
                "intents": {"coverage": "complete", "acts": [{
                    "segmentId": 0,
                    "evidenceQuote": text, "intent": "question",
                    "targetId": None if fixed and stage == "source_repair" else old,
                }]},
                "factCoverage": [{"segmentId": 0, "disposition": "none"}],
            }
        if stage == "fact_assess":
            return {"assessments": [], "sourceKinds": [{"segmentId": 0, "kind": "no_fact"}]}
        if stage == "exchange_extract":
            return {"operations": [{"operation": "add", "type": "response_obligation",
                                     "evidenceQuote": text,
                                    "supersedesAtomId": None, "expiresAt": None}]}
        if stage == "exchange_support":
            return {"intent": "question", "verdict": "supported"}
        raise AssertionError(stage)

    worker = UserFactWorker(store, model, clock=lambda: 14, resolution_protocol=True,
                            resolution_strategy="intent_first", joint_proposals=True)
    if fixed:
        receipt = json.loads(worker.run_one(A, "worker"))
        assert receipt["resolution"]["status"] == "complete_with_effects"
        assert not store.memory_barrier(A)
        assert len(store.pending_exchanges(A, now_us=15)) == 2
        assert stages == ["source_propose", "source_repair", "fact_assess",
                          "exchange_extract", "exchange_support"]
    else:
        with pytest.raises(SourceResolutionUnresolved):
            worker.run_one(A, "worker")
        assert stages == ["source_propose", "source_repair"]
        assert store.memory_barrier(A)
        assert len(store.pending_exchanges(A, now_us=15)) == 1
