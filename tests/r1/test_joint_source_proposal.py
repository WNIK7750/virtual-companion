"""Joint proposals share evidence but never bypass independent domain support."""

import json
import sqlite3
from dataclasses import replace
from hashlib import sha256

import pytest
from jsonschema import Draft202012Validator
from test_evidence_memory import operation
from test_rebuilt_presentation import approved
from test_rebuilt_turns import A, config
from test_source_resolution import seeded, source_status
from virtual_companion.boundary.contracts import Scope, VersionConflict
from virtual_companion.boundary.presentation import MessageUnit, Presentation
from virtual_companion.boundary.sqlite_background_worker import SQLiteResolvedBackgroundStore
from virtual_companion.boundary.turns import CognitiveResult
from virtual_companion.cognition.background_worker import (
    AssistantExchangeWorker,
    BackgroundScheduler,
    UserFactWorker,
)


@pytest.mark.parametrize("failure", [None, "omitted_fact", "false_none", "unknown",
                                     "unsupported", "sqlite"])
def test_joint_source_keeps_mixed_fact_and_request_atomic(tmp_path, failure):
    store, item, correction, request = seeded(tmp_path, mixed=True)
    stages = []
    if failure == "sqlite":
        with store.transaction() as db:
            db.execute("CREATE TRIGGER reject_completion BEFORE UPDATE ON background_sources "
                       "WHEN NEW.status='completed' BEGIN SELECT RAISE(ABORT,'injected'); END")

    def model(stage, messages, schema, cancelled):
        stages.append(stage)
        if stage == "source_propose":
            data = json.loads(messages[-1]["content"])
            assert data["sources"][-1]["text"] == correction + "。" + request
            fact = {**operation(operation="supersede", content=correction,
                               evidenceQuote=correction, evidenceTurnId=item.input_id,
                               supersedesAtomId="old-drink"), "factKey": None}
            fact.pop("content")
            return {
                "facts": {"operations": [] if failure == "omitted_fact" else [fact]},
                "intents": {"acts": [
                    {"segmentId": 0,
                     "evidenceQuote": correction, "intent": "statement", "targetId": None},
                    {"segmentId": 1,
                     "evidenceQuote": request, "intent": "request", "targetId": None},
                ], "coverage": "complete"},
                "factCoverage": [
                    {"segmentId": 0, "disposition": "unknown" if failure == "unknown"
                     else "none" if failure == "false_none" else "candidate"},
                    {"segmentId": 1, "disposition": "none"},
                ],
            }
        if stage == "source_repair":
            return json.loads(messages[-1]["content"])["rejectedProposal"]
        if stage == "fact_support":
            data = json.loads(messages[-1]["content"])
            assert data["conversation"][-1]["id"] == item.input_id
            return {"intent": "correction", "verdict": "supported"}
        if stage == "fact_assess":
            return {"assessments": [{"candidateIndex": 0, "disposition": "accepted",
                                     "reason": "fact"}], "sourceKinds": [
                {"segmentId": 0, "kind": "fact"}, {"segmentId": 1, "kind": "no_fact"}]}
        if stage == "exchange_extract":
            return {"operations": [{"operation": "add", "type": "response_obligation",
                                     "evidenceQuote": request,
                                    "supersedesAtomId": None, "expiresAt": None}]}
        if stage == "exchange_support":
            return {"intent": "request", "verdict": "unsupported"
                    if failure == "unsupported" else "supported"}
        raise AssertionError(stage)

    worker = UserFactWorker(store, model, clock=lambda: 10, resolution_protocol=True,
                            resolution_strategy="intent_first", joint_proposals=True)
    if failure:
        with pytest.raises((ValueError, sqlite3.IntegrityError)):
            worker.run_one(A, "worker")
        expected = (
            "failed"
            if failure in {"omitted_fact", "false_none", "unknown"}
            else "retryable"
        )
        assert source_status(store) == expected and store.memory_barrier(A)
        if expected == "failed":
            assert store.background_sources(A)[0]["error"] == (
                "needs_clarification.source_coverage"
            )
        assert len(store.history(A, "old-drink")) == 1
        assert store.pending_exchanges(A, now_us=10) == []
        if failure in {"omitted_fact", "false_none", "unknown"}:
            assert stages == ["source_propose", "source_repair"]
    else:
        scheduler = BackgroundScheduler(
            store, user_invoke=model, assistant_invoke=model, clock=lambda: 10,
            resolution_protocol=True, resolution_strategy="intent_first", joint_proposals=True)
        receipt = json.loads(scheduler.run_next("worker"))
        assert receipt["evaluator"] == "joint-source.v9.fact-model.v7"
        assert source_status(store) == "completed" and not store.memory_barrier(A)
        assert len(store.history(A, "old-drink")) == 2
        assert len(store.pending_exchanges(A, now_us=10)) == 1
        assert stages == ["source_propose", "fact_assess", "fact_support", "exchange_extract",
                          "exchange_support"]


@pytest.mark.parametrize("initial", ["empty_candidate", "false_fact"])
def test_joint_recall_question_does_not_invent_an_unknown_fact(tmp_path, initial):
    store = SQLiteResolvedBackgroundStore(tmp_path / "recall-question.sqlite3")
    text = "你还记得我住在哪座城市吗？"
    item = store.accept(config(), "recall-question", text)
    stages = []

    def model(stage, messages, schema, cancelled):
        stages.append(stage)
        data = json.loads(messages[-1]["content"])
        if stage == "source_propose":
            instruction = messages[0]["content"]
            assert "只询问未知值" in instruction
            assert text in instruction
            invalid_empty_candidate = {
                "facts": {"operations": []},
                "intents": {"acts": [{
                    "segmentId": 0,

                    "evidenceQuote": text,
                    "intent": "question",
                    "targetId": None,
                }], "coverage": "complete"},
                "factCoverage": [{"segmentId": 0, "disposition": "candidate"}],
            }
            assert Draft202012Validator(schema).is_valid(invalid_empty_candidate)
            facts = []
            if initial == "false_fact":
                fact = {
                    **operation(
                        content=text,
                        evidenceQuote=text,
                        evidenceTurnId=item.input_id,
                    ),
                    "factKey": {"subject": "user", "predicate": "home"},
                }
                fact.pop("content")
                facts.append(fact)
            return {
                "facts": {"operations": facts},
                "intents": {"acts": [{
                    "segmentId": 0,

                    "evidenceQuote": text,
                    "intent": "question",
                    "targetId": None,
                }], "coverage": "complete"},
                "factCoverage": [{"segmentId": 0, "disposition": "candidate"}],
            }
        if stage == "source_repair":
            assert initial == "empty_candidate"
            assert data["reason"] == "fact_binding_conflict"
            assert data["rejectedProposal"]["facts"]["operations"] == []
            return {
                "facts": {"operations": []},
                "intents": {"acts": [{
                    "segmentId": 0,

                    "evidenceQuote": text,
                    "intent": "question",
                    "targetId": None,
                }], "coverage": "complete"},
                "factCoverage": [{"segmentId": 0, "disposition": "none"}],
            }
        if stage == "fact_assess":
            assert "只询问未知值" in messages[0]["content"]
            if initial == "false_fact":
                assert len(data["candidates"]) == 1
                return {
                    "assessments": [{
                        "candidateIndex": 0,
                        "disposition": "accepted",
                        "reason": "fact",
                    }],
                    "sourceKinds": [{"segmentId": 0, "kind": "fact"}],
                }
            assert data["candidates"] == []
            return {
                "assessments": [],
                "sourceKinds": [{"segmentId": 0, "kind": "no_fact"}],
            }
        if stage == "fact_support":
            assert initial == "false_fact"
            assert data["proposal"]["evidenceQuote"] == text
            return {"intent": "question", "verdict": "supported"}
        if stage == "exchange_extract":
            return {"operations": [{
                "operation": "add",
                "type": "response_obligation",

                "evidenceQuote": text,
                "supersedesAtomId": None,
                "expiresAt": None,
            }]}
        if stage == "exchange_support":
            return {"intent": "request", "verdict": "supported"}
        raise AssertionError(stage)

    receipt = json.loads(UserFactWorker(
        store,
        model,
        clock=lambda: 10,
        resolution_protocol=True,
        resolution_strategy="intent_first",
        joint_proposals=True,
    ).run_one(A, "worker"))

    assert receipt["operations"] == 0
    assert receipt["fact_resolution"]["segment_kinds"] == ["no_fact"]
    if initial == "false_fact":
        assert receipt["fact_resolution"]["dispositions"] == ["rejected"]
        assert receipt["fact_resolution"]["reasons"] == ["non_factual"]
        assert receipt["fact_resolution"]["evaluator"] == (
            "fact-source-assess+support.v1"
        )
    assert len(store.pending_exchanges(A, now_us=11)) == 1
    assert not store.memory_barrier(A)
    expected = ["source_propose"]
    if initial == "empty_candidate":
        expected.append("source_repair")
    expected.extend(["fact_assess"])
    if initial == "false_fact":
        expected.append("fact_support")
    expected.extend(["exchange_extract", "exchange_support"])
    assert stages == expected


def test_joint_support_uses_original_index_after_assessment_filters_candidate(tmp_path):
    store = SQLiteResolvedBackgroundStore(tmp_path / "filtered-candidate.sqlite3")
    parts = ("今天先聊旅行。", "你还记得我住在哪座城市吗？")
    item = store.accept(config(), "filtered-candidate", "".join(parts))

    def model(stage, messages, schema, cancelled):
        data = json.loads(messages[-1]["content"])
        if stage == "source_propose":
            facts = []
            for index, part in enumerate(parts):
                fact = {**operation(content=part, evidenceQuote=part,
                                    evidenceTurnId=item.input_id),
                        "factKey": {"subject": "user", "predicate": f"candidate-{index}"}}
                fact.pop("content")
                facts.append(fact)
            return {"facts": {"operations": facts},
                    "factCoverage": [{"segmentId": index, "disposition": "candidate"}
                                     for index in range(2)],
                    "intents": {"coverage": "complete", "acts": [
                        {"segmentId": index,
                         "evidenceQuote": part,
                         "intent": "statement" if index == 0 else "question",
                         "targetId": None}
                        for index, part in enumerate(parts)]}}
        if stage == "fact_assess":
            return {"assessments": [
                {"candidateIndex": 0, "disposition": "rejected", "reason": "transient"},
                {"candidateIndex": 1, "disposition": "accepted", "reason": "fact"}],
                "sourceKinds": [{"segmentId": 0, "kind": "no_fact"},
                                {"segmentId": 1, "kind": "fact"}]}
        if stage == "fact_support":
            assert data["proposal"]["evidenceQuote"] == parts[1]
            return {"intent": "question", "verdict": "supported"}
        if stage == "exchange_extract":
            return {"operations": [{"operation": "add", "type": "response_obligation",

                                    "evidenceQuote": parts[1],
                                    "supersedesAtomId": None, "expiresAt": None}]}
        if stage == "exchange_support":
            return {"intent": "request", "verdict": "supported"}
        raise AssertionError(stage)

    receipt = json.loads(UserFactWorker(
        store, model, clock=lambda: 10, resolution_protocol=True,
        resolution_strategy="intent_first", joint_proposals=True,
    ).run_one(A, "worker"))
    assert receipt["fact_resolution"]["dispositions"] == ["rejected", "rejected"]
    assert receipt["fact_resolution"]["segment_kinds"] == ["no_fact", "no_fact"]
    assert receipt["operations"] == 0
    assert not store.memory_barrier(A)


def question_store(tmp_path):
    store = SQLiteResolvedBackgroundStore(tmp_path / "answer.sqlite3")
    item = store.accept(config(), "greeting", "你好")

    def greeting(stage, messages, schema, cancelled):
        if stage == "exchange_intent":
            return {"coverage": "complete", "acts": [
                {"segmentId": 0,  "evidenceQuote": "你好",
                 "intent": "statement", "targetId": None}]}
        return {"operations": []}

    UserFactWorker(store, greeting, clock=lambda: 8, resolution_protocol=True,
                   resolution_strategy="intent_first").run_one(A, "seed")
    run = store.claim(A, item.input_id, "turn", now_us=10, lease_us=100)
    said = "你今天想聊什么？"
    candidate = Presentation(Scope(A, item.session_id), item.input_id, run.attempt_id,
                             "c1", 1, said, (MessageUnit("one", "text", said),))
    receipt = store.commit(CognitiveResult(
        run, "c1", 1, said, "review", sha256(said.encode()).hexdigest(),
        presentation=approved(candidate)), now_us=11)
    store.publish(A, receipt["delivery_id"])
    store.acknowledge(A, receipt["delivery_id"], candidate_id="c1", revision=1,
                      sha256=receipt["sha256"], presentation_sha256=receipt["presentation_sha256"],
                      sequence=1, displayed_end=len(said))

    def question(stage, messages, schema, cancelled):
        if stage == "exchange_intent":
            return {"coverage": "complete", "acts": [
                {"segmentId": 0,
                 "evidenceQuote": said, "intent": "question", "targetId": None}]}
        if stage == "exchange_extract":
            return {"operations": [{"operation": "add", "type": "open_loop",

                                    "evidenceQuote": said, "supersedesAtomId": None,
                                    "expiresAt": None}]}
        return {"intent": "question", "verdict": "supported"}

    AssistantExchangeWorker(store, question, clock=lambda: 12, resolution_protocol=True,
                            resolution_strategy="intent_first").run_one(A, "seed")
    return store, item.session_id


@pytest.mark.parametrize("kind", ["answer", "mixed", "quotation", "cancel", "wrong_target",
                                  "false_fact", "unresolved_fact", "mixed_omitted", "tampered"])
def test_joint_answer_preserves_independent_fact_and_rejects_false_resolution(tmp_path, kind):
    store, session = question_store(tmp_path)
    quote = "聊旅行吧" if kind != "quotation" else "“聊旅行吧”只是别人说的话。"
    text = quote + ("。我住在北京。" if kind in {"mixed", "mixed_omitted"} else "")
    item = store.accept(config(), "answer", text, session_id=session)
    target = store.pending_exchanges(A, now_us=13)[0]["exchange_id"]
    stages = []
    completions = []
    original_finish = store.finish_resolved_background

    def capture_finish(*args, **kwargs):
        completions.append((args, kwargs))
        if kind == "tampered":
            kwargs["fact_resolution"] = replace(kwargs["fact_resolution"], context_sha256="0" * 64)
        return original_finish(*args, **kwargs)

    store.finish_resolved_background = capture_finish

    def model(stage, messages, schema, cancelled):
        stages.append(stage)
        data = json.loads(messages[-1]["content"])
        if stage == "source_propose":
            assert any(s["text"] == "你今天想聊什么？" for s in data["sources"])
            assert data["targets"][0]["exchange_id"] == target
            assert schema["properties"]["intents"]["properties"]["acts"]["items"][
                "properties"]["targetId"] == {"enum": [None, target]}
            facts = []
            acts = [{"segmentId": 0,
                     "evidenceQuote": quote, "intent": "answer",
                     "targetId": "invented" if kind == "wrong_target" else target}]
            coverage = [{"segmentId": 0, "disposition": "none"}]
            if kind in {"false_fact", "unresolved_fact"}:
                fact = {**operation(content=quote, evidenceQuote=quote,
                                   evidenceTurnId=item.input_id),
                        "factKey": {"subject": "user", "predicate": "topic"}}
                fact.pop("content")
                facts.append(fact)
                coverage[0]["disposition"] = "candidate"
            if kind in {"mixed", "mixed_omitted"}:
                fact = {**operation(content="我住在北京。", evidenceQuote="我住在北京。",
                                   evidenceTurnId=item.input_id),
                        "factKey": {"subject": "user", "predicate": "home"}}
                fact.pop("content")
                if kind == "mixed":
                    facts.append(fact)
                acts.append({"segmentId": 1,
                             "evidenceQuote": "我住在北京。", "intent": "statement",
                             "targetId": None})
                coverage.append({"segmentId": 1, "disposition":
                                 "candidate" if kind == "mixed" else "none"})
            if kind == "cancel":
                store.accept(config(), "newer", "先等一下", session_id=session)
            return {"facts": {"operations": facts},
                    "intents": {"coverage": "complete", "acts": acts},
                    "factCoverage": coverage}
        if stage == "source_repair":
            return data["rejectedProposal"]
        if stage == "fact_assess":
            Draft202012Validator.check_schema(schema)
            assessments = [
                {"candidateIndex": i,
                 "disposition": "unresolved" if kind == "unresolved_fact" else
                 "rejected" if kind == "false_fact" else "accepted",
                 "reason": "unclear" if kind == "unresolved_fact" else
                 "transient" if kind == "false_fact" else "fact"}
                for i, _ in enumerate(data["candidates"])]
            return {"assessments": assessments, "sourceKinds": [
                {"segmentId": s["id"], "kind": "fact"
                 if kind in {"mixed", "mixed_omitted"} and s["id"] == 1
                 else "unknown" if kind == "unresolved_fact" else "no_fact"}
                for s in data["segments"]]}
        if stage == "fact_support":
            return {"intent": "assertion", "verdict": "supported"}
        if stage == "exchange_extract":
            return {"operations": [{"operation": "resolve", "type": None,
                                     "evidenceQuote": quote,
                                    "supersedesAtomId": target, "expiresAt": None}]}
        if stage == "exchange_support":
            return {"intent": "quotation" if kind == "quotation" else "answer",
                    "verdict": "unsupported" if kind == "quotation" else "supported"}
        raise AssertionError(stage)

    worker = UserFactWorker(store, model, clock=lambda: 14, resolution_protocol=True,
                            resolution_strategy="intent_first", joint_proposals=True)
    blocked = {"quotation", "cancel", "wrong_target", "unresolved_fact",
               "mixed_omitted", "tampered"}
    if kind in blocked - {"cancel"}:
        with pytest.raises(ValueError):
            worker.run_one(A, "worker")
    else:
        result = worker.run_one(A, "worker")
        assert (result is None) == (kind == "cancel")
    reopened = SQLiteResolvedBackgroundStore(store.path)
    assert len(reopened.pending_exchanges(A, now_us=15)) == int(
        kind in blocked)
    assert reopened.memory_barrier(A) == (
        kind in blocked)
    assert len(reopened.recall(A, now_us=15)) == int(kind == "mixed")
    assert "fact_extract" not in stages and "exchange_intent" not in stages
    if kind in {"answer", "mixed"}:
        assert worker.run_one(A, "worker") is None
    if kind == "wrong_target":
        assert stages == ["source_propose", "source_repair"]
    if kind == "false_fact":
        assert "fact_support" not in stages
        receipt = json.loads(result)
        assert receipt["fact_resolution"]["dispositions"] == ["rejected"]
        assert receipt["fact_resolution"]["reasons"] == ["transient"]
    if kind in {"answer", "mixed", "false_fact"}:
        args, kwargs = completions[-1]
        assert reopened.finish_resolved_background(*args, **kwargs) == result
        with pytest.raises(ValueError, match="fact source resolution"):
            reopened.finish_resolved_background(*args, **{**kwargs, "fact_resolution": None})
        changed = replace(kwargs["fact_resolution"], commands_sha256="0" * 64)
        with pytest.raises(VersionConflict, match="replay"):
            reopened.finish_resolved_background(*args, **{**kwargs, "fact_resolution": changed})
