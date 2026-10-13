"""RF2 source completion must not mistake candidate rejection for source coverage."""

import json
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
from threading import Event

import pytest
from test_evidence_memory import operation
from test_rebuilt_presentation import approved
from test_rebuilt_turns import A, config
from virtual_companion.boundary.contracts import Scope
from virtual_companion.boundary.evidence import EvidenceTurn
from virtual_companion.boundary.exchange_resolution import digest
from virtual_companion.boundary.memory import FactKey, MemoryMutation
from virtual_companion.boundary.presentation import MessageUnit, Presentation
from virtual_companion.boundary.sqlite_background_worker import (
    SQLiteBackgroundWorkerStore,
    SQLiteResolvedBackgroundStore,
)
from virtual_companion.boundary.turns import CognitiveResult
from virtual_companion.cognition.background_worker import AssistantExchangeWorker, UserFactWorker
from virtual_companion.cognition.exchange_model import INTENT_RESOLUTION_VERSION, ExchangeModel


@pytest.mark.parametrize("named_target", ["previous-question", "foreign-question"])
def test_question_target_is_a_reference_not_permission_to_resolve_the_prior_request(named_target):
    from copy import deepcopy

    text = "按我现在的音乐偏好，有什么适合听着放松的？"
    scope = Scope(A, "session")
    source = EvidenceTurn(scope, "current-query", "user", text, len(text))
    target = {"companion_id": A, "exchange_id": "previous-question", "revision": 1,
              "direction": "assistant_response",
              "content": "依据以前的偏好，有什么适合听着放松的？",
              "status": "pending", "expires_us": None}
    targets = {target["exchange_id"]: target}
    original, stages, extraction = deepcopy(targets), [], []

    def invoke(stage, messages, schema, cancelled):
        stages.append(stage)
        data = json.loads(messages[-1]["content"])
        if stage == "exchange_intent":
            return {"acts": [{"segmentId": 0, "evidenceQuote": text,
                              "intent": "question", "targetId": named_target}],
                    "coverage": "complete"}
        if stage == "exchange_extract":
            extraction.append(data)
            return {"operations": [{"operation": "add", "type": "response_obligation",
                                    "evidenceQuote": text, "supersedesAtomId": None,
                                    "expiresAt": None}]}
        assert stage == "exchange_support"
        assert data["action"] == "open" and data["pending_for"] == "assistant"
        assert data["evidence_quote"] == data["proposed_content"] == text
        return {"intent": "question", "verdict": "supported"}

    model = ExchangeModel(invoke, Event())
    params = dict(scope=scope, batch_id=source.source_id, now_us=100, turns=(source,),
                  targets=targets, eligible_source_ids=(source.source_id,), strategy="intent_first")
    if named_target == "foreign-question":
        with pytest.raises(ValueError, match="bound schema"):
            model.resolve_source(**params)
        assert stages == ["exchange_intent"]
        assert targets == original
        return
    prepared = model.prepare_source_intents(
        turns=(source,), targets=targets, eligible_source_ids=(source.source_id,))
    effects, resolution = model.resolve_source(**params, _prepared_intents=prepared)
    assert json.loads(prepared.interpretation)["acts"][0]["targetId"] == named_target
    assert extraction[0]["independent_acts"][0]["targetId"] is None
    command, _ = effects[0]
    assert len(effects) == 1 and command.action == "open"
    assert command.direction == "assistant_response" and command.content == text
    assert command.exchange_id != named_target and command.evidence.source_id == source.source_id
    assert resolution.source_id == source.source_id and not resolution.no_exchange
    assert resolution.target_refs == ((named_target, 1),)
    assert resolution.evaluator == INTENT_RESOLUTION_VERSION
    assert stages == ["exchange_intent", "exchange_extract", "exchange_support"]
    assert targets == original  # User question never resolves its own prior response debt.


def test_referencing_different_targets_cannot_duplicate_one_nontransition_speech_act():
    from virtual_companion.boundary.contracts import SourceResolutionUnresolved

    scope = Scope(A, "session")
    text = "现在适合听什么？"
    source = EvidenceTurn(scope, "source", "user", text, len(text))
    stages = []

    def invoke(stage, messages, schema, cancelled):
        stages.append(stage)
        assert stage == "exchange_intent"
        return {"acts": [{"segmentId": 0, "evidenceQuote": text,
                          "intent": "question", "targetId": target}
                         for target in (None, "previous-question")], "coverage": "complete"}

    with pytest.raises(SourceResolutionUnresolved, match="independent intent coverage"):
        ExchangeModel(invoke, Event()).resolve_source(
            scope=scope, batch_id=source.source_id, now_us=100, turns=(source,),
            targets={"previous-question": {"exchange_id": "previous-question"}},
            eligible_source_ids=(source.source_id,), strategy="intent_first")
    assert stages == ["exchange_intent"]


@pytest.mark.parametrize("intent", ["question", "request"])
def test_context_then_question_role_error_uses_one_bound_repair_and_commits(tmp_path, intent):
    from copy import deepcopy

    context = "Background music matters;"
    question = "What suits a relaxed evening, without getting sleepy?"
    text = context + " " + question
    store = SQLiteResolvedBackgroundStore(tmp_path / "context-question.sqlite3")
    item = store.accept(config(), "input", text)
    claim = store.claim_background(A, "worker", now_us=10, lease_us=100, role="user")
    turns = store.background_exchange_context(claim)
    calls, returned = [], []

    def invoke(stage, messages, schema, cancelled):
        data = json.loads(messages[-1]["content"])
        calls.append((stage, deepcopy(data), deepcopy(schema)))
        if stage == "exchange_intent":
            assert [s["id"] for s in data["segments"]] == [0, 1, 2]
            return {"acts": [
                {"segmentId": 0, "endSegmentId": 0, "intent": "context", "targetIndex": None},
                {"segmentId": 1, "endSegmentId": 2, "intent": intent, "targetIndex": None},
            ], "coverage": "complete"}
        if stage == "exchange_extract":
            assert data["eligible_source_ids"] == [item.input_id]
            assert [act["intent"] for act in data["independent_acts"]] == [intent]
            assert data["independent_acts"][0]["evidenceQuote"] == question
            assert schema["properties"]["operations"]["items"]["properties"]["type"] == {
                "enum": ["response_obligation", None]}
            result = {"operations": [{
                "operation": "add", "type": (
                    "response_obligation" if "repair" in data else "open_loop"),
                "expiresAt": None, "targetIndex": None, "evidenceQuoteIndex": 0,
            }]}
            returned.append(deepcopy(result))
            return result
        assert stage == "exchange_support"
        assert data["source_role"] == "user" and data["pending_for"] == "assistant"
        assert data["confirmed_text"] == text and data["evidence_quote"] == question
        return {"intent": intent, "verdict": "supported"}

    invoke.supports_bound_evidence = True
    model = ExchangeModel(invoke, Event())
    prepared = model.prepare_source_intents(
        turns=turns, targets={}, eligible_source_ids=(item.input_id,))
    assert prepared.complete and prepared.fact_statement_quotes == ()
    effects, resolution = model.resolve_source(
        scope=Scope(A, item.session_id), batch_id=item.input_id, now_us=10,
        turns=turns, targets={}, eligible_source_ids=(item.input_id,), strategy="intent_first",
        _prepared_intents=prepared)

    assert [call[0] for call in calls] == [
        "exchange_intent", "exchange_extract", "exchange_extract", "exchange_support"]
    first, repaired = calls[1], calls[2]
    assert repaired[2] == first[2]
    assert {k: v for k, v in repaired[1].items() if k != "repair"} == first[1]
    assert repaired[1]["repair"]["previous"]["operations"][0]["type"] == "open_loop"
    assert [result["operations"][0]["type"] for result in returned] == [
        "open_loop", "response_obligation"]
    assert len(effects) == 1 and not resolution.no_exchange
    command, support = effects[0]
    assert command.action == "open" and command.direction == "assistant_response"
    assert command.content == command.evidence.quote == question
    assert command.evidence.role == "user" and command.evidence.source_id == item.input_id
    assert command.evidence.confirmed_end == len(text)
    assert (
        command.evidence.prefix_sha256
        == resolution.source_sha256 == sha256(text.encode()).hexdigest())
    assert support.command_sha256 == command.fingerprint
    assert resolution.source_id == item.input_id
    store.finish_resolved_background(
        claim, (), evaluator=INTENT_RESOLUTION_VERSION, now_us=11,
        exchanges=effects, sources=turns, resolution=resolution)
    assert source_status(store) == "completed" and not store.memory_barrier(A)
    pending = store.pending_exchanges(A, now_us=11)
    assert len(pending) == 1 and pending[0]["exchange_id"] == command.exchange_id
    assert pending[0]["direction"] == "assistant_response" and pending[0]["content"] == question
    with store.transaction(write=False) as db:
        assert db.execute("SELECT COUNT(*) FROM memory_versions").fetchone()[0] == 0
    assert source_status(SQLiteResolvedBackgroundStore(store.path)) == "completed"


@pytest.mark.parametrize("boundary", ["second_question", "incomplete_quote"])
def test_context_then_question_role_repair_rejects_extra_action_or_incomplete_quote(
        tmp_path, boundary):
    context = "Background music matters;"
    question = "What suits a relaxed evening, without getting sleepy?"
    extra = "Which track comes next?" if boundary == "second_question" else ""
    text = context + " " + question + extra
    store = SQLiteResolvedBackgroundStore(tmp_path / "rejected-context-question.sqlite3")
    item = store.accept(config(), "input", text)
    claim = store.claim_background(A, "worker", now_us=10, lease_us=100, role="user")
    turns = store.background_exchange_context(claim)
    stages = []

    def invoke(stage, messages, schema, cancelled):
        stages.append(stage)
        if stage == "exchange_intent":
            return {"acts": [
                {"segmentId": 0, "evidenceQuote": context, "intent": "context", "targetId": None},
                {"segmentId": 1, "evidenceQuote": question, "intent": "question", "targetId": None},
                *([{"segmentId": 3, "evidenceQuote": extra, "intent": "question", "targetId": None}]
                  if extra else []),
            ], "coverage": "complete"}
        assert stage == "exchange_extract"
        assert "repair" not in json.loads(messages[-1]["content"])
        return {"operations": [{
            "operation": "add", "type": "open_loop", "expiresAt": None,
            "evidenceQuote": question[:-1] if boundary == "incomplete_quote" else question,
            "supersedesAtomId": None,
        }]}

    with pytest.raises(ValueError, match="bound schema"):
        ExchangeModel(invoke, Event()).resolve_source(
            scope=Scope(A, item.session_id), batch_id=item.input_id, now_us=10,
            turns=turns, targets={}, eligible_source_ids=(item.input_id,), strategy="intent_first")
    assert stages == ["exchange_intent", "exchange_extract"]
    assert source_status(store) == "running" and store.memory_barrier(A)
    assert store.pending_exchanges(A, now_us=11) == []
    with store.transaction(write=False) as db:
        assert db.execute("SELECT COUNT(*) FROM memory_versions").fetchone()[0] == 0


def seeded(tmp_path, *, mixed):
    store = SQLiteResolvedBackgroundStore(tmp_path / "source.sqlite3")
    store.register(A)
    owner = store.claim_owner(A, "seed", expected_epoch=0)
    store.mutate(
        MemoryMutation(
            Scope(A, "seed"),
            "seed",
            "assert_fact",
            "old-drink",
            "我喜欢茶",
            "manual",
            "seed-source",
            "seed",
            datetime.fromtimestamp(0, UTC),
            FactKey("user", "drink", "profile"),
        ),
        owner,
        expected_version=0,
    )
    correction = "更正，我现在不喜欢茶"
    request = "请给我推荐不含茶的饮品。"
    text = correction + ("。" + request if mixed else "")
    item = store.accept(config(), "input", text)
    return store, item, correction, request


def source_status(store):
    with store.transaction(write=False) as db:
        return db.execute("SELECT status FROM background_sources").fetchone()["status"]


def test_intent_first_does_not_silently_discard_a_conflicting_candidate(tmp_path):
    store, item, correction, _ = seeded(tmp_path, mixed=False)
    source = store.background_exchange_context(
        store.claim_background(A, "worker", now_us=10, lease_us=100, role="user")
    )
    stages = []

    def model(stage, messages, schema, cancelled):
        stages.append(stage)
        data = json.loads(messages[-1]["content"])
        if stage == "exchange_intent":
            assert "candidates" not in data and "operations" not in data
            return {"acts": [{"segmentId": 0,
                              "evidenceQuote": correction, "intent": "statement",
                              "targetId": None}], "coverage": "complete"}
        if stage == "exchange_extract":
            return {"operations": [{"operation": "add", "type": "response_obligation",

                                    "evidenceQuote": correction,
                                    "supersedesAtomId": None, "expiresAt": None}]}
        if stage == "exchange_support":
            return {"intent": "unclear", "verdict": "unknown"}
        raise AssertionError(stage)

    with pytest.raises(ValueError, match="contradictory.*coverage"):
        ExchangeModel(model, Event()).resolve_source(
            scope=Scope(A, item.session_id), batch_id=item.input_id, now_us=10,
            turns=source, targets={}, eligible_source_ids=(item.input_id,),
            strategy="intent_first",
        )
    assert stages == ["exchange_intent", "exchange_extract", "exchange_support"]


def test_intent_first_detects_mixed_request_missing_from_candidates(tmp_path):
    store, item, correction, request = seeded(tmp_path, mixed=True)
    source = store.background_exchange_context(
        store.claim_background(A, "worker", now_us=10, lease_us=100, role="user")
    )

    def model(stage, messages, schema, cancelled):
        if stage == "exchange_intent":
            return {"acts": [{"segmentId": 0,
                              "evidenceQuote": correction, "intent": "statement",
                              "targetId": None},
                             {"segmentId": 1,
                              "evidenceQuote": request, "intent": "request",
                              "targetId": None}], "coverage": "complete"}
        if stage == "exchange_extract":
            return {"operations": []}
        raise AssertionError(stage)

    with pytest.raises(ValueError, match="coverage"):
        ExchangeModel(model, Event()).resolve_source(
            scope=Scope(A, item.session_id), batch_id=item.input_id, now_us=10,
            turns=source, targets={}, eligible_source_ids=(item.input_id,),
            strategy="intent_first",
        )


def test_intent_first_candidate_quote_is_bound_to_prior_independent_acts(tmp_path):
    store, item, correction, request = seeded(tmp_path, mixed=True)
    source = store.background_exchange_context(
        store.claim_background(A, "worker", now_us=10, lease_us=100, role="user")
    )

    def model(stage, messages, schema, cancelled):
        if stage == "exchange_intent":
            return {"acts": [
                {"segmentId": 0,
                 "evidenceQuote": correction, "intent": "statement", "targetId": None},
                {"segmentId": 1,
                 "evidenceQuote": request, "intent": "request", "targetId": None},
            ], "coverage": "complete"}
        assert stage == "exchange_extract"
        data = json.loads(messages[-1]["content"])
        assert data["independent_acts"][1]["evidenceQuote"] == request
        allowed = schema["properties"]["operations"]["items"]["properties"]["evidenceQuote"]["enum"]
        assert set(allowed) == {correction, request}
        return {"operations": [{"operation": "add", "type": "response_obligation",

                                "evidenceQuote": correction + "。" + request,
                                "supersedesAtomId": None, "expiresAt": None}]}

    with pytest.raises(ValueError, match="bound schema"):
        ExchangeModel(model, Event()).resolve_source(
            scope=Scope(A, item.session_id), batch_id=item.input_id, now_us=10,
            turns=source, targets={}, eligible_source_ids=(item.input_id,),
            strategy="intent_first",
        )


def test_whole_statement_cannot_silently_erase_an_extracted_request(tmp_path):
    store, item, correction, request = seeded(tmp_path, mixed=True)
    source = store.background_exchange_context(
        store.claim_background(A, "worker", now_us=10, lease_us=100, role="user")
    )

    def model(stage, messages, schema, cancelled):
        if stage == "exchange_intent":
            data = json.loads(messages[-1]["content"])
            assert [part["text"] for part in data["segments"]] == [
                correction + "。", request]
            return {"acts": [{"segmentId": 0,
                              "evidenceQuote": correction + "。" + request,
                              "intent": "statement", "targetId": None}],
                    "coverage": "complete"}
        if stage == "exchange_support":
            return {"intent": "unclear", "verdict": "unknown"}
        assert stage == "exchange_extract"
        return {"operations": [{"operation": "add", "type": "response_obligation",

                                "evidenceQuote": correction + "。" + request,
                                "supersedesAtomId": None, "expiresAt": None}]}

    with pytest.raises(ValueError, match="contradictory.*coverage"):
        ExchangeModel(model, Event()).resolve_source(
            scope=Scope(A, item.session_id), batch_id=item.input_id, now_us=10,
            turns=source, targets={}, eligible_source_ids=(item.input_id,),
            strategy="intent_first",
        )


def test_comma_joined_statement_and_request_conflict_is_not_no_exchange():
    scope = Scope(A, "session")
    text = "我不喜欢茶了，推荐别的吧。"
    source = EvidenceTurn(scope, "source", "user", text, len(text))

    def model(stage, messages, schema, cancelled):
        if stage == "exchange_intent":
            return {"acts": [{"segmentId": 0,
                              "evidenceQuote": text, "intent": "statement",
                              "targetId": None}], "coverage": "complete"}
        if stage == "exchange_support":
            return {"intent": "unclear", "verdict": "unknown"}
        assert stage == "exchange_extract"
        return {"operations": [{"operation": "add", "type": "response_obligation",
                                 "evidenceQuote": text,
                                "supersedesAtomId": None, "expiresAt": None}]}

    with pytest.raises(ValueError, match="contradictory.*coverage"):
        ExchangeModel(model, Event()).resolve_source(
            scope=scope, batch_id="source", now_us=10, turns=(source,), targets={},
            eligible_source_ids=("source",), strategy="intent_first")


def test_intent_first_schema_binds_quote_to_current_source_before_extraction():
    scope = Scope(A, "session")
    source = EvidenceTurn(scope, "current", "user", "聊旅行吧", 4)
    prior = EvidenceTurn(scope, "old-question", "assistant", "你今天想聊什么？", 8)
    stages = []

    def model(stage, messages, schema, cancelled):
        stages.append(stage)
        assert stage == "exchange_intent"
        quote_schema = schema["properties"]["acts"]["items"]["properties"]["evidenceQuote"]
        assert quote_schema == {"enum": ["聊旅行吧"]}
        return {"acts": [{"segmentId": 0,
                          "evidenceQuote": prior.text, "intent": "answer",
                          "targetId": "question"}], "coverage": "complete"}

    with pytest.raises(ValueError, match="bound schema"):
        ExchangeModel(model, Event()).resolve_source(
            scope=scope, batch_id="current", now_us=10, turns=(prior, source),
            targets={"question": {"exchange_id": "question"}},
            eligible_source_ids=("current",), strategy="intent_first")
    assert stages == ["exchange_intent"]


def test_intent_first_keeps_quoted_clause_and_negation_together():
    scope = Scope(A, "session")
    text = "朋友说“请给我推荐饮品”，但我不需要。"
    source = EvidenceTurn(scope, "source", "user", text, len(text))

    def model(stage, messages, schema, cancelled):
        if stage == "exchange_intent":
            data = json.loads(messages[-1]["content"])
            assert [part["text"] for part in data["segments"]] == [text]
            return {"acts": [{"segmentId": 0,
                              "evidenceQuote": text, "intent": "decline",
                              "targetId": None}], "coverage": "complete"}
        return {"operations": []}

    effects, resolution = ExchangeModel(model, Event()).resolve_source(
        scope=scope, batch_id="source", now_us=10, turns=(source,), targets={},
        eligible_source_ids=("source",), strategy="intent_first")
    assert effects == () and resolution.no_exchange


@pytest.mark.parametrize("strategy", ["candidate_first", "intent_first"])
def test_repeated_pending_request_is_a_host_bound_no_change_completion(strategy):
    scope = Scope(A, "session")
    text = "请给我推荐不含茶的饮品。"
    source = EvidenceTurn(scope, "repeat", "user", text, len(text))
    targets = {
        "pending": {
            "companion_id": A,
            "exchange_id": "pending",
            "revision": 1,
            "direction": "assistant_response",
            "content": text,
            "status": "pending",
            "expires_us": None,
        }
    }
    stages = []

    def model(stage, messages, schema, cancelled):
        stages.append(stage)
        if stage == "exchange_intent":
            return {"acts": [{"segmentId": 0,
                              "evidenceQuote": text, "intent": "question",
                              "targetId": None}], "coverage": "complete"}
        if stage == "exchange_extract":
            return {"operations": []}
        raise AssertionError(stage)

    effects, resolution = ExchangeModel(model, Event()).resolve_source(
        scope=scope, batch_id=source.source_id, now_us=10, turns=(source,),
        targets=targets, eligible_source_ids=(source.source_id,), strategy=strategy,
    )

    assert effects == ()
    assert resolution.no_exchange
    assert resolution.status == "complete_no_exchange"
    assert stages == (["exchange_extract"] if strategy == "candidate_first"
                      else ["exchange_intent", "exchange_extract"])


def test_intent_first_worker_commits_supported_fact_and_real_request(tmp_path):
    store, item, correction, request = seeded(tmp_path, mixed=True)
    stages = []

    def model(stage, messages, schema, cancelled):
        stages.append(stage)
        if stage == "fact_extract":
            candidate = {**operation(
                operation="supersede", content=correction, evidenceQuote=correction,
                evidenceTurnId=item.input_id, supersedesAtomId="old-drink"),
                "factKey": None}
            candidate.pop("content")
            return {"operations": [candidate]}
        if stage == "fact_support":
            return {"intent": "correction", "verdict": "supported"}
        if stage == "exchange_intent":
            assert "candidates" not in json.loads(messages[-1]["content"])
            return {"acts": [
                {"segmentId": 0,
                 "evidenceQuote": correction, "intent": "statement", "targetId": None},
                {"segmentId": 1,
                 "evidenceQuote": request, "intent": "request", "targetId": None},
            ], "coverage": "complete"}
        if stage == "exchange_extract":
            return {"operations": [
                {"operation": "add", "type": "response_obligation",
                  "evidenceQuote": request,
                 "supersedesAtomId": None, "expiresAt": None}]}
        if stage == "exchange_support":
            assert json.loads(messages[-1]["content"])["evidence_quote"] == request
            return {"intent": "request", "verdict": "supported"}
        raise AssertionError(stage)

    receipt = UserFactWorker(store, model, clock=lambda: 10,
                             resolution_protocol=True,
                             resolution_strategy="intent_first").run_one(A, "worker")
    assert stages == ["fact_extract", "fact_support", "exchange_intent",
                      "exchange_extract", "exchange_support"]
    assert source_status(store) == "completed" and not store.memory_barrier(A)
    assert len(store.history(A, "old-drink")) == 2
    assert json.loads(receipt)["evaluator"] == "fact-model.v7"
    pending = store.pending_exchanges(A, now_us=10)
    assert len(pending) == 1 and pending[0]["content"] == request
    assert json.loads(receipt)["resolution"]["evaluator"] == INTENT_RESOLUTION_VERSION


def test_intent_first_keeps_assistant_question_direction():
    scope = Scope(A, "session")
    source = EvidenceTurn(scope, "assistant-source", "assistant", "你今天好吗？", 6)
    stages = []

    def model(stage, messages, schema, cancelled):
        stages.append(stage)
        if stage == "exchange_intent":
            return {"acts": [{"segmentId": 0,
                              "evidenceQuote": source.text, "intent": "question",
                              "targetId": None}], "coverage": "complete"}
        if stage == "exchange_extract":
            return {"operations": [{"operation": "add", "type": "open_loop",

                                    "evidenceQuote": source.text,
                                    "supersedesAtomId": None, "expiresAt": None}]}
        if stage == "exchange_support":
            assert set(schema["properties"]["intent"]["enum"]) == {
                "question", "request", "other", "unclear",
            }
            return {"intent": "question", "verdict": "supported"}
        raise AssertionError(stage)

    effects, resolution = ExchangeModel(model, Event()).resolve_source(
        scope=scope, batch_id=source.source_id, now_us=10, turns=(source,),
        targets={}, eligible_source_ids=(source.source_id,), strategy="intent_first")
    assert len(effects) == 1 and not resolution.no_exchange
    assert effects[0][0].direction == "user_question"
    assert stages == ["exchange_intent", "exchange_extract", "exchange_support"]


def test_assistant_request_label_cannot_erase_actual_question_candidate():
    scope = Scope(A, "session")
    said = "你今天想聊什么？"
    source = EvidenceTurn(scope, "assistant-source", "assistant", said, len(said))
    stages = []

    def model(stage, messages, schema, cancelled):
        stages.append(stage)
        if stage == "exchange_intent":
            return {"acts": [{"segmentId": 0,
                              "evidenceQuote": said, "intent": "request",
                              "targetId": None}], "coverage": "complete"}
        if stage == "exchange_extract":
            return {"operations": [{"operation": "add", "type": "open_loop",

                                    "evidenceQuote": said,
                                    "supersedesAtomId": None, "expiresAt": None}]}
        if stage == "exchange_support":
            return {"intent": "question", "verdict": "supported"}
        raise AssertionError(stage)

    effects, resolution = ExchangeModel(model, Event()).resolve_source(
        scope=scope, batch_id=source.source_id, now_us=10, turns=(source,),
        targets={}, eligible_source_ids=(source.source_id,), strategy="intent_first",
    )
    assert len(effects) == 1 and effects[0][0].direction == "user_question"
    assert not resolution.no_exchange
    assert stages == ["exchange_intent", "exchange_extract", "exchange_support"]


def test_assistant_request_label_without_candidate_cannot_complete_source():
    scope = Scope(A, "session")
    said = "你今天想聊什么？"
    source = EvidenceTurn(scope, "assistant-source", "assistant", said, len(said))

    def model(stage, messages, schema, cancelled):
        if stage == "exchange_intent":
            return {"acts": [{"segmentId": 0,
                              "evidenceQuote": said, "intent": "request",
                              "targetId": None}], "coverage": "complete"}
        assert stage == "exchange_extract"
        return {"operations": []}

    with pytest.raises(ValueError, match="coverage"):
        ExchangeModel(model, Event()).resolve_source(
            scope=scope, batch_id=source.source_id, now_us=10, turns=(source,),
            targets={}, eligible_source_ids=(source.source_id,), strategy="intent_first",
        )


def test_pure_correction_rejected_candidate_can_finish_with_evidenced_no_exchange(tmp_path):
    store, item, correction, _ = seeded(tmp_path, mixed=False)
    stages = []

    def model(stage, messages, schema, cancelled):
        stages.append(stage)
        if stage == "fact_extract":
            return {
                "operations": [
                    {
                        **operation(
                            operation="supersede",
                            content=correction,
                            evidenceQuote=correction,
                            evidenceTurnId=item.input_id,
                            supersedesAtomId="old-drink",
                        ),
                        "factKey": None,
                    }
                ]
            }
        if stage == "fact_support":
            return {"intent": "correction", "verdict": "supported"}
        if stage == "exchange_extract":
            return {
                "operations": [
                    {
                        "operation": "add",
                        "type": "response_obligation",

                        "evidenceQuote": correction,
                        "supersedesAtomId": None,
                        "expiresAt": None,
                    }
                ]
            }
        if stage == "exchange_assess":
            data = json.loads(messages[-1]["content"])
            candidate = data["candidates"][0]
            return {
                "assessments": [
                    {
                        "candidate_id": candidate["candidate_id"],
                        "disposition": "rejected",
                        "reason": "statement",
                    }
                ],
                "coverage": "complete",
                "uncovered": [],
                "no_exchange": True,
            }
        raise AssertionError(stage)

    assert UserFactWorker(store, model, clock=lambda: 10, resolution_protocol=True).run_one(
        A, "worker"
    )
    assert stages == ["fact_extract", "fact_support", "exchange_extract", "exchange_assess"]
    assert source_status(store) == "completed"
    assert not store.memory_barrier(A)
    assert len(store.history(A, "old-drink")) == 2
    assert store.pending_exchanges(A, now_us=10) == []


def test_mixed_request_omitted_by_extractor_cannot_complete_source(tmp_path):
    store, item, correction, request = seeded(tmp_path, mixed=True)
    stages = []

    def model(stage, messages, schema, cancelled):
        stages.append(stage)
        if stage == "fact_extract":
            return {
                "operations": [
                    {
                        **operation(
                            operation="supersede",
                            content=correction,
                            evidenceQuote=correction,
                            evidenceTurnId=item.input_id,
                            supersedesAtomId="old-drink",
                        ),
                        "factKey": None,
                    }
                ]
            }
        if stage == "fact_support":
            return {"intent": "correction", "verdict": "supported"}
        if stage == "exchange_extract":
            return {"operations": []}
        if stage == "exchange_assess":
            return {
                "assessments": [],
                "coverage": "incomplete",
                "uncovered": [{"evidenceTurnId": item.input_id, "evidenceQuote": request}],
                "no_exchange": False,
            }
        raise AssertionError(stage)

    with pytest.raises(ValueError, match="coverage"):
        UserFactWorker(store, model, clock=lambda: 10, resolution_protocol=True).run_one(
            A, "worker"
        )
    assert stages == ["fact_extract", "fact_support", "exchange_extract", "exchange_assess"]
    assert source_status(store) == "failed"
    source = store.background_sources(A)[0]
    assert source["error"] == "needs_clarification.source_coverage"
    assert store.memory_status(A) == "needs_clarification"
    assert store.memory_barrier(A)
    assert len(store.history(A, "old-drink")) == 1
    assert store.pending_exchanges(A, now_us=10) == []


def test_mixed_rejected_correction_keeps_real_request_and_original_candidate_id(tmp_path):
    store, item, correction, request = seeded(tmp_path, mixed=True)

    def model(stage, messages, schema, cancelled):
        if stage == "fact_extract":
            return {
                "operations": [
                    {
                        **operation(
                            operation="supersede",
                            content=correction,
                            evidenceQuote=correction,
                            evidenceTurnId=item.input_id,
                            supersedesAtomId="old-drink",
                        ),
                        "factKey": None,
                    }
                ]
            }
        if stage == "fact_support":
            return {"intent": "correction", "verdict": "supported"}
        if stage == "exchange_extract":
            return {
                "operations": [
                    {
                        "operation": "add",
                        "type": "response_obligation",

                        "evidenceQuote": quote,
                        "supersedesAtomId": None,
                        "expiresAt": None,
                    }
                    for quote in (correction, request)
                ]
            }
        if stage == "exchange_assess":
            candidates = json.loads(messages[-1]["content"])["candidates"]
            return {
                "assessments": [
                    {
                        "candidate_id": candidates[0]["candidate_id"],
                        "disposition": "rejected",
                        "reason": "statement",
                    },
                    {
                        "candidate_id": candidates[1]["candidate_id"],
                        "disposition": "accepted",
                        "reason": "supported",
                    },
                ],
                "coverage": "complete",
                "uncovered": [],
                "no_exchange": False,
            }
        if stage == "exchange_support":
            assert json.loads(messages[-1]["content"])["evidence_quote"] == request
            return {"intent": "request", "verdict": "supported"}
        raise AssertionError(stage)

    receipt = UserFactWorker(store, model, clock=lambda: 10, resolution_protocol=True).run_one(
        A, "worker"
    )
    assert receipt
    assert source_status(store) == "completed"
    assert not store.memory_barrier(A)
    pending = store.pending_exchanges(A, now_us=10)
    assert len(pending) == 1 and pending[0]["content"] == request
    assert pending[0]["exchange_id"] == f"{item.input_id}-exchange-1"
    assert len(store.history(A, "old-drink")) == 2
    assert json.loads(receipt)["resolution"]["assessments"][0]["disposition"] == "rejected"


@pytest.mark.parametrize("case", ["unresolved", "duplicate", "false_complete"])
def test_unsettled_or_unbound_assessment_never_completes(tmp_path, case):
    store, item, correction, _ = seeded(tmp_path, mixed=False)

    def model(stage, messages, schema, cancelled):
        if stage == "fact_extract":
            return {
                "operations": [
                    {
                        **operation(
                            operation="supersede",
                            content=correction,
                            evidenceQuote=correction,
                            evidenceTurnId=item.input_id,
                            supersedesAtomId="old-drink",
                        ),
                        "factKey": None,
                    }
                ]
            }
        if stage == "fact_support":
            return {"intent": "correction", "verdict": "supported"}
        if stage == "exchange_extract":
            return {
                "operations": [
                    {
                        "operation": "add",
                        "type": "response_obligation",

                        "evidenceQuote": correction,
                        "supersedesAtomId": None,
                        "expiresAt": None,
                    }
                ]
            }
        if stage == "exchange_assess":
            candidate = json.loads(messages[-1]["content"])["candidates"][0]
            identifier = candidate["candidate_id"]
            if case == "duplicate":
                return {
                    "assessments": [
                        {
                            "candidate_id": identifier,
                            "disposition": "rejected",
                            "reason": "statement",
                        },
                        {
                            "candidate_id": identifier,
                            "disposition": "rejected",
                            "reason": "statement",
                        },
                    ],
                    "coverage": "complete",
                    "uncovered": [],
                    "no_exchange": True,
                }
            return {
                "assessments": [
                    {
                        "candidate_id": identifier,
                        "disposition": "unresolved" if case == "unresolved" else "rejected",
                        "reason": "unclear" if case == "unresolved" else "statement",
                    }
                ],
                "coverage": "complete" if case == "unresolved" else "unknown",
                "uncovered": [],
                "no_exchange": True,
            }
        raise AssertionError(stage)

    with pytest.raises(ValueError):
        UserFactWorker(store, model, clock=lambda: 10, resolution_protocol=True).run_one(
            A, "worker"
        )
    expected = "retryable" if case == "duplicate" else "failed"
    assert source_status(store) == expected
    if expected == "failed":
        assert store.background_sources(A)[0]["error"] == (
            "needs_clarification.source_coverage"
        )
    assert store.memory_barrier(A)
    assert len(store.history(A, "old-drink")) == 1


def test_v16_receipt_is_required_replay_bound_and_rejects_legacy_open(tmp_path):
    path = tmp_path / "resolved.sqlite3"
    store = SQLiteResolvedBackgroundStore(path)
    item = store.accept(config(), "input", "你好")
    claim = store.claim_background(A, "worker", now_us=10, lease_us=100, role="user")
    sources = store.background_exchange_context(claim)

    def model(stage, messages, schema, cancelled):
        if stage == "exchange_extract":
            return {"operations": []}
        if stage == "exchange_assess":
            return {"assessments": [], "coverage": "complete", "uncovered": [], "no_exchange": True}
        raise AssertionError(stage)

    effects, resolution = ExchangeModel(model, Event()).resolve_source(
        scope=Scope(A, item.session_id),
        batch_id=item.input_id,
        now_us=10,
        turns=sources,
        targets={},
        eligible_source_ids=(item.input_id,),
    )
    assert effects == ()
    with pytest.raises(ValueError, match="requires a resolution"):
        store.finish_background(claim, (), evaluator="fact-model.v2", now_us=11)
    with pytest.raises(Exception, match="target snapshot"):
        store.finish_resolved_background(
            claim, (), evaluator="fact-model.v2", now_us=11,
            exchanges=effects, sources=sources,
            resolution=replace(resolution, targets_sha256=digest(["stale"])),
        )
    assert source_status(store) == "running" and store.memory_barrier(A)
    receipt = store.finish_resolved_background(
        claim,
        (),
        evaluator="fact-model.v2",
        now_us=11,
        exchanges=effects,
        sources=sources,
        resolution=resolution,
    )
    assert (
        store.finish_resolved_background(
            claim,
            (),
            evaluator="fact-model.v2",
            now_us=12,
            exchanges=effects,
            sources=sources,
            resolution=resolution,
        )
        == receipt
    )
    with pytest.raises(Exception, match="replay"):
        store.finish_resolved_background(
            claim,
            (),
            evaluator="fact-model.v2",
            now_us=12,
            exchanges=effects,
            sources=sources,
            resolution=replace(resolution, candidates_sha256=digest(["other"])),
        )
    with pytest.raises(ValueError):
        SQLiteBackgroundWorkerStore(path)
    legacy_path = tmp_path / "legacy.sqlite3"
    SQLiteBackgroundWorkerStore(legacy_path)
    with pytest.raises(ValueError):
        SQLiteResolvedBackgroundStore(legacy_path)
    assert not SQLiteResolvedBackgroundStore(path).memory_barrier(A)


@pytest.mark.parametrize("strategy", ["candidate_first", "intent_first"])
def test_v16_completion_fault_rolls_back_fact_and_source_then_reopens(tmp_path, strategy):
    store, item, correction, _ = seeded(tmp_path, mixed=False)

    def model(stage, messages, schema, cancelled):
        if stage == "fact_extract":
            candidate = {
                **operation(
                    operation="supersede",
                    content=correction,
                    evidenceQuote=correction,
                    evidenceTurnId=item.input_id,
                    supersedesAtomId="old-drink",
                ),
                "factKey": None,
            }
            if strategy == "intent_first":
                candidate.pop("content")
            return {
                "operations": [candidate]
            }
        if stage == "fact_support":
            return {"intent": "correction", "verdict": "supported"}
        if stage == "exchange_intent":
            return {"acts": [{"segmentId": 0,
                              "evidenceQuote": correction, "intent": "statement",
                              "targetId": None}], "coverage": "complete"}
        if stage == "exchange_extract":
            return {"operations": []}
        if stage == "exchange_assess":
            return {"assessments": [], "coverage": "complete", "uncovered": [], "no_exchange": True}
        raise AssertionError(stage)

    with store.transaction() as db:
        db.execute(
            "CREATE TRIGGER fail_completed BEFORE UPDATE ON background_sources "
            "WHEN NEW.status='completed' BEGIN SELECT RAISE(ABORT,'injected'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="injected"):
        UserFactWorker(store, model, clock=lambda: 10, resolution_protocol=True,
                       resolution_strategy=strategy).run_one(
            A, "worker"
        )
    assert len(store.history(A, "old-drink")) == 1
    assert source_status(store) == "retryable" and store.memory_barrier(A)
    with store.transaction() as db:
        db.execute("DROP TRIGGER fail_completed")
    reopened = SQLiteResolvedBackgroundStore(store.path)
    assert UserFactWorker(
        reopened, model, clock=lambda: 2_000_000, resolution_protocol=True,
        resolution_strategy=strategy,
    ).run_one(A, "worker")
    assert len(reopened.history(A, "old-drink")) == 2
    assert source_status(reopened) == "completed" and not reopened.memory_barrier(A)


def test_new_input_during_source_assessment_cancels_without_partial_write(tmp_path):
    store, item, correction, _ = seeded(tmp_path, mixed=False)
    once = False

    def model(stage, messages, schema, cancelled):
        nonlocal once
        if stage == "fact_extract":
            return {
                "operations": [
                    {
                        **operation(
                            operation="supersede",
                            content=correction,
                            evidenceQuote=correction,
                            evidenceTurnId=item.input_id,
                            supersedesAtomId="old-drink",
                        ),
                        "factKey": None,
                    }
                ]
            }
        if stage == "fact_support":
            return {"intent": "correction", "verdict": "supported"}
        if stage == "exchange_extract":
            return {"operations": []}
        if stage == "exchange_assess":
            if not once:
                once = True
                store.accept(config(), "later", "新的用户消息")
            return {"assessments": [], "coverage": "complete", "uncovered": [], "no_exchange": True}
        raise AssertionError(stage)

    assert (
        UserFactWorker(store, model, clock=lambda: 10, resolution_protocol=True).run_one(
            A, "worker"
        )
        is None
    )
    assert len(store.history(A, "old-drink")) == 1
    assert source_status(store) == "queued" and store.memory_barrier(A)


@pytest.mark.parametrize(
    "change",
    [
        {"type": "open_loop"},
        {"evidenceQuote": "编造的请求"},
        {"expiresAt": 0},
    ],
)
def test_even_rejected_candidate_cannot_bypass_role_quote_or_ttl(change):
    scope = Scope(A, "session")
    text = "更正，我不喜欢茶"
    stages = []

    def model(stage, messages, schema, cancelled):
        stages.append(stage)
        if stage == "exchange_extract":
            return {
                "operations": [
                    {
                        **{
                            "operation": "add",
                            "type": "response_obligation",

                            "evidenceQuote": text,
                            "supersedesAtomId": None,
                            "expiresAt": None,
                        },
                        **change,
                    }
                ]
            }
        raise AssertionError(stage)

    with pytest.raises(ValueError):
        ExchangeModel(model, Event()).resolve_source(
            scope=scope,
            batch_id="source",
            now_us=10,
            turns=(EvidenceTurn(scope, "source", "user", text, len(text)),),
            targets={},
            eligible_source_ids=("source",),
        )
    assert stages == ["exchange_extract"]


@pytest.mark.parametrize("strategy", ["candidate_first", "intent_first"])
def test_v16_assistant_ack_question_uses_same_resolution_and_domain_gates(tmp_path, strategy):
    store = SQLiteResolvedBackgroundStore(tmp_path / "assistant.sqlite3")
    item = store.accept(config(), "input", "你好")
    run = store.claim(A, item.input_id, "turn", now_us=10, lease_us=100)
    said = "你今天想聊什么？"
    candidate = Presentation(
        Scope(A, item.session_id),
        item.input_id,
        run.attempt_id,
        "c1",
        1,
        said,
        (MessageUnit("one", "text", said),),
    )
    delivery = store.commit(
        CognitiveResult(
            run,
            "c1",
            1,
            said,
            "review",
            sha256(said.encode()).hexdigest(),
            presentation=approved(candidate),
        ),
        now_us=11,
    )
    store.publish(A, delivery["delivery_id"])
    store.acknowledge(
        A,
        delivery["delivery_id"],
        candidate_id="c1",
        revision=1,
        sha256=delivery["sha256"],
        presentation_sha256=delivery["presentation_sha256"],
        sequence=1,
        displayed_end=len(said),
    )

    def model(stage, messages, schema, cancelled):
        data = json.loads(messages[-1]["content"])
        if stage == "exchange_intent":
            source = data["sources"][-1]
            return {"acts": [{"segmentId": 0,
                              "evidenceQuote": said, "intent": "question",
                              "targetId": None}], "coverage": "complete"}
        if stage == "exchange_extract":
            source = data["sources"][-1]
            assert source["role"] == "assistant" and source["text"] == said
            return {
                "operations": [
                    {
                        "operation": "add",
                        "type": "open_loop",

                        "evidenceQuote": said,
                        "supersedesAtomId": None,
                        "expiresAt": None,
                    }
                ]
            }
        if stage == "exchange_assess":
            return {
                "assessments": [
                    {
                        "candidate_id": data["candidates"][0]["candidate_id"],
                        "disposition": "accepted",
                        "reason": "supported",
                    }
                ],
                "coverage": "complete",
                "uncovered": [],
                "no_exchange": False,
            }
        if stage == "exchange_support":
            return {"intent": "question", "verdict": "supported"}
        raise AssertionError(stage)

    receipt = AssistantExchangeWorker(
        store,
        model,
        clock=lambda: 12,
        resolution_protocol=True,
        resolution_strategy=strategy,
    ).run_one(A, "assistant-worker")
    assert receipt and json.loads(receipt)["resolution"]["protocol"] == "source-resolution.v2"
    pending = store.pending_exchanges(A, now_us=13)
    assert len(pending) == 1 and pending[0]["direction"] == "user_question"
    with store.transaction(write=False) as db:
        row = db.execute("SELECT status FROM background_sources WHERE role='assistant'").fetchone()
    assert row["status"] == "completed"


def test_assessor_false_acceptance_cannot_overrule_independent_domain_support(tmp_path):
    store, item, correction, _ = seeded(tmp_path, mixed=False)
    stages = []

    def model(stage, messages, schema, cancelled):
        stages.append(stage)
        if stage == "fact_extract":
            return {
                "operations": [
                    {
                        **operation(
                            operation="supersede",
                            content=correction,
                            evidenceQuote=correction,
                            evidenceTurnId=item.input_id,
                            supersedesAtomId="old-drink",
                        ),
                        "factKey": None,
                    }
                ]
            }
        if stage == "fact_support":
            return {"intent": "correction", "verdict": "supported"}
        if stage == "exchange_extract":
            return {
                "operations": [
                    {
                        "operation": "add",
                        "type": "response_obligation",

                        "evidenceQuote": correction,
                        "supersedesAtomId": None,
                        "expiresAt": None,
                    }
                ]
            }
        if stage == "exchange_assess":
            candidate = json.loads(messages[-1]["content"])["candidates"][0]
            return {
                "assessments": [
                    {
                        "candidate_id": candidate["candidate_id"],
                        "disposition": "accepted",
                        "reason": "supported",
                    }
                ],
                "coverage": "complete",
                "uncovered": [],
                "no_exchange": False,
            }
        if stage == "exchange_support":
            return {"intent": "statement", "verdict": "unsupported"}
        raise AssertionError(stage)

    with pytest.raises(ValueError, match="support"):
        UserFactWorker(store, model, clock=lambda: 10, resolution_protocol=True).run_one(
            A, "worker"
        )
    assert stages[-1] == "exchange_support"
    assert source_status(store) == "retryable" and store.memory_barrier(A)
    assert len(store.history(A, "old-drink")) == 1
