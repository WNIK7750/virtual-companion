import json
import threading

import pytest
from jsonschema import Draft202012Validator
from test_exchange_extraction import operation as compiled_operation
from test_rebuilt_turns import A
from virtual_companion.boundary.contracts import Scope
from virtual_companion.cognition.evidence_memory import EvidenceTurn
from virtual_companion.cognition.exchange_model import ExchangeModel


def operation(**changes):
    result = compiled_operation(**changes)
    del result["content"]
    return result


def inputs():
    text = "周末想聊什么？"
    return dict(
        scope=Scope(A, "session"),
        batch_id="batch",
        now_us=100,
        turns=(
            EvidenceTurn(
                Scope(A, "session"), "assistant", "assistant", text + "不能发送的尾巴", len(text)
            ),
        ),
        targets={},
    )


class Model:
    def __init__(self, verdict="supported"):
        self.calls, self.verdict = [], verdict

    def __call__(self, stage, messages, schema, cancelled):
        self.calls.append((stage, messages, schema))
        if stage == "exchange_extract":
            return {"operations": [operation()]}
        return {
            "verdict": self.verdict,
            "intent": "question"
            if json.loads(messages[-1]["content"])["source_role"] == "assistant"
            else "request",
        }


def test_extraction_and_support_use_same_controlled_port_without_unshown_text():
    invoke = Model()
    proposals = ExchangeModel(invoke, threading.Event()).extract(**inputs())
    assert [call[0] for call in invoke.calls] == ["exchange_extract", "exchange_support"]
    assert len(proposals) == 1
    assert "不能发送的尾巴" not in str(invoke.calls)
    assert proposals[0][1].command_sha256 == proposals[0][0].fingerprint
    assert proposals[0][0].content == proposals[0][0].evidence.quote
    assert "content" not in invoke.calls[0][2]["properties"]["operations"]["items"]["properties"]
    assert json.loads(invoke.calls[1][1][-1]["content"])["evidence_quote"] == "周末想聊什么？"


@pytest.mark.parametrize("repair_quote", ["周末想聊什么？", "编造的请求"])
def test_statement_contradiction_has_one_grounded_repair(repair_quote):
    class Mixed(Model):
        supports = 0

        def __call__(self, stage, messages, schema, cancelled):
            if stage != "exchange_support":
                return super().__call__(stage, messages, schema, cancelled)
            self.supports += 1
            data = json.loads(messages[-1]["content"])
            if self.supports == 1:
                return {"intent": "statement", "verdict": "supported"}
            assert data["repair"]["previous"]["intent"] == "statement"
            return {"intent": "question", "verdict": "supported", "supporting_quote": repair_quote}

    model = Mixed()
    if repair_quote == "编造的请求":
        with pytest.raises(ValueError):
            ExchangeModel(model, threading.Event()).extract(**inputs())
    else:
        assert len(ExchangeModel(model, threading.Event()).extract(**inputs())) == 1
    assert model.supports == 2


@pytest.mark.parametrize("verdict", ["unknown", "unsupported", True])
def test_missing_support_never_becomes_a_storage_proposal(verdict):
    with pytest.raises(ValueError):
        ExchangeModel(Model(verdict), threading.Event()).extract(**inputs())


def test_repair_budget_is_shared_by_the_whole_batch():
    class Batch(Model):
        supports = 0

        def __call__(self, stage, messages, schema, cancelled):
            if stage == "exchange_extract":
                return {"operations": [operation(), operation()]}
            self.supports += 1
            if self.supports == 2:
                return {
                    "intent": "question",
                    "verdict": "supported",
                    "supporting_quote": "周末想聊什么？",
                }
            return {"intent": "statement", "verdict": "supported"}

    model = Batch()
    with pytest.raises(ValueError, match="intent"):
        ExchangeModel(model, threading.Event()).extract(**inputs())
    assert model.supports == 3


def test_cancellation_after_contradiction_prevents_repair_call():
    cancelled = threading.Event()
    calls = []

    def invoke(stage, messages, schema, cancellation):
        calls.append(stage)
        if stage == "exchange_extract":
            return {"operations": [operation()]}
        cancelled.set()
        return {"intent": "statement", "verdict": "supported"}

    with pytest.raises(InterruptedError):
        ExchangeModel(invoke, cancelled).extract(**inputs())
    assert calls == ["exchange_extract", "exchange_support"]


def test_cancel_stops_before_support_or_returning_effects():
    cancelled = threading.Event()
    invoke = Model()

    def cancel_after_extract(*args):
        result = invoke(*args)
        cancelled.set()
        return result

    with pytest.raises(InterruptedError):
        ExchangeModel(cancel_after_extract, cancelled).extract(**inputs())
    assert len(invoke.calls) == 1


def test_resolution_classifies_current_answer_and_retains_original_question():
    params = inputs()
    scope = params["scope"]
    params["turns"] = (EvidenceTurn(scope, "answer", "user", "想聊旅行", 4),)
    params["targets"] = {
        "question": {
            "companion_id": A,
            "exchange_id": "question",
            "revision": 1,
            "direction": "user_question",
            "content": "周末想聊什么？",
            "status": "pending",
            "expires_us": None,
        }
    }

    def invoke(stage, messages, schema, cancelled):
        if stage == "exchange_extract":
            return {
                "operations": [
                    operation(
                        operation="resolve",
                        type=None,
                        evidenceTurnId="answer",
                        evidenceQuote="想聊旅行",
                        supersedesAtomId="question",
                    )
                ]
            }
        data = json.loads(messages[-1]["content"])
        assert data["evidence_quote"] == "想聊旅行"
        assert data["proposed_content"] == "周末想聊什么？"
        return {"intent": "answer", "verdict": "supported"}

    proposals = ExchangeModel(invoke, threading.Event()).extract(**params)
    assert proposals[0][0].action == "resolve"
    assert proposals[0][0].content == "周末想聊什么？"


def test_foreign_target_is_rejected_before_any_model_call():
    invoke = Model()
    params = inputs()
    params["targets"] = {"bad": {"companion_id": "foreign"}}
    with pytest.raises(ValueError):
        ExchangeModel(invoke, threading.Event()).extract(**params)
    assert invoke.calls == []


def test_request_schema_does_not_offer_transitions_without_existing_targets():
    invoke = Model()
    ExchangeModel(invoke, threading.Event()).extract(**inputs())
    props = invoke.calls[0][2]["properties"]["operations"]["items"]["properties"]
    assert props["operation"]["enum"] == ["add"]
    assert props["supersedesAtomId"]["enum"] == [None]


def test_user_source_schema_rejects_assistant_only_open_loop_before_support():
    params = inputs()
    scope = params["scope"]
    params["turns"] = (EvidenceTurn(scope, "user-source", "user", "帮我安排周末", 6),)
    calls = []

    def invoke(stage, messages, schema, cancelled):
        calls.append(stage)
        assert stage == "exchange_extract"
        assert not Draft202012Validator(schema).is_valid({"operations": [operation(
            type="open_loop", evidenceTurnId="user-source", evidenceQuote="帮我安排周末",
        )]})
        assert Draft202012Validator(schema).is_valid({"operations": [operation(
            type="response_obligation", evidenceTurnId="user-source",
            evidenceQuote="帮我安排周末",
        )]})
        return {"operations": [operation(
            type="open_loop", evidenceTurnId="user-source", evidenceQuote="帮我安排周末",
        )]}

    with pytest.raises(ValueError, match="schema"):
        ExchangeModel(invoke, threading.Event()).extract(**params)
    assert calls == ["exchange_extract"]


def test_supported_flag_cannot_turn_declining_intent_into_a_new_request():
    class Declining(Model):
        def __call__(self, stage, messages, schema, cancelled):
            if stage == "exchange_support":
                return {"intent": "decline", "verdict": "supported"}
            return super().__call__(stage, messages, schema, cancelled)

    with pytest.raises(ValueError, match="intent"):
        ExchangeModel(Declining(), threading.Event()).extract(**inputs())




@pytest.mark.parametrize("use_stage_wrapper", [False, True], ids=["direct", "background"])
def test_structured_memory_wire_reconstructs_long_quotes_from_bound_spans(use_stage_wrapper):
    from virtual_companion.cognition.background_worker import _StageInvoke

    text = "帮我查一下" + ("周末活动" * 120) + "，明天告诉我好吗？"
    calls = []

    class Compact(Model):
        supports_bound_evidence = True

        def __call__(self, stage, messages, schema, cancelled):
            calls.append((stage, messages, schema))
            if stage == "exchange_intent":
                props = schema["properties"]["acts"]["items"]["properties"]
                assert "evidenceQuote" not in props
                assert props["endSegmentId"]["enum"] == [0, 1]
                assert "targetId" not in props
                return {"coverage": "complete", "acts": [{
                    "segmentId": 0,
                    "endSegmentId": 1,
                    "intent": "question",
                    "targetIndex": None,
                }]}
            if stage == "exchange_extract":
                props = schema["properties"]["operations"]["items"]["properties"]
                assert "evidenceQuote" not in props
                assert props["evidenceQuoteIndex"]["enum"] == [0]
                assert "supersedesAtomId" not in props
                assert props["targetIndex"]["enum"] == [None]
                assert props["expiresAt"] == {
                    "type": ["integer", "null"],
                    "minimum": 0,
                    "maximum": 253_402_300_799_999,
                }
                candidate = {
                    "operation": "add", "type": "open_loop", "evidenceQuoteIndex": 0,
                    "targetIndex": None, "expiresAt": 253_402_300_799_999,
                }
                assert Draft202012Validator(schema).is_valid({"operations": [candidate]})
                for invalid_expiry in (0.5, 253_402_300_800_000):
                    invalid = {**candidate, "expiresAt": invalid_expiry}
                    assert not Draft202012Validator(schema).is_valid({"operations": [invalid]})
                assert len(json.loads(messages[-1]["content"])["independent_acts"][0][
                    "evidenceQuote"]) == len(text)
                return {"operations": [{
                    "operation": "add",
                    "type": "open_loop",
                    "evidenceQuoteIndex": 0,
                    "targetIndex": None,
                    "expiresAt": None,
                }]}
            assert stage == "exchange_support"
            return {"intent": "question", "verdict": "supported"}

    scope = inputs()["scope"]
    source = EvidenceTurn(scope, "long-source", "assistant", text, len(text))
    invoke = _StageInvoke(Compact()) if use_stage_wrapper else Compact()
    effects, receipt = ExchangeModel(invoke, threading.Event()).resolve_source(
        scope=scope,
        batch_id="long-batch",
        now_us=100,
        turns=(source,),
        targets={},
        eligible_source_ids=(source.source_id,),
        strategy="intent_first",
    )

    assert len(effects) == 1
    assert effects[0][0].content == text
    assert effects[0][0].evidence.quote == text
    assert not receipt.no_exchange
    intent_response = {"coverage": "complete", "acts": [{
        "segmentId": 0, "endSegmentId": 1, "intent": "question", "targetIndex": None,
    }]}
    assert len(json.dumps(intent_response, ensure_ascii=False)) < 256
    assert [stage for stage, _, _ in calls] == [
        "exchange_intent", "exchange_extract", "exchange_support",
    ]


def test_structured_memory_wire_binds_targets_by_index_not_provider_authored_id():
    scope = inputs()["scope"]
    source = EvidenceTurn(scope, "answer-source", "user", "想聊旅行", 4)
    target_id = "💡" * 128
    targets = {target_id: {
        "companion_id": scope.companion_id,
        "exchange_id": target_id,
        "revision": 1,
        "direction": "user_question",
        "content": "周末想聊什么？",
        "status": "pending",
        "expires_us": None,
    }}
    calls = []

    class Compact(Model):
        supports_bound_evidence = True

        def __call__(self, stage, messages, schema, cancelled):
            calls.append((stage, messages, schema))
            props = (schema["properties"]["acts"]["items"]["properties"]
                     if stage == "exchange_intent" else
                     schema["properties"]["operations"]["items"]["properties"]
                     if stage == "exchange_extract" else {})
            if stage in {"exchange_intent", "exchange_extract"}:
                assert "targetId" not in props and "supersedesAtomId" not in props
                assert target_id not in json.dumps(schema, ensure_ascii=False)
                assert props["targetIndex"]["enum"] == [None, 0]
            if stage == "exchange_intent":
                return {"coverage": "complete", "acts": [{
                    "segmentId": 0, "endSegmentId": 0, "intent": "answer", "targetIndex": 0,
                }]}
            if stage == "exchange_extract":
                return {"operations": [{
                    "operation": "resolve", "type": None, "evidenceQuoteIndex": 0,
                    "targetIndex": 0, "expiresAt": None,
                }]}
            assert stage == "exchange_support"
            return {"intent": "answer", "verdict": "supported"}

    effects, receipt = ExchangeModel(Compact(), threading.Event()).resolve_source(
        scope=scope,
        batch_id="target-index-batch",
        now_us=100,
        turns=(source,),
        targets=targets,
        eligible_source_ids=(source.source_id,),
        strategy="intent_first",
    )

    assert len(effects) == 1
    assert effects[0][0].action == "resolve"
    assert effects[0][0].exchange_id == target_id
    assert receipt.target_refs == ((target_id, 1),)
    assert [stage for stage, _, _ in calls] == [
        "exchange_intent", "exchange_extract", "exchange_support",
    ]


@pytest.mark.parametrize("use_stage_wrapper", [False, True], ids=["direct", "background"])
def test_compact_extraction_preserves_original_independent_act_quote_indexes(use_stage_wrapper):
    from virtual_companion.cognition.background_worker import _StageInvoke
    from virtual_companion.cognition.exchange_model import EXTRACT, extraction_schema

    shared_quote = "我周末想放松，帮我想个安排。"
    other_quote = "适合去哪儿？"
    text = shared_quote + other_quote
    scope = inputs()["scope"]
    source = EvidenceTurn(scope, "quote-index-source", "user", text, len(text))
    acts = [
        {"segmentId": segment, "evidenceTurnId": source.source_id,
         "evidenceQuote": quote, "intent": intent, "targetId": None}
        for segment, quote, intent in (
            (0, shared_quote, "statement"),
            (0, shared_quote, "request"),
            (2, other_quote, "question"),
        )
    ]
    calls = []

    class Compact:
        supports_bound_evidence = True
        quote_indexes = (1, 2)

        def __call__(self, stage, messages, schema, cancelled):
            calls.append((stage, messages, schema))
            if stage == "exchange_intent":
                return {"coverage": "complete", "acts": [
                    {"segmentId": act["segmentId"],
                     "endSegmentId": 1 if act["segmentId"] == 0 else 2,
                     "intent": act["intent"], "targetIndex": None}
                    for act in acts
                ]}
            if stage == "exchange_extract":
                return {"operations": [
                    {"operation": "add", "type": "response_obligation",
                     "evidenceQuoteIndex": index, "targetIndex": None, "expiresAt": None}
                    for index in self.quote_indexes
                ]}
            assert stage == "exchange_support"
            quote = json.loads(messages[-1]["content"])["evidence_quote"]
            return {"intent": "request" if quote == shared_quote else "question",
                    "verdict": "supported"}

    provider = Compact()
    invoke = _StageInvoke(provider) if use_stage_wrapper else provider
    model = ExchangeModel(invoke, threading.Event())
    schema = extraction_schema(
        [source.source_id], user_only=True, quote_options=[shared_quote, other_quote],
    )

    def extract_indexes(indexes):
        provider.quote_indexes = indexes
        return model._call(
            "exchange_extract", EXTRACT, {"independent_acts": acts}, schema,
            invoke=model._bound_source_invoke(source.source_id),
        )

    for index, expected_quote in ((1, shared_quote), (2, other_quote)):
        result = extract_indexes((index,))
        assert result["operations"][0]["evidenceQuote"] == expected_quote
        assert result["operations"][0]["evidenceTurnId"] == source.source_id
        assert "evidenceQuoteIndex" not in result["operations"][0]
        _, messages, wire_schema = calls[-1]
        props = wire_schema["properties"]["operations"]["items"]["properties"]
        assert props["evidenceQuoteIndex"]["enum"] == [0, 1, 2]
        assert "evidenceTurnId" not in props
        assert json.loads(messages[0]["content"].split("\nJSON schema:\n")[1]) == wire_schema
        wire_acts = json.loads(messages[-1]["content"])["independent_acts"]
        assert [act["evidenceQuote"] for act in wire_acts] == [
            shared_quote, shared_quote, other_quote,
        ]
        assert all("evidenceTurnId" not in act for act in wire_acts)

    for invalid_index in (-1, 3):
        with pytest.raises(ValueError, match="Bound extraction evidence choice is invalid"):
            extract_indexes((invalid_index,))

    calls.clear()
    provider.quote_indexes = (1, 2)
    effects, receipt = model.resolve_source(
        scope=scope, batch_id="quote-index-batch", now_us=100,
        turns=(source,), targets={}, eligible_source_ids=(source.source_id,),
        strategy="intent_first",
    )
    assert [effect[0].evidence.quote for effect in effects] == [shared_quote, other_quote]
    assert all(effect[0].evidence.source_id == source.source_id for effect in effects)
    assert all(item.disposition == "accepted" for item in receipt.assessments)
    assert not receipt.no_exchange
    assert [stage for stage, _, _ in calls] == [
        "exchange_intent", "exchange_extract", "exchange_support", "exchange_support",
    ]
