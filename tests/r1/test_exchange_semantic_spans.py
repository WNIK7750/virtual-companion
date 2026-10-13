import json
from threading import Event

import pytest
from virtual_companion.boundary.contracts import Scope, SourceResolutionUnresolved
from virtual_companion.boundary.evidence import EvidenceTurn
from virtual_companion.cognition.exchange_model import ExchangeModel

SCOPE = Scope("11111111-1111-4111-8111-111111111111", "session")


def resolve(text, invoke, targets=None):
    return ExchangeModel(invoke, Event()).resolve_source(
        scope=SCOPE, batch_id="batch", now_us=100,
        turns=(EvidenceTurn(SCOPE, "current", "user", text, len(text)),),
        targets=targets or {}, eligible_source_ids=("current",), strategy="intent_first")


@pytest.mark.parametrize("text,intent", [
    ("前天我提到的是《城南旧事》，作者是林海音。", "statement"),
    ("我不喜欢茶了，现在更喜欢白水。", "statement"),
    ("不用推荐了，我已经选好了。", "decline"),
])
def test_one_nonactionable_act_can_cover_its_complete_original_utterance(text, intent):
    calls = []

    def invoke(stage, messages, schema, cancelled):
        calls.append(stage)
        if stage == "exchange_intent":
            return {"coverage": "complete", "acts": [{"segmentId": 0,
                    "evidenceQuote": text,
                    "intent": intent, "targetId": None}]}
        assert stage == "exchange_extract"
        return {"operations": []}

    effects, receipt = resolve(text, invoke)
    assert not effects and receipt.no_exchange
    assert calls == ["exchange_intent", "exchange_extract"]


@pytest.mark.parametrize("text,intent", [
    ("我前天提到那本书的作者是谁？查不到就直接说查不到。", "question"),
    ("从公交车上的声音谈谈城市节奏，可以举例或类比，我不急着要结论。", "request"),
])
def test_request_and_its_response_conditions_form_one_exact_supported_obligation(text, intent):
    calls = []

    def invoke(stage, messages, schema, cancelled):
        calls.append(stage)
        if stage == "exchange_intent":
            return {"coverage": "complete", "acts": [{"segmentId": 0,
                    "evidenceQuote": text,
                    "intent": intent, "targetId": None}]}
        if stage == "exchange_extract":
            data = json.loads(messages[-1]["content"])
            assert data["independent_acts"][0]["evidenceQuote"] == text
            return {"operations": [{"operation": "add", "type": "response_obligation",
                    "evidenceQuote": text,
                    "supersedesAtomId": None, "expiresAt": None}]}
        assert stage == "exchange_support"
        return {"intent": intent, "verdict": "supported"}

    effects, receipt = resolve(text, invoke)
    assert len(effects) == 1 and effects[0][0].content == text
    assert not receipt.no_exchange
    assert calls == ["exchange_intent", "exchange_extract", "exchange_support"]


def test_grouped_request_cannot_silently_drop_a_separate_request():
    first = "推荐一本书，短一点。"
    second = "也告诉我明天怎么安排。"

    def invoke(stage, messages, schema, cancelled):
        if stage == "exchange_intent":
            return {"coverage": "complete", "acts": [
                {"segmentId": 0, "evidenceQuote": first,
                 "intent": "request", "targetId": None},
                {"segmentId": 2, "evidenceQuote": second,
                 "intent": "request", "targetId": None}]}
        assert stage == "exchange_extract"
        return {"operations": [{"operation": "add", "type": "response_obligation",
                "evidenceQuote": first,
                "supersedesAtomId": None, "expiresAt": None}]}

    with pytest.raises(SourceResolutionUnresolved, match="coverage"):
        resolve(first + second, invoke)


def test_extractor_cannot_drop_conditions_from_a_grouped_act():
    text = "作者是谁？查不到就直接说查不到。"

    def invoke(stage, messages, schema, cancelled):
        if stage == "exchange_intent":
            return {"coverage": "complete", "acts": [{"segmentId": 0,
                    "evidenceQuote": text,
                    "intent": "question", "targetId": None}]}
        assert stage == "exchange_extract"
        quotes = schema["properties"]["operations"]["items"]["properties"]["evidenceQuote"]
        assert quotes == {"enum": [text]}
        return {"operations": [{"operation": "add", "type": "response_obligation",
                "evidenceQuote": "作者是谁？",
                "supersedesAtomId": None, "expiresAt": None}]}

    with pytest.raises(ValueError, match="bound schema"):
        resolve(text, invoke)


def test_intent_target_wire_does_not_offer_command_identity_as_exchange_identity():
    text = "取消那个请求。"
    target = {"companion_id": SCOPE.companion_id, "exchange_id": "exchange-1",
              "command_id": "operation-1", "revision": 1, "direction": "assistant_response",
              "content": "推荐一本书", "status": "pending", "expires_us": None}

    def invoke(stage, messages, schema, cancelled):
        assert stage == "exchange_intent"
        data = json.loads(messages[-1]["content"])
        assert "command_id" not in data["targets"][0]
        assert schema["properties"]["acts"]["items"]["properties"]["targetId"] == {
            "enum": [None, "exchange-1"]}
        return {"coverage": "complete", "acts": [{"segmentId": 0,
                "evidenceQuote": text,
                "intent": "end", "targetId": "operation-1"}]}

    with pytest.raises(ValueError, match="bound schema"):
        resolve(text, invoke, {"exchange-1": target})


def test_prepared_query_intents_are_reused_for_exchange_settlement():
    text = "忙完一整天了，依据以前的偏好，有什么适合听着放松的？"
    turns = (EvidenceTurn(SCOPE, "current", "user", text, len(text)),)
    stages = []

    def invoke(stage, messages, schema, cancelled):
        stages.append(stage)
        if stage == "exchange_intent":
            data = json.loads(messages[-1]["content"])
            assert len(data["segments"]) == 3
            assert "evidenceTurnId" not in schema["properties"]["acts"]["items"]["properties"]
            return {"coverage": "complete", "acts": [{
                "segmentId": 0, "evidenceQuote": text, "intent": "question", "targetId": None,
            }]}
        if stage == "exchange_extract":
            return {"operations": [{
                "operation": "add", "type": "response_obligation", "evidenceQuote": text,
                "supersedesAtomId": None, "expiresAt": None,
            }]}
        assert stage == "exchange_support"
        return {"intent": "question", "verdict": "supported"}

    model = ExchangeModel(invoke, Event())
    prepared = model.prepare_source_intents(
        turns=turns, targets={}, eligible_source_ids=("current",),
    )
    assert prepared.fact_statement_quotes == ()
    effects, receipt = model.resolve_source(
        scope=SCOPE, batch_id="batch", now_us=100, turns=turns, targets={},
        eligible_source_ids=("current",), strategy="intent_first", _prepared_intents=prepared,
    )
    assert len(effects) == 1 and not receipt.no_exchange
    assert stages == ["exchange_intent", "exchange_extract", "exchange_support"]


def test_context_and_request_have_no_current_fact_evidence():
    text = "刚忙完手头的事，按以前的兴趣，推荐一个放松办法。"
    turns = (EvidenceTurn(SCOPE, "current", "user", text, len(text)),)

    def invoke(stage, messages, schema, cancelled):
        assert stage == "exchange_intent"
        return {"coverage": "complete", "acts": [
            {"segmentId": 0, "evidenceQuote": "刚忙完手头的事，",
             "intent": "context", "targetId": None},
            {"segmentId": 1, "evidenceQuote": "按以前的兴趣，",
             "intent": "context", "targetId": None},
            {"segmentId": 2, "evidenceQuote": "推荐一个放松办法。",
             "intent": "request", "targetId": None},
        ]}

    prepared = ExchangeModel(invoke, Event()).prepare_source_intents(
        turns=turns, targets={}, eligible_source_ids=("current",),
    )
    assert prepared.complete
    assert prepared.fact_statement_quotes == ()


def test_a_statement_and_request_in_one_segment_cannot_skip_facts():
    text = "请记住我喜欢爵士乐。"
    turns = (EvidenceTurn(SCOPE, "current", "user", text, len(text)),)

    def invoke(stage, messages, schema, cancelled):
        assert stage == "exchange_intent"
        return {"coverage": "complete", "acts": [
            {"segmentId": 0, "evidenceQuote": text, "intent": "statement", "targetId": None},
            {"segmentId": 0, "evidenceQuote": text, "intent": "request", "targetId": None},
        ]}

    prepared = ExchangeModel(invoke, Event()).prepare_source_intents(
        turns=turns, targets={}, eligible_source_ids=("current",),
    )
    assert prepared.complete
    assert prepared.fact_statement_quotes == (text,)


@pytest.mark.parametrize("coverage,quote", [
    ("unknown", "你记得吗？我后来搬到杭州了。"),
    ("complete", "你记得吗？"),
])
def test_incomplete_query_intents_cannot_skip_a_source_fact(coverage, quote):
    text = "你记得吗？我后来搬到杭州了。"
    turns = (EvidenceTurn(SCOPE, "current", "user", text, len(text)),)

    def invoke(stage, messages, schema, cancelled):
        assert stage == "exchange_intent"
        return {"coverage": coverage, "acts": [{
            "segmentId": 0, "evidenceQuote": quote, "intent": "question", "targetId": None,
        }]}

    model = ExchangeModel(invoke, Event())
    prepared = model.prepare_source_intents(
        turns=turns, targets={}, eligible_source_ids=("current",),
    )
    assert not prepared.complete
    assert prepared.fact_statement_quotes is None
    with pytest.raises(SourceResolutionUnresolved, match="intent coverage"):
        model.resolve_source(
            scope=SCOPE, batch_id="batch", now_us=100, turns=turns, targets={},
            eligible_source_ids=("current",), strategy="intent_first", _prepared_intents=prepared,
        )


def test_prepared_intents_cannot_be_reused_for_another_context():
    text = "你记得我喜欢什么吗？"
    source = EvidenceTurn(SCOPE, "current", "user", text, len(text))
    stages = []

    def invoke(stage, messages, schema, cancelled):
        stages.append(stage)
        assert stage == "exchange_intent"
        return {"coverage": "complete", "acts": [{
            "segmentId": 0, "evidenceQuote": text, "intent": "question", "targetId": None,
        }]}

    model = ExchangeModel(invoke, Event())
    prepared = model.prepare_source_intents(
        turns=(source,), targets={}, eligible_source_ids=("current",),
    )
    prior = EvidenceTurn(SCOPE, "prior", "user", "我喜欢爵士乐。", len("我喜欢爵士乐。"))
    with pytest.raises(ValueError, match="source context or targets"):
        model.resolve_source(
            scope=SCOPE, batch_id="batch", now_us=100, turns=(prior, source), targets={},
            eligible_source_ids=("current",), strategy="intent_first", _prepared_intents=prepared,
        )
    assert stages == ["exchange_intent"]
