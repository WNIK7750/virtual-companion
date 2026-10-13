"""Empty proposals cannot erase independently grounded requests."""

import json
from threading import Event

import pytest
from virtual_companion.boundary.contracts import Scope, SourceResolutionUnresolved
from virtual_companion.boundary.evidence import EvidenceTurn
from virtual_companion.cognition.exchange_model import ExchangeModel


def resolve(invoke, cancelled=None):
    scope = Scope("11111111-1111-4111-8111-111111111111", "session")
    text = "What is my dietary preference?"
    return ExchangeModel(invoke, cancelled or Event()).resolve_source(
        scope=scope, batch_id="batch", now_us=100,
        turns=(EvidenceTurn(scope, "source", "user", text, len(text)),),
        targets={}, eligible_source_ids=("source",), strategy="intent_first",
    )


class Recording:
    def __init__(self, *, second="valid", support="question", cancel=None):
        self.calls = []
        self.second, self.support, self.cancel = second, support, cancel

    def __call__(self, stage, messages, schema, cancelled):
        data = json.loads(messages[-1]["content"])
        self.calls.append((stage, data, schema))
        if stage == "exchange_intent":
            return {"coverage": "complete", "acts": [{
                "segmentId": 0,
                "evidenceQuote": data["sources"][0]["text"],
                "intent": "question", "targetId": None,
            }]}
        if stage == "exchange_extract":
            if self.cancel:
                self.cancel.set()
            if sum(s == stage for s, _, _ in self.calls) == 1 or self.second == "empty":
                return {"operations": []}
            return {"operations": [{
                "operation": "add", "type": "response_obligation",
                **({"evidenceTurnId": "other"} if self.second == "foreign" else {}),
                "evidenceQuote": data["sources"][0]["text"],
                "supersedesAtomId": None, "expiresAt": None,
            }]}
        assert stage == "exchange_support"
        return {"intent": self.support, "verdict": "supported"}


def test_empty_question_proposal_has_one_bound_repair_then_independent_support():
    model = Recording()
    effects, receipt = resolve(model)
    assert len(effects) == 1 and not receipt.no_exchange
    assert [s for s, _, _ in model.calls] == [
        "exchange_intent", "exchange_extract", "exchange_extract", "exchange_support"]
    _, first, schema = model.calls[1]
    _, repaired, repaired_schema = model.calls[2]
    assert repaired_schema == schema
    assert {k: v for k, v in repaired.items() if k != "repair"} == first
    assert repaired["repair"]["previous"] == {"operations": []}
    assert repaired["repair"]["uncovered"] == [{
        "evidenceTurnId": "source", "evidenceQuote": first["sources"][0]["text"]}]


def test_persistent_empty_does_not_assert_answered_or_clear_source():
    model = Recording(second="empty")
    with pytest.raises(SourceResolutionUnresolved, match="coverage"):
        resolve(model)
    assert [s for s, _, _ in model.calls] == [
        "exchange_intent", "exchange_extract", "exchange_extract"]


def test_extraction_and_support_share_one_repair_budget():
    model = Recording(support="statement")
    with pytest.raises(ValueError, match="intent"):
        resolve(model)
    assert [s for s, _, _ in model.calls].count("exchange_support") == 1


def test_repaired_candidate_still_cannot_change_source():
    model = Recording(second="foreign")
    with pytest.raises(ValueError, match="schema"):
        resolve(model)
    assert [s for s, _, _ in model.calls].count("exchange_extract") == 2


def test_cancelled_empty_response_does_not_make_repair_call():
    cancelled = Event()
    model = Recording(cancel=cancelled)
    with pytest.raises(InterruptedError):
        resolve(model, cancelled)
    assert len(model.calls) == 2


@pytest.mark.parametrize("error", [TimeoutError("deadline"), RuntimeError("budget")])
def test_repair_cannot_bypass_controlled_invoker_failure(error):
    model = Recording()

    def invoke(stage, messages, schema, cancelled):
        if "repair" in json.loads(messages[-1]["content"]):
            raise error
        return model(stage, messages, schema, cancelled)

    with pytest.raises(type(error), match=str(error)):
        resolve(invoke)
    assert len(model.calls) == 2


def test_repair_candidate_requires_positive_independent_support():
    model = Recording()

    def invoke(stage, messages, schema, cancelled):
        result = model(stage, messages, schema, cancelled)
        if stage == "exchange_support":
            result["verdict"] = "unsupported"
        return result

    with pytest.raises(ValueError):
        resolve(invoke)
    assert [s for s, _, _ in model.calls].count("exchange_extract") == 2
