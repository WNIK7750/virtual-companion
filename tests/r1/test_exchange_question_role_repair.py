"""Replay the v17 user-question role error without a real model or external fixture.

Witness: stage1-delivery-memory-20260927/memory-outcome-v3-tail/call-004/005.json.
The original outgoing schema already excludes user open_loop; keep it strict.
"""

import json
from copy import deepcopy
from dataclasses import replace
from threading import Event

import pytest
from virtual_companion.boundary.contracts import Scope, SourceResolutionUnresolved
from virtual_companion.boundary.evidence import EvidenceTurn
from virtual_companion.cognition.exchange_model import ExchangeModel

QUESTION = "我的饮食偏好是什么？"
BAD = {"operations": [{
    "operation": "add", "type": "open_loop",
    "evidenceQuote": QUESTION, "supersedesAtomId": None, "expiresAt": None,
}]}
GOOD = {"operations": [{**BAD["operations"][0], "type": "response_obligation"}]}
ACT = {"segmentId": 0, "evidenceQuote": QUESTION,
       "intent": "question", "targetId": None}


def inputs():
    scope = Scope("11111111-1111-4111-8111-111111111111", "session")
    return dict(scope=scope, batch_id="batch", now_us=100,
                turns=(EvidenceTurn(scope, "current", "user", QUESTION, len(QUESTION)),),
                targets={}, eligible_source_ids=("current",), strategy="intent_first")


class Recorded:
    def __init__(self, *, first=BAD, replacement=GOOD, acts=None, coverage="complete",
                 support_intent="question", support_verdict="supported",
                 cancel_after_first=False, repair_error=None):
        self.first, self.replacement = deepcopy(first), deepcopy(replacement)
        self.acts = deepcopy([ACT] if acts is None else acts)
        self.coverage = coverage
        self.support_intent, self.support_verdict = support_intent, support_verdict
        self.cancel_after_first, self.repair_error = cancel_after_first, repair_error
        self.calls, self.returned = [], []

    def __call__(self, stage, messages, schema, cancelled):
        data = json.loads(messages[-1]["content"])
        self.calls.append((stage, data, deepcopy(schema)))
        if stage == "exchange_intent":
            return {"coverage": self.coverage, "acts": deepcopy(self.acts)}
        if stage == "exchange_extract":
            if "repair" in data:
                if self.repair_error is not None:
                    raise self.repair_error
                result = deepcopy(self.replacement)
            else:
                if self.cancel_after_first:
                    cancelled.set()
                result = deepcopy(self.first)
            self.returned.append(result)
            return result
        assert stage == "exchange_support", "Only exact supported proposals may advance"
        return {"intent": self.support_intent, "verdict": self.support_verdict}

    def count(self, stage):
        return sum(call[0] == stage for call in self.calls)


@pytest.mark.parametrize("intent", ["question", "request"])
def test_recorded_user_question_role_error_gets_one_bound_model_repair(intent):
    model = Recorded(acts=[{**ACT, "intent": intent}], support_intent=intent)
    proposals, receipt = ExchangeModel(model, Event()).resolve_source(**inputs())
    assert len(proposals) == 1 and not receipt.no_exchange
    command, support = proposals[0]
    assert command.direction == "assistant_response"
    assert command.evidence.role == "user"
    assert command.evidence.source_id == "current" and command.evidence.quote == QUESTION
    assert support.command_sha256 == command.fingerprint
    assert support.evaluator == receipt.evaluator == "exchange-model.v22"
    assert [call[0] for call in model.calls] == [
        "exchange_intent", "exchange_extract", "exchange_extract", "exchange_support",
    ]
    _, first, schema = model.calls[1]
    _, repaired, repaired_schema = model.calls[2]
    assert schema["properties"]["operations"]["items"]["properties"]["type"] == {
        "enum": ["response_obligation", None]}
    assert repaired_schema == schema
    assert {k: v for k, v in repaired.items() if k != "repair"} == first
    assert repaired["repair"]["previous"] == BAD
    assert model.returned[0] == BAD  # No host rewrite of the original illegal operation.
    assert "open_loop" in repaired["repair"]["error"]


@pytest.mark.parametrize("replacement, error", [
    (BAD, ValueError),
    ({"operations": []}, SourceResolutionUnresolved),
    ({"operations": [{**GOOD["operations"][0], "evidenceTurnId": "other"}]}, ValueError),
    ({"operations": [{**GOOD["operations"][0], "evidenceQuote": "无来源的问题？"}]}, ValueError),
])
def test_role_repair_rejects_bad_or_missing_replacement_without_third_call(replacement, error):
    model = Recorded(replacement=replacement)
    with pytest.raises(error):
        ExchangeModel(model, Event()).resolve_source(**inputs())
    assert model.count("exchange_extract") == 2 and model.count("exchange_support") == 0


@pytest.mark.parametrize("changed", [
    {"evidenceTurnId": "other"}, {"evidenceQuote": QUESTION[:-1]},
    {"supersedesAtomId": "other"}, {"expiresAt": 1234}, {"operation": "resolve"},
    {"content": QUESTION}, {"type": "unknown"}, {"type": None},
])
def test_wrong_identity_or_additional_shape_error_is_not_given_role_retry(changed):
    model = Recorded(first={"operations": [{**BAD["operations"][0], **changed}]})
    with pytest.raises(ValueError):
        ExchangeModel(model, Event()).resolve_source(**inputs())
    assert model.count("exchange_extract") == 1 and model.count("exchange_support") == 0


def test_missing_field_is_not_given_role_retry():
    first = deepcopy(BAD)
    del first["operations"][0]["expiresAt"]
    model = Recorded(first=first)
    with pytest.raises(ValueError, match="schema"):
        ExchangeModel(model, Event()).resolve_source(**inputs())
    assert model.count("exchange_extract") == 1


def test_mixed_operations_are_not_given_role_retry():
    model = Recorded(first={"operations": [*BAD["operations"], *GOOD["operations"]]})
    with pytest.raises(ValueError, match="schema"):
        ExchangeModel(model, Event()).resolve_source(**inputs())
    assert model.count("exchange_extract") == 1


@pytest.mark.parametrize("acts, coverage, extract_calls", [
    ([ACT, {**ACT, "intent": "statement"}], "complete", 1),
    ([{**ACT, "intent": "statement"}], "complete", 1),
    ([{**ACT, "intent": "quotation"}], "complete", 1),
    ([{**ACT, "intent": "unclear"}], "complete", 0),
    ([ACT], "unknown", 0),
    ([{**ACT, "targetId": "unknown"}], "complete", 0),
])
def test_mixed_or_unbound_intent_does_not_acquire_role_retry(acts, coverage, extract_calls):
    model = Recorded(acts=acts, coverage=coverage)
    with pytest.raises((ValueError, SourceResolutionUnresolved)):
        ExchangeModel(model, Event()).resolve_source(**inputs())
    assert model.count("exchange_extract") == extract_calls


def test_partially_confirmed_source_is_not_a_whole_source_role_repair():
    params = inputs()
    params["turns"] = (replace(params["turns"][0], text=QUESTION + "还有别的意思"),)
    model = Recorded()
    with pytest.raises(ValueError, match="schema"):
        ExchangeModel(model, Event()).resolve_source(**params)
    assert model.count("exchange_extract") == 1


def test_candidate_first_does_not_gain_role_retry_without_independent_intent():
    model, params = Recorded(), inputs()
    params["strategy"] = "candidate_first"
    with pytest.raises(ValueError, match="schema"):
        ExchangeModel(model, Event()).resolve_source(**params)
    assert model.count("exchange_extract") == 1 and model.count("exchange_intent") == 0


def test_assistant_question_open_loop_remains_valid_without_role_retry():
    params = inputs()
    params["turns"] = (replace(params["turns"][0], role="assistant"),)
    model = Recorded()
    proposals, receipt = ExchangeModel(model, Event()).resolve_source(**params)
    assert proposals[0][0].direction == "user_question" and not receipt.no_exchange
    assert model.count("exchange_extract") == 1 and model.count("exchange_support") == 1


@pytest.mark.parametrize("intent, verdict", [
    ("statement", "supported"), ("question", "unsupported"), ("question", "unknown"),
])
def test_role_repair_requires_positive_support_and_shares_existing_total_budget(intent, verdict):
    model = Recorded(support_intent=intent, support_verdict=verdict)
    with pytest.raises(ValueError):
        ExchangeModel(model, Event()).resolve_source(**inputs())
    assert model.count("exchange_extract") == 2 and model.count("exchange_support") == 1


def test_empty_proposal_repair_cannot_then_acquire_additional_role_repair():
    model = Recorded(first={"operations": []}, replacement=BAD)
    with pytest.raises(ValueError, match="schema"):
        ExchangeModel(model, Event()).resolve_source(**inputs())
    assert model.count("exchange_extract") == 2 and model.count("exchange_support") == 0


def test_cancel_after_invalid_role_cannot_start_repair():
    model = Recorded(cancel_after_first=True)
    with pytest.raises(InterruptedError):
        ExchangeModel(model, Event()).resolve_source(**inputs())
    assert model.count("exchange_extract") == 1


@pytest.mark.parametrize("error", [TimeoutError("deadline"), RuntimeError("budget")])
def test_role_repair_does_not_bypass_controlled_invoker_failure(error):
    model = Recorded(repair_error=error)
    with pytest.raises(type(error), match=str(error)):
        ExchangeModel(model, Event()).resolve_source(**inputs())
    assert model.count("exchange_extract") == 2 and model.count("exchange_support") == 0
