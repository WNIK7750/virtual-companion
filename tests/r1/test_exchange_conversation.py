import json
import threading
from dataclasses import replace

import pytest
from test_exchange_model import Model, inputs
from virtual_companion.boundary.exchanges import ExchangeSupport
from virtual_companion.cognition.evidence_memory import EvidenceTurn
from virtual_companion.cognition.exchange_model import ExchangeModel


def conversation(answer):
    params = inputs()
    params["turns"] += (EvidenceTurn(params["scope"], "user-answer", "user", answer, len(answer)),)
    return params


def test_support_receives_later_context_without_unshown_tail():
    invoke = Model()
    proposals = ExchangeModel(invoke, threading.Event()).extract(**conversation("想聊旅行"))
    support = json.loads(invoke.calls[1][1][-1]["content"])
    assert support["conversation"][-1]["text"] == "想聊旅行"
    assert "不能发送的尾巴" not in str(invoke.calls)
    assert proposals[0][1].context_sha256


def test_semantic_support_is_bound_to_the_actual_conversation():
    first = ExchangeModel(Model(), threading.Event()).extract(**conversation("想聊旅行"))
    second = ExchangeModel(Model(), threading.Event()).extract(**conversation("这个先不回答"))
    assert first[0][0].fingerprint == second[0][0].fingerprint
    assert first[0][1].context_sha256 != second[0][1].context_sha256
    with pytest.raises(ValueError):
        replace(first[0][1], context_sha256="invalid")


def test_already_processed_source_cannot_be_proposed_again():
    invoke = Model()  # Deliberately proposes the old assistant question.
    with pytest.raises(ValueError):
        ExchangeModel(invoke, threading.Event()).extract(
            **conversation("换个话题"), eligible_source_ids=("user-answer",)
        )
    assert len(invoke.calls) == 1  # Rejected before semantic evaluation.


def test_legacy_support_receipt_does_not_gain_a_null_context_field():
    support = ExchangeSupport("a" * 64, "recorded")
    assert support.to_mapping() == {"command_sha256": "a" * 64, "evaluator": "recorded"}
