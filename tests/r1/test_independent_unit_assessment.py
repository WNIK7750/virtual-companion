from threading import Event

import pytest
from virtual_companion.boundary.presentation import digest


def packet(text="如果你愿意", user="请给我明确回答。"):
    payload = {
        "version": "stream-unit-review.v1",
        "user_text": user,
        "context": None,
        "candidate_prefix": text,
        "displayed_prefix": "",
        "candidate_sha256": digest(text),
        "observed_text": text + "\n\n",
        "final": False,
        "basis": {"input_id": "synthetic"},
        "forbidden_characters": "",
    }
    return {"payload": payload, "sha256": digest(payload)}


def test_independent_unit_contract_classifies_utterance_before_host_decision():
    from virtual_companion.cognition.independent_unit_assessment import IndependentUnitAssessor

    calls = []

    def invoke(stage, messages, schema, cancelled):
        calls.append((stage, messages, schema, cancelled))
        assert "candidate_sha256" not in messages[1]["content"]
        assert "如果你愿意" in messages[1]["content"]
        return {
            "request_sha256": schema["properties"]["request_sha256"]["const"],
            "utterance_state": "unfinished",
            "evidence_span": "如果你愿意",
        }

    cancelled = Event()
    assessor = IndependentUnitAssessor(invoke, cancelled)
    value = assessor(packet())
    assert value["decision"] == "hold"
    assert value["no_open_dependency"] is False
    assert calls[0][0] == "stream_unit_independent"
    assert calls[0][3] is cancelled


def test_independent_unit_contract_maps_complete_and_conflict_without_model_approval_field():
    from virtual_companion.cognition.independent_unit_assessment import IndependentUnitAssessor

    def judge(state):
        def invoke(stage, messages, schema, cancelled):
            return {
                "request_sha256": schema["properties"]["request_sha256"]["const"],
                "utterance_state": state,
                "evidence_span": "我在这里。",
            }

        return IndependentUnitAssessor(invoke, Event())(packet("我在这里。"))

    assert judge("complete")["decision"] == "approve"
    assert judge("conflict")["decision"] == "reject"
    assert judge("uncertain")["decision"] == "hold"


@pytest.mark.parametrize("fault", ["foreign_hash", "outside_span", "extra_field", "bad_state"])
def test_independent_unit_contract_rejects_unbound_model_output(fault):
    from virtual_companion.cognition.independent_unit_assessment import IndependentUnitAssessor

    def invoke(stage, messages, schema, cancelled):
        result = {
            "request_sha256": schema["properties"]["request_sha256"]["const"],
            "utterance_state": "complete",
            "evidence_span": "我在这里。",
        }
        if fault == "foreign_hash":
            result["request_sha256"] = "0" * 64
        elif fault == "outside_span":
            result["evidence_span"] = "不存在"
        elif fault == "extra_field":
            result["approved"] = True
        else:
            result["utterance_state"] = "approved"
        return result

    with pytest.raises(ValueError):
        IndependentUnitAssessor(invoke, Event())(packet("我在这里。"))


def test_independent_unit_contract_observes_cancellation_after_model_return():
    from virtual_companion.cognition.independent_unit_assessment import IndependentUnitAssessor

    cancelled = Event()

    def invoke(stage, messages, schema, event):
        event.set()
        return {
            "request_sha256": schema["properties"]["request_sha256"]["const"],
            "utterance_state": "complete",
            "evidence_span": "我在这里。",
        }

    with pytest.raises(InterruptedError):
        IndependentUnitAssessor(invoke, cancelled)(packet("我在这里。"))
