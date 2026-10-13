"""RF3-S2b: state-only prefix opinion and evidence-bound final opinion."""

from threading import Event

import pytest
from test_rebuilt_turns import claim, config
from virtual_companion.boundary.contracts import canonical
from virtual_companion.boundary.sqlite_streaming import SQLiteStreamingStore
from virtual_companion.cognition.independent_unit_assessment import StreamingUnitAssessor
from virtual_companion.cognition.short_dialogue import ShortRequest
from virtual_companion.cognition.stream_unit_review import BoundedUnitReviewer


def packet_for(utterance, *, final, user_text="你好", shown=""):
    from virtual_companion.boundary.presentation import digest

    payload = {"candidate_prefix": utterance, "user_text": user_text,
               "displayed_prefix": shown, "final": final}
    return {"payload": payload, "sha256": digest(payload)}


def test_compact_evidence_is_optional_but_never_a_foreign_one():
    from virtual_companion.cognition.independent_unit_assessment import CompactUnitAssessor

    seen = []

    def invoke(stage, messages, schema, cancelled):
        seen.append(schema)
        return {"utterance_state": "complete"}

    assessor = CompactUnitAssessor(invoke, Event())
    for final in (False, True):
        packet = packet_for("甲句。", final=final)
        result = assessor(packet)
        assert result["decision"] == "approve"
        assert result["request_sha256"] == packet["sha256"]
    assert all(schema["required"] == ["utterance_state"] for schema in seen)

    def quoted(stage, messages, schema, cancelled):
        return {"utterance_state": "complete", "evidence_span": "甲句"}

    quoted_assessor = CompactUnitAssessor(quoted, Event())
    assert quoted_assessor(packet_for("甲句。", final=True))["decision"] == "approve"
    assert quoted_assessor.last_evidence == "verbatim"

    def foreign(stage, messages, schema, cancelled):
        return {"utterance_state": "complete", "evidence_span": "不在候选中的片段"}

    # A volunteered but unusable span is recorded, not fatal: the host binds by
    # call identity, while a strict contract still refuses it.
    tolerant = CompactUnitAssessor(foreign, Event())
    assert tolerant(packet_for("甲句。", final=False))["decision"] == "approve"
    assert tolerant.last_evidence == "mismatch"
    assert CompactUnitAssessor(invoke, Event()).last_evidence == "absent"
    from virtual_companion.cognition.independent_unit_assessment import (
        IndependentUnitAssessor,
    )

    with pytest.raises(ValueError):
        IndependentUnitAssessor(foreign, Event())(packet_for("甲句。", final=False))


def test_prefix_opinion_uses_a_single_state_field_and_binds_the_host_hash():
    seen = []

    def invoke(stage, messages, schema, cancelled):
        seen.append((messages, schema))
        return {"state": "complete"}

    assessor = StreamingUnitAssessor(invoke, Event())
    packet = packet_for("甲句。", final=False)
    result = assessor(packet)
    schema = seen[0][1]
    assert set(schema["properties"]) == {"state"} and schema["required"] == ["state"]
    assert schema["properties"]["state"]["enum"] == [
        "complete", "unfinished", "conflict", "uncertain"]
    assert "evidence" not in canonical(schema)
    assert packet["sha256"] not in canonical(seen[0][0])
    assert result == {"request_sha256": packet["sha256"], "decision": "approve",
                      "self_contained": True, "no_open_dependency": True,
                      "consistent_with_displayed": True, "constraints_satisfied": True}
    assert len(canonical(seen[0][1])) < 200


def test_final_opinion_still_requires_verbatim_evidence():
    seen = []

    def invoke(stage, messages, schema, cancelled):
        seen.append(schema)
        return {"state": "complete", "evidence_span": "甲句"}

    assessor = StreamingUnitAssessor(invoke, Event())
    result = assessor(packet_for("甲句。乙句。", final=True))
    assert set(seen[0]["properties"]) == {"state", "evidence_span"}
    assert seen[0]["properties"]["evidence_span"]["maxLength"] == 8
    assert result["decision"] == "approve"

    def foreign(stage, messages, schema, cancelled):
        return {"state": "complete", "evidence_span": "不在候选中的片段"}

    with pytest.raises(ValueError, match="evidence"):
        StreamingUnitAssessor(foreign, Event())(packet_for("甲句。", final=True))


@pytest.mark.parametrize("state,decision", [
    ("unfinished", "hold"), ("conflict", "reject"), ("uncertain", "hold")])
def test_state_only_opinion_maps_to_the_host_gate(state, decision):
    def invoke(stage, messages, schema, cancelled):
        return {"state": state}

    result = StreamingUnitAssessor(invoke, Event())(packet_for("甲句。", final=False))
    assert result["decision"] == decision
    assert result["self_contained"] is (state == "complete")
    assert result["consistent_with_displayed"] is (state != "conflict")


def test_state_only_opinion_rejects_unknown_or_extra_output():
    def unknown(stage, messages, schema, cancelled):
        return {"state": "maybe"}

    with pytest.raises(ValueError):
        StreamingUnitAssessor(unknown, Event())(packet_for("甲句。", final=False))

    def extra(stage, messages, schema, cancelled):
        return {"state": "complete", "evidence_span": "甲"}

    with pytest.raises(ValueError):
        StreamingUnitAssessor(extra, Event())(packet_for("甲句。", final=False))


def test_reviewer_uses_the_minimal_prefix_opinion_then_the_bound_final_one(tmp_path):
    store = SQLiteStreamingStore(tmp_path / "minimal.sqlite3")
    run = claim(store, store.accept(config(), "first", "你好"))
    schemas = []

    def invoke(stage, messages, schema, cancelled):
        schemas.append((schema["required"], "evidence_span" in schema["properties"]))
        if "evidence_span" in schema["properties"]:
            return {"state": "complete", "evidence_span": "甲句"}
        return {"state": "complete"}

    reviewer = BoundedUnitReviewer(run, ShortRequest("你好"),
                                   StreamingUnitAssessor(invoke, Event()))
    early = reviewer("甲句。", False)
    assert early is not None and early.candidate.text == "甲句。"
    final = reviewer("甲句。乙句。", True)
    assert final is not None and final.candidate.text == "甲句。乙句。"
    assert schemas == [(["state"], False), (["state", "evidence_span"], True)]
