"""Frozen RF3 self-containment set: host-deterministic withholding vs model opinion.

The host may only refuse a candidate on evidence it can check itself (dangling
punctuation, unclosed quote/group). Everything else must reach an explicit model
opinion; the fixture therefore records which layer owns each decision.
"""

import json
from pathlib import Path

import pytest
from test_rebuilt_turns import claim, config
from virtual_companion.boundary.sqlite_streaming import SQLiteStreamingStore
from virtual_companion.cognition.short_dialogue import ShortRequest
from virtual_companion.cognition.stream_unit_review import BoundedUnitReviewer

FIXTURE = Path(__file__).with_name("fixtures") / "rf3-unit-semantics.v1.json"


def frozen_cases():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]


def approving(packet):
    return {
        "request_sha256": packet["sha256"],
        "decision": "approve",
        "self_contained": True,
        "no_open_dependency": True,
        "consistent_with_displayed": True,
        "constraints_satisfied": True,
    }


def reviewer_for(tmp_path, case, assess, tag="review"):
    store = SQLiteStreamingStore(tmp_path / f"semantics-{case['id']}-{tag}.sqlite3")
    run = claim(store, store.accept(config(), "first", case["user_request"]))
    # Seeding the already-shown prefix consumes one early opinion, so the frozen
    # probes run with the two-opinion budget to keep host/model ownership visible.
    return BoundedUnitReviewer(run, ShortRequest(case["user_request"]), assess, early_limit=2)


def test_frozen_set_is_complete_and_mixed():
    cases = frozen_cases()
    assert len(cases) == len({case["id"] for case in cases}) >= 16
    for case in cases:
        assert case["host_gate"] in {"withhold", "defer"}
        assert type(case["gold_presentable"]) is bool
        assert case["utterance"].endswith("\n\n")
        if case["host_gate"] == "withhold":
            assert case["gold_presentable"] is False
    deferred = [case for case in cases if case["host_gate"] == "defer"]
    assert {case["gold_presentable"] for case in deferred} == {True, False}
    assert {case["dependency"] for case in cases} >= {
        "none", "conditional", "negation_scope", "reference", "enumeration",
        "quoted_speech", "contradiction", "constraint",
    }


def test_host_withholds_only_what_it_can_decide_and_defers_the_rest(tmp_path):
    for case in frozen_cases():
        calls = []

        def assess(packet, calls=calls):
            calls.append(packet["payload"]["candidate_prefix"])
            return approving(packet)

        reviewer = reviewer_for(tmp_path, case, assess)
        if case["shown"]:
            reviewer(case["shown"] + "\n\n", False)
            calls.clear()
        result = reviewer(case["utterance"], False)
        if case["host_gate"] == "withhold":
            assert result is None and calls == [], case["id"]
            assert reviewer.approved_text == (case["shown"] + "\n\n"
                                              if case["shown"] else ""), case["id"]
        else:
            assert calls and result is not None, case["id"]
            # The candidate is a faithful prefix ending at the earliest boundary,
            # so the fixture's trailing paragraph mark may stay with the next unit.
            assert case["utterance"].startswith(result.candidate.text), case["id"]
            assert result.candidate.text.strip(), case["id"]


@pytest.mark.parametrize("case_id", [
    "neg-conditional-without-punctuation",
    "neg-dangling-connective",
    "neg-unresolved-reference",
    "neg-enumeration-continuation",
    # "neg-contradicts-user-limit" moved to the host-withheld group once the host
    # could verify the length constraint itself; the model counterexample stays in
    # the development EVD and the frozen real-model reports.
    "neg-contradicts-shown",
])
def test_model_required_negatives_keep_the_original_punctuation_decision_open(tmp_path, case_id):
    """These read as finished sentences; only a model opinion can refuse them."""

    case = next(entry for entry in frozen_cases() if entry["id"] == case_id)
    calls, seeding = [], {"on": bool(case["shown"])}

    def hold(packet):
        if seeding["on"]:
            return approving(packet)
        calls.append(packet["payload"]["candidate_prefix"])
        verdict = approving(packet)
        verdict["decision"] = "hold"
        verdict["self_contained"] = False
        return verdict

    reviewer = reviewer_for(tmp_path, case, hold)
    if case["shown"]:
        assert reviewer(case["shown"] + "\n\n", False) is not None
        seeding["on"] = False
    assert reviewer(case["utterance"], False) is None
    assert calls and case["utterance"].startswith(calls[0]) and calls[0].strip()
    assert reviewer.approved_text == case["shown"]

    def reject(packet):
        verdict = approving(packet)
        verdict["decision"] = "reject"
        verdict["consistent_with_displayed"] = False
        return verdict

    conflict = reviewer_for(tmp_path, case, reject, tag="conflict")
    with pytest.raises(ValueError, match="rejected"):
        conflict(case["utterance"], False)
