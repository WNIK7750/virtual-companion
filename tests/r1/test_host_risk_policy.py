"""RF4 short path: risk-routed review behind an explicit host-policy receipt.

An ordinary turn should not pay a fixed model opinion. Publication must still be
accountable, so an unflagged candidate is published under the host risk policy
receipt -- ``approval_source="host_policy"``, ``model_approved=False``, revision
pinned -- and every candidate the scan flags keeps the conservative path: an
explicit model opinion before anything is shown.

``review_policy`` defaults to ``"model"``: this file also pins that the default
entry does not change silently.
"""

import hashlib
import json
from pathlib import Path

import pytest
from test_rebuilt_turns import claim, config
from virtual_companion.boundary.presentation import ReviewedPresentation, verify_delivery
from virtual_companion.boundary.sqlite_streaming import SQLiteStreamingStore
from virtual_companion.cognition.short_dialogue import ShortRequest
from virtual_companion.cognition.stream_unit_review import (
    HOST_RISK_POLICY_REVISION,
    BoundedUnitReviewer,
    UserConstraintViolation,
)

FIXTURE = Path(__file__).with_name("fixtures") / "rf3-unit-semantics.v1.json"
ORDINARY = "你好，今天过得怎么样？\n\n"
DANGLING_REFERENCE = "它其实很有意思。\n\n"
SHORT_LIMIT = "最多三个字。"
TOO_LONG = "好的，我明白。"


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


def reviewer_for(tmp_path, user_text, assess, *, policy="host-risk", tag="risk"):
    store = SQLiteStreamingStore(tmp_path / f"host-risk-{tag}-{abs(hash(user_text))}.sqlite3")
    run = claim(store, store.accept(config(), "first", user_text))
    return BoundedUnitReviewer(run, ShortRequest(user_text), assess, early_limit=2,
                               review_policy=policy)


def receipt_for(text, result):
    """The durable delivery receipt the desktop reader revalidates on reopen."""
    candidate = result.candidate
    return {
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "presentation": result.to_mapping(),
        "presentation_sha256": candidate.fingerprint,
        "companion_id": candidate.scope.companion_id,
        "input_id": candidate.input_id,
        "attempt_id": candidate.attempt_id,
        "candidate_id": candidate.candidate_id,
        "revision": candidate.revision,
        "review_id": result.review.review_id,
    }


def test_unflagged_ordinary_text_publishes_under_the_host_risk_receipt(tmp_path):
    calls = []

    def assess(packet):
        calls.append(packet)
        return approving(packet)

    reviewer = reviewer_for(tmp_path, "陪我聊两句。", assess)
    result = reviewer(ORDINARY, True)
    assert calls == [], "ordinary unflagged text must not spend a model opinion"
    assert result.review.approval_source == "host_policy"
    assert result.review.model_approved is False
    assert result.review.policy_revision == HOST_RISK_POLICY_REVISION
    assert result.review.passed and result.candidate.text == ORDINARY
    assert ReviewedPresentation.from_mapping(result.to_mapping()) == result
    verify_delivery(ORDINARY, receipt_for(ORDINARY, result))


def test_unflagged_early_candidate_is_shown_before_the_model_stops(tmp_path):
    calls = []
    reviewer = reviewer_for(tmp_path, "陪我聊两句。",
                            lambda packet: (calls.append(packet), approving(packet))[1])
    result = reviewer(ORDINARY + "后面还有很多内容", False)
    assert calls == []
    assert result.review.approval_source == "host_policy"
    # Earliest faithful boundary wins: the sentence end, not the blank line.
    assert result.candidate.text == "你好，今天过得怎么样？"


def test_flagged_text_still_needs_a_real_opinion(tmp_path):
    calls = []

    def assess(packet):
        calls.append(packet)
        return approving(packet)

    reviewer = reviewer_for(tmp_path, "给我推荐一本书吧。", assess)
    result = reviewer(DANGLING_REFERENCE, True)
    assert len(calls) == 1
    assert calls[0]["payload"]["candidate_prefix"] == DANGLING_REFERENCE
    assert result.review.approval_source == "model"
    assert result.review.model_approved is True

    def hold(packet):
        verdict = approving(packet)
        verdict["decision"] = "hold"
        verdict["self_contained"] = False
        return verdict

    holding = reviewer_for(tmp_path, "给我推荐一本书吧。", hold, tag="hold")
    assert holding(DANGLING_REFERENCE, True) is None


def test_default_review_policy_is_unchanged(tmp_path):
    calls = []
    reviewer = reviewer_for(tmp_path, "陪我聊两句。",
                            lambda packet: (calls.append(packet), approving(packet))[1],
                            policy="model", tag="default")
    result = reviewer(ORDINARY, True)
    assert len(calls) == 1, "the default entry keeps paying the model opinion"
    assert result.review.approval_source == "model"


def test_unknown_review_policy_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="policy"):
        reviewer_for(tmp_path, "陪我聊两句。", approving, policy="host")


def test_flagging_negatives_still_buy_an_opinion_and_withheld_ones_do_not(tmp_path):
    """The safety direction of the routing decision, on the frozen set."""

    for case in frozen_cases():
        calls = []

        def assess(packet, calls=calls):
            calls.append(packet["payload"]["candidate_prefix"])
            return approving(packet)

        reviewer = reviewer_for(tmp_path, case["user_request"], assess,
                                tag=case["id"])
        if case["shown"]:
            assert reviewer(case["shown"] + "\n\n", False) is not None
            calls.clear()
        result = reviewer(case["utterance"], False)
        if case["host_gate"] == "withhold":
            assert result is None and calls == [], case["id"]
        elif case["gold_presentable"] is False:
            # Never shown without a real opinion, even though the host can read the
            # risk form: the scan routes the opinion, it does not replace it.
            assert calls, case["id"]
            assert result is None or result.review.approval_source == "model", case["id"]
        else:
            assert calls == [] and result is not None, case["id"]
            assert result.review.approval_source == "host_policy", case["id"]


def test_host_verifiable_constraints_keep_their_own_path(tmp_path):
    calls = []
    reviewer = reviewer_for(tmp_path, SHORT_LIMIT,
                            lambda packet: (calls.append(packet), approving(packet))[1])
    with pytest.raises(UserConstraintViolation):
        reviewer(TOO_LONG, True)
    assert calls == [], "a countable violation is refused without an opinion"

    satisfied = reviewer_for(tmp_path, SHORT_LIMIT,
                             lambda packet: (calls.append(packet), approving(packet))[1],
                             tag="satisfied")
    result = satisfied("好的。", True)
    assert result.review.approval_source == "host_policy"
    assert calls == []


def test_unparsed_constraint_like_text_keeps_the_conservative_path(tmp_path):
    calls = []
    reviewer = reviewer_for(tmp_path, "只回答一个数字就行。",
                            lambda packet: (calls.append(packet), approving(packet))[1])
    result = reviewer("大概是三四个吧。\n\n", True)
    assert len(calls) == 1
    assert result.review.approval_source == "model"
