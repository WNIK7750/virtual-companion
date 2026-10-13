"""RF4 short path: host risk routing decides when a model opinion must be spent.

Ordinary no-tool turns currently pay one early and one final model opinion per
attempt. ARC-006 asks for one main generation and no fixed model review, so the
open question is which candidates still need an explicit opinion. This scan answers
only that: it marks text carrying a risk form the host can point at, with the
provenance it saw.

Two deliberate limits:

* the scan never approves and never refuses a candidate; publication stays a
  separate decision with its own receipt, so this module cannot become
  punctuation-based release;
* every frozen negative candidate must be flagged, including the nine the host
  cannot decide alone, while the frozen ordinary positives must stay unflagged --
  a router that flags everything saves nothing, and one that misses a negative
  would publish unverified text.
"""

import json
from dataclasses import fields
from pathlib import Path

import pytest
from virtual_companion.cognition.unit_risk import (
    RISK_CODES,
    RiskScan,
    RiskSignal,
    has_open_group,
    scan_unit_risk,
)

FIXTURE = Path(__file__).with_name("fixtures") / "rf3-unit-semantics.v1.json"

# The risk form each frozen negative is caught by. Extra codes on the same case are
# allowed; a different code means the reason changed and must be re-justified.
EXPECTED_CODE = {
    "neg-dangling-comma": "dangling_ending",
    "neg-dangling-colon": "dangling_ending",
    "neg-open-quote": "open_group",
    "neg-open-group": "open_group",
    "neg-contradicts-user-limit": "constraint_violation",
    "neg-constraint-maximum-exceeded": "constraint_violation",
    "neg-unparsed-format-constraint": "unparsed_constraint",
    "neg-conditional-without-punctuation": "unclosed_condition",
    "neg-dangling-connective": "dangling_connective",
    "neg-unclosed-contrast": "dangling_connective",
    "neg-unresolved-reference": "unbound_reference",
    "neg-enumeration-continuation": "enumeration_promise",
    "neg-partial-requested-count": "requested_count_unmet",
    "neg-misquoted-user-source": "non_verbatim_user_quote",
    "neg-contradicts-shown": "polarity_conflict",
    "neg-cross-segment-preference-conflict": "polarity_conflict",
}


def frozen_cases():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]


def scan(case):
    return scan_unit_risk(
        case["user_request"], case["utterance"], shown_text=case.get("shown", ""))


def test_the_frozen_set_assigns_a_risk_form_to_every_negative():
    cases = {case["id"]: case for case in frozen_cases()}
    negatives = {case_id for case_id, case in cases.items()
                 if case["gold_presentable"] is False}
    assert negatives == set(EXPECTED_CODE)
    assert set(EXPECTED_CODE.values()) <= RISK_CODES


@pytest.mark.parametrize("case_id", sorted(EXPECTED_CODE))
def test_every_frozen_negative_candidate_is_flagged_with_its_risk_form(case_id):
    case = next(entry for entry in frozen_cases() if entry["id"] == case_id)
    result = scan(case)
    assert result.flagged is True, case_id
    assert EXPECTED_CODE[case_id] in {signal.code for signal in result.signals}, case_id
    for signal in result.signals:
        assert signal.code in RISK_CODES
        assert signal.source in {"candidate", "shown_text", "user_text"}
        assert signal.span and len(signal.span) <= 64
        assert signal.span in case["utterance"] + case["user_request"] + case["shown"]


def test_frozen_ordinary_positives_stay_unflagged():
    """Flagging ordinary text buys nothing back; the router must stay quiet."""

    for case in frozen_cases():
        if case["gold_presentable"] is not True:
            continue
        result = scan(case)
        assert result.flagged is False, (case["id"], result.signals)


def test_the_scan_reports_no_publication_decision():
    assert {field.name for field in fields(RiskScan)} == {"flagged", "signals", "version"}
    assert {field.name for field in fields(RiskSignal)} == {"code", "span", "source"}
    plain = scan_unit_risk("陪我聊两句。", "你好，今天过得怎么样？\n\n")
    assert plain.flagged is False and plain.signals == ()
    assert plain.version == "unit-risk-scan.v1"


def test_the_scan_is_deterministic_and_bounded():
    case = next(entry for entry in frozen_cases() if entry["id"] == "neg-dangling-connective")
    assert scan(case) == scan(case)
    with pytest.raises(ValueError):
        scan_unit_risk("陪我聊两句。", None)
    with pytest.raises(ValueError):
        scan_unit_risk("陪我聊两句。", "好" * 64001)
    with pytest.raises(ValueError):
        scan_unit_risk("陪我聊两句。", "好的。", shown_text=None)
    with pytest.raises(ValueError):
        RiskSignal("not-a-code", "好", "candidate")


@pytest.mark.parametrize("text,expected", [
    ("他说：“今天", True),
    ("他说：“今天很高兴。”", False),
    ("这（可能", True),
    ("这（可能）。", False),
    ("已经结束了）", True),
    ("没有任何括号。", False),
])
def test_open_group_detection_covers_quotes_brackets_and_mismatches(text, expected):
    assert has_open_group(text) is expected


# The next two cases are recorded real-model output (rf4-adversarial-scan-01) that the
# first scan version flagged without the answer carrying the risk: a numbered list does
# satisfy a requested count, and a short shared span with the user text that merely
# flips "不X"/"X" is ordinary disagreement, not a contradiction. Over-flagging only buys
# needless opinions, but it is what the policy pays for, so it is pinned here.
def test_a_numbered_list_satisfies_a_requested_count():
    three = "1. 读一本你一直想读但还没读的书。  \n2. 尝试学习一项新技能。  \n3. 写一封真诚的信件。"
    assert scan_unit_risk("给我三个建议。", three).flagged is False
    five = "\n".join(f"{index}. 第{index}件事" for index in range(1, 6))
    assert scan_unit_risk("列一下今天要做的五件事。", five).flagged is False
    short = "1. 只有一件事。  \n2. 还有一件。"
    assert scan_unit_risk("给我三个建议。", short).flagged is True
    assert scan_unit_risk("说说你的建议。", "我先说第一点。\n\n").flagged is True


def test_a_short_shared_span_with_the_user_text_does_not_route_a_review():
    answer = "我平时不喝茶，但可以为你介绍一些茶的知识或推荐茶品。需要的话可以告诉我。"
    assert scan_unit_risk("你平时喝茶吗？", answer).flagged is False
    agreeing = "明白了，你更喜欢安静的音乐。需要我推荐一些安静的音乐类型或歌单吗？"
    assert scan_unit_risk("我不喜欢吵闹的音乐，平时都听安静的。", agreeing).flagged is False
    # The frozen contradiction cases keep their longer shared span and stay flagged.
    conflict = scan_unit_risk("我不喜欢吵闹的音乐，平时都听安静的。",
                              "我很喜欢吵闹的音乐，越吵越好。\n\n")
    assert [signal.code for signal in conflict.signals] == ["polarity_conflict"]
    shown = scan_unit_risk("你平时喝茶吗？", "我不喜欢喝茶。\n\n其实我很喜欢茶。\n\n",
                           shown_text="我不喜欢喝茶。")
    assert "polarity_conflict" in shown.codes


# Recorded real output (rf5-risk-truncation-01): repeating "我周三下午三点在图书馆等朋友，
# 别迟到。" was routed because "三点" was read as a request for three items. A clock time is
# not a count request, and flagging it buys an opinion for nothing.
def test_a_clock_time_is_not_a_requested_item_count():
    repeat = "我周三下午三点在图书馆等朋友，别迟到。"
    assert scan_unit_risk(f"原样重复：“{repeat}”", repeat).flagged is False
    assert scan_unit_risk("晚上八点开始。", "好的，我记住了。").flagged is False
    assert scan_unit_risk("给我三个建议。", "首先注意休息。").flagged is True
    assert scan_unit_risk("说三点建议。", "首先注意休息。").flagged is True
