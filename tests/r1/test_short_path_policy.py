"""Controlled comparison of early-opinion budgets on identical observed streams.

Streams are replayed step by step so the comparison is not confounded by model
latency or by sampling variance between runs: one early opinion per turn is the
ordinary-chat policy, and the terminal opinion always owns the complete text.
"""

import pytest
from test_rebuilt_turns import claim, config
from virtual_companion.boundary.sqlite_streaming import SQLiteStreamingStore
from virtual_companion.cognition.short_dialogue import ShortRequest
from virtual_companion.cognition.stream_unit_review import BoundedUnitReviewer

ONE_UNIT = "甲句。"
TWO_UNITS = "甲句。乙句。"
THREE_UNITS = "甲句。乙句。丙句。"


def approve(packet):
    return {"request_sha256": packet["sha256"], "decision": "approve",
            "self_contained": True, "no_open_dependency": True,
            "consistent_with_displayed": True, "constraints_satisfied": True}


def reviewer_for(tmp_path, early_limit, opinions):
    store = SQLiteStreamingStore(tmp_path / f"budget-{early_limit}.sqlite3")
    run = claim(store, store.accept(config(), "first", "你好"))

    def assess(packet):
        opinions.append(packet["payload"]["final"])
        return approve(packet)

    return BoundedUnitReviewer(run, ShortRequest("你好"), assess, early_limit=early_limit)


@pytest.mark.parametrize("full,prefixes,expected", [
    (ONE_UNIT, (), {0: 1, 1: 1, 2: 1}),
    (TWO_UNITS, ("甲句。",), {0: 1, 1: 2, 2: 2}),
    (THREE_UNITS, ("甲句。", "甲句。乙句。"), {0: 1, 1: 2, 2: 3}),
])
def test_opinions_per_turn_for_each_early_budget(tmp_path, full, prefixes, expected):
    """One terminal opinion plus at most ``early_limit`` early opinions per turn."""
    for limit in (0, 1, 2):
        opinions = []
        reviewer = reviewer_for(tmp_path, limit, opinions)
        for prefix in prefixes:
            reviewer(prefix, False)
        assert reviewer(full, True) is not None
        assert len(opinions) == expected[limit], (full, limit, opinions)
        assert opinions[-1] is True, "the terminal opinion always owns the end"
        assert sum(1 for final in opinions if not final) <= limit


def test_bounded_policy_removes_one_opinion_from_a_three_unit_reply(tmp_path):
    """Ordinary chat pays one early opinion, not one opinion per unit."""
    baseline, bounded = [], []
    for limit, collected in ((2, baseline), (1, bounded)):
        reviewer = reviewer_for(tmp_path, limit, collected)
        reviewer("甲句。", False)
        reviewer("甲句。乙句。", False)
        assert reviewer(THREE_UNITS, True) is not None
    assert len(baseline) == 3 and len(bounded) == 2
    assert baseline.count(False) == 2 and bounded.count(False) == 1


def test_terminal_only_budget_still_approves_the_whole_answer(tmp_path):
    """The conservative option: no early opinion, the terminal one owns everything."""
    opinions = []
    reviewer = reviewer_for(tmp_path, 0, opinions)
    assert reviewer("甲句。", False) is None
    assert reviewer(TWO_UNITS, False) is None
    result = reviewer(TWO_UNITS, True)
    assert result is not None and result.candidate.text == TWO_UNITS
    assert opinions == [True]


def test_early_opinions_never_exceed_the_budget_on_a_long_stream(tmp_path):
    """A long reply with many boundaries cannot spend an opinion per boundary."""
    opinions = []
    reviewer = reviewer_for(tmp_path, 1, opinions)
    stream = ""
    for index in range(6):
        stream += f"第{index + 1}句。"
        reviewer(stream, False)
    assert reviewer(stream, True) is not None
    assert len(opinions) == 2, opinions
    assert opinions == [False, True]
