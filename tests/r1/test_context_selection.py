from dataclasses import replace

import pytest
from virtual_companion.text.context_selection import RecordedSummary, Turn, select_context


def count(messages):
    return sum(len(message["content"]) for message in messages)


def test_recent_keeps_whole_turns_and_current_input_exactly():
    history = (Turn("old", "a" * 30, "b" * 30), Turn("new", "question", "answer"))
    plan = select_context(history, "  当前输入🙂  ", budget=30, count=count, strategy="recent")
    assert plan.included_turn_ids == ("new",)
    assert plan.omitted_turn_ids == ("old",)
    assert plan.messages[-1] == {"role": "user", "content": "  当前输入🙂  "}
    assert plan.fits
    assert history[0].user == "a" * 30


def test_full_history_reports_overflow_instead_of_silently_selecting():
    history = (Turn("old", "long" * 20, "reply"),)
    plan = select_context(history, "now", budget=10, count=count, strategy="full")
    assert not plan.fits
    assert plan.included_turn_ids == ("old",)
    assert plan.omitted_turn_ids == ()


@pytest.mark.parametrize("strategy", ["full", "recent", "summary_recall"])
def test_current_input_alone_overflow_is_explicit(strategy):
    with pytest.raises(ValueError, match="Current input"):
        select_context((), "cannot truncate this", budget=3, count=count, strategy=strategy)


def test_summary_and_recall_have_provenance_and_no_duplicate_turns():
    history = (Turn("fact", "我喜欢茉莉花。", "记住了。"), Turn("recent", "你好", "你好"))
    summary = RecordedSummary("summary-1", "用户喜欢茉莉花。", (("fact", history[0].digest),))
    plan = select_context(
        history,
        "喜欢什么？",
        budget=100,
        count=count,
        strategy="summary_recall",
        summaries=(summary,),
        recalled_turn_ids=("fact", "fact"),
    )
    assert plan.evidence_ids == ("summary-1",)
    assert plan.included_turn_ids == ("fact", "recent")
    assert sum(m["content"] == history[0].user for m in plan.messages) == 1
    assert "录制摘要" in plan.messages[0]["content"]


def test_changed_or_invalidated_summary_is_not_reused():
    turn = Turn("fact", "old fact", "reply")
    summary = RecordedSummary("summary-1", "old summary", ((turn.id, turn.digest),))
    for history, invalidated in [
        ((replace(turn, user="corrected fact"),), ()),
        ((turn,), ("summary-1",)),
    ]:
        plan = select_context(
            history,
            "now",
            budget=100,
            count=count,
            strategy="summary_recall",
            summaries=(summary,),
            invalidated_evidence_ids=invalidated,
        )
        assert plan.evidence_ids == ()
        assert plan.rejected_evidence_ids == ("summary-1",)
        assert "old summary" not in str(plan.messages)


def test_unknown_sources_or_recall_ids_fail_closed():
    summary = RecordedSummary("bad", "unsupported", (("missing", "0" * 64),))
    plan = select_context(
        (), "now", budget=100, count=count, strategy="summary_recall", summaries=(summary,)
    )
    assert plan.rejected_evidence_ids == ("bad",)
    with pytest.raises(ValueError, match="Unknown recalled"):
        select_context(
            (),
            "now",
            budget=100,
            count=count,
            strategy="summary_recall",
            recalled_turn_ids=("missing",),
        )


def test_cancelled_turn_never_invents_an_assistant_message():
    plan = select_context(
        (Turn("cancelled", "unanswered", None),), "next", budget=100, count=count, strategy="recent"
    )
    assert [message["role"] for message in plan.messages] == ["user", "user"]


def test_counter_failure_is_not_replaced_by_a_character_estimate():
    def broken(messages):
        raise OSError("counter unavailable")

    with pytest.raises(OSError):
        select_context((), "now", budget=100, count=broken, strategy="recent")
