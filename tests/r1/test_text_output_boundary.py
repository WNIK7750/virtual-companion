from datetime import UTC, datetime

import pytest
from virtual_companion.text.output_boundary import (
    PresentationAck,
    TextCandidate,
    TextOutputBoundary,
)


def boundary():
    return TextOutputBoundary(TextCandidate("turn-1", "attempt-1", "你好🙂", 1))


def ack(sequence, end, attempt="attempt-1"):
    return PresentationAck(
        attempt,
        sequence,
        end,
        datetime(2026, 9, 9, tzinfo=UTC),
        1,
        TextCandidate("turn-1", "attempt-1", "你好🙂", 1).sha256,
    )


def approve(value):
    value.review("review-1", value.candidate.sha256, approved=True)


def test_generated_or_reviewed_text_alone_cannot_commit():
    value = boundary()
    with pytest.raises(ValueError):
        value.commit(lambda record: pytest.fail("not presented"))
    approve(value)
    with pytest.raises(ValueError):
        value.commit(lambda record: pytest.fail("not presented"))
    value.acknowledge(ack(1, 2))
    with pytest.raises(ValueError):
        value.commit(lambda record: pytest.fail("only partly presented"))


def test_complete_approved_presentation_commits_once():
    value = boundary()
    approve(value)
    assert value.acknowledge(ack(1, 3)) == "accepted"
    writes = []
    first = value.commit(writes.append)
    assert value.commit(writes.append) == first
    assert len(writes) == 1 and first.text == "你好🙂"
    assert first.candidate_sha256 == value.candidate.sha256
    assert first.review_id == "review-1"


def test_foreign_late_and_duplicate_acks_do_not_extend_displayed_range():
    value = boundary()
    approve(value)
    assert value.acknowledge(ack(2, 1)) == "accepted"
    assert value.acknowledge(ack(1, 3)) == "late_sequence"
    assert value.acknowledge(ack(3, 3, "old-attempt")) == "foreign_attempt"
    assert value.acknowledge(ack(2, 1)) == "duplicate"
    assert value.displayed_end == 1
    with pytest.raises(ValueError, match="Conflicting"):
        value.acknowledge(ack(2, 3))


def test_cancel_keeps_partial_fact_but_blocks_late_completion_and_commit():
    value = boundary()
    approve(value)
    value.acknowledge(ack(1, 1))
    value.cancel()
    assert value.acknowledge(ack(2, 3)) == "terminal"
    assert value.displayed_end == 1 and value.status == "cancelled"
    with pytest.raises(ValueError):
        value.commit(lambda _: None)


def test_review_must_bind_exact_candidate_and_cannot_be_replaced():
    value = boundary()
    with pytest.raises(ValueError):
        value.review("bad", "0" * 64, approved=True)
    value.review("reject", value.candidate.sha256, approved=False)
    with pytest.raises(ValueError):
        approve(value)
    assert value.acknowledge(ack(1, 3)) == "terminal"


def test_failed_commit_can_retry_with_same_idempotency_key():
    value = boundary()
    approve(value)
    value.acknowledge(ack(1, 3))
    durable = {}
    calls = []

    def writer(record):
        calls.append(record.idempotency_key)
        durable.setdefault(record.idempotency_key, record)
        if len(calls) == 1:
            raise OSError("commit succeeded but response lost")

    with pytest.raises(OSError):
        value.commit(writer)
    assert value.status != "committed"
    with pytest.raises(RuntimeError, match="uncertain"):
        value.cancel()
    value.commit(writer)
    assert len(durable) == 1 and len(set(calls)) == 1


@pytest.mark.parametrize("sequence,end", [(True, 1), (0, 1), (1, -1), (1, 4), (1, 1.5)])
def test_invalid_ack_range_fails_without_mutation(sequence, end):
    value = boundary()
    approve(value)
    with pytest.raises(ValueError):
        value.acknowledge(ack(sequence, end))
    assert value.displayed_end == 0


def test_ack_from_another_candidate_revision_is_not_reused():
    value = TextOutputBoundary(TextCandidate("turn-1", "attempt-1", "你好🙂", 2))
    approve(value)
    assert value.acknowledge(ack(1, 3)) == "foreign_candidate"
    assert value.displayed_end == 0
