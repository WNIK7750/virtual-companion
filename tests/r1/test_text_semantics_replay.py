"""Synthetic combined replay, with a real SQLite idempotent commit adapter."""

import json
import sqlite3
from dataclasses import asdict
from datetime import UTC, datetime

import pytest
from virtual_companion.text.chronology import ChronologyGate, TimeEvent
from virtual_companion.text.output_boundary import (
    PresentationAck,
    TextCandidate,
    TextOutputBoundary,
)


def presented(candidate):
    output = TextOutputBoundary(candidate)
    output.review("recorded-review", candidate.sha256, approved=True)
    output.acknowledge(
        PresentationAck(
            candidate.attempt_id,
            1,
            len(candidate.text),
            datetime(2026, 9, 9, tzinfo=UTC),
            candidate.revision,
            candidate.sha256,
        )
    )
    return output


def test_late_input_is_retained_but_does_not_create_new_output():
    chronology = ChronologyGate()
    events = [
        TimeEvent(
            "current", datetime(2026, 9, 9, tzinfo=UTC), datetime(2026, 9, 9, tzinfo=UTC), "a" * 64
        ),
        TimeEvent(
            "late", datetime(2026, 9, 8, tzinfo=UTC), datetime(2026, 9, 9, tzinfo=UTC), "b" * 64
        ),
    ]
    outputs = []
    for event in events:
        receipt = chronology.accept(event)
        if receipt.status == "accepted":
            outputs.append(presented(TextCandidate(event.id, "attempt-" + event.id, "合成回复", 1)))
    assert len(outputs) == 1
    assert [item.status for item in chronology.receipts] == ["accepted", "late"]


def test_uncertain_sqlite_commit_and_boundary_reconstruction_do_not_duplicate_text(tmp_path):
    database = sqlite3.connect(tmp_path / "synthetic-output.sqlite3")
    database.execute("CREATE TABLE commits(id TEXT PRIMARY KEY, record TEXT NOT NULL)")
    candidate = TextCandidate("turn", "attempt", "合成已审查文本🙂", 1)
    first = presented(candidate)
    calls = []

    def writer(record):
        value = json.dumps(asdict(record), sort_keys=True, ensure_ascii=False)
        with database:
            database.execute(
                "INSERT OR IGNORE INTO commits VALUES(?,?)", (record.idempotency_key, value)
            )
            stored = database.execute(
                "SELECT record FROM commits WHERE id=?", (record.idempotency_key,)
            ).fetchone()[0]
            if stored != value:
                raise ValueError("Conflicting idempotency key")
        calls.append(record.idempotency_key)
        if len(calls) == 1:
            raise OSError("synthetic response loss after COMMIT")

    try:
        with pytest.raises(OSError):
            first.commit(writer)
        assert first.status == "commit_pending"
        database.close()
        database = sqlite3.connect(tmp_path / "synthetic-output.sqlite3")
        replay = presented(candidate)
        record = replay.commit(writer)
        assert database.execute("SELECT count(*) FROM commits").fetchone()[0] == 1
        assert first.commit(writer) == record
        assert first.status == replay.status == "committed"
    finally:
        database.close()
