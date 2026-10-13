from datetime import UTC, datetime

import pytest
from virtual_companion.text.chronology import ChronologyGate, TimeEvent, utc_microseconds, utc_text


def event(identity, when, digest="a" * 64):
    return TimeEvent(
        identity, datetime.fromisoformat(when), datetime(2026, 9, 9, tzinfo=UTC), digest
    )


def test_mixed_offsets_are_ordered_by_absolute_time_not_text():
    local = datetime.fromisoformat("2026-08-16T23:30:00+08:00")
    utc = datetime.fromisoformat("2026-08-16T16:00:00+00:00")
    assert local.isoformat() > utc.isoformat()  # Characterize the old raw TEXT ordering risk.
    assert utc_microseconds(local) < utc_microseconds(utc)
    assert utc_text(local) < utc_text(utc)


def test_equal_instants_and_microseconds_roundtrip_without_float_precision_loss():
    one = datetime.fromisoformat("2026-09-09T00:00:00.000001+08:00")
    two = datetime.fromisoformat("2026-09-08T16:00:00.000001Z")
    assert utc_text(one) == utc_text(two)
    assert utc_microseconds(one) == utc_microseconds(two)
    assert utc_microseconds(datetime(1969, 12, 31, 23, 59, 59, 999999, tzinfo=UTC)) == -1
    end = datetime(9999, 12, 31, 23, 59, 59, 999999, tzinfo=UTC)
    assert utc_microseconds(end) - utc_microseconds(end.replace(microsecond=999998)) == 1


def test_late_event_is_recorded_without_rewriting_applied_state():
    gate = ChronologyGate()
    first = gate.accept(event("initial", "2026-09-09T00:00:00Z"))
    late = gate.accept(event("old", "2026-08-16T10:00:00+08:00"))
    assert first.status == "accepted" and late.status == "late"
    assert gate.applied == (first,)
    assert gate.receipts == (first, late)
    assert gate.watermark == utc_microseconds(first.event.occurred_at)


def test_duplicate_event_is_idempotent_but_conflicting_payload_is_rejected():
    gate = ChronologyGate()
    first = gate.accept(event("one", "2026-09-09T00:00:00+08:00"))
    assert gate.accept(event("one", "2026-09-08T16:00:00Z")) == first
    with pytest.raises(ValueError, match="Conflicting"):
        gate.accept(event("one", "2026-09-08T16:00:00Z", "b" * 64))
    assert gate.receipts == (first,)


def test_equal_event_times_have_stable_receive_sequence():
    gate = ChronologyGate()
    first = gate.accept(event("z", "2026-09-09T00:00:00Z"))
    second = gate.accept(event("a", "2026-09-09T08:00:00+08:00"))
    assert first.sequence < second.sequence
    assert gate.applied == (first, second)


def test_naive_times_fail_before_any_state_change():
    gate = ChronologyGate()
    with pytest.raises(ValueError):
        gate.accept(event("bad", "2026-09-09T00:00:00"))
    assert gate.receipts == () and gate.watermark is None


def test_replay_preserves_late_receipts_and_applied_order():
    gate = ChronologyGate()
    for item in [
        event("first", "2026-09-09T00:00:00Z"),
        event("late", "2026-09-08T00:00:00Z"),
        event("next", "2026-09-09T00:00:00.000001Z"),
    ]:
        gate.accept(item)
    replay = ChronologyGate()
    for receipt in gate.receipts:
        replay.accept(receipt.event)
    assert replay.receipts == gate.receipts
    assert replay.applied == gate.applied
    assert replay.watermark == gate.watermark


def test_sqlite_normalized_text_and_integer_match_python_order():
    import sqlite3

    values = [
        "2026-08-16T23:30:00+08:00",
        "2026-08-16T16:00:00Z",
        "2026-08-16T16:00:00.000001Z",
        "2026-08-17T00:00:00+08:00",
    ]
    times = [datetime.fromisoformat(value) for value in values]
    database = sqlite3.connect(":memory:")
    try:
        database.execute(
            "CREATE TABLE events(seq INTEGER PRIMARY KEY, raw TEXT, utc TEXT, us INTEGER)"
        )
        database.executemany(
            "INSERT INTO events VALUES(?,?,?,?)",
            [
                (i, value, utc_text(time), utc_microseconds(time))
                for i, (value, time) in enumerate(zip(values, times, strict=True))
            ],
        )
        expected = sorted(range(len(times)), key=lambda i: (utc_microseconds(times[i]), i))
        for column in ("utc", "us"):
            query = (
                "SELECT seq FROM events ORDER BY utc,seq"
                if column == "utc"
                else "SELECT seq FROM events ORDER BY us,seq"
            )
            assert [row[0] for row in database.execute(query)] == expected
        assert [
            row[0] for row in database.execute("SELECT seq FROM events ORDER BY raw,seq")
        ] != expected
    finally:
        database.close()
