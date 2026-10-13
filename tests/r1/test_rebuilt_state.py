from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime
from threading import Barrier

import pytest
from virtual_companion.boundary.contracts import OwnerToken, Scope, StateEvent, VersionConflict
from virtual_companion.boundary.sqlite_state import SQLiteStateStore

A = "00000000-0000-0000-0000-000000000001"
B = "00000000-0000-0000-0000-000000000002"


def event(identifier, when, *, companion=A, session="one", kind="presence", **payload):
    return StateEvent(
        Scope(companion, session),
        identifier,
        kind,
        datetime.fromisoformat(when),
        datetime.fromisoformat(when),
        **payload,
    )


@pytest.fixture
def state(tmp_path):
    store = SQLiteStateStore(tmp_path / "state.sqlite3")
    store.register(A)
    return store, store.claim_owner(A, "worker", expected_epoch=0)


def apply(store, owner, command):
    return store.apply(
        command, owner, expected_version=store.snapshot(owner.companion_id).write_version
    )


def test_late_presence_retains_receipt_without_negative_period(state):
    store, owner = state
    apply(store, owner, event("first", "2026-09-09T12:00:00Z", availability="busy"))
    late = event("late", "2026-09-09T11:00:00Z", availability="available")
    receipt = apply(store, owner, late)
    assert receipt.disposition == "late"
    assert store.snapshot(A).availability == "busy"
    assert len(store.periods(A)) == 1
    assert store.periods(A)[0]["ended_us"] is None
    assert store.apply(late, owner, expected_version=0) == receipt


def test_mixed_offsets_and_equal_time_have_stable_period_order(state):
    store, owner = state
    for identifier, when, status in [
        ("one", "2026-09-09T23:30:00+08:00", "busy"),
        ("two", "2026-09-09T16:00:00Z", "available"),
        ("three", "2026-09-10T00:00:00+08:00", "sleeping"),
    ]:
        apply(store, owner, event(identifier, when, availability=status))
    periods = store.periods(A)
    assert [p["availability"] for p in periods] == ["sleeping", "available", "busy"]
    assert all(p["ended_us"] is None or p["ended_us"] >= p["started_us"] for p in periods)


def test_interruption_replay_returns_original_period_after_state_changed_and_reopen(state):
    store, owner = state
    apply(store, owner, event("busy", "2026-09-09T12:00:00Z", availability="busy"))
    hit = event("input", "2026-09-09T12:01:00Z", kind="interrupt")
    before = apply(store, owner, hit)
    apply(store, owner, event("available", "2026-09-09T12:02:00Z", availability="available"))
    reopened = SQLiteStateStore(store.path)
    assert reopened.apply(hit, owner, expected_version=0) == before
    assert [p["interruptions"] for p in reopened.periods(A)] == [0, 1]


def test_state_cas_uses_persistent_write_version_not_business_revision(state):
    store, owner = state
    first = event(
        "initial",
        "2026-09-09T12:00:00Z",
        kind="state",
        values_json='{"valence":0.9}',
        business_revision=0,
    )
    apply(store, owner, first)
    base = store.snapshot(A)
    newer = event(
        "newer",
        "2026-09-09T12:10:00Z",
        kind="state",
        values_json='{"valence":0.15}',
        business_revision=0,
    )
    store.apply(newer, owner, expected_version=base.write_version)
    stale = event(
        "older",
        "2026-09-09T12:01:00Z",
        kind="state",
        values_json='{"valence":0.42}',
        business_revision=0,
    )
    with pytest.raises(VersionConflict):
        store.apply(stale, owner, expected_version=base.write_version)
    assert store.snapshot(A).business_revision == 0
    assert store.snapshot(A).values_json == '{"valence":0.15}'
    assert store.receipt(A, "older") is None


def test_clock_rollback_cannot_reapply_decay_or_move_watermark(state):
    store, owner = state
    initial = event("initial", "2026-09-09T12:00:00Z", kind="state", values_json='{"v":0.9}')
    apply(store, owner, initial)
    watermark = store.snapshot(A).decay_watermark_us
    back = event("back", "2026-09-09T11:50:00Z", kind="state", values_json='{"v":0.7}')
    assert apply(store, owner, back).disposition == "late"
    assert store.snapshot(A).decay_watermark_us == watermark
    assert store.snapshot(A).values_json == '{"v":0.9}'
    assert store.apply(initial, owner, expected_version=0).sequence == 1


def test_companion_scope_and_cross_session_version_owner(state):
    store, owner = state
    store.register(B)
    other = store.claim_owner(B, "worker", expected_epoch=0)
    apply(store, owner, event("same", "2026-09-09T12:00:00Z", availability="busy"))
    apply(store, other, event("same", "2026-09-09T12:00:00Z", companion=B, availability="sleeping"))
    apply(
        store, owner, event("next", "2026-09-09T12:01:00Z", session="two", availability="available")
    )
    assert store.snapshot(A).write_version == 2
    assert store.snapshot(B).availability == "sleeping"
    with pytest.raises(ValueError):
        store.apply(
            event("bad", "2026-09-09T12:02:00Z", companion=B, availability="busy"),
            owner,
            expected_version=1,
        )


def test_owner_takeover_fences_old_writer_and_allows_receipt_lookup(state):
    store, old = state
    command = event("original", "2026-09-09T12:00:00Z", availability="busy")
    receipt = apply(store, old, command)
    new = store.claim_owner(A, "replacement", expected_epoch=old.epoch)
    with pytest.raises(VersionConflict):
        apply(store, old, event("stale", "2026-09-09T12:01:00Z", availability="available"))
    assert store.apply(command, old, expected_version=0) == receipt
    assert new.epoch == old.epoch + 1


def test_conflicting_duplicate_is_rejected_and_equivalent_offset_deduplicates(state):
    store, owner = state
    original = event("id", "2026-09-09T12:00:00Z", availability="busy")
    receipt = apply(store, owner, original)
    equivalent = replace(original, occurred_at=datetime.fromisoformat("2026-09-09T20:00:00+08:00"))
    assert apply(store, owner, equivalent) == receipt
    with pytest.raises(ValueError):
        apply(store, owner, replace(original, availability="sleeping"))


def test_concurrent_presence_and_interrupt_are_one_atomic_order(state):
    store, owner = state
    apply(store, owner, event("busy", "2026-09-09T12:00:00Z", availability="busy"))
    base = store.snapshot(A).write_version
    barrier = Barrier(2)
    commands = [
        event("switch", "2026-09-09T12:01:00Z", availability="available"),
        event("hit", "2026-09-09T12:01:00Z", kind="interrupt"),
    ]

    def run(command):
        local = SQLiteStateStore(store.path)
        barrier.wait(timeout=5)
        try:
            return local.apply(command, owner, expected_version=base)
        except VersionConflict:
            return apply(local, owner, command)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run, commands))
    hit = store.receipt(A, "hit")
    periods = store.periods(A)
    assert sum(p["interruptions"] for p in periods) == int(hit.disposition == "interrupted")
    if hit.disposition == "interrupted":
        assert next(p for p in periods if p["period_id"] == hit.period_id)["availability"] == "busy"


def test_receipt_failure_rolls_back_state_period_and_counter(state):
    store, owner = state
    apply(store, owner, event("busy", "2026-09-09T12:00:00Z", availability="busy"))
    before = store.snapshot(A), store.periods(A)
    with store.transaction() as connection:
        connection.execute(
            "CREATE TRIGGER fail_receipt BEFORE INSERT ON state_receipts "
            "BEGIN SELECT RAISE(ABORT, 'synthetic disk boundary'); END"
        )
    with pytest.raises(Exception, match="synthetic disk boundary"):
        apply(store, owner, event("hit", "2026-09-09T12:01:00Z", kind="interrupt"))
    assert (store.snapshot(A), store.periods(A)) == before


@pytest.mark.parametrize(
    "change",
    [
        dict(kind="unknown"),
        dict(values_json='{"v":NaN}'),
        dict(business_revision=-1),
        dict(occurred_at=datetime(2026, 9, 9)),
    ],
)
def test_invalid_commands_fail_before_mutation(change):
    original = event("state", "2026-09-09T12:00:00Z", kind="state", values_json='{"v":0.2}')
    with pytest.raises(ValueError):
        replace(original, **change)


def test_invalid_owner_and_scope():
    with pytest.raises(ValueError):
        Scope("../other", "session")
    with pytest.raises(ValueError):
        OwnerToken(A, "worker", 0)


def test_uninitialized_presence_is_explicit_not_assumed_available(state):
    store, owner = state
    result = apply(store, owner, event("first", "2026-09-09T12:00:00Z", kind="interrupt"))
    assert result.disposition == "uninitialized"
    assert result.period_id is None
    assert store.periods(A) == []
