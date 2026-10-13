import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from test_rebuilt_turns import A, claim, config
from virtual_companion.boundary.contracts import Scope, VersionConflict
from virtual_companion.boundary.presentation import (
    MessageUnit,
    Presentation,
    PresentationPolicy,
    ReviewedPresentation,
    review_by_policy,
)
from virtual_companion.boundary.sqlite_background_worker import SQLiteResolvedBackgroundStore
from virtual_companion.boundary.sqlite_segments import SQLiteSegmentStore


def setup(tmp_path):
    store = SQLiteSegmentStore(tmp_path / "segments.sqlite3")
    item = store.accept(config(), "first", "你好")
    return store, claim(store, item)


def reviewed(run, texts=("我在这里。",)):
    candidate = Presentation(Scope(A, run.session_id), run.input_id, run.attempt_id,
                             "candidate", 1, "\n\n".join(texts),
                             tuple(MessageUnit(f"unit{i}", "text", t)
                                   for i, t in enumerate(texts, 1)))
    policy = PresentationPolicy(10, 1000, 32, False)
    return ReviewedPresentation(candidate, policy,
        review_by_policy(candidate, policy, f"review{len(texts)}",
                         policy_revision="synthetic-unit-policy-v1"))


def test_dispatch_is_durable_unknown_and_never_replayed_after_reopen(tmp_path):
    store, run = setup(tmp_path)
    receipt = store.approve_segment(run, reviewed(run), now_us=100)
    assert receipt["status"] == "pending"
    event = store.dispatch_segment(run, 1, now_us=101)
    assert event["text"] == "我在这里。"
    reopened = SQLiteSegmentStore(store.path)
    assert reopened.dispatch_segment(run, 1, now_us=102) is None
    assert reopened.segments(A, run.input_id)[0]["status"] == "unknown"
    assert reopened.inspect(A, run.input_id)["commit"] is None
    confirmed = reopened.confirm_segment(A, run.input_id, 1,
        fingerprint=event["fingerprint"], host_receipt="device-record-1", channel="text_visible")
    assert confirmed["status"] == "confirmed"
    assert reopened.confirm_segment(A, run.input_id, 1,
        fingerprint=event["fingerprint"], host_receipt="device-record-1",
        channel="text_visible") == confirmed


def test_prefix_cannot_change_and_unknown_predecessor_blocks_next_dispatch(tmp_path):
    store, run = setup(tmp_path)
    store.approve_segment(run, reviewed(run), now_us=100)
    with pytest.raises(ValueError):
        store.approve_segment(run, reviewed(run, ("修改前段", "下一段")), now_us=100)
    store.approve_segment(run, reviewed(run, ("我在这里。", "你想聊些什么？")), now_us=100)
    with pytest.raises(ValueError):
        store.dispatch_segment(run, 2, now_us=100)
    first = store.dispatch_segment(run, 1, now_us=100)
    with pytest.raises(ValueError):
        store.dispatch_segment(run, 2, now_us=100)
    store.confirm_segment(A, run.input_id, 1, fingerprint=first["fingerprint"],
                          host_receipt="one", channel="text_visible")
    assert store.dispatch_segment(run, 2, now_us=100)["sequence"] == 2


def test_cancel_stops_new_dispatch_but_allows_actual_prior_receipt(tmp_path):
    store, run = setup(tmp_path)
    store.approve_segment(run, reviewed(run), now_us=100)
    event = store.dispatch_segment(run, 1, now_us=100)
    store.cancel(A, run.input_id)
    with pytest.raises(VersionConflict):
        store.approve_segment(run, reviewed(run, ("我在这里。", "后来")), now_us=101)
    assert store.confirm_segment(A, run.input_id, 1, fingerprint=event["fingerprint"],
        host_receipt="late-but-real", channel="text_visible")["status"] == "confirmed"


def test_stale_context_expired_owner_and_unapproved_values_rejected(tmp_path):
    store, run = setup(tmp_path)
    with pytest.raises(ValueError):
        store.approve_segment(run, "raw token", now_us=100)
    with pytest.raises(VersionConflict):
        store.approve_segment(run, reviewed(run), now_us=151)
    with pytest.raises(VersionConflict):
        store.approve_segment(replace(run, state_version=999), reviewed(run), now_us=100)
    store.approve_segment(run, reviewed(run), now_us=100)
    store.accept(config(), "correction", "更正前文")
    with pytest.raises(VersionConflict):
        store.dispatch_segment(run, 1, now_us=100)


def test_ack_cannot_confirm_pending_wrong_hash_or_audio(tmp_path):
    store, run = setup(tmp_path)
    receipt = store.approve_segment(run, reviewed(run), now_us=100)
    args = dict(fingerprint=receipt["fingerprint"], host_receipt="host", channel="text_visible")
    with pytest.raises(ValueError):
        store.confirm_segment(A, run.input_id, 1, **args)
    store.dispatch_segment(run, 1, now_us=100)
    for changes in ({"fingerprint": "0" * 64}, {"channel": "audio_played"}):
        with pytest.raises(ValueError):
            store.confirm_segment(A, run.input_id, 1, **(args | changes))
    assert store.segments(A, run.input_id)[0]["status"] == "unknown"


def test_sqlite_fault_rolls_back_dispatch_and_old_reader_refuses(tmp_path):
    store, run = setup(tmp_path)
    store.approve_segment(run, reviewed(run), now_us=100)
    with store.transaction() as db:
        db.execute("CREATE TRIGGER fail_dispatch BEFORE UPDATE ON stream_segments "
                   "BEGIN SELECT RAISE(ABORT,'synthetic disk failure'); END")
    with pytest.raises(sqlite3.DatabaseError):
        store.dispatch_segment(run, 1, now_us=100)
    assert store.segments(A, run.input_id)[0]["status"] == "pending"
    with pytest.raises(ValueError):
        SQLiteResolvedBackgroundStore(store.path)


def test_concurrent_dispatch_has_one_winner_and_no_automatic_whole_fallback(tmp_path):
    store, run = setup(tmp_path)
    first = store.approve_segment(run, reviewed(run), now_us=100)
    assert store.approve_segment(run, reviewed(run), now_us=100) == first
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: store.dispatch_segment(run, 1, now_us=100),
                                    range(2)))
    assert sum(event is not None for event in results) == 1
    with pytest.raises(ValueError, match="whole-turn"):
        store.commit(None, now_us=100)


@pytest.mark.parametrize("operation", ["approve", "confirm"])
def test_storage_failure_never_fabricates_approval_or_confirmation(tmp_path, operation):
    store, run = setup(tmp_path)
    if operation == "confirm":
        store.approve_segment(run, reviewed(run), now_us=100)
        event = store.dispatch_segment(run, 1, now_us=100)
    with store.transaction() as db:
        verb = "INSERT" if operation == "approve" else "UPDATE"
        db.execute(f"CREATE TRIGGER fail_write BEFORE {verb} ON stream_segments "
                   "BEGIN SELECT RAISE(ABORT,'synthetic failure'); END")
    with pytest.raises(sqlite3.DatabaseError):
        if operation == "approve":
            store.approve_segment(run, reviewed(run), now_us=100)
        else:
            store.confirm_segment(A, run.input_id, 1, fingerprint=event["fingerprint"],
                                  host_receipt="receipt", channel="text_visible")
    reopened = SQLiteSegmentStore(store.path)
    rows = reopened.segments(A, run.input_id)
    assert rows == [] if operation == "approve" else rows[0]["status"] == "unknown"


def test_cancel_before_dispatch_and_locked_clock_expiry_block_publication(tmp_path):
    store, run = setup(tmp_path)
    store.approve_segment(run, reviewed(run), now_us=100)
    with pytest.raises(VersionConflict):
        store.dispatch_segment(run, 1, now_us=100, clock=lambda: 151)
    store.cancel(A, run.input_id)
    with pytest.raises(VersionConflict):
        store.dispatch_segment(run, 1, now_us=100)
    assert store.segments(A, run.input_id)[0]["status"] == "pending"


def test_old_database_is_not_migrated_and_tampered_manifest_not_released(tmp_path):
    old = SQLiteResolvedBackgroundStore(tmp_path / "old.sqlite3")
    old.accept(config(), "old", "合成")
    before = old.path.read_bytes()
    with pytest.raises(ValueError):
        SQLiteSegmentStore(old.path)
    assert old.path.read_bytes() == before
    store, run = setup(tmp_path)
    store.approve_segment(run, reviewed(run), now_us=100)
    with store.transaction() as db:
        db.execute("UPDATE stream_segments SET fingerprint=?", ("0" * 64,))
    with pytest.raises(ValueError):
        store.dispatch_segment(run, 1, now_us=100)
