"""Explicit zero-output stream retry preserves intake and fences old effects."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest
from test_harness_recovery import Model, execute, response, tool_call
from test_rebuilt_turns import A, claim, config
from test_segment_store import reviewed
from virtual_companion.boundary.contracts import VersionConflict
from virtual_companion.boundary.sqlite_harness import harness_binding
from virtual_companion.boundary.sqlite_memory_management import SQLiteExplicitMemoryStore


def setup(tmp_path):
    store = SQLiteExplicitMemoryStore(tmp_path / "retry.sqlite3")
    item = store.accept(config(), "stable-turn", "你好")
    return store, item, claim(store, item)


def retry(store, run, retry_id="retry-1", *, now=151, chosen=None):
    return store.retry_stream(A, run.input_id, run.attempt_id, retry_id,
                              chosen or config(), now_us=now)


@pytest.mark.parametrize("ending", ["failed", "cancelled", "cancel", "expired", "fail"])
def test_zero_unit_retry_reuses_input_and_archives_old_attempt(tmp_path, ending):
    store, item, run = setup(tmp_path)
    terminal = None
    if ending in {"failed", "cancelled"}:
        terminal = store.finish_stream(run, outcome=ending, now_us=101)
    elif ending == "cancel":
        store.cancel(A, item.input_id)
    elif ending == "fail":
        store.fail(run, reason="generation_failed", now_us=101)
    receipt = retry(store, run)
    reopened = SQLiteExplicitMemoryStore(store.path)
    assert retry(reopened, run, now=152) == receipt
    assert receipt["input_id"] == item.input_id
    assert reopened.stream_state(A, item.input_id)["settlement"] is None
    with reopened.transaction(write=False) as db:
        assert db.execute("SELECT count(*) FROM turn_inputs").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM background_sources").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM turn_attempt_history").fetchone()[0] == 1
    archived = reopened.attempts(A, item.input_id)[0]
    assert archived["attempt"] == run.attempt_id
    if terminal is not None:
        assert receipt["previous_stream_terminal"] == terminal
    fresh = claim(reopened, item, worker="next-worker", now=153)
    assert fresh.attempt_id != run.attempt_id and fresh.owner.epoch > run.owner.epoch
    assert retry(reopened, run, now=154) == receipt
    with pytest.raises(VersionConflict):
        reopened.finish_stream(run, outcome="failed", now_us=154)
    with pytest.raises(VersionConflict):
        reopened.approve_segment(run, reviewed(run), now_us=154)
    assert reopened.inspect(A, item.input_id)["run"]["attempt"] == fresh.attempt_id


def test_live_run_requires_cancel_or_lease_expiry(tmp_path):
    store, item, run = setup(tmp_path)
    with pytest.raises(VersionConflict):
        retry(store, run, now=149)
    assert store.inspect(A, item.input_id)["status"] == "running"
    retry(store, run, now=150)


@pytest.mark.parametrize("change", ["config", "attempt", "input"])
def test_reused_retry_identity_rejects_changed_parameters(tmp_path, change):
    store, _, run = setup(tmp_path)
    retry(store, run)
    args = [A, run.input_id, run.attempt_id, "retry-1", config()]
    if change == "config":
        args[-1] = config(version="2")
    elif change == "attempt":
        args[2] = "other-attempt"
    else:
        args[1] = "other-input"
    with pytest.raises(ValueError, match="Conflicting"):
        store.retry_stream(*args, now_us=152)


@pytest.mark.parametrize("state", ["pending", "unknown", "confirmed"])
def test_any_approved_unit_blocks_whole_turn_retry(tmp_path, state):
    store, item, run = setup(tmp_path)
    store.approve_segment(run, reviewed(run), now_us=101)
    if state != "pending":
        event = store.dispatch_segment(run, 1, now_us=102)
    if state == "confirmed":
        store.confirm_segment(A, item.input_id, 1, fingerprint=event["fingerprint"],
                              host_receipt="device-1", channel="text_visible")
    terminal = store.finish_stream(run, outcome="failed", now_us=103)
    before = store.segments(A, item.input_id)
    with pytest.raises(VersionConflict, match="reconcil"):
        retry(store, run)
    assert store.segments(A, item.input_id) == before
    assert store.stream_state(A, item.input_id)["settlement"] == terminal


def test_retry_preserves_unknown_harness_effect_fence(tmp_path):
    store, item, run = setup(tmp_path)
    journal = store.harness_journal(run, "tool", clock=lambda: 101)
    execute(Model([response("", [tool_call()]), response()]), journal,
            binding=harness_binding(run),
            dispatch=lambda _: {"outcome": "unknown", "content": "lost receipt"})
    terminal = store.finish_stream(run, outcome="failed", now_us=102)
    with pytest.raises(VersionConflict, match="unresolved Harness"):
        retry(store, run)
    assert store.stream_state(A, item.input_id)["settlement"] == terminal
    assert store.attempts(A, item.input_id)[0]["attempt"] == run.attempt_id


def test_retry_fault_rolls_back_terminal_and_archive(tmp_path):
    store, item, run = setup(tmp_path)
    terminal = store.finish_stream(run, outcome="failed", now_us=101)
    with store.transaction() as db:
        db.execute("CREATE TRIGGER refuse_retry BEFORE INSERT ON turn_retry_receipts "
                   "BEGIN SELECT RAISE(ABORT,'retry-fault'); END")
    with pytest.raises(sqlite3.DatabaseError):
        retry(store, run)
    assert store.stream_state(A, item.input_id)["settlement"] == terminal
    with store.transaction(write=False) as db:
        assert db.execute("SELECT count(*) FROM turn_attempt_history").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM turn_retry_bindings").fetchone()[0] == 0
        assert json.loads(db.execute("SELECT receipt FROM stream_terminals").fetchone()[0]) \
            == terminal


def test_concurrent_same_retry_is_one_transition_and_new_config_is_bound(tmp_path):
    store, item, run = setup(tmp_path)
    store.cancel(A, item.input_id)
    chosen = config(version="2")
    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts = list(pool.map(lambda _: retry(store, run, chosen=chosen), range(2)))
    assert receipts[0] == receipts[1]
    fresh = claim(store, item, now=152)
    assert json.loads(fresh.config_json)["companion_config"] == chosen.to_mapping()
    assert store.inspect(A, item.input_id)["config"] == item.config_json
    assert len(store.attempts(A, item.input_id)) == 2


def test_cancelled_zero_unit_retry_does_not_release_unrelated_active_owner(tmp_path):
    store, item, run = setup(tmp_path)
    store.cancel(A, item.input_id)
    other = store.accept(config(), "other-turn", "稍等", session_id=item.session_id)
    active = claim(store, other, worker="other-worker", now=151)
    retry(store, run, now=152)
    assert store.owner(A) == active.owner
    assert store.inspect(A, other.input_id)["status"] == "running"
    with pytest.raises(VersionConflict, match="active run"):
        claim(store, item, now=153)


def test_locked_clock_is_used_for_expired_attempt_retry(tmp_path):
    store, item, run = setup(tmp_path)
    receipt = store.retry_stream(A, item.input_id, run.attempt_id, "retry-clock", config(),
                                 now_us=149, clock=lambda: 150)
    assert receipt["status"] == "queued"
