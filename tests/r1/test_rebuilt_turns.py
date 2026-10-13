import hashlib
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from datetime import datetime
from threading import Barrier

import pytest
from virtual_companion.boundary.contracts import Scope, StateEvent, VersionConflict
from virtual_companion.boundary.sqlite_state import SQLiteStateStore
from virtual_companion.boundary.sqlite_turns import SQLiteTurnStore
from virtual_companion.boundary.turns import CognitiveResult
from virtual_companion.text.identity import CompanionConfig, VersionedReference

A = "00000000-0000-0000-0000-000000000001"
B = "00000000-0000-0000-0000-000000000002"


def config(companion=A, version="1"):
    return CompanionConfig(companion, VersionedReference("persona", version, "a" * 64))


@pytest.fixture
def store(tmp_path):
    return SQLiteTurnStore(tmp_path / "turns.sqlite3")


def intake(store, key="first", **kwargs):
    return store.accept(config(), key, "你好", **kwargs)


def claim(store, item, worker="worker", now=100):
    return store.claim(A, item.input_id, worker, now_us=now, lease_us=50)


def result(run, text="你好。", event=None):
    return CognitiveResult(
        run, "candidate", 1, text, "review", hashlib.sha256(text.encode()).hexdigest(), event
    )


def test_first_retry_reuses_original_session_input_and_config(store):
    first = intake(store)
    reopened = SQLiteTurnStore(store.path)
    assert reopened.accept(config(version="2"), "first", "你好") == first
    assert first.config_json != reopened.accept(config(version="2"), "next", "你好").config_json
    with store.transaction(write=False) as db:
        assert db.execute("SELECT count(*) FROM turn_sessions").fetchone()[0] == 2


def test_request_conflicts_and_session_ownership(store):
    first = intake(store)
    for text, session in [("different", None), ("你好", first.session_id)]:
        with pytest.raises(ValueError, match="Conflicting"):
            store.accept(config(), "first", text, session_id=session)
    with pytest.raises(ValueError, match="Unknown session"):
        store.accept(config(B), "first", "你好", session_id=first.session_id)
    other = store.accept(config(B), "first", "你好")
    assert other.companion_id == B
    assert other.input_id != first.input_id


def test_concurrent_first_retry_has_one_identity(store):
    barrier = Barrier(2)

    def send(_):
        barrier.wait()
        return intake(SQLiteTurnStore(store.path))

    with ThreadPoolExecutor(2) as pool:
        items = list(pool.map(send, range(2)))
    assert items[0] == items[1]
    with store.transaction(write=False) as db:
        assert db.execute("SELECT count(*) FROM turn_sessions").fetchone()[0] == 1


def test_intake_insert_failure_rolls_back_new_session_and_registration(store):
    with store.transaction() as db:
        db.execute(
            "CREATE TRIGGER break_input BEFORE INSERT ON turn_inputs BEGIN SELECT "
            "RAISE(ABORT,'injected'); END"
        )
    with pytest.raises(sqlite3.IntegrityError):
        intake(store)
    with store.transaction(write=False) as db:
        assert db.execute("SELECT count(*) FROM turn_sessions").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM companion_state").fetchone()[0] == 0


def test_claim_insert_failure_has_no_orphan_or_owner_change(store):
    item = intake(store)
    with store.transaction() as db:
        db.execute(
            "CREATE TRIGGER break_run BEFORE INSERT ON turn_runs BEGIN SELECT "
            "RAISE(ABORT,'injected'); END"
        )
    with pytest.raises(sqlite3.IntegrityError):
        claim(store, item)
    assert store.inspect(A, item.input_id)["status"] == "queued"
    assert store.owner(A) is None


def test_expired_claim_recovers_same_attempt_with_new_fence_and_original_binding(store):
    item = intake(store)
    first = claim(store, item)
    with pytest.raises(VersionConflict):
        claim(store, item, "next", 149)
    second = claim(SQLiteTurnStore(store.path), item, "next", 150)
    assert second.attempt_id == first.attempt_id
    assert second.owner.epoch == first.owner.epoch + 1
    assert second.config_json == item.config_json
    with pytest.raises(VersionConflict):
        store.commit(result(first), now_us=151)
    assert store.commit(result(second), now_us=151)["status"] == "committed"


def test_single_companion_lane_blocks_other_sessions_but_not_other_companion(store):
    first, second = intake(store), intake(store, "second")
    run = claim(store, first)
    with pytest.raises(VersionConflict):
        claim(store, second)
    other = store.accept(config(B), "first", "hi")
    assert store.claim(B, other.input_id, "worker", now_us=100, lease_us=50)
    store.commit(result(run), now_us=101)
    with pytest.raises(VersionConflict, match="delivery"):
        claim(store, second, now=1000)


def test_business_commit_state_and_terminal_are_atomic(store):
    item = intake(store)
    run = claim(store, item)
    event = StateEvent(
        Scope(A, item.session_id),
        "assessment",
        "state",
        datetime.fromisoformat("2026-09-09T12:00:00Z"),
        datetime.fromisoformat("2026-09-09T12:00:00Z"),
        values_json='{"mood":1}',
        business_revision=1,
    )
    proposal = result(run, event=event)
    with store.transaction() as db:
        db.execute(
            "CREATE TRIGGER break_commit BEFORE INSERT ON turn_commits BEGIN SELECT "
            "RAISE(ABORT,'injected'); END"
        )
    with pytest.raises(sqlite3.IntegrityError):
        store.commit(proposal, now_us=101)
    assert store.snapshot(A).write_version == 0
    assert store.receipt(A, "assessment") is None
    assert store.inspect(A, item.input_id)["status"] == "running"
    with store.transaction() as db:
        db.execute("DROP TRIGGER break_commit")
    receipt = store.commit(proposal, now_us=101)
    assert store.snapshot(A).values_json == '{"mood":1}'
    assert store.inspect(A, item.input_id)["status"] == "committed"
    assert receipt["delivery_status"] == "pending"
    assert receipt["displayed_end"] == 0
    assert SQLiteTurnStore(store.path).commit(proposal, now_us=999) == receipt
    assert store.snapshot(A).write_version == 1


def test_recovery_checks_committed_result_before_lease(store):
    item = intake(store)
    run = claim(store, item)
    store.commit(result(run), now_us=101)
    assert store.inspect(A, item.input_id)["status"] == "committed"
    with pytest.raises(VersionConflict, match="committed"):
        claim(SQLiteTurnStore(store.path), item, "next", 10000)


def test_commit_rejects_changed_candidate_and_stale_state_before_delivery(store):
    item = intake(store)
    run = claim(store, item)
    store.commit(result(run), now_us=101)
    with pytest.raises(ValueError, match="Conflicting"):
        store.commit(result(run, "另一个回答"), now_us=102)


def test_review_hash_scope_and_expired_lease_rejected(store):
    item = intake(store)
    run = claim(store, item)
    with pytest.raises(ValueError, match="review"):
        CognitiveResult(run, "candidate", 1, "text", "review", "a" * 64)
    with pytest.raises(VersionConflict, match="expired"):
        store.commit(result(run), now_us=150)
    assert store.inspect(A, item.input_id)["status"] == "running"


def test_v1_upgrade_is_explicit_backup_restores_and_old_reader_refuses(store, tmp_path):
    path = tmp_path / "v1.sqlite3"
    old = SQLiteStateStore(path)
    old.register(A)
    owner = old.claim_owner(A, "prior", expected_epoch=0)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        SQLiteTurnStore(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    backup = tmp_path / "backup.sqlite3"
    with closing(sqlite3.connect(path)) as source, closing(sqlite3.connect(backup)) as dest:
        source.backup(dest)
    upgraded = SQLiteTurnStore(path, upgrade_v1=True)
    assert upgraded.owner(A) == owner
    with pytest.raises(ValueError):
        old.snapshot(A)
    assert SQLiteStateStore(backup).owner(A) == owner


def test_future_schema_rejected_without_writes(store):
    with sqlite3.connect(store.path) as db:
        db.execute("PRAGMA user_version=99")
    before = hashlib.sha256(store.path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        SQLiteTurnStore(store.path, upgrade_v1=True)
    assert hashlib.sha256(store.path.read_bytes()).hexdigest() == before


def committed(store):
    item = intake(store)
    run = claim(store, item)
    return item, run, store.commit(result(run, "你🙂好"), now_us=101)


def ack(store, receipt, sequence=1, end=1, stopped=False, **overrides):
    args = dict(
        candidate_id=receipt["candidate_id"],
        revision=receipt["revision"],
        sha256=receipt["sha256"],
        sequence=sequence,
        displayed_end=end,
        stopped=stopped,
    )
    args.update(overrides)
    return store.acknowledge(A, receipt["delivery_id"], **args)


def test_cancel_before_publish_has_no_event_and_releases_lane(store):
    item, run, receipt = committed(store)
    store.cancel(A, item.input_id)
    assert store.publish(A, receipt["delivery_id"]) is None
    assert store.delivery(A, receipt["delivery_id"])["status"] == "cancelled"
    with store.transaction(write=False) as db:
        assert db.execute("SELECT count(*) FROM turn_publications").fetchone()[0] == 0
    next_item = intake(store, "second")
    assert claim(store, next_item)
    assert store.commit(result(run, "你🙂好"), now_us=999) == receipt


def test_publish_then_cancel_requires_actual_stop_receipt_and_preserves_prefix(store):
    item, _, receipt = committed(store)
    published = store.publish(A, receipt["delivery_id"])
    assert published["delivery_id"] == receipt["delivery_id"]
    assert store.publish(A, receipt["delivery_id"]) is None
    assert ack(store, receipt, end=2)["displayed_end"] == 2
    assert store.cancel(A, item.input_id)["status"] == "stopping"
    other = intake(store, "second")
    with pytest.raises(VersionConflict):
        claim(store, other)
    stopped = ack(SQLiteTurnStore(store.path), receipt, sequence=2, end=2, stopped=True)
    assert stopped["status"] == "cancelled"
    assert store.delivery(A, receipt["delivery_id"])["displayed_text"] == "你🙂"
    assert claim(store, other)
    with pytest.raises(ValueError):
        ack(store, receipt, sequence=3, end=3)


def test_ack_duplicate_conflict_late_wrong_candidate_and_unknown_scope(store):
    _, _, receipt = committed(store)
    with pytest.raises(ValueError):
        ack(store, receipt)
    store.publish(A, receipt["delivery_id"])
    first = ack(store, receipt, sequence=2, end=2)
    assert ack(store, receipt, sequence=2, end=2) == first
    with pytest.raises(ValueError, match="Conflicting"):
        ack(store, receipt, sequence=2, end=1)
    assert ack(store, receipt, sequence=1, end=3)["disposition"] == "late"
    assert store.delivery(A, receipt["delivery_id"])["displayed_end"] == 2
    for changes in [
        {"candidate_id": "wrong"},
        {"revision": 2},
        {"sha256": "a" * 64},
        {"displayed_end": 4},
    ]:
        with pytest.raises(ValueError):
            ack(store, receipt, sequence=3, **changes)
    with pytest.raises(ValueError):
        store.delivery(B, receipt["delivery_id"])
    assert ack(store, receipt, sequence=3, end=3)["status"] == "completed"


def test_cancel_running_blocks_late_commit_and_recovery(store):
    item = intake(store)
    run = claim(store, item)
    assert store.cancel(A, item.input_id)["status"] == "cancelled"
    with pytest.raises(VersionConflict):
        store.commit(result(run), now_us=101)
    with pytest.raises(VersionConflict):
        claim(store, item, now=1000)
    assert claim(store, intake(store, "second"))


def test_cancel_publish_concurrent_winner_has_matching_event(store):
    item, _, receipt = committed(store)
    barrier = Barrier(2)

    def publish():
        barrier.wait()
        return store.publish(A, receipt["delivery_id"])

    def cancel():
        barrier.wait()
        return store.cancel(A, item.input_id)

    with ThreadPoolExecutor(2) as pool:
        publication, cancellation = pool.submit(publish), pool.submit(cancel)
        event = publication.result()
        state = cancellation.result()
    assert state["status"] == ("stopping" if event else "cancelled")
    with store.transaction(write=False) as db:
        assert db.execute("SELECT count(*) FROM turn_publications").fetchone()[0] == bool(event)


def test_publication_event_insert_failure_rolls_back_transition(store):
    _, _, receipt = committed(store)
    with store.transaction() as db:
        db.execute(
            "CREATE TRIGGER break_publish BEFORE INSERT ON turn_publications BEGIN SELECT "
            "RAISE(ABORT,'injected'); END"
        )
    with pytest.raises(sqlite3.IntegrityError):
        store.publish(A, receipt["delivery_id"])
    assert store.delivery(A, receipt["delivery_id"])["status"] == "pending"


def test_silence_commits_without_claiming_output_and_stale_state_cannot_publish(store):
    item = intake(store)
    run = claim(store, item)
    receipt = store.commit(result(run, ""), now_us=101)
    assert receipt["delivery_status"] == "completed"
    assert store.publish(A, receipt["delivery_id"]) is None
    second = intake(store, "second")
    later = claim(store, second)
    event = StateEvent(
        Scope(A, second.session_id),
        "background",
        "presence",
        datetime.fromisoformat("2026-09-09T12:00:00Z"),
        datetime.fromisoformat("2026-09-09T12:00:00Z"),
        availability="busy",
    )
    store.apply(event, later.owner, expected_version=later.state_version)
    with pytest.raises(VersionConflict, match="snapshot"):
        store.commit(result(later), now_us=101)
    assert store.inspect(A, second.input_id)["commit"] is None


def test_upgrade_failure_rolls_back_all_ddl(tmp_path, monkeypatch):
    import virtual_companion.boundary.sqlite_turns as module

    path = tmp_path / "v1.sqlite3"
    old = SQLiteStateStore(path)
    old.register(A)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(module, "_TURN_SCHEMA", module._TURN_SCHEMA + ("INVALID SQL",))
    with pytest.raises(sqlite3.OperationalError):
        SQLiteTurnStore(path, upgrade_v1=True)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert SQLiteStateStore(path).snapshot(A).write_version == 0


def test_new_claims_respect_companion_intake_order(store):
    first, second = intake(store), intake(store, "second")
    with pytest.raises(VersionConflict, match="earlier"):
        claim(store, second)
    store.cancel(A, first.input_id)
    assert claim(store, second)


def test_lease_renewal_is_fenced_and_does_not_rewind(store):
    item = intake(store)
    run = claim(store, item)
    renewed = store.renew(run, now_us=120, lease_us=100)
    assert renewed.lease_until_us == 220
    assert store.renew(run, now_us=121, lease_us=10).lease_until_us == 220
    with pytest.raises(VersionConflict):
        claim(store, item, "other", now=219)
    next_run = claim(store, item, "other", now=220)
    with pytest.raises(VersionConflict):
        store.renew(run, now_us=221, lease_us=100)
    assert store.commit(result(next_run), now_us=221)


def test_ack_receipt_insert_failure_rolls_back_displayed_range(store):
    _, _, receipt = committed(store)
    store.publish(A, receipt["delivery_id"])
    with store.transaction() as db:
        db.execute(
            "CREATE TRIGGER break_ack BEFORE INSERT ON turn_acks BEGIN SELECT "
            "RAISE(ABORT,'injected'); END"
        )
    with pytest.raises(sqlite3.IntegrityError):
        ack(store, receipt, end=3)
    assert store.delivery(A, receipt["delivery_id"])["displayed_end"] == 0
    assert store.delivery(A, receipt["delivery_id"])["status"] == "published"


def test_competing_workers_cannot_create_two_runs(store):
    item = intake(store)
    barrier = Barrier(2)

    def take(worker):
        barrier.wait()
        try:
            return claim(store, item, worker)
        except VersionConflict:
            return None

    with ThreadPoolExecutor(2) as pool:
        claims = list(pool.map(take, ["one", "two"]))
    assert len([run for run in claims if run]) == 1
    with store.transaction(write=False) as db:
        assert db.execute("SELECT count(*) FROM turn_runs").fetchone()[0] == 1


@pytest.mark.parametrize("table", ["turn_deliveries", "turn_inputs", "turn_runs"])
def test_failure_after_commit_record_rolls_back_entire_business_commit(store, table):
    item = intake(store)
    run = claim(store, item)
    operation = "INSERT" if table == "turn_deliveries" else "UPDATE"
    with store.transaction() as db:
        db.execute(
            f"CREATE TRIGGER break_terminal BEFORE {operation} ON {table} "
            "BEGIN SELECT RAISE(ABORT,'injected'); END"
        )
    with pytest.raises(sqlite3.IntegrityError):
        store.commit(result(run), now_us=101)
    fact = store.inspect(A, item.input_id)
    assert fact["status"] == fact["run"]["status"] == "running"
    assert fact["commit"] is None
    assert fact["delivery"] is None


def test_forged_attempt_scope_basis_config_and_owner_cannot_commit(store):
    item = intake(store)
    run = claim(store, item)
    for changes in [
        dict(attempt_id="other"),
        dict(session_id="other"),
        dict(state_version=1),
        dict(config_json="{}"),
    ]:
        with pytest.raises(VersionConflict):
            store.commit(result(replace(run, **changes)), now_us=101)
    with pytest.raises(ValueError):
        replace(run, companion_id=B)
    with pytest.raises(ValueError):
        replace(run, state_version=True)
    assert store.inspect(A, item.input_id)["commit"] is None
