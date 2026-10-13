import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from test_rebuilt_presentation import approved, candidate
from test_rebuilt_turns import A, B, claim, config, intake, result
from virtual_companion.boundary.contracts import Scope, VersionConflict
from virtual_companion.boundary.sqlite_lifecycle import SQLiteLifecycleStore
from virtual_companion.boundary.sqlite_presentation import SQLitePresentationStore


class V5SchemaStore(SQLitePresentationStore):
    """Synthetic v5 format fixture; no retired policy implementation."""

    schema_version = 5


def proposal(run):
    value = replace(
        candidate(),
        scope=Scope(A, run.session_id),
        input_id=run.input_id,
        attempt_id=run.attempt_id,
    )
    return replace(result(run), presentation=approved(value))


def test_failed_retry_preserves_intake_and_attempt_configs(tmp_path):
    store = SQLiteLifecycleStore(tmp_path / "runs.db")
    item = intake(store)
    first = claim(store, item)
    failure = store.fail(first, reason="model_timeout", now_us=101)
    assert store.inspect(A, item.input_id)["status"] == "failed"
    assert store.fail(first, reason="model_timeout", now_us=900) == failure
    with pytest.raises(VersionConflict):
        claim(store, item, now=200)
    retry = store.retry(A, item.input_id, first.attempt_id, "retry-1", config(version="2"))
    store = SQLiteLifecycleStore(store.path)
    second = claim(store, item, now=200)
    assert second.attempt_id != first.attempt_id
    assert second.config_json != first.config_json
    assert intake(store) == item
    assert store.retry(A, item.input_id, first.attempt_id, "retry-1", config(version="2")) == retry
    with pytest.raises(ValueError):
        store.retry(A, item.input_id, first.attempt_id, "retry-1", config(version="3"))
    with pytest.raises(VersionConflict):
        store.commit(proposal(first), now_us=201)
    committed = store.commit(proposal(second), now_us=201)
    assert store.fail(second, reason="response_lost", now_us=999) == committed
    history = store.attempts(A, item.input_id)
    assert [r["status"] for r in history] == ["failed", "committed"]
    assert [r["config"] for r in history] == [first.config_json, second.config_json]


def test_expiry_is_recovery_not_failure_or_retry(tmp_path):
    store = SQLiteLifecycleStore(tmp_path / "runs.db")
    item = intake(store)
    first = claim(store, item)
    with pytest.raises(VersionConflict):
        store.fail(first, reason="timeout", now_us=150)
    with pytest.raises(VersionConflict):
        store.retry(A, item.input_id, first.attempt_id, "retry", config())
    second = claim(store, item, worker="replacement", now=150)
    assert second.attempt_id == first.attempt_id
    with pytest.raises(VersionConflict):
        store.fail(first, reason="timeout", now_us=151)
    store.fail(second, reason="model_error", now_us=151)
    later = intake(store, "later")
    assert claim(store, later, now=152).input_id == later.input_id


def test_failure_and_retry_rollback_and_scope(tmp_path):
    store = SQLiteLifecycleStore(tmp_path / "runs.db")
    item = intake(store)
    run = claim(store, item)
    with store.transaction() as db:
        db.execute(
            "CREATE TRIGGER fail_archive BEFORE INSERT ON turn_attempt_history "
            "BEGIN SELECT RAISE(ABORT,'injected'); END"
        )
    with pytest.raises(sqlite3.IntegrityError):
        store.fail(run, reason="error", now_us=101)
    assert store.inspect(A, item.input_id)["status"] == "running"
    with store.transaction() as db:
        db.execute("DROP TRIGGER fail_archive")
    store.fail(run, reason="error", now_us=101)
    with pytest.raises(ValueError):
        store.retry(A, item.input_id, run.attempt_id, "r", config(B))
    with store.transaction() as db:
        db.execute(
            "CREATE TRIGGER fail_retry BEFORE INSERT ON turn_retry_receipts "
            "BEGIN SELECT RAISE(ABORT,'injected'); END"
        )
    with pytest.raises(sqlite3.IntegrityError):
        store.retry(A, item.input_id, run.attempt_id, "r", config())
    assert store.inspect(A, item.input_id)["status"] == "failed"
    assert len(store.attempts(A, item.input_id)) == 1
    store.cancel(A, item.input_id)
    with pytest.raises(VersionConflict):
        store.retry(A, item.input_id, run.attempt_id, "r2", config())


def test_upgrade_populated_v5_and_backup_restore(tmp_path):
    path = tmp_path / "runs.db"
    old = V5SchemaStore(path)
    item = intake(old)
    run = claim(old, item)
    original = old.commit(proposal(run), now_us=101)
    with old.transaction(write=False) as db, sqlite3.connect(tmp_path / "backup.db") as target:
        db.backup(target)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        SQLiteLifecycleStore(path)
    assert path.read_bytes() == before
    store = SQLiteLifecycleStore(path, upgrade_v5=True)
    assert store.commit(proposal(run), now_us=999) == original
    with store.transaction(write=False) as db:
        assert list(db.execute("PRAGMA foreign_key_check")) == []
    with pytest.raises(ValueError):
        V5SchemaStore(path)
    restored = V5SchemaStore(tmp_path / "backup.db")
    assert restored.inspect(A, item.input_id)["commit"] == original
    assert json.loads(store.attempts(A, item.input_id)[0]["config"])


def test_concurrent_retry_has_one_new_attempt(tmp_path):
    store = SQLiteLifecycleStore(tmp_path / "runs.db")
    item = intake(store)
    run = claim(store, item)
    store.fail(run, reason="error", now_us=101)

    def retry(_):
        return SQLiteLifecycleStore(store.path).retry(
            A, item.input_id, run.attempt_id, "retry", config()
        )

    with ThreadPoolExecutor(2) as pool:
        replies = list(pool.map(retry, range(2)))
    assert replies[0] == replies[1]
    new = claim(store, item, now=200)
    assert len(store.attempts(A, item.input_id)) == 2
    with pytest.raises(VersionConflict):
        store.retry(A, item.input_id, run.attempt_id, "different", config())
    store.cancel(A, item.input_id)
    assert store.attempts(A, item.input_id)[-1]["status"] == "cancelled"
    assert new.attempt_id != run.attempt_id


def test_upgrade_failure_restores_entire_original_schema(tmp_path):
    path = tmp_path / "runs.db"
    old = V5SchemaStore(path)
    item = intake(old)
    run = claim(old, item)
    with old.transaction() as db:
        db.execute("CREATE INDEX original_input_index ON turn_inputs(client_key)")
    before = path.read_bytes()

    class BrokenUpgrade(SQLiteLifecycleStore):
        def _upgrade(self, db, *args):
            super()._upgrade(db, *args)
            raise RuntimeError("crash before commit")

    with pytest.raises(RuntimeError):
        BrokenUpgrade(path, upgrade_v5=True)
    assert path.read_bytes() == before
    assert V5SchemaStore(path).inspect(A, item.input_id)["run"]["attempt"] == run.attempt_id
    upgraded = SQLiteLifecycleStore(path, upgrade_v5=True)
    with upgraded.transaction(write=False) as db:
        assert db.execute(
            "SELECT name FROM sqlite_schema WHERE name='original_input_index'"
        ).fetchone()
        assert db.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_failure_after_cancel_does_not_reclassify_input(tmp_path):
    store = SQLiteLifecycleStore(tmp_path / "runs.db")
    item = intake(store)
    run = claim(store, item)
    store.cancel(A, item.input_id)
    with pytest.raises(VersionConflict):
        store.fail(run, reason="cancelled_model", now_us=101)
    assert store.inspect(A, item.input_id)["status"] == "cancelled"
