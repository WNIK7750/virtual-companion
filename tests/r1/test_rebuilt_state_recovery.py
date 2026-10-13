import hashlib
import json
import sqlite3
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pytest
from virtual_companion.boundary.application import CompanionState
from virtual_companion.boundary.contracts import Scope, StateEvent, VersionConflict
from virtual_companion.boundary.sqlite_state import SQLiteStateStore
from virtual_companion.text.identity import CompanionConfig, VersionedReference

A = "00000000-0000-0000-0000-000000000001"
WHEN = datetime.fromisoformat("2026-09-09T12:00:00Z")


@pytest.mark.parametrize("format_kind", ["foreign", "future"])
def test_unknown_database_rejected_without_mutation(tmp_path, format_kind):
    path = tmp_path / "old.sqlite3"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE old_presence(started_at TEXT, ended_at TEXT)")
        db.execute("INSERT INTO old_presence VALUES(?,?)", ("2026-09-09T12:00:00Z", None))
        if format_kind == "future":
            db.execute("PRAGMA user_version=99")
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        SQLiteStateStore(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    # Existing-format reader remains usable after rejected candidate adapter startup.
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT count(*) FROM old_presence").fetchone()[0] == 1


def test_backup_restores_snapshot_owner_and_idempotency_receipts(tmp_path):
    store = SQLiteStateStore(tmp_path / "new.sqlite3")
    store.register(A)
    owner = store.claim_owner(A, "worker", expected_epoch=0)
    event = StateEvent(Scope(A, "session"), "input", "presence", WHEN, WHEN, availability="busy")
    receipt = store.apply(event, owner, expected_version=0)
    backup = tmp_path / "restore.sqlite3"
    source = sqlite3.connect(store.path)
    destination = sqlite3.connect(backup)
    try:
        source.backup(destination)
    finally:
        source.close()
        destination.close()
    restored = SQLiteStateStore(backup)
    assert restored.snapshot(A) == store.snapshot(A)
    assert restored.owner(A) == owner
    assert restored.apply(event, owner, expected_version=0) == receipt


def test_config_replacement_keeps_state_and_optimistic_basis(tmp_path):
    store = SQLiteStateStore(tmp_path / "state.sqlite3")
    store.register(A)
    owner = store.claim_owner(A, "worker", expected_epoch=0)
    config = CompanionConfig(A, VersionedReference("synthetic", "1", "a" * 64))
    app = CompanionState(config, store)
    basis = app.snapshot()
    event = StateEvent(Scope(A, "one"), "input", "state", WHEN, WHEN, values_json='{"v":0.9}')
    app.commit(event, basis, owner)
    replacement = config.with_resources({"avatar": VersionedReference("avatar", "2", "b" * 64)})
    other_session = CompanionState(replacement, store)
    assert other_session.snapshot() == app.snapshot()
    later = StateEvent(Scope(A, "two"), "next", "state", WHEN, WHEN, values_json='{"v":0.1}')
    with pytest.raises(VersionConflict):
        other_session.commit(later, basis, owner)
    assert other_session.commit(later, app.snapshot(), owner).disposition == "same_time"
    assert app.snapshot().values_json == '{"v":0.9}'


_CRASH_WORKER = """
import sys, os
sys.path.insert(0, sys.argv[1])
from contextlib import contextmanager
from datetime import datetime
from virtual_companion.boundary.sqlite_state import SQLiteStateStore
from virtual_companion.boundary.contracts import Scope, StateEvent
class CrashStore(SQLiteStateStore):
    @contextmanager
    def transaction(self, **kwargs):
        with super().transaction(**kwargs) as db:
            db.create_function("crash", 0, lambda: os._exit(23))
            yield db
store = CrashStore(sys.argv[2])
owner = store.owner(sys.argv[3])
when = datetime.fromisoformat("2026-09-09T12:00:00Z")
event = StateEvent(Scope(sys.argv[3], "session"), "input", "presence", when, when,
                   availability="busy")
store.apply(event, owner, expected_version=0)
os._exit(23)
"""


@pytest.mark.parametrize("crash_point", ["before_receipt", "after_commit"])
def test_process_exit_at_commit_boundary_recovers_atomic_facts(tmp_path, crash_point):
    path = tmp_path / "state.sqlite3"
    store = SQLiteStateStore(path)
    store.register(A)
    owner = store.claim_owner(A, "worker", expected_epoch=0)
    if crash_point == "before_receipt":
        with store.transaction() as db:
            db.execute(
                "CREATE TRIGGER crash_receipt BEFORE INSERT ON state_receipts "
                "BEGIN SELECT crash(); END"
            )
    process = subprocess.run(  # noqa: S603 - fixed isolated child with synthetic database
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            _CRASH_WORKER,
            str(Path(__file__).resolve().parents[2] / "apps/backend/src"),
            str(path),
            A,
        ],
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert process.returncode == 23, process.stderr.decode(errors="replace")
    recovered = SQLiteStateStore(path)
    if crash_point == "before_receipt":
        assert recovered.snapshot(A).write_version == 0
        assert recovered.periods(A) == []
        assert recovered.receipt(A, "input") is None
        with recovered.transaction() as db:
            db.execute("DROP TRIGGER crash_receipt")
    else:
        assert recovered.snapshot(A).write_version == 1
        assert len(recovered.periods(A)) == 1
    event = StateEvent(Scope(A, "session"), "input", "presence", WHEN, WHEN, availability="busy")
    receipt = recovered.apply(event, owner, expected_version=0)
    assert receipt.write_version == 1
    assert len(recovered.periods(A)) == 1
    assert json.loads(json.dumps(asdict(receipt)))["event_id"] == "input"
