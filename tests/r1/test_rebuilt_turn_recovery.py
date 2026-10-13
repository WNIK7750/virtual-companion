import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest
from virtual_companion.boundary.contracts import OwnerToken, Scope, StateEvent, VersionConflict
from virtual_companion.boundary.sqlite_turns import SQLiteTurnStore
from virtual_companion.boundary.turns import CognitiveResult, RunClaim
from virtual_companion.text.identity import CompanionConfig, VersionedReference

A = "00000000-0000-0000-0000-000000000001"

_WORKER = """
import sys, os
sys.path.insert(0, sys.argv[1])
from contextlib import contextmanager
from datetime import datetime
from hashlib import sha256
from virtual_companion.boundary.sqlite_turns import SQLiteTurnStore
from virtual_companion.boundary.turns import CognitiveResult
from virtual_companion.boundary.contracts import Scope, StateEvent
class CrashStore(SQLiteTurnStore):
    @contextmanager
    def transaction(self, **kwargs):
        with super().transaction(**kwargs) as db:
            db.create_function('crash', 0, lambda: os._exit(23))
            yield db
store = CrashStore(sys.argv[2])
companion, input_id, point = sys.argv[3:]
run = store.claim(companion, input_id, 'child', now_us=100, lease_us=50)
if point == 'after_claim':
    os._exit(23)
when = datetime.fromisoformat('2026-09-09T12:00:00Z')
event = StateEvent(Scope(companion, run.session_id), 'assessment', 'state', when, when,
                   values_json='{"mood":1}', business_revision=1)
receipt = store.commit(CognitiveResult(run, 'candidate', 1, 'hello', 'review',
                       sha256(b'hello').hexdigest(), event), now_us=101)
if point == 'after_publish':
    store.publish(companion, receipt['delivery_id'])
os._exit(23)
"""


@pytest.mark.parametrize(
    "point", ["before_run", "after_claim", "before_commit", "after_commit", "after_publish"]
)
def test_abrupt_child_exit_recovers_atomic_turn_facts(tmp_path, point):
    store = SQLiteTurnStore(tmp_path / "turns.sqlite3")
    item = store.accept(
        CompanionConfig(A, VersionedReference("persona", "1", "a" * 64)), "first", "hi"
    )
    table = {"before_run": "turn_runs", "before_commit": "turn_commits"}.get(point)
    if table:
        with store.transaction() as db:
            db.execute(
                f"CREATE TRIGGER crash_write BEFORE INSERT ON {table} BEGIN SELECT crash(); END"
            )
    child = subprocess.run(  # noqa: S603 - fixed isolated child and synthetic database
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            _WORKER,
            str(Path(__file__).resolve().parents[2] / "apps/backend/src"),
            str(store.path),
            A,
            item.input_id,
            point,
        ],
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert child.returncode == 23, child.stderr.decode(errors="replace")
    reopened = SQLiteTurnStore(store.path)
    fact = reopened.inspect(A, item.input_id)
    if table:
        with reopened.transaction() as db:
            db.execute("DROP TRIGGER crash_write")
    if point == "before_run":
        assert fact["status"] == "queued"
        assert fact["run"] is None
        assert reopened.owner(A) is None
        assert reopened.claim(A, item.input_id, "recovery", now_us=200, lease_us=50)
    elif point in {"after_claim", "before_commit"}:
        assert fact["status"] == "running"
        assert fact["commit"] is None
        assert reopened.snapshot(A).write_version == 0
        assert reopened.receipt(A, "assessment") is None
        recovered = reopened.claim(A, item.input_id, "recovery", now_us=200, lease_us=50)
        assert recovered.attempt_id == fact["run"]["attempt"]
        assert recovered.owner.epoch > fact["run"]["epoch"]
    else:
        assert fact["status"] == fact["run"]["status"] == "committed"
        assert reopened.snapshot(A).values_json == '{"mood":1}'
        with reopened.transaction(write=False) as db:
            payload = json.loads(db.execute("SELECT payload FROM turn_commits").fetchone()[0])
        payload["claim"]["owner"] = OwnerToken(**payload["claim"]["owner"])
        payload["claim"] = RunClaim(**payload["claim"])
        payload["assessment"]["scope"] = Scope(**payload["assessment"]["scope"])
        for field in ("occurred_at", "received_at"):
            payload["assessment"][field] = datetime.fromisoformat(payload["assessment"][field])
        payload["assessment"] = StateEvent(**payload["assessment"])
        proposal = CognitiveResult(**payload)
        assert reopened.commit(proposal, now_us=1000) == fact["commit"]
        assert reopened.snapshot(A).write_version == 1
        delivery = fact["commit"]["delivery_id"]
        assert reopened.delivery(A, delivery)["displayed_end"] == 0
        assert reopened.delivery(A, delivery)["status"] == (
            "published" if point == "after_publish" else "pending"
        )
        with pytest.raises(VersionConflict, match="committed"):
            reopened.claim(A, item.input_id, "recovery", now_us=200, lease_us=50)
        if point == "after_publish":
            assert reopened.publication(A, delivery)["text"] == "hello"
            assert reopened.publish(A, delivery) is None
