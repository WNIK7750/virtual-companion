import subprocess
import sys
from pathlib import Path

import pytest
from test_rebuilt_memory import A, ack, output_effect, turn
from virtual_companion.boundary.sqlite_memory import SQLiteMemoryStore

_WORKER = """
import os, sys, json
sys.path.insert(0, sys.argv[1])
from contextlib import contextmanager
from virtual_companion.boundary.sqlite_memory import SQLiteMemoryStore
class CrashStore(SQLiteMemoryStore):
    @contextmanager
    def transaction(self, **kwargs):
        with super().transaction(**kwargs) as db:
            db.create_function('crash', 0, lambda: os._exit(23))
            yield db
store = CrashStore(sys.argv[2])
companion, delivery = sys.argv[3:]
event = store.publication(companion, delivery)
store.acknowledge(companion, delivery, candidate_id=event['candidate_id'],
                  revision=event['revision'], sha256=event['sha256'],
                  sequence=1, displayed_end=2)
os._exit(23)
"""


@pytest.mark.parametrize("point", ["before_memory_receipt", "after_ack_commit"])
def test_process_exit_keeps_ack_and_output_memory_atomic(tmp_path, point):
    store = SQLiteMemoryStore(tmp_path / "memory.sqlite3")
    _, _, receipt = turn(store, effects=(output_effect(),))
    store.publish(A, receipt["delivery_id"])
    if point == "before_memory_receipt":
        with store.transaction() as db:
            db.execute(
                "CREATE TRIGGER crash_memory BEFORE INSERT ON memory_commands "
                "BEGIN SELECT crash(); END"
            )
    child = subprocess.run(  # noqa: S603 - fixed isolated child, synthetic SQLite only
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            _WORKER,
            str(Path(__file__).resolve().parents[2] / "apps/backend/src"),
            str(store.path),
            A,
            receipt["delivery_id"],
        ],
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert child.returncode == 23, child.stderr.decode(errors="replace")
    reopened = SQLiteMemoryStore(store.path)
    if point == "before_memory_receipt":
        assert reopened.delivery(A, receipt["delivery_id"])["displayed_end"] == 0
        assert reopened.recall(A, now_us=0) == []
        assert reopened.snapshot(A).write_version == 0
        assert reopened.effects(A, receipt["delivery_id"])[0]["status"] == "pending"
        with reopened.transaction() as db:
            db.execute("DROP TRIGGER crash_memory")
    else:
        assert reopened.delivery(A, receipt["delivery_id"])["displayed_end"] == 2
        assert len(reopened.recall(A, now_us=0)) == 1
        assert reopened.snapshot(A).write_version == 1
    ack(reopened, receipt, 2)
    ack(reopened, receipt, 2)
    assert len(reopened.history(A, "said")) == 1
    assert reopened.snapshot(A).write_version == 1
