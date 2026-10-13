import json
import sqlite3
import subprocess
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest
from test_rebuilt_lifecycle import proposal
from test_rebuilt_memory import mutation
from test_rebuilt_turns import A, B, config, intake
from virtual_companion.boundary.contracts import Scope, StateEvent, VersionConflict
from virtual_companion.boundary.memory import MemoryEffect
from virtual_companion.boundary.runtime import CompanionRuntime, KernelFailure
from virtual_companion.boundary.sqlite_harness import SQLiteHarnessStore

NOW = datetime.fromisoformat("2026-09-27T10:00:00Z")


def setup(tmp_path):
    store = SQLiteHarnessStore(tmp_path / "current-runtime.db")
    return store, intake(store)


def effect(item):
    return MemoryEffect(replace(mutation(), scope=Scope(A, item.session_id),
                                source_kind="output", source_id="candidate"), 2)


def test_kernel_io_is_outside_transaction_and_unknown_commit_reconciles(tmp_path):
    class LostResponse(SQLiteHarnessStore):
        def commit(self, result, **kwargs):
            super().commit(result, **kwargs)
            raise OSError("lost response after commit")

    store = LostResponse(tmp_path / "runtime.db")
    runtime = CompanionRuntime(config(), store)
    item = runtime.accept("one", "你好")

    class Kernel:
        calls = 0

        def generate(self, run, snapshot, text):
            self.calls += 1
            assert text == "你好" and snapshot.write_version == run.state_version
            with sqlite3.connect(store.path, timeout=0.05) as connection:
                connection.execute("BEGIN IMMEDIATE")
            return proposal(run)

    kernel = Kernel()
    with pytest.raises(OSError):
        runtime.execute(item.input_id, "worker", kernel, clock=lambda: 100, lease_us=50)
    runtime = CompanionRuntime(config(), SQLiteHarnessStore(store.path))
    committed = runtime.execute(item.input_id, "recovery", kernel, clock=lambda: 500, lease_us=50)
    assert committed["status"] == "committed"
    assert kernel.calls == 1


def test_known_failure_needs_explicit_retry_unknown_failure_retains_attempt(tmp_path):
    store = SQLiteHarnessStore(tmp_path / "runtime.db")
    runtime = CompanionRuntime(config(), store)
    item = runtime.accept("one", "你好")

    class Kernel:
        error = KernelFailure("model_timeout")

        def generate(self, *args):
            raise self.error

    kernel = Kernel()
    failed = runtime.execute(item.input_id, "worker", kernel, clock=lambda: 100, lease_us=50)
    assert failed["status"] == "failed"
    runtime.retry(item.input_id, failed["attempt"], "retry", config(version="2"))
    kernel.error = OSError("unknown interruption")
    with pytest.raises(OSError):
        runtime.execute(item.input_id, "worker", kernel, clock=lambda: 200, lease_us=50)
    assert store.inspect(A, item.input_id)["status"] == "running"
    assert len(store.attempts(A, item.input_id)) == 2


def test_scope_and_kernel_claim_mismatch_rejected_before_commit(tmp_path):
    store = SQLiteHarnessStore(tmp_path / "runtime.db")
    runtime = CompanionRuntime(config(), store)
    other = CompanionRuntime(config(B), store).accept("one", "other")
    with pytest.raises(ValueError):
        runtime.execute(other.input_id, "worker", None, clock=lambda: 100, lease_us=50)
    item = runtime.accept("one", "你好")

    class Kernel:
        def generate(self, run, *args):
            return proposal(replace(run, input_id="different-input"))

    with pytest.raises(ValueError):
        runtime.execute(item.input_id, "worker", Kernel(), clock=lambda: 100, lease_us=50)
    assert store.inspect(A, item.input_id)["commit"] is None


def test_model_wait_cancellation_fences_late_result(tmp_path):
    store = SQLiteHarnessStore(tmp_path / "runtime.db")
    runtime = CompanionRuntime(config(), store)
    item = runtime.accept("one", "你好")

    class Kernel:
        def generate(self, run, *args):
            runtime.cancel(item.input_id)
            return proposal(run)

    with pytest.raises(VersionConflict):
        runtime.execute(item.input_id, "worker", Kernel(), clock=lambda: 100, lease_us=50)
    assert store.inspect(A, item.input_id)["commit"] is None


@pytest.mark.parametrize("crash_before_commit", [True, False])
def test_real_child_exit_keeps_ack_and_memory_atomic(tmp_path, crash_before_commit):
    store, item = setup(tmp_path)
    run = store.claim(A, item.input_id, "worker", now_us=100, lease_us=50)
    receipt = store.commit(
        replace(proposal(run), memories=(effect(item),)),
        now_us=101,
    )
    store.publish(A, receipt["delivery_id"])
    before = store.snapshot(A)
    args = dict(
        candidate_id="candidate",
        revision=1,
        sha256=receipt["sha256"],
        presentation_sha256=receipt["presentation_sha256"],
        sequence=1,
        displayed_end=2,
    )
    arguments = tmp_path / "ack.json"
    arguments.write_text(json.dumps(args), encoding="utf-8")
    if crash_before_commit:
        with store.transaction() as db:
            db.execute(
                "CREATE TRIGGER crash_ack BEFORE INSERT ON memory_commands "
                "BEGIN SELECT crash_process(); END"
            )
    script = """
import json, os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[5])
from virtual_companion.boundary.sqlite_harness import SQLiteHarnessStore
class CrashStore(SQLiteHarnessStore):
    def _prepare_connection(self, db, **kwargs):
        super()._prepare_connection(db, **kwargs)
        db.create_function('crash_process', 0, lambda: os._exit(73))
store = CrashStore(Path(sys.argv[1]))
store.acknowledge(sys.argv[2], sys.argv[3], **json.loads(Path(sys.argv[4]).read_text()))
os._exit(74)
"""
    process = subprocess.run(  # noqa: S603 - fixed diagnostic code and synthetic paths
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            script,
            str(store.path),
            A,
            receipt["delivery_id"],
            str(arguments),
            str(Path(__file__).resolve().parents[2] / "apps/backend/src"),
        ],
        timeout=20,
        check=False,
        capture_output=True,
    )
    assert process.returncode == (73 if crash_before_commit else 74), process.stderr
    reopened = SQLiteHarnessStore(store.path)
    if crash_before_commit:
        assert reopened.snapshot(A) == before
        assert reopened.inspect(A, item.input_id)["delivery"]["displayed_end"] == 0
        with reopened.transaction() as db:
            db.execute("DROP TRIGGER crash_ack")
    reopened.acknowledge(A, receipt["delivery_id"], **args)
    assert reopened.history(A, "event-one")[0]["content"] == mutation().content
    assert reopened.snapshot(A).write_version == before.write_version + 1


def test_incompatible_background_state_cannot_wedge_confirmed_ack(tmp_path):
    from datetime import timedelta

    store, item = setup(tmp_path)
    run = store.claim(A, item.input_id, "worker", now_us=100, lease_us=50)
    receipt = store.commit(
        replace(proposal(run), memories=(effect(item),)),
        now_us=101,
    )
    store.publish(A, receipt["delivery_id"])
    later = NOW + timedelta(minutes=1)
    store.apply(
        StateEvent(
            Scope(A, item.session_id),
            "different-state-shape",
            "state",
            later,
            later,
            values_json='{"revision":0}',
        ),
        store.owner(A),
        expected_version=store.snapshot(A).write_version,
    )
    store.acknowledge(
        A,
        receipt["delivery_id"],
        candidate_id="candidate",
        revision=1,
        sha256=receipt["sha256"],
        presentation_sha256=receipt["presentation_sha256"],
        sequence=1,
        displayed_end=3,
    )
    assert store.history(A, "event-one")[0]["content"] == mutation().content
    assert store.snapshot(A).values_json == '{"revision":0}'
    assert store.inspect(A, item.input_id)["delivery"]["status"] == "completed"
