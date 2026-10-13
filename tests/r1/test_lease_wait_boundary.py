import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from test_cognition_preparation import basis
from test_cognitive_kernel import RecordedModel, kernel
from virtual_companion.boundary.contracts import VersionConflict
from virtual_companion.cognition.preparation import CallControl, CallDeadline


def test_second_inspection_cannot_release_expired_work(tmp_path):
    _, _, run, _ = basis(tmp_path)
    now = [100]
    inspections = []

    def inspect():
        inspections.append(True)
        if len(inspections) == 2:
            now[0] = 150
        return False

    control = CallControl(
        run,
        clock=lambda: now[0],
        cancelled=inspect,
        renew=lambda *_: pytest.fail("unexpected renewal"),
        deadline_us=300,
        renewal_margin_us=15,
    )
    with pytest.raises(VersionConflict, match="Expired"):
        control.checkpoint()


def test_renewal_return_cannot_exceed_absolute_deadline(tmp_path):
    _, _, run, _ = basis(tmp_path)
    now = [140]

    def renew(claim, before):
        now[0] = 310
        return replace(claim, lease_until_us=400)

    control = CallControl(
        run,
        clock=lambda: now[0],
        cancelled=lambda: False,
        renew=renew,
        deadline_us=300,
        renewal_margin_us=15,
    )
    with pytest.raises(CallDeadline):
        control.checkpoint()


def test_sqlite_renew_rechecks_clock_after_waiting_for_write_lock(tmp_path):
    store, item, run, _ = basis(tmp_path)
    now = [140]
    entered = threading.Event()

    def renew():
        entered.set()
        return store.renew(run, now_us=140, lease_us=50, clock=lambda: now[0])

    with sqlite3.connect(store.path, isolation_level=None) as blocker:
        blocker.execute("BEGIN IMMEDIATE")
        with ThreadPoolExecutor(1) as pool:
            result = pool.submit(renew)
            assert entered.wait(1)
            time.sleep(0.05)
            now[0] = 151
            blocker.execute("ROLLBACK")
            with pytest.raises(VersionConflict, match="expired"):
                result.result(timeout=2)
    assert store.inspect(item.companion_id, item.input_id)["run"]["lease_until"] == 150


def test_sqlite_control_read_has_explicit_lock_wait_limit(tmp_path):
    store, item, _, _ = basis(tmp_path)
    with sqlite3.connect(store.path, isolation_level=None) as blocker:
        blocker.execute("BEGIN EXCLUSIVE")
        started = time.monotonic()
        try:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                store.inspect(item.companion_id, item.input_id, timeout=0.025)
            assert time.monotonic() - started < 0.5
        finally:
            blocker.execute("ROLLBACK")


def test_commit_cannot_publish_a_candidate_using_time_before_lock_wait(tmp_path):
    store, item, run, persona = basis(tmp_path)
    result = kernel(store, persona, RecordedModel()).generate(
        run, store.snapshot(run.companion_id), "input"
    )
    now = [140]
    entered = threading.Event()

    def commit():
        entered.set()
        return store.commit(result, now_us=140, clock=lambda: now[0])

    with sqlite3.connect(store.path, isolation_level=None) as blocker:
        blocker.execute("BEGIN IMMEDIATE")
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(commit)
            assert entered.wait(1)
            time.sleep(0.05)
            now[0] = 151
            blocker.execute("ROLLBACK")
            with pytest.raises(VersionConflict, match="expired"):
                future.result(timeout=2)
    assert store.inspect(item.companion_id, item.input_id)["commit"] is None
