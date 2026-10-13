import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path

import pytest
from virtual_companion.text.a3_budget import ExperimentLedger


def limits():
    return {
        "seeds": [1, 2],
        "total_budget": {"max_calls": 3, "max_output_tokens": 30, "max_wall_seconds": 20},
        "per_seed_session_budget": {
            "max_calls": 2,
            "max_output_tokens": 20,
            "max_wall_seconds": 10,
        },
    }


def test_protocol_invalid_keeps_reservation_and_unknown_semantics_after_reopen(tmp_path):
    path = tmp_path / "protocol.sqlite3"
    with ExperimentLedger(path, "experiment", "a" * 64, limits()) as ledger:
        session = ledger.start_session(1)
        key = ledger.reserve(session, "control", "invalid", 1, "causal_review", 10)
        ledger.finish(key, "protocol_invalid", {"semantic_status": "unknown"})
        ledger.end_session(session, 1)
    with ExperimentLedger(path, "experiment", "a" * 64, limits()) as ledger:
        assert ledger.totals()["calls"] == 1
        assert ledger.records()[0]["status"] == "protocol_invalid"
        assert json.loads(ledger.records()[0]["metrics"])["semantic_status"] == "unknown"


def test_smaller_pilot_wall_reservation_survives_unclean_reopen(tmp_path):
    path = tmp_path / "pilot.sqlite3"
    with ExperimentLedger(path, "experiment", "a" * 64, limits()) as ledger:
        session = ledger.start_session(1, wall_cap=3)
        assert ledger.wall_remaining(session) == 3
    with ExperimentLedger(path, "experiment", "a" * 64, limits()) as ledger:
        assert ledger.totals()["wall_seconds"] == 3
        with pytest.raises(ValueError):
            ledger.start_session(1, wall_cap=0)


def test_budget_atomic_concurrent_reservations_and_terminal_retention(tmp_path):
    with ExperimentLedger(tmp_path / "ledger.sqlite3", "experiment", "a" * 64, limits()) as ledger:
        session = ledger.start_session(1)

        def call(i):
            try:
                return ledger.reserve(session, "candidate", f"c{i}", 1, "reply", 10)
            except ValueError:
                return None

        with concurrent.futures.ThreadPoolExecutor(4) as pool:
            ids = [i for i in pool.map(call, range(4)) if i is not None]
        assert len(ids) == 2
        ledger.finish(ids[0], "cancelled", {"actual_output_tokens": None})
        ledger.finish(ids[1], "failed", {})
        assert ledger.totals()["calls"] == 2
        ledger.end_session(session, 2)
        with pytest.raises(ValueError, match="budget"):
            ledger.reserve(ledger.start_session(1), "control", "c5", 1, "reply", 10)


def test_reopen_keeps_uncertain_reservation_and_prevents_duplicate_attempt(tmp_path):
    path = tmp_path / "ledger.sqlite3"
    with ExperimentLedger(path, "experiment", "a" * 64, limits()) as ledger:
        session = ledger.start_session(1)
        ledger.reserve(session, "candidate", "case", 1, "reply", 10)
    with ExperimentLedger(path, "experiment", "a" * 64, limits()) as ledger:
        assert ledger.totals() == {"calls": 1, "output_tokens": 10, "wall_seconds": 10.0}
        rows = ledger.records()
        assert rows[0]["status"] == "interrupted"
        with pytest.raises(ValueError, match="budget"):
            ledger.start_session(1)
        session = ledger.start_session(2)
        key = ledger.reserve(session, "control", "case", 1, "reply", 10)
        with pytest.raises(ValueError, match="duplicate"):
            ledger.reserve(session, "control", "case", 1, "reply", 10)
        ledger.finish(key, "completed", {"actual_output_tokens": 3})
        assert json.loads(ledger.records()[-1]["metrics"])["actual_output_tokens"] == 3


def test_ledger_rejects_new_freeze_and_simultaneous_owner(tmp_path):
    path = tmp_path / "ledger.sqlite3"
    with ExperimentLedger(path, "experiment", "a" * 64, limits()):
        with pytest.raises(RuntimeError, match="owned"):
            with ExperimentLedger(path, "experiment", "a" * 64, limits()):
                pass
    with pytest.raises(ValueError, match="identity"):
        with ExperimentLedger(path, "experiment", "b" * 64, limits()):
            pass


def test_actual_process_crash_retains_charge_and_releases_os_lock(tmp_path):
    path = tmp_path / "ledger.sqlite3"
    code = """
import json, os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[3])
from virtual_companion.text.a3_budget import ExperimentLedger
with ExperimentLedger(Path(sys.argv[1]), 'crash', 'a'*64, json.loads(sys.argv[2])) as ledger:
    session = ledger.start_session(1)
    ledger.reserve(session, 'control', 'case', 1, 'reply', 10)
    os._exit(7)
"""
    child = subprocess.run(  # noqa: S603 - fixed child probe, no shell
        [
            sys.executable,
            "-c",
            code,
            str(path),
            json.dumps(limits()),
            str(Path(__file__).resolve().parents[2] / "apps/backend/src"),
        ],
        check=False,
        timeout=10,
    )
    assert child.returncode == 7
    with ExperimentLedger(path, "crash", "a" * 64, limits()) as ledger:
        assert ledger.records()[0]["status"] == "interrupted"
        assert ledger.totals() == {"calls": 1, "output_tokens": 10, "wall_seconds": 10.0}


def test_explicit_freeze_successor_preserves_totals_and_per_call_provenance(tmp_path):
    path = tmp_path / "ledger.sqlite3"
    with ExperimentLedger(path, "experiment", "a" * 64, limits()) as ledger:
        session = ledger.start_session(1)
        call = ledger.reserve(session, "control", "old-case", 1, "reply", 10)
        ledger.finish(call, "failed", {})
        ledger.end_session(session, 2)
    with ExperimentLedger(
        path, "experiment", "b" * 64, limits(), previous_freeze_hash="a" * 64
    ) as ledger:
        assert ledger.totals() == {"calls": 1, "output_tokens": 10, "wall_seconds": 2.0}
        session = ledger.start_session(1)
        call = ledger.reserve(session, "candidate", "new-case", 1, "reply", 10)
        ledger.finish(call, "completed", {})
        assert [r["freeze_hash"] for r in ledger.records()] == ["a" * 64, "b" * 64]


def test_successor_cannot_increase_caps_and_failed_transition_is_atomic(tmp_path):
    path = tmp_path / "ledger.sqlite3"
    with ExperimentLedger(path, "experiment", "a" * 64, limits()):
        pass
    raised = limits()
    raised["total_budget"]["max_calls"] += 1
    with pytest.raises(ValueError, match="budget"):
        with ExperimentLedger(path, "experiment", "b" * 64, raised, previous_freeze_hash="a" * 64):
            pass
    with ExperimentLedger(path, "experiment", "a" * 64, limits()) as ledger:
        assert ledger.totals()["calls"] == 0
