"""Current ownership/storage works with retired policy modules inaccessible."""

import json
import subprocess
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest
from test_rebuilt_lifecycle import proposal
from test_rebuilt_memory import mutation
from test_rebuilt_turns import A, claim, intake
from virtual_companion.boundary.contracts import Scope, StateEvent
from virtual_companion.boundary.memory import MemoryEffect
from virtual_companion.boundary.sqlite_harness import SQLiteHarnessStore
from virtual_companion.boundary.sqlite_memory_management import SQLiteExplicitMemoryStore


def test_current_store_import_does_not_load_retired_policy():
    script = '''
import importlib.abc, sys
sys.path.insert(0, sys.argv[1])
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.rsplit('.', 1)[-1] in {
            'dynamics', 'relationship', 'sqlite_dynamics', 'sqlite_relationship'
        }:
            raise AssertionError('retired import: ' + fullname)
sys.meta_path.insert(0, Block())
from virtual_companion.boundary.sqlite_memory_management import SQLiteExplicitMemoryStore
from virtual_companion.boundary import ports, turns
assert SQLiteExplicitMemoryStore.schema_version == 19
'''
    result = subprocess.run(  # noqa: S603 - fixed script and repository source path
        [sys.executable, "-I", "-B", "-c", script,
         str(Path(__file__).resolve().parents[2] / "apps/backend/src")],
        capture_output=True, text=True, check=False, timeout=20,
    )
    assert result.returncode == 0, result.stderr


def test_current_interruption_records_no_retired_discovery(tmp_path):
    store = SQLiteExplicitMemoryStore(tmp_path / "current.db")
    item = intake(store)
    owner = store.claim_owner(A, "state-worker", expected_epoch=0)
    when = datetime.fromisoformat("2026-09-27T10:00:00Z")
    for key, kind, availability in [("busy", "presence", "busy"), ("input", "interrupt", None)]:
        event = StateEvent(Scope(A, item.session_id), key, kind, when, when,
                           availability=availability)
        receipt = store.apply(event, owner, expected_version=store.snapshot(A).write_version)
    assert receipt.discovery_probability is None
    assert receipt.draw_value is None
    assert receipt.reaction_code is None
    assert store.periods(A)[0]["interruptions"] == 1
    assert SQLiteExplicitMemoryStore(store.path).receipt(A, "input") == receipt


def test_nonempty_retired_relationship_effects_are_rejected(tmp_path):
    store = SQLiteExplicitMemoryStore(tmp_path / "current.db")
    item = intake(store)
    run = claim(store, item)
    with pytest.raises(ValueError, match="[Rr]etired"):
        replace(proposal(run), relationships=({"old_effect": "opaque"},))
    with pytest.raises(ValueError, match="[Rr]etired"):
        store.mutate_relationship(object(), run.owner, expected_version=run.state_version)
    assert store.inspect(A, item.input_id)["commit"] is None


def test_pending_retired_effect_is_opaque_across_current_memory_ack_and_reopen(tmp_path):
    store = SQLiteHarnessStore(tmp_path / "current.db")
    item = intake(store)
    run = claim(store, item)
    value = proposal(run)
    memory = replace(mutation(), scope=Scope(A, item.session_id),
                     source_kind="output", source_id=value.candidate_id)
    receipt = store.commit(replace(value, memories=(MemoryEffect(memory, 2),)), now_us=101)
    with store.transaction() as db:
        db.execute("INSERT INTO relationship_effects VALUES(?,?,?,?,?,?)",
                   (A, receipt["delivery_id"], "retired", 2, "opaque old payload", "pending"))
        db.execute("INSERT INTO relationship_commands VALUES(?,?,?,?,?)",
                   (A, "old-applied", "opaque", "opaque", json.dumps({"original": True})))
    store.publish(A, receipt["delivery_id"])
    args = dict(candidate_id=value.candidate_id, revision=value.revision,
                sha256=receipt["sha256"], presentation_sha256=receipt["presentation_sha256"],
                sequence=1, displayed_end=len(value.text))
    store.acknowledge(A, receipt["delivery_id"], **args)
    reopened = SQLiteHarnessStore(store.path)
    assert reopened.history(A, memory.record_id)[0]["content"] == memory.content
    assert reopened.relationship_receipt(A, "old-applied") == {"original": True}
    with reopened.transaction(write=False) as db:
        assert tuple(db.execute("SELECT payload,status FROM relationship_effects").fetchone()) == (
            "opaque old payload", "pending",
        )
    assert reopened.inspect(A, item.input_id)["delivery"]["status"] == "completed"
    assert reopened.acknowledge(A, receipt["delivery_id"], **args)
