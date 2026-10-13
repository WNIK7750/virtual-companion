"""Current command outcomes are source-bound evidence, never forgotten content."""

import json
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256

import pytest
from test_memory_command_worker import command, seed
from test_rebuilt_turns import A, B, config
from test_segment_store import reviewed
from virtual_companion.boundary.contracts import Scope, VersionConflict, canonical
from virtual_companion.boundary.history_readset import (
    evidence_hash,
    history_readset,
    validate_readset,
)
from virtual_companion.boundary.memory_management import ExplicitMemoryCommand
from virtual_companion.boundary.sqlite_memory_management import SQLiteExplicitMemoryStore
from virtual_companion.cognition.background_worker import UserFactWorker

SOURCE = "请忘记我的音乐偏好这条记忆。"
FORGOTTEN_TEXT = "我喜欢安静的音乐"


def completed_command(tmp_path, *, action="forget", repeated=False):
    store = SQLiteExplicitMemoryStore(tmp_path / "operations.sqlite3")
    seed(store)
    source = SOURCE if action == "forget" else "把我的音乐偏好改成我喜欢摇滚音乐。"
    item = store.accept(config(), "command", source)

    def invoke(stage, messages, schema, cancelled):
        if stage == "memory_command":
            value = command(source)
            if action == "correct":
                replacement = "我喜欢摇滚音乐"
                start = source.index(replacement)
                value.update(action=action, replacementQuote=replacement,
                             replacementStart=start, replacementEnd=start + len(replacement))
            return value
        if stage == "memory_command_support":
            return {"meaning": "explicit_single_forget" if action == "forget"
                    else "explicit_single_correction", "verdict": "supported"}
        raise AssertionError(stage)

    receipt = UserFactWorker(store, invoke, clock=lambda: 10, memory_commands=True).run_one(
        A, "worker", source=item.input_id)
    assert receipt["status"] == "committed"
    if repeated:
        with store.transaction(write=False) as db:
            payload = json.loads(db.execute(
                "SELECT payload FROM memory_management_operations "
                "WHERE companion=? AND operation=?",
                (A, receipt["operation_id"])).fetchone()[0])
        previous = ExplicitMemoryCommand(**{**payload, "scope": Scope(**payload["scope"])})
        store.cancel(A, item.input_id)
        item = store.accept(config(), "repeated-command", source)
        child = store.claim_background(A, "worker", role="user", source=item.input_id,
                                       now_us=11, lease_us=100)
        operation = replace(previous, scope=Scope(A, item.session_id),
            operation_id="same-desired-state", source_id=item.input_id,
            source_sha256=sha256(source.encode()).hexdigest())
        receipt = store.execute_memory_command(operation, now_us=12, claim=child)
        assert receipt["status"] == "already_applied"
    claim = store.claim(A, item.input_id, "new-foreground-owner", now_us=20, lease_us=100)
    return store, claim, receipt


def context(operation=None):
    value = dict(history=[], memories=[], exchanges=[], memory_status="current")
    if operation is not None:
        value["current_operation"] = operation
    return canonical(value)


@pytest.mark.parametrize("action", ["forget", "correct"])
@pytest.mark.parametrize("repeated", [False, True])
def test_successful_current_operation_is_content_free_and_same_source(tmp_path, action, repeated):
    store, claim, receipt = completed_command(tmp_path, action=action, repeated=repeated)
    result = store.current_memory_operation(claim)
    assert result == dict(protocol="memory-command-outcome.v1", source_id=claim.input_id,
        source_sha256=sha256(store.inspect(A, claim.input_id)["text"].encode()).hexdigest(),
        operation_id=receipt["operation_id"], action=action, status=receipt["status"],
        receipt_sha256=evidence_hash(receipt))
    assert FORGOTTEN_TEXT not in canonical(result)
    assert "我喜欢摇滚音乐" not in canonical(result)
    if action == "forget":
        assert store.recall(A, now_us=21) == []
    # The current claim is a new owner after the command finished: old worker
    # identity must not be required to read the persistent outcome.
    assert SQLiteExplicitMemoryStore(store.path).current_memory_operation(claim) == result
    later = store.accept(config(), "later", "聊点别的")
    assert store.current_memory_operation(replace(claim, input_id=later.input_id,
                                                  session_id=later.session_id)) is None
    other = store.accept(config(B), "other", SOURCE)
    other_claim = store.claim(B, other.input_id, "other-owner", now_us=20, lease_us=100)
    assert store.current_memory_operation(other_claim) is None
    with pytest.raises((ValueError, VersionConflict)):
        store.current_memory_operation(replace(claim, session_id=other.session_id))


@pytest.mark.parametrize("mutation", ["fingerprint", "source_hash", "source_identity",
    "receipt_status", "receipt_version", "source_receipt", "entry", "duplicate_source"])
def test_corrupt_or_foreign_operation_cannot_be_projected(tmp_path, mutation):
    store, claim, receipt = completed_command(tmp_path)
    with store.transaction() as db:
        row = db.execute("SELECT * FROM memory_management_operations "
                         "WHERE companion=? AND operation=?",
                         (A, receipt["operation_id"])).fetchone()
        payload, persisted = json.loads(row["payload"]), json.loads(row["receipt"])
        if mutation == "fingerprint":
            db.execute("UPDATE memory_management_operations SET fingerprint=? WHERE companion=?",
                       ("f" * 64, A))
        elif mutation == "source_hash":
            db.execute("UPDATE background_sources SET source_hash=? WHERE companion=? AND source=?",
                       ("f" * 64, A, claim.input_id))
        elif mutation == "source_receipt":
            work = db.execute("SELECT job,receipt FROM background_work WHERE companion=?",
                              (A,)).fetchone()
            evidence = json.loads(work["receipt"])
            evidence["memory_receipt"]["record_version"] += 1
            db.execute("UPDATE background_work SET receipt=? WHERE companion=? AND job=?",
                       (canonical(evidence), A, work["job"]))
        elif mutation == "duplicate_source":
            row = db.execute("SELECT * FROM background_sources WHERE companion=? AND source=?",
                             (A, claim.input_id)).fetchone()
            db.execute("INSERT INTO background_sources VALUES(?,?,?,?,?,?,?,?,?)",
                       (A, row["session"], "duplicate-job", row["role"], row["source"],
                        "f" * 64, row["confirmed_end"], row["status"], row["generation"]))
        else:
            if mutation == "source_identity":
                payload["source_id"] = "other-input"
            elif mutation == "entry":
                payload.update(entry="list", source_coverage="explicit_selection",
                               evidence_quote="", quote_start=None, quote_end=None)
            elif mutation == "receipt_status":
                persisted["status"] = "version_changed"
            else:
                persisted["record_version"] = True
            db.execute("UPDATE memory_management_operations SET payload=?,fingerprint=?,receipt=? "
                       "WHERE companion=? AND operation=?",
                       (canonical(payload), evidence_hash(payload), canonical(persisted),
                        A, receipt["operation_id"]))
    with pytest.raises((ValueError, VersionConflict)):
        store.current_memory_operation(claim)


def test_only_persisted_success_is_an_outcome(tmp_path):
    store, claim, _ = completed_command(tmp_path)
    with store.transaction() as db:
        db.execute("UPDATE background_sources SET status='failed' WHERE companion=? AND source=?",
                   (A, claim.input_id))
        db.execute("UPDATE background_work SET receipt=? WHERE companion=?",
                   (canonical({"status": "needs_clarification", "unresolved": ["target"]}), A))
    assert store.current_memory_operation(claim) is None


def test_readset_v1_is_unchanged_and_v2_binds_current_operation(tmp_path):
    store, claim, _ = completed_command(tmp_path)
    old = history_readset(A, claim.input_id, claim.state_version, context())
    assert old == dict(protocol="history-readset.v1", companion=A, input=claim.input_id,
        state_version=claim.state_version, context_sha256=evidence_hash(json.loads(context())),
        facts=[], history=[], exchanges=[])
    operation = store.current_memory_operation(claim)
    new = history_readset(A, claim.input_id, claim.state_version, context(operation))
    assert new["protocol"] == "history-readset.v2"
    assert new["current_operation"] == operation
    assert new["context_sha256"] != old["context_sha256"]
    candidate = reviewed(claim, ("本次遗忘已经提交。",))
    event = store.approve_segment(claim, candidate, now_us=21, readset=new)
    assert event["readset"] == new
    assert store.dispatch_segment(claim, 1, now_us=22)["readset"] == new


@pytest.mark.parametrize("mutation", ["wrong_input", "extra_text", "wrong_status", "bad_digest",
                                    "v1_with_operation", "v2_without_operation"])
def test_readset_rejects_unknown_or_unbound_operation_shape(tmp_path, mutation):
    store, claim, _ = completed_command(tmp_path)
    operation = store.current_memory_operation(claim)
    value = history_readset(A, claim.input_id, claim.state_version, context(operation))
    if mutation == "wrong_input":
        value["current_operation"]["source_id"] = "not-current"
    elif mutation == "extra_text":
        value["current_operation"]["forgotten_content"] = FORGOTTEN_TEXT
    elif mutation == "wrong_status":
        value["current_operation"]["status"] = "needs_clarification"
    elif mutation == "bad_digest":
        value["current_operation"]["receipt_sha256"] = "invalid"
    elif mutation == "v1_with_operation":
        value["protocol"] = "history-readset.v1"
    else:
        del value["current_operation"]
    with pytest.raises(ValueError):
        validate_readset(value)


@pytest.mark.parametrize("phase", ["approval", "dispatch"])
def test_readset_rechecks_persistent_outcome_before_publication(tmp_path, phase):
    store, claim, _ = completed_command(tmp_path)
    operation = store.current_memory_operation(claim)
    value = history_readset(A, claim.input_id, claim.state_version, context(operation))
    candidate = reviewed(claim, ("本次遗忘已经提交。",))
    if phase == "dispatch":
        store.approve_segment(claim, candidate, now_us=21, readset=value)
    with store.transaction() as db:
        db.execute("UPDATE background_sources SET status='failed' WHERE companion=? AND source=?",
                   (A, claim.input_id))
    with pytest.raises(VersionConflict, match="readset"):
        if phase == "approval":
            store.approve_segment(claim, candidate, now_us=22, readset=value)
        else:
            store.dispatch_segment(claim, 1, now_us=22)


def test_readset_wrong_receipt_digest_is_not_valid_provenance(tmp_path):
    store, claim, _ = completed_command(tmp_path)
    operation = deepcopy(store.current_memory_operation(claim))
    operation["receipt_sha256"] = "f" * 64
    value = history_readset(A, claim.input_id, claim.state_version, context(operation))
    with pytest.raises(VersionConflict, match="readset"):
        store.approve_segment(claim, reviewed(claim), now_us=21, readset=value)
