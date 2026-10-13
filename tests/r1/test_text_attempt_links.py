import json
from contextlib import contextmanager
from dataclasses import replace
from threading import Event

import pytest
from test_rebuilt_turns import A, B, claim, config
from test_review_carrier_selection import factory as factory
from virtual_companion.boundary.contracts import VersionConflict, canonical
from virtual_companion.boundary.sqlite_received import SQLiteReceivedStore
from virtual_companion.feedback.errors import MessageOwnershipError
from virtual_companion.text.attempt_links import (
    _boundary_key,
    load_attempt_link,
    persist_attempt_link,
)
from virtual_companion.text.feedback_receipts import FeedbackReceiptCatalog
from virtual_companion.text.sqlite_store import SQLiteConversationStore


def prepared(factory):
    _, _, runtime, memory, mirror, _ = factory
    attempt = mirror.begin("今天先聊聊。", runtime.config.attempt_binding({"test": "synthetic"}))
    runtime.prepare_turn(attempt, mirror.session_id, mirror.history()["messages"], Event())
    return runtime, memory, mirror, attempt


def clear_attempt_links(mirror):
    with mirror.transaction():
        mirror.connection.execute(
            "DELETE FROM metadata WHERE key LIKE 'text-attempt-link:v1:%'"
        )


def attempt_link_count(mirror):
    with mirror.lock:
        return mirror.connection.execute(
            "SELECT COUNT(*) FROM metadata WHERE key LIKE 'text-attempt-link:v1:%'"
        ).fetchone()[0]


def test_actual_initial_boundary_attempt_is_linked_before_any_generation(factory):
    runtime, memory, mirror, attempt = prepared(factory)
    assert runtime.claim.attempt_id != attempt["attempt_id"]
    with memory.store.transaction(write=False) as db:
        tables = {row[0] for row in db.execute(
            "SELECT name FROM sqlite_schema WHERE type='table'")}
        source = db.execute(
            "SELECT session,client_key FROM turn_inputs WHERE companion=? AND input=?",
            (runtime.companion, runtime.claim.input_id),
        ).fetchone()
    assert "text_attempt_links" not in tables
    assert source["client_key"] == attempt["turn_id"]
    row = load_attempt_link(mirror, attempt["attempt_id"])
    assert row["attempt"] == runtime.claim.attempt_id
    assert row["text_attempt"] == attempt["attempt_id"]
    assert row["input"] == runtime.claim.input_id
    assert row["session"] == source["session"]
    assert row["turn_id"] == attempt["turn_id"]
    assert json.loads(row["binding"]) == attempt["binding"]
    assert memory.store.segments(runtime.companion, runtime.claim.input_id) == []


def test_boundary_write_lock_is_held_through_mirror_commit(factory, monkeypatch):
    runtime, memory, mirror, attempt = prepared(factory)
    clear_attempt_links(mirror)
    boundary_transaction_open = False
    original_boundary_transaction = memory.store.transaction
    original_mirror_transaction = mirror.transaction

    @contextmanager
    def tracked_boundary_transaction(*args, **kwargs):
        nonlocal boundary_transaction_open
        with original_boundary_transaction(*args, **kwargs) as db:
            boundary_transaction_open = True
            try:
                yield db
            finally:
                boundary_transaction_open = False

    @contextmanager
    def require_boundary_lock_during_mirror_commit():
        assert boundary_transaction_open
        with original_mirror_transaction():
            yield
        assert boundary_transaction_open

    monkeypatch.setattr(memory.store, "transaction", tracked_boundary_transaction)
    monkeypatch.setattr(mirror, "transaction", require_boundary_lock_during_mirror_commit)
    persist_attempt_link(
        mirror, memory.store, runtime.claim,
        boundary_session_id=runtime.claim.session_id, turn_id=attempt["turn_id"],
        text_attempt_id=attempt["attempt_id"], binding=attempt["binding"],
        now_us=runtime.clock(), clock=runtime.clock,
    )

    assert not boundary_transaction_open
    assert load_attempt_link(mirror, attempt["attempt_id"]) is not None


def test_link_is_idempotent_and_rejects_unbound_reassignment(factory):
    runtime, memory, mirror, attempt = prepared(factory)
    with memory.store.transaction(write=False) as db:
        source = db.execute(
            "SELECT session FROM turn_inputs WHERE companion=? AND input=?",
            (runtime.companion, runtime.claim.input_id),
        ).fetchone()
    values = dict(boundary_session_id=source["session"], turn_id=attempt["turn_id"],
                  text_attempt_id=attempt["attempt_id"], binding=attempt["binding"],
                  now_us=runtime.clock(), clock=runtime.clock)
    persist_attempt_link(mirror, memory.store, runtime.claim, **values)
    with pytest.raises(ValueError, match="Live text mirror attempt"):
        persist_attempt_link(mirror, memory.store, runtime.claim,
                             **{**values, "text_attempt_id": "wrong"})


def test_link_rejects_a_conflicting_reverse_key_without_overwriting(factory):
    runtime, memory, mirror, attempt = prepared(factory)
    with memory.store.transaction(write=False) as db:
        source = db.execute(
            "SELECT session FROM turn_inputs WHERE companion=? AND input=?",
            (runtime.companion, runtime.claim.input_id),
        ).fetchone()
    reverse_key = _boundary_key(runtime.companion, runtime.claim.input_id,
                                runtime.claim.attempt_id)
    conflicting = canonical({
        "protocol": "text-attempt-link.v1",
        "companion": runtime.companion,
        "input": runtime.claim.input_id,
        "attempt": runtime.claim.attempt_id,
        "text_attempt": "another-text-attempt",
    })
    with mirror.transaction():
        mirror.connection.execute("UPDATE metadata SET value=? WHERE key=?",
                                  (conflicting, reverse_key))
    with pytest.raises(ValueError, match="Immutable attempt link conflict"):
        persist_attempt_link(
            mirror, memory.store, runtime.claim,
            boundary_session_id=source["session"], turn_id=attempt["turn_id"],
            text_attempt_id=attempt["attempt_id"], binding=attempt["binding"],
            now_us=runtime.clock(), clock=runtime.clock,
        )
    with mirror.lock:
        assert mirror.connection.execute(
            "SELECT value FROM metadata WHERE key=?", (reverse_key,)
        ).fetchone()[0] == conflicting
        assert mirror.connection.execute(
            "SELECT 1 FROM metadata WHERE key LIKE 'text-attempt-link:v1:text:%'"
        ).fetchone() is not None


def test_link_rejects_a_mirror_owned_by_another_companion(factory, tmp_path):
    runtime, memory, _, attempt = prepared(factory)
    wrong_owner = SQLiteConversationStore(tmp_path / "wrong-owner", companion_id=B)
    try:
        with memory.store.transaction(write=False) as db:
            source = db.execute(
                "SELECT session FROM turn_inputs WHERE companion=? AND input=?",
                (runtime.companion, runtime.claim.input_id),
            ).fetchone()
        with pytest.raises(ValueError, match="mirror owner differs"):
            persist_attempt_link(
                wrong_owner, memory.store, runtime.claim,
                boundary_session_id=source["session"], turn_id=attempt["turn_id"],
                text_attempt_id=attempt["attempt_id"], binding=attempt["binding"],
                now_us=runtime.clock(), clock=runtime.clock,
            )
    finally:
        wrong_owner.connection.close()


def test_metadata_crosswalk_survives_reopen_and_is_resolved_from_mirror(tmp_path):
    mirror_path = tmp_path / "mirror"
    boundary_path = tmp_path / "memory.sqlite3"
    mirror = SQLiteConversationStore(mirror_path, companion_id=A)
    version_binding = {"version": "synthetic"}
    attempt = mirror.begin("输入", {"version_binding": version_binding})
    receipts = SQLiteReceivedStore(boundary_path)
    receipts.register(A)
    boundary_session = "boundary:" + mirror.session_id
    receipts.ensure_session(A, boundary_session)
    item = receipts.accept(config(), attempt["turn_id"], "输入",
                           session_id=boundary_session)
    run = claim(receipts, item)
    persist_attempt_link(mirror, receipts, run, boundary_session_id=boundary_session,
                         turn_id=attempt["turn_id"], text_attempt_id=attempt["attempt_id"],
                         binding=attempt["binding"], now_us=101)
    mirror.close()

    reopened_mirror = SQLiteConversationStore(mirror_path, companion_id=A)
    reopened_receipts = SQLiteReceivedStore(boundary_path)
    catalog = FeedbackReceiptCatalog(reopened_mirror, lambda: reopened_receipts)
    target = catalog.resolve(
        session_id=reopened_mirror.session_id, turn_id=attempt["turn_id"],
        assistant_message_id=None,
        target={"kind": "attempt", "attempt_id": attempt["attempt_id"]},
    )
    assert target["target_binding"]["status"] == "attempt-only"
    assert target["target_binding"]["attempt"]["input_id"] == item.input_id
    assert target["target_binding"]["attempt"]["boundary_attempt_id"] == run.attempt_id
    assert target["target_binding"]["attempt"]["text_attempt_id"] == attempt["attempt_id"]
    reopened_mirror.close()


def test_incomplete_metadata_crosswalk_is_rejected_not_repaired(factory):
    runtime, memory, mirror, attempt = prepared(factory)
    with mirror.transaction():
        row = mirror.connection.execute(
            "SELECT key FROM metadata WHERE key LIKE 'text-attempt-link:v1:boundary:%'"
        ).fetchone()
        assert row is not None
        mirror.connection.execute("DELETE FROM metadata WHERE key=?", (row["key"],))
    with pytest.raises((MessageOwnershipError, ValueError), match="(?i)link|crosswalk"):
        load_attempt_link(mirror, attempt["attempt_id"])


def test_boundary_reopen_still_rejects_unknown_tables(tmp_path):
    path = tmp_path / "memory.sqlite3"
    SQLiteReceivedStore(path)
    with SQLiteReceivedStore(path).transaction() as db:
        db.execute("CREATE TABLE text_attempt_links (attempt TEXT)")
    with pytest.raises(ValueError, match="Reliable reception accepts only the exact v19 schema"):
        SQLiteReceivedStore(path)


@pytest.mark.parametrize("mutation", ["config", "state_version"])
def test_stale_claim_config_or_basis_is_rejected_without_metadata(factory, mutation):
    runtime, memory, mirror, attempt = prepared(factory)
    live_claim = runtime.claim
    clear_attempt_links(mirror)
    if mutation == "config":
        stale_claim = replace(live_claim, config_json=canonical({"changed": True}))
    else:
        stale_claim = replace(live_claim, state_version=live_claim.state_version + 1)
    with pytest.raises(VersionConflict):
        persist_attempt_link(
            mirror, memory.store, stale_claim,
            boundary_session_id=live_claim.session_id, turn_id=attempt["turn_id"],
            text_attempt_id=attempt["attempt_id"], binding=attempt["binding"],
            now_us=runtime.clock(), clock=runtime.clock,
        )
    assert attempt_link_count(mirror) == 0


def test_expired_claim_reowned_with_same_attempt_cannot_write_link(factory):
    runtime, memory, mirror, attempt = prepared(factory)
    previous_claim = runtime.claim
    clear_attempt_links(mirror)
    current_claim = memory.store.claim(
        runtime.companion, previous_claim.input_id, "replacement-worker",
        now_us=previous_claim.lease_until_us, lease_us=120_000_000,
    )
    assert current_claim.attempt_id == previous_claim.attempt_id
    assert current_claim.owner != previous_claim.owner
    with pytest.raises(VersionConflict):
        persist_attempt_link(
            mirror, memory.store, previous_claim,
            boundary_session_id=previous_claim.session_id, turn_id=attempt["turn_id"],
            text_attempt_id=attempt["attempt_id"], binding=attempt["binding"],
            now_us=previous_claim.lease_until_us + 1,
        )
    assert attempt_link_count(mirror) == 0


def test_cancelled_claim_cannot_write_link(factory):
    runtime, memory, mirror, attempt = prepared(factory)
    live_claim = runtime.claim
    clear_attempt_links(mirror)
    memory.store.cancel(runtime.companion, live_claim.input_id)
    with pytest.raises(VersionConflict):
        persist_attempt_link(
            mirror, memory.store, live_claim,
            boundary_session_id=live_claim.session_id, turn_id=attempt["turn_id"],
            text_attempt_id=attempt["attempt_id"], binding=attempt["binding"],
            now_us=runtime.clock(), clock=runtime.clock,
        )
    assert attempt_link_count(mirror) == 0
