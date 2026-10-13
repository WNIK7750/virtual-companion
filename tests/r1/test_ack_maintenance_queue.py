import pytest
from test_pending_exchanges import question
from test_rebuilt_turns import A, config
from virtual_companion.boundary.sqlite_maintenance import SQLiteMaintenanceStore


def acknowledge(store, event, sequence, end, stopped=False):
    return store.acknowledge(
        A,
        event["delivery_id"],
        candidate_id=event["candidate_id"],
        revision=event["revision"],
        sha256=event["sha256"],
        sequence=sequence,
        displayed_end=end,
        stopped=stopped,
        presentation_sha256=event["presentation_sha256"],
    )


def test_terminal_ack_enqueues_one_durable_internal_job(tmp_path):
    store = SQLiteMaintenanceStore(tmp_path / "auto.sqlite3", automatic_maintenance=True)
    command = question(store, displayed=7)
    queued = store.queued_maintenance(A)
    assert len(queued) == 1
    row = store.inspect(A, queued[0])
    assert row["trigger"]["source_id"] == command.evidence.source_id
    assert row["text"] == ""
    assert SQLiteMaintenanceStore(store.path).queued_maintenance(A) == queued


def test_unshown_output_never_enqueues(tmp_path):
    store = SQLiteMaintenanceStore(tmp_path / "unshown.sqlite3", automatic_maintenance=True)
    question(store)
    assert store.queued_maintenance(A) == []


def test_only_terminal_ack_enqueues_and_duplicate_ack_is_idempotent(tmp_path):
    store = SQLiteMaintenanceStore(tmp_path / "terminal.sqlite3", automatic_maintenance=True)
    command = question(store)
    event = store.publication(A, command.evidence.source_id)
    acknowledge(store, event, 1, 7)
    assert store.queued_maintenance(A) == []
    receipt = acknowledge(store, event, 2, len(event["text"]))
    assert acknowledge(store, event, 2, len(event["text"])) == receipt
    queued = store.queued_maintenance(A)
    assert len(queued) == 1
    assert store.inspect(A, queued[0])["trigger"]["confirmed_end"] == len(event["text"])


def test_intervening_user_does_not_lose_late_actual_display_ack(tmp_path):
    store = SQLiteMaintenanceStore(tmp_path / "intervening.sqlite3", automatic_maintenance=True)
    command = question(store)
    event = store.publication(A, command.evidence.source_id)
    store.accept(config(), "intervening", "想聊旅行", session_id=command.scope.session_id)
    store.cancel(A, event["input_id"])
    receipt = acknowledge(store, event, 1, 7, stopped=True)
    assert receipt["displayed_end"] == 7
    assert store.queued_maintenance(A) == []
    assert (
        store.exchange_sources(
            command.scope, references=(("assistant", command.evidence.source_id),)
        )[0].text
        == command.content
    )


def test_old_source_after_new_user_requires_context_reconciliation(tmp_path):
    store = SQLiteMaintenanceStore(tmp_path / "stale.sqlite3")
    command = question(store, displayed=7)
    store.accept(config(), "answer", "想聊旅行", session_id=command.scope.session_id)
    with pytest.raises(ValueError, match="subsequent user"):
        store.accept_maintenance(config(), "stale", command.scope, command.evidence)
    assert store.queued_maintenance(A) == []


def test_auto_queue_failure_rolls_back_ack_for_safe_reconciliation(tmp_path, monkeypatch):
    store = SQLiteMaintenanceStore(tmp_path / "rollback.sqlite3", automatic_maintenance=True)
    original = store._accepted

    def fail(db, receipt, trigger):
        original(db, receipt, trigger)
        if trigger is not None:
            raise OSError("synthetic enqueue failure")

    monkeypatch.setattr(store, "_accepted", fail)
    with pytest.raises(OSError):
        question(store, displayed=7)
    with store.transaction(write=False) as db:
        assert db.execute("SELECT count(*) FROM turn_acks").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM turn_triggers").fetchone()[0] == 0
        assert db.execute("SELECT displayed_end FROM turn_deliveries").fetchone()[0] == 0
