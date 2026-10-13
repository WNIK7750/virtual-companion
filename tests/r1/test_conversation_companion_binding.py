"""Synthetic roots only: default conversation ownership precedes recovery."""

import io
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from virtual_companion.text import __main__ as entry
from virtual_companion.text.application import TextSession
from virtual_companion.text.sqlite_store import SQLiteConversationStore

A = "00000000-0000-0000-0000-000000000001"
B = "00000000-0000-0000-0000-000000000002"


class RecordingModel:
    def __init__(self):
        self.requests = []

    def stream(self, messages, cancelled):
        self.requests.append(messages)
        yield "synthetic answer"

    def cancel(self):
        pass


def send(store, model, text, binding=None):
    session = TextSession(model, store, binding or {})
    session.send(text)
    session.worker.join(2)
    assert session.snapshot()["attempt"]["status"] == "completed"


def dump(root):
    with sqlite3.connect(root / "conversations.sqlite3") as connection:
        return list(connection.iterdump())


def test_same_root_different_companion_rejected_before_history_send(tmp_path):
    first = SQLiteConversationStore(tmp_path, companion_id=A)
    send(first, RecordingModel(), "COMPANION_A_SYNTHETIC_MARKER")
    first.begin("pending synthetic message", {})
    first.close()
    with sqlite3.connect(tmp_path / "conversations.sqlite3") as connection:
        connection.execute("PRAGMA journal_mode=DELETE")
    original_bytes = (tmp_path / "conversations.sqlite3").read_bytes()
    original = dump(tmp_path)
    model = RecordingModel()
    with pytest.raises(ValueError, match="companion|Companion"):
        second = SQLiteConversationStore(tmp_path, companion_id=B)
        send(second, model, "hello from B")
    assert model.requests == []
    assert (tmp_path / "conversations.sqlite3").read_bytes() == original_bytes
    assert dump(tmp_path) == original
    with sqlite3.connect(tmp_path / "conversations.sqlite3") as connection:
        assert connection.execute("SELECT status FROM attempts ORDER BY seq DESC").fetchone() == (
            "running",
        )


def test_distinct_roots_do_not_send_other_companion_history(tmp_path):
    first = SQLiteConversationStore(tmp_path / "a", companion_id=A)
    send(first, RecordingModel(), "COMPANION_A_SYNTHETIC_MARKER")
    first.close()
    second = SQLiteConversationStore(tmp_path / "b", companion_id=B)
    model = RecordingModel()
    try:
        send(second, model, "hello from B")
        assert model.requests == [[{"role": "user", "content": "hello from B"}]]
    finally:
        second.close()


def test_same_identity_keeps_history_across_provider_or_appearance_change(tmp_path):
    first = SQLiteConversationStore(tmp_path, companion_id=A)
    send(first, RecordingModel(), "same identity marker", {"provider": "local", "avatar": "one"})
    session_id = first.session_id
    first.close()
    second = SQLiteConversationStore(tmp_path, companion_id=A)
    model = RecordingModel()
    try:
        assert second.session_id == session_id
        send(second, model, "again", {"provider": "api", "avatar": "two"})
        assert model.requests[0][0]["content"] == "same identity marker"
    finally:
        second.close()


def test_unbound_nonempty_history_is_not_assigned_or_recovered(tmp_path):
    legacy = SQLiteConversationStore(tmp_path)
    legacy.begin("legacy synthetic history", {})
    legacy.close()
    original = dump(tmp_path)
    with pytest.raises(ValueError, match="unbound|Unbound|ownership"):
        SQLiteConversationStore(tmp_path, companion_id=A)
    assert dump(tmp_path) == original


def test_empty_legacy_store_can_bind_without_changing_session(tmp_path):
    legacy = SQLiteConversationStore(tmp_path)
    session_id = legacy.session_id
    legacy.close()
    bound = SQLiteConversationStore(tmp_path, companion_id=A)
    assert bound.session_id == session_id
    assert bound.connection.execute(
        "SELECT value FROM metadata WHERE key='companion_id'"
    ).fetchone()[0] == A
    bound.close()


def test_omitted_identity_cannot_reopen_bound_database(tmp_path):
    bound = SQLiteConversationStore(tmp_path, companion_id=A)
    bound.begin("running synthetic input", {})
    bound.close()
    original = dump(tmp_path)
    with pytest.raises(ValueError, match="companion|Companion"):
        SQLiteConversationStore(tmp_path)
    assert dump(tmp_path) == original


def test_same_companion_recovers_pending_attempt_and_backup_keeps_owner(tmp_path):
    first = SQLiteConversationStore(tmp_path, companion_id=A)
    attempt = first.begin("synthetic pending", {})
    first.close()
    reopened = SQLiteConversationStore(tmp_path, companion_id=A)
    try:
        assert reopened.latest_attempt()["attempt_id"] == attempt["attempt_id"]
        assert reopened.latest_attempt()["status"] == "interrupted"
        backup = reopened.backup()
        with sqlite3.connect(backup) as connection:
            assert connection.execute(
                "SELECT value FROM metadata WHERE key='companion_id'"
            ).fetchone() == (A,)
    finally:
        reopened.close()


def test_concurrent_first_binding_has_one_owner(tmp_path):
    barrier = threading.Barrier(2)

    def bind(companion):
        barrier.wait(timeout=2)
        try:
            store = SQLiteConversationStore(tmp_path, companion_id=companion)
        except (ValueError, sqlite3.OperationalError):
            return None
        store.close()
        return companion

    with ThreadPoolExecutor(max_workers=2) as pool:
        winners = list(pool.map(bind, [A, B]))
    winners = [winner for winner in winners if winner is not None]
    assert len(winners) == 1
    with sqlite3.connect(tmp_path / "conversations.sqlite3") as connection:
        assert connection.execute(
            "SELECT value FROM metadata WHERE key='companion_id'"
        ).fetchone() == (winners[0],)
        assert connection.execute("SELECT COUNT(*) FROM sessions").fetchone() == (1,)


@pytest.mark.parametrize("identity", ["", "invalid", A.replace("-", ""), 1])
def test_invalid_explicit_identity_does_not_create_root(tmp_path, identity):
    root = tmp_path / "not-created"
    with pytest.raises(ValueError):
        SQLiteConversationStore(root, companion_id=identity)
    assert not root.exists()


def prepare_main(monkeypatch, tmp_path):
    monkeypatch.setattr(entry.sys, "stdin", io.StringIO("a" * 32 + "\n"))
    monkeypatch.setattr(entry.sys, "argv", ["synthetic", "--fake"])
    monkeypatch.setattr(entry, "contain_current_process", lambda: None)
    monkeypatch.setattr(entry.threading, "Thread", lambda **_: SimpleNamespace(start=lambda: None))
    monkeypatch.setenv("VC_DATA_ROOT", str(tmp_path / "default"))
    monkeypatch.delenv("VC_MEMORY_MANAGEMENT_DB", raising=False)
    monkeypatch.delenv("VC_MEMORY_COMPANION_ID", raising=False)


@pytest.mark.parametrize("overrides", [
    {"VC_MEMORY_MANAGEMENT_DB": "synthetic.sqlite3"},
    {"VC_MEMORY_COMPANION_ID": A},
    {"VC_MEMORY_COMPANION_ID": ""},
    {"VC_MEMORY_MANAGEMENT_DB": "", "VC_MEMORY_COMPANION_ID": ""},
    {"VC_MEMORY_MANAGEMENT_DB": "synthetic.sqlite3", "VC_MEMORY_COMPANION_ID": "bad"},
    {"VC_MEMORY_MANAGEMENT_DB": " ", "VC_MEMORY_COMPANION_ID": A},
    {"VC_MEMORY_MANAGEMENT_DB": "synthetic.sqlite3", "VC_MEMORY_COMPANION_ID": " "},
])
def test_default_bootstrap_validates_configuration_before_opening_store(
    tmp_path, monkeypatch, overrides,
):
    prepare_main(monkeypatch, tmp_path)
    for key, value in overrides.items():
        monkeypatch.setenv(key, value)
    opened = []

    def capture(*args, **kwargs):
        opened.append((args, kwargs))
        raise RuntimeError("Store opened before configuration validation")

    monkeypatch.setattr(entry, "SQLiteConversationStore", capture)
    with pytest.raises(ValueError):
        entry.main()
    assert opened == []
    assert not (tmp_path / "default").exists()


@pytest.mark.parametrize("override", [False, True])
def test_default_bootstrap_passes_resolved_companion_to_store(tmp_path, monkeypatch, override):
    prepare_main(monkeypatch, tmp_path)
    if override:
        monkeypatch.setenv("VC_MEMORY_MANAGEMENT_DB", str(tmp_path / "synthetic.sqlite3"))
        monkeypatch.setenv("VC_MEMORY_COMPANION_ID", B)
    opened = []

    def capture(root, **kwargs):
        opened.append((root, kwargs))
        raise RuntimeError("captured store boundary")

    monkeypatch.setattr(entry, "SQLiteConversationStore", capture)
    with pytest.raises(RuntimeError, match="captured store boundary"):
        entry.main()
    assert opened == [(tmp_path / "default", {"companion_id": B if override else A})]


def test_default_bootstrap_mismatch_does_not_open_memory_or_recover_attempt(tmp_path, monkeypatch):
    root = tmp_path / "default"
    first = SQLiteConversationStore(root, companion_id=A)
    first.begin("synthetic pending", {})
    first.close()
    original = dump(root)
    prepare_main(monkeypatch, tmp_path)
    monkeypatch.setenv("VC_MEMORY_MANAGEMENT_DB", str(tmp_path / "memory.sqlite3"))
    monkeypatch.setenv("VC_MEMORY_COMPANION_ID", B)
    opened = []
    monkeypatch.setattr(entry, "TextMemoryRuntime", lambda *args, **kwargs: opened.append(args))
    with pytest.raises(ValueError, match="companion"):
        entry.main()
    assert opened == []
    assert dump(root) == original
