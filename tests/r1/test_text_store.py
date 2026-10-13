from __future__ import annotations

import sqlite3

import pytest
from virtual_companion.text.sqlite_store import SQLiteConversationStore


def test_restart_marks_pending_attempt_interrupted_and_retry_does_not_double_write(tmp_path):
    store = SQLiteConversationStore(tmp_path)
    first = store.begin("hello", {"version": "one"})
    session_id = store.history()["session_id"]
    store.close()
    recovered = SQLiteConversationStore(tmp_path)
    assert recovered.latest_attempt()["status"] == "interrupted"
    assert recovered.history()["session_id"] == session_id
    retry = recovered.begin(None, {"version": "one"}, retry=True)
    assert retry["attempt_id"] != first["attempt_id"]
    recovered.finish(retry["attempt_id"], "completed", "answer")
    assert [m["content"] for m in recovered.history()["messages"]] == ["hello", "answer"]
    with pytest.raises(ValueError):
        recovered.finish(retry["attempt_id"], "completed", "duplicate")
    recovered.close()


def test_failed_assistant_commit_rolls_back_attempt_status_and_message(tmp_path):
    store = SQLiteConversationStore(tmp_path)
    attempt = store.begin("hello", {})
    store.connection.execute("""CREATE TRIGGER fail_assistant BEFORE INSERT ON messages
        WHEN NEW.role='assistant' BEGIN SELECT RAISE(ABORT, 'synthetic failure'); END""")
    with pytest.raises(sqlite3.DatabaseError):
        store.finish(attempt["attempt_id"], "completed", "answer")
    assert store.latest_attempt()["status"] == "running"
    assert len(store.history()["messages"]) == 1
    store.close()


def test_backup_is_consistent_readonly_and_new_session_preserves_old_session(tmp_path):
    store = SQLiteConversationStore(tmp_path)
    attempt = store.begin("hello", {})
    store.finish(attempt["attempt_id"], "completed", "answer")
    original = store.history()["session_id"]
    backup = store.backup()
    with sqlite3.connect(backup.as_uri() + "?mode=ro", uri=True) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("DELETE FROM messages")
    store.new_session()
    assert store.history()["session_id"] != original
    assert store.connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2
    assert original in [session["id"] for session in store.sessions()]
    store.select_session(original)
    assert len(store.history()["messages"]) == 2
    with pytest.raises(ValueError):
        store.select_session("unknown")
    store.close()


def test_future_schema_is_not_silently_migrated(tmp_path):
    with sqlite3.connect(tmp_path / "conversations.sqlite3") as connection:
        connection.execute("PRAGMA user_version=99")
    with pytest.raises(ValueError, match="schema"):
        SQLiteConversationStore(tmp_path)


def test_session_pages_reach_oldest_without_duplicates(tmp_path):
    store = SQLiteConversationStore(tmp_path)
    first = store.session_id
    for _ in range(100):
        store.new_session()
    page = store.sessions()
    older = store.sessions(offset=100)
    assert len(page) == 100
    assert [item["id"] for item in older] == [first]
    assert not set(item["id"] for item in page) & {first}
    store.close()
