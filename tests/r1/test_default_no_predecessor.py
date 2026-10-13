"""The actual default adapter completes without predecessor cognitive behavior."""

from test_default_text_delivery import setup, wait_for
from virtual_companion.text.memory_runtime import DEFAULT_COMPANION_ID
from virtual_companion.text.sqlite_store import SQLiteConversationStore


def test_default_ack_and_reopen_complete_with_predecessor_behavior_poisoned(tmp_path, monkeypatch):
    import sys
    from importlib.abc import MetaPathFinder

    class DenyPredecessor(MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if any(part in fullname.split(".") for part in (
                "native", "native_port", "business_port", "sqlite_dynamics",
                "sqlite_relationship", "relationship", "dynamics",
            )):
                raise AssertionError("Retired behavior imported by current default: " + fullname)
            return None

    monkeypatch.setattr(sys, "meta_path", [DenyPredecessor(), *sys.meta_path])

    runtime, session, memory, mirror, model = setup(tmp_path)
    try:
        attempt = session.send("合成普通聊天，不查询旧记忆")
        event = wait_for(lambda: runtime.poll()["event"])
        assert session.snapshot()["partial"] == ""
        runtime.ack({"sequence": event["sequence"], "fingerprint": event["fingerprint"],
                     "host_receipt": "synthetic-current-default-visible"})
        wait_for(lambda: not session.snapshot()["busy"])
        assert session.snapshot()["attempt"]["status"] == "completed"
        assert memory.store.stream_state(
            DEFAULT_COMPANION_ID, event["basis"]["input_id"])["outcome"] == "completed"
        assert len(model.calls) == 2
        assert mirror.latest_attempt()["turn_id"] == attempt["turn_id"]
        assert mirror.history()["messages"][-1]["content"] == "这是合成答案。"
    finally:
        session.stop()
        wait_for(lambda: not session.snapshot()["busy"])
        mirror.close()
    reopened = SQLiteConversationStore(tmp_path / "chat", companion_id=DEFAULT_COMPANION_ID)
    try:
        assert reopened.latest_attempt()["status"] == "completed"
        assert [m["role"] for m in reopened.history()["messages"]] == ["user", "assistant"]
    finally:
        reopened.close()
