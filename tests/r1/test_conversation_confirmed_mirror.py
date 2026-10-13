"""The conversation database is a repairable mirror of canonical text ACKs."""

import pytest
from test_rebuilt_turns import A, claim, config
from test_segment_store import reviewed
from virtual_companion.boundary.sqlite_memory_management import SQLiteExplicitMemoryStore
from virtual_companion.text.sqlite_store import SQLiteConversationStore


def opened(tmp_path):
    return SQLiteConversationStore(tmp_path, companion_id=A)


def test_reopen_repairs_missing_mirror_without_inventing_another_user(tmp_path):
    store = opened(tmp_path)
    attempt = store.begin("你好", {"synthetic": True})
    canonical = SQLiteExplicitMemoryStore(tmp_path / "canonical.sqlite3")
    canonical.register(A)
    canonical.ensure_session(A, store.session_id)
    item = canonical.accept(config(), attempt["turn_id"], "你好", session_id=store.session_id)
    run = claim(canonical, item)
    canonical.approve_segment(run, reviewed(run, ("我在。",)), now_us=101)
    event = canonical.dispatch_segment(run, 1, now_us=102)
    canonical.confirm_segment(A, item.input_id, 1, fingerprint=event["fingerprint"],
                              host_receipt="device-confirmed", channel="text_visible")
    canonical.finish_stream(run, outcome="completed", now_us=103)
    store.close()  # v19 ACK committed, process died before its display mirror.
    store = opened(tmp_path)
    assert store.latest_attempt()["status"] == "interrupted"
    canonical = SQLiteExplicitMemoryStore(canonical.path)
    state = canonical.stream_state(A, item.input_id)
    receipt = store.reconcile_confirmed(attempt["turn_id"], state["confirmed_text"],
                                        completed=state["outcome"] == "completed")
    assert receipt["status"] == "completed"
    assert receipt == store.reconcile_confirmed(attempt["turn_id"], "我在。", completed=True)
    rows = store.history()["messages"]
    assert [(row["role"], row["content"]) for row in rows] == [
        ("user", "你好"), ("assistant", "我在。"),
    ]
    assert rows[-1]["id"] == receipt["message_id"]
    store.close()


@pytest.mark.parametrize("ending", ["failed", "cancelled", "interrupted"])
def test_late_ack_mirrors_actual_text_without_promoting_noncompleted_terminal(tmp_path, ending):
    store = opened(tmp_path)
    attempt = store.begin("你好", {})
    if ending == "interrupted":
        store.close()
        store = opened(tmp_path)
    else:
        store.finish(attempt["attempt_id"], ending, "unconfirmed")
    receipt = store.reconcile_confirmed(attempt["turn_id"], "实际呈现。", completed=False)
    assert receipt["status"] == ending
    assert store.latest_attempt()["status"] == ending
    assert store.history()["messages"][-1]["content"] == "实际呈现。"
    store.close()


def test_running_mirror_leaves_finish_owner_and_deduplicates_same_completion(tmp_path):
    store = opened(tmp_path)
    attempt = store.begin("你好", {})
    receipt = store.reconcile_confirmed(attempt["turn_id"], "我在。", completed=True)
    assert receipt["status"] == store.latest_attempt()["status"] == "running"
    store.finish(attempt["attempt_id"], "completed", "我在。")
    assert store.latest_attempt()["status"] == "completed"
    assert len(store.history()["messages"]) == 2
    assert store.history()["messages"][-1]["id"] == receipt["message_id"]
    store.close()


@pytest.mark.parametrize("replacement", ["我", "不同文本"])
def test_mirror_never_shortens_or_replaces_confirmed_text(tmp_path, replacement):
    store = opened(tmp_path)
    attempt = store.begin("你好", {})
    store.reconcile_confirmed(attempt["turn_id"], "我在。", completed=False)
    with pytest.raises(ValueError, match="conflict|Conflicting"):
        store.reconcile_confirmed(attempt["turn_id"], replacement, completed=True)
    with pytest.raises(ValueError, match="conflict|Conflicting"):
        store.finish(attempt["attempt_id"], "completed", replacement)
    assert store.latest_attempt()["status"] == "running"
    assert store.history()["messages"][-1]["content"] == "我在。"
    store.close()


def test_mirror_extends_only_the_same_confirmed_prefix_and_keeps_identity(tmp_path):
    store = opened(tmp_path)
    attempt = store.begin("你好", {})
    first = store.reconcile_confirmed(attempt["turn_id"], "我在。", completed=False)
    extended = store.reconcile_confirmed(
        attempt["turn_id"], "我在。今天想接着说。", completed=False)

    assert extended["message_id"] == first["message_id"]
    assert extended["status"] == "running"
    assert store.history()["messages"][-1]["content"] == "我在。今天想接着说。"
    with pytest.raises(ValueError, match="conflict|Conflicting"):
        store.finish(attempt["attempt_id"], "completed", "我在。今天想接着说。还有别的。")
    store.finish(attempt["attempt_id"], "completed", "我在。今天想接着说。")
    assert store.latest_attempt()["status"] == "completed"
    assert store.history()["messages"][-1]["id"] == first["message_id"]
    store.close()


def test_unknown_turn_and_invalid_confirmation_do_not_create_history(tmp_path):
    store = opened(tmp_path)
    with pytest.raises(ValueError):
        store.reconcile_confirmed("missing", "不应出现", completed=True)
    attempt = store.begin("你好", {})
    for text, completed in (("", True), ("text", 1), ("x" * 64001, False)):
        with pytest.raises(ValueError):
            store.reconcile_confirmed(attempt["turn_id"], text, completed=completed)
    assert len(store.history()["messages"]) == 1
    store.close()


def test_records_and_repair_bind_original_turn_not_active_latest(tmp_path):
    store = opened(tmp_path)
    first_session = store.session_id
    first = store.begin("第一轮", {"turn": 1})
    store.finish(first["attempt_id"], "failed", "")
    second = store.begin(None, {"turn": 2}, retry=True)
    store.finish(second["attempt_id"], "failed", "")
    second_session = store.new_session()["session_id"]
    latest = store.begin("另一会话", {"turn": 3})
    records = store.turn_records()
    assert records == [
        {"turn_id": first["turn_id"], "session_id": first_session,
         "latest_attempt": {**second, "status": "failed"}},
        {"turn_id": latest["turn_id"], "session_id": second_session,
         "latest_attempt": latest},
    ]
    receipt = store.reconcile_confirmed(first["turn_id"], "第一轮已显示", completed=False)
    assert receipt["attempt_id"] == second["attempt_id"]
    assert store.latest_attempt() == latest
    assert len(store.history()["messages"]) == 1
    store.select_session(first_session)
    assert store.history()["messages"][-1]["content"] == "第一轮已显示"
    store.close()
