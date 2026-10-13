import json

import pytest
from virtual_companion.text.application import TextSession
from virtual_companion.text.memory import MemoryConversationStore
from virtual_companion.text.sqlite_store import SQLiteConversationStore
from virtual_companion.text.transport import validate


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    value = (
        MemoryConversationStore()
        if request.param == "memory"
        else SQLiteConversationStore(tmp_path)
    )
    yield value
    if request.param == "sqlite":
        value.close()


def populate(store, count, content="synthetic"):
    for index in range(count):
        attempt = store.begin(f"{index}: {content}", {})
        store.finish(attempt["attempt_id"], "completed", content)


def test_history_pages_reach_every_message_without_duplicates(store):
    populate(store, 31)
    full = store.history()["messages"]
    page = store.history_page()
    assert len(page["messages"]) == 20
    found = page["messages"]
    while page["next_before"]:
        page = store.history_page(before=page["next_before"], session_id=store.session_id)
        found = page["messages"] + found
    assert found == full
    assert len({m["id"] for m in found}) == len(full)


def test_new_messages_dont_shift_older_cursor_and_incremental_reads_are_bounded(store):
    populate(store, 15)
    page = store.history_page()
    original = store.history()["messages"]
    last = original[-1]["id"]
    populate(store, 14)
    older = store.history_page(before=page["next_before"])
    assert older["messages"] == original[:10]
    newer = store.history_page(after=last)
    assert len(newer["messages"]) == 20
    rest = store.history_page(after=newer["next_after"])
    assert newer["messages"] + rest["messages"] == store.history()["messages"][30:]
    assert rest["next_after"] is None


def test_escaped_large_messages_have_a_bounded_wire_page(store):
    populate(store, 25, "\x00" * 64000)
    page = store.history_page()
    assert len(json.dumps(page, ensure_ascii=False).encode()) < 1_001_000
    assert page["next_before"]
    assert len(store.history()["messages"]) == 50


def test_cursor_is_bound_to_active_session(store):
    populate(store, 1)
    old = store.history()["messages"][-1]["id"]
    old_session = store.session_id
    store.new_session()
    with pytest.raises(ValueError):
        store.history_page(before=old)
    with pytest.raises(ValueError):
        store.history_page(session_id=old_session)
    with pytest.raises(ValueError):
        store.history_page(before=old, after=old)


def test_presentation_paging_does_not_truncate_generation_context(store):
    populate(store, 21)
    seen = []

    class Model:
        def stream(self, messages, cancelled):
            seen.extend(messages)
            yield "new reply"

        def cancel(self):
            pass

    session = TextSession(Model(), store, {})
    assert len(session.history()["messages"]) == 20
    session.send("new question")
    session.worker.join(2)
    assert not session.snapshot()["busy"]
    assert len(seen) == 43
    assert seen[0]["content"] == "0: synthetic"
    assert seen[-1]["content"] == "new question"


@pytest.mark.parametrize(
    "payload",
    [
        {"before": "a", "after": "b"},
        {"before": -1},
        {"after": ""},
        {"session_id": "a" * 129},
        {"limit": 99999},
    ],
)
def test_transport_rejects_invalid_history_window(payload):
    with pytest.raises(ValueError):
        validate("history", payload)
