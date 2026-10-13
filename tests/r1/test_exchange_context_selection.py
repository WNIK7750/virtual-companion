from dataclasses import replace

import pytest
from test_pending_exchanges import question
from test_rebuilt_presentation import approved
from test_rebuilt_turns import A, claim, config, result
from virtual_companion.boundary.presentation import Presentation
from virtual_companion.boundary.sqlite_conversation import SQLiteConversationStore


def prepared(tmp_path):
    store = SQLiteConversationStore(tmp_path / "conversation.sqlite3", evidence_clock=lambda: 100)
    command = question(store, displayed=7)
    item = store.accept(config(), "answer", "想聊旅行", session_id=command.scope.session_id)
    run = claim(store, item, now=200)
    return store, command, item, run


def complete(store, item, run, sources):
    from virtual_companion.boundary.contracts import Scope

    display = Presentation(
        Scope(A, item.session_id), item.input_id, run.attempt_id, "candidate", 1, "", ()
    )
    value = replace(result(run, ""), presentation=approved(display), exchange_sources=sources)
    return store.commit(value, now_us=201)


def test_context_is_ordered_actual_text_and_ends_at_current_user(tmp_path):
    store, command, item, run = prepared(tmp_path)
    sources, eligible = store.exchange_context(run)
    assert [(s.role, s.text) for s in sources] == [
        ("user", "你好"),
        ("assistant", command.content),
        ("user", "想聊旅行"),
    ]
    assert sources[-1].source_id == item.input_id
    assert eligible == tuple(s.source_id for s in sources)


def test_empty_extraction_consumes_sources_and_reopen_does_not_repeat_them(tmp_path):
    store, _, item, run = prepared(tmp_path)
    sources, _ = store.exchange_context(run)
    complete(store, item, run, sources)
    store = SQLiteConversationStore(store.path, evidence_clock=lambda: 100)
    next_item = store.accept(config(), "next", "换个话题", session_id=item.session_id)
    next_run = claim(store, next_item, now=300)
    context, eligible = store.exchange_context(next_run)
    assert len(context) == 4
    assert eligible == (next_item.input_id,)


def test_failed_commit_does_not_mark_sources_consumed(tmp_path):
    store, _, _, run = prepared(tmp_path)
    sources, eligible = store.exchange_context(run)
    with pytest.raises(ValueError):
        store.commit(replace(result(run, ""), exchange_sources=sources), now_us=201)
    assert store.exchange_context(run)[1] == eligible


def test_future_input_does_not_enter_current_source_window(tmp_path):
    store, _, _, run = prepared(tmp_path)
    future = store.accept(config(), "future", "后面的输入")
    sources, eligible = store.exchange_context(run)
    assert future.input_id not in eligible
    assert "后面的输入" not in [s.text for s in sources]
