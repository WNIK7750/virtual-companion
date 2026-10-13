from dataclasses import replace
from hashlib import sha256

import pytest
from test_pending_exchanges import apply, question, user_command
from test_rebuilt_turns import A, B
from virtual_companion.boundary.contracts import Scope
from virtual_companion.boundary.sqlite_exchanges import SQLiteExchangeStore


def test_source_port_returns_only_real_acknowledged_assistant_prefix(tmp_path):
    store = SQLiteExchangeStore(tmp_path / "sources.sqlite3")
    command = question(store, displayed=7)
    sources = store.exchange_sources(
        command.scope, references=(("assistant", command.evidence.source_id),)
    )
    assert len(sources) == 1
    assert sources[0].text == "晚上想聊什么？"
    assert sources[0].confirmed_end == 7
    assert sources[0].role == "assistant"


def test_unshown_delivery_yields_no_evidence_text(tmp_path):
    store = SQLiteExchangeStore(tmp_path / "unshown.sqlite3")
    command = question(store)
    assert (
        store.exchange_sources(
            command.scope, references=(("assistant", command.evidence.source_id),)
        )
        == ()
    )


def test_user_cancelling_generation_does_not_retract_actual_user_utterance(tmp_path):
    store = SQLiteExchangeStore(tmp_path / "cancel.sqlite3")
    command = user_command(store)
    store.cancel(A, command.evidence.source_id)
    sources = store.exchange_sources(
        command.scope, references=(("user", command.evidence.source_id),)
    )
    assert sources[0].text == command.content
    assert sources[0].confirmed_end == len(command.content)


def test_user_source_cannot_exclude_a_later_retraction(tmp_path):
    store = SQLiteExchangeStore(tmp_path / "truncated.sqlite3")
    prefix = "帮我安排周末。"
    command = user_command(store, text=prefix + "算了，不需要安排了。")
    command = replace(
        command,
        content=prefix,
        evidence=replace(
            command.evidence,
            quote=prefix,
            confirmed_end=len(prefix),
            prefix_sha256=sha256(prefix.encode()).hexdigest(),
        ),
    )
    with pytest.raises(ValueError, match="current confirmed"):
        apply(store, command)
    assert store.pending_exchanges(A, now_us=100) == []


def test_assistant_evaluation_must_cover_current_displayed_prefix(tmp_path):
    store = SQLiteExchangeStore(tmp_path / "expanded.sqlite3")
    prefix = "晚上想聊什么？"
    text = prefix + "先不用回答，我去忙一会。"
    command = question(store, text=text, displayed=len(text), quote=prefix)
    with pytest.raises(ValueError, match="current confirmed"):
        apply(store, command)
    assert store.pending_exchanges(A, now_us=100) == []


@pytest.mark.parametrize("failure", ["companion", "session", "role", "missing", "duplicate"])
def test_sources_cannot_be_relabelled_or_cross_scope(tmp_path, failure):
    store = SQLiteExchangeStore(tmp_path / "isolation.sqlite3")
    command = user_command(store)
    scope = command.scope
    refs = (("user", command.evidence.source_id),)
    if failure == "companion":
        scope = Scope(B, scope.session_id)
    elif failure == "session":
        scope = Scope(A, "other")
    elif failure == "role":
        refs = (("assistant", command.evidence.source_id),)
    elif failure == "missing":
        refs = (("user", "missing"),)
    else:
        refs *= 2
    with pytest.raises(ValueError):
        store.exchange_sources(scope, references=refs)
