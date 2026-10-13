from dataclasses import replace

import pytest
from test_pending_exchanges import question
from test_rebuilt_presentation import approved
from test_rebuilt_turns import A, claim, config, result
from virtual_companion.boundary.evidence import evidence_digest
from virtual_companion.boundary.exchanges import ExchangeSupport
from virtual_companion.boundary.presentation import Presentation
from virtual_companion.boundary.sqlite_conversation import SQLiteConversationStore
from virtual_companion.boundary.sqlite_exchanges import SQLiteExchangeStore


def prepared(tmp_path, store_type=SQLiteConversationStore):
    store = store_type(tmp_path / "context.sqlite3", evidence_clock=lambda: 100)
    command = question(store, displayed=7)
    item = store.accept(config(), "next", "我先喝口水", session_id=command.scope.session_id)
    run = claim(store, item, now=200)
    sources = store.exchange_sources(
        command.scope,
        references=(
            ("assistant", command.evidence.source_id),
            ("user", item.input_id),
        ),
    )
    support = ExchangeSupport(command.fingerprint, "synthetic", evidence_digest(sources))
    display = Presentation(command.scope, item.input_id, run.attempt_id, "candidate", 1, "", ())
    proposal = replace(
        result(run, ""),
        presentation=approved(display),
        exchanges=((command, support),),
        exchange_sources=sources,
    )
    return store, command, sources, proposal


def test_prior_actual_assistant_source_can_commit_with_current_user_context(tmp_path):
    store, command, _, proposal = prepared(tmp_path)
    store.commit(proposal, now_us=201)
    assert store.pending_exchanges(A, now_us=100)[0]["content"] == command.content
    assert "exchange_sources" in proposal.payload


@pytest.mark.parametrize("change", ["drop", "text", "reorder"])
def test_context_cannot_be_changed_after_semantic_support(tmp_path, change):
    _, _, sources, proposal = prepared(tmp_path)
    changed = (
        sources[:1]
        if change == "drop"
        else sources[::-1]
        if change == "reorder"
        else (
            sources[0],
            replace(sources[1], text="伪造来源", confirmed_end=4),
        )
    )
    with pytest.raises(ValueError):
        replace(proposal, exchange_sources=changed)


def test_invented_context_rejected_by_storage_even_with_matching_synthetic_digest(tmp_path):
    store, command, sources, proposal = prepared(tmp_path)
    sources = (sources[0], replace(sources[1], text="伪造来源", confirmed_end=4))
    proposal = replace(
        proposal,
        exchange_sources=sources,
        exchanges=(
            (command, ExchangeSupport(command.fingerprint, "synthetic", evidence_digest(sources))),
        ),
    )
    with pytest.raises(ValueError):
        store.commit(proposal, now_us=201)
    assert store.pending_exchanges(A, now_us=100) == []


def test_even_empty_extraction_must_bind_actual_sources(tmp_path):
    store, _, sources, proposal = prepared(tmp_path)
    sources = (sources[0], replace(sources[1], text="伪造来源", confirmed_end=4))
    proposal = replace(proposal, exchanges=(), exchange_sources=sources)
    with pytest.raises(ValueError):
        store.commit(proposal, now_us=201)


def test_old_exchange_adapter_rejects_contextual_effects_without_migration(tmp_path):
    store, _, _, proposal = prepared(tmp_path, SQLiteExchangeStore)
    with pytest.raises(ValueError, match="Contextual exchanges"):
        store.commit(proposal, now_us=201)
