from dataclasses import replace

import pytest
from test_pending_exchanges import user_command
from test_rebuilt_presentation import approved
from test_rebuilt_turns import A, claim, config, result
from virtual_companion.boundary.contracts import Scope, VersionConflict
from virtual_companion.boundary.exchanges import ExchangeSupport
from virtual_companion.boundary.presentation import MessageUnit, Presentation
from virtual_companion.boundary.sqlite_exchanges import SQLiteExchangeStore
from virtual_companion.boundary.sqlite_harness import SQLiteHarnessStore


def prepared(store):
    command = user_command(store)
    item = store.accept(config(), "request", command.content)
    run = claim(store, item)
    text = "我先想一想。"
    presentation = Presentation(
        Scope(A, item.session_id),
        item.input_id,
        run.attempt_id,
        "candidate",
        1,
        text,
        (MessageUnit("one", "text", text),),
    )
    proposal = replace(
        result(run, text),
        presentation=approved(presentation),
        exchanges=((command, ExchangeSupport(command.fingerprint, "synthetic")),),
    )
    return command, proposal


def test_exchange_and_reply_commit_and_reopen_as_one_transaction(tmp_path):
    store = SQLiteExchangeStore(tmp_path / "commit.sqlite3", evidence_clock=lambda: 100)
    command, proposal = prepared(store)
    receipt = store.commit(proposal, now_us=101)
    reopened = SQLiteExchangeStore(store.path, evidence_clock=lambda: 100)
    assert reopened.commit(proposal, now_us=9999) == receipt
    assert reopened.pending_exchanges(A, now_us=100)[0]["exchange_id"] == command.exchange_id
    assert receipt["displayed_end"] == 0  # No claim that the assistant has spoken.
    assert reopened.snapshot(A).write_version == 1
    with pytest.raises(ValueError, match="Conflicting"):
        reopened.commit(replace(proposal, exchanges=()), now_us=9999)


def test_invalid_second_exchange_rolls_back_reply_and_first_exchange(tmp_path):
    store = SQLiteExchangeStore(tmp_path / "rollback.sqlite3", evidence_clock=lambda: 100)
    command, proposal = prepared(store)
    bad = replace(
        command, command_id="bad", exchange_id="missing", action="close", expected_revision=1
    )
    proposal = replace(
        proposal,
        exchanges=(*proposal.exchanges, (bad, ExchangeSupport(bad.fingerprint, "synthetic"))),
    )
    with pytest.raises(VersionConflict):
        store.commit(proposal, now_us=101)
    assert store.pending_exchanges(A, now_us=100) == []
    assert store.snapshot(A).write_version == 0
    with store.transaction(write=False) as db:
        assert db.execute("SELECT count(*) FROM turn_commits").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM turn_deliveries").fetchone()[0] == 0


def test_old_adapter_cannot_silently_drop_new_exchange_effects(tmp_path):
    store = SQLiteHarnessStore(tmp_path / "old.sqlite3")
    _, proposal = prepared(store)
    with pytest.raises(ValueError, match="exchange"):
        store.commit(proposal, now_us=101)


def test_expiry_uses_utc_clock_not_the_turn_lease_clock(tmp_path):
    store = SQLiteExchangeStore(tmp_path / "expiry.sqlite3", evidence_clock=lambda: 1001)
    _, proposal = prepared(store)
    with pytest.raises(VersionConflict, match="Expired"):
        store.commit(proposal, now_us=101)
    assert store.pending_exchanges(A, now_us=100) == []


def test_cancelled_turn_cannot_commit_input_exchange(tmp_path):
    store = SQLiteExchangeStore(tmp_path / "cancel.sqlite3", evidence_clock=lambda: 100)
    _, proposal = prepared(store)
    store.cancel(A, proposal.claim.input_id)
    with pytest.raises(VersionConflict):
        store.commit(proposal, now_us=101)
    assert store.pending_exchanges(A, now_us=100) == []


def test_expiry_during_effect_application_rolls_back_whole_commit(tmp_path):
    now = [100]

    class SlowStore(SQLiteExchangeStore):
        def _commit_exchanges(self, db, proposal, receipt):
            super()._commit_exchanges(db, proposal, receipt)
            now[0] = 200

    store = SlowStore(tmp_path / "slow.sqlite3", evidence_clock=lambda: 100)
    _, proposal = prepared(store)
    with pytest.raises(VersionConflict, match="lease"):
        store.commit(proposal, now_us=100, clock=lambda: now[0])
    assert store.pending_exchanges(A, now_us=100) == []
