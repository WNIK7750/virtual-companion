from dataclasses import replace

import pytest
from test_pending_exchanges import question
from test_rebuilt_turns import A, claim, config, result
from virtual_companion.boundary.contracts import VersionConflict
from virtual_companion.boundary.exchanges import ExchangeSupport
from virtual_companion.boundary.sqlite_maintenance import SQLiteMaintenanceStore


def prepared(tmp_path):
    store = SQLiteMaintenanceStore(tmp_path / "maintenance.sqlite3", evidence_clock=lambda: 100)
    command = question(store, displayed=7)
    item = store.accept_maintenance(config(), "maintain", command.scope, command.evidence)
    return store, command, item


def test_internal_work_has_no_fabricated_user_text_and_reopens(tmp_path):
    store, command, item = prepared(tmp_path)
    row = store.inspect(A, item.input_id)
    assert row["text"] == ""
    assert row["trigger"]["kind"] == "exchange_maintenance"
    assert row["trigger"]["source_id"] == command.evidence.source_id
    reopened = SQLiteMaintenanceStore(store.path)
    assert (
        reopened.accept_maintenance(config(), "maintain", command.scope, command.evidence) == item
    )
    with pytest.raises(ValueError):
        store.exchange_sources(command.scope, references=(("user", item.input_id),))


def test_maintenance_commits_exchanges_without_a_reply_and_replays(tmp_path):
    store, command, item = prepared(tmp_path)
    run = claim(store, item, now=200)
    proposals = ((command, ExchangeSupport(command.fingerprint, "synthetic")),)
    receipt = store.finish_maintenance(run, proposals, now_us=201)
    assert store.finish_maintenance(run, proposals, now_us=9999) == receipt
    assert store.pending_exchanges(A, now_us=100)[0]["direction"] == "user_question"
    with store.transaction(write=False) as db:
        assert db.execute("SELECT count(*) FROM turn_deliveries").fetchone()[0] == 1
    assert store.inspect(A, item.input_id)["status"] == "committed"
    with pytest.raises(ValueError):
        store.finish_maintenance(run, (), now_us=202)


def test_user_input_cancels_internal_generation_and_preserves_question(tmp_path):
    store, command, item = prepared(tmp_path)
    run = claim(store, item, now=200)
    user = store.accept(config(), "next-user", "换个话题吧", session_id=item.session_id)
    assert store.inspect(A, item.input_id)["status"] == "cancelled"
    with pytest.raises(VersionConflict):
        store.finish_maintenance(
            run, ((command, ExchangeSupport(command.fingerprint, "synthetic")),), now_us=201
        )
    assert store.pending_exchanges(A, now_us=100) == []
    assert claim(store, user, now=201).input_id == user.input_id


def test_unshown_or_nonterminal_assistant_source_cannot_enqueue(tmp_path):
    store = SQLiteMaintenanceStore(tmp_path / "unshown.sqlite3")
    command = question(store)
    with pytest.raises(ValueError):
        store.accept_maintenance(config(), "bad", command.scope, command.evidence)


def test_same_ack_cannot_queue_or_commit_twice_with_different_client_keys(tmp_path):
    store, command, item = prepared(tmp_path)
    with pytest.raises(VersionConflict):
        store.accept_maintenance(config(), "duplicate", command.scope, command.evidence)
    run = claim(store, item, now=200)
    store.finish_maintenance(run, (), now_us=201)
    with pytest.raises(VersionConflict):
        store.accept_maintenance(config(), "duplicate-after", command.scope, command.evidence)


def test_cancelled_maintenance_can_be_enqueued_later_without_losing_source(tmp_path):
    store, command, item = prepared(tmp_path)
    store.cancel(A, item.input_id)
    next_item = store.accept_maintenance(config(), "resume", command.scope, command.evidence)
    assert next_item.input_id != item.input_id
    assert store.inspect(A, next_item.input_id)["status"] == "queued"


def test_maintenance_cannot_change_bound_model_configuration(tmp_path):
    store, _, item = prepared(tmp_path)
    run = claim(store, item, now=200)
    with pytest.raises(VersionConflict):
        store.finish_maintenance(replace(run, config_json="{}"), (), now_us=201)


def test_late_cancel_or_failure_reconciles_completed_maintenance(tmp_path):
    store, _, item = prepared(tmp_path)
    run = claim(store, item, now=200)
    receipt = store.finish_maintenance(run, (), now_us=201)
    assert store.cancel(A, item.input_id) == receipt
    assert store.fail(run, reason="late_failure", now_us=202) == receipt


def test_retry_of_prior_user_intake_does_not_cancel_new_maintenance(tmp_path):
    store, command, item = prepared(tmp_path)
    # question() accepted this exact request before the maintenance existed.
    store.accept(config(), "start", "你好")
    assert store.inspect(A, item.input_id)["status"] == "queued"
    assert (
        store.exchange_sources(
            command.scope, references=(("assistant", command.evidence.source_id),)
        )[0].text
        == command.content
    )


def test_normal_reply_commit_cannot_disguise_internal_maintenance(tmp_path):
    store, _, item = prepared(tmp_path)
    run = claim(store, item, now=200)
    with pytest.raises(ValueError, match="maintenance"):
        store.commit(result(run, ""), now_us=201)


@pytest.mark.parametrize("failure", ["expired", "owner", "bad_second"])
def test_failed_maintenance_has_no_partial_effects(tmp_path, failure):
    store, command, item = prepared(tmp_path)
    run = claim(store, item, now=200)
    proposals = ((command, ExchangeSupport(command.fingerprint, "synthetic")),)
    now = 251 if failure == "expired" else 201
    if failure == "owner":
        store.claim_owner(A, "other", expected_epoch=run.owner.epoch)
    if failure == "bad_second":
        bad = replace(
            command, command_id="bad", exchange_id="missing", action="resolve", expected_revision=1
        )
        proposals += ((bad, ExchangeSupport(bad.fingerprint, "synthetic")),)
    with pytest.raises((ValueError, VersionConflict)):
        store.finish_maintenance(run, proposals, now_us=now)
    assert store.pending_exchanges(A, now_us=100) == []
    assert store.snapshot(A).write_version == 0
