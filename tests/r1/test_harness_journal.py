import copy

import pytest
from test_harness_port import Model, execute, response, tool_call
from test_rebuilt_turns import A, claim, intake
from virtual_companion.boundary.contracts import VersionConflict
from virtual_companion.boundary.sqlite_harness import SQLiteHarnessStore, harness_binding
from virtual_companion.boundary.sqlite_lifecycle import SQLiteLifecycleStore
from virtual_companion.cognition.harness import recover_transcript


class V7SchemaStore(SQLiteLifecycleStore):
    """Synthetic old format only; no retired relationship projection."""

    schema_version = 7
    schema = SQLiteHarnessStore.schema[:-1]
    tables = SQLiteHarnessStore.tables - {"harness_checkpoints"}


def setup_store(tmp_path):
    store = SQLiteHarnessStore(tmp_path / "harness.sqlite3")
    item = intake(store)
    run = claim(store, item)
    return store, item, run


def test_real_checkpoint_reopen_and_recovery_preserve_pending_original(tmp_path):
    store, _, run = setup_store(tmp_path)
    journal = store.harness_journal(run, "reply", clock=lambda: 100)
    model = Model([response("", [tool_call(body={"recipient": "fixture"})])])

    def crash(_):
        raise KeyboardInterrupt("synthetic process interruption")

    with pytest.raises(KeyboardInterrupt):
        execute(model, journal, dispatch=crash, binding=harness_binding(run))
    reopened = SQLiteHarnessStore(store.path)
    saved = reopened.harness_journal(run, "reply", clock=lambda: 101).load()
    assert saved["pending"]["arguments"] == '{"body": {"recipient": "fixture"}}'
    recovered = recover_transcript(saved, binding=harness_binding(run))
    assert recovered["uncertain"]
    assert reopened.snapshot(A).write_version == 0
    assert reopened.inspect(A, run.input_id)["status"] == "running"
    with reopened.transaction(write=False) as db:
        assert db.execute("SELECT count(*) FROM turn_commits").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM harness_checkpoints").fetchone()[0] >= 3


@pytest.mark.parametrize("cause", ["expiry", "takeover", "cancel", "state_owner"])
def test_stale_journal_cannot_save_or_load(tmp_path, cause):
    store, item, run = setup_store(tmp_path)
    now = [100]
    journal = store.harness_journal(run, "reply", clock=lambda: now[0])
    execute(Model([response()]), journal, binding=harness_binding(run))
    value = journal.load()
    if cause == "expiry":
        now[0] = 150
    elif cause == "takeover":
        store.claim(A, item.input_id, "new", now_us=150, lease_us=50)
    elif cause == "cancel":
        store.cancel(A, item.input_id)
    else:
        store.claim_owner(A, "new", expected_epoch=run.owner.epoch)
    with pytest.raises(VersionConflict):
        journal.save(value)
    with pytest.raises(VersionConflict):
        journal.load()


def test_concurrent_journal_handles_cannot_overwrite_newer_checkpoint(tmp_path):
    store, _, run = setup_store(tmp_path)
    first = store.harness_journal(run, "reply", clock=lambda: 100)
    second = store.harness_journal(run, "reply", clock=lambda: 100)
    execute(Model([response()]), first, binding=harness_binding(run))
    state = first.load()
    with pytest.raises(VersionConflict, match="checkpoint revision"):
        second.save(state)
    assert second.load() == state
    second.save(state)
    with pytest.raises(VersionConflict, match="checkpoint revision"):
        first.save(state)


def test_new_owner_reconciles_old_snapshot_explicitly_without_executing(tmp_path):
    store, item, old = setup_store(tmp_path)
    journal = store.harness_journal(old, "reply", clock=lambda: 100)
    execute(Model([response()]), journal, binding=harness_binding(old))
    previous = copy.deepcopy(journal.load())
    new = store.claim(A, item.input_id, "new", now_us=150, lease_us=50)
    fresh = store.harness_journal(new, "reply", clock=lambda: 151)
    with pytest.raises(VersionConflict, match="binding"):
        fresh.load()
    original = fresh.load_for_reconciliation()
    assert original == previous
    recovered = recover_transcript(original, binding=original["binding"])
    recovered["binding"] = harness_binding(new)
    fresh.save(recovered)
    assert fresh.load()["cache_epoch"] == 1
    with store.transaction(write=False) as db:
        assert (
            db.execute("SELECT count(DISTINCT epoch) FROM harness_checkpoints").fetchone()[0] == 2
        )


def test_existing_v7_is_rejected_without_schema_migration(tmp_path):
    path = tmp_path / "old.sqlite3"
    old = V7SchemaStore(path)
    intake(old)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        SQLiteHarnessStore(path)
    assert path.read_bytes() == before
