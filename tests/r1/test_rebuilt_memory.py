import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from datetime import datetime, timedelta
from hashlib import sha256
from threading import Barrier

import pytest
from virtual_companion.boundary.contracts import Scope, VersionConflict
from virtual_companion.boundary.memory import FactKey, MemoryEffect, MemoryMutation
from virtual_companion.boundary.sqlite_memory import SQLiteMemoryStore
from virtual_companion.boundary.turns import CognitiveResult
from virtual_companion.text.identity import CompanionConfig, VersionedReference

A = "00000000-0000-0000-0000-000000000001"
B = "00000000-0000-0000-0000-000000000002"
WHEN = datetime.fromisoformat("2026-09-09T12:00:00Z")


def mutation(key="one", *, kind="append_event", record="event-one", content="一起散步", **kwargs):
    return MemoryMutation(
        Scope(A, "session"),
        key,
        kind,
        record,
        content,
        "input",
        "synthetic-input",
        "user statement",
        WHEN,
        **kwargs,
    )


@pytest.fixture
def store(tmp_path):
    db = SQLiteMemoryStore(tmp_path / "memory.sqlite3")
    db.register(A)
    db.claim_owner(A, "worker", expected_epoch=0)
    return db


def apply(store, command):
    return store.mutate(command, store.owner(A), expected_version=store.snapshot(A).write_version)


def test_independent_experiences_do_not_share_fact_slot(store):
    apply(store, mutation())
    apply(store, mutation("two", record="event-two", content="一起看电影"))
    rows = store.recall(A, now_us=0)
    assert {row["content"] for row in rows} == {"一起散步", "一起看电影"}
    assert len(store.history(A, "event-one")) == 1


def fact(store):
    command = mutation(
        kind="assert_fact",
        record="home",
        content="住在北京",
        fact_key=FactKey("user", "home-city", "profile"),
    )
    apply(store, command)
    return command


def test_correction_retains_source_chain_and_filters_stale_index(store):
    fact(store)
    corrected = mutation(
        "fix", kind="correct", record="home", content="住在上海", expected_record_version=1
    )
    apply(store, corrected)
    assert [r["content"] for r in store.recall(A, now_us=0)] == ["住在上海"]
    assert store.recall(A, now_us=0, candidates=(("home", 1),)) == []
    assert store.recall(A, now_us=0, candidates=(("home", 2),))[0]["content"] == "住在上海"
    history = store.history(A, "home")
    assert [r["version"] for r in history] == [1, 2]
    assert history[1]["previous_version"] == 1
    assert history[0]["source_id"] == "synthetic-input"


def test_event_time_is_optional_and_revision_does_not_use_audit_time_as_event_time(store):
    created = mutation(
        "dated",
        kind="assert_fact",
        record="dated-home",
        content="以前住在北京，具体哪年记不清",
        fact_key=FactKey("user", "home-city", "profile"),
        event_time=None,
    )
    apply(store, created)
    first = store.history(A, "dated-home")[0]
    assert first["event_time"] is None
    assert first["occurred_at"] == WHEN.isoformat()

    corrected = mutation(
        "dated-fix",
        kind="correct",
        record="dated-home",
        content="以前住在北京，大约是上学时",
        expected_record_version=1,
        event_time=None,
    )
    apply(store, corrected)
    current = store.recall(A, now_us=0)[0]
    assert current["event_time"] is None
    assert current["occurred_at"] == WHEN.isoformat()


def test_unknown_correction_remains_pending_and_retry_does_not_retarget(store):
    command = mutation(
        "fix", kind="correct", record="home", content="上海", expected_record_version=1
    )
    receipt = apply(store, command)
    assert receipt["disposition"] == "pending_target"
    fact(store)
    assert apply(store, command) == receipt
    assert store.recall(A, now_us=0)[0]["content"] == "住在北京"


def test_correction_with_unknown_details_retains_expiry_in_payload_and_index(store):
    from virtual_companion.text.chronology import utc_microseconds

    expiry = WHEN + timedelta(days=1)
    apply(store, mutation(
        kind="assert_fact", record="home", content="这周暂住北京",
        fact_key=FactKey("user", "home-city", "profile"), expires_at=expiry,
    ))
    correction = mutation(
        "fix-ttl", kind="correct", record="home", content="这周暂住上海，抵达时间不详",
        expected_record_version=1,
    )
    receipt = apply(store, correction)
    assert receipt["disposition"] == "applied"
    assert apply(store, correction) == receipt
    assert store.history(A, "home")[-1]["expires_at"] == expiry.isoformat()
    assert len(store.recall(A, now_us=utc_microseconds(expiry) - 1)) == 1
    assert store.recall(A, now_us=utc_microseconds(expiry)) == []
    assert len(store.history(A, "home")) == 2


@pytest.mark.parametrize("new_expiry", [None, WHEN + timedelta(days=2)])
def test_correction_cannot_revive_expired_fact_through_single_mutation(store, new_expiry):
    from virtual_companion.text.chronology import utc_microseconds

    expiry = WHEN + timedelta(days=1)
    apply(store, mutation(
        kind="assert_fact", record="home", content="暂住北京",
        fact_key=FactKey("user", "home-city", "profile"), expires_at=expiry,
    ))
    correction = replace(mutation(
        "expired-fix", kind="correct", record="home", content="暂住上海",
        expected_record_version=1, expires_at=new_expiry,
    ), occurred_at=expiry)
    assert apply(store, correction)["disposition"] == "conflict"
    assert store.recall(A, now_us=utc_microseconds(expiry)) == []
    assert len(store.history(A, "home")) == 1


def test_revocation_expiry_and_stale_rewrite_cannot_resurrect(store):
    fact(store)
    apply(
        store,
        mutation("revoke", kind="revoke", record="home", content="", expected_record_version=1),
    )
    assert store.recall(A, now_us=0) == []
    receipt = apply(
        store, mutation("stale", kind="correct", record="home", expected_record_version=1)
    )
    assert receipt["disposition"] == "conflict"
    assert store.recall(A, now_us=0, candidates=(("home", 1),)) == []
    apply(store, mutation("ttl", record="temporary", expires_at=WHEN))
    from virtual_companion.text.chronology import utc_microseconds

    assert store.recall(A, now_us=utc_microseconds(WHEN)) == []


def test_shared_state_version_and_owner_fence_cover_memory_writes(store):
    basis, owner = store.snapshot(A), store.owner(A)
    command = mutation()
    receipt = store.mutate(command, owner, expected_version=basis.write_version)
    assert store.snapshot(A).write_version == basis.write_version + 1
    assert store.mutate(command, owner, expected_version=0) == receipt
    with pytest.raises(VersionConflict):
        store.mutate(mutation("two", record="two"), owner, expected_version=0)
    store.claim_owner(A, "next", expected_epoch=owner.epoch)
    with pytest.raises(VersionConflict):
        store.mutate(mutation("two", record="two"), owner, expected_version=1)
    with pytest.raises(ValueError):
        apply(store, replace(command, content="conflicting retry"))


def turn(store, *, effects=()):
    config = CompanionConfig(A, VersionedReference("persona", "1", "a" * 64))
    item = store.accept(config, "first", "你好")
    run = store.claim(A, item.input_id, "turn-worker", now_us=100, lease_us=100)
    # Effects must use the actual session selected by intake.
    effects = tuple(
        replace(e, mutation=replace(e.mutation, scope=Scope(A, item.session_id))) for e in effects
    )
    proposal = CognitiveResult(
        run,
        "candidate",
        1,
        "你🙂好",
        "review",
        sha256("你🙂好".encode()).hexdigest(),
        memories=effects,
    )
    return item, proposal, store.commit(proposal, now_us=101)


def output_effect(key="said", end=2):
    command = replace(mutation(key, record=key), source_kind="output", source_id="candidate")
    return MemoryEffect(command, end)


def ack(store, receipt, end, sequence=1, stopped=False):
    return store.acknowledge(
        A,
        receipt["delivery_id"],
        candidate_id=receipt["candidate_id"],
        revision=receipt["revision"],
        sha256=receipt["sha256"],
        sequence=sequence,
        displayed_end=end,
        stopped=stopped,
    )


def test_output_effect_requires_confirmed_prefix_not_commit_or_publish(store):
    _, proposal, receipt = turn(store, effects=(MemoryEffect(mutation()), output_effect()))
    assert len(store.recall(A, now_us=0)) == 1
    store.publish(A, receipt["delivery_id"])
    assert len(store.recall(A, now_us=0)) == 1
    ack(store, receipt, 1)
    assert len(store.recall(A, now_us=0)) == 1
    ack(store, receipt, 2, 2)
    assert len(store.recall(A, now_us=0)) == 2
    version = store.snapshot(A).write_version
    assert SQLiteMemoryStore(store.path).commit(proposal, now_us=1000) == receipt
    ack(store, receipt, 2, 2)
    assert store.snapshot(A).write_version == version


def test_cancel_retains_only_confirmed_output_effects(store):
    item, _, receipt = turn(store, effects=(output_effect("prefix", 1), output_effect("suffix", 3)))
    store.publish(A, receipt["delivery_id"])
    ack(store, receipt, 1)
    store.cancel(A, item.input_id)
    ack(store, receipt, 1, 2, stopped=True)
    assert [r["record_id"] for r in store.recall(A, now_us=0)] == ["prefix"]
    assert store.effects(A, receipt["delivery_id"])[1]["status"] == "cancelled"


def test_cancel_before_publish_never_applies_effects(store):
    item, _, receipt = turn(store, effects=(output_effect(),))
    store.cancel(A, item.input_id)
    assert store.publish(A, receipt["delivery_id"]) is None
    assert store.recall(A, now_us=0) == []
    assert store.effects(A, receipt["delivery_id"])[0]["status"] == "cancelled"


def test_fact_slot_is_explicit_and_different_companions_remain_isolated(store):
    original = fact(store)
    conflict = apply(store, replace(original, mutation_id="other", record_id="new-id"))
    assert conflict["disposition"] == "conflict"
    store.register(B)
    owner = store.claim_owner(B, "other", expected_epoch=0)
    other = replace(original, scope=Scope(B, "session"))
    assert store.mutate(other, owner, expected_version=0)["disposition"] == "applied"
    assert store.history(B, "home")[0]["content"] == "住在北京"
    assert len(store.recall(A, now_us=0)) == 1
    with pytest.raises(ValueError):
        store.mutate(original, owner, expected_version=0)


def test_two_writers_cannot_both_correct_same_version(store):
    fact(store)
    owner, basis = store.owner(A), store.snapshot(A).write_version
    barrier = Barrier(2)

    def correct(key):
        barrier.wait()
        command = mutation(
            key, kind="correct", record="home", content=key, expected_record_version=1
        )
        try:
            return store.mutate(command, owner, expected_version=basis)
        except VersionConflict:
            return None

    with ThreadPoolExecutor(2) as pool:
        receipts = list(pool.map(correct, ["上海", "深圳"]))
    assert sum(r is not None for r in receipts) == 1
    assert len(store.history(A, "home")) == 2


def test_unknown_target_without_identity_is_queryable_and_never_overwrites(store):
    command = mutation("unknown", kind="correct", record=None, content="新的住址")
    receipt = apply(store, command)
    assert receipt["disposition"] == "pending_target"
    assert store.memory_receipt(A, "unknown") == receipt
    assert store.memory_receipt(B, "unknown") is None
    assert store.recall(A, now_us=0) == []


def test_failed_memory_write_rolls_back_whole_cognitive_commit(store):
    with store.transaction() as db:
        db.execute(
            "CREATE TRIGGER break_memory BEFORE INSERT ON memory_commands "
            "BEGIN SELECT RAISE(ABORT,'injected'); END"
        )
    with pytest.raises(sqlite3.IntegrityError):
        turn(store, effects=(MemoryEffect(mutation()),))
    assert store.snapshot(A).write_version == 0
    assert store.recall(A, now_us=0) == []
    with store.transaction(write=False) as db:
        assert db.execute("SELECT count(*) FROM turn_commits").fetchone()[0] == 0
        assert db.execute("SELECT status FROM turn_runs").fetchone()[0] == "running"


def test_failed_output_memory_rolls_back_ack_then_retry_applies_once(store):
    _, _, receipt = turn(store, effects=(output_effect(),))
    store.publish(A, receipt["delivery_id"])
    with store.transaction() as db:
        db.execute(
            "CREATE TRIGGER break_memory BEFORE INSERT ON memory_commands "
            "BEGIN SELECT RAISE(ABORT,'injected'); END"
        )
    with pytest.raises(sqlite3.IntegrityError):
        ack(store, receipt, 2)
    assert store.delivery(A, receipt["delivery_id"])["displayed_end"] == 0
    assert store.history(A, "said") == []
    assert store.effects(A, receipt["delivery_id"])[0]["status"] == "pending"
    with store.transaction() as db:
        db.execute("DROP TRIGGER break_memory")
    ack(store, receipt, 2)
    assert len(store.history(A, "said")) == 1
    assert store.memory_receipt(A, "said")["delivery_id"] == receipt["delivery_id"]


def test_stale_output_correction_records_conflict_without_losing_ack(store):
    fact(store)
    command = replace(
        mutation(
            "spoken-fix", kind="correct", record="home", content="上海", expected_record_version=1
        ),
        source_kind="output",
        source_id="candidate",
    )
    _, _, receipt = turn(store, effects=(MemoryEffect(command, 2),))
    apply(
        store,
        mutation(
            "manual-fix", kind="correct", record="home", content="深圳", expected_record_version=1
        ),
    )
    store.publish(A, receipt["delivery_id"])
    ack(store, receipt, 3)
    assert store.delivery(A, receipt["delivery_id"])["status"] == "completed"
    assert store.effects(A, receipt["delivery_id"])[0]["status"] == "conflict"
    assert store.recall(A, now_us=0)[0]["content"] == "深圳"


def test_reserved_output_id_cannot_be_used_by_background_command(store):
    _, _, receipt = turn(store, effects=(output_effect(),))
    with pytest.raises(VersionConflict):
        apply(store, mutation("said", record="background"))
    with pytest.raises(ValueError):
        apply(store, output_effect().mutation)
    assert store.effects(A, receipt["delivery_id"])[0]["status"] == "pending"


def test_v2_upgrade_preserves_unresolved_turn_and_backup_can_restore(tmp_path):
    from virtual_companion.boundary.sqlite_turns import SQLiteTurnStore

    path = tmp_path / "v2.sqlite3"
    old = SQLiteTurnStore(path)
    _, proposal, receipt = turn(old)
    before = sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        SQLiteMemoryStore(path)
    assert sha256(path.read_bytes()).hexdigest() == before
    backup = tmp_path / "backup.sqlite3"
    with closing(sqlite3.connect(path)) as source, closing(sqlite3.connect(backup)) as destination:
        source.backup(destination)
    upgraded = SQLiteMemoryStore(path, upgrade_v2=True)
    assert upgraded.commit(proposal, now_us=999) == receipt
    assert upgraded.delivery(A, receipt["delivery_id"])["status"] == "pending"
    with pytest.raises(ValueError):
        old.snapshot(A)
    assert SQLiteTurnStore(backup).commit(proposal, now_us=999) == receipt


def test_upgrade_ddl_failure_and_future_schema_do_not_change_database(tmp_path, monkeypatch):
    import virtual_companion.boundary.sqlite_memory as module
    from virtual_companion.boundary.sqlite_turns import SQLiteTurnStore

    path = tmp_path / "v2.sqlite3"
    SQLiteTurnStore(path).register(A)
    before = sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(module, "_MEMORY_SCHEMA", module._MEMORY_SCHEMA + ("INVALID SQL",))
    with pytest.raises(sqlite3.OperationalError):
        SQLiteMemoryStore(path, upgrade_v2=True)
    assert sha256(path.read_bytes()).hexdigest() == before
    with closing(sqlite3.connect(path)) as db:
        db.execute("PRAGMA user_version=99")
    before = sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        SQLiteMemoryStore(path, upgrade_v2=True)
    assert sha256(path.read_bytes()).hexdigest() == before


@pytest.mark.parametrize(
    "changes",
    [
        {"source_kind": "persona"},
        {"reason": ""},
        {"content": ""},
        {"fact_key": FactKey("user", "home", "profile")},
        {"expected_record_version": True},
        {"occurred_at": WHEN.replace(tzinfo=None)},
        {"content": "好" * 6000},
        {"expires_at": WHEN.replace(year=2025)},
    ],
)
def test_invalid_memory_contract_rejected_before_storage(changes):
    with pytest.raises(ValueError):
        replace(mutation(), **changes)


def test_effects_cannot_bypass_output_gate_or_borrow_other_session(store):
    with pytest.raises(ValueError):
        MemoryEffect(output_effect().mutation)
    with pytest.raises(ValueError):
        MemoryEffect(mutation(), 2)
    _, proposal, _ = turn(store)
    with pytest.raises(ValueError):
        replace(proposal, memories=(output_effect(),))
    scoped = replace(
        output_effect(),
        mutation=replace(output_effect().mutation, scope=Scope(A, proposal.claim.session_id)),
    )
    with pytest.raises(ValueError):
        replace(proposal, memories=(replace(scoped, after_codepoints=4),))
    with pytest.raises(ValueError):
        replace(proposal, memories=(scoped, scoped))


@pytest.mark.parametrize("prefixes", [(3,), (1, 2, 3)])
def test_coalesced_ack_matches_incremental_effect_order(store, prefixes):
    asserted = replace(
        mutation(
            "assert",
            kind="assert_fact",
            record="home",
            content="北京",
            fact_key=FactKey("user", "home", "profile"),
        ),
        source_kind="output",
        source_id="candidate",
    )
    corrected = replace(
        mutation(
            "correct", kind="correct", record="home", content="上海", expected_record_version=1
        ),
        source_kind="output",
        source_id="candidate",
    )
    # Plan storage order is not the order in which the host displayed each prefix.
    _, _, receipt = turn(store, effects=(MemoryEffect(corrected, 3), MemoryEffect(asserted, 1)))
    store.publish(A, receipt["delivery_id"])
    for sequence, end in enumerate(prefixes, 1):
        ack(store, receipt, end, sequence)
    assert [r["content"] for r in store.history(A, "home")] == ["北京", "上海"]
    assert store.recall(A, now_us=0)[0]["content"] == "上海"
