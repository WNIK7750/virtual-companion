from dataclasses import replace

import pytest
from test_rebuilt_presentation import prepared
from test_rebuilt_turns import A, B, config
from virtual_companion.boundary.sqlite_background import SQLiteBackgroundStore


def test_user_intake_and_barrier_survive_reopen_and_ten_continuous_inputs(tmp_path):
    store = SQLiteBackgroundStore(tmp_path / "queue.sqlite3")
    first = store.accept(config(), "u0", "我喜欢甜饮料")
    assert store.memory_barrier(A)
    with store.transaction() as db:
        db.execute("UPDATE background_sources SET status='running' WHERE companion=?", (A,))
    for index in range(1, 11):
        store.accept(config(), f"u{index}", "更正，我现在不喜欢甜饮料")
    reopened = SQLiteBackgroundStore(store.path)
    jobs = reopened.background_sources(A)
    assert len(jobs) == 11 and all(j["status"] == "queued" for j in jobs)
    assert jobs[0]["source"] == first.input_id and jobs[0]["generation"] == 1
    assert reopened.memory_barrier(A)
    assert reopened.accept(config(), "u0", "我喜欢甜饮料") == first
    assert len(reopened.background_sources(A)) == 11


def test_intake_failure_rolls_back_source_job_and_user_input(tmp_path):
    store = SQLiteBackgroundStore(tmp_path / "queue.sqlite3")
    import sqlite3

    with store.transaction() as db:
        db.execute("CREATE TRIGGER fail_job BEFORE INSERT ON background_sources "
                   "BEGIN SELECT RAISE(ABORT,'injected'); END")
    with pytest.raises(sqlite3.IntegrityError):
        store.accept(config(), "u0", "你好")
    with store.transaction(write=False) as db:
        assert db.execute("SELECT count(*) FROM turn_inputs").fetchone()[0] == 0


def test_ack_queues_only_actual_prefix_without_fake_user_or_cross_companion(tmp_path):
    store = SQLiteBackgroundStore(tmp_path / "queue.sqlite3")
    item, result = prepared(store)
    receipt = store.commit(result, now_us=101)
    assert [j["role"] for j in store.background_sources(A)] == ["user"]
    assert store.background_sources(B) == [] and not store.memory_barrier(B)
    store.publish(A, receipt["delivery_id"])
    store.cancel(A, item.input_id)
    args = dict(candidate_id=receipt["candidate_id"], revision=1, sha256=receipt["sha256"],
                presentation_sha256=receipt["presentation_sha256"], sequence=1,
                displayed_end=2, stopped=True)
    ack = store.acknowledge(A, receipt["delivery_id"], **args)
    assert store.acknowledge(A, receipt["delivery_id"], **args) == ack
    jobs = store.background_sources(A)
    assert len(jobs) == 2 and jobs[1]["role"] == "assistant"
    assert jobs[1]["confirmed_end"] == 2
    assert store.background_source_text(A, jobs[1]["job"]) == "你🙂"
    with store.transaction(write=False) as db:
        assert db.execute("SELECT count(*) FROM turn_inputs").fetchone()[0] == 1


def test_backpressure_never_silently_accepts_an_unqueued_source(tmp_path):
    store = SQLiteBackgroundStore(tmp_path / "queue.sqlite3")
    # Use a smaller explicit test capacity; production default remains bounded.
    store.max_pending_sources = 2
    store.accept(config(), "u1", "第一条")
    store.accept(config(), "u2", "第二条")
    with pytest.raises(ValueError, match="backpressure"):
        store.accept(config(), "u3", "不能悄悄丢掉")
    assert len(store.background_sources(A)) == 2
    store.accept(replace(config(), companion_id=B), "u3", "另一个伴侣")
    assert len(store.background_sources(B)) == 1
