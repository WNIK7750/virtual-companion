import sqlite3

import pytest
from test_rebuilt_turns import A, claim, config
from test_segment_store import reviewed
from virtual_companion.boundary.contracts import Scope, VersionConflict
from virtual_companion.boundary.sqlite_streaming import SQLiteStreamingStore


def setup(tmp_path):
    store = SQLiteStreamingStore(tmp_path / "stream.sqlite3")
    item = store.accept(config(), "first", "你好")
    run = claim(store, item)
    store.approve_segment(run, reviewed(run), now_us=100)
    event = store.dispatch_segment(run, 1, now_us=100)
    return store, run, event


def confirm(store, run, event):
    return store.confirm_segment(A, run.input_id, 1, fingerprint=event["fingerprint"],
                                 host_receipt="real-visible", channel="text_visible")


def test_silent_terminal_requires_no_approved_prefix_and_never_creates_assistant_source(tmp_path):
    store = SQLiteStreamingStore(tmp_path / "silent.sqlite3")
    item = store.accept(config(), "listen", "我还没说完。")
    owner = claim(store, item)
    receipt = store.finish_stream(owner, outcome="listened", now_us=100)
    assert receipt["response_type"] == "listen" and receipt["approved_units"] == 0
    assert store.finish_stream(owner, outcome="listened", now_us=9999) == receipt
    assert store.stream_state(A, item.input_id)["queue_closed"]
    assert [s["role"] for s in store.background_sources(A)] == ["user"]
    store = SQLiteStreamingStore(store.path)
    assert store.stream_state(A, item.input_id)["outcome"] == "listened"
    next_item = store.accept(config(), "next", "现在补充好了。", session_id=item.session_id)
    assert claim(store, next_item).input_id == next_item.input_id


def test_listening_cannot_hide_an_approved_spoken_unit(tmp_path):
    store, owner, _ = setup(tmp_path)
    with pytest.raises(ValueError, match="Listening"):
        store.finish_stream(owner, outcome="listened", now_us=100)
    assert store.stream_state(A, owner.input_id)["settlement"] is None


def test_confirmed_unit_enters_history_and_source_queue_atomically(tmp_path):
    store, run, event = setup(tmp_path)
    confirm(store, run, event)
    jobs = [s for s in store.background_sources(A) if s["role"] == "assistant"]
    assert len(jobs) == 1
    source = store.exchange_sources(Scope(A, run.session_id),
                                    references=(("assistant", jobs[0]["source"]),))
    assert source[0].text == "我在这里。"
    store.finish_stream(run, outcome="failed", now_us=100)
    next_item = store.accept(config(), "next", "继续", session_id=run.session_id)
    next_run = claim(store, next_item, now=200)
    history, _ = store.exchange_context(next_run)
    assert [s.text for s in history] == ["你好", "我在这里。", "继续"]
    assert store.inspect(A, run.input_id)["delivery"] is None


def test_unknown_is_not_history_and_late_receipt_survives_failed_terminal(tmp_path):
    store, run, event = setup(tmp_path)
    terminal = store.finish_stream(run, outcome="failed", now_us=100)
    assert terminal == store.finish_stream(run, outcome="failed", now_us=100)
    assert not [s for s in store.background_sources(A) if s["role"] == "assistant"]
    reopened = SQLiteStreamingStore(store.path)
    confirm(reopened, run, event)
    confirm(reopened, run, event)
    assert len([s for s in reopened.background_sources(A) if s["role"] == "assistant"]) == 1
    with pytest.raises(VersionConflict):
        reopened.dispatch_segment(run, 1, now_us=101)
    with pytest.raises(ValueError):
        reopened.finish_stream(run, outcome="completed", now_us=100)


def test_confirmation_queue_failure_rolls_back_receipt_and_history(tmp_path):
    store, run, event = setup(tmp_path)
    with store.transaction() as db:
        db.execute("CREATE TRIGGER fail_source BEFORE INSERT ON background_sources "
                   "WHEN NEW.role='assistant' BEGIN SELECT RAISE(ABORT,'fault'); END")
    with pytest.raises(sqlite3.DatabaseError):
        confirm(store, run, event)
    assert store.segments(A, run.input_id)[0]["status"] == "unknown"
    with store.transaction(write=False) as db:
        assert db.execute("SELECT count(*) FROM stream_history").fetchone()[0] == 0


def test_completed_requires_all_confirmed_and_cancelled_cannot_complete(tmp_path):
    store, run, event = setup(tmp_path)
    with pytest.raises(ValueError):
        store.finish_stream(run, outcome="completed", now_us=100)
    confirm(store, run, event)
    store.cancel(A, run.input_id)
    with pytest.raises(VersionConflict):
        store.finish_stream(run, outcome="completed", now_us=100)
    assert store.finish_stream(run, outcome="cancelled", now_us=100)["outcome"] == "cancelled"


def test_background_context_stops_at_exact_segment_and_never_reads_future_unit(tmp_path):
    store, run, first = setup(tmp_path)
    confirm(store, run, first)
    store.approve_segment(run, reviewed(run, ("我在这里。", "你想聊些什么？")), now_us=100)
    second = store.dispatch_segment(run, 2, now_us=100)
    store.confirm_segment(A, run.input_id, 2, fingerprint=second["fingerprint"],
                          host_receipt="second-visible", channel="text_visible")
    assert store.finish_stream(run, outcome="completed", now_us=100)["confirmed_at_settlement"] == 2
    work = store.claim_background(A, "background", now_us=200, lease_us=50, role="assistant")
    context = store.background_exchange_context(work)
    assert [s.text for s in context] == ["你好", "我在这里。"]
    jobs = [s for s in store.background_sources(A) if s["role"] == "assistant"]
    assert len(jobs) == 2 and len({s["source"] for s in jobs}) == 2


def test_terminal_fault_leaves_run_and_receipts_unchanged(tmp_path):
    store, run, event = setup(tmp_path)
    confirm(store, run, event)
    with store.transaction() as db:
        db.execute("CREATE TRIGGER fail_terminal BEFORE INSERT ON stream_terminals "
                   "BEGIN SELECT RAISE(ABORT,'fault'); END")
    with pytest.raises(sqlite3.DatabaseError):
        store.finish_stream(run, outcome="failed", now_us=100)
    assert store.inspect(A, run.input_id)["status"] == "running"
    assert store.segments(A, run.input_id)[0]["status"] == "confirmed"


def test_expired_stream_attempt_cannot_be_replaced_before_reconciliation(tmp_path):
    store, run, _ = setup(tmp_path)
    with pytest.raises(VersionConflict, match="reconciliation"):
        claim(store, store.accept(config(), "first", "你好"), worker="replacement", now=200)
