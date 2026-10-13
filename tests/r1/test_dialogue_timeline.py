import pytest
from test_pending_exchanges import question
from test_rebuilt_presentation import acknowledge
from test_rebuilt_turns import A, claim, config
from virtual_companion.boundary.contracts import Scope
from virtual_companion.boundary.sqlite_timeline import SQLiteTimelineStore
from virtual_companion.cognition.exchange_planning import ExchangePlanner


def test_actual_input_and_terminal_prefix_times_survive_reopen(tmp_path):
    now = [100]
    store = SQLiteTimelineStore(tmp_path / "timeline.sqlite3", evidence_clock=lambda: now[0])
    item = store.accept(config(), "first", "你好")
    scope = Scope(A, item.session_id)
    now[0] = 200
    command = question(store, item=item, displayed=7)
    assert store.source_time(scope, "user", item.input_id) == 100
    assert store.source_time(scope, "assistant", command.evidence.source_id) == 200
    reopened = SQLiteTimelineStore(store.path, evidence_clock=lambda: 999)
    assert reopened.accept(config(), "first", "你好") == item
    assert reopened.source_time(scope, "user", item.input_id) == 100
    assert reopened.source_time(Scope(A, "other"), "user", item.input_id) is None


def test_unconfirmed_reply_has_no_display_time(tmp_path):
    store = SQLiteTimelineStore(tmp_path / "timeline.sqlite3", evidence_clock=lambda: 100)
    command = question(store)
    assert store.source_time(command.scope, "assistant", command.evidence.source_id) is None


def test_invalid_calendar_clock_rolls_back_input(tmp_path):
    store = SQLiteTimelineStore(tmp_path / "timeline.sqlite3", evidence_clock=lambda: True)
    with pytest.raises(ValueError):
        store.accept(config(), "first", "你好")
    with store.transaction(write=False) as db:
        assert db.execute("SELECT count(*) FROM turn_inputs").fetchone()[0] == 0


def test_internal_maintenance_is_not_a_user_timestamp(tmp_path):
    store = SQLiteTimelineStore(
        tmp_path / "timeline.sqlite3",
        evidence_clock=lambda: 100,
        automatic_maintenance=True,
    )
    command = question(store, displayed=7)
    internal = store.queued_maintenance(A)[0]
    assert store.source_time(command.scope, "user", internal) is None
    item = store.accept(config(), "next", "你好", session_id=command.scope.session_id)
    run = claim(store, item, now=200)
    assert store.exchange_context(run)[0][-1].source_id == item.input_id
    history = ExchangePlanner(store, None, evidence_clock=lambda: 100).history(run)
    assert [h.occurred_at for h in history] == [
        "1970-01-01T00:00:00.000100+00:00",
        "1970-01-01T00:00:00.000100+00:00",
    ]


def test_ack_time_failure_rolls_back_display_and_maintenance_then_retry_is_stable(tmp_path):
    now = [100]
    store = SQLiteTimelineStore(
        tmp_path / "timeline.sqlite3",
        evidence_clock=lambda: now[0],
        automatic_maintenance=True,
    )
    item = store.accept(config(), "first", "你好")
    reply = "晚上想聊什么？"
    command = question(store, item=item, text=reply)
    receipt = store.inspect(A, item.input_id)["commit"]
    now[0] = True
    with pytest.raises(ValueError):
        acknowledge(store, receipt, displayed_end=len(reply))
    assert store.source_time(command.scope, "assistant", command.evidence.source_id) is None
    assert not store.queued_maintenance(A)
    now[0] = 200
    done = acknowledge(store, receipt, displayed_end=len(reply))
    now[0] = 300
    assert acknowledge(store, receipt, displayed_end=len(reply)) == done
    assert store.source_time(command.scope, "assistant", command.evidence.source_id) == 200
    assert len(store.queued_maintenance(A)) == 1
