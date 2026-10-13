"""Current recorded calls preserve owner, journal, budget and cancellation contracts."""

import json
import sqlite3
import time

import pytest
from provider_recording import RecordedProviderModels
from test_rebuilt_turns import A, config
from virtual_companion.boundary.sqlite_exchanges import SQLiteExchangeStore
from virtual_companion.cognition.preparation import CallCancelled, CallControl
from virtual_companion.cognition.structured_harness import StructuredHarnessInvoker
from virtual_companion.cognition.turn_guardian import TurnGuardian


@pytest.mark.parametrize("mode", ["completed", "failed", "cancelled"])
def test_current_structured_call_keeps_budget_journal_and_no_presentation_claim(tmp_path, mode):
    clock = lambda: int(time.monotonic() * 1_000_000)  # noqa: E731
    store = SQLiteExchangeStore(tmp_path / "current.sqlite3")
    item = store.accept(config(), "current", "合成当前输入")
    run = store.claim(A, item.input_id, "worker", now_us=clock(), lease_us=5_000_000)

    class Source:
        calls = 0

        def cancel(self):
            pass

        def __call__(self, stage, messages, schema, cancelled):
            self.calls += 1
            assert stage == "current_projection"
            if mode == "failed":
                raise ValueError("recorded provider failure")
            return {"accepted": True}

    source = Source()
    models = RecordedProviderModels(source, tmp_path / "budget.sqlite3")
    invoke = StructuredHarnessInvoker(models, store, clock)
    control = CallControl(
        run, clock=clock,
        cancelled=lambda: store.inspect(A, item.input_id)["status"] != "running",
        renew=lambda owner, now: store.renew(owner, now_us=now, lease_us=5_000_000),
        deadline_us=clock() + 10_000_000, renewal_margin_us=1_000_000,
    )
    guard = TurnGuardian(control, invoke)
    schema = {"type": "object", "required": ["accepted"],
              "properties": {"accepted": {"type": "boolean"}}, "additionalProperties": False}
    def execute():
        with guard:
            invoke.bind_turn(run, guard)
            if mode == "cancelled":
                store.cancel(A, item.input_id)
                invoke("current_projection", [{"role": "user", "content": "synthetic"}],
                       schema, guard.cancelled)
            elif mode == "failed":
                invoke("current_projection", [{"role": "user", "content": "synthetic"}],
                       schema, guard.cancelled)
            else:
                assert invoke(
                    "current_projection", [{"role": "user", "content": "synthetic"}],
                    schema, guard.cancelled,
                ) == {"accepted": True}
    if mode == "cancelled":
        with pytest.raises(CallCancelled):
            execute()
    elif mode == "failed":
        with pytest.raises(ValueError, match="recorded provider failure"):
            execute()
    else:
        execute()
    records = models.ledger.records()
    assert len(records) == source.calls == (0 if mode == "cancelled" else 1)
    assert all(row["transmission"] == "not_sent" for row in records)
    with sqlite3.connect(store.path) as db:
        checkpoints = db.execute("SELECT payload FROM harness_checkpoints").fetchall()
        assert bool(checkpoints) is (mode != "cancelled")
        for (payload,) in checkpoints:
            assert json.loads(payload)["binding"]["attempt"] == run.attempt_id
        assert db.execute("SELECT count(*) FROM turn_commits").fetchone()[0] == 0
        assert db.execute("SELECT count(*) FROM turn_deliveries").fetchone()[0] == 0
