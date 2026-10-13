import json

import pytest
from test_exchange_maintenance import prepared
from test_rebuilt_turns import A, config
from virtual_companion.boundary.contracts import VersionConflict
from virtual_companion.cognition.maintenance import ExchangeMaintenance


class Invoker:
    def __init__(self, after_extract=None, malformed=False):
        self.calls = []
        self.after_extract, self.malformed = after_extract, malformed

    def bind_turn(self, claim, guardian):
        self.claim, self.guardian = claim, guardian

    def cancel(self):
        pass

    def __call__(self, stage, messages, schema, cancelled):
        self.guardian.checkpoint()
        self.calls.append((stage, messages))
        data = json.loads(messages[-1]["content"])
        if stage == "exchange_support":
            return {"intent": "question", "verdict": "supported"}
        if self.after_extract:
            self.after_extract()
        if self.malformed:
            return {"operations": "invalid"}
        source = data["sources"][0]
        return {
            "operations": [
                {
                    "operation": "add",
                    "type": "open_loop",
                    "evidenceQuote": source["text"],
                    "evidenceTurnId": source["id"],
                    "supersedesAtomId": None,
                    "expiresAt": None,
                }
            ]
        }


def runtime(store, model):
    return ExchangeMaintenance(store, model, clock=lambda: 200, evidence_clock=lambda: 100)


def test_model_maintenance_uses_actual_ack_and_reconciles_without_recalling_model(tmp_path):
    store, command, item = prepared(tmp_path)
    invoke = Invoker()
    app = runtime(store, invoke)
    receipt = app.execute(A, item.input_id, "maintenance", deadline_us=5_000_200)
    assert app.execute(A, item.input_id, "maintenance", deadline_us=5_000_200) == receipt
    assert [c[0] for c in invoke.calls] == ["exchange_extract", "exchange_support"]
    assert "也可以先休息" not in str(invoke.calls)
    assert store.pending_exchanges(A, now_us=100)[0]["content"] == command.content


def test_new_user_during_extraction_cancels_before_support_and_commit(tmp_path):
    store, _, item = prepared(tmp_path)
    invoke = Invoker(lambda: store.accept(config(), "interrupt", "先停一下"))
    receipt = runtime(store, invoke).execute(A, item.input_id, "maintenance", deadline_us=5_000_200)
    assert receipt["status"] == "cancelled"
    assert len(invoke.calls) == 1
    assert store.pending_exchanges(A, now_us=100) == []


def test_invalid_model_output_is_failed_not_silent_success(tmp_path):
    store, _, item = prepared(tmp_path)
    receipt = runtime(store, Invoker(malformed=True)).execute(
        A, item.input_id, "maintenance", deadline_us=5_000_200
    )
    assert receipt["status"] == "failed"
    assert receipt["reason"] == "exchange_model_contract"
    assert store.pending_exchanges(A, now_us=100) == []


def test_user_input_cannot_be_executed_as_maintenance(tmp_path):
    store, _, _ = prepared(tmp_path)
    item = store.accept(config(), "normal", "你好")
    model = Invoker()
    with pytest.raises(ValueError):
        runtime(store, model).execute(A, item.input_id, "maintenance", deadline_us=5_000_200)
    assert model.calls == []


def test_stale_worker_does_not_cancel_new_owner(tmp_path, monkeypatch):
    store, _, item = prepared(tmp_path)
    successors = []
    failures = []
    original_fail = store.fail

    def fail(*args, **kwargs):
        failures.append(kwargs["reason"])
        return original_fail(*args, **kwargs)

    monkeypatch.setattr(store, "fail", fail)

    def takeover():
        successors.append(
            store.claim(A, item.input_id, "successor", now_us=5_000_201, lease_us=5_000_000)
        )

    model = Invoker(takeover)
    with pytest.raises(VersionConflict):
        runtime(store, model).execute(A, item.input_id, "old", deadline_us=5_000_200)
    assert store.inspect(A, item.input_id)["status"] == "running"
    assert store.owner(A) == successors[0].owner
    assert store.pending_exchanges(A, now_us=100) == []
    assert failures == []


def test_lost_commit_response_reconciles_without_repeating_model(tmp_path, monkeypatch):
    store, _, item = prepared(tmp_path)
    finish = store.finish_maintenance

    def lose_response(*args, **kwargs):
        finish(*args, **kwargs)
        raise OSError("synthetic lost response")

    monkeypatch.setattr(store, "finish_maintenance", lose_response)
    model = Invoker()
    app = runtime(store, model)
    with pytest.raises(OSError):
        app.execute(A, item.input_id, "worker", deadline_us=5_000_200)
    assert app.execute(A, item.input_id, "worker", deadline_us=5_000_200)["effects"]
    assert len(model.calls) == 2
