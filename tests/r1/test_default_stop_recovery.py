import threading

import pytest
from test_default_runtime_recovery import completed
from test_default_text_delivery import Model, setup, wait_for


@pytest.mark.parametrize("prior_completed", [False, True])
def test_stop_targets_latest_accepted_input_when_claim_failed(
    tmp_path, monkeypatch, prior_completed,
):
    runtime, text, memory, mirror, _ = setup(tmp_path)
    old_input = old_state = None
    if prior_completed:
        text.send("已完成的合成前轮")
        completed(runtime, text)
        old_input = runtime.claim.input_id
        old_state = memory.store.stream_state(memory.companion, old_input)
    claim = memory.store.claim

    def unavailable(*args, **kwargs):
        raise OSError("synthetic claim failure after durable acceptance")

    monkeypatch.setattr(memory.store, "claim", unavailable)
    current = text.send("接纳成功但claim失败的当前输入")
    wait_for(lambda: not text.snapshot()["busy"])
    monkeypatch.setattr(memory.store, "claim", claim)
    with memory.store.transaction(write=False) as db:
        input_id = db.execute("SELECT input FROM turn_inputs WHERE companion=? AND client_key=?",
                              (memory.companion, current["turn_id"])).fetchone()[0]
    assert memory.store.inspect(memory.companion, input_id)["status"] == "queued"
    runtime.stop()
    text.stop()
    assert memory.store.inspect(memory.companion, input_id)["status"] == "cancelled"
    assert memory.store.segments(memory.companion, input_id) == []
    runtime.before_start(False)
    if old_input is not None:
        assert memory.store.stream_state(memory.companion, old_input) == old_state
    mirror.close()


def test_stop_still_cancels_bound_active_generation(tmp_path):
    class Blocking(Model):
        def __init__(self):
            super().__init__()
            self.entered, self.release = threading.Event(), threading.Event()

        def stream(self, messages, cancelled):
            self.entered.set()
            assert self.release.wait(3)
            if not cancelled.is_set():
                yield from super().stream(messages, cancelled)

        def cancel(self):
            self.release.set()

    model = Blocking()
    runtime, text, memory, mirror, _ = setup(tmp_path, model)
    text.send("合成执行中输入")
    assert model.entered.wait(2)
    input_id = runtime.claim.input_id
    runtime.stop()
    text.stop()
    wait_for(lambda: not text.snapshot()["busy"])
    assert text.snapshot()["attempt"]["status"] == "cancelled"
    assert memory.store.inspect(memory.companion, input_id)["status"] == "cancelled"
    assert memory.store.segments(memory.companion, input_id) == []
    runtime.before_start(False)
    mirror.close()


def test_stop_keeps_dispatched_unknown_receipt_reconcilable(tmp_path):
    runtime, text, memory, mirror, _ = setup(tmp_path)
    text.send("合成已派发输入")
    event = wait_for(lambda: runtime.poll()["event"])
    runtime.stop()
    text.stop()
    wait_for(lambda: not text.snapshot()["busy"])
    assert memory.store.stream_state(memory.companion, event["basis"]["input_id"])[
        "confirmed_units"
    ] == 0
    assert runtime.state()["reconcile"][0]["fingerprint"] == event["fingerprint"]
    with pytest.raises(ValueError):
        runtime.before_start(False)
    runtime.ack({"sequence": event["sequence"], "fingerprint": event["fingerprint"],
                 "host_receipt": "synthetic-late-stop-receipt"})
    assert mirror.history()["messages"][-1]["content"] == "这是合成答案。"
    runtime.before_start(False)
    mirror.close()
