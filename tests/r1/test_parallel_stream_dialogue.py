from threading import Event

import pytest
from test_rebuilt_turns import A, claim, config
from test_segment_store import reviewed
from virtual_companion.boundary.sqlite_streaming import SQLiteStreamingStore
from virtual_companion.cognition.short_dialogue import ShortRequest
from virtual_companion.cognition.stream_dialogue import StreamDeliveryUnknown, StreamDialogue
from virtual_companion.providers.text_deltas import UnapprovedTextDelta


def setup(tmp_path, *, missing_receipt=False):
    store = SQLiteStreamingStore(tmp_path / "parallel.sqlite3")
    run = claim(store, store.accept(config(), "first", "你好"))
    displayed, interrupted = Event(), Event()
    seen = []

    class Model:
        def count(self, body):
            return 10

        def invoke(self, body):
            self.receive(UnapprovedTextDelta(1, "我在这里。\n\n"))
            if missing_receipt:
                assert interrupted.wait(2), "Missing receipt did not stop generation"
                raise InterruptedError("model cancelled")
            assert displayed.wait(2), "First unit was not visible before model terminal"
            self.receive(UnapprovedTextDelta(2, "你想聊什么？"))
            return {"text": "我在这里。\n\n你想聊什么？",
                    "tool_calls": [], "finish_reason": "stop"}

        def cancel(self):
            interrupted.set()

    model = Model()

    def factory(receive):
        model.receive = receive
        return model

    def review(text, final):
        return reviewed(run, ("我在这里。", "你想聊什么？")) if final else (
            reviewed(run) if text.startswith("我在这里。\n\n") else None
        )

    def present(event):
        seen.append(event)
        displayed.set()
        return None if missing_receipt else f"device-{len(seen)}"

    dialogue = StreamDialogue(store=store, model_factory=factory,
        review_prefix=review, present=present, checkpoint=lambda: None,
        cancelled=Event(), clock=lambda: 100, parallel_review=True)
    return store, run, dialogue, seen, model, displayed


def test_parallel_review_can_confirm_complete_unit_before_model_terminal(tmp_path):
    store, run, dialogue, seen, _, _ = setup(tmp_path)
    assert dialogue.generate(run, ShortRequest("你好"))["outcome"] == "completed"
    assert [event["text"] for event in seen] == ["我在这里。", "你想聊什么？"]
    assert store.stream_state(A, run.input_id)["confirmed_units"] == 2


def test_parallel_review_unknown_receipt_cancels_model_and_never_replays(tmp_path):
    store, run, dialogue, seen, _, _ = setup(tmp_path, missing_receipt=True)
    with pytest.raises(StreamDeliveryUnknown):
        dialogue.generate(run, ShortRequest("你好"))
    state = store.stream_state(A, run.input_id)
    assert state["unknown_units"] == 1 and state["confirmed_units"] == 0
    # The coordinator stops the producer because delivery failed. That local
    # abort is not a user cancellation and must keep the failed terminal.
    assert not dialogue.cancelled.is_set()
    assert state["outcome"] == "failed"
    with pytest.raises(StreamDeliveryUnknown):
        dialogue.generate(run, ShortRequest("你好"))
    assert len(seen) == 1


def test_parallel_model_failure_keeps_only_actually_confirmed_prefix(tmp_path):
    store, run, dialogue, seen, model, displayed = setup(tmp_path)

    def invoke(body):
        model.receive(UnapprovedTextDelta(1, "我在这里。\n\n"))
        assert displayed.wait(2)
        raise RuntimeError("synthetic model failure after first display")

    model.invoke = invoke
    with pytest.raises(RuntimeError, match="synthetic model failure"):
        dialogue.generate(run, ShortRequest("你好"))
    state = store.stream_state(A, run.input_id)
    assert state["outcome"] == "failed"
    assert state["confirmed_units"] == 1
    assert state["confirmed_text"] == "我在这里。"
    assert len(seen) == 1


def test_parallel_cancel_during_display_keeps_receipt_and_blocks_tail(tmp_path):
    store, run, dialogue, seen, _, _ = setup(tmp_path)
    original = dialogue.present

    def present(event):
        receipt = original(event)
        dialogue.cancelled.set()
        return receipt

    dialogue.present = present
    with pytest.raises(InterruptedError):
        dialogue.generate(run, ShortRequest("你好"))
    state = store.stream_state(A, run.input_id)
    assert state["outcome"] == "cancelled"
    assert state["confirmed_units"] == 1
    assert len(seen) == 1
