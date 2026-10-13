"""Deterministic A3-REVIEW-002 scheduling through actual ControlledCalls."""

import json
import threading

import pytest
from virtual_companion.cognition.kernel import ControlledCalls
from virtual_companion.cognition.message_assembly import StageMessageModel


class CancelledMarker(Exception):
    pass


@pytest.mark.parametrize("window", ["entry", "assembly", "capacity", "ready", "stream"])
def test_cancel_lifecycle_releases_owner_and_records_one_terminal(window):
    original_window = window
    entered, interrupted = threading.Event(), threading.Event()
    cause = CancelledMarker("original durable cause")
    events = []
    request = {"stage": "reply", "context_sha256": "a" * 64}
    messages = [{"role": "user", "content": json.dumps(request)}]

    def pause():
        entered.set()
        assert interrupted.wait(2), "cancellation did not finish"

    class Control:
        def checkpoint(self):
            if entered.is_set():
                raise cause

    class Transport:
        def check_context(self, messages):
            if window == "capacity":
                pause()

        def stream(self, messages, cancelled):
            if window == "stream":
                pause()
                raise OSError("socket interrupted")
            yield '{"ok":true}'

        def cancel(self):
            pass

    class Scheduled(StageMessageModel):
        def check_context(self, messages):
            if window == "entry":
                pause()
            super().check_context(messages)
            if window == "ready":
                pause()

        def _assemble(self, messages):
            result = super()._assemble(messages)
            if window == "assembly":
                pause()
            return result

        def cancel(self):
            super().cancel()
            interrupted.set()

    model = Scheduled(Transport(), systems={"reply": "synthetic"}, observe=events.append)
    with pytest.raises(CancelledMarker) as caught:
        ControlledCalls(model, Control()).request(request)
    assert caught.value is cause
    # Disable scheduling and prove an independent call on the same instance works.
    window = None
    model.check_context(messages)
    assert "".join(model.stream(messages, threading.Event())) == '{"ok":true}'
    terminals = [e for e in events if e["status"] in {"completed", "cancelled", "failed"}]
    assert [(e["call_id"], e["status"]) for e in terminals] == [
        (1, "cancelled"), (2, "completed")
    ]
    if original_window == "stream":
        assert terminals[0]["error_type"] == "OSError"


def test_ready_cancel_emits_once_even_when_cancel_repeated():
    events = []

    class Transport:
        def check_context(self, messages):
            pass

        def cancel(self):
            pass

    model = StageMessageModel(Transport(), systems={"reply": "synthetic"}, observe=events.append)
    model.check_context([{"role": "user", "content": '{"stage":"reply"}'}])
    model.cancel()
    model.cancel()
    assert [e["status"] for e in events] == ["checking_capacity", "cancelled"]


def test_stream_release_does_not_bypass_inflight_cancel_guard():
    entered, release = threading.Event(), threading.Event()
    events = []

    class Transport:
        def check_context(self, messages):
            pass

        def stream(self, messages, cancelled):
            yield "first"

        def cancel(self):
            entered.set()
            assert release.wait(2)

    model = StageMessageModel(Transport(), systems={"reply": "synthetic"}, observe=events.append)
    messages = [{"role": "user", "content": '{"stage":"reply"}'}]
    model.check_context(messages)
    stream = model.stream(messages, threading.Event())
    assert next(stream) == "first"
    worker = threading.Thread(target=model.cancel)
    worker.start()
    try:
        assert entered.wait(2)
        stream.close()
        with pytest.raises(RuntimeError, match="owns"):
            model.check_context(messages)
    finally:
        release.set()
        worker.join(2)
    assert not worker.is_alive()
    model.check_context(messages)
    assert list(model.stream(messages, threading.Event())) == ["first"]
    assert [e["status"] for e in events] == [
        "checking_capacity", "cancelled", "checking_capacity", "completed"
    ]


def test_cancel_signal_alone_preserves_stream_exception():
    cancelled, events = threading.Event(), []

    class Transport:
        def check_context(self, messages):
            pass

        def stream(self, messages, signal):
            signal.set()
            raise OSError("interrupted")
            yield

    model = StageMessageModel(Transport(), systems={"reply": "synthetic"}, observe=events.append)
    messages = [{"role": "user", "content": '{"stage":"reply"}'}]
    model.check_context(messages)
    with pytest.raises(OSError):
        list(model.stream(messages, cancelled))
    assert events[-1]["status"] == "cancelled"
    assert events[-1]["error_type"] == "OSError"
