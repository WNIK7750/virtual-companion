import importlib
import json
import threading

import pytest


def messages(stage="reply", text="原文<think>保留</think>"):
    return [
        {
            "role": "user",
            "content": json.dumps({"stage": stage, "context_sha256": "a" * 64, "text": text}),
        }
    ]


class Transport:
    def __init__(self):
        self.checked, self.sent, self.cancelled = [], [], 0

    def check_context(self, value):
        self.checked.append(value)

    def stream(self, value, cancelled):
        self.sent.append(value)
        yield '{"text":"<think>保留</think>"}'

    def cancel(self):
        self.cancelled += 1


def bridge(transport=None, events=None):
    cls = importlib.import_module("virtual_companion.cognition.message_assembly").StageMessageModel
    systems = {"reply": "共同人格与回复协议"}
    model = cls(
        transport or Transport(),
        systems=systems,
        observe=(events.append if events is not None else None),
    )
    return model, systems


def test_final_messages_bound_to_capacity_and_raw_response():
    events = []
    model, systems = bridge(events=events)
    original = messages()
    model.check_context(original)
    systems["reply"] = "mutated outside"
    response = "".join(model.stream(original, threading.Event()))
    assert response == '{"text":"<think>保留</think>"}'
    assert model.transport.checked == model.transport.sent
    assert model.transport.sent[0] == [
        {"role": "system", "content": "共同人格与回复协议"},
        *original,
    ]
    assert events[-1]["status"] == "completed"
    assert events[-1]["response_content"] == response
    assert events[-1]["request_sha256"] == events[0]["request_sha256"]


def test_swapped_request_is_rejected_before_generation():
    model, _ = bridge()
    model.check_context(messages())
    with pytest.raises(ValueError):
        list(model.stream(messages(text="changed"), threading.Event()))
    assert not model.transport.sent


def test_generation_requires_capacity_check_and_unknown_stage_is_rejected():
    model, _ = bridge()
    with pytest.raises(ValueError):
        list(model.stream(messages(), threading.Event()))
    with pytest.raises(ValueError):
        model.check_context(messages(stage="unknown"))
    assert not model.transport.checked


def test_cancelled_prepared_call_cannot_be_sent_but_next_call_can():
    model, _ = bridge()
    model.check_context(messages())
    model.cancel()
    with pytest.raises(ValueError):
        list(model.stream(messages(), threading.Event()))
    model.check_context(messages())
    assert list(model.stream(messages(), threading.Event()))
    assert model.transport.cancelled == 1


def test_old_cancel_cannot_interrupt_a_new_call():
    entered, release = threading.Event(), threading.Event()
    transport = Transport()

    def cancel():
        entered.set()
        assert release.wait(2)

    transport.cancel = cancel
    model, _ = bridge(transport)
    model.check_context(messages())
    thread = threading.Thread(target=model.cancel)
    thread.start()
    try:
        assert entered.wait(2)
        with pytest.raises(RuntimeError):
            model.check_context(messages())
    finally:
        release.set()
        thread.join(2)
    assert not thread.is_alive()
    model.check_context(messages())
    assert list(model.stream(messages(), threading.Event()))


def test_generator_close_releases_underlying_stream_and_owner():
    closed = []
    transport = Transport()

    def stream(value, cancelled):
        try:
            yield "first"
            yield "second"
        finally:
            closed.append(True)

    transport.stream = stream
    model, _ = bridge(transport)
    model.check_context(messages())
    iterator = model.stream(messages(), threading.Event())
    assert next(iterator) == "first"
    with pytest.raises(RuntimeError):
        model.check_context(messages())
    iterator.close()
    assert closed == [True]
    model.check_context(messages())
    assert list(model.stream(messages(), threading.Event())) == ["first", "second"]


@pytest.mark.parametrize("response", [123, "x" * 64001], ids=["nontext", "oversized"])
def test_invalid_response_is_a_model_contract_failure(response):
    from virtual_companion.boundary.runtime import KernelFailure

    transport = Transport()
    transport.stream = lambda *_: iter([response])
    events = []
    model, _ = bridge(transport, events)
    model.check_context(messages())
    with pytest.raises(KernelFailure, match="model_contract"):
        list(model.stream(messages(), threading.Event()))
    assert events[-1]["status"] == "failed"
    model.check_context(messages())


def test_actual_kernel_rewrite_uses_assembled_and_bound_requests(tmp_path):
    from test_cognition_preparation import basis
    from test_cognitive_kernel import RecordedModel, kernel
    from test_rebuilt_turns import A

    cls = importlib.import_module("virtual_companion.cognition.message_assembly").StageMessageModel
    store, _, run, persona = basis(tmp_path)
    recorded, events = RecordedModel(rejects=1), []
    model = cls(
        recorded,
        systems={s: "synthetic stage protocol " + s for s in ("prepare", "reply", "review")},
        observe=events.append,
    )
    candidate = kernel(store, persona, model)
    result = candidate.generate(run, store.snapshot(A), "合成输入")
    assert result.revision == 2
    assert recorded.requests == recorded.checked
    done = [event for event in events if event["status"] == "completed"]
    assert [event["stage"] for event in done] == ["prepare", "reply", "review", "reply", "review"]
    assert done[-1]["candidate_sha256"] == result.presentation.candidate.fingerprint
    assert json.loads(done[-2]["messages"][-1]["content"])["feedback"] is not None
    assert all(len(event["messages"]) == 2 for event in done)


def test_capacity_failure_releases_call_ownership():
    transport = Transport()
    normal = transport.check_context
    transport.check_context = lambda _: (_ for _ in ()).throw(TimeoutError())
    events = []
    model, _ = bridge(transport, events)
    with pytest.raises(TimeoutError):
        model.check_context(messages())
    assert events[-1]["status"] == "failed"
    assert events[-1]["error_type"] == "TimeoutError"
    transport.check_context = normal
    model.check_context(messages())
    with pytest.raises(RuntimeError):
        model.check_context(messages())
    assert list(model.stream(messages(), threading.Event()))
