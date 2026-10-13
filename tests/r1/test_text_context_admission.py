from __future__ import annotations

import pytest
from virtual_companion.text.application import TextSession
from virtual_companion.text.errors import ContextCheckError, ContextLimitError
from virtual_companion.text.fake import FakeModel
from virtual_companion.text.llama import LlamaModel
from virtual_companion.text.memory import MemoryConversationStore


def test_rejected_context_does_not_write_user_or_attempt():
    store = MemoryConversationStore()
    seen = []

    def check(messages):
        seen.append(messages)
        raise ValueError("context full")

    session = TextSession(FakeModel(), store, {}, preflight=check)
    with pytest.raises(ValueError, match="context full"):
        session.send("  keep my draft  ")
    assert seen == [[{"role": "user", "content": "keep my draft"}]]
    assert store.history()["messages"] == []
    assert store.latest_attempt() is None
    assert not session.snapshot()["busy"]


def test_retry_admission_includes_full_history_without_duplicate_user():
    store = MemoryConversationStore()
    prior = store.begin("first", {})
    store.finish(prior["attempt_id"], "completed", "reply")
    stopped = store.begin("second", {})
    store.finish(stopped["attempt_id"], "cancelled", "partial")
    original = store.history()

    def check(messages):
        assert messages == [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
        ]
        raise OSError("preflight unavailable")

    session = TextSession(FakeModel(), store, {}, preflight=check)
    with pytest.raises(OSError):
        session.retry()
    assert store.history() == original
    assert store.latest_attempt()["attempt_id"] == stopped["attempt_id"]


@pytest.mark.parametrize("count,accepted", [(3903, True), (3904, False), (8000, False)])
def test_exact_template_tokens_reserve_output_capacity(monkeypatch, count, accepted):
    model = LlamaModel(
        {"server_arguments": ["--ctx-size", "4096"], "sampling": {"max_tokens": 192}}, None
    )
    messages = [{"role": "user", "content": "中文 and English"}]
    payload = {"messages": messages, "chat_template_kwargs": {"enable_thinking": False}}
    monkeypatch.setattr(model, "_payload", lambda actual: payload if actual == messages else None)
    calls = []

    def request(route, body):
        calls.append((route, body))
        return (
            {"prompt": "formatted chat"} if route == "/apply-template" else {"tokens": [1] * count}
        )

    monkeypatch.setattr(model, "_context_request", request)
    if accepted:
        model.check_context(messages)
    else:
        with pytest.raises(ContextLimitError):
            model.check_context(messages)
    assert calls == [
        ("/apply-template", payload),
        (
            "/tokenize",
            {
                "content": "formatted chat",
                "add_special": True,
                "parse_special": True,
            },
        ),
    ]


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"prompt": []},
        {"prompt": "x", "tokens": [True]},
        {"prompt": "x", "tokens": []},
    ],
)
def test_invalid_counter_response_fails_closed_without_exposing_details(monkeypatch, response):
    model = LlamaModel({}, None)
    monkeypatch.setattr(model, "_payload", lambda _: {})
    monkeypatch.setattr(model, "_context_request", lambda *_: response)
    with pytest.raises(ContextCheckError, match="尚未发送"):
        model.check_context([])


def test_transport_exposes_only_safe_admission_error():
    import http.client
    import json
    import threading

    from virtual_companion.text.transport import create_server

    def dispatch(*_):
        raise ContextLimitError()

    server = create_server("a" * 48, dispatch, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
    try:
        connection.request(
            "POST",
            "/v1/send",
            json.dumps({"text": "hello"}),
            {
                "Authorization": "Bearer " + "a" * 48,
                "Content-Type": "application/json",
            },
        )
        response = connection.getresponse()
        assert response.status == 409
        assert json.loads(response.read())["error"] == str(ContextLimitError())
    finally:
        connection.close()
        server.shutdown()
        server.server_close()
        thread.join(2)


def test_prepared_llama_request_freezes_wire_payload_not_only_user_messages(tmp_path, monkeypatch):
    from copy import deepcopy
    from types import SimpleNamespace

    from test_exact_llama_messages import setup

    profile, messages = setup(tmp_path)
    model = LlamaModel(profile, tmp_path)
    model.process = SimpleNamespace(poll=lambda: None)
    counted, sent = [], []

    def context(route, payload):
        if route == "/apply-template":
            counted.append(deepcopy(payload))
            return {"prompt": "frozen"}
        return {"tokens": [1]}

    def stream(payload, cancelled):
        sent.append(deepcopy(payload))
        yield "answer"

    monkeypatch.setattr(model, "_context_request", context)
    monkeypatch.setattr(model, "_stream_payload", stream, raising=False)
    request = model.prepare_request(messages)
    request.check_context()
    messages[-1]["content"] = "changed caller"
    profile["sampling"]["seed"] = 99
    profile["sampling"]["max_tokens"] = 3000
    (tmp_path / "prompt.txt").write_text("changed prompt", encoding="utf-8")
    import threading

    assert list(request.stream(threading.Event())) == ["answer"]
    assert sent == counted
    assert sent[0]["seed"] == 1
    with pytest.raises(ValueError):
        list(request.stream(threading.Event()))


@pytest.mark.parametrize("fault", [
    "no_preflight", "process_replaced", "cancelled", "process_replaced_during_stream",
])
def test_prepared_llama_request_blocks_unchecked_or_replaced_process(tmp_path, monkeypatch, fault):
    from threading import Event
    from types import SimpleNamespace

    from test_exact_llama_messages import setup

    profile, messages = setup(tmp_path)
    model = LlamaModel(profile, tmp_path)
    model.process = SimpleNamespace(poll=lambda: None)
    monkeypatch.setattr(model, "_context_request", lambda route, payload:
                        {"prompt": "x"} if route == "/apply-template" else {"tokens": [1]})
    sent = []
    def stream(*args):
        sent.append(1)
        if fault == "process_replaced_during_stream":
            model.process = SimpleNamespace(poll=lambda: None)
        yield "x"

    monkeypatch.setattr(model, "_stream_payload", stream, raising=False)
    request = model.prepare_request(messages)
    if fault != "no_preflight":
        request.check_context()
    cancelled = Event()
    if fault == "process_replaced":
        model.process = SimpleNamespace(poll=lambda: None)
    if fault == "cancelled":
        cancelled.set()
        assert list(request.stream(cancelled)) == []
    else:
        with pytest.raises((ValueError, RuntimeError)):
            list(request.stream(cancelled))
    assert sent == ([1] if fault == "process_replaced_during_stream" else [])
