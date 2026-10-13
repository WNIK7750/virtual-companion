import json
import threading
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator
from test_exact_llama_messages import setup
from virtual_companion.text import llama
from virtual_companion.text.errors import ContextLimitError
from virtual_companion.text.llama import LlamaModel
from virtual_companion.text.memory_runtime import _PriorityGenerationGate, _StructuredMemoryInvoker


def stage_model(tmp_path):
    profile, messages = setup(tmp_path)
    model = LlamaModel(profile, tmp_path)
    model.process = SimpleNamespace(poll=lambda: None)
    return model, messages


@pytest.mark.parametrize("fault", [
    "missing_system", "extra_system", "empty_system", "extra_field", "tool", "last_assistant",
])
def test_stage_request_rejects_unsupported_messages(tmp_path, fault):
    model, messages = stage_model(tmp_path)
    if fault == "missing_system":
        messages.pop(0)
    elif fault == "extra_system":
        messages.insert(1, messages[0].copy())
    elif fault == "empty_system":
        messages[0]["content"] = " "
    elif fault == "extra_field":
        messages[-1]["tool_calls"] = []
    elif fault == "tool":
        messages[-1]["role"] = "tool"
    else:
        messages[-1]["role"] = "assistant"
    with pytest.raises(ValueError):
        model.prepare_stage_request(messages)


def test_stage_request_keeps_process_binding_and_does_not_change_default(tmp_path):
    model, messages = stage_model(tmp_path)
    model._count_payload = lambda payload: 2
    request = model.prepare_stage_request(messages)
    request.check_context()
    model.process = SimpleNamespace(poll=lambda: None)
    with pytest.raises(RuntimeError, match="process changed"):
        list(request.stream(threading.Event()))
    ordinary = model.prepare_request(messages)
    assert ordinary._payload["messages"][0]["content"] == "synthetic baseline"
    assert len(ordinary._payload["messages"]) == 3


def test_stage_schema_is_frozen_into_the_exact_counted_local_wire(tmp_path):
    model, messages = stage_model(tmp_path)
    schema = {"type": "object", "properties": {"action": {"const": "answer"},
        "text": {"type": "string"}}, "required": ["action", "text"],
        "additionalProperties": False}
    original = deepcopy(schema)
    request = model.prepare_stage_request(messages, stage="demand_control", schema=schema)
    schema["properties"]["action"]["const"] = "mutated"
    counted = []
    model._count_payload = lambda payload: counted.append(payload) or 2
    request.check_context()
    assert model.supports_stage_metadata is True
    assert counted == [request._payload]
    assert counted[0]["response_format"] == {"type": "json_schema", "json_schema": {
        "name": "demand_control", "strict": True, "schema": original}}
    assert counted[0]["messages"] == messages
    assert counted[0]["max_tokens"] == model.profile["sampling"]["max_tokens"]
    assert counted[0]["seed"] == model.profile["sampling"]["seed"]


@pytest.mark.parametrize("total,slots,capacity", [(4096, 1, 4096), (4096, 2, 2048),
                                                  (8192, 2, 4096)])
def test_local_preflight_uses_effective_per_slot_context(tmp_path, total, slots, capacity):
    model, messages = stage_model(tmp_path)
    model.profile["server_arguments"] = ["--ctx-size", str(total), "--parallel", str(slots)]
    request = model.prepare_stage_request(messages)
    assert request.capacity == capacity
    reserve = model.profile["sampling"]["max_tokens"]
    model._count_payload = lambda _: capacity - reserve - 1
    request.check_context()
    model.check_context(messages)
    model._count_payload = lambda _: capacity - reserve
    with pytest.raises(ContextLimitError):
        request.check_context()
    with pytest.raises(ContextLimitError):
        model.check_context(messages)


@pytest.mark.parametrize("entry", ["prepared", "memory_worker"])
def test_stage_loopback_counts_and_sends_identical_frozen_complete_messages(
    tmp_path, monkeypatch, entry,
):
    model, messages = stage_model(tmp_path)
    expected_messages = deepcopy(messages)
    received = []
    literal = "literal <think>inside JSON</think> must survive"
    content = json.dumps({"action": "answer", "text": literal})

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append((self.path, payload))
            if self.path == "/apply-template":
                body = b'{"prompt":"rendered stage"}'
            elif self.path == "/tokenize":
                body = b'{"tokens":[1,2]}'
            else:
                chunks = [
                    {"choices": [{"index": 0, "delta": {"reasoning_content": "hidden"}}]},
                    {"choices": [{"index": 0, "delta": {"content": content}}]},
                    {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                ]
                body = ("".join("data: " + json.dumps(item) + "\n\n" for item in chunks)
                        + "data: [DONE]\n\n").encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
    thread.start()
    original = llama.http.client.HTTPConnection
    monkeypatch.setattr(llama.http.client, "HTTPConnection",
                        lambda host, port, timeout:
                        original("127.0.0.1", server.server_port, timeout=timeout))
    try:
        if entry == "prepared":
            request = model.prepare_stage_request(messages)
            request.check_context()
            messages[0]["content"] = "caller changed"
            model.profile["sampling"]["max_tokens"] = 1
            result = json.loads("".join(request.stream(threading.Event())))
        else:
            # Native JSON schema validation stays with the host stage caller;
            # no provider tool or schema wire capability is inferred.
            schema = {"type": "object", "properties": {
                "action": {"const": "answer"}, "text": {"type": "string"}},
                "required": ["action", "text"], "additionalProperties": False}
            gate = _PriorityGenerationGate()
            invoke = _StructuredMemoryInvoker(model, gate)
            result = invoke("memory_command", messages, schema, threading.Event())
            Draft202012Validator(schema).validate(result)
            assert gate._active is None
        assert result["text"] == literal
        assert [route for route, _ in received] == [
            "/apply-template", "/tokenize", "/v1/chat/completions",
        ]
        assert received[0][1] == received[2][1]
        assert received[2][1]["messages"] == expected_messages
        assert sum(m["role"] == "system" for m in received[2][1]["messages"]) == 1
        assert received[2][1]["max_tokens"] == 512
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert not thread.is_alive()


@pytest.mark.parametrize("stage", ["exchange_intent", "exchange_extract"])
def test_local_memory_output_override_is_one_frozen_counted_and_sent_payload(tmp_path, stage):
    model, messages = stage_model(tmp_path)
    model.profile["sampling"]["max_tokens"] = 192
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}},
              "required": ["ok"], "additionalProperties": False}
    request = model.prepare_stage_request(messages, stage=stage, schema=schema,
                                          output_tokens=512)
    assert model.prepare_request(messages)._payload["max_tokens"] == 192
    counted, sent = [], []
    model._count_payload = lambda payload: counted.append(payload) or 2

    def stream(payload, cancelled, **metadata):
        sent.append((payload, metadata))
        yield '{"ok":true}'

    model._stream_payload = stream
    model.profile["sampling"]["max_tokens"] = 1
    request.check_context()
    assert list(request.stream(threading.Event())) == ['{"ok":true}']
    assert counted == [sent[0][0]]
    assert sent[0][0]["max_tokens"] == request.reserve == 512
    assert sent[0][1] == {"stage": True}
    assert model.profile["sampling"]["max_tokens"] == 1
    with pytest.raises(ValueError, match="checked request"):
        list(request.stream(threading.Event()))


def test_local_memory_output_override_uses_frozen_capacity_reserve(tmp_path):
    model, messages = stage_model(tmp_path)
    model.profile["sampling"]["max_tokens"] = 192
    request = model.prepare_stage_request(messages, output_tokens=512)
    model._count_payload = lambda _: request.capacity - 512 - 1
    request.check_context()
    model.profile["sampling"]["max_tokens"] = 1
    model._count_payload = lambda _: request.capacity - 512
    with pytest.raises(ContextLimitError):
        request.check_context()


@pytest.mark.parametrize("output_tokens", [True, 0, 4097])
def test_local_stage_rejects_unbounded_output_override(tmp_path, output_tokens):
    model, messages = stage_model(tmp_path)
    with pytest.raises(ValueError, match="output"):
        model.prepare_stage_request(messages, output_tokens=output_tokens)


@pytest.mark.parametrize("stage", ["exchange_intent", "exchange_extract"])
def test_local_memory_invoker_propagates_exchange_reserve_without_changing_default(tmp_path, stage):
    model, messages = stage_model(tmp_path)
    model.profile["sampling"]["max_tokens"] = 192
    model.profile["server_arguments"] = ["--ctx-size", "8192"]
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}},
              "required": ["ok"], "additionalProperties": False}
    counted, sent = [], []
    model._count_payload = lambda payload: counted.append(payload) or 2

    def stream(payload, cancelled, **metadata):
        sent.append(payload)
        yield '{"ok":true}'

    model._stream_payload = stream
    invoke = _StructuredMemoryInvoker(model, _PriorityGenerationGate())
    assert invoke(stage, messages, schema, threading.Event()) == {"ok": True}
    assert counted == sent
    assert sent[0]["max_tokens"] == 1024
    assert model.profile["sampling"]["max_tokens"] == 192
