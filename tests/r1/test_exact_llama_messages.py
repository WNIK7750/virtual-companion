import hashlib
import importlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from virtual_companion.text.llama import LlamaModel


def setup(tmp_path):
    baseline = "synthetic baseline"
    (tmp_path / "prompt.txt").write_text(baseline, encoding="utf-8")
    profile = {
        "prompt_path": "prompt.txt",
        "prompt_sha256": hashlib.sha256(baseline.encode()).hexdigest(),
        "sampling": {"max_tokens": 512, "seed": 1},
        "server_arguments": ["--ctx-size", "4096"],
    }
    messages = [
        {"role": "system", "content": "pinned persona and protocol"},
        {"role": "user", "content": "synthetic context"},
    ]
    return profile, messages


def model(tmp_path):
    cls = importlib.import_module("virtual_companion.text.exact_llama").ExactLlamaModel
    profile, messages = setup(tmp_path)
    expected = hashlib.sha256(messages[0]["content"].encode()).hexdigest()
    instance = cls(profile, tmp_path, system_sha256=frozenset({expected}))
    instance.process = SimpleNamespace(poll=lambda: None)
    return instance, profile, messages


def test_exact_path_does_not_prepend_baseline_or_mutate_caller(tmp_path):
    instance, profile, messages = model(tmp_path)
    payload = instance._payload(messages)
    assert payload["messages"] == messages
    assert len(payload["messages"]) == 2
    payload["messages"][0]["content"] = "changed"
    profile["sampling"]["max_tokens"] = 1
    assert instance._payload(messages)["max_tokens"] == 512
    assert messages[0]["content"] == "pinned persona and protocol"


@pytest.mark.parametrize(
    "fault", ["missing_system", "extra_system", "changed_system", "tool_role", "extra_field"]
)
def test_exact_path_rejects_unfrozen_or_unsupported_messages(tmp_path, fault):
    instance, _, messages = model(tmp_path)
    if fault == "missing_system":
        messages = messages[1:]
    elif fault == "extra_system":
        messages.insert(1, messages[0].copy())
    elif fault == "changed_system":
        messages[0]["content"] += "changed"
    elif fault == "tool_role":
        messages[-1]["role"] = "tool"
    else:
        messages[0]["name"] = "unexpected"
    with pytest.raises(ValueError):
        instance._payload(messages)


def test_token_count_receives_the_same_final_messages_as_generation(tmp_path):
    instance, _, messages = model(tmp_path)
    requests = []

    def request(route, payload):
        requests.append((route, payload))
        return {"prompt": "exact rendered"} if route == "/apply-template" else {"tokens": [1, 2]}

    instance._context_request = request
    instance.check_context(messages)
    assert requests[0][1] == instance._payload(messages)
    assert requests[1][1]["content"] == "exact rendered"


def test_default_adapter_still_prepends_its_frozen_prompt(tmp_path):
    profile, messages = setup(tmp_path)
    instance = LlamaModel(profile, tmp_path)
    instance.process = SimpleNamespace(poll=lambda: None)
    assert instance._payload(messages)["messages"][0]["content"] == "synthetic baseline"
    assert len(instance._payload(messages)["messages"]) == 3


@pytest.mark.parametrize(
    "chunk_mode", ["ordinary", "whole", "characters", "reasoning", "prefix", "desktop"]
)
@pytest.mark.parametrize("prepared", [False, True])
def test_real_loopback_receives_exact_messages_after_capacity_check(
    tmp_path, monkeypatch, chunk_mode, prepared
):
    from virtual_companion.text import llama

    instance, _, messages = model(tmp_path)
    received = []
    literal = "用户输入里有 <think>这个标签</think>，我照原样记下。以及尾部<thin"
    expected = (
        "valid"
        if chunk_mode == "ordinary"
        else json.dumps(
            {"semantic_text": literal, "units": [{"text": literal}]}, ensure_ascii=False
        )
    )
    if chunk_mode == "prefix":
        expected = "<think>ambiguous prefix</think>" + expected
    raw_content = expected
    if chunk_mode == "desktop":
        instance = LlamaModel(instance.profile, tmp_path)
        instance.process = SimpleNamespace(poll=lambda: None)
        raw_content, expected = "<think>hidden reasoning</think>visible", "visible"
    fragments = list(raw_content) if chunk_mode == "characters" else [raw_content]
    deltas = [{"content": part} for part in fragments]
    if chunk_mode == "reasoning":
        deltas.insert(0, {"reasoning_content": "separate reasoning must not enter JSON"})

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append((self.path, payload))
            if self.path == "/apply-template":
                body = json.dumps({"prompt": "rendered fixture"}).encode()
            elif self.path == "/tokenize":
                body = json.dumps({"tokens": [1, 2]}).encode()
            else:
                body = (
                    b"".join(
                        (
                            "data: "
                            + json.dumps(
                                {"choices": [{"index": 0, "delta": delta}]}, ensure_ascii=False
                            )
                            + "\n\n"
                        ).encode()
                        for delta in deltas
                    )
                    + b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
                    + b"data: [DONE]\n\n"
                )
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
    monkeypatch.setattr(
        llama.http.client,
        "HTTPConnection",
        lambda host, port, timeout: original("127.0.0.1", server.server_port, timeout=timeout),
    )
    try:
        if prepared:
            request = instance.prepare_request(messages)
            request.check_context()
            actual = "".join(request.stream(threading.Event()))
        else:
            instance.check_context(messages)
            actual = "".join(instance.stream(messages, threading.Event()))
        assert actual == expected
        if chunk_mode == "prefix":
            with pytest.raises(json.JSONDecodeError):
                json.loads(actual)
        elif chunk_mode not in {"ordinary", "desktop"}:
            assert json.loads(actual)["units"][0]["text"] == literal
        assert [route for route, _ in received] == [
            "/apply-template",
            "/tokenize",
            "/v1/chat/completions",
        ]
        assert received[0][1]["messages"] == received[2][1]["messages"]
        if prepared:
            assert received[0][1] == received[2][1]
        assert received[2][1]["messages"] == instance._payload(messages)["messages"]
        assert received[2][1]["max_tokens"] == 512
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert not thread.is_alive()
