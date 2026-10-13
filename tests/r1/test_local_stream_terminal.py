"""Local HTTP stream failures must not promote displayed prefixes to history."""

import hashlib
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from virtual_companion.providers.openai_compatible import StreamFailure, decode_stream
from virtual_companion.text import llama
from virtual_companion.text.application import TextSession
from virtual_companion.text.memory import MemoryConversationStore


def event(delta, finish=None, index=0):
    return {"choices": [{"index": index, "delta": delta, "finish_reason": finish}]}


def wire(events, *, done=True):
    return b"".join(
        ("data: " + json.dumps(item, ensure_ascii=False) + "\n\n").encode()
        for item in events
    ) + (b"data: [DONE]\n\n" if done else b"")


START = event({"content": "合成前缀"})
STOP = event({}, "stop")
USAGE = {"choices": [], "usage": {
    "prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5,
}}
INVALID = {
    "late_content": [START, STOP, event({"content": "illegal tail"})],
    "refusal_with_content": [event({"content": "unsafe", "refusal": "refused"}), STOP],
    "wrong_index": [event({"content": "wrong"}, index=1), STOP],
    "missing_index": [{"choices": [{"delta": {"content": "wrong"}}]}, STOP],
    "multiple_choices": [{"choices": [START["choices"][0], START["choices"][0]]}, STOP],
    "duplicate_finish": [START, STOP, STOP],
    "bad_choices": [START, {"choices": {}}, STOP],
    "bad_delta": [START, event([]), STOP],
    "bad_content": [START, event({"content": []}), STOP],
    "bad_usage": [START, STOP, {"choices": [], "usage": {"total_tokens": -1}}],
    "conflicting_usage": [START, STOP, USAGE, {"choices": [], "usage": {
        "prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6,
    }}],
    "length": [START, event({}, "length")],
    "missing_finish": [START],
    "error": [START, {"error": {"message": "synthetic"}}, STOP],
}


@contextmanager
def loopback(tmp_path, monkeypatch, raw, *, pause=False, split=False):
    entered, release = threading.Event(), threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            try:
                for chunk in ([bytes([byte]) for byte in raw] if split else [raw]):
                    self.wfile.write(chunk)
                    self.wfile.flush()
                entered.set()
                if pause:
                    release.wait(3)
            except ConnectionError:
                pass

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    listener = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True,
    )
    listener.start()
    original = llama.http.client.HTTPConnection
    monkeypatch.setattr(
        llama.http.client, "HTTPConnection",
        lambda host, port, timeout: original("127.0.0.1", server.server_port, timeout=timeout),
    )
    (tmp_path / "prompt.txt").write_text("synthetic", encoding="utf8")
    model = llama.LlamaModel({
        "prompt_path": "prompt.txt",
        "prompt_sha256": hashlib.sha256(b"synthetic").hexdigest(),
        "sampling": {"max_tokens": 64},
    }, tmp_path)
    model.process = SimpleNamespace(poll=lambda: None)
    try:
        yield model, entered
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        listener.join(2)


@pytest.mark.parametrize("case", INVALID)
def test_invalid_local_stream_never_commits_assistant(tmp_path, monkeypatch, case):
    raw = wire(INVALID[case])
    with pytest.raises(StreamFailure):
        decode_stream(raw)
    with loopback(tmp_path, monkeypatch, raw) as (model, _):
        session = TextSession(model, MemoryConversationStore(), {})
        session.send("synthetic input")
        session.worker.join(2)
        state = session.snapshot()
        assert not state["busy"]
        assert state["attempt"]["status"] == "failed"
        assert state["error"]
        assert [m["role"] for m in session.history()["messages"]] == ["user"]
        if case in {"refusal_with_content", "wrong_index", "missing_index", "multiple_choices"}:
            assert state["partial"] == ""
        if case in {"late_content", "duplicate_finish", "bad_usage", "length", "missing_finish"}:
            assert state["partial"] == "合成前缀"


@pytest.mark.parametrize("usage", [False, True])
@pytest.mark.parametrize("split", [False, True])
def test_normal_stop_optional_usage_utf8(tmp_path, monkeypatch, usage, split):
    raw = wire([START, STOP] + ([USAGE] if usage else []))
    with loopback(tmp_path, monkeypatch, raw, split=split) as (model, _):
        session = TextSession(model, MemoryConversationStore(), {})
        session.send("synthetic input")
        session.worker.join(2)
        assert session.snapshot()["attempt"]["status"] == "completed"
        assert session.history()["messages"][-1]["content"] == "合成前缀"


def test_increment_is_available_before_terminal_and_cancel_closes_stream(tmp_path, monkeypatch):
    with loopback(tmp_path, monkeypatch, wire([START], done=False), pause=True) as (model, _):
        cancelled = threading.Event()
        stream = model.stream([], cancelled)
        assert next(stream) == "合成前缀"
        cancelled.set()
        assert list(stream) == []
        assert model.connection is None and model.socket is None


def test_eof_without_done_fails_after_prefix(tmp_path, monkeypatch):
    with loopback(tmp_path, monkeypatch, wire([START, STOP], done=False)) as (model, _):
        stream = model.stream([], threading.Event())
        assert next(stream) == "合成前缀"
        with pytest.raises((ValueError, RuntimeError)):
            list(stream)


@pytest.mark.parametrize("bad_record", [
    b'data: {"choices":[],"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
    b'data: {"choices":[{"index":false,"delta":{},"finish_reason":"stop"}]}\n\n',
    b'data: {"choices":[{"index":0,"delta":{"content":"\xff"}}]}\n\n',
    b'data: {"choices":[{"index":0,"delta":{"tool_calls":[{}]}}]}\n\n',
])
def test_malformed_wire_rejects_after_valid_prefix(tmp_path, monkeypatch, bad_record):
    raw = wire([START], done=False) + bad_record + wire([STOP])
    with loopback(tmp_path, monkeypatch, raw) as (model, _):
        stream = model.stream([], threading.Event())
        assert next(stream) == "合成前缀"
        with pytest.raises((ValueError, RuntimeError)):
            list(stream)
