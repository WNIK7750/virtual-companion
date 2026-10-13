from __future__ import annotations

import hashlib
import http.client
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from virtual_companion.text.application import TextSession
from virtual_companion.text.llama import LlamaModel
from virtual_companion.text.memory import MemoryConversationStore


def model_fixture(tmp_path):
    prompt = "synthetic"
    (tmp_path / "prompt.txt").write_text(prompt, encoding="utf-8")
    model = LlamaModel(
        {
            "prompt_path": "prompt.txt",
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "sampling": {},
        },
        tmp_path,
    )
    model.process = SimpleNamespace(poll=lambda: None)
    return model


def test_cancel_interrupts_blocked_request_write(tmp_path, monkeypatch):
    model = model_fixture(tmp_path)
    entered, interrupted, release = threading.Event(), threading.Event(), threading.Event()
    cancelled = threading.Event()
    errors = []

    class Socket:
        def dup(self):
            return self

        def close(self):
            pass

        def setblocking(self, enabled):
            pass

        def shutdown(self, how):
            interrupted.set()
            release.set()

    class Connection:
        def __init__(self, *args, **kwargs):
            self.sock = None

        def connect(self):
            self.sock = Socket()

        def request(self, *args, **kwargs):
            if self.sock is None:
                self.connect()
            entered.set()
            release.wait(2)
            raise OSError("synthetic blocked write")

        def close(self):
            self.sock = None

    monkeypatch.setattr("virtual_companion.text.llama.http.client.HTTPConnection", Connection)

    def consume():
        try:
            list(model.stream([{"role": "user", "content": "hello"}], cancelled))
        except OSError as error:
            errors.append(str(error))

    worker = threading.Thread(target=consume)
    worker.start()
    try:
        assert entered.wait(1)
        cancelled.set()
        model.cancel()
        assert interrupted.wait(0.5), "blocked request socket was not registered for cancellation"
        worker.join(1)
        assert not worker.is_alive()
    finally:
        release.set()
        worker.join(3)
    assert model.connection is None and model.socket is None


def test_cancel_during_connect_does_not_send_or_reconnect(tmp_path, monkeypatch):
    model = model_fixture(tmp_path)
    entered, release = threading.Event(), threading.Event()
    cancelled = threading.Event()
    sent = []

    class Socket:
        def dup(self):
            return self

        def close(self):
            pass

        def setblocking(self, enabled):
            pass

    class Connection:
        def __init__(self, *args, **kwargs):
            self.sock = None

        def connect(self):
            entered.set()
            release.wait(2)
            self.sock = Socket()

        def request(self, *args, **kwargs):
            if self.sock is None:
                self.connect()
            sent.append(True)

        def getresponse(self):
            return SimpleNamespace(status=200)

        def close(self):
            self.sock = None

    monkeypatch.setattr("virtual_companion.text.llama.http.client.HTTPConnection", Connection)
    worker = threading.Thread(target=lambda: list(model.stream([], cancelled)))
    worker.start()
    try:
        assert entered.wait(1)
        cancelled.set()
        model.cancel()
    finally:
        release.set()
        worker.join(3)
    assert not worker.is_alive()
    assert sent == [], "cancelled connection must not submit a new model request"


@pytest.mark.parametrize(
    "phase", ["headers", "body", "disconnect", "length", "content_filter", "missing_finish"]
)
def test_real_loopback_stall_or_disconnect_preserves_retry_semantics(tmp_path, monkeypatch, phase):
    entered, release = threading.Event(), threading.Event()
    requests = []
    http_connection = http.client.HTTPConnection

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):  # noqa: N802
            requests.append(self.rfile.read(int(self.headers["Content-Length"])))
            try:
                first = len(requests) == 1
                if first and phase == "headers":
                    entered.set()
                    release.wait(5)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                if first:
                    self.wfile.write(
                        b'data: {"choices":[{"index":0,"delta":{"content":"partial"}}]}\n\n'
                    )
                    self.wfile.flush()
                    entered.set()
                    if phase == "missing_finish":
                        self.wfile.write(b"data: [DONE]\n\n")
                        return
                    if phase in {"length", "content_filter"}:
                        self.wfile.write(
                            ('data: {"choices":[{"index":0,"delta":{},"finish_reason":"'
                             + phase + '"}]}\n\ndata: [DONE]\n\n').encode()
                        )
                        return
                    if phase != "disconnect":
                        release.wait(5)
                    return  # EOF without DONE must never commit a complete reply.
                self.wfile.write(
                    b'data: {"choices":[{"index":0,"delta":{"content":"complete"},'
                    b'"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
                )
            except ConnectionError:
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    listener = threading.Thread(target=server.serve_forever, daemon=True)
    listener.start()

    def connection_factory(host, port, **kwargs):
        assert (host, port) == ("127.0.0.1", 18083)
        return http_connection("127.0.0.1", server.server_port, **kwargs)

    monkeypatch.setattr(
        "virtual_companion.text.llama.http.client.HTTPConnection", connection_factory
    )
    model = model_fixture(tmp_path)
    session = TextSession(model, MemoryConversationStore(), {})
    try:
        first = session.send("synthetic question")
        assert entered.wait(2)
        if phase in {"disconnect", "length", "content_filter", "missing_finish"}:
            session.worker.join(2)
            assert session.snapshot()["attempt"]["status"] == "failed"
            assert session.snapshot()["error"]
        else:
            started = time.monotonic()
            state = session.stop()
            assert time.monotonic() - started < 2.5
            assert not state["busy"] and state["attempt"]["status"] == "cancelled"
        assert [m["role"] for m in session.history()["messages"]] == ["user"]
        release.set()
        second = session.retry()
        session.worker.join(2)
        assert not session.snapshot()["busy"]
        assert second["attempt_id"] != first["attempt_id"]
        assert second["turn_id"] == first["turn_id"]
        assert session.snapshot()["attempt"]["status"] == "completed"
        assert [m["content"] for m in session.history()["messages"]] == [
            "synthetic question",
            "complete",
        ]
        assert len(requests) == 2
    finally:
        release.set()
        model.cancel()
        if session.worker:
            session.worker.join(3)
        server.shutdown()
        server.server_close()
        listener.join(2)
