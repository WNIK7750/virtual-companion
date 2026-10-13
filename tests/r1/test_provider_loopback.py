import asyncio
import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
from test_model_provider_contract import event, route
from virtual_companion.providers.bound_call import BoundProviderCall
from virtual_companion.providers.budget import ProviderBudgetLedger
from virtual_companion.providers.httpx_transport import HttpxProviderTransport, ProviderFailure
from virtual_companion.providers.openai_compatible import build_request

# An opaque synthetic DPAPI reference ID; the Secrets fixture supplies fake bytes.
SYNTHETIC_DPAPI_REFERENCE = "dpapi:0123456789abcdef0123456789abcdef"


class Ledger:
    def __init__(self, limit=1):
        self.calls, self.finished, self.limit = [], [], limit

    def reserve(self, **record):
        if len(self.calls) >= self.limit:
            raise ValueError("budget exhausted")
        self.calls.append(record)
        return str(len(self.calls))

    def finish(self, receipt, **record):
        self.finished.append({"receipt": receipt, **record})


@contextmanager
def server(mode):
    observed, arrived, release = [], threading.Event(), threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):  # noqa: N802
            observed.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            arrived.set()
            if mode == "headers_stall":
                release.wait(3)
                return
            if mode == "error":
                self.send_response(401)
                self.end_headers()
                self.wfile.write(b"SECRET-ECHO-MUST-NOT-REACH-ERROR")
                return
            if mode == "redirect":
                self.send_response(307)
                self.send_header("Location", "/unexpected")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            if mode == "body_stall":
                self.wfile.write(b": waiting\n\n")
                self.wfile.flush()
                release.wait(3)
                return
            if mode == "content_stall":
                self.wfile.write(event({"choices": [{"index": 0, "delta": {"content": "片段🌿"}}]}))
                self.wfile.flush()
                release.wait(3)
                return
            if mode == "first_stall" and len(observed) == 1:
                release.wait(3)
                return
            raw = event(
                {"choices": [{"index": 0, "delta": {"content": "合成🌿"}, "finish_reason": None}]}
            )
            raw += event(
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "length" if mode == "truncated" else "stop",
                        }
                    ]
                }
            )
            if mode == "truncated":
                raw += event(
                    {
                        "choices": [],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
                    }
                )
            raw += b"data: [DONE]\n\n"
            for byte in raw:
                self.wfile.write(bytes([byte]))
            self.wfile.flush()

    service = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=service.serve_forever)
    thread.start()
    try:
        yield service.server_port, observed, arrived
    finally:
        release.set()
        service.shutdown()
        service.server_close()
        thread.join(2)


def client(port, ledger):
    configured = route(endpoint=f"http://127.0.0.1:{port}/v1/chat/completions")
    transport = HttpxProviderTransport(configured, secrets=None, ledger=ledger)
    body = build_request(configured, [{"role": "user", "content": "synthetic"}], output_tokens=32)
    return transport, body


def invoke(transport, body, cancelled=None, stage="reply"):
    return transport.invoke(
        body,
        cancelled or threading.Event(),
        stage=stage,
        data_scope=("current",),
        input_tokens=20,
        timeout_seconds=2,
    )


def test_unapproved_delta_arrives_before_terminal_and_cancel_keeps_failure():
    with server("content_stall") as (port, observed, _):
        ledger = Ledger()
        transport, body = client(port, ledger)
        cancelled, deltas = threading.Event(), []

        def receive(delta):
            assert not ledger.finished
            deltas.append(delta)
            cancelled.set()

        call = BoundProviderCall(
            transport.route, transport=transport, counter=lambda body: 20,
            stage="reply", cancelled=cancelled, clock=time.monotonic,
            deadline=time.monotonic() + 2, output_tokens=32, data_scope=("current",),
            on_content=receive)
        request = {"messages": [{"role": "user", "content": "synthetic"}],
                   "tools": [], "max_output_tokens": 32}
        call.count(request)
        with pytest.raises(ProviderFailure, match="cancelled"):
            call.invoke(request)
        assert len(observed) == 1
        assert [d.text for d in deltas] == ["片段🌿"]
        assert deltas[0].sequence == 1 and deltas[0].approved is False
        assert ledger.finished[0]["status"] == "cancelled"


@pytest.mark.parametrize("mode", ["normal", "truncated", "consumer_failure"])
def test_provisional_consumer_does_not_replace_final_protocol_or_budget(mode):
    with server("normal" if mode == "consumer_failure" else mode) as (port, _, _):
        ledger = Ledger()
        transport, body = client(port, ledger)
        deltas = []

        def receive(delta):
            deltas.append(delta)
            if mode == "consumer_failure":
                raise ValueError("private-consumer-error")

        def run():
            return transport.invoke(body, threading.Event(), stage="reply",
                                    data_scope=("current",), input_tokens=20,
                                    timeout_seconds=2, on_content=receive)

        if mode == "normal":
            assert run()["text"] == "".join(d.text for d in deltas) == "合成🌿"
            assert ledger.finished[0]["status"] == "completed"
        else:
            with pytest.raises(ProviderFailure) as error:
                run()
            assert "private-consumer-error" not in str(error.value)
            assert ledger.finished[0]["status"] != "completed"
        assert len(ledger.calls) == len(ledger.finished) == 1


def test_real_loopback_stream_reserves_before_send_and_keeps_unknown_usage(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    with server("normal") as (port, observed, _):
        ledger = Ledger()
        transport, body = client(port, ledger)
        result = invoke(transport, body)
        assert result["text"] == "合成🌿"
        assert len(ledger.calls) == len(observed) == 1
        assert ledger.calls[0]["output_tokens"] == 32
        assert ledger.finished[0]["usage"] is None
        assert ledger.finished[0]["status"] == "completed"
        assert ledger.finished[0]["transmission"] == "response_received"
        with pytest.raises(ValueError, match="budget"):
            invoke(transport, body)
        assert len(observed) == 1


def test_read_timeout_after_headers_has_content_free_failure_node(monkeypatch):
    ledger = Ledger()
    transport, body = client(1, ledger)

    async def stalled(payload, headers, cancelled, timeout, outcome, on_content=None):
        outcome["transmission"] = "response_received"
        raise httpx.ReadTimeout("private server text")

    monkeypatch.setattr(transport, "_exchange", stalled)
    with pytest.raises(ProviderFailure, match="provider_read_timeout") as error:
        invoke(transport, body)
    assert "private server text" not in str(error.value)
    assert error.value.__cause__ is None
    assert ledger.finished[0]["status"] == "failed"
    assert ledger.finished[0]["transmission"] == "response_received"
    assert ledger.finished[0]["usage"] is None


@pytest.mark.parametrize(
    ("failure_type", "failure_code"),
    [
        (httpx.ConnectTimeout, "provider_connect_timeout"),
        (httpx.ConnectError, "provider_connect_error"),
    ],
)
def test_connect_failure_has_safe_code_and_retains_unknown_charge(tmp_path, monkeypatch,
                                                                failure_type, failure_code):
    configured = route(
        kind="api", endpoint="https://example.test/v1/chat/completions",
        auth="bearer", secret_ref=SYNTHETIC_DPAPI_REFERENCE,
        explicit=True,
    )
    ledger = ProviderBudgetLedger(tmp_path / "calls.sqlite3", {
        "id": "synthetic-connect-failure.v1", "max_calls": 1,
        "max_input_tokens": 20, "max_output_tokens": 32, "max_cost_microusd": 1,
        "routes": {configured.fingerprint: {
            "kind": "api", "input_rate": 1000, "output_rate": 2000,
        }},
    })

    class Secrets:
        def get(self, reference, endpoint):
            return "synthetic-credential"

    transport = HttpxProviderTransport(configured, secrets=Secrets(), ledger=ledger)
    body = build_request(configured, [{"role": "user", "content": "synthetic"}],
                         output_tokens=32)
    attempts = []

    async def failed_connect(http_transport, request):
        assert len(ledger.records()) == 1 and ledger.records()[0]["status"] == "reserved"
        attempts.append(request.method)
        raise failure_type("private network detail", request=request)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", failed_connect)
    with pytest.raises(ProviderFailure) as error:
        invoke(transport, body, stage="demand_control")

    assert str(error.value) == failure_code
    assert error.value.__cause__ is None
    assert attempts == ["POST"]
    assert transport.metrics["headers"] is transport.metrics["first_content"] is None
    assert transport.metrics["finished"] is not None
    records = ledger.records()
    assert len(records) == 1
    assert records[0]["status"] == "failed"
    assert records[0]["transmission"] == "may_have_been_sent"
    assert records[0]["usage"] is records[0]["estimated_cost_microusd"] is None
    reopened = ProviderBudgetLedger(ledger.path, ledger.policy)
    assert reopened.summary() == {
        "reserved_calls": 1, "reserved_output_tokens": 32,
        "unknown_usage_calls": 1, "reserved_cost_microusd": 1,
    }


@pytest.mark.parametrize("mode", ["headers_stall", "body_stall"])
def test_actual_socket_cancel_returns_and_records_unknown_usage(mode):
    with server(mode) as (port, _, arrived):
        ledger = Ledger()
        transport, body = client(port, ledger)
        errors = []

        def run():
            try:
                invoke(transport, body)
            except ProviderFailure as error:
                errors.append(str(error))

        worker = threading.Thread(target=run)
        worker.start()
        assert arrived.wait(2)
        started = time.monotonic()
        transport.cancel()
        worker.join(1)
        assert not worker.is_alive()
        assert time.monotonic() - started < 1
        assert errors == ["provider_cancelled"]
        assert ledger.finished[0]["status"] == "cancelled"
        assert ledger.finished[0]["usage"] is None


@pytest.mark.parametrize("mode", ["error", "redirect"])
def test_error_body_is_not_exposed_and_redirects_are_not_followed(mode):
    with server(mode) as (port, observed, _):
        ledger = Ledger()
        transport, body = client(port, ledger)
        with pytest.raises(ProviderFailure) as error:
            invoke(transport, body)
        assert "SECRET-ECHO" not in str(error.value)
        assert error.value.__cause__ is None
        assert len(observed) == 1
        assert ledger.finished[0]["status"] == "failed"


def test_cancelled_or_unapproved_context_never_sends_or_reserves():
    ledger = Ledger()
    transport, body = client(1, ledger)
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(ProviderFailure, match="cancelled"):
        invoke(transport, body, cancelled)
    with pytest.raises(ValueError, match="scope"):
        transport.invoke(
            body,
            threading.Event(),
            stage="reply",
            data_scope=("history",),
            input_tokens=20,
            timeout_seconds=2,
        )
    assert ledger.calls == []


def test_truncated_response_retains_reported_usage_in_failure_ledger():
    with server("truncated") as (port, _, _):
        ledger = Ledger()
        transport, body = client(port, ledger)
        with pytest.raises(ProviderFailure):
            invoke(transport, body)
        assert ledger.finished[0]["status"] == "failed"
        assert ledger.finished[0]["usage"] == {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18,
        }


def test_budget_reservation_wait_consumes_absolute_call_deadline():
    class DelayedLedger(Ledger):
        def reserve(self, **record):
            time.sleep(0.05)
            return super().reserve(**record)

    with server("normal") as (port, observed, _):
        ledger = DelayedLedger()
        transport, body = client(port, ledger)
        with pytest.raises(ProviderFailure, match="deadline"):
            transport.invoke(
                body,
                threading.Event(),
                stage="reply",
                data_scope=("current",),
                input_tokens=20,
                timeout_seconds=0.01,
            )
        assert observed == []
        assert ledger.finished[0]["transmission"] == "not_sent"


def test_sampling_override_cannot_change_after_route_freeze():
    from dataclasses import replace

    from virtual_companion.providers.contracts import ProviderSampling

    ledger = Ledger()
    transport, body = client(1, ledger)
    transport.route = replace(transport.route, sampling=ProviderSampling(0.2, 0.8, 1))
    body.update(temperature=0.2, top_p=0.8, seed=2)
    with pytest.raises(ValueError, match="sampling"):
        invoke(transport, body)
    assert ledger.calls == []


def test_generation_and_review_reserve_one_shared_ledger_while_both_in_flight():
    """Parallel review needs two real transports and one pre-send budget ledger."""

    with server("first_stall") as (port, observed, arrived):
        ledger = Ledger(limit=2)
        generation, body = client(port, ledger)
        review, review_body = client(port, ledger)
        cancelled, failures, results = threading.Event(), [], []

        def call():
            try:
                results.append(invoke(generation, body, cancelled))
            except ProviderFailure as error:
                failures.append(str(error))

        worker = threading.Thread(target=call)
        worker.start()
        assert arrived.wait(2)
        review_result = invoke(review, review_body, stage="stream_unit_review")
        assert review_result["text"] == "合成🌿"
        assert [record["stage"] for record in ledger.calls] == ["reply", "stream_unit_review"]
        assert worker.is_alive(), "Generation must still be in flight during review"
        assert len(ledger.calls) == 2 and len(ledger.finished) == 1
        generation.cancel()
        worker.join(2)
        assert not worker.is_alive()
        assert failures == ["provider_cancelled"] and results == []
        statuses = {record["receipt"]: record["status"] for record in ledger.finished}
        assert sorted(statuses.values()) == ["cancelled", "completed"]
        assert len(observed) == 2


def test_first_content_is_observed_before_completion_and_cancel_is_timed():
    with server("content_stall") as (port, _, _):
        transport, body = client(port, Ledger())
        failures = []

        def run():
            try:
                invoke(transport, body)
            except ProviderFailure as error:
                failures.append(error)

        thread = threading.Thread(target=run)
        thread.start()
        try:
            deadline = time.monotonic() + 1
            while time.monotonic() < deadline and not transport.metrics.get("first_content"):
                time.sleep(0.01)
            snapshot = transport.metrics
            assert snapshot["started"] <= snapshot["first_content"]
            assert snapshot["finished"] is None
        finally:
            transport.cancel()
            thread.join(2)
        assert not thread.is_alive() and failures
        assert transport.metrics["finished"] >= snapshot["first_content"]


@contextmanager
def incomplete_stream_server(prefix, suffix):
    """Send an unfinished suffix only after the caller confirms the complete prefix."""
    observed, continue_stream = [], threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):  # noqa: N802
            observed.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(event({"choices": [{"index": 0,
                "delta": {"content": prefix}, "finish_reason": None}]}))
            self.wfile.flush()
            if not continue_stream.wait(3):
                return
            self.wfile.write(event({"choices": [{"index": 0,
                "delta": {"content": suffix}, "finish_reason": None}]}))
            self.wfile.flush()
            # HTTP EOF without finish_reason or [DONE] is an incomplete SSE stream.

    service = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=service.serve_forever)
    thread.start()
    try:
        yield service.server_port, observed, continue_stream
    finally:
        continue_stream.set()
        service.shutdown()
        service.server_close()
        thread.join(2)
        assert not thread.is_alive(), "Loopback server must stop after EOF"


@pytest.mark.parametrize("observe_content", [False, True], ids=["terminal", "content-observer"])
def test_length_terminal_persists_known_usage_in_sqlite_ledger(tmp_path, monkeypatch,
                                                            observe_content):
    import sqlite3
    from hashlib import sha256

    from virtual_companion.boundary.contracts import canonical

    usage = {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}
    raw = event({"choices": [{"index": 0, "delta": {"content": "合成🌿"}}]})
    raw += event({"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
                  "usage": usage}) + b"data: [DONE]\n\n"
    received, original_bytes = [], httpx.Response.aiter_bytes

    async def observe_bytes(response, *args, **kwargs):
        async for chunk in original_bytes(response, *args, **kwargs):
            received.append(chunk)
            yield chunk

    monkeypatch.setattr(httpx.Response, "aiter_bytes", observe_bytes)
    with complete_sse_server(raw) as (port, requests, written):
        configured = route(endpoint=f"http://127.0.0.1:{port}/v1/chat/completions")
        ledger = ProviderBudgetLedger(tmp_path / "length.sqlite3", {
            "id": "synthetic-length-terminal.v1", "max_calls": 1,
            "max_input_tokens": 20, "max_output_tokens": 32, "max_cost_microusd": 0,
            "routes": {configured.fingerprint: {
                "kind": "local", "input_rate": 0, "output_rate": 0,
            }},
        })
        transport = HttpxProviderTransport(configured, secrets=None, ledger=ledger)
        body = build_request(configured, [{"role": "user", "content": "synthetic"}],
                             output_tokens=32)
        deltas = []
        with pytest.raises(ProviderFailure) as error:
            transport.invoke(body, threading.Event(), stage="reply", data_scope=("current",),
                input_tokens=20, timeout_seconds=2,
                **({"on_content": deltas.append} if observe_content else {}))
        assert error.value.__cause__ is None
        assert len(requests) == 1
        assert transport.metrics["finished"] is not None
        assert written.wait(1) and b"".join(received) == raw

    reopened = ProviderBudgetLedger(ledger.path, ledger.policy)
    with sqlite3.connect(reopened.path) as db:
        db.row_factory = sqlite3.Row
        rows = [dict(row) for row in db.execute("SELECT * FROM provider_calls")]
    assert len(rows) == 1
    record = rows[0]
    assert record["status"] == "failed" and record["transmission"] == "response_received"
    assert record["request_sha256"] == sha256(canonical(requests[0]).encode()).hexdigest()
    assert record["usage"] == canonical(usage), {"failure": str(error.value), "sql": record}
    assert reopened.records()[0]["usage"] == usage
    assert json.loads(record["terminal"]) == ["failed", "response_received", usage]
    assert record["overrun"] == 0 and reopened.summary()["unknown_usage_calls"] == 0
    assert str(error.value) == "provider_output_limit_reached"


@contextmanager
def complete_sse_server(raw):
    """Provide the entire bounded SSE body in one write, including its known usage."""
    observed, written = [], threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):  # noqa: N802
            observed.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            self.wfile.flush()
            written.set()

    service = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=service.serve_forever)
    thread.start()
    try:
        yield service.server_port, observed, written
    finally:
        service.shutdown()
        service.server_close()
        thread.join(2)
        assert not thread.is_alive(), "Complete-response loopback server must stop"


def test_complete_terminal_cannot_override_provisional_text_rejection():
    raw = event({"choices": [{"index": 0, "delta": {"content": "不可呈现",
        "tool_calls": [{"index": 0, "id": "call-1",
                        "function": {"name": "test_tool", "arguments": "{}"}}]}}]})
    raw += event({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
    raw += b"data: [DONE]\n\n"
    with complete_sse_server(raw) as (port, _, _):
        ledger, deltas = Ledger(), []
        transport, body = client(port, ledger)
        with pytest.raises(ProviderFailure):
            transport.invoke(body, threading.Event(), stage="reply", data_scope=("current",),
                             input_tokens=20, timeout_seconds=2, on_content=deltas.append)
        assert deltas == []
        assert len(ledger.calls) == len(ledger.finished) == 1
        assert ledger.finished[0]["status"] == "failed"


@pytest.mark.parametrize("ending", ["cancelled", "completed", "terminal_persist_failure"])
def test_cleanup_await_with_late_cancel_retires_handles_and_keeps_sqlite_terminal(
    tmp_path, monkeypatch, ending
):
    """Widen the real watcher's cleanup await, then cancel through the public API.

    This is an actual HTTP exchange/asyncio task, not a preinstalled closed-loop
    handle. A second cancellation used to skip the later handle withdrawal.
    """
    cleanup_entered, cleanup_release = threading.Event(), threading.Event()
    cleanup_loops, cleanup_tasks = [], []
    original_gather = asyncio.gather

    async def delayed_watcher_cleanup(*tasks, **kwargs):
        if len(tasks) == 1 and isinstance(tasks[0], asyncio.Task) and (
            tasks[0].get_coro().__qualname__.endswith("._exchange.<locals>.watch")
        ):
            cleanup_loops.append(asyncio.get_running_loop())
            cleanup_tasks.append(asyncio.current_task())
            cleanup_entered.set()
            while not cleanup_release.is_set():
                await asyncio.sleep(0.001)
        return await original_gather(*tasks, **kwargs)

    monkeypatch.setattr(asyncio, "gather", delayed_watcher_cleanup)
    usage = {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}
    raw = event({"choices": [{"index": 0, "delta": {"content": "合成关闭验证。"}}]})
    raw += event({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                  "usage": usage}) + b"data: [DONE]\n\n"
    endpoint = server("body_stall") if ending == "cancelled" else complete_sse_server(raw)
    with endpoint as (port, observed, arrived):
        configured = route(endpoint=f"http://127.0.0.1:{port}/v1/chat/completions")
        policy = {
            "id": "synthetic-cleanup-cancel.v1", "max_calls": 1,
            "max_input_tokens": 20, "max_output_tokens": 32, "max_cost_microusd": 0,
            "routes": {configured.fingerprint: {"kind": "local", "input_rate": 0,
                                                 "output_rate": 0}},
        }
        ledger = ProviderBudgetLedger(tmp_path / "cleanup.sqlite3", policy)
        finish_attempts = []
        if ending == "terminal_persist_failure":
            def fail_before_commit(receipt, **record):
                finish_attempts.append({"receipt": receipt, **record})
                raise RuntimeError("synthetic terminal persistence failure")

            monkeypatch.setattr(ledger, "finish", fail_before_commit)
        transport = HttpxProviderTransport(configured, secrets=None, ledger=ledger)
        body = build_request(configured, [{"role": "user", "content": "synthetic"}],
                             output_tokens=32)
        results, errors = [], []

        def call():
            try:
                results.append(invoke(transport, body))
            except Exception as error:
                errors.append(error)

        worker = threading.Thread(target=call)
        worker.start()
        try:
            assert arrived.wait(2)
            if ending == "cancelled":
                assert transport._loop is not None and transport._task is not None
                transport.cancel()  # Still cancels real active I/O.
            assert cleanup_entered.wait(2), "The real exchange never reached watcher cleanup"
            assert worker.is_alive() and not ledger.records()[0]["terminal"]
            assert len(cleanup_loops) == len(cleanup_tasks) == 1
            # Queue cancellation ahead of releasing the await. On the old code it
            # interrupts that await and leaves the now-closing loop published.
            transport.cancel()
            cleanup_loops[0].call_soon_threadsafe(cleanup_release.set)
            worker.join(2)
            assert not worker.is_alive()
        finally:
            cleanup_release.set()
            if worker.is_alive():
                transport.cancel()
            worker.join(2)
        assert cleanup_loops[0].is_closed() and cleanup_tasks[0].done()
        transport.cancel()
        transport.cancel()
        assert transport._loop is None and transport._task is None
        assert len(observed) == 1 and transport.metrics["finished"] is not None
        reopened = ProviderBudgetLedger(ledger.path, policy)
        record = reopened.records()[0]
        assert len(reopened.records()) == 1
        if ending == "terminal_persist_failure":
            assert record["status"] == "reserved" and record["terminal"] is None
            assert record["usage"] is None and reopened.summary()["unknown_usage_calls"] == 1
            assert finish_attempts[0]["status"] == "completed"
            assert finish_attempts[0]["usage"] == usage
        else:
            assert record["status"] == ("cancelled" if ending == "cancelled" else "completed")
            assert record["usage"] == (None if ending == "cancelled" else usage)
            assert record["terminal"] is not None
        if ending == "cancelled":
            assert results == [] and len(errors) == 1
            assert isinstance(errors[0], ProviderFailure) and str(errors[0]) == "provider_cancelled"
            assert reopened.summary()["unknown_usage_calls"] == 1
        elif ending == "terminal_persist_failure":
            assert results == [] and len(errors) == 1
            assert type(errors[0]) is RuntimeError
            assert str(errors[0]) == "synthetic terminal persistence failure"
        else:
            assert errors == [] and results[0]["usage"] == usage
