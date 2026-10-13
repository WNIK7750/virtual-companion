import json
import os
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from virtual_companion.providers.contracts import ProviderCapabilities, ProviderRoute
from virtual_companion.text.provider_model import ProviderModel

SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}


class Counter:
    def __call__(self, body):
        return 17

    def local_byte_bound(self, body):
        return 200


class Transport:
    def __init__(self):
        self.calls = []
        self.tokens = 17

    def invoke(self, body, cancelled, **kwargs):
        self.calls.append((body, kwargs))
        return {
            "text": '{"ok":true}',
            "tool_calls": [],
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": self.tokens,
                "completion_tokens": 5,
                "total_tokens": self.tokens + 5,
            },
        }

    def cancel(self):
        pass


def model(wire_profile="deepseek-flash.v1"):
    route = ProviderRoute(
        "test-api",
        1,
        "api",
        "https://api.deepseek.com/chat/completions",
        "deepseek-flash",
        "bearer",
        "dpapi:" + "a" * 32,
        True,
        ("current", "history", "valid_memory", "derived"),
        ProviderCapabilities(True, False, True, 1048576, "max_tokens"),
        wire_profile=wire_profile,
    )
    transport = Transport()
    return ProviderModel(route, counter=Counter(), transport=transport), transport


@pytest.mark.parametrize("stage", ["demand_control", "final_review", "fact_extract"])
def test_stage_schema_count_and_send_are_one_frozen_request(stage):
    value, transport = model()
    messages = [{"role": "user", "content": "synthetic"}]
    request = value.prepare_stage_request(messages, stage=stage, schema=SCHEMA)
    messages[0]["content"] = "changed"
    request.check_context()
    assert list(request.stream(threading.Event())) == ['{"ok":true}']
    body, options = transport.calls[0]
    assert body["messages"][-1]["content"] == "synthetic"
    assert body["response_format"] == {"type": "json_object"}
    assert options["stage"] == stage
    assert options["input_tokens"] == 17
    with pytest.raises(ValueError):
        list(request.stream(threading.Event()))


@pytest.mark.parametrize("stage,temperature", [("demand_control", 0.2), ("demand_repair", 0.2),
                                             ("stream_unit_review", 0), ("fact_repair", 0),
                                             ("exchange_extract", 0), ("memory_command", 0)])
def test_focused_provider_preflight_freezes_the_actual_stage(stage, temperature):
    value, transport = model("deepseek-flash.focused.v2")
    request = value.prepare_stage_request([{"role": "user", "content": "synthetic"}],
                                          stage=stage, schema=SCHEMA)
    assert request._body["temperature"] == temperature
    request.check_context()
    assert list(request.stream(threading.Event())) == ['{"ok":true}']
    body, options = transport.calls[0]
    assert body == request._body
    assert options["stage"] == stage
    assert body["temperature"] == temperature


def test_missing_schema_or_preflight_or_cancel_never_sends():
    value, transport = model()
    with pytest.raises(ValueError):
        value.prepare_stage_request([], stage="demand_control")
    request = value.prepare_stage_request(
        [{"role": "user", "content": "x"}], stage="demand_control", schema=SCHEMA
    )
    with pytest.raises(ValueError):
        list(request.stream(threading.Event()))
    request.check_context()
    signal = threading.Event()
    signal.set()
    with pytest.raises(InterruptedError):
        list(request.stream(signal))
    assert transport.calls == []


def test_counter_mismatch_disables_route_without_releasing_text():
    value, transport = model()
    transport.tokens = 18
    request = value.prepare_stage_request(
        [{"role": "user", "content": "x"}], stage="final_review", schema=SCHEMA
    )
    request.check_context()
    with pytest.raises(ValueError, match="count"):
        list(request.stream(threading.Event()))
    with pytest.raises(ValueError):
        value.prepare_stage_request(
            [{"role": "user", "content": "x"}], stage="final_review", schema=SCHEMA
        )
    assert len(transport.calls) == 1


def test_unknown_usage_disables_route_and_mutated_payload_never_sends():
    value, transport = model()
    request = value.prepare_stage_request(
        [{"role": "user", "content": "x"}], stage="final_review", schema=SCHEMA
    )
    request.check_context()
    request._body["messages"][-1]["content"] = "changed"
    with pytest.raises(ValueError, match="Frozen"):
        list(request.stream(threading.Event()))
    assert transport.calls == []
    transport.invoke = lambda *a, **k: {
        "text": '{"ok":true}',
        "tool_calls": [],
        "finish_reason": "stop",
        "usage": None,
    }
    request = value.prepare_stage_request(
        [{"role": "user", "content": "x"}], stage="final_review", schema=SCHEMA
    )
    request.check_context()
    with pytest.raises(ValueError, match="count"):
        list(request.stream(threading.Event()))
    assert value.failed


@pytest.mark.parametrize("stage", ["fact_extract", "exchange_intent", "exchange_extract"])
def test_loopback_provider_uses_stage_budget_and_full_terminal(stage):
    import json
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from test_model_provider_contract import event, route
    from test_provider_loopback import Ledger
    from virtual_companion.providers.httpx_transport import HttpxProviderTransport
    from virtual_companion.text.provider_model import ConservativeReservation

    observed = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):  # noqa: N802
            observed.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(
                event(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": '{"ok":true}'},
                                "finish_reason": "stop",
                            }
                        ]
                    }
                )
            )
            self.wfile.write(
                event(
                    {
                        "choices": [],
                        "usage": {"prompt_tokens": 17, "completion_tokens": 5, "total_tokens": 22},
                    }
                )
            )
            self.wfile.write(b"data: [DONE]\n\n")

    service = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=service.serve_forever)
    worker.start()
    ledger = Ledger()
    try:
        configured = route(endpoint=f"http://127.0.0.1:{service.server_port}/chat/completions")
        transport = HttpxProviderTransport(
            configured, secrets=None, ledger=ConservativeReservation(ledger)
        )
        from virtual_companion.text.memory_runtime import (
            _PriorityGenerationGate,
            _StructuredMemoryInvoker,
        )

        value = ProviderModel(configured, counter=Counter(), transport=transport, output_tokens=192)
        invoke = _StructuredMemoryInvoker(value, _PriorityGenerationGate())
        assert invoke(stage, [{"role": "user", "content": "synthetic"}],
                      SCHEMA, threading.Event()) == {"ok": True}
        expected_output = 1024 if stage in {"exchange_intent", "exchange_extract"} else 192
        assert ledger.calls[0]["output_tokens"] == observed[0]["max_tokens"] == expected_output
        assert value.output_tokens == 192
        assert ledger.calls[0]["input_tokens"] == 200
        assert ledger.calls[0]["stage"] == stage
        assert ledger.finished[0]["status"] == "completed"
        assert ledger.finished[0]["usage"]["total_tokens"] == 22
        assert observed[0]["response_format"]["type"] == "json_schema"
    finally:
        service.shutdown()
        service.server_close()
        worker.join(2)


def test_worker_passes_exact_host_schema_and_stage():
    from virtual_companion.text.memory_runtime import (
        _PriorityGenerationGate,
        _StructuredMemoryInvoker,
    )

    value, transport = model()
    result = _StructuredMemoryInvoker(value, _PriorityGenerationGate())(
        "fact_source_resolution",
        [{"role": "user", "content": "synthetic"}],
        SCHEMA,
        threading.Event(),
    )
    assert result == {"ok": True}
    assert transport.calls[0][1]["stage"] == "fact_source_resolution"
    assert '"required":["ok"]' in transport.calls[0][0]["messages"][0]["content"]


def test_cumulative_allowance_includes_prior_unknown_reservations(tmp_path):
    import sqlite3

    from virtual_companion.text.provider_model import _InheritedBudget

    prior, current = tmp_path / "prior.sqlite3", tmp_path / "current.sqlite3"
    for path, count in ((prior, 99), (current, 0)):
        with sqlite3.connect(path) as db:
            db.execute("CREATE TABLE provider_calls(input_tokens INTEGER, output_tokens INTEGER)")
            db.execute("INSERT INTO provider_calls VALUES (?,0)", (count,))
    ledger = type("Ledger", (), {"path": current})()
    value = _InheritedBudget(ledger, [prior], 100)
    value.check(1)
    with pytest.raises(ValueError, match="Cumulative"):
        value.check(2)


@pytest.mark.parametrize("stage", ["exchange_intent", "exchange_extract"])
def test_memory_output_override_is_frozen_for_context_and_wire(stage):
    from dataclasses import replace

    value, transport = model()
    value.output_tokens = 192
    value.route = replace(value.route, capabilities=replace(
        value.route.capabilities, context_tokens=530))
    messages = [{"role": "user", "content": "synthetic"}]
    ordinary = value.prepare_stage_request(messages, stage="demand_control", schema=SCHEMA)
    request = value.prepare_stage_request(messages, stage=stage, schema=SCHEMA,
                                          output_tokens=512)
    assert ordinary._body["max_tokens"] == 192
    value.output_tokens = 4096
    request.check_context()
    assert list(request.stream(threading.Event())) == ['{"ok":true}']
    body, options = transport.calls[0]
    assert body["max_tokens"] == 512
    assert options["stage"] == stage
    assert value.output_tokens == 4096
    with pytest.raises(ValueError, match="consumed"):
        list(request.stream(threading.Event()))
    assert len(transport.calls) == 1


def test_memory_output_override_keeps_one_context_position_free():
    from dataclasses import replace

    value, transport = model()
    value.route = replace(value.route, capabilities=replace(
        value.route.capabilities, context_tokens=529))
    request = value.prepare_stage_request(
        [{"role": "user", "content": "synthetic"}], stage="exchange_intent",
        schema=SCHEMA, output_tokens=512)
    value.output_tokens = 1
    with pytest.raises(ValueError, match="context"):
        request.check_context()
    assert transport.calls == []


@pytest.mark.parametrize("output_tokens", [True, 0, 4097])
def test_provider_stage_rejects_unbounded_output_override(output_tokens):
    value, transport = model()
    with pytest.raises(ValueError, match="output"):
        value.prepare_stage_request(
            [{"role": "user", "content": "synthetic"}], stage="exchange_intent",
            schema=SCHEMA, output_tokens=output_tokens)
    assert transport.calls == []


@pytest.mark.parametrize("terminal", ["length", "cancelled"])
def test_memory_override_preserves_unfinished_rejection_without_retry(terminal):
    value, transport = model()
    signal = threading.Event()
    original_invoke = transport.invoke

    def invoke(*args, **kwargs):
        result = original_invoke(*args, **kwargs)
        if terminal == "cancelled":
            signal.set()
        else:
            result["finish_reason"] = terminal
        return result

    transport.invoke = invoke
    request = value.prepare_stage_request(
        [{"role": "user", "content": "synthetic"}], stage="exchange_intent",
        schema=SCHEMA, output_tokens=512)
    request.check_context()
    expected = InterruptedError if terminal == "cancelled" else ValueError
    with pytest.raises(expected):
        list(request.stream(signal))
    with pytest.raises(ValueError, match="consumed"):
        list(request.stream(signal))
    assert len(transport.calls) == 1


@pytest.fixture
def provisional_provider(tmp_path, monkeypatch):
    """Real loopback I/O; finished observes Httpx's own settlement/EOF boundary."""
    from test_model_provider_contract import event, route
    from test_provider_loopback import Ledger
    from virtual_companion.providers import httpx_transport
    from virtual_companion.providers.httpx_transport import HttpxProviderTransport
    from virtual_companion.text.provider_model import ConservativeReservation

    @contextmanager
    def run(parts, *, stage="demand_control", timeout=3, finish_reason="stop", done=True,
            split_terminal=False, coalesce_raw=False, pause_before_terminal=False,
            pause_after_first=False):
        sent, finished, first_received = threading.Event(), threading.Event(), threading.Event()
        release_terminal, abort_server = threading.Event(), threading.Event()
        observed, marks, checkpoints, deliveries = [], [], [], []
        raw_events, server_events = [], []
        call_id, pid = "synthetic-loopback-call-1", os.getpid()
        content_parts = [event({"choices": [{
            "index": 0, "delta": {"content": part}, "finish_reason": None,
        }]}) for part in parts]
        prefix = b"".join(content_parts[:1] if pause_after_first else content_parts)
        tail = event({"choices": [{
            "index": 0, "delta": {}, "finish_reason": finish_reason,
        }]})
        tail += event({
            "choices": [],
            "usage": {"prompt_tokens": 17, "completion_tokens": 5, "total_tokens": 22},
        })
        if done:
            tail += b"data: [DONE]\n\n"
        if pause_after_first:
            tail = b"".join(content_parts[1:]) + tail
        payload = prefix + tail

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):  # noqa: N802
                observed.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                try:
                    # Only the explicit slow-server case gates production of terminal bytes.
                    if split_terminal or pause_before_terminal or pause_after_first:
                        self.wfile.write(prefix)
                        self.wfile.flush()
                        # Split only at first socket receipt, never at consumer ACK.
                        assert first_received.wait(1)
                        if pause_before_terminal or pause_after_first:
                            while not release_terminal.wait(0.01):
                                if abort_server.is_set():
                                    return
                            if abort_server.is_set():
                                return
                        server_events.append({"event": "terminal_write_attempt",
                                              "monotonic_ns": str(time.monotonic_ns())})
                        self.wfile.write(tail)
                    else:
                        server_events.append({"event": "terminal_write_attempt",
                                              "monotonic_ns": str(time.monotonic_ns())})
                        self.wfile.write(payload)
                    self.wfile.flush()
                    server_events.append({"event": "terminal_write_finished",
                                          "monotonic_ns": str(time.monotonic_ns())})
                    sent.set()
                except OSError:
                    if not abort_server.is_set():
                        raise

        class ObservedTransport(HttpxProviderTransport):
            def invoke(self, *args, **kwargs):
                on_content = kwargs["on_content"]

                def deliver(delta):
                    delivery = {"text": delta.text, "thread": threading.get_ident(),
                                "entered_at": time.monotonic(), "exited_at": None}
                    deliveries.append(delivery)
                    try:
                        on_content(delta)
                    finally:
                        delivery["exited_at"] = time.monotonic()

                kwargs["on_content"] = deliver
                return super().invoke(*args, **kwargs)

            def _mark(self, **values):
                super()._mark(**values)
                marks.append({"thread": threading.get_ident(), **values})
                if values.get("first_content") is not None:
                    first_received.set()
                if values.get("finished") is not None:
                    finished.set()

        service = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        service.daemon_threads = True
        server_worker = threading.Thread(
            target=lambda: service.serve_forever(poll_interval=0.01),
            name="synthetic-provider-http", daemon=True,
        )
        server_worker.start()
        configured = route(endpoint=f"http://127.0.0.1:{service.server_port}/chat/completions")
        ledger = Ledger()
        transport = ObservedTransport(
            configured, secrets=None, ledger=ConservativeReservation(ledger),
        )
        value = ProviderModel(configured, counter=Counter(), transport=transport,
                              timeout_seconds=timeout)
        request = value.prepare_stage_request(
            [{"role": "user", "content": "synthetic handshake"}], stage=stage, schema=SCHEMA,
        )
        request.check_context()
        signal = threading.Event()
        iterator = request.stream(signal)
        threads_before = set(threading.enumerate())
        original_stream = httpx_transport.httpx.AsyncClient.stream

        @asynccontextmanager
        async def observed_stream(client, *args, **kwargs):
            async with original_stream(client, *args, **kwargs) as response:
                original_iter = response.aiter_bytes

                async def observed_bytes(*read_args, **read_kwargs):
                    pending, chunk_index = b"", 0
                    if coalesce_raw:
                        read_kwargs["chunk_size"] = len(payload)
                    async for chunk in original_iter(*read_args, **read_kwargs):
                        chunk_index += 1
                        raw_events.append({"event": "raw_chunk", "chunk": chunk_index,
                                           "bytes": len(chunk), "call_id": call_id,
                                           "pid": pid, "monotonic_ns": str(time.monotonic_ns())})
                        pending += chunk
                        while b"\n" in pending:
                            line, pending = pending.split(b"\n", 1)
                            if not line.startswith(b"data: "):
                                continue
                            data = line[6:].strip()
                            if data == b"[DONE]":
                                fields = {"event": "sse_done"}
                            else:
                                record = json.loads(data)
                                choices = record.get("choices", [])
                                reason = choices[0].get("finish_reason") if choices else None
                                if not reason:
                                    continue
                                fields = {"event": "sse_stop", "finish_reason": reason}
                            raw_events.append({**fields, "chunk": chunk_index, "call_id": call_id,
                                               "pid": pid,
                                               "monotonic_ns": str(time.monotonic_ns())})
                        # Match the production observer: record SSE before yielding bytes.
                        yield chunk

                response.aiter_bytes = observed_bytes
                yield response

        monkeypatch.setattr(httpx_transport.httpx.AsyncClient, "stream", observed_stream)

        def record(name):
            checkpoints.append({"name": name, "at": time.monotonic(),
                                "metrics": transport.metrics, "settlements": list(ledger.finished)})

        def settled(status):
            assert finished.wait(1), "Actual Httpx transport never settled"
            assert transport.metrics["finished"] is not None
            assert len(ledger.calls) == len(ledger.finished) == 1
            assert ledger.calls[0]["request_sha256"] == request._fingerprint
            assert ledger.calls[0]["input_tokens"] == 200
            assert ledger.calls[0]["output_tokens"] == 512
            assert ledger.calls[0]["stage"] == stage
            assert ledger.finished[0]["receipt"] == "1"
            assert ledger.finished[0]["status"] == status
            for thread in set(threading.enumerate()) - threads_before:
                if thread.name == "desktop-provider-stream":
                    thread.join(1)
                    assert not thread.is_alive(), "Provider worker leaked"
            assert transport._loop is None and transport._task is None
            assert not transport._active.locked()
            assert value._streams == set()
            record("settled:" + status)

        state = SimpleNamespace(iterator=iterator, signal=signal, model=value,
                                transport=transport, ledger=ledger, sent=sent,
                                finished=finished, record=record, settled=settled,
                                deliveries=deliveries, raw_events=raw_events,
                                server_events=server_events, call_id=call_id, pid=pid,
                                first_received=first_received, release_terminal=release_terminal)
        try:
            yield state
        finally:
            iterator.close()
            abort_server.set()
            release_terminal.set()
            monkeypatch.setattr(httpx_transport.httpx.AsyncClient, "stream", original_stream)
            service.shutdown()
            service.server_close()
            server_worker.join(1)
            assert not server_worker.is_alive()
            (tmp_path / "actual-transport-evidence.json").write_text(json.dumps({
                "stage": stage, "timeout": timeout, "parts": parts,
                "split_terminal": split_terminal,
                "coalesce_raw": coalesce_raw,
                "pause_before_terminal": pause_before_terminal,
                "pause_after_first": pause_after_first,
                "server_sent_all_bytes": sent.is_set(), "requests": observed,
                "transport_marks": marks, "checkpoints": checkpoints,
                "synchronous_callbacks": deliveries,
                "raw_events_before_yield": raw_events, "server_events": server_events,
                "reservations": ledger.calls, "settlements": ledger.finished,
                "transport_metrics": transport.metrics,
                "live_api_attempts": 0, "credentials_read": 0,
            }, indent=2) + "\n", encoding="utf-8")

    return run


@pytest.mark.parametrize("stage", ["demand_control", "demand_repair"])
def test_provisional_deltas_preserve_order_without_pausing_actual_transport_terminal(
    provisional_provider, stage,
):
    parts = ['{"ok":', "tr", "ue}"]
    with provisional_provider(parts, stage=stage) as state:
        for index, expected in enumerate(parts):
            assert next(state.iterator) == expected
            assert state.sent.wait(1)
            state.record("consumer_holds_delta:" + str(index))
            assert state.finished.wait(1), "I/O completion must not await semantic consumption"
            assert state.transport.metrics["finished"] is not None
            assert len(state.ledger.finished) == 1
            assert len(state.deliveries) == len(parts)
            assert all(row["exited_at"] is not None for row in state.deliveries)
            assert state.deliveries[-1]["thread"] != threading.get_ident()
        assert list(state.iterator) == []
        state.settled("completed")
        assert [delivery["text"] for delivery in state.deliveries] == parts
        assert state.ledger.finished[0]["usage"]["total_tokens"] == 22


def test_provisional_one_chunk_response_finishes_while_consumer_is_paused(provisional_provider):
    with provisional_provider(['{"ok":true}']) as state:
        assert next(state.iterator) == '{"ok":true}'
        assert state.sent.wait(1)
        state.record("consumer_holds_only_delta")
        assert state.finished.wait(1)
        assert list(state.iterator) == []
        state.settled("completed")
        assert state.ledger.finished[0]["usage"]["total_tokens"] == 22


@pytest.mark.parametrize("interrupt", ["signal", "model_cancel", "close", "throw"])
def test_provisional_held_delta_cancellation_or_close_settles_once_without_worker_leak(
    provisional_provider, interrupt,
):
    from virtual_companion.providers.httpx_transport import ProviderFailure

    with provisional_provider(['{"ok":', "true}"], pause_before_terminal=True) as state:
        assert next(state.iterator) == '{"ok":'
        assert state.first_received.wait(1)
        assert not state.sent.is_set()
        state.record("consumer_holds_first_delta")
        assert not state.finished.wait(0.1)
        if interrupt == "signal":
            state.signal.set()
            assert state.finished.wait(1), "Cancellation required consumer resumption"
            with pytest.raises(InterruptedError):
                next(state.iterator)
        elif interrupt == "model_cancel":
            state.model.cancel()
            assert state.finished.wait(1), "Model cancellation required consumer resumption"
            with pytest.raises(ProviderFailure, match="provider_cancelled"):
                next(state.iterator)
        elif interrupt == "close":
            state.iterator.close()
        else:
            with pytest.raises(ValueError, match="synthetic consumer failure"):
                state.iterator.throw(ValueError("synthetic consumer failure"))
        state.settled("cancelled")
        assert state.ledger.finished[0]["usage"] is None


def test_provisional_held_delta_deadline_settles_without_consumer_resumption(provisional_provider):
    from virtual_companion.providers.httpx_transport import ProviderFailure

    with provisional_provider(['{"ok":', "true}"], timeout=0.75,
                              pause_before_terminal=True) as state:
        assert next(state.iterator) == '{"ok":'
        assert state.first_received.wait(1)
        assert not state.sent.is_set()
        state.record("consumer_holds_until_deadline")
        assert not state.finished.wait(0.1)
        assert state.finished.wait(1.5), "Deadline required consumer resumption"
        with pytest.raises(ProviderFailure, match="provider_deadline_exceeded"):
            next(state.iterator)
        state.settled("deadline_exceeded")
        assert state.ledger.finished[0]["usage"] is None


@pytest.mark.parametrize("failure", ["length", "missing_done"])
def test_provisional_stream_keeps_full_terminal_rejection(provisional_provider, failure):
    from virtual_companion.providers.httpx_transport import ProviderFailure

    options = {"finish_reason": "length"} if failure == "length" else {"done": False}
    with provisional_provider(['{"ok":true}'], pause_before_terminal=True, **options) as state:
        assert next(state.iterator) == '{"ok":true}'
        assert not state.finished.wait(0.1)
        state.release_terminal.set()
        assert state.sent.wait(1)
        assert state.finished.wait(1), "Invalid terminal must settle without consumer resumption"
        with pytest.raises(ProviderFailure):
            next(state.iterator)
        state.settled("failed")
        assert state.ledger.finished[0]["usage"]["total_tokens"] == 22
