"""Independent bounded I/O completion; synthetic real HTTP and durable host ACK only."""

import asyncio
import json
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import FrozenInstanceError

import pytest
from test_default_generation_task_contract import PLAIN
from test_default_generation_task_contract import provider_host as provider_host
from test_default_text_delivery import wait_for
from test_text_provider_model import provisional_provider as provisional_provider
from unit_review_recording import bound_unit_packets as bound_unit_packets
from virtual_companion.providers.contracts import ProviderStreamCapacityError
from virtual_companion.providers.httpx_transport import ProviderFailure
from virtual_companion.text import provider_model
from virtual_companion.text.provider_model import _ContentBuffer, _ProducerTerminal


@pytest.mark.parametrize("wire", ["same_chunk", "split_chunk"])
def test_io_terminal_does_not_wait_for_paused_consumer(provisional_provider, wire):
    with provisional_provider(["{\"ok\":true}"], split_terminal=wire == "split_chunk",
                              coalesce_raw=wire == "same_chunk") as state:
        assert next(state.iterator) == '{"ok":true}'
        assert state.sent.wait(1), "Synthetic server must finish independently of the consumer"
        state.record("consumer_paused_after_first_delta")
        assert state.finished.wait(0.5), "I/O terminal still depends on consumer resumption"
        assert state.transport.metrics["finished"] is not None
        assert len(state.ledger.finished) == 1
        assert state.ledger.finished[0]["status"] == "completed"
        assert state.ledger.finished[0]["usage"]["total_tokens"] == 22
        assert len([row for row in state.raw_events if row["event"] == "sse_stop"]) == 1
        assert len([row for row in state.raw_events if row["event"] == "sse_done"]) == 1
        assert all(row["exited_at"] is not None for row in state.deliveries)
        assert list(state.iterator) == []
        state.settled("completed")


@pytest.mark.parametrize("limit", ["bytes", "events"])
def test_buffer_capacity_is_exact_and_does_not_wait_for_consumer(limit):
    buffer = _ContentBuffer(4 if limit == "bytes" else 20, 1 if limit == "events" else 2)
    buffer.append("🌿")
    before = time.monotonic()
    with pytest.raises(ProviderStreamCapacityError):
        buffer.append("x")
    assert time.monotonic() - before < 0.1
    assert buffer.take(0) == ("delta", "🌿")
    buffer.append("x")
    terminal = _ProducerTerminal(text="🌿x")
    buffer.finish(terminal)
    assert buffer.take(0) == ("delta", "x")
    assert buffer.take(0) == ("terminal", terminal)
    with pytest.raises(FrozenInstanceError):
        terminal.text = "changed"
    with pytest.raises(ValueError, match="already finished"):
        buffer.finish(_ProducerTerminal(text="changed"))
    with pytest.raises(InterruptedError, match="closed"):
        buffer.append("late")


def test_producer_error_discards_unconsumed_content_and_preserves_the_original_error():
    buffer = _ContentBuffer(20, 2)
    buffer.append("unapproved")
    error = ValueError("synthetic producer failure")
    terminal = _ProducerTerminal(error=error)
    buffer.finish(terminal)
    assert not buffer._events and buffer._bytes == 0
    assert buffer.take(0) == ("terminal", terminal)
    assert terminal.error is error
    buffer.close()
    with pytest.raises(InterruptedError, match="closed"):
        buffer.take(0)


def test_default_event_capacity_accepts_4096_and_rejects_the_next_pending_event():
    assert provider_model.MAX_PENDING_STREAM_EVENTS == 4096
    buffer = _ContentBuffer(provider_model.MAX_PROVIDER_RESPONSE_BYTES,
                            provider_model.MAX_PENDING_STREAM_EVENTS)
    for index in range(4096):
        buffer.append(str(index))
    with pytest.raises(ProviderStreamCapacityError):
        buffer.append("overflow")
    for index in range(4096):
        assert buffer.take(0) == ("delta", str(index))
    terminal = _ProducerTerminal(text="synthetic completed producer")
    buffer.finish(terminal)
    assert buffer.take(0) == ("terminal", terminal)


def test_pending_bytes_share_the_existing_httpx_response_bound():
    from virtual_companion.providers import contracts, httpx_transport

    assert contracts.MAX_PROVIDER_RESPONSE_BYTES == 2_000_000
    assert provider_model.MAX_PROVIDER_RESPONSE_BYTES == contracts.MAX_PROVIDER_RESPONSE_BYTES
    assert httpx_transport.MAX_PROVIDER_RESPONSE_BYTES == contracts.MAX_PROVIDER_RESPONSE_BYTES
    buffer = _ContentBuffer(contracts.MAX_PROVIDER_RESPONSE_BYTES, 2)
    buffer.append("a" * 2_000_000)
    assert buffer._bytes == 2_000_000
    with pytest.raises(ProviderStreamCapacityError):
        buffer.append("x")
    buffer.close()
    assert buffer._bytes == 0 and not buffer._events


def test_real_httpx_response_over_shared_bound_keeps_unknown_reservation(provisional_provider):
    with provisional_provider(["a" * 2_000_001], coalesce_raw=True) as state:
        with pytest.raises(ProviderFailure, match="provider_response_too_large"):
            next(state.iterator)
        state.settled("failed")
        assert state.deliveries == []
        assert state.ledger.finished[0]["usage"] is None
        assert state.ledger.calls[0]["input_tokens"] == 200
        assert state.ledger.calls[0]["output_tokens"] == 512


@pytest.mark.parametrize("stage", ["demand_control", "demand_repair"])
def test_actual_focused_profile_uses_bounded_buffer_and_the_same_frozen_wire(
    tmp_path, monkeypatch, stage,
):
    from test_deepseek_focused import (
        MESSAGES,
        SCHEMA,
        assert_frozen_send,
        focused_model,
        persist_recording,
        recording_loopback,
    )

    buffers = []
    original = provider_model._ContentBuffer

    def record_buffer(max_bytes, max_events):
        buffer = original(max_bytes, max_events)
        buffers.append(buffer)
        return buffer

    monkeypatch.setattr(provider_model, "_ContentBuffer", record_buffer)
    value, ledger = focused_model(tmp_path)
    with recording_loopback(monkeypatch, ledger) as (raw, network, _, _):
        request = value.prepare_stage_request(MESSAGES, stage=stage, schema=SCHEMA)
        request.check_context()
        iterator = request.stream(threading.Event())
        assert next(iterator) == '{"answer":"synthetic"}'
        wait_for(lambda: ledger.records()[0]["status"] == "completed")
        assert_frozen_send(request, ledger.records(), raw, network)
        assert len(buffers) == 1
        assert buffers[0].max_bytes == 2_000_000 and buffers[0].max_events == 4096
        wait_for(lambda: buffers[0].terminal is not None)
        terminal = buffers[0].terminal
        assert terminal.error is None and terminal.text == '{"answer":"synthetic"}'
        assert value.transport.metrics["finished"] is not None
        assert list(iterator) == []
        assert not value._streams and not value.transport._active.locked()
    assert ledger.summary()["unknown_usage_calls"] == 0
    persist_recording(tmp_path, ledger.records(), raw, network)


@pytest.mark.parametrize("limit", ["bytes", "events"])
def test_httpx_buffer_overflow_aborts_without_pending_release_or_refunded_reservation(
    provisional_provider, monkeypatch, limit,
):
    if limit == "bytes":
        monkeypatch.setattr(provider_model, "MAX_PROVIDER_RESPONSE_BYTES", 4)
        parts = ["a", "🌿", "b"]
    else:
        monkeypatch.setattr(provider_model, "MAX_PENDING_STREAM_EVENTS", 1)
        parts = ["a", "b", "c"]
    with provisional_provider(parts, pause_after_first=True) as state:
        assert next(state.iterator) == "a"
        assert not state.finished.is_set()
        state.release_terminal.set()
        assert state.sent.wait(1)
        assert state.finished.wait(1), "Buffer overflow must end I/O while consumption is paused"
        with pytest.raises(ProviderFailure, match="provider_stream_buffer_capacity_exceeded"):
            next(state.iterator)
        state.settled("failed")
        assert state.ledger.finished[0]["usage"] is None
        assert state.ledger.finished[0]["transmission"] == "response_received"
        assert state.ledger.calls[0]["input_tokens"] == 200
        assert state.ledger.calls[0]["output_tokens"] == 512


@pytest.mark.parametrize("interrupt", ["signal", "model_cancel", "close", "throw"])
def test_late_consumer_cancellation_discards_pending_content_without_rewriting_io_terminal(
    provisional_provider, interrupt,
):
    with provisional_provider(['{"ok":', "true}"]) as state:
        assert next(state.iterator) == '{"ok":'
        assert state.finished.wait(1)
        before = list(state.ledger.finished)
        if interrupt == "signal":
            state.signal.set()
            with pytest.raises(InterruptedError):
                next(state.iterator)
        elif interrupt == "model_cancel":
            state.model.cancel()
            with pytest.raises(InterruptedError):
                next(state.iterator)
        elif interrupt == "close":
            state.iterator.close()
        else:
            with pytest.raises(ValueError, match="synthetic consumer failure"):
                state.iterator.throw(ValueError("synthetic consumer failure"))
        assert list(state.iterator) == []
        state.settled("completed")
        assert state.ledger.finished == before
        assert before[0]["usage"]["total_tokens"] == 22


def test_zero_content_rejects_empty_fallback_and_preserves_reported_usage(provisional_provider):
    with provisional_provider([]) as state:
        with pytest.raises(ProviderFailure, match="provider_protocol_rejected"):
            next(state.iterator)
        state.settled("failed")
        assert state.deliveries == []
        assert state.ledger.finished[0]["usage"]["total_tokens"] == 22


@pytest.mark.parametrize("stage", ["demand_control", "demand_repair"])
def test_completed_io_buffer_still_obeys_the_original_consumer_deadline(
    provisional_provider, stage,
):
    with provisional_provider(['{"ok":', "true}"], stage=stage, timeout=0.5) as state:
        assert next(state.iterator) == '{"ok":'
        assert state.finished.wait(0.5)
        before = list(state.ledger.finished)
        assert before[0]["status"] == "completed"
        assert before[0]["usage"]["total_tokens"] == 22
        threading.Event().wait(0.6)
        with pytest.raises(TimeoutError, match="Provider request deadline"):
            next(state.iterator)
        assert list(state.iterator) == []
        state.settled("completed")
        assert state.ledger.finished == before


def test_actual_host_cancellation_keeps_confirmed_prefix_and_cancels_pending_producer(
    provider_host,
):
    with provider_host(revision="short-dialogue.v23", ending="cancel") as state:
        state.session.send(PLAIN)
        assert state.prefix_received.wait(2)
        event = None
        until = time.monotonic() + 2
        while event is None and time.monotonic() < until:
            event = state.runtime.poll()["event"]
            time.sleep(0.005)
        assert event is not None, "Actual sealed review must approve and dispatch the prefix"
        assert event["sequence"] == 1 and event["display_text"] == "合成先导首句。"
        assert state.runtime.generation_terminal_us is None
        assert not any(item["event"] == "server_response_finished"
                       and item["kind"] == "generation" for item in state.timeline)
        host_receipt = "synthetic-independent-io-visible-" + event["basis"]["input_id"]
        assert state.runtime.ack({"sequence": 1, "fingerprint": event["fingerprint"],
                                  "host_receipt": host_receipt}) == {"status": "confirmed"}
        durable = state.memory.store.stream_state(state.memory.companion,
                                                  state.runtime.claim.input_id)
        assert durable["confirmed_units"] == 1
        assert durable["confirmed_text"] == event["display_text"]
        state.mark("host_ack_returned", sequence=1)
        state.runtime.stop()
        state.session.stop()
        wait_for(lambda: not state.session.snapshot()["busy"])
        durable = state.memory.store.stream_state(state.memory.companion,
                                                  state.runtime.claim.input_id)
        assert state.session.snapshot()["attempt"]["status"] == durable["outcome"] == "cancelled"
        assert durable["approved_units"] == durable["confirmed_units"] == 1
        assert durable["confirmed_text"] == "合成先导首句。"
        assert state.runtime.poll()["event"] is None
        assert state.mirror.history()["messages"][-1]["content"] == "合成先导首句。"
        assert [item["stage"] for item in state.ledger.calls] \
            == ["demand_control", "stream_unit_review"]
        assert len(state.ledger.finished) == 2
        by_receipt = {item["receipt"]: item for item in state.ledger.finished}
        assert by_receipt["1"]["status"] == "cancelled" and by_receipt["1"]["usage"] is None
        assert by_receipt["2"]["status"] == "completed"
        assert by_receipt["2"]["usage"]["total_tokens"] == 22
        assert not any(thread.name == "desktop-provider-stream" and thread.is_alive()
                       for thread in threading.enumerate())


def test_actual_host_consumer_deadline_keeps_confirmed_prefix_and_known_io_usage(
    provider_host, monkeypatch,
):
    from virtual_companion.text.stream_host import StreamHostBridge

    resume, ack_returned = threading.Event(), threading.Event()
    with provider_host(revision="short-dialogue.v23", ending="cancel") as state:
        state.runtime.model.timeout_seconds = 0.8
        original_present = StreamHostBridge.present

        def hold_after_ack(bridge, *args, **kwargs):
            result = original_present(bridge, *args, **kwargs)
            ack_returned.set()
            assert resume.wait(2), "Synthetic consumer deadline gate never released"
            return result

        monkeypatch.setattr(StreamHostBridge, "present", hold_after_ack)
        try:
            state.session.send(PLAIN)
            assert state.prefix_received.wait(2)
            event = wait_for(lambda: state.runtime.poll()["event"])
            assert event["display_text"] == "合成先导首句。"
            host_receipt = "synthetic-independent-deadline-visible-" + event["basis"]["input_id"]
            assert state.runtime.ack({"sequence": 1, "fingerprint": event["fingerprint"],
                                      "host_receipt": host_receipt}) == {"status": "confirmed"}
            assert ack_returned.wait(1)
            state.release.set()
            wait_for(lambda: len(state.ledger.finished) == 2)
            before = list(state.ledger.finished)
            assert all(item["status"] == "completed" for item in before)
            assert state.runtime.generation_terminal_us is None
            threading.Event().wait(1)
            resume.set()
            wait_for(lambda: not state.session.snapshot()["busy"])
            durable = state.memory.store.stream_state(state.memory.companion,
                                                      state.runtime.claim.input_id)
            assert state.session.snapshot()["attempt"]["status"] == durable["outcome"] == "failed"
            assert durable["approved_units"] == durable["confirmed_units"] == 1
            assert durable["confirmed_text"] == "合成先导首句。"
            assert state.runtime.poll()["event"] is None
            assert state.mirror.history()["messages"][-1]["content"] == "合成先导首句。"
            assert [item["stage"] for item in state.ledger.calls] \
                == ["demand_control", "stream_unit_review"]
            assert state.ledger.finished == before
            assert all(item["usage"]["total_tokens"] == 22 for item in before)
            assert state.runtime.snapshot()["retry_allowed"] is False
        finally:
            resume.set()


def test_actual_host_cancel_ends_pending_generation_and_same_ledger_review_peer(
    provider_host, monkeypatch,
):
    from virtual_companion.providers import httpx_transport

    review_arrived, release_review = threading.Event(), threading.Event()
    original_stream = httpx_transport.httpx.AsyncClient.stream
    original_peer = httpx_transport.HttpxProviderTransport.review_peer
    peers = []

    def record_peer(transport):
        peer = original_peer(transport)
        peers.append(peer)
        return peer

    @asynccontextmanager
    async def hold_review_bytes(client, *args, **kwargs):
        async with original_stream(client, *args, **kwargs) as response:
            body = json.loads(kwargs["content"])
            if "你只审查候选文字" in body["messages"][0]["content"]:
                original_iter = response.aiter_bytes

                async def pending_review(*read_args, **read_kwargs):
                    review_arrived.set()
                    while not release_review.is_set():
                        await asyncio.sleep(0.005)
                    async for chunk in original_iter(*read_args, **read_kwargs):
                        yield chunk

                response.aiter_bytes = pending_review
            yield response

    monkeypatch.setattr(httpx_transport.httpx.AsyncClient, "stream", hold_review_bytes)
    monkeypatch.setattr(httpx_transport.HttpxProviderTransport, "review_peer", record_peer)
    try:
        with provider_host(revision="short-dialogue.v23", ending="cancel") as state:
            state.session.send(PLAIN)
            assert review_arrived.wait(2), "Actual same-route review must be reserved and sent"
            assert len(state.ledger.calls) == 2 and state.ledger.finished == []
            assert state.runtime.poll()["event"] is None
            state.runtime.stop()
            state.session.stop()
            wait_for(lambda: not state.session.snapshot()["busy"])
            wait_for(lambda: len(state.ledger.finished) == 2)
            assert [item["stage"] for item in state.ledger.calls] \
                == ["demand_control", "stream_unit_review"]
            assert len({item["route"] for item in state.ledger.calls}) == 1
            assert all(item["status"] == "cancelled" and item["usage"] is None
                       for item in state.ledger.finished)
            durable = state.memory.store.stream_state(state.memory.companion,
                                                      state.runtime.claim.input_id)
            assert durable["confirmed_units"] == durable["approved_units"] == 0
            assert durable["confirmed_text"] == "" and state.runtime.poll()["event"] is None
            assert len(peers) == 1 and peers[0].ledger.ledger is state.ledger
            for transport in (state.runtime.model.transport, peers[0]):
                assert transport._loop is None and transport._task is None
                assert not transport._active.locked()
            assert not any(thread.name == "desktop-provider-stream" and thread.is_alive()
                           for thread in threading.enumerate())
    finally:
        release_review.set()
