"""Offline focused-profile contracts; recording loopback is not a DeepSeek effect test."""

import json
import threading
from contextlib import contextmanager
from dataclasses import replace
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
from test_deepseek_provider import MESSAGES, SCHEMA, NeverLedger, NeverSecrets, deepseek_route
from test_rebuilt_turns import claim, config
from test_text_provider_bootstrap import application as application
from test_text_provider_bootstrap import runtime_observer as runtime_observer
from test_text_provider_configuration import configured as configured
from test_text_provider_model import Counter
from unit_review_recording import bound_unit_packets as bound_unit_packets
from unit_review_recording import recorded_review_packet
from virtual_companion.boundary.contracts import canonical
from virtual_companion.boundary.sqlite_streaming import SQLiteStreamingStore
from virtual_companion.cognition.short_dialogue import ShortRequest
from virtual_companion.cognition.stream_unit_review import (
    BoundedUnitReviewer,
    CandidateRejected,
    StructuredUnitAssessor,
)
from virtual_companion.providers.budget import ProviderBudgetLedger
from virtual_companion.providers.httpx_transport import HttpxProviderTransport, ProviderFailure
from virtual_companion.providers.openai_compatible import build_request, validate_wire_profile
from virtual_companion.text import __main__ as bootstrap
from virtual_companion.text.fake import FakeModel
from virtual_companion.text.provider_counter import DeepSeekCounter, _render
from virtual_companion.text.provider_model import (
    ConservativeReservation,
    ProviderModel,
    load_provider_model,
)

FOCUSED = "deepseek-flash.focused.v2"
TEMPERATURES = {
    "demand_control": 0.2, "demand_repair": 0.2,
    "stream_unit_review": 0, "fact_target_select": 0, "fact_extract": 0,
    "fact_support": 0, "fact_repair": 0, "fact_assess": 0,
    "exchange_intent": 0, "exchange_extract": 0, "exchange_assess": 0,
    "exchange_support": 0, "source_propose": 0, "source_repair": 0,
    "memory_command": 0, "memory_command_support": 0,
}


def focused_route(**changes):
    return deepseek_route(wire_profile=FOCUSED, **changes)


@pytest.mark.parametrize("stage,temperature", TEMPERATURES.items())
def test_focused_stage_owns_exact_sampling_before_freeze(stage, temperature):
    old = build_request(deepseek_route(), MESSAGES, output_tokens=100, schema=SCHEMA)
    body = build_request(focused_route(), MESSAGES, output_tokens=100, schema=SCHEMA, stage=stage)
    assert body == {**old, "temperature": temperature}
    validate_wire_profile(focused_route(), body, stage=stage)
    assert not {"seed", "top_p", "reasoning_effort"}.intersection(body)


@pytest.mark.parametrize("stage", [None, "unknown", "final_review", "source_read", "fact_targets",
                                   "exchange_context", "commit", "fact_extract_extra"])
def test_focused_unknown_or_host_only_stage_rejects_before_preflight(stage):
    with pytest.raises(ValueError, match="stage"):
        build_request(focused_route(), MESSAGES, output_tokens=100, stage=stage)


@pytest.mark.parametrize("mutation", ["missing", "wrong_stage", "bool", "nan", "extra",
                                     "seed", "top_p", "thinking"])
def test_focused_transport_matches_this_stage_before_credentials_or_budget(mutation):
    route = focused_route()
    body = build_request(route, MESSAGES, output_tokens=100, schema=SCHEMA,
                         stage="stream_unit_review")
    if mutation == "missing":
        del body["temperature"]
    elif mutation == "wrong_stage":
        body["temperature"] = 0.2
    elif mutation == "bool":
        body["temperature"] = False
    elif mutation == "nan":
        body["temperature"] = float("nan")
    elif mutation == "extra":
        body["unapproved_sampling"] = 0
    elif mutation in {"seed", "top_p"}:
        body[mutation] = 1
    else:
        body["thinking"] = {"type": "enabled"}
    transport = HttpxProviderTransport(route, secrets=NeverSecrets(), ledger=NeverLedger())
    with pytest.raises(ValueError):
        transport.invoke(body, threading.Event(), stage="stream_unit_review",
                         data_scope=("current",), input_tokens=100, timeout_seconds=10)


def test_focused_stage_cannot_be_relabelled_at_transport():
    route = focused_route()
    body = build_request(route, MESSAGES, output_tokens=100, stage="demand_control")
    transport = HttpxProviderTransport(route, secrets=NeverSecrets(), ledger=NeverLedger())
    for stage in ("fact_repair", "unknown"):
        with pytest.raises(ValueError, match="stage|profile"):
            transport.invoke(body, threading.Event(), stage=stage, data_scope=("current",),
                             input_tokens=100, timeout_seconds=10)


def test_profile_bound_counter_keeps_temperature_out_of_rendered_input():
    old = build_request(deepseek_route(), MESSAGES, output_tokens=100, schema=SCHEMA)
    original = _render(old, ())
    for stage in ("demand_control", "stream_unit_review"):
        body = build_request(focused_route(), MESSAGES, output_tokens=100, schema=SCHEMA,
                             stage=stage)
        assert _render(body, (), wire_profile=FOCUSED) == original
        with pytest.raises(ValueError):
            _render(body, ())
    # Use the real counter public surface without loading a remote/tokenizer artifact.
    counter = object.__new__(DeepSeekCounter)
    counter.forbidden_tokens, counter.wire_profile = (), FOCUSED
    assert counter.render(body) == original


@pytest.mark.parametrize("field,value", [("temperature", 1), ("temperature", False),
                                       ("seed", 1), ("top_p", 1), ("tools", []),
                                       ("extra_body", {}), ("reasoning_effort", "none")])
def test_focused_counter_never_discards_other_unsupported_fields(field, value):
    body = build_request(focused_route(), MESSAGES, output_tokens=100, stage="stream_unit_review")
    body[field] = value
    with pytest.raises(ValueError):
        _render(body, (), wire_profile=FOCUSED)


def test_managed_configuration_changes_route_config_and_budget_identity(configured, monkeypatch):
    from virtual_companion.text import provider_counter

    values, write, _, root = configured
    old = load_provider_model(write(), data_root=root)
    monkeypatch.setattr(provider_counter, "DeepSeekCounter", lambda *a, **k: Counter())
    old_policy = canonical(values["policy"])
    values["revision"] += 1
    values["route"]["wire_profile"] = FOCUSED
    route = replace(old.route, wire_profile=FOCUSED)
    values["ledger"] = str(root / "focused.sqlite3")
    rates = next(iter(values["policy"]["routes"].values()))
    values["policy"] = {**values["policy"], "id": "provider-config-focused-test",
                        "routes": {route.fingerprint: rates}}
    value = load_provider_model(write(), data_root=root)
    assert value.route.wire_profile == FOCUSED
    assert value.route.fingerprint != old.route.fingerprint
    assert value.config_fingerprint != old.config_fingerprint
    assert canonical(values["policy"]) != old_policy
    assert value.transport.ledger.ledger.ledger.summary()["reserved_calls"] == 0


def test_provider_selection_exposes_actual_profile_in_existing_identity(application, monkeypatch):
    dispatch, memories = application

    class Focused(FakeModel):
        config_revision, config_fingerprint = 3, "c" * 64
        route = focused_route()

    monkeypatch.setattr(bootstrap, "load_provider_model", lambda *a, **k: Focused())
    result = dispatch("selectProvider", {"provider_id": "deepseek-api", "confirmed": True})
    assert result["status"] == "selected"
    state = dispatch("state", {})
    assert state["provider"]["active"]["model_profile"] == FOCUSED
    assert len(memories) == 2


def sse(value):
    return ("data: " + json.dumps(value, ensure_ascii=False) + "\n\n").encode()


@contextmanager
def recording_loopback(monkeypatch, ledger, *, mode="normal", reply=None):
    """Keep official route/request bytes; substitute only the test network transport."""
    raw_requests, network_requests, replies = [], [], []
    arrived, release = threading.Event(), threading.Event()
    if reply is not None:
        replies.append(reply)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):  # noqa: N802 - standard HTTP server callback
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            rows = ledger.records()
            assert rows[-1]["request_sha256"] == sha256(raw).hexdigest()
            raw_requests.append(raw)
            arrived.set()
            if mode == "error":
                self.send_response(503)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            if mode == "stall":
                self.wfile.write(b": waiting\n\n")
                self.wfile.flush()
                release.wait(2)
                return
            answer = replies.pop(0) if replies else {"answer": "synthetic"}
            raw = sse({"choices": [{"index": 0, "delta": {"content": canonical(answer)},
                                    "finish_reason": None}]})
            raw += sse({"choices": [{"index": 0, "delta": {},
                                     "finish_reason": "length" if mode == "length" else "stop"}]})
            if mode != "unknown_usage":
                raw += sse({"choices": [], "usage": {
                    "prompt_tokens": 17, "completion_tokens": 5, "total_tokens": 22,
                }})
            if mode != "incomplete":
                raw += b"data: [DONE]\n\n"
            self.wfile.write(raw)
            self.wfile.flush()

    service = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    service.daemon_threads = True
    worker = threading.Thread(target=service.serve_forever)
    worker.start()
    real_transport = httpx.AsyncHTTPTransport

    class NetworkTransport(httpx.AsyncBaseTransport):
        def __init__(self, **options):
            assert options == {"retries": 0, "trust_env": False}
            self.inner = real_transport(**options)

        async def handle_async_request(self, request):
            assert str(request.url) == "https://api.deepseek.com/chat/completions"
            raw = await request.aread()
            network_requests.append({"url": str(request.url), "raw": raw})
            recording = httpx.Request(request.method,
                                      f"http://127.0.0.1:{service.server_port}/chat/completions",
                                      headers=request.headers, content=raw)
            return await self.inner.handle_async_request(recording)

        async def aclose(self):
            await self.inner.aclose()

    monkeypatch.setattr(httpx, "AsyncHTTPTransport", NetworkTransport)
    try:
        yield raw_requests, network_requests, replies, arrived
    finally:
        release.set()
        service.shutdown()
        service.server_close()
        worker.join(2)


class SyntheticSecrets:
    def get(self, reference, endpoint):
        assert reference.startswith("dpapi:")
        assert endpoint == "https://api.deepseek.com/chat/completions"
        return "synthetic-only"  # noqa: S105 - test port, never a managed key


def focused_model(tmp_path, *, timeout=2):
    route = focused_route()
    policy = {"id": "focused-loopback", "max_calls": 32, "max_input_tokens": 100000,
              "max_output_tokens": 50000, "max_cost_microusd": 100000,
              "routes": {route.fingerprint: {"kind": "api", "input_rate": 1, "output_rate": 1}}}
    ledger = ProviderBudgetLedger(tmp_path / "provider.sqlite3", policy)
    transport = HttpxProviderTransport(route, secrets=SyntheticSecrets(),
                                       ledger=ConservativeReservation(ledger))
    value = ProviderModel(route, counter=Counter(), transport=transport, timeout_seconds=timeout)
    return value, ledger


def assert_frozen_send(request, rows, raw, network):
    expected = canonical(request._body).encode()
    assert raw[-1] == network[-1]["raw"] == expected
    assert rows[-1]["request_sha256"] == request._fingerprint == sha256(expected).hexdigest()
    assert rows[-1]["input_tokens"] == request.bound == 200
    assert json.loads(raw[-1])["temperature"] == TEMPERATURES[rows[-1]["stage"]]


def persist_recording(tmp_path, rows, raw, network):
    (tmp_path / "focused-recording.json").write_text(json.dumps({
        "schema": "focused-offline-httpx-recording.v1", "api_calls": 0,
        "network_boundary": "test HTTPX transport to real loopback, not DeepSeek",
        "requests": [{"official_url": item["url"], "raw_body_utf8": item["raw"].decode(),
                      "same_loopback_bytes": item["raw"] == loopback,
                      "request_sha256": sha256(loopback).hexdigest()}
                     for item, loopback in zip(network, raw, strict=True)],
        "ledger": rows,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def test_checked_body_httpx_and_all_stages_keep_one_budget_route(tmp_path, monkeypatch):
    value, ledger = focused_model(tmp_path)
    peer = value.review_peer()
    assert peer.route is value.route
    assert peer.transport.ledger.ledger is ledger
    with recording_loopback(monkeypatch, ledger) as (raw, network, _, _):
        for stage in TEMPERATURES:
            owner = value if stage in {"demand_control", "demand_repair"} else peer
            request = owner.prepare_stage_request(MESSAGES, stage=stage, schema=SCHEMA)
            request.check_context()
            assert json.loads("".join(request.stream(threading.Event()))) == {"answer": "synthetic"}
            assert_frozen_send(request, ledger.records(), raw, network)
    rows = ledger.records()
    assert len(rows) == len(TEMPERATURES)
    assert {row["route"] for row in rows} == {value.route.fingerprint}
    assert {row["status"] for row in rows} == {"completed"}
    assert ledger.summary()["unknown_usage_calls"] == 0
    persist_recording(tmp_path, rows, raw, network)


@pytest.mark.parametrize("case,status,unknown", [("error", "failed", 1),
                                                ("incomplete", "failed", 0),
                                                ("length", "failed", 0),
                                                ("unknown_usage", "completed", 1),
                                                ("cancel", "cancelled", 1),
                                                ("deadline", "deadline_exceeded", 1)])
def test_focused_terminal_keeps_reservation_and_unknown_usage(tmp_path, monkeypatch,
                                                             case, status, unknown):
    value, ledger = focused_model(tmp_path, timeout=0.15 if case == "deadline" else 2)
    mode = "stall" if case in {"cancel", "deadline"} else case
    cancelled = threading.Event()
    with recording_loopback(monkeypatch, ledger, mode=mode) as (raw, network, _, arrived):
        request = value.prepare_stage_request(MESSAGES, stage="stream_unit_review", schema=SCHEMA)
        request.check_context()
        stopper = None
        if case == "cancel":
            def cancel_after_send():
                assert arrived.wait(1)
                cancelled.set()
            stopper = threading.Thread(target=cancel_after_send)
            stopper.start()
        with pytest.raises((ProviderFailure, ValueError, InterruptedError)):
            list(request.stream(cancelled))
        if stopper is not None:
            stopper.join(1)
        assert_frozen_send(request, ledger.records(), raw, network)
    summary, rows = ledger.summary(), ledger.records()
    assert len(rows) == summary["reserved_calls"] == 1
    assert rows[0]["status"] == status
    assert summary["unknown_usage_calls"] == unknown
    assert summary["reserved_output_tokens"] == 512
    assert rows[0]["overrun"] == 0
    if case == "unknown_usage":
        assert value.failed
    persist_recording(tmp_path, rows, raw, network)


@pytest.mark.parametrize("case", ["approve", "unsupported", "foreign_claim", "foreign_source"])
def test_focused_structured_review_retains_sealed_host_audit(tmp_path, monkeypatch, case):
    value, ledger = focused_model(tmp_path)
    store = SQLiteStreamingStore(tmp_path / "companion.sqlite3")
    user = "刚整理完桌上的旧票根，想聊聊。"
    candidate = "这句话已经完整。" if case == "approve" else "你一直收藏这些票根。"
    run = claim(store, store.accept(config(), "focused-input", user))
    requests, packets = [], []
    with recording_loopback(monkeypatch, ledger) as (raw, network, replies, _):
        def invoke(stage, messages, schema, cancelled):
            packet = recorded_review_packet(messages)
            packets.append(packet)
            opinion = {"decision": "approve", "self_contained": True,
                       "no_open_dependency": True, "consistent_with_displayed": True,
                       "constraints_satisfied": True, "audit_version": "candidate-propositions.v3",
                       "segments": [{"segment_id": item["segment_id"], "commitments": [],
                                     "questions": []} for item in packet["review_segments"]]}
            if case != "approve":
                source = packet["personal_sources"][0]
                ref = {key: source[key] for key in ("id", "owner", "source_id", "role", "revision")}
                ref["quote"] = user
                if case == "foreign_source":
                    ref["revision"] += 1
                opinion["segments"][0]["commitments"] = [{
                    "kind": "assertion", "claim": user if case == "foreign_claim" else candidate,
                    "support": "supported" if case == "foreign_source" else "unsupported",
                    "sources": [ref] if case == "foreign_source" else [],
                }]
            replies.append(opinion)
            request = value.prepare_stage_request(messages, stage=stage, schema=schema,
                                                  output_tokens=1536)
            requests.append(request)
            request.check_context()
            return json.loads("".join(request.stream(cancelled)))

        assessor = StructuredUnitAssessor(invoke, threading.Event())
        verdicts = []

        def assess(envelope):
            verdict = assessor(envelope)
            verdicts.append(verdict)
            return verdict

        reviewer = BoundedUnitReviewer(run, ShortRequest(user), assess)
        if case.startswith("foreign_"):
            with pytest.raises(ValueError, match="Commitment|source"):
                reviewer(candidate, True)
        elif case == "unsupported":
            with pytest.raises(CandidateRejected):
                reviewer(candidate, True)
            assert verdicts[-1]["decision"] == "reject"
            assert verdicts[-1]["model_opinion"]["decision"] == "approve"
        else:
            assert reviewer(candidate, True).candidate.text == candidate
        assert_frozen_send(requests[0], ledger.records(), raw, network)
    assert packets[0]["user_text"] == user
    assert ledger.records()[0]["output_tokens"] == 1536
    assert ledger.records()[0]["status"] == "completed"
    persist_recording(tmp_path, ledger.records(), raw, network)
