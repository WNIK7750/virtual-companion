"""Recorded N08 format failure through actual runtime/reviewer/SQLite, with CPU suppliers."""

import json
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pytest
from test_default_text_delivery import Model, setup, wait_for
from unit_review_recording import bound_unit_packets as bound_unit_packets
from unit_review_recording import recorded_review_packet
from virtual_companion.cognition.stream_unit_review import StructuredUnitAssessor
from virtual_companion.text.application import TextSession
from virtual_companion.text.default_runtime import DefaultTextRuntime
from virtual_companion.text.fake import synthetic_unit_opinion

# Exact raw_text from stage1-natural-v33-pilot-20261010/attempts.jsonl, N08/v33.
# Original replay receipt remains n08-original-raw-replay-01/receipt.json.
RAW = (Path(__file__).parent / "fixtures/n08-v33-malformed-carrier.txt").read_text("utf-8")
RAW_SHA256 = "f93fcd5d328f25a53c813a0dc78304e23637247875cebdc301ab407fa0c9e454"
TEXT = (
    "你俩争的其实不是最后两步本身，而是“规则到底算不算数”这件事。"
    "对方说“规则本来就该灵活”，如果指的是对局里没写清的情况可以商量着来，那还有道理；"
    "但要是拿它当理由，把已经定好的走法、胜负条件也随时改掉，那确实就没法算了——"
    "因为输赢得建立在同一套标准上，标准一变，结果就悬空了。\n\n"
    "所以我的看法是：灵活可以留在“规则没覆盖到的地方”，比如赛前说好遇到争议怎么处理；"
    "一旦进入对局，尤其是最后两步这种直接决定胜负的阶段，就该按事先认的规则走。"
    "不然争的就不是棋，而是谁嗓门大。\n\n"
    "你们平时是下之前会先约定好规则，还是默认按某套标准来？"
)
SOURCE = ("我们下棋时总在最后两步争个不停。对方说‘规则本来就该灵活’，"
          "我觉得这样输赢没法算。你怎么看？")


class RecordedCarrierModel(Model):
    def __init__(self, chunking, *, raw=RAW, recovery=None, reject_final=False):
        super().__init__()
        self.chunking, self.raw = chunking, raw
        self.recovery = (json.dumps({"action": "answer", "text": TEXT}, ensure_ascii=False)
                         if recovery is None else recovery)
        self.reject_final, self.on_recovery, self.initial_error = reject_final, None, None
        self.generations, self.reviews = [], []
        self.initial_closed = False

    def review_peer(self):
        owner = self

        class Peer(Model):
            def stream(self, messages, cancelled):
                packet = recorded_review_packet(messages)
                owner.reviews.append(deepcopy(packet))
                opinion = synthetic_unit_opinion(packet)
                if owner.reject_final and packet["final"]:
                    opinion.update(decision="reject", checks=["pass", "pass", "pass", "fail"],
                        failure={"scope": "accuracy", "segment_id": None,
                                 "basis": "explicit CPU final rejection"})
                yield json.dumps(opinion, ensure_ascii=False)

        return Peer()

    def stream(self, messages, cancelled):
        self.generations.append(deepcopy(messages))
        if len(self.generations) == 1:
            try:
                yield from ([self.raw] if self.chunking == "one_fragment"
                            else [self.raw[:-1], self.raw[-1:]])
                if self.initial_error is not None:
                    raise self.initial_error
            finally:
                self.initial_closed = True
            return
        assert self.initial_closed, "Original generation must end before recovery"
        if self.on_recovery is not None:
            self.on_recovery()
        yield self.recovery


def carrier_setup(tmp_path, chunking="one_fragment", *, model=None):
    model = model or RecordedCarrierModel(chunking)
    _, _, memory, mirror, _ = setup(tmp_path, model)
    runtime = DefaultTextRuntime(memory, mirror, model, binding={"test": "cpu-recorded-N08"},
                                 prompt_revision="short-dialogue.v33", delivery_timeout=1.5)
    session = TextSession(model, mirror, {}, prepare_turn=runtime.prepare_turn,
                          before_start=runtime.before_start)
    return runtime, session, memory, mirror, model


def acknowledge_all(runtime, session):
    events = []
    while session.snapshot()["busy"]:
        event = wait_for(lambda: runtime.poll()["event"] or not session.snapshot()["busy"])
        if event is True:
            break
        events.append(deepcopy(event))
        runtime.ack(dict(sequence=event["sequence"], fingerprint=event["fingerprint"],
                         host_receipt=f"synthetic-N08-format-recovery-{event['sequence']}"))
    return events


@pytest.mark.parametrize("chunking", ["one_fragment", "terminal_split"])
def test_recorded_malformed_terminal_recovers_once_without_replaying_ack_prefix(
    tmp_path, monkeypatch, chunking,
):
    assert sha256(RAW.encode()).hexdigest() == RAW_SHA256
    with pytest.raises(json.JSONDecodeError, match="Extra data"):
        json.loads(RAW)
    runtime, session, memory, mirror, model = carrier_setup(tmp_path, chunking)
    packets, observed, stages, turns = [], [], [], []
    assess = StructuredUnitAssessor.__call__
    prepare, prepare_turn = runtime._prepare_turn_stage, runtime.prepare_turn

    def capture(assessor, envelope):
        packets.append(deepcopy(envelope["payload"]))
        return assess(assessor, envelope)

    monkeypatch.setattr(StructuredUnitAssessor, "__call__", capture)
    runtime.observe_stream = lambda event: observed.append(deepcopy(event))

    def capture_stage(claim, *, final, repair=None, recall_query=None):
        if repair is not None:
            assert model.initial_closed and memory.generation_gate._active is None
        stage = prepare(claim, final=final, repair=repair, recall_query=recall_query)
        stages.append(stage)
        return stage

    def capture_turn(*args):
        turn = prepare_turn(*args)
        turns.append(turn)
        return turn

    monkeypatch.setattr(runtime, "_prepare_turn_stage", capture_stage)
    session.prepare_turn = capture_turn
    try:
        session.send(SOURCE)
        events = acknowledge_all(runtime, session)
        state = memory.store.stream_state(memory.companion, runtime.claim.input_id)
        assert session.snapshot()["attempt"]["status"] == "completed", session.snapshot()
        assert len(model.generations) == 2
        assert turns[0].result.generation_calls == 2 and turns[0].result.memory_calls == 0
        assert len(stages) == 2 and stages[0].request is stages[1].request
        assert stages[0].access_status == stages[1].access_status
        assert [event["sequence"] for event in events] == [1, 2]
        assert "".join(event["text"] for event in events) == TEXT
        assert events[1]["display_text"] == TEXT
        assert state["outcome"] == "completed"
        assert state["approved_units"] == state["confirmed_units"] == 2
        assert state["confirmed_text"] == TEXT
        segments = memory.store.segments(memory.companion, runtime.claim.input_id)
        assert segments[0]["fingerprint"] == events[0]["fingerprint"]
        assert all(segment["status"] == "confirmed" for segment in segments)
        assert mirror.history()["messages"][-1]["content"] == TEXT
        assert len(model.reviews) == 2
        assert packets[1]["final"] is True
        assert packets[1]["displayed_prefix"] == events[0]["display_text"]
        assert packets[1]["candidate_prefix"] == TEXT
        assert packets[0]["context"] == packets[1]["context"]
        assert packets[0]["basis"] == packets[1]["basis"]
        message = next(message for message in model.generations[1]
                       if '{"carrier_recovery":' in message["content"])
        payload = json.loads(message["content"][message["content"].index('{"carrier_recovery":'):])
        assert payload["carrier_recovery"] == dict(
            malformed_carrier=RAW, parser_error="Extra data at char 295",
            action="answer", text=TEXT)
        assert [m for m in model.generations[1] if m is not message] == model.generations[0]
        assert [e["original_sha256"] for e in observed
                if e["event"] == "carrier_recovery"] == [RAW_SHA256]
    finally:
        runtime.stop()
        session.stop()
        memory.close()
        mirror.close()


@pytest.mark.parametrize("recovery", [
    json.dumps({"action": "answer", "text": TEXT + "新增文字。"}, ensure_ascii=False),
    json.dumps({"action": "ack", "text": TEXT}, ensure_ascii=False),
    RAW,
    json.dumps({"action": "memory"}),
    json.dumps({"action": "answer", "text": TEXT, "text_note": "inert"}, ensure_ascii=False),
], ids=["changed_text", "changed_action", "malformed_again", "memory", "extra_field"])
def test_failed_carrier_recovery_preserves_only_ack_prefix_without_another_attempt(
    tmp_path, recovery,
):
    model = RecordedCarrierModel("one_fragment", recovery=recovery)
    runtime, session, memory, mirror, _ = carrier_setup(tmp_path, model=model)
    try:
        session.send(SOURCE)
        events = acknowledge_all(runtime, session)
        assert session.snapshot()["attempt"]["status"] == "failed"
        assert len(model.generations) == 2 and len(model.reviews) == 1
        assert len(events) == 1
        state = memory.store.stream_state(memory.companion, runtime.claim.input_id)
        assert state["outcome"] == "failed" and state["confirmed_units"] == 1
        assert state["confirmed_text"] == mirror.history()["messages"][-1]["content"] == events[0][
            "display_text"]
        assert not runtime.snapshot()["retry_allowed"]
    finally:
        runtime.stop()
        session.stop()
        memory.close()
        mirror.close()


@pytest.mark.parametrize("failure", ["unclosed_string", "transport", "cancel", "final_rejection"])
def test_carrier_recovery_keeps_termination_and_final_reviewer_boundaries(tmp_path, failure):
    model = RecordedCarrierModel("one_fragment",
                                 raw=RAW[:-3] if failure == "unclosed_string" else RAW,
                                 reject_final=failure == "final_rejection")
    runtime, session, memory, mirror, _ = carrier_setup(tmp_path, model=model)
    if failure == "transport":
        model.initial_error = OSError("explicit CPU supplier terminal failure")
    if failure == "cancel":
        model.on_recovery = lambda: runtime.cancelled.set()
    try:
        session.send(SOURCE)
        events = acknowledge_all(runtime, session)
        expected = "cancelled" if failure == "cancel" else "failed"
        assert session.snapshot()["attempt"]["status"] == expected
        assert len(model.generations) == (1 if failure in {"unclosed_string", "transport"} else 2)
        assert len(model.reviews) == (2 if failure == "final_rejection" else 1)
        assert len(events) == 1
        state = memory.store.stream_state(memory.companion, runtime.claim.input_id)
        assert state["outcome"] == expected and state["confirmed_units"] == 1
        assert state["confirmed_text"] == mirror.history()["messages"][-1]["content"] == events[0][
            "display_text"]
    finally:
        runtime.stop()
        session.stop()
        memory.close()
        mirror.close()


@pytest.mark.parametrize("budget", [4, 2], ids=["reserved_recovery", "budget_exhausted"])
def test_real_provider_reservation_and_ledger_include_the_single_carrier_recovery(
    tmp_path, monkeypatch, budget,
):
    """Actual ProviderModel/HTTPX/SSE/SQLite ledger with a no-socket AsyncClient."""
    import httpx
    from test_deepseek_focused import Counter, SyntheticSecrets, focused_route
    from test_model_provider_contract import event
    from virtual_companion.providers.budget import ProviderBudgetLedger
    from virtual_companion.providers.httpx_transport import HttpxProviderTransport
    from virtual_companion.text.provider_model import ConservativeReservation, ProviderModel

    route = focused_route()
    policy = {"id": "CPU-carrier-recovery", "max_calls": budget, "max_input_tokens": 100000,
              "max_output_tokens": 100000, "max_cost_microusd": 100000,
              "routes": {route.fingerprint: {"kind": "api", "input_rate": 1, "output_rate": 1}}}
    ledger = ProviderBudgetLedger(tmp_path / "provider.sqlite3", policy)
    wire = []

    class Response:
        status_code = 200
        headers = {"Content-Type": "text/event-stream"}

        def __init__(self, body):
            messages = body["messages"]
            content = messages[-1]["content"]
            view = json.loads(content) if content.startswith("{") else None
            if isinstance(view, dict) and "view_version" in view:
                self.raw = json.dumps(synthetic_unit_opinion(view))
            elif any('{"carrier_recovery":' in message["content"] for message in messages):
                self.raw = json.dumps({"action": "answer", "text": TEXT}, ensure_ascii=False)
            else:
                self.raw = RAW

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def aiter_bytes(self):
            yield event({"choices": [{"index": 0, "delta": {"content": self.raw}}]})
            yield event({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            yield event({"choices": [], "usage": {"prompt_tokens": 17,
                "completion_tokens": 10, "total_tokens": 27}})
            yield b"data: [DONE]\n\n"

    class Client:
        def __init__(self, **options):
            assert options["trust_env"] is False and options["follow_redirects"] is False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        def stream(self, method, endpoint, *, content, headers):
            assert method == "POST" and endpoint == route.endpoint
            wire.append(sha256(content).hexdigest())
            return Response(json.loads(content))

    monkeypatch.setattr(httpx, "AsyncClient", Client)
    monkeypatch.setattr(httpx, "AsyncHTTPTransport", lambda **_: object())
    transport = HttpxProviderTransport(route, secrets=SyntheticSecrets(),
                                       ledger=ConservativeReservation(ledger))
    model = ProviderModel(route, counter=Counter(), transport=transport, timeout_seconds=4)
    runtime, session, memory, mirror, _ = carrier_setup(tmp_path, model=model)
    errors = []
    deliver = runtime.deliver

    def capture(stage, answer):
        try:
            return deliver(stage, answer)
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")
            raise

    monkeypatch.setattr(runtime, "deliver", capture)
    try:
        session.send(SOURCE)
        events = acknowledge_all(runtime, session)
        assert session.snapshot()["attempt"]["status"] == (
            "completed" if budget == 4 else "failed"), errors
        rows = ledger.records()
        assert len(rows) == len(wire) == budget
        assert [row["stage"] for row in rows] == ["demand_control", "stream_unit_review"] + (
            ["demand_repair", "stream_unit_review"] if budget == 4 else [])
        assert [row["request_sha256"] for row in rows] == wire
        assert {row["route"] for row in rows} == {route.fingerprint}
        assert {row["status"] for row in rows} == {"completed"}
        assert ledger.summary()["unknown_usage_calls"] == 0
        assert len(events) == (2 if budget == 4 else 1)
        if budget == 4:
            assert "".join(item["text"] for item in events) == TEXT
    finally:
        runtime.stop()
        session.stop()
        memory.close()
        mirror.close()
