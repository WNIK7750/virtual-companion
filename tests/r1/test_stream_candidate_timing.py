"""RF3.1 measurement contract: when is a real complete candidate actually available?

The coordinator must expose an ordered, monotonic observation trace so a frozen
real-model run can locate the first eligible unit, the review boundary and the
actual visibility receipt instead of inferring them. A candidate boundary remains
an observation only: approval still requires the trusted review port.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event

import pytest
from test_rebuilt_turns import A, claim, config
from test_segment_store import reviewed
from virtual_companion.boundary.sqlite_streaming import SQLiteStreamingStore
from virtual_companion.cognition.short_dialogue import ShortRequest
from virtual_companion.cognition.stream_dialogue import StreamDeliveryUnknown, StreamDialogue
from virtual_companion.providers.text_deltas import UnapprovedTextDelta
from virtual_companion.text.fake import synthetic_unit_opinion


@pytest.mark.parametrize('terminal', ['stop', 'length'])
def test_default_provider_wire_confirms_unit_before_generation_terminal(
    tmp_path, terminal, record_property,
):
    """Actual SSE, ProviderModel and default v19/ACK path; no remote model call."""
    from test_default_text_delivery import setup, wait_for
    from test_model_provider_contract import event, route
    from test_provider_loopback import Ledger
    from test_text_provider_model import Counter
    from virtual_companion.providers.httpx_transport import HttpxProviderTransport
    from virtual_companion.text.provider_model import ConservativeReservation, ProviderModel

    release, terminal_sent = threading.Event(), threading.Event()
    calls, times = [], {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send_content(self, text, *, finish=None):
            self.wfile.write(event({'choices': [{'index': 0, 'delta': {'content': text},
                                                 'finish_reason': finish}]}))
            self.wfile.flush()

        def send_terminal(self, reason='stop'):
            self.send_content('', finish=reason)
            self.wfile.write(event({'choices': [], 'usage': {
                'prompt_tokens': 17, 'completion_tokens': 10, 'total_tokens': 27}}))
            self.wfile.write(b'data: [DONE]\n\n')
            self.wfile.flush()

        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            calls.append(body)
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            if '你只审查候选文字' in body['messages'][0]['content']:
                packet = json.loads(body['messages'][-1]['content'])
                self.send_content(json.dumps(synthetic_unit_opinion(packet), ensure_ascii=False))
                self.send_terminal()
                return
            self.send_content('{"action":"answer","text":"当前回应已形成。')
            if not release.wait(4):
                return
            self.send_content('后句继续展开。"}')
            times['provider_terminal_us'] = time.time_ns() // 1000
            terminal_sent.set()
            self.send_terminal(terminal)

    service = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=service.serve_forever)
    worker.start()
    runtime = memory = mirror = None
    try:
        configured = route(endpoint=f'http://127.0.0.1:{service.server_port}/chat/completions')
        ledger = Ledger(limit=8)
        model = ProviderModel(configured, counter=Counter(), transport=HttpxProviderTransport(
            configured, secrets=None, ledger=ConservativeReservation(ledger)))
        runtime, text, memory, mirror, _ = setup(tmp_path, model)
        text.send('合成输入')
        first = wait_for(lambda: runtime.poll()['event'])
        assert first['text'] == '当前回应已形成。'
        assert not terminal_sent.is_set()
        runtime.ack(dict(sequence=1, fingerprint=first['fingerprint'],
                         host_receipt='synthetic-real-wire-first'))
        times['confirmed_us'] = time.time_ns() // 1000
        assert memory.store.stream_state(memory.companion, runtime.claim.input_id)[
            'confirmed_units'] == 1
        assert not terminal_sent.is_set()
        release.set()
        if terminal == 'stop':
            last = wait_for(lambda: (event if (event := runtime.poll()['event'])
                                    and event['sequence'] == 2 else None))
            runtime.ack(dict(sequence=2, fingerprint=last['fingerprint'],
                             host_receipt='synthetic-real-wire-final'))
        wait_for(lambda: not text.snapshot()['busy'])
        state = memory.store.stream_state(memory.companion, runtime.claim.input_id)
        assert state['outcome'] == ('completed' if terminal == 'stop' else 'failed')
        assert state['confirmed_text'] == ('当前回应已形成。后句继续展开。'
                                           if terminal == 'stop' else first['text'])
        assert times['confirmed_us'] < times['provider_terminal_us']
        for name, value in times.items():
            record_property(name, value)
        record_property('first_fingerprint', first['fingerprint'])
        record_property('outcome', state['outcome'])
        record_property('confirmed_units', state['confirmed_units'])
        assert all(body['response_format']['type'] == 'json_schema' for body in calls)
        assert len(ledger.calls) == len(ledger.finished) == (3 if terminal == 'stop' else 2)
        assert all(call['input_tokens'] == 200 for call in ledger.calls)
        assert all(call['route'] == configured.fingerprint for call in ledger.calls)
    finally:
        release.set()
        if memory is not None:
            memory.close()
        if mirror is not None:
            mirror.close()
        service.shutdown()
        service.server_close()
        worker.join(2)

FIRST = "我在这里。"
TAIL = "你想聊什么？"
FULL = FIRST + "\n\n" + TAIL


def approved_prefix(run, observed, final, approved_text=""):
    """Explicit test approval for the observed prefix; not a production decision.

    A final opinion with nothing approved yet owns the whole text as one unit, which
    is what the real reviewer produces; otherwise it extends the approved units.
    """

    if final:
        if not approved_text:
            return reviewed(run, (observed.rstrip("\n") or observed,))
        parts = tuple(part for part in observed.rstrip("\n").split("\n\n") if part)
        return reviewed(run, parts) if parts else None
    boundary = observed.find("\n\n")
    return None if boundary < 0 else reviewed(run, (observed[:boundary],))


def default_present(event):
    return "host-" + event["fingerprint"][:8]


def dialogue_for(store, run, model, *, observe=None, present=None, parallel=True, review=None):
    approved = {"text": ""}

    def review_prefix(text, final):
        if review is not None:
            return review(text, final)
        result = approved_prefix(run, text, final, approved["text"])
        if result is not None:
            approved["text"] = result.candidate.text
        return result

    options = {} if observe is None else {"observe": observe}
    return StreamDialogue(
        store=store,
        model_factory=lambda receive: model.attach(receive),
        review_prefix=review_prefix,
        present=default_present if present is None else present,
        checkpoint=lambda: None,
        cancelled=Event(),
        clock=lambda: 100,
        parallel_review=parallel,
        **options,
    )


class ScriptedModel:
    """Synthetic producer whose deltas are explicit; no model call is made."""

    def __init__(self, script):
        self.script, self.receive, self.cancelled = script, None, Event()

    def attach(self, receive):
        self.receive = receive
        return self

    def count(self, body):
        return 10

    def cancel(self):
        self.cancelled.set()


def test_coordinator_records_candidate_timing_before_any_decision(tmp_path):
    events = []
    store = SQLiteStreamingStore(tmp_path / "timing.sqlite3")
    run = claim(store, store.accept(config(), "first", "你好"))
    displayed = Event()

    class Model(ScriptedModel):
        def invoke(self, body):
            self.receive(UnapprovedTextDelta(1, FIRST))
            self.receive(UnapprovedTextDelta(2, "\n\n"))
            assert displayed.wait(2), "First unit was not visible before model terminal"
            self.receive(UnapprovedTextDelta(3, TAIL))
            return {"text": FULL, "tool_calls": [], "finish_reason": "stop"}

    model = Model(None)

    def present(event):
        displayed.set()
        return "host-" + event["fingerprint"][:8]

    dialogue = dialogue_for(store, run, model, observe=events.append, present=present)

    assert dialogue.generate(run, ShortRequest("你好"))["outcome"] == "completed"

    names = [event["event"] for event in events]
    sequences = [e["sequence"] for e in events if e["event"] == "delta"]
    assert names[0] == "delta" and sequences == [1, 2, 3]
    boundaries = [(event["kind"], event["offset"])
                  for event in events if event["event"] == "observed_boundary"]
    # The earliest faithful boundary is the sentence ending and stays the same
    # observation once the blank line arrives, so it is recorded exactly once.
    assert boundaries == [("sentence", len(FIRST))]
    for required in ("review_begin", "review_end", "approved", "visible", "confirmed",
                     "model_terminal"):
        assert required in names, required
    assert names.index("observed_boundary") < names.index("review_begin")
    assert names.index("review_begin") < names.index("review_end")
    assert names.index("approved") < names.index("visible") < names.index("confirmed")
    assert names.index("visible") < names.index("model_terminal")
    assert names[-1] == "settled"
    times = [event["at"] for event in events]
    assert times == sorted(times)
    assert all(event["at"] == 100 for event in events)
    assert [e["sequence"] for e in events if e["event"] == "visible"] == [1, 2]
    assert [e["sequence"] for e in events if e["event"] == "confirmed"] == [1, 2]


def test_boundary_observed_before_terminal_is_not_dropped_when_review_was_in_flight(tmp_path):
    """The first review and durable ACK precede consumption of the next delta."""
    from virtual_companion.cognition.stream_unit_review import BoundedUnitReviewer

    store = SQLiteStreamingStore(tmp_path / "drain.sqlite3")
    run = claim(store, store.accept(config(), "first", "你好"))
    request = ShortRequest("你好")
    review_started, model_done, first_confirmed = Event(), Event(), Event()
    calls, seen, events = [], [], []

    class Model(ScriptedModel):
        def count(self, body):
            self.receive(UnapprovedTextDelta(1, FIRST))
            assert review_started.is_set() and first_confirmed.is_set()
            assert store.stream_state(A, run.input_id)["confirmed_text"] == FIRST
            self.receive(UnapprovedTextDelta(2, "\n\n" + TAIL))
            model_done.set()
            return 10

        def invoke(self, body):
            return {"text": FULL, "tool_calls": [], "finish_reason": "stop"}

    def assess(envelope):
        packet = envelope["payload"]
        calls.append((packet["observed_text"], packet["final"]))
        if not packet["final"]:
            review_started.set()
            assert not model_done.is_set() and not first_confirmed.is_set()
        return {"request_sha256": envelope["sha256"], "decision": "approve",
                "self_contained": True, "no_open_dependency": True,
                "consistent_with_displayed": True, "constraints_satisfied": True}

    def observe(event):
        events.append(event)
        if event["event"] == "confirmed" and event["sequence"] == 1:
            assert not model_done.is_set()
            assert store.stream_state(A, run.input_id)["confirmed_units"] == 1
            first_confirmed.set()

    model = Model(None)
    reviewer = BoundedUnitReviewer(run, request, assess, early_limit=1)
    dialogue = dialogue_for(store, run, model, review=reviewer, observe=observe,
                            present=lambda event: seen.append(event) or f"host-{len(seen)}")

    assert dialogue.generate(run, request)["outcome"] == "completed"
    state = store.stream_state(A, run.input_id)
    assert state["confirmed_text"] == FULL and state["confirmed_units"] == 2
    assert [event["text"] for event in seen] == [FIRST, "\n\n" + TAIL]
    assert calls == [(FIRST, False), (FULL, True)]
    assert reviewer.early_calls == reviewer.final_calls == 1
    names = [event["event"] for event in events]
    assert names.index("review_end") < names.index("approved") < names.index("visible")
    assert names.index("visible") < names.index("confirmed") < names.index("model_terminal")
    assert next(i for i, event in enumerate(events)
                if event["event"] == "delta" and event["sequence"] == 2) > names.index("confirmed")
    assert model_done.is_set() and names[-1] == "settled"


def test_parallel_review_does_not_spend_an_early_opinion_after_model_terminal(
    tmp_path, monkeypatch,
):
    """Without a faithful boundary, a terminal producer spends only its final opinion."""
    from virtual_companion.cognition.stream_unit_review import BoundedUnitReviewer

    store = SQLiteStreamingStore(tmp_path / "already-terminal.sqlite3")
    run = claim(store, store.accept(config(), "first", "你好"))
    request, complete = ShortRequest("你好"), "No faithful boundary"
    terminal = Event()
    real_thread = threading.Thread
    calls, events = [], []

    def delayed_review_thread(*, target, **options):
        def after_terminal():
            assert terminal.wait(2)
            target()
        return real_thread(target=after_terminal, **options)

    monkeypatch.setattr(
        "virtual_companion.cognition.stream_dialogue.threading.Thread", delayed_review_thread,
    )

    class Model(ScriptedModel):
        def invoke(self, body):
            self.receive(UnapprovedTextDelta(1, complete))
            return {"text": complete, "tool_calls": [], "finish_reason": "stop"}

    def observe(event):
        events.append(event["event"])
        if event["event"] == "model_terminal":
            terminal.set()

    def assess(envelope):
        packet = envelope["payload"]
        assert terminal.is_set() and packet["final"]
        calls.append((packet["observed_text"], packet["final"]))
        return {"request_sha256": envelope["sha256"], "decision": "approve",
                "self_contained": True, "no_open_dependency": True,
                "consistent_with_displayed": True, "constraints_satisfied": True}

    reviewer = BoundedUnitReviewer(run, request, assess, early_limit=1)
    dialogue = dialogue_for(store, run, Model(None), observe=observe, review=reviewer)
    assert dialogue.generate(run, request)["outcome"] == "completed"
    assert calls == [(complete, True)] and terminal.is_set()
    assert reviewer.early_calls == 0 and reviewer.final_calls == 1
    assert "observed_boundary" not in events
    assert events.index("model_terminal") < events.index("review_begin")
    assert events[-1] == "settled"
    state = store.stream_state(A, run.input_id)
    assert state["confirmed_units"] == 1 and state["confirmed_text"] == complete


def test_stream_kernel_forwards_observation_port_without_changing_decisions(tmp_path):
    from virtual_companion.cognition.short_context import ShortContextAdmission
    from virtual_companion.cognition.stream_kernel import StreamKernel

    events = []
    store = SQLiteStreamingStore(tmp_path / "kernel-timing.sqlite3")
    run = claim(store, store.accept(config(), "first", "你好"))

    class Model(ScriptedModel):
        def invoke(self, body):
            self.receive(UnapprovedTextDelta(1, FIRST))
            return {"text": FIRST, "tool_calls": [], "finish_reason": "stop"}

    model = Model(None)

    def factory(cancelled, receive):
        return model.attach(receive)

    kernel = StreamKernel(
        store=store,
        model_factory=factory,
        admit=ShortContextAdmission(
            store, lambda run, state, text: ShortRequest(text), evidence_clock=lambda: 100
        ),
        review_prefix=lambda run, request, text, final: reviewed(run),
        present=lambda event: "actual-host",
        clock=lambda: 100,
        lease_us=50,
        deadline_us=1000,
        renewal_margin_us=10,
        observe=events.append,
    )
    assert kernel.generate(run, store.snapshot(A), "你好")["outcome"] == "completed"
    assert [event["event"] for event in events][-1] == "settled"
    assert [e["sequence"] for e in events if e["event"] == "delta"] == [1]
    assert store.stream_state(A, run.input_id)["confirmed_text"] == FIRST


def test_parallel_cancel_during_in_flight_review_publishes_nothing(tmp_path):
    store = SQLiteStreamingStore(tmp_path / "cancel-review.sqlite3")
    run = claim(store, store.accept(config(), "first", "你好"))
    review_started, release, seen = Event(), Event(), []

    class Model(ScriptedModel):
        def invoke(self, body):
            self.receive(UnapprovedTextDelta(1, FIRST + "\n\n"))
            assert review_started.wait(2)
            return {"text": FIRST, "tool_calls": [], "finish_reason": "stop"}

        def cancel(self):
            release.set()

    def review(text, final):
        review_started.set()
        assert release.wait(2)
        return approved_prefix(run, text, final)

    model = Model(None)
    dialogue = dialogue_for(store, run, model, review=review,
                            present=lambda event: seen.append(event) or "host-1")

    def cancel():  # Cancellation arrives while the assessment is still running.
        assert review_started.wait(2)
        dialogue.cancelled.set()
        release.set()

    import threading
    cancelling = threading.Thread(target=cancel)
    cancelling.start()
    with pytest.raises(InterruptedError):
        dialogue.generate(run, ShortRequest("你好"))
    cancelling.join(2)
    state = store.stream_state(A, run.input_id)
    assert state["outcome"] == "cancelled"
    assert state["approved_units"] == state["confirmed_units"] == 0
    assert seen == []


@pytest.mark.parametrize("failure", ["review_error", "unknown_ack"])
def test_parallel_internal_failure_is_failed_without_user_cancel(
    tmp_path, failure, record_property,
):
    store = SQLiteStreamingStore(tmp_path / "internal-failure.sqlite3")
    run = claim(store, store.accept(config(), "first", "你好"))
    events, seen = [], []

    class Model(ScriptedModel):
        def invoke(self, body):
            self.receive(UnapprovedTextDelta(1, FIRST + "\n\n"))
            assert self.cancelled.wait(2), "Internal failure did not abort generation"
            # The aborted producer must not publish another observation or unit.
            self.receive(UnapprovedTextDelta(2, TAIL))
            pytest.fail("Generation continued after the review/presentation failure")

    def review(text, final):
        if failure == "review_error":
            raise OSError("synthetic review transport failure")
        return approved_prefix(run, text, final)

    model = Model(None)
    dialogue = dialogue_for(
        store, run, model, review=review, observe=events.append,
        present=lambda event: seen.append(event),
    )
    error = OSError if failure == "review_error" else StreamDeliveryUnknown
    with pytest.raises(error):
        dialogue.generate(run, ShortRequest("你好"))
    state = store.stream_state(A, run.input_id)
    record_property("failure", failure)
    record_property("external_cancelled", dialogue.cancelled.is_set())
    record_property("stream_state", json.dumps(state, ensure_ascii=False))
    record_property("timeline", json.dumps(events, ensure_ascii=False))
    assert state["outcome"] == "failed"
    assert not dialogue.cancelled.is_set()
    assert model.cancelled.is_set()
    assert state["confirmed_units"] == 0 and state["confirmed_text"] == ""
    assert state["approved_units"] == state["unknown_units"] == len(seen) == (
        0 if failure == "review_error" else 1
    )
    assert [event["sequence"] for event in events if event["event"] == "delta"] == [1]
    assert events[-1]["event"] == "settled" and events[-1]["outcome"] == "failed"


@pytest.mark.parametrize("ending", ["review_error", "unknown_ack", "cancel"])
def test_default_stream_observation_matches_persisted_terminal(
    tmp_path, ending, record_property,
):
    from test_default_text_delivery import StreamingModel, setup, wait_for

    runtime, text, memory, mirror, model = setup(
        tmp_path, StreamingModel(early="transport" if ending == "review_error" else "approve")
    )
    events = []
    observe = runtime.observe_stream

    def observe_stream(event):
        observe(event)
        events.append(event)

    runtime.observe_stream = observe_stream
    try:
        text.send("合成输入")
        if ending != "review_error":
            first = wait_for(lambda: runtime.poll()["event"])
            assert not model.terminal.is_set()
            if ending == "cancel":
                runtime.ack(dict(sequence=first["sequence"], fingerprint=first["fingerprint"],
                                 host_receipt="synthetic-known-prefix"))
                runtime.stop()
                text.stop()
        wait_for(lambda: not text.snapshot()["busy"])
        state = memory.store.stream_state(memory.companion, runtime.claim.input_id)
        session = text.snapshot()
        record_property("ending", ending)
        record_property("external_cancelled", runtime.cancelled.is_set())
        record_property("stream_state", json.dumps(state, ensure_ascii=False))
        record_property("session_state", json.dumps(session, ensure_ascii=False))
        record_property("timeline", json.dumps(events, ensure_ascii=False))
        outcome = "cancelled" if ending == "cancel" else "failed"
        assert state["outcome"] == session["attempt"]["status"] == outcome
        assert runtime.cancelled.is_set() == (ending == "cancel")
        settled = [event["outcome"] for event in events if event["event"] == "settled"]
        assert settled == [outcome]
        assert state["confirmed_units"] == (1 if ending == "cancel" else 0)
        if ending == "cancel":
            assert state["confirmed_text"] == first["text"]
            assert mirror.history()["messages"][-1]["content"] == first["text"]
        else:
            assert state["confirmed_text"] == ""
    finally:
        model.release.set()
        memory.close()
        mirror.close()


def test_parallel_provider_failure_retains_only_the_confirmed_prefix(tmp_path):
    store = SQLiteStreamingStore(tmp_path / "provider-failure.sqlite3")
    run = claim(store, store.accept(config(), "first", "你好"))
    confirmed, events, seen = Event(), [], []

    class Model(ScriptedModel):
        def invoke(self, body):
            self.receive(UnapprovedTextDelta(1, FIRST + "\n\n"))
            assert confirmed.wait(2), "First unit was not confirmed while generating"
            self.receive(UnapprovedTextDelta(2, TAIL))
            raise OSError("synthetic generation transport failure")

    def observe(event):
        events.append(event)
        if event["event"] == "confirmed":
            confirmed.set()

    dialogue = dialogue_for(
        store, run, Model(None), observe=observe,
        present=lambda event: seen.append(event) or "host-first",
    )
    with pytest.raises(OSError, match="synthetic generation transport failure"):
        dialogue.generate(run, ShortRequest("你好"))
    state = store.stream_state(A, run.input_id)
    assert state["outcome"] == "failed" and not dialogue.cancelled.is_set()
    assert state["confirmed_text"] == FIRST and state["confirmed_units"] == 1
    assert state["approved_units"] == 1 and state["unknown_units"] == 0
    assert [event["text"] for event in seen] == [FIRST]
    names = [event["event"] for event in events]
    assert names.index("confirmed") < names.index("model_terminal")
    assert events[-1]["outcome"] == "failed"


def test_parallel_unknown_receipt_after_confirmed_prefix_never_replays(tmp_path):
    store = SQLiteStreamingStore(tmp_path / "unknown-tail.sqlite3")
    run = claim(store, store.accept(config(), "first", "你好"))
    displayed, seen = Event(), []

    class Model(ScriptedModel):
        def invoke(self, body):
            self.receive(UnapprovedTextDelta(1, FIRST + "\n\n"))
            assert displayed.wait(2)
            self.receive(UnapprovedTextDelta(2, TAIL))
            return {"text": FULL, "tool_calls": [], "finish_reason": "stop"}

    def present(event):
        seen.append(event)
        if len(seen) == 1:
            displayed.set()
            return "host-1"
        return None

    dialogue = dialogue_for(store, run, Model(None), present=present)
    with pytest.raises(StreamDeliveryUnknown):
        dialogue.generate(run, ShortRequest("你好"))
    state = store.stream_state(A, run.input_id)
    assert state["confirmed_units"] == 1 and state["unknown_units"] == 1
    assert state["confirmed_text"] == FIRST
    assert len(seen) == 2
    with pytest.raises(StreamDeliveryUnknown):
        dialogue.generate(run, ShortRequest("你好"))
    assert len(seen) == 2
