import time
from threading import Event

import pytest
from test_rebuilt_turns import A, claim, config
from test_segment_store import reviewed
from virtual_companion.boundary.sqlite_streaming import SQLiteStreamingStore
from virtual_companion.cognition.short_dialogue import ShortRequest
from virtual_companion.providers.text_deltas import UnapprovedTextDelta


def test_stream_state_reports_real_outcome_instead_of_legacy_cancel_flag(tmp_path):
    store = SQLiteStreamingStore(tmp_path / "stream.sqlite3")
    run = claim(store, store.accept(config(), "first", "你好"))
    store.finish_stream(run, outcome="failed", now_us=100)
    assert store.stream_state(A, run.input_id)["outcome"] == "failed"


def scenario(tmp_path, *, missing_ack=False, fail=False):
    from virtual_companion.cognition.stream_dialogue import StreamDialogue
    store = SQLiteStreamingStore(tmp_path / "stream.sqlite3")
    run = claim(store, store.accept(config(), "first", "你好"))
    observed, requests = [], []

    class Model:
        def count(self, body):
            requests.append(body)
            return 10

        def invoke(self, body):
            assert body == requests[0]
            self.receive(UnapprovedTextDelta(1, "我在这里。"))
            assert observed  # Real delivery before final model return.
            if fail:
                raise RuntimeError("synthetic model failure")
            return {"text": "我在这里。", "tool_calls": [], "finish_reason": "stop"}

        def cancel(self):
            self.was_cancelled = True

    model = Model()

    def factory(receive):
        model.receive = receive
        return model

    def present(event):
        assert store.segments(A, run.input_id)[0]["status"] == "unknown"
        observed.append(event)
        return None if missing_ack else "host-visible-1"

    coordinator = StreamDialogue(store=store, model_factory=factory,
        review_prefix=lambda text, final: reviewed(run) if text == "我在这里。" else None,
        present=present, checkpoint=lambda: None, cancelled=Event(), clock=lambda: 100)
    return store, run, model, coordinator, observed


def test_model_delta_is_approved_persisted_displayed_and_confirmed_before_terminal(tmp_path):
    store, run, model, coordinator, observed = scenario(tmp_path)
    result = coordinator.generate(run, ShortRequest("你好"))
    assert result["outcome"] == "completed"
    state = store.stream_state(A, run.input_id)
    assert state["outcome"] == "completed" and state["confirmed_units"] == 1
    assert state["confirmed_text"] == "我在这里。"
    assert len(observed) == 1


@pytest.mark.parametrize("missing_ack,fail", [(True, False), (False, True)])
def test_partial_failure_never_replays_or_fabricates_history(tmp_path, missing_ack, fail):
    store, run, model, coordinator, observed = scenario(tmp_path, missing_ack=missing_ack,
                                                       fail=fail)
    with pytest.raises(RuntimeError):
        coordinator.generate(run, ShortRequest("你好"))
    state = store.stream_state(A, run.input_id)
    assert state["outcome"] == "failed"
    assert state["confirmed_units"] == int(not missing_ack)
    assert state["unknown_units"] == int(missing_ack)
    assert model.was_cancelled and len(observed) == 1


def test_real_bound_transport_reaches_confirmed_history(tmp_path):
    from test_provider_loopback import Ledger, client, server
    from virtual_companion.cognition.stream_dialogue import StreamDialogue
    from virtual_companion.providers.bound_call import BoundProviderCall
    store = SQLiteStreamingStore(tmp_path / "real-http.sqlite3")
    run = claim(store, store.accept(config(), "first", "你好"))
    cancelled, ledger = Event(), Ledger()
    with server("normal") as (port, _, _):
        transport, _ = client(port, ledger)

        def factory(receive):
            return BoundProviderCall(transport.route, transport=transport,
                counter=lambda body: 20, stage="reply", cancelled=cancelled,
                clock=time.monotonic, deadline=time.monotonic() + 2,
                output_tokens=32, data_scope=("current",), on_content=receive)

        def present(event):
            assert not ledger.finished
            assert event["text"] == "合成🌿"
            return "loopback-host-visible"

        coordinator = StreamDialogue(store=store, model_factory=factory,
            review_prefix=lambda text, final: reviewed(run, (text,)), present=present,
            checkpoint=lambda: None, cancelled=cancelled, clock=lambda: 100, output_tokens=32)
        assert coordinator.generate(run, ShortRequest("你好"))["outcome"] == "completed"
    assert len(ledger.calls) == len(ledger.finished) == 1
    assert store.stream_state(A, run.input_id)["confirmed_text"] == "合成🌿"


def test_raw_unapproved_content_is_not_presented(tmp_path):
    store, run, model, coordinator, observed = scenario(tmp_path)
    coordinator.review_prefix = lambda text, final: None

    def invoke(body):
        model.receive(UnapprovedTextDelta(1, "我在这里。"))
        return {"text": "我在这里。", "tool_calls": [], "finish_reason": "stop"}

    model.invoke = invoke
    with pytest.raises(ValueError, match="approval"):
        coordinator.generate(run, ShortRequest("你好"))
    assert not observed
    assert store.stream_state(A, run.input_id)["approved_units"] == 0


def test_cancel_during_actual_display_keeps_receipt_and_stops_generation(tmp_path):
    store, run, model, coordinator, observed = scenario(tmp_path)
    original = coordinator.present

    def present(event):
        receipt = original(event)
        coordinator.cancelled.set()
        return receipt

    coordinator.present = present
    with pytest.raises(InterruptedError):
        coordinator.generate(run, ShortRequest("你好"))
    state = store.stream_state(A, run.input_id)
    assert state["outcome"] == "cancelled" and state["confirmed_units"] == 1
    assert len(observed) == 1


def test_late_confirmation_updates_live_read_model_not_original_settlement(tmp_path):
    store, run, model, coordinator, observed = scenario(tmp_path, missing_ack=True)
    with pytest.raises(RuntimeError):
        coordinator.generate(run, ShortRequest("你好"))
    event = observed[0]
    store.confirm_segment(A, run.input_id, 1, fingerprint=event["fingerprint"],
                          host_receipt="late-device-receipt", channel="text_visible")
    state = store.stream_state(A, run.input_id)
    assert state["outcome"] == "failed" and state["confirmed_units"] == 1
    assert state["settlement"]["confirmed_at_settlement"] == 0
    with pytest.raises(RuntimeError):
        coordinator.generate(run, ShortRequest("你好"))
    assert len(observed) == 1


def test_parallel_first_boundary_blocks_fast_producer_until_durable_ack(tmp_path):
    from hashlib import sha256

    from virtual_companion.cognition.stream_dialogue import StreamDialogue
    from virtual_companion.cognition.stream_unit_review import BoundedUnitReviewer

    prefix, suffix = "First faithful sentence!", " A final continuation."
    store = SQLiteStreamingStore(tmp_path / "parallel-backpressure.sqlite3")
    run = claim(store, store.accept(config(), "first", "synthetic input"))
    request = ShortRequest("synthetic input")
    review_started, producer_advanced, terminal = Event(), Event(), Event()
    lifecycle, ack_snapshots, opinions = [], [], []

    class Model:
        def count(self, body):
            return 10

        def invoke(self, body):
            self.receive(UnapprovedTextDelta(1, prefix))
            producer_advanced.set()
            assert review_started.wait(1)
            self.receive(UnapprovedTextDelta(2, suffix))
            return {"text": prefix + suffix, "tool_calls": [], "finish_reason": "stop"}

        def cancel(self):
            pass

    model = Model()

    def factory(receive):
        model.receive = receive
        return model

    def assess(envelope):
        packet = envelope["payload"]
        opinions.append(packet)
        if not packet["final"]:
            review_started.set()
            # The old producer reaches EOF while the first opinion is pending.
            # Backpressure must instead keep it inside receive until this ACK.
            terminal.wait(0.1)
        return {"request_sha256": envelope["sha256"], "decision": "approve",
                "self_contained": True, "no_open_dependency": True,
                "consistent_with_displayed": True, "constraints_satisfied": True}

    def observe(event):
        lifecycle.append(event["event"])
        if event["event"] == "model_terminal":
            terminal.set()
        if event["event"] == "confirmed" and event["sequence"] == 1:
            state = store.stream_state(A, run.input_id)
            ack_snapshots.append({"terminal": terminal.is_set(),
                                  "producer_advanced": producer_advanced.is_set(),
                                  "outcome": state["outcome"],
                                  "prefix_sha256": sha256(
                                      state["confirmed_text"].encode()).hexdigest()})

    reviewer = BoundedUnitReviewer(run, request, assess, early_limit=1)
    coordinator = StreamDialogue(store=store, model_factory=factory,
        review_prefix=reviewer, present=lambda event: f"visible-{event['sequence']}",
        checkpoint=lambda: None, cancelled=Event(), clock=lambda: 100,
        parallel_review=True, observe=observe)
    assert coordinator.generate(run, request)["outcome"] == "completed"
    assert ack_snapshots[0]["terminal"] is False, (
        f"Terminal preceded first durable ACK: {lifecycle}")
    assert ack_snapshots[0]["producer_advanced"] is False
    assert ack_snapshots[0]["outcome"] == "unsettled"
    assert ack_snapshots[0]["prefix_sha256"] == sha256(prefix.encode()).hexdigest()
    assert lifecycle.index("confirmed") < lifecycle.index("model_terminal")
    assert lifecycle[lifecycle.index("confirmed") + 1:].count("delta") == 1
    assert [packet["final"] for packet in opinions] == [False, True]
    assert reviewer.early_calls == reviewer.final_calls == 1


def test_parallel_without_boundary_keeps_the_existing_final_opinion(tmp_path):
    from virtual_companion.cognition.stream_unit_review import BoundedUnitReviewer

    store, run, model, coordinator, visible = scenario(tmp_path)
    text, opinions = "No sentence boundary", []
    request = ShortRequest("你好")

    def assess(envelope):
        opinions.append(envelope["payload"])
        return {"request_sha256": envelope["sha256"], "decision": "approve",
                "self_contained": True, "no_open_dependency": True,
                "consistent_with_displayed": True, "constraints_satisfied": True}

    def invoke(body):
        model.receive(UnapprovedTextDelta(1, text))
        assert visible == []
        return {"text": text, "tool_calls": [], "finish_reason": "stop"}

    reviewer = BoundedUnitReviewer(run, request, assess, early_limit=1)
    model.invoke = invoke
    coordinator.parallel_review = True
    coordinator.review_prefix = reviewer
    assert coordinator.generate(run, request)["outcome"] == "completed"
    assert [packet["final"] for packet in opinions] == [True]
    assert reviewer.early_calls == 0 and reviewer.final_calls == 1
    assert store.stream_state(A, run.input_id)["confirmed_text"] == text


@pytest.mark.parametrize("failure", ["reject", "cancel"])
def test_parallel_first_boundary_rejection_or_external_cancel_never_consumes_tail(
    tmp_path, failure,
):
    from threading import Thread

    from virtual_companion.cognition.stream_dialogue import StreamDialogue
    from virtual_companion.cognition.stream_unit_review import UserConstraintViolation

    prefix = "First faithful sentence!"
    store = SQLiteStreamingStore(tmp_path / "parallel-abort.sqlite3")
    run = claim(store, store.accept(config(), "first", "synthetic input"))
    cancelled, reviewing, release, advanced = Event(), Event(), Event(), Event()
    errors, visible = [], []

    class Model:
        def count(self, body):
            return 10

        def invoke(self, body):
            self.receive(UnapprovedTextDelta(1, prefix))
            advanced.set()
            self.receive(UnapprovedTextDelta(2, " Unapproved suffix"))
            return {"text": prefix + " Unapproved suffix", "tool_calls": [],
                    "finish_reason": "stop"}

        def cancel(self):
            release.set()

    model = Model()

    def factory(receive):
        model.receive = receive
        return model

    def review(text, final):
        reviewing.set()
        assert release.wait(2)
        if failure == "reject":
            raise UserConstraintViolation("Synthetic unit rejected")
        return reviewed(run, (prefix,))

    coordinator = StreamDialogue(store=store, model_factory=factory,
        review_prefix=review, present=lambda event: visible.append(event) or "visible",
        checkpoint=lambda: None, cancelled=cancelled, clock=lambda: 100, parallel_review=True)

    def generate():
        try:
            coordinator.generate(run, ShortRequest("synthetic input"))
        except Exception as error:
            errors.append(error)

    worker = Thread(target=generate, daemon=True)
    worker.start()
    try:
        assert reviewing.wait(1)
        if failure == "cancel":
            cancelled.set()
        release.set()
        worker.join(2)
        assert not worker.is_alive()
    finally:
        cancelled.set()
        release.set()
        worker.join(2)
    assert len(errors) == 1
    assert isinstance(
        errors[0], UserConstraintViolation if failure == "reject" else InterruptedError,
    )
    assert not advanced.is_set() and visible == []
    reopened = SQLiteStreamingStore(store.path)
    state = reopened.stream_state(A, run.input_id)
    assert state["outcome"] == ("failed" if failure == "reject" else "cancelled")
    assert state["confirmed_text"] == "" and state["approved_units"] == 0


@pytest.mark.parametrize("parallel", [False, True])
def test_real_stream_eof_preserves_confirmed_prefix_without_replay(tmp_path, parallel):
    from threading import Thread

    from test_provider_loopback import Ledger, client, incomplete_stream_server
    from virtual_companion.cognition.stream_dialogue import StreamDeliveryUnknown, StreamDialogue
    from virtual_companion.providers.bound_call import BoundProviderCall
    from virtual_companion.providers.httpx_transport import ProviderFailure

    prefix, suffix = "合成🌿。", "还有未完成的"
    store = SQLiteStreamingStore(tmp_path / "stream-eof.sqlite3")
    run = claim(store, store.accept(config(), "first", "你好"))
    cancelled, ledger = Event(), Ledger()
    visible, reviews, lifecycle, results, failures = [], [], [], [], []

    with incomplete_stream_server(prefix, suffix) as (port, requests, continue_stream):
        transport, _ = client(port, ledger)

        def factory(receive):
            return BoundProviderCall(transport.route, transport=transport,
                counter=lambda body: 20, stage="reply", cancelled=cancelled,
                clock=time.monotonic, deadline=time.monotonic() + 2,
                output_tokens=32, data_scope=("current",), on_content=receive)

        def review(text, final):
            reviews.append((text, final))
            return reviewed(run, (prefix,)) if text == prefix else None

        def present(event):
            assert not ledger.finished and transport.metrics["finished"] is None
            assert event["text"] == prefix
            assert store.segments(A, run.input_id)[0]["status"] == "unknown"
            visible.append(event)
            return "loopback-visible-before-eof"

        def observe(event):
            lifecycle.append(event["event"])
            if event["event"] == "confirmed":
                assert not ledger.finished and transport.metrics["finished"] is None
                state = store.stream_state(A, run.input_id)
                assert state["outcome"] == "unsettled" and state["confirmed_units"] == 1
                assert state["confirmed_text"] == prefix
                continue_stream.set()

        coordinator = StreamDialogue(store=store, model_factory=factory,
            review_prefix=review, present=present, checkpoint=lambda: None,
            cancelled=cancelled, clock=lambda: 100, output_tokens=32, observe=observe,
            parallel_review=parallel)

        def generate():
            try:
                results.append(coordinator.generate(run, ShortRequest("你好")))
            except Exception as error:
                failures.append(error)

        worker = Thread(target=generate, daemon=True)
        worker.start()
        try:
            worker.join(3)
            assert not worker.is_alive(), "EOF failure must settle without hanging"
        finally:
            if worker.is_alive():
                cancelled.set()
                transport.cancel()
                worker.join(2)

        assert results == [] and len(failures) == 1
        assert isinstance(failures[0], ProviderFailure)
        assert str(failures[0]) == "provider_stream_incomplete"
        assert failures[0].__cause__ is None and not cancelled.is_set()
        assert reviews[0] == (prefix, False)
        if not parallel:
            assert reviews == [(prefix, False), (prefix + suffix, False)]
        assert lifecycle.count("delta") == 2
        assert lifecycle.count("confirmed") == lifecycle.count("approved") == 1
        assert lifecycle[lifecycle.index("confirmed") + 1:].count("delta") == 1
        assert lifecycle.index("confirmed") < lifecycle.index("model_terminal")
        assert len(visible) == len(requests) == len(ledger.calls) == len(ledger.finished) == 1
        assert ledger.calls[0]["stage"] == "reply"
        assert ledger.finished == [{"receipt": "1", "status": "failed", "usage": None,
                                    "transmission": "response_received"}]
        assert transport.metrics["first_content"] < transport.metrics["finished"]

        with pytest.raises(StreamDeliveryUnknown, match="reconciliation"):
            coordinator.generate(run, ShortRequest("你好"))
        assert len(visible) == len(requests) == len(ledger.calls) == len(ledger.finished) == 1

    reopened = SQLiteStreamingStore(store.path)
    state = reopened.stream_state(A, run.input_id)
    assert state["outcome"] == "failed" and state["queue_closed"]
    assert state["confirmed_text"] == prefix
    assert state["confirmed_units"] == state["approved_units"] == 1
    assert state["unknown_units"] == state["pending_units"] == 0
    assert state["settlement"]["confirmed_at_settlement"] == 1
    assert [segment["status"] for segment in reopened.segments(A, run.input_id)] == ["confirmed"]
    assert reopened.inspect(A, run.input_id)["delivery"] is None
    jobs = [source for source in reopened.background_sources(A) if source["role"] == "assistant"]
    assert len(jobs) == 1
    from hashlib import sha256
    assert jobs[0]["source_hash"] == sha256(prefix.encode()).hexdigest()
    assert jobs[0]["source"] == "segment-" + visible[0]["fingerprint"]
    next_item = reopened.accept(config(), "next", "继续", session_id=run.session_id)
    history, _ = reopened.exchange_context(claim(reopened, next_item, now=200))
    assert [source.text for source in history] == ["你好", prefix, "继续"]
