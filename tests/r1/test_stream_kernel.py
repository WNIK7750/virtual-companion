from threading import Event, get_ident

import pytest
from test_rebuilt_turns import A, claim, config
from test_segment_store import reviewed
from virtual_companion.boundary.contracts import VersionConflict
from virtual_companion.boundary.sqlite_streaming import SQLiteStreamingStore
from virtual_companion.cognition.preparation import CallCancelled
from virtual_companion.cognition.short_context import ShortContextAdmission
from virtual_companion.cognition.short_dialogue import ShortRequest
from virtual_companion.providers.text_deltas import UnapprovedTextDelta


def setup(tmp_path, *, correction_at=None):
    from virtual_companion.cognition.stream_kernel import StreamKernel

    store = SQLiteStreamingStore(tmp_path / "kernel.sqlite3")
    run = claim(store, store.accept(config(), "first", "你好"))
    seen, sent = [], []
    interrupted = Event()

    def correction():
        store.accept(config(), "correction", "更正，我不喜欢茶")

    class Model:
        def count(self, body):
            if correction_at == "count":
                correction()
            return 10

        def invoke(self, body):
            sent.append(body)
            self.receive(UnapprovedTextDelta(1, "我在这里。"))
            return {"text": "我在这里。", "tool_calls": [], "finish_reason": "stop"}

        def cancel(self):
            interrupted.set()

    model = Model()

    def factory(cancelled, receive):
        model.receive, model.cancelled = receive, cancelled
        return model

    def review(run, request, text, final):
        if correction_at == "review":
            correction()
        return reviewed(run)

    kernel = StreamKernel(
        store=store,
        model_factory=factory,
        admit=ShortContextAdmission(
            store, lambda run, state, text: ShortRequest(text), evidence_clock=lambda: 100
        ),
        review_prefix=review,
        present=lambda event: seen.append(event) or "actual-host",
        clock=lambda: 100,
        lease_us=50,
        deadline_us=1000,
        renewal_margin_us=10,
    )
    return store, run, kernel, model, seen, sent, interrupted


def test_stream_kernel_uses_real_context_and_settles_without_self_cancellation(tmp_path):
    store, run, kernel, model, seen, sent, interrupted = setup(tmp_path)
    result = kernel.generate(run, store.snapshot(A), "你好")
    assert result["outcome"] == "completed"
    assert len(seen) == len(sent) == 1
    # The adopted prose rendering states the same barrier fact without JSON syntax.
    evidence = sent[0]["messages"][1]["content"]
    assert "以下是已确认的背景资料" in evidence and "尚未完成" in evidence
    assert "{" not in evidence and '"' not in evidence
    assert not interrupted.is_set()
    assert store.stream_state(A, run.input_id)["confirmed_text"] == "我在这里。"


@pytest.mark.parametrize("arrival", ["count", "review"])
def test_correction_during_count_or_review_blocks_stale_stream(tmp_path, arrival):
    store, run, kernel, model, seen, sent, interrupted = setup(tmp_path, correction_at=arrival)
    with pytest.raises(VersionConflict):
        kernel.generate(run, store.snapshot(A), "你好")
    assert not seen
    assert len(sent) == (arrival == "review")
    assert store.stream_state(A, run.input_id)["outcome"] == "failed"


def test_guardian_cancels_blocking_model_without_waiting_for_delta(tmp_path):
    store, run, kernel, model, seen, sent, interrupted = setup(tmp_path)

    def blocked(body):
        store.cancel(A, run.input_id)
        assert interrupted.wait(2), "Guardian failed to interrupt a silent transport"
        raise InterruptedError("transport cancelled")

    model.invoke = blocked
    with pytest.raises(CallCancelled):
        kernel.generate(run, store.snapshot(A), "你好")
    assert model.cancelled.is_set()
    assert not seen
    assert store.stream_state(A, run.input_id)["outcome"] == "cancelled"


def test_rejected_admission_closes_run_without_model_or_display(tmp_path):
    store, run, kernel, model, seen, sent, interrupted = setup(tmp_path)
    kernel.admit = lambda *args: ShortRequest("伪造输入")
    with pytest.raises(ValueError, match="replace"):
        kernel.generate(run, store.snapshot(A), "你好")
    assert not sent and not seen
    assert interrupted.is_set()
    assert store.stream_state(A, run.input_id)["outcome"] == "failed"


@pytest.mark.parametrize("field,value", [
    ("companion", "00000000-0000-0000-0000-000000000002"),
    ("input_id", "foreign-input"),
    ("state_version", 999),
])
def test_stream_context_must_belong_to_current_claim(tmp_path, field, value):
    from dataclasses import replace

    store, run, kernel, model, seen, sent, interrupted = setup(tmp_path)
    original = kernel.admit

    class WrongBinding:
        def __call__(self, claim, snapshot, text):
            request = original(claim, snapshot, text)
            return replace(request, context=replace(request.context, **{field: value}))

        def validate(self, context):
            pass  # A read-set check alone cannot prove request ownership.

    kernel.admit = WrongBinding()
    with pytest.raises(ValueError, match="context"):
        kernel.generate(run, store.snapshot(A), "你好")
    assert not sent and not seen
    assert store.stream_state(A, run.input_id)["outcome"] == "failed"


def test_lease_renewed_during_count_before_streaming_dispatch(tmp_path):
    store, run, kernel, model, seen, sent, interrupted = setup(tmp_path)
    now = [100]
    kernel.clock = lambda: now[0]
    original = model.count

    def count(body):
        now[0] = 145  # Original lease expires at 150.
        return original(body)

    model.count = count
    original_review = kernel.review_prefix

    def review(*args):
        now[0] = 160
        return original_review(*args)

    kernel.review_prefix = review
    assert kernel.generate(run, store.snapshot(A), "你好")["outcome"] == "completed"
    assert seen and not interrupted.is_set()


def test_review_session_factory_binds_claim_request_and_model_cancellation(tmp_path):
    from virtual_companion.cognition.stream_unit_review import BoundedUnitReviewer

    store, run, kernel, model, seen, _, _ = setup(tmp_path)
    bindings = []

    def session_factory(bound_claim, request, cancelled):
        bindings.append((bound_claim, request, cancelled))

        def assess(packet):
            return {
                "request_sha256": packet["sha256"],
                "decision": "approve",
                "self_contained": True,
                "no_open_dependency": True,
                "consistent_with_displayed": True,
                "constraints_satisfied": True,
            }

        return BoundedUnitReviewer(bound_claim, request, assess)

    kernel.review_prefix = None
    kernel.review_session_factory = session_factory
    assert kernel.generate(run, store.snapshot(A), "你好")["outcome"] == "completed"
    assert len(bindings) == 1 and bindings[0][0] == run
    assert bindings[0][1].user_text == "你好"
    assert bindings[0][2] is model.cancelled
    assert len(seen) == 1


def test_review_session_cancellation_during_assessment_never_dispatches(tmp_path):
    from virtual_companion.cognition.stream_unit_review import BoundedUnitReviewer

    store, run, kernel, _, seen, _, _ = setup(tmp_path)

    def session_factory(bound_claim, request, cancelled):
        def assess(packet):
            cancelled.set()
            return {
                "request_sha256": packet["sha256"],
                "decision": "approve",
                "self_contained": True,
                "no_open_dependency": True,
                "consistent_with_displayed": True,
                "constraints_satisfied": True,
            }

        return BoundedUnitReviewer(bound_claim, request, assess)

    kernel.review_prefix = None
    kernel.review_session_factory = session_factory
    with pytest.raises((InterruptedError, CallCancelled)):
        kernel.generate(run, store.snapshot(A), "你好")
    assert not seen
    assert store.stream_state(A, run.input_id)["outcome"] == "cancelled"


def test_stream_kernel_can_opt_into_parallel_review_with_same_owner_checks(tmp_path):
    store, run, kernel, model, seen, _, _ = setup(tmp_path)
    first_visible = Event()
    model_thread = []
    kernel.parallel_review = True

    def review(claim, request, text, final):
        if not final:
            assert get_ident() != model_thread[0], "Review reentered model callback thread"
        return (reviewed(run, ("我在这里。", "你想聊什么？")) if final else
                reviewed(run) if text.startswith("我在这里。\n\n") else None)

    kernel.review_prefix = review

    def present(event):
        seen.append(event)
        first_visible.set()
        return f"device-{len(seen)}"

    kernel.present = present

    def invoke(body):
        model_thread.append(get_ident())
        model.receive(UnapprovedTextDelta(1, "我在这里。\n\n"))
        assert first_visible.wait(2), "Kernel did not present before model terminal"
        model.receive(UnapprovedTextDelta(2, "你想聊什么？"))
        return {"text": "我在这里。\n\n你想聊什么？",
                "tool_calls": [], "finish_reason": "stop"}

    model.invoke = invoke
    assert kernel.generate(run, store.snapshot(A), "你好")["outcome"] == "completed"
    assert [event["text"] for event in seen] == ["我在这里。", "你想聊什么？"]
