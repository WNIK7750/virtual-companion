"""Cross-module regressions from the September 27 independent audit."""

import json
from datetime import UTC, datetime

import pytest
from test_rebuilt_turns import A, B, claim, config
from test_segment_store import reviewed
from virtual_companion.boundary.contracts import Scope
from virtual_companion.boundary.memory import FactKey, MemoryMutation
from virtual_companion.boundary.sqlite_memory_management import SQLiteExplicitMemoryStore
from virtual_companion.cognition.background_worker import UserFactWorker
from virtual_companion.cognition.short_context import ShortContextAdmission
from virtual_companion.cognition.short_dialogue import ShortRequest
from virtual_companion.text.memory_runtime import _fake_memory_result


def complete_source(store, item):
    def model(stage, messages, schema, cancelled):
        if stage == "fact_target_select":
            return {"selected": [], "uncertain": False, "intent": "other"}
        return _fake_memory_result(stage, messages, schema, cancelled)

    receipt = UserFactWorker(
        store, model, clock=lambda: 10,
        resolution_protocol=True, resolution_strategy="intent_first", memory_commands=True,
    ).run_one(A, "worker", source=item.input_id)
    assert receipt is not None


def assistant_backlog(store, count):
    for batch, start in enumerate(range(0, count, 26)):
        item = store.accept(config(), f"turn-{batch}", "你好")
        complete_source(store, item)
        run = claim(store, item)
        texts = tuple(f"第{index}段。" for index in range(start, min(start + 26, count)))
        for sequence in range(1, len(texts) + 1):
            store.approve_segment(run, reviewed(run, texts[:sequence]), now_us=100)
            event = store.dispatch_segment(run, sequence, now_us=100)
            store.confirm_segment(
                A, item.input_id, sequence, fingerprint=event["fingerprint"],
                host_receipt=f"receipt-{batch}-{sequence}", channel="text_visible",
            )
        store.finish_stream(run, outcome="completed", now_us=100)


@pytest.mark.parametrize("count", [99, 100, 101, 255, 256])
def test_source_status_sees_user_beyond_assistant_display_page(tmp_path, count):
    store = SQLiteExplicitMemoryStore(tmp_path / "memory.db")
    assistant_backlog(store, count)
    assert store.memory_status(A) == "current"
    assert not store.memory_barrier(A)
    if count == store.max_pending_sources:
        with pytest.raises(ValueError, match="backpressure"):
            store.accept(config(), "correction", "之前的偏好说错了")
        assert store.memory_status(A) == "current"
        assert not store.memory_barrier(A)
        return
    current = store.accept(config(), "correction", "之前的偏好说错了")
    assert store.memory_barrier(A)
    assert store.memory_status(A) == "pending_source"
    assert len(store.background_sources(A)) <= 100
    run = claim(store, current)
    admission = ShortContextAdmission(
        store, lambda *args: ShortRequest(args[-1]), evidence_clock=lambda: 100,
    )
    request = admission(run, store.snapshot(A), current_text(store, current))
    context = json.loads(request.context.payload)
    assert context["memory_status"] == "pending_source"
    assert context["exchanges"] == context["memories"] == []


def current_text(store, item):
    return store.inspect(A, item.input_id)["text"]


def test_clarification_precedence_is_independent_of_page_and_companion(tmp_path):
    store = SQLiteExplicitMemoryStore(tmp_path / "memory.db")
    assistant_backlog(store, 104)
    item = store.accept(config(), "clarify", "是另一条，先别用了")
    work = store.claim_background(A, "worker", now_us=10, lease_us=100,
                                  role="user", source=item.input_id)
    store.protect_background(work, error="needs_clarification.memory_command", now_us=11)
    store.accept(config(), "later", "等一下")
    store.register(B)
    assert store.memory_status(A) == "needs_clarification"
    assert store.memory_status(B) == "current"
    assert SQLiteExplicitMemoryStore(store.path).memory_status(A) == "needs_clarification"


@pytest.mark.parametrize("tail", [9, 105])
def test_candidate_recalls_relevant_fact_beyond_early_rows(tmp_path, tail):
    store = SQLiteExplicitMemoryStore(tmp_path / "memory.db")
    store.register(A)
    owner = store.claim_owner(A, "seed", expected_epoch=0)
    for index in range(tail):
        content = "我的小猫叫团团" if index == tail - 1 else f"第{index}条无关资料"
        mutation = MemoryMutation(
            Scope(A, "seed-session"), f"seed-{index}", "assert_fact", f"fact-{index}",
            content, "manual", f"source-{index}", "synthetic",
            datetime(2026, 9, 27, tzinfo=UTC), FactKey("user", f"key-{index}", "profile"),
        )
        store.mutate(mutation, owner, expected_version=store.snapshot(A).write_version)
    item = store.accept(config(), "query", "我的小猫叫什么？")
    complete_source(store, item)
    run = claim(store, item)
    admission = ShortContextAdmission(
        store, lambda *args: ShortRequest(args[-1]), evidence_clock=lambda: 100,
    )
    context = admission(run, store.snapshot(A), "我的小猫叫什么？").context
    assert [value["content"] for value in json.loads(context.payload)["memories"]] == [
        "我的小猫叫团团",
    ]
    admission.validate(context)
