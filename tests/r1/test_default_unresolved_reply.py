"""A semantic refusal preserves the source without blocking grounded conversation."""

import json

import pytest
from test_default_demand_integration import ack_and_finish, demand_setup
from test_default_text_delivery import wait_for
from unit_review_recording import bound_unit_packets as bound_unit_packets
from unit_review_recording import recorded_review_packet
from virtual_companion.boundary.history_readset import evidence_hash, history_readset
from virtual_companion.cognition.stream_unit_review import semantic_review_view
from virtual_companion.text.default_runtime import ACCESS_UNRESOLVED

BOOK = "前天我提到的是《城南旧事》，作者是林海音。"


def book_worker(runtime, memory, *, verdict="unsupported", fault=None):
    original = memory.scheduler.user.invoke

    def invoke(stage, messages, schema, cancelled):
        if stage == "memory_command":
            return {
                "action": "none", "requestedScope": "none", "targetIds": [],
                "evidenceQuote": "", "quoteStart": None, "quoteEnd": None,
                "replacementQuote": "", "replacementStart": None, "replacementEnd": None,
                "coverage": "not_command", "unresolved": [],
            }
        if stage == "exchange_intent":
            opinion = original(stage, messages, schema, cancelled)
            # Keep the original intent; bind only its quote to this input.
            opinion["acts"][0]["evidenceQuote"] = BOOK
            return opinion
        if stage == "fact_extract":
            return {"operations": [{
                "operation": "add", "type": "long_term", "evidenceQuote": BOOK,
                "evidenceTurnId": runtime.claim.input_id, "supersedesAtomId": None,
                "expiresAt": None, "factKey": {"subject": "user", "predicate": "book"},
            }]}
        if stage == "fact_support":
            if fault == "transport":
                raise OSError("synthetic support transport failure")
            if fault == "cancel":
                runtime.cancelled.set()
            return {"intent": "correction", "verdict": verdict}
        return original(stage, messages, schema, cancelled)

    memory.scheduler.user.invoke = invoke


@pytest.mark.parametrize("verdict", ["unsupported", "unknown"])
def test_semantic_refusal_keeps_barrier_and_uses_one_restricted_readset(tmp_path, verdict):
    runtime, text, memory, mirror, model, source, _ = demand_setup(tmp_path)
    text.send(source)
    event = wait_for(lambda: runtime.poll()["event"])
    runtime.ack({"sequence": event["sequence"], "fingerprint": event["fingerprint"],
                 "host_receipt": "synthetic-prior-history"})
    wait_for(lambda: not text.snapshot()["busy"])
    book_worker(runtime, memory, verdict=verdict)
    model.calls.clear()
    model.primary.clear()
    prepared = []
    original_prepare = runtime.prepare

    def prepare(claim, *, final):
        result = original_prepare(claim, final=final)
        prepared.append(result.request.context)
        return result

    runtime.prepare = prepare
    text.send(BOOK)
    ack_and_finish(runtime, text)
    assert len(model.primary) == 2 and len(model.calls) == 3
    assert prepared[0].state_version == prepared[1].state_version
    initial, final = [json.loads(context.payload) for context in prepared]
    assert initial["history"]  # Regression: pending source used to expose history.
    assert final["history"] == initial["history"]
    assert final["memories"] == memory.store.committed_recall(
        memory.companion, now_us=runtime.clock(), limit=8,
        candidates=tuple((item["record_id"], item["version"]) for item in final["memories"]))
    assert final["memories"] and all(item["content"] != BOOK for item in final["memories"])
    assert final["exchanges"] == memory.store.pending_exchanges(
        memory.companion, now_us=runtime.clock(), limit=8,
        candidates=tuple((item["exchange_id"], item["revision"]) for item in final["exchanges"]))
    assert all(item["source_id"] != runtime.claim.input_id for item in final["memories"])
    assert final["memory_access"] == "unresolved"
    assert final["source_processing"]["status"] == "failed"
    assert final["memory_status"] == "needs_clarification"
    assert "current_operation" not in final
    review = json.loads(model.calls[-1][-1]["content"])
    sealed = recorded_review_packet(model.calls[-1])
    assert semantic_review_view(sealed) == review
    assert sealed["context"] == prepared[-1].payload
    # Compact wire refers to the sealed personal catalog without duplicating words.
    assert review["personal_sources"] == sealed["personal_sources"]
    for key in ("memory_status", "source_processing", "memory_access", "exchanges"):
        assert review["context"][key] == final[key]
    assert [r["personal_source_id"] for r in review["context"]["memories"]] == [
        "memory:" + r["record_id"] for r in final["memories"]]
    assert ACCESS_UNRESOLVED in model.primary[-1][0]["content"]
    assert ACCESS_UNRESOLVED in model.calls[-1][0]["content"]
    assert model.primary[-1][-1]["content"] == BOOK
    sources = memory.store.background_sources(memory.companion)
    assert next(s for s in sources if s["source"] == runtime.claim.input_id)["status"] == "failed"
    assert memory.store.memory_barrier(memory.companion)
    segment = memory.store.segments(memory.companion, runtime.claim.input_id)[0]
    readset = segment["readset"]
    expected = history_readset(prepared[-1].companion, prepared[-1].input_id,
                               prepared[-1].state_version, prepared[-1].payload)
    assert readset == expected
    assert [item["id"] for item in readset["history"]] == [item["id"] for item in final["history"]]
    assert readset["context_sha256"] == evidence_hash(final)


@pytest.mark.parametrize("fault", ["transport", "cancel"])
def test_nonsemantic_failure_never_enters_restricted_answer(tmp_path, fault):
    runtime, text, memory, mirror, model, _, _ = demand_setup(tmp_path)
    book_worker(runtime, memory, fault=fault)
    text.send(BOOK)
    wait_for(lambda: not text.snapshot()["busy"])
    assert text.snapshot()["attempt"]["status"] == ("cancelled" if fault == "cancel" else "failed")
    assert len(model.primary) == 1
    assert not memory.store.segments(memory.companion, runtime.claim.input_id)
    assert memory.store.memory_barrier(memory.companion)


def test_new_input_after_semantic_refusal_cannot_rebase_reply(tmp_path, monkeypatch):
    runtime, text, memory, mirror, model, _, _ = demand_setup(tmp_path)
    book_worker(runtime, memory)
    protect = memory.store.protect_background

    def replace_owner(*args, **kwargs):
        protect(*args, **kwargs)
        memory.store.accept(runtime.config, "newer-input", "另一条输入",
                            session_id=runtime.claim.session_id)

    monkeypatch.setattr(memory.store, "protect_background", replace_owner)
    text.send(BOOK)
    wait_for(lambda: not text.snapshot()["busy"])
    assert text.snapshot()["attempt"]["status"] == "failed"
    assert len(model.primary) == 1
    assert not memory.store.segments(memory.companion, runtime.claim.input_id)
