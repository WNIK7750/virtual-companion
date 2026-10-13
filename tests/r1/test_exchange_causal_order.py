"""Late background processing cannot let an old answer settle a future question."""

import json
from dataclasses import replace
from hashlib import sha256

import pytest
from test_effective_history import NOW, admission, emit
from test_exchange_capacity import recording, run
from test_pending_context_ranking import pending_catalog
from test_pending_exchanges import apply, user_command
from test_rebuilt_turns import A, config
from test_segment_store import reviewed
from virtual_companion.boundary.contracts import Scope
from virtual_companion.boundary.exchanges import ExchangeCommand, ExchangeEvidence
from virtual_companion.boundary.history_readset import history_readset
from virtual_companion.cognition.background_worker import AssistantExchangeWorker


@pytest.mark.parametrize("action,intent", [("resolve", "answer"), ("close", "end")])
def test_typed_ack_cannot_settle_user_obligation_even_with_supported_model_intent(
    tmp_path, action, intent,
):
    from virtual_companion.boundary.presentation import ReviewedPresentation, review_by_policy

    store = pending_catalog(tmp_path, 1)
    target = store.pending_exchanges(A, now_us=NOW)[0]
    item = store.accept(config(), "listening", "我还没说完。")
    run(store, item, recording())
    owner, request = admission(store, item)
    bound = reviewed(owner, ("嗯，我在听。",))
    candidate = replace(bound.candidate, response_type="ack")
    bound = ReviewedPresentation(candidate, bound.policy, review_by_policy(
        candidate, bound.policy, "ack-review", policy_revision="synthetic-unit-policy-v1"))
    store.approve_segment(owner, bound, now_us=NOW + 11,
        readset=history_readset(A, item.input_id, request.context.state_version,
                               request.context.payload))
    event = store.dispatch_segment(owner, 1, now_us=NOW + 12)
    store.confirm_segment(A, item.input_id, 1, fingerprint=event["fingerprint"],
                          host_receipt="visible-ack", channel="text_visible")
    store.finish_stream(owner, outcome="completed", now_us=NOW + 13)
    source = "segment-" + event["fingerprint"]
    text = "嗯，我在听。"
    forged = ExchangeCommand(Scope(A, item.session_id), "ack-transition", target["exchange_id"],
        action, target["direction"], target["content"], ExchangeEvidence("assistant", source,
            text, len(text), sha256(text.encode()).hexdigest()), NOW + 20,
        target["revision"], target["expires_us"])
    before = store.snapshot(A).write_version
    with pytest.raises(ValueError, match="Acknowledgement"):
        apply(store, forged, now_us=NOW + 20)
    assert store.snapshot(A).write_version == before

    recorded = recording(target=target, intent=intent)
    wire_types = []

    def misclassified(stage, messages, schema, cancelled):
        if stage == "exchange_intent":
            data = json.loads(messages[-1]["content"])
            wire_types.append(data["sources"][-1]["response_type"])
            return {"coverage": "complete", "acts": [{"segmentId": data["segments"][0]["id"],
                "evidenceQuote": data["sources"][-1]["text"], "intent": intent,
                "targetId": target["exchange_id"]}]}
        return recorded(stage, messages, schema, cancelled)

    worker = AssistantExchangeWorker(store, misclassified, clock=lambda: NOW + 20,
        resolution_protocol=True, resolution_strategy="intent_first")
    with pytest.raises(ValueError, match="Acknowledgement"):
        worker.run_one(A, "assistant")
    assert wire_types == ["ack"]
    assert store.exchange_history(A, target["exchange_id"])[-1]["status"] == "pending"
    with store.transaction(write=False) as db:
        assert db.execute("SELECT receipt FROM background_work WHERE companion=? AND job=("
            "SELECT job FROM background_sources WHERE companion=? AND source=?)",
            (A, A, source)).fetchone()[0] is None


@pytest.mark.parametrize("future_question", [True, False])
def test_source_order_filters_future_targets_and_guards_commit(tmp_path, future_question):
    store = pending_catalog(tmp_path, 0)

    def open_question():
        command = replace(user_command(store, text="《城南旧事》的作者是谁？"),
                          occurred_us=NOW, expires_us=None)
        apply(store, command, now_us=NOW)
        store.cancel(A, command.evidence.source_id)
        return command

    if not future_question:
        question = open_question()
    seed = store.accept(config(), "seed", "《城南旧事》的作者是林海音。")
    run(store, seed, recording())
    emit(store, seed, "《城南旧事》的作者是林海音。")
    if future_question:
        question = open_question()
    target = store.exchange_history(A, question.exchange_id)[-1]
    source = next(s for s in store.background_sources(A) if s["role"] == "assistant")
    if future_question:
        text = store.background_source_text(A, source["job"])
        forged = replace(question, command_id="older-answer", action="resolve",
                         scope=Scope(A, seed.session_id),
                         expected_revision=1, occurred_us=NOW + 20,
                         evidence=ExchangeEvidence("assistant", source["source"], text,
                             len(text), sha256(text.encode()).hexdigest()))
        # Even independently supported commands cannot bypass causal ordering.
        with pytest.raises(ValueError, match="originates after"):
            apply(store, forged, now_us=NOW + 20)

    worker = AssistantExchangeWorker(store,
        recording(target=None if future_question else target,
                  intent=None if future_question else "answer"),
        clock=lambda: NOW + 20, resolution_protocol=True, resolution_strategy="intent_first")
    receipt = json.loads(worker.run_one(A, "assistant"))
    refs = receipt["resolution"]["target_refs"]
    assert bool(refs) is not future_question
    assert store.exchange_history(A, question.exchange_id)[-1]["status"] == (
        "pending" if future_question else "resolved")


def test_earlier_assistant_unit_cannot_resolve_later_unit_in_same_turn(tmp_path):
    store = pending_catalog(tmp_path, 0)
    item = store.accept(config(), "multi-unit", "说说这本书。")
    run(store, item, recording())
    owner, request = admission(store, item)
    context = request.context
    units = ("作者是林海音。", "接下来我会介绍故事背景。")
    sources = []
    for sequence in (1, 2):
        store.approve_segment(owner, reviewed(owner, units[:sequence]), now_us=NOW + 11,
            readset=history_readset(A, item.input_id, context.state_version, context.payload))
        event = store.dispatch_segment(owner, sequence, now_us=NOW + 12)
        store.confirm_segment(A, item.input_id, sequence, fingerprint=event["fingerprint"],
                              host_receipt=f"unit-{sequence}", channel="text_visible")
        sources.append("segment-" + event["fingerprint"])
    store.finish_stream(owner, outcome="completed", now_us=NOW + 13)
    scope = Scope(A, item.session_id)
    with store.transaction(write=False) as db:
        earlier, later = [store._exchange_source(db, scope, "assistant", source)
                          for source in sources]
        assert earlier["ordinal"] == later["ordinal"]
        assert (earlier["sequence"], later["sequence"]) == (1, 2)

    def evidence(index):
        text = units[index]
        return ExchangeEvidence("assistant", sources[index], text, len(text),
                                sha256(text.encode()).hexdigest())

    promise = ExchangeCommand(scope, "promise", "promise", "open", "assistant_response",
                              units[1], evidence(1), NOW + 20, None, None)
    apply(store, promise, now_us=NOW + 20)
    assert store.ranked_pending_exchanges(A, now_us=NOW + 21, score=lambda _: 1,
        before_source=(scope, "assistant", sources[0])) == []
    premature = replace(promise, command_id="premature", action="resolve", expected_revision=1,
                        occurred_us=NOW + 21, evidence=evidence(0))
    with pytest.raises(ValueError, match="originates after"):
        apply(store, premature, now_us=NOW + 21)
    assert store.exchange_history(A, "promise")[-1]["status"] == "pending"
