"""Real SQLite/TextSession wiring with synthetic model actions, never an API.

These checks prove evidence identity/qualification and bounded orchestration;
the supplied model opinions do not prove semantic entailment or naturalness.
"""

import json
from copy import deepcopy
from dataclasses import replace

import pytest
from jsonschema import Draft202012Validator
from test_default_text_delivery import wait_for
from test_rebuilt_memory import apply, mutation
from unit_review_recording import bound_unit_packets as bound_unit_packets
from unit_review_recording import recorded_review_packet
from virtual_companion.boundary.contracts import Scope, canonical
from virtual_companion.boundary.history_readset import history_readset, validate_readset
from virtual_companion.boundary.memory import FactKey
from virtual_companion.cognition.background_worker import BackgroundScheduler
from virtual_companion.cognition.semantic_retrieval import SemanticMemoryRetriever
from virtual_companion.text.__main__ import diagnostic_options, request_contract_sha256
from virtual_companion.text.application import TextSession
from virtual_companion.text.default_runtime import DefaultTextRuntime
from virtual_companion.text.evidence_recall import VERSION
from virtual_companion.text.fake import synthetic_unit_opinion
from virtual_companion.text.memory_runtime import DEFAULT_COMPANION_ID, TextMemoryRuntime
from virtual_companion.text.sqlite_store import SQLiteConversationStore


class RecordingModel:
    supports_stage_metadata = True

    def __init__(self, actions, *, reject_once=False):
        self.actions, self.calls = list(actions), []
        self.reject_once = reject_once
        self.before_generation = None

    def prepare_stage_request(self, messages, *, stage, schema):
        owner = self
        messages, schema = deepcopy(messages), deepcopy(schema)

        class Prepared:
            def check_context(self):
                pass

            def stream(self, cancelled):
                owner.calls.append((stage, messages, schema))
                if stage != "stream_unit_review":
                    if owner.before_generation is not None:
                        owner.before_generation(len(owner.generations))
                    action = owner.actions.pop(0)
                    assert Draft202012Validator(schema).is_valid(action), action
                    yield canonical(action)
                    return
                result = synthetic_unit_opinion(json.loads(messages[-1]["content"]))
                if owner.reject_once:
                    owner.reject_once = False
                    result.update(decision="reject", constraints_satisfied=False)
                yield canonical(result)

        return Prepared()

    @property
    def generations(self):
        return [call for call in self.calls if call[0] != "stream_unit_review"]

    def cancel(self):
        pass


ANSWER = {"action": "answer", "text": "这是合成答案。"}


def recall(query):
    return {"action": "recall", "query": query}


@pytest.fixture
def factory(tmp_path):
    opened = []

    def create(actions, *, enabled=True, large=False, retriever=None, reject_once=False):
        model = RecordingModel(actions, reject_once=reject_once)
        memory = TextMemoryRuntime(tmp_path / f"memory-{len(opened)}.db",
                                   DEFAULT_COMPANION_ID, model=model, fake=True)
        mirror = SQLiteConversationStore(tmp_path / f"chat-{len(opened)}",
                                         companion_id=memory.companion)
        contents = [f"{['alpha', 'beta', 'gamma'][i // 8]} #{i}: 不喝冰水。"
                    + ("完整原句。" * (850 if i >= 8 else 10) if large else "")
                    for i in range(24)]
        source = memory.store.accept(memory.config, "manual-seed",
                                     contents[0] if large else "\n".join(contents))
        memory.store.cancel(memory.companion, source.input_id)
        memory.store.owner(memory.companion) or memory.store.claim_owner(
            memory.companion, "manual-seed", expected_epoch=0)
        for i, content in enumerate(contents):
            row_source = source
            if large and i:
                row_source = memory.store.accept(memory.config, f"manual-seed-{i}", content)
                memory.store.cancel(memory.companion, row_source.input_id)
            apply(memory.store, replace(mutation(
                f"seed-{i}", kind="assert_fact", record=f"fact-{i:02}", content=content,
                fact_key=FactKey("user", f"fact-{i:02}", "profile")),
                scope=Scope(memory.companion, source.session_id),
                source_kind="manual", source_id=row_source.input_id))
        runtime = DefaultTextRuntime(memory, mirror, model, binding={"test": "synthetic"},
            delivery_timeout=1, semantic_retriever=retriever,
            **({"readonly_recall": VERSION} if enabled else {}))
        text = TextSession(model, mirror, {}, prepare_turn=runtime.prepare_turn,
                           before_start=runtime.before_start)
        opened.append((runtime, text, memory, mirror))
        rows = memory.store.committed_recall(memory.companion, now_us=runtime.clock(),
            candidates=tuple((f"fact-{i:02}", 1) for i in range(24)), limit=24)
        return runtime, text, memory, mirror, model, rows, source

    yield create
    for runtime, text, memory, mirror in opened:
        runtime.stop()
        text.stop()
        if text.worker is not None:
            text.worker.join(2)
        memory.close()
        mirror.close()


def finish(runtime, text):
    wait_for(lambda: runtime.poll()["event"] or not text.snapshot()["busy"])
    event = runtime.poll()["event"]
    assert event is not None, text.snapshot()
    runtime.ack({"sequence": event["sequence"], "fingerprint": event["fingerprint"],
                 "host_receipt": "synthetic-readonly-dom"})
    wait_for(lambda: not text.snapshot()["busy"])
    assert text.snapshot()["attempt"]["status"] == "completed", text.snapshot()
    return event


def final_packet(model):
    return recorded_review_packet([call for call in model.calls
                                   if call[0] == "stream_unit_review"][-1][1])


def test_actual_default_turn_retains_all_24_originals_in_generation_review_and_readset(factory):
    runtime, text, memory, _, model, rows, seed = factory([recall("beta"), recall("gamma"), ANSWER])
    before = memory.store.committed_catalog_page(memory.companion, now_us=runtime.clock(),
                                                 limit=100)
    text.send("alpha")
    event = finish(runtime, text)
    packet = final_packet(model)
    context = json.loads(packet["context"])
    assert context["memories"] == rows
    assert context["evidence_access"]["coverage"] == "partial"
    assert context["evidence_access"]["queries"] == ["beta", "gamma"]
    assert context["evidence_access"]["calls"] == 2
    assert context["evidence_access"]["stop_reason"] == "call_limit"
    assert len(packet["context"].encode()) <= 64000
    rendered = canonical(model.generations[-1][1])
    assert all(row["content"] in rendered for row in rows)
    assert "部分" in rendered and "readonly-recall.v1" in rendered
    assert event["readset"] == history_readset(runtime.companion, runtime.claim.input_id,
                                               runtime.claim.state_version, packet["context"])
    assert event["readset"]["protocol"] == "history-readset.v6"
    assert len(event["readset"]["facts"]) == 24
    assert memory.store.committed_catalog_page(memory.companion, now_us=runtime.clock(),
                                               limit=100) == before
    assert all(row["source_id"] == seed.input_id and row["version"] == 1 for row in rows)
    assert memory.store.inspect(memory.companion, seed.input_id)["text"] == "\n".join(
        row["content"] for row in rows)
    assert len(model.generations) == 3
    assert len(model.calls) == 4


def test_candidate_selector_is_bound_and_default_rejects_recall(factory):
    runtime, text, _, _, model, _, _ = factory([recall("beta")], enabled=False)
    selected, _, _, _, _, _, _ = factory([ANSWER])
    assert dict(runtime.config.resources)["model"] != dict(selected.config.resources)["model"]
    # The recording validates real prepared schema before returning its action.
    text.send("alpha")
    wait_for(lambda: not text.snapshot()["busy"])
    assert text.snapshot()["attempt"]["status"] == "failed"
    assert runtime.poll()["event"] is None and len(model.generations) == 1
    options = diagnostic_options({"VC_TEXT_DIAGNOSTIC_READONLY_RECALL": VERSION})
    assert options == {"readonly_recall": VERSION}
    assert request_contract_sha256("control", options) != request_contract_sha256("control", {})
    with pytest.raises(ValueError):
        diagnostic_options({"VC_TEXT_DIAGNOSTIC_READONLY_RECALL": "unknown"})


def test_duplicate_query_stops_without_replaying_or_dropping_first_hop(factory):
    runtime, text, _, _, model, rows, _ = factory([recall("beta"), recall("beta"), ANSWER])
    text.send("alpha")
    finish(runtime, text)
    data = json.loads(final_packet(model)["context"])
    assert data["memories"] == rows[:16]
    assert data["evidence_access"]["calls"] == 1
    assert data["evidence_access"]["stop_reason"] == "duplicate_query"


def test_semantic_repair_reuses_one_remaining_recall_and_same_originals(factory):
    runtime, text, _, _, model, rows, _ = factory(
        [recall("beta"), ANSWER, recall("gamma"), ANSWER], reject_once=True)
    text.send("alpha")
    finish(runtime, text)
    data = json.loads(final_packet(model)["context"])
    assert data["memories"] == rows
    assert data["evidence_access"]["calls"] == 2
    assert len(model.generations) == 4


def test_one_writable_memory_does_not_consume_or_reset_readonly_budget(factory):
    runtime, text, memory, _, model, rows, _ = factory(
        [{"action": "memory"}, recall("beta"), recall("gamma"), ANSWER])
    text.send("alpha")
    finish(runtime, text)
    data = json.loads(final_packet(model)["context"])
    assert data["memory_access"] == "completed"
    assert data["memories"] == rows
    assert data["evidence_access"]["calls"] == 2
    assert memory.store.current_source_processing(runtime.claim)["status"] == "completed"
    assert len(model.generations) == 4


def test_full_context_budget_stops_new_evidence_and_keeps_complete_initial_originals(factory):
    runtime, text, _, _, model, rows, _ = factory([recall("beta"), ANSWER], large=True)
    text.send("alpha")
    finish(runtime, text)
    packet = final_packet(model)
    data = json.loads(packet["context"])
    assert data["memories"][:8] == rows[:8]
    assert 8 < len(data["memories"]) < 16
    assert all(row in rows for row in data["memories"])
    assert len(packet["context"].encode()) <= 64000
    assert data["evidence_access"]["stop_reason"] == "content_limit"
    assert data["evidence_access"]["coverage"] == "partial"


@pytest.mark.parametrize("fault", ["cancel", "deadline"])
def test_normal_semantic_admission_checks_cancel_and_absolute_deadline(factory, fault):
    documents = []

    class Encoder:
        def query_embed(self, query):
            if fault == "cancel":
                runtime.cancelled.set()
            else:
                clock[0] = runtime.deadline
            yield [1.0]

        def embed(self, texts):
            documents.extend(texts)
            yield from ([1.0] for _ in texts)

    retriever = SemanticMemoryRetriever(Encoder(), model_id="synthetic", model_revision="1",
                                        synthetic_model_identity=True)
    runtime, text, _, _, model, _, _ = factory([ANSWER], enabled=False, retriever=retriever)
    clock = [runtime.clock()]
    runtime.clock = lambda: clock[0]
    runtime.admit.evidence_clock = runtime.clock
    text.send("alpha")
    wait_for(lambda: not text.snapshot()["busy"])
    assert text.snapshot()["attempt"]["status"] == ("cancelled" if fault == "cancel" else "failed")
    assert not model.calls and runtime.poll()["event"] is None
    assert not documents


def test_late_correction_cannot_leave_first_hop_original_in_final_approved_readset(factory):
    runtime, text, memory, _, model, _, seed = factory([recall("beta"), ANSWER])

    def correct(generation):
        if generation == 2:
            apply(memory.store, replace(mutation(
                "correct-first-hop", kind="correct", record="fact-00", content="alpha 已更正。",
                expected_record_version=1), scope=Scope(memory.companion, seed.session_id),
                source_kind="manual", source_id=seed.input_id))

    model.before_generation = correct
    text.send("alpha")
    wait_for(lambda: not text.snapshot()["busy"])
    assert text.snapshot()["attempt"]["status"] == "failed"
    assert runtime.poll()["event"] is None
    assert memory.store.segments(runtime.companion, runtime.claim.input_id) == []


def test_legacy_readset_limits_are_not_widened_by_explicit_candidate(factory):
    runtime, text, _, _, model, _, _ = factory([recall("beta"), recall("gamma"), ANSWER])
    text.send("alpha")
    event = finish(runtime, text)
    changed = deepcopy(event["readset"])
    changed["protocol"] = "history-readset.v1"
    changed.pop("evidence_access", None)
    with pytest.raises(ValueError):
        validate_readset(changed)


@pytest.mark.parametrize("repair", [False, True])
def test_memory_refresh_merges_a_real_new_saved_original_without_losing_read_hop(factory, repair):
    from test_default_cross_turn_memory import Scenario

    source = "alpha 我喜欢温水"
    runtime, text, memory, _, model, rows, _ = factory(
        [recall("beta"), {"action": "memory", "query": "温水"}, ANSWER]
        + ([recall("gamma"), ANSWER] if repair else []), reject_once=repair)
    scenario = Scenario(original=source, predicate="warm-water")
    memory.scheduler = BackgroundScheduler(memory.store, user_invoke=scenario.invoke,
        assistant_invoke=scenario.invoke, clock=runtime.clock, resolution_protocol=True,
        resolution_strategy="intent_first", memory_commands=True)
    text.send(source)
    finish(runtime, text)
    data = json.loads(final_packet(model)["context"])
    assert data["memories"][:16] == rows[:16]
    added = [row for row in data["memories"] if row["source_id"] == runtime.claim.input_id]
    assert len(added) == 1 and added[0]["content"] == source
    assert data["memory_access"] == "completed"
    assert data["evidence_access"]["calls"] == (2 if repair else 1)
    assert data["evidence_access"]["queries"] == (["beta", "gamma"] if repair else ["beta"])
    assert len(data["memories"]) <= 24
    assert sum(call[0] == "fact_extract" for call in scenario.worker_calls) == 1


def test_initial_metadata_reserves_two_maximum_queries_without_trimming_a_read_original(
    factory, monkeypatch,
):
    queries = ["beta " + "甲" * 251, "gamma " + "乙" * 250]
    runtime, text, memory, _, model, rows, seed = factory(
        [recall(queries[0]), recall(queries[1]), ANSWER])
    # Eight legal near-capacity initial facts; beta/gamma queries bring new rows.
    length = (62600 - len(canonical(rows[:8]).encode())
              + sum(len(row['content'].encode()) for row in rows[:8]) - 8 * len('alpha ')) // 24
    for i in range(8):
        apply(memory.store, replace(mutation(f"large-{i}", kind="correct",
            record=f"fact-{i:02}", content="alpha " + "原" * length,
            expected_record_version=1), scope=Scope(memory.companion, seed.session_id),
            source_kind="manual", source_id=seed.input_id))
    observed = []
    admit = type(runtime.admit).__call__

    def observe(adapter, *args, **kwargs):
        request = admit(adapter, *args, **kwargs)
        data = json.loads(request.context.payload)
        if adapter is runtime.admit and data.get('evidence_access', {}).get('calls') == 0:
            observed.append(data)
        return request

    monkeypatch.setattr(type(runtime.admit), '__call__', observe)
    text.send("alpha")
    finish(runtime, text)
    data = json.loads(final_packet(model)["context"])
    assert len(final_packet(model)["context"].encode()) <= 64000
    assert data["evidence_access"]["coverage"] == "partial"
    assert data["evidence_access"]["queries"] == queries
    assert 1 <= len(observed[0]['memories']) <= 8
    assert data['memories'][:len(observed[0]['memories'])] == observed[0]['memories']
    assert all(len(query) == 256 for query in queries)


def test_v6_access_rejects_forged_coverage_queries_limits_and_legacy_widening(factory):
    runtime, _, _, _, _, rows, _ = factory([ANSWER])
    access = {"protocol": VERSION, "scope": "current_committed_facts", "coverage": "partial",
              "queries": ["beta", "gamma"], "calls": 2, "stop_reason": "call_limit",
              "limits": {"calls": 2, "records": 24, "evidence_bytes": 64000}}
    data = {"memories": rows, "history": [], "exchanges": [], "memory_status": "current",
            "evidence_access": access}
    good = history_readset(runtime.companion, "synthetic-input", 0, canonical(data))
    assert validate_readset(good) == good
    for field, bad in [('protocol', 'unknown'), ('scope', 'all_facts'), ('coverage', 'complete'),
                       ('calls', 3), ('calls', True), ('queries', ['beta', 'beta']),
                       ('queries', ['beta']), ('stop_reason', None), ('stop_reason', 'unknown'),
                       ('limits', {'calls': 3, 'records': 24, 'evidence_bytes': 64000}),
                       ('limits', {'calls': 2, 'records': 25, 'evidence_bytes': 64000})]:
        changed = deepcopy(good)
        changed['evidence_access'][field] = bad
        with pytest.raises(ValueError):
            validate_readset(changed)
    changed = deepcopy(data)
    changed['evidence_access']['limits']['evidence_bytes'] = 1
    with pytest.raises(ValueError):
        history_readset(runtime.companion, 'synthetic-input', 0, canonical(changed))
