"""Default entry exercises real source fences with synthetic model replies only."""

import json
from copy import deepcopy

import pytest
from test_default_text_delivery import Model, setup, wait_for
from test_memory_command_worker import command, seed
from unit_review_recording import bound_unit_packets as bound_unit_packets
from unit_review_recording import recorded_review_packet
from virtual_companion.boundary.history_readset import evidence_hash, history_readset
from virtual_companion.cognition.background_worker import UserFactWorker
from virtual_companion.cognition.stream_unit_review import semantic_review_view
from virtual_companion.text.fake import synthetic_unit_opinion

RECALL_QUERY = "忙完一整天了，依据以前的偏好，有什么适合听着放松的？"
# Independent extraction checks the current source; this question yields no facts.
QUERY_STAGES = ["memory_command", "exchange_intent", "fact_extract",
                "exchange_extract", "exchange_support"]


class DemandModel(Model):
    def __init__(self, *, first_action="memory"):
        super().__init__()
        self.first_action = first_action
        self.primary = []
        self.on_final = None
        self.fail_final_once = False

    def stream(self, messages, cancelled):
        if "你只审查候选文字" in messages[0]["content"]:
            yield from super().stream(messages, cancelled)
            return
        self.calls.append(deepcopy(messages))
        self.primary.append(deepcopy(messages))
        final = "本次只能返回answer" in messages[0]["content"]
        if final and self.on_final is not None:
            self.on_final(messages)
        if final and self.fail_final_once:
            self.fail_final_once = False
            raise OSError("synthetic final transport failure")
        yield json.dumps({"action": "answer", "text": "这是合成答案。"}
                         if final or self.first_action == "answer"
                         else {"action": "memory"}, ensure_ascii=False)


class TypedTurnModel(Model):
    def __init__(self, response_type):
        super().__init__()
        self.response_type = response_type

    @property
    def reply(self):
        return "先留半小时读书，再整理任务。" if self.response_type == "answer" else "嗯，我在听。"

    def stream(self, messages, cancelled):
        if "你只审查候选文字" in messages[0]["content"]:
            yield from super().stream(messages, cancelled)
            return
        self.calls.append(deepcopy(messages))
        yield json.dumps({"action": self.response_type, **({"text": self.reply}
            if self.response_type != "listen" else {})}, ensure_ascii=False)


def demand_setup(tmp_path, mode="query", *, source=None, model=None, seed_memory=True):
    model = model or DemandModel()
    runtime, text, memory, mirror, _ = setup(tmp_path, model)
    if seed_memory:
        seed(memory.store)
    sources = {
        "query": "我喜欢什么音乐？",
        "forget": "请忘记我喜欢安静音乐这件事。",
        "correct": "把我喜欢安静的音乐改成我喜欢摇滚。",
        "unclear": "请忘记那一件事。",
        "error": "我喜欢什么音乐？",
    }
    source = sources[mode] if source is None else source
    stages = []

    def invoke(stage, messages, schema, cancelled):
        stages.append(stage)
        assert runtime.claim.input_id in {
            item["source"] for item in memory.store.background_sources(memory.companion)
        }
        # A locked foreground model gate here would deadlock the real invoker.
        assert memory.generation_gate._active is None
        if mode == "error":
            raise OSError("synthetic source interpretation failure")
        if stage == "memory_command":
            if mode in {"forget", "correct", "unclear"}:
                result = command(source, targets=() if mode == "unclear" else ("music",))
                if mode == "correct":
                    replacement = "我喜欢摇滚"
                    start = source.index(replacement)
                    result.update(action="correct", replacementQuote=replacement,
                                  replacementStart=start, replacementEnd=start + len(replacement))
                return result
            return {
                "action": "none", "requestedScope": "none", "targetIds": [],
                "evidenceQuote": "", "quoteStart": None, "quoteEnd": None,
                "replacementQuote": "", "replacementStart": None, "replacementEnd": None,
                "coverage": "not_command", "unresolved": [],
            }
        if stage == "memory_command_support":
            return {"meaning": "explicit_single_correction" if mode == "correct"
                    else "explicit_single_forget", "verdict": "supported"}
        if stage == "fact_extract":
            return {"operations": []}
        if stage == "exchange_intent":
            return {"coverage": "complete", "acts": [{
                "segmentId": 0,
                "evidenceQuote": source, "intent": "question", "targetId": None,
            }]}
        if stage == "exchange_extract":
            return {"operations": [{
                "operation": "add", "type": "response_obligation",
                 "evidenceQuote": source,
                "supersedesAtomId": None, "expiresAt": None,
            }]}
        if stage == "exchange_support":
            return {"intent": "question", "verdict": "supported"}
        raise AssertionError(stage)

    memory.scheduler.user = UserFactWorker(
        memory.store, invoke, clock=runtime.clock, memory_commands=True,
        resolution_protocol=True, resolution_strategy="intent_first",
    )
    return runtime, text, memory, mirror, model, source, stages


def ack_and_finish(runtime, text, *, receipt="synthetic-demand-dom-receipt"):
    wait_for(lambda: runtime.poll()["event"] or not text.snapshot()["busy"])
    event = runtime.poll()["event"]
    assert event is not None, text.snapshot()
    assert text.snapshot()["partial"] == ""
    runtime.ack({"sequence": event["sequence"], "fingerprint": event["fingerprint"],
                 "host_receipt": receipt})
    wait_for(lambda: not text.snapshot()["busy"])
    assert text.snapshot()["attempt"]["status"] == "completed", text.snapshot()
    return event


@pytest.mark.parametrize("response_type", ["listen", "ack", "answer"])
def test_real_typed_turn_keeps_delivery_receipts_and_pending_semantics(tmp_path, response_type):
    from virtual_companion.boundary.contracts import Scope
    from virtual_companion.cognition.background_worker import AssistantExchangeWorker

    runtime, text, memory, mirror, model, source, _ = demand_setup(
        tmp_path, source="先帮我安排一下今天的学习。", seed_memory=False,
        model=TypedTurnModel(response_type))
    try:
        text.send(source)
        if response_type == "listen":
            wait_for(lambda: not text.snapshot()["busy"])
            assert runtime.poll()["event"] is None
            assert text.snapshot()["attempt"]["status"] == "listened", text.snapshot()
            assert not text.snapshot()["error"] and text.snapshot()["partial"] == ""
            assert [m["role"] for m in mirror.history()["messages"]] == ["user"]
            assert len(model.calls) == 1
        else:
            event = ack_and_finish(runtime, text)
            assert event["presentation"]["candidate"].get(
                "response_type", "answer") == response_type
            message = mirror.history()["messages"][-1]
            assert message.get("response_type", "answer") == response_type
            if response_type == "answer":
                assert set(message) == {"id", "turn_id", "role", "content",
                                        "timeline_order", "bubble_ranges", "bubble_orders"}
                assert mirror.history_page()["messages"][-1] == message
            assert len(model.calls) == 2
        item = runtime.claim.input_id
        terminal = memory.store.stream_state(memory.companion, item)
        assert terminal["outcome"] == ("listened" if response_type == "listen" else "completed")
        assert terminal["approved_units"] == terminal["confirmed_units"] == int(
            response_type != "listen")
        assert memory.scheduler.user.run_one(memory.companion, "synthetic-user", source=item)
        target = memory.store.pending_exchanges(memory.companion, now_us=runtime.clock())[0]
        assistants = [s for s in memory.store.background_sources(memory.companion)
                      if s["role"] == "assistant"]
        assert len(assistants) == int(response_type != "listen")
        if assistants:
            turns = memory.store.exchange_sources(Scope(memory.companion, mirror.session_id),
                references=(("assistant", assistants[0]["source"]),))
            assert turns[0].response_type == response_type

            def invoke(stage, messages, schema, cancelled):
                data = json.loads(messages[-1]["content"])
                if stage == "exchange_intent":
                    assert data["sources"][-1].get("response_type", "answer") == response_type
                    return {"coverage": "complete", "acts": [{"segmentId": 0,
                        "evidenceQuote": model.reply,
                        "intent": "answer" if response_type == "answer" else "statement",
                        "targetId": target["exchange_id"] if response_type == "answer" else None}]}
                if stage == "exchange_extract":
                    return {"operations": ([{"operation": "resolve", "type": None,
                        "evidenceQuote": model.reply, "supersedesAtomId": target["exchange_id"],
                        "expiresAt": None}] if response_type == "answer" else [])}
                assert stage == "exchange_support"
                return {"intent": "answer", "verdict": "supported"}

            worker = AssistantExchangeWorker(memory.store, invoke, clock=runtime.clock,
                resolution_protocol=True, resolution_strategy="intent_first")
            receipt = json.loads(worker.run_one(memory.companion, "synthetic-assistant"))
            assert receipt["operations"] == int(response_type == "answer")
            assert receipt.get("response_type", "answer") == response_type
            with memory.store.transaction(write=False) as db:
                row = db.execute("SELECT status FROM background_sources WHERE companion=? "
                    "AND source=?", (memory.companion, assistants[0]["source"])).fetchone()
                assert row[0] == "completed"
        assert memory.store.exchange_history(memory.companion, target["exchange_id"])[-1][
            "status"] == ("resolved" if response_type == "answer" else "pending")
    finally:
        memory.close()
        mirror.close()


def test_ack_type_survives_reopen_and_the_actual_next_demand_and_review_wire(tmp_path):
    from hashlib import sha256

    from virtual_companion.boundary.contracts import Scope, canonical
    from virtual_companion.boundary.evidence import EvidenceTurn, evidence_digest, evidence_payload
    from virtual_companion.text.application import TextSession
    from virtual_companion.text.default_runtime import DefaultTextRuntime
    from virtual_companion.text.sqlite_store import SQLiteConversationStore

    runtime, text, memory, mirror, model = setup(tmp_path, TypedTurnModel("ack"))
    try:
        text.send("我还在说。")
        event = ack_and_finish(runtime, text)
        assert memory.scheduler.user.run_one(memory.companion, "synthetic-user",
                                             source=runtime.claim.input_id)
        old = mirror.history()["messages"][-1]
        assert old["response_type"] == "ack"
        assert mirror.connection.execute("PRAGMA user_version").fetchone()[0] == 1
        assert mirror.connection.execute("SELECT value FROM metadata WHERE key=?",
            ("response-type:" + old["id"],)).fetchone()[0] == "ack"
        mirror.close()
        mirror = SQLiteConversationStore(tmp_path / "chat", companion_id=memory.companion)
        assert mirror.history()["messages"][-1] == old
        runtime = DefaultTextRuntime(memory, mirror, model, binding={"test": "synthetic"},
                                     delivery_timeout=0.8)
        text = TextSession(model, mirror, {}, prepare_turn=runtime.prepare_turn,
                           before_start=runtime.before_start)
        model.response_type = "answer"
        text.send("继续说。")
        next_event = ack_and_finish(runtime, text, receipt="synthetic-reopened-dom-receipt")
        assert next_event["readset"]["history"][-1]["response_type"] == "ack"
        primary = model.calls[-2]
        reviewer = model.calls[-1]
        from virtual_companion.cognition.short_dialogue import prose_evidence

        semantic = json.loads(reviewer[-1]["content"])
        sealed = recorded_review_packet(reviewer)
        assert semantic_review_view(sealed) == semantic
        review_context = sealed["context"]
        assert primary[1]["content"] == prose_evidence(review_context)
        assert (
            "2. 助手接话（ack，仅表示收到/在听，待答未解决）：嗯，我在听。"
            in primary[1]["content"]
        )
        assert json.loads(review_context)["history"][-1] == {
            "id": "segment-" + event["fingerprint"], "role": "assistant",
            "text": "嗯，我在听。", "response_type": "ack"}
        # Removing the type while keeping the ACK identity/text cannot turn the
        # referenced acknowledgement into answer evidence at the commit fence.
        from virtual_companion.boundary.contracts import VersionConflict

        stripped = deepcopy(next_event["readset"])
        del stripped["history"][-1]["response_type"]
        with memory.store.transaction(write=False) as db:
            with pytest.raises(VersionConflict, match="history changed"):
                memory.store._validate_segment_readset(db, runtime.claim, stripped,
                                                       now_us=runtime.clock())
        # Ordinary assistant evidence keeps the legacy exact-wire representation.
        turn = EvidenceTurn(Scope(memory.companion, mirror.session_id), "old-answer", "assistant",
                            "旧答案。", 4)
        legacy = [{"id": "old-answer", "role": "assistant", "text": "旧答案。"}]
        assert evidence_payload((turn,)) == legacy
        assert evidence_digest((turn,)) == sha256(canonical(legacy).encode()).hexdigest()
    finally:
        memory.close()
        mirror.close()


@pytest.mark.parametrize("action", ["answer", "memory"])
def test_default_entry_orchestrates_only_bound_host_callbacks(tmp_path, monkeypatch, action):
    from virtual_companion.text.turn_orchestrator import TurnOrchestrator

    runtime, text, memory, mirror, model, source, _ = demand_setup(
        tmp_path, source="普通合成输入", model=DemandModel(first_action=action),
        seed_memory=action == "memory",
    )
    calls, prepared_turns = [], []
    original_turn = runtime.prepare_turn
    originals = {name: getattr(runtime, name)
                 for name in ("prepare", "resolve", "refresh", "deliver")}

    def prepare(claim, *, final):
        calls.append(("prepare", final))
        return originals["prepare"](claim, final=final)

    def resolve(claim):
        calls.append(("resolve", claim.input_id))
        return originals["resolve"](claim)

    def refresh(claim):
        refreshed = originals["refresh"](claim)
        calls.append(("refresh", refreshed.state_version))
        return refreshed

    def deliver(stage, answer):
        calls.append(("deliver", stage.request.context.payload))
        return originals["deliver"](stage, answer)

    def capture_turn(*args):
        prepared = original_turn(*args)
        prepared_turns.append(prepared)
        return prepared

    for name, callback in (("prepare", prepare), ("resolve", resolve), ("refresh", refresh),
                           ("deliver", deliver)):
        monkeypatch.setattr(runtime, name, callback)
    text.prepare_turn = capture_turn
    try:
        text.send(source)
        event = ack_and_finish(runtime, text)
        assert len(prepared_turns) == 1 and isinstance(prepared_turns[0], TurnOrchestrator)
        assert [name for name, _ in calls] == (["prepare", "deliver"] if action == "answer"
            else ["prepare", "resolve", "refresh", "prepare", "deliver"])
        assert [value for name, value in calls if name == "prepare"] == (
            [False] if action == "answer" else [False, True])
        assert len(model.primary) == (1 if action == "answer" else 2)
        context = json.loads(calls[-1][1])
        if action == "memory":
            assert context["memory_status"] == "current" and context["memories"]
        assert memory.store.stream_state(memory.companion, event["basis"]["input_id"])[
            "outcome"] == "completed"
        assert mirror.history()["messages"][-1]["content"] == event["display_text"]
    finally:
        memory.close()
        mirror.close()


def test_default_memory_demand_refreshes_effective_context_then_approves_and_acks(tmp_path):
    runtime, text, memory, mirror, model, source, stages = demand_setup(tmp_path)
    original_facts = memory.store.history(memory.companion, "music")
    observed_basis = []
    original = runtime.prepare

    def prepare(claim, *, final):
        observed_basis.append((final, runtime.claim.state_version))
        return original(claim, final=final)

    runtime.prepare = prepare
    text.send(source)
    event = ack_and_finish(runtime, text)
    assert stages == QUERY_STAGES
    assert memory.store.history(memory.companion, "music") == original_facts
    assert len(model.primary) == 2 and len(model.calls) == 3
    first_system = model.primary[0][0]["content"]
    first_evidence = model.primary[0][1]["content"]
    assert "desktop-demand.v4" in first_system
    assert "我喜欢安静的音乐" in first_evidence
    assert "尚未额外补查或委托" in first_evidence
    assert "相关长期记忆：无" not in first_evidence
    assert "本次只能返回answer" in model.primary[1][0]["content"]
    assert "相关长期记忆：暂未提供" not in model.primary[1][1]["content"]
    assert "我喜欢安静的音乐" in json.dumps(model.primary[0], ensure_ascii=False)
    assert "我喜欢安静的音乐" in json.dumps(model.primary[1], ensure_ascii=False)
    assert observed_basis[1][1] > observed_basis[0][1]
    assert not memory.store.memory_barrier(memory.companion)
    assert mirror.history()["messages"][-1]["content"] == "这是合成答案。"
    assert memory.store.stream_state(memory.companion, event["basis"]["input_id"])[
        "confirmed_text"
    ] == "这是合成答案。"


def test_explicit_recall_after_reopen_buffers_initial_answer_until_source_resolution(tmp_path):
    _, _, memory, mirror, _ = setup(tmp_path)
    seed(memory.store)
    memory.close()
    mirror.close()
    runtime, text, memory, mirror, model, source, stages = demand_setup(
        tmp_path, source=RECALL_QUERY, model=DemandModel(first_action="memory"),
        seed_memory=False,
    )
    saved, = memory.store.recall(memory.companion, now_us=runtime.clock())
    assert saved["content"] == "我喜欢安静的音乐"
    assert saved["status"] == "active" and saved["occurred_at"].startswith("2026-09-21")
    assert saved["event_time"] is None
    original_facts = memory.store.history(memory.companion, "music")
    admitted, committed_catalogs = [], []
    prepare = runtime.prepare

    def record(claim, *, final):
        stage = prepare(claim, final=final)
        admitted.append((final, stage.request))
        committed_catalogs.append(memory.store.committed_recall(
            memory.companion, now_us=runtime.clock()))
        return stage

    runtime.prepare = record
    text.send(source)
    event = ack_and_finish(runtime, text)
    assert stages == QUERY_STAGES
    assert committed_catalogs == [[saved], [saved]]
    assert memory.store.history(memory.companion, "music") == original_facts
    assert [final for final, _ in admitted] == [False, True]
    initial, final = (request for _, request in admitted)
    assert json.loads(initial.context.payload)["memory_status"] != "current"
    assert json.loads(initial.context.payload)["memories"] == []
    assert json.loads(final.context.payload)["memory_status"] == "current"
    assert json.loads(final.context.payload)["memories"] == [saved]
    assert len(model.primary) == 2 and len(model.calls) == 3
    assert "本次只能返回answer" not in model.primary[0][0]["content"]
    assert "我喜欢安静的音乐" not in json.dumps(model.primary[0], ensure_ascii=False)
    assert "本次只能返回answer" in model.primary[1][0]["content"]
    assert "我喜欢安静的音乐" in json.dumps(model.primary[1], ensure_ascii=False)
    assert event["readset"] == history_readset(
        final.context.companion, final.context.input_id,
        final.context.state_version, final.context.payload,
    )
    assert event["readset"]["facts"] == [{
        "record_id": saved["record_id"], "version": saved["version"],
        "source_id": saved["source_id"], "sha256": evidence_hash(saved),
    }]
    assert event["readset"]["state_version"] > initial.context.state_version
    assert not memory.store.memory_barrier(memory.companion)
    assert memory.store.stream_state(memory.companion, runtime.claim.input_id)[
        "confirmed_text"
    ] == "这是合成答案。"


@pytest.mark.parametrize("source", [
    "忙完一整天了，有什么适合听着放松的？",
    "以前的耳机坏了，今天解释一下耳机怎么发声。",
    "我以前喜欢安静的音乐，今天想试试白噪声。",
    "不用依据以前的偏好，按当前要求推荐一段白噪声。",
    "不要忘记我喜欢安静的音乐。",
    "我会忘掉那件事。",
    "忘记那件事了吗？",
])
def test_current_direct_reply_does_not_force_old_facts_or_source_work(tmp_path, source):
    runtime, text, memory, mirror, model, _, stages = demand_setup(
        tmp_path, source=source, model=DemandModel(first_action="answer"),
    )
    original_facts = memory.store.history(memory.companion, "music")
    text.send(source)
    event = ack_and_finish(runtime, text)
    assert stages == []
    assert len(model.primary) == 1 and len(model.calls) == 2
    from virtual_companion.cognition.short_dialogue import OPERATION_TRUTH_SYSTEM
    from virtual_companion.text.default_runtime import ACCESS_INITIAL

    generation_system = model.primary[0][0]["content"]
    review_system = model.calls[-1][0]["content"]
    assert "尚未请求memory补查" in generation_system
    assert ACCESS_INITIAL in generation_system and ACCESS_INITIAL in review_system
    assert "已准入的有效背景足够时可直接answer" in ACCESS_INITIAL
    assert "不宣称未实际执行的额外查询或持久保存" in ACCESS_INITIAL
    assert OPERATION_TRUTH_SYSTEM in generation_system and OPERATION_TRUTH_SYSTEM in review_system
    assert "尚无持久保存、更正或遗忘成功证据" in generation_system
    assert "尚无持久保存、更正或遗忘成功证据" in review_system
    assert "宿主本次访问状态与实际操作证据共同约束操作陈述" not in review_system
    packet = recorded_review_packet(model.calls[-1])
    context = json.loads(packet["context"])
    assert context["source_processing"]["status"] == "queued"
    if source in {"我以前喜欢安静的音乐，今天想试试白噪声。", "不要忘记我喜欢安静的音乐。"}:
        assert context["memories"]
    else:
        assert context["memories"] == []
    assert event["readset"] == history_readset(
        runtime.claim.companion_id, runtime.claim.input_id,
        runtime.claim.state_version, packet["context"])
    assert all(item["content"] == "我喜欢安静的音乐" and item["version"] == 1
               for item in context["memories"])
    assert context["memory_access"] == "not_accessed"
    state = memory.store.stream_state(memory.companion, event["basis"]["input_id"])
    assert state["approved_units"] == state["confirmed_units"] == 1
    assert state["outcome"] == "completed"
    assert memory.store.history(memory.companion, "music") == original_facts
    assert memory.store.memory_barrier(memory.companion)


def test_direct_default_reply_admits_semantic_fact_through_readonly_context(tmp_path):
    from test_rebuilt_memory import apply, mutation
    from virtual_companion.boundary.memory import FactKey
    from virtual_companion.cognition.semantic_retrieval import SemanticMemoryRetriever
    from virtual_companion.cognition.short_context import ShortContextAdmission

    query = "我偏爱的动物是哪种？"
    runtime, text, memory, mirror, model, _, stages = demand_setup(
        tmp_path, source=query, model=DemandModel(first_action="answer"), seed_memory=False,
    )
    try:
        seed(memory.store)
        facts = [
            ("favorite-animal", "我最喜欢猫"),
            ("animal-dog", "我喜欢狗"),
            ("animal-rabbit", "我喜欢兔子"),
            ("animal-bird", "我喜欢鸟"),
            ("animal-hamster", "我喜欢仓鼠"),
            ("animal-fox", "我喜欢狐狸"),
            ("animal-turtle", "我喜欢乌龟"),
            ("animal-fish", "我喜欢鱼"),
            ("animal-horse", "我喜欢马"),
        ]
        for index, (record, content) in enumerate(facts):
            apply(memory.store, mutation(
                f"semantic-seed-{index}", kind="assert_fact", record=record,
                content=content,
                fact_key=FactKey("user", f"animal-preference-{index}", "profile"),
            ))

        class FixedEmbedder:
            def __init__(self):
                self.queries = []
                self.documents = []

            def query_embed(self, value):
                self.queries.append(value)
                return iter(([1.0, 0.0],))

            def embed(self, values):
                values = list(values)
                self.documents.append(tuple(values))
                return iter([1.0, 0.0] if value in {"我最喜欢猫", "我最喜欢狗"}
                            else [0.0, 1.0]
                            for value in values)

        embedder = FixedEmbedder()
        initial = runtime.admit
        runtime.admit = ShortContextAdmission(
            runtime.store, initial.admit, evidence_clock=initial.evidence_clock,
            semantic_retriever=SemanticMemoryRetriever(
                embedder, model_id="fake-contract-test", model_revision="fixed",
                preprocessing_revision="raw-content.v1", page_size=3,
                candidate_overfetch=8, synthetic_model_identity=True,
            ),
        )

        text.send(query)
        event = ack_and_finish(runtime, text)
        packet = recorded_review_packet(model.calls[-1])
        context = json.loads(packet["context"])
        selected = [item for item in context["memories"]
                    if item["record_id"] == "favorite-animal"]

        assert len(selected) == 1
        assert selected[0]["version"] == 1
        assert selected[0]["content"] == "我最喜欢猫"
        assert context["memory_access"] == "not_accessed"
        assert stages == []
        assert len(model.primary) == 1 and len(model.calls) == 2
        assert embedder.queries == [query]
        assert sum(len(batch) for batch in embedder.documents) == 10
        assert event["readset"] == history_readset(
            runtime.claim.companion_id, runtime.claim.input_id,
            runtime.claim.state_version, packet["context"],
        )
        assert (memory.store.history(memory.companion, "favorite-animal")[0]["content"]
                == "我最喜欢猫")

        retriever = runtime.admit.semantic_retriever
        retriever.recall(memory.store, memory.companion, now_us=runtime.clock(),
                         query=query, limit=8)
        assert sum(len(batch) for batch in embedder.documents) == 10
        apply(memory.store, mutation(
            "correct-favorite-animal", kind="correct", record="favorite-animal",
            content="我最喜欢狗", expected_record_version=1,
        ))
        revised = retriever.recall(memory.store, memory.companion,
                                   now_us=runtime.clock(), query=query, limit=8)
        revised_target = next(item for item in revised
                              if item["record_id"] == "favorite-animal")
        assert (revised_target["version"], revised_target["content"]) == (2, "我最喜欢狗")
        assert sum(len(batch) for batch in embedder.documents) == 11
    finally:
        runtime.stop()
        text.stop()
        memory.close()
        mirror.close()


@pytest.mark.parametrize("race", ["correct", "revoke", "expiry"])
def test_direct_semantic_recall_rejects_stale_fact_during_embedding(tmp_path, race):
    from datetime import UTC, datetime

    from test_rebuilt_memory import apply, mutation
    from virtual_companion.boundary.memory import FactKey
    from virtual_companion.cognition.semantic_retrieval import SemanticMemoryRetriever
    from virtual_companion.cognition.short_context import ShortContextAdmission

    query = "我偏爱的动物是哪种？"
    runtime, text, memory, mirror, model, _, _ = demand_setup(
        tmp_path, source=query, model=DemandModel(first_action="answer"), seed_memory=False,
    )
    try:
        seed(memory.store)
        now = runtime.clock()
        expiry_us = now + 1_000_000
        expiry = datetime.fromtimestamp(expiry_us / 1_000_000, tz=UTC)
        apply(memory.store, mutation(
            "semantic-race-seed", kind="assert_fact", record="favorite-animal",
            content="我最喜欢猫", fact_key=FactKey("user", "favorite-animal", "profile"),
            **({"expires_at": expiry} if race == "expiry" else {}),
        ))
        admission_clock = [now]

        class RacingEmbedder:
            changed = False

            def query_embed(self, value):
                return iter(([1.0, 0.0],))

            def embed(self, values):
                values = list(values)
                if not self.changed:
                    self.changed = True
                    if race == "correct":
                        apply(memory.store, mutation(
                            "semantic-race-correction", kind="correct",
                            record="favorite-animal", content="我最喜欢狗",
                            expected_record_version=1,
                        ))
                    elif race == "revoke":
                        apply(memory.store, mutation(
                            "semantic-race-revoke", kind="revoke",
                            record="favorite-animal", content="",
                            expected_record_version=1,
                        ))
                    else:
                        admission_clock[0] = expiry_us
                return iter([1.0, 0.0] if value == "我最喜欢猫" else [0.0, 1.0]
                            for value in values)

        initial = runtime.admit
        runtime.admit = ShortContextAdmission(
            runtime.store, initial.admit, evidence_clock=lambda: admission_clock[0],
            semantic_retriever=SemanticMemoryRetriever(
                RacingEmbedder(), model_id="fake-race-test", model_revision="fixed",
                page_size=2, candidate_overfetch=8, synthetic_model_identity=True,
            ),
        )

        text.send(query)
        wait_for(lambda: not text.snapshot()["busy"])

        assert text.snapshot()["attempt"]["status"] == "failed"
        assert model.primary == []
        assert runtime.poll()["event"] is None
        if race == "correct":
            assert memory.store.committed_recall(
                memory.companion, now_us=now,
                candidates=(("favorite-animal", 2),),
            )[0]["content"] == "我最喜欢狗"
        elif race == "revoke":
            assert memory.store.committed_recall(
                memory.companion, now_us=now,
                candidates=(("favorite-animal", 1),),
            ) == []
        else:
            assert memory.store.committed_recall(
                memory.companion, now_us=expiry_us,
                candidates=(("favorite-animal", 1),),
            ) == []
    finally:
        runtime.stop()
        text.stop()
        memory.close()
        mirror.close()


@pytest.mark.parametrize("fault", ["cancel", "error"])
def test_required_recall_source_failure_never_generates_final_or_approves(
    tmp_path, monkeypatch, fault,
):
    runtime, text, memory, mirror, model, source, stages = demand_setup(
        tmp_path, "error" if fault == "error" else "query", source=RECALL_QUERY,
        model=DemandModel(first_action="memory"),
    )
    if fault == "cancel":
        run_one = memory.scheduler.user.run_one

        def cancel(*args, **kwargs):
            runtime.cancelled.set()
            return run_one(*args, **kwargs)

        monkeypatch.setattr(memory.scheduler.user, "run_one", cancel)
    text.send(source)
    wait_for(lambda: not text.snapshot()["busy"])
    assert text.snapshot()["attempt"]["status"] == ("cancelled" if fault == "cancel" else "failed")
    assert len(model.calls) == 1 and len(model.primary) == 1
    assert "本次只能返回answer" not in model.primary[0][0]["content"]
    assert memory.store.memory_barrier(memory.companion)
    assert runtime.poll()["event"] is None
    assert memory.store.segments(memory.companion, runtime.claim.input_id) == []
    assert [message["role"] for message in mirror.history()["messages"]] == ["user"]


def test_required_recall_reuses_completed_source_after_final_transport_failure(tmp_path):
    runtime, text, memory, mirror, model, source, stages = demand_setup(
        tmp_path, source="还记得我以前说过什么吗？", model=DemandModel(first_action="memory"),
    )
    original_facts = memory.store.history(memory.companion, "music")
    model.fail_final_once = True
    text.send(source)
    wait_for(lambda: not text.snapshot()["busy"])
    assert text.snapshot()["attempt"]["status"] == "failed"
    assert len(model.primary) == 2
    assert stages == QUERY_STAGES
    assert memory.store.history(memory.companion, "music") == original_facts
    completed_stages = list(stages)
    source_id = runtime.claim.input_id
    assert not memory.store.memory_barrier(memory.companion)
    assert runtime.poll()["event"] is None
    before_retry = len(model.primary)
    text.retry()
    event = ack_and_finish(runtime, text)
    assert len(model.primary) - before_retry == 2
    assert "本次只能返回answer" not in model.primary[-2][0]["content"]
    assert "本次只能返回answer" in model.primary[-1][0]["content"]
    assert runtime.claim.input_id == source_id
    assert stages == completed_stages
    assert memory.store.history(memory.companion, "music") == original_facts
    assert event["readset"]["facts"][0]["record_id"] == "music"
    assert "我喜欢安静的音乐" in json.dumps(model.primary[-1], ensure_ascii=False)


def test_required_recall_rejection_cannot_delegate_a_second_memory_action(tmp_path):
    class RepairMemoryModel(DemandModel):
        def stream(self, messages, cancelled):
            if "你只审查候选文字" in messages[0]["content"]:
                self.calls.append(deepcopy(messages))
                yield json.dumps(synthetic_unit_opinion(json.loads(messages[-1]["content"]),
                    decision="reject",
                    constraints_satisfied=False))
            elif "候选未呈现" in messages[-2]["content"]:
                self.calls.append(deepcopy(messages))
                self.primary.append(deepcopy(messages))
                yield json.dumps({"action": "memory"})
            else:
                yield from super().stream(messages, cancelled)

    runtime, text, memory, mirror, model, source, stages = demand_setup(
        tmp_path, source=RECALL_QUERY, model=RepairMemoryModel(first_action="memory"),
    )
    original_facts = memory.store.history(memory.companion, "music")
    text.send(source)
    wait_for(lambda: not text.snapshot()["busy"])
    assert text.snapshot()["attempt"]["status"] == "failed"
    assert stages == QUERY_STAGES
    assert memory.store.history(memory.companion, "music") == original_facts
    assert len(model.calls) == 4
    assert not memory.store.memory_barrier(memory.companion)
    assert runtime.poll()["event"] is None
    assert memory.store.segments(memory.companion, runtime.claim.input_id) == []
    assert [message["role"] for message in mirror.history()["messages"]] == ["user"]


@pytest.mark.parametrize("mode", ["forget", "correct"])
def test_default_current_management_never_reintroduces_old_memory_in_final_context(tmp_path, mode):
    runtime, text, memory, mirror, model, source, stages = demand_setup(tmp_path, mode)
    text.send(source)
    ack_and_finish(runtime, text)
    assert stages == ["memory_command", "memory_command_support"]
    # Current raw request is preserved, including a quoted old value. Only its
    # independent evidence context must exclude that superseded/forgotten fact.
    evidence_messages = model.primary[-1][1:-1]
    assert "我喜欢安静的音乐" not in json.dumps(evidence_messages, ensure_ascii=False)
    if mode == "correct":
        assert "我喜欢摇滚" in json.dumps(evidence_messages, ensure_ascii=False)
    assert model.primary[-1][-1]["content"] == source
    assert not memory.store.memory_barrier(memory.companion)


@pytest.mark.parametrize("mode", ["unclear", "error"])
def test_unresolved_source_has_no_false_success_and_transport_failure_has_no_approval(
    tmp_path, mode,
):
    class ClarifyingModel(DemandModel):
        def stream(self, messages, cancelled):
            if ("你只审查候选文字" not in messages[0]["content"]
                    and "本次只能返回answer" in messages[0]["content"]):
                self.calls.append(deepcopy(messages))
                self.primary.append(deepcopy(messages))
                yield json.dumps(
                    {"action": "answer", "text": "你指的是哪条记忆？遗忘尚未确认完成。"},
                    ensure_ascii=False)
                return
            yield from super().stream(messages, cancelled)

    runtime, text, memory, mirror, model, source, stages = demand_setup(
        tmp_path, mode, model=ClarifyingModel(first_action="memory"))
    original_facts = memory.store.history(memory.companion, "music")
    text.send(source)
    if mode == "unclear":
        event = ack_and_finish(runtime, text)
        packet = recorded_review_packet(model.calls[-1])
        data = json.loads(packet["context"])
        assert data["memory_access"] == "needs_clarification"
        assert data["source_processing"]["status"] == "failed"
        assert data["source_processing"]["error"].startswith("needs_clarification.")
        assert "current_operation" not in data
        assert event["display_text"] == "你指的是哪条记忆？遗忘尚未确认完成。"
        assert event["readset"] == history_readset(
            runtime.claim.companion_id, runtime.claim.input_id,
            runtime.claim.state_version, packet["context"])
        assert len(model.primary) == 2 and len(model.calls) == 3
        assert stages == ["memory_command"]
    else:
        wait_for(lambda: not text.snapshot()["busy"])
        assert text.snapshot()["attempt"]["status"] == "failed"
        assert memory.store.segments(memory.companion, runtime.claim.input_id) == []
        assert len(model.calls) == len(model.primary) == 1
        assert runtime.poll()["event"] is None
        assert [m["role"] for m in mirror.history()["messages"]] == ["user"]
    assert memory.store.history(memory.companion, "music") == original_facts
    assert stages and memory.store.memory_barrier(memory.companion)


def test_completed_source_receipt_on_retry_does_not_repeat_worker_effects(tmp_path):
    runtime, text, memory, mirror, model, source, stages = demand_setup(tmp_path, "correct")
    model.fail_final_once = True
    text.send(source)
    wait_for(lambda: not text.snapshot()["busy"])
    assert text.snapshot()["attempt"]["status"] == "failed"
    assert stages == ["memory_command", "memory_command_support"]
    source_id = runtime.claim.input_id
    history = memory.store.history(memory.companion, "music")
    assert not memory.store.memory_barrier(memory.companion)
    text.retry()
    ack_and_finish(runtime, text)
    assert runtime.claim.input_id == source_id
    assert stages == ["memory_command", "memory_command_support"]
    assert memory.store.history(memory.companion, "music") == history


@pytest.mark.parametrize("fault", ["cancel", "lease"])
@pytest.mark.parametrize("required", [False, True])
def test_cancel_or_expired_lease_during_final_generation_prevents_approval(
    tmp_path, monkeypatch, fault, required,
):
    from virtual_companion.boundary.contracts import VersionConflict
    from virtual_companion.text import default_runtime

    guardians, faults, cancel_calls = [], [], []

    class ObservedGuardian(default_runtime.TurnGuardian):
        def __enter__(self):
            entered = super().__enter__()
            guardians.append(entered)
            return entered

    monkeypatch.setattr(default_runtime, "TurnGuardian", ObservedGuardian)
    runtime, text, memory, mirror, model, source, stages = demand_setup(
        tmp_path, source=RECALL_QUERY if required else None,
        model=DemandModel(first_action="memory"),
    )
    original_cancel = model.cancel

    def cancel():
        cancel_calls.append(True)
        return original_cancel()

    monkeypatch.setattr(model, "cancel", cancel)

    def interrupt(messages):
        guardian = guardians[-1]
        assert not guardian.stopped.is_set()
        if fault == "cancel":
            runtime.cancelled.set()
        else:
            with memory.store.transaction() as db:
                db.execute("UPDATE turn_runs SET lease_until=? WHERE companion=? AND input=?",
                           (runtime.clock() - 1, memory.companion, runtime.claim.input_id))
        # Exercise the real keeper checkpoint before the final delta can reach
        # the runtime's next check. Polling cadence cannot decide the outcome.
        try:
            guardian.checkpoint()
        except (InterruptedError, VersionConflict) as error:
            faults.append((guardian, type(error), runtime.cancelled.is_set()))
            raise
        raise AssertionError("Injected control failure was not observed")

    model.on_final = interrupt
    text.send(source)
    wait_for(lambda: not text.snapshot()["busy"])
    assert text.snapshot()["attempt"]["status"] == ("cancelled" if fault == "cancel" else "failed")
    assert len(faults) == 1
    guardian, error_type, user_cancelled = faults[0]
    assert error_type is (InterruptedError if fault == "cancel" else VersionConflict)
    assert user_cancelled is (fault == "cancel")
    assert guardian.cancelled is not runtime.cancelled
    assert guardian.cancelled.is_set() and cancel_calls
    assert all(item.stopped.is_set() and not item.observer.is_alive() for item in guardians)
    assert memory.generation_gate._active is None
    assert len(model.calls) == 2
    assert runtime.poll()["event"] is None
    assert memory.store.segments(memory.companion, runtime.claim.input_id) == []
    state = memory.store.stream_state(memory.companion, runtime.claim.input_id)
    assert state["approved_units"] == state["confirmed_units"] == 0
    assert state["confirmed_text"] == ""
    assert [m["role"] for m in mirror.history()["messages"]] == ["user"]
