import json
import sqlite3
import threading
import time
from datetime import UTC, datetime, timedelta

import pytest
from virtual_companion.boundary.contracts import Scope
from virtual_companion.boundary.memory import FactKey, MemoryMutation
from virtual_companion.boundary.sqlite_memory_management import SQLiteExplicitMemoryStore
from virtual_companion.cognition.exchange_model import extraction_schema as exchange_schema
from virtual_companion.cognition.fact_model import extraction_schema
from virtual_companion.text.chronology import utc_microseconds
from virtual_companion.text.fake import FakeModel
from virtual_companion.text.memory_runtime import (
    DEFAULT_COMPANION_ID,
    MemoryAwareModel,
    TextMemoryRuntime,
    _PriorityGenerationGate,
    _StructuredMemoryInvoker,
)

SESSION = "00000000-0000-0000-0000-000000000010"
OTHER_COMPANION = "00000000-0000-0000-0000-000000000002"


@pytest.mark.parametrize("mode", ["query", "forget", "correct", "unclear"])
@pytest.mark.parametrize("entry", ["direct", "session"])
def test_on_demand_recall_resolves_only_bound_current_source(tmp_path, mode, entry):
    from test_memory_command_worker import command, seed
    from virtual_companion.cognition.background_worker import BackgroundScheduler

    runtime = TextMemoryRuntime(tmp_path / "memory.db", DEFAULT_COMPANION_ID)
    seed(runtime.store)
    source = "我喜欢什么音乐？" if mode == "query" else "请忘记我喜欢安静音乐这件事。"
    if mode == "correct":
        source = "把我喜欢安静的音乐改成我喜欢摇滚。"
    current = runtime.accept_user("current", source, SESSION) if entry == "direct" else None
    stages = []

    def invoke(stage, messages, schema, cancelled):
        stages.append(stage)
        if stage == "memory_command":
            if mode == "correct":
                replacement = "我喜欢摇滚"
                start = source.index(replacement)
                return {**command(source), "action": "correct", "replacementQuote": replacement,
                        "replacementStart": start, "replacementEnd": start + len(replacement)}
            if mode != "query":
                return command(source, targets=() if mode == "unclear" else ("music",))
            return {"action": "none", "requestedScope": "none", "targetIds": [],
                    "evidenceQuote": "", "quoteStart": None, "quoteEnd": None,
                    "replacementQuote": "", "replacementStart": None, "replacementEnd": None,
                    "coverage": "not_command", "unresolved": []}
        if stage == "memory_command_support":
            return {"meaning": "explicit_single_correction" if mode == "correct"
                    else "explicit_single_forget", "verdict": "supported"}
        if stage == "fact_extract":
            return {"operations": []}
        if stage == "exchange_intent":
            return {"coverage": "complete", "acts": [{"segmentId": 0,
                     "evidenceQuote": source,
                    "intent": "question", "targetId": None}]}
        if stage == "exchange_extract":
            return {"operations": [{"operation": "add", "type": "response_obligation",
                     "evidenceQuote": source,
                    "supersedesAtomId": None, "expiresAt": None}]}
        if stage == "exchange_support":
            return {"intent": "question", "verdict": "supported"}
        raise AssertionError(stage)

    runtime.scheduler = BackgroundScheduler(
        runtime.store, user_invoke=invoke, assistant_invoke=invoke, clock=lambda: 10,
        memory_commands=True, resolution_protocol=True, resolution_strategy="intent_first",
    )
    if entry == "direct":
        assert runtime.store.recall(DEFAULT_COMPANION_ID, now_us=10) == []
        status = runtime.resolve_for_recall(current.input_id, threading.Event())
    else:
        from test_text_session import wait_done
        from virtual_companion.text.application import TextSession
        from virtual_companion.text.memory import MemoryConversationStore

        captured, statuses = [], []

        class Model:
            def check_context(self, messages):
                captured.append(messages)

            def stream(self, messages, cancelled):
                captured.append(messages)
                yield "synthetic response, not presentation evidence"

            def cancel(self):
                pass

        model = Model()
        wrapped = MemoryAwareModel(model, runtime, runtime.generation_gate)

        def prepare(attempt, session_id, records, cancelled):
            nonlocal current
            message = next(m for m in records if m["turn_id"] == attempt["turn_id"])
            current = runtime.accept_user(attempt["turn_id"], message["content"], session_id)
            assert runtime.store.recall(DEFAULT_COMPANION_ID, now_us=10) == []
            # Explicit memory-demand fixture; no per-turn classifier or default switch.
            statuses.append(runtime.resolve_for_recall(current.input_id, cancelled))
            if statuses[-1] != "current":
                raise ValueError("Source unresolved")
            return wrapped.prepare_request([{"role": "user", "content": message["content"]}])

        session = TextSession(model, MemoryConversationStore(), {}, prepare_turn=prepare)
        session.send(source)
        wait_done(session)
        status = statuses[-1]
        if mode == "unclear":
            assert captured == []
            assert session.snapshot()["attempt"]["status"] == "failed"
        else:
            assert len(captured) == 2 and captured[0] == captured[1]
            if mode == "query":
                assert "我喜欢安静的音乐" in captured[1][0]["content"]
            else:
                assert all("我喜欢安静的音乐" not in m["content"]
                           for m in captured[1] if m["role"] == "system")
    if mode == "unclear":
        assert status != "current"
        assert runtime.store.memory_barrier(DEFAULT_COMPANION_ID)
        assert runtime.store.recall(DEFAULT_COMPANION_ID, now_us=11) == []
    else:
        assert status == "current"
        facts = runtime.store.recall(DEFAULT_COMPANION_ID, now_us=11)
        expected = {"query": ["我喜欢安静的音乐"], "correct": ["我喜欢摇滚"], "forget": []}
        assert [f["content"] for f in facts] == expected[mode]
        count = len(stages)
        assert runtime.resolve_for_recall(current.input_id, threading.Event()) == "current"
        assert len(stages) == count
        assert count == (5 if mode == "query" else 2)
        assert stages == (["memory_command", "exchange_intent", "fact_extract",
                           "exchange_extract", "exchange_support"] if mode == "query"
                          else ["memory_command", "memory_command_support"])
        projected = runtime.with_context([{"role": "user", "content": "我喜欢什么音乐？"}])
        contains_fact = "我喜欢安静的音乐" in json.dumps(projected, ensure_ascii=False)
        assert contains_fact == (mode == "query")
        if mode == "query":
            from test_rebuilt_turns import claim
            from test_segment_store import reviewed
            from test_stream_history import confirm

            run = claim(runtime.store, current)
            runtime.store.approve_segment(
                run, reviewed(run, ("你喜欢安静的音乐。",)), now_us=100
            )
            event = runtime.store.dispatch_segment(run, 1, now_us=100)
            confirm(runtime.store, run, event)
            runtime.store.finish_stream(run, outcome="completed", now_us=100)
            source = "你还记得我的音乐偏好吗？"
            current = runtime.accept_user("next-query", source, SESSION)
            runtime.scheduler.user.clock = lambda: 200
            assert runtime.resolve_for_recall(current.input_id, threading.Event()) == "current"
            remaining = runtime.store.background_sources(DEFAULT_COMPANION_ID)
            assert len(remaining) == 1 and remaining[0]["role"] == "assistant"
            # Reading memory did not drain or impersonate the ACK maintenance job.
            assert remaining[0]["status"] == "queued"


@pytest.mark.parametrize("scenario", [
    "older", "pre_cancel", "during_cancel", "cancel_error", "new_input", "error",
])
def test_on_demand_recall_preserves_unresolved_sources(tmp_path, scenario):
    from types import SimpleNamespace

    from test_memory_command_worker import seed
    from virtual_companion.cognition.background_worker import UserFactWorker

    runtime = TextMemoryRuntime(tmp_path / "memory.db", DEFAULT_COMPANION_ID)
    seed(runtime.store)
    if scenario == "older":
        runtime.accept_user("older", "更正，是另一条", SESSION)
    current = runtime.accept_user("query", "我喜欢什么音乐？", SESSION)
    cancelled = threading.Event()
    calls = []

    def invoke(*args):
        calls.append(1)
        if scenario in {"during_cancel", "cancel_error"}:
            cancelled.set()
        if scenario == "new_input":
            runtime.accept_user("new-input", "请忘掉这个偏好", SESSION)
        if scenario in {"error", "cancel_error"}:
            raise OSError("synthetic model unavailable")
        return {}

    runtime.scheduler = SimpleNamespace(user=UserFactWorker(
        runtime.store, invoke, clock=lambda: 10, memory_commands=True,
    ))
    if scenario == "pre_cancel":
        cancelled.set()
    if scenario == "error":
        # An invalid/rejected provider response is never a safe source result.
        with pytest.raises((ValueError, OSError)):
            runtime.resolve_for_recall(current.input_id, cancelled)
    else:
        status = runtime.resolve_for_recall(current.input_id, cancelled)
        assert status == (
            "pending_source" if scenario in {"older", "new_input"} else "cancelled"
        )
    assert len(calls) == int(scenario not in {"older", "pre_cancel"})
    reopened = type(runtime.store)(runtime.path)
    assert reopened.memory_barrier(DEFAULT_COMPANION_ID)
    assert reopened.recall(DEFAULT_COMPANION_ID, now_us=11) == []
    assert all(row["status"] != "completed"
               for row in reopened.background_sources(DEFAULT_COMPANION_ID))


class _StructuredModel:
    def __init__(self, value):
        self.value = value

    def stream(self, messages, cancelled):
        yield self.value


def test_structured_memory_adapter_only_fills_safe_fact_defaults():
    raw = json.dumps({
        "operations": [{
            "type": "long_term",
            "evidenceTurnId": "source-1",
            "evidenceQuote": "我喜欢喝茶",
            "operation": "add",
            "factKey": {"subject": "user", "predicate": "饮品偏好"},
            "supersedesAtomId": None,
        }],
    },
        ensure_ascii=False,
    )
    schema = extraction_schema("source-1", (), bind_content=True)
    invoke = _StructuredMemoryInvoker(_StructuredModel(raw), threading.Lock())
    value = invoke("fact_extract", [], schema, threading.Event())
    assert value["operations"][0]["expiresAt"] is None
    assert "content" not in value["operations"][0]


def test_structured_memory_adapter_drops_only_directionless_exchange_candidates():
    raw = json.dumps(
        {
            "operations": [{
                "operation": "add",
                "type": None,

                "evidenceQuote": "我喜欢喝茶",
                "supersedesAtomId": None,
                "expiresAt": None,
            }],
        },
        ensure_ascii=False,
    )
    schema = exchange_schema(["source-1"], (), user_only=True)
    invoke = _StructuredMemoryInvoker(_StructuredModel(raw), threading.Lock())
    value = invoke("exchange_extract", [], schema, threading.Event())
    assert value == {"operations": []}


def test_text_memory_runtime_keeps_conversation_session_and_context_scope(tmp_path):
    runtime = TextMemoryRuntime(
        tmp_path / "memory.sqlite3",
        DEFAULT_COMPANION_ID,
        model=FakeModel(),
        fake=True,
    )
    runtime.attach_foreground(lambda: False)
    owner = runtime.store.claim_owner(DEFAULT_COMPANION_ID, "seed-owner", expected_epoch=0)
    runtime.store.mutate(
        MemoryMutation(
            Scope(DEFAULT_COMPANION_ID, SESSION),
            "seed-home",
            "assert_fact",
            "home",
            "我住在上海",
            "manual",
            "source-home",
            "stage1 test",
            datetime(2026, 9, 21, tzinfo=UTC),
            FactKey("user", "home", "profile"),
        ),
        owner,
        expected_version=runtime.store.snapshot(DEFAULT_COMPANION_ID).write_version,
    )
    context = runtime.with_context([{"role": "user", "content": "我住哪里？"}])
    assert context[0]["role"] == "system"
    assert "我住在上海" in context[0]["content"]
    first = runtime.accept_user("turn-00000000000000000000000000000001", "我住哪里？", SESSION)
    second = runtime.accept_user("turn-00000000000000000000000000000001", "我住哪里？", SESSION)

    assert first == second
    deadline = time.monotonic() + 3
    while (
        time.monotonic() < deadline
        and runtime.store.memory_status(DEFAULT_COMPANION_ID) != "current"
    ):
        time.sleep(0.05)
    assert runtime.store.memory_status(DEFAULT_COMPANION_ID) == "current"
    runtime.close()


def test_text_memory_runtime_fake_worker_closes_admitted_source(tmp_path):
    runtime = TextMemoryRuntime(
        tmp_path / "memory.sqlite3",
        DEFAULT_COMPANION_ID,
        model=FakeModel(),
        fake=True,
    )
    runtime.attach_foreground(lambda: False)
    runtime.accept_user(
        "turn-00000000000000000000000000000002", "今天下雨。", SESSION
    )
    deadline = time.monotonic() + 3
    while (
        time.monotonic() < deadline
        and runtime.store.memory_status(DEFAULT_COMPANION_ID) != "current"
    ):
        time.sleep(0.05)
    assert runtime.store.memory_status(DEFAULT_COMPANION_ID) == "current"
    assert runtime.store.background_sources(DEFAULT_COMPANION_ID) == []
    runtime.close()


def test_runtime_ranks_all_valid_memories_before_limiting_context(tmp_path):
    runtime = TextMemoryRuntime(tmp_path / "memory.sqlite3", DEFAULT_COMPANION_ID)
    owner = runtime.store.claim_owner(DEFAULT_COMPANION_ID, "many-owner", expected_epoch=0)
    for index in range(105):
        runtime.store.mutate(
            MemoryMutation(
                Scope(DEFAULT_COMPANION_ID, SESSION), f"seed-{index}", "assert_fact",
                f"record-{index}", "我住在杭州" if index == 104 else f"编号{index}的书是蓝色",
                "manual", f"source-{index}", "synthetic large catalog", datetime.now(UTC),
                FactKey("user", f"property-{index}", "profile"),
            ), owner,
            expected_version=runtime.store.snapshot(DEFAULT_COMPANION_ID).write_version,
        )
    messages = [{"role": "user", "content": "我住哪里？"}]
    context = runtime.with_context(messages)
    assert context[0]["role"] == "system"
    assert "我住在杭州" in context[0]["content"]
    assert "编号" not in context[0]["content"]
    # Selecting across the catalog must not bypass pending-source barriers.
    runtime.accept_user("pending-request", "忘掉我的住址", SESSION)
    assert runtime.with_context(messages) == messages
    runtime.close()


def test_memory_relevance_keeps_single_character_topics_without_grammar_only_matches():
    from virtual_companion.boundary.memory_relevance import lexical_relevance

    drawing = {"content": "我喜欢画画"}
    assert lexical_relevance("画点什么比较好？")(drawing) > 0
    assert lexical_relevance("你还记得我的事吗？")(drawing) == 0
    assert lexical_relevance("画点什么比较好？")({"content": "我喜欢安静的音乐"}) == 0
    assert lexical_relevance("cycling gear")({"content": "I cycle outdoors"}) > 0


@pytest.mark.parametrize("query, predicate, content", [
    ("忙完一整天了，依据以前的偏好，有什么适合听着放松的？",
     "音乐偏好", "我喜欢安静的音乐。"),
    ("忙了一整天，依据以前的兴趣，有什么适合放松的？", "休闲兴趣", "我爱素描。"),
])
def test_memory_relevance_matches_explicit_predicate_phrases(query, predicate, content):
    from virtual_companion.boundary.memory_relevance import lexical_relevance

    fact = {"content": content, "fact_key": {"predicate": predicate}}
    literal = {"content": query}
    unmatched = {"content": "编号零"}
    assert lexical_relevance(query)({"content": content}) == 0
    assert lexical_relevance(query)(literal) > lexical_relevance(query)(fact) > 0
    candidates = lexical_relevance(query, include_unmatched=True)
    assert candidates(literal) > candidates(fact) > candidates(unmatched) > 0


@pytest.mark.parametrize("query", [
    "你还记得我的事吗？",
    "忙完一整天了，想放松一下。",
    "偏偏今天好累。",
    "画点什么比较好？",
    "兴趣两个字怎么写？",
])
def test_memory_predicate_does_not_match_grammar_mood_or_unrelated_topics(query):
    from virtual_companion.boundary.memory_relevance import lexical_relevance

    music = {"content": "我喜欢安静的音乐。", "fact_key": {"predicate": "音乐偏好"}}
    assert lexical_relevance(query)(music) == 0


def test_unmatched_memory_candidates_are_opt_in_and_rank_after_literal_matches():
    from virtual_companion.boundary.memory_relevance import lexical_relevance

    query = "空闲时能怎样放松？"
    drawing = {"content": "我喜欢画画"}
    assert not set(query) & set(drawing["content"])
    assert lexical_relevance(query)(drawing) == 0
    candidates = lexical_relevance(query, include_unmatched=True)
    assert 0 < candidates(drawing) < candidates({"content": "空闲时可以散步"})
    assert candidates({"content": "我住在上海"}) == candidates(drawing)


@pytest.mark.parametrize("change", [None, "correct", "forget", "expiry"])
def test_recall_query_selects_a_tail_head_without_expanding_context_or_reviving_old_values(
    tmp_path, change,
):
    from dataclasses import replace

    from test_effective_history import NOW, WHEN, complete
    from test_rebuilt_turns import A, claim, config
    from virtual_companion.cognition.short_context import ShortContextAdmission
    from virtual_companion.cognition.short_dialogue import ShortRequest

    store = SQLiteExplicitMemoryStore(tmp_path / "tail-query.sqlite3")
    original = store.accept(config(), "drawing-source", "我喜欢画画")
    complete(store, original)
    store.cancel(A, original.input_id)
    owner = store.owner(A) or store.claim_owner(A, "candidate-owner", expected_epoch=0)
    for index in range(8):
        store.mutate(MemoryMutation(
            Scope(A, original.session_id), f"seed-{index}", "assert_fact", f"other-{index}",
            f"topic_{index}", "manual", f"source-{index}", "synthetic distractor", WHEN,
            FactKey("user", f"other-{index}", "profile"),
        ), owner, expected_version=store.snapshot(A).write_version)
    seed = MemoryMutation(
        Scope(A, original.session_id), "seed-drawing", "assert_fact", "drawing", "我喜欢画画",
        "manual", original.input_id, "synthetic tail fact", WHEN,
        FactKey("user", "hobby", "profile"),
        expires_at=WHEN + timedelta(seconds=1) if change == "expiry" else None,
    )
    store.mutate(seed, owner, expected_version=store.snapshot(A).write_version)
    if change in {"correct", "forget"}:
        store.mutate(replace(
            seed, mutation_id="change-drawing", kind="correct" if change == "correct" else "revoke",
            content="我偏爱素描" if change == "correct" else "", fact_key=None,
            source_id="changed-source", expected_record_version=1,
        ), owner, expected_version=store.snapshot(A).write_version)
    text = "空闲时能怎样放松？"
    assert not set(text) & set(seed.content)
    store.ensure_session(A, "new-session")
    current = store.accept(config(), "semantic-query", text, session_id="new-session")
    complete(store, current)
    now = NOW + (1_000_000 if change == "expiry" else 11)
    run = claim(store, current, now=now - 1)
    admission = ShortContextAdmission(store, lambda *args: ShortRequest(args[-1]),
                                      evidence_clock=lambda: now)
    snapshot = store.snapshot(A)
    without_hint = admission(run, snapshot, text, include_unmatched=True)
    assert len(json.loads(without_hint.context.payload)["memories"]) == 8
    assert all(row["record_id"] != "drawing" for row in
               json.loads(without_hint.context.payload)["memories"])
    selected = admission(run, snapshot, text, include_unmatched=True,
                         recall_query="绘画 画画 素描 爱好")
    data = json.loads(selected.context.payload)
    assert len(data["memories"]) == 8 and data["history"] == []
    drawing = [row for row in data["memories"] if row["record_id"] == "drawing"]
    if change in {"forget", "expiry"}:
        assert drawing == []
    else:
        assert data["memories"][0] == drawing[0]
        assert drawing[0]["content"] == ("我偏爱素描" if change == "correct" else seed.content)
        assert drawing[0]["version"] == (2 if change == "correct" else 1)
    assert selected.user_text == text
    admission.validate(selected.context)
    assert store.snapshot(A).write_version == snapshot.write_version


def test_recall_query_matches_fact_text_without_interpreting_record_ids_as_targets():
    from virtual_companion.boundary.memory_relevance import lexical_relevance

    scorer = lexical_relevance("空闲时能怎样放松？", include_unmatched=True,
                               recall_query="drawing-secret-id 绘画 画画")
    assert scorer({"record_id": "drawing-secret-id", "content": "topic_0"}) == 0.5
    assert scorer({"record_id": "unrelated-id", "content": "我喜欢画画"}) > 0.5


def test_short_admission_exposes_bounded_candidates_after_literal_matches(tmp_path):
    from test_effective_history import NOW, WHEN, complete
    from test_rebuilt_turns import A, claim, config
    from virtual_companion.cognition.short_context import ShortContextAdmission
    from virtual_companion.cognition.short_dialogue import ShortRequest

    store = SQLiteExplicitMemoryStore(tmp_path / "candidate-admission.sqlite3")
    original = store.accept(config(), "drawing-source", "我喜欢画画")
    complete(store, original)
    store.cancel(A, original.input_id)
    owner = store.owner(A) or store.claim_owner(A, "candidate-owner", expected_epoch=0)
    contents = [("drawing", "我喜欢画画"),
                *((f"other-{index}", f"topic_{index}") for index in range(8)),
                ("literal", "空闲时可以散步")]
    for record, content in contents:
        store.mutate(MemoryMutation(
            Scope(A, original.session_id), f"seed-{record}", "assert_fact", record,
            content, "manual", f"source-{record}", "synthetic candidate", WHEN,
            FactKey("user", record, "profile"),
        ), owner, expected_version=store.snapshot(A).write_version)
    store.register(OTHER_COMPANION)
    store.ensure_session(OTHER_COMPANION, SESSION)
    other_owner = store.claim_owner(OTHER_COMPANION, "other-owner", expected_epoch=0)
    store.mutate(MemoryMutation(
        Scope(OTHER_COMPANION, SESSION), "foreign-op", "assert_fact", "foreign-record",
        "空闲时能放松", "manual", "foreign-source", "synthetic other companion", WHEN,
        FactKey("user", "hobby", "profile"),
    ), other_owner, expected_version=store.snapshot(OTHER_COMPANION).write_version)
    text = "空闲时能怎样放松？"
    current = store.accept(config(), "query", text, session_id=original.session_id)
    complete(store, current)
    run = claim(store, current, now=NOW)
    admission = ShortContextAdmission(store, lambda *args: ShortRequest(args[-1]),
                                      evidence_clock=lambda: NOW + 11)
    snapshot = store.snapshot(A)
    default = admission(run, snapshot, text)
    explicit_false = admission(run, snapshot, text, include_unmatched=False)
    assert default.context.payload == explicit_false.context.payload
    assert [row["record_id"] for row in json.loads(default.context.payload)["memories"]] \
        == ["literal"]
    requested = admission(run, snapshot, text, include_unmatched=True)
    memories = json.loads(requested.context.payload)["memories"]
    assert len(memories) == 8
    assert memories[0]["record_id"] == "literal"
    assert any(row["content"] == "我喜欢画画" for row in memories)
    assert all(row["record_id"] != "foreign-record" for row in memories)
    admission.validate(requested.context)
    assert store.snapshot(A).write_version == snapshot.write_version


@pytest.mark.parametrize("change", [None, "expiry", "correct", "forget"])
def test_short_admission_candidate_filter_uses_current_store_heads(tmp_path, change):
    from dataclasses import replace

    from test_effective_history import NOW, WHEN, complete
    from test_rebuilt_turns import A, claim, config
    from virtual_companion.cognition.short_context import ShortContextAdmission
    from virtual_companion.cognition.short_dialogue import ShortRequest

    store = SQLiteExplicitMemoryStore(tmp_path / "current-candidates.sqlite3")
    original = store.accept(config(), "drawing-source", "我喜欢画画")
    complete(store, original)
    store.cancel(A, original.input_id)
    owner = store.owner(A) or store.claim_owner(A, "candidate-owner", expected_epoch=0)
    seed = MemoryMutation(
        Scope(A, original.session_id), "seed-hobby", "assert_fact", "hobby", "我喜欢画画",
        "manual", original.input_id, "synthetic candidate", WHEN,
        FactKey("user", "hobby", "profile"),
        expires_at=WHEN + timedelta(seconds=1) if change == "expiry" else None,
    )
    store.mutate(seed, owner, expected_version=store.snapshot(A).write_version)
    if change in {"correct", "forget"}:
        store.mutate(replace(
            seed, mutation_id="change-hobby", kind="correct" if change == "correct" else "revoke",
            content="我偏爱散步" if change == "correct" else "", fact_key=None,
            source_id="changed-source", expected_record_version=1,
        ), owner, expected_version=store.snapshot(A).write_version)
    text = "空闲时能怎样放松？"
    assert not set(text) & set(seed.content)
    current = store.accept(config(), "query", text, session_id=original.session_id)
    complete(store, current)
    now = NOW + (1_000_000 if change == "expiry" else 11)
    run = claim(store, current, now=now - 1)
    admission = ShortContextAdmission(store, lambda *args: ShortRequest(args[-1]),
                                      evidence_clock=lambda: now)
    snapshot = store.snapshot(A)
    requested = admission(run, snapshot, text, include_unmatched=True)
    data = json.loads(requested.context.payload)
    memories = data["memories"]
    if change in {"expiry", "forget"}:
        assert memories == []
    else:
        assert len(memories) == 1
        assert memories[0]["content"] == ("我偏爱散步" if change == "correct" else seed.content)
        assert memories[0]["version"] == (2 if change == "correct" else 1)
        assert memories[0]["event_time"] is None
        rendered = "\n".join(item["content"] for item in requested.model_request(128)["messages"])
        assert "发生时间：不详" in rendered
        assert WHEN.date().isoformat() not in rendered
    if change is not None:
        assert seed.content not in {source["text"] for source in data["history"]}
    admission.validate(requested.context)
    assert store.snapshot(A).write_version == snapshot.write_version


def test_text_memory_runtime_waits_for_a_foreground_idle_window_before_background_work(
    tmp_path,
):
    busy = threading.Event()
    busy.set()
    runtime = TextMemoryRuntime(
        tmp_path / "memory.sqlite3",
        DEFAULT_COMPANION_ID,
        model=FakeModel(),
        fake=True,
        background_idle_grace=0.2,
    )
    runtime.attach_foreground(busy.is_set)
    runtime.accept_user(
        "turn-00000000000000000000000000000003", "我后来搬到杭州了。", SESSION
    )
    try:
        time.sleep(0.05)
        assert runtime.store.memory_status(DEFAULT_COMPANION_ID) == "pending_source"
        busy.clear()
        time.sleep(0.05)
        assert runtime.store.memory_status(DEFAULT_COMPANION_ID) == "pending_source"
        deadline = time.monotonic() + 2
        while (
            time.monotonic() < deadline
            and runtime.store.memory_status(DEFAULT_COMPANION_ID) != "current"
        ):
            time.sleep(0.02)
        assert runtime.store.memory_status(DEFAULT_COMPANION_ID) == "current"
    finally:
        runtime.close()


@pytest.mark.parametrize("content, predicate, query", [
    ("我住在上海", "home", "我住哪里？"),
    ("我喜欢安静的音乐。", "音乐偏好",
     "忙完一整天了，依据以前的偏好，有什么适合听着放松的？"),
])
def test_text_memory_runtime_reopens_cross_day_memory_without_companion_leak(
    tmp_path, content, predicate, query,
):
    path = tmp_path / "memory.sqlite3"
    runtime = TextMemoryRuntime(path, DEFAULT_COMPANION_ID, model=FakeModel(), fake=True)
    owner = runtime.store.claim_owner(DEFAULT_COMPANION_ID, "cross-day-owner", expected_epoch=0)
    runtime.store.mutate(
        MemoryMutation(
            Scope(DEFAULT_COMPANION_ID, "cross-day-session"),
            "cross-day-home",
            "assert_fact",
            "home",
            content,
            "manual",
            "source-cross-day",
            "cross-day fixture",
            datetime(2026, 9, 21, 8, tzinfo=UTC),
            FactKey("user", predicate, "profile"),
            event_time=datetime(2026, 9, 20, 18, tzinfo=UTC),
        ),
        owner,
        expected_version=runtime.store.snapshot(DEFAULT_COMPANION_ID).write_version,
    )
    runtime.close()

    reopened = TextMemoryRuntime(path, DEFAULT_COMPANION_ID, model=FakeModel(), fake=True)
    other = TextMemoryRuntime(path, OTHER_COMPANION, model=FakeModel(), fake=True)
    try:
        messages = [{"role": "user", "content": query}]
        context = reopened.with_context(messages)
        assert content in context[0]["content"]
        assert "2026-09-20" in context[0]["content"]
        assert "2026-09-21" not in context[0]["content"]
        assert other.with_context(messages) == messages
        page = reopened.list_memories(limit=20)
        assert page["stats"]["effective_count"] == 1
        assert page["stats"]["unknown_event_time_count"] == 0
    finally:
        other.close()
        reopened.close()


def test_text_memory_runtime_expires_a_current_fact_without_erasing_history(tmp_path):
    path = tmp_path / "memory.sqlite3"
    runtime = TextMemoryRuntime(path, DEFAULT_COMPANION_ID, model=FakeModel(), fake=True)
    owner = runtime.store.claim_owner(DEFAULT_COMPANION_ID, "ttl-owner", expected_epoch=0)
    expiry = datetime(2026, 9, 22, tzinfo=UTC)
    runtime.store.mutate(
        MemoryMutation(
            Scope(DEFAULT_COMPANION_ID, "ttl-session"),
            "temporary-home",
            "assert_fact",
            "temporary-home",
            "我这周住在上海",
            "manual",
            "source-ttl",
            "ttl fixture",
            datetime(2026, 9, 20, tzinfo=UTC),
            FactKey("user", "temporary_home", "profile"),
            expires_at=expiry,
        ),
        owner,
        expected_version=runtime.store.snapshot(DEFAULT_COMPANION_ID).write_version,
    )
    before_expiry = utc_microseconds(expiry - timedelta(minutes=1))
    at_expiry = utc_microseconds(expiry)
    assert (
        runtime.store.recall(DEFAULT_COMPANION_ID, now_us=before_expiry)[0]["content"]
        == "我这周住在上海"
    )
    assert runtime.store.recall(DEFAULT_COMPANION_ID, now_us=at_expiry) == []
    page = runtime.store.memory_records(DEFAULT_COMPANION_ID, now_us=at_expiry)
    stats = runtime.store.memory_statistics(DEFAULT_COMPANION_ID, now_us=at_expiry)
    assert page["total"] == 1
    assert page["rows"][0]["status"] == "expired"
    assert stats["current_record_count"] == 1
    assert stats["effective_count"] == 0
    assert stats["history_version_count"] == 1
    assert stats["status_counts"]["expired"] == 1
    runtime.close()


def test_memory_backup_restores_current_fact_and_version_history(tmp_path):
    path = tmp_path / "memory.sqlite3"
    runtime = TextMemoryRuntime(path, DEFAULT_COMPANION_ID, model=FakeModel(), fake=True)
    owner = runtime.store.claim_owner(DEFAULT_COMPANION_ID, "backup-owner", expected_epoch=0)
    runtime.store.mutate(
        MemoryMutation(
            Scope(DEFAULT_COMPANION_ID, "backup-session"),
            "home",
            "assert_fact",
            "home",
            "我住在上海",
            "manual",
            "source-backup",
            "backup fixture",
            datetime(2026, 9, 21, tzinfo=UTC),
            FactKey("user", "home", "profile"),
            event_time=datetime(2026, 9, 20, 18, tzinfo=UTC),
        ),
        owner,
        expected_version=runtime.store.snapshot(DEFAULT_COMPANION_ID).write_version,
    )
    runtime.store.mutate(
        MemoryMutation(
            Scope(DEFAULT_COMPANION_ID, "backup-session"),
            "backup-home-correction",
            "correct",
            "home",
            "我住在苏州",
            "manual",
            "source-backup-correction",
            "backup correction",
            datetime(2026, 9, 22, tzinfo=UTC),
            expected_record_version=1,
        ),
        owner,
        expected_version=runtime.store.snapshot(DEFAULT_COMPANION_ID).write_version,
    )
    backup = tmp_path / "memory-backup.sqlite3"
    try:
        runtime.store.backup_to(backup)
    finally:
        runtime.close()

    restored = type(runtime.store)(backup)
    items = restored.memory_records(
        DEFAULT_COMPANION_ID,
        limit=20,
        now_us=utc_microseconds(datetime(2026, 9, 22, tzinfo=UTC)),
    )
    stats = restored.memory_statistics(
        DEFAULT_COMPANION_ID,
        now_us=utc_microseconds(datetime(2026, 9, 22, tzinfo=UTC)),
    )
    assert stats["effective_count"] == 1
    assert stats["history_version_count"] == 2
    assert items["rows"][0]["content"] == "我住在苏州"
    assert items["rows"][0]["event_time"] == "2026-09-20T18:00:00+00:00"


def test_memory_backup_does_not_overwrite_a_destination_created_during_backup(
    tmp_path, monkeypatch,
):
    store = SQLiteExplicitMemoryStore(tmp_path / "memory.sqlite3")
    destination = tmp_path / "competing-backup.sqlite3"
    connect = sqlite3.connect

    def competing_connect(path, *args, **kwargs):
        if kwargs.get("uri") and "mode=ro" in str(path):
            destination.write_bytes(b"another owner's backup")
        return connect(path, *args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", competing_connect)
    with pytest.raises(FileExistsError):
        store.backup_to(destination)
    assert destination.read_bytes() == b"another owner's backup"


def test_foreground_generation_preempts_an_inflight_background_generation():
    gate = _PriorityGenerationGate()
    background_started = threading.Event()
    background_cancelled = threading.Event()
    foreground_entered = threading.Event()

    def background():
        with gate.background(background_cancelled.set):
            background_started.set()
            while not background_cancelled.wait(0.01):
                pass

    worker = threading.Thread(target=background)
    worker.start()
    assert background_started.wait(1)

    def foreground():
        with gate.foreground():
            foreground_entered.set()

    waiting = threading.Thread(target=foreground)
    waiting.start()
    assert foreground_entered.wait(1)
    assert background_cancelled.is_set()
    waiting.join(1)
    worker.join(1)
    assert not worker.is_alive()


def test_foreground_preemption_marks_background_cancel_before_model_socket_exists():
    gate = _PriorityGenerationGate()
    background_started = threading.Event()
    background_cancelled = threading.Event()
    foreground_entered = threading.Event()

    def background():
        with gate.background(lambda: None, background_cancelled):
            background_started.set()
            background_cancelled.wait(1)

    worker = threading.Thread(target=background)
    worker.start()
    assert background_started.wait(1)

    def foreground():
        with gate.foreground():
            foreground_entered.set()

    waiting = threading.Thread(target=foreground)
    waiting.start()
    assert foreground_entered.wait(1)
    assert background_cancelled.is_set()
    waiting.join(1)
    worker.join(1)
    assert not worker.is_alive()


def test_context_preflight_preempts_background_generation():
    gate = _PriorityGenerationGate()
    background_started = threading.Event()
    background_cancelled = threading.Event()
    context_checked = threading.Event()

    def background():
        with gate.background(lambda: None, background_cancelled):
            background_started.set()
            background_cancelled.wait(1)

    class _ContextModel:
        def check_context(self, messages):
            assert background_cancelled.is_set()
            context_checked.set()

        def stream(self, messages, cancelled):
            yield "unused"

    class _Memory:
        def with_context(self, messages):
            return messages

    worker = threading.Thread(target=background)
    worker.start()
    assert background_started.wait(1)

    MemoryAwareModel(_ContextModel(), _Memory(), gate).check_context(
        [{"role": "user", "content": "hello"}]
    )
    assert context_checked.wait(1)
    worker.join(1)
    assert not worker.is_alive()


@pytest.mark.parametrize("kind", ["direct", "prepared_stream", "prepared_count", "background"])
def test_cancelled_gate_waiter_exits_without_waiting_for_active_owner(kind):
    gate = _PriorityGenerationGate()
    cancelled, done = threading.Event(), threading.Event()
    queued = threading.Event()
    calls, errors = [], []
    original_wait = gate._condition.wait

    def observe_wait(timeout=None):
        queued.set()
        return original_wait(timeout)

    gate._condition.wait = observe_wait

    class Model:
        def check_context(self, messages):
            calls.append("count")

        def stream(self, messages, event):
            calls.append("stream")
            yield "must not generate"

    class Memory:
        def with_context(self, messages):
            return messages

        def capture_context(self, messages):
            return messages, lambda: None

    wrapper = MemoryAwareModel(Model(), Memory(), gate)
    messages = [{"role": "user", "content": "hello"}]
    request = None
    if kind.startswith("prepared"):
        request = wrapper.prepare_request(messages, cancelled=cancelled)
        if kind == "prepared_stream":
            request.check_context()
            calls.clear()

    def waiter():
        try:
            if kind == "direct":
                list(wrapper.stream(messages, cancelled))
            elif kind == "prepared_stream":
                list(request.stream(cancelled))
            elif kind == "prepared_count":
                request.check_context()
            else:
                with gate.background(lambda: None, cancelled):
                    calls.append("background")
        except InterruptedError:
            pass
        except Exception as error:
            errors.append(error)
        finally:
            done.set()

    # Keep the first owner alive until after the assertion: completing the
    # owner must not be what lets a cancelled waiter leave the queue.
    with gate.foreground():
        worker = threading.Thread(target=waiter)
        worker.start()
        assert queued.wait(1), "Waiter did not reach the occupied gate"
        cancelled.set()
        exited_while_owned = done.wait(0.5)
    worker.join(1)
    assert exited_while_owned
    assert not errors and not calls
    assert gate._foreground_waiters == 0
    with gate.background(lambda: None):
        pass  # Cancelled foreground reservations cannot starve later work.


def test_prepared_request_cannot_escape_bound_cancellation_with_another_event():
    cancelled = threading.Event()
    sent = []

    class Memory:
        def capture_context(self, messages):
            return messages, lambda: None

    class Model:
        def check_context(self, messages):
            pass

        def stream(self, messages, event):
            sent.append(messages)
            yield "stale reply"

    request = MemoryAwareModel(Model(), Memory(), _PriorityGenerationGate()).prepare_request(
        [{"role": "user", "content": "hello"}], cancelled=cancelled
    )
    request.check_context()
    cancelled.set()
    with pytest.raises(ValueError, match="cancellation"):
        list(request.stream(threading.Event()))
    assert not sent


def test_prepared_memory_request_binds_preflight_and_stream_to_one_projection():
    class Memory:
        calls = 0

        def capture_context(self, messages):
            return self.with_context(messages), lambda: None

        def with_context(self, messages):
            self.calls += 1
            return [{"role": "system", "content": f"snapshot-{self.calls}"}, *messages]

    class Model:
        def check_context(self, messages):
            self.checked = messages

        def stream(self, messages, cancelled):
            self.sent = messages
            yield "answer"

    memory, model = Memory(), Model()
    wrapper = MemoryAwareModel(model, memory, _PriorityGenerationGate())
    request = wrapper.prepare_request([{"role": "user", "content": "question"}])
    request.check_context()
    assert list(request.stream(threading.Event())) == ["answer"]
    assert model.checked == model.sent
    assert memory.calls == 1


def test_prepared_memory_request_rejects_revision_after_preflight(tmp_path):
    from virtual_companion.boundary.contracts import VersionConflict

    class Model:
        sent = False

        def check_context(self, messages):
            pass

        def stream(self, messages, cancelled):
            self.sent = True
            yield "stale"

    model = Model()
    memory = TextMemoryRuntime(tmp_path / "memory.sqlite3", DEFAULT_COMPANION_ID)
    wrapper = MemoryAwareModel(model, memory, memory.generation_gate)
    request = wrapper.prepare_request([{"role": "user", "content": "question"}])
    request.check_context()
    memory.accept_user("concurrent-input", "先等等", SESSION)
    with pytest.raises(VersionConflict):
        list(request.stream(threading.Event()))
    assert not model.sent


def test_prepared_request_is_immutable_cancelled_and_single_use(tmp_path):
    class Model:
        sent = False

        def check_context(self, messages):
            messages[0]["content"] = "model mutation"

        def stream(self, messages, cancelled):
            self.sent = True
            yield messages[0]["content"]

    model = Model()
    memory = TextMemoryRuntime(tmp_path / "memory.sqlite3", DEFAULT_COMPANION_ID)
    wrapper = MemoryAwareModel(model, memory, memory.generation_gate)
    messages = [{"role": "user", "content": "original"}]
    request = wrapper.prepare_request(messages)
    messages[0]["content"] = "caller mutation"
    with pytest.raises(ValueError, match="preflight"):
        list(request.stream(threading.Event()))
    request.check_context()
    assert list(request.stream(threading.Event())) == ["original"]
    with pytest.raises(ValueError, match="preflight"):
        list(request.stream(threading.Event()))
    model.sent = False
    request = wrapper.prepare_request(messages)
    request.check_context()
    cancelled = threading.Event()
    cancelled.set()
    assert list(request.stream(cancelled)) == []
    assert not model.sent


def test_prepared_request_stops_stale_delta_after_revision(tmp_path):
    memory = TextMemoryRuntime(tmp_path / "memory.sqlite3", DEFAULT_COMPANION_ID)

    class Model:
        closed = False

        def check_context(self, messages):
            pass

        def stream(self, messages, cancelled):
            try:
                yield "prefix"
                memory.accept_user("revision-during-stream", "改一下", SESSION)
                yield "stale suffix"
            finally:
                self.closed = True

    from virtual_companion.boundary.contracts import VersionConflict

    model = Model()
    request = MemoryAwareModel(model, memory, memory.generation_gate).prepare_request(
        [{"role": "user", "content": "question"}]
    )
    request.check_context()
    stream = request.stream(threading.Event())
    assert next(stream) == "prefix"
    with pytest.raises(VersionConflict):
        next(stream)
    assert model.closed


def test_prepared_request_does_not_fall_back_on_projection_error(tmp_path, monkeypatch):
    memory = TextMemoryRuntime(tmp_path / "memory.sqlite3", DEFAULT_COMPANION_ID)

    def unavailable(*args, **kwargs):
        raise sqlite3.OperationalError("synthetic unavailable memory")

    monkeypatch.setattr(memory.store, "ranked_recall", unavailable)
    with pytest.raises(sqlite3.OperationalError):
        MemoryAwareModel(FakeModel(), memory, memory.generation_gate).prepare_request(
            [{"role": "user", "content": "question"}]
        )


def test_prepared_request_rejects_expiry_without_store_write(tmp_path, monkeypatch):
    from virtual_companion.boundary.contracts import VersionConflict
    from virtual_companion.text import memory_runtime

    expiry = datetime(2026, 10, 1, tzinfo=UTC)

    class Clock:
        value = expiry - timedelta(seconds=1)
        fromisoformat = staticmethod(datetime.fromisoformat)

        @classmethod
        def now(cls, zone):
            return cls.value

    monkeypatch.setattr(memory_runtime, "datetime", Clock)
    memory = TextMemoryRuntime(tmp_path / "memory.sqlite3", DEFAULT_COMPANION_ID)
    owner = memory.store.claim_owner(DEFAULT_COMPANION_ID, "expiry-owner", expected_epoch=0)
    memory.store.mutate(
        MemoryMutation(
            Scope(DEFAULT_COMPANION_ID, SESSION), "expiry-op", "assert_fact", "home",
            "我这周住在上海", "manual", "expiry-source", "synthetic",
            Clock.value, FactKey("user", "home", "profile"), expires_at=expiry,
        ),
        owner,
        expected_version=memory.store.snapshot(DEFAULT_COMPANION_ID).write_version,
    )

    class Model:
        sent = False

        def check_context(self, messages):
            assert "我这周住在上海" in messages[0]["content"]

        def stream(self, messages, cancelled):
            self.sent = True
            yield "expired"

    model = Model()
    request = MemoryAwareModel(model, memory, memory.generation_gate).prepare_request(
        [{"role": "user", "content": "我住哪里？"}]
    )
    request.check_context()
    version = memory.store.snapshot(DEFAULT_COMPANION_ID).write_version
    Clock.value = expiry
    with pytest.raises(VersionConflict):
        list(request.stream(threading.Event()))
    assert memory.store.snapshot(DEFAULT_COMPANION_ID).write_version == version
    assert not model.sent


def test_prepared_request_reads_catalog_once(tmp_path, monkeypatch):
    memory = TextMemoryRuntime(tmp_path / "memory.sqlite3", DEFAULT_COMPANION_ID)
    original = memory.store.ranked_recall
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(memory.store, "ranked_recall", counted)

    class Model:
        def check_context(self, messages):
            pass

        def stream(self, messages, cancelled):
            yield "first"
            yield "second"

    request = MemoryAwareModel(Model(), memory, memory.generation_gate).prepare_request(
        [{"role": "user", "content": "question"}]
    )
    request.check_context()
    assert list(request.stream(threading.Event())) == ["first", "second"]
    assert len(calls) == 1


def test_prepared_request_rejects_revision_at_terminal(tmp_path):
    from virtual_companion.boundary.contracts import VersionConflict

    memory = TextMemoryRuntime(tmp_path / "memory.sqlite3", DEFAULT_COMPANION_ID)

    class Model:
        def check_context(self, messages):
            pass

        def stream(self, messages, cancelled):
            yield "prefix"
            memory.accept_user("revision-at-terminal", "先别说了", SESSION)

    request = MemoryAwareModel(Model(), memory, memory.generation_gate).prepare_request(
        [{"role": "user", "content": "question"}]
    )
    request.check_context()
    stream = request.stream(threading.Event())
    assert next(stream) == "prefix"
    with pytest.raises(VersionConflict):
        next(stream)


@pytest.mark.parametrize("stage", ["exchange_intent", "exchange_extract",
                                   "fact_extract", "memory_command",
                                   "demand_control", "final_review"])
def test_structured_memory_output_budget_only_changes_exchange_stages(stage):
    limits, checked, sent = [], [], []

    class StageModel:
        supports_stage_metadata = True
        supports_stage_output_override = True

        def prepare_stage_request(self, messages, *, stage, schema, output_tokens=192):
            limits.append(output_tokens)

            class Request:
                def check_context(self):
                    checked.append(output_tokens)

                def stream(self, cancelled):
                    sent.append(output_tokens)
                    yield '{"ok":true}'

            return Request()

    invoke = _StructuredMemoryInvoker(StageModel(), _PriorityGenerationGate())
    assert invoke(stage, [], {"type": "object"}, threading.Event()) == {"ok": True}
    expected = 1024 if stage in {"exchange_intent", "exchange_extract"} else 192
    assert limits == checked == sent == [expected]
