"""CPU evidence actions prove bounded orchestration and store identity, not semantics."""

import json
from dataclasses import replace
from threading import Event

import pytest
from jsonschema import Draft202012Validator
from virtual_companion.boundary.contracts import VersionConflict
from virtual_companion.cognition.stream_unit_review import CandidateRejected
from virtual_companion.text import demand_memory
from virtual_companion.text.turn_orchestrator import TurnOrchestrator


def fact(index, **changes):
    return {"record_id": f"fact-{index}", "version": 1,
            "content": f"记录{index}：不喝冰水。", "source_id": f"source-{index}", **changes}


def session(lookup, *, cancelled=None, initial=(), checkpoint=None, **limits):
    from virtual_companion.text.evidence_recall import EvidenceRecallSession

    cancelled = cancelled or Event()
    return EvidenceRecallSession(lookup=lookup, cancelled=cancelled,
        checkpoint=checkpoint or (lambda signal: None), initial=initial, **limits)


def test_recall_is_explicit_and_can_follow_memory_final_without_becoming_writable():
    with pytest.raises(ValueError):
        demand_memory._action('{"action":"recall","query":"旧票"}', final=False)
    decoded = demand_memory._action(
        '{"action":"recall","query":"  旧票  "}', final=True, allow_recall=True)
    assert decoded == {"action": "recall", "query": "旧票"}
    schema = demand_memory.action_schema(allow_memory=False, allow_recall=True)
    assert Draft202012Validator(schema).is_valid(decoded)
    assert not Draft202012Validator(schema).is_valid({"action": "memory"})


@pytest.mark.parametrize("body", [
    '{"action":"recall"}', '{"action":"recall","query":""}',
    '{"action":"recall","query":null}', '{"action":"recall","query":"x","limit":24}',
    '{"action":"recall","query":"x","query":"x"}',
])
def test_recall_query_shape_is_strict(body):
    with pytest.raises(ValueError):
        demand_memory._action(body, final=True, allow_recall=True)


def test_session_preserves_first_hop_and_both_eight_record_hops():
    calls, cancelled = [], Event()

    def lookup(query, *, cancelled):
        calls.append(query)
        start = 8 if query == "第一跳" else 16
        return [fact(i) for i in range(start, start + 8)]

    current = session(lookup, cancelled=cancelled, initial=[fact(i) for i in range(8)])
    first = current.recall("第一跳", cancelled=cancelled)
    assert len(first.memories) == 16 and first.can_recall
    second = current.recall("第二跳", cancelled=cancelled)
    assert list(second.memories) == [fact(i) for i in range(24)]
    assert second.evidence_access["coverage"] == "partial"
    assert second.evidence_access["stop_reason"] == "call_limit"
    assert not second.can_recall and calls == ["第一跳", "第二跳"]
    assert current.recall("第三跳", cancelled=cancelled).memories == second.memories
    assert calls == ["第一跳", "第二跳"]


@pytest.mark.parametrize("mode", ["duplicate_query", "no_new_evidence", "empty"])
def test_repeat_or_no_new_evidence_stops_without_replaying_lookup(mode):
    calls, cancelled = [], Event()

    def lookup(query, *, cancelled):
        calls.append(query)
        return [] if mode == "empty" else [fact(0)]

    current = session(lookup, cancelled=cancelled,
                      initial=[fact(0)] if mode == "no_new_evidence" else [])
    first = current.recall("不是每天", cancelled=cancelled)
    stopped = (current.recall("  不是每天  ", cancelled=cancelled)
               if mode == "duplicate_query" else first)
    assert not stopped.can_recall
    assert len(calls) == 1
    assert stopped.evidence_access["stop_reason"] == mode


def test_content_limit_keeps_existing_complete_original_and_marks_partial():
    cancelled = Event()
    current = session(lambda q, **kw: [fact(1, content="新" * 1000)],
                      cancelled=cancelled, initial=[fact(0)], max_bytes=300)
    result = current.recall("更多", cancelled=cancelled)
    assert list(result.memories) == [fact(0)]
    assert result.evidence_access["stop_reason"] == "content_limit"
    assert result.evidence_access["coverage"] == "partial" and not result.can_recall


@pytest.mark.parametrize("changes", [{"content": "改写原文"}, {"version": 2}])
def test_conflicting_identity_does_not_silently_replace_first_hop(changes):
    cancelled = Event()
    current = session(lambda q, **kw: [fact(0, **changes)], cancelled=cancelled, initial=[fact(0)])
    with pytest.raises(VersionConflict):
        current.recall("同记录", cancelled=cancelled)
    assert list(current.selection.memories) == [fact(0)]


def test_returned_rows_do_not_mutate_private_evidence():
    cancelled = Event()
    original = fact(0)
    current = session(lambda q, **kw: [original], cancelled=cancelled)
    result = current.recall("查原文", cancelled=cancelled)
    original["content"] = "lookup后来改写"
    result.memories[0]["content"] = "调用者后来改写"
    assert current.selection.memories[0]["content"] == fact(0)["content"]


@pytest.mark.parametrize("mode", ["cancel", "deadline", "wrong_event", "failure"])
def test_cancel_deadline_and_lookup_error_are_not_empty_results(mode):
    cancelled, calls = Event(), []

    def checkpoint(signal):
        if signal.is_set():
            raise InterruptedError("same turn cancelled")
        if mode == "deadline":
            raise TimeoutError("same turn deadline")

    def lookup(query, *, cancelled):
        calls.append(query)
        if mode == "cancel":
            cancelled.set()
        if mode == "failure":
            raise RuntimeError("retriever original failure")
        return [fact(0)]

    current = session(lookup, cancelled=cancelled, checkpoint=checkpoint)
    error = {"cancel": InterruptedError, "deadline": TimeoutError,
             "wrong_event": ValueError, "failure": RuntimeError}[mode]
    with pytest.raises(error):
        current.recall("查询", cancelled=Event() if mode == "wrong_event" else cancelled)
    assert current.selection.memories == ()
    assert len(calls) == (0 if mode in {"deadline", "wrong_event"} else 1)


class Prepared:
    def __init__(self, action):
        self.action, self.request = action, object()

    def check_context(self):
        pass

    def stream(self, cancelled):
        assert not cancelled.is_set()
        yield json.dumps(self.action, ensure_ascii=False)


def turn(actions, *, reject_once=False, prefix=False):
    actions = iter(actions)
    cancelled, prepares, lookups, writes, deliveries = Event(), [], [], [], []
    evidence = session(lambda q, **kw: lookups.append(q) or [fact(len(lookups))],
                       cancelled=cancelled, initial=[fact(0)])
    rejected = False

    def prepare(claim, **kwargs):
        prepares.append(kwargs)
        return Prepared(next(actions))

    def deliver(prepared, candidate):
        nonlocal rejected
        text = "".join(candidate)
        if reject_once and not rejected:
            rejected = True
            raise CandidateRejected({"decision": "reject"}, text)
        deliveries.append((text, prepares[-1].get("evidence")))
        return text

    orchestrator = TurnOrchestrator(object(), cancelled, prepare=prepare,
        resolve_current=lambda c: writes.append(c), refresh=lambda c: c,
        check=lambda *args: None, has_approved_prefix=lambda: prefix,
        deliver=deliver, settle_failure=lambda c: None, settle_listen=lambda p: None,
        recall_evidence=lambda c, q, *, cancelled: evidence.recall(q, cancelled=cancelled),
        max_recall_calls=2)
    return orchestrator, cancelled, prepares, lookups, writes, deliveries


def test_existing_material_answers_without_any_tool():
    driver, cancelled, prepares, lookups, writes, deliveries = turn([
        {"action": "answer", "text": "已有材料足够。"}])
    assert list(driver.stream(cancelled)) == ["已有材料足够。"]
    assert not lookups and not writes and len(prepares) == 1
    assert driver.result.recall_calls == 0


def test_readonly_recall_keeps_memory_delegation_available_once():
    driver, cancelled, prepares, lookups, writes, deliveries = turn([
        {"action": "memory"}, {"action": "recall", "query": "相关旧事"},
        {"action": "answer", "text": "依据原文回应。"}])
    assert list(driver.stream(cancelled)) == ["依据原文回应。"]
    assert len(writes) == 1 and lookups == ["相关旧事"]
    assert prepares[1]["final"] is True and prepares[1]["allow_recall"]
    assert deliveries[-1][1].memories == (fact(0), fact(1))
    assert (driver.result.memory_calls, driver.result.recall_calls) == (1, 1)


def test_repair_uses_same_remaining_recall_budget_and_original_evidence():
    driver, cancelled, prepares, lookups, writes, deliveries = turn([
        {"action": "recall", "query": "第一条线索"}, {"action": "answer", "text": "被拒绝草稿。"},
        {"action": "recall", "query": "第二条线索"},
        {"action": "answer", "text": "按两条原文回答。"},
    ], reject_once=True)
    assert list(driver.stream(cancelled)) == ["按两条原文回答。"]
    assert lookups == ["第一条线索", "第二条线索"] and not writes
    assert deliveries[-1][1].memories == (fact(0), fact(1), fact(2))
    assert not prepares[-1]["allow_recall"] and driver.result.recall_calls == 2


def test_repair_cannot_reset_exhausted_recall_budget():
    driver, cancelled, prepares, lookups, writes, deliveries = turn([
        {"action": "recall", "query": "第一条"}, {"action": "recall", "query": "第二条"},
        {"action": "answer", "text": "被拒绝草稿。"}, {"action": "recall", "query": "第三条"},
    ], reject_once=True)
    with pytest.raises(ValueError):
        list(driver.stream(cancelled))
    assert lookups == ["第一条", "第二条"]
    assert not prepares[-1]["allow_recall"]


def test_approved_prefix_does_not_start_recall_repair():
    driver, cancelled, prepares, lookups, writes, deliveries = turn([
        {"action": "answer", "text": "部分已显示。"}, {"action": "recall", "query": "不应执行"},
    ], reject_once=True, prefix=True)
    with pytest.raises(CandidateRejected):
        list(driver.stream(cancelled))
    assert len(prepares) == 1 and not lookups


def test_real_store_originals_survive_two_readonly_hops_without_writes(tmp_path):
    from test_rebuilt_memory import A, apply, mutation
    from test_rebuilt_turns import config
    from virtual_companion.boundary.memory import FactKey
    from virtual_companion.boundary.sqlite_memory_management import SQLiteExplicitMemoryStore

    store = SQLiteExplicitMemoryStore(tmp_path / "facts.sqlite3")
    contents = [fact(i)["content"] for i in range(24)]
    source = store.accept(config(), "manual-seed", "\n".join(contents))
    store.cancel(A, source.input_id)
    store.owner(A) or store.claim_owner(A, "manual", expected_epoch=0)
    for index, content in enumerate(contents):
        apply(store, replace(
            mutation(f"seed-{index}", kind="assert_fact", record=f"fact-{index}",
                     content=content, fact_key=FactKey("user", f"fact-{index}", "profile")),
            source_kind="manual", source_id=source.input_id))
    before = store.snapshot(A).write_version
    all_rows = store.committed_recall(
        A, now_us=0, candidates=tuple((f"fact-{i}", 1) for i in range(24)), limit=24)
    assert len(all_rows) == 24
    cancelled = Event()

    def lookup(query, *, cancelled):
        start = 8 if query == "first" else 16
        return store.committed_recall(A, now_us=0,
            candidates=tuple((f"fact-{i}", 1) for i in range(start, start + 8)), limit=8)

    current = session(lookup, cancelled=cancelled, initial=all_rows[:8])
    current.recall("first", cancelled=cancelled)
    selected = current.recall("second", cancelled=cancelled)
    assert list(selected.memories) == all_rows
    assert store.snapshot(A).write_version == before
    assert store.inspect(A, source.input_id)["text"] == "\n".join(contents)
    assert all(row["source_id"] == source.input_id for row in selected.memories)


def test_cancel_flag_is_respected_even_when_host_checkpoint_only_checks_owner():
    cancelled, calls = Event(), []
    current = session(lambda q, **kw: calls.append(q) or [fact(0)], cancelled=cancelled)
    cancelled.set()
    with pytest.raises(InterruptedError):
        current.recall("不应开始", cancelled=cancelled)
    assert not calls
