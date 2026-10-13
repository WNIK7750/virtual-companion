"""Real default admission/action wiring with explicitly supplied CPU opinions.

These requests use the real prepared ProviderModel, one source worker, sealed
review, approval and ACK. The supplier does not demonstrate model semantics.
"""

import json
import os
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from hashlib import sha256
from pathlib import Path
from threading import Event

import pytest
from test_default_cross_turn_memory import Scenario
from test_default_text_delivery import wait_for
from test_unit_scope_review_v4 import (
    recorded_calls,
    recording_runtime,
)
from test_unit_scope_review_v4 import (
    sealed_capture as sealed_capture,
)
from virtual_companion.boundary.contracts import VersionConflict, canonical
from virtual_companion.boundary.history_readset import history_readset
from virtual_companion.cognition.background_worker import BackgroundScheduler
from virtual_companion.cognition.errors import FactSupportUnresolved
from virtual_companion.cognition.short_dialogue import PROMPTS, ShortContext

SEED = "我后来搬到杭州了。"
QUERY = "一句话回答我搬去了哪里。"
REPLY = "你后来搬到杭州了。"


def record(label, runtime, transport, **extra):
    directory = os.environ.get("P102_ADMISSION_RECORD_DIR")
    if directory:
        value = {
            "protocol": "cpu-memory-admission-recording.v1",
            "semantic_effect_proven": False,
            "source_id": runtime.claim.input_id,
            "calls": recorded_calls(transport),
            "sealed": transport.sealed,
            "opinions": transport.opinions,
            **extra,
        }
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / (label + ".json")).write_text(canonical(value) + "\n", encoding="utf-8")


def build(tmp_path, packets, actions):
    runtime, session, memory, mirror, transport = recording_runtime(
        tmp_path, packets, revision="short-dialogue.v27", actions=actions)
    worker = Scenario(original=SEED, query=QUERY, recalled=REPLY, predicate="city",
                      updated="我后来搬到苏州了。", updated_reply="你后来搬到苏州了。")
    runtime.clock = worker.clock
    memory.scheduler = BackgroundScheduler(
        memory.store, user_invoke=worker.invoke, assistant_invoke=worker.invoke,
        clock=worker.clock, resolution_protocol=True, resolution_strategy="intent_first",
        memory_commands=True)
    stages = []
    original = runtime.prepare

    def prepare(*args, **kwargs):
        stage = original(*args, **kwargs)
        stages.append(stage)
        return stage

    runtime.prepare = prepare
    return runtime, session, memory, mirror, transport, worker, stages


def send(runtime, session, text):
    session.send(text)
    return finish(runtime, session)


def finish(runtime, session):
    event = wait_for(lambda: runtime.poll()["event"] or not session.snapshot()["busy"])
    assert isinstance(event, dict), session.snapshot()
    assert session.snapshot()["partial"] == ""
    runtime.ack({"sequence": event["sequence"], "fingerprint": event["fingerprint"],
                 "host_receipt": "synthetic-admission-" + event["fingerprint"]})
    wait_for(lambda: not session.snapshot()["busy"])
    assert session.snapshot()["attempt"]["status"] == "completed"
    return event


def seed_and_queue(runtime, session, memory):
    first = send(runtime, session, SEED)
    assert memory.store.memory_status(memory.companion) == "current"
    fact, = memory.store.recall(memory.companion, now_us=runtime.clock())
    assert fact["source_id"] == first["basis"]["input_id"] and fact["content"] == SEED
    queued = send(runtime, session, "只依据已有来源回答，不确定的细节请保留为未知。")
    assert memory.store.memory_status(memory.companion) == "pending_source"
    return first, fact, queued


@pytest.mark.parametrize("delegate", [False, True], ids=["initial", "one-current-delegation"])
def test_committed_city_survives_prior_queue_in_real_initial_final_review_and_ack(
    tmp_path, sealed_capture, delegate
):
    actions = [{"action": "memory"}, {"action": "answer", "text": "本次记录了搬家城市。"},
               {"action": "ack", "text": "收到。"}]
    if delegate:
        actions += [{"action": "memory", "query": "搬家 城市"}]
    actions += [{"action": "answer", "text": REPLY}]
    runtime, session, memory, mirror, transport, worker, stages = build(
        tmp_path, sealed_capture, actions)
    try:
        first, fact, queued = seed_and_queue(runtime, session, memory)
        previous_queue = {r["source"]: (r["status"], r["attempts"])
                          for r in memory.store.background_sources(memory.companion)}
        before = len(stages), len(worker.worker_calls)
        event = send(runtime, session, QUERY)
        prepared = stages[before[0]:]
        assert len(prepared) == (2 if delegate else 1)
        for index, stage in enumerate(prepared):
            data = json.loads(stage.request.context.payload)
            assert data["memory_status"] == "pending_source"
            assert fact in data["memories"]
            assert first["basis"]["input_id"] in {r["id"] for r in data["history"]}
            assert queued["basis"]["input_id"] in {r["id"] for r in data["history"]}
            assert data["source_processing"]["source_id"] == event["basis"]["input_id"]
            assert data["source_processing"]["source_sha256"] == sha256(QUERY.encode()).hexdigest()
            assert data["source_processing"]["status"] == ("completed" if index else "queued")
            assert data["memory_access"] == ("completed" if index else "not_accessed")
            assert SEED in str(stage.messages)
        envelope = transport.sealed[-1]
        assert json.loads(envelope["payload"]["context"])["memories"] == (
            json.loads(prepared[-1].request.context.payload)["memories"])
        assert any(s["kind"] == "memory" and s["source_id"] == fact["source_id"]
                   and s["revision"] == fact["version"] and s["text"] == SEED
                   for s in envelope["payload"]["personal_sources"])
        assert event["readset"]["facts"][0]["record_id"] == fact["record_id"]
        assert event["display_text"] == REPLY
        assert mirror.history()["messages"][-1]["content"] == REPLY
        remaining = {r["source"]: (r["status"], r["attempts"])
                     for r in memory.store.background_sources(memory.companion)}
        assert all(remaining[source] == state for source, state in previous_queue.items())
        delegated_sources = {
            packet["source"]["id"] for _, packet in worker.worker_calls[before[1]:]
            if isinstance(packet.get("source"), dict)
        }
        assert delegated_sources == ({event["basis"]["input_id"]} if delegate else set())
        record("city-" + str(delegate), runtime, transport, event=event,
               prepared_contexts=[json.loads(s.request.context.payload) for s in prepared],
               queue_before=previous_queue, queue_after=remaining)
    finally:
        memory.close()
        mirror.close()


@pytest.mark.parametrize("user", [
    "请把“记得备份”改成更口语的提醒。", "按以前的话题聊两句。", "请记住我刚才的要求。",
])
def test_same_generator_action_is_not_replaced_by_host_string_routing(
    tmp_path, sealed_capture, user
):
    runtime, session, memory, mirror, transport, worker, _ = build(
        tmp_path, sealed_capture, [{"action": "answer", "text": "出门前，备份一下文件。"}])
    try:
        event = send(runtime, session, user)
        assert [row["stage"] for row in recorded_calls(transport)] == [
            "demand_control", "stream_unit_review"]
        assert not worker.worker_calls
        assert event["display_text"] == "出门前，备份一下文件。"
        assert json.loads(transport.sealed[0]["payload"]["context"])["memory_access"] == (
            "not_accessed")
        record("action-" + sha256(user.encode()).hexdigest()[:8], runtime, transport, event=event)
    finally:
        memory.close()
        mirror.close()


@pytest.mark.parametrize("outcome", ["clarification", "unresolved", "unknown"])
def test_one_delegation_reports_actual_noncompletion_without_rebase_or_erasing_committed_evidence(
    tmp_path, sealed_capture, outcome
):
    runtime, session, memory, mirror, transport, worker, stages = build(
        tmp_path, sealed_capture,
        [{"action": "memory"}, {"action": "answer", "text": "本次记录了搬家城市。"},
         {"action": "ack", "text": "收到。"}, {"action": "memory"},
         {"action": "answer", "text": "当前记忆操作尚未确认。"}])
    try:
        _, fact, _ = seed_and_queue(runtime, session, memory)
        original = worker.invoke

        def invoke(stage, messages, schema, cancelled):
            if stage == "memory_command" and outcome == "clarification":
                text = json.loads(messages[-1]["content"])["source"]
                return {"action": "unsupported", "requestedScope": "none", "targetIds": [],
                        "evidenceQuote": text, "quoteStart": 0, "quoteEnd": len(text),
                        "replacementQuote": "", "replacementStart": None,
                        "replacementEnd": None, "coverage": "not_command", "unresolved": ["intent"]}
            if stage == "fact_extract" and outcome == "unresolved":
                raise FactSupportUnresolved("declared CPU extraction unresolved")
            return original(stage, messages, schema, cancelled)

        memory.scheduler.user.invoke = invoke
        if outcome == "unknown":
            memory.scheduler.user.run_one = lambda *args, **kwargs: None
        start = len(stages)
        session.send("那次搬家情况呢？")
        if outcome == "unknown":
            wait_for(lambda: not session.snapshot()["busy"])
            assert session.snapshot()["attempt"]["status"] == "failed"
            assert len(stages) == start + 1
            assert not memory.store.segments(memory.companion, runtime.claim.input_id)
            assert mirror.history()["messages"][-1]["role"] == "user"
            return
        event = finish(runtime, session)
        initial, final = stages[start:]
        assert initial.request.context.state_version == final.request.context.state_version
        data = json.loads(final.request.context.payload)
        assert data["memory_access"] == (
            "needs_clarification" if outcome == "clarification" else "unresolved")
        assert data["source_processing"]["status"] == "failed"
        assert data["source_processing"]["error"].startswith("needs_clarification.")
        assert data["memory_status"] == "needs_clarification"
        assert fact in data["memories"] and data["history"]
        assert "current_operation" not in data
        assert fact["record_id"] in {r["record_id"] for r in event["readset"]["facts"]}
        assert "当前输入已有有效的已保存事实记录" not in final.request.operation_constraints()
        record(outcome, runtime, transport, event=event, final_context=data)
    finally:
        memory.close()
        mirror.close()


@pytest.mark.parametrize("change", ["correct", "forget", "expiry"])
def test_real_head_changes_and_dependent_ack_stay_out_under_old_pending(
    tmp_path, sealed_capture, change
):
    actions = [{"action": "memory"}, {"action": "answer", "text": REPLY},
               {"action": "ack", "text": "收到。"}]
    if change != "expiry":
        actions += [{"action": "memory"}, {"action": "answer", "text": "本次操作已完成。"}]
    actions += [{"action": "answer", "text": "只按当前有效来源回答。"}]
    runtime, session, memory, mirror, transport, worker, stages = build(
        tmp_path, sealed_capture, actions)
    if change == "expiry":
        worker.expiry = worker.now + timedelta(days=1)
    try:
        first, fact, _ = seed_and_queue(runtime, session, memory)
        if change != "expiry":
            send(runtime, session, worker.correct if change == "correct" else worker.forget)
        else:
            worker.now += timedelta(days=2)
        event = send(runtime, session, QUERY)
        data = json.loads(stages[-1].request.context.payload)
        ids = {r["id"] for r in data["history"]}
        assert first["basis"]["input_id"] not in ids
        assert "segment-" + first["fingerprint"] not in ids
        assert not any(r["record_id"] == fact["record_id"] and r["version"] == 1
                       for r in event["readset"]["facts"])
        if change == "correct":
            current, = data["memories"]
            assert current["record_id"] == fact["record_id"] and current["version"] == 2
            assert current["content"] == worker.updated
        else:
            assert data["memories"] == []
        assert memory.store.memory_status(memory.companion) == (
            "pending_source" if change == "expiry" else "needs_clarification")
        record("head-" + change, runtime, transport, event=event, final_context=data)
    finally:
        memory.close()
        mirror.close()


def test_new_metadata_is_current_source_bound_without_changing_reference_protocol(
    tmp_path, sealed_capture
):
    runtime, session, memory, mirror, _, _, stages = build(
        tmp_path, sealed_capture, [{"action": "answer", "text": "收到。"}])
    original_prepare = runtime.prepare
    checked = []
    failures = []

    def check_context(context):
        # Source metadata is checked while its runtime owner is still live.
        runtime.admit.validate(context)
        data = json.loads(context.payload)
        assert "source_processing" in data and "memory_access" in data
        readset = history_readset(context.companion, context.input_id, context.state_version,
                                  context.payload)
        assert readset["protocol"] == "history-readset.v1"
        assert readset["context_sha256"] == sha256(canonical(data).encode()).hexdigest()
        altered = deepcopy(data)
        altered["source_processing"]["source_id"] = "another-source"
        with pytest.raises(ValueError, match="current input"):
            history_readset(context.companion, context.input_id, context.state_version,
                            canonical(altered))
        altered = deepcopy(data)
        altered["source_processing"]["status"] = "completed"
        forged = ShortContext(context.companion, context.input_id, context.state_version,
                              canonical(altered))
        with pytest.raises(VersionConflict, match="processing"):
            runtime.admit.validate(forged)
        checked.append(forged)

    def check_prepared(*args, **kwargs):
        stage = original_prepare(*args, **kwargs)
        try:
            check_context(stage.request.context)
        except BaseException as error:
            # Propagate worker-thread assertion failures to this test's thread.
            failures.append(error)
        return stage

    runtime.prepare = check_prepared
    try:
        session.send("当前输入")
        wait_for(lambda: runtime.poll()["event"] or not session.snapshot()["busy"])
        assert not failures, repr(failures)
        assert session.snapshot()["busy"], repr(session.snapshot())
        finish(runtime, session)
        assert not failures, repr(failures)
        assert len(checked) == len(stages) == 1
        # A completed owner must still fail before any later context validation.
        with pytest.raises(VersionConflict, match="Segment owner"):
            runtime.admit.validate(checked[0])
    finally:
        memory.close()
        mirror.close()


def test_prompt_registry_is_unchanged_by_admission_refactor():
    legacy = {key: value for key, value in PROMPTS.items()
              if key not in {
                  "short-dialogue.v28", "short-dialogue.v29", "short-dialogue.v30",
                  "short-dialogue.v31", "short-dialogue.v32", "short-dialogue.v33",
                  "short-dialogue.v34", "short-dialogue.v35", "short-dialogue.v36"}}
    assert sha256(canonical(legacy).encode()).hexdigest() == (
        "29e6e4de5a03ef467f71f21105fdca5f741e2d5e373b778e99bc70722bb4eddc")


@pytest.mark.parametrize("action", ["更正", "遗忘"])
def test_unperformed_operation_remains_rejected_by_same_review_and_single_repair(
    tmp_path, sealed_capture, action
):
    runtime, session, memory, mirror, transport, worker, _ = build(
        tmp_path, sealed_capture,
        [{"action": "answer", "text": "这项记忆操作已经完成。"},
         {"action": "answer", "text": "这项操作还没有完成。"}])
    # Explicit test-author opinions; no host classifier certifies these words.
    transport.review_options = [{"decision": "reject", "constraints_satisfied": False}, {}]
    try:
        event = send(runtime, session, f"请{action}先前那条记忆。")
        assert not worker.worker_calls
        assert event["display_text"] == "这项操作还没有完成。"
        assert [r["stage"] for r in recorded_calls(transport)] == [
            "demand_control", "stream_unit_review", "demand_repair", "stream_unit_review"]
        assert transport.opinions[0]["decision"] == "reject"
        assert all("current_operation" not in json.loads(e["payload"]["context"])
                   for e in transport.sealed)
        assert len(memory.store.segments(memory.companion, runtime.claim.input_id)) == 1
        record("unperformed-" + action, runtime, transport, event=event)
    finally:
        memory.close()
        mirror.close()


@pytest.mark.parametrize("branch", ["initial", "repair"])
def test_successful_guard_renewal_preserves_one_memory_delegation_and_receipt_ack(
    tmp_path, sealed_capture, monkeypatch, branch
):
    from virtual_companion.text import default_runtime

    actions = [{"action": "memory"}, {"action": "answer", "text": REPLY}]
    if branch == "repair":
        actions.insert(0, {"action": "answer", "text": "这项记忆操作已经完成。"})
    runtime, session, memory, mirror, transport, worker, stages = build(
        tmp_path, sealed_capture, actions)
    if branch == "repair":
        transport.review_options = [{"decision": "reject", "constraints_satisfied": False}, {}]
    guardians, renewals, delegated = [], [], []

    class ObservedGuardian(default_runtime.TurnGuardian):
        def __enter__(self):
            entered = super().__enter__()
            guardians.append(entered)
            return entered

    monkeypatch.setattr(default_runtime, "TurnGuardian", ObservedGuardian)
    original_invoke = transport.invoke
    target_stage = "demand_control" if branch == "initial" else "demand_repair"

    def invoke(body, cancelled, **options):
        if options["stage"] == target_stage and not renewals:
            guardian = guardians[-1]
            original_claim = runtime.claim
            worker.now += timedelta(seconds=91)
            renewed = guardian.checkpoint()
            assert renewed.lease_until_us > original_claim.lease_until_us
            assert replace(renewed, lease_until_us=original_claim.lease_until_us) == original_claim
            renewals.append((original_claim, renewed))
        return original_invoke(body, cancelled, **options)

    monkeypatch.setattr(transport, "invoke", invoke)
    original_delegate = memory.scheduler.user.run_one

    def delegate(*args, **kwargs):
        delegated.append(kwargs["foreground"])
        return original_delegate(*args, **kwargs)

    monkeypatch.setattr(memory.scheduler.user, "run_one", delegate)
    original_resolve, resolve_errors = runtime.resolve, []

    def resolve(claim):
        try:
            return original_resolve(claim)
        except VersionConflict as error:
            resolve_errors.append(str(error))
            raise

    monkeypatch.setattr(runtime, "resolve", resolve)
    try:
        try:
            event = send(runtime, session, SEED)
        except AssertionError:
            if resolve_errors:
                pytest.fail("Memory delegation after successful renewal: " + str(resolve_errors),
                            pytrace=False)
            raise
        original_claim, renewed = renewals[0]
        assert delegated == [renewed]
        assert original_claim != renewed
        final = json.loads(stages[-1].request.context.payload)
        assert final["memory_access"] == "completed"
        assert final["source_processing"] == memory.store.current_source_processing(runtime.claim)
        assert final["source_processing"]["status"] == "completed"
        assert final["memories"][0]["source_id"] == event["basis"]["input_id"]
        assert runtime.claim.state_version > renewed.state_version
        with memory.store.transaction(write=False) as db:
            receipt = json.loads(db.execute(
                "SELECT w.receipt FROM background_work w JOIN background_sources s "
                "ON w.companion=s.companion AND w.job=s.job "
                "WHERE s.companion=? AND s.source=? AND s.role='user'",
                (memory.companion, renewed.input_id)).fetchone()[0])
        assert receipt["foreground_binding"] == memory.store._delegated_worker(renewed)
        assert receipt["foreground_basis"] == runtime.claim.state_version
        assert final["source_processing"]["receipt_sha256"] == sha256(
            canonical(receipt).encode()).hexdigest()
        assert event["readset"]["context_sha256"] == sha256(canonical(final).encode()).hexdigest()
        assert json.loads(transport.sealed[-1]["payload"]["context"]) == final
        assert event["display_text"] == REPLY
        assert memory.store.segments(memory.companion, renewed.input_id)[0]["status"] == "confirmed"
        assert not runtime.cancelled.is_set()
        assert all(g.stopped.is_set() and not g.observer.is_alive() for g in guardians)
        record("renewed-" + branch, runtime, transport, event=event, final_context=final,
               original_lease_until_us=original_claim.lease_until_us,
               renewed_lease_until_us=renewed.lease_until_us, delegated_receipt=receipt)
    finally:
        memory.close()
        mirror.close()


@pytest.mark.parametrize("mutation", ["owner", "epoch", "state", "config", "input", "session",
                                     "attempt", "expired", "deadline"])
def test_renewal_normalization_never_authorizes_changed_identity_or_inactive_run(
    tmp_path, sealed_capture, mutation
):
    runtime, _, memory, mirror, _, worker, _ = build(tmp_path, sealed_capture, [])
    try:
        attempt = mirror.begin("当前输入", {})
        runtime.prepare_turn(attempt, mirror.session_id, mirror.history()["messages"], Event())
        claim = runtime.claim
        altered = {
            "owner": lambda: replace(claim, owner=replace(claim.owner, owner_id="other-owner")),
            "epoch": lambda: replace(
                claim, owner=replace(claim.owner, epoch=claim.owner.epoch + 1)),
            "state": lambda: replace(claim, state_version=claim.state_version + 1),
            "config": lambda: replace(claim, config_json=claim.config_json + " "),
            "input": lambda: replace(claim, input_id="another-input"),
            "session": lambda: replace(claim, session_id="another-session"),
            "attempt": lambda: replace(claim, attempt_id="another-attempt"),
        }
        if mutation in altered:
            claim = altered[mutation]()
        else:
            worker.now += timedelta(seconds=121 if mutation == "expired" else 181)
        with pytest.raises(TimeoutError if mutation == "deadline" else VersionConflict):
            runtime.resolve(claim)
        assert not worker.worker_calls
        assert runtime._memory_access == "not_accessed"
    finally:
        memory.close()
        mirror.close()
