"""ACK-derived evidence is invalidated across turns without deleting display history."""

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from test_explicit_memory_management import list_command
from test_rebuilt_turns import A, claim, config
from test_segment_store import reviewed
from unit_review_recording import bound_unit_packets as bound_unit_packets
from unit_review_recording import recorded_review_packet
from virtual_companion.boundary.contracts import Scope, VersionConflict
from virtual_companion.boundary.memory import FactKey, MemoryMutation
from virtual_companion.boundary.sqlite_memory_management import SQLiteExplicitMemoryStore
from virtual_companion.cognition.background_worker import UserFactWorker
from virtual_companion.cognition.short_context import ShortContextAdmission
from virtual_companion.cognition.short_dialogue import ShortRequest
from virtual_companion.text.chronology import utc_microseconds
from virtual_companion.text.memory_runtime import _fake_memory_result

WHEN = datetime(2026, 9, 27, tzinfo=UTC)
NOW = utc_microseconds(WHEN)


def complete(store, item):
    UserFactWorker(store, _fake_memory_result, clock=lambda: NOW,
                   resolution_protocol=True, resolution_strategy="intent_first",
                   memory_commands=True).run_one(A, "worker", source=item.input_id)


def admission(store, item, *, now=NOW + 11):
    run = claim(store, item, now=now - 1)
    admit = ShortContextAdmission(store, lambda *args: ShortRequest(args[-1]),
                                  evidence_clock=lambda: now)
    return run, admit(run, store.snapshot(A), store.inspect(A, item.input_id)["text"])


def emit(store, item, text, *, legacy=False):
    run, request = admission(store, item)
    kwargs = {}
    if not legacy:
        from virtual_companion.boundary.history_readset import history_readset

        context = request.context
        kwargs["readset"] = history_readset(context.companion, context.input_id,
                                            context.state_version, context.payload)
    saved = store.approve_segment(run, reviewed(run, (text,)), now_us=NOW + 11, **kwargs)
    event = store.dispatch_segment(run, 1, now_us=NOW + 12)
    store.confirm_segment(A, item.input_id, 1, fingerprint=event["fingerprint"],
                          host_receipt=item.input_id, channel="text_visible")
    store.finish_stream(run, outcome="completed", now_us=NOW + 13)
    return saved


def seeded(tmp_path, *, legacy=False, expires=False):
    store = SQLiteExplicitMemoryStore(tmp_path / "history.sqlite3")
    original = store.accept(config(), "original", "我喜欢安静的音乐")
    complete(store, original)
    store.cancel(A, original.input_id)
    seed = MemoryMutation(Scope(A, original.session_id), "seed", "assert_fact", "music",
                          "我喜欢安静的音乐", "manual", original.input_id, "synthetic", WHEN,
                          FactKey("user", "music", "profile"),
                          expires_at=WHEN + timedelta(seconds=1) if expires else None)
    owner = store.owner(A) or store.claim_owner(A, "seed-owner", expected_epoch=0)
    store.mutate(seed, owner, expected_version=store.snapshot(A).write_version)
    query = store.accept(config(), "recall", "我喜欢什么音乐？", session_id=original.session_id)
    complete(store, query)
    emit(store, query, "你喜欢安静的音乐。", legacy=legacy)
    return store, original, query, seed


@pytest.mark.parametrize("change", ["correct", "forget", "expiry"])
@pytest.mark.parametrize("legacy", [False, True])
def test_cross_turn_ack_is_not_effective_after_fact_change(tmp_path, change, legacy):
    store, original, query, seed = seeded(tmp_path, legacy=legacy, expires=change == "expiry")
    if change == "correct":
        store.mutate(replace(seed, mutation_id="change", kind="correct", content="我喜欢热闹的音乐",
                             source_id="new-source", expected_record_version=1, fact_key=None),
                     store.owner(A), expected_version=store.snapshot(A).write_version)
    elif change == "forget":
        store.execute_memory_command(list_command(), now_us=NOW + 20)
    store = SQLiteExplicitMemoryStore(store.path)
    current = store.accept(config(), "next", "聊聊音乐吧", session_id=original.session_id)
    complete(store, current)
    _, request = admission(store, current, now=NOW + 2_000_000)
    history = json.loads(request.context.payload)["history"]
    assert not any("安静" in row["text"] for row in history)
    assert store.segments(A, query.input_id)[0]["text"] == "你喜欢安静的音乐。"


def test_second_generation_history_inherits_first_ack_dependencies(tmp_path):
    store, original, _, _ = seeded(tmp_path)
    derived = store.accept(config(), "derived", "hello", session_id=original.session_id)
    complete(store, derived)
    saved = emit(store, derived, "记得，你偏好安静。　")
    assert saved["readset"]["facts"] == []
    store.execute_memory_command(list_command(), now_us=NOW + 20)
    current = store.accept(config(), "next", "hello again", session_id=original.session_id)
    complete(store, current)
    _, request = admission(store, current)
    assert not any("安静" in row["text"] for row in json.loads(request.context.payload)["history"])


def test_pending_source_hides_fact_history_but_keeps_current_request_and_unrelated_user(tmp_path):
    # The directory remains available; the latest correction still prohibits
    # answering with the old value. The recorded opinion proves wiring only.
    from threading import Event

    from virtual_companion.cognition.stream_unit_review import (
        BoundedUnitReviewer,
        CandidateRejected,
        StructuredUnitAssessor,
        semantic_review_view,
    )
    from virtual_companion.text.fake import synthetic_unit_opinion

    store, original, _, _ = seeded(tmp_path)
    safe = store.accept(config(), "safe", "聊聊电影", session_id=original.session_id)
    complete(store, safe)
    store.cancel(A, safe.input_id)
    text = "安静的音乐那句话有误，先不要用旧偏好"
    current = store.accept(config(), "pending", text, session_id=original.session_id)
    run, request = admission(store, current)
    data = json.loads(request.context.payload)
    assert any("安静" in row["text"] for row in data["history"])
    assert any(row["id"] == safe.input_id for row in data["history"])
    assert request.model_request(192)["messages"][-1]["content"] == text
    assert data["memories"] and data["memory_status"] == "pending_source"
    assert data["source_processing"]["status"] == "queued"
    assert store.current_memory_operation(run) is None
    calls = []

    def invoke(stage, messages, schema, cancelled):
        assert stage == "stream_unit_review"
        packet = recorded_review_packet(messages)
        assert semantic_review_view(packet) == json.loads(messages[-1]["content"])
        assert packet["user_text"] == text and packet["context"] == request.context.payload
        assert packet["basis"]["input_id"] == current.input_id
        assert packet["candidate_prefix"] == "你仍喜欢安静的音乐。"
        assert any(source["kind"] == "current" and source["text"] == text
                   for source in packet["personal_sources"])
        assert any(source["kind"] == "memory" and source["text"] == "我喜欢安静的音乐"
                   for source in packet["personal_sources"])
        calls.append(packet)
        return synthetic_unit_opinion(packet, decision="reject", constraints_satisfied=False)

    reviewer = BoundedUnitReviewer(run, request, StructuredUnitAssessor(invoke, Event()))
    with pytest.raises(CandidateRejected) as rejected:
        reviewer("你仍喜欢安静的音乐。", True)
    assert rejected.value.verdict["decision"] == "reject"
    assert rejected.value.verdict["constraints_satisfied"] is False
    assert len(calls) == 1 and reviewer.approved_text == ""
    assert store.segments(A, current.input_id) == []
    assert store.memory_barrier(A)


def test_unrelated_pending_keeps_committed_evidence_and_original_source(tmp_path):
    store, original, _, _ = seeded(tmp_path)
    current = store.accept(config(), "pending-unrelated", "聊聊电影",
                           session_id=original.session_id)
    _, request = admission(store, current)
    data = json.loads(request.context.payload)
    assert data["memory_status"] == "pending_source"
    assert any(row["id"] == original.input_id for row in data["history"])
    assert any("安静" in row["text"] for row in data["history"])
    assert any(source["kind"] == "history" and source["source_id"] == original.input_id
               and source["text"] == "我喜欢安静的音乐" for source in request.personal_sources(
                   A, current.input_id, request.context.state_version))


def test_background_projection_filters_expired_history_but_keeps_processing_source(tmp_path):
    store, original, _, _ = seeded(tmp_path, expires=True)
    current = store.accept(config(), "worker-next", "安静的音乐是刚才的说法",
                           session_id=original.session_id)
    work = store.claim_background(A, "worker", now_us=NOW + 2_000_000,
                                  lease_us=100, role="user", source=current.input_id)
    sources = store.background_exchange_context(work, now_us=NOW + 2_000_001)
    assert sources[-1].source_id == current.input_id
    assert sources[-1].text == "安静的音乐是刚才的说法"
    assert not any("安静" in row.text for row in sources[:-1])


def test_live_readset_is_validated_before_approval(tmp_path):
    from virtual_companion.boundary.history_readset import history_readset

    store, original, _, _ = seeded(tmp_path)
    item = store.accept(config(), "next", "我喜欢什么音乐", session_id=original.session_id)
    complete(store, item)
    run, request = admission(store, item)
    data = json.loads(request.context.payload)
    data["memories"][0]["content"] = "伪造的有效记忆"
    readset = history_readset(A, item.input_id, run.state_version, json.dumps(data))
    with pytest.raises(VersionConflict, match="readset"):
        store.approve_segment(run, reviewed(run), now_us=NOW + 11, readset=readset)
    assert store.segments(A, item.input_id) == []


def test_expiry_between_approval_and_dispatch_does_not_publish(tmp_path):
    from virtual_companion.boundary.history_readset import history_readset

    store, original, _, _ = seeded(tmp_path, expires=True)
    item = store.accept(config(), "next", "我喜欢什么音乐", session_id=original.session_id)
    complete(store, item)
    run, request = admission(store, item)
    run = store.renew(run, now_us=NOW + 12, lease_us=3_000_000)
    readset = history_readset(A, item.input_id, run.state_version, request.context.payload)
    store.approve_segment(run, reviewed(run), now_us=NOW + 13, readset=readset)
    with pytest.raises(VersionConflict, match="readset"):
        store.dispatch_segment(run, 1, now_us=NOW + 2_000_000)
    assert store.segments(A, item.input_id)[0]["status"] == "pending"


@pytest.mark.parametrize("field,value", [("protocol", "history-readset.v999"),
                                        ("state_version", True), ("facts", "verified")])
def test_readset_does_not_accept_version_or_verified_flags(tmp_path, field, value):
    from virtual_companion.boundary.history_readset import history_readset, validate_readset

    store, original, _, _ = seeded(tmp_path)
    item = store.accept(config(), "next", "hello", session_id=original.session_id)
    complete(store, item)
    run, request = admission(store, item)
    data = history_readset(A, item.input_id, run.state_version, request.context.payload)
    data[field] = value
    with pytest.raises(ValueError):
        validate_readset(data)


def test_real_stream_coordinator_binds_the_context_it_sent(tmp_path):
    from threading import Event

    from virtual_companion.cognition.stream_dialogue import StreamDialogue
    from virtual_companion.providers.text_deltas import UnapprovedTextDelta

    store, original, _, _ = seeded(tmp_path)
    item = store.accept(config(), "next", "我喜欢什么音乐", session_id=original.session_id)
    complete(store, item)
    run, request = admission(store, item)
    sent = []

    class Model:
        def count(self, body):
            sent.append(body)
            return 10

        def invoke(self, body):
            assert body == sent[0]
            self.receive(UnapprovedTextDelta(1, "我在这里。"))
            return {"text": "我在这里。", "tool_calls": [], "finish_reason": "stop"}

        def cancel(self):
            pass

    model = Model()

    def factory(receive):
        model.receive = receive
        return model

    coordinator = StreamDialogue(store=store, model_factory=factory,
        review_prefix=lambda text, final: reviewed(run), present=lambda event: "actual-host",
        checkpoint=lambda: None, cancelled=Event(), clock=lambda: NOW + 11)
    assert coordinator.generate(run, request)["outcome"] == "completed"
    saved = store.segments(A, item.input_id)[0]
    assert saved["readset"]["facts"][0]["record_id"] == "music"
    assert "安静" in sent[0]["messages"][1]["content"]


def test_foreign_companion_readset_is_rejected(tmp_path):
    from test_rebuilt_turns import B
    from virtual_companion.boundary.history_readset import history_readset

    store, original, _, _ = seeded(tmp_path)
    item = store.accept(config(), "next", "hello", session_id=original.session_id)
    complete(store, item)
    run, request = admission(store, item)
    readset = history_readset(B, item.input_id, run.state_version, request.context.payload)
    with pytest.raises(VersionConflict, match="readset"):
        store.approve_segment(run, reviewed(run), now_us=NOW + 11, readset=readset)


def test_background_commit_rechecks_expired_source_context(tmp_path):
    from threading import Event

    from virtual_companion.cognition.exchange_model import ExchangeModel

    store, _, query, _ = seeded(tmp_path, expires=True)
    source = "segment-" + store.segments(A, query.input_id)[0]["fingerprint"]
    work = store.claim_background(A, "worker", now_us=NOW + 20, lease_us=3_000_000,
                                  role="assistant", source=source)
    sources = store.background_exchange_context(work, now_us=NOW + 21)
    assert sources[-1].source_id == source
    proposals, resolution = ExchangeModel(_fake_memory_result, Event()).resolve_source(
        scope=Scope(A, work.session), batch_id=source, now_us=NOW + 21,
        turns=sources, targets={}, eligible_source_ids=(source,), strategy="intent_first")
    with pytest.raises(VersionConflict, match="context changed"):
        store.finish_assistant_background(work, proposals, sources, evaluator=resolution.evaluator,
                                           now_us=NOW + 2_000_000, resolution=resolution)


def test_proven_unrelated_assistant_is_kept_under_other_pending_source(tmp_path):
    store, original, _, _ = seeded(tmp_path)
    independent = store.accept(config(), "independent", "hello")
    complete(store, independent)
    saved = emit(store, independent, "你好。")
    assert saved["readset"]["facts"] == saved["readset"]["history"] == []
    store.accept(config(), "pending", "刚才偏好有误", session_id=original.session_id)
    assert "segment-" + saved["fingerprint"] not in store.ineffective_history_sources(A, now_us=NOW)


@pytest.mark.parametrize("route", ["foreground", "worker"])
def test_store_injected_clock_governs_projection_without_explicit_now(tmp_path, route):
    store, original, query, _ = seeded(tmp_path, expires=True)
    clock = [NOW + 20]
    store = SQLiteExplicitMemoryStore(store.path, evidence_clock=lambda: clock[0])
    source = "segment-" + store.segments(A, query.input_id)[0]["fingerprint"]
    if route == "foreground":
        current = store.accept(config(), "next", "hello", session_id=original.session_id)
        complete(store, current)
        run = claim(store, current, now=NOW + 20)
        before, _ = store.exchange_context(run)
        assert source in {turn.source_id for turn in before}
        clock[0] = NOW + 2_000_000
        after, _ = store.exchange_context(run)
        assert source not in {turn.source_id for turn in after}
    else:
        work = store.claim_background(A, "worker", now_us=NOW + 20,
                                      lease_us=3_000_000, role="assistant", source=source)
        assert store.background_exchange_context(work)[-1].source_id == source
        clock[0] = NOW + 2_000_000
        with pytest.raises(VersionConflict, match="context changed"):
            store.background_exchange_context(work)
