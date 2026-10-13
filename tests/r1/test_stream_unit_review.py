import json
from copy import deepcopy
from threading import Event

import pytest
from test_rebuilt_turns import A, claim, config
from unit_review_recording import bound_unit_packets as bound_unit_packets
from unit_review_recording import recorded_review_packet
from virtual_companion.boundary.contracts import canonical
from virtual_companion.boundary.presentation import digest
from virtual_companion.boundary.sqlite_streaming import SQLiteStreamingStore
from virtual_companion.cognition.short_dialogue import ShortRequest
from virtual_companion.cognition.stream_dialogue import StreamDialogue
from virtual_companion.cognition.stream_unit_review import candidate_segments
from virtual_companion.providers.text_deltas import UnapprovedTextDelta

S01 = "今天雨好大，我忽然想起小时候放学的路。"
S05 = "我想去散步但在下雨，请给我一个简单安排。"
# Frozen v21 candidate and early opinion from the synthetic DeepSeek diagnosis.
# Receipt SHA256s: s-01 e99ccf73cb71bffa349ca7c5b67aea9146eb087bafa2c405473bd8a12187df47;
# s-08 9979bbc1b59b27f92ef71c15f5366769631d4f64fd4947e4c9ff52c65c37cb8c.
V21_RAIN = (
    "雨大到让记忆翻出放学那条路，往往是因为雨声、踩水的触感和当年放学路上的某种节奏对上了。"
)
V21_TICKETS_INPUT = "刚整理完桌上的一堆旧票根，想聊聊看到它们的感觉。"
V21_TICKETS_PREFIX = (
    "看到旧票根时，那种感觉像是把散落的日期又摆回了眼前——纸片上的地点、座位和数字还在，"
    "但当时看完的心情，只有你自己清楚。"
)
V21_APPROVE = dict(decision="approve", self_contained=True, no_open_dependency=True,
                   consistent_with_displayed=True, constraints_satisfied=True)
V22_TICKETS = "票根攒到能堆成一堆，翻的时候大概每张都对应着一次出门。"
V22_QUIET = "那种安静是待在房间里才明显，还是从外面回来一进门时更明显？"


def audit_opinion(packet, *, mode="nonpersonal", support="not_applicable", refs=(), **checks):
    """Recorded meanings for historical fixtures, not a semantic classifier."""
    personal = mode in {"personal_fact", "fiction"}
    return {**V21_APPROVE, **checks, "audit_version": "candidate-propositions.v3",
            "segments": [{"segment_id": item["segment_id"], "commitments": [
                {"kind": "assertion" if mode == "personal_fact" else mode,
                 "claim": item["text"], "support": support, "sources": list(refs)}
            ] if personal or support != "not_applicable" or refs else [], "questions": []}
                         for item in packet["review_segments"]]}


def segment_payload(text):
    return [{"segment_id": item["segment_id"], "text": item["text"]}
            for item in candidate_segments(text)]


def relation_wire(propositions):
    return [{key: item[key] for key in ("segment_id", "commitments", "questions")}
            for item in propositions]


def personal_ref(source, quote):
    return {key: source[key] for key in ("id", "owner", "source_id", "role", "revision")} | {
        "quote": quote}


@pytest.mark.parametrize("text,support", [(V22_TICKETS, "unsupported"),
                                         (V22_QUIET, "not_applicable")])
def test_v22_review_wire_delegates_only_semantics_to_model(tmp_path, text, support):
    # Frozen s-07 raw reject had end=31 for 27 codepoints; s-03 approve had
    # end=31/34 for 29 codepoints. Both failed before semantic handling in v1.
    # Receipts: 1e68661552b6d965100534353a67934bbc50903b4ede019ebd6ebec220162ff8
    # and ba76f74ba61b61b8c6e7e1214538ca70901ad3c75dc21dffa5a4b6b707dc9713.
    from virtual_companion.cognition.stream_unit_review import (
        BoundedUnitReviewer,
        StructuredUnitAssessor,
    )
    from virtual_companion.text.turn_orchestrator import CandidateRejected

    store = SQLiteStreamingStore(tmp_path / "v22-semantic-only.sqlite3")
    run = claim(store, store.accept(config(), "v22-input", V21_TICKETS_INPUT))
    seen = []

    def invoke(stage, messages, schema, cancelled):
        packet = recorded_review_packet(messages)
        seen.append(packet)
        assert "start" not in schema["properties"]["segments"]["items"]["properties"]
        assert "end" not in schema["properties"]["segments"]["items"]["properties"]
        assert "text" not in schema["properties"]["segments"]["items"]["properties"]
        assert "".join(row["text"] for row in packet["review_segments"]) == text
        return audit_opinion(packet, mode="personal_fact" if support == "unsupported"
                             else "nonpersonal", support=support)

    assessor = StructuredUnitAssessor(invoke, Event())

    def assess(packet):
        verdict = assessor(packet)
        if verdict["decision"] == "reject":
            raise CandidateRejected(verdict)
        return verdict

    reviewer = BoundedUnitReviewer(run, ShortRequest(V21_TICKETS_INPUT), assess)
    if support == "unsupported":
        with pytest.raises(CandidateRejected) as rejected:
            reviewer(text, True)
        result = rejected.value.verdict
        assert result["decision"] == "reject" and result["model_opinion"] == V21_APPROVE
        assert result["propositions"][0]["start"] == 0
        assert result["propositions"][-1]["end"] == len(text)
        assert not reviewer.approved_text
    else:
        assert reviewer(text, True).candidate.text == text
    assert len(seen) == reviewer.final_calls == 1


# Verbatim synthetic failures from remote-ded72a81 and local-cedcba96;
# their history/memory/exchange seeds were all empty.
UNSOURCED_REPLIES = [
    (S01, "雨声一大，那种放学路上的记忆就自己冒出来了——踩着水洼，"
     "书包带子滑到胳膊肘，空气里全是湿土和树叶的味道。"),
    (S01, "雨声像小时候放学时的铃声，总让人想起那条湿漉漉的路。"),
    (S05, "那就撑着伞在楼下或小区里走一小圈，听听雨声，十分钟左右就回。"),
]


def replay_v21_boolean_approval(tmp_path, user_text, prefix, *, suffix="", final_hold=False,
                               mode="nonpersonal", support="not_applicable", quote=None):
    """Actual default owner/review/delivery/SQLite/ACK; supplier bytes are recorded.

    Both bad candidates and controls retain the original all-true early opinion.
    Their explicit recorded proposition labels test host enforcement, not model
    accuracy. The producer completes after its first real review, independently of
    approval, so a rejection cannot be hidden by an ACK-dependent supplier deadlock.
    Electron DOM/visibility is checked separately; this receipt is a synthetic host.
    """
    from test_default_text_delivery import Model, wait_for
    from test_default_text_delivery import setup as default_setup
    from virtual_companion.text.application import TextSession
    from virtual_companion.text.default_runtime import DefaultTextRuntime

    packets, opinions = [], []
    released, terminal = Event(), Event()

    class RecordedV21(Model):
        def review_peer(self):
            class Peer(Model):
                def stream(self, messages, cancelled):
                    packet = recorded_review_packet(messages)
                    packets.append(packet)
                    refs = ([] if mode not in {"personal_fact", "fiction"} else [personal_ref(
                        packet["personal_sources"][0], quote or user_text)])
                    opinion = audit_opinion(packet, mode=mode, support=support, refs=refs)
                    if suffix and packet["candidate_prefix"] == prefix + suffix:
                        opinion["segments"][-1].update(
                            commitments=[], questions=[suffix])
                    if final_hold and packet["final"]:
                        opinion.update(decision="hold", self_contained=False,
                                       no_open_dependency=False)
                    opinions.append(deepcopy(opinion))
                    yield json.dumps(opinion)
                    released.set()
            return Peer()

        def stream(self, messages, cancelled):
            self.calls.append(deepcopy(messages))
            try:
                yield json.dumps({"action": "answer", "text": prefix},
                                 ensure_ascii=False, separators=(",", ":"))[:-2]
                while not released.wait(0.01):
                    if cancelled.is_set():
                        raise InterruptedError("Recorded v21 producer cancelled")
                if cancelled.is_set():
                    raise InterruptedError("Recorded v21 producer cancelled")
                yield json.dumps(suffix, ensure_ascii=False)[1:] + "}"
            finally:
                terminal.set()

        def cancel(self):
            released.set()

    model = RecordedV21()
    _, _, memory, mirror, _ = default_setup(tmp_path, model)
    runtime = DefaultTextRuntime(memory, mirror, model, binding={"test": "synthetic-v21"},
                                 delivery_timeout=0.8, prompt_revision="short-dialogue.v20")
    session = TextSession(model, mirror, {}, prepare_turn=runtime.prepare_turn,
                          before_start=runtime.before_start)
    try:
        session.send(user_text)
        cutpoint = wait_for(lambda: runtime.poll()["event"] or
                            (not session.snapshot()["busy"] and {"blocked": True}))
        published, ack_status = [], []
        active_at_ack = False
        if not cutpoint.get("blocked"):
            published.append(cutpoint["display_text"])
            active_at_ack = not terminal.is_set()
            result = runtime.ack(dict(sequence=cutpoint["sequence"],
                fingerprint=cutpoint["fingerprint"], host_receipt="synthetic-v21-visible"))
            ack_status.append(result["status"])
        released.set()
        wait_for(lambda: not session.snapshot()["busy"])
        assert packets and packets[0]["user_text"] == user_text
        assert packets[0]["candidate_prefix"] == prefix and not packets[0]["final"]
        assert not packets[0]["displayed_prefix"]
        state = memory.store.stream_state(memory.companion, runtime.claim.input_id)
        assistants = [row["content"] for row in mirror.history()["messages"]
                      if row["role"] == "assistant"]
        return {
            "delivery": {"approved_texts": published, "confirmed_units": state["confirmed_units"],
                         "confirmed_text": state["confirmed_text"], "ack_status": ack_status,
                         "assistant_history": assistants},
            "active_at_ack": active_at_ack, "packets": packets,
            "opinions": opinions, "generation_calls": len(model.calls),
            "attempt_status": session.snapshot()["attempt"]["status"],
        }
    finally:
        released.set()
        memory.close()
        mirror.close()


@pytest.mark.parametrize("user_text,prefix,suffix,final_hold,quote", [
    (S01, V21_RAIN, "", False, "小时候放学的路"),
    (V21_TICKETS_INPUT, V21_TICKETS_PREFIX,
     "你想说说看到它们时具体是什么感觉吗？", True, "旧票根"),
], ids=["s01-entire-answer", "s08-early-prefix-before-final-hold"])
def test_v21_boolean_approval_must_not_publish_unsourced_personal_details(
    tmp_path, user_text, prefix, suffix, final_hold, quote,
):
    observed = replay_v21_boolean_approval(tmp_path, user_text, prefix,
                                         suffix=suffix, final_hold=final_hold,
                                         mode="personal_fact", support="unsupported", quote=quote)
    assert observed["delivery"] == {
        "approved_texts": [], "confirmed_units": 0, "confirmed_text": "",
        "ack_status": [], "assistant_history": [],
    }
    assert observed["attempt_status"] == "failed"
    assert observed["packets"][0]["personal_sources"][0]["text"] == user_text
    assert observed["opinions"][0]["decision"] == "approve"
    assert observed["opinions"][0]["constraints_satisfied"] is True
    assert observed["opinions"][0]["segments"][0]["commitments"][0]["support"] == "unsupported"
    assert observed["generation_calls"] == 2  # Existing single repair, same supplier bytes.


@pytest.mark.parametrize("user_text,prefix,mode,support", [
    ("这次大雨让我想起小时候放学的路，原因正是雨声、踩水的触感和当年放学路上的某种节奏对上了。",
     V21_RAIN, "personal_fact", "supported"),
    ("请写一个虚构的雨天放学片段。", "虚构一幕：小孩踩过水洼，闻到湿土的气味。",
     "fiction", "supported"),
    ("下雨天出门，有什么建议？", "出门前看看降雨和路况，绕开有积水的路段。",
     "nonpersonal", "not_applicable"),
    (S01, "雨声像一只翻开旧相册的手。", "metaphor", "not_applicable"),
    (S01, "一场大雨，把现在和那条放学的路连在了一起。", "personal_fact", "supported"),
], ids=["explicit-current-source", "requested-fiction", "general-advice", "metaphor", "reflection"])
def test_v21_source_and_nonpersonal_controls_use_the_same_delivery_chain(
    tmp_path, user_text, prefix, mode, support,
):
    observed = replay_v21_boolean_approval(tmp_path, user_text, prefix, mode=mode, support=support)
    assert observed["delivery"] == {
        "approved_texts": [prefix], "confirmed_units": 1, "confirmed_text": prefix,
        "ack_status": ["confirmed"], "assistant_history": [prefix],
    }
    assert observed["attempt_status"] == "completed"
    assert observed["generation_calls"] == 1


def test_ack_review_wire_and_candidate_fingerprint_preserve_host_type(tmp_path):
    from virtual_companion.cognition.stream_unit_review import (
        BoundedUnitReviewer,
        StructuredUnitAssessor,
    )

    store = SQLiteStreamingStore(tmp_path / "ack-review.sqlite3")
    run = claim(store, store.accept(config(), "ack", "我还没说完。"))
    seen = []

    def invoke(stage, messages, schema, cancelled):
        packet = recorded_review_packet(messages)
        seen.append(packet)
        assert packet["response_type"] == "ack" and packet["candidate_prefix"] == "我在听。"
        return audit_opinion(packet)

    reviewer = BoundedUnitReviewer(run, ShortRequest("我还没说完。"),
        StructuredUnitAssessor(invoke, Event()), response_type="ack")
    bound = reviewer("我在听。", True)
    assert bound.candidate.response_type == "ack"
    assert bound.candidate.fingerprint == seen[0]["candidate_sha256"]
    assert bound.review.candidate_sha256 == bound.candidate.fingerprint


def source_contract(system):
    # Assert the general policy on the actual request, not an expected answer.
    assert "用户个人事实" in system
    assert "user_text" in system and "context" in system
    assert "感官细节" in system and "情绪归因" in system
    assert "constraints_satisfied=false" in system and "decision=reject" in system
    assert "一般隐喻" in system and "明确虚构" in system


@pytest.mark.parametrize("user_text,reply,background,supported,mode", [
    *((user, reply, None, False, "personal_fact") for user, reply in UNSOURCED_REPLIES),
    ("房间很安静，我想聊聊这种感觉。", "你想念以前陪你住的家人，所以才孤独。",
     None, False, "personal_fact"),
    ("小时候放学踩过水洼，书包带滑到胳膊肘，闻到了湿土和树叶的味道。",
     "那段放学记忆里，有水洼、滑落的书包带，以及湿土和树叶的气味。", None, True, "personal_fact"),
    (S01, "你提过放学经过水洼，也闻到过湿土的味道。",
     {"memory_status": "current", "history": [], "exchanges": [], "memories": [{
         "record_id": "rain-fact", "version": 2, "status": "active", "source_kind": "input",
         "source_id": "past-rain", "content": "小时候放学经过水洼，闻到湿土的味道。"}]},
     True, "personal_fact"),
    (S01, "雨声像一只翻开旧相册的手。", None, True, "metaphor"),
    ("请写一个虚构的雨天放学片段。", "虚构一幕：小孩踩过水洼，闻到湿土的气味。",
     None, True, "fiction"),
    (S05, "如果雨小且路面安全，就沿平整路线散步十分钟再回来；否则等天气安全后再散步。",
     None, True, "hypothesis"),
    (S05, "如果附近有带顶棚的步道，就走十分钟；没有就先在室内活动。", None, False, "nonpersonal"),
], ids=["remote-s01", "local-s01", "remote-s05", "emotion-attribution", "user-source",
        "confirmed-background", "general-metaphor", "explicit-fiction", "conditional-proposal",
        "substitute-activity"])
def test_personal_source_contract_reaches_bound_review_without_rejecting_normal_examples(
    tmp_path, user_text, reply, background, supported, mode,
):
    from virtual_companion.cognition.short_dialogue import ShortContext
    from virtual_companion.cognition.stream_unit_review import (
        BoundedUnitReviewer,
        StructuredUnitAssessor,
    )

    store = SQLiteStreamingStore(tmp_path / "source-review.sqlite3")
    run = claim(store, store.accept(config(), "source-input", user_text))
    context = None if background is None else ShortContext(
        run.companion_id, run.input_id, run.state_version, canonical(background))
    request = ShortRequest(user_text, context=context)
    calls = []

    def invoke(stage, messages, schema, cancelled):
        calls.append((stage, deepcopy(messages)))
        source_contract(messages[0]["content"])
        payload = recorded_review_packet(messages)
        assert payload["user_text"] == user_text
        assert payload["context"] == (None if context is None else context.payload)
        assert payload["candidate_prefix"] == reply
        if user_text == S05:
            contract = payload["response_task_contract"]
            assert contract["kind"] == "activity_plan" and contract["activity"] == "散步"
            assert contract["span"] == S05 and contract["substitute_activities"] is False
            assert payload["final"] and "天气等待" in messages[0]["content"]
        # Recorded strict reviewer opinions test routing, not model accuracy.
        refs = []
        if mode in {"personal_fact", "fiction"}:
            source = payload["personal_sources"][-1 if background else 0]
            refs = [personal_ref(source, source["text"])]
        support = ("supported" if supported else "unsupported") if refs else "not_applicable"
        return audit_opinion(payload, mode=mode, support=support, refs=refs,
                             decision="approve" if supported else "reject",
                             constraints_satisfied=supported)

    reviewer = BoundedUnitReviewer(run, request, StructuredUnitAssessor(invoke, Event()))
    if supported:
        assert reviewer(reply, True).candidate.text == reply
    else:
        with pytest.raises(ValueError, match="candidate explicitly rejected"):
            reviewer(reply, True)
        assert not reviewer.approved_text
    assert len(calls) == reviewer.final_calls == 1
    assert calls[0][0] == "stream_unit_review"
    assert reviewer.early_calls == 0


@pytest.mark.parametrize("user_text,rejected_reply", UNSOURCED_REPLIES,
                         ids=["remote-s01", "local-s01", "remote-s05"])
def test_default_personal_source_rejection_uses_existing_single_repair(
    tmp_path, user_text, rejected_reply,
):
    from test_default_text_delivery import Model, wait_for
    from test_default_text_delivery import setup as default_setup

    repaired_reply = ("雨声把你带回了那条放学的路。" if user_text == S01
                      else "先等雨小些，再散步十分钟。")

    class RecordedSourceReview(Model):
        def stream(self, messages, cancelled):
            self.calls.append(deepcopy(messages))
            if "你只审查候选文字" in messages[0]["content"]:
                payload = recorded_review_packet(messages)
                supported = payload["candidate_prefix"] == repaired_reply
                personal = user_text == S01 or not supported
                refs = ([personal_ref(payload["personal_sources"][0], user_text)]
                        if personal else [])
                yield json.dumps(audit_opinion(payload,
                    mode="personal_fact" if personal else "nonpersonal",
                    support=("supported" if supported else "unsupported") if personal
                    else "not_applicable", refs=refs,
                    decision="approve" if supported else "reject", constraints_satisfied=supported))
            else:
                repair = len(self.calls) > 1
                yield json.dumps({"action": "answer",
                    "text": repaired_reply if repair else rejected_reply}, ensure_ascii=False)

    model = RecordedSourceReview()
    runtime, session, memory, mirror, _ = default_setup(tmp_path, model)
    try:
        session.send(user_text)
        event = wait_for(lambda: runtime.poll()["event"])
        assert len(model.calls) == 4  # generation/review, one existing repair/review
        initial, first_review, repair, final_review = model.calls
        source_contract(first_review[0]["content"])
        source_contract(final_review[0]["content"])
        before, after = (recorded_review_packet(call) for call in (first_review, final_review))
        assert before["user_text"] == after["user_text"] == user_text
        assert before["context"] == after["context"] and before["basis"] == after["basis"]
        assert before["candidate_prefix"] == rejected_reply and not before["displayed_prefix"]
        assert repair[:-2] == initial[:-1] and repair[-1] == initial[-1]
        assert rejected_reply in repair[-2]["content"]
        assert '"constraints_satisfied":false' in repair[-2]["content"]
        assert event["display_text"] == after["candidate_prefix"] == repaired_reply
        assert len(memory.store.segments(memory.companion, runtime.claim.input_id)) == 1
        runtime.ack(dict(sequence=event["sequence"], fingerprint=event["fingerprint"],
                         host_receipt="synthetic-source-repaired-visible"))
        wait_for(lambda: not session.snapshot()["busy"])
        assert session.snapshot()["attempt"]["status"] == "completed"
        assert mirror.history()["messages"][-1]["content"] == repaired_reply
    finally:
        memory.close()
        mirror.close()


def test_structured_assessor_binds_host_hash_without_model_echo():
    from virtual_companion.cognition.stream_unit_review import StructuredUnitAssessor

    received = []

    def invoke(stage, messages, schema, cancelled):
        received.append((stage, messages, schema, cancelled))
        return audit_opinion(recorded_review_packet(messages), mode="hypothesis",
                             decision="hold", self_contained=False,
                             no_open_dependency=False, constraints_satisfied=False)

    cancelled = Event()
    assessor = StructuredUnitAssessor(invoke, cancelled)
    payload = {"candidate_prefix": "如果你愿意，", "final": False,
               "review_segments": segment_payload("如果你愿意，")}
    packet = {"payload": payload, "sha256": digest(payload)}
    result = assessor(packet)
    assert result["decision"] == "hold"
    assert result["request_sha256"] == packet["sha256"]
    assert received[0][0] == "stream_unit_review"
    assert received[0][3] is cancelled
    assert "先独立判断" in received[0][1][0]["content"]
    assert "如果你愿意" in received[0][1][1]["content"]
    from virtual_companion.cognition.stream_unit_review import semantic_review_view

    assert received[0][1][1]["content"] == canonical(semantic_review_view(payload))
    assert packet["sha256"] not in canonical(received[0][1])
    assert received[0][2]["additionalProperties"] is False
    model_fields = {"decision", "self_contained", "no_open_dependency",
                    "consistent_with_displayed", "constraints_satisfied",
                    "audit_version", "segments"}
    assert set(received[0][2]["properties"]) == model_fields
    assert set(received[0][2]["required"]) == model_fields
    cancelled.set()
    with pytest.raises(InterruptedError):
        assessor(packet)


def test_structured_assessor_rejects_model_supplied_hash():
    from virtual_companion.cognition.stream_unit_review import StructuredUnitAssessor

    def invoke(stage, messages, schema, cancelled):
        return {
            "request_sha256": "0" * 64,
            "decision": "approve",
            "self_contained": True,
            "no_open_dependency": True,
            "consistent_with_displayed": True,
            "constraints_satisfied": True,
        }

    with pytest.raises(ValueError):
        StructuredUnitAssessor(invoke, Event())({"payload": {}, "sha256": digest({})})


def test_structured_assessor_binds_the_frozen_packet_when_only_payload_is_sent():
    from virtual_companion.cognition.stream_unit_review import StructuredUnitAssessor

    payload = {"candidate_prefix": "我在这里。", "final": False,
               "review_segments": segment_payload("我在这里。")}
    packet = {"payload": payload, "sha256": digest(payload)}
    sent_hash = packet["sha256"]
    model_reply = audit_opinion(payload)

    def invoke(stage, messages, schema, cancelled):
        from virtual_companion.cognition.stream_unit_review import semantic_review_view

        assert messages[1]["content"] == canonical(semantic_review_view(payload))
        assert sent_hash not in canonical(messages)
        # The invoker receives immutable serialized input. Caller mutation during
        # the call cannot rebind the returned opinion to a different packet.
        packet["sha256"] = "0" * 64
        return model_reply

    result = StructuredUnitAssessor(invoke, Event())(packet)
    assert result["model_opinion"] == V21_APPROVE and result["request_sha256"] == sent_hash
    assert relation_wire(result["propositions"]) == model_reply["segments"]
    assert result["propositions"][0]["text"] == payload["candidate_prefix"]
    assert "request_sha256" not in model_reply


def test_v19_task_rejection_uses_the_existing_single_repair_and_confirmed_ack(tmp_path):
    from test_default_text_delivery import Model, wait_for
    from test_default_text_delivery import setup as default_setup
    from virtual_companion.text.application import TextSession
    from virtual_companion.text.default_runtime import DefaultTextRuntime

    original = (
        "既然正在下雨，先别急着出门，在家做点别的过渡一下，比如整理下桌面或者听会儿歌，"
        "等雨停了再出去。如果雨小了、路面没积水，可以撑伞在附近走一圈，选中短途路线，"
        "穿防滑鞋，避开湿滑和积水路段。"
    )
    repaired = "等雨小且路面安全时，带伞沿平整路线散步十分钟再回来；条件不合适就等候。"
    packets = []

    class RecordedTaskReview(Model):
        def stream(self, messages, cancelled):
            self.calls.append(deepcopy(messages))
            if "你只审查候选文字" in messages[0]["content"]:
                payload = recorded_review_packet(messages)
                packets.append(payload)
                assert payload["final"] and not payload["displayed_prefix"]
                assert payload["response_task_contract"]["activity"] == "散步"
                assert "天气等待" in messages[0]["content"]
                supported = payload["candidate_prefix"] == repaired
                # Recorded opinions verify contract/repair routing, not model quality.
                yield json.dumps(audit_opinion(payload,
                    decision="approve" if supported else "reject", constraints_satisfied=supported))
            else:
                assert "short-dialogue.v19" in messages[0]["content"]
                yield json.dumps({"action": "answer",
                    "text": original if len(self.calls) == 1 else repaired}, ensure_ascii=False)

    model = RecordedTaskReview()
    _, _, memory, mirror, _ = default_setup(tmp_path, model)
    runtime = DefaultTextRuntime(memory, mirror, model, binding={"test": "synthetic"},
                                 delivery_timeout=0.8, prompt_revision="short-dialogue.v19")
    session = TextSession(model, mirror, {}, prepare_turn=runtime.prepare_turn,
                          before_start=runtime.before_start)
    try:
        session.send(S05)
        event = wait_for(lambda: runtime.poll()["event"])
        assert len(model.calls) == 4  # generation/review + existing single repair/review
        assert len(packets) == 2
        assert packets[0]["response_task_contract"] == packets[1]["response_task_contract"]
        assert packets[0]["candidate_prefix"] == original
        assert event["display_text"] == repaired and original not in event["display_text"]
        assert len(memory.store.segments(memory.companion, runtime.claim.input_id)) == 1
        runtime.ack(dict(sequence=event["sequence"], fingerprint=event["fingerprint"],
                         host_receipt="synthetic-task-repaired-visible"))
        wait_for(lambda: not session.snapshot()["busy"])
        assert session.snapshot()["attempt"]["status"] == "completed"
        assert mirror.history()["messages"][-1]["content"] == repaired
    finally:
        memory.close()
        mirror.close()


def approved(packet):
    return {
        "request_sha256": packet["sha256"],
        "decision": "approve",
        "self_contained": True,
        "no_open_dependency": True,
        "consistent_with_displayed": True,
        "constraints_satisfied": True,
    }


def setup(tmp_path, assess=approved, early_limit=1, user_text="你好"):
    from virtual_companion.cognition.stream_unit_review import BoundedUnitReviewer

    store = SQLiteStreamingStore(tmp_path / "units.sqlite3")
    run = claim(store, store.accept(config(), "first", user_text))
    calls = []

    def invoke(packet):
        calls.append(packet)
        return assess(packet)

    reviewer = BoundedUnitReviewer(run, ShortRequest(user_text), invoke, early_limit=early_limit)
    return store, run, reviewer, calls


def test_only_complete_observed_boundary_or_terminal_can_be_considered(tmp_path):
    _, _, reviewer, calls = setup(tmp_path)
    for value in ("如果", "如果你", "如果你愿意，"):
        assert reviewer(value, False) is None
    assert not calls
    first = reviewer("我在这里。\n\n", False)
    assert first.candidate.text == "我在这里。"
    assert len(calls) == 1
    assert reviewer("我在这里。\n\n你想聊", False) is None
    assert len(calls) == 1
    second = reviewer("我在这里。\n\n你想聊什么？", True)
    assert second.candidate.text == "我在这里。\n\n你想聊什么？"
    assert second.candidate.units[0] == first.candidate.units[0]
    assert second.review.approval_source == "model"
    assert len(calls) == 2


def test_incomplete_or_contradictory_assessment_cannot_publish(tmp_path):
    def hold(packet):
        result = approved(packet)
        result["decision"] = "hold"
        result["no_open_dependency"] = False
        return result

    _, _, reviewer, _ = setup(tmp_path, hold)
    assert reviewer("如果你有时间，\n\n", False) is None
    assert reviewer("如果你有时间，我可以接着说。", True) is None
    assert reviewer.approved_text == ""


@pytest.mark.parametrize("tail", ["如果你愿意，", "条件如下：", "我想说、"])
def test_dangling_tail_is_not_offered_as_semantic_unit_even_if_model_would_approve(tmp_path, tail):
    _, _, reviewer, calls = setup(tmp_path)
    assert reviewer(tail + "\n\n", False) is None
    assert reviewer(tail, True) is None
    assert not calls


def test_dangling_tail_can_be_completed_before_first_unit_review(tmp_path):
    _, _, reviewer, calls = setup(tmp_path)
    assert reviewer("如果你愿意，\n\n", False) is None
    result = reviewer("如果你愿意，\n\n我可以继续。\n\n", False)
    assert result.candidate.text == "如果你愿意，\n\n我可以继续。"
    assert len(calls) == 1


@pytest.mark.parametrize("text", ["他说：“今天", "这（可能", "请看《第一章"])
def test_open_quote_or_group_is_not_offered_even_with_model_approval(tmp_path, text):
    _, _, reviewer, calls = setup(tmp_path)
    assert reviewer(text + "\n\n", False) is None
    assert reviewer(text, True) is None
    assert not calls


def test_open_quote_can_close_before_next_review(tmp_path):
    _, _, reviewer, calls = setup(tmp_path)
    assert reviewer("他说：“今天\n\n", False) is None
    result = reviewer("他说：“今天\n\n很高兴。”\n\n", False)
    assert result.candidate.text == "他说：“今天\n\n很高兴。”"
    assert len(calls) == 1


@pytest.mark.parametrize("fault", ["foreign_hash", "extra_field", "false_bool", "wrong_type"])
def test_host_rejects_unbound_or_malformed_model_approval(tmp_path, fault):
    def invalid(packet):
        value = approved(packet)
        if fault == "foreign_hash":
            value["request_sha256"] = "0" * 64
        elif fault == "extra_field":
            value["unknown"] = True
        elif fault == "false_bool":
            value["self_contained"] = False
        else:
            value["self_contained"] = "true"
        return value

    _, _, reviewer, _ = setup(tmp_path, invalid)
    with pytest.raises(ValueError):
        reviewer("我在这里。\n\n", False)
    assert reviewer.approved_text == ""


def test_reviewer_caps_early_assessments_and_reserves_final_call(tmp_path):
    # The default budget is one early opinion; this test exercises the upper bound.
    _, _, reviewer, calls = setup(tmp_path, early_limit=2)
    assert reviewer("一。\n\n", False)
    assert reviewer("一。\n\n二。\n\n", False)
    assert reviewer("一。\n\n二。\n\n三。\n\n", False) is None
    assert len(calls) == 2
    final = reviewer("一。\n\n二。\n\n三。", True)
    assert final.candidate.text == "一。\n\n二。\n\n三。"
    assert len(calls) == 3


def test_final_only_review_does_not_reenter_active_model_transport(tmp_path):
    from virtual_companion.cognition.stream_unit_review import BoundedUnitReviewer

    store = SQLiteStreamingStore(tmp_path / "final-only.sqlite3")
    run = claim(store, store.accept(config(), "first", "你好"))
    calls = []

    def assess(packet):
        calls.append(packet)
        return approved(packet)

    reviewer = BoundedUnitReviewer(run, ShortRequest("你好"), assess, early_limit=0)
    assert reviewer("我在这里。\n\n", False) is None
    assert not calls
    result = reviewer("我在这里。", True)
    assert result.candidate.text == "我在这里。"
    assert len(calls) == 1


def test_held_candidate_is_not_reassessed_until_new_complete_boundary(tmp_path):
    decisions = []

    def first_hold_then_approve(packet):
        decisions.append(packet["payload"]["candidate_prefix"])
        if len(decisions) == 1:
            result = approved(packet)
            result["decision"] = "hold"
            result["self_contained"] = False
            return result
        return approved(packet)

    _, _, reviewer, calls = setup(tmp_path, first_hold_then_approve, early_limit=2)
    assert reviewer("如果你愿意\n\n", False) is None
    assert reviewer("如果你愿意\n\n我", False) is None
    assert reviewer("如果你愿意\n\n我可以", False) is None
    assert len(calls) == 1
    result = reviewer("如果你愿意\n\n我可以继续。\n\n", False)
    assert result.candidate.text == "如果你愿意\n\n我可以继续。"
    assert len(calls) == 2


def test_stream_coordinator_waits_for_review_and_confirms_units_before_model_eof(tmp_path):
    store, run, reviewer, calls = setup(tmp_path, early_limit=2)
    observed = []

    class Model:
        def count(self, body):
            return 10

        def invoke(self, body):
            self.receive(UnapprovedTextDelta(1, "我在这里。"))
            # A complete sentence is already a faithful candidate boundary.
            assert len(observed) == len(calls) == 1
            self.receive(UnapprovedTextDelta(2, "\n\n"))
            # The separator alone adds no content, so no new unit is published.
            assert len(observed) == len(calls) == 1
            self.receive(UnapprovedTextDelta(3, "你想聊什么？"))
            assert len(observed) == len(calls) == 2
            return {"text": "我在这里。\n\n你想聊什么？", "tool_calls": [], "finish_reason": "stop"}

        def cancel(self):
            pass

    model = Model()

    def factory(receive):
        model.receive = receive
        return model

    def present(event):
        observed.append(event)
        return f"host-{len(observed)}"

    dialogue = StreamDialogue(
        store=store,
        model_factory=factory,
        review_prefix=reviewer,
        present=present,
        checkpoint=lambda: None,
        cancelled=Event(),
        clock=lambda: 100,
    )
    assert dialogue.generate(run, ShortRequest("你好"))["outcome"] == "completed"
    assert len(observed) == len(calls) == 2
    assert store.stream_state(A, run.input_id)["confirmed_text"] == "我在这里。\n\n你想聊什么？"


# Verbatim synthetic s-01 candidate from the v20 receipt, rejected early and
# approved unchanged at final review (candidate SHA-256 9c417cff...16d5d65).
S01_V20_REPLY = (
    "雨大得让人容易把眼前和记忆里的路叠在一起——同样是落雨，"
    "小时候放学那条路上的水洼和屋檐声，总会自己冒出来。"
)


@pytest.mark.parametrize("rejection_port", ["opinion", "exception"])
def test_early_rejected_identical_final_candidate_cannot_publish(tmp_path, rejection_port):
    from virtual_companion.text.turn_orchestrator import CandidateRejected

    def reject_then_approve(packet):
        verdict = approved(packet)
        if not packet["payload"]["final"]:
            verdict.update(decision="reject", constraints_satisfied=False)
            if rejection_port == "exception":
                raise CandidateRejected(verdict)
        else:
            assert packet["payload"]["candidate_prefix"] == calls[0]["payload"]["candidate_prefix"]
            assert packet["payload"]["candidate_sha256"] == calls[0]["payload"]["candidate_sha256"]
        return verdict

    _, _, reviewer, calls = setup(tmp_path, reject_then_approve, user_text=S01)
    with pytest.raises(CandidateRejected) as early:
        reviewer(S01_V20_REPLY, False)
    assert early.value.text == S01_V20_REPLY
    with pytest.raises(CandidateRejected) as final:
        reviewer(S01_V20_REPLY, True)
    assert final.value is early.value
    assert not reviewer.approved_text
    assert len(calls) == 1
    assert calls[0]["payload"]["candidate_prefix"] == S01_V20_REPLY
    assert calls[0]["payload"]["final"] is False


def test_default_early_rejected_identical_candidate_uses_existing_repair(tmp_path):
    from test_default_text_delivery import Model, wait_for
    from test_default_text_delivery import setup as default_setup

    repaired_reply = "雨声把你带回了那条放学的路。"

    class FlippingReviewModel(Model):
        def review_peer(self):
            owner = self

            class Peer(Model):
                def stream(self, messages, cancelled):
                    owner.calls.append(deepcopy(messages))
                    packet = recorded_review_packet(messages)
                    reject = packet["candidate_prefix"] == S01_V20_REPLY and not packet["final"]
                    yield json.dumps(audit_opinion(packet,
                        decision="reject" if reject else "approve",
                        constraints_satisfied=not reject))

            return Peer()

        def stream(self, messages, cancelled):
            self.calls.append(deepcopy(messages))
            repairing = any("rejected_candidate" in message["content"] for message in messages)
            reply = repaired_reply if repairing else S01_V20_REPLY
            yield '{"action":"answer","text":' + json.dumps(reply, ensure_ascii=False)[:-1]
            yield '"}'

    model = FlippingReviewModel()
    runtime, session, memory, mirror, _ = default_setup(tmp_path, model)
    try:
        session.send(S01)
        event = wait_for(lambda: runtime.poll()["event"])
        assert event["display_text"] == repaired_reply
        assert len(model.calls) == 4  # initial generation/reject, existing repair/approve
        initial, early_review, repair, repaired_review = model.calls
        before, after = (recorded_review_packet(call) for call in
                         (early_review, repaired_review))
        assert before["candidate_prefix"] == S01_V20_REPLY and before["final"] is False
        assert after["candidate_prefix"] == repaired_reply
        assert before["basis"] == after["basis"] and before["context"] == after["context"]
        assert repair[:-2] == initial[:-1] and repair[-1] == initial[-1]
        assert S01_V20_REPLY in repair[-2]["content"]
        assert '"constraints_satisfied":false' in repair[-2]["content"]
        runtime.ack(dict(sequence=event["sequence"], fingerprint=event["fingerprint"],
                         host_receipt="synthetic-identical-reject-repair"))
        wait_for(lambda: not session.snapshot()["busy"])
        assert session.snapshot()["attempt"]["status"] == "completed"
        assert mirror.history()["messages"][-1]["content"] == repaired_reply
        segments = memory.store.segments(memory.companion, runtime.claim.input_id)
        assert len(segments) == 1 and segments[0]["status"] == "confirmed"
    finally:
        memory.close()
        mirror.close()


@pytest.mark.parametrize("rejection_port", ["opinion", "exception"])
def test_early_rejected_candidate_cannot_reopen_by_appending_text(tmp_path, rejection_port):
    from virtual_companion.text.turn_orchestrator import CandidateRejected

    def assess(packet):
        verdict = approved(packet)
        if not packet["payload"]["final"]:
            verdict.update(decision="reject", constraints_satisfied=False)
            if rejection_port == "exception":
                raise CandidateRejected(verdict)
        return verdict

    _, _, reviewer, calls = setup(tmp_path, assess, user_text=S01)
    with pytest.raises(CandidateRejected) as early:
        reviewer(S01_V20_REPLY, False)
    changed = S01_V20_REPLY + "这里只是我虚构的雨景联想，不代表你的经历。"
    with pytest.raises(CandidateRejected) as final:
        reviewer(changed, True)
    assert final.value is early.value and final.value.text == S01_V20_REPLY
    assert not reviewer.approved_text
    assert len(calls) == 1
    assert calls[0]["payload"]["candidate_prefix"] == S01_V20_REPLY


@pytest.mark.parametrize("early_opinion", [
    "hold", "missing_check", "wrong_check", "foreign_hash", "false_approval",
    "raised_hold", "raised_malformed", "raised_foreign", "schema_error", "transport_error",
])
def test_only_valid_early_hold_may_continue_without_repair(tmp_path, early_opinion):
    from virtual_companion.text.turn_orchestrator import CandidateRejected

    def assess(packet):
        verdict = approved(packet)
        if packet["payload"]["final"]:
            return verdict
        if early_opinion == "schema_error":
            raise ValueError("Unit model response violates bound schema")
        if early_opinion == "transport_error":
            raise OSError("Synthetic review transport error")
        if early_opinion in {"hold", "raised_hold"}:
            verdict.update(decision="hold", self_contained=False)
        elif early_opinion == "false_approval":
            verdict["constraints_satisfied"] = False
        else:
            verdict.update(decision="reject", constraints_satisfied=False)
            if early_opinion in {"missing_check", "raised_malformed"}:
                del verdict["self_contained"]
            elif early_opinion == "wrong_check":
                verdict["self_contained"] = "true"
            elif early_opinion in {"foreign_hash", "raised_foreign"}:
                verdict["request_sha256"] = "0" * 64
        if early_opinion.startswith("raised_"):
            raise CandidateRejected(verdict)
        return verdict

    _, _, reviewer, calls = setup(tmp_path, assess, user_text=S01)
    if early_opinion not in {"hold", "raised_hold", "transport_error"}:
        with pytest.raises(ValueError):
            reviewer(S01_V20_REPLY, False)
    elif early_opinion == "transport_error":
        with pytest.raises(OSError):
            reviewer(S01_V20_REPLY, False)
    else:
        assert reviewer(S01_V20_REPLY, False) is None
    if early_opinion not in {"hold", "raised_hold"}:
        assert not reviewer.approved_text and len(calls) == 1
        return
    result = reviewer(S01_V20_REPLY, True)
    assert result.candidate.text == reviewer.approved_text == S01_V20_REPLY
    assert len(calls) == 2
    assert calls[0]["payload"]["candidate_sha256"] == calls[-1]["payload"]["candidate_sha256"]
    assert calls[-1]["payload"]["final"] is True


def protocol_packet(user_text=S01, candidate=V21_RAIN):
    source = {"id": "current:source-input", "kind": "current", "owner": A,
              "source_id": "source-input", "role": "user", "revision": 1,
              "text": user_text}
    payload = {"version": "stream-unit-review.v7", "candidate_prefix": candidate,
               "review_segments": segment_payload(candidate),
               "final": True, "user_text": user_text,
               "basis": {"companion_id": A, "input_id": "source-input", "state_version": 1},
               "personal_sources": [source]}
    return {"payload": payload, "sha256": digest(payload)}


def legacy_wire_snapshot(audit_version, task):
    """Characterize the unchanged v3/v4 Port through its actual frozen body."""
    from virtual_companion.cognition import stream_unit_review as port
    from virtual_companion.providers.contracts import ProviderCapabilities, ProviderRoute
    from virtual_companion.text.provider_model import ProviderModel

    packet = protocol_packet(
        "A general observation is welcome.", "General observations vary. Fine.",
    )
    payload = packet["payload"]
    if audit_version == port.UNIT_SCOPE_AUDIT_VERSION:
        payload["audit_version"] = audit_version
    source = payload["personal_sources"][0]
    payload["personal_sources"].append({**source, "id": "history:old-input", "kind": "history",
        "source_id": "old-input", "text": "A corrected observation remains relevant."})
    payload["context"] = canonical({
        "history": [{"id": "old-input", "role": "user",
                     "text": "A corrected observation remains relevant."}],
        "memories": [], "exchanges": [], "memory_status": "pending_source",
        "memory_access": "not_accessed", "current_operation": {"status": "committed"},
        "source_processing": {"protocol": "current-source-processing.v1", "status": "queued",
                              "source_id": "source-input", "receipt_sha256": None},
    })
    payload["displayed_prefix"] = payload["candidate_prefix"][:5]
    if task:
        payload["response_task_contract"] = {"kind": "exact_items", "count": 2, "span": "two"}
    packet["sha256"] = digest(payload)
    model_opinion = {"audit_version": audit_version, "decision": "approve",
                     **dict.fromkeys(port.CHECKS, True)}
    if audit_version == port.UNIT_SCOPE_AUDIT_VERSION:
        model_opinion["units"] = [
            {"segment_id": item["segment_id"], "personal_support": "none",
             "personal_source_ids": [], "fiction_support": "none", "fiction_source_ids": []}
            for item in payload["review_segments"]]
    else:
        model_opinion["segments"] = [
            {"segment_id": item["segment_id"], "commitments": [], "questions": []}
            for item in payload["review_segments"]]
    route = ProviderRoute("legacy-wire", 1, "api", "https://api.deepseek.com/chat/completions",
        "deepseek-flash", "bearer", "dpapi:" + "a" * 32, True,
        ("current", "history", "valid_memory", "derived"),
        ProviderCapabilities(True, False, True, 1_048_576, "max_tokens"),
        wire_profile="deepseek-flash.focused.v2")
    class Counter:
        def __call__(self, body):
            return 17

        def local_byte_bound(self, body):
            return len(canonical(body).encode())

    model = ProviderModel(route, counter=Counter(), transport=None)
    snapshots = []

    def invoke(stage, messages, schema, cancelled):
        assert not cancelled.is_set() and stage == "stream_unit_review"
        prepared = model.prepare_stage_request(messages, stage=stage, schema=schema,
                                               output_tokens=1536)
        snapshots.append({"schema": digest(schema), "messages": digest(messages),
                          "frozen_body": digest(prepared._body)})
        return model_opinion

    verdict = port.StructuredUnitAssessor(invoke, Event(), audit_version=audit_version)(packet)
    assert len(snapshots) == 1
    return {**snapshots[0], "semantic_view": digest(port.semantic_review_view(payload)),
            "bound_verdict": digest(verdict)}


def test_semantic_view_keeps_one_candidate_and_exact_evidence_without_host_metadata():
    from virtual_companion.cognition.stream_unit_review import semantic_review_view

    packet = protocol_packet(candidate="纸张可以留住一个日期。你看它时想到什么？")
    payload = packet["payload"]
    current = payload["personal_sources"][0]
    history = {**current, "id": "history:past-input", "kind": "history",
               "source_id": "past-input", "text": "我上周去了书展。"}
    memory = {**current, "id": "memory:fact-record", "kind": "memory", "revision": 3,
              "source_id": "fact-input", "text": "我喜欢步行。"}
    payload["personal_sources"] += [history, memory]
    context = {
        "memory_status": "current",
        "current_operation": {"status": "committed", "action": "correct"},
        "history": [{"id": "past-input", "role": "user", "text": history["text"]},
                    {"id": "assistant-input", "role": "assistant", "text": "已经收到。",
                     "response_type": "ack"}],
        "memories": [{"record_id": "fact-record", "content": memory["text"], "version": 3,
                      "event_time": None, "status": "active"}],
        "exchanges": [{"content": "上次待续问题", "direction": "assistant_response"}],
    }
    payload.update(context=canonical(context),
                   observed_text=payload["candidate_prefix"] + "尚未审查的尾部",
                   displayed_prefix="纸张可以留住一个日期。", candidate_sha256="host-fingerprint",
                   forbidden_characters="!",
                   constraints=[{"kind": "max_chars", "limit": 80, "span": "80字以内"}],
                   unparsed_constraints=["保留条件"], response_type="ack",
                   response_task_contract={"kind": "exact_items", "count": 2, "span": "给两条建议"})
    before = deepcopy(payload)
    view = semantic_review_view(payload)
    assert payload == before
    assert view["user_source_id"] == current["id"]
    assert view["personal_sources"] == payload["personal_sources"]
    assert view["review_segments"] == payload["review_segments"]
    assert "".join(row["text"] for row in view["review_segments"]) == payload["candidate_prefix"]
    assert not {"basis", "candidate_sha256", "candidate_prefix",
                "observed_text", "user_text"} & view.keys()
    assert "尚未审查的尾部" not in canonical(view)
    assert view["displayed_segments"] == [payload["review_segments"][0]["segment_id"]]
    assert view["context"]["history"][0] == {
        "id": "past-input", "role": "user", "personal_source_id": history["id"]}
    assert view["context"]["memories"][0] == {
        "record_id": "fact-record", "version": 3, "event_time": None, "status": "active",
        "personal_source_id": memory["id"]}
    assert view["context"]["history"][1] == context["history"][1]
    assert view["context"]["current_operation"] == context["current_operation"]
    assert view["context"]["exchanges"] == context["exchanges"]
    for key in ("final", "forbidden_characters", "constraints", "unparsed_constraints",
                "response_task_contract", "response_type"):
        assert view[key] == payload[key]


def test_assessor_uses_semantic_view_but_binds_the_complete_original_packet():
    from virtual_companion.cognition.stream_unit_review import (
        StructuredUnitAssessor,
        semantic_review_view,
    )

    packet = protocol_packet(candidate="我上周去了书展。")
    source = packet["payload"]["personal_sources"][0]
    opinion = audit_opinion(packet["payload"], mode="personal_fact", support="unsupported",
                            refs=[personal_ref(source, "放学的路")])
    seen = []

    def invoke(stage, messages, schema, cancelled):
        seen.append(deepcopy(messages))
        assert json.loads(messages[-1]["content"]) == semantic_review_view(packet["payload"])
        assert "JSON schema:" not in messages[0]["content"]
        assert schema["properties"]["audit_version"]["const"] == "candidate-propositions.v3"
        return opinion

    result = StructuredUnitAssessor(invoke, Event())(packet)
    assert len(seen) == 1 and result["request_sha256"] == packet["sha256"]
    assert result["request_sha256"] != digest(json.loads(seen[0][-1]["content"]))
    assert result["decision"] == "reject" and result["model_opinion"] == V21_APPROVE


def invoke_protocol(packet, opinion):
    from virtual_companion.cognition.stream_unit_review import StructuredUnitAssessor

    calls = []

    def invoke(stage, messages, schema, cancelled):
        calls.append((stage, messages, schema))
        return deepcopy(opinion)

    result = StructuredUnitAssessor(invoke, Event())(packet)
    assert len(calls) == 1 and calls[0][0] == "stream_unit_review"
    from virtual_companion.cognition.stream_unit_review import semantic_review_view

    assert json.loads(calls[0][1][1]["content"]) == semantic_review_view(packet["payload"])
    assert result["request_sha256"] == packet["sha256"]
    return result


@pytest.mark.parametrize("support", ["unsupported", "uncertain"])
def test_protocol_valid_quote_does_not_override_unsupported_relation(support):
    packet = protocol_packet()
    source = packet["payload"]["personal_sources"][0]
    # The quote exists, but it does not supply rain sounds, water sensations or
    # their causal relation. That insufficiency is still a model semantic verdict.
    opinion = audit_opinion(packet["payload"], mode="personal_fact", support=support,
                            refs=[personal_ref(source, "小时候放学的路")])
    result = invoke_protocol(packet, opinion)
    assert result["decision"] == "reject" and result["constraints_satisfied"] is False
    assert result["model_opinion"] == V21_APPROVE
    assert relation_wire(result["propositions"]) == opinion["segments"]


@pytest.mark.parametrize("mutation", ["empty", "missing", "duplicate", "unknown",
                                    "numeric-span", "wrong-version", "foreign-payload"])
def test_protocol_requires_exact_nonempty_candidate_coverage(mutation):
    packet = protocol_packet(candidate="雨声像翻相册的手。先看看路况。")
    opinion = audit_opinion(packet["payload"], mode="metaphor")
    item = opinion["segments"][0]
    if mutation == "empty":
        opinion["segments"] = []
    elif mutation == "missing":
        opinion["segments"].pop()
    elif mutation == "duplicate":
        opinion["segments"].append(deepcopy(item))
    elif mutation == "unknown":
        item["segment_id"] = "segment-from-another-candidate"
    elif mutation == "numeric-span":
        item.update(start=0, end=31, text="模型自行计数字符")
    elif mutation == "foreign-payload":
        packet["payload"]["review_segments"][0]["text"] = "另一个候选。"
        packet["sha256"] = digest(packet["payload"])
    else:
        opinion["audit_version"] = "candidate-propositions.v1"
    with pytest.raises(ValueError):
        invoke_protocol(packet, opinion)


def test_protocol_covers_every_codepoint_and_preserves_each_recorded_relation():
    packet = protocol_packet(candidate="雨声会让你想到什么？\n\n建议：\n先看看路况🌧️。")
    text = packet["payload"]["candidate_prefix"]
    opinion = audit_opinion(packet["payload"])
    opinion["segments"][0]["questions"] = ["雨声会让你想到什么？"]
    # ID coverage is independent of model response order; host restores original ranges.
    opinion["segments"].reverse()
    result = invoke_protocol(packet, opinion)
    propositions = result["propositions"]
    assert len(propositions) == 2 and propositions[0]["questions"] == ["雨声会让你想到什么？"]
    assert propositions[1]["commitments"] == propositions[1]["questions"] == []
    assert "".join(item["text"] for item in propositions) == text
    assert propositions[0]["start"] == 0 and propositions[-1]["end"] == len(text)
    assert all(item["text"] == text[item["start"]:item["end"]] for item in propositions)
    assert candidate_segments(text) == candidate_segments(text)


def test_host_review_segments_bound_long_answers_without_splitting_commas():
    text = "一条建议，含执行细节，仍在同一句。\n" * 12
    segments = candidate_segments(text)
    assert len(segments) == 8
    assert "".join(item["text"] for item in segments) == text
    assert segments[0]["text"] == "一条建议，含执行细节，仍在同一句。\n"
    assert segments[-1]["text"].count("。") == 5
    assert len({item["segment_id"] for item in segments}) == 8
    assert segments[0]["segment_id"] != candidate_segments(text.replace("一条", "另一条", 1))[0][
        "segment_id"]


@pytest.mark.parametrize("field,value", [
    ("id", "memory:unknown"), ("owner", "another-owner"), ("source_id", "another-input"),
    ("revision", 2), ("role", "assistant"), ("quote", "小时候踩水的触感"), ("quote", ""),
])
def test_protocol_reference_must_match_admitted_identity_and_exact_quote(field, value):
    packet = protocol_packet()
    source = packet["payload"]["personal_sources"][0]
    ref = personal_ref(source, "小时候放学的路")
    ref[field] = value
    opinion = audit_opinion(packet["payload"], mode="personal_fact", support="supported",
                            refs=[ref])
    with pytest.raises(ValueError):
        invoke_protocol(packet, opinion)


def test_protocol_old_boolean_wire_is_not_an_audit():
    with pytest.raises(ValueError):
        invoke_protocol(protocol_packet(), V21_APPROVE)


@pytest.mark.parametrize("invalid", ["foreign-segment", "old-source-revision"])
def test_invalid_segment_or_source_never_reaches_default_delivery_or_ack(tmp_path, invalid):
    from test_default_text_delivery import Model, setup, wait_for

    class InvalidReview(Model):
        def stream(self, messages, cancelled):
            self.calls.append(deepcopy(messages))
            if "你只审查候选文字" in messages[0]["content"]:
                payload = recorded_review_packet(messages)
                ref = personal_ref(payload["personal_sources"][0], S01)
                opinion = audit_opinion(payload, mode="personal_fact", support="supported",
                                        refs=[ref])
                if invalid == "foreign-segment":
                    opinion["segments"][0]["segment_id"] = "foreign-candidate-segment"
                else:
                    ref["revision"] += 1
                yield json.dumps(opinion)
            else:
                yield json.dumps({"action": "answer", "text": "你想起小时候放学的路。"})

    runtime, session, memory, mirror, model = setup(tmp_path, InvalidReview())
    try:
        session.send(S01)
        wait_for(lambda: not session.snapshot()["busy"])
        assert session.snapshot()["attempt"]["status"] == "failed"
        assert len(model.calls) == 2  # Invalid audit does not authorize a repair.
        assert runtime.poll()["event"] is None  # No renderer delivery or ACK authority.
        state = memory.store.stream_state(memory.companion, runtime.claim.input_id)
        assert state["approved_units"] == state["confirmed_units"] == 0
        assert state["confirmed_text"] == ""
        assert memory.store.segments(memory.companion, runtime.claim.input_id) == []
        assert [row["role"] for row in mirror.history()["messages"]] == ["user"]
    finally:
        memory.close()
        mirror.close()


@pytest.mark.parametrize("mode,support,refs", [
    ("personal_fact", "supported", []), ("personal_fact", "not_applicable", []),
    ("fiction", "not_applicable", []), ("nonpersonal", "supported", []),
])
def test_protocol_personal_or_fictional_content_cannot_omit_its_relation(mode, support, refs):
    packet = protocol_packet()
    with pytest.raises(ValueError):
        invoke_protocol(packet, audit_opinion(packet["payload"], mode=mode,
                                             support=support, refs=refs))


def test_protocol_current_user_details_can_support_same_specific_memory():
    source_text = ("这次大雨让我想起小时候放学的路，原因正是雨声、踩水的触感和当年放学路上的"
                   "某种节奏对上了。")
    packet = protocol_packet(source_text)
    opinion = audit_opinion(packet["payload"], mode="personal_fact", support="supported",
        refs=[personal_ref(packet["payload"]["personal_sources"][0], source_text)])
    result = invoke_protocol(packet, opinion)
    assert result["decision"] == "approve" and result["model_opinion"] == V21_APPROVE


def test_protocol_freezes_candidate_and_source_text_before_model_call():
    from virtual_companion.cognition.stream_unit_review import StructuredUnitAssessor

    packet = protocol_packet()
    bound_hash = packet["sha256"]
    source = packet["payload"]["personal_sources"][0]
    opinion = audit_opinion(packet["payload"], mode="personal_fact", support="unsupported",
        refs=[personal_ref(source, "小时候放学的路")])

    def invoke(stage, messages, schema, cancelled):
        assert recorded_review_packet(messages)["candidate_prefix"] == V21_RAIN
        packet["payload"]["candidate_prefix"] = "被调用者改写的文字。"
        source["text"] = "被调用者改写的来源。"
        packet["sha256"] = digest(packet["payload"])
        return opinion

    result = StructuredUnitAssessor(invoke, Event())(packet)
    assert result["request_sha256"] == bound_hash != packet["sha256"]
    assert result["decision"] == "reject"
    assert relation_wire(result["propositions"]) == opinion["segments"]
    assert "".join(item["text"] for item in result["propositions"]) == V21_RAIN


def test_protocol_source_match_is_not_a_host_proof_of_semantic_entailment():
    packet = protocol_packet()
    # Intentionally lying semantic label is an interface counterexample. Host can
    # show exactly what the model claimed, but cannot derive entailment from it.
    opinion = audit_opinion(packet["payload"], mode="personal_fact", support="supported",
        refs=[personal_ref(packet["payload"]["personal_sources"][0], "小时候放学的路")])
    result = invoke_protocol(packet, opinion)
    assert result["decision"] == "approve"
    assert result["propositions"][0]["commitments"][0]["sources"][0]["quote"] == "小时候放学的路"


@pytest.mark.parametrize("proof", ["current", "history", "memory", "old_revision",
                                 "revoked", "assistant"])
def test_protocol_source_view_uses_only_current_admitted_user_records(tmp_path, proof):
    from virtual_companion.cognition.short_dialogue import ShortContext
    from virtual_companion.cognition.stream_unit_review import (
        BoundedUnitReviewer,
        StructuredUnitAssessor,
    )

    store = SQLiteStreamingStore(tmp_path / "source-view.sqlite3")
    run = claim(store, store.accept(config(), "source-input", S01))
    data = {"memory_status": "current", "exchanges": [], "history": [
        {"id": "past-user", "role": "user", "text": "放学时踩过水洼。"},
        {"id": "past-assistant", "role": "assistant", "text": "你还闻到湿土。"},
    ], "memories": [
        {"record_id": "active-fact", "version": 3, "status": "active", "source_kind": "input",
         "source_id": "past-user", "content": "放学时听到雨声。"},
        {"record_id": "revoked-fact", "version": 2, "status": "revoked", "source_kind": "input",
         "source_id": "past-user", "content": "不再有效的细节。"},
        {"record_id": "assistant-fact", "version": 1, "status": "active", "source_kind": "output",
         "source_id": "past-assistant", "content": "助手杜撰的细节。"},
    ]}
    request = ShortRequest(S01, context=ShortContext(A, run.input_id, run.state_version,
                                                   canonical(data)))
    seen = []

    def invoke(stage, messages, schema, cancelled):
        payload = recorded_review_packet(messages)
        seen.append(payload)
        source = payload["personal_sources"][{"current": 0, "history": 1}.get(proof, 2)]
        ref = personal_ref(source, source["text"])
        if proof == "old_revision":
            ref["revision"] = 2
        elif proof == "revoked":
            ref.update(id="memory:revoked-fact", revision=2, quote="不再有效的细节。")
        elif proof == "assistant":
            ref.update(id="history:past-assistant", source_id="past-assistant", role="assistant",
                       revision=1, quote="你还闻到湿土。")
        return audit_opinion(payload, mode="personal_fact", support="supported", refs=[ref])

    reviewer = BoundedUnitReviewer(run, request, StructuredUnitAssessor(invoke, Event()))
    reply = {"current": S01, "history": "放学时踩过水洼。"}.get(proof, "放学时听到雨声。")
    if proof in {"old_revision", "revoked", "assistant"}:
        with pytest.raises(ValueError):
            reviewer(reply, True)
        assert reviewer.approved_text == ""
    else:
        assert reviewer(reply, True).candidate.text == reply
    assert len(seen) == reviewer.final_calls == 1
    sources = seen[0]["personal_sources"]
    assert [item["id"] for item in sources] == [f"current:{run.input_id}",
                                               "history:past-user", "memory:active-fact"]
    assert all(item["owner"] == A and item["role"] == "user" for item in sources)
    assert sources[-1]["revision"] == 3 and sources[-1]["source_id"] == "past-user"
    assert seen[0]["context"] == request.context.payload
    pending = deepcopy(data)
    pending["memory_status"] = "pending_source"
    unavailable = ShortRequest(S01, context=ShortContext(A, run.input_id, run.state_version,
                                                        canonical(pending)))
    assert [item["kind"] for item in unavailable.personal_sources(
        A, run.input_id, run.state_version)] == ["current", "history"]


def test_protocol_fiction_requires_current_authorization_reference():
    packet = protocol_packet("今天下雨了。", "虚构一幕：小孩踩过水洼。")
    old_authorization = {"id": "history:past-fiction", "kind": "history", "owner": A,
                         "source_id": "past-fiction", "role": "user", "revision": 1,
                         "text": "请写一个虚构的雨天放学片段。"}
    packet["payload"]["personal_sources"].append(old_authorization)
    packet["sha256"] = digest(packet["payload"])
    opinion = audit_opinion(packet["payload"], mode="fiction", support="supported",
        refs=[personal_ref(old_authorization, old_authorization["text"])])
    with pytest.raises(ValueError, match="current user authorization"):
        invoke_protocol(packet, opinion)


@pytest.mark.parametrize("binding", ["owner", "input", "state"])
def test_protocol_context_source_view_cannot_use_foreign_binding(tmp_path, binding):
    from virtual_companion.cognition.short_dialogue import ShortContext
    from virtual_companion.cognition.stream_unit_review import BoundedUnitReviewer

    store = SQLiteStreamingStore(tmp_path / "foreign-source.sqlite3")
    run = claim(store, store.accept(config(), "source-input", S01))
    owner = "00000000-0000-4000-8000-000000000099" if binding == "owner" else A
    input_id = "another-input" if binding == "input" else run.input_id
    state = run.state_version + 1 if binding == "state" else run.state_version
    context = ShortContext(owner, input_id, state, canonical({"history": []}))
    request = ShortRequest(S01, context=context)
    seen = []
    reviewer = BoundedUnitReviewer(run, request,
        lambda packet: seen.append(packet) or approved(packet))
    with pytest.raises(ValueError, match="context.*binding"):
        reviewer("下雨了。", True)
    assert seen == [] and reviewer.approved_text == ""



@pytest.mark.parametrize("audit_version,task,expected", [
    ('candidate-propositions.v3', False, {
        'schema': '9ddf6fca1ae6a0625be1b57fe671b861d29ce68e562dc8cb3f1a026f51cca121',
        'messages': '9a358c490622ecfca65419ee539540ea9626d010247aeca3d067aa08d8247484',
        'frozen_body': '6beb2b96f1bc28fe415b4b2366fdf60b4542d8174bce846fc07198437435088d',
        'semantic_view': '6a56442385fb1b4bd813a5ffa3c589a362f6e0023bf611f96628481e775052bb',
        'bound_verdict': '980171029761d21d83074b644a633651053de1a1981b0c9ab9a16b17570d5d85',
    }),
    ('candidate-propositions.v3', True, {
        'schema': '9ddf6fca1ae6a0625be1b57fe671b861d29ce68e562dc8cb3f1a026f51cca121',
        'messages': 'fdf5f75f318697b03abe31de617beec3dc9c0ffad48db0c658f657b2a7c133f1',
        'frozen_body': '97a44c1e5d82d0a46c8f92e7e49987c81faf92cc12138b57290c8a6ceabc1e6d',
        'semantic_view': '195349b5d7979c38853f6c3edf6a296944d3cf79ad83f3b288a69a644ef1335d',
        'bound_verdict': '67055c9e4dfb1690880e9a2de4b4a3d3be7c366de1933a4e165a8c3cca9c3e71',
    }),
    ('candidate-unit-scope.v4', False, {
        'schema': '7a718a7c2fb8e003b7a34d01e3a4e8e04d3e190ba2c30a657c98cfc47fa687f1',
        'messages': 'cfa7444e861a083aee4a628eacfe39ac20f29559f6e32c162f762383fbac8de5',
        'frozen_body': '8642ede4d9c6630fa75faaecf0f10515ec8af660baf4f019f381dad4a82badcf',
        'semantic_view': 'af9fcb5a65af7aeabb6bb499e9ece411c3e4b9d01b8c2cfec83930868147c901',
        'bound_verdict': 'c93a988374b3a68875ac86e3f0b8e37e52d327ee71636c1f89a2f12c6ba2dfc2',
    }),
    ('candidate-unit-scope.v4', True, {
        'schema': '7a718a7c2fb8e003b7a34d01e3a4e8e04d3e190ba2c30a657c98cfc47fa687f1',
        'messages': '1da56928551c71cb708da1efa74a8bdd51ea62cb6a37d98d0f265188f731456c',
        'frozen_body': 'd9a76d0ba063bcb5be956766a65c7664552fcdaf3066471cb83d547b7bef70f1',
        'semantic_view': '645fb4e43fabef9a4827ec316cbbfb454ea6cc782980a4c2d9dd31dc7f393771',
        'bound_verdict': 'de2f4db7310198712302cb0c5215fed870fa1cf2907775a91e1681f9d08bd509',
    }),
])
def test_legacy_v3_v4_schema_messages_view_frozen_body_and_verdict_stay_identical(
    audit_version, task, expected,
):
    # These exact snapshots were recorded from the preimage, not the compact implementation.
    assert legacy_wire_snapshot(audit_version, task) == expected
