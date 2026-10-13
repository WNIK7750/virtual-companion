"""RF3-S2: host-bound review identity and host-verifiable user constraints.

The model no longer copies a long host hash, and a length constraint the host can
count is enforced on the actual display surface instead of trusting the model's
self-reported ``constraints_satisfied``.
"""

import json
from threading import Event

import pytest
from test_natural_context_responsibility import write_record
from test_rebuilt_turns import A, claim, config
from test_unit_scope_review_v4 import (
    finish,
    recorded_calls,
    recording_runtime,
)
from test_unit_scope_review_v4 import (
    sealed_capture as sealed_capture,
)
from unit_review_recording import bound_unit_packets as bound_unit_packets
from unit_review_recording import recorded_review_packet
from virtual_companion.boundary.contracts import canonical
from virtual_companion.boundary.presentation import digest
from virtual_companion.boundary.sqlite_streaming import SQLiteStreamingStore
from virtual_companion.cognition.independent_unit_assessment import CompactUnitAssessor
from virtual_companion.cognition.short_dialogue import ShortRequest
from virtual_companion.cognition.stream_dialogue import StreamDialogue
from virtual_companion.cognition.stream_unit_review import (
    BoundedUnitReviewer,
    UserConstraintViolation,
    semantic_review_view,
)
from virtual_companion.cognition.user_constraints import (
    UserConstraint,
    character_count,
    constraint_issues,
    recognize_user_constraints,
)
from virtual_companion.providers.text_deltas import UnapprovedTextDelta

LONG = "好，我明白你的意思了，我们可以慢慢聊。"
V18_PLAN_INPUT = "我想去散步但在下雨，请给我一个简单安排。"
V18_PLAN_PREFIX = (
    "既然正在下雨，先别急着出门，在家做点别的过渡一下，"
    "比如整理下桌面或者听会儿歌，等雨停了再出去。"
)


def task_review(tmp_path, user_text, *, review_policy="model"):
    from virtual_companion.cognition.stream_unit_review import StructuredUnitAssessor
    from virtual_companion.text.fake import synthetic_unit_opinion

    store = SQLiteStreamingStore(tmp_path / "task-contract.sqlite3")
    run = claim(store, store.accept(config(), "task-input", user_text))
    calls = []

    def invoke(stage, messages, schema, cancelled):
        calls.append((stage, messages, schema))
        return synthetic_unit_opinion(json.loads(messages[-1]["content"]))

    reviewer = BoundedUnitReviewer(run, ShortRequest(user_text),
        StructuredUnitAssessor(invoke, Event()), review_policy=review_policy)
    return reviewer, calls


def test_v19_activity_contract_withholds_actual_v18_prefix_before_any_opinion(tmp_path):
    reviewer, calls = task_review(tmp_path, V18_PLAN_INPUT)
    assert reviewer(V18_PLAN_PREFIX, False) is None
    assert calls == [] and reviewer.early_calls == 0 and reviewer.approved_text == ""
    assert reviewer.task_contract.to_payload() == {
        "version": "response-task-contract.v1", "kind": "activity_plan",
        "source": "user_text", "span": V18_PLAN_INPUT,
        "requires_complete_answer": True, "activity": "散步",
        "waiting": "condition_of_same_path", "substitute_activities": False,
        "numeric_claims": "supported_or_explicit_plan_choice",
    }
    # An approve fixture proves routing, not that this old response is acceptable.
    result = reviewer(V18_PLAN_PREFIX, True)
    assert result.candidate.text == V18_PLAN_PREFIX
    semantic = json.loads(calls[0][1][-1]["content"])
    packet = recorded_review_packet(calls[0][1])
    assert semantic_review_view(packet) == semantic
    assert semantic["response_task_contract"] == packet["response_task_contract"] \
        == reviewer.task_contract.to_payload()
    assert packet["final"] is True and not packet["displayed_prefix"]
    assert semantic["final"] is True and semantic["displayed_segments"] == []
    assert "displayed_partial" not in semantic
    system = calls[0][1][0]["content"]
    assert "主行动" in system and "天气等待" in system and "替代活动" in system


def test_v19_exact_two_items_contract_reaches_real_assessor_wire(tmp_path):
    user_text = "给我两条下雨天通勤的实用建议。"
    reviewer, calls = task_review(tmp_path, user_text)
    assert reviewer("1. 提前确认路线。", False) is None
    assert calls == []
    result = reviewer("1. 提前确认路线。\n2. 走路时避开积水。", True)
    assert result is not None
    packet = json.loads(calls[0][1][1]["content"])
    contract = packet["response_task_contract"]
    assert contract["kind"] == "exact_items" and contract["count"] == 2
    assert contract["span"] == user_text and contract["source"] == "user_text"
    assert contract["numeric_claims"] == "supported_or_explicit_plan_choice"
    assert "独立做法" in calls[0][1][0]["content"]
    assert "精确阈值" in calls[0][1][0]["content"]


def test_v19_numbered_count_violation_cannot_be_approved_by_a_model(tmp_path):
    reviewer, calls = task_review(tmp_path, "给我两条下雨天通勤的实用建议。")
    with pytest.raises(UserConstraintViolation, match="exact_items"):
        reviewer("1. 提前确认路线。\n2. 带伞。\n3. 走路时避开积水。", True)
    assert not calls and reviewer.approved_text == ""


def test_v33_only_write_two_opening_sentences_rejects_explanation_before_model_review(tmp_path):
    user_text = "我想给一年没联系的表姐寄张明信片。只写开头两句，别替我编我们一起做过什么。"
    reviewer, calls = task_review(tmp_path, user_text)
    over_answer = ("可以。开头两句可以这样写：\n\n“表姐，好久不见。"
                   "忽然想给你寄张明信片。”\n\n要不要再来一版？")
    assert reviewer("可以。", False) is None
    assert not calls and reviewer.early_calls == 0
    assert reviewer.task_contract.to_payload() == {
        "version": "response-task-contract.v1", "kind": "exact_sentences",
        "source": "user_text", "span": user_text, "requires_complete_answer": True,
        "count": 2, "content_only": True,
    }
    with pytest.raises(UserConstraintViolation, match="exact_sentences"):
        reviewer(over_answer, True)
    assert not calls and reviewer.approved_text == ""


def test_v33_only_write_two_opening_sentences_accepts_only_the_requested_text(tmp_path):
    user_text = "我想给一年没联系的表姐寄张明信片。只写开头两句，别替我编我们一起做过什么。"
    reviewer, calls = task_review(tmp_path, user_text)
    text = "“表姐，好久不见。忽然想给你寄张明信片。”"
    result = reviewer(text, True)
    assert result.candidate.text == text
    assert len(calls) == 1
    packet = recorded_review_packet(calls[0][1])
    assert packet["response_task_contract"] == reviewer.task_contract.to_payload()


@pytest.mark.parametrize("text", [
    "他说：‘写作要克制。只写两句。’你怎么看？",
    "如果我说‘只写两句’，你会怎么理解？",
    "陪我聊两句。",
])
def test_sentence_contract_does_not_take_quoted_or_casual_counts_as_a_request(text):
    from virtual_companion.cognition.user_constraints import recognize_response_task
    assert recognize_response_task(text) is None


def test_sentence_contract_allows_quotes_within_the_requested_writing():
    from virtual_companion.cognition.user_constraints import (
        recognize_response_task,
        task_contract_issues,
    )
    contract = recognize_response_task("只写两句。")
    assert task_contract_issues('表姐，忽然想说声“好久不见”。想给你寄张明信片。', contract) == ()


@pytest.mark.parametrize("reply", [
    "一、提前确认路线。二、走路时避开积水。",
    "建议如下：1. 提前确认路线。\n2. 走路时避开积水。",
    "5. 提前确认路线。\n6. 走路时避开积水。",
])
def test_v19_count_does_not_impose_a_numbering_format(tmp_path, reply):
    reviewer, calls = task_review(tmp_path, "给我两条下雨天通勤的实用建议。")
    result = reviewer(reply, True)
    assert result is not None and len(calls) == 1
    payload = json.loads(calls[0][1][1]["content"])
    assert payload["response_task_contract"]["count"] == 2


def test_v19_general_explanation_contract_has_no_personal_attribution(tmp_path):
    user_text = "我在想，为什么旧照片会让人同时开心又难过？"
    reviewer, calls = task_review(tmp_path, user_text)
    assert reviewer("旧照片唤起快乐。", False) is None
    assert not calls
    reviewer("旧照片能唤起快乐，也让人看见变化；两种反应可以并存。", True)
    packet = json.loads(calls[0][1][1]["content"])
    contract = packet["response_task_contract"]
    assert contract["kind"] == "general_explanation" and contract["topic"] == "旧照片"
    assert contract["aspects"] == ["开心", "难过"]
    assert contract["attribution"] == "general_not_user_personal"
    system = calls[0][1][0]["content"]
    assert "泛指" in system and "用户本人" in system and "情绪归因" in system


def test_v19_task_contract_cannot_be_bypassed_by_host_risk_policy(tmp_path):
    reviewer, calls = task_review(tmp_path, V18_PLAN_INPUT, review_policy="host-risk")
    result = reviewer("等雨小且路面安全时，沿平整路线散步十分钟再回来。", True)
    assert len(calls) == 1 and result.review.model_approved is True


@pytest.mark.parametrize("text,kind", [
    ("给我两条下雨天通勤的实用建议。", "exact_items"),
    ("请列出3条通勤建议。", "exact_items"),
    (V18_PLAN_INPUT, "activity_plan"),
    ("我想骑车但在下雨，请给我一条简单安排。", "activity_plan"),
    ("我在想，为什么旧照片会让人同时开心又难过？", "general_explanation"),
    ("为什么搬家让人同时期待又难过？", "general_explanation"),
    ("刚整理完桌上的一堆旧票根。", None),
    ("我今天有点累，先让我说说，不用急着给建议。", None),
    ("朋友说‘给我两条通勤建议’。", None),
    ("如果有空，给我两条通勤建议。", None),
    ("别给我两条通勤建议。", None),
    ("给我两条建议，再写个故事。", None),
    ("我想散步和骑车但在下雨，请给我一个简单安排。", None),
    ("我想去散步但在下雨，也想听歌，请给我一个简单安排。", None),
    ("为什么旧照片会让我同时开心又难过？", None),
    ("请解释我为什么舍不得这些照片。", None),
])
def test_v19_task_recognition_is_bounded_to_explicit_request_shapes(text, kind):
    from virtual_companion.cognition.user_constraints import recognize_response_task

    contract = recognize_response_task(text)
    assert (None if contract is None else contract.kind) == kind
    if contract is not None:
        assert contract.span == text and contract.requires_complete_answer
        assert contract.to_payload()["source"] == "user_text"


def packet_for(utterance, user_text="只回答一个字，不要说别的。", shown="", final=False):
    payload = {"candidate_prefix": utterance, "user_text": user_text,
               "displayed_prefix": shown, "observed_text": utterance, "final": final}
    return {"payload": payload, "sha256": digest(payload)}


def verdict(packet, state="complete", evidence=None):
    text = packet["payload"]["candidate_prefix"]
    return {"utterance_state": state, "evidence_span": evidence if evidence is not None else text}


def test_compact_assessor_never_asks_the_model_for_the_host_hash():
    seen = []

    def invoke(stage, messages, schema, cancelled):
        seen.append((stage, messages, schema, cancelled))
        return verdict(packet_for("好"))

    assessor = CompactUnitAssessor(invoke, Event())
    packet = packet_for("好")
    result = assessor(packet)
    schema = seen[0][2]
    assert set(schema["properties"]) == {"utterance_state", "evidence_span"}
    # Evidence is optional everywhere; the host binds by call identity when absent.
    assert schema["required"] == ["utterance_state"]
    assert "request_sha256" not in canonical(schema)
    assert packet["sha256"] not in canonical(seen[0][1]), "host hash must not be echoed"
    assert result == {"request_sha256": packet["sha256"], "decision": "approve",
                      "self_contained": True, "no_open_dependency": True,
                      "consistent_with_displayed": True, "constraints_satisfied": True}
    assert seen[0][0] == "stream_unit_independent"


def test_compact_assessor_is_smaller_than_the_hash_echoing_contract():
    from virtual_companion.cognition.independent_unit_assessment import IndependentUnitAssessor

    schemas = {}

    def invoke(stage, messages, schema, cancelled):
        schemas["compact"] = canonical(schema)
        return verdict(packet_for("好"))

    CompactUnitAssessor(invoke, Event())(packet_for("好"))

    def foreign(stage, messages, schema, cancelled):
        schemas["full"] = canonical(schema)
        return {"request_sha256": schema["properties"]["request_sha256"]["const"],
                "utterance_state": "complete", "evidence_span": "好"}

    IndependentUnitAssessor(foreign, Event())(packet_for("好"))
    assert len(schemas["compact"]) < len(schemas["full"])
    assert "64-hex" not in schemas["compact"]


@pytest.mark.parametrize("fault", [
    "extra_field", "unknown_state", "foreign_evidence", "blank_evidence"])
def test_compact_assessor_rejects_unbound_or_stale_output(fault):
    def invoke(stage, messages, schema, cancelled):
        value = verdict(packet_for("好"))
        if fault == "extra_field":
            value["request_sha256"] = "0" * 64
        elif fault == "unknown_state":
            value["utterance_state"] = "maybe"
        elif fault == "foreign_evidence":
            value["evidence_span"] = "另一个候选的证据"
        else:
            value["evidence_span"] = "   "
        return value

    assessor = CompactUnitAssessor(invoke, Event())
    if fault in {"foreign_evidence", "blank_evidence"}:
        # Unusable volunteered evidence is recorded and bound by call identity.
        assert assessor(packet_for("好"))["decision"] == "approve"
        assert assessor.last_evidence == "mismatch"
        return
    with pytest.raises(ValueError):
        assessor(packet_for("好"))

    def omitted(stage, messages, schema, cancelled):
        value = verdict(packet_for("好"))
        del value["evidence_span"]
        return value

    # An omitted span is bound by call identity; a foreign span is always refused.
    assert CompactUnitAssessor(omitted, Event())(
        packet_for("好", final=False))["decision"] == "approve"
    assert CompactUnitAssessor(omitted, Event())(
        packet_for("好", final=True))["decision"] == "approve"


def test_compact_assessor_respects_cancellation():
    cancelled = Event()

    def invoke(stage, messages, schema, flag):
        if flag.is_set():
            raise InterruptedError("model cancelled")
        return verdict(packet_for("好"))

    assessor = CompactUnitAssessor(invoke, cancelled)
    cancelled.set()
    with pytest.raises(InterruptedError):
        assessor(packet_for("好"))


def test_bounded_reviewer_accepts_the_compact_assessor(tmp_path):
    store = SQLiteStreamingStore(tmp_path / "compact.sqlite3")
    run = claim(store, store.accept(config(), "first", "你好"))
    reviewer = BoundedUnitReviewer(run, ShortRequest("你好"),
        CompactUnitAssessor(lambda stage, messages, schema, cancelled: verdict(
            packet_for("甲句。", "你好"), "complete", "甲句。"), Event()))
    result = reviewer("甲句。", False)
    assert result is not None and result.candidate.text == "甲句。"
    assert result.review.model_approved is True


@pytest.mark.parametrize("text,expected", [
    ("只回答一个字，不要说别的。", (("exact_characters", 1),)),
    ("只回答两个字", (("exact_characters", 2),)),
    ("最多5个字", (("max_characters", 5),)),
    ("不超过十个字，谢谢", (("max_characters", 10),)),
    ("三个字以内就行", (("max_characters", 3),)),
    ("你好，今天过得怎么样？", ()),
    ("我刚刚写了三千字", ()),
])
def test_recognized_length_constraints_are_bounded_and_provenanced(text, expected):
    constraints, unparsed = recognize_user_constraints(text)
    assert tuple((item.kind, item.limit) for item in constraints) == expected
    assert not unparsed or expected
    for item in constraints:
        assert item.span in text and item.source == "user_text"


@pytest.mark.parametrize("text", ["用一句话回答我", "尽量简短一点", "只回答我就好", "不要超过"])
def test_constraint_like_phrases_without_a_parse_stay_conservative(text):
    constraints, unparsed = recognize_user_constraints(text)
    assert constraints == () and unparsed, text


def test_character_count_and_issue_checking_use_the_display_surface():
    constraint = UserConstraint("exact_characters", 1, "只回答一个字")
    assert character_count("好") == 1
    # A trailing sentence mark adds no content and is not counted.
    assert character_count("好。") == 1
    assert character_count("好，我明白。") == 5
    assert constraint_issues("好", (constraint,)) == ()
    assert constraint_issues("好。", (constraint,)) == ()
    assert constraint_issues("好的", (constraint,)) == ("exact_characters",)
    maximum = UserConstraint("max_characters", 3, "最多3个字")
    assert constraint_issues("三个字", (maximum,)) == ()
    assert constraint_issues("四个字呢", (maximum,)) == ("max_characters",)


def test_unbound_early_opinion_fails_without_becoming_a_semantic_hold(tmp_path):
    store = SQLiteStreamingStore(tmp_path / "unbound.sqlite3")
    run = claim(store, store.accept(config(), "first", "你好"))
    calls = []

    def sometimes_unbound(packet):
        calls.append(packet["payload"]["final"])
        if not packet["payload"]["final"]:
            raise ValueError("Classification evidence absent from observed utterance")
        return {"request_sha256": packet["sha256"], "decision": "approve",
                "self_contained": True, "no_open_dependency": True,
                "consistent_with_displayed": True, "constraints_satisfied": True}

    reviewer = BoundedUnitReviewer(run, ShortRequest("你好"), sometimes_unbound)
    with pytest.raises(ValueError, match="Classification evidence absent"):
        reviewer("甲句。", False)
    assert reviewer.approved_text == ""
    assert calls == [False]


def test_unbound_final_opinion_still_fails_closed(tmp_path):
    store = SQLiteStreamingStore(tmp_path / "unbound-final.sqlite3")
    run = claim(store, store.accept(config(), "first", "你好"))

    def unbound(packet):
        raise ValueError("Classification evidence absent from observed utterance")

    reviewer = BoundedUnitReviewer(run, ShortRequest("你好"), unbound)
    with pytest.raises(ValueError, match="evidence"):
        reviewer("甲句。", True)
    assert reviewer.approved_text == ""


def test_compact_review_evidence_is_capped_and_still_bound():
    seen = []

    def invoke(stage, messages, schema, cancelled):
        seen.append(schema)
        return {"utterance_state": "complete", "evidence_span": "好"}

    assessor = CompactUnitAssessor(invoke, Event())
    assessor(packet_for("好"))
    evidence = seen[0]["properties"]["evidence_span"]
    assert evidence["maxLength"] == 8 and evidence["minLength"] == 1
    assert "最短" in evidence.get("description", "")

    def verbose(stage, messages, schema, cancelled):
        text = "甲" * 40
        return {"utterance_state": "complete", "evidence_span": text}

    with pytest.raises(ValueError):
        CompactUnitAssessor(verbose, Event())(packet_for("甲" * 40))


def test_length_constraint_withholds_early_units_and_fails_on_violation(tmp_path):
    store = SQLiteStreamingStore(tmp_path / "constraint.sqlite3")
    run = claim(store, store.accept(config(), "first", "只回答一个字，不要说别的。"))
    calls = []

    def approve(packet):
        calls.append(packet["payload"]["candidate_prefix"])
        return {"request_sha256": packet["sha256"], "decision": "approve",
                "self_contained": True, "no_open_dependency": True,
                "consistent_with_displayed": True, "constraints_satisfied": True}

    reviewer = BoundedUnitReviewer(run, ShortRequest("只回答一个字，不要说别的。"), approve)
    assert [item.kind for item in reviewer.constraints] == ["exact_characters"]
    # A whole-answer constraint cannot be verified early, so nothing is offered.
    assert reviewer(LONG + "\n\n", False) is None and calls == []
    with pytest.raises(UserConstraintViolation, match="characters"):
        reviewer(LONG, True)
    assert calls == [], "the host decides this without consulting the model"


def test_satisfied_length_constraint_is_published_normally(tmp_path):
    store = SQLiteStreamingStore(tmp_path / "satisfied.sqlite3")
    run = claim(store, store.accept(config(), "first", "只回答一个字，不要说别的。"))

    def approve(packet):
        return {"request_sha256": packet["sha256"], "decision": "approve",
                "self_contained": True, "no_open_dependency": True,
                "consistent_with_displayed": True, "constraints_satisfied": True}

    reviewer = BoundedUnitReviewer(run, ShortRequest("只回答一个字，不要说别的。"), approve)
    result = reviewer("好", True)
    assert result is not None and result.candidate.text == "好"


def test_unparsed_constraint_like_phrase_only_withholds_early_units(tmp_path):
    store = SQLiteStreamingStore(tmp_path / "unparsed.sqlite3")
    run = claim(store, store.accept(config(), "first", "用一句话回答我"))
    calls = []

    def approve(packet):
        calls.append(packet["payload"]["candidate_prefix"])
        return {"request_sha256": packet["sha256"], "decision": "approve",
                "self_contained": True, "no_open_dependency": True,
                "consistent_with_displayed": True, "constraints_satisfied": True}

    reviewer = BoundedUnitReviewer(run, ShortRequest("用一句话回答我"), approve)
    assert reviewer.unparsed_constraints and reviewer.constraints == ()
    assert reviewer("甲句。", False) is None and calls == []
    result = reviewer("甲句。乙句。", True)
    assert result is not None and calls == ["甲句。乙句。"]


def test_coordinator_fails_conservatively_without_publishing_a_violating_answer(tmp_path):
    store = SQLiteStreamingStore(tmp_path / "violating.sqlite3")
    run = claim(store, store.accept(config(), "first", "只回答一个字，不要说别的。"))
    seen = []

    def approve(packet):
        return {"request_sha256": packet["sha256"], "decision": "approve",
                "self_contained": True, "no_open_dependency": True,
                "consistent_with_displayed": True, "constraints_satisfied": True}

    class Model:
        def __init__(self):
            self.receive = None

        def count(self, body):
            return 10

        def cancel(self):
            pass

        def invoke(self, body):
            self.receive(UnapprovedTextDelta(1, LONG))
            return {"text": LONG, "tool_calls": [], "finish_reason": "stop"}

    model = Model()
    reviewer = BoundedUnitReviewer(run, ShortRequest("只回答一个字，不要说别的。"), approve)
    dialogue = StreamDialogue(
        store=store, model_factory=lambda receive: (setattr(model, "receive", receive), model)[1],
        review_prefix=reviewer, present=lambda event: seen.append(event) or "host-1",
        checkpoint=lambda: None, cancelled=Event(), clock=lambda: 100, parallel_review=True)
    with pytest.raises(UserConstraintViolation):
        dialogue.generate(run, ShortRequest("只回答一个字，不要说别的。"))
    state = store.stream_state(A, run.input_id)
    assert state["outcome"] == "failed"
    assert state["approved_units"] == state["confirmed_units"] == 0
    assert seen == []


def test_unconstrained_short_reply_still_streams_early(tmp_path):
    store = SQLiteStreamingStore(tmp_path / "free.sqlite3")
    run = claim(store, store.accept(config(), "first", "陪我聊两句。"))
    calls = []

    def approve(packet):
        calls.append(packet["payload"]["candidate_prefix"])
        return {"request_sha256": packet["sha256"], "decision": "approve",
                "self_contained": True, "no_open_dependency": True,
                "consistent_with_displayed": True, "constraints_satisfied": True}

    reviewer = BoundedUnitReviewer(run, ShortRequest("陪我聊两句。"), approve)
    assert reviewer.constraints == () and reviewer.unparsed_constraints == ()
    assert reviewer("好呀。", False) is not None
    assert calls == ["好呀。"]


SCOPED_BREAKFAST_INPUT = "请列出三种早餐，每种不超过十个字。"
SCOPED_BREAKFAST_REPLY = "1. 燕麦粥配水煮蛋\n2. 全麦面包夹生菜\n3. 豆浆配小笼包"


@pytest.mark.parametrize("text,scope", [
    (SCOPED_BREAKFAST_INPUT, "每种不超过十个字"),
    ("分两段回答，每段只回答三个字。", "每段只回答三个字"),
    ("标题不超过5个字，正文详细说明。", "标题不超过5个字"),
    ("不超过十个字的标题，再写正文。", "不超过十个字的标题"),
    ("他说“只回答三个字”。", "他说“只回答三个字”"),
])
def test_scoped_length_clauses_keep_their_original_current_user_scope(text, scope):
    constraints, unparsed = recognize_user_constraints(text)
    assert constraints == ()
    assert unparsed == (scope,)
    assert scope in text


def test_scoped_length_and_explicit_whole_reply_limit_are_distinct():
    text = "每项不超过五个字，全文不超过十个字。"
    constraints, unparsed = recognize_user_constraints(text)
    assert [(item.kind, item.limit) for item in constraints] == [("max_characters", 10)]
    assert constraints[0].span == "不超过十个字" and constraints[0].source == "user_text"
    assert unparsed == ("每项不超过五个字",)
    assert constraint_issues(SCOPED_BREAKFAST_REPLY, constraints) == ("max_characters",)


@pytest.mark.parametrize("revision", ["short-dialogue.v12", "short-dialogue.v26"])
def test_scoped_length_original_breakfast_reaches_one_review_and_bound_ack(
    tmp_path, sealed_capture, revision
):
    runtime, session, memory, mirror, transport = recording_runtime(
        tmp_path, sealed_capture, revision=revision,
        actions=[{"action": "answer", "text": SCOPED_BREAKFAST_REPLY}],
    )
    try:
        attempt = session.send(SCOPED_BREAKFAST_INPUT)
        event = finish(runtime, session)
        assert len(transport.calls) == 2 and len(transport.sealed) == 1
        assert [options["stage"] for _, options in transport.calls] == [
            "demand_control", "stream_unit_review"
        ]
        envelope = transport.sealed[0]
        assert digest(envelope["payload"]) == envelope["sha256"]
        packet = envelope["payload"]
        assert packet["user_text"] == SCOPED_BREAKFAST_INPUT
        assert packet["constraints"] == []
        assert packet["unparsed_constraints"] == ["每种不超过十个字"]
        assert packet["final"] is True and packet["displayed_prefix"] == ""
        assert packet["candidate_prefix"] == SCOPED_BREAKFAST_REPLY
        assert packet["basis"]["input_id"] == event["basis"]["input_id"] == runtime.claim.input_id
        assert packet["basis"]["attempt_id"] == event["basis"]["attempt_id"]
        assert packet["candidate_sha256"] == event["presentation"]["review"]["candidate_sha256"]
        current = next(
            source for source in packet["personal_sources"] if source["kind"] == "current"
        )
        assert current["text"] == SCOPED_BREAKFAST_INPUT
        assert current["source_id"] == runtime.claim.input_id
        wire_view = json.loads(transport.calls[1][0]["messages"][-1]["content"])
        assert wire_view == semantic_review_view(packet)
        assert wire_view["unparsed_constraints"] == packet["unparsed_constraints"]
        assert event["sequence"] == 1 and event["text"] == SCOPED_BREAKFAST_REPLY
        assert mirror.history()["messages"][-1]["content"] == SCOPED_BREAKFAST_REPLY
        saved = memory.store.segments(runtime.companion, runtime.claim.input_id)
        assert len(saved) == 1 and saved[0]["status"] == "confirmed"
        write_record(f"scoped-breakfast-{revision}", {
            "cpu_scope": "Declared approve tests wiring and ACK; not model semantic quality",
            "attempt": attempt, "requests": recorded_calls(transport), "sealed_packet": envelope,
            "declared_opinions": transport.opinions,
            "semantic_view": wire_view, "dispatch": event, "segments": saved,
            "history": mirror.history(), "terminal": session.snapshot(),
        })
    finally:
        memory.close()
        mirror.close()


def test_scoped_length_declared_item_violation_uses_existing_reject_repair_limit(
    tmp_path, sealed_capture
):
    from test_default_text_delivery import wait_for

    reply = "1. 一碗热燕麦粥配两个水煮鸡蛋\n2. 全麦面包夹生菜\n3. 豆浆配小笼包"
    runtime, session, memory, mirror, transport = recording_runtime(
        tmp_path, sealed_capture, revision="short-dialogue.v26",
        actions=[{"action": "answer", "text": reply}, {"action": "answer", "text": reply}],
        review_options=[{"decision": "reject", "constraints_satisfied": False}] * 2,
    )
    try:
        session.send(SCOPED_BREAKFAST_INPUT)
        wait_for(lambda: not session.snapshot()["busy"])
        assert session.snapshot()["attempt"]["status"] == "failed"
        assert [options["stage"] for _, options in transport.calls] == [
            "demand_control", "stream_unit_review", "demand_repair", "stream_unit_review"
        ]
        assert len(transport.sealed) == 2 and not transport.actions
        for envelope in transport.sealed:
            assert digest(envelope["payload"]) == envelope["sha256"]
            packet = envelope["payload"]
            assert packet["candidate_prefix"] == reply and packet["constraints"] == []
            assert packet["unparsed_constraints"] == ["每种不超过十个字"]
        assert runtime.poll()["event"] is None
        assert memory.store.segments(runtime.companion, runtime.claim.input_id) == []
        assert [row["role"] for row in mirror.history()["messages"]] == ["user"]
        write_record("scoped-item-declared-reject", {
            "cpu_scope": "Two declared reject opinions; not model semantic quality",
            "requests": recorded_calls(transport), "sealed_packets": transport.sealed,
            "declared_opinions": transport.opinions,
            "terminal": session.snapshot(), "history": mirror.history(),
        })
    finally:
        memory.close()
        mirror.close()


@pytest.mark.parametrize("user", ["只回答十个字。", "请列出三种早餐，全文不超过十个字。"])
def test_scoped_length_true_whole_reply_count_still_rejects_before_review(
    tmp_path, sealed_capture, user
):
    from test_default_text_delivery import wait_for

    runtime, session, memory, mirror, transport = recording_runtime(
        tmp_path, sealed_capture, revision="short-dialogue.v26",
        actions=[{"action": "answer", "text": SCOPED_BREAKFAST_REPLY}],
    )
    try:
        session.send(user)
        wait_for(lambda: not session.snapshot()["busy"])
        assert session.snapshot()["attempt"]["status"] == "failed"
        assert len(transport.calls) == 1 and transport.calls[0][1]["stage"] == "demand_control"
        assert transport.sealed == [] and runtime.poll()["event"] is None
        assert memory.store.segments(runtime.companion, runtime.claim.input_id) == []
        assert [row["role"] for row in mirror.history()["messages"]] == ["user"]
    finally:
        memory.close()
        mirror.close()
