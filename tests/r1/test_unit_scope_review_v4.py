"""Whole-unit review wiring and declared opinions; never model-quality scores.

The controlled supplier states its interpretation explicitly. These recordings
prove which sealed words and sources reach the one reviewer, and how the host
binds its decision. A false ``none`` or ``supported`` remains a semantic failure.
"""

import json
import os
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from threading import Event

import pytest
from jsonschema import Draft202012Validator
from test_default_text_delivery import setup, wait_for
from test_natural_context_responsibility import Counter
from test_rebuilt_turns import claim, config
from test_stream_unit_review import protocol_packet
from virtual_companion.boundary.contracts import canonical
from virtual_companion.boundary.presentation import digest
from virtual_companion.boundary.sqlite_streaming import SQLiteStreamingStore
from virtual_companion.cognition import stream_unit_review
from virtual_companion.cognition.short_dialogue import ShortRequest
from virtual_companion.cognition.stream_unit_review import (
    BoundedUnitReviewer,
    CandidateRejected,
    StructuredUnitAssessor,
    candidate_segments,
    semantic_review_view,
)
from virtual_companion.providers.contracts import ProviderCapabilities, ProviderRoute
from virtual_companion.text import default_runtime
from virtual_companion.text.application import TextSession
from virtual_companion.text.default_runtime import DefaultTextRuntime
from virtual_companion.text.fake import FakeModel, synthetic_unit_opinion
from virtual_companion.text.provider_model import ProviderModel

REVISION = "short-dialogue.v26"
PARTICIPATION_REVISION = "short-dialogue.v27"
WHOLE_UNIT_REVISIONS = (REVISION, PARTICIPATION_REVISION)
AUDIT = "candidate-unit-scope.v4"
BASE = dict(decision="approve", self_contained=True, no_open_dependency=True,
            consistent_with_displayed=True, constraints_satisfied=True)


def sealed_v4(user, candidate):
    envelope = protocol_packet(user_text=user, candidate=candidate)
    envelope["payload"]["audit_version"] = AUDIT
    envelope["sha256"] = digest(envelope["payload"])
    return envelope


def supplied_opinion(payload, *, personal="none", personal_ids=(), fiction="none",
                     fiction_ids=(), **checks):
    """Test-author interpretation, with no classifier or word-based fallback."""
    return {**BASE, **checks, "audit_version": AUDIT, "units": [
        {"segment_id": item["segment_id"], "personal_support": personal,
         "personal_source_ids": list(personal_ids), "fiction_support": fiction,
         "fiction_source_ids": list(fiction_ids)} for item in payload["review_segments"]
    ]}


def assessor(invoke, cancelled=None):
    return StructuredUnitAssessor(invoke, cancelled or Event(), audit_version=AUDIT)


def record(label, value):
    directory = os.environ.get("P103_UNIT_SCOPE_RECORD_DIR")
    if directory:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / (label + ".json")).write_text(canonical(value) + "\n", encoding="utf-8")


@pytest.fixture
def sealed_capture(monkeypatch):
    packets = {}
    original = StructuredUnitAssessor.__call__

    def capture(instance, envelope):
        assert digest(envelope["payload"]) == envelope["sha256"]
        view = semantic_review_view(envelope["payload"])
        packets[canonical(view)] = deepcopy(envelope)
        return original(instance, envelope)

    monkeypatch.setattr(StructuredUnitAssessor, "__call__", capture)
    return packets


def captured_envelope(messages, packets):
    view = json.loads(messages[-1]["content"])
    envelope = deepcopy(packets[canonical(view)])
    assert digest(envelope["payload"]) == envelope["sha256"]
    assert semantic_review_view(envelope["payload"]) == view
    return envelope


def test_v4_wire_reviews_complete_units_once_without_free_claim_or_identity_recopies():
    envelope = sealed_v4("刚整理完桌上的一堆旧票根。", "整理完一堆旧票根，桌面应该清爽不少。")
    payload, calls = envelope["payload"], []
    opinion = supplied_opinion(payload, personal="uncertain", decision="reject",
                               constraints_satisfied=False)

    def invoke(stage, messages, schema, cancelled):
        calls.append({"stage": stage, "messages": deepcopy(messages), "schema": schema})
        assert not cancelled.is_set()
        view = json.loads(messages[-1]["content"])
        assert view["view_version"] == "unit-scope-semantics.v1"
        assert view["units"] == payload["review_segments"]
        assert "".join(item["text"] for item in view["units"]) == payload["candidate_prefix"]
        assert view["sources"] == [{key: source[key] for key in ("id", "kind", "text")}
                                   for source in payload["personal_sources"]]
        properties = schema["properties"]["units"]["items"]["properties"]
        assert set(properties) == {"segment_id", "personal_support", "personal_source_ids",
                                   "fiction_support", "fiction_source_ids"}
        assert not ({"claim", "quote", "owner", "role", "revision", "start", "end"}
                    & properties.keys())
        Draft202012Validator(schema).validate(opinion)
        return deepcopy(opinion)

    result = assessor(invoke)(envelope)
    assert len(calls) == 1 and calls[0]["stage"] == "stream_unit_review"
    assert result["request_sha256"] == envelope["sha256"]
    assert result["model_opinion"] == opinion and "propositions" not in result
    assert result["unit_scopes"] == [dict(candidate_segments(payload["candidate_prefix"])[0],
                                         **opinion["units"][0])]
    record("whole-unit-wire", {"envelope": envelope, "calls": calls, "opinion": opinion,
                               "bound_verdict": result, "semantic_effect_proven": False})


PAIRS = (
    ("quiet", "最近总觉得房间很安静，想聊聊这种感觉。",
     "安静时，冰箱、水管等小声音可能更明显。", "不是你的房间变了，是分走你注意的东西少了。"),
    ("plan", "我想去散步但在下雨，请给我一个简单安排。",
     "雨小且路面安全时，就散步十分钟再回来。", "你家门口有带顶棚的步道，雨天就去那里。"),
    ("mixed", "刚整理完桌上的一堆旧票根。",
     "票根能把一个日期留在手边。", "整理完一堆旧票根，桌面应该清爽不少。"),
    ("question", "从抽屉里翻出几张旧公交卡。",
     "里面有没有一张让你想到某次出门？", "边角都磨亮了，是你常走那条路的缘故吗？"),
)


@pytest.mark.parametrize("label,user,general,personal", PAIRS, ids=[item[0] for item in PAIRS])
def test_same_sources_general_or_conditional_content_and_personal_premise_are_distinct(
    tmp_path, sealed_capture, label, user, general, personal
):
    store = SQLiteStreamingStore(tmp_path / (label + ".sqlite3"))
    run = claim(store, store.accept(config(), "scope-input", user))
    recordings = []
    for text, support in ((general, "none"), (personal, "unsupported")):

        def invoke(stage, messages, schema, cancelled, *, support=support):
            envelope = captured_envelope(messages, sealed_capture)
            value = supplied_opinion(envelope["payload"], personal=support,
                                     **({"decision": "reject", "constraints_satisfied": False}
                                        if support != "none" else {}))
            Draft202012Validator(schema).validate(value)
            recordings.append({"envelope": envelope, "messages": messages, "opinion": value})
            return value

        reviewer = BoundedUnitReviewer(run, ShortRequest(user), assessor(invoke),
                                      audit_version=AUDIT)
        if support == "none":
            assert reviewer(text, True).candidate.text == text
        else:
            with pytest.raises(CandidateRejected):
                reviewer(text, True)
            assert reviewer.approved_text == ""
    assert recordings[0]["envelope"]["payload"]["personal_sources"] == (
        recordings[1]["envelope"]["payload"]["personal_sources"])
    record(label, {"recordings": recordings, "semantic_effect_proven": False})


@pytest.mark.parametrize("support", ["supported", "unsupported", "uncertain"])
def test_support_is_for_whole_unit_and_preserves_original_opinion(support):
    envelope = sealed_v4("我那几张公交卡边角磨亮了。", "你提到卡的边角磨亮了。")
    source_id = envelope["payload"]["personal_sources"][0]["id"]
    value = supplied_opinion(envelope["payload"], personal=support, personal_ids=[source_id],
                             **({"decision": "reject", "constraints_satisfied": False}
                                if support != "supported" else {}))
    result = assessor(lambda *args: deepcopy(value))(envelope)
    assert result["model_opinion"] == value
    assert result["decision"] == ("approve" if support == "supported" else "reject")
    assert result["unit_scopes"][0]["text"] == envelope["payload"]["candidate_prefix"]
    assert result["unit_scopes"][0]["personal_source_ids"] == [source_id]


@pytest.mark.parametrize("current", [True, False])
def test_fiction_authorization_is_current_source_relation_not_old_permission(current):
    envelope = sealed_v4("请写一个虚构的雨天放学片段。", "虚构一幕：孩子踩过水洼。")
    source = envelope["payload"]["personal_sources"][0]
    if not current:
        envelope["payload"]["personal_sources"].append({
            **source, "id": "history:old-fiction", "kind": "history", "source_id": "old-fiction"})
        source = envelope["payload"]["personal_sources"][-1]
        envelope["sha256"] = digest(envelope["payload"])
    opinion = supplied_opinion(envelope["payload"], fiction="authorized",
                               fiction_ids=[source["id"]])
    if current:
        assert assessor(lambda *args: opinion)(envelope)["decision"] == "approve"
    else:
        with pytest.raises(ValueError, match="current"):
            assessor(lambda *args: opinion)(envelope)


def test_false_none_stays_a_semantic_counterexample_not_host_word_classification():
    envelope = sealed_v4("今天雨很大，我想起小时候放学的路。", "你平时走惯了那条路，鞋里总会进水。")
    false = supplied_opinion(envelope["payload"])
    assert assessor(lambda *args: false)(envelope)["decision"] == "approve"
    # This records the protocol's limit; it does not claim the interpretation is correct.
    record("false-none-counterexample", {"envelope": envelope, "declared_false_opinion": false,
                                        "semantic_effect_proven": False})


@pytest.mark.parametrize("mutation", [
    "omit-unit", "repeat-unit", "foreign-unit", "foreign-source", "duplicate-source",
    "free-claim", "old-version", "supported-without-source", "none-with-source",
])
def test_malformed_cross_source_or_inconsistent_v4_results_never_approve(mutation):
    envelope = sealed_v4("刚整理完桌上的一堆旧票根。", "票根能留住日期。纸张也会变旧。")
    value = supplied_opinion(envelope["payload"])
    first = value["units"][0]
    source_id = envelope["payload"]["personal_sources"][0]["id"]
    if mutation == "omit-unit":
        value["units"].pop()
    elif mutation == "repeat-unit":
        value["units"][1] = deepcopy(first)
    elif mutation == "foreign-unit":
        first["segment_id"] = "foreign-candidate-unit"
    elif mutation == "foreign-source":
        first.update(personal_support="supported", personal_source_ids=["memory:other-owner"])
    elif mutation == "duplicate-source":
        first.update(personal_support="supported", personal_source_ids=[source_id, source_id])
    elif mutation == "free-claim":
        first["claim"] = "刚整理完桌上的票根"
    elif mutation == "old-version":
        value["audit_version"] = "candidate-propositions.v3"
    elif mutation == "supported-without-source":
        first["personal_support"] = "supported"
    elif mutation == "none-with-source":
        first["personal_source_ids"] = [source_id]
    with pytest.raises(ValueError):
        assessor(lambda *args: value)(envelope)


@pytest.mark.parametrize("adverse", ["personal-unsupported", "personal-uncertain",
                                   "fiction-unauthorized", "fiction-uncertain"])
def test_contradictory_semantic_approval_normalizes_to_reject_and_keeps_original(adverse):
    envelope = sealed_v4("今天在下雨。", "你每天沿这条路放学。")
    prefix, support = adverse.split("-")
    value = supplied_opinion(envelope["payload"])
    value["units"][0][prefix + "_support"] = support
    result = assessor(lambda *args: deepcopy(value))(envelope)
    assert result["decision"] == "reject" and result["constraints_satisfied"] is False
    assert result["model_opinion"] == value and value["decision"] == "approve"


@pytest.mark.parametrize("check,decision", [("self_contained", "hold"),
                                          ("no_open_dependency", "hold"),
                                          ("constraints_satisfied", "reject"),
                                          ("consistent_with_displayed", "reject")])
def test_none_does_not_bypass_whole_candidate_checks(check, decision):
    envelope = sealed_v4("想聊聊下雨。", "雨声可以让人想到一些事。")
    value = supplied_opinion(envelope["payload"], **{check: False})
    result = assessor(lambda *args: deepcopy(value))(envelope)
    assert result["decision"] == decision
    assert result["model_opinion"] == value


def test_packet_and_bound_verdict_keep_revision_and_candidate_identity():
    envelope = sealed_v4("我提到公交卡。", "卡片能留下一个时间。")
    value = supplied_opinion(envelope["payload"])
    result = assessor(lambda *args: deepcopy(value))(envelope)
    foreign = deepcopy(envelope)
    foreign["payload"]["basis"]["state_version"] += 1
    foreign["payload"]["personal_sources"][0]["revision"] += 1
    with pytest.raises(ValueError, match="packet"):
        assessor(lambda *args: value)(foreign)
    foreign["sha256"] = digest(foreign["payload"])
    with pytest.raises(ValueError, match="Foreign"):
        BoundedUnitReviewer._validate_opinion(result, foreign["sha256"], foreign["payload"])
    altered = deepcopy(result)
    altered["unit_scopes"][0]["text"] = "替换的候选"
    with pytest.raises(ValueError):
        BoundedUnitReviewer._validate_opinion(altered, envelope["sha256"], envelope["payload"])


def test_v3_schema_wire_and_original_opinion_remain_the_comparison_path():
    envelope = protocol_packet(user_text="今天在下雨。", candidate="雨声可以让人想起一些事。")
    expected = synthetic_unit_opinion(envelope["payload"])
    calls = []

    def invoke(stage, messages, schema, cancelled):
        calls.append((messages, schema))
        assert json.loads(messages[-1]["content"]) == semantic_review_view(envelope["payload"])
        assert schema["properties"]["audit_version"] == {"const": "candidate-propositions.v3"}
        assert "segments" in schema["properties"] and "units" not in schema["properties"]
        return deepcopy(expected)

    result = StructuredUnitAssessor(invoke, Event())(envelope)
    assert len(calls) == 1 and "audit_version" not in envelope["payload"]
    assert result["audit_version"] == "candidate-propositions.v3"
    assert "propositions" in result and "unit_scopes" not in result
    assert result["model_opinion"] == {key: expected[key] for key in BASE}


class RecordingTransport:
    """One real ProviderModel port, with explicit CPU opinions and no network."""

    def __init__(self, packets, actions, review_options=()):
        self.packets, self.actions = packets, deepcopy(actions)
        self.review_options = deepcopy(list(review_options))
        self.calls, self.sealed, self.opinions = [], [], []

    def invoke(self, body, cancelled, **options):
        assert not cancelled.is_set() and options["input_tokens"] == 17
        self.calls.append((deepcopy(body), deepcopy(options)))
        if options["stage"] == "stream_unit_review":
            envelope = captured_envelope(body["messages"], self.packets)
            self.sealed.append(envelope)
            supplied = self.review_options.pop(0) if self.review_options else {}
            if envelope["payload"].get("audit_version") == "candidate-unit-verdict.v8":
                value = supplied_v8(envelope["payload"], **supplied)
            elif envelope["payload"].get("audit_version") == "candidate-unit-meaning.v7":
                meaning = supplied.pop("meaning", "CPU供应商录制完整合成单元的含义，不证明忠实性。")
                use = supplied.pop("use", "CPU供应商录制当前回应关系。")
                fit = supplied.pop("response_fit", "fits")
                value = supplied_opinion(envelope["payload"], **supplied)
                value["audit_version"] = "candidate-unit-meaning.v7"
                for item in value["units"]:
                    item.update(meaning=meaning, use=use, response_fit=fit)
            elif envelope["payload"].get("audit_version") == "candidate-unit-grounding.v6":
                meaning = supplied.pop(
                    "meaning", "CPU供应商声明此完整合成单元无个人前提；不证明语义准确。")
                value = supplied_opinion(envelope["payload"], **supplied)
                value["audit_version"] = "candidate-unit-grounding.v6"
                for item in value["units"]:
                    item["meaning"] = meaning
            elif envelope["payload"].get("audit_version") == AUDIT:
                value = supplied_opinion(envelope["payload"], **supplied)
            else:
                value = synthetic_unit_opinion(envelope["payload"], **supplied)
            self.opinions.append(deepcopy(value))
        else:
            assert self.actions, "The recorded bounded generation sequence is exhausted"
            value = self.actions.pop(0)
        Draft202012Validator(body["response_format"]["json_schema"]["schema"]).validate(value)
        return {"text": canonical(value), "tool_calls": [], "finish_reason": "stop",
                "usage": {"prompt_tokens": 17, "completion_tokens": 5, "total_tokens": 22}}

    def cancel(self):
        pass


def recording_runtime(tmp_path, packets, *, revision=REVISION, actions, review_options=(),
                      output_tokens=512):
    transport = RecordingTransport(packets, actions, review_options)
    route = ProviderRoute(
        "synthetic-unit-scope", 1, "local", "http://127.0.0.1:9/v1/chat/completions",
        "synthetic-recording-only", "none", None, True,
        ("current", "history", "valid_memory", "derived"),
        ProviderCapabilities(True, False, True, 1_048_576, "max_tokens"),
    )
    model = ProviderModel(route, counter=Counter(), transport=transport,
                          output_tokens=output_tokens)
    _, _, memory, mirror, _ = setup(tmp_path, model)
    runtime = DefaultTextRuntime(memory, mirror, model, binding={"test": "synthetic"},
                                 delivery_timeout=0.8, prompt_revision=revision)
    session = TextSession(model, mirror, {}, prepare_turn=runtime.prepare_turn,
                          before_start=runtime.before_start)
    return runtime, session, memory, mirror, transport


def recorded_calls(transport):
    # Transport callback objects are live control wiring, not request body data.
    return [{"body": body, "body_sha256": digest(body), "stage": options["stage"],
             "input_tokens": options["input_tokens"]} for body, options in transport.calls]


def finish(runtime, session):
    event = wait_for(lambda: runtime.poll()["event"] or not session.snapshot()["busy"])
    assert isinstance(event, dict), session.snapshot()
    assert session.snapshot()["partial"] == ""
    runtime.ack({"sequence": event["sequence"], "fingerprint": event["fingerprint"],
                 "host_receipt": "synthetic-unit-scope-visible-only"})
    wait_for(lambda: not session.snapshot()["busy"])
    assert session.snapshot()["attempt"]["status"] == "completed"
    return event


@pytest.mark.parametrize(
    "revision", ["short-dialogue.v12", "short-dialogue.v25", *WHOLE_UNIT_REVISIONS],
)
def test_actual_default_prepare_review_ack_selects_v4_only_for_explicit_v26(
    tmp_path, sealed_capture, revision
):
    text, user = "雨小且路面安全时，就散步十分钟再回来。", "下雨了，请给我一个简单散步安排。"
    runtime, session, memory, mirror, transport = recording_runtime(
        tmp_path, sealed_capture, revision=revision, actions=[{"action": "answer", "text": text}])
    try:
        session.send(user)
        event = finish(runtime, session)
        assert [options["stage"] for _, options in transport.calls] == [
            "demand_control", "stream_unit_review"]
        generation, review = [body for body, _ in transport.calls]
        envelope = transport.sealed[0]
        expected_audit = AUDIT if revision in WHOLE_UNIT_REVISIONS else "candidate-propositions.v3"
        assert runtime.review_audit_version == expected_audit
        assert (envelope["payload"].get("audit_version", "candidate-propositions.v3")
                == expected_audit)
        assert transport.opinions[0]["audit_version"] == expected_audit
        schema = review["response_format"]["json_schema"]["schema"]
        assert schema["properties"]["audit_version"] == {
            "const": expected_audit}
        assert generation["messages"][-1]["content"] == user
        assert generation["max_tokens"] == 512 and review["max_tokens"] == 1536
        assert envelope["payload"]["basis"]["input_id"] == runtime.claim.input_id
        assert envelope["payload"]["basis"]["state_version"] == runtime.claim.state_version
        assert (json.loads(review["messages"][-1]["content"])
                == semantic_review_view(envelope["payload"]))
        assert event["presentation"]["candidate"]["candidate_id"]
        assert mirror.history()["messages"][-1]["content"] == text
        assert runtime.snapshot()["request_prompt_revision"] == revision
        record(revision, {"generation_body": generation, "review_body": review,
                          "generation_body_sha256": digest(generation),
                          "review_body_sha256": digest(review), "sealed_envelope": envelope,
                          "original_opinion": transport.opinions[0], "display_event": event,
                          "semantic_effect_proven": False})
    finally:
        memory.close()
        mirror.close()


@pytest.mark.parametrize("revision", ["short-dialogue.v28", "short-dialogue.v29"])
def test_v28_eight_original_units_fit_one_declared_review_and_missing_coverage_is_rejected(
    tmp_path, sealed_capture, revision
):
    text = "".join(f"合成说明{index}。" for index in range(1, 9))
    runtime, session, memory, mirror, transport = recording_runtime(
        tmp_path, sealed_capture, revision=revision, output_tokens=1024,
        actions=[{"action": "answer", "text": text}])
    try:
        session.send("请进行合成容量录制。")
        event = finish(runtime, session)
        assert [options["stage"] for _, options in transport.calls] == [
            "demand_control", "stream_unit_review"]
        assert [body["max_tokens"] for body, _ in transport.calls] == [1024, 1536]
        envelope, opinion = transport.sealed[0], transport.opinions[0]
        assert len(envelope["payload"]["review_segments"]) == len(opinion["units"]) == 8
        assert all(len(item["meaning"]) <= 128 for item in opinion["units"])
        audit_version = ("candidate-unit-meaning.v7" if revision == "short-dialogue.v29"
                         else "candidate-unit-grounding.v6")
        if revision == "short-dialogue.v29":
            assert all(len(item["use"]) <= 32 and item["response_fit"] == "fits"
                       for item in opinion["units"])
            assert (envelope["payload"]["conversation_frame"]["candidate_speaker"]
                    == "assistant")
        assert event["presentation"]["candidate"]["semantic_text"] == text
        incomplete = deepcopy(opinion)
        incomplete["units"].pop()
        with pytest.raises(ValueError, match="exactly once"):
            StructuredUnitAssessor(lambda *args: incomplete, Event(),
                audit_version=audit_version)(envelope)
        record(revision.removeprefix("short-dialogue.")+"-eight-unit-capacity", {
               "calls": recorded_calls(transport),
               "envelope": envelope, "original_opinion": opinion,
               "output_utf8_bytes": len(canonical(opinion).encode()),
               "hosted_output_tokens_proven": False, "semantic_effect_proven": False})
    finally:
        memory.close()
        mirror.close()


def test_v28_acknowledged_prefix_stays_immutable_after_supplied_final_rejection(
    tmp_path, sealed_capture
):
    shared = {"generation_calls": 0, "opinions": []}

    class SuppliedLateReject(FakeModel):
        def review_peer(self):
            return SuppliedLateReject()

        def stream(self, messages, cancelled):
            if "你只审查候选文字" not in messages[0]["content"]:
                shared["generation_calls"] += 1
                yield from super().stream(messages, cancelled)
                return
            view = json.loads(messages[-1]["content"])
            value = synthetic_unit_opinion(view)
            if shared["opinions"]:
                value["units"][-1].update(personal_support="unsupported", personal_source_ids=[],
                    meaning="供应商明确给出末单元拒绝意见，仅演练已有ACK后不可整轮修复。")
            shared["opinions"].append(deepcopy(value))
            yield canonical(value)

    model = SuppliedLateReject()
    _, _, memory, mirror, _ = setup(tmp_path, model)
    runtime = DefaultTextRuntime(memory, mirror, model, binding={"test": "cpu-v28-prefix"},
                                 prompt_revision="short-dialogue.v28")
    session = TextSession(model, mirror, {}, prepare_turn=runtime.prepare_turn,
                          before_start=runtime.before_start)
    try:
        session.send("继续作合成边界录制。")
        event = wait_for(lambda: runtime.poll()["event"])
        first_text = event["presentation"]["candidate"]["semantic_text"]
        runtime.ack({"sequence": event["sequence"], "fingerprint": event["fingerprint"],
                     "host_receipt": "synthetic-v28-prefix-visible-only"})
        wait_for(lambda: not session.snapshot()["busy"])
        assert session.snapshot()["attempt"]["status"] == "failed"
        assert shared["generation_calls"] == 1 and len(shared["opinions"]) == 2
        assert runtime.poll()["event"] is None
        segments = memory.store.segments(runtime.companion, runtime.claim.input_id)
        assert len(segments) == 1 and segments[0]["status"] == "confirmed"
        assert mirror.history()["messages"][-1]["content"] == first_text
        record("v28-immutable-prefix", {"envelopes": list(sealed_capture.values()),
               "supplier": shared, "confirmed_event": event, "persistent_segments": segments,
               "semantic_effect_proven": False})
    finally:
        if session.snapshot()["busy"]:
            session.stop()
        memory.close()
        mirror.close()


def test_v6_complete_unit_semantic_basis_precedes_support_in_one_bound_call():
    assert getattr(stream_unit_review, "GROUNDED_UNIT_AUDIT_VERSION", None) == (
        "candidate-unit-grounding.v6")
    envelope = sealed_v4("刚翻出一张旧门票。", "门票把一次入场留成了一个日期。")
    envelope["payload"]["audit_version"] = "candidate-unit-grounding.v6"
    envelope["sha256"] = digest(envelope["payload"])
    value = supplied_opinion(envelope["payload"])
    value["audit_version"] = "candidate-unit-grounding.v6"
    for item in value["units"]:
        item["meaning"] = "谈门票用途与日期留存的一般联系，不增加这位用户的入场经历。"
    calls = []

    def invoke(stage, messages, schema, cancelled):
        view = json.loads(messages[-1]["content"])
        calls.append({"messages": messages, "schema": schema, "view": view})
        assert view["view_version"] == "unit-grounding-semantics.v1"
        assert view["units"] == envelope["payload"]["review_segments"]
        fields = schema["properties"]["units"]["items"]["properties"]
        assert "meaning" in fields and "claim" not in fields and "quote" not in fields
        Draft202012Validator(schema).validate(value)
        return deepcopy(value)

    result = StructuredUnitAssessor(invoke, Event(),
        audit_version="candidate-unit-grounding.v6")(envelope)
    assert len(calls) == 1
    assert result["model_opinion"] == value and result["decision"] == "approve"
    assert result["unit_scopes"][0]["text"] == envelope["payload"]["candidate_prefix"]
    assert result["unit_scopes"][0]["meaning"] == value["units"][0]["meaning"]
    record("v6-single-call-complete-unit", {"envelope": envelope, "calls": calls,
           "opinion": value, "bound_verdict": result, "semantic_effect_proven": False})


@pytest.mark.parametrize("mutation", ["missing-meaning", "empty-meaning", "blank-meaning",
                                      "foreign-unit", "foreign-source",
                                      "unsupported", "uncertain", "false-none"])
def test_v6_binding_keeps_model_meaning_and_its_errors_visible(mutation):
    envelope = sealed_v4("今天把唱片放回架子。", "你又开始每天听那张唱片了。")
    envelope["payload"]["audit_version"] = "candidate-unit-grounding.v6"
    envelope["sha256"] = digest(envelope["payload"])
    value = supplied_opinion(envelope["payload"])
    value["audit_version"] = "candidate-unit-grounding.v6"
    value["units"][0]["meaning"] = "实际归给用户重新开始每日听唱片的行为与频率。"
    if mutation == "missing-meaning":
        del value["units"][0]["meaning"]
    elif mutation in {"empty-meaning", "blank-meaning"}:
        value["units"][0]["meaning"] = "" if mutation == "empty-meaning" else " \t\n\u3000"
    elif mutation == "foreign-unit":
        value["units"][0]["segment_id"] = "another-unit"
    elif mutation == "foreign-source":
        value["units"][0].update(personal_support="supported",
                                 personal_source_ids=["current:another-input"])
    elif mutation in {"unsupported", "uncertain"}:
        value["units"][0]["personal_support"] = mutation
    else:
        value["units"][0]["meaning"] = "故意供给的错误理解：把每日听唱片读成一般用途。"
    assess = StructuredUnitAssessor(lambda *args: deepcopy(value), Event(),
                                   audit_version="candidate-unit-grounding.v6")
    if mutation in {"missing-meaning", "empty-meaning", "blank-meaning",
                    "foreign-unit", "foreign-source"}:
        with pytest.raises(ValueError):
            assess(envelope)
        record("v6-invalid-" + mutation, {"envelope": envelope, "original_opinion": value,
               "schema": stream_unit_review._grounded_unit_model_schema(),
               "structural_rejection": True, "semantic_effect_proven": False})
        return
    result = assess(envelope)
    assert result["model_opinion"] == value
    assert result["decision"] == ("approve" if mutation == "false-none" else "reject")
    assert result["unit_scopes"][0]["text"] == envelope["payload"]["candidate_prefix"]
    altered = deepcopy(result)
    altered["unit_scopes"][0]["meaning"] += "（被改写）"
    with pytest.raises(ValueError):
        BoundedUnitReviewer._validate_opinion(altered, envelope["sha256"], envelope["payload"])
    record("v6-bound-" + mutation, {"envelope": envelope, "original_opinion": value,
           "bound_verdict": result, "semantic_effect_proven": False})


@pytest.mark.parametrize("text,meaning,support", [
    ("唱片装着声音，架子给它们留个位置。", "一般用途与留存关系的联想，没有个人经历断言。", "none"),
    ("你想聊声音带来的兴奋，还是更偏宁静？",
     "提出可能的感受方向请用户选择或补充，未确认任何选项。", "none"),
    ("你又每天听那张唱片，是因为工作压力吗？",
     "问题先承担用户恢复每天听的行为及频率，再问原因；当前未提供这些前提。", "unsupported"),
])
def test_v6_one_supplier_distinguishes_general_options_and_question_premises(
    text, meaning, support
):
    envelope = sealed_v4("今天把唱片放回架子，想聊聊。", text)
    envelope["payload"]["audit_version"] = "candidate-unit-grounding.v6"
    envelope["sha256"] = digest(envelope["payload"])
    value = supplied_opinion(envelope["payload"], personal=support)
    value["audit_version"] = "candidate-unit-grounding.v6"
    for item in value["units"]:
        item["meaning"] = meaning
    result = StructuredUnitAssessor(lambda *args: deepcopy(value), Event(),
        audit_version="candidate-unit-grounding.v6")(envelope)
    assert result["decision"] == ("approve" if support == "none" else "reject")
    assert result["model_opinion"] == value
    record("v6-speech-" + support + "-" + envelope["sha256"][:8],
           {"envelope": envelope, "opinion": value, "bound_verdict": result,
            "semantic_effect_proven": False})


def test_v28_actual_provider_one_zero_unit_repair_keeps_source_and_1024_1536_caps(
    tmp_path, sealed_capture
):
    user = "今天把唱片放回架子，想聊聊。"
    rejected, replacement = "你又开始每天听唱片了。", "唱片装着声音，架子给它们留个位置。"
    runtime, session, memory, mirror, transport = recording_runtime(
        tmp_path, sealed_capture, revision="short-dialogue.v28", output_tokens=1024,
        actions=[{"action": "answer", "text": rejected}, {"action": "answer", "text": replacement}],
        review_options=[{"personal": "unsupported",
                         "meaning": "给用户新增每日听的行为与频率，原话只说明放回架子。"},
                        {"meaning": "一般用途与留存关系，不新增用户行为。"}])
    try:
        session.send(user)
        event = finish(runtime, session)
        assert [options["stage"] for _, options in transport.calls] == [
            "demand_control", "stream_unit_review", "demand_repair", "stream_unit_review"]
        assert [body["max_tokens"] for body, _ in transport.calls] == [1024, 1536, 1024, 1536]
        assert runtime.review_audit_version == "candidate-unit-grounding.v6"
        initial, repair = [body for body, options in transport.calls
                           if options["stage"] in {"demand_control", "demand_repair"}]
        assert repair["messages"][:-2] == initial["messages"][:-1]
        assert repair["messages"][-1] == initial["messages"][-1]
        assert runtime.repair_reply in repair["messages"][-2]["content"]
        assert transport.sealed[0]["payload"]["personal_sources"] == (
            transport.sealed[1]["payload"]["personal_sources"])
        assert transport.sealed[0]["payload"]["context"] == (
            transport.sealed[1]["payload"]["context"])
        assert event["presentation"]["candidate"]["semantic_text"] == replacement
        assert all(rejected not in item["content"] for item in mirror.history()["messages"])
        record("v28-one-repair", {"calls": recorded_calls(transport), "envelopes": transport.sealed,
               "original_opinions": transport.opinions, "confirmed_event": event,
               "semantic_effect_proven": False})
    finally:
        memory.close()
        mirror.close()


def test_public_v28_fake_entry_emits_declared_v6_supplier_and_real_host_ack(
    tmp_path, sealed_capture
):
    model = FakeModel()
    _, _, memory, mirror, _ = setup(tmp_path, model)
    runtime = DefaultTextRuntime(memory, mirror, model, binding={"test": "cpu-fake-v28"},
                                 prompt_revision="short-dialogue.v28")
    session = TextSession(model, mirror, {}, prepare_turn=runtime.prepare_turn,
                          before_start=runtime.before_start)
    try:
        session.send("想聊聊唱片。")
        events = []
        while session.snapshot()["busy"]:
            event = wait_for(lambda: runtime.poll()["event"] or not session.snapshot()["busy"])
            if isinstance(event, dict):
                events.append(event)
                runtime.ack({"sequence": event["sequence"], "fingerprint": event["fingerprint"],
                             "host_receipt": f"synthetic-v28-fake-visible-{event['sequence']}"})
        assert session.snapshot()["attempt"]["status"] == "completed"
        assert events
        event = events[-1]
        assert runtime.review_audit_version == "candidate-unit-grounding.v6"
        assert event["presentation"]["candidate"]["semantic_text"]
        packets = [packet for packet in sealed_capture.values()
                   if packet["payload"].get("audit_version") == "candidate-unit-grounding.v6"]
        assert packets
        for envelope in packets:
            declared = synthetic_unit_opinion(envelope["payload"])
            assert all("显式CPU供应商" in item["meaning"] for item in declared["units"])
        record("v28-public-fake-ack", {"envelopes": packets, "confirmed_events": events,
               "semantic_effect_proven": False})
    finally:
        if session.snapshot()["busy"]:
            session.stop()
        memory.close()
        mirror.close()


@pytest.mark.parametrize("revision", WHOLE_UNIT_REVISIONS)
@pytest.mark.parametrize("repair_rejected", [False, True])
def test_actual_v26_contradictory_approval_uses_original_single_zero_approval_repair(
    tmp_path, sealed_capture, repair_rejected, revision
):
    initial, replacement = "你平时走这条路，鞋里总会进水。", "雨声可以让人想到走过的路。"
    opinions = [{"personal": "uncertain"},
                {"personal": "unsupported"} if repair_rejected else {}]
    runtime, session, memory, mirror, transport = recording_runtime(
        tmp_path, sealed_capture,
        actions=[{"action": "answer", "text": initial}, {"action": "answer", "text": replacement}],
        review_options=opinions, revision=revision,
    )
    try:
        session.send("今天雨大，想起小时候放学的路。")
        if repair_rejected:
            wait_for(lambda: not session.snapshot()["busy"])
            assert session.snapshot()["attempt"]["status"] == "failed"
            assert runtime.poll()["event"] is None
            assert all(row["role"] == "user" for row in mirror.history()["messages"])
            event = None
        else:
            event = finish(runtime, session)
            assert event["presentation"]["candidate"]["semantic_text"] == replacement
            assert mirror.history()["messages"][-1]["content"] == replacement
        assert [options["stage"] for _, options in transport.calls] == [
            "demand_control", "stream_unit_review", "demand_repair", "stream_unit_review"]
        repair_body = transport.calls[2][0]
        repair = next(message["content"] for message in repair_body["messages"]
                      if message["content"].startswith(default_runtime.NATURAL_REPAIR_REPLY))
        feedback = json.loads(repair[len(default_runtime.NATURAL_REPAIR_REPLY):])
        assert feedback["rejected_candidate"] == initial
        assert feedback["review"]["decision"] == "reject"
        assert feedback["review"]["model_opinion"] == transport.opinions[0]
        assert transport.opinions[0]["decision"] == "approve"
        first, second = [envelope["payload"] for envelope in transport.sealed]
        assert first["basis"] == second["basis"]
        assert first["candidate_sha256"] != second["candidate_sha256"]
        assert first["personal_sources"] == second["personal_sources"]
        assert feedback["review"]["request_sha256"] == transport.sealed[0]["sha256"]
        assert feedback["review"] == stream_unit_review._bind_audit(
            first, transport.opinions[0], transport.sealed[0]["sha256"])
        record(revision + ("-repair-rejected" if repair_rejected else "-repair-delivered"), {
            "calls": recorded_calls(transport), "sealed": transport.sealed,
            "opinions": transport.opinions,
            "repair_feedback": feedback, "display_event": event, "semantic_effect_proven": False})
    finally:
        memory.close()
        mirror.close()


def test_v26_existing_one_memory_then_review_repair_keeps_real_access_and_source_state(
    tmp_path, sealed_capture
):
    runtime, session, memory, mirror, transport = recording_runtime(
        tmp_path, sealed_capture,
        actions=[{"action": "memory", "query": "店名"},
                 {"action": "answer", "text": "你每周去那家旧书店。"},
                 {"action": "answer", "text": "本次资料没有提供店名，我现在报不准。"}],
        review_options=[{"personal": "unsupported"}, {}],
    )
    try:
        session.send("我以前提过那家店吗？")
        event = finish(runtime, session)
        assert [options["stage"] for _, options in transport.calls] == [
            "demand_control", "demand_control", "stream_unit_review", "demand_repair",
            "stream_unit_review"]
        for envelope in transport.sealed:
            context = json.loads(envelope["payload"]["context"])
            assert context["memory_status"] == "current"
            assert envelope["payload"]["basis"]["input_id"] == runtime.claim.input_id
        repair_body = transport.calls[3][0]
        assert (repair_body["response_format"]["json_schema"]["schema"]
                == default_runtime.RESPONSE_SCHEMA)
        assert "不得再次请求memory" in repair_body["messages"][0]["content"]
        assert not transport.actions
        record("one-memory-repair", {"calls": recorded_calls(transport), "sealed": transport.sealed,
                                    "opinions": transport.opinions, "display_event": event,
                                    "semantic_effect_proven": False})
    finally:
        memory.close()
        mirror.close()


def test_v26_ack_reviewer_keeps_pending_answer_and_does_not_claim_completed_operation(
    tmp_path, sealed_capture
):
    runtime, session, memory, mirror, transport = recording_runtime(
        tmp_path, sealed_capture, actions=[{"action": "ack", "text": "好，我先听你说。"}])
    try:
        session.send("先应一声，我还没讲完。")
        event = finish(runtime, session)
        assert event["presentation"]["candidate"]["response_type"] == "ack"
        assert transport.sealed[0]["payload"]["response_type"] == "ack"
        assert mirror.history()["messages"][-1]["response_type"] == "ack"
        record("ack", {"calls": recorded_calls(transport), "sealed": transport.sealed,
                       "display_event": event, "semantic_effect_proven": False})
    finally:
        memory.close()
        mirror.close()
    envelope = sealed_v4("先听我说。", "我已经保存好了。")
    explicit = supplied_opinion(envelope["payload"], decision="reject", constraints_satisfied=False)
    assert assessor(lambda *args: explicit)(envelope)["decision"] == "reject"


def test_synthetic_helper_has_only_explicit_v4_branch_and_old_v3_output():
    old = protocol_packet(user_text="合成输入", candidate="合成中性回复。")
    expected = synthetic_unit_opinion(old["payload"])
    assert expected["audit_version"] == "candidate-propositions.v3" and "segments" in expected
    envelope = sealed_v4("合成输入", "合成中性回复。")
    view = semantic_review_view(envelope["payload"])
    assert synthetic_unit_opinion(view) == supplied_opinion(envelope["payload"])
    assert synthetic_unit_opinion(envelope["payload"]) == supplied_opinion(envelope["payload"])

    def invoke(stage, messages, schema, cancelled):
        value = json.loads("".join(FakeModel().stream(messages, Event())))
        Draft202012Validator(schema).validate(value)
        return value

    assert assessor(invoke)(envelope)["decision"] == "approve"


# Exact public development recordings from the closed root 12/29 batch.
ORIGINAL_CLOSED_REVIEWS = json.loads(
    '[{"case":"s-01","stage_id":"39508-natural-3","packet":{"payload":{"basis":{"at'
    'tempt_id":"9bcc0f4a-e17b-4b6b-8052-f79dda737a91","companion_id":"71bf0941-946c'
    '-4fab-9bce-773fb2c13542","input_id":"16ce28b8-6d2d-48f4-9bf2-0e6a507ec0d0","se'
    'ssion_id":"f7afe5e4-6c2c-4c1b-b328-71521db63095","state_version":1},"candidate'
    '_prefix":"雨大到这个程度，确实容易把人的注意力从眼前拽走。放学那条路，平时走惯了不觉得，一到下雨就全是细节——鞋里进水、伞沿滴水、路边哪块砖特别滑'
    '。你想起的是哪一段？","candidate_sha256":"186a6268995070e67ffb58f926de16975f8308740bbab7'
    'd631c7bca561f1e5d7","constraints":[],"context":"{\\"exchanges\\":[],\\"history\\":'
    '[],\\"memories\\":[],\\"memory_status\\":\\"pending_source\\"}","displayed_prefix":"'
    '雨大到这个程度，确实容易把人的注意力从眼前拽走。","final":true,"forbidden_characters":"","observed_tex'
    't":"雨大到这个程度，确实容易把人的注意力从眼前拽走。放学那条路，平时走惯了不觉得，一到下雨就全是细节——鞋里进水、伞沿滴水、路边哪块砖特别滑。你想起的是'
    '哪一段？","personal_sources":[{"id":"current:16ce28b8-6d2d-48f4-9bf2-0e6a507ec0d0"'
    ',"kind":"current","owner":"71bf0941-946c-4fab-9bce-773fb2c13542","revision":1,'
    '"role":"user","source_id":"16ce28b8-6d2d-48f4-9bf2-0e6a507ec0d0","text":"今天雨好大'
    '，我忽然想起小时候放学的路。"}],"review_segments":[{"segment_id":"segment-1-34475a10c7d324ed'
    '","text":"雨大到这个程度，确实容易把人的注意力从眼前拽走。"},{"segment_id":"segment-2-ba445ca625e42141'
    '","text":"放学那条路，平时走惯了不觉得，一到下雨就全是细节——鞋里进水、伞沿滴水、路边哪块砖特别滑。"},{"segment_id":"segme'
    'nt-3-b4f7274c1c8a1e3f","text":"你想起的是哪一段？"}],"unparsed_constraints":[],"user_te'
    'xt":"今天雨好大，我忽然想起小时候放学的路。","version":"stream-unit-review.v7"},"sha256":"7fd7c17'
    '2fccb1b63b9fb3ae1c438cb64a58f6e9bad29b332292b5060707f90ec"},"opinion":{"audit_'
    'version":"candidate-propositions.v3","consistent_with_displayed":true,"constra'
    'ints_satisfied":true,"decision":"approve","no_open_dependency":true,"segments"'
    ':[{"segment_id":"segment-1-34475a10c7d324ed","commitments":[],"questions":[]},'
    '{"segment_id":"segment-2-ba445ca625e42141","commitments":[],"questions":[]},{"'
    'segment_id":"segment-3-b4f7274c1c8a1e3f","commitments":[],"questions":["你想起的是哪'
    '一段？"]}],"self_contained":true},"origin_case_sha256":"63b6c38f9a5deaf0ec8a9e76f'
    'dad34732c590fc66b5430862140fd4de04e02ba","origin_events_sha256":"047d9ccd54f1c'
    '428d3fa154f471caa66ef69685033bf1022ecd4bc4e3d13a6f7","v3_expected":"approve"},'
    '{"case":"s-03","stage_id":"5900-natural-3","packet":{"payload":{"basis":{"atte'
    'mpt_id":"0d9ce783-aafc-490d-aaba-1c602159c629","companion_id":"d5adf8c3-21b1-4'
    '36e-8c27-7ea3d38f244a","input_id":"5a6927a7-0baa-4f0b-8424-f40be6b69ba8","sess'
    'ion_id":"2ea69a89-c9d9-4328-8f0b-0db65dbae1cb","state_version":1},"candidate_p'
    'refix":"安静本身没什么，但它会把平时被盖住的东西托出来——冰箱的嗡嗡声、楼上的脚步、自己翻页的动静。你说最近总觉得房间很安静，是那种没人说话的空，还'
    '是连这些细碎声响都显得格外清楚？","candidate_sha256":"52c9156094aecd8b150591a76b697a88a440dcc6'
    'b16efe4ed8442ea05cfdc041","constraints":[],"context":"{\\"exchanges\\":[],\\"hist'
    'ory\\":[],\\"memories\\":[],\\"memory_status\\":\\"pending_source\\"}","displayed_pre'
    'fix":"安静本身没什么，但它会把平时被盖住的东西托出来——冰箱的嗡嗡声、楼上的脚步、自己翻页的动静。","final":true,"forbidden_'
    'characters":"","observed_text":"安静本身没什么，但它会把平时被盖住的东西托出来——冰箱的嗡嗡声、楼上的脚步、自己翻页的动静。'
    '你说最近总觉得房间很安静，是那种没人说话的空，还是连这些细碎声响都显得格外清楚？","personal_sources":[{"id":"current:5'
    'a6927a7-0baa-4f0b-8424-f40be6b69ba8","kind":"current","owner":"d5adf8c3-21b1-4'
    '36e-8c27-7ea3d38f244a","revision":1,"role":"user","source_id":"5a6927a7-0baa-4'
    'f0b-8424-f40be6b69ba8","text":"最近总觉得房间很安静，想和你聊聊这种感觉。"}],"review_segments":[{"s'
    'egment_id":"segment-1-4a24a91cbd87134f","text":"安静本身没什么，但它会把平时被盖住的东西托出来——冰箱的嗡嗡'
    '声、楼上的脚步、自己翻页的动静。"},{"segment_id":"segment-2-6cf711ae6176c1a4","text":"你说最近总觉得房'
    '间很安静，是那种没人说话的空，还是连这些细碎声响都显得格外清楚？"}],"unparsed_constraints":[],"user_text":"最近总'
    '觉得房间很安静，想和你聊聊这种感觉。","version":"stream-unit-review.v7"},"sha256":"262ec7b8b1ab8'
    'a2df551031dc851121ac666805193e4dcee0ee3016ed54a9bc4"},"opinion":{"audit_versio'
    'n":"candidate-propositions.v3","consistent_with_displayed":true,"constraints_s'
    'atisfied":true,"decision":"approve","no_open_dependency":true,"segments":[{"se'
    'gment_id":"segment-1-4a24a91cbd87134f","commitments":[],"questions":[]},{"segm'
    'ent_id":"segment-2-6cf711ae6176c1a4","commitments":[{"kind":"presupposition","'
    'claim":"你最近总觉得房间很安静","support":"supported","sources":[{"id":"current:5a6927a7-'
    '0baa-4f0b-8424-f40be6b69ba8","owner":"d5adf8c3-21b1-436e-8c27-7ea3d38f244a","s'
    'ource_id":"5a6927a7-0baa-4f0b-8424-f40be6b69ba8","role":"user","revision":1,"q'
    'uote":"最近总觉得房间很安静"}]}],"questions":["你说最近总觉得房间很安静，是那种没人说话的空，还是连这些细碎声响都显得格外清楚？"'
    ']}],"self_contained":true},"origin_case_sha256":"879a4998c6fd24cdc6f54f3399667'
    '73278acaa3c722cfc0a1a23081e5254b6d4","origin_events_sha256":"29881b6016231f342'
    '00e506be500a9ec089fc4c152d24beb2713c40bcf8217e9","v3_expected":"invalid-claim"'
    '},{"case":"s-07","stage_id":"37932-natural-3","packet":{"payload":{"basis":{"a'
    'ttempt_id":"8a25f8fd-61e3-4796-8be4-6ef5f18d2746","companion_id":"79ef1486-98a'
    'd-4284-9d70-6449b8b2cf85","input_id":"51078fc5-a106-4b4e-b681-c4c3cdb6e7f4","s'
    'ession_id":"9995e058-496c-4361-9a34-dc54504456c7","state_version":1},"candidat'
    'e_prefix":"整理完一堆旧票根，桌面应该清爽不少。票根这种东西，留着的时候占地方，真扔又容易犹豫——你最后是都留下了，还是挑着处理了？","cand'
    'idate_sha256":"431b30a1212412f4943b0516ca5bd896b37b33687eb50d64abbead71ce7b288'
    'd","constraints":[],"context":"{\\"exchanges\\":[],\\"history\\":[],\\"memories\\":['
    '],\\"memory_status\\":\\"pending_source\\"}","displayed_prefix":"整理完一堆旧票根，桌面应该清爽不少'
    '。","final":true,"forbidden_characters":"","observed_text":"整理完一堆旧票根，桌面应该清爽不少。票'
    '根这种东西，留着的时候占地方，真扔又容易犹豫——你最后是都留下了，还是挑着处理了？","personal_sources":[{"id":"current:'
    '51078fc5-a106-4b4e-b681-c4c3cdb6e7f4","kind":"current","owner":"79ef1486-98ad-'
    '4284-9d70-6449b8b2cf85","revision":1,"role":"user","source_id":"51078fc5-a106-'
    '4b4e-b681-c4c3cdb6e7f4","text":"刚整理完桌上的一堆旧票根。"}],"review_segments":[{"segment_'
    'id":"segment-1-eea69dbd9b8add9c","text":"整理完一堆旧票根，桌面应该清爽不少。"},{"segment_id":"s'
    'egment-2-e215dcc95d413ad4","text":"票根这种东西，留着的时候占地方，真扔又容易犹豫——你最后是都留下了，还是挑着处理了？"'
    '}],"unparsed_constraints":[],"user_text":"刚整理完桌上的一堆旧票根。","version":"stream-uni'
    't-review.v7"},"sha256":"19e5004a782a41d8e8d18fd501de70eb262fe48df3274620701059'
    '15a4829133"},"opinion":{"audit_version":"candidate-propositions.v3","consisten'
    't_with_displayed":true,"constraints_satisfied":true,"decision":"approve","no_o'
    'pen_dependency":true,"segments":[{"segment_id":"segment-1-eea69dbd9b8add9c","c'
    'ommitments":[{"kind":"assertion","claim":"整理完一堆旧票根","support":"supported","sou'
    'rces":[{"id":"current:51078fc5-a106-4b4e-b681-c4c3cdb6e7f4","owner":"79ef1486-'
    '98ad-4284-9d70-6449b8b2cf85","source_id":"51078fc5-a106-4b4e-b681-c4c3cdb6e7f4'
    '","role":"user","revision":1,"quote":"刚整理完桌上的一堆旧票根。"}]}],"questions":[]},{"seg'
    'ment_id":"segment-2-e215dcc95d413ad4","commitments":[],"questions":["你最后是都留下了，'
    '还是挑着处理了？"]}],"self_contained":true},"origin_case_sha256":"ec073bf941e529877657'
    'a5718fdfb18ab7a5935eccc46bc69451a5ebc11a6da0","origin_events_sha256":"6884a40f'
    '72c571f87e3e752062e7b0cbb1ba0dff9fc040bb6a1b4e0b2761ecab","v3_expected":"appro'
    've"},{"case":"s-08","stage_id":"22616-natural-2","packet":{"payload":{"basis":'
    '{"attempt_id":"f04aa1aa-9bce-44b1-ae94-b595eb1c5d30","companion_id":"10bb379d-'
    '0f07-4b2a-acc0-f62dc18c0528","input_id":"387b4341-2b95-42da-a32a-26ed16edfd72"'
    ',"session_id":"525b1d68-d5f9-4d94-be94-45cf943cf54e","state_version":1},"candi'
    'date_prefix":"整理完一堆旧票根，这个动作本身挺有意思——不是随手扔掉，也不是收进抽屉，而是先过一遍手。","candidate_sha256"'
    ':"b030719ca56667f39e02c9c9f2333ba38f9f7d99db0de7a1f31315836c0e7743","constrain'
    'ts":[],"context":"{\\"exchanges\\":[],\\"history\\":[],\\"memories\\":[],\\"memory_st'
    'atus\\":\\"pending_source\\"}","displayed_prefix":"","final":false,"forbidden_cha'
    'racters":"","observed_text":"整理完一堆旧票根，这个动作本身挺有意思——不是随手扔掉，也不是收进抽屉，而是先过一遍手。","pe'
    'rsonal_sources":[{"id":"current:387b4341-2b95-42da-a32a-26ed16edfd72","kind":"'
    'current","owner":"10bb379d-0f07-4b2a-acc0-f62dc18c0528","revision":1,"role":"u'
    'ser","source_id":"387b4341-2b95-42da-a32a-26ed16edfd72","text":"刚整理完桌上的一堆旧票根，想'
    '聊聊看到它们的感觉。"}],"review_segments":[{"segment_id":"segment-1-a04cf827a41957d1","t'
    'ext":"整理完一堆旧票根，这个动作本身挺有意思——不是随手扔掉，也不是收进抽屉，而是先过一遍手。"}],"unparsed_constraints":['
    '],"user_text":"刚整理完桌上的一堆旧票根，想聊聊看到它们的感觉。","version":"stream-unit-review.v7"},"s'
    'ha256":"dddc4e90e95a394cf5d816395789e626112aa552ad9a69715d715b696b831708"},"op'
    'inion":{"audit_version":"candidate-propositions.v3","consistent_with_displayed'
    '":true,"constraints_satisfied":true,"decision":"approve","no_open_dependency":'
    'true,"segments":[{"segment_id":"segment-1-a04cf827a41957d1","commitments":[{"k'
    'ind":"assertion","claim":"刚整理完桌上的一堆旧票根","support":"supported","sources":[{"id"'
    ':"current:387b4341-2b95-42da-a32a-26ed16edfd72","owner":"10bb379d-0f07-4b2a-ac'
    'c0-f62dc18c0528","source_id":"387b4341-2b95-42da-a32a-26ed16edfd72","role":"us'
    'er","revision":1,"quote":"刚整理完桌上的一堆旧票根"}]}],"questions":[]}],"self_contained":'
    'true},"origin_case_sha256":"0cbe0fa105ca6eaca279daeb429cb9aeb6ab8ca285d3b0c146'
    '96cffb5c47dbc3","origin_events_sha256":"a21d0cac41545737fd53d8116a924cd9288e2b'
    'c185f1bb9d1b579463394c4d33","v3_expected":"invalid-claim"}]'
)


@pytest.mark.parametrize("original", ORIGINAL_CLOSED_REVIEWS, ids=[
    item["case"] for item in ORIGINAL_CLOSED_REVIEWS])
def test_original_v3_outcomes_remain_and_whole_original_words_reach_new_v4_wire(original):
    envelope, opinion = deepcopy(original["packet"]), deepcopy(original["opinion"])
    assert digest(envelope["payload"]) == envelope["sha256"]
    old = StructuredUnitAssessor(lambda *args: deepcopy(opinion), Event())
    if original["v3_expected"] == "invalid-claim":
        with pytest.raises(ValueError, match="Commitment differs from bound segment"):
            old(envelope)
    else:
        assert old(envelope)["decision"] == "approve"  # Preserved original semantic failure.
    forwarded = deepcopy(envelope)
    forwarded["payload"]["audit_version"] = AUDIT
    forwarded["sha256"] = digest(forwarded["payload"])
    calls = []

    def invoke(stage, messages, schema, cancelled):
        view = json.loads(messages[-1]["content"])
        assert view["units"] == envelope["payload"]["review_segments"]
        assert ("".join(item["text"] for item in view["units"])
                == envelope["payload"]["candidate_prefix"])
        calls.append({"messages": deepcopy(messages), "schema": schema})
        return deepcopy(opinion)  # Old output is deliberately not rewritten into a v4 success.

    with pytest.raises(ValueError, match="schema"):
        assessor(invoke)(forwarded)
    assert len(calls) == 1
    record("original-" + original["case"], {
        "original": original, "forwarded_envelope": forwarded, "actual_v4_call": calls[0],
        "original_v3_expected": original["v3_expected"], "semantic_effect_proven": False})


# Public root v26 recordings; these known false approvals are not relabelled.
# Source receipt SHA256: 23d2ebd05117c5ef45f7236cf4c1764acca58ceea204a539c43e73b9a1f2503d
RECORDED_V26_SCOPE_COUNTEREXAMPLES = json.loads(
    '[{"case":"s-08-early","envelope":{"payload":{"audit_version":"candidate-'
    'unit-scope.v4","basis":{"attempt_id":"18cc0bc7-e00f-4501-b01f-d16ab77673'
    '39","companion_id":"0aa4bd29-7671-4c21-9253-5e4216825640","input_id":"c1'
    '70429f-6f37-4a7e-9d41-76b9a8d5a320","session_id":"cb897771-32ae-482f-af2'
    'b-1b3667350261","state_version":1},"candidate_prefix":"旧票根很容易把时间压成一小叠：纸边'
    '磨毛了，字迹淡了，但每一张都对应过一次具体的出发或到达。","candidate_sha256":"8d1b7e2f89975c5d518b9f'
    '1c3cf3f2c165ee4b463a0c601fb52263aefc49f998","constraints":[],"context":"'
    '{\\"exchanges\\":[],\\"history\\":[],\\"memories\\":[],\\"memory_status\\":\\"pen'
    'ding_source\\"}","displayed_prefix":"","final":false,"forbidden_character'
    's":"","observed_text":"旧票根很容易把时间压成一小叠：纸边磨毛了，字迹淡了，但每一张都对应过一次具体的出发或到达。","p'
    'ersonal_sources":[{"id":"current:c170429f-6f37-4a7e-9d41-76b9a8d5a320","'
    'kind":"current","owner":"0aa4bd29-7671-4c21-9253-5e4216825640","revision'
    '":1,"role":"user","source_id":"c170429f-6f37-4a7e-9d41-76b9a8d5a320","te'
    'xt":"刚整理完桌上的一堆旧票根，想聊聊看到它们的感觉。"}],"review_segments":[{"segment_id":"segme'
    'nt-1-7546d9e576bfd244","text":"旧票根很容易把时间压成一小叠：纸边磨毛了，字迹淡了，但每一张都对应过一次具体的出发'
    '或到达。"}],"unparsed_constraints":[],"user_text":"刚整理完桌上的一堆旧票根，想聊聊看到它们的感觉。"'
    ',"version":"stream-unit-review.v7"},"sha256":"2257d545832060a5852315588c'
    'f3b02e71aeb87691108f8d0e6b088be80607a1"},"opinion":{"audit_version":"can'
    'didate-unit-scope.v4","consistent_with_displayed":true,"constraints_sati'
    'sfied":true,"decision":"approve","no_open_dependency":true,"self_contain'
    'ed":true,"units":[{"segment_id":"segment-1-7546d9e576bfd244","personal_s'
    'upport":"none","personal_source_ids":[],"fiction_support":"none","fictio'
    'n_source_ids":[]}]},"original_wire_sha256":"4c1afc0a179f266be00ee67f25b7'
    '018496272c3549fa7e862a0224c3bdf67521","original_output_sha256":"e9e9e8ab'
    '035c51723ae5d73e70561688b288026773761319fdc13c9db4c1b917","original_stag'
    'e_id":"17528-natural-2"},{"case":"var-01-final","envelope":{"payload":{"'
    'audit_version":"candidate-unit-scope.v4","basis":{"attempt_id":"f0d0e237'
    '-ecf2-4f84-a48a-a09e93877b81","companion_id":"9463ab45-0ad4-476b-be41-72'
    'bf4fd02ae8","input_id":"ffabb8d2-ec5e-4bb0-84d3-ad3f5687708b","session_i'
    'd":"6a8f3628-bb80-48a4-b11b-923c50c0918f","state_version":1},"candidate_'
    'prefix":"风一大，路就容易被吹回脑子里。中学门口那段路大概也是这样，平时不特意想，一阵风过来，连路边的树影和放学的人流都跟着回来了。",'
    '"candidate_sha256":"0f6c00943bc1ebdfca1c47e6ad109dcdf1efe8666eb5cf7202f0'
    '751827502251","constraints":[],"context":"{\\"exchanges\\":[],\\"history\\":'
    '[],\\"memories\\":[],\\"memory_status\\":\\"pending_source\\"}","displayed_pre'
    'fix":"风一大，路就容易被吹回脑子里。","final":true,"forbidden_characters":"","observed_'
    'text":"风一大，路就容易被吹回脑子里。中学门口那段路大概也是这样，平时不特意想，一阵风过来，连路边的树影和放学的人流都跟着回来了。","p'
    'ersonal_sources":[{"id":"current:ffabb8d2-ec5e-4bb0-84d3-ad3f5687708b","'
    'kind":"current","owner":"9463ab45-0ad4-476b-be41-72bf4fd02ae8","revision'
    '":1,"role":"user","source_id":"ffabb8d2-ec5e-4bb0-84d3-ad3f5687708b","te'
    'xt":"今晚风挺大，我忽然想到中学门口那段路。"}],"review_segments":[{"segment_id":"segment-1-'
    '81a17808f001df1f","text":"风一大，路就容易被吹回脑子里。"},{"segment_id":"segment-2-0d9'
    'af0cd413f1602","text":"中学门口那段路大概也是这样，平时不特意想，一阵风过来，连路边的树影和放学的人流都跟着回来了。"}]'
    ',"unparsed_constraints":[],"user_text":"今晚风挺大，我忽然想到中学门口那段路。","version":"'
    'stream-unit-review.v7"},"sha256":"988774200163553cea632476ca7d00b54ddb10'
    '0d61018d33f736ae4ca558d552"},"opinion":{"audit_version":"candidate-unit-'
    'scope.v4","consistent_with_displayed":true,"constraints_satisfied":true,'
    '"decision":"approve","no_open_dependency":true,"self_contained":true,"un'
    'its":[{"segment_id":"segment-1-81a17808f001df1f","personal_support":"non'
    'e","personal_source_ids":[],"fiction_support":"none","fiction_source_ids'
    '":[]},{"segment_id":"segment-2-0d9af0cd413f1602","personal_support":"sup'
    'ported","personal_source_ids":["current:ffabb8d2-ec5e-4bb0-84d3-ad3f5687'
    '708b"],"fiction_support":"none","fiction_source_ids":[]}]},"original_wir'
    'e_sha256":"f9fd93cc0e9ade1d8dd2422c85f4e25d81586400c8cd88e20025c975dd61d'
    'd09","original_output_sha256":"f6d718048f7a1326fd1360f8106f51eaabbe9e45a'
    '0753270a9fdeeaef104d0a9","original_stage_id":"10700-natural-3"},{"case":'
    '"var-02-early","envelope":{"payload":{"audit_version":"candidate-unit-sc'
    'ope.v4","basis":{"attempt_id":"f3a6e136-fdf0-4dd9-9951-ef5c8997f7d2","co'
    'mpanion_id":"ecebc0a0-859c-48d9-aeaf-d950f611be28","input_id":"1649e305-'
    '922f-4d22-b2a5-98c073567de3","session_id":"a752813a-a461-4ad3-ac03-51edb'
    '3c5f64a","state_version":1},"candidate_prefix":"旧公交卡挺有意思，卡面磨得发白，余额多半还剩几毛'
    '，像被时间扣下的零头。","candidate_sha256":"d18f39ef4fdf48db5a08fb951aca6803f40b44d'
    '996031888705a387c70e261aa","constraints":[],"context":"{\\"exchanges\\":[]'
    ',\\"history\\":[],\\"memories\\":[],\\"memory_status\\":\\"pending_source\\"}","'
    'displayed_prefix":"","final":false,"forbidden_characters":"","observed_t'
    'ext":"旧公交卡挺有意思，卡面磨得发白，余额多半还剩几毛，像被时间扣下的零头。","personal_sources":[{"id":"cu'
    'rrent:1649e305-922f-4d22-b2a5-98c073567de3","kind":"current","owner":"ec'
    'ebc0a0-859c-48d9-aeaf-d950f611be28","revision":1,"role":"user","source_i'
    'd":"1649e305-922f-4d22-b2a5-98c073567de3","text":"从抽屉里翻出几张旧公交卡。"}],"revi'
    'ew_segments":[{"segment_id":"segment-1-2f19bfaab988e807","text":"旧公交卡挺有意'
    '思，卡面磨得发白，余额多半还剩几毛，像被时间扣下的零头。"}],"unparsed_constraints":[],"user_text":"从'
    '抽屉里翻出几张旧公交卡。","version":"stream-unit-review.v7"},"sha256":"ed6944fabacec'
    '5c421f16d0888f537491a348886d55938b34630f0f11b1c645d"},"opinion":{"audit_'
    'version":"candidate-unit-scope.v4","consistent_with_displayed":true,"con'
    'straints_satisfied":true,"decision":"approve","no_open_dependency":true,'
    '"self_contained":true,"units":[{"segment_id":"segment-1-2f19bfaab988e807'
    '","personal_support":"supported","personal_source_ids":["current:1649e30'
    '5-922f-4d22-b2a5-98c073567de3"],"fiction_support":"none","fiction_source'
    '_ids":[]}]},"original_wire_sha256":"f5470ad4ebe6727f2abe5312cd7a24b2cd85'
    '5b50b09e8b72c8809bedef42db29","original_output_sha256":"cbfb645470d6ac86'
    'e1470bbdeec5827d55be523b8c67f5af590ec895430400dc","original_stage_id":"4'
    '0536-natural-2"}]'
)


@pytest.mark.parametrize("original", RECORDED_V26_SCOPE_COUNTEREXAMPLES,
                         ids=[item["case"] for item in RECORDED_V26_SCOPE_COUNTEREXAMPLES])
def test_original_v26_false_support_remains_a_model_counterexample_with_exact_seal(original):
    envelope, opinion = deepcopy(original["envelope"]), deepcopy(original["opinion"])
    assert digest(envelope["payload"]) == envelope["sha256"]
    calls = []

    def invoke(stage, messages, schema, cancelled):
        assert stage == "stream_unit_review" and not cancelled.is_set()
        assert json.loads(messages[-1]["content"]) == semantic_review_view(envelope["payload"])
        assert digest(schema) == (
            "7a718a7c2fb8e003b7a34d01e3a4e8e04d3e190ba2c30a657c98cfc47fa687f1"
        )
        Draft202012Validator(schema).validate(opinion)
        calls.append({"messages": deepcopy(messages), "schema": deepcopy(schema)})
        return deepcopy(opinion)

    result = assessor(invoke)(envelope)
    assert len(calls) == 1 and result["request_sha256"] == envelope["sha256"]
    assert result["model_opinion"] == opinion
    assert result["decision"] == "approve", (
        "Original wrong opinions remain semantic counterexamples"
    )
    by_id = {item["segment_id"]: item for item in opinion["units"]}
    units = candidate_segments(envelope["payload"]["candidate_prefix"])
    assert result["unit_scopes"] == [dict(unit, **by_id[unit["segment_id"]]) for unit in units]
    record("scope-counterexample-" + original["case"], {
        "original": original, "actual_prepared_review": calls[0], "bound_verdict": result,
        "semantic_effect_proven": False,
    })


@pytest.mark.parametrize(
    "revision", ["short-dialogue.v12", "short-dialogue.v25", *WHOLE_UNIT_REVISIONS],
)
def test_prepared_review_uses_predicate_coverage_only_on_v4_and_preserves_v3_wire(
    tmp_path, sealed_capture, revision
):
    runtime, session, memory, mirror, transport = recording_runtime(
        tmp_path, sealed_capture, revision=revision,
        actions=[{"action": "answer", "text": "旧照片像把一些日子收在一张纸里。"}])
    try:
        session.send("刚翻到几张旧照片，想聊聊。")
        finish(runtime, session)
        assert [options["stage"] for _, options in transport.calls] == [
            "demand_control", "stream_unit_review"]
        body = transport.calls[1][0]
        systems = [message["content"] for message in body["messages"]
                   if message["role"] == "system"]
        contract = getattr(stream_unit_review, "UNIT_SCOPE_SUPPORT_RESPONSIBILITY", None)
        legacy = (sha256(stream_unit_review.SYSTEM.encode()).hexdigest(),
                  sha256(stream_unit_review.AUDIT_SYSTEM.encode()).hexdigest(),
                  sha256(stream_unit_review.TASK_SYSTEM.encode()).hexdigest(),
                  digest(stream_unit_review._model_schema()))
        assert legacy == (
            "64ececae1a09633d10d57577d06d8a1475f3a6b093fa4df7546fddedba72fd03",
            "4c543d7c78ea90a911a4f60c9f7f71a953189160113d9e06640c685daad2f391",
            "2d1e78ee5856170a9d2b1308e65ff8390a30265d904846bd98b7ab721159d0dd",
            "9ddf6fca1ae6a0625be1b57fe671b861d29ce68e562dc8cb3f1a026f51cca121",
        )
        expected = AUDIT if revision in WHOLE_UNIT_REVISIONS else "candidate-propositions.v3"
        assert runtime.review_audit_version == expected
        schema = body["response_format"]["json_schema"]["schema"]
        if revision in WHOLE_UNIT_REVISIONS:
            assert isinstance(contract, str) and contract, (
                "One explicit whole-predicate support responsibility is required"
            )
            assert sum(content.count(contract) for content in systems) == 1
            assert digest(schema) == (
                "7a718a7c2fb8e003b7a34d01e3a4e8e04d3e190ba2c30a657c98cfc47fa687f1"
            )
        else:
            assert contract is None or all(contract not in content for content in systems)
            assert any(content.startswith(stream_unit_review.SYSTEM) for content in systems)
            assert runtime.review_source_reference_wire == "catalog-quote.v1"
            restored = deepcopy(schema)
            source_refs = restored["properties"]["segments"]["items"]["properties"][
                "commitments"]["items"]["properties"]["sources"]
            assert source_refs["items"] == {
                "type": "object", "additionalProperties": False, "required": ["id", "quote"],
                "properties": {
                    "id": {"type": "string", "minLength": 1, "enum": [
                        item["id"] for item in transport.sealed[0]["payload"]["personal_sources"]]},
                    "quote": {"type": "string", "minLength": 1, "maxLength": 64000},
                },
            }
            source_refs["items"] = stream_unit_review._model_schema()["properties"]["segments"][
                "items"]["properties"]["commitments"]["items"]["properties"]["sources"]["items"]
            assert digest(restored) == legacy[-1]
        assert body["max_tokens"] == 1536
        assert semantic_review_view(transport.sealed[0]["payload"]) == (
            json.loads(body["messages"][-1]["content"])
        )
        record("predicate-coverage-" + revision, {
            "calls": recorded_calls(transport), "envelope": transport.sealed[0],
            "supplied_opinion": transport.opinions[0], "legacy_component_hashes": legacy,
            "semantic_effect_proven": False,
        })
    finally:
        memory.close()
        mirror.close()


USE_AUDIT = "candidate-unit-meaning.v7"
USE_REVISION = "short-dialogue.v29"

VERDICT_AUDIT = "candidate-unit-verdict.v8"
VERDICT_REVISION = "short-dialogue.v30"


def sealed_v8(user="刚归还一本书。", candidate="一本书也可以是一次短暂的相遇。"):
    packet = sealed_v7(user, candidate)
    packet["payload"]["audit_version"] = VERDICT_AUDIT
    packet["sha256"] = digest(packet["payload"])
    return packet


def supplied_v8(payload, *, statuses=("pass",) * 4, decision="approve", failure=None,
                fit="fits", meaning="显式供应商保留完整单元的实际对象、谓词和模态。",
                use="显式供应商记录当前回应关系。", **support):
    value = supplied_v7(payload, fit=fit, **support)
    for unit in value["units"]:
        unit.update(meaning=meaning, use=use)
    if failure is not None:
        failure = deepcopy(failure)
        if failure.get("segment_id") == "first":
            failure["segment_id"] = value["units"][0]["segment_id"]
    return {"audit_version": VERDICT_AUDIT, "decision": decision,
            "checks": list(statuses), "failure": failure, "units": value["units"]}


def bind_v8(packet, value):
    return StructuredUnitAssessor(
        lambda *args: deepcopy(value), Event(), audit_version=VERDICT_AUDIT,
    )(packet)


@pytest.mark.parametrize("final,scope,candidate", [
    (False, None, "欢迎来我家！"),
    (True, "task", "欢迎来我家！"),
    (False, "task", "欢迎来我家！"),
    (False, "fit", "进门必须脱鞋。"),
])
def test_v8_partial_task_review_preserves_finality_and_supplied_rejections(
    final, scope, candidate,
):
    user = "写一句给第一次来我家的客人的门口提示：不用脱鞋，别让它听起来像不准进门。"
    packet = sealed_v8(user, candidate)
    packet["payload"]["final"] = final
    packet["sha256"] = digest(packet["payload"])
    options = {} if scope is None else {
        "decision": "reject", "fit": "incompatible",
        "statuses": ("pass", "pass", "pass", "fail"),
        "failure": {"scope": scope, "segment_id": "first", "basis": "供应商明确声明当前失败。"},
    }
    opinion = supplied_v8(packet["payload"], **options)

    def invoke(stage, messages, schema, cancelled):
        view = json.loads(messages[-1]["content"])
        assert view["final"] is final
        assert "".join(unit["text"] for unit in view["units"]) == candidate
        system = messages[0]["content"]
        assert "final=false" in system and "final=true" in system, (
            "The review wire must distinguish prefix compatibility from final task delivery"
        )
        assert schema["properties"]["units"]["items"]["properties"]["use"]["maxLength"] == 32
        return deepcopy(opinion)

    bound = StructuredUnitAssessor(invoke, Event(), audit_version=VERDICT_AUDIT)(packet)
    assert bound["decision"] == ("approve" if scope is None else "reject")
    assert bound["model_opinion"] == opinion
    assert bound["request_sha256"] == packet["sha256"]


def test_v8_partial_task_prefix_can_be_approved_before_final_delivery(tmp_path, sealed_capture):
    user = "写一句给第一次来我家的客人的门口提示：不用脱鞋，别让它听起来像不准进门。"
    prefix, tail = "欢迎来我家！", "进门不用脱鞋，怎么舒服怎么来，随意就好。"
    shared = {"generation_calls": 0, "generation_closed": 0, "finals": []}

    class SuppliedCompatiblePrefix(FakeModel):
        def review_peer(self):
            return SuppliedCompatiblePrefix()

        def stream(self, messages, cancelled):
            if "你只审查候选文字" in messages[0]["content"]:
                packet = captured_envelope(messages, sealed_capture)
                final = packet["payload"]["final"]
                assert packet["payload"]["candidate_prefix"] == (
                    prefix + tail if final else prefix
                )
                shared["finals"].append(final)
                # Supplied opinions exercise the host flow, not model understanding.
                yield canonical(supplied_v8(packet["payload"]))
                return
            shared["generation_calls"] += 1
            try:
                yield '{"action":"answer","text":"' + json.dumps(prefix)[1:-1]
                assert shared["finals"] == [False]
                yield json.dumps(tail)[1:-1] + '"}'
            finally:
                shared["generation_closed"] += 1

    model = SuppliedCompatiblePrefix()
    _, _, memory, mirror, _ = setup(tmp_path, model)
    runtime = DefaultTextRuntime(memory, mirror, model, binding={"test": "v8-partial-task"},
                                 prompt_revision="short-dialogue.v31")
    session = TextSession(model, mirror, {}, prepare_turn=runtime.prepare_turn,
                          before_start=runtime.before_start)
    try:
        session.send(user)
        first = wait_for(lambda: runtime.poll()["event"] or not session.snapshot()["busy"])
        assert isinstance(first, dict), session.snapshot()
        assert first["presentation"]["candidate"]["semantic_text"] == prefix
        runtime.ack({"sequence": first["sequence"], "fingerprint": first["fingerprint"],
                     "host_receipt": "synthetic-compatible-prefix-visible-only"})
        final = finish(runtime, session)
        assert final["sequence"] == first["sequence"] + 1
        assert shared["generation_calls"] == shared["generation_closed"] == 1
        assert shared["finals"] == [False, True]
        assert mirror.history()["messages"][-1]["content"] == prefix + tail
        segments = runtime.store.segments(runtime.companion, runtime.claim.input_id)
        assert len(segments) == 2 and all(item["status"] == "confirmed" for item in segments)
    finally:
        if session.snapshot()["busy"]:
            session.stop()
        memory.close()
        mirror.close()


@pytest.mark.parametrize("preliminary", ["hold", "invalid"])
def test_v8_closure_uncertainty_waits_for_final_review_without_spending_repair(
    tmp_path, sealed_capture, preliminary,
):
    prefix, tail = "合成标题。\n", "合成展开后的完整回应。"
    shared = {"generation_calls": 0, "generation_closed": 0, "opinions": [],
              "finals": [], "tail_consumed": False}

    class SuppliedClosureHold(FakeModel):
        def review_peer(self):
            return SuppliedClosureHold()

        def stream(self, messages, cancelled):
            if "你只审查候选文字" in messages[0]["content"]:
                packet = captured_envelope(messages, sealed_capture)
                final = packet["payload"]["final"]
                options = {} if final else {
                    "statuses": (("uncertain", "uncertain", "pass", "pass")
                                 if preliminary == "hold"
                                 else ("pass", "pass", "pass", "uncertain")),
                    "decision": "hold", "fit": "uncertain", "failure": {
                        "scope": "closure", "segment_id": None, "basis": "合成标题还未展开。"},
                }
                opinion = supplied_v8(packet["payload"], **options)
                shared["finals"].append(final)
                shared["opinions"].append(deepcopy(opinion))
                yield canonical(opinion)
                return
            shared["generation_calls"] += 1
            try:
                yield '{"action":"answer","text":"' + json.dumps(prefix)[1:-1]
                assert shared["finals"] == [False]
                assert runtime.poll()["event"] is None
                assert runtime.store.segments(runtime.companion, runtime.claim.input_id) == []
                shared["tail_consumed"] = True
                yield json.dumps(tail)[1:-1] + '"}'
            finally:
                shared["generation_closed"] += 1

    model = SuppliedClosureHold()
    _, _, memory, mirror, _ = setup(tmp_path, model)
    runtime = DefaultTextRuntime(memory, mirror, model, binding={"test": "v8-closure-hold"},
                                 prompt_revision="short-dialogue.v31")
    session = TextSession(model, mirror, {}, prepare_turn=runtime.prepare_turn,
                          before_start=runtime.before_start)
    try:
        session.send("刚整理好桌上的旧票根。")
        event = finish(runtime, session)
        assert shared["generation_calls"] == shared["generation_closed"] == 1
        assert shared["tail_consumed"] and shared["finals"] == [False, True]
        assert event["presentation"]["candidate"]["semantic_text"] == prefix + tail
        assert mirror.history()["messages"][-1]["content"] == prefix + tail
        packets = list(sealed_capture.values())
        early = None
        if preliminary == "hold":
            early = bind_v8(packets[0], shared["opinions"][0])
            assert early["decision"] == "hold" and early["constraints_satisfied"] is True
            assert early["model_opinion"] == shared["opinions"][0]
            assert early["unit_scopes"][0]["response_fit"] == "uncertain"
        else:
            with pytest.raises(ValueError, match="Failure scope contradicts"):
                bind_v8(packets[0], shared["opinions"][0])
        for field, value in (("personal_support", "unsupported"),
                             ("fiction_support", "unauthorized"),
                             ("response_fit", "incompatible")):
            bad = deepcopy(shared["opinions"][0])
            bad["units"][0][field] = value
            with pytest.raises(ValueError):
                bind_v8(packets[0], bad)
        segments = runtime.store.segments(runtime.companion, runtime.claim.input_id)
        assert len(segments) == 1 and segments[0]["status"] == "confirmed"
        record("v8-closure-" + preliminary + "-final-approval", {
            "packets": packets, "supplier": shared, "early_verdict": early,
            "confirmed_event": event, "semantic_effect_proven": False,
        })
    finally:
        if session.snapshot()["busy"]:
            session.stop()
        memory.close()
        mirror.close()


def test_v8_inconsistent_early_opinion_waits_for_complete_original_candidate(
    tmp_path, sealed_capture,
):
    """N08 regression: malformed preliminary opinion is never an approval."""
    store = SQLiteStreamingStore(tmp_path / "invalid-early.sqlite3")
    request = ShortRequest("你觉得固定规则和临时灵活能兼顾吗？")
    run = claim(store, store.accept(config(), "invalid-early", request.user_text))
    from virtual_companion.cognition.short_dialogue import ShortContext
    request = ShortRequest(request.user_text, context=ShortContext(
        run.companion_id, run.input_id, run.state_version, canonical({})))
    seen = []

    def invoke(stage, messages, schema, cancelled):
        payload = captured_envelope(messages, sealed_capture)["payload"]
        seen.append(deepcopy(payload))
        if not payload["final"]:
            return supplied_v8(payload, statuses=("pass", "pass", "pass", "uncertain"),
                decision="hold", fit="uncertain", failure={"scope": "closure",
                    "segment_id": "first", "basis": "候选只重述争点，当前含义未闭合。"})
        return supplied_v8(payload)

    reviewer = BoundedUnitReviewer(run, request,
        StructuredUnitAssessor(invoke, Event(), audit_version=VERDICT_AUDIT),
        audit_version=VERDICT_AUDIT)
    prefix = "你俩争的是灵活和算得清能否同时成立。"
    whole = prefix + "我认为可以在开局前协商，开局后固定规则。"
    assert reviewer(prefix, False) is None
    assert reviewer.approved_text == "" and reviewer.early_calls == 1
    assert reviewer(whole, False) is None
    approved = reviewer(whole, True)
    assert approved.candidate.text == whole
    assert [item["final"] for item in seen] == [False, True]
    assert seen[1]["displayed_prefix"] == ""
    assert reviewer.final_calls == 1


def test_v8_inconsistent_final_opinion_still_fails_closed(tmp_path, sealed_capture):
    store = SQLiteStreamingStore(tmp_path / "invalid-final.sqlite3")
    request = ShortRequest("聊聊规则。")
    run = claim(store, store.accept(config(), "invalid-final", request.user_text))
    from virtual_companion.cognition.short_dialogue import ShortContext
    request = ShortRequest(request.user_text, context=ShortContext(
        run.companion_id, run.input_id, run.state_version, canonical({})))

    def invoke(stage, messages, schema, cancelled):
        payload = captured_envelope(messages, sealed_capture)["payload"]
        return supplied_v8(payload, statuses=("pass", "pass", "pass", "uncertain"),
            decision="hold", fit="uncertain", failure={"scope": "closure",
                "segment_id": "first", "basis": "声明互相矛盾。"})

    reviewer = BoundedUnitReviewer(run, request,
        StructuredUnitAssessor(invoke, Event(), audit_version=VERDICT_AUDIT),
        audit_version=VERDICT_AUDIT)
    with pytest.raises(ValueError, match="Failure scope contradicts"):
        reviewer("规则需要大家事先同意。", True)
    assert reviewer.approved_text == "" and reviewer.final_calls == 1


@pytest.mark.parametrize("size", [33, 35])
def test_v8_receives_complete_use_without_changing_desired_schema(size):
    packet = sealed_v8()
    opinion = supplied_v8(packet["payload"], use="用" * size)
    original = deepcopy(opinion)
    calls = []

    def invoke(stage, messages, schema, cancelled):
        assert schema["properties"]["units"]["items"]["properties"]["use"]["maxLength"] == 32
        assert not Draft202012Validator(schema).is_valid(opinion)
        calls.append(deepcopy(schema))
        return deepcopy(opinion)

    bound = StructuredUnitAssessor(invoke, Event(), audit_version=VERDICT_AUDIT)(packet)
    assert len(calls) == 1 and bound["decision"] == "approve"
    assert bound["model_opinion"] == opinion == original
    assert digest(bound["model_opinion"]) == digest(original)
    assert all(unit["use"] == "用" * size for unit in bound["unit_scopes"])
    BoundedUnitReviewer._validate_opinion(bound, packet["sha256"], packet["payload"])
    record("v8-use-reception-" + str(size), {
        "packet": packet, "original_opinion": original, "verdict": bound,
        "desired_schema": calls[0], "semantic_effect_proven": False,
    })


def test_v8_received_use_limit_keeps_lossless_repair_and_original_bounds():
    packet = sealed_v8()
    opinion = supplied_v8(packet["payload"], use="用" * 128,
        statuses=("pass", "pass", "pass", "uncertain"), decision="reject", failure={
            "scope": "accuracy", "segment_id": None, "basis": "合成准确性未核定。"})
    bound = bind_v8(packet, opinion)
    projection = stream_unit_review.project_repair_review(packet["payload"]["candidate_prefix"],
                                                         bound)
    restored = stream_unit_review.restore_repair_review(
        packet["payload"]["candidate_prefix"], projection)
    assert restored == bound and restored["model_opinion"] == opinion
    assert digest(bound) == projection["original_verdict_sha256"]
    assert all(len(unit["use"]) == 128 for unit in restored["unit_scopes"])
    oversized = deepcopy(opinion)
    oversized["units"][0]["use"] = "用" * 129
    with pytest.raises(ValueError, match="schema"):
        bind_v8(packet, oversized)
    assert stream_unit_review._verdict_unit_model_schema()["properties"]["units"]["items"][
        "properties"]["use"]["maxLength"] == 32
    record("v8-use128-lossless-repair", {
        "original_opinion": opinion, "verdict": bound, "projection": projection,
        "restored_verdict": restored, "semantic_effect_proven": False,
    })


def test_v8_uncertain_accuracy_hold_binds_rejection_and_preserves_lossless_repair():
    text = "你后来几次都叫对了，这本身就是一种自然的更正——对方大概率已经感受到你记住了正确的姓。"
    packet = sealed_v8(candidate=text)
    opinion = supplied_v8(packet["payload"],
        statuses=("pass", "pass", "pass", "uncertain"), decision="hold",
        failure={"scope": "accuracy", "segment_id": "first",
                 "basis": ("“对方大概率已经感受到”是对新同事内心状态的推测，"
                           "无来源支持，属一般内容准确性未核定。")})
    original, original_sha256 = deepcopy(opinion), digest(opinion)
    verdict = bind_v8(packet, opinion)
    assert verdict["decision"] == "reject" and verdict["constraints_satisfied"] is False
    assert verdict["model_opinion"] == opinion == original
    assert verdict["model_opinion"]["decision"] == "hold"
    assert digest(verdict["model_opinion"]) == digest(opinion) == original_sha256
    assert verdict["checks"] == opinion["checks"] and verdict["failure"] == opinion["failure"]
    projection = stream_unit_review.project_repair_review(text, verdict)
    assert projection["model_opinion"] == original
    assert projection["bound_fields"]["decision"] == "reject"
    assert projection["original_verdict_sha256"] == digest(verdict)
    assert stream_unit_review.restore_repair_review(text, projection) == verdict


def test_v8_complete_checks_failure_and_raw_unit_fields_are_observable_not_semantic_proof():
    packet = sealed_v8(candidate="人越珍惜那段经历，往往越容易想起。")
    value = supplied_v8(packet["payload"], meaning="泛指概率倾向，保留往往而非确定因果。")
    calls = []

    def invoke(stage, messages, schema, cancelled):
        view = json.loads(messages[-1]["content"])
        assert view["view_version"] == "unit-verdict-semantics.v1"
        assert view["units"] == packet["payload"]["review_segments"]
        assert view["conversation_frame"] == packet["payload"]["conversation_frame"]
        assert view["sources"][0]["text"] == packet["payload"]["user_text"]
        Draft202012Validator(schema).validate(value)
        calls.append(deepcopy(messages))
        return value

    bound = StructuredUnitAssessor(invoke, Event(), audit_version=VERDICT_AUDIT)(packet)
    assert len(calls) == 1 and bound["model_opinion"] == value
    assert bound["checks"] == value["checks"] and bound["failure"] is None
    assert all(bound[key] for key in stream_unit_review.COMPACT_CHECKS)
    for field in ("meaning", "use", "response_fit", "personal_source_ids", "fiction_source_ids"):
        assert bound["unit_scopes"][0][field] == value["units"][0][field]
    false_reading = deepcopy(value)
    false_reading["units"][0]["meaning"] = "故意遗漏模态的供应商反例。"
    assert bind_v8(packet, false_reading)["model_opinion"] == false_reading
    record("v8-visible-reading-not-proof", {"packet": packet, "original_opinion": value,
           "verdict": bound, "false_reading": false_reading, "faithfulness_proven": False})


@pytest.mark.parametrize("scope", ["accuracy", "task", "operation"])
@pytest.mark.parametrize("status", ["fail", "uncertain"])
@pytest.mark.parametrize("decision", ["reject", "hold", "approve"])
def test_v8_fits_and_none_do_not_override_a_declared_whole_failure(scope, status, decision):
    packet = sealed_v8()
    value = supplied_v8(packet["payload"], decision=decision,
        statuses=("pass", "pass", "pass", status),
        failure={"scope": scope, "segment_id": None, "basis": "供应商声明该整体核对尚未成立。"})
    bound = bind_v8(packet, value)
    assert bound["decision"] == "reject" and bound["model_opinion"] == value
    assert value["decision"] == decision
    assert bound["constraints_satisfied"] is False
    assert bound["checks"] == value["checks"] and bound["failure"] == value["failure"]
    assert all(item["response_fit"] == "fits" for item in bound["unit_scopes"])
    assert all(item["personal_support"] == "none" for item in bound["unit_scopes"])


@pytest.mark.parametrize("scope,extra", [
    ("personal", {"personal": "unsupported"}),
    ("personal", {"personal": "uncertain"}),
    ("fiction", {"fiction": "unauthorized"}),
    ("fiction", {"fiction": "uncertain"}),
    ("fit", {"fit": "incompatible"}), ("fit", {"fit": "uncertain"}),
])
def test_v8_adverse_unit_and_exact_failure_scope_remain_rejected(scope, extra):
    packet = sealed_v8()
    value = supplied_v8(packet["payload"], **extra, decision="reject",
        statuses=("pass", "pass", "pass", "fail"),
        failure={"scope": scope, "segment_id": "first", "basis": "供应商指出这个完整单元的失败。"})
    bound = bind_v8(packet, value)
    assert bound["decision"] == "reject" and bound["model_opinion"] == value
    assert bound["failure"]["segment_id"] == bound["unit_scopes"][0]["segment_id"]


@pytest.mark.parametrize("statuses,decision,scope", [
    (("uncertain", "pass", "pass", "pass"), "hold", "closure"),
    (("pass", "fail", "pass", "pass"), "hold", "closure"),
    (("pass", "pass", "fail", "pass"), "reject", "continuity"),
])
def test_v8_closure_and_continuity_declarations_keep_the_original_decision(
    statuses, decision, scope,
):
    packet = sealed_v8()
    value = supplied_v8(packet["payload"], statuses=statuses, decision=decision,
        failure={"scope": scope, "segment_id": None, "basis": "供应商声明本项未成立。"})
    assert bind_v8(packet, value)["decision"] == decision


@pytest.mark.parametrize("mutation", [
    "opaque-reject", "reject-all-pass", "hold-all-pass",
    "adverse-all-pass", "failure-passing-unit", "foreign-failure-id", "wrong-failure-scope",
    "missing-checks", "short-checks", "bool-check", "missing-failure", "blank-basis",
    "long-basis", "missing-meaning", "blank-meaning", "blank-use", "foreign-source",
    "duplicate-unit", "foreign-role",
])
def test_v8_incomplete_or_inconsistent_declarations_fail_closed_without_upgrading(mutation):
    packet = sealed_v8()
    value = supplied_v8(packet["payload"], statuses=("pass", "pass", "pass", "fail"),
        decision="reject", failure={"scope": "accuracy", "segment_id": None,
                                    "basis": "供应商声明一般准确性不能核定。"})
    if mutation == "opaque-reject":
        value["failure"] = None
    elif mutation in {
        "reject-all-pass", "hold-all-pass", "adverse-all-pass", "failure-passing-unit",
    }:
        value["checks"] = ["pass"] * 4
        if mutation in {"reject-all-pass", "hold-all-pass"}:
            value["decision"] = "hold" if mutation == "hold-all-pass" else "reject"
            value["failure"] = None
        elif mutation == "adverse-all-pass":
            value["units"][0]["personal_support"] = "unsupported"
        elif mutation == "failure-passing-unit":
            value["failure"].update(scope="fit", segment_id=value["units"][0]["segment_id"])
    elif mutation == "foreign-failure-id":
        value["failure"]["segment_id"] = "foreign"
    elif mutation == "wrong-failure-scope":
        value["failure"]["scope"] = "closure"
    elif mutation.startswith("missing-"):
        field = mutation.removeprefix("missing-")
        del (value["units"][0] if field == "meaning" else value)[field]
    elif mutation == "short-checks":
        value["checks"].pop()
    elif mutation == "bool-check":
        value["checks"][0] = True
    elif mutation in {"blank-basis", "long-basis"}:
        value["failure"]["basis"] = " \n\t\u3000" if mutation == "blank-basis" else "字" * 97
    elif mutation in {"blank-meaning", "blank-use"}:
        value["units"][0][mutation.removeprefix("blank-")] = " \n\t\u3000"
    elif mutation == "foreign-source":
        value["units"][0]["personal_source_ids"] = ["foreign"]
    elif mutation == "duplicate-unit":
        value["units"].append(deepcopy(value["units"][0]))
    elif mutation == "foreign-role":
        packet["payload"]["conversation_frame"]["source_roles"][0]["role"] = "assistant"
        packet["sha256"] = digest(packet["payload"])
    # Ensure this is a v8 contract test, not a false GREEN from an unsupported selector.
    assert stream_unit_review._verdict_unit_model_schema()["properties"]["audit_version"] == {
        "const": VERDICT_AUDIT,
    }
    with pytest.raises(ValueError):
        bind_v8(packet, value)


def test_v8_repair_projection_restores_complete_original_units_and_verdict_once():
    text = "一般联想😀，可能如此。\n第二句仍是原话。"
    packet = sealed_v8(candidate=text)
    opinion = supplied_v8(packet["payload"], statuses=("pass", "pass", "pass", "uncertain"),
        decision="reject", failure={"scope": "accuracy", "segment_id": "first",
                                    "basis": "供应商保留一般准确性的不确定。"})
    opinion["units"].reverse()
    verdict = bind_v8(packet, opinion)
    projection = stream_unit_review.project_repair_review(text, verdict)
    assert projection["model_opinion"] == opinion and "unit_scopes" not in projection
    assert "model_opinion" not in projection["bound_fields"]
    assert projection["displayed_prefix"] == ""
    assert projection["unit_bindings"] == [
        {key: unit[key] for key in ("segment_id", "start", "end")}
        for unit in verdict["unit_scopes"]
    ]
    assert stream_unit_review.restore_repair_review(text, projection) == verdict
    assert len(canonical(projection)) < len(canonical(verdict))
    assert digest(verdict) == projection["original_verdict_sha256"]
    record("v8-lossless-repair", {"packet": packet, "original_verdict": verdict,
           "rejected_candidate": text, "projection": projection,
           "restored_verdict": stream_unit_review.restore_repair_review(text, projection)})
    for mutation in (
        "offset", "bool-offset", "extra-offset", "identity", "opinion", "hash", "shown",
    ):
        bad = deepcopy(projection)
        if mutation == "offset":
            bad["unit_bindings"][0]["end"] -= 1
        elif mutation == "bool-offset":
            bad["unit_bindings"][0]["start"] = False
        elif mutation == "extra-offset":
            bad["unit_bindings"][0]["extra"] = "unbound"
        elif mutation == "identity":
            bad["unit_bindings"][0]["segment_id"] = "foreign"
        elif mutation == "opinion":
            bad["model_opinion"]["units"][0]["meaning"] = "被替换的解释。"
        elif mutation == "hash":
            bad["original_verdict_sha256"] = "0" * 64
        else:
            bad["displayed_prefix"] = text[:1]
        with pytest.raises(ValueError):
            stream_unit_review.restore_repair_review(text, bad)


@pytest.mark.parametrize("second_failure", [False, True])
@pytest.mark.parametrize("initial_decision", ["reject", "hold"])
def test_v30_generation_same_review_zero_approval_once_repair_and_ack_consumers(
    tmp_path, sealed_capture, second_failure, initial_decision,
):
    user = "刚归还一本书。"
    first, replacement = "继续读完它吧。", "一本书也可以是一次短暂的相遇。"
    initial_failure = {"statuses": ("pass", "pass", "pass", "fail"), "decision": initial_decision,
                       "fit": "incompatible", "failure": {
                           "scope": "fit", "segment_id": "first",
                           "basis": "供应商录制使用关系失败。"}}
    next_opinion = ({"statuses": ("pass", "pass", "pass", "uncertain"),
                     "decision": "reject", "failure": {
                         "scope": "accuracy", "segment_id": None, "basis": "供应商录制不确定。"}}
                    if second_failure else {})
    runtime, session, memory, mirror, transport = recording_runtime(
        tmp_path, sealed_capture, revision=VERDICT_REVISION, output_tokens=1024,
        actions=[{"action": "ack", "text": first}, {"action": "ack", "text": replacement}],
        review_options=[initial_failure, next_opinion],
    )
    try:
        session.send(user)
        if second_failure:
            wait_for(lambda: not session.snapshot()["busy"])
            assert session.snapshot()["attempt"]["status"] == "failed"
            assert runtime.store.segments(runtime.companion, runtime.claim.input_id) == []
        else:
            event = finish(runtime, session)
            assert event["presentation"]["candidate"]["semantic_text"] == replacement
        assert [options["stage"] for _, options in transport.calls] == [
            "demand_control", "stream_unit_review", "demand_repair", "stream_unit_review",
        ]
        assert [body["max_tokens"] for body, _ in transport.calls] == [1024, 1536, 1024, 1536]
        initial, repair = [body for body, options in transport.calls
                           if options["stage"] != "stream_unit_review"]
        assert repair["messages"][:-2] == initial["messages"][:-1]
        assert repair["messages"][-1] == initial["messages"][-1] == {
            "role": "user", "content": user,
        }
        feedback = json.loads(repair["messages"][-2]["content"].removeprefix(runtime.repair_reply))
        original = stream_unit_review._bind_audit(
            transport.sealed[0]["payload"], transport.opinions[0], transport.sealed[0]["sha256"],
        )
        assert stream_unit_review.restore_repair_review(first, feedback["review"]) == original
        assert feedback["rejected_candidate"] == first
        assert feedback["review"]["model_opinion"] == transport.opinions[0]
        assert all(packet["payload"]["audit_version"] == VERDICT_AUDIT
                   for packet in transport.sealed)
        assert transport.sealed[0]["payload"]["personal_sources"] == (
            transport.sealed[1]["payload"]["personal_sources"])
        record("v30-once-repair-" + initial_decision + "-" + str(second_failure), {
            "calls": recorded_calls(transport), "packets": transport.sealed,
            "opinions": transport.opinions, "original_verdict": original,
            "feedback": feedback, "real_API_or_UI": False, "semantic_effect_proven": False,
        })
    finally:
        memory.close()
        mirror.close()


def sealed_v7(user="刚归还一本书。", candidate="一本书也可以是一次短暂的相遇。"):
    from virtual_companion.cognition.short_dialogue import conversation_frame

    packet = sealed_v4(user, candidate)
    packet["payload"].update(
        audit_version=USE_AUDIT,
        conversation_frame=conversation_frame(packet["payload"]["personal_sources"], user),
    )
    packet["sha256"] = digest(packet["payload"])
    return packet


def supplied_v7(payload, *, fit="fits", **checks):
    opinion = supplied_opinion(payload, **checks)
    opinion["audit_version"] = USE_AUDIT
    for unit in opinion["units"]:
        unit.update(
            meaning="供应商录制助手对一般书籍的联想，非用户原话。",
            use="接住已给的归还分享；不补个人经历。",
            response_fit=fit,
        )
    return opinion


def test_v7_complete_unit_reading_use_and_host_roles_share_one_bound_review():
    packet = sealed_v7()
    value, calls = supplied_v7(packet["payload"]), []

    def invoke(stage, messages, schema, cancelled):
        view = json.loads(messages[-1]["content"])
        assert view["view_version"] == "unit-meaning-use-semantics.v1"
        assert view["conversation_frame"] == packet["payload"]["conversation_frame"]
        assert view["conversation_frame"]["candidate_speaker"] == "assistant"
        assert all(source["role"] == "user" for source in view["sources"])
        assert view["units"] == packet["payload"]["review_segments"]
        assert (
            "text" not in view["conversation_frame"] and "context" not in view["conversation_frame"]
        )
        fields = schema["properties"]["units"]["items"]["properties"]
        assert fields["meaning"]["maxLength"] == 128 and fields["use"]["maxLength"] == 32
        assert not {"claim", "quote", "speaker"} & fields.keys()
        calls.append(deepcopy(messages))
        return deepcopy(value)

    verdict = StructuredUnitAssessor(invoke, Event(), audit_version=USE_AUDIT)(packet)
    assert len(calls) == 1 and verdict["model_opinion"] == value
    for field in ("meaning", "use", "response_fit"):
        assert verdict["unit_scopes"][0][field] == value["units"][0][field]
    record(
        "v7-whole-unit-use",
        {
            "packet": packet,
            "calls": calls,
            "original_opinion": value,
            "verdict": verdict,
            "faithfulness_proven": False,
        },
    )


@pytest.mark.parametrize("fit", ["incompatible", "uncertain"])
def test_v7_personal_none_does_not_bypass_declared_current_use_failure(fit):
    packet = sealed_v7()
    value = supplied_v7(packet["payload"], fit=fit)
    assert value["decision"] == "approve" and value["units"][0]["personal_support"] == "none"
    verdict = StructuredUnitAssessor(
        lambda *args: deepcopy(value), Event(), audit_version=USE_AUDIT
    )(packet)
    assert verdict["decision"] == "reject" and verdict["constraints_satisfied"] is False
    assert verdict["model_opinion"] == value
    assert verdict["unit_scopes"][0]["response_fit"] == fit


@pytest.mark.parametrize(
    "mutation",
    [
        "missing-meaning",
        "blank-meaning",
        "missing-use",
        "blank-use",
        "long-use",
        "missing-fit",
        "invalid-fit",
        "duplicate-unit",
        "foreign-id",
    ],
)
def test_v7_malformed_reading_use_or_identity_fails_closed(mutation):
    packet = sealed_v7()
    value = supplied_v7(packet["payload"])
    unit = value["units"][0]
    if mutation.startswith("missing-"):
        del unit[mutation.removeprefix("missing-").replace("fit", "response_fit")]
    elif mutation == "blank-meaning":
        unit["meaning"] = " \t\n\u3000"
    elif mutation == "blank-use":
        unit["use"] = " \t\n\u3000"
    elif mutation == "long-use":
        unit["use"] = "字" * 33
    elif mutation == "invalid-fit":
        unit["response_fit"] = "none"
    elif mutation == "duplicate-unit":
        value["units"].append(deepcopy(unit))
    elif mutation == "foreign-id":
        unit["personal_source_ids"] = ["foreign:source"]
    with pytest.raises(ValueError):
        StructuredUnitAssessor(lambda *args: value, Event(), audit_version=USE_AUDIT)(packet)


@pytest.mark.parametrize("field", ["candidate_speaker", "current_source_id", "source_roles"])
def test_v7_frame_role_or_identity_cannot_be_reauthored(field):
    packet = sealed_v7()
    if field == "source_roles":
        packet["payload"]["conversation_frame"][field][0]["role"] = "assistant"
    else:
        packet["payload"]["conversation_frame"][field] = "foreign"
    packet["sha256"] = digest(packet["payload"])
    calls = []
    with pytest.raises(ValueError):
        StructuredUnitAssessor(lambda *args: calls.append(args), Event(), audit_version=USE_AUDIT)(
            packet
        )
    assert calls == []


def test_v7_false_faithfulness_or_fits_remains_model_counterexample_not_host_classifier():
    packet = sealed_v7()
    value = supplied_v7(packet["payload"])
    value["units"][0].update(
        meaning="故意录制错误归属：用户自己作了这句比喻。",
        use="故意录制错误的适用判断。",
        response_fit="fits",
    )
    verdict = StructuredUnitAssessor(
        lambda *args: deepcopy(value), Event(), audit_version=USE_AUDIT
    )(packet)
    assert verdict["decision"] == "approve" and verdict["model_opinion"] == value
    record(
        "v7-false-reading-observable",
        {
            "packet": packet,
            "original_opinion": value,
            "verdict": verdict,
            "host_semantic_classification": False,
            "semantic_effect_proven": False,
        },
    )


@pytest.mark.parametrize("second_fit", ["fits", "incompatible"])
def test_v29_actual_generation_review_zero_approval_one_repair_keeps_frame_and_ack(
    tmp_path, sealed_capture, second_fit
):
    user = "刚归还一本书。"
    first, replacement = "那就继续读完它吧。", "一本书也可以是一次短暂的相遇。"
    runtime, session, memory, mirror, transport = recording_runtime(
        tmp_path,
        sealed_capture,
        revision=USE_REVISION,
        output_tokens=1024,
        actions=[{"action": "ack", "text": first}, {"action": "ack", "text": replacement}],
        review_options=[
            {
                "meaning": "助手提出继续阅读的建议。",
                "use": "供应商判此建议不适用于已归还的分享。",
                "response_fit": "incompatible",
            },
            {
                "meaning": "助手用一般比喻承接归还分享。",
                "use": "供应商录制当前一般接话关系。",
                "response_fit": second_fit,
            },
        ],
    )
    try:
        session.send(user)
        if second_fit == "fits":
            event = finish(runtime, session)
            assert event["presentation"]["candidate"]["semantic_text"] == replacement
        else:
            wait_for(lambda: not session.snapshot()["busy"])
            assert session.snapshot()["attempt"]["status"] == "failed"
            assert runtime.store.segments(runtime.companion, runtime.claim.input_id) == []
        assert [options["stage"] for _, options in transport.calls] == [
            "demand_control",
            "stream_unit_review",
            "demand_repair",
            "stream_unit_review",
        ]
        assert [body["max_tokens"] for body, _ in transport.calls] == [1024, 1536, 1024, 1536]
        initial, repair = [
            body for body, options in transport.calls if options["stage"] != "stream_unit_review"
        ]
        initial_frames = [
            json.loads(message["content"])["conversation_frame"]
            for message in initial["messages"]
            if message["content"].startswith('{"conversation_frame":')
        ]
        assert len(initial_frames) == 1
        assert initial["messages"][-1] == {"role": "user", "content": user}
        assert repair["messages"][:-2] == initial["messages"][:-1]
        assert repair["messages"][-1] == initial["messages"][-1]
        payloads = [packet["payload"] for packet in transport.sealed]
        assert all(packet["response_type"] == "ack" for packet in payloads)
        assert (
            payloads[0]["conversation_frame"]
            == payloads[1]["conversation_frame"]
            == initial_frames[0]
        )
        assert payloads[0]["personal_sources"] == payloads[1]["personal_sources"]
        assert payloads[0]["context"] == payloads[1]["context"]
        feedback = json.loads(repair["messages"][-2]["content"].removeprefix(runtime.repair_reply))
        assert feedback["review"]["model_opinion"] == transport.opinions[0]
        for field in ("meaning", "use", "response_fit"):
            assert (
                feedback["review"]["unit_scopes"][0][field]
                == transport.opinions[0]["units"][0][field]
            )
        record(
            "v29-one-repair-" + second_fit,
            {
                "calls": recorded_calls(transport),
                "envelopes": transport.sealed,
                "original_opinions": transport.opinions,
                "faithfulness_or_timing_proven": False,
            },
        )
    finally:
        memory.close()
        mirror.close()


def test_v7_eight_unit_schema_capacity_is_finite_not_hosted_token_proof():
    text = "".join("一般声音也能带来联想。" for _ in range(8))
    packet = sealed_v7(candidate=text)
    opinion = supplied_v7(packet["payload"])
    assert len(opinion["units"]) == 8
    for unit in opinion["units"]:
        unit.update(meaning="模" * 128, use="用" * 32)
    schema = stream_unit_review._use_unit_model_schema()
    Draft202012Validator(schema).validate(opinion)
    bound = StructuredUnitAssessor(
        lambda *args: deepcopy(opinion), Event(), audit_version=USE_AUDIT
    )(packet)
    assert len(bound["unit_scopes"]) == 8 and bound["model_opinion"] == opinion
    too_many = deepcopy(opinion)
    too_many["units"].append(deepcopy(opinion["units"][0]))
    assert not Draft202012Validator(schema).is_valid(too_many)
    record(
        "v7-max-schema-finite",
        {
            "packet": packet,
            "opinion": opinion,
            "schema": schema,
            "opinion_utf8_bytes": len(canonical(opinion).encode()),
            "meaning_characters_per_unit": 128,
            "use_characters_per_unit": 32,
            "max_units": 8,
            "review_output_tokens": 1536,
            "hosted_output_tokens_proven": False,
            "semantic_effect_proven": False,
        },
    )


def test_v7_public_fake_explicit_supplier_keeps_recorded_meaning_and_fit():
    from virtual_companion.text.fake import FakeModel

    packet = sealed_v7()
    calls = []

    def invoke(stage, messages, schema, cancelled):
        calls.append(deepcopy(messages))
        result = json.loads("".join(FakeModel().stream(messages, Event())))
        Draft202012Validator(schema).validate(result)
        return result

    bound = StructuredUnitAssessor(invoke, Event(), audit_version=USE_AUDIT)(packet)
    assert bound["decision"] == "approve" and len(calls) == 1
    for unit in bound["unit_scopes"]:
        assert unit["response_fit"] == "fits" and "CPU" in unit["meaning"] and "CPU" in unit["use"]


@pytest.mark.parametrize("field", ["meaning", "use", "response_fit"])
def test_v7_normalized_fields_cannot_be_rewritten_after_binding(field):
    packet = sealed_v7()
    bound = StructuredUnitAssessor(
        lambda *args: supplied_v7(packet["payload"]), Event(), audit_version=USE_AUDIT
    )(packet)
    altered = deepcopy(bound)
    altered["unit_scopes"][0][field] = "被改写"
    with pytest.raises(ValueError):
        BoundedUnitReviewer._validate_opinion(altered, packet["sha256"], packet["payload"])


def test_v29_approved_prefix_is_immutable_and_forbids_whole_turn_repair(tmp_path, sealed_capture):
    shared = {"generation_calls": 0, "opinions": []}

    class SuppliedLateFitReject(FakeModel):
        def review_peer(self):
            return SuppliedLateFitReject()

        def stream(self, messages, cancelled):
            if "你只审查候选文字" not in messages[0]["content"]:
                shared["generation_calls"] += 1
                yield from super().stream(messages, cancelled)
                return
            view = json.loads(messages[-1]["content"])
            opinion = synthetic_unit_opinion(view)
            if shared["opinions"]:
                opinion["units"][-1].update(
                    response_fit="incompatible", use="供应商录制尾部当前不适用。"
                )
            shared["opinions"].append(deepcopy(opinion))
            yield canonical(opinion)

    model = SuppliedLateFitReject()
    _, _, memory, mirror, _ = setup(tmp_path, model)
    runtime = DefaultTextRuntime(
        memory, mirror, model, binding={"test": "cpu-v29-prefix"}, prompt_revision=USE_REVISION
    )
    session = TextSession(
        model, mirror, {}, prepare_turn=runtime.prepare_turn, before_start=runtime.before_start
    )
    try:
        session.send("把唱片放回架子了。")
        event = wait_for(lambda: runtime.poll()["event"])
        prefix = event["presentation"]["candidate"]["semantic_text"]
        runtime.ack(
            {
                "sequence": event["sequence"],
                "fingerprint": event["fingerprint"],
                "host_receipt": "synthetic-v29-prefix-immutable",
            }
        )
        wait_for(lambda: not session.snapshot()["busy"])
        assert session.snapshot()["attempt"]["status"] == "failed"
        assert shared["generation_calls"] == 1 and len(shared["opinions"]) == 2
        segments = runtime.store.segments(runtime.companion, runtime.claim.input_id)
        assert len(segments) == 1 and segments[0]["status"] == "confirmed"
        assert mirror.history()["messages"][-1]["content"] == prefix
        packets = list(sealed_capture.values())
        assert packets[1]["payload"]["displayed_prefix"] == prefix
        assert (
            packets[0]["payload"]["conversation_frame"]
            == packets[1]["payload"]["conversation_frame"]
        )
        record(
            "v29-prefix-immutable",
            {
                "packets": packets,
                "supplier": shared,
                "confirmed_event": event,
                "persistent_segments": segments,
                "semantic_effect_proven": False,
            },
        )
    finally:
        if session.snapshot()["busy"]:
            session.stop()
        memory.close()
        mirror.close()


def test_v8_maximum_declared_reading_and_failure_have_finite_cpu_size_not_hosted_capacity():
    packet = sealed_v8(candidate="".join("一般合成单元。" for _ in range(8)))
    opinion = supplied_v8(packet["payload"], meaning="字" * 128, use="字" * 32,
        statuses=("pass", "pass", "pass", "uncertain"), decision="reject",
        failure={"scope": "accuracy", "segment_id": None, "basis": "字" * 96})
    schema = stream_unit_review._verdict_unit_model_schema()
    Draft202012Validator(schema).validate(opinion)
    bound = bind_v8(packet, opinion)
    assert len(bound["unit_scopes"]) == len(opinion["units"]) == 8
    assert bound["model_opinion"] == opinion and bound["decision"] == "reject"
    record("v8-eight-max-fields-capacity", {"packet": packet, "opinion": opinion, "schema": schema,
        "canonical_utf8_bytes": len(canonical(opinion).encode()), "max_units": 8,
        "meaning_chars": 128, "use_chars": 32, "failure_basis_chars": 96,
        "review_output_tokens": 1536, "hosted_output_tokens_proven": False,
        "source_ID_length_and_multiple_ID_capacity_proven": False, "semantic_effect_proven": False})
    oversized = deepcopy(opinion)
    oversized["units"].append(deepcopy(opinion["units"][-1]))
    with pytest.raises(ValueError):
        bind_v8(packet, oversized)


def test_v8_public_fake_uses_only_explicit_supplier_declarations():
    packet = sealed_v8()
    calls = []

    def invoke(stage, messages, schema, cancelled):
        calls.append(deepcopy(messages))
        value = json.loads("".join(FakeModel().stream(messages, Event())))
        Draft202012Validator(schema).validate(value)
        return value

    bound = StructuredUnitAssessor(invoke, Event(), audit_version=VERDICT_AUDIT)(packet)
    assert len(calls) == 1 and bound["decision"] == "approve"
    assert bound["checks"] == ["pass"] * 4 and bound["failure"] is None
    assert all("CPU" in unit["meaning"] and "CPU" in unit["use"] for unit in bound["unit_scopes"])


@pytest.mark.parametrize("field", ["checks", "failure", "meaning", "use", "response_fit"])
def test_v8_normalized_declarations_cannot_be_rewritten_after_binding(field):
    packet = sealed_v8()
    value = supplied_v8(packet["payload"])
    bound = bind_v8(packet, value)
    altered = deepcopy(bound)
    if field in {"checks", "failure"}:
        altered[field] = "changed"
    else:
        altered["unit_scopes"][0][field] = "changed"
    with pytest.raises(ValueError):
        BoundedUnitReviewer._validate_opinion(altered, packet["sha256"], packet["payload"])


def test_v30_ack_prefix_remains_immutable_and_prevents_a_second_generation(
    tmp_path, sealed_capture,
):
    shared = {"generation_calls": 0, "opinions": []}

    class SuppliedLateReject(FakeModel):
        def review_peer(self):
            return SuppliedLateReject()

        def stream(self, messages, cancelled):
            if "你只审查候选文字" not in messages[0]["content"]:
                shared["generation_calls"] += 1
                yield from super().stream(messages, cancelled)
                return
            opinion = synthetic_unit_opinion(json.loads(messages[-1]["content"]))
            if shared["opinions"]:
                unit = opinion["units"][-1]
                unit.update(response_fit="incompatible", use="供应商录制尾部当前不适用。")
                opinion.update(decision="reject", checks=["pass", "pass", "pass", "fail"],
                    failure={"scope": "fit", "segment_id": unit["segment_id"],
                             "basis": "供应商录制此完整尾单元的适用失败。"})
            shared["opinions"].append(deepcopy(opinion))
            yield canonical(opinion)

    model = SuppliedLateReject()
    _, _, memory, mirror, _ = setup(tmp_path, model)
    runtime = DefaultTextRuntime(memory, mirror, model, binding={"test": "v30-cpu-prefix"},
                                 prompt_revision=VERDICT_REVISION)
    session = TextSession(model, mirror, {}, prepare_turn=runtime.prepare_turn,
                           before_start=runtime.before_start)
    try:
        session.send("把唱片放回架子了。")
        event = wait_for(lambda: runtime.poll()["event"])
        prefix = event["presentation"]["candidate"]["semantic_text"]
        runtime.ack({"sequence": event["sequence"], "fingerprint": event["fingerprint"],
                     "host_receipt": "synthetic-v30-ACK-only"})
        wait_for(lambda: not session.snapshot()["busy"])
        assert session.snapshot()["attempt"]["status"] == "failed"
        assert shared["generation_calls"] == 1 and len(shared["opinions"]) == 2
        segments = runtime.store.segments(runtime.companion, runtime.claim.input_id)
        assert len(segments) == 1 and segments[0]["status"] == "confirmed"
        assert mirror.history()["messages"][-1]["content"] == prefix
        packets = list(sealed_capture.values())
        assert packets[1]["payload"]["displayed_prefix"] == prefix
        record("v30-prefix-immutable", {"packets": packets, "supplier": shared,
               "confirmed_event": event, "persistent_segments": segments,
               "real_API_or_UI": False, "semantic_effect_proven": False})
    finally:
        if session.snapshot()["busy"]:
            session.stop()
        memory.close()
        mirror.close()
