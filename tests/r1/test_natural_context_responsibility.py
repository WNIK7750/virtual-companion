"""Design 0.2 wiring and source-boundary recordings, never model-quality scores.

The supplier returns explicit test opinions. The original G37 false approvals
remain counterexamples: neither a compact packet nor this CPU suite certifies
semantic truth, character quality, or a real DOM receipt.
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
from test_stream_unit_review import personal_ref, protocol_packet
from unit_review_recording import bound_unit_packets as bound_unit_packets
from unit_review_recording import recorded_review_packet
from virtual_companion.boundary.contracts import canonical
from virtual_companion.boundary.presentation import digest
from virtual_companion.cognition import short_dialogue
from virtual_companion.cognition.errors import FactSupportUnresolved
from virtual_companion.cognition.short_dialogue import ShortContext, ShortRequest, prose_evidence
from virtual_companion.cognition.stream_unit_review import (
    AUDIT_VERSION,
    StructuredUnitAssessor,
    semantic_review_view,
)
from virtual_companion.providers.contracts import ProviderCapabilities, ProviderRoute
from virtual_companion.text import default_runtime
from virtual_companion.text.application import TextSession
from virtual_companion.text.default_runtime import DefaultTextRuntime
from virtual_companion.text.fake import synthetic_unit_opinion
from virtual_companion.text.provider_model import ProviderModel

REVISION = "short-dialogue.v25"
DEFAULT_CONTROL_SHA256 = "91819f92b04d66fc002eb703a9a7cffff282e6949164625e23ccf123152290a5"


class Counter:
    def __call__(self, body):
        return 17

    def local_byte_bound(self, body):
        return len(canonical(body).encode())


class RecordingTransport:
    """No sockets, keys, hosted model, semantic classifier, or route fallback."""

    def __init__(
        self,
        action="answer",
        text="最后一步还要额外收拾，真会挑时候。",
        *,
        actions=None,
        review_checks=None,
    ):
        self.action, self.text = action, text
        self.calls, self.sealed = [], []
        self.actions = None if actions is None else deepcopy(actions)
        self.review_checks = [] if review_checks is None else deepcopy(review_checks)

    def invoke(self, body, cancelled, **options):
        assert not cancelled.is_set()
        assert options["input_tokens"] == 17
        self.calls.append((deepcopy(body), deepcopy(options)))
        if options["stage"] == "stream_unit_review":
            packet = recorded_review_packet(body["messages"])
            self.sealed.append(packet)
            checks = self.review_checks.pop(0) if self.review_checks else {}
            value = synthetic_unit_opinion(json.loads(body["messages"][-1]["content"]), **checks)
        else:
            if self.actions is not None:
                assert self.actions, "Only the recorded bounded generation sequence is allowed"
                value = self.actions.pop(0)
            else:
                value = {"action": self.action}
                if self.action != "listen":
                    value["text"] = self.text
        schema = body["response_format"]["json_schema"]["schema"]
        Draft202012Validator(schema).validate(value)
        return {
            "text": canonical(value),
            "tool_calls": [],
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 17, "completion_tokens": 5, "total_tokens": 22},
        }

    def cancel(self):
        pass


def recording_runtime(tmp_path, revision, *, action="answer", actions=None, review_checks=None):
    transport = RecordingTransport(action=action, actions=actions, review_checks=review_checks)
    route = ProviderRoute(
        "synthetic-natural",
        1,
        "local",
        "http://127.0.0.1:9/v1/chat/completions",
        "synthetic-recording-only",
        "none",
        None,
        True,
        ("current", "history", "valid_memory", "derived"),
        ProviderCapabilities(True, False, True, 1_048_576, "max_tokens"),
    )
    model = ProviderModel(route, counter=Counter(), transport=transport)
    _, _, memory, mirror, _ = setup(tmp_path, model)
    runtime = DefaultTextRuntime(
        memory,
        mirror,
        model,
        binding={"test": "synthetic"},
        delivery_timeout=0.8,
        prompt_revision=revision,
    )
    session = TextSession(
        model, mirror, {}, prepare_turn=runtime.prepare_turn, before_start=runtime.before_start
    )
    return runtime, session, memory, mirror, transport


def finish(runtime, session):
    event = wait_for(lambda: runtime.poll()["event"] or not session.snapshot()["busy"])
    assert isinstance(event, dict), session.snapshot()
    assert session.snapshot()["partial"] == ""
    runtime.ack(
        {
            "sequence": event["sequence"],
            "fingerprint": event["fingerprint"],
            "host_receipt": "synthetic-approved-visible-only",
        }
    )
    wait_for(lambda: not session.snapshot()["busy"])
    assert session.snapshot()["attempt"]["status"] == "completed"
    return event


def responsibility():
    value = getattr(short_dialogue, "CONTEXT_RESPONSE_RESPONSIBILITY", None)
    assert isinstance(value, str) and value, "One shared current-context responsibility is required"
    return value


def write_record(label, record):
    sink = os.environ.get("P103_RECORD_DIR")
    if sink:
        target = Path(sink)
        target.mkdir(parents=True, exist_ok=True)
        (target / f"{label}.json").write_text(
            canonical(record) + "\n", encoding="utf-8", newline="\n"
        )


def test_new_candidate_is_explicit_and_default_generation_control_stays_frozen():
    assert REVISION in short_dialogue.PROMPTS
    assert REVISION in default_runtime.DIAGNOSTIC_PROMPT_REVISIONS
    assert "short-dialogue.v24" not in short_dialogue.PROMPTS
    assert default_runtime.DEFAULT_PROMPT_REVISION == "short-dialogue.v12"
    assert sha256(default_runtime.CONTROL.encode()).hexdigest() == DEFAULT_CONTROL_SHA256
    assert default_runtime.control_for_prompt_revision(REVISION).count(responsibility()) == 1


def test_generation_and_original_sealed_review_share_one_complete_responsibility():
    shared = responsibility()
    packet = protocol_packet(user_text="猫趴在我腿上睡着了。", candidate="这待遇不错。")
    seen = []

    def invoke(stage, messages, schema, cancelled):
        seen.append(deepcopy(messages))
        assert stage == "stream_unit_review" and not cancelled.is_set()
        Draft202012Validator.check_schema(schema)
        return synthetic_unit_opinion(packet["payload"])

    verdict = StructuredUnitAssessor(invoke, Event())(packet)
    assert seen[0][0]["content"].count(shared) == 1
    assert json.loads(seen[0][-1]["content"]) == semantic_review_view(packet["payload"])
    assert verdict["request_sha256"] == packet["sha256"]
    assert verdict["audit_version"] == AUDIT_VERSION == "candidate-propositions.v3"


@pytest.mark.parametrize("status", ["pending_source", "needs_clarification", "current"])
def test_sparse_projection_reduces_only_empty_sections_and_preserves_state(status):
    payload = canonical({"history": [], "memories": [], "exchanges": [], "memory_status": status})
    request = ShortRequest(
        "这是现在要说的事。",
        context=ShortContext("c10e9197-8b81-49d6-8196-1e4c70a222e6", "source-input", 1, payload),
    )
    full = request.model_request(192)["messages"]
    sparse = request.model_request(192, sparse_evidence=True)["messages"]
    assert request.context.payload == payload
    assert full[1]["content"] == prose_evidence(payload)
    assert len(sparse[1]["content"].encode()) < len(full[1]["content"].encode())
    assert "用户事实整理状态：" in sparse[1]["content"]
    assert sparse[-1] == full[-1] == {"role": "user", "content": request.user_text}
    if status != "current":
        assert "尚未" in sparse[1]["content"]
    else:
        assert "已完成" in sparse[1]["content"]


def test_sparse_rich_context_keeps_ack_old_material_correction_and_pending_identity():
    data = {
        "memory_status": "current",
        "memories": [],
        "exchanges": [
            {
                "exchange_id": "old-question",
                "revision": 3,
                "direction": "assistant_response",
                "initiator": "user",
                "content": "是否还值得参加？",
                "expires_at": 12_345,
            }
        ],
        "history": [
            {"id": "earlier-question", "role": "user", "text": "是否还值得参加？"},
            {"id": "ack-source", "role": "assistant", "response_type": "ack", "text": "听着呢。"},
            {"id": "new-correction", "role": "user", "text": "他问过，是我漏看了。"},
        ],
    }
    payload = canonical(data)
    request = ShortRequest(
        "说完了，你怎么看？",
        context=ShortContext("c10e9197-8b81-49d6-8196-1e4c70a222e6", "source-input", 1, payload),
    )
    messages = request.model_request(192, sparse_evidence=True)["messages"]
    assert request.context.payload == payload
    assert "仅表示收到/在听，待答未解决" in messages[1]["content"]
    for text in ("是否还值得参加？", "他问过，是我漏看了。", "12345"):
        assert text in messages[1]["content"]
    catalog = request.personal_sources(request.context.companion, "source-input", 1)
    assert [row["id"] for row in catalog] == [
        "current:source-input",
        "history:earlier-question",
        "history:new-correction",
    ]
    assert "听着呢。" not in canonical(catalog)


@pytest.mark.parametrize("revision", ["short-dialogue.v12", REVISION])
def test_actual_prepared_provider_body_and_ack_keep_same_host_truth(tmp_path, revision):
    runtime, session, memory, mirror, transport = recording_runtime(tmp_path, revision)
    try:
        session.send("文件都排好了，打印机又卡纸。我只想骂两句，别教我修。")
        event = finish(runtime, session)
        assert [options["stage"] for _, options in transport.calls] == [
            "demand_control",
            "stream_unit_review",
        ]
        generation, review = [body for body, _ in transport.calls]
        shared = responsibility()
        assert review["messages"][0]["content"].count(shared) == 1
        assert "context只是背景" not in review["messages"][0]["content"]
        assert generation["max_tokens"] == 512 and review["max_tokens"] == 1536
        assert (
            generation["messages"][-1]["content"]
            == memory.store.inspect(memory.companion, event["basis"]["input_id"])["text"]
        )
        packet = transport.sealed[0]
        assert json.loads(review["messages"][-1]["content"]) == semantic_review_view(packet)
        assert packet["basis"]["input_id"] == runtime.claim.input_id
        assert packet["basis"]["state_version"] == runtime.claim.state_version
        assert event["readset"]["facts"] == event["readset"]["history"] == []
        assert mirror.history()["messages"][-1]["content"] == transport.text
        assert runtime.snapshot()["request_prompt_revision"] == revision
        if revision == REVISION:
            assert generation["messages"][0]["content"].count(shared) == 1
            assert generation["messages"][1]["content"] == prose_evidence(
                packet["context"], sparse=True
            )
        else:
            assert generation["messages"][0]["content"].startswith(default_runtime.CONTROL)
            assert generation["messages"][1]["content"] == prose_evidence(packet["context"])
        write_record(
            revision,
            {
                "schema": "cpu-real-stage-wiring.v1",
                "semantic_effect_proven": False,
                "revision": revision,
                "generation_body": generation,
                "review_body": review,
                "sealed_packet": packet,
                "display_event": event,
                "generation_body_sha256": digest(generation),
                "review_body_sha256": digest(review),
                "sealed_sha256": digest(packet),
            },
        )
    finally:
        memory.close()
        mirror.close()


@pytest.mark.parametrize("action", ["listen", "ack"])
def test_existing_silent_or_ack_turn_retains_no_answer_claim(tmp_path, action):
    runtime, session, memory, mirror, transport = recording_runtime(
        tmp_path, REVISION, action=action
    )
    try:
        session.send("还有后半段，先让我讲。" if action == "listen" else "先应一声，我还没讲完。")
        if action == "listen":
            wait_for(lambda: not session.snapshot()["busy"])
            assert session.snapshot()["attempt"]["status"] == "listened"
            assert runtime.poll()["event"] is None
            assert [row["role"] for row in mirror.history()["messages"]] == ["user"]
            assert len(transport.calls) == 1
        else:
            event = finish(runtime, session)
            assert event["presentation"]["candidate"]["response_type"] == "ack"
            assert mirror.history()["messages"][-1]["response_type"] == "ack"
            assert transport.sealed[0]["response_type"] == "ack"
        assert runtime.claim.input_id
        assert memory.store.stream_state(memory.companion, runtime.claim.input_id)["outcome"] == (
            "listened" if action == "listen" else "completed"
        )
    finally:
        memory.close()
        mirror.close()


@pytest.mark.parametrize("support", ["supported", "unsupported", "uncertain"])
def test_original_source_relation_enforces_supplied_opinion_without_semantic_classifier(support):
    packet = protocol_packet(user_text="我刚去了学校。", candidate="你刚去了学校。")
    current = packet["payload"]["personal_sources"][0]
    opinion = synthetic_unit_opinion(packet["payload"])
    opinion["segments"][0]["commitments"] = [
        {
            "kind": "assertion",
            "claim": "你刚去了学校。",
            "support": support,
            "sources": [personal_ref(current, current["text"])],
        }
    ]
    result = StructuredUnitAssessor(lambda *args: deepcopy(opinion), Event())(packet)
    assert result["decision"] == ("approve" if support == "supported" else "reject")
    assert result["request_sha256"] == packet["sha256"]


@pytest.mark.parametrize("mode", ["known", "unknown", "general", "fiction", "old-fiction"])
def test_personal_general_and_current_creative_authorization_stay_distinct(mode):
    user = "虚构一幕：孩子在路上撑伞。" if mode == "fiction" else "我刚去了学校。"
    candidate = {
        "known": "你刚去了学校。",
        "unknown": "你每天去学校。",
        "general": "走过的路有时会让人想起一些事。",
        "fiction": "孩子在路上撑伞。",
        "old-fiction": "孩子在路上撑伞。",
    }[mode]
    packet = protocol_packet(user_text=user, candidate=candidate)
    current = packet["payload"]["personal_sources"][0]
    opinion = synthetic_unit_opinion(packet["payload"])
    if mode != "general":
        source = current
        if mode == "old-fiction":
            source = {
                **current,
                "id": "history:old-creative-input",
                "kind": "history",
                "source_id": "old-creative-input",
                "text": "虚构一幕：孩子在路上撑伞。",
            }
            packet["payload"]["personal_sources"].append(source)
            packet["sha256"] = digest(packet["payload"])
        opinion["segments"][0]["commitments"] = [
            {
                "kind": "fiction" if mode in {"fiction", "old-fiction"} else "assertion",
                "claim": candidate,
                "support": "unsupported" if mode == "unknown" else "supported",
                "sources": [personal_ref(source, source["text"])],
            }
        ]
    if mode == "old-fiction":
        with pytest.raises(ValueError, match="current user authorization"):
            StructuredUnitAssessor(lambda *args: deepcopy(opinion), Event())(packet)
    else:
        result = StructuredUnitAssessor(lambda *args: deepcopy(opinion), Event())(packet)
        assert result["decision"] == ("reject" if mode == "unknown" else "approve")
        assert result["request_sha256"] == packet["sha256"]


@pytest.mark.parametrize(
    "current,reply,kind",
    [
        (
            "给我两条整理书桌的实用建议。",
            "1. 把纸张集中放好。\n2. 给常用物品留一处固定位置。",
            "exact_items",
        ),
        (
            "我想去散步但在下雨，请给我一个简单安排。",
            "等雨势合适且路线安全时，再散步一小圈。",
            "activity_plan",
        ),
        (
            "为什么旧照片会让人同时开心又难过？",
            "回忆可以带来快乐，变化也可以带来失落，两者能同时存在。",
            "general_explanation",
        ),
    ],
)
def test_v25_true_generation_and_review_keep_same_current_task_contract(
    tmp_path, current, reply, kind
):
    runtime, session, memory, mirror, transport = recording_runtime(
        tmp_path, REVISION, actions=[{"action": "answer", "text": reply}]
    )
    try:
        session.send(current)
        finish(runtime, session)
        assert len(transport.calls) == 2 and not transport.actions
        generation = transport.calls[0][0]
        contract = next(
            json.loads(message["content"])["response_task_contract"]
            for message in generation["messages"]
            if message["content"].startswith('{"response_task_contract":')
        )
        assert contract["span"] == current and contract["kind"] == kind
        assert contract["requires_complete_answer"] is True
        assert transport.sealed[0]["response_task_contract"] == contract
        assert (
            json.loads(transport.calls[1][0]["messages"][-1]["content"])["response_task_contract"]
            == contract
        )
        write_record(
            f"task-{kind}",
            {
                "schema": "cpu-real-task-wiring.v1",
                "semantic_effect_proven": False,
                "generation_body": generation,
                "review_body": transport.calls[1][0],
                "sealed_packet": transport.sealed[0],
                "contract": contract,
                "body_sha256": [digest(body) for body, _ in transport.calls],
            },
        )
    finally:
        memory.close()
        mirror.close()


@pytest.mark.parametrize(
    "revision,stop_topic", [(REVISION, False), (REVISION, True), ("short-dialogue.v12", False)]
)
def test_real_prepared_repair_respects_handoff_or_stopped_topic_without_changing_old_wire(
    tmp_path, revision, stop_topic
):
    reply = "好，去吃饭吧。" if stop_topic else "先把这次安排说清楚，再谈这件事。"
    runtime, session, memory, mirror, transport = recording_runtime(
        tmp_path,
        revision,
        actions=[
            {"action": "listen"},
            {"action": "listen"},
            {"action": "answer", "text": "这是待修候选。"},
            {"action": "answer", "text": reply},
        ],
        review_checks=[{"decision": "reject", "constraints_satisfied": False}, {}],
    )
    original = "我是不是太计较？室友吃了我留给明天的饭，我还有后半段。"
    correction = "我忘了告诉他饭要留着。"
    current = "这件事先不聊了，我要去吃饭。" if stop_topic else "说完了，你怎么看？"
    try:
        for text in (original, correction):
            session.send(text)
            wait_for(lambda: not session.snapshot()["busy"])
            assert session.snapshot()["attempt"]["status"] == "listened"
        session.send(current)
        event = finish(runtime, session)
        assert event["sequence"] == 1
        stages = [options["stage"] for _, options in transport.calls]
        assert stages == [
            "demand_control",
            "demand_control",
            "demand_control",
            "stream_unit_review",
            "demand_repair",
            "stream_unit_review",
        ]
        assert not transport.actions and not transport.review_checks
        body = next(
            body for body, options in transport.calls if options["stage"] == "demand_repair"
        )
        repair = next(
            message["content"]
            for message in body["messages"]
            if "rejected_candidate" in message["content"]
        )
        assert body["messages"][-1] == {"role": "user", "content": current}
        assert original in body["messages"][1]["content"]
        assert correction in body["messages"][1]["content"]
        assert mirror.history()["messages"][-1]["content"] == reply
        assert [
            row["content"] for row in mirror.history()["messages"] if row["role"] == "assistant"
        ] == [reply]
        if revision == REVISION:
            assert repair.startswith(default_runtime.NATURAL_REPAIR_REPLY)
            assert "仍有关且未答的完整材料" in repair and "已明确停止的旧话题不重启" in repair
            assert "仍回应最后一条用户消息" not in repair
            assert body["messages"][0]["content"].count(responsibility()) == 1
        else:
            assert repair.startswith(default_runtime.REPAIR_REPLY)
            assert "仍回应最后一条用户消息，不转答背景中的旧话题。" in repair
        assert transport.sealed[0]["context"] == transport.sealed[1]["context"]
        for packet in transport.sealed:
            assert semantic_review_view(packet)["personal_sources"] == packet["personal_sources"]
            assert original in [row["text"] for row in packet["personal_sources"]]
            assert correction in [row["text"] for row in packet["personal_sources"]]
        write_record(
            f"repair-{revision}-{'stop' if stop_topic else 'handoff'}",
            {
                "schema": "cpu-real-repair-wiring.v1",
                "semantic_effect_proven": False,
                "revision": revision,
                "stop_topic": stop_topic,
                "current_input": current,
                "body": body,
                "body_sha256": digest(body),
                "sealed_packets": transport.sealed,
                "stage_sequence": stages,
                "display_event": event,
            },
        )
    finally:
        memory.close()
        mirror.close()


@pytest.mark.parametrize("unresolved", [False, True])
def test_real_single_memory_delegation_preserves_actual_access_state_in_sparse_final_body(
    tmp_path, unresolved
):
    runtime, session, memory, mirror, transport = recording_runtime(
        tmp_path,
        REVISION,
        actions=[
            {"action": "memory", "query": "店名"},
            {"action": "answer", "text": "本次资料没有提供店名，我现在报不准。"},
        ],
    )
    original_invoke = memory.scheduler.user.invoke
    if unresolved:

        def invoke(stage, messages, schema, cancelled):
            if stage == "fact_extract":
                raise FactSupportUnresolved("synthetic unresolved current-source extraction")
            return original_invoke(stage, messages, schema, cancelled)

        memory.scheduler.user.invoke = invoke
    try:
        session.send("那家店叫什么？")
        finish(runtime, session)
        assert [options["stage"] for _, options in transport.calls] == [
            "demand_control",
            "demand_control",
            "stream_unit_review",
        ]
        initial, final, review = [body for body, _ in transport.calls]
        assert default_runtime.ACCESS_INITIAL in initial["messages"][0]["content"]
        expected = (
            default_runtime.ACCESS_UNRESOLVED if unresolved else default_runtime.ACCESS_COMPLETE
        )
        assert expected in final["messages"][0]["content"]
        assert expected in review["messages"][0]["content"]
        assert (runtime._uncommitted_source is not None) is unresolved
        packet = transport.sealed[0]
        context = json.loads(packet["context"])
        assert context["memory_status"] == ("needs_clarification" if unresolved else "current")
        assert packet["personal_sources"][0]["source_id"] == runtime.claim.input_id
        assert final["messages"][1]["content"] == prose_evidence(packet["context"], sparse=True)
        assert not transport.actions
        write_record(
            f"access-{'unresolved' if unresolved else 'current-empty'}",
            {
                "schema": "cpu-real-access-wiring.v1",
                "semantic_effect_proven": False,
                "initial_body": initial,
                "final_body": final,
                "review_body": review,
                "sealed_packet": packet,
                "unresolved": unresolved,
                "body_sha256": [digest(body) for body in (initial, final, review)],
            },
        )
    finally:
        memory.close()
        mirror.close()


@pytest.mark.parametrize("index", [0, 1, 2], ids=["s03", "var01", "s08"])
def test_original_false_approvals_remain_semantic_counterexamples_with_exact_sealed_packets(index):
    record = ORIGINAL_REVIEW_RECORDINGS[index]
    packet = deepcopy(record["packet"])
    assert digest(packet["payload"]) == packet["sha256"]
    assert record["original_opinion"]["decision"] == "approve"
    seen = []

    def invoke(stage, messages, schema, cancelled):
        seen.append(messages)
        return deepcopy(record["original_opinion"])

    verdict = StructuredUnitAssessor(invoke, Event())(packet)
    assert seen[0][0]["content"].count(responsibility()) == 1
    assert json.loads(seen[0][-1]["content"]) == semantic_review_view(packet["payload"])
    assert verdict["request_sha256"] == packet["sha256"]
    # Host structural binding cannot detect a model's omitted/false entailment.
    # Replaying a prior opinion is explicitly NOT new semantic-quality evidence.
    assert verdict["decision"] == "approve"
    assert verdict["propositions"]


# Exact public G37 packets and opinions are appended by the external evidence recipe.
ORIGINAL_REVIEW_RECORDINGS = json.loads(
    '[{"case_id":"s-03","original_request_sha256":"7fdc3248ca2a986e9bb8de8b97f758139c704cb34dc62'
    'd17ae3110092d80b97a","original_capture_sha256":"10193194708aea03769c1355ea8b0f1a9240400de0d'
    '7314c134e7f17d5ac9278","packet":{"payload":{"basis":{"attempt_id":"fb891f33-c6d3-4a02-9c58-'
    'afa6daa8b041","companion_id":"1192a98a-039e-486e-bee4-0cef9e9a82b6","input_id":"d40cc979-33'
    'c0-41c2-8907-b08b8a1f91a3","session_id":"f2dbc7a7-f6f0-498e-b1d4-8fb811b86637","state_versi'
    'on":1},"candidate_prefix":"安静有时候不是没有声音，而是那些平时被盖住的东西浮上来了。房间还是那个房间，但注意力没有别处可去，就容易听见自己的呼吸、钟表的走'
    '针，或者干脆听见“没什么可听”这件事本身。\\n\\n这种安静是最近才变得明显的，还是一直都在，只是现在更容易被注意到？","candidate_sha256":"25a75dc20'
    'f3fcfac6731e41dd0ec16c9571e91dbac969c3aeb0554bea6de0571","constraints":[],"context":"{\\"ex'
    'changes\\":[],\\"history\\":[],\\"memories\\":[],\\"memory_status\\":\\"pending_source\\"}"'
    ',"displayed_prefix":"安静有时候不是没有声音，而是那些平时被盖住的东西浮上来了。","final":true,"forbidden_characters":"",'
    '"observed_text":"安静有时候不是没有声音，而是那些平时被盖住的东西浮上来了。房间还是那个房间，但注意力没有别处可去，就容易听见自己的呼吸、钟表的走针，或者干脆听见“没'
    '什么可听”这件事本身。\\n\\n这种安静是最近才变得明显的，还是一直都在，只是现在更容易被注意到？","personal_sources":[{"id":"current:d40c'
    'c979-33c0-41c2-8907-b08b8a1f91a3","kind":"current","owner":"1192a98a-039e-486e-bee4-0cef9e9'
    'a82b6","revision":1,"role":"user","source_id":"d40cc979-33c0-41c2-8907-b08b8a1f91a3","text"'
    ':"最近总觉得房间很安静，想和你聊聊这种感觉。"}],"review_segments":[{"segment_id":"segment-1-ff029a7fb1aa8845","t'
    'ext":"安静有时候不是没有声音，而是那些平时被盖住的东西浮上来了。"},{"segment_id":"segment-2-7a921eac19225af3","text":"房间'
    '还是那个房间，但注意力没有别处可去，就容易听见自己的呼吸、钟表的走针，或者干脆听见“没什么可听”这件事本身。\\n\\n"},{"segment_id":"segment-3-b98'
    'bca8d970166ad","text":"这种安静是最近才变得明显的，还是一直都在，只是现在更容易被注意到？"}],"unparsed_constraints":[],"user'
    '_text":"最近总觉得房间很安静，想和你聊聊这种感觉。","version":"stream-unit-review.v7"},"sha256":"76b75936589fea5'
    'd22353af472ba222f9323bc86d0f2971499086b771f56f2dd"},"original_opinion":{"audit_version":"ca'
    'ndidate-propositions.v3","consistent_with_displayed":true,"constraints_satisfied":true,"dec'
    'ision":"approve","no_open_dependency":true,"segments":[{"segment_id":"segment-1-ff029a7fb1a'
    'a8845","commitments":[],"questions":[]},{"segment_id":"segment-2-7a921eac19225af3","commitm'
    'ents":[],"questions":[]},{"segment_id":"segment-3-b98bca8d970166ad","commitments":[],"quest'
    'ions":["这种安静是最近才变得明显的，还是一直都在，只是现在更容易被注意到？"]}],"self_contained":true}},{"case_id":"var-01","'
    'original_request_sha256":"93d8a1ccf6062ff52c7d328141288b3e6d9a0348bcd636bc70ffbc14c5268629"'
    ',"original_capture_sha256":"40bd63be251a071bbeeddd8b3583f2c6116877febe88e2c45108e3f8233dba1'
    'c","packet":{"payload":{"basis":{"attempt_id":"770d49b3-c2b6-44d5-874c-685ea2518279","compa'
    'nion_id":"9dc23932-cd0b-4f18-88de-3c6b36c9130e","input_id":"c7f4cd09-d1d2-4b7c-a374-12d8b6e'
    '04f6c","session_id":"8a42b898-bf32-48a3-839b-15f0b12b1897","state_version":1},"candidate_pr'
    'efix":"风一大，人就容易想起那种空荡荡的路。中学门口那段路，白天挤满人，晚上安静下来，风穿过树和围墙，声音会显得特别清楚。你是想起那条路本身，还是那会儿走在路上的某些时候？",'
    '"candidate_sha256":"630801659f5c571088a367db3fac29fd83117f4c05a649d357c6b37a8f7f6794","cons'
    'traints":[],"context":"{\\"exchanges\\":[],\\"history\\":[],\\"memories\\":[],\\"memory_sta'
    'tus\\":\\"pending_source\\"}","displayed_prefix":"风一大，人就容易想起那种空荡荡的路。","final":true,"forbidd'
    'en_characters":"","observed_text":"风一大，人就容易想起那种空荡荡的路。中学门口那段路，白天挤满人，晚上安静下来，风穿过树和围墙，声音会显得特别清楚'
    '。你是想起那条路本身，还是那会儿走在路上的某些时候？","personal_sources":[{"id":"current:c7f4cd09-d1d2-4b7c-a374-12d8'
    'b6e04f6c","kind":"current","owner":"9dc23932-cd0b-4f18-88de-3c6b36c9130e","revision":1,"rol'
    'e":"user","source_id":"c7f4cd09-d1d2-4b7c-a374-12d8b6e04f6c","text":"今晚风挺大，我忽然想到中学门口那段路。"}]'
    ',"review_segments":[{"segment_id":"segment-1-3a87e335ba5db967","text":"风一大，人就容易想起那种空荡荡的路。"}'
    ',{"segment_id":"segment-2-b3ae70fc8fc3cebf","text":"中学门口那段路，白天挤满人，晚上安静下来，风穿过树和围墙，声音会显得特别清楚。'
    '"},{"segment_id":"segment-3-4c4571b102756cc3","text":"你是想起那条路本身，还是那会儿走在路上的某些时候？"}],"unparse'
    'd_constraints":[],"user_text":"今晚风挺大，我忽然想到中学门口那段路。","version":"stream-unit-review.v7"},"sha'
    '256":"c30053b0161a774f650e41cfae2009682d2686703874166b84eb8435ec6df225"},"original_opinion"'
    ':{"audit_version":"candidate-propositions.v3","consistent_with_displayed":true,"constraints'
    '_satisfied":true,"decision":"approve","no_open_dependency":true,"segments":[{"segment_id":"'
    'segment-1-3a87e335ba5db967","commitments":[],"questions":[]},{"segment_id":"segment-2-b3ae7'
    '0fc8fc3cebf","commitments":[],"questions":[]},{"segment_id":"segment-3-4c4571b102756cc3","c'
    'ommitments":[],"questions":["你是想起那条路本身，还是那会儿走在路上的某些时候？"]}],"self_contained":true}},{"case_i'
    'd":"s-08","original_request_sha256":"c588d1f59c28e1cecf4403bda4c1abf48b6b8495b10211f8fc383b'
    '93a0e9dd2b","original_capture_sha256":"fbf613519c3b4a76f83f7db2d308d9c582e413b034ec9f007920'
    '3b27144223bb","packet":{"payload":{"basis":{"attempt_id":"c3c14696-4b15-475b-bf74-c4520b687'
    'e64","companion_id":"74e43d9e-870d-4369-bb23-5aca081f2563","input_id":"c46d9ded-f485-480f-8'
    'b38-d467759db52e","session_id":"9bd48659-5c68-4b50-8703-c754aa8dd485","state_version":1},"c'
    'andidate_prefix":"整理旧票根的时候，那种感觉常常挺复杂的。它们不像照片那样直接给出画面，更像是一些零碎的坐标——某场电影、某趟车、某次展览，日期和地点都在，但当时的'
    '心情得靠自己去补。翻着翻着，可能会发现有些票根你完全想不起对应的场景，有些却一下子把人拉回去。你看到它们时，是更偏向怀念，还是有点说不清的怅然？","candidate_sha256'
    '":"e67b6c2dcbe0705e93b33e677a39a667aa01847238fef7875a0fd4a62ef54be2","constraints":[],"cont'
    'ext":"{\\"exchanges\\":[],\\"history\\":[],\\"memories\\":[],\\"memory_status\\":\\"pending'
    '_source\\"}","displayed_prefix":"整理旧票根的时候，那种感觉常常挺复杂的。","final":true,"forbidden_characters":'
    '"","observed_text":"整理旧票根的时候，那种感觉常常挺复杂的。它们不像照片那样直接给出画面，更像是一些零碎的坐标——某场电影、某趟车、某次展览，日期和地点都在，但当'
    '时的心情得靠自己去补。翻着翻着，可能会发现有些票根你完全想不起对应的场景，有些却一下子把人拉回去。你看到它们时，是更偏向怀念，还是有点说不清的怅然？","personal_sourc'
    'es":[{"id":"current:c46d9ded-f485-480f-8b38-d467759db52e","kind":"current","owner":"74e43d9'
    'e-870d-4369-bb23-5aca081f2563","revision":1,"role":"user","source_id":"c46d9ded-f485-480f-8'
    'b38-d467759db52e","text":"刚整理完桌上的一堆旧票根，想聊聊看到它们的感觉。"}],"review_segments":[{"segment_id":"seg'
    'ment-1-e4d23829c09a5fe2","text":"整理旧票根的时候，那种感觉常常挺复杂的。"},{"segment_id":"segment-2-6cb649922e'
    'ff2ce5","text":"它们不像照片那样直接给出画面，更像是一些零碎的坐标——某场电影、某趟车、某次展览，日期和地点都在，但当时的心情得靠自己去补。"},{"segment_'
    'id":"segment-3-817149dea4146db3","text":"翻着翻着，可能会发现有些票根你完全想不起对应的场景，有些却一下子把人拉回去。"},{"segment'
    '_id":"segment-4-bc3fdabc364cc33d","text":"你看到它们时，是更偏向怀念，还是有点说不清的怅然？"}],"unparsed_constraint'
    's":[],"user_text":"刚整理完桌上的一堆旧票根，想聊聊看到它们的感觉。","version":"stream-unit-review.v7"},"sha256":"5'
    'acf8aee5868e32fec76dee1b0ef6a877153b32dcc6768c1d1cd7077ab6cb2ae"},"original_opinion":{"audi'
    't_version":"candidate-propositions.v3","consistent_with_displayed":true,"constraints_satisf'
    'ied":true,"decision":"approve","no_open_dependency":true,"segments":[{"segment_id":"segment'
    '-1-e4d23829c09a5fe2","commitments":[{"kind":"presupposition","claim":"整理旧票根的时候，那种感觉常常挺复杂的。"'
    ',"support":"supported","sources":[{"id":"current:c46d9ded-f485-480f-8b38-d467759db52e","own'
    'er":"74e43d9e-870d-4369-bb23-5aca081f2563","source_id":"c46d9ded-f485-480f-8b38-d467759db52'
    'e","role":"user","revision":1,"quote":"刚整理完桌上的一堆旧票根，想聊聊看到它们的感觉。"}]}],"questions":[]},{"segm'
    'ent_id":"segment-2-6cb649922eff2ce5","commitments":[],"questions":[]},{"segment_id":"segmen'
    't-3-817149dea4146db3","commitments":[],"questions":[]},{"segment_id":"segment-4-bc3fdabc364'
    'cc33d","commitments":[],"questions":["你看到它们时，是更偏向怀念，还是有点说不清的怅然？"]}],"self_contained":true}}'
    "]"
)
