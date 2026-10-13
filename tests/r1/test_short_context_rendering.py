"""The model-facing evidence rendering is a product choice, never a silent one.

The streaming path attaches the admission's canonical JSON payload as a user message.
A 4B local model sometimes answers *in* that shape (the recorded desktop counterexample
`推荐一部电影。` returned a JSON body while 100 plain-path samples without attached
evidence did not), so the evidence gets an opt-in prose rendering. The default stays
the JSON message until a same-condition comparison adopts the other, and the host keeps
parsing the stored JSON payload for its read-set recheck in both cases.
"""

import json

import pytest
from virtual_companion.boundary.contracts import canonical
from virtual_companion.cognition.short_dialogue import (
    DEFAULT_CONTEXT_RENDERING,
    SYSTEM,
    ShortContext,
    ShortRequest,
)

COMPANION = "00000000-0000-0000-0000-000000000001"
EVIDENCE = {
    "history": [{"id": "turn-1", "role": "user", "text": "我今天有点累。"},
                {"id": "turn-2", "role": "assistant", "text": "那就早点休息。"}],
    "memory_status": "current",
    "memories": [{"record_id": "fact-1", "version": 2,
                  "content": "用户不喜欢吵闹的音乐。"}],
    "exchanges": [
        {"exchange_id": "ask-1", "revision": 1, "direction": "user_question",
         "initiator": "assistant", "content": "周末想去哪？"},
        {"exchange_id": "request-1", "revision": 1, "direction": "assistant_response",
         "initiator": "user", "content": "帮我想想周末怎么安排。"},
    ],
}
ENGLISH_KEYS = ("history", "memory_status", "memories", "exchanges", "record_id",
                "exchange_id", "role", "text", "version", "revision")


def context(payload=None, rendering=None):
    values = {"companion": COMPANION, "input_id": "input-1", "state_version": 3,
              "payload": canonical(EVIDENCE) if payload is None else payload}
    if rendering is not None:
        values["rendering"] = rendering
    return ShortContext(**values)


def request(rendering=None):
    return ShortRequest("推荐一部电影。", constraint_source="test",
                        context=context(rendering=rendering))


def evidence_message(rendering):
    messages = request(rendering).model_request(192)["messages"]
    assert messages[0]["role"] == "system" and messages[-1]["role"] == "user"
    return messages[1:-1]


def test_the_adopted_default_is_prose_and_json_stays_selectable():
    """Adopted 2026-09-20 after the richer-evidence verification (EVD S11)."""

    assert DEFAULT_CONTEXT_RENDERING == "prose"
    body = request()
    messages = body.model_request(192)["messages"]
    assert len(messages) == 3
    assert messages[1]["role"] == "user"
    assert messages[1]["content"].startswith("以下是已确认的背景资料")
    assert "{" not in messages[1]["content"] and '"' not in messages[1]["content"]
    assert messages[2] == {"role": "user", "content": body.user_text}
    assert messages[0]["content"].startswith(
        SYSTEM.replace("不声称已执行或保存。", "操作反馈须符合本次实际证据。")
    )
    assert messages[0]["content"].endswith(body.operation_constraints())


def test_the_recorded_rollback_rendering_still_emits_the_json_evidence_message():
    messages = request("json").model_request(192)["messages"]
    assert messages[1] == {"role": "user", "content": canonical(EVIDENCE)}
    assert "附带JSON" in messages[0]["content"]


def test_prose_rendering_carries_no_json_syntax_or_english_field_names():
    messages = evidence_message("prose")
    assert len(messages) == 1
    content = messages[0]["content"]
    for banned in ("{", "}", "[", "]", '"', "'", ":", "_"):
        assert banned not in content, banned
    for key in ENGLISH_KEYS:
        assert key not in content, key
    assert "我今天有点累。" in content
    assert "那就早点休息。" in content
    assert "用户不喜欢吵闹的音乐。" in content
    assert "周末想去哪？" in content
    assert "帮我想想周末怎么安排。" in content
    assert "助手曾提问，需结合后续对话核验是否仍待回应" in content
    assert "用户曾请求，需结合已确认对话核验是否已回应" in content
    assert "历史待续线索（是否仍待回应，须结合已确认对话核验）" in content
    assert "不要复述清单" in content
    assert messages[0]["role"] == "user"


def test_prose_rendering_states_pending_memory_and_empty_collections():
    payload = canonical({**EVIDENCE, "memory_status": "pending_source", "memories": [],
                         "exchanges": []})
    content = ShortContext(COMPANION, "input-1", 3, payload,
                           rendering="prose").payload
    assert json.loads(content)["memory_status"] == "pending_source"
    messages = ShortRequest("推荐一部电影。", constraint_source="test",
                            context=context(payload, "prose")).model_request(192)["messages"]
    rendered = messages[1]["content"]
    assert "尚未完成" in rendered or "未完成" in rendered
    assert "暂未提供" in rendered


@pytest.mark.parametrize("status", ["pending_source", "needs_clarification", None, "failed"])
def test_unavailable_empty_projection_does_not_claim_no_personal_records(status):
    payload = canonical({"history": [], "memory_status": status,
                         "memories": [], "exchanges": []})
    messages = ShortRequest("我的饮食偏好是什么？", context=context(payload)).model_request(
        192)["messages"]
    evidence = messages[1]["content"]
    for label in ("已确认的对话历史", "相关长期记忆", "历史待续线索"):
        assert f"{label}：无" not in evidence
        line = evidence[evidence.index(label):].splitlines()[0]
        assert "未提供" in line or "暂未提供" in line
    assert "不能据此判断有无记录" in evidence
    assert json.loads(context(payload).payload)["memories"] == []


def test_current_empty_projection_remains_no_available_related_evidence():
    payload = canonical({"history": [], "memory_status": "current",
                         "memories": [], "exchanges": []})
    messages = ShortRequest("我的饮食偏好是什么？", context=context(payload)).model_request(
        192)["messages"]
    evidence = messages[1]["content"]
    assert "用户事实整理状态：本次用户事实整理已完成" in evidence
    assert "相关长期记忆：无" in evidence
    assert "暂未提供" not in evidence
    assert "本次相关上下文" in messages[0]["content"]


@pytest.mark.parametrize("rendering", ["prose", "json"])
def test_evidence_system_preserves_missing_versus_absent_for_both_renderings(rendering):
    payload = canonical({"history": [], "memory_status": "pending_source",
                         "memories": [], "exchanges": []})
    messages = ShortRequest("我的饮食偏好是什么？", context=context(payload, rendering)
                            ).model_request(192)["messages"]
    assert "不等于没有记录" in messages[0]["content"]
    if rendering == "json":
        assert json.loads(messages[1]["content"]) == json.loads(payload)


def test_prose_rendering_distinguishes_clarification_from_background_progress():
    payload = canonical({**EVIDENCE, "memory_status": "needs_clarification",
                         "memories": [], "exchanges": []})
    rendered = ShortRequest(
        "你还记得吗？", constraint_source="test", context=context(payload, "prose")
    ).model_request(192)["messages"][1]["content"]
    assert "需要用户澄清" in rendered
    assert "尚未保存" in rendered
    assert "只有当前消息确在请求更正或遗忘时，才询问目标" in rendered


def test_the_stored_payload_stays_parseable_for_the_read_set_recheck():
    bound = context(rendering="prose")
    assert json.loads(bound.payload) == EVIDENCE
    assert bound.rendering == "prose"


@pytest.mark.parametrize("status", [None, "unknown", "unavailable", "failed"])
def test_unknown_memory_status_is_never_rendered_as_completed(status):
    payload = canonical({**EVIDENCE, "memory_status": status,
                         "memories": [], "exchanges": []})
    rendered = ShortRequest("还记得吗？", context=context(payload)).model_request(192)
    evidence = rendered["messages"][1]["content"]
    assert "用户事实整理状态：本次用户事实整理已完成" not in evidence
    assert "未确认" in evidence


@pytest.mark.parametrize("event_time", [None, "2024-04-01T00:00:00+00:00"])
def test_memory_rendering_distinguishes_event_time_from_audit_time(event_time):
    payload = canonical({**EVIDENCE, "memories": [{
        "record_id": "trip", "version": 2, "content": "我去过杭州",
        "event_time": event_time, "recorded_at": "2026-09-27T00:00:00Z",
    }]})
    body = ShortRequest("我什么时候去过杭州？", context=context(payload)).model_request(192)
    evidence = body["messages"][1]["content"]
    assert f"发生时间：{event_time or '不详'}" in evidence
    assert "2026-09-27" not in evidence
    assert "录入时间" in body["messages"][0]["content"]


def test_unknown_rendering_is_rejected():
    with pytest.raises(ValueError, match="rendering"):
        context(rendering="markdown")
    with pytest.raises(ValueError, match="rendering"):
        ShortContext(COMPANION, "input-1", 3, canonical(EVIDENCE), rendering="")


@pytest.mark.parametrize("action", ["forget", "correct"])
@pytest.mark.parametrize("status", ["committed", "already_applied"])
def test_current_operation_rendering_distinguishes_effect_from_empty_recall(action, status):
    operation = {
        "protocol": "memory-command-outcome.v1", "source_id": "input-1",
        "source_sha256": "a" * 64, "operation_id": "operation-1",
        "action": action, "status": status, "receipt_sha256": "b" * 64,
    }
    payload = canonical({"history": [], "memory_status": "current", "memories": [],
                         "exchanges": [], "current_operation": operation})
    body = ShortRequest("请处理这条记忆", context=context(payload)).model_request(192)
    rendered = body["messages"][1]["content"]
    assert "本次请求的记忆操作结果" in rendered
    assert ("遗忘" if action == "forget" else "更正") in rendered
    assert ("已完成并生效" if status == "committed" else "无需重复更改") in rendered
    assert "operation-1" not in rendered and "a" * 64 not in rendered
    assert "物理删除" in rendered
    assert "本次实际操作" in body["messages"][0]["content"]
    assert "不降格为排队" in body["messages"][0]["content"]
    assert "不夸大为再次修改" in body["messages"][0]["content"]
    assert "某个附近地点，不声称已执行或保存。" not in body["messages"][0]["content"]
    assert json.loads(context(payload).payload)["current_operation"] == operation
    as_json = ShortRequest("请处理这条记忆", context=context(payload, "json"))
    assert json.loads(as_json.model_request(192)["messages"][1]["content"]) == json.loads(payload)
