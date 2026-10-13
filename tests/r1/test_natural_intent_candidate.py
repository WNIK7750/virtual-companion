"""P1-03 candidate on the actual demand, preparation, review and ACK path.

The inputs include recorded natural-dialogue failures and normal controls. Model
outputs/opinions here are recordings: these tests prove which instructions and
evidence reach generation/review, not whether a real model speaks naturally.
"""

from copy import deepcopy

import pytest
from test_default_cross_turn_memory import (
    CORRECT,
    FORGET,
    ORIGINAL,
    QUERY,
    Scenario,
    build,
    turn,
)
from test_default_text_delivery import Model, setup
from test_short_dialogue_prompt_versions import control_for_revision
from virtual_companion.boundary.memory_relevance import MAX_RECALL_QUERY_CHARS
from virtual_companion.cognition.short_dialogue import PROMPTS, ShortRequest
from virtual_companion.text import default_runtime

CANDIDATE = "short-dialogue.v13"
COMPARISON = "short-dialogue.v12"


class PreparedRecordingModel(Model):
    """Freeze and record the same messages during preflight and generation."""

    supports_stage_metadata = True

    def __init__(self):
        super().__init__()
        self.prepared = []

    def prepare_stage_request(self, messages, *, stage, schema):
        owner = self
        record = dict(stage=stage, schema=deepcopy(schema), messages=deepcopy(messages),
                      checked=None, sent=None)
        self.prepared.append(record)

        class BoundRequest:
            def check_context(self):
                record["checked"] = deepcopy(record["messages"])

            def stream(self, cancelled):
                assert record["checked"] == record["messages"]
                record["sent"] = deepcopy(record["messages"])
                yield from owner.stream(record["messages"], cancelled)

        return BoundRequest()


@pytest.mark.parametrize("user_text", [
    "今天雨好大，我忽然想起小时候放学的路。",
    "我今天有点累，先让我说说，不用急着给建议。",
    "最近总觉得房间很安静，想和你聊聊这种感觉。",
    "给我两条下雨天通勤的实用建议。",
    "我想去散步但在下雨，请给我一个简单安排。",
    "我在想，为什么旧照片会让人同时开心又难过？",
    "先听我说，不给建议，也别提问。",
    "详细解释风扇为什么能让人觉得凉，列出三种因素，不要给购买建议。",
    "不是我搬家，是我朋友搬家。只帮她列两件今晚能做的事，别建议买东西。",
], ids=["share", "listen", "discuss", "two-suggestions", "one-arrangement",
        "two-sided-explanation", "no-advice-or-question", "requested-detail", "subject-and-scope"])
def test_candidate_reaches_the_same_checked_and_sent_default_request(
    tmp_path, monkeypatch, user_text,
):
    control = control_for_revision(CANDIDATE)
    previous = control_for_revision(COMPARISON)
    monkeypatch.setattr(default_runtime, "CONTROL", control)
    captured = []
    model_request = ShortRequest.model_request

    def capture(request, output_tokens, system=None):
        body = model_request(request, output_tokens, system)
        if system is not None and system.startswith(control):
            comparison = model_request(request, output_tokens, previous + system[len(control):])
            captured.append((request, body, comparison))
        return body

    monkeypatch.setattr(ShortRequest, "model_request", capture)
    model = PreparedRecordingModel()
    runtime, session, memory, mirror, _ = setup(tmp_path, model)
    try:
        event = turn(runtime, session, user_text)
        assert len(captured) == 1
        assert [record["stage"] for record in model.prepared] == [
            "demand_control", "stream_unit_review"]
        assert len(model.calls) == 2
        request, body, comparison = captured[0]
        generation, review = model.prepared
        assert generation["checked"] == generation["sent"] == body["messages"] == model.calls[0]
        assert body["messages"][0]["content"].startswith(control)
        assert body["messages"][1:] == comparison["messages"][1:]
        assert body["messages"][-1] == {"role": "user", "content": user_text}
        assert body["messages"][0]["content"][len(control):] == (
            comparison["messages"][0]["content"][len(previous):]
        )
        assert body["tools"] == comparison["tools"] == []
        assert body["max_output_tokens"] == comparison["max_output_tokens"] == 192
        branches = generation["schema"]["oneOf"]
        branches_by_action = {
            branch["properties"]["action"]["const"]: branch for branch in branches
        }
        assert set(branches_by_action) == {"answer", "ack", "listen", "memory"}
        assert branches_by_action["ack"]["required"] == ["action", "text"]
        assert branches_by_action["ack"]["properties"]["text"] == {
            "type": "string", "minLength": 1,
        }
        memory_branch = branches_by_action["memory"]
        assert memory_branch["properties"] == {
            "action": {"const": "memory"},
            "query": {
                "type": "string", "minLength": 1, "maxLength": MAX_RECALL_QUERY_CHARS,
            },
        }
        assert memory_branch["required"] == ["action"]
        assert memory_branch["additionalProperties"] is False
        assert review["checked"] == review["sent"] == model.calls[1]
        assert request.context.input_id == event["basis"]["input_id"]
        assert event["readset"]["facts"] == event["readset"]["history"] == []
        # Prompt demonstrations cannot become user history or source evidence.
        examples = PROMPTS[CANDIDATE].split("\n接话示意", 1)[1].split("\n事实与范围：", 1)[0]
        assert examples not in body["messages"][1]["content"]
        assert mirror.history()["messages"][0]["content"] == user_text
    finally:
        memory.close()
        mirror.close()


@pytest.mark.parametrize("management", [CORRECT, FORGET], ids=["correct", "forget"])
def test_candidate_keeps_operation_evidence_and_invalidated_history_out_of_recall(
    tmp_path, monkeypatch, management,
):
    control = control_for_revision(CANDIDATE)
    previous = control_for_revision(COMPARISON)
    monkeypatch.setattr(default_runtime, "CONTROL", control)
    captured = []
    model_request = ShortRequest.model_request

    def capture(request, output_tokens, system=None):
        body = model_request(request, output_tokens, system)
        if system is not None and system.startswith(control):
            baseline = model_request(request, output_tokens, previous + system[len(control):])
            captured.append((request.user_text, body, baseline))
        return body

    monkeypatch.setattr(ShortRequest, "model_request", capture)
    model = Scenario()
    memory, mirror, runtime, session = build(tmp_path, model)
    try:
        saved = turn(runtime, session, ORIGINAL)
        before = turn(runtime, session, QUERY)
        changed = turn(runtime, session, management)
        after = turn(runtime, session, QUERY)
        assert before["display_text"] == model.recalled
        assert changed["readset"]["current_operation"]["status"] == "committed"
        assert saved["basis"]["input_id"] in memory.store.ineffective_history_sources(
            memory.companion, now_us=model.clock())
        assert after["display_text"] == (
            model.updated_reply if management == CORRECT else model.missing)
        assert ORIGINAL not in model.primary[-1][1]["content"]
        assert len(after["readset"]["facts"]) == (1 if management == CORRECT else 0)
        for user_text, body, baseline in captured:
            assert body["messages"][0]["content"].startswith(control)
            assert body["messages"][1:] == baseline["messages"][1:]
            assert body["messages"][-1]["content"] == user_text
            assert body["messages"][0]["content"][len(control):] == (
                baseline["messages"][0]["content"][len(previous):]
            )
        boundary = "\n事实与范围：\n"
        assert PROMPTS[CANDIDATE].split(boundary, 1)[1] == (
            PROMPTS[COMPARISON].split(boundary, 1)[1])
    finally:
        memory.close()
        mirror.close()
