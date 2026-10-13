"""Recorded P1-03 failures and test-only v14 on the production host/ACK path.

Prompt-conditioned outputs and reviewer opinions are synthetic recordings. They
prove request/approval/delivery behavior, not new real-model naturalness results.
The rain before-answer and both opinions are verbatim s-01 model_output evidence.
Other before-answers keep the recorded output, with tight synthetic rejection
opinions where the real reviewer approved content that failed the intent criteria.
"""

import json
import threading
from copy import deepcopy
from hashlib import sha256

import pytest
from test_default_text_delivery import Model, wait_for
from unit_review_recording import bound_unit_packets as bound_unit_packets
from unit_review_recording import recorded_review_packet
from virtual_companion.cognition.short_dialogue import PROMPTS
from virtual_companion.cognition.stream_unit_review import SYSTEM as REVIEW_SYSTEM
from virtual_companion.text import default_runtime
from virtual_companion.text.application import TextSession
from virtual_companion.text.fake import synthetic_unit_opinion
from virtual_companion.text.memory_runtime import DEFAULT_COMPANION_ID, TextMemoryRuntime
from virtual_companion.text.sqlite_store import SQLiteConversationStore

CANDIDATE = "short-dialogue.v14"
RAIN = "今天雨好大，我忽然想起小时候放学的路。"
RAIN_PREFIX = "雨一大，就容易把放学的路带出来。"
RAIN_BEFORE = RAIN_PREFIX + "那段路你记得最清楚的是哪一段？"
QUIET = "最近总觉得房间很安静，想和你聊聊这种感觉。"
QUIET_BEFORE = (
    "房间安静下来的时候，那种空的感觉会变得很明显，好像连自己的动静都被放大了。"
    "你是最近才注意到这种安静，还是之前就有、只是现在更在意了？"
)
WALK = "我想去散步但在下雨，请给我一个简单安排。"
WALK_BEFORE = (
    "带把伞，穿防滑的鞋，选附近有遮挡或能随时折返的路线走二三十分钟；"
    "雨大或打雷就在室内走几圈。"
)
# This is the exact parsed s-01 JSON, not a newly permissive reviewer policy.
RAIN_APPROVE = dict(consistent_with_displayed=True, constraints_satisfied=True,
                   decision="approve", no_open_dependency=True, self_contained=True)
RAIN_REJECT = {**RAIN_APPROVE, "constraints_satisfied": False, "decision": "reject"}
CASES = (
    ("rain-share", RAIN, RAIN_BEFORE, RAIN_PREFIX),
    ("quiet-explore", QUIET, QUIET_BEFORE, "你在房间里做什么的时候，最注意到这种安静？"),
    ("simple-walk", WALK, WALK_BEFORE,
     "等雨小且没有雷电、积水时，带伞沿好走的路走十分钟再折返；天气不安全就等一等。"),
    ("listen", "我今天有点累，先让我说说，不用急着给建议。", None, "你说，我听着。"),
    ("two-suggestions", "给我两条下雨天通勤的实用建议。", None,
     "1. 留出些余量，路上避开积水，经过湿滑处放慢。\n"
     "2. 用能封口的袋子装好怕湿的东西，取用前先擦干手。"),
    ("general-explanation", "我在想，为什么旧照片会让人同时开心又难过？", None,
     "开心可以来自对美好时刻的回想；难过可以来自意识到那个时刻已经过去。"
     "同一张照片既让记忆变得鲜明，也显出时间的距离，所以两种感受能够并存。"),
    ("new-share", "刚整理完桌上的一堆旧票根。", None,
     "票根像给过去的小事留了个书签。"),
    ("new-explore", "刚整理完桌上的一堆旧票根，想聊聊看到它们的感觉。", None,
     "你最先留意的是哪一张？"),
)


def intent_contract(system):
    """Check operative general guidance on the sent request, not an answer template."""
    for rule in ("分享不默认变成追问", "话题还能继续不是提问理由", "用户明确想聊",
                 "让用户自己描述", "不预设空落、孤独", "保持用户指定的活动",
                 "天气不安全就等待", "解释一般现象时用泛指", "相关背景不必说出来"):
        assert rule in system, rule


class IntentRecordingModel(Model):
    def __init__(self, case):
        super().__init__()
        self.name, self.user_text, self.before, self.after = case
        self.release, self.terminal = threading.Event(), threading.Event()
        self.reviews = []

    def review_peer(self):
        owner = self

        class Reviewer(Model):
            def stream(self, messages, cancelled):
                owner.calls.append(deepcopy(messages))
                assert messages[0]["content"].startswith(REVIEW_SYSTEM)
                packet = recorded_review_packet(messages)
                assert packet["user_text"] == owner.user_text
                text = packet["candidate_prefix"]
                supported = text == owner.after or (
                    owner.name == "rain-share" and text == RAIN_PREFIX)
                if not packet["final"] and owner.after.startswith(text) and not supported:
                    # A partial multi-part answer is still pending, not a rejected
                    # completed answer. Keep the original final requirements.
                    opinion = dict(decision="hold", self_contained=False,
                        no_open_dependency=False, consistent_with_displayed=True,
                        constraints_satisfied=False)
                else:
                    opinion = deepcopy(RAIN_APPROVE if supported else RAIN_REJECT)
                # Retain the recorded five fields separately from the v3 wire.
                wire_opinion = synthetic_unit_opinion(packet, **opinion)
                owner.reviews.append(dict(
                    packet=packet, opinion=opinion, wire_opinion=wire_opinion))
                yield json.dumps(wire_opinion)

        return Reviewer()

    def stream(self, messages, cancelled):
        self.calls.append(deepcopy(messages))
        assert messages[-1] == {"role": "user", "content": self.user_text}
        is_candidate = "回应方式（short-dialogue.v14）" in messages[0]["content"]
        if is_candidate:
            intent_contract(messages[0]["content"])
        reply = self.after if is_candidate else self.before
        assert reply is not None
        try:
            if self.name == "rain-share":
                yield '{"action":"answer","text":"' + RAIN_PREFIX
                while not self.release.wait(0.01):
                    if cancelled.is_set():
                        raise InterruptedError("synthetic rain generation cancelled")
                yield reply[len(RAIN_PREFIX):] + '"}'
            else:
                yield json.dumps({"action": "answer", "text": reply}, ensure_ascii=False)
        finally:
            self.terminal.set()

    def cancel(self):
        self.release.set()


def run_recording(tmp_path, case, revision):
    model = IntentRecordingModel(case)
    memory = TextMemoryRuntime(tmp_path / "memory.db", DEFAULT_COMPANION_ID,
                               model=model, fake=True)
    mirror = SQLiteConversationStore(tmp_path / "chat", companion_id=DEFAULT_COMPANION_ID)
    runtime = default_runtime.DefaultTextRuntime(
        memory, mirror, model, binding={"test": "synthetic-intent"},
        delivery_timeout=0.8, prompt_revision=revision)
    session = TextSession(model, mirror, {}, prepare_turn=runtime.prepare_turn,
                          before_start=runtime.before_start)
    events = []
    try:
        session.send(model.user_text)

        def next_event_or_end():
            event = runtime.poll()["event"]
            return event if event else (True if not session.snapshot()["busy"] else None)

        while True:
            event = wait_for(next_event_or_end)
            if event is True:
                break
            assert session.snapshot()["partial"] == ""
            if model.name == "rain-share":
                assert not model.terminal.is_set()
                assert runtime.snapshot()["generation_terminal_us"] is None
            events.append(deepcopy(event))
            runtime.ack(dict(sequence=event["sequence"], fingerprint=event["fingerprint"],
                             host_receipt=f"synthetic-intent-{revision}-{event['sequence']}"))
            wait_for(lambda: runtime.poll()["event"] is None)
            model.release.set()
        state = memory.store.stream_state(memory.companion, runtime.claim.input_id)
        return dict(calls=model.calls, reviews=model.reviews, events=events,
                    state=state, attempt=session.snapshot()["attempt"],
                    history=mirror.history()["messages"],
                    prompt_revision=runtime.snapshot()["request_prompt_revision"])
    finally:
        model.release.set()
        memory.close()
        mirror.close()


def test_recorded_rain_final_rejection_preserves_exact_review_and_confirmed_prefix(
    tmp_path, record_property,
):
    result = run_recording(tmp_path, CASES[0], "short-dialogue.v13")
    assert [r["opinion"] for r in result["reviews"]] == [RAIN_APPROVE, RAIN_REJECT]
    early, final = (r["packet"] for r in result["reviews"])
    assert early["candidate_prefix"] == RAIN_PREFIX and not early["final"]
    assert final["candidate_prefix"] == RAIN_BEFORE and final["final"]
    assert final["displayed_prefix"] == RAIN_PREFIX
    assert early["context"] == final["context"] and early["basis"] == final["basis"]
    assert result["state"]["outcome"] == result["attempt"]["status"] == "failed"
    assert result["state"]["confirmed_text"] == result["history"][-1]["content"] == RAIN_PREFIX
    assert result["state"]["approved_units"] == result["state"]["confirmed_units"] == 1
    assert len(result["calls"]) == 3  # generation, early opinion, final rejection; no rewrite
    record_property("replay", json.dumps(result, ensure_ascii=False))


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_v14_intent_candidate_uses_same_generation_review_display_and_ack(
    tmp_path, monkeypatch, record_property, case,
):
    # Select only this CPU diagnostic. Product default and startup allowlist stay frozen.
    monkeypatch.setattr(default_runtime, "DIAGNOSTIC_PROMPT_REVISION", CANDIDATE)
    selected = default_runtime.control_for_prompt_revision(CANDIDATE)
    assert default_runtime.DEFAULT_PROMPT_REVISION == "short-dialogue.v12"
    result = run_recording(tmp_path, case, CANDIDATE)
    assert result["prompt_revision"] == CANDIDATE
    assert result["calls"][0][0]["content"].startswith(selected)
    intent_contract(result["calls"][0][0]["content"])
    assert result["state"]["outcome"] == result["attempt"]["status"] == "completed"
    assert result["state"]["confirmed_text"] == result["history"][-1]["content"] == case[3]
    assert result["state"]["confirmed_units"] == result["state"]["approved_units"] == 1
    approved = [r for r in result["reviews"] if r["opinion"]["decision"] != "hold"]
    assert [r["opinion"] for r in approved] == [RAIN_APPROVE]
    assert sum(not r["packet"]["final"] for r in result["reviews"]) <= 1
    assert sum(r["packet"]["final"] for r in result["reviews"]) <= 1
    assert len(result["calls"]) == 1 + len(result["reviews"])
    assert sum(not m[0]["content"].startswith(REVIEW_SYSTEM) for m in result["calls"]) == 1
    event = result["events"][0]
    assert event["display_text"] == case[3] and event["sequence"] == 1
    assert event["readset"]["facts"] == event["readset"]["history"] == []
    packet = approved[0]["packet"]
    assert packet["candidate_prefix"] == case[3] and packet["basis"] == {
        key: event["basis"][key] for key in packet["basis"]}
    record_property("replay", json.dumps(result, ensure_ascii=False))


def test_v14_keeps_frozen_prompts_fact_scope_and_production_selection():
    boundary = "\n事实与范围：\n"
    assert PROMPTS[CANDIDATE].split(boundary, 1)[1] == (
        PROMPTS["short-dialogue.v13"].split(boundary, 1)[1])
    assert sha256(PROMPTS["short-dialogue.v13"].encode()).hexdigest() == (
        "6359b67e8ded09d87071607d0e0636d2210e632ef8f6f24eb9385927144cf228")
    assert default_runtime.DEFAULT_PROMPT_REVISION == "short-dialogue.v12"
    assert default_runtime.DIAGNOSTIC_PROMPT_REVISION == "short-dialogue.v13"
    assert CANDIDATE in default_runtime.DIAGNOSTIC_PROMPT_REVISIONS
    assert default_runtime.control_for_prompt_revision(CANDIDATE).endswith(
        default_runtime.DESKTOP_PROTOCOL
    )
    with pytest.raises(ValueError, match="Unsupported desktop prompt revision"):
        default_runtime.control_for_prompt_revision("short-dialogue.v999")
