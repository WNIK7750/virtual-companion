"""Prompt revisions are explicit, versioned and never switch silently."""

import json
from hashlib import sha256
from threading import Event

import pytest
from test_default_cross_turn_memory import QUERY, Scenario, build, turn
from test_default_text_delivery import setup
from virtual_companion.cognition.short_dialogue import (
    POLICY,
    PROMPTS,
    SYSTEM,
    SYSTEM_V4,
    ShortContext,
    ShortDialogue,
    ShortRequest,
    prose_evidence,
)
from virtual_companion.text import default_runtime
from virtual_companion.text.application import TextSession
from virtual_companion.text.default_runtime import CONTROL, DEFAULT_PROMPT_REVISION


class Model:
    def __init__(self, text="甲句。"):
        self.text = text
        self.body = None

    def count(self, body):
        self.body = body
        return 10

    def invoke(self, body):
        return {"text": self.text, "tool_calls": [], "finish_reason": "stop"}


def control_for_revision(revision):
    """Select a candidate prefix while keeping the actual desktop protocol tail."""
    selected = PROMPTS[DEFAULT_PROMPT_REVISION].replace(
        "只输出回复正文", "回复正文放在text字段"
    ).replace("不声称已执行或保存。", "操作反馈须符合本次实际证据。")
    candidate = PROMPTS[revision].replace(
        "只输出回复正文", "回复正文放在text字段"
    ).replace("不声称已执行或保存。", "操作反馈须符合本次实际证据。")
    assert CONTROL.startswith(selected)
    return candidate + CONTROL[len(selected):]


def test_default_prompt_is_unchanged_until_a_revision_is_adopted():
    model = Model()
    request = ShortRequest("你好")
    result = ShortDialogue(model, cancelled=Event()).generate(request)
    assert result.policy == POLICY == "short-dialogue.v3"
    system = model.body["messages"][0]["content"]
    assert system.startswith(SYSTEM.replace("不声称已执行或保存。", "操作反馈须符合本次实际证据。"))
    assert system.endswith(request.operation_constraints())


def test_v12_is_the_selected_desktop_candidate():
    assert DEFAULT_PROMPT_REVISION == "short-dialogue.v12"


@pytest.mark.parametrize("revision", [
    "short-dialogue.v3", "short-dialogue.v4", "short-dialogue.v5", "short-dialogue.v6",
    "short-dialogue.v8", "short-dialogue.v9", "short-dialogue.v10", "short-dialogue.v11",
    "short-dialogue.v12", "short-dialogue.v13", "short-dialogue.v14", "short-dialogue.v15",
])
def test_revision_is_selectable_and_versioned(revision):
    model = Model()
    dialogue = ShortDialogue(model, cancelled=Event(), policy=revision)
    request = ShortRequest("你好")
    result = dialogue.generate(request)
    assert result.policy == revision
    system = model.body["messages"][0]["content"]
    selected = PROMPTS[revision].replace("不声称已执行或保存。", "操作反馈须符合本次实际证据。")
    assert system.startswith(selected)
    assert system.endswith(request.operation_constraints())


def test_actual_default_request_uses_its_selected_prompt_and_scoped_empty_memory(tmp_path):
    model = Scenario()
    memory, mirror, runtime, session = build(tmp_path, model)
    try:
        event = turn(runtime, session, QUERY)
        assert len(model.primary) == 2  # Initial control, then the existing memory refresh.
        selected = PROMPTS[DEFAULT_PROMPT_REVISION].replace(
            "只输出回复正文", "回复正文放在text字段"
        ).replace("不声称已执行或保存。", "操作反馈须符合本次实际证据。")
        for messages in model.primary:
            assert messages[0]["content"].startswith(selected)
            assert messages[-1] == {"role": "user", "content": QUERY}
        related_memory = next(line for line in model.primary[-1][1]["content"].splitlines()
                              if line.startswith("相关长期记忆"))
        assert "本次" in related_memory
        assert event["readset"]["facts"] == []
    finally:
        memory.close()
        mirror.close()


@pytest.mark.parametrize("user_text", [
    "最近总觉得房间很安静，想和你聊聊这种感觉。",
    "我想去散步但在下雨，请给我一个简单安排。",
    "请详细解释为什么备份对回滚重要，给两个例子。",
], ids=["share", "one-arrangement", "requested-detail"])
def test_v9_remains_selectable_without_extra_stages_or_context_changes(
    tmp_path, monkeypatch, user_text,
):
    control = control_for_revision("short-dialogue.v9")
    monkeypatch.setattr(default_runtime, "CONTROL", control)
    prepared = []
    model_request = ShortRequest.model_request

    def record_request(request, output_tokens, system=None):
        body = model_request(request, output_tokens, system)
        if system is not None and system.startswith(control):
            prepared.append((request, output_tokens, system, body))
        return body

    monkeypatch.setattr(ShortRequest, "model_request", record_request)
    runtime, session, memory, mirror, model = setup(tmp_path)
    try:
        event = turn(runtime, session, user_text)
        assert len(prepared) == 1 and len(model.calls) == 2
        request, output_tokens, system, body = prepared[0]
        assert body["messages"] == model.calls[0]
        sent_system = body["messages"][0]["content"]
        assert sent_system.startswith(control)
        assert "回应方式（short-dialogue.v9）" in sent_system
        for rule in ("短承接也可以是完整回应", "相关联想", "不汇报自己将如何回应",
                     "不默认总结、反问或追加帮助邀请", "用户明确要求详细",
                     "联想不能补成用户的具体经历"):
            assert rule in sent_system, rule
        assert "\n表达示意：\n" not in sent_system
        previous = control_for_revision("short-dialogue.v6")
        baseline = model_request(request, output_tokens, previous + system[len(control):])
        assert body["messages"][1:] == baseline["messages"][1:]
        assert body["tools"] == baseline["tools"] == []
        assert body["max_output_tokens"] == baseline["max_output_tokens"] == 192
        assert sent_system[len(control):] == baseline["messages"][0]["content"][len(previous):]
        assert sent_system.endswith(request.operation_constraints())
        assert body["messages"][-1] == {"role": "user", "content": user_text}
        assert request.context.input_id == event["basis"]["input_id"]
        assert event["readset"]["facts"] == event["readset"]["history"] == []
    finally:
        memory.close()
        mirror.close()


@pytest.mark.parametrize("user_text", [
    "今天翻出了以前写的一个本子。",
    "为什么金属勺子摸起来比木勺子凉？",
    "今晚想整理桌面，给我一个具体建议。",
    "请比较两种备份方式，详细说明各自用途，给两个例子。",
], ids=["share", "explain", "one-suggestion", "requested-comparison-detail"])
def test_v10_remains_selectable_with_grounded_response_rules(
    tmp_path, monkeypatch, user_text,
):
    control = control_for_revision("short-dialogue.v10")
    monkeypatch.setattr(default_runtime, "CONTROL", control)
    prepared = []
    model_request = ShortRequest.model_request

    def record_request(request, output_tokens, system=None):
        body = model_request(request, output_tokens, system)
        if system is not None and system.startswith(control):
            prepared.append((request, output_tokens, system, body))
        return body

    monkeypatch.setattr(ShortRequest, "model_request", record_request)
    runtime, session, memory, mirror, model = setup(tmp_path)
    try:
        event = turn(runtime, session, user_text)
        assert len(prepared) == 1 and len(model.calls) == 2
        request, output_tokens, system, body = prepared[0]
        assert body["messages"] == model.calls[0]
        sent_system = body["messages"][0]["content"]
        # The v9 failure is this missing distinction, before the revision check.
        assert "未说的经历、感受和评价保持未知" in sent_system
        assert sent_system.startswith(control)
        assert "回应方式（short-dialogue.v10）" in sent_system
        for rule in ("接已给出的内容", "不靠情绪选项替用户解释自己",
                     "先给主要原因、区别或判断", "题干改述、同义铺陈",
                     "只要一个建议就先选定一条", "不堆替代清单",
                     "用户要求比较或多个方案时再按要求展开", "不为简短漏答",
                     "不套固定句式或句数", "内容说完可以停",
                     "不例行附和、总结、追问或帮助邀请"):
            assert rule in sent_system, rule
        assert "\n表达示意：\n" not in sent_system
        boundary = "\n事实与范围：\n"
        assert PROMPTS["short-dialogue.v10"].split(boundary, 1)[1] == (
            PROMPTS["short-dialogue.v9"].split(boundary, 1)[1]
        )
        previous = control_for_revision("short-dialogue.v9")
        baseline = model_request(request, output_tokens, previous + system[len(control):])
        assert body["messages"][1:] == baseline["messages"][1:]
        assert body["tools"] == baseline["tools"] == []
        assert body["max_output_tokens"] == baseline["max_output_tokens"] == 192
        assert sent_system[len(control):] == baseline["messages"][0]["content"][len(previous):]
        assert sent_system.endswith(request.operation_constraints())
        assert body["messages"][-1] == {"role": "user", "content": user_text}
        assert request.context.input_id == event["basis"]["input_id"]
        assert event["readset"]["facts"] == event["readset"]["history"] == []
    finally:
        memory.close()
        mirror.close()


@pytest.mark.parametrize("user_text,required_rule", [
    ("今天翻出了以前写的一个本子。", "“可能”和问句也不能充当来源"),
    ("新换的办公室感觉空旷，想聊聊这个。", "不让用户在你猜的情绪里选"),
    ("为什么门锁更牢靠，却出入更麻烦？", "问题有几面，就分别讲清各自缘由"),
    ("想把这一盒书搬到楼上，给我一个简单办法。", "能实现原目标的做法"),
    ("给我两条整理书桌的建议。", "用户限定的数量、时间、范围和禁止事项"),
    ("先让我说完，别给建议，也不用提问。", "用户只想说时不给建议"),
    ("请比较两种备份方式，详细说明各自用途，给两个例子。", "不为简短漏答"),
], ids=["share-source", "open-discussion", "both-causes", "one-original-goal",
        "two-suggestions", "listen-without-advice", "requested-detail"])
def test_v11_remains_selectable_with_compact_intent_guidance_without_changing_evidence(
    tmp_path, monkeypatch, user_text, required_rule,
):
    control = control_for_revision("short-dialogue.v11")
    monkeypatch.setattr(default_runtime, "CONTROL", control)
    prepared = []
    model_request = ShortRequest.model_request

    def record_request(request, output_tokens, system=None):
        body = model_request(request, output_tokens, system)
        if system is not None and system.startswith(control):
            prepared.append((request, output_tokens, system, body))
        return body

    monkeypatch.setattr(ShortRequest, "model_request", record_request)
    runtime, session, memory, mirror, model = setup(tmp_path)
    try:
        event = turn(runtime, session, user_text)
        assert len(prepared) == 1 and len(model.calls) == 2
        request, output_tokens, system, body = prepared[0]
        assert body["messages"] == model.calls[0]
        sent = body["messages"][0]["content"]
        assert required_rule in sent, required_rule
        assert sent.startswith(control)
        previous = control_for_revision("short-dialogue.v10")
        guidance, boundary, tail = PROMPTS["short-dialogue.v11"].partition("\n事实与范围：\n")
        prior_guidance, _, prior_tail = PROMPTS["short-dialogue.v10"].partition(boundary)
        assert boundary and tail == prior_tail
        assert len(guidance) < len(prior_guidance), (
            "rewrite the active guidance, do not stack rules"
        )
        assert "不换目标、不列替代活动" in guidance
        assert "不例行反问、总结或邀聊" in guidance
        assert "用户想聊时沿原话展开或提问" in guidance
        _, marker, examples = guidance.partition("\n对比示意")
        assert marker and "不是用户事实" in examples
        assert examples.count("\n用户：") == 1
        assert examples.count("\n不合适：") == examples.count("\n合适：") == 1
        baseline = model_request(request, output_tokens, previous + system[len(control):])
        assert body["messages"][1:] == baseline["messages"][1:]
        assert examples not in body["messages"][1]["content"]
        assert body["tools"] == baseline["tools"] == []
        assert body["max_output_tokens"] == baseline["max_output_tokens"] == 192
        assert sent[len(control):] == baseline["messages"][0]["content"][len(previous):]
        assert sent.endswith(request.operation_constraints())
        assert body["messages"][-1] == {"role": "user", "content": user_text}
        assert request.context.input_id == event["basis"]["input_id"]
        assert event["readset"]["facts"] == event["readset"]["history"] == []
    finally:
        memory.close()
        mirror.close()


@pytest.mark.parametrize("user_text,required_rule", [
    ("今天雨好大，我忽然想起小时候放学的路。", "只用用户给出的锚点"),
    ("最近总觉得房间很安静，想和你聊聊这种感觉。", "不诊断情绪"),
    ("我想去散步但在下雨，请给我一个简单安排。", "指定对象、数量和形式"),
    ("先听我说说，不用给建议，也不用提问。", "只有下一问能推动当下内容时才问"),
    ("请详细解释为什么备份对回滚重要，给两个例子。", "不为简短漏答"),
], ids=["anchored-memory", "open-discussion", "requested-walk", "listen", "requested-detail"])
def test_v12_candidate_control_binds_anchored_and_direct_response_rules(
    tmp_path, monkeypatch, user_text, required_rule,
):
    control = control_for_revision("short-dialogue.v12")
    monkeypatch.setattr(default_runtime, "CONTROL", control)
    prepared = []
    model_request = ShortRequest.model_request

    def record_request(request, output_tokens, system=None):
        body = model_request(request, output_tokens, system)
        if system is not None and system.startswith(control):
            prepared.append((request, output_tokens, system, body))
        return body

    monkeypatch.setattr(ShortRequest, "model_request", record_request)
    runtime, session, memory, mirror, model = setup(tmp_path)
    try:
        event = turn(runtime, session, user_text)
        assert len(prepared) == 1 and len(model.calls) == 2
        request, output_tokens, system, body = prepared[0]
        assert body["messages"] == model.calls[0]
        sent = body["messages"][0]["content"]
        assert sent.startswith(control)
        assert "回应方式（short-dialogue.v12）" in sent
        assert required_rule in sent, required_rule
        guidance, boundary, tail = PROMPTS["short-dialogue.v12"].partition("\n事实与范围：\n")
        assert boundary and tail == PROMPTS["short-dialogue.v11"].partition(boundary)[2]
        for rule in ("围绕当前一条主线", "不编既往细节、习惯、情绪或因果",
                     "不等于有放学铃声", "不等于孤独或需要陪伴",
                     "先给可直接执行的结果", "指定对象、数量和形式",
                     "散步安排、天气安全边界和需携带物", "不改成室内活动",
                     "只有安全或可行性的关键条件不明时", "先接已说的具体内容",
                     "不泛泛附和、不诊断情绪", "不例行反问或邀聊",
                     "不汇报处理流程", "不长篇扩展、不强制总结"):
            assert rule in guidance, rule
        _, marker, examples = guidance.partition("\n安排示意")
        assert marker and "不是用户事实" in examples
        assert examples.count("\n用户：") == 1
        assert examples.count("\n合适：") == examples.count("\n不合适：") == 1
        for detail in ("雨小且无雷电、积水时", "带伞", "十分钟再折返",
                       "否则等天气安全再出门", "不合适：改在家快走或慢跑"):
            assert detail in examples, detail
        previous = control_for_revision("short-dialogue.v11")
        baseline = model_request(request, output_tokens, previous + system[len(control):])
        assert body["messages"][1:] == baseline["messages"][1:]
        assert examples not in body["messages"][1]["content"]
        assert body["tools"] == baseline["tools"] == []
        assert body["max_output_tokens"] == baseline["max_output_tokens"] == 192
        assert sent[len(control):] == baseline["messages"][0]["content"][len(previous):]
        assert sent.endswith(request.operation_constraints())
        assert body["messages"][-1] == {"role": "user", "content": user_text}
        assert request.context.input_id == event["basis"]["input_id"]
        assert event["readset"]["facts"] == event["readset"]["history"] == []
    finally:
        memory.close()
        mirror.close()


def test_v5_explanation_rule_is_preserved_for_comparison():
    request = ShortRequest("为什么熟悉的地方有时会让人感触很多？")
    body = request.model_request(192, control_for_revision("short-dialogue.v5"))
    system = body["messages"][0]["content"]
    for rule in ("简单解释或闲聊", "同一意思换措辞重复", "百科背景", "泛泛可能性",
                 "用户要求细节、例子或多种原因时充分展开"):
        assert rule in system, rule
    assert body["messages"][-1] == {"role": "user", "content": request.user_text}
    assert system.endswith(request.operation_constraints())


@pytest.mark.parametrize("revision,expected_hash", [
    ("short-dialogue.v12", "485c8a2e3cddfa55d88bd9e107b6f19be9d1b39ba9a5ae4589d9502d797bd874"),
    ("short-dialogue.v13", "6359b67e8ded09d87071607d0e0636d2210e632ef8f6f24eb9385927144cf228"),
    ("short-dialogue.v14", "1912887ca1d162b627b8fba459b908a8e4aa2578a0094ab7399acc0d52fcbeb8"),
])
def test_frozen_v12_v13_v14_prompt_bytes(revision, expected_hash):
    assert sha256(PROMPTS[revision].encode()).hexdigest() == expected_hash


def test_v15_rewrites_expression_and_preserves_fact_task_and_operation_rules():
    guidance, boundary, tail = PROMPTS["short-dialogue.v15"].partition("\n事实与范围：\n")
    prior_guidance, _, prior_tail = PROMPTS["short-dialogue.v14"].partition(boundary)
    assert boundary and tail == prior_tail
    assert len(guidance) < len(prior_guidance), "rewrite guidance rather than stack rules"
    assert "回应方式（short-dialogue.v15）" in guidance
    for rule in (
        "围绕当前主线回应", "贴切的观察、联想或不同看法", "不只换词复述、泛泛赞同",
        "未说的经历、感受、习惯、感官细节和原因仍未知", "“可能”和问句也不是依据",
        "用户想聊不等于必须追问", "先一起谈已经给出的内容", "当前关键处时再问",
        "话题可以暂时悬着", "不在每次末尾追加总结、反问、邀聊或帮助邀请",
        "不汇报处理过程", "不教用户怎样表达", "满足指定对象、数量、时间和形式",
        "简单安排保留原活动", "未知环境和物品用条件表述", "涉及两面时讲清怎样并存",
        "一般原因用泛指", "只在关键处或用户要求细节、例子、比较时深化",
        "完整回答优先", "不套固定句数、口头禅或文艺铺陈",
    ):
        assert rule in guidance, rule
    assert DEFAULT_PROMPT_REVISION == "short-dialogue.v12"


def test_v15_candidate_control_reaches_existing_default_request_without_extra_calls(
    tmp_path, monkeypatch,
):
    revision = "short-dialogue.v15"
    monkeypatch.setattr(default_runtime, "DIAGNOSTIC_PROMPT_REVISIONS",
                        (*default_runtime.DIAGNOSTIC_PROMPT_REVISIONS, revision))
    control = default_runtime.control_for_prompt_revision(revision)
    prepared = []
    model_request = ShortRequest.model_request

    def record_request(request, output_tokens, system=None):
        body = model_request(request, output_tokens, system)
        if system is not None and system.startswith(control):
            prepared.append((request, output_tokens, system, body))
        return body

    monkeypatch.setattr(ShortRequest, "model_request", record_request)
    default, _, memory, mirror, model = setup(tmp_path)
    candidate = default_runtime.DefaultTextRuntime(
        memory, mirror, model, binding={"test": "synthetic"},
        delivery_timeout=0.8, prompt_revision=revision,
    )
    session = TextSession(model, mirror, {}, prepare_turn=candidate.prepare_turn,
                          before_start=candidate.before_start)
    user_text = "刚整理完桌上的一堆旧票根，想聊聊看到它们的感觉。"
    try:
        event = turn(candidate, session, user_text)
        assert len(prepared) == 1 and len(model.calls) == 2
        request, output_tokens, system, body = prepared[0]
        prior_control = default_runtime.control_for_prompt_revision("short-dialogue.v14")
        baseline = model_request(request, output_tokens, prior_control + system[len(control):])
        assert body["messages"] == model.calls[0]
        assert body["messages"][1:] == baseline["messages"][1:]
        assert body["tools"] == baseline["tools"] == []
        assert body["max_output_tokens"] == baseline["max_output_tokens"] == 192
        sent = body["messages"][0]["content"]
        assert sent.startswith(control) and control.endswith(default_runtime.DESKTOP_PROTOCOL)
        assert sent[len(control):] == baseline["messages"][0]["content"][len(prior_control):]
        assert sent.endswith(request.operation_constraints())
        assert body["messages"][-1] == {"role": "user", "content": user_text}
        assert request.context.input_id == event["basis"]["input_id"]
        assert event["readset"]["facts"] == event["readset"]["history"] == []
        assert candidate.prompt_revision == revision
        assert default.control == default_runtime.CONTROL == (
            control_for_revision("short-dialogue.v12"))
    finally:
        memory.close()
        mirror.close()


def test_typed_ack_prose_preserves_pending_meaning_and_other_history_bytes():
    data = {
        "history": [
            {"role": "user", "text": "先听我说完。"},
            {"role": "assistant", "text": "这是旧版普通回复。"},
            {"role": "assistant", "response_type": "answer", "text": "这是完整回答。"},
            {"role": "assistant", "response_type": "ack", "text": "嗯，我听着。"},
        ],
        "memory_status": "current", "memories": [], "exchanges": [],
    }
    payload = json.dumps(data, ensure_ascii=False)
    context = ShortContext("00000000-0000-0000-0000-000000000001", "input-ack", 1, payload)
    request = ShortRequest("继续说刚才的事。", context=context)
    rendered = request.model_request(192)["messages"][1]["content"]
    assert rendered.splitlines()[2:6] == [
        "1. 用户说：先听我说完。",
        "2. 助手说：这是旧版普通回复。",
        "3. 助手说：这是完整回答。",
        "4. 助手接话（ack，仅表示收到/在听，待答未解决）：嗯，我听着。",
    ]
    data["history"][-1].pop("response_type")
    assert rendered.replace("助手接话（ack，仅表示收到/在听，待答未解决）", "助手说") == (
        prose_evidence(json.dumps(data, ensure_ascii=False)))


def test_v6_candidate_control_reaches_default_request_with_direct_answer_rule(
    tmp_path, monkeypatch,
):
    control = control_for_revision("short-dialogue.v6")
    monkeypatch.setattr(default_runtime, "CONTROL", control)
    model = Scenario()
    memory, mirror, runtime, session = build(tmp_path, model)
    try:
        turn(runtime, session, QUERY)
        assert len(model.primary) == 2
        for messages in model.primary:
            system = messages[0]["content"]
            assert system.startswith(control)
            for rule in ("第一句就说清关键", "不先换词复述用户的问题", "后句只补必要的新信息",
                         "不重复前句", "用户要求细节、例子或多种原因时充分展开"):
                assert rule in system, rule
            assert messages[-1] == {"role": "user", "content": QUERY}
    finally:
        memory.close()
        mirror.close()


@pytest.mark.parametrize("user_text", [
    "刚听完一张很吵的专辑。",
    "详细解释风扇为什么能让人觉得凉，列出三种因素，不要给购买建议。",
])
def test_v8_examples_reach_default_control_without_entering_evidence(
    tmp_path, monkeypatch, user_text,
):
    control = control_for_revision("short-dialogue.v8")
    monkeypatch.setattr(default_runtime, "CONTROL", control)
    prepared = []
    model_request = ShortRequest.model_request

    def record_request(request, output_tokens, system=None):
        body = model_request(request, output_tokens, system)
        if system is None or not system.startswith(control):
            return body  # StreamDialogue also projects its already generated answer.
        baseline = model_request(request, output_tokens, CONTROL + system[len(control):])
        prepared.append((request, body, baseline))
        return body

    monkeypatch.setattr(ShortRequest, "model_request", record_request)
    runtime, session, memory, mirror, model = setup(tmp_path)
    try:
        event = turn(runtime, session, user_text)
        assert len(model.calls) == 2  # One answer and the existing review, no style agent.
        messages = model.calls[0]
        assert len(prepared) == 1
        request, body, baseline = prepared[0]
        assert request.user_text == user_text
        assert request.context.input_id == event["basis"]["input_id"]
        assert body["messages"] == messages
        assert body["messages"][1:] == baseline["messages"][1:]
        assert body["tools"] == baseline["tools"] == []
        assert body["max_output_tokens"] == baseline["max_output_tokens"] == 192
        assert [message["role"] for message in messages] == ["system", "user", "user"]
        system = messages[0]["content"]
        assert system.startswith(control)
        assert system[len(control):] == baseline["messages"][0]["content"][len(CONTROL):]
        guidance, heading, remainder = system.partition("\n表达示意：\n")
        assert heading, "The recorded v6 prompt has no separated response demonstrations"
        examples, boundary, _ = remainder.partition("\n事实与范围：\n")
        assert boundary and "不是当前用户的事实或历史" in guidance
        assert examples.count("用户：") == examples.count("回应：") == 2
        assert system.endswith(ShortRequest(user_text).operation_constraints())
        assert messages[-1] == {"role": "user", "content": user_text}
        assert examples not in messages[1]["content"]
        assert event["readset"]["facts"] == event["readset"]["history"] == []
    finally:
        memory.close()
        mirror.close()


def test_revision_rules_cover_the_recorded_failures():
    for rule in ("JSON", "相同的语言", "不要介绍或说明自己的身份", "先给出有用的一句"):
        assert rule in SYSTEM_V4, rule
    assert SYSTEM_V4.startswith(SYSTEM), "a revision extends the frozen baseline"


def test_unknown_revision_or_conflicting_system_is_refused():
    with pytest.raises(ValueError):
        ShortDialogue(Model(), cancelled=Event(),
                      policy="short-dialogue.v999")
    with pytest.raises(ValueError):
        ShortDialogue(Model(), cancelled=Event(),
                      policy="short-dialogue.v4", system="explicit")
    with pytest.raises(ValueError):
        ShortDialogue(Model(), cancelled=Event(), system="   ")


def test_v18_integrates_intent_delivery_instead_of_stacking_prohibitions():
    previous = PROMPTS["short-dialogue.v17"]
    candidate = PROMPTS["short-dialogue.v18"]
    assert sha256(previous.encode()).hexdigest() == (
        "fcb432b429ce3939d3208bd6f3d2a1c3463c584994b2657a3f58526e4d6d07bc")
    boundary = "\n事实与范围：\n"
    guidance, marker, tail = candidate.partition(boundary)
    previous_guidance, _, previous_tail = previous.partition(boundary)
    assert marker and tail == previous_tail
    assert len(guidance) < len(previous_guidance)
    assert guidance.count("回应方式（short-dialogue.v18）") == 1
    assert "short-dialogue.v17" not in guidance
    assert "虚构归于作品" in guidance
    assert "用户现实经历仍按事实来源" in guidance
    assert "一到两句" not in guidance and "表达示意" not in guidance
    assert DEFAULT_PROMPT_REVISION == "short-dialogue.v12"
    assert sha256(CONTROL.encode()).hexdigest() == (
        "91819f92b04d66fc002eb703a9a7cffff282e6949164625e23ccf123152290a5")


def test_v19_is_explicit_and_keeps_generation_guidance_and_default_unchanged():
    candidate = PROMPTS["short-dialogue.v19"]
    assert candidate == PROMPTS["short-dialogue.v18"].replace(
        "short-dialogue.v18", "short-dialogue.v19")
    assert default_runtime.control_for_prompt_revision("short-dialogue.v19") == (
        control_for_revision("short-dialogue.v19"))
    assert DEFAULT_PROMPT_REVISION == "short-dialogue.v12"
    assert sha256(CONTROL.encode()).hexdigest() == (
        "91819f92b04d66fc002eb703a9a7cffff282e6949164625e23ccf123152290a5")


V18_INTENT_CASES = [
    ("s-01", "普通分享", "只谈原话中事物的一处联系", "一场雨也能把一段很久以前的路带回脑海。"),
    ("s-02", "先让用户说", "把话语空间留给用户", "嗯，我在听。"),
    ("s-03", "明确想聊感受", "让用户自行描述感受", "你说的这份安静，给你的感觉是什么？"),
    ("s-04", "指定几条", "与指定数量相同的独立做法",
     "一、出门前确认路线和天气；二、走路时避开积水并放慢脚步。"),
    ("s-05", "一个简单安排", "保留原活动并串成一条可行路径",
     "等雨小且路面安全时，带伞沿平整的路走十分钟再回来。"),
    ("s-06", "解释原因", "各面怎样产生及怎样同时成立",
     "旧照片保留快乐的片段，让人开心；时间带来的变化又让人难过，"
     "同一段过去可以承载两种感受。"),
    ("s-07", "普通分享", "自然停下", "票根也是一种小小的时间记录。"),
    ("s-08", "明确想聊感受", "一个从原话出发的开放问题", "看着这些票根，你有什么感受？"),
]
