"""Explicit v26 generation wiring; CPU suppliers make no semantic-quality claim."""

import json
from copy import deepcopy
from dataclasses import replace
from threading import Event
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator
from test_effective_history import NOW, admission, complete, seeded
from test_explicit_memory_management import list_command
from test_rebuilt_turns import A, config
from test_segment_store import reviewed
from virtual_companion.boundary.contracts import canonical
from virtual_companion.cognition import short_dialogue
from virtual_companion.cognition.short_dialogue import ShortContext, ShortDialogue, ShortRequest
from virtual_companion.providers.contracts import ProviderCapabilities, ProviderRoute
from virtual_companion.text import default_runtime
from virtual_companion.text.demand_memory import ACTION_SCHEMA
from virtual_companion.text.provider_model import ProviderModel

REVISION = "short-dialogue.v26"


class RecordingModel:
    def __init__(self):
        self.counted, self.sent = [], []

    def count(self, body):
        self.counted.append(deepcopy(body))
        return 17

    def invoke(self, body):
        self.sent.append(deepcopy(body))
        return {"text": "这是指定的合成候选。", "tool_calls": [], "finish_reason": "stop"}


def generated(request):
    model = RecordingModel()
    result = ShortDialogue(model, cancelled=Event(), policy=REVISION).generate(request)
    assert result.status == "candidate" and result.policy == REVISION
    assert model.counted == model.sent and len(model.sent) == 1
    return model.sent[0]


def test_v26_selects_one_current_content_and_participation_responsibility():
    assert getattr(short_dialogue, "UNIT_SCOPE_PROMPT_REVISION", None) == REVISION
    system = short_dialogue.PROMPTS[REVISION]
    assert system == short_dialogue.SYSTEM_V26 and system != short_dialogue.SYSTEM_V25
    assert "当前内容与参与方式" in system
    assert "具体贡献" in system and "一般观察、联系、解释、比喻或观点" in system
    assert "有用的未知量" in system and "短句或自然停下" in system
    assert "完整当前输入" in system and "仍有关且未答" in system
    assert "当前原话或有效来源" in system and "提问的前提" in system
    assert "ack不结清待答" in system
    assert short_dialogue.CONTEXT_RESPONSE_RESPONSIBILITY not in system
    assert short_dialogue.RESPONSE_TASK_RESPONSIBILITY in system
    assert short_dialogue.NATURAL_PROMPT_REVISION == "short-dialogue.v25"


@pytest.mark.parametrize("status", ["pending_source", "needs_clarification", "current"])
def test_v26_sparse_body_keeps_exact_current_user_and_actual_access_status(status):
    payload = canonical({"history": [], "memories": [], "exchanges": [], "memory_status": status})
    current = "刚完成这一小件事。\n还有一处想和你说。"
    context = ShortContext(A, "current-input", 4, payload)
    request = ShortRequest(current, context=context)
    body = generated(request)
    assert body["messages"][-1] == {"role": "user", "content": current}
    assert body["messages"][1]["content"] == short_dialogue.prose_evidence(payload, sparse=True)
    assert "用户事实整理状态：" in body["messages"][1]["content"]
    assert "相关长期记忆：无。" not in body["messages"][1]["content"]
    assert body["messages"][0]["content"].endswith(request.operation_constraints())
    assert request.context.payload == payload
    assert body["tools"] == [] and body["max_output_tokens"] == 192


def test_v26_rich_view_keeps_effective_background_ack_pending_question_and_new_correction():
    payload = canonical({
        "memory_status": "current",
        "history": [
            {"id": "question", "role": "user", "text": "帮我安排读书时间，我还没讲完。"},
            {"id": "ack", "role": "assistant", "response_type": "ack", "text": "嗯，你继续。"},
            {"id": "correction", "role": "user", "text": "更正一下，早上不方便。"},
        ],
        "memories": [{
            "record_id": "reading", "version": 2, "status": "active", "source_kind": "input",
            "source_id": "preference-source", "content": "用户偏好安静地读书。",
            "expires_at": 123456789,
        }],
        "exchanges": [{
            "exchange_id": "still-owed", "revision": 2, "direction": "assistant_response",
            "initiator": "user", "content": "帮我安排读书时间。", "expires_at": 123456790,
        }],
    })
    request = ShortRequest("说完了，现在你说吧。", context=ShortContext(A, "handoff", 7, payload))
    body = generated(request)
    evidence = body["messages"][1]["content"]
    for text in ("早上不方便", "用户偏好安静地读书", "帮我安排读书时间", "123456789"):
        assert text in evidence
    assert "ack，仅表示收到/在听，待答未解决" in evidence
    catalog = request.personal_sources(A, "handoff", 7)
    assert [row["id"] for row in catalog] == [
        "current:handoff", "history:question", "history:correction", "memory:reading",
    ]
    assert catalog[-1]["revision"] == 2 and catalog[-1]["source_id"] == "preference-source"
    assert "嗯，你继续。" not in canonical(catalog)
    assert body["messages"][-1]["content"] == request.user_text


@pytest.mark.parametrize("change", ["correct", "forget", "expiry"])
def test_v26_actual_admission_does_not_restore_invalid_history_or_confirmed_old_fact(
    tmp_path, change,
):
    store, original, query, seed = seeded(tmp_path, expires=change == "expiry")
    if change == "correct":
        store.mutate(
            replace(seed, mutation_id="change", kind="correct", content="我喜欢热闹的音乐",
                    source_id="new-source", expected_record_version=1, fact_key=None),
            store.owner(A), expected_version=store.snapshot(A).write_version,
        )
    elif change == "forget":
        store.execute_memory_command(list_command(), now_us=NOW + 20)
    current = store.accept(config(), "next", "聊聊音乐吧", session_id=original.session_id)
    complete(store, current)
    _, request = admission(store, current, now=NOW + 2_000_000)
    body = generated(request)
    assert "安静的音乐" not in body["messages"][1]["content"]
    assert store.segments(A, query.input_id)[0]["text"] == "你喜欢安静的音乐。"
    assert not any("安静" in row["text"] for row in json.loads(request.context.payload)["history"])


def test_v26_default_plain_candidate_and_explicit_constraints_keep_existing_truth():
    model = RecordingModel()
    request = ShortRequest(
        "请不要用数字。", forbidden_characters="123", constraint_source="current-user",
    )
    generated(request)
    default = ShortDialogue(model, cancelled=Event()).generate(ShortRequest("你好"))
    assert default.policy == short_dialogue.POLICY == "short-dialogue.v3"
    assert model.sent[0]["messages"][0]["content"].startswith(short_dialogue.SYSTEM.replace(
        "不声称已执行或保存。", "操作反馈须符合本次实际证据。"))


def test_v26_actual_admission_keeps_confirmed_prefix_and_excludes_unsent_candidate_tail(tmp_path):
    store, original, _, _ = seeded(tmp_path)
    earlier = store.accept(config(), "earlier", "还有一段没说完。", session_id=original.session_id)
    complete(store, earlier)
    run, _ = admission(store, earlier)
    store.approve_segment(run, reviewed(run, ("这是已确认的接话。",)), now_us=NOW + 11)
    event = store.dispatch_segment(run, 1, now_us=NOW + 12)
    store.confirm_segment(A, earlier.input_id, 1, fingerprint=event["fingerprint"],
                          host_receipt="synthetic-prefix-only", channel="text_visible")
    store.approve_segment(run, reviewed(run, ("这是已确认的接话。", "这是未发送的候选尾段。")),
                          now_us=NOW + 12)
    store.cancel(A, earlier.input_id)
    current = store.accept(config(), "current", "现在你说吧。",
                           session_id=original.session_id)
    complete(store, current)
    _, request = admission(store, current)
    body = generated(request)
    assert "这是已确认的接话。" in body["messages"][1]["content"]
    assert "这是未发送的候选尾段。" not in canonical(body)
    assert store.segments(A, earlier.input_id)[1]["text"] == "这是未发送的候选尾段。"
    assert body["messages"][-1]["content"] == "现在你说吧。"


class Counter:
    def __call__(self, body):
        return 17

    def local_byte_bound(self, body):
        return len(canonical(body).encode())


class StageRecorder:
    """Declared action fixture, without a classifier, socket, credential or ledger."""

    def __init__(self, action):
        self.action, self.calls = action, []

    def invoke(self, body, cancelled, **options):
        assert not cancelled.is_set()
        self.calls.append((deepcopy(body), deepcopy(options)))
        Draft202012Validator(ACTION_SCHEMA).validate(self.action)
        count = options["input_tokens"]
        return {
            "text": canonical(self.action), "tool_calls": [], "finish_reason": "stop",
            "usage": {"prompt_tokens": count, "completion_tokens": 5, "total_tokens": count + 5},
        }

    def cancel(self):
        pass


def prepared_generation(request, action):
    transport = StageRecorder(action)
    route = ProviderRoute(
        "synthetic-v26-wiring", 1, "local", "http://127.0.0.1:9/v1/chat/completions",
        "synthetic-recording-only", "none", None, True,
        ("current", "history", "valid_memory", "derived"),
        ProviderCapabilities(True, False, True, 1_048_576, "max_tokens"),
    )
    model = ProviderModel(route, counter=Counter(), transport=transport)
    runtime = default_runtime.DefaultTextRuntime.__new__(default_runtime.DefaultTextRuntime)
    runtime.model, runtime.cancelled = model, Event()
    runtime.prompt_revision = REVISION
    runtime.control = default_runtime.control_for_prompt_revision(REVISION)
    runtime.claim = SimpleNamespace(input_id=request.context.input_id)
    runtime.companion = request.context.companion
    runtime._uncommitted_source = None
    runtime.check = lambda *args, **kwargs: None
    # Admission itself is separately exercised with the actual SQLite store.
    runtime.admit = lambda *args, **kwargs: request
    runtime.store = SimpleNamespace(snapshot=lambda companion: {},
                                    inspect=lambda companion, input_id: {"text": request.user_text})
    prepared = runtime.prepare(runtime.claim, final=False)
    prepared.wire.check_context()
    checked = deepcopy(prepared.wire._body)
    prepared.messages[-1]["content"] = "这是一段尚未提交的编辑。"
    assert json.loads("".join(prepared.wire.stream(Event()))) == action
    assert len(transport.calls) == 1 and transport.calls[0][0] == checked
    assert transport.calls[0][1]["stage"] == "demand_control"
    assert checked["max_tokens"] == 512
    selected = short_dialogue.SYSTEM_V26.replace("只输出回复正文", "回复正文放在text字段")
    assert sum(row["role"] == "system" and row["content"].startswith(selected)
               for row in checked["messages"]) == 1
    assert checked["messages"][-1] == {"role": "user", "content": request.user_text}
    return checked


@pytest.mark.parametrize("current,action,task_kind", [
    ("刚把一件小事做完了。", {"action": "answer", "text": "这是指定的短候选。"}, None),
    ("还有后半段，先让我讲，不用回应。", {"action": "listen"}, None),
    ("先应一声，我还没讲完。", {"action": "ack", "text": "嗯，你继续。"}, None),
    ("给我两条整理书桌的实用建议。", {"action": "answer", "text": "一条。另一条。"}, "exact_items"),
    ("我想去散步但在下雨，请给我一个简单安排。",
     {"action": "answer", "text": "这是指定安排。"}, "activity_plan"),
    ("为什么旧照片会让人同时开心又难过？",
     {"action": "answer", "text": "这是指定解释。"}, "general_explanation"),
])
def test_v26_real_prepared_wire_preserves_three_actions_and_current_task_contract(
    current, action, task_kind,
):
    payload = canonical({
        "history": [], "memories": [], "exchanges": [], "memory_status": "pending_source",
    })
    request = ShortRequest(current, context=ShortContext(A, "current", 1, payload))
    body = prepared_generation(request, action)
    contracts = [json.loads(row["content"])["response_task_contract"] for row in body["messages"]
                 if row["content"].startswith('{"response_task_contract":')]
    if task_kind is None:
        assert not contracts
    else:
        assert len(contracts) == 1 and contracts[0]["kind"] == task_kind
        assert contracts[0]["span"] == current and contracts[0]["requires_complete_answer"] is True
        if task_kind == "exact_items":
            assert contracts[0]["count"] == 2
        elif task_kind == "activity_plan":
            assert contracts[0]["activity"] == "散步"
            assert contracts[0]["substitute_activities"] is False
        else:
            assert contracts[0]["attribution"] == "general_not_user_personal"
