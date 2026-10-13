"""Explicit v27 body and existing schemas; supplied text proves no model quality."""

import json
from copy import deepcopy
from hashlib import sha256
from threading import Event
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator
from test_natural_v26_participation import Counter, RecordingModel, StageRecorder
from test_rebuilt_turns import A
from virtual_companion.boundary.contracts import canonical
from virtual_companion.cognition import short_dialogue
from virtual_companion.cognition.short_dialogue import ShortContext, ShortDialogue, ShortRequest
from virtual_companion.providers.contracts import ProviderCapabilities, ProviderRoute
from virtual_companion.text import default_runtime
from virtual_companion.text.demand_memory import ACTION_SCHEMA
from virtual_companion.text.provider_model import ProviderModel

REVISION = "short-dialogue.v27"
LEGACY_PROMPTS_SHA256 = "d37cd25c837bf6c28fdb452f6eb5c0518c71f12cbca06c62fe39e46ec2cb5999"
SHARED_RESPONSIBILITY_SHA256 = "7e1ce3b2b408dc57442424ba0371e2a4ea931c235dca864dcf27373c3853d739"


def request(current, **changes):
    payload = {"history": [], "memories": [], "exchanges": [], "memory_status": "pending_source"}
    payload.update(changes)
    return ShortRequest(current, context=ShortContext(A, "current", 3, canonical(payload)))


def generated(current):
    model = RecordingModel()
    candidate = ShortDialogue(model, cancelled=Event(), policy=REVISION).generate(current)
    assert candidate.status == "candidate" and candidate.policy == REVISION
    assert model.counted == model.sent and len(model.sent) == 1
    return model.sent[0]


def test_v27_explicit_content_selection_precedes_current_delivery():
    assert getattr(short_dialogue, "CONTENT_PARTICIPATION_PROMPT_REVISION", None) == REVISION
    system = short_dialogue.PROMPTS[REVISION]
    assert system == short_dialogue.SYSTEM_V27
    headings = ["先确定有效材料", "再选择当前贡献", "完成本轮所求"]
    positions = [system.index(text) for text in headings]
    assert positions == sorted(positions)
    assert "物件的类别不提供这一件的状态" in system
    assert "一般关系保持一般范围" in system
    assert "合同约束你的交付" in system
    assert "ack不结清待答" in system and "短结束" in system
    assert short_dialogue.RESPONSE_TASK_RESPONSIBILITY not in system


def test_v27_keeps_all_legacy_prompts_shared_responsibilities_and_default_body():
    old = {key: value for key, value in short_dialogue.PROMPTS.items()
           if key not in {
               REVISION, "short-dialogue.v28", "short-dialogue.v29", "short-dialogue.v30",
               "short-dialogue.v31", "short-dialogue.v32", "short-dialogue.v33",
               "short-dialogue.v34", "short-dialogue.v35", "short-dialogue.v36"}}
    assert len(old) == 22 and sha256(canonical(old).encode()).hexdigest() == LEGACY_PROMPTS_SHA256
    shared = {"context": short_dialogue.CONTEXT_RESPONSE_RESPONSIBILITY,
              "task": short_dialogue.RESPONSE_TASK_RESPONSIBILITY}
    assert sha256(canonical(shared).encode()).hexdigest() == SHARED_RESPONSIBILITY_SHA256
    assert short_dialogue.NATURAL_PROMPT_REVISION == "short-dialogue.v25"
    assert short_dialogue.UNIT_SCOPE_PROMPT_REVISION == "short-dialogue.v26"
    model = RecordingModel()
    current = ShortRequest("你好")
    default = ShortDialogue(model, cancelled=Event()).generate(current)
    assert default.policy == short_dialogue.POLICY == "short-dialogue.v3"
    assert model.sent == [current.model_request(192, short_dialogue.SYSTEM)]


@pytest.mark.parametrize("revision", ["short-dialogue.v3", "short-dialogue.v12"])
def test_legacy_short_generator_preserves_three_argument_request_recorder(monkeypatch, revision):
    original, calls = ShortRequest.model_request, []

    def record_request(current, output_tokens, system=None):
        body = original(current, output_tokens, system)
        calls.append(body)
        return body

    monkeypatch.setattr(ShortRequest, "model_request", record_request)
    model = RecordingModel()
    candidate = ShortDialogue(model, cancelled=Event(), policy=revision).generate(
        ShortRequest("刚归还一本书。"))
    assert candidate.status == "candidate" and len(calls) == 1
    assert model.sent == calls


@pytest.mark.parametrize("status", ["pending_source", "needs_clarification", "current"])
def test_v27_one_generation_keeps_exact_user_and_sparse_access_boundary(status):
    current = request("整理盒子时找出几张书签。\n还想说一句。", memory_status=status)
    body = generated(current)
    assert body["messages"][-1] == {"role": "user", "content": current.user_text}
    assert body["messages"][1]["content"] == short_dialogue.prose_evidence(
        current.context.payload, sparse=True,
    )
    assert "相关长期记忆：无。" not in body["messages"][1]["content"]
    assert body["messages"][0]["content"].endswith(current.operation_constraints())
    assert body["tools"] == [] and body["max_output_tokens"] == 192
    sources = current.personal_sources(A, "current", 3)
    assert len(sources) == 1 and sources[0]["text"] == current.user_text


def test_v27_effective_view_retains_new_correction_pending_answer_and_ack_scope():
    current = request(
        "说完了，现在按更正后的情况说吧。", memory_status="current",
        history=[
            {"id": "owed", "role": "user", "text": "帮我安排读书时间，我还没讲完。"},
            {"id": "ack", "role": "assistant", "response_type": "ack", "text": "嗯，你继续。"},
            {"id": "correction", "role": "user", "text": "更正一下，早上不方便。"},
        ],
        memories=[{"record_id": "reading", "version": 2, "status": "active",
                   "source_kind": "input", "source_id": "preference",
                   "content": "用户偏好安静地读书。",
                   "expires_at": 123456789}],
        exchanges=[{"exchange_id": "still-owed", "revision": 2, "direction": "assistant_response",
                    "initiator": "user", "content": "帮我安排读书时间。", "expires_at": 123456790}],
    )
    body = generated(current)
    evidence = body["messages"][1]["content"]
    assert "早上不方便" in evidence and "用户偏好安静地读书" in evidence
    assert "ack，仅表示收到/在听，待答未解决" in evidence
    sources = current.personal_sources(A, "current", 3)
    assert [row["id"] for row in sources] == [
        "current:current", "history:owed", "history:correction", "memory:reading",
    ]
    assert sources[-1]["revision"] == 2 and sources[-1]["source_id"] == "preference"
    assert "嗯，你继续。" not in canonical(sources)


def prepared_generation(current, action, *, repair=False):
    transport = StageRecorder(action)
    route = ProviderRoute(
        "synthetic-v27-wiring", 1, "local", "http://127.0.0.1:9/v1/chat/completions",
        "synthetic-recording-only", "none", None, True,
        ("current", "history", "valid_memory", "derived"),
        ProviderCapabilities(True, False, True, 1_048_576, "max_tokens"),
    )
    model = ProviderModel(route, counter=Counter(), transport=transport)
    runtime = default_runtime.DefaultTextRuntime.__new__(default_runtime.DefaultTextRuntime)
    runtime.model, runtime.cancelled = model, Event()
    runtime.prompt_revision = REVISION
    runtime.review_audit_version = default_runtime.UNIT_SCOPE_AUDIT_VERSION
    runtime.control = default_runtime.control_for_prompt_revision(REVISION)
    runtime.repair_reply = default_runtime.NATURAL_REPAIR_REPLY
    runtime.claim = SimpleNamespace(input_id=current.context.input_id)
    runtime.companion = current.context.companion
    runtime._uncommitted_source = None
    runtime.check = lambda *args, **kwargs: None
    runtime.admit = lambda *args, **kwargs: current
    runtime.store = SimpleNamespace(snapshot=lambda companion: {},
                                    inspect=lambda companion, input_id: {"text": current.user_text})
    if repair:
        original = runtime.prepare(runtime.claim, final=False)
        feedback = SimpleNamespace(prepared=original, text="指定的未呈现候选。",
                                   verdict="指定来源覆盖拒绝。")
        prepared = runtime._prepare_turn_stage(runtime.claim, final=False, repair=feedback)
    else:
        prepared = runtime.prepare(runtime.claim, final=False)
    prepared.wire.check_context()
    checked = deepcopy(prepared.wire._body)
    prepared.messages[-1]["content"] = "这是尚未提交的编辑。"
    Draft202012Validator(ACTION_SCHEMA).validate(action)
    assert json.loads("".join(prepared.wire.stream(Event()))) == action
    assert len(transport.calls) == 1 and transport.calls[0][0] == checked
    assert transport.calls[0][1]["stage"] == ("demand_repair" if repair else "demand_control")
    assert checked["max_tokens"] == 512
    assert sum(row["role"] == "system" and row["content"].startswith(short_dialogue.SYSTEM_V27)
               for row in checked["messages"]) == 1
    return checked


@pytest.mark.parametrize("current,action,kind", [
    ("整理盒子时找出几张书签。", {"action": "answer", "text": "指定短候选。"}, None),
    ("还有后半段，先让我讲，不用回应。", {"action": "listen"}, None),
    ("先应一声，我还没讲完。", {"action": "ack", "text": "嗯，你继续。"}, None),
    ("我以前喜欢什么？", {"action": "memory", "query": "过去 兴趣"}, None),
    ("给我两条整理书桌的实用建议。", {"action": "answer", "text": "指定两条。"}, "exact_items"),
    ("我想去散步但在下雨，请给我一个简单安排。",
     {"action": "answer", "text": "指定安排。"}, "activity_plan"),
    ("为什么旧照片会让人同时开心又难过？",
     {"action": "answer", "text": "指定解释。"}, "general_explanation"),
])
def test_v27_actual_runtime_prepares_existing_actions_and_typed_task(current, action, kind):
    body = prepared_generation(request(current), action)
    assert body["messages"][-1] == {"role": "user", "content": current}
    contracts = [json.loads(row["content"])["response_task_contract"] for row in body["messages"]
                 if row["content"].startswith('{"response_task_contract":')]
    if kind is None:
        assert not contracts
    else:
        assert len(contracts) == 1 and contracts[0]["kind"] == kind
        assert contracts[0]["span"] == current
        assert contracts[0]["requires_complete_answer"] is True
        if kind == "exact_items":
            assert contracts[0]["count"] == 2
        elif kind == "activity_plan":
            assert contracts[0]["activity"] == "散步"
            assert contracts[0]["substitute_activities"] is False
        else:
            assert contracts[0]["attribution"] == "general_not_user_personal"


def test_v27_actual_repair_stays_typed_one_supplied_call_without_claiming_quality():
    body = prepared_generation(request("请继续回答刚才的问题。"),
                               {"action": "answer", "text": "指定修正。"}, repair=True)
    assert any(default_runtime.NATURAL_REPAIR_REPLY in row["content"] for row in body["messages"])
    assert "指定的未呈现候选。" in canonical(body)
    assert "指定来源覆盖拒绝。" in canonical(body)


def test_v28_generation_and_one_review_share_versioned_semantic_responsibility():
    from virtual_companion.cognition import stream_unit_review

    assert getattr(short_dialogue, "GROUNDED_PARTICIPATION_PROMPT_REVISION", None) == (
        "short-dialogue.v28")
    shared = short_dialogue.GROUNDED_CONVERSATION_RESPONSIBILITY
    assert shared in short_dialogue.PROMPTS["short-dialogue.v28"]
    assert shared in stream_unit_review.GROUNDED_UNIT_SYSTEM
    assert "short-dialogue.v28" in default_runtime.DIAGNOSTIC_PROMPT_REVISIONS
    assert default_runtime.DEFAULT_PROMPT_REVISION == "short-dialogue.v12"
    assert sha256(default_runtime.CONTROL.encode()).hexdigest() == (
        "91819f92b04d66fc002eb703a9a7cffff282e6949164625e23ccf123152290a5")


def test_v30_original_frame_and_counted_short_request_keep_the_same_roles_and_raw_current():
    from virtual_companion.cognition import stream_unit_review

    current = request("请按更正后的情况说吧。", memory_status="current", history=[
        {"id": "user-raw", "role": "user", "text": "上午原先方便。"},
        {"id": "assistant-raw", "role": "assistant", "text": "助手先前的比喻。"},
    ])
    model = RecordingModel()
    ShortDialogue(model, cancelled=Event(), policy="short-dialogue.v30").generate(current)
    body = model.sent[0]
    assert model.counted == model.sent and len(model.sent) == 1
    assert body["messages"][-1] == {"role": "user", "content": current.user_text}
    frame = json.loads(body["messages"][-2]["content"])["conversation_frame"]
    assert frame == current.conversation_frame() and frame["candidate_speaker"] == "assistant"
    assert all(row["role"] == "user" for row in frame["source_roles"])
    assert not {"text", "context", "meaning", "use", "response_fit"} & frame.keys()
    shared = short_dialogue.CONVERSATION_VERDICT_RESPONSIBILITY
    assert shared in short_dialogue.PROMPTS["short-dialogue.v30"]
    assert shared in stream_unit_review.VERDICT_UNIT_SYSTEM
    assert default_runtime.DEFAULT_PROMPT_REVISION == "short-dialogue.v12"
    assert sha256(default_runtime.CONTROL.encode()).hexdigest() == (
        "91819f92b04d66fc002eb703a9a7cffff282e6949164625e23ccf123152290a5")



def test_v29_new_shared_frame_is_only_known_identity_and_same_raw_sources():
    from virtual_companion.cognition import stream_unit_review

    current = request(
        "请按更正后的情况说吧。",
        memory_status="current",
        history=[
            {"id": "old-user", "role": "user", "text": "上午原先方便。"},
            {"id": "old-assistant", "role": "assistant", "text": "助手的比喻。"},
        ],
        memories=[
            {
                "record_id": "record",
                "version": 2,
                "status": "active",
                "source_kind": "input",
                "source_id": "correction",
                "content": "现在上午不方便。",
            }
        ],
    )
    model = RecordingModel()
    ShortDialogue(model, cancelled=Event(), policy="short-dialogue.v29").generate(current)
    body = model.sent[0]
    assert model.counted == model.sent and len(model.sent) == 1
    assert body["messages"][-1] == {"role": "user", "content": current.user_text}
    assert body["messages"][1]["content"] == short_dialogue.prose_evidence(
        current.context.payload, sparse=True
    )
    frame = json.loads(body["messages"][-2]["content"])["conversation_frame"]
    sources = current.personal_sources(A, "current", 3)
    assert frame == current.conversation_frame()
    assert (
        frame["candidate_speaker"] == "assistant" and frame["current_source_id"] == sources[0]["id"]
    )
    assert frame["source_roles"] == [{"id": row["id"], "role": row["role"]} for row in sources]
    assert all(row["role"] == "user" for row in frame["source_roles"])
    assert not {"text", "context", "meaning", "use", "response_fit"} & frame.keys()
    shared = short_dialogue.CONVERSATION_USE_RESPONSIBILITY
    assert (
        shared in short_dialogue.PROMPTS["short-dialogue.v29"]
        and shared in stream_unit_review.USE_UNIT_SYSTEM
    )
    assert default_runtime.DEFAULT_PROMPT_REVISION == "short-dialogue.v12"
