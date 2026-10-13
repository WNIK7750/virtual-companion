"""Recording/SQLite checks; canned semantic opinions do not prove model quality."""

import json
import os
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from threading import Event, Lock

import pytest
from test_character_context import character
from test_default_flash_route import boot as boot
from test_explicit_memory_management import seed
from test_memory_command_worker import command
from test_rebuilt_turns import A, B, config
from virtual_companion.boundary.sqlite_memory_management import SQLiteExplicitMemoryStore
from virtual_companion.cognition.background_worker import UserFactWorker
from virtual_companion.cognition.errors import FactSupportUnresolved
from virtual_companion.text import __main__ as bootstrap
from virtual_companion.text.memory_management import MemoryManagementService
from virtual_companion.text.memory_runtime import TextMemoryRuntime, _StructuredMemoryInvoker
from virtual_companion.text.runtime_skills import load_memory_skills

VERSION = "memory-skills.v2"
MAINTENANCE = "vc-memory-maintenance"
SESSION = "00000000-0000-0000-0000-000000000010"
NOW = 1_791_174_580_000_000


class RecordingModel:
    supports_stage_metadata = True
    instances = []

    def __init__(self, respond):
        self.respond, self.calls, self.results = respond, [], []
        self.instances.append(self)

    def prepare_stage_request(self, messages, *, stage, schema):
        frozen = deepcopy(messages)
        self.calls.append((stage, frozen, deepcopy(schema)))
        owner = self

        class Request:
            def check_context(self):
                pass

            def stream(self, cancelled):
                result = owner.respond(stage, json.loads(frozen[-1]["content"]))
                owner.results.append({"stage": stage, "result": deepcopy(result)})
                yield json.dumps(result, ensure_ascii=False)

        return Request()


@pytest.fixture(autouse=True)
def preserve_recorded_witness(request, tmp_path):
    start = len(RecordingModel.instances)
    yield
    target = os.environ.get("VC_R2_WITNESS_DIR")
    if target is None:
        return
    root = Path(target)
    root.mkdir(exist_ok=True)
    suffix = sha256(request.node.nodeid.encode()).hexdigest()[:12]
    recordings = [
        {"stage": stage, "messages": messages, "schema": schema}
        for model in RecordingModel.instances[start:] for stage, messages, schema in model.calls
    ]
    databases = [{"path": str(path), "sha256": sha256(path.read_bytes()).hexdigest()}
                 for path in tmp_path.rglob("*.sqlite3*") if path.is_file()]
    (root / f"{request.node.originalname}-{suffix}.json").write_text(json.dumps({
        "test": request.node.nodeid,
        "scope": "synthetic recording + actual fresh SQLite; semantic opinions are canned",
        "calls": recordings,
        "results": [result for model in RecordingModel.instances[start:]
                    for result in model.results],
        "database_files": databases,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def no_command():
    return {
        "action": "none", "requestedScope": "none", "targetIds": [],
        "evidenceQuote": "", "quoteStart": None, "quoteEnd": None,
        "replacementQuote": "", "replacementStart": None, "replacementEnd": None,
        "coverage": "not_command", "unresolved": [],
    }


def selected_invoker(model):
    return _StructuredMemoryInvoker(model, Lock(), memory_skills=load_memory_skills(VERSION))


def stage_names(model):
    return [stage for stage, _, _ in model.calls]


def maintenance_text():
    bundle = load_memory_skills(VERSION)
    return next(text for name, text, _ in bundle.instructions if name == MAINTENANCE)


def assert_same_source_and_policy(model, source):
    policy = maintenance_text()
    for _, messages, _ in model.calls:
        assert sum(message["content"] == policy for message in messages) == 1
        # Source remains the original protocol payload, not a rewritten Skill example.
        assert source in json.dumps(json.loads(messages[-1]["content"]), ensure_ascii=False)


def test_v2_is_explicit_and_r2_never_enters_generation_messages():
    legacy = load_memory_skills()
    selected = load_memory_skills(VERSION)
    assert legacy.version == "memory-skills.v1"
    assert selected.version == VERSION
    assert legacy.names == ("vc-memory-retrieval", "vc-context-composition")
    assert selected.names == (*legacy.names, MAINTENANCE)
    assert selected.fingerprint != legacy.fingerprint
    assert legacy.identity == selected.identity[:2]
    assert legacy.messages(navigation=True) == selected.messages(navigation=True)
    assert legacy.messages(navigation=False) == selected.messages(navigation=False)
    assert legacy.maintenance_messages(stage="memory_command", source_role="user") == []
    assert len(selected.maintenance_messages(stage="memory_command", source_role="user")) == 1
    assert selected.maintenance_messages(stage="stream_unit_review", source_role="user") == []
    assert selected.maintenance_messages(stage="memory_command", source_role="assistant") == []
    with pytest.raises(ValueError, match="Unsupported"):
        load_memory_skills("invented")
    assert bootstrap.diagnostic_options({"VC_TEXT_RUNTIME_MEMORY_SKILLS": VERSION}) == {
        "memory_skills": VERSION,
    }


def test_actual_for_source_binding_survives_policy_injection_and_is_user_only(tmp_path):
    store = SQLiteExplicitMemoryStore(tmp_path / "source.sqlite3")
    source = "我不是每天喝咖啡，只在夜班时偶尔喝。"
    item = store.accept(config(), "source", source)
    claim = store.claim_background(A, "recording", now_us=NOW, lease_us=300_000_000,
                                   role="user", source=item.input_id)
    model = RecordingModel(lambda stage, data: {"operations": []})
    invoke = selected_invoker(model).for_source(store, claim)
    messages = [{"role": "system", "content": "original bounded stage"},
                {"role": "user", "content": json.dumps({"source": source})}]
    invoke("fact_extract", messages, {}, Event())
    assert messages[0]["content"] == "original bounded stage"
    assert_same_source_and_policy(model, source)
    # This exact invoker is not bound to a user source; no maintenance policy leaks in.
    other = RecordingModel(lambda stage, data: {"operations": []})
    selected_invoker(other)("fact_extract", messages, {}, Event())
    assert all(message["content"] != maintenance_text() for message in other.calls[0][1])


@pytest.mark.parametrize("replacement", [
    "我每月最多听两次音乐，而且只在失眠时；这不代表我喜欢音乐。",
    "妈妈每天听音乐，我自己只是可能下周试试，并没有已经开始。",
])
def test_sqlite_correction_keeps_full_qualified_original_and_reopen(tmp_path, replacement):
    store = SQLiteExplicitMemoryStore(tmp_path / "correction.sqlite3")
    seed(store)
    source = "更正音乐那条记录：" + replacement
    item = store.accept(config(), "correction", source)
    start = source.index(replacement)

    def respond(stage, data):
        if stage == "memory_command":
            return {**command(source), "action": "correct", "replacementQuote": replacement,
                    "replacementStart": start, "replacementEnd": start + len(replacement)}
        assert stage == "memory_command_support"
        return {"meaning": "explicit_single_correction", "verdict": "supported"}

    model = RecordingModel(respond)
    receipt = UserFactWorker(store, selected_invoker(model), clock=lambda: NOW,
                            memory_commands=True).run_one(A, "worker", source=item.input_id)
    assert receipt["status"] == "committed"
    assert stage_names(model) == ["memory_command", "memory_command_support"]
    assert_same_source_and_policy(model, source)
    reopened = SQLiteExplicitMemoryStore(store.path)
    assert reopened.recall(A, now_us=NOW + 1)[0]["content"] == replacement
    assert [row["content"] for row in reopened.history(A, "music")] == [
        "我喜欢安静的音乐", replacement,
    ]
    assert reopened.recall(B, now_us=NOW + 1) == []


def test_ordinary_sharing_closes_source_without_forced_fact_or_obligation(tmp_path):
    store = SQLiteExplicitMemoryStore(tmp_path / "share.sqlite3")
    text = "今天只是想随便聊聊。"
    item = store.accept(config(), "ordinary", text)

    def respond(stage, data):
        if stage == "memory_command":
            return no_command()
        if stage == "exchange_intent":
            return {"coverage": "complete", "acts": [{
                "segmentId": 0, "endSegmentId": len(data["segments"]) - 1,
                "intent": "context", "targetIndex": None,
            }]}
        assert stage == "fact_extract"
        return {"operations": []}

    model = RecordingModel(respond)
    receipt = UserFactWorker(store, selected_invoker(model), clock=lambda: NOW,
                            memory_commands=True, resolution_protocol=True,
                            resolution_strategy="intent_first").run_one(
                                A, "worker", source=item.input_id)
    assert receipt
    assert stage_names(model) == ["memory_command", "exchange_intent", "fact_extract"]
    assert_same_source_and_policy(model, text)
    assert store.memory_status(A) == "current"
    assert store.recall(A, now_us=NOW + 1) == []
    assert store.pending_exchanges(A, now_us=NOW + 1) == []


@pytest.mark.parametrize("source,meaning", [
    ("不要忘记我喜欢安静的音乐。", "negated"),
    ("她说‘请忘记我喜欢安静的音乐’。", "quotation"),
    ("如果以后让你忘记音乐偏好，你会怎样？", "hypothetical"),
])
def test_bound_false_forget_still_requires_semantic_support(tmp_path, source, meaning):
    store = SQLiteExplicitMemoryStore(tmp_path / "false-forget.sqlite3")
    seed(store)
    item = store.accept(config(), "false-command", source)

    def respond(stage, data):
        if stage == "memory_command":
            return command(source)
        if stage == "memory_command_support":
            return {"meaning": meaning, "verdict": "unsupported"}
        if stage == "exchange_intent":
            return {"coverage": "complete", "acts": [{
                "segmentId": 0, "endSegmentId": len(data["segments"]) - 1,
                "intent": "context", "targetIndex": None,
            }]}
        if stage == "fact_extract":
            return {"operations": []}
        if stage == "exchange_extract":
            return {"operations": []}
        raise AssertionError(stage)

    model = RecordingModel(respond)
    UserFactWorker(store, selected_invoker(model), clock=lambda: NOW, memory_commands=True,
                   resolution_protocol=True, resolution_strategy="intent_first"
                   ).run_one(A, "worker", source=item.input_id)
    assert stage_names(model)[:2] == ["memory_command", "memory_command_support"]
    assert_same_source_and_policy(model, source)
    assert store.recall(A, now_us=NOW + 1)[0]["content"] == "我喜欢安静的音乐"
    assert len(store.history(A, "music")) == 1


def test_current_source_processing_is_once_and_ambiguous_target_stays_pending(tmp_path):
    source = "请忘记那个音乐偏好。"
    model = RecordingModel(lambda stage, data: command(source, targets=()))
    runtime = TextMemoryRuntime(tmp_path / "pending.sqlite3", A, model=model,
                                memory_skills=VERSION)
    try:
        seed(runtime.store)
        item = runtime.accept_user("unclear", source, SESSION)
        assert runtime.resolve_for_recall(item.input_id, Event()) != "current"
        assert stage_names(model) == ["memory_command"]
        assert runtime.store.memory_barrier(A)
        assert len(runtime.store.history(A, "music")) == 1
        assert_same_source_and_policy(model, source)
    finally:
        runtime.close()


def test_successful_same_source_call_does_not_execute_twice(tmp_path):
    source = "请忘记我喜欢安静的音乐这件事。"

    def respond(stage, data):
        if stage == "memory_command":
            return command(source)
        assert stage == "memory_command_support"
        return {"meaning": "explicit_single_forget", "verdict": "supported"}

    model = RecordingModel(respond)
    runtime = TextMemoryRuntime(tmp_path / "once.sqlite3", A, model=model, memory_skills=VERSION)
    try:
        seed(runtime.store)
        item = runtime.accept_user("forget", source, SESSION)
        assert runtime.resolve_for_recall(item.input_id, Event()) == "current"
        assert runtime.resolve_for_recall(item.input_id, Event()) == "current"
        assert stage_names(model) == ["memory_command", "memory_command_support"]
        assert runtime.store.recall(A, now_us=NOW) == []
        assert len(runtime.store.history(A, "music")) == 2
    finally:
        runtime.close()


def test_already_applied_and_version_changed_are_current_state_receipts(tmp_path):
    store = SQLiteExplicitMemoryStore(tmp_path / "outcomes.sqlite3")
    seed(store)
    service = MemoryManagementService(store, A, clock=lambda: NOW)
    value = "我仅在失眠时偶尔听音乐。"
    payload = {"operation_id": "first", "action": "correct", "record_id": "music",
               "expected_version": 1, "replacement": value}
    assert service.apply_action(payload)["status"] == "committed"
    repeated = service.apply_action({**payload, "operation_id": "repeat"})
    assert repeated["status"] == "already_applied"
    assert repeated["message"] == "当前记忆已经是你要求的状态，无需重复处理。"
    stale = service.apply_action({**payload, "operation_id": "stale", "replacement": "我每天听"})
    assert stale["status"] == "version_changed"
    assert len(store.history(A, "music")) == 2
    assert SQLiteExplicitMemoryStore(store.path).recall(A, now_us=NOW + 1)[0]["content"] == value


def test_raw_quote_binding_is_not_semantic_support_for_a_question(tmp_path):
    store = SQLiteExplicitMemoryStore(tmp_path / "support.sqlite3")
    source = "我每月最多听两次音乐吗？"
    item = store.accept(config(), "question", source)

    def respond(stage, data):
        if stage == "memory_command":
            return no_command()
        if stage == "exchange_intent":
            return {"coverage": "complete", "acts": [{
                "segmentId": 0, "endSegmentId": 0, "intent": "question", "targetIndex": None,
            }]}
        if stage == "fact_extract":
            return {"operations": [{
                "operation": "add", "type": "long_term", "evidenceTurnId": item.input_id,
                "evidenceQuote": source, "supersedesAtomId": None, "expiresAt": None,
                "factKey": {"subject": "user", "predicate": "音乐频率"},
            }]}
        assert stage == "fact_support"
        return {"intent": "question", "verdict": "unsupported"}

    model = RecordingModel(respond)
    with pytest.raises(FactSupportUnresolved):
        UserFactWorker(store, selected_invoker(model), clock=lambda: NOW, memory_commands=True,
                       resolution_protocol=True, resolution_strategy="intent_first").run_one(
                           A, "worker", source=item.input_id)
    assert stage_names(model) == [
        "memory_command", "exchange_intent", "fact_extract", "fact_support",
    ]
    assert_same_source_and_policy(model, source)
    assert store.memory_barrier(A)
    assert store.recall(A, now_us=NOW + 1) == []


def test_v2_startup_freezes_one_bundle_before_memory_and_generation(boot, monkeypatch):
    bundle = load_memory_skills(VERSION)
    calls, memories, runtimes = [], [], []
    original_memory, original_runtime = bootstrap.TextMemoryRuntime, bootstrap.DefaultTextRuntime

    def load(version):
        calls.append(version)
        return bundle

    def memory(*args, **kwargs):
        result = original_memory(*args, **kwargs)
        memories.append(result)
        return result

    def runtime(*args, **kwargs):
        result = original_runtime(*args, **kwargs)
        runtimes.append(result)
        return result

    monkeypatch.setenv("VC_TEXT_RUNTIME_MEMORY_SKILLS", VERSION)
    monkeypatch.setattr(bootstrap, "load_memory_skills", load)
    monkeypatch.setattr(bootstrap, "TextMemoryRuntime", memory)
    monkeypatch.setattr(bootstrap, "DefaultTextRuntime", runtime)
    dispatch, api, _ = boot()
    startup = dispatch("state", {})["startup"]
    assert startup["can_send"]
    assert startup["components"]["runtime_skills"]["protocol"] == VERSION
    assert startup["components"]["runtime_skills"]["bundle_sha256"] == bundle.fingerprint
    assert calls == [VERSION]
    assert memories[0].memory_skills is runtimes[0].memory_skills is bundle
    assert memories[0].scheduler.user.invoke.memory_skills is bundle
    assert api.sent == []


def test_missing_v2_resource_blocks_send_without_model_calls(boot, monkeypatch):
    monkeypatch.setenv("VC_TEXT_RUNTIME_MEMORY_SKILLS", VERSION)

    def unavailable(version):
        assert version == VERSION
        raise OSError("synthetic missing maintenance policy")

    monkeypatch.setattr(bootstrap, "load_memory_skills", unavailable)
    dispatch, api, _ = boot()
    state = dispatch("state", {})
    assert not state["startup"]["can_send"]
    assert "runtime_skills" in state["startup"]["blocked_reasons"]
    assert state["startup"]["components"]["runtime_skills"]["protocol"] == VERSION
    assert api.sent == []


def test_v2_selection_rejects_a_preloaded_legacy_bundle(tmp_path):
    with pytest.raises(ValueError, match="selected capability"):
        TextMemoryRuntime(tmp_path / "invalid.sqlite3", A, memory_skills=VERSION,
                          loaded_memory_skills=load_memory_skills())
    assert not (tmp_path / "invalid.sqlite3").exists()


def test_reopened_pending_source_uses_its_original_role_with_selected_policy(tmp_path):
    first = character()
    path = tmp_path / "role.sqlite3"
    initial = TextMemoryRuntime(path, A, character=first)
    source = "我只是想聊聊旧书，并没有换工作。"
    item = initial.accept_user("before-role-change", source, SESSION)
    initial.close()
    runtime = TextMemoryRuntime(path, A, character=character(personality="沉稳"))
    try:
        claim = runtime.store.claim_background(A, "worker", now_us=NOW,
                                               lease_us=300_000_000, role="user")
        model = RecordingModel(lambda stage, data: {"operations": []})
        selected_invoker(model).for_source(runtime.store, claim)(
            "fact_extract", [{"role": "user", "content": json.dumps({"source": source})}],
            {}, Event())
        role = model.calls[0][1][0]["content"]
        assert first.payload in role
        assert "沉稳" not in role
        assert_same_source_and_policy(model, source)
        saved = json.loads(runtime.store.inspect(A, item.input_id)["config"])
        assert saved["companion_config"]["character_data"] == first.to_mapping()
    finally:
        runtime.close()


def test_request_creates_pending_obligation_without_completed_action_or_user_fact(tmp_path):
    store = SQLiteExplicitMemoryStore(tmp_path / "request.sqlite3")
    text = "请下周提醒妈妈别每天听音乐。"
    item = store.accept(config(), "request", text)

    def respond(stage, data):
        if stage == "memory_command":
            return no_command()
        if stage == "exchange_intent":
            return {"coverage": "complete", "acts": [{
                "segmentId": 0, "endSegmentId": len(data["segments"]) - 1,
                "intent": "request", "targetIndex": None,
            }]}
        if stage == "fact_extract":
            return {"operations": []}
        if stage == "exchange_extract":
            return {"operations": [{
                "operation": "add", "type": "response_obligation", "expiresAt": None,
                "targetIndex": None, "evidenceQuoteIndex": 0,
            }]}
        assert stage == "exchange_support"
        return {"intent": "request", "verdict": "supported"}

    model = RecordingModel(respond)
    UserFactWorker(store, selected_invoker(model), clock=lambda: NOW, memory_commands=True,
                   resolution_protocol=True, resolution_strategy="intent_first").run_one(
                       A, "worker", source=item.input_id)
    assert stage_names(model) == ["memory_command", "exchange_intent", "fact_extract",
                                 "exchange_extract", "exchange_support"]
    assert_same_source_and_policy(model, text)
    assert store.recall(A, now_us=NOW + 1) == []
    pending = store.pending_exchanges(A, now_us=NOW + 1)
    assert len(pending) == 1 and pending[0]["content"] == text
    assert pending[0]["status"] == "pending"


def test_recorded_wrong_subject_support_can_still_pass_exact_quote_binding(tmp_path):
    """A limitation witness: intentionally wrong semantic fakes are NOT product success."""
    store = SQLiteExplicitMemoryStore(tmp_path / "wrong-subject.sqlite3")
    source = "妈妈只在失眠时偶尔听音乐，并不代表我也会听。"
    item = store.accept(config(), "wrong-subject", source)

    def respond(stage, data):
        if stage == "memory_command":
            return no_command()
        if stage == "exchange_intent":
            return {"coverage": "complete", "acts": [{
                "segmentId": 0, "endSegmentId": len(data["segments"]) - 1,
                "intent": "statement", "targetIndex": None,
            }]}
        if stage == "fact_extract":
            return {"operations": [{
                "operation": "add", "type": "long_term", "evidenceTurnId": item.input_id,
                "evidenceQuote": source, "supersedesAtomId": None, "expiresAt": None,
                "factKey": {"subject": "user", "predicate": "音乐习惯"},
            }]}
        if stage == "fact_support":
            # Deliberately wrong support: the full original cannot justify subject=user.
            return {"intent": "assertion", "verdict": "supported"}
        assert stage == "exchange_extract"
        return {"operations": []}

    model = RecordingModel(respond)
    UserFactWorker(store, selected_invoker(model), clock=lambda: NOW, memory_commands=True,
                   resolution_protocol=True, resolution_strategy="intent_first").run_one(
                       A, "worker", source=item.input_id)
    row, = store.recall(A, now_us=NOW + 1)
    assert row["content"] == source
    assert row["fact_key"]["subject"] == "user", "record the known semantic failure unchanged"
    assert_same_source_and_policy(model, source)


def test_confirmed_assistant_source_never_gets_user_maintenance_policy(tmp_path):
    from test_rebuilt_turns import claim
    from test_segment_store import reviewed
    from test_stream_history import confirm

    store = SQLiteExplicitMemoryStore(tmp_path / "assistant.sqlite3")
    item = store.accept(config(), "user", "你好。")
    run = claim(store, item)
    store.approve_segment(run, reviewed(run, ("我只是提出一个假设。",)), now_us=100)
    event = store.dispatch_segment(run, 1, now_us=100)
    confirm(store, run, event)
    store.finish_stream(run, outcome="completed", now_us=100)
    assistant = store.claim_background(A, "assistant-worker", now_us=NOW,
                                       lease_us=300_000_000, role="assistant")
    assert assistant is not None and assistant.role == "assistant"
    model = RecordingModel(lambda stage, data: {"operations": []})
    selected_invoker(model).for_source(store, assistant)(
        "exchange_extract", [{"role": "user", "content": json.dumps({
            "source": "我只是提出一个假设。"})}], {}, Event())
    assert all(message["content"] != maintenance_text() for message in model.calls[0][1])


def test_pre_cancelled_source_makes_no_model_call_or_completion(tmp_path):
    model = RecordingModel(lambda stage, data: no_command())
    runtime = TextMemoryRuntime(tmp_path / "cancel.sqlite3", A, model=model, memory_skills=VERSION)
    try:
        item = runtime.accept_user("cancel", "请记住我只在失眠时偶尔听音乐。", SESSION)
        cancelled = Event()
        cancelled.set()
        assert runtime.resolve_for_recall(item.input_id, cancelled) == "cancelled"
        assert model.calls == []
        assert runtime.store.memory_barrier(A)
        assert runtime.store.recall(A, now_us=NOW) == []
    finally:
        runtime.close()
