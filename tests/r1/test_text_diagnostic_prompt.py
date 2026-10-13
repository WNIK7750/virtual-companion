"""Process-only natural candidate selection on the normal desktop request path."""

from hashlib import sha256

import pytest
from test_default_cross_turn_memory import turn
from test_default_text_delivery import setup
from test_natural_intent_candidate import PreparedRecordingModel
from test_short_dialogue_prompt_versions import control_for_revision
from test_text_provider_bootstrap import application  # noqa: F401
from test_text_provider_bootstrap import runtime_observer as runtime_observer
from virtual_companion.cognition.short_dialogue import PROMPTS
from virtual_companion.text import __main__ as bootstrap
from virtual_companion.text import default_runtime
from virtual_companion.text.application import TextSession

CANDIDATES = (
    "short-dialogue.v13", "short-dialogue.v14", "short-dialogue.v15", "short-dialogue.v16",
    "short-dialogue.v17",
    "short-dialogue.v18",
)


def test_v18_is_explicitly_selectable_without_switching_the_product_default():
    revision = "short-dialogue.v18"
    assert default_runtime.DIAGNOSTIC_PROMPT_REVISIONS.count(revision) == 1
    assert default_runtime.control_for_prompt_revision(revision) == control_for_revision(revision)
    assert default_runtime.DEFAULT_PROMPT_REVISION == "short-dialogue.v12"
    assert sha256(default_runtime.CONTROL.encode()).hexdigest() == (
        "91819f92b04d66fc002eb703a9a7cffff282e6949164625e23ccf123152290a5")


def test_v16_keeps_frozen_comparison_and_fact_operation_obligation_boundaries():
    previous = PROMPTS["short-dialogue.v15"]
    candidate = PROMPTS["short-dialogue.v16"]
    assert sha256(previous.encode()).hexdigest() == (
        "45d503530a655197ac47ccc7d1eb2b9774649627b071e37cccba9295d4d38647")
    assert candidate != previous
    boundary = "\n事实与范围：\n"
    assert candidate.partition(boundary)[1:] == previous.partition(boundary)[1:]
    assert sha256(default_runtime.CONTROL.encode()).hexdigest() == (
        "91819f92b04d66fc002eb703a9a7cffff282e6949164625e23ccf123152290a5")


def test_v17_keeps_v16_frozen_and_limits_chat_questions_to_explicit_feeling_invites():
    previous = PROMPTS["short-dialogue.v16"]
    candidate = PROMPTS["short-dialogue.v17"]
    assert sha256(previous.encode()).hexdigest() == (
        "714ed08b8b924f446342d68c42256e764509a866ce3977074d3b5cea238502d5")
    boundary = "\n事实与范围：\n"
    assert candidate.partition(boundary)[1:] == previous.partition(boundary)[1:]
    guidance = candidate.partition(boundary)[0]
    assert "普通分享先给贴题、有内容的观察" in guidance
    assert "只有原话明确邀请聊某个感受时" in guidance
    assert "一个开放、直接的问题" in guidance
    assert "不提供猜测的场景或情绪选项" in guidance
    assert "只想说时不提问、不分析、不替用户追问" in guidance
    assert "指定数量" in guidance and "安全条件" in guidance and "分别解释" in guidance


@pytest.mark.parametrize("revision", CANDIDATES)
def test_candidate_is_instance_bound_and_reaches_checked_sent_request(tmp_path, revision):
    model = PreparedRecordingModel()
    default, _, memory, mirror, _ = setup(tmp_path, model)
    candidate = default_runtime.DefaultTextRuntime(
        memory, mirror, model, binding={"test": "synthetic"},
        delivery_timeout=0.8, prompt_revision=revision,
    )
    session = TextSession(model, mirror, {}, prepare_turn=candidate.prepare_turn,
                          before_start=candidate.before_start)
    try:
        control = control_for_revision(revision)
        event = turn(candidate, session, "我想去散步但在下雨，请给我一个简单安排。")
        generation, review = model.prepared
        assert generation["checked"] == generation["sent"]
        assert generation["sent"][0]["content"].startswith(control)
        assert review["checked"] == review["sent"]
        assert event["display_text"] == "这是合成答案。"
        assert candidate.control == control
        assert candidate.snapshot()["request_prompt_revision"] == revision
        assert candidate.snapshot()["request_control_sha256"] == (
            sha256(control.encode()).hexdigest())
        assert default.control == default_runtime.CONTROL == (
            control_for_revision("short-dialogue.v12"))
        assert default.prompt_revision == default_runtime.DEFAULT_PROMPT_REVISION == (
            "short-dialogue.v12")
        assert candidate.config != default.config
    finally:
        memory.close()
        mirror.close()


@pytest.mark.parametrize("revision", [None, *CANDIDATES])
def test_process_selection_survives_provider_rebuild_and_recovery(monkeypatch, request, revision):
    if revision is None:
        monkeypatch.delenv("VC_TEXT_DIAGNOSTIC_PROMPT_REVISION", raising=False)
    else:
        monkeypatch.setenv("VC_TEXT_DIAGNOSTIC_PROMPT_REVISION", revision)
    runtimes = []
    factory = bootstrap.DefaultTextRuntime

    def record(*args, **kwargs):
        value = factory(*args, **kwargs)
        runtimes.append(value)
        return value

    monkeypatch.setattr(bootstrap, "DefaultTextRuntime", record)
    dispatch, _ = request.getfixturevalue("application")
    expected = revision or "short-dialogue.v12"
    control_sha256 = sha256(control_for_revision(expected).encode()).hexdigest()
    assert dispatch("state", {})["request_prompt_revision"] == expected
    assert dispatch("state", {})["request_control_sha256"] == control_sha256
    selected = dispatch("selectProvider", {"provider_id": "deepseek-api", "confirmed": True})
    assert selected["status"] == "selected"
    assert dispatch("state", {})["request_prompt_revision"] == expected
    assert dispatch("selectProvider", {"provider_id": "local"})["status"] == "selected"
    memory_factory = bootstrap.TextMemoryRuntime
    attempts = 0

    def fail_once(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ValueError("synthetic rebuild failure")
        return memory_factory(*args, **kwargs)

    monkeypatch.setattr(bootstrap, "TextMemoryRuntime", fail_once)
    unavailable = dispatch("selectProvider", {"provider_id": "deepseek-api", "confirmed": True})
    assert unavailable["status"] == "unavailable"
    assert dispatch("state", {})["request_prompt_revision"] == expected
    assert len(runtimes) == 4
    assert all(runtime.prompt_revision == expected for runtime in runtimes)
    assert all(runtime.control == control_for_revision(expected) for runtime in runtimes)
    assert all(runtime.snapshot()["request_control_sha256"] == control_sha256
               for runtime in runtimes)


@pytest.mark.parametrize("revision", ["", "short-dialogue.v12", "unknown", " short-dialogue.v13"])
def test_invalid_process_revision_refuses_start_before_launch_or_state_write(monkeypatch, revision):
    class NoRead:
        def readline(self):
            raise AssertionError("Invalid prompt must fail before reading launch credential")

    monkeypatch.setenv("VC_TEXT_DIAGNOSTIC_PROMPT_REVISION", revision)
    monkeypatch.setattr(bootstrap.sys, "stdin", NoRead())
    with pytest.raises(ValueError, match="diagnostic prompt revision"):
        bootstrap.main()


def test_unknown_runtime_revision_is_rejected(tmp_path):
    _, _, memory, mirror, model = setup(tmp_path)
    try:
        with pytest.raises(ValueError, match="prompt revision"):
            default_runtime.DefaultTextRuntime(memory, mirror, model,
                binding={"test": "synthetic"}, prompt_revision="short-dialogue.unknown")
    finally:
        memory.close()
        mirror.close()
