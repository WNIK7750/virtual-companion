"""Default model messages keep current intent and operation outcomes legible.

These checks drive the real desktop runtime against its synthetic SQLite store;
they inspect the request assembled from confirmed history, pending exchanges and
the content-free operation receipt, not a model's literal reply wording.
"""

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
from test_short_dialogue_prompt_versions import control_for_revision
from virtual_companion.text import default_runtime


@pytest.fixture(params=[None, "short-dialogue.v13"], ids=["default", "candidate-v13"])
def natural_prompt_revision(request, monkeypatch):
    if request.param is not None:
        monkeypatch.setattr(default_runtime, "CONTROL", control_for_revision(request.param))
    return default_runtime.CONTROL


def test_default_request_keeps_current_topic_ahead_of_a_stale_pending_exchange(
    tmp_path, natural_prompt_revision,
):
    model = Scenario()
    memory, mirror, runtime, session = build(tmp_path, model)
    try:
        earlier = turn(runtime, session, QUERY)
        current = "我喜欢爵士音乐。"
        event = turn(runtime, session, current)
        messages = model.primary[-1]
        system, evidence = messages[0]["content"], messages[1]["content"]

        assert messages[-1] == {"role": "user", "content": current}
        assert QUERY in evidence
        assert "没有有效的音乐偏好记录。" in evidence
        assert any(ref["source_id"] == earlier["basis"]["input_id"]
                   for ref in event["readset"]["exchanges"])
        assert "等待助手回应" not in evidence
        assert "结合已确认对话" in evidence
        assert system.startswith(natural_prompt_revision)
        assert "历史待续" in system and "结合已确认对话" in system
    finally:
        memory.close()
        mirror.close()


def test_default_forget_request_explains_committed_effect_without_reintroducing_target(
    tmp_path, natural_prompt_revision,
):
    model = Scenario()
    memory, mirror, runtime, session = build(tmp_path, model)
    try:
        turn(runtime, session, ORIGINAL)
        event = turn(runtime, session, FORGET)
        messages = model.primary[-1]
        system, evidence = messages[0]["content"], messages[1]["content"]

        assert event["readset"]["current_operation"]["status"] == "committed"
        assert "遗忘操作已完成并生效" in evidence
        assert ORIGINAL not in evidence
        assert "按此结果明确反馈" in system
        assert "遗忘不复述已遗忘内容" in system
        assert messages[-1] == {"role": "user", "content": FORGET}
    finally:
        memory.close()
        mirror.close()


def test_default_request_distinguishes_already_applied_from_a_new_change(
    tmp_path, natural_prompt_revision,
):
    model = Scenario()
    memory, mirror, runtime, session = build(tmp_path, model)
    try:
        turn(runtime, session, ORIGINAL)
        turn(runtime, session, CORRECT)
        event = turn(runtime, session, CORRECT)
        messages = model.primary[-1]
        system, evidence = messages[0]["content"], messages[1]["content"]

        assert event["readset"]["current_operation"]["status"] == "already_applied"
        assert "目标已处于所要求状态，无需重复更改" in evidence
        assert "不夸大为再次修改" in system
    finally:
        memory.close()
        mirror.close()
