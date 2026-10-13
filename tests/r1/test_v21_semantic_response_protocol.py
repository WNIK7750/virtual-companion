"""The diagnostic generation/review messages stay on the existing demand path."""

from hashlib import sha256

from virtual_companion.cognition.short_dialogue import PROMPTS
from virtual_companion.text import default_runtime


def test_v21_replaces_expression_responsibility_without_rewriting_frozen_versions():
    prompt = PROMPTS["short-dialogue.v21"]
    assert "回应契约（short-dialogue.v21）" in prompt
    assert "已给的起点" in prompt and "待用户回答的未知量" in prompt
    assert "问句也承担前提" in prompt
    assert "一般观察" in prompt and "事物类别" in prompt
    assert prompt.split("\n事实与范围：\n", 1)[1] == PROMPTS["short-dialogue.v20"].split(
        "\n事实与范围：\n", 1)[1]
    assert sha256(PROMPTS["short-dialogue.v20"].encode()).hexdigest() == (
        "8262c5622337fb8d30c971cbb51f8ee3bb14c04fe85ff9575f5bd3fa0ff8e9a1")
    assert default_runtime.DEFAULT_PROMPT_REVISION == "short-dialogue.v12"
    assert sha256(default_runtime.CONTROL.encode()).hexdigest() == (
        "91819f92b04d66fc002eb703a9a7cffff282e6949164625e23ccf123152290a5")


def test_v22_keeps_content_and_user_commitments_on_the_same_existing_path():
    prompt = PROMPTS["short-dialogue.v22"]
    assert "回应契约（short-dialogue.v22）" in prompt
    assert "连贯内容" in prompt and "本轮意图" in prompt
    assert "事物类别" in prompt and "未知量" in prompt
    assert "情绪选项" in prompt and "邀请" in prompt
    assert prompt.split("\n事实与范围：\n", 1)[1] == PROMPTS["short-dialogue.v21"].split(
        "\n事实与范围：\n", 1)[1]
    assert sha256(PROMPTS["short-dialogue.v21"].encode()).hexdigest() == (
        "042d45f28e6c39c3e13e8557b8cbdc9ca4ba6acbf9e1918bd263d3366f7d369f")
    assert default_runtime.DEFAULT_PROMPT_REVISION == "short-dialogue.v12"
    assert default_runtime.control_for_prompt_revision("short-dialogue.v22").startswith(
        prompt.replace("只输出回复正文", "回复正文放在text字段").replace(
            "不声称已执行或保存。", "操作反馈须符合本次实际证据。"))


def test_v23_is_one_standalone_generation_responsibility_with_frozen_default():
    prompt = PROMPTS["short-dialogue.v23"]
    assert "回应与事实责任（short-dialogue.v23）" in prompt
    assert "当前任务合同" not in prompt and "表达示意" not in prompt
    assert "个人承诺" in prompt and "一般联系" in prompt and "未知量" in prompt
    assert "详细" in prompt and "倾听" in prompt and "历史待续" in prompt
    assert default_runtime.DIAGNOSTIC_PROMPT_REVISIONS.count("short-dialogue.v23") == 1
    assert default_runtime.DEFAULT_PROMPT_REVISION == "short-dialogue.v12"
    assert sha256(default_runtime.CONTROL.encode()).hexdigest() == (
        "91819f92b04d66fc002eb703a9a7cffff282e6949164625e23ccf123152290a5")
    assert sha256(PROMPTS["short-dialogue.v22"].encode()).hexdigest() == (
        "3c0108f94a6231b9f831bd1675a03c9b9f2ae12d3ea76a4d23aee3af0a12c037")
    assert default_runtime.control_for_prompt_revision("short-dialogue.v23").startswith(
        prompt.replace("只输出回复正文", "回复正文放在text字段"))
