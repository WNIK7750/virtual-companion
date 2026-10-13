from hashlib import sha256
from pathlib import Path

import pytest
from virtual_companion.text.default_runtime import (
    CONTROL,
    DEFAULT_PROMPT_REVISION,
    DIAGNOSTIC_PROMPT_REVISIONS,
    control_for_prompt_revision,
)

ROOT = Path(__file__).resolve().parents[2]


def test_neutral_prompt_covers_stage1_natural_dialogue_contract():
    prompt = (ROOT / "configs" / "text-baseline-prompt.txt").read_text(encoding="utf-8")

    assert "分享或倾诉" in prompt
    assert "不主动列建议" in prompt
    assert all(phrase in prompt for phrase in ("例子", "类比", "不同角度"))
    assert "不替用户宣布共识" in prompt
    assert "转题" in prompt and "接回" in prompt
    assert "明确要求任务、步骤或格式" in prompt
    assert "不要固定" in prompt


@pytest.mark.parametrize("revision", ["short-dialogue.v16", "short-dialogue.v17"])
def test_natural_diagnostic_keeps_production_control(revision):
    assert revision in DIAGNOSTIC_PROMPT_REVISIONS
    assert DEFAULT_PROMPT_REVISION == "short-dialogue.v12"
    assert CONTROL == control_for_prompt_revision(DEFAULT_PROMPT_REVISION)
    assert sha256(CONTROL.encode()).hexdigest() == (
        "91819f92b04d66fc002eb703a9a7cffff282e6949164625e23ccf123152290a5")
    assert control_for_prompt_revision(revision) != CONTROL
