from __future__ import annotations

import hashlib

import pytest
from virtual_companion.text.llama import VisibleText, verify_file
from virtual_companion.text.profile import load_profile


def test_thinking_tags_split_across_chunks_never_leak():
    visible = VisibleText()
    chunks = ["<thi", "nk>secret", " reasoning</th", "ink>你", "好"]
    assert "".join(visible.feed(chunk) for chunk in chunks) + visible.finish() == "你好"


def test_ordinary_text_is_streamed_and_incomplete_markup_is_suppressed():
    visible = VisibleText()
    assert visible.feed("你好") == "你好"
    assert visible.feed("<thi") == ""
    assert visible.finish() == ""


def test_artifact_drift_is_rejected_before_process_launch(tmp_path):
    artifact = tmp_path / "model"
    artifact.write_bytes(b"fixed")
    digest = hashlib.sha256(b"fixed").hexdigest()
    verify_file(artifact, digest)
    artifact.write_bytes(b"drift")
    with pytest.raises(ValueError, match="hash"):
        verify_file(artifact, digest)


def test_runtime_parameter_drift_is_rejected(tmp_path):
    config = tmp_path / "profile.json"
    config.write_text('{"sampling":{"temperature":9}}', encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        load_profile(config)
