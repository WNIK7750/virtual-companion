from __future__ import annotations

import hashlib
import json
from pathlib import Path

FIXTURE_DIRECTORY = Path(__file__).resolve().parents[1] / "fixtures" / "r1"
HTML_PATH = FIXTURE_DIRECTORY / "screen-scene.v1.html"
GROUND_TRUTH_PATH = FIXTURE_DIRECTORY / "screen-scene.v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_screen_scene_is_offline_fixed_size_and_contains_test_cases() -> None:
    html = HTML_PATH.read_text(encoding="utf-8")

    assert 'data-scene-version="1"' in html
    assert 'width: 1280px' in html
    assert 'height: 720px' in html
    assert "普通话识别：今天是 2026 年 9 月 2 日。" in html
    assert "Order VC-2048 / ￥39.90" in html
    assert 'data-case="small-text"' in html
    assert 'data-case="occluder"' in html
    assert 'data-privacy="do-not-persist"' in html
    assert "http://" not in html
    assert "https://" not in html


def test_ground_truth_separates_observation_ttl_and_privacy_sentinel() -> None:
    truth = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))

    assert truth["schema_version"] == 1
    assert truth["viewport"] == {"width": 1280, "height": 720}
    assert truth["observation_contract"]["required"] == [
        "source",
        "confidence",
        "captured_at",
        "ttl_ms",
    ]
    assert truth["observation_contract"]["ttl_ms"] == 5_000
    assert truth["privacy_sentinel"]["persist_allowed"] is False
    assert truth["expected_text"]["small_text"] == "small text 7px: do not overclaim OCR"


def test_fixture_hashes_are_characterized_in_ground_truth() -> None:
    truth = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))

    assert truth["artifacts"]["html_sha256"] == _sha256(HTML_PATH)
