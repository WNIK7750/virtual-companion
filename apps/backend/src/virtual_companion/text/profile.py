from __future__ import annotations

import json
from pathlib import Path

from .llama import verify_file

PROFILE_SHA256 = "fee3c6b28660c1db335be5bd952d8d5e47b2d713515582458484bdb89c347310"


def load_profile(path: Path) -> dict:
    verify_file(path, PROFILE_SHA256)
    return json.loads(path.read_text(encoding="utf-8"))
