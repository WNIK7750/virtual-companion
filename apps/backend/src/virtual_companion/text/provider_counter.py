"""Pinned narrow DeepSeek V4.1 text counter; every hosted response is cross-checked.

Renderer adapted from DeepSeek recipe 8cadfede7063c896b944e7bae05daa3549ae97ea
(MIT, Copyright 2026 DeepSeek). See THIRD_PARTY_NOTICES.md. No model execution.
"""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

from virtual_companion.providers.contracts import (
    DEEPSEEK_FLASH_PROFILE,
    DEEPSEEK_FOCUSED_PROFILE,
    DEEPSEEK_TEXT_PROFILES,
)

TOKENIZER_SHA256 = "81f64d1248a68ce3663e07ab3ee48b851e5df0e32d27cb98e4c9a268151e8d99"
BOS, EOS = "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"
ROLE = {"system": "<｜System｜>", "user": "<｜User｜>", "assistant": "<｜Assistant｜>"}
END_THINK = "</think>"
FORMAT = (
    "\n\n## Response Format:\n\n"
    "You MUST strictly adhere to the following schema to reply:\n"
    '{"type": "json_object"}'
)
REQUIRED = {"model", "messages", "thinking", "stream", "stream_options", "max_tokens"}


def _read_verified(path):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != TOKENIZER_SHA256:
        raise ValueError("Provider tokenizer artifact hash differs")
    data = json.loads(raw)
    markers = tuple(
        sorted(
            {item["content"] for item in data["added_tokens"] if item["content"]}
            | {"<think>", "</think>", "｜DSML｜"}
        )
    )
    return raw, markers


def _render(body, markers, *, wire_profile=DEEPSEEK_FLASH_PROFILE):
    if wire_profile not in DEEPSEEK_TEXT_PROFILES:
        raise ValueError("Unsupported provider counter wire profile")
    optional = {"response_format"}
    if wire_profile == DEEPSEEK_FOCUSED_PROFILE:
        optional.add("temperature")
        if (not isinstance(body, dict) or type(body.get("temperature")) not in {int, float}
                or body["temperature"] not in {0, 0.2}):
            raise ValueError("Request outside focused counter wire domain")
    if (
        not isinstance(body, dict)
        or not REQUIRED <= set(body)
        or set(body) - REQUIRED - optional
        or body["model"] != "deepseek-flash"
        or body["thinking"] != {"type": "disabled"}
        or body["stream"] is not True
        or body["stream_options"] != {"include_usage": True}
        or type(body["max_tokens"]) is not int
        or not 1 <= body["max_tokens"] <= 393216
    ):
        raise ValueError("Request outside candidate text wire domain")
    messages = body["messages"]
    if not isinstance(messages, list) or not 0 < len(messages) <= 512:
        raise ValueError("Bounded plain-text messages required")
    normalized = []
    for message in messages:
        if (
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or message["role"] not in ROLE
            or not isinstance(message["content"], str)
        ):
            raise ValueError("Plain system/user/assistant messages only")
        content, role = message["content"], message["role"]
        try:
            content.encode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise ValueError("Unicode outside candidate domain") from exc
        if any(marker in content for marker in markers):
            raise ValueError("Special-token literal outside candidate domain")
        if role == "system" and not content:
            continue
        if role == "user" and normalized and normalized[-1]["role"] == "user":
            normalized[-1]["content"] += "\n\n" + content
        else:
            normalized.append(deepcopy(message))
    if not normalized or normalized[-1]["role"] == "assistant":
        raise ValueError("Assistant-prefix completion outside candidate domain")
    if "response_format" in body:
        if body["response_format"] != {"type": "json_object"} or not any(
            "json" in message["content"].lower() for message in normalized
        ):
            raise ValueError("Explicit JSON-object instruction required")
        if normalized[0]["role"] != "system":
            normalized.insert(0, {"role": "system", "content": ""})
        normalized[0]["content"] += FORMAT
    parts = [BOS]
    for message in normalized:
        role, content = message["role"], message["content"]
        parts.extend((ROLE[role], END_THINK if role == "assistant" else "", content))
        if role == "assistant":
            parts.append(EOS)
    parts.extend((ROLE["assistant"], END_THINK))
    result = "".join(parts)
    if len(result.encode("utf-8")) > 2_000_000:
        raise ValueError("Candidate prompt byte limit exceeded")
    return result


class DeepSeekCounter:
    def __init__(self, tokenizer_path, *, wire_profile=DEEPSEEK_FLASH_PROFILE):
        import importlib.metadata

        from tokenizers import Tokenizer

        if importlib.metadata.version("tokenizers") != "0.23.2":
            raise ValueError("Candidate tokenizers runtime version differs")
        if wire_profile not in DEEPSEEK_TEXT_PROFILES:
            raise ValueError("Unsupported provider counter wire profile")
        self.wire_profile = wire_profile
        raw, self.forbidden_tokens = _read_verified(str(Path(tokenizer_path).resolve()))
        self._tokenizer = Tokenizer.from_str(raw.decode("utf-8"))

    def render(self, body):
        return _render(body, self.forbidden_tokens, wire_profile=self.wire_profile)

    def __call__(self, body):
        prompt = self.render(body)
        count = len(self._tokenizer.encode(prompt, add_special_tokens=False).ids)
        if count > len(prompt.encode("utf-8")):
            raise ValueError("Pinned local byte-bound invariant failed")
        return count

    def local_byte_bound(self, body):
        """BPE byte bound for this rendered prompt, NOT a hosted billing guarantee."""
        return len(self.render(body).encode("utf-8"))
