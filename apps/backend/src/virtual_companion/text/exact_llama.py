"""Opt-in local adapter for experiments that already assemble final messages.

The desktop baseline continues to use LlamaModel. This adapter shares its verified
runtime, capacity checks, streaming and cancellation, but never prepends another
system message. A frozen allowlist binds each stage's actual system text.
"""

from copy import deepcopy
from hashlib import sha256

from .llama import LlamaModel


class ExactLlamaModel(LlamaModel):
    def _filter_content(self, text, display_filter):
        # JSON strings may legitimately contain literal reasoning delimiters.
        # Dedicated reasoning delta fields are separate from content upstream;
        # ambiguous prefixes in content must reach the parser unchanged and fail
        # there, never be silently repaired by presentation filtering.
        if not isinstance(text, str):
            raise ValueError("Text content delta required")
        return text

    def __init__(self, profile, repository, *, system_sha256):
        if (
            not isinstance(system_sha256, frozenset)
            or not system_sha256
            or any(
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
                for value in system_sha256
            )
        ):
            raise ValueError("Frozen system prompt hashes required")
        super().__init__(deepcopy(profile), repository)
        self.system_sha256 = system_sha256

    def _payload(self, messages):
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError("Explicit system and user messages required")
        for index, message in enumerate(messages):
            if (
                not isinstance(message, dict)
                or set(message) != {"role", "content"}
                or not isinstance(message["content"], str)
                or not message["content"].strip()
                or message["role"] not in ({"system"} if index == 0 else {"user", "assistant"})
            ):
                raise ValueError("One initial system and explicit text messages required")
        if messages[-1]["role"] != "user":
            raise ValueError("Final user input required")
        if sha256(messages[0]["content"].encode()).hexdigest() not in self.system_sha256:
            raise ValueError("Unfrozen system prompt")
        # Reuse fixed model/thinking/sampling and artifact verification. The
        # baseline prompt is still hash checked, but is not part of this payload.
        payload = super()._payload([])
        payload["messages"] = deepcopy(messages)
        return payload
