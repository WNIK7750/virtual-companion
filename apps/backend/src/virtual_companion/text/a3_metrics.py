"""Opt-in exact-message metrics; never changes the desktop model adapter."""

import json
import time
from copy import deepcopy
from hashlib import sha256

from .exact_llama import ExactLlamaModel


class MeasuredExactModel(ExactLlamaModel):
    def count_context(self, messages):
        self.input_tokens = super().count_context(messages)
        return self.input_tokens

    def stream(self, messages, cancelled):
        self.metrics = {
            "started_monotonic": time.monotonic(),
            "first_content_monotonic": None,
            "input_tokens": getattr(self, "input_tokens", None),
            "actual_output_tokens": None,
            "usage_source": "missing",
            "provider_usage": None,
            "token_measurement_error": None,
        }
        chunks = []
        try:
            for content in super().stream(messages, cancelled):
                if content and self.metrics["first_content_monotonic"] is None:
                    self.metrics["first_content_monotonic"] = time.monotonic()
                chunks.append(content)
                yield content
        finally:
            self.metrics["generation_end_monotonic"] = time.monotonic()
            self.metrics["cancelled"] = cancelled.is_set()
            try:
                tokens = self._context_request(
                    "/tokenize",
                    {"content": "".join(chunks), "add_special": False, "parse_special": False},
                )["tokens"]
                if not isinstance(tokens, list) or any(type(t) is not int for t in tokens):
                    raise ValueError("Invalid tokenizer response")
                self.metrics["actual_output_tokens"] = len(tokens)
                self.metrics["usage_source"] = "fixed_tokenizer_received_content"
            except Exception as error:
                self.metrics["token_measurement_error"] = type(error).__name__
            # Partial responses cannot reveal tokens generated after disconnect.
            self.metrics["unobserved_server_tokens"] = "unknown"

    def check_context(self, messages):
        self.metrics = {}
        super().check_context(messages)


class ConstrainedExactModel(MeasuredExactModel):
    """Explicit b10516 schema grammar; both A3 routes use the same frozen schema."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.schema_sha256 = frozenset()
        self._schema = None

    def bind_schema(self, schema):
        self._schema = None
        raw = json.dumps(
            schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        if (
            not isinstance(schema, dict)
            or sha256(raw.encode()).hexdigest() not in self.schema_sha256
        ):
            raise ValueError("Unfrozen A3 response schema")
        self._schema = deepcopy(schema)

    def _payload(self, messages):
        if self._schema is None:
            raise ValueError("A3 response schema must be bound before capacity and generation")
        return {
            **super()._payload(messages),
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "a3", "strict": True, "schema": deepcopy(self._schema)},
            },
        }
