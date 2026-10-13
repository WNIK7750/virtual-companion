"""No-network Model Port fixture; synthetic counting is never a tokenizer claim."""

import time
from hashlib import sha256

from virtual_companion.boundary.contracts import canonical
from virtual_companion.providers.bound_call import BoundProviderCall
from virtual_companion.providers.budget import ProviderBudgetLedger
from virtual_companion.providers.contracts import ProviderCapabilities, ProviderRoute


class RecordedProviderModels:
    def __init__(self, source, path):
        self.source = source
        self.route = ProviderRoute(
            "synthetic",
            1,
            "local",
            "http://127.0.0.1:1/v1/chat/completions",
            "synthetic-recording",
            "none",
            None,
            False,
            ("current",),
            ProviderCapabilities(True, False, True, 131072, "max_tokens"),
        )
        self.ledger = ProviderBudgetLedger(
            path,
            {
                "id": "synthetic-provider-recording.v1",
                "max_calls": 32,
                "max_input_tokens": 2_000_000,
                "max_output_tokens": 131072,
                "max_cost_microusd": 0,
                "routes": {
                    self.route.fingerprint: {"kind": "local", "input_rate": 0, "output_rate": 0}
                },
            },
        )

    def __call__(self, stage, schema, cancelled):
        return BoundProviderCall(
            self.route,
            transport=self,
            counter=lambda body: len(canonical(body)),
            stage=stage,
            cancelled=cancelled,
            clock=time.monotonic,
            deadline=time.monotonic() + 10,
            output_tokens=4096,
            data_scope=("current",),
            schema=schema,
        )

    def cancel(self):
        self.source.cancel()

    def invoke(self, body, cancelled, *, stage, data_scope, input_tokens, timeout_seconds):
        receipt = self.ledger.reserve(
            route=self.route.fingerprint,
            stage=stage,
            request_sha256=sha256(canonical(body).encode()).hexdigest(),
            input_tokens=input_tokens,
            output_tokens=body["max_tokens"],
        )
        status = "failed"
        try:
            raw = self.source(
                stage, body["messages"], body["response_format"]["json_schema"]["schema"], cancelled
            )
            status = "completed"
            return {
                "text": canonical(raw),
                "tool_calls": [],
                "finish_reason": "stop",
                "usage": None,
            }
        finally:
            self.ledger.finish(receipt, status=status, transmission="not_sent", usage=None)
