"""Route structured functional calls through the adopted Harness without a second owner."""

import json
import threading
from hashlib import sha256

from virtual_companion.boundary.contracts import canonical

from .harness import HarnessLimits, run_harness


class StructuredHarnessInvoker:
    def __init__(self, models, store, clock):
        self.models, self.store, self.clock = models, store, clock
        self.claim, self.guard, self.ordinal = None, None, 0
        self.candidate_observations = []
        self._current, self._lock = None, threading.Lock()

    def bind_turn(self, claim, guard):
        with self._lock:
            if self._current is not None:
                raise RuntimeError("Structured Harness call still active")
            self.claim, self.guard, self.ordinal = claim, guard, 0
            self.candidate_observations = []

    def validated_plan(self, plan):
        """Parser/schema completion, distinct from semantic review and UI display."""
        self.candidate_observations.append(
            {
                "monotonic_us": self.clock(),
                "plan_sha256": sha256(canonical(plan).encode()).hexdigest(),
            }
        )

    def cancel(self):
        with self._lock:
            if self._current is not None:
                self._current.cancel()

    def __call__(self, stage, messages, schema, cancelled):
        if self.claim is None or self.guard is None:
            raise RuntimeError("Structured Harness requires the owning whole-turn guard")
        self.guard.checkpoint()
        self.ordinal += 1
        model = self.models(stage, schema, cancelled)
        journal = self.store.harness_journal(
            self.claim, f"{stage}-{self.ordinal}", clock=self.clock
        )
        with self._lock:
            if self._current is not None:
                raise RuntimeError("Structured Harness model already active")
            self._current = model
        try:
            result = run_harness(
                binding=journal.binding,
                messages=messages,
                tools=[],
                model=model,
                dispatch=self._no_external_tools,
                journal=journal,
                guard=self.guard,
                limits=HarnessLimits(1, model.context_tokens, model.output_tokens),
            )

            def unique(pairs):
                value = {}
                for key, item in pairs:
                    if key in value:
                        raise ValueError("Duplicate structured model field")
                    value[key] = item
                return value

            raw = json.loads(result["text"], object_pairs_hook=unique)
            canonical(raw)
            if not isinstance(raw, dict):
                raise ValueError("Structured structured model object required")
            return raw
        finally:
            with self._lock:
                self._current = None

    @staticmethod
    def _no_external_tools(_):
        raise RuntimeError("External tools are not enabled for structured functional calls")
