"""The same versioned Port recordings are consumed by the TS carrier trial."""

import json
from pathlib import Path

import pytest
from test_harness_port import Guard, Journal, Model
from virtual_companion.cognition.harness import HarnessLimits, run_harness

FIXTURE = json.loads(
    (Path(__file__).resolve().parent / "fixtures/harness-carrier-cases.json").read_text(
        encoding="utf-8"
    )
)


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda c: c["id"])
def test_shared_carrier_recording(case):
    model = Model(case["responses"])
    dispatched = []

    class BoundGuard(Guard):
        def checkpoint(self):
            if case.get("cancel_after_model") and model.requests:
                raise InterruptedError("synthetic cancellation")

    class BoundJournal(Journal):
        def save(self, value):
            if case.get("checkpoint_failure"):
                raise OSError("synthetic disk failure")
            super().save(value)

    def dispatch(call):
        dispatched.append(call)
        return case["dispatch"]

    try:
        result = run_harness(
            binding=FIXTURE["binding"],
            messages=FIXTURE["messages"],
            tools=FIXTURE["tools"],
            model=model,
            journal=BoundJournal(),
            guard=BoundGuard(),
            dispatch=dispatch,
            limits=HarnessLimits(4, 16000, 256),
        )
        outcome = {"status": result["status"], "text": result["text"]}
    except (InterruptedError, OSError):
        outcome = {"status": "interrupted", "text": ""}
    outcome.update(model_calls=len(model.requests), dispatches=len(dispatched))
    assert outcome == case["expected"]
