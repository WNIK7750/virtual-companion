"""Read the sealed host sidecar for existing semantic recording suppliers.

The model's actual messages remain the compact wire in each recording. Old
fixture labels and assertions can inspect their original host packet, only after
checking that its semantic projection exactly equals the sent model view. This
does not infer meanings, invent binding fields or relax the product audit.
"""

import json
from copy import deepcopy
from threading import RLock

import pytest
from virtual_companion.boundary.contracts import canonical
from virtual_companion.boundary.presentation import digest
from virtual_companion.cognition.stream_unit_review import (
    StructuredUnitAssessor,
    semantic_review_view,
)

_packets = {}
_lock = RLock()


@pytest.fixture(autouse=True)
def bound_unit_packets(monkeypatch):
    original = StructuredUnitAssessor.__call__
    with _lock:
        _packets.clear()

    def assess(assessor, envelope):
        if isinstance(envelope, dict) and digest(envelope.get("payload")) == envelope.get("sha256"):
            packet = json.loads(canonical(envelope["payload"]))
            view = semantic_review_view(packet)
            with _lock:
                _packets[canonical(view)] = packet
        return original(assessor, envelope)

    monkeypatch.setattr(StructuredUnitAssessor, "__call__", assess)
    yield
    with _lock:
        _packets.clear()


def recorded_review_packet(messages):
    view = json.loads(messages[-1]["content"])
    if view.get("view_version") != "unit-review-semantics.v1":
        return view  # Explicit legacy-schema negative fixtures, never approval.
    with _lock:
        packet = deepcopy(_packets.get(canonical(view)))
    assert packet is not None, "Actual original host review packet was not captured"
    assert semantic_review_view(packet) == view
    return packet
