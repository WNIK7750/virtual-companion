"""Default wire must describe its actual memory action, not infer it from emptiness.

The real C-forget witness committed a revoke and then said no record existed.
Use the real current SQLite store, source worker, admission and approved delivery
helpers; the model only records decisions. No mirror receipt is fabricated.
"""

import json
from hashlib import sha256
from uuid import uuid4

import pytest
from test_default_cross_turn_memory import (
    CORRECT,
    FORGET,
    ORIGINAL,
    QUERY,
    Scenario,
    build,
    turn,
)
from test_default_text_delivery import wait_for
from unit_review_recording import bound_unit_packets as bound_unit_packets
from unit_review_recording import recorded_review_packet
from virtual_companion.boundary.contracts import canonical
from virtual_companion.cognition.stream_unit_review import semantic_review_view


class CaptureAdmission:
    """Observe the real admitted payload without changing checks or ownership."""

    def __init__(self, original):
        self.original, self.contexts = original, []

    def __call__(self, *args, **kwargs):
        request = self.original(*args, **kwargs)
        self.contexts.append(request.context)
        return request

    def __getattr__(self, name):
        return getattr(self.original, name)


def observe(runtime):
    capture = CaptureAdmission(runtime.admit)
    runtime.admit = capture
    return capture


@pytest.mark.parametrize("command,action,repeat,status", [
    (FORGET, "forget", False, "committed"),
    (CORRECT, "correct", False, "committed"),
    (CORRECT, "correct", True, "already_applied"),
])
def test_default_final_request_has_bound_current_command_receipt(
    tmp_path, command, action, repeat, status,
):
    model = Scenario()
    memory, mirror, runtime, session = build(tmp_path, model)
    capture = observe(runtime)
    try:
        turn(runtime, session, ORIGINAL)
        if repeat:
            turn(runtime, session, command)
        event = turn(runtime, session, command)
        context = capture.contexts[-1]
        payload = json.loads(context.payload)
        with memory.store.transaction(write=False) as db:
            row = db.execute(
                "SELECT payload,receipt FROM memory_management_operations "
                "WHERE companion=? AND json_extract(payload,'$.source_id')=?",
                (memory.companion, event["basis"]["input_id"]),
            ).fetchone()
        operation, receipt = json.loads(row["payload"]), json.loads(row["receipt"])
        assert receipt["status"] == status and receipt["action"] == action
        assert operation["source_id"] == event["basis"]["input_id"]
        projected = payload.get("current_operation")
        assert projected is not None, "empty post-action memory is not the command outcome"
        assert projected == {
            "protocol": "memory-command-outcome.v1",
            "source_id": operation["source_id"],
            "source_sha256": operation["source_sha256"],
            "operation_id": receipt["operation_id"],
            "action": action,
            "status": status,
            "receipt_sha256": sha256(canonical(receipt).encode()).hexdigest(),
        }
        # The provenance reference contains no removed content or replacement.
        assert ORIGINAL not in canonical(projected)
        if action == "correct":
            assert operation["replacement"] not in canonical(projected)
        assert event["readset"].get("current_operation") == projected
        assert event["readset"]["protocol"] == "history-readset.v2"
        review_messages = [messages for messages in model.calls
                           if "你只审查候选文字" in messages[0]["content"]][-1]
        reviewed = json.loads(review_messages[-1]["content"])
        sealed = recorded_review_packet(review_messages)
        assert semantic_review_view(sealed) == reviewed
        assert sealed["context"] == context.payload
        assert json.loads(sealed["context"]) == payload
        assert reviewed["context"]["current_operation"] == projected
        assert event["readset"]["context_sha256"] == sha256(context.payload.encode()).hexdigest()
        final_wire = model.primary[-1]
        evidence_wire = "\n".join(message["content"] for message in final_wire[1:-1])
        assert receipt["operation_id"] not in evidence_wire
        if action == "forget":
            assert ORIGINAL not in evidence_wire
            assert payload["memories"] == []
        assert final_wire[-1]["content"] == command  # Current request stays intact.
    finally:
        memory.close()
        mirror.close()


def test_completed_old_command_does_not_become_next_turn_or_other_companion_result(tmp_path):
    model = Scenario()
    memory, mirror, runtime, session = build(tmp_path, model)
    capture = observe(runtime)
    try:
        turn(runtime, session, ORIGINAL)
        turn(runtime, session, FORGET)
        assert json.loads(capture.contexts[-1].payload).get("current_operation")
        turn(runtime, session, QUERY)
        assert json.loads(capture.contexts[-1].payload).get("current_operation") is None
    finally:
        memory.close()
        mirror.close()
    other, other_mirror, other_runtime, other_session = build(
        tmp_path, model, companion=str(uuid4()))
    other_capture = observe(other_runtime)
    try:
        turn(other_runtime, other_session, QUERY)
        assert json.loads(other_capture.contexts[-1].payload).get("current_operation") is None
    finally:
        other.close()
        other_mirror.close()


def test_unresolved_command_cannot_supply_success_result_or_complete_reply(tmp_path):
    class Unresolved(Scenario):
        def invoke(self, stage, messages, schema, cancelled):
            result = super().invoke(stage, messages, schema, cancelled)
            if stage == "memory_command_support":
                return {**result, "verdict": "unknown"}
            return result

    model = Unresolved()
    memory, mirror, runtime, session = build(tmp_path, model)
    capture = observe(runtime)
    try:
        turn(runtime, session, ORIGINAL)
        capture.contexts.clear()
        session.send(FORGET)
        wait_for(lambda: not session.snapshot()["busy"])
        assert session.snapshot()["attempt"]["status"] == "failed"
        assert memory.store.memory_barrier(memory.companion)
        assert all(json.loads(context.payload).get("current_operation") is None
                   for context in capture.contexts)
        with memory.store.transaction(write=False) as db:
            count = db.execute("SELECT count(*) FROM memory_management_operations").fetchone()[0]
            assert count == 0
    finally:
        memory.close()
        mirror.close()


def test_list_operation_is_not_projected_as_a_later_conversation_command(tmp_path):
    model = Scenario()
    memory, mirror, runtime, session = build(tmp_path, model)
    capture = observe(runtime)
    try:
        turn(runtime, session, ORIGINAL)
        item = memory.list_memories()["items"][0]
        result = memory.apply_action({
            "operation_id": str(uuid4()), "action": "forget",
            "record_id": item["record_id"], "expected_version": item["version"],
            "replacement": "",
        })
        assert result["status"] == "committed"
        turn(runtime, session, QUERY)
        assert json.loads(capture.contexts[-1].payload).get("current_operation") is None
    finally:
        memory.close()
        mirror.close()
