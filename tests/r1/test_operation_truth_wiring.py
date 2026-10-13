"""Operation truth is shared by actual generation/review wires, without a new call."""

import json
from dataclasses import replace

import pytest
from test_current_command_outcome import observe
from test_default_cross_turn_memory import (
    CORRECT,
    FORGET,
    ORIGINAL,
    Scenario,
    build,
    turn,
)
from test_default_text_delivery import setup
from unit_review_recording import bound_unit_packets as bound_unit_packets
from unit_review_recording import recorded_review_packet
from virtual_companion.boundary.contracts import canonical
from virtual_companion.cognition.short_dialogue import ShortRequest
from virtual_companion.cognition.stream_unit_review import semantic_review_view
from virtual_companion.text import default_runtime


def assert_shared(model, context):
    constraint = ShortRequest("probe", context=context).operation_constraints()
    assert "只有宣称持久保存、更正或遗忘成功时" not in constraint
    assert "操作是否已执行、是否成功或当前状态" in constraint
    assert "过去助手原话只证明当时的表达" in constraint
    review = model.calls[-1]
    generation = model.calls[-2]
    assert constraint in generation[0]["content"]
    assert constraint in review[0]["content"]
    packet = recorded_review_packet(review)
    assert packet["context"] == context.payload
    assert json.loads(review[-1]["content"]) == semantic_review_view(packet)
    return constraint


def test_effective_operation_policy_changes_the_runtime_resource_identity(tmp_path, monkeypatch):
    (tmp_path / 'before').mkdir()
    (tmp_path / 'after').mkdir()
    first = setup(tmp_path / 'before')
    monkeypatch.setattr(default_runtime, 'OPERATION_TRUTH_SYSTEM',
                        default_runtime.OPERATION_TRUTH_SYSTEM + '合成身份变化。')
    second = setup(tmp_path / 'after')
    try:
        assert dict(first[0].config.resources)['model'] != dict(second[0].config.resources)['model']
    finally:
        for runtime, text, memory, mirror, _ in (first, second):
            runtime.stop()
            text.stop()
            if text.worker is not None:
                text.worker.join(2)
            memory.close()
            mirror.close()


def test_direct_share_has_same_unconfirmed_operation_constraint_in_both_calls(tmp_path):
    runtime, session, memory, mirror, model = setup(tmp_path)
    captured = observe(runtime)
    try:
        turn(runtime, session, "前天我提到的是《城南旧事》，作者是林海音。")
        assert len(model.calls) == 2
        context = captured.contexts[-1]
        assert json.loads(context.payload)["memory_status"] == "pending_source"
        constraint = assert_shared(model, context)
        # Source completion with zero writes is still no save evidence. Neither
        # older facts nor another source's successful operation can stand in.
        for memories in ([], [{"source_id": "other-source", "content": "old fact"}]):
            no_save = replace(context, payload=canonical({
                "memory_status": "current", "memories": memories,
                "history": [], "exchanges": [],
            }))
            assert ShortRequest("probe", context=no_save).operation_constraints() == constraint
    finally:
        memory.close()
        mirror.close()


@pytest.mark.parametrize("command", [None, CORRECT, FORGET])
def test_real_saved_fact_and_commands_share_scoped_success_with_review(tmp_path, command):
    model = Scenario()
    memory, mirror, runtime, session = build(tmp_path, model)
    captured = observe(runtime)
    try:
        turn(runtime, session, ORIGINAL)
        if command:
            turn(runtime, session, command)
        context = captured.contexts[-1]
        data = json.loads(context.payload)
        if command:
            assert data["current_operation"]["status"] == "committed"
        else:
            assert any(row["source_id"] == context.input_id for row in data["memories"])
        success = assert_shared(model, context)
        # A later turn cannot reuse this turn's receipt or fact as its own save.
        data.pop("current_operation", None)
        later = replace(context, input_id="later-input", payload=canonical(data))
        assert success != ShortRequest("probe", context=later).operation_constraints()
    finally:
        memory.close()
        mirror.close()
