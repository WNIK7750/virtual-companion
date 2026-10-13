"""A captured pending-task opinion cannot stand in for an explicit rejection."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
from test_default_text_delivery import setup
from test_unit_scope_review_v4 import (
    VERDICT_AUDIT,
    bind_v8,
    captured_envelope,
    finish,
    supplied_v8,
)
from test_unit_scope_review_v4 import (
    sealed_capture as sealed_capture,
)
from virtual_companion.boundary.contracts import canonical
from virtual_companion.boundary.presentation import digest
from virtual_companion.cognition.stream_unit_review import InvalidUnitVerdict
from virtual_companion.text.application import TextSession
from virtual_companion.text.default_runtime import DefaultTextRuntime
from virtual_companion.text.fake import FakeModel


def captured_task_hold():
    fixture = Path(__file__).parent / "fixtures" / "v35-preliminary-task-hold.json"
    return json.loads(fixture.read_text(encoding="utf-8-sig"))


def test_captured_preliminary_task_hold_is_not_a_bound_rejection():
    recorded = captured_task_hold()
    envelope, opinion = recorded["envelope"], recorded["opinion"]
    assert envelope["sha256"] == digest(envelope["payload"])
    original = deepcopy(opinion)
    with pytest.raises(InvalidUnitVerdict, match="Preliminary task uncertainty") as caught:
        bind_v8(envelope, opinion)
    assert caught.value.opinion == opinion == original
    # The prefix has an unsupported operation claim. Deferral cannot approve it.
    assert envelope["payload"]["candidate_prefix"] == "好，提醒关了。"


@pytest.mark.parametrize("change", [
    "final", "explicit-reject", "task-fail", "operation", "accuracy",
    "personal-unsupported", "personal-uncertain", "fit-incompatible", "fit-uncertain",
])
def test_deferral_does_not_reopen_final_or_adverse_task_opinions(change):
    recorded = captured_task_hold()
    envelope, opinion = recorded["envelope"], recorded["opinion"]
    if change == "final":
        envelope["payload"]["final"] = True
        envelope["sha256"] = digest(envelope["payload"])
    elif change == "explicit-reject":
        opinion["decision"] = "reject"
    elif change == "task-fail":
        opinion["checks"][3] = "fail"
    elif change in {"operation", "accuracy"}:
        opinion["failure"]["scope"] = change
    elif change.startswith("personal-"):
        opinion["units"][0]["personal_support"] = change.removeprefix("personal-")
    else:
        opinion["units"][0]["response_fit"] = change.removeprefix("fit-")
    verdict = bind_v8(envelope, opinion)
    assert verdict["decision"] == "reject"
    assert verdict["model_opinion"] == opinion
    assert verdict["constraints_satisfied"] is False


def test_deferred_task_opinion_still_requires_original_unit_and_source_binding():
    recorded = captured_task_hold()
    envelope, opinion = recorded["envelope"], recorded["opinion"]
    opinion["units"][0].update(personal_support="supported",
                                personal_source_ids=["foreign-source"])
    with pytest.raises(ValueError) as caught:
        bind_v8(envelope, opinion)
    assert not isinstance(caught.value, InvalidUnitVerdict)


@pytest.mark.parametrize("final_reject", [False, True])
def test_original_stream_reaches_final_review_before_any_repair(
    tmp_path, sealed_capture, final_reject,
):
    prefix = "好，提醒关了。" if final_reject else "说到这件事。"
    tail, replacement = "没有进去过，也可以对一个地方有所留恋。", "提醒要在界面里关闭。"
    shared = {"generations": 0, "closed": 0, "reviews": [], "repair": []}

    class PendingTaskModel(FakeModel):
        def review_peer(self):
            return PendingTaskModel()

        def stream(self, messages, cancelled):
            if "你只审查候选文字" in messages[0]["content"]:
                packet = captured_envelope(messages, sealed_capture)["payload"]
                shared["reviews"].append(deepcopy(packet))
                if not packet["final"] and packet["candidate_prefix"] == prefix:
                    if final_reject:
                        # Exact captured opinion, including its actual unit identity.
                        opinion = captured_task_hold()["opinion"]
                    else:
                        opinion = supplied_v8(packet, decision="hold",
                            statuses=("pass", "pass", "pass", "uncertain"), failure={
                                "scope": "task", "segment_id": None,
                                "basis": "前缀尚未包含后续解释，完整任务未核定。"})
                elif final_reject and packet["candidate_prefix"] == prefix + tail:
                    opinion = supplied_v8(packet, decision="reject",
                        statuses=("pass", "pass", "pass", "fail"), failure={
                            "scope": "operation", "segment_id": "first",
                            "basis": "没有提醒设置操作结果，不能宣称已经关闭。"})
                else:
                    opinion = supplied_v8(packet)
                yield canonical(opinion)
                return
            shared["generations"] += 1
            repair = any(item["content"].startswith(runtime.repair_reply) for item in messages)
            shared["repair"].append(repair)
            try:
                if repair:
                    # A real final operation rejection permits the existing one repair.
                    assert [packet["final"] for packet in shared["reviews"]] == [False, True]
                    yield canonical({"action": "answer", "text": replacement})
                    return
                yield '{"action":"answer","text":"' + json.dumps(prefix)[1:-1]
                assert len(shared["reviews"]) == 1
                assert runtime.poll()["event"] is None
                assert runtime.store.segments(runtime.companion, runtime.claim.input_id) == []
                yield json.dumps(tail)[1:-1] + '"}'
            finally:
                shared["closed"] += 1

    model = PendingTaskModel()
    _, _, memory, mirror, _ = setup(tmp_path, model)
    runtime = DefaultTextRuntime(memory, mirror, model, binding={"test": "pending-task"},
                                 prompt_revision="short-dialogue.v35")
    session = TextSession(model, mirror, {}, prepare_turn=runtime.prepare_turn,
                          before_start=runtime.before_start)
    try:
        session.send("先把提醒关了，你接着说说，为什么没进去过也会舍不得？")
        event = finish(runtime, session)
        expected = replacement if final_reject else prefix + tail
        assert event["presentation"]["candidate"]["semantic_text"] == expected
        assert mirror.history()["messages"][-1]["content"] == expected
        assert shared["generations"] == shared["closed"] == 1 + int(final_reject)
        assert shared["repair"] == ([False, True] if final_reject else [False])
        assert len(shared["reviews"]) == 2 + int(final_reject)
        assert [packet["final"] for packet in shared["reviews"][:2]] == [False, True]
        if final_reject:
            assert shared["reviews"][-1]["candidate_prefix"] == replacement
        assert shared["reviews"][1]["candidate_prefix"] == prefix + tail
        assert all(packet["displayed_prefix"] == "" for packet in shared["reviews"])
        assert all(packet["audit_version"] == VERDICT_AUDIT for packet in shared["reviews"])
        assert len(runtime.store.segments(runtime.companion, runtime.claim.input_id)) == 1
    finally:
        if session.snapshot()["busy"]:
            session.stop()
        memory.close()
        mirror.close()
