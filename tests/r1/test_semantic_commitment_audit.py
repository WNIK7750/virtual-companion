"""Recorded v23 meanings exercise the wire/repair contract, not LLM accuracy."""

import json
from copy import deepcopy

import pytest
from test_stream_unit_review import V21_APPROVE, invoke_protocol, personal_ref, protocol_packet
from unit_review_recording import bound_unit_packets as bound_unit_packets
from unit_review_recording import recorded_review_packet

QUIET_INPUT = "最近总觉得房间很安静，想和你聊聊这种感觉。"
QUIET_QUESTION = "你说最近总有这种感觉，是它一直待在那儿，还是某个时候特别明显？"
TICKET_INPUT = "刚整理完桌上的一堆旧票根，想聊聊看到它们的感觉。"
TICKET_QUESTION = "你说想聊看到它们的感觉，那这些票根里，哪一张最先让你停下来多看了一会儿？"
RAIN_REPLY = "雨大到某个程度，就会把人拽回某条熟悉的路上——大概是因为放学那条路走过太多次了。"


def opinion(packet, commitments=(), questions=()):
    return {**V21_APPROVE, "audit_version": "candidate-propositions.v3", "segments": [
        {"segment_id": row["segment_id"], "commitments": deepcopy(list(commitments)),
         "questions": list(questions)} for row in packet["review_segments"]]}


def commitment(kind, claim, support, sources=()):
    return {"kind": kind, "claim": claim, "support": support, "sources": list(sources)}


def test_open_question_retains_supported_anchor_and_unknown_without_asserting_it():
    packet = protocol_packet(QUIET_INPUT, QUIET_QUESTION)
    ref = personal_ref(packet["payload"]["personal_sources"][0], "最近总觉得房间很安静")
    raw = opinion(packet["payload"], [commitment(
        "assertion", "你说最近总有这种感觉", "supported", [ref])],
        ["是它一直待在那儿，还是某个时候特别明显？"])
    result = invoke_protocol(packet, raw)
    assert result["decision"] == "approve"
    assert result["propositions"][0]["commitments"] == raw["segments"][0]["commitments"]
    assert result["propositions"][0]["questions"] == raw["segments"][0]["questions"]


def test_question_presupposition_is_a_separate_commitment_not_an_unknown_exemption():
    packet = protocol_packet(TICKET_INPUT, TICKET_QUESTION)
    source = packet["payload"]["personal_sources"][0]
    raw = opinion(packet["payload"], [
        commitment("assertion", "你说想聊看到它们的感觉", "supported",
                   [personal_ref(source, "想聊聊看到它们的感觉")]),
        commitment("presupposition", "最先让你停下来多看了一会儿", "unsupported"),
    ], ["哪一张最先让你停下来多看了一会儿？"])
    result = invoke_protocol(packet, raw)
    assert result["model_opinion"] == V21_APPROVE
    assert result["decision"] == "reject" and not result["constraints_satisfied"]
    assert result["propositions"][0]["commitments"][1]["kind"] == "presupposition"


def test_each_frequency_and_cause_commitment_keeps_its_own_support_relation():
    packet = protocol_packet(candidate=RAIN_REPLY)
    source = packet["payload"]["personal_sources"][0]
    raw = opinion(packet["payload"], [
        commitment("assertion", "放学那条路", "supported",
                   [personal_ref(source, "小时候放学的路")]),
        commitment("assertion", "放学那条路走过太多次了", "unsupported"),
        commitment("assertion", "大概是因为放学那条路走过太多次了", "uncertain"),
    ])
    result = invoke_protocol(packet, raw)
    assert result["decision"] == "reject"
    assert [row["support"] for row in result["propositions"][0]["commitments"]] == [
        "supported", "unsupported", "uncertain"]
    assert result["model_opinion"] == V21_APPROVE


@pytest.mark.parametrize("field", ["claim", "questions"])
def test_semantic_diagnostic_cannot_add_text_absent_from_bound_segment(field):
    packet = protocol_packet(QUIET_INPUT, QUIET_QUESTION)
    raw = opinion(packet["payload"])
    if field == "claim":
        raw["segments"][0]["commitments"] = [commitment("assertion", "用户已经哭了", "unsupported")]
    else:
        raw["segments"][0]["questions"] = ["你已经哭了多久？"]
    with pytest.raises(ValueError, match="bound segment"):
        invoke_protocol(packet, raw)


def test_generic_observation_has_no_personal_commitment_and_no_forced_question():
    packet = protocol_packet("刚整理完一叠明信片。", "小小的纸片，也能留住一个时期的印记。")
    result = invoke_protocol(packet, opinion(packet["payload"]))
    assert result["decision"] == "approve"
    assert result["propositions"][0]["commitments"] == []
    assert result["propositions"][0]["questions"] == []


def test_empty_commitments_are_a_recorded_model_claim_not_a_host_semantic_proof():
    packet = protocol_packet(candidate=RAIN_REPLY)
    # Deliberately dishonest extraction. No deterministic host classifier is added.
    result = invoke_protocol(packet, opinion(packet["payload"]))
    assert result["decision"] == "approve"
    assert result["propositions"][0]["text"] == RAIN_REPLY


def test_v2_whole_segment_labels_are_not_silently_reinterpreted_as_v3():
    packet = protocol_packet()
    raw = {**V21_APPROVE, "audit_version": "candidate-propositions.v2", "segments": [
        {"segment_id": row["segment_id"], "mode": "nonpersonal", "support": "not_applicable",
         "sources": []} for row in packet["payload"]["review_segments"]]}
    with pytest.raises(ValueError):
        invoke_protocol(packet, raw)


@pytest.mark.parametrize("check,decision", [
    ("self_contained", "hold"), ("no_open_dependency", "hold"),
    ("constraints_satisfied", "reject"), ("consistent_with_displayed", "reject"),
])
def test_valid_conflicting_approval_only_normalizes_toward_stricter_decision(check, decision):
    # s08's complete v24 wire had approve/no_open_dependency=false. Binding is
    # valid, but this is neither permission to publish nor an untyped exception.
    packet = protocol_packet(TICKET_INPUT, "纸片也能留下一个时期的印记。")
    raw = opinion(packet["payload"])
    raw[check] = False
    result = invoke_protocol(packet, raw)
    assert result["decision"] == decision
    assert result["model_opinion"] == {**V21_APPROVE, check: False}
    assert result[check] is False


def test_negative_constraint_wins_over_incomplete_approval_without_fabricating_checks():
    packet = protocol_packet()
    raw = opinion(packet["payload"])
    raw.update(self_contained=False, constraints_satisfied=False)
    result = invoke_protocol(packet, raw)
    assert result["decision"] == "reject"
    assert not result["self_contained"] and not result["constraints_satisfied"]
    assert result["model_opinion"]["decision"] == "approve"


def test_commitment_feedback_reaches_only_one_repair_then_durable_ack(tmp_path):
    from test_default_text_delivery import setup, wait_for
    from test_text_provider_model import Counter
    from test_text_provider_model import model as provider_fixture
    from virtual_companion.providers.text_deltas import UnapprovedTextDelta
    from virtual_companion.text.provider_model import ProviderModel

    safe = "一阵雨也能让很久以前的一条路重新鲜明起来。"

    class RecordedTransport:
        def __init__(self):
            self.calls = []

        def invoke(self, body, cancelled, **options):
            self.calls.append((deepcopy(body), options))
            messages = body["messages"]
            if options["stage"] == "stream_unit_review":
                packet = recorded_review_packet(messages)
                rows = [] if packet["candidate_prefix"] == safe else [commitment(
                    "assertion", "放学那条路走过太多次了", "unsupported")]
                response = opinion(packet, rows)
            else:
                response = {"action": "answer",
                            "text": safe if options["stage"] == "demand_repair" else RAIN_REPLY}
            raw = json.dumps(response)
            if "on_content" in options:
                options["on_content"](UnapprovedTextDelta(1, raw))
            return {"text": raw, "tool_calls": [], "finish_reason": "stop",
                    "usage": {"prompt_tokens": 17, "completion_tokens": 5, "total_tokens": 22}}

        def cancel(self):
            pass

    route = provider_fixture()[0].route
    transport = RecordedTransport()
    model = ProviderModel(route, counter=Counter(), transport=transport, output_tokens=512)

    runtime, session, memory, mirror, model = setup(tmp_path, model)
    try:
        session.send("今天雨好大，我忽然想起小时候放学的路。")
        event = wait_for(lambda: runtime.poll()["event"])
        assert len(transport.calls) == 4
        assert [options["stage"] for _, options in transport.calls] == [
            "demand_control", "stream_unit_review", "demand_repair", "stream_unit_review"]
        assert [body["max_tokens"] for body, _ in transport.calls] == [512, 1536, 512, 1536]
        assert model.output_tokens == 512
        repair = transport.calls[2][0]["messages"][-2]["content"]
        feedback = json.loads(repair[repair.index("{"):])["review"]
        assert feedback["decision"] == "reject" and feedback["model_opinion"] == V21_APPROVE
        assert feedback["propositions"][0]["commitments"][0]["claim"] == "放学那条路走过太多次了"
        assert event["display_text"] == safe
        runtime.ack({"sequence": event["sequence"], "fingerprint": event["fingerprint"],
                     "host_receipt": "synthetic-commitment-repair-visible"})
        wait_for(lambda: not session.snapshot()["busy"])
        assert session.snapshot()["attempt"]["status"] == "completed"
        state = memory.store.stream_state(memory.companion, runtime.claim.input_id)
        assert state["confirmed_text"] == safe
        assert mirror.history()["messages"][-1]["content"] == safe
    finally:
        memory.close()
        mirror.close()
