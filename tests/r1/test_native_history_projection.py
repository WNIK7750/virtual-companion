"""A diagnostic wire projection keeps admission and received evidence unchanged."""
import json
from dataclasses import replace

import pytest
from test_review_carrier_selection import factory as factory
from test_review_carrier_selection import resource
from virtual_companion.boundary.contracts import canonical
from virtual_companion.cognition.short_dialogue import ShortContext, ShortRequest
from virtual_companion.text.native_history import native_history_request

COMPANION = "00000000-0000-0000-0000-000000000001"


def request():
    data = {
        "history": [
            {"id": "u1", "role": "user", "text": "我今天没买票。"},
            {"id": "a1", "role": "assistant", "text": "那先别急着出发。"},
        ],
        "memory_status": "current", "memories": [], "exchanges": [],
        "received_output": [{"input_id": "other", "sequence": 1, "text": "未见前缀"}],
        "receiving_context": {"protocol": "receiving-context.v1", "source": "receiving-client",
                              "scope": "this_client", "observation": "at_input_submission",
                              "quiet": True},
    }
    context = ShortContext(COMPANION, "current", 7, canonical(data))
    return ShortRequest("还是先回家吧。", context=context)


def test_native_projection_uses_roles_once_and_valid_json_assistant_history():
    source = request()
    frozen = source.context.payload
    result = native_history_request(source, 192, "只返回JSON。", sparse_evidence=True)
    messages = result["messages"]
    assert messages[-1] == {"role": "user", "content": source.user_text}
    assert messages[-3] == {"role": "user", "content": "我今天没买票。"}
    assert messages[-2]["role"] == "assistant"
    assert json.loads(messages[-2]["content"]) == {
        "action": "answer", "text": "那先别急着出发。"}
    serialized = canonical(messages)
    assert serialized.count("我今天没买票。") == 1
    assert serialized.count("那先别急着出发。") == 1
    assert source.context.payload == frozen
    assert result["tools"] == [] and result["max_output_tokens"] == 192


def test_received_output_stays_background_and_never_becomes_native_assistant_history():
    source = request()
    result = native_history_request(source, 192, "只返回JSON。", sparse_evidence=True)
    assistants = [m for m in result["messages"] if m["role"] == "assistant"]
    assert len(assistants) == 1
    assert "未见前缀" not in assistants[0]["content"]
    assert "未见前缀" in canonical(result["messages"])
    assert json.loads(source.context.payload)["receiving_context"]["quiet"] is True


def test_empty_history_preserves_original_body_exactly():
    source = ShortRequest("你好。", context=ShortContext(COMPANION, "current", 0,
        canonical({"history": [], "memory_status": "current", "memories": [], "exchanges": []})))
    assert native_history_request(source, 192, "只返回JSON。") == source.model_request(
        192, "只返回JSON。")


def test_ack_history_keeps_receipt_only_meaning():
    source = request()
    data = json.loads(source.context.payload)
    data["history"][1]["response_type"] = "ack"
    source = replace(source, context=replace(source.context, payload=canonical(data)))
    body = native_history_request(source, 192, "只返回JSON。", sparse_evidence=True)
    assert json.loads(body["messages"][-2]["content"])["action"] == "ack"


def test_explicit_projection_has_distinct_runtime_resource(factory):
    create, _, initial, *_ = factory
    runtime = create(history_projection="native-json.v1")
    assert runtime.control == initial.control
    assert runtime.review_audit_version == initial.review_audit_version
    assert resource(runtime) != resource(initial)
    with pytest.raises(ValueError, match="history projection"):
        create(history_projection="unknown")
