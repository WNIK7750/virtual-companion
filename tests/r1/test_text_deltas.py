import pytest
from test_model_provider_contract import event
from virtual_companion.providers.openai_compatible import decode_stream
from virtual_companion.providers.text_deltas import TextDeltas


def test_every_byte_boundary_preserves_utf8_and_delta_order():
    raw = event({"choices": [{"index": 0, "delta": {"content": "你好🌿"}}]})
    raw += event({"choices": [{"index": 0, "delta": {"content": "继续。"},
                               "finish_reason": "stop"}]}) + b"data: [DONE]\n\n"
    raw = raw.replace(b"\n", b"\r\n")
    for split in range(len(raw) + 1):
        decoder = TextDeltas()
        deltas = (*decoder.feed(raw[:split]), *decoder.feed(raw[split:]))
        assert [d.sequence for d in deltas] == [1, 2]
        assert all(d.approved is False for d in deltas)
        assert "".join(d.text for d in deltas) == decode_stream(raw)["text"]


@pytest.mark.parametrize("delta", [{"refusal": "no"}, {"tool_calls": [{"index": 0}]},
                                  {"content": 3}])
def test_invalid_or_nontext_event_does_not_emit(delta):
    with pytest.raises(ValueError):
        TextDeltas().feed(event({"choices": [{"index": 0, "delta": delta}]}))


def test_unfinished_content_is_not_a_successful_terminal():
    raw = event({"choices": [{"index": 0, "delta": {"content": "未完"}}]})
    assert TextDeltas().feed(raw)[0].text == "未完"
    with pytest.raises(ValueError):
        decode_stream(raw)


def test_late_content_cannot_be_emitted_after_finish():
    decoder = TextDeltas()
    decoder.feed(event({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}))
    with pytest.raises(ValueError):
        decoder.feed(event({"choices": [{"index": 0, "delta": {"content": "迟到"}}]}))
