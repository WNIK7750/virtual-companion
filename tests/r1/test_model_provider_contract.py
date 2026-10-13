from dataclasses import replace

import pytest
from virtual_companion.providers.contracts import ProviderCapabilities, ProviderRoute
from virtual_companion.providers.httpx_transport import _stream_failure_code
from virtual_companion.providers.openai_compatible import (
    StreamFailure,
    build_request,
    decode_stream,
)


def route(**changes):
    fields = dict(
        route_id="local",
        revision=1,
        kind="local",
        endpoint="http://127.0.0.1:18083/v1/chat/completions",
        model="frozen-local-model",
        auth="none",
        secret_ref=None,
        explicit=False,
        data_scope=("current",),
        capabilities=ProviderCapabilities(True, True, True, 32000, "max_tokens"),
    )
    return ProviderRoute(**{**fields, **changes})


def test_local_no_key_and_exact_messages_no_hidden_system_or_extra_override():
    messages = [{"role": "system", "content": "neutral"}, {"role": "user", "content": "hello"}]
    body = build_request(route(), messages, output_tokens=256)
    assert body["messages"] == messages
    assert body["max_tokens"] == 256
    assert body["stream_options"] == {"include_usage": True}
    assert "api_key" not in str(body)
    messages[0]["content"] = "changed"
    assert body["messages"][0]["content"] == "neutral"


def test_frozen_sampling_is_part_of_route_and_actual_request():
    from virtual_companion.providers.contracts import ProviderSampling

    configured = route(sampling=ProviderSampling(0.2, 0.8, 20260907))
    body = build_request(configured, [{"role": "user", "content": "x"}], output_tokens=256)
    assert (body["temperature"], body["top_p"], body["seed"]) == (0.2, 0.8, 20260907)
    assert configured.fingerprint != route(sampling=ProviderSampling(0.2, 0.8, 1)).fingerprint
    with pytest.raises(ValueError):
        ProviderSampling(float("nan"), 0.8, 1)
    with pytest.raises(ValueError):
        ProviderSampling(0.2, 0.8, True)


@pytest.mark.parametrize(
    "changes",
    [
        {"kind": "api", "endpoint": "https://example.test/v1/chat/completions"},
        {"kind": "local", "endpoint": "https://example.test/v1/chat/completions"},
        {"endpoint": "http://127.0.0.1/v1/chat/completions?key=secret"},
        {"endpoint": "http://user:password@127.0.0.1/v1/chat/completions"},
        {"auth": "bearer", "secret_ref": None},
        {"explicit": "yes"},
    ],
)
def test_implicit_remote_missing_key_or_unsafe_endpoint_rejected(changes):
    with pytest.raises(ValueError):
        route(**changes)


def test_explicit_api_route_keeps_secret_reference_and_declared_output_field():
    configured = route(
        kind="api",
        endpoint="https://example.test/v1/chat/completions",
        auth="bearer",
        secret_ref="dpapi:0123456789abcdef0123456789abcdef",  # noqa: S106 - synthetic reference
        explicit=True,
        capabilities=ProviderCapabilities(True, False, False, 8192, "max_completion_tokens"),
    )
    body = build_request(configured, [{"role": "user", "content": "synthetic"}], output_tokens=128)
    assert body["model"] == "frozen-local-model"
    assert body["max_completion_tokens"] == 128
    assert "max_tokens" not in body
    with pytest.raises(ValueError, match="structured"):
        build_request(
            configured,
            [{"role": "user", "content": "x"}],
            output_tokens=128,
            schema={"type": "object"},
        )
    with pytest.raises(ValueError, match="tools"):
        build_request(
            configured,
            [{"role": "user", "content": "x"}],
            output_tokens=128,
            tools=[{"name": "read", "parameters": {"type": "object"}}],
        )
    assert configured.fingerprint != replace(configured, revision=2).fingerprint


def test_tool_messages_and_nested_arguments_are_preserved_on_wire():
    messages = [
        {"role": "user", "content": "x"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "a", "name": "read", "arguments": '{"body":{"value":1}}'}],
        },
        {"role": "tool", "tool_call_id": "a", "content": "evidence"},
    ]
    body = build_request(
        route(),
        messages,
        output_tokens=256,
        tools=[{"name": "read", "parameters": {"type": "object"}}],
    )
    assert body["messages"][1]["tool_calls"][0]["function"]["arguments"] == '{"body":{"value":1}}'
    assert body["messages"][2]["tool_call_id"] == "a"


def event(value):
    import json

    return ("data: " + json.dumps(value, ensure_ascii=False) + "\n\n").encode()


def test_stream_usage_unknown_is_not_zero_and_unicode_chunks_are_exact():
    raw = event({"choices": [{"index": 0, "delta": {"content": "你好🌿"}, "finish_reason": None}]})
    raw += event({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    raw += b"data: [DONE]\n\n"
    result = decode_stream(raw)
    assert result["text"] == "你好🌿"
    assert result["usage"] is None
    usage = {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16}
    raw = raw.replace(b"data: [DONE]", event({"choices": [], "usage": usage}) + b"data: [DONE]")
    assert decode_stream(raw)["usage"] == usage


@pytest.mark.parametrize(
    "raw",
    [
        b'data: {"choices":[]}\n\n',
        b'data: {"choices":[], "usage":{"prompt_tokens":false,"completion_tokens":0,'
        b'"total_tokens":0}}\n\ndata: [DONE]\n\n',
        b'data: {"choices":[{"index":0,"delta":{"content":"partial"},'
        b'"finish_reason":"length"}]}\n\ndata: [DONE]\n\n',
        b'data: {"choices":[{"index":0,"delta":{"refusal":"no"},'
        b'"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n',
    ],
)
def test_incomplete_refused_or_invalid_stream_fails_closed(raw):
    with pytest.raises(ValueError):
        decode_stream(raw)


def test_stream_failure_preserves_safe_terminal_reason_without_response_content():
    limited = event(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "private partial"},
                    "finish_reason": "length",
                }
            ]
        }
    ) + b"data: [DONE]\n\n"
    with pytest.raises(StreamFailure) as error:
        decode_stream(limited)
    assert error.value.reason == "output_limit_reached"
    assert "private partial" not in str(error.value)

    incomplete = event(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "private partial"},
                    "finish_reason": "stop",
                }
            ]
        }
    )
    with pytest.raises(StreamFailure) as error:
        decode_stream(incomplete)
    assert error.value.reason == "stream_incomplete"
    assert "private partial" not in str(error.value)

    assert _stream_failure_code("output_limit_reached") == "provider_output_limit_reached"
    assert _stream_failure_code("invalid_stream") == "provider_protocol_rejected"
