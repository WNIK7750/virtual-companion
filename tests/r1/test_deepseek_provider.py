"""Offline contracts for the explicit, non-thinking DeepSeek Flash wire profile."""

import json
import threading
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256

import pytest
from virtual_companion.boundary.contracts import canonical
from virtual_companion.providers.bound_call import BoundProviderCall
from virtual_companion.providers.contracts import (
    ProviderCapabilities,
    ProviderRoute,
    ProviderSampling,
)
from virtual_companion.providers.httpx_transport import HttpxProviderTransport
from virtual_companion.providers.openai_compatible import build_request

ENDPOINT = "https://api.deepseek.com/chat/completions"
SCHEMA = {
    "type": "object", "properties": {"answer": {"type": "string"}},
    "required": ["answer"], "additionalProperties": False,
}
MESSAGES = [{"role": "system", "content": "Neutral companion."},
            {"role": "user", "content": "Synthetic request."}]


def deepseek_route(**changes):
    fields = dict(
        route_id="deepseek-pilot", revision=1, kind="api", endpoint=ENDPOINT,
        model="deepseek-flash", auth="bearer",
        secret_ref="dpapi:0123456789abcdef0123456789abcdef",  # noqa: S106 - fixture
        explicit=True, data_scope=("current",),
        capabilities=ProviderCapabilities(True, False, True, 32000, "max_tokens"),
        wire_profile="deepseek-flash.v1",
    )
    return ProviderRoute(**{**fields, **changes})


def local_route():
    return ProviderRoute(
        "local", 1, "local", "http://127.0.0.1:18083/v1/chat/completions",
        "frozen-local-model", "none", None, False, ("current",),
        ProviderCapabilities(True, True, True, 32000, "max_tokens"),
    )


def test_legacy_profile_keeps_preexisting_frozen_fingerprint_and_wire():
    route = local_route()
    assert route.fingerprint == "8b305afc237971aeb0d9fe1816a5d9baacdb8d6e34a4983be4ae503b70310cde"
    body = build_request(route, MESSAGES, output_tokens=100, schema=SCHEMA)
    assert body["messages"] == MESSAGES
    assert body["response_format"]["type"] == "json_schema"
    assert "thinking" not in body
    assert sha256(canonical(body).encode()).hexdigest() == (
        "2af7a570da1ad6a512c971ca258dc6bfe929f40ee0399fc7666d4c5bb78bff36"
    )


def test_v1_and_no_profile_api_keep_original_body_and_route_hash_with_stage_metadata():
    for route, fingerprint, body_sha in (
        (deepseek_route(), "0fb203c31e924aa922d4a2c4588d9157b4d6f2128c391c9cc6f2d47fa3c78ca3",
         "4cce46f132d9ac945e7d05be2efd9a11ff2ce354d0b200cecc3dd230102e3572"),
        (replace(deepseek_route(), wire_profile=None),
         "f02382d0ca43c23c239862ce6d7ab23f7ad3fc7461c891d4554c71ae820c70e8",
         "59314883d36a81a09270aff1e88ab5a2956f7a0c7fb0d2fbbf09a1b69dd18704"),
    ):
        original = build_request(route, MESSAGES, output_tokens=100, schema=SCHEMA)
        assert route.fingerprint == fingerprint
        assert sha256(canonical(original).encode()).hexdigest() == body_sha
        for stage in (None, "demand_control", "stream_unit_review", "unregistered_old_stage"):
            assert build_request(route, MESSAGES, output_tokens=100, schema=SCHEMA,
                                 stage=stage) == original
        assert "temperature" not in original


def test_deepseek_profile_uses_json_object_and_explicit_schema_without_mutating_input():
    messages, schema = deepcopy(MESSAGES), deepcopy(SCHEMA)
    body = build_request(deepseek_route(), messages, output_tokens=100, schema=schema)
    assert body["thinking"] == {"type": "disabled"}
    assert body["response_format"] == {"type": "json_object"}
    assert body["max_tokens"] == 100
    assert not set(body).intersection({
        "seed", "top_p", "reasoning_effort", "max_completion_tokens",
    })
    instruction = body["messages"][0]
    assert instruction["role"] == "system"
    assert "JSON" in instruction["content"]
    assert json.loads(instruction["content"].split("\n", 1)[1]) == SCHEMA
    assert body["messages"][1:] == MESSAGES
    assert messages == MESSAGES and schema == SCHEMA
    assert deepseek_route().fingerprint != replace(deepseek_route(), wire_profile=None).fingerprint


@pytest.mark.parametrize("changes", [
    {"wire_profile": "deepseek-flash.v999"},
    {"sampling": ProviderSampling(0.2, 1, 1)},
    {"model": "some-other-model"},
    {"endpoint": "https://other.example/chat/completions"},
    {"capabilities": ProviderCapabilities(True, True, True, 32000, "max_tokens")},
    {"capabilities": ProviderCapabilities(True, False, True, 32000, "max_completion_tokens")},
])
def test_profile_rejects_unverified_route_or_sampling(changes):
    with pytest.raises(ValueError, match="profile"):
        deepseek_route(**changes)


def test_plain_text_and_unverified_tools_are_explicit():
    route = deepseek_route()
    body = build_request(route, MESSAGES, output_tokens=100)
    assert body["messages"] == MESSAGES
    assert body["thinking"] == {"type": "disabled"}
    assert "response_format" not in body
    with pytest.raises(ValueError, match="tools"):
        build_request(route, MESSAGES, output_tokens=100,
                      tools=[{"name": "read", "parameters": {"type": "object"}}])
    with pytest.raises(ValueError, match="object"):
        build_request(route, MESSAGES, output_tokens=100, schema={"type": "array"})


class NeverSecrets:
    def get(self, *args):
        pytest.fail("Rejected wire profile must not access credentials")


class NeverLedger:
    def reserve(self, **kwargs):
        pytest.fail("Rejected wire profile must not reserve or send")


@pytest.mark.parametrize("mutation", [
    "missing_thinking", "enabled_thinking", "extra_thinking_field", "reasoning_effort",
    "seed", "top_p", "json_schema", "missing_instruction", "removed_format", "tools",
])
def test_profile_wire_tampering_fails_before_credentials_budget_or_network(mutation):
    route = deepseek_route()
    body = build_request(route, MESSAGES, output_tokens=100, schema=SCHEMA)
    if mutation == "missing_thinking":
        body.pop("thinking")
    elif mutation == "enabled_thinking":
        body["thinking"] = {"type": "enabled"}
    elif mutation == "extra_thinking_field":
        body["thinking"]["reasoning_effort"] = "high"
    elif mutation in {"seed", "top_p", "reasoning_effort"}:
        body[mutation] = 1
    elif mutation == "json_schema":
        body["response_format"] = {"type": "json_schema"}
    elif mutation == "missing_instruction":
        body["messages"].pop(0)
    elif mutation == "removed_format":
        body.pop("response_format")
    elif mutation == "tools":
        body["tools"] = []
    transport = HttpxProviderTransport(route, secrets=NeverSecrets(), ledger=NeverLedger())
    with pytest.raises(ValueError):
        transport.invoke(body, threading.Event(), stage="reply", data_scope=("current",),
                         input_tokens=100, timeout_seconds=10)


def test_local_wire_still_rejects_provider_specific_thinking():
    route = local_route()
    body = build_request(route, MESSAGES, output_tokens=100)
    body["thinking"] = {"type": "disabled"}
    transport = HttpxProviderTransport(route, secrets=NeverSecrets(), ledger=NeverLedger())
    with pytest.raises(ValueError):
        transport.invoke(body, threading.Event(), stage="reply", data_scope=("current",),
                         input_tokens=100, timeout_seconds=10)


@pytest.mark.parametrize("wire_profile,temperature", [("deepseek-flash.v1", None),
                                                     ("deepseek-flash.focused.v2", 0)])
def test_bound_call_counts_and_sends_identical_profile_body_and_schema_is_host_checked(
        wire_profile, temperature):
    from virtual_companion.cognition.fact_model import FactModel

    class RecordingTransport:
        def invoke(self, body, cancelled, **kwargs):
            sent.append(body)
            return {
                "text": '{"answer":17}', "tool_calls": [], "finish_reason": "stop", "usage": None,
            }

    counted, sent = [], []

    def invoke(stage, messages, schema, cancelled):
        call = BoundProviderCall(
            deepseek_route(wire_profile=wire_profile), transport=RecordingTransport(),
            counter=lambda body: counted.append(body) or 100,
            stage=stage, cancelled=cancelled, clock=lambda: 1, deadline=10,
            output_tokens=100, data_scope=("current",), schema=schema,
        )
        request = {"messages": messages, "tools": [], "max_output_tokens": 100}
        call.count(request)
        return json.loads(call.invoke(request)["text"])

    model = FactModel(invoke, threading.Event())
    with pytest.raises(ValueError, match="schema"):
        model._call("fact_extract", "Return JSON", {"source": "synthetic"}, SCHEMA)
    assert len(sent) == 1 and counted == sent
    assert sent[0]["response_format"] == {"type": "json_object"}
    assert sent[0].get("temperature") == temperature
