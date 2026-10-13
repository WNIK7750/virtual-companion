"""Pure Chat Completions wire adaptation; no credentials or provider selection.

Behavioral mapping from Cyrene vendors/openai-adapter.ts at 1ce59f2 (MIT).
Keep explicit output caps, exact schema capabilities, unknown usage and strict
termination instead of the donor's global overrides/default-zero usage.
"""

import json
import re
from copy import deepcopy
from dataclasses import asdict

from jsonschema import Draft202012Validator

from virtual_companion.boundary.contracts import canonical, identifier

from .contracts import (
    DEEPSEEK_FLASH_PROFILE,
    DEEPSEEK_FOCUSED_PROFILE,
    DEEPSEEK_TEXT_PROFILES,
    ProviderRoute,
    focused_stage_temperature,
)

_JSON_OBJECT_INSTRUCTION = "Return exactly one JSON object matching this JSON schema:\n"


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate provider JSON field")
        value[key] = item
    return value


def strict_json(raw):
    result = json.loads(raw, object_pairs_hook=_unique)
    canonical(result)
    return result


def validate_wire_profile(route, body, *, stage=None):
    """Check optional wire invariants before credentials, reservation or transmission.

    JSON-object mode constrains JSON syntax only. Business callers retain their
    bound host-side JSON Schema validation; the provider is not a schema authority.
    """
    if route.wire_profile not in DEEPSEEK_TEXT_PROFILES:
        return
    forbidden = {"tools", "tool_choice", "seed", "top_p", "reasoning_effort"}
    if route.wire_profile == DEEPSEEK_FLASH_PROFILE:
        forbidden.add("temperature")
    else:
        temperature = focused_stage_temperature(stage)
        required = {"model", "messages", "stream", "stream_options", "max_tokens",
                    "thinking", "temperature"}
        if (not required <= set(body) or set(body) - required - {"response_format"}
                or type(body.get("temperature")) not in {int, float}
                or body["temperature"] != temperature):
            raise ValueError("Request differs from the focused DeepSeek stage profile")
    if (
        body.get("thinking") != {"type": "disabled"}
        or body.get("stream_options") != {"include_usage": True}
        or set(body).intersection(forbidden)
    ):
        raise ValueError("Request differs from the frozen DeepSeek wire profile")
    messages = body.get("messages")
    if not isinstance(messages, list) or not 0 < len(messages) <= 512 or any(
        not isinstance(item, dict)
        or set(item) != {"role", "content"}
        or item["role"] not in {"system", "user", "assistant"}
        or not isinstance(item["content"], str)
        for item in messages
    ):
        raise ValueError("DeepSeek wire profile requires bounded plain-text messages")
    instruction = messages[0]
    has_schema = instruction["role"] == "system" and instruction["content"].startswith(
        _JSON_OBJECT_INSTRUCTION
    )
    if "response_format" not in body:
        if has_schema:
            raise ValueError("DeepSeek schema instruction requires JSON-object mode")
        return
    if (not route.capabilities.structured_output
            or body["response_format"] != {"type": "json_object"} or not has_schema):
        raise ValueError("DeepSeek JSON-object mode requires its explicit schema instruction")
    encoded = instruction["content"][len(_JSON_OBJECT_INSTRUCTION):]
    schema = strict_json(encoded)
    Draft202012Validator.check_schema(schema)
    if (not isinstance(schema, dict) or schema.get("type") != "object"
            or canonical(schema) != encoded):
        raise ValueError("DeepSeek JSON-object mode requires a canonical object schema")


def _call(call):
    if not isinstance(call, dict) or set(call) != {"id", "name", "arguments"}:
        raise ValueError("Exact provider tool call required")
    identifier(call["id"])
    identifier(call["name"])
    if (
        not isinstance(call["arguments"], str)
        or len(call["arguments"].encode()) > 64000
        or not isinstance(strict_json(call["arguments"]), dict)
    ):
        raise ValueError("Complete tool argument object required")
    return {
        "id": call["id"],
        "type": "function",
        "function": {"name": call["name"], "arguments": call["arguments"]},
    }


def build_request(route, messages, *, output_tokens, tools=(), schema=None, stage=None):
    if not isinstance(route, ProviderRoute):
        raise ValueError("Typed model route required")
    if not route.capabilities.streaming:
        raise ValueError("Provider streaming is unsupported")
    if type(output_tokens) is not int or not 0 < output_tokens < route.capabilities.context_tokens:
        raise ValueError("Positive bounded output reserve required")
    if not isinstance(messages, list) or not 0 < len(messages) <= 512:
        raise ValueError("Bounded complete messages required")
    wire, outstanding, all_calls = [], set(), set()
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {
            "system",
            "user",
            "assistant",
            "tool",
        }:
            raise ValueError("Explicit message role required")
        role = message["role"]
        fields = {"role", "content"}
        if role == "assistant" and "tool_calls" in message:
            fields.add("tool_calls")
        if role == "tool":
            fields.add("tool_call_id")
        if set(message) != fields or not isinstance(message.get("content"), str):
            raise ValueError("Exact text message fields required")
        item = {"role": role, "content": message["content"]}
        if role == "tool":
            if message["tool_call_id"] not in outstanding:
                raise ValueError("Unmatched tool result")
            outstanding.remove(message["tool_call_id"])
            item["tool_call_id"] = message["tool_call_id"]
        elif outstanding:
            raise ValueError("Unclosed tool round")
        if "tool_calls" in message:
            calls = message["tool_calls"]
            if not isinstance(calls, list) or not 0 < len(calls) <= 16:
                raise ValueError("Bounded tool calls required")
            item["tool_calls"] = [_call(c) for c in calls]
            ids = [c["id"] for c in calls]
            if len(set(ids)) != len(ids) or all_calls.intersection(ids):
                raise ValueError("Duplicate tool call IDs")
            all_calls.update(ids)
            outstanding.update(ids)
        wire.append(item)
    if outstanding:
        raise ValueError("Complete tool results required before the next model request")
    if (tools or all_calls) and not route.capabilities.tools:
        raise ValueError("Provider tools are unsupported")
    body = {
        "model": route.model,
        "messages": wire,
        "stream": True,
        "stream_options": {"include_usage": True},
        route.capabilities.output_field: output_tokens,
    }
    if route.sampling is not None:
        body.update(asdict(route.sampling))
    if route.wire_profile in DEEPSEEK_TEXT_PROFILES:
        # Explicit disable mirrors the pinned pi DeepSeek compatibility branch.
        # No generic extra_body escape hatch or provider-name inference is used.
        body["thinking"] = {"type": "disabled"}
    if route.wire_profile == DEEPSEEK_FOCUSED_PROFILE:
        body["temperature"] = focused_stage_temperature(stage)
    if tools:
        if not isinstance(tools, (tuple, list)) or len(tools) > 32:
            raise ValueError("Bounded tool declarations required")
        specs = []
        for tool in tools:
            if (
                not isinstance(tool, dict)
                or set(tool) - {"name", "parameters", "description", "effect"}
                or not {"name", "parameters"}.issubset(tool)
            ):
                raise ValueError("Explicit function schema required")
            identifier(tool["name"])
            Draft202012Validator.check_schema(tool["parameters"])
            specs.append(
                {
                    "type": "function",
                    "function": {k: deepcopy(v) for k, v in tool.items() if k != "effect"},
                }
            )
        if len({s["function"]["name"] for s in specs}) != len(specs):
            raise ValueError("Duplicate tools")
        body.update(tools=specs, tool_choice="auto")
    if schema is not None:
        if not route.capabilities.structured_output:
            raise ValueError("Provider structured output is unsupported")
        if route.wire_profile in DEEPSEEK_TEXT_PROFILES:
            if not isinstance(schema, dict) or schema.get("type") != "object":
                raise ValueError("DeepSeek JSON-object mode requires an object schema")
            wire.insert(0, {
                "role": "system", "content": _JSON_OBJECT_INSTRUCTION + canonical(schema),
            })
            body["response_format"] = {"type": "json_object"}
        else:
            Draft202012Validator.check_schema(schema)
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "companion_result", "strict": True, "schema": deepcopy(schema),
                },
            }
    validate_wire_profile(route, body, stage=stage)
    if len(canonical(body).encode()) > 2_000_000:
        raise ValueError("Provider request byte limit exceeded")
    return body


class StreamFailure(ValueError):
    def __init__(self, reason, usage):
        super().__init__("Invalid provider stream")
        self.reason = reason
        self.usage = deepcopy(usage)


class _StreamTerminalFailure(ValueError):
    def __init__(self, reason):
        super().__init__("Invalid provider stream terminal")
        self.reason = reason


class ContentObservation:
    """Observe the first nonempty content delta; never approve a partial result.

    Timing supports the LF/CRLF SSE framing used by the pinned runtime. Final
    validation remains decode_stream, including after any malformed early event.
    """

    def __init__(self):
        self.pending, self.found = b"", False

    def feed(self, chunk):
        if self.found:
            return False
        records = re.split(rb"\r?\n\r?\n", self.pending + chunk)
        self.pending = records.pop()
        for record in records:
            data = b"\n".join(
                line[5:].removeprefix(b" ")
                for line in record.splitlines()
                if line.startswith(b"data:")
            )
            try:
                item = strict_json(data.decode("utf-8"))
                choices = item.get("choices", [])
                if not isinstance(choices, list):
                    continue
                for choice in choices:
                    content = choice.get("delta", {}).get("content")
                    if choice.get("index") == 0 and isinstance(content, str) and content:
                        self.found, self.pending = True, b""
                        return True
            except (ValueError, AttributeError, TypeError):
                continue
        return False


def decode_stream(raw):
    observed = {"usage": None}
    try:
        return _decode_stream(raw, observed)
    except (ValueError, TypeError, AttributeError, KeyError) as error:
        reason = error.reason if isinstance(error, _StreamTerminalFailure) else "invalid_stream"
        raise StreamFailure(reason, observed["usage"]) from None


def _decode_stream(raw, observed):
    """Decode a bounded complete SSE transcript. The transport owns cancellation.

    UTF-8 is decoded strictly after incremental bytes are collected, so a split
    multibyte code point is preserved. A dropped stream cannot become success.
    """
    if not isinstance(raw, bytes) or len(raw) > 2_000_000:
        raise ValueError("Bounded provider stream required")
    records = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n").split("\n\n")
    text, calls, usage, finish, done = [], {}, None, None, False
    for record in records:
        data = "\n".join(
            line[5:].removeprefix(" ") for line in record.split("\n") if line.startswith("data:")
        )
        if not data:
            continue
        if done:
            raise ValueError("Data after provider terminator")
        if data == "[DONE]":
            done = True
            continue
        item = strict_json(data)
        if not isinstance(item, dict) or not isinstance(item.get("choices"), list):
            raise ValueError("Invalid provider event")
        if item.get("usage") is not None:
            reported = item["usage"]
            keys = ("prompt_tokens", "completion_tokens", "total_tokens")
            if (
                not isinstance(reported, dict)
                or any(type(reported.get(k)) is not int or reported[k] < 0 for k in keys)
                or reported["total_tokens"]
                != reported["prompt_tokens"] + reported["completion_tokens"]
                or (usage is not None and usage != reported)
            ):
                raise ValueError("Invalid or conflicting provider usage")
            usage = reported
            observed["usage"] = reported
        choices = item["choices"]
        if not choices:
            continue
        if len(choices) != 1 or choices[0].get("index") != 0 or finish is not None:
            raise ValueError("Multiple or late completion choices")
        choice = choices[0]
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            raise ValueError("Invalid or refused provider response")
        if delta.get("refusal"):
            raise _StreamTerminalFailure("response_refused")
        if delta.get("content") is not None:
            if not isinstance(delta["content"], str):
                raise ValueError("Invalid provider content")
            text.append(delta["content"])
        for part in delta.get("tool_calls", []):
            index = part.get("index")
            if type(index) is not int or not 0 <= index < 16:
                raise ValueError("Invalid streamed tool index")
            current = calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
            if part.get("id"):
                if current["id"] or not isinstance(part["id"], str):
                    raise ValueError("Repeated streamed tool ID")
                current["id"] = part["id"]
            function = part.get("function", {})
            if not isinstance(function, dict):
                raise ValueError("Invalid streamed function")
            for key in ("name", "arguments"):
                if key in function and function[key] is not None:
                    if not isinstance(function[key], str):
                        raise ValueError("Invalid streamed tool fragment")
                    current[key] += function[key]
        if choice.get("finish_reason") is not None:
            finish = choice["finish_reason"]
    if not done:
        raise _StreamTerminalFailure("stream_incomplete")
    if finish == "length":
        raise _StreamTerminalFailure("output_limit_reached")
    if finish not in {"stop", "tool_calls"}:
        raise _StreamTerminalFailure("finish_reason_rejected")
    ordered = [calls[i] for i in sorted(calls)]
    for call in ordered:
        _call(call)
    if len({call["id"] for call in ordered}) != len(ordered):
        raise ValueError("Duplicate streamed tool IDs")
    content = "".join(text)
    if len(content.encode()) > 64000 or (not content.strip() and not ordered):
        raise ValueError("Empty or oversized provider result")
    return {"text": content, "tool_calls": ordered, "finish_reason": finish, "usage": usage}
