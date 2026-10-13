"""Fixed-source Harness behavior at project Ports; no provider or external action."""

import copy
import json

import pytest
from virtual_companion.cognition.harness import (
    HarnessLimits,
    recover_transcript,
    run_harness,
    tool_identity,
)


class Journal:
    def __init__(self):
        self.entries = []

    def save(self, snapshot):
        self.entries.append(copy.deepcopy(snapshot))


class Guard:
    def checkpoint(self):
        pass


class Model:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def count(self, request):
        # Synthetic exact counter for the test wire, not a tokenizer claim.
        return len(json.dumps(request, ensure_ascii=False))

    def invoke(self, request):
        self.requests.append(copy.deepcopy(request))
        return next(self.responses)


def response(text="candidate", calls=None):
    return {"text": text, "tool_calls": calls or [], "finish_reason": "stop"}


def tool_call(call_id="call", **args):
    return {"id": call_id, "name": "lookup", "arguments": json.dumps(args)}


def execute(model, journal, *, dispatch=None, guard=None, **changes):
    arguments = dict(
        binding={
            "companion": "synthetic",
            "session": "s",
            "input": "i",
            "attempt": "a",
            "owner": "o",
            "epoch": 1,
        },
        messages=[{"role": "user", "content": "synthetic input"}],
        tools=[{"name": "lookup", "parameters": {"type": "object"}, "effect": "read_only"}],
        model=model,
        dispatch=dispatch or (lambda _: {"outcome": "success", "content": "evidence"}),
        journal=journal,
        guard=guard or Guard(),
        limits=HarnessLimits(4, 16000, 256),
    )
    arguments.update(changes)
    return run_harness(**arguments)


def test_tool_transcript_is_checkpointed_before_dispatch_and_result_is_only_candidate():
    journal = Journal()
    first = tool_call(recipient={"name": "fixture"})
    model = Model([response("checking", [first]), response("supported candidate")])

    def dispatch(call):
        saved = journal.entries[-1]
        assert saved["pending"] == call
        assert saved["messages"][-1]["tool_calls"] == [first]
        return {"outcome": "success", "content": "evidence"}

    result = execute(model, journal, dispatch=dispatch)
    assert result["status"] == "candidate"
    assert result["text"] == "supported candidate"
    assert "commit" not in result
    assert model.requests[1]["messages"][-2:][0]["tool_calls"] == [first]
    assert model.requests[1]["messages"][-1]["tool_call_id"] == "call"
    assert journal.entries[-1]["status"] == "candidate"
    assert model.requests[0]["max_output_tokens"] == 256
    assert [tool["name"] for tool in model.requests[0]["tools"]] == ["lookup"]


def test_unknown_effect_stops_round_and_cannot_be_replayed_under_new_id():
    first = tool_call("first", body={"target": "fixture"})
    next_call = {**first, "id": "second"}
    model = Model(
        [
            response("", [first, tool_call("later")]),
            response("", [next_call]),
            response("cannot confirm"),
        ]
    )
    invoked = []

    def dispatch(call):
        invoked.append(call)
        raise TimeoutError("effect may already have happened")

    result = execute(model, Journal(), dispatch=dispatch)
    assert len(invoked) == 1
    assert result["uncertain"] == [tool_identity(first)]
    results = [
        json.loads(m["content"])["outcome"] for m in result["messages"] if m["role"] == "tool"
    ]
    assert results == ["unknown", "not_executed", "unknown"]


def test_nested_identity_and_strict_argument_types():
    assert tool_identity(tool_call(body={"target": "a"})) != tool_identity(
        tool_call(body={"target": "b"})
    )
    assert tool_identity(tool_call(a=1, b=2)) == tool_identity(tool_call(b=2, a=1))
    for invalid in ('{"a":1,"a":2}', "[]", '{"value":NaN}'):
        with pytest.raises(ValueError):
            tool_identity({**tool_call(), "arguments": invalid})


@pytest.mark.parametrize("arguments", [{}, {"query": 3}, {"query": "city", "companion": "foreign"}])
def test_tool_schema_is_enforced_before_any_dispatch(arguments):
    tools = [{"name": "lookup", "effect": "read_only", "parameters": {
        "type": "object", "properties": {"query": {"type": "string"}},
        "required": ["query"], "additionalProperties": False,
    }}]
    sent = []
    model = Model([response("", [tool_call(**arguments)]), response()])
    journal = Journal()
    with pytest.raises(ValueError, match="arguments"):
        execute(model, journal, tools=tools,
                dispatch=lambda call: sent.append(call) or {"outcome": "success", "content": ""})
    assert not sent
    assert journal.entries[-1]["status"] == "interrupted"
    assert len(model.requests) == 1


def test_tool_schema_is_frozen_before_model_callbacks():
    tools = [{"name": "lookup", "effect": "read_only", "parameters": {
        "type": "object", "properties": {"query": {"type": "string"}},
        "required": ["query"], "additionalProperties": False,
    }}]
    model = Model([response("", [tool_call()]), response()])
    count = model.count

    def mutating_count(request):
        tools[0]["parameters"].clear()
        return count(request)

    model.count = mutating_count
    sent = []
    with pytest.raises(ValueError, match="arguments"):
        execute(model, Journal(), tools=tools,
                dispatch=lambda call: sent.append(call) or {"outcome": "success", "content": ""})
    assert not sent


def test_valid_local_reference_schema_still_dispatches_original_arguments():
    parameters = {"type": "object", "$defs": {"query": {"type": "string"}},
                  "properties": {"query": {"$ref": "#/$defs/query"}},
                  "required": ["query"], "additionalProperties": False}
    tools = [{"name": "lookup", "effect": "read_only", "parameters": parameters}]
    call = tool_call(query="住在哪座城市？")
    seen = []
    result = execute(Model([response("", [call]), response("candidate")]), Journal(),
                     tools=tools, dispatch=lambda value: seen.append(value) or {
                         "outcome": "success", "content": "source-bound evidence"})
    assert seen == [call]
    assert result["text"] == "candidate"  # Still unreviewed, not delivered.


def test_tool_schema_never_fetches_remote_reference(monkeypatch):
    import urllib.request

    fetched, sent = [], []
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: fetched.append(a))
    tools = [{"name": "lookup", "effect": "read_only", "parameters": {
        "$ref": "https://untrusted.invalid/tool-schema.json",
    }}]
    with pytest.raises(ValueError, match="schema could not be resolved"):
        execute(Model([response("", [tool_call()])]), Journal(), tools=tools,
                dispatch=lambda value: sent.append(value))
    assert not fetched and not sent


def test_call_id_cannot_be_reused_with_new_arguments_in_a_later_round():
    model = Model([response("", [tool_call(value=1)]), response("", [tool_call(value=2)])])
    dispatched = []

    def dispatch(call):
        dispatched.append(call)
        return {"outcome": "success", "content": "evidence"}

    with pytest.raises(ValueError, match="reused"):
        execute(model, Journal(), dispatch=dispatch)
    assert len(dispatched) == 1


def test_complete_request_including_tool_arguments_is_counted_before_more_calls():
    model = Model([response("", [tool_call(payload="x" * 1500)])])
    with pytest.raises(ValueError, match="capacity"):
        execute(model, Journal(), limits=HarnessLimits(4, 1000, 256))
    assert len(model.requests) == 1


def test_failed_checkpoint_prevents_first_model_or_tool_send():
    class Broken(Journal):
        def save(self, snapshot):
            raise OSError("synthetic disk full")

    model = Model([response()])
    with pytest.raises(OSError):
        execute(model, Broken())
    assert model.requests == []


def test_cancel_after_model_returns_never_releases_candidate():
    class Cancelled(Guard):
        def checkpoint(self):
            if model.requests:
                raise InterruptedError("cancelled")

    journal, model = Journal(), Model([response()])
    with pytest.raises(InterruptedError):
        execute(model, journal, guard=Cancelled())
    assert journal.entries[-1]["status"] == "interrupted"


def test_round_limit_never_makes_an_extra_model_request_or_successful_final():
    model = Model([response("", [tool_call()])])
    with pytest.raises(ValueError, match="round limit"):
        execute(model, Journal(), limits=HarnessLimits(1, 16000, 256))
    assert len(model.requests) == 1


def test_recovery_requires_original_arguments_and_binding_and_never_reexecutes():
    journal, model = Journal(), Model([response("", [tool_call(body={"a": 1})]), response()])
    execute(model, journal)
    started = next(s for s in journal.entries if s["pending"] is not None)
    before = copy.deepcopy(started)
    recovered = recover_transcript(started, binding=started["binding"])
    assert recovered["uncertain"] == [tool_identity(started["pending"])]
    assert json.loads(recovered["messages"][-1]["content"])["outcome"] == "unknown"
    assert recovered["cache_epoch"] == started["cache_epoch"] + 1
    assert started == before
    with pytest.raises(ValueError, match="binding"):
        recover_transcript(started, binding={**started["binding"], "epoch": 2})
    started["messages"] = started["messages"][:-1]
    with pytest.raises(ValueError, match="original"):
        recover_transcript(started, binding=started["binding"])
