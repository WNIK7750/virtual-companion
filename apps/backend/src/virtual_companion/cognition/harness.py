"""Cyrene's cohesive round/transcript/checkpoint loop adapted to project Ports.

Source: Cyrene 1ce59f2, harness/cyrene-harness.ts, tool-round.ts and run-recovery.ts
(MIT; see THIRD_PARTY_NOTICES.md). This carrier is opt-in until whole-kernel
compatibility is complete. It returns an unreviewed candidate, never a commit.

Differences: explicit tools only; serial dispatch; no retries or fallback; complete
arguments checkpointed before dispatch; unknown effects retain canonical identity.
The injected whole-turn guard owns time/cancellation/lease, and the model Port
owns exact final-wire counting, reservation, interruptible I/O and terminal usage.
The journal must persist bound snapshots before returning. No business writes,
persona, private data, global usage store, auto-summary or external tool registry.
"""

import json
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256

from jsonschema import Draft202012Validator
from referencing import Registry

from virtual_companion.boundary.contracts import canonical, identifier


@dataclass(frozen=True)
class HarnessLimits:
    max_rounds: int
    context_tokens: int
    output_tokens: int

    def __post_init__(self):
        if (
            any(
                type(x) is not int or x <= 0
                for x in (self.max_rounds, self.context_tokens, self.output_tokens)
            )
            or self.output_tokens + 1 >= self.context_tokens
        ):
            raise ValueError("Positive explicit Harness limits required")


@dataclass(frozen=True, init=False)
class HarnessRestore:
    """Immutable execution checkpoint; messages alone are not recovery authority.

    The journal checks live ownership before rebinding a durable snapshot. This
    carrier neither grants ownership nor resets the provider's usage ledger.
    """

    payload: str

    def __init__(self, snapshot):
        _binding(snapshot.get("binding"))
        for field in ("rounds", "cache_epoch"):
            if type(snapshot.get(field)) is not int or snapshot[field] < 0:
                raise ValueError("Complete execution restore counters required")
        if not isinstance(snapshot.get("uncertain"), list):
            raise ValueError("Complete execution restore uncertainty required")
        object.__setattr__(self, "payload", canonical(snapshot))

    def snapshot(self):
        return json.loads(self.payload)


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate tool argument")
        result[key] = value
    return result


def tool_identity(call):
    if not isinstance(call, dict) or set(call) != {"id", "name", "arguments"}:
        raise ValueError("Exact tool call required")
    identifier(call["id"])
    identifier(call["name"])
    raw = call["arguments"]
    if not isinstance(raw, str) or len(raw.encode()) > 64000:
        raise ValueError("Bounded original tool arguments required")
    arguments = json.loads(raw, object_pairs_hook=_unique)
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be an object")
    return sha256(canonical([call["name"], arguments]).encode()).hexdigest()


def _binding(value):
    if not isinstance(value, dict) or set(value) != {
        "companion",
        "session",
        "input",
        "attempt",
        "owner",
        "epoch",
    }:
        raise ValueError("Complete Harness binding required")
    for name in ("companion", "session", "input", "attempt", "owner"):
        identifier(value[name])
    if type(value["epoch"]) is not int or value["epoch"] < 1:
        raise ValueError("Positive owner epoch required")


def _response(value, allowed):
    if not isinstance(value, dict) or set(value) != {"text", "tool_calls", "finish_reason"}:
        raise ValueError("Exact Harness response required")
    if not isinstance(value["text"], str) or len(value["text"].encode()) > 64000:
        raise ValueError("Bounded response text required")
    calls = value["tool_calls"]
    if not isinstance(calls, list) or len(calls) > 16:
        raise ValueError("Bounded tool round required")
    seen = set()
    for call in calls:
        tool_identity(call)
        if call["id"] in seen or call["name"] not in allowed:
            raise ValueError("Unknown tool or duplicate tool-call identity")
        # Validate against the frozen host declaration, not a model's schema or
        # a mutable caller object. An empty registry never retrieves remote refs.
        validator = Draft202012Validator(allowed[call["name"]]["parameters"], registry=Registry())
        try:
            valid = validator.is_valid(json.loads(call["arguments"]))
        except Exception as error:
            raise ValueError("Tool arguments schema could not be resolved") from error
        if not valid:
            raise ValueError("Tool arguments do not match the declared schema")
        seen.add(call["id"])
    if value["finish_reason"] not in {"stop", "tool_calls"}:
        # Unlike donor UI suffixes, truncated structured output cannot pass review.
        raise ValueError("Incomplete model output")
    if not calls and not value["text"].strip():
        raise ValueError("Empty candidate")


def _tool_message(call, result):
    return {"role": "tool", "tool_call_id": call["id"], "content": canonical(result)}


def run_harness(*, binding, messages, tools, model, dispatch, journal, guard, limits, restore=None):
    """Run one candidate subrun under the caller's unique whole-turn ownership.

    Model.count receives the complete request, including tool arguments and output
    reserve. Model.invoke must reserve before sending and account for all outcomes.
    No network SDK is selected here. An exception cannot become a display reply.
    """
    _binding(binding)
    if not isinstance(limits, HarnessLimits):
        raise ValueError("Explicit Harness limits required")
    if not isinstance(messages, list) or not messages or len(messages) > 256:
        raise ValueError("Bounded input transcript required")
    if not isinstance(tools, list) or len(tools) > 32:
        raise ValueError("Bounded explicit tools required")
    allowed = {}
    for tool in tools:
        if not isinstance(tool, dict) or set(tool) != {"name", "parameters", "effect"}:
            raise ValueError("Explicit tool schema/effect required")
        identifier(tool["name"])
        if (
            tool["name"] in allowed
            or tool["effect"] not in {"read_only", "idempotent", "non_idempotent", "unknown"}
            or not isinstance(tool["parameters"], dict)
        ):
            raise ValueError("Invalid tool declaration")
        Draft202012Validator.check_schema(tool["parameters"])
        allowed[tool["name"]] = tool
    # Copy before any callback: neither provider nor journal may mutate caller data.
    canonical([messages, tools])
    tools = deepcopy(tools)
    allowed = {tool["name"]: tool for tool in tools}
    guard.checkpoint()
    loader = getattr(journal, "load_for_execution", None)
    durable = loader() if callable(loader) else None
    if durable is not None:
        if restore is not None:
            raise ValueError("Durable journal owns execution restore")
        restore = HarnessRestore(durable)
    if restore is None and any(m.get("tool_calls") or m.get("role") == "tool" for m in messages):
        raise ValueError("Tool transcript requires typed execution restore")
    state = {
        "binding": deepcopy(binding),
        "operation_origin": deepcopy(binding),
        "initial_messages": deepcopy(messages),
        "messages": deepcopy(messages),
        "status": "running",
        "rounds": 0,
        "cache_epoch": 0,
        "pending": None,
        "uncertain": [],
        "request": None,
        "tools": tools,
        "max_rounds": limits.max_rounds,
    }
    completed = False
    if restore is not None:
        if not isinstance(restore, HarnessRestore):
            raise ValueError("Typed execution restore required")
        prior = restore.snapshot()
        if prior["binding"] != binding:
            raise ValueError("Recovery binding mismatch")
        # Legacy effect checkpoints lack a frozen capability contract; refusing
        # execution preserves evidence without pretending current tools are safe.
        if prior.get("tools") != tools:
            raise ValueError("Recovery tool contract mismatch")
        if messages not in (prior.get("initial_messages"), prior.get("messages")):
            raise ValueError("Recovery input transcript mismatch")
        completed = prior["status"] == "candidate"
        state = prior if completed else recover_transcript(prior, binding=binding)
        state["status"] = "candidate" if completed else "running"

    def save():
        if len(canonical(state).encode()) > 2_000_000:
            raise ValueError("Harness checkpoint capacity exceeded")
        journal.save(deepcopy(state))

    try:
        if completed:
            guard.checkpoint()
            save()
            return {
                "status": "candidate", "text": state["messages"][-1]["content"],
                "messages": deepcopy(state["messages"]), "uncertain": list(state["uncertain"]),
            }
        for round_number in range(state["rounds"], min(limits.max_rounds, state["max_rounds"])):
            guard.checkpoint()
            request = {
                "messages": deepcopy(state["messages"]),
                "tools": deepcopy(tools),
                "max_output_tokens": limits.output_tokens,
            }
            count = model.count(deepcopy(request))
            guard.checkpoint()
            if type(count) is not int or count < 0:
                raise ValueError("Exact request count required")
            if count + limits.output_tokens + 1 > limits.context_tokens:
                raise ValueError("Final request capacity exceeded")
            state["request"] = {
                "round": round_number,
                "status": "prepared",
                "sha256": sha256(canonical(request).encode()).hexdigest(),
            }
            # A prepared request may have reached the provider. Count it before
            # invoking so crash/recovery cannot replenish the execution budget.
            state["rounds"] = round_number + 1
            save()
            guard.checkpoint()
            result = model.invoke(request)
            guard.checkpoint()
            _response(result, allowed)
            state["request"]["status"] = "returned"
            calls = deepcopy(result["tool_calls"])
            seen_calls = {
                call["id"]
                for message in state["messages"]
                for call in message.get("tool_calls", [])
            }
            if any(call["id"] in seen_calls for call in calls):
                raise ValueError("Tool-call identity reused across rounds")
            state["messages"].append(
                {
                    "role": "assistant",
                    "content": result["text"],
                    **({"tool_calls": calls} if calls else {}),
                }
            )
            # A no-tool response ends this loop; the caller still reviews its candidate.
            if not calls:
                state["status"] = "candidate"
                save()
                guard.checkpoint()
                return {
                    "status": "candidate",
                    "text": result["text"],
                    "messages": deepcopy(state["messages"]),
                    "uncertain": list(state["uncertain"]),
                }
            blocked = False
            for call in calls:
                guard.checkpoint()
                fingerprint = tool_identity(call)
                if blocked:
                    outcome = {"outcome": "not_executed", "content": "prior tool uncertain"}
                elif fingerprint in state["uncertain"]:
                    outcome = {"outcome": "unknown", "content": "unresolved prior effect"}
                else:
                    state["pending"] = deepcopy(call)
                    save()  # Includes the assistant and complete original argument JSON.
                    guard.checkpoint()
                    try:
                        outcome = dispatch(deepcopy(call))
                    except Exception:
                        # Even a declared read may be implemented incorrectly. Never retry
                        # from an exception or stringify an untrusted error/credential.
                        outcome = {"outcome": "unknown", "content": "dispatch interrupted"}
                    guard.checkpoint()
                    if (
                        not isinstance(outcome, dict)
                        or set(outcome) != {"outcome", "content"}
                        or outcome["outcome"] not in {"success", "failure", "unknown"}
                        or not isinstance(outcome["content"], str)
                        or len(outcome["content"].encode()) > 64000
                    ):
                        outcome = {"outcome": "unknown", "content": "invalid tool result"}
                if outcome["outcome"] == "unknown":
                    if fingerprint not in state["uncertain"]:
                        state["uncertain"].append(fingerprint)
                    blocked = True
                elif outcome["outcome"] == "failure":
                    blocked = True
                state["messages"].append(_tool_message(call, outcome))
                state["pending"] = None
                save()
            save()
        raise ValueError("Harness round limit reached")
    except BaseException as error:
        state["status"] = "interrupted"
        try:
            save()
        except Exception as journal_error:
            error.add_note(f"Terminal checkpoint failed: {type(journal_error).__name__}")
        raise


def recover_transcript(snapshot, *, binding):
    """Pure reconciliation only; never invoke a model/tool or authorize a new owner.

    The storage Port verifies current ownership before providing a snapshot. A new
    epoch requires an explicit owner-checked rebind outside this pure function.
    Unfinished calls are closed honestly; original missing evidence fails closed.
    """
    _binding(binding)
    if snapshot.get("binding") != binding:
        raise ValueError("Recovery binding mismatch")
    state = deepcopy(snapshot)
    pending = state.get("pending")
    if pending is not None:
        fingerprint = tool_identity(pending)
        originals = [
            call
            for message in state["messages"]
            for call in message.get("tool_calls", [])
            if call["id"] == pending["id"]
        ]
        if originals != [pending]:
            raise ValueError("Missing or ambiguous original tool arguments")
        resolved = {m["tool_call_id"] for m in state["messages"] if m["role"] == "tool"}
        if pending["id"] in resolved:
            raise ValueError("Pending tool already has a result")
        if fingerprint not in state["uncertain"]:
            state["uncertain"].append(fingerprint)
    for message in list(state["messages"]):
        resolved = {m["tool_call_id"] for m in state["messages"] if m["role"] == "tool"}
        for call in message.get("tool_calls", []):
            if call["id"] in resolved:
                continue
            tool_identity(call)
            started = pending is not None and call["id"] == pending["id"]
            state["messages"].append(
                _tool_message(
                    call,
                    {
                        "outcome": "unknown" if started else "not_executed",
                        "content": "interrupted; explicit reconciliation required",
                    },
                )
            )
    state.update(pending=None, status="reconciled", cache_epoch=state["cache_epoch"] + 1)
    return state
