"""Cyrene SocialAtom extraction shape adapted to two evidence-bound directions.

Source: social-context/extractor.ts at 1ce59f2 (MIT; THIRD_PARTY_NOTICES).
All operations validate before returning; no partial acceptance, guessed targets,
automatic 72-hour TTL, or storage side effect. Fact operations use their own Port.
"""

import json
import math
from hashlib import sha256

from virtual_companion.boundary.contracts import Scope, canonical, identifier
from virtual_companion.boundary.exchanges import (
    ExchangeCommand,
    ExchangeEvidence,
    evaluate_exchange,
)

from .evidence_memory import EvidenceTurn, _unique

_FIELDS = {
    "operation",
    "type",
    "content",
    "evidenceTurnId",
    "evidenceQuote",
    "supersedesAtomId",
    "expiresAt",
}
_DIRECTIONS = {"open_loop": "user_question", "response_obligation": "assistant_response"}
_ACTIONS = {
    "add": "open",
    "resolve": "resolve",
    "defer": "defer",
    "resume": "resume",
    "revoke": "close",
}


def compile_exchanges(raw, *, scope, batch_id, now_us, turns, targets, support,
                      original_indexes=None):
    """targets are the fresh companion projection; storage repeats revision checks."""
    if not isinstance(scope, Scope) or type(now_us) is not int or now_us < 0:
        raise ValueError("Typed scope and UTC time required")
    identifier(batch_id)
    if not isinstance(raw, str) or len(raw.encode()) > 64000:
        raise ValueError("Bounded extraction JSON required")
    document = json.loads(raw, object_pairs_hook=_unique)
    canonical(document)
    if not isinstance(document, dict) or set(document) != {"operations"}:
        raise ValueError("Exact extraction envelope required")
    operations = document["operations"]
    if not isinstance(operations, list) or len(operations) > 16:
        raise ValueError("Bounded exchange extraction required")
    if original_indexes is None:
        original_indexes = tuple(range(len(operations)))
    if (not isinstance(original_indexes, (tuple, list))
            or len(original_indexes) != len(operations)
            or any(type(index) is not int or not 0 <= index < 16
                   for index in original_indexes)
            or len(set(original_indexes)) != len(original_indexes)):
        raise ValueError("Unique original exchange candidate indexes required")
    sources = {}
    for turn in turns:
        if not isinstance(turn, EvidenceTurn) or turn.scope != scope or turn.source_id in sources:
            raise ValueError("Unique same-scope evidence sources required")
        sources[turn.source_id] = turn
    commands = []
    touched = set()
    for index, operation in zip(original_indexes, operations, strict=True):
        if not isinstance(operation, dict) or set(operation) != _FIELDS:
            raise ValueError("Exact exchange operation fields required")
        action = operation["operation"]
        if not isinstance(action, str) or action not in _ACTIONS:
            raise ValueError("Unsupported exchange action")
        source_id, quote = operation["evidenceTurnId"], operation["evidenceQuote"]
        if not isinstance(source_id, str) or source_id not in sources:
            raise ValueError("Unknown exchange evidence")
        turn = sources[source_id]
        prefix = turn.text[: turn.confirmed_end]
        if not isinstance(quote, str) or not quote.strip() or quote not in prefix:
            raise ValueError("Exchange evidence must be an exact confirmed quote")
        target_id = operation["supersedesAtomId"]
        if action == "add":
            kind = operation["type"]
            if target_id is not None or not isinstance(kind, str) or kind not in _DIRECTIONS:
                raise ValueError("New exchange requires its direction and no target")
            direction, content = _DIRECTIONS[kind], operation["content"]
            if content != quote:
                raise ValueError("New pending content must retain the verbatim evidence quote")
            expiry = operation["expiresAt"]
            if expiry is not None:
                if type(expiry) not in {int, float} or not math.isfinite(expiry):
                    raise ValueError("Finite explicit millisecond expiry required")
                expiry = int(expiry * 1000)
                if expiry <= now_us:
                    raise ValueError("Expired exchange proposal")
            key, revision = f"{batch_id}-exchange-{index}", None
        else:
            if not isinstance(target_id, str) or target_id not in targets:
                raise ValueError("Locate the current exchange target before compiling")
            target = targets[target_id]
            if (
                target.get("companion_id") != scope.companion_id
                or target.get("exchange_id") != target_id
                or target.get("status") not in {"pending", "deferred"}
                or (target.get("expires_us") is not None and target["expires_us"] <= now_us)
                or any(operation[k] is not None for k in ("type", "content", "expiresAt"))
            ):
                raise ValueError("Invalid target; transitions cannot rewrite its content or TTL")
            if target_id in touched:
                raise ValueError("One action per target per extraction")
            touched.add(target_id)
            direction, content = target["direction"], target["content"]
            key, revision, expiry = target_id, target["revision"], target["expires_us"]
        if action == "resolve" and turn.role != (
            "user" if direction == "user_question" else "assistant"
        ):
            raise ValueError("Resolution needs the responding party's evidence")
        command = ExchangeCommand(
            scope,
            f"{batch_id}-operation-{index}",
            key,
            _ACTIONS[action],
            direction,
            content,
            ExchangeEvidence(
                turn.role,
                turn.source_id,
                quote,
                turn.confirmed_end,
                sha256(prefix.encode()).hexdigest(),
            ),
            now_us,
            revision,
            expiry,
        )
        commands.append(command)
    # Schema/source validation completes before any evaluator receives a proposal.
    return tuple(
        (
            c,
            evaluate_exchange(
                c,
                support=support,
                confirmed_text=sources[c.evidence.source_id].text[: c.evidence.confirmed_end],
            ),
        )
        for c in commands
    )
