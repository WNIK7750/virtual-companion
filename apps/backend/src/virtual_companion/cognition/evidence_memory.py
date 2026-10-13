"""SocialAtom operation adaptation; provenance/support are required before mutation.

Behavior adapted from Cyrene social-context/extractor.ts at 1ce59f2 (MIT).
See THIRD_PARTY_NOTICES.md. Unlike the donor, reject the entire invalid batch,
require a bound support verdict, and keep the project's stable record revisions.
This compiler does not infer support or write storage.
"""

import json
from datetime import UTC, datetime
from hashlib import sha256

from virtual_companion.boundary.contracts import Scope, canonical, identifier
from virtual_companion.boundary.evidence import EvidenceTurn
from virtual_companion.boundary.memory import MemoryMutation
from virtual_companion.text.chronology import utc_microseconds


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate extraction field")
        value[key] = item
    return value


def compile_operations(raw, *, scope, batch_id, now, turns, targets, support, fact_keys):
    """Compile a complete recording/model proposal. Support callback is a trusted Port.

    targets is a fresh companion-scoped projection with companion_id attached by
    the caller, not arbitrary model content. Revisions are checked again by storage.
    Open-loop/resolve remains an obligation proposal for B; it cannot write facts.
    """
    if not isinstance(scope, Scope):
        raise ValueError("Typed memory scope required")
    identifier(batch_id)
    now_us = utc_microseconds(now)
    if not isinstance(raw, str) or len(raw.encode()) > 64000:
        raise ValueError("Bounded extraction object required")
    document = json.loads(raw, object_pairs_hook=_unique)
    canonical(document)
    if not isinstance(document, dict) or set(document) != {"operations"}:
        raise ValueError("Exact extraction envelope required")
    operations = document["operations"]
    if not isinstance(operations, list) or len(operations) > 16:
        raise ValueError("Bounded operation batch required")
    sources = {}
    for turn in turns:
        if not isinstance(turn, EvidenceTurn) or turn.source_id in sources:
            raise ValueError("Unique typed evidence sources required")
        if turn.scope != scope:
            raise ValueError("Evidence belongs to another companion/session")
        sources[turn.source_id] = turn
    mutations = []
    fields = {
        "operation",
        "type",
        "content",
        "evidenceTurnId",
        "evidenceQuote",
        "supersedesAtomId",
        "expiresAt",
    }
    for index, operation in enumerate(operations):
        if not isinstance(operation, dict) or set(operation) != fields:
            raise ValueError("Exact operation fields required")
        kind = operation["operation"]
        if kind not in {"add", "supersede", "revoke"}:
            raise ValueError("Obligation operations require the obligation Port")
        if operation["type"] not in {"long_term", "short_term"}:
            raise ValueError("Only user fact types can enter this memory compiler")
        source_id, quote = operation["evidenceTurnId"], operation["evidenceQuote"]
        if not isinstance(source_id, str) or source_id not in sources:
            raise ValueError("Unknown evidence source")
        turn = sources[source_id]
        if (
            turn.role != "user"
            or not isinstance(quote, str)
            or not quote.strip()
            or quote not in turn.text[: turn.confirmed_end]
        ):
            raise ValueError("User evidence must be a confirmed exact quote")
        content = operation["content"]
        if not isinstance(content, str) or (kind != "revoke" and not content.strip()):
            raise ValueError("Explicit fact content required")
        if len(content.encode()) > 16000 or (kind == "revoke" and content):
            raise ValueError("Invalid fact or revocation content")
        if support is None:
            raise ValueError("Missing semantic support evaluator")
        verdict = support(content, turn, quote)
        if (
            not isinstance(verdict, dict)
            or verdict.get("verdict") != "supported"
            or verdict.get("content_sha256") != sha256(content.encode()).hexdigest()
            or verdict.get("source_sha256") != sha256(turn.text.encode()).hexdigest()
            or not isinstance(verdict.get("evaluator"), str)
            or not verdict["evaluator"]
        ):
            raise ValueError("Fact support is unknown, unsupported or bound to other evidence")
        expires = operation["expiresAt"]
        expiry = None
        if operation["type"] == "short_term":
            if type(expires) not in {int, float}:
                raise ValueError("Short-term expiry required")
            expiry = datetime.fromtimestamp(expires / 1000, UTC)
            if utc_microseconds(expiry) <= now_us:
                raise ValueError("Expired proposal")
        elif expires is not None:
            raise ValueError("Long-term operation cannot conceal a TTL")
        target_id = operation["supersedesAtomId"]
        expected = None
        if kind == "add":
            if target_id is not None:
                raise ValueError("New fact cannot claim a replacement target")
            record_id = f"{batch_id}-record-{index}"
        else:
            if not isinstance(target_id, str) or target_id not in targets:
                raise ValueError("Unknown revision target; locate it before compiling")
            target = targets[target_id]
            if (
                target.get("companion_id") != scope.companion_id
                or target.get("record_id") != target_id
                or target.get("status") != "active"
                or target.get("kind") != "fact"
            ):
                raise ValueError("Invalid or foreign revision target")
            if (
                target.get("expires_at")
                and utc_microseconds(datetime.fromisoformat(target["expires_at"])) <= now_us
            ):
                raise ValueError("Expired revision target")
            record_id, expected = target_id, target["version"]
        mutations.append(
            MemoryMutation(
                scope=scope,
                mutation_id=f"{batch_id}-operation-{index}",
                kind={"add": "assert_fact", "supersede": "correct", "revoke": "revoke"}[kind],
                record_id=record_id,
                content=content,
                source_kind="input",
                source_id=turn.source_id,
                reason=canonical({"quote": quote, "support": verdict}),
                occurred_at=now,
                fact_key=fact_keys.get(index) if kind == "add" else None,
                expected_record_version=expected,
                expires_at=expiry,
            )
        )
    return tuple(mutations)
