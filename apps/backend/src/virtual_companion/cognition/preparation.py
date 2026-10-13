"""Immutable A1 input and cooperative-call contracts; no model, storage or framework I/O.

The application supplies a current source-of-truth memory read and explicitly
selected history. This does not enable automatic recall or choose a prompt strategy.
"""

import json
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from hashlib import sha256

from virtual_companion.boundary.contracts import (
    Scope,
    StateSnapshot,
    VersionConflict,
    canonical,
    companion_id,
    identifier,
)
from virtual_companion.boundary.memory import FactKey
from virtual_companion.boundary.turns import CognitiveResult, RunClaim
from virtual_companion.text.chronology import utc_microseconds
from virtual_companion.text.identity import CompanionConfig

DUTIES = (
    "main_reply",
    "engagement",
    "state_appraisal",
    "causal_review",
    "expression_director",
    "expression_review",
)
MAX_REVIEW_ATTEMPTS = 3  # Includes the first candidate, not three extra rewrites.


def integer(value, minimum=0):
    if type(value) is not int or not minimum <= value < 2**63:
        raise ValueError("Bounded integer required")


def text(value):
    if not isinstance(value, str) or len(value.encode()) > 64000:
        raise ValueError("Bounded UTF-8 text required")


@dataclass(frozen=True)
class DisplayedTurn:
    scope: Scope
    input_id: str
    role: str
    text: str
    displayed_end: int | None
    occurred_at: str | None = None

    def __post_init__(self):
        if not isinstance(self.scope, Scope) or self.role not in {"user", "assistant"}:
            raise ValueError("Typed history scope and role required")
        identifier(self.input_id)
        text(self.text)
        if self.occurred_at is not None:
            utc_microseconds(datetime.fromisoformat(self.occurred_at))
        if self.displayed_end is not None:
            integer(self.displayed_end)
            if self.displayed_end > len(self.text):
                raise ValueError("Display prefix exceeds content")
        if self.role == "user" and self.displayed_end != len(self.text):
            raise ValueError("User input must be complete")


@dataclass(frozen=True)
class MemoryEvidence:
    companion_id: str
    record_id: str
    version: int
    status: str
    source_kind: str
    source_id: str
    content: str
    expires_us: int | None
    kind: str = "unclassified"
    fact_key: FactKey | None = None

    def __post_init__(self):
        companion_id(self.companion_id)
        identifier(self.record_id)
        identifier(self.source_id)
        integer(self.version, 1)
        if self.kind not in {"unclassified", "fact", "event"}:
            raise ValueError("Explicit memory record kind required")
        if self.kind == "fact":
            if not isinstance(self.fact_key, FactKey):
                raise ValueError("Fact memory requires typed fact identity")
        elif self.fact_key is not None:
            raise ValueError("Only fact memory may carry fact identity")
        if self.status not in {"active", "revoked"} or self.source_kind not in {
            "input",
            "assessment",
            "manual",
            "output",
            "background",
        }:
            raise ValueError("Explicit memory status and evidence source required")
        text(self.content)
        if self.expires_us is not None:
            integer(self.expires_us)


@dataclass(frozen=True)
class CognitiveContext:
    claim: RunClaim
    state: StateSnapshot
    current_text: str
    # Canonical strings keep nested content immutable between duties.
    payload: str
    excluded: tuple[tuple[str, int, str], ...]

    def __post_init__(self):
        if not isinstance(self.claim, RunClaim) or not isinstance(self.state, StateSnapshot):
            raise ValueError("Typed context bindings required")
        data = json.loads(self.payload)
        if (
            data.get("claim") != asdict(self.claim)
            or data.get("state") != asdict(self.state)
            or data.get("current_text") != self.current_text
            or data.get("excluded") != [list(row) for row in self.excluded]
            or canonical(data) != self.payload
        ):
            raise ValueError("Context envelope differs from its immutable bindings")

    @property
    def fingerprint(self):
        return sha256(self.payload.encode()).hexdigest()


def assemble_context(
    claim,
    state,
    input_id,
    current_text,
    persona,
    *,
    now_us,
    history=(),
    memories=(),
    recall_refs=(),
):
    if not isinstance(claim, RunClaim) or not isinstance(state, StateSnapshot):
        raise ValueError("Typed run and state required")
    integer(now_us)
    text(current_text)
    if not current_text.strip():
        raise ValueError("Current input must be nonempty and preserved")
    text(persona)
    if input_id != claim.input_id or state.companion_id != claim.companion_id:
        raise ValueError("Cognitive identity mismatch")
    if state.write_version != claim.state_version:
        raise VersionConflict("State changed after claim")
    config = json.loads(claim.config_json)
    bound_config = CompanionConfig.from_mapping(config["companion_config"])
    if (
        bound_config.companion_id != claim.companion_id
        or config["companion_config_sha256"] != bound_config.snapshot_hash
    ):
        raise ValueError("Attempt configuration identity or hash mismatch")
    if sha256(persona.encode()).hexdigest() != config["companion_config"]["character"]["sha256"]:
        raise ValueError("Persona text differs from frozen attempt asset")
    if any(not isinstance(items, tuple) for items in (history, memories, recall_refs)):
        raise ValueError("Immutable input snapshots required")
    if len(history) > 256 or len(memories) > 100 or len(recall_refs) > 100:
        raise ValueError("Explicit A1 snapshot capacity exceeded; do not silently trim")
    selected_history, excluded, seen = [], [], set()
    for turn in history:
        if not isinstance(turn, DisplayedTurn) or turn.scope.companion_id != claim.companion_id:
            raise ValueError("Cross-companion or untyped history")
        key = (turn.scope.session_id, turn.input_id, turn.role)
        if key in seen or turn.input_id == input_id:
            raise ValueError("Duplicate history or current input repeated in history")
        seen.add(key)
        if turn.displayed_end is None or turn.displayed_end == 0:
            excluded.append((turn.input_id, 0, "unconfirmed_display"))
            continue
        selected_history.append({**asdict(turn), "text": turn.text[: turn.displayed_end]})
    current = {}
    for record in memories:
        if not isinstance(record, MemoryEvidence) or record.companion_id != claim.companion_id:
            raise ValueError("Cross-companion or untyped memory")
        if record.record_id in current:
            raise ValueError("Memory source must provide exactly one current version per record")
        current[record.record_id] = record
    selected_memory, seen = [], set()
    for ref in recall_refs:
        if not isinstance(ref, tuple) or len(ref) != 2:
            raise ValueError("Versioned recall reference required")
        key, version = ref
        identifier(key)
        integer(version, 1)
        if ref in seen:
            raise ValueError("Duplicate recall reference")
        seen.add(ref)
        record = current.get(key)
        reason = (
            "unknown_record"
            if record is None
            else "stale_version"
            if version != record.version
            else "revoked"
            if record.status != "active"
            else "expired"
            if record.expires_us is not None and record.expires_us <= now_us
            else None
        )
        if reason:
            excluded.append((key, version, reason))
        else:
            selected_memory.append(asdict(record))
    payload = canonical(
        {
            "schema_version": 1,
            "claim": asdict(claim),
            "state": asdict(state),
            "current_text": current_text,
            "persona": persona,
            "history": selected_history,
            "memories": selected_memory,
            "excluded": excluded,
            "observed_at_us": now_us,
        }
    )
    return CognitiveContext(claim, state, current_text, payload, tuple(excluded))


class CallCancelled(Exception):
    pass


class CallDeadline(Exception):
    pass


class CallControl:
    """Cooperative checkpoints for A2 adapters, not a background thread or lease scheduler.

    The caller wires durable cancel observation and TurnPort.renew. Model transports
    must interrupt blocking I/O themselves; checkpoints alone do not guarantee that.
    """

    def __init__(self, claim, *, clock, cancelled, renew, deadline_us, renewal_margin_us):
        if not isinstance(claim, RunClaim):
            raise ValueError("Typed claim required")
        integer(deadline_us)
        integer(renewal_margin_us, 1)
        self.claim, self.clock, self.cancelled, self.renew = claim, clock, cancelled, renew
        self.deadline_us, self.margin = deadline_us, renewal_margin_us
        self.last_now = None
        self.diagnostics = {"stage": "lease_control", "operations": {}, "recent": []}

    def checkpoint(self):
        return self._measure("checkpoint", self._checkpoint)

    def _measure(self, phase, operation):
        """Bounded metadata only; never retain prompts, inputs or exception messages."""
        start = time.monotonic()
        stats = self.diagnostics["operations"].setdefault(
            phase,
            {
                "count": 0,
                "failed": 0,
                "max_duration_seconds": 0,
                "max_start_gap_seconds": 0,
                "last_start_monotonic": start,
            },
        )
        stats["count"] += 1
        stats["max_start_gap_seconds"] = max(
            stats["max_start_gap_seconds"], start - stats["last_start_monotonic"]
        )
        stats["last_start_monotonic"] = start
        row = {
            "phase": phase,
            "start_monotonic": start,
            "lease_before_us": self.claim.lease_until_us,
        }
        try:
            return operation()
        except BaseException as error:
            stats["failed"] += 1
            row["error"] = type(error).__name__
            raise
        finally:
            end = time.monotonic()
            stats["max_duration_seconds"] = max(stats["max_duration_seconds"], end - start)
            row.update(
                end_monotonic=end,
                observed_now_us=self.last_now,
                lease_after_us=self.claim.lease_until_us,
            )
            recent = self.diagnostics["recent"]
            recent.append(row)
            del recent[:-16]

    def _checkpoint(self):
        if self._measure("inspect", self.cancelled):
            raise CallCancelled("Cancellation observed before more model work")
        now = self._check_time()
        if self.claim.lease_until_us - now <= self.margin:
            renewed = self._measure("renew", lambda: self.renew(self.claim, now))
            if (
                not isinstance(renewed, RunClaim)
                or replace(renewed, lease_until_us=self.claim.lease_until_us) != self.claim
                or renewed.lease_until_us <= self.claim.lease_until_us
            ):
                raise VersionConflict("Renewal changed identity or failed to extend lease")
            self.claim = renewed
        if self._measure("inspect", self.cancelled):
            raise CallCancelled("Cancellation observed during renewal")
        # Both storage operations can wait. A successful earlier check is not
        # permission to send work after the lease or absolute deadline elapsed.
        self._check_time()
        return self.claim

    def _check_time(self):
        now = self.clock()
        integer(now)
        if self.last_now is not None and now < self.last_now:
            raise VersionConflict("Call clock moved backwards")
        self.last_now = now
        if now >= self.deadline_us:
            raise CallDeadline("Cognitive time budget exhausted")
        if now >= self.claim.lease_until_us:
            raise VersionConflict("Expired owner must reconcile; never silently renew")
        return now


def classify_difference(control, candidate, rule=None):
    if canonical(control) == canonical(candidate):
        return "equivalent"
    if (
        isinstance(rule, dict)
        and rule.get("id")
        and rule.get("reason")
        and "before" in rule
        and "after" in rule
        and canonical(control) == canonical(rule["before"])
        and canonical(candidate) == canonical(rule["after"])
    ):
        return "intentional_fix"
    return "unjudged"


@dataclass(frozen=True)
class DutyAssessment:
    duty: str
    disposition: str
    context_sha256: str
    candidate_sha256: str
    reason: str
    call_id: str | None = None

    def __post_init__(self):
        if self.duty not in DUTIES or self.disposition not in {
            "complete",
            "not_required",
            "failed",
        }:
            raise ValueError("Known duty and explicit disposition required")
        identifier(self.reason)
        if self.call_id is not None:
            identifier(self.call_id)
        for digest in (self.context_sha256, self.candidate_sha256):
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise ValueError("SHA-256 binding required")


def validate_candidate(context, result, assessments, *, review_attempt):
    """Validate structural readiness only; this is not an assessment of model quality."""
    integer(review_attempt, 1)
    if review_attempt > MAX_REVIEW_ATTEMPTS:
        raise ValueError("Candidate review budget exhausted")
    if (
        not isinstance(context, CognitiveContext)
        or not isinstance(result, CognitiveResult)
        or result.claim != context.claim
        or result.presentation is None
    ):
        raise ValueError("Candidate must bind the prepared run and actual reviewed display")
    if not isinstance(assessments, tuple) or len(assessments) != len(DUTIES):
        raise ValueError("All six duties require an explicit disposition")
    seen = set()
    for item in assessments:
        if not isinstance(item, DutyAssessment) or item.duty in seen:
            raise ValueError("Duplicate or untyped duty disposition")
        seen.add(item.duty)
        if (
            item.context_sha256 != context.fingerprint
            or item.candidate_sha256 != result.presentation.candidate.fingerprint
            or item.disposition == "failed"
        ):
            raise ValueError("Duty result failed or belongs to another context/candidate")
        if item.disposition == "not_required" and item.duty not in {
            "causal_review",
            "expression_review",
        }:
            if result.text or item.duty not in {"main_reply", "expression_director"}:
                raise ValueError("Required cognitive responsibility cannot be omitted")
    return result
