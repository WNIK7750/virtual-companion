"""Typed host-bound commands for explicit user memory management.

The model may propose meaning, but only the host fills record identity, version,
scope and source identity.  These commands are deliberately narrower than the
automatic fact extraction protocol.
"""

import re
from dataclasses import asdict, dataclass
from hashlib import sha256

from .contracts import Scope, canonical, identifier


@dataclass(frozen=True)
class ExplicitMemoryCommand:
    scope: Scope
    operation_id: str
    action: str
    record_id: str
    expected_record_version: int
    entry: str
    source_id: str
    source_sha256: str
    requested_scope: str
    source_coverage: str
    interpreter: str
    interpretation_sha256: str
    evidence_quote: str = ""
    quote_start: int | None = None
    quote_end: int | None = None
    replacement: str = ""
    resolves_job: str | None = None

    def __post_init__(self):
        if not isinstance(self.scope, Scope):
            raise ValueError("Memory command scope required")
        for value in (self.operation_id, self.record_id, self.source_id):
            identifier(value)
        identifier(self.interpreter)
        if self.resolves_job is not None:
            identifier(self.resolves_job)
        if self.action not in {"correct", "forget"}:
            raise ValueError("Explicit memory command action required")
        if self.entry not in {"conversation", "list"}:
            raise ValueError("Explicit memory command entry required")
        if type(self.expected_record_version) is not int or self.expected_record_version < 1:
            raise ValueError("Positive target version required")
        if not isinstance(self.source_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.source_sha256
        ):
            raise ValueError("Bound source digest required")
        if not isinstance(self.interpretation_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.interpretation_sha256
        ):
            raise ValueError("Bound memory interpretation digest required")
        if self.requested_scope != "single_record":
            raise ValueError("Only single-record memory management is supported")
        expected_coverage = {
            "conversation": "pure_command",
            "list": "explicit_selection",
        }[self.entry]
        if self.source_coverage != expected_coverage:
            raise ValueError("Memory command source coverage differs from its entry")
        if not isinstance(self.evidence_quote, str) or len(self.evidence_quote.encode()) > 16000:
            raise ValueError("Bounded memory command evidence required")
        if not isinstance(self.replacement, str) or len(self.replacement.encode()) > 16000:
            raise ValueError("Bounded memory replacement required")
        if self.entry == "conversation":
            if (
                not self.evidence_quote
                or type(self.quote_start) is not int
                or type(self.quote_end) is not int
                or not 0 <= self.quote_start < self.quote_end
            ):
                raise ValueError("Conversation command requires exact source evidence")
        elif self.evidence_quote or self.quote_start is not None or self.quote_end is not None:
            raise ValueError("List command identity must not imitate dialogue evidence")
        if self.action == "forget" and self.replacement:
            raise ValueError("Forget command cannot assert replacement content")
        if self.action == "correct" and not self.replacement.strip():
            raise ValueError("Correction requires replacement content")

    def to_mapping(self):
        return asdict(self)

    @property
    def fingerprint(self):
        return sha256(canonical(self.to_mapping()).encode()).hexdigest()


def list_source_digest(*, action, record_id, expected_version, replacement=""):
    """Bind a UI action to the stable selection without inventing a chat quote."""
    return sha256(
        canonical(
            {
                "action": action,
                "record_id": record_id,
                "expected_version": expected_version,
                "replacement": replacement,
            }
        ).encode()
    ).hexdigest()
