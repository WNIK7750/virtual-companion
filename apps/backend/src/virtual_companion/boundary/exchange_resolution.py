"""Host-bound evidence that one source's exchange interpretation is complete.

This is a semantic assessment receipt, not an authority to bypass domain support.
"""

from dataclasses import asdict, dataclass
from hashlib import sha256

from virtual_companion.boundary.contracts import canonical, identifier

PROTOCOL = "source-resolution.v1"
NAMED_TARGETS_PROTOCOL = "source-resolution.v2"


def digest(value):
    return sha256(canonical(value).encode()).hexdigest()


def _sha(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


@dataclass(frozen=True)
class CandidateAssessment:
    candidate_id: str
    candidate_sha256: str
    disposition: str
    reason: str

    def __post_init__(self):
        identifier(self.candidate_id)
        if not _sha(self.candidate_sha256):
            raise ValueError("Bound candidate digest required")
        if self.disposition not in {"accepted", "rejected", "unresolved"}:
            raise ValueError("Explicit candidate disposition required")
        if self.reason not in {
            "supported",
            "statement",
            "quotation",
            "decline",
            "already_answered",
            "already_pending",
            "unsupported",
            "unclear",
        }:
            raise ValueError("Known assessment reason required")
        if self.disposition == "accepted" and self.reason != "supported":
            raise ValueError("Accepted candidate needs positive assessment")
        if self.disposition == "rejected" and self.reason in {"supported", "unclear"}:
            raise ValueError("Rejected candidate needs a definite negative reason")
        if self.disposition == "unresolved" and self.reason != "unclear":
            raise ValueError("Unresolved candidate needs explicit uncertainty")


@dataclass(frozen=True)
class SourceResolution:
    source_id: str
    source_sha256: str
    context_sha256: str
    targets_sha256: str
    candidates_sha256: str
    assessments: tuple[CandidateAssessment, ...]
    effects_sha256: str
    support_sha256: str
    no_exchange: bool
    evaluator: str
    protocol: str = PROTOCOL
    coverage: str = "complete"
    uncovered: tuple = ()
    # v1 certifies the complete catalog; v2 certifies only the supplied targets.
    target_refs: tuple[tuple[str, int], ...] | None = None

    def __post_init__(self):
        identifier(self.source_id)
        identifier(self.evaluator)
        if (
            self.protocol not in {PROTOCOL, NAMED_TARGETS_PROTOCOL}
            or type(self.no_exchange) is not bool
            or self.coverage != "complete"
            or self.uncovered != ()
        ):
            raise ValueError("Explicit current source resolution protocol required")
        if self.protocol == PROTOCOL:
            if self.target_refs is not None:
                raise ValueError("Legacy resolution cannot carry a named target scope")
        else:
            if not isinstance(self.target_refs, tuple) or len(self.target_refs) > 100:
                raise ValueError("Bounded explicit target references required")
            for reference in self.target_refs:
                if (not isinstance(reference, tuple) or len(reference) != 2
                        or type(reference[1]) is not int or reference[1] < 1):
                    raise ValueError("Typed target identity and revision required")
                identifier(reference[0])
            if (tuple(sorted(self.target_refs)) != self.target_refs
                    or len({ref[0] for ref in self.target_refs}) != len(self.target_refs)):
                raise ValueError("Ordered unique target references required")
        if any(
            not _sha(value)
            for value in (
                self.source_sha256,
                self.context_sha256,
                self.targets_sha256,
                self.candidates_sha256,
                self.effects_sha256,
                self.support_sha256,
            )
        ):
            raise ValueError("Bound source resolution digests required")
        if (
            not isinstance(self.assessments, tuple)
            or len(self.assessments) > 16
            or any(not isinstance(item, CandidateAssessment) for item in self.assessments)
            or len({item.candidate_id for item in self.assessments}) != len(self.assessments)
            or any(item.disposition == "unresolved" for item in self.assessments)
            or self.no_exchange != all(item.disposition != "accepted" for item in self.assessments)
        ):
            raise ValueError("Complete settled candidate assessment required")

    @property
    def fingerprint(self):
        return digest(self.to_mapping())

    @property
    def status(self):
        return "complete_no_exchange" if self.no_exchange else "complete_with_effects"

    def to_mapping(self):
        value = {**asdict(self), "status": self.status}
        if self.protocol == PROTOCOL:
            value.pop("target_refs")  # Preserve historical v1 receipt fingerprints.
        return value
