"""Immutable synthetic intake/run/commit contracts; no model or storage imports.

Internal assessments commit with the turn. Optional memory effects declare their
input or confirmed-output timing; only the opt-in memory adapter accepts them.
"""

from dataclasses import asdict, dataclass
from hashlib import sha256

from .contracts import OwnerToken, Scope, StateEvent, canonical, companion_id, identifier
from .evidence import EvidenceTurn, evidence_digest
from .exchanges import ExchangeCommand, ExchangeSupport
from .memory import MemoryEffect
from .presentation import ReviewedPresentation


@dataclass(frozen=True)
class IntakeReceipt:
    companion_id: str
    session_id: str
    input_id: str
    client_key: str
    request_hash: str
    config_json: str


@dataclass(frozen=True)
class RunClaim:
    companion_id: str
    session_id: str
    input_id: str
    attempt_id: str
    owner: OwnerToken
    state_version: int
    config_json: str
    lease_until_us: int

    def __post_init__(self):
        companion_id(self.companion_id)
        for value in (self.session_id, self.input_id, self.attempt_id):
            identifier(value)
        if not isinstance(self.owner, OwnerToken) or self.owner.companion_id != self.companion_id:
            raise ValueError("Claim owner must belong to companion")
        if (
            type(self.state_version) is not int
            or self.state_version < 0
            or type(self.lease_until_us) is not int
            or not 0 <= self.lease_until_us < 2**63
            or not isinstance(self.config_json, str)
        ):
            raise ValueError("Invalid claim version, lease or configuration")


@dataclass(frozen=True)
class CognitiveResult:
    claim: RunClaim
    candidate_id: str
    revision: int
    text: str
    review_id: str
    reviewed_sha256: str
    assessment: StateEvent | None = None
    memories: tuple[MemoryEffect, ...] = ()
    presentation: ReviewedPresentation | None = None
    relationships: tuple[()] = ()
    exchanges: tuple[tuple[ExchangeCommand, ExchangeSupport], ...] = ()
    exchange_sources: tuple[EvidenceTurn, ...] = ()

    def __post_init__(self):
        if not isinstance(self.claim, RunClaim):
            raise ValueError("Run claim required")
        identifier(self.candidate_id)
        identifier(self.review_id)
        if type(self.revision) is not int or self.revision < 1:
            raise ValueError("Positive candidate revision required")
        if not isinstance(self.text, str) or len(self.text.encode()) > 64000:
            raise ValueError("Text exceeds synthetic UTF-8 contract")
        # Empty text represents a deliberate silence, not an unreviewed fallback.
        if sha256(self.text.encode()).hexdigest() != self.reviewed_sha256:
            raise ValueError("Candidate does not match approved review hash")
        if self.presentation is not None:
            if not isinstance(self.presentation, ReviewedPresentation):
                raise ValueError("Typed presentation review required")
            candidate = self.presentation.candidate
            if (
                candidate.scope.companion_id != self.claim.companion_id
                or candidate.scope.session_id != self.claim.session_id
                or candidate.input_id != self.claim.input_id
                or candidate.attempt_id != self.claim.attempt_id
                or candidate.candidate_id != self.candidate_id
                or candidate.revision != self.revision
                or candidate.text != self.text
                or self.presentation.review.review_id != self.review_id
            ):
                raise ValueError("Cognitive result and presentation binding mismatch")
        if self.assessment is not None and (
            not isinstance(self.assessment, StateEvent)
            or self.assessment.scope.companion_id != self.claim.companion_id
            or self.assessment.scope.session_id != self.claim.session_id
            or self.assessment.kind != "state"
        ):
            raise ValueError("Internal state assessment must belong to this run")
        if not isinstance(self.memories, tuple) or len(self.memories) > 32:
            raise ValueError("At most 32 immutable memory effects per result")
        keys = set()
        for effect in self.memories:
            if not isinstance(effect, MemoryEffect):
                raise ValueError("Typed memory effect required")
            mutation = effect.mutation
            if (
                mutation.scope.companion_id != self.claim.companion_id
                or mutation.scope.session_id != self.claim.session_id
                or mutation.mutation_id in keys
                or (
                    effect.after_codepoints is not None
                    and (
                        effect.after_codepoints > len(self.text)
                        or mutation.source_id != self.candidate_id
                    )
                )
            ):
                raise ValueError("Memory effect scope, identity or presentation range mismatch")
            keys.add(mutation.mutation_id)
        if not isinstance(self.relationships, tuple) or self.relationships:
            raise ValueError("Retired relationship effects are unsupported")
        if not isinstance(self.exchanges, tuple) or len(self.exchanges) > 16:
            raise ValueError("Bounded immutable exchange effects required")
        if not isinstance(self.exchange_sources, tuple) or len(self.exchange_sources) > 32:
            raise ValueError("Bounded immutable exchange context required")
        scope = (
            Scope(self.claim.companion_id, self.claim.session_id)
            if self.exchange_sources or self.exchanges else None
        )
        source_map = {}
        for source in self.exchange_sources:
            if (
                not isinstance(source, EvidenceTurn)
                or source.scope != scope
                or source.source_id in source_map
                or source.confirmed_end != len(source.text)
            ):
                raise ValueError("Unique complete confirmed exchange context required")
            source_map[source.source_id] = source
        if self.exchange_sources and (
            self.exchange_sources[-1].source_id != self.claim.input_id
            or self.exchange_sources[-1].role != "user"
        ):
            raise ValueError("Exchange context must end at the current user input")
        context_sha256 = (
            evidence_digest(self.exchange_sources)
            if self.exchanges and self.exchange_sources else None
        )
        keys = set()
        for pair in self.exchanges:
            if (
                not isinstance(pair, tuple)
                or len(pair) != 2
                or not isinstance(pair[0], ExchangeCommand)
                or not isinstance(pair[1], ExchangeSupport)
            ):
                raise ValueError("Typed exchange command and support required")
            command, support = pair
            if (
                command.scope != scope
                or command.command_id in keys
                or support.command_sha256 != command.fingerprint
            ):
                raise ValueError("Current user-source exchange effects only; output must await ACK")
            if self.exchange_sources:
                source = source_map.get(command.evidence.source_id)
                if (
                    source is None
                    or source.role != command.evidence.role
                    or source.confirmed_end != command.evidence.confirmed_end
                    or sha256(source.text.encode()).hexdigest() != command.evidence.prefix_sha256
                    or support.context_sha256 != context_sha256
                ):
                    raise ValueError("Exchange effect must bind its full actual conversation")
            elif (
                command.evidence.role != "user" or command.evidence.source_id != self.claim.input_id
            ):
                raise ValueError("Current user-source exchange effects only; output must await ACK")
            keys.add(command.command_id)

    @property
    def payload(self):
        assessment = asdict(self.assessment) if self.assessment else None
        if assessment:
            assessment["occurred_at"] = self.assessment.occurred_at.isoformat()
            assessment["received_at"] = self.assessment.received_at.isoformat()
        return canonical(
            {
                "claim": asdict(self.claim),
                "candidate_id": self.candidate_id,
                "revision": self.revision,
                "text": self.text,
                "review_id": self.review_id,
                "reviewed_sha256": self.reviewed_sha256,
                "assessment": assessment,
                **(
                    {"exchange_sources": [asdict(t) for t in self.exchange_sources]}
                    if self.exchange_sources
                    else {}
                ),
                **(
                    {
                        "exchanges": [
                            {"command": asdict(c), "support": s.to_mapping()}
                            for c, s in self.exchanges
                        ]
                    }
                    if self.exchanges
                    else {}
                ),
                **({"presentation": self.presentation.to_mapping()} if self.presentation else {}),
                **(
                    {
                        "memories": [
                            {
                                "mutation": e.mutation.to_mapping(),
                                "after_codepoints": e.after_codepoints,
                            }
                            for e in self.memories
                        ]
                    }
                    if self.memories
                    else {}
                ),
            }
        )

    @property
    def fingerprint(self):
        # Ownership/lease are transient. Recovery queries the original business identity.
        return sha256(
            canonical(
                [
                    self.claim.companion_id,
                    self.claim.session_id,
                    self.claim.input_id,
                    self.claim.attempt_id,
                    self.claim.config_json,
                    self.candidate_id,
                    self.revision,
                    self.text,
                    self.review_id,
                    self.reviewed_sha256,
                    self.assessment.fingerprint if self.assessment else None,
                    *(
                        [["exchange_sources", evidence_digest(self.exchange_sources)]]
                        if self.exchange_sources
                        else []
                    ),
                    *(
                        [
                            [
                                "exchanges",
                                [[c.fingerprint, s.to_mapping()] for c, s in self.exchanges],
                            ]
                        ]
                        if self.exchanges
                        else []
                    ),
                    *([[e.mutation.fingerprint, e.after_codepoints] for e in self.memories]),
                    *([self.presentation.to_mapping()] if self.presentation else []),
                ]
            ).encode()
        ).hexdigest()
