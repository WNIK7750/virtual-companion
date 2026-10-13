"""Evidence-bound pending questions and assistant responses, separate from facts.

The caller obtains semantic support through a trusted evaluator before writing.
Storage verifies actual source identity/prefix, state ownership and target revision.
Neither silence nor elapsed time is evidence that a party answered or consented.
"""

from dataclasses import asdict, dataclass
from hashlib import sha256

from .contracts import Scope, canonical, identifier


@dataclass(frozen=True)
class ExchangeEvidence:
    role: str
    source_id: str
    quote: str
    confirmed_end: int
    prefix_sha256: str

    def __post_init__(self):
        identifier(self.source_id)
        if self.role not in {"user", "assistant"}:
            raise ValueError("Explicit source role required")
        if not isinstance(self.quote, str) or not self.quote.strip() or len(self.quote) > 16000:
            raise ValueError("Bounded exact evidence quote required")
        if type(self.confirmed_end) is not int or not 0 < self.confirmed_end <= 64000:
            raise ValueError("Bounded confirmed prefix required")
        if (
            not isinstance(self.prefix_sha256, str)
            or len(self.prefix_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.prefix_sha256)
        ):
            raise ValueError("Source prefix digest required")


@dataclass(frozen=True)
class ExchangeCommand:
    scope: Scope
    command_id: str
    exchange_id: str
    action: str
    direction: str
    content: str
    evidence: ExchangeEvidence
    occurred_us: int
    expected_revision: int | None
    expires_us: int | None

    def __post_init__(self):
        if not isinstance(self.scope, Scope) or not isinstance(self.evidence, ExchangeEvidence):
            raise ValueError("Typed exchange scope and evidence required")
        for key in (self.command_id, self.exchange_id):
            identifier(key)
        if self.action not in {"open", "resolve", "defer", "resume", "close"}:
            raise ValueError("Explicit exchange action required")
        if self.direction not in {"user_question", "assistant_response"}:
            raise ValueError("Explicit exchange direction required")
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("Exchange content required")
        if len(self.content.encode()) > 16000:
            raise ValueError("Bounded exchange content required")
        if type(self.occurred_us) is not int or self.occurred_us < 0:
            raise ValueError("UTC occurrence required")
        if self.expires_us is not None and (
            type(self.expires_us) is not int or self.expires_us <= self.occurred_us
        ):
            raise ValueError("Explicit future expiry required")
        if self.action == "open":
            if self.expected_revision is not None:
                raise ValueError("New exchange cannot claim a previous revision")
            if self.direction == "user_question" and self.evidence.role != "assistant":
                raise ValueError("A pending user question needs an actual assistant question")
        elif type(self.expected_revision) is not int or self.expected_revision < 1:
            raise ValueError("Current target revision required")

    @property
    def fingerprint(self):
        return sha256(canonical(asdict(self)).encode()).hexdigest()


@dataclass(frozen=True)
class ExchangeSupport:
    command_sha256: str
    evaluator: str
    context_sha256: str | None = None

    def __post_init__(self):
        identifier(self.evaluator)
        if (
            not isinstance(self.command_sha256, str)
            or len(self.command_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.command_sha256)
        ):
            raise ValueError("Bound semantic support digest required")
        if self.context_sha256 is not None and (
            not isinstance(self.context_sha256, str)
            or len(self.context_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.context_sha256)
        ):
            raise ValueError("Bound conversation support digest required")

    def to_mapping(self):
        # Keep legacy single-source receipt identities unchanged when no context
        # binding was recorded. New contextual judgments explicitly include it.
        return {
            "command_sha256": self.command_sha256,
            "evaluator": self.evaluator,
            **({"context_sha256": self.context_sha256} if self.context_sha256 is not None else {}),
        }


def evaluate_exchange(command, *, support, confirmed_text):
    """Trusted semantic Port; quote matching alone never establishes the action.

    Include the complete confirmed source so quotes inside negation or reported
    speech cannot silently lose their context. Unacknowledged assistant tails
    are excluded. Storage independently verifies the source before writing.
    """
    if not isinstance(command, ExchangeCommand) or not callable(support):
        raise ValueError("Typed command and semantic support evaluator required")
    evidence = command.evidence
    if (
        not isinstance(confirmed_text, str)
        or len(confirmed_text) != evidence.confirmed_end
        or sha256(confirmed_text.encode()).hexdigest() != evidence.prefix_sha256
        or evidence.quote not in confirmed_text
    ):
        raise ValueError("Semantic support needs the bound complete confirmed source")
    verdict = support({"command": asdict(command), "confirmed_text": confirmed_text})
    if (
        not isinstance(verdict, dict)
        or set(verdict) != {"verdict", "command_sha256", "evaluator"}
        or verdict["verdict"] != "supported"
        or verdict["command_sha256"] != command.fingerprint
    ):
        raise ValueError("Unknown, unsupported or unbound exchange support")
    return ExchangeSupport(verdict["command_sha256"], verdict["evaluator"])
