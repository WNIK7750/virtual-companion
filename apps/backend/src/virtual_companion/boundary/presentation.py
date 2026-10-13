"""Review the actual immutable display surface, independently of semantic text.

Policy budgets are supplied by the caller, not new Persona defaults. Model approval
is a trusted recording/port result, not model quality evidence or authentication.

Two display representations exist. ``joined-v1`` is the legacy surface whose text
is the unit texts joined by a blank line; ``spans-v1`` binds each unit to a
contiguous code-point range of the immutable original text and inserts nothing, so
reassembling the units reproduces the original bytes exactly. Legacy mappings
never carry the new fields, which keeps their hashes byte-identical.
"""

import re
from dataclasses import asdict, dataclass
from hashlib import sha256

from .contracts import Scope, canonical, identifier

JOINED = "joined-v1"
SPANS = "spans-v1"
SEPARATOR = "\n\n"
REPRESENTATIONS = (JOINED, SPANS)


def digest(value):
    return sha256(canonical(value).encode()).hexdigest()


def semantic_bubble_ranges(text):
    """Group the source's paragraphs without editing it or splitting sentences.

    Offsets are code points, as with faithful display units. Separators belong to
    the preceding bubble; a trailing separator never creates an empty bubble.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Nonempty source text required")
    ranges, start = [], 0
    for separator in re.finditer(r"\r?\n[ \t]*(?:\r?\n)+", text):
        end = separator.end()
        if end < len(text) and text[start:end].strip():
            ranges.append({"start": start, "end": end})
            start = end
    ranges.append({"start": start, "end": len(text)})
    return ranges


def unit_mapping(unit):
    """Legacy-compatible unit mapping; spans are absent when the unit has none."""
    value = asdict(unit)
    if value["start"] is None and value["end"] is None:
        del value["start"], value["end"]
    return value


def span_units(source_text, cuts):
    """Contiguous display units ending at the given code-point cuts.

    The final cut must be ``len(source_text)``; every unit owns the characters up
    to its cut, including any separator the model itself produced.
    """
    if not isinstance(source_text, str) or not source_text:
        raise ValueError("Immutable source text required")
    if not isinstance(cuts, (tuple, list)) or not cuts:
        raise ValueError("At least one display cut required")
    units, start = [], 0
    for index, end in enumerate(cuts, 1):
        if type(end) is not int or not start < end <= len(source_text):
            raise ValueError("Ordered in-range display cuts required")
        units.append(MessageUnit(f"unit-{index}", "text", source_text[start:end], start, end))
        start = end
    if start != len(source_text):
        raise ValueError("Display cuts must cover the whole source text")
    return tuple(units)


def spanned_presentation(scope, input_id, attempt_id, candidate_id, revision, source_text,
                         cuts, *, intent="speak"):
    """Faithful candidate whose display text is exactly the immutable source text."""
    return Presentation(scope, input_id, attempt_id, candidate_id, revision, source_text,
                        span_units(source_text, cuts), intent, SPANS, source_text)


@dataclass(frozen=True)
class MessageUnit:
    unit_id: str
    kind: str
    text: str
    start: int | None = None
    end: int | None = None

    def __post_init__(self):
        identifier(self.unit_id)
        if self.kind not in {"text", "action"}:
            raise ValueError("Unknown display unit kind")
        if (
            not isinstance(self.text, str)
            or not self.text.strip()
            or len(self.text.encode()) > 64000
        ):
            raise ValueError("Nonempty bounded display text required")
        if (self.start is None) != (self.end is None):
            raise ValueError("Display range requires both bounds")
        if self.start is not None and (
            type(self.start) is not int
            or type(self.end) is not int
            or self.start < 0
            or self.end <= self.start
        ):
            raise ValueError("Bounded display range required")


@dataclass(frozen=True)
class Presentation:
    scope: Scope
    input_id: str
    attempt_id: str
    candidate_id: str
    revision: int
    semantic_text: str
    units: tuple[MessageUnit, ...]
    intent: str = "speak"
    representation: str = JOINED
    source_text: str | None = None
    response_type: str = "answer"

    def __post_init__(self):
        if not isinstance(self.scope, Scope):
            raise ValueError("Presentation scope required")
        for value in (self.input_id, self.attempt_id, self.candidate_id):
            identifier(value)
        if type(self.revision) is not int or self.revision < 1:
            raise ValueError("Positive presentation revision required")
        if not isinstance(self.semantic_text, str) or len(self.semantic_text.encode()) > 64000:
            raise ValueError("Bounded semantic text required")
        if not isinstance(self.units, tuple) or len(self.units) > 32:
            raise ValueError("At most 32 immutable display units")
        if self.intent not in {"speak", "silent", "defer"} or (
            self.intent != "speak" and (self.units or self.semantic_text)
        ):
            raise ValueError("Silent or deferred presentation cannot contain spoken text")
        if self.response_type not in {"answer", "ack"} or (
            self.response_type == "ack" and self.intent != "speak"
        ):
            raise ValueError("Explicit spoken response type required")
        if self.representation not in REPRESENTATIONS:
            raise ValueError("Unknown display representation version")
        keys = set()
        for unit in self.units:
            if not isinstance(unit, MessageUnit) or unit.unit_id in keys:
                raise ValueError("Typed uniquely identified units required")
            keys.add(unit.unit_id)
        self._validate_representation()
        if len(self.text.encode()) > 64000:
            raise ValueError("Aggregate display text exceeds UTF-8 contract")

    def _validate_representation(self):
        if self.representation == JOINED:
            if self.source_text is not None or any(
                unit.start is not None or unit.end is not None for unit in self.units
            ):
                raise ValueError("Joined display cannot carry source ranges")
            return
        if not isinstance(self.source_text, str) or not self.source_text:
            raise ValueError("Faithful display requires the immutable source text")
        if len(self.source_text.encode()) > 64000:
            raise ValueError("Faithful source text exceeds UTF-8 contract")
        if not self.units:
            raise ValueError("Faithful display requires at least one unit")
        cursor = 0
        for unit in self.units:
            if unit.start != cursor or unit.end is None or unit.end > len(self.source_text):
                raise ValueError("Faithful units must be contiguous, ordered and in range")
            if self.source_text[unit.start:unit.end] != unit.text:
                raise ValueError("Faithful unit text must equal its source range")
            cursor = unit.end
        if cursor != len(self.source_text):
            raise ValueError("Faithful units must cover the whole source text")

    @property
    def text(self):
        # Faithful display inserts nothing; the legacy surface joins with a blank line.
        if self.representation == SPANS:
            return self.source_text
        return SEPARATOR.join(unit.text for unit in self.units)

    @property
    def fingerprint(self):
        return digest(self.to_mapping())

    def to_mapping(self):
        value = {**asdict(self), "units": [unit_mapping(unit) for unit in self.units]}
        if self.intent == "speak":
            del value["intent"]  # Preserve legacy candidate hashes exactly.
        if self.response_type == "answer":
            del value["response_type"]  # Legacy spoken candidates are answers.
        if self.representation == JOINED:
            del value["representation"]
            del value["source_text"]
        return value

    def unit_end(self, unit_id):
        end = 0
        for index, unit in enumerate(self.units):
            end += (2 if index else 0) + len(unit.text)
            if unit.unit_id == unit_id:
                return end
        raise ValueError("Unknown display unit")


@dataclass(frozen=True)
class PresentationPolicy:
    max_exclamations: int
    max_unit_codepoints: int
    max_units: int
    allow_actions: bool

    def __post_init__(self):
        for value, minimum, maximum in (
            (self.max_exclamations, 0, 64000),
            (self.max_unit_codepoints, 1, 64000),
            (self.max_units, 0, 32),
        ):
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError("Invalid explicit presentation budget")
        if type(self.allow_actions) is not bool:
            raise ValueError("Explicit action policy required")

    @property
    def fingerprint(self):
        return digest(asdict(self))


@dataclass(frozen=True)
class PresentationReview:
    review_id: str
    candidate_sha256: str
    policy_sha256: str
    model_approved: bool
    issues: tuple[str, ...]
    approval_source: str = "model"
    policy_revision: str | None = None

    def __post_init__(self):
        identifier(self.review_id)
        for value in (self.candidate_sha256, self.policy_sha256):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError("Review hash required")
        if type(self.model_approved) is not bool or not isinstance(self.issues, tuple):
            raise ValueError("Immutable review result required")
        for issue in self.issues:
            identifier(issue)
        if self.approval_source == "host_policy":
            identifier(self.policy_revision)
            if self.model_approved:
                raise ValueError("Host policy cannot fabricate a model approval")
        elif self.approval_source != "model" or self.policy_revision is not None:
            raise ValueError("Unknown presentation approval contract")

    @property
    def passed(self):
        return (self.model_approved or self.approval_source == "host_policy") and not self.issues


def surface_issues(candidate, policy):
    issues = []
    if (
        sum(unit.text.count("!") + unit.text.count("！") for unit in candidate.units)
        > policy.max_exclamations
    ):
        issues.append("exclamation_budget")
    if len(candidate.units) > policy.max_units:
        issues.append("unit_count")
    if any(len(unit.text) > policy.max_unit_codepoints for unit in candidate.units):
        issues.append("unit_length")
    if not policy.allow_actions and any(unit.kind == "action" for unit in candidate.units):
        issues.append("actions_not_allowed")
    return tuple(issues)


def review(candidate, policy, review_id, *, model_approved):
    if not isinstance(candidate, Presentation) or not isinstance(policy, PresentationPolicy):
        raise ValueError("Typed candidate and policy required")
    return PresentationReview(
        review_id,
        candidate.fingerprint,
        policy.fingerprint,
        model_approved,
        surface_issues(candidate, policy),
    )


def review_by_policy(candidate, policy, review_id, *, policy_revision):
    """Trusted host risk policy, not a claim that an LLM reviewed this surface.

    The caller must establish admission before invoking this helper. As with the
    existing model review Port, this receipt binds a decision, not authentication.
    """
    if not isinstance(candidate, Presentation) or not isinstance(policy, PresentationPolicy):
        raise ValueError("Typed candidate and policy required")
    return PresentationReview(
        review_id, candidate.fingerprint, policy.fingerprint, False,
        surface_issues(candidate, policy), "host_policy", policy_revision,
    )


@dataclass(frozen=True)
class ReviewedPresentation:
    candidate: Presentation
    policy: PresentationPolicy
    review: PresentationReview

    def __post_init__(self):
        if (
            not isinstance(self.candidate, Presentation)
            or not isinstance(self.policy, PresentationPolicy)
            or not isinstance(self.review, PresentationReview)
        ):
            raise ValueError("Typed display review binding required")
        if (
            self.review.candidate_sha256 != self.candidate.fingerprint
            or self.review.policy_sha256 != self.policy.fingerprint
            or not self.review.passed
            or surface_issues(self.candidate, self.policy)
        ):
            raise ValueError("Display review rejected or binding changed")

    def to_mapping(self):
        verdict = {**asdict(self.review), "issues": list(self.review.issues)}
        if self.review.approval_source == "model":
            # Preserve the exact legacy serialization and its persisted hashes.
            del verdict["approval_source"]
            del verdict["policy_revision"]
        return {
            "candidate": self.candidate.to_mapping(),
            "policy": asdict(self.policy),
            "review": verdict,
        }

    @classmethod
    def from_mapping(cls, value):
        candidate = dict(value["candidate"])
        candidate["scope"] = Scope(**candidate["scope"])
        candidate["units"] = tuple(MessageUnit(**unit) for unit in candidate["units"])
        candidate.setdefault("representation", JOINED)
        candidate.setdefault("source_text", None)
        verdict = dict(value["review"])
        verdict["issues"] = tuple(verdict["issues"])
        return cls(
            Presentation(**candidate),
            PresentationPolicy(**value["policy"]),
            PresentationReview(**verdict),
        )


def verify_delivery(text, receipt):
    """Revalidate persisted review/manifest before publishing or acknowledging it."""
    if sha256(text.encode()).hexdigest() != receipt["sha256"]:
        raise ValueError("Persisted delivery text digest mismatch")
    if "presentation" not in receipt:
        if "presentation_sha256" in receipt:
            raise ValueError("Missing presentation manifest")
        return
    bound = ReviewedPresentation.from_mapping(receipt["presentation"])
    candidate = bound.candidate
    if (
        text != candidate.text
        or receipt.get("presentation_sha256") != candidate.fingerprint
        or receipt["companion_id"] != candidate.scope.companion_id
        or receipt["input_id"] != candidate.input_id
        or receipt["attempt_id"] != candidate.attempt_id
        or receipt["candidate_id"] != candidate.candidate_id
        or receipt["revision"] != candidate.revision
        or receipt["review_id"] != bound.review.review_id
    ):
        raise ValueError("Persisted delivery does not match reviewed presentation")
