"""Review opinion adjudication, separate from final candidate approval.

Re-expresses the pinned control's grounding truth table over the new immutable
display hash. It does not implement the deterministic semantic judges themselves.
A caller must supply their results; no missing judge silently defaults to passing.
"""

from dataclasses import asdict, dataclass

from virtual_companion.boundary.presentation import Presentation

DUTIES = ("causal_review", "expression_review")


class ReviewContractError(ValueError):
    """Malformed model review, distinct from an internal judge-port failure."""


def fields(value, names):
    if not isinstance(value, dict) or set(value) != set(names.split()):
        raise ValueError("Exact review object fields required")


def reason(value):
    if not isinstance(value, str) or not value.strip() or len(value.encode()) > 64000:
        raise ValueError("Bounded review reason required")


@dataclass(frozen=True)
class ReviewOpinion:
    candidate_sha256: str
    passed: bool
    evidence_quotes: tuple[str, ...]
    reason: str

    def __post_init__(self):
        digest = self.candidate_sha256
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("Review candidate hash required")
        if type(self.passed) is not bool:
            raise ValueError("Boolean review required")
        reason(self.reason)
        if not isinstance(self.evidence_quotes, tuple) or len(self.evidence_quotes) > 8:
            raise ValueError("Bounded immutable evidence quotes required")
        if any(not isinstance(q, str) or len(q.encode()) > 64000 for q in self.evidence_quotes):
            raise ValueError("Bounded quote text required")


@dataclass(frozen=True)
class Adjudication:
    passed: bool
    disposition: str
    candidate_sha256: str
    model_received: bool
    candidate_matches: bool
    evidence_valid: bool


def adjudicate(candidate, deterministic_passed, opinion):
    if not isinstance(candidate, Presentation) or type(deterministic_passed) is not bool:
        raise ValueError("Typed surface and explicit deterministic verdict required")
    if opinion is None:
        return Adjudication(
            deterministic_passed,
            "accepted" if deterministic_passed else "rejected",
            candidate.fingerprint,
            False,
            True,
            True,
        )
    if not isinstance(opinion, ReviewOpinion):
        raise ValueError("Typed model opinion required")
    matches = opinion.candidate_sha256 == candidate.fingerprint
    grounded = bool(opinion.evidence_quotes) and all(
        quote.strip() and any(quote in unit.text for unit in candidate.units)
        for quote in opinion.evidence_quotes
    )
    if opinion.passed:
        # "accepted" describes receiving a positive opinion, not permission to
        # display. The independent deterministic failure remains a failure.
        disposition = "accepted" if matches else "ignored_ungrounded"
        passed = deterministic_passed
    elif matches and grounded:
        disposition, passed = "rejected", False
    else:
        disposition = "ignored_ungrounded" if deterministic_passed else "rejected"
        passed = deterministic_passed
    return Adjudication(passed, disposition, candidate.fingerprint, True, matches, grounded)


def envelope(context, candidate, verdict):
    try:
        fields(verdict, "context_sha256 candidate_sha256 causal_review expression_review")
    except ValueError as error:
        raise ReviewContractError(str(error)) from error
    if (
        verdict["context_sha256"] != context.fingerprint
        or verdict["candidate_sha256"] != candidate.fingerprint
    ):
        raise ReviewContractError("Foreign review envelope")


class StrictReviewGate:
    """Retain the initial opt-in A2 boolean contract for existing callers."""

    @property
    def response_contract(self):
        return {
            duty: {"passed": "boolean", "reason": "grounded review assessment"} for duty in DUTIES
        }

    def evaluate(self, context, candidate, verdict):
        envelope(context, candidate, verdict)
        result = {}
        for duty in DUTIES:
            try:
                fields(verdict[duty], "passed reason")
                if type(verdict[duty]["passed"]) is not bool:
                    raise ValueError("Boolean verdict required")
                reason(verdict[duty]["reason"])
            except ValueError as error:
                raise ReviewContractError(str(error)) from error
            result[duty] = {
                "passed": verdict[duty]["passed"],
                "disposition": "strict",
                "candidate_sha256": candidate.fingerprint,
            }
        return result


class GroundedReviewGate:
    """Explicit alternative for comparison; needs a deterministic two-judge port.

    The expression opinion's passed value means hard semantic/format checks,
    excluding advisory preferred unit count. No model label replaces hard checks.
    """

    def __init__(self, deterministic):
        if not callable(deterministic):
            raise ValueError("Deterministic review port required")
        self.deterministic = deterministic

    @property
    def response_contract(self):
        return {
            duty: {
                "passed": "boolean; expression: hard checks only, exclude advisory unit count",
                "reason": "review assessment",
                "candidate_sha256": "hash of the candidate actually reviewed",
                "evidence_quotes": [
                    "exact quote from one actual display unit; required for rejection"
                ],
            }
            for duty in DUTIES
        }

    def evaluate(self, context, candidate, verdict):
        envelope(context, candidate, verdict)
        deterministic = self.deterministic(context, candidate)
        if (
            not isinstance(deterministic, tuple)
            or len(deterministic) != 2
            or any(type(v) is not bool for v in deterministic)
        ):
            raise ValueError("Both deterministic judge outcomes required")
        result = {}
        for index, duty in enumerate(DUTIES):
            row = verdict[duty]
            try:
                fields(row, "passed reason candidate_sha256 evidence_quotes")
                if not isinstance(row["evidence_quotes"], list):
                    raise ValueError("JSON quote array required")
                opinion = ReviewOpinion(
                    row["candidate_sha256"],
                    row["passed"],
                    tuple(row["evidence_quotes"]),
                    row["reason"],
                )
            except ValueError as error:
                raise ReviewContractError(str(error)) from error
            result[duty] = asdict(adjudicate(candidate, deterministic[index], opinion))
        return result
