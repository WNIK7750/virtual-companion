"""ARC-002 single-owner text boundary pilot, not the current streaming default.

ACKs confirm a cumulative Unicode-code-point prefix, never audio playback.
The commit callback must atomically deduplicate by the supplied idempotency key.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime

from .chronology import utc_microseconds


def _identifier(value):
    if not isinstance(value, str) or not 1 <= len(value) <= 128:
        raise ValueError("Invalid output identity")


@dataclass(frozen=True)
class TextCandidate:
    turn_id: str
    attempt_id: str
    text: str
    revision: int

    def __post_init__(self):
        _identifier(self.turn_id)
        _identifier(self.attempt_id)
        if not isinstance(self.text, str) or not self.text.strip() or len(self.text) > 64000:
            raise ValueError("Invalid candidate text")
        if type(self.revision) is not int or self.revision < 1:
            raise ValueError("Invalid candidate revision")

    @property
    def sha256(self):
        return hashlib.sha256(self.text.encode()).hexdigest()


@dataclass(frozen=True)
class PresentationAck:
    attempt_id: str
    sequence: int
    displayed_end: int
    received_at: datetime
    candidate_revision: int
    candidate_sha256: str


@dataclass(frozen=True)
class TextCommit:
    idempotency_key: str
    turn_id: str
    attempt_id: str
    revision: int
    candidate_sha256: str
    review_id: str
    text: str


class TextOutputBoundary:
    def __init__(self, candidate: TextCandidate):
        if not isinstance(candidate, TextCandidate):
            raise ValueError("TextCandidate required")
        self._candidate = candidate
        self.status = "generated"
        self.displayed_end = 0
        self._review = None
        self._acks = {}
        self._last_sequence = 0
        self._commit = None

    @property
    def candidate(self):
        return self._candidate

    def review(self, evidence_id, candidate_sha256, *, approved):
        _identifier(evidence_id)
        if type(approved) is not bool or candidate_sha256 != self.candidate.sha256:
            raise ValueError("Review does not bind the candidate")
        review = (evidence_id, candidate_sha256, approved)
        if self._review == review:
            return
        if self._review is not None or self.status != "generated":
            raise ValueError("Review replacement requires a new candidate")
        self._review = review
        self.status = "approved" if approved else "rejected"

    def acknowledge(self, ack: PresentationAck):
        if not isinstance(ack, PresentationAck):
            raise ValueError("PresentationAck required")
        _identifier(ack.attempt_id)
        utc_microseconds(ack.received_at)
        if (
            type(ack.sequence) is not int
            or ack.sequence < 1
            or type(ack.displayed_end) is not int
            or not 0 <= ack.displayed_end <= len(self.candidate.text)
        ):
            raise ValueError("Invalid presentation sequence or range")
        if ack.attempt_id != self.candidate.attempt_id:
            return "foreign_attempt"
        if (
            type(ack.candidate_revision) is not int
            or ack.candidate_revision != self.candidate.revision
            or ack.candidate_sha256 != self.candidate.sha256
        ):
            return "foreign_candidate"
        previous = self._acks.get(ack.sequence)
        if previous is not None:
            if previous.displayed_end != ack.displayed_end:
                raise ValueError("Conflicting acknowledgement sequence")
            return "duplicate"
        if self.status in {"cancelled", "rejected", "committed", "commit_pending"}:
            return "terminal"
        if self.status != "approved":
            raise ValueError("Presentation requires approved review")
        if ack.sequence < self._last_sequence:
            return "late_sequence"
        if ack.displayed_end < self.displayed_end:
            raise ValueError("Presentation range cannot move backwards")
        self._acks[ack.sequence] = ack
        self._last_sequence = ack.sequence
        self.displayed_end = ack.displayed_end
        return "accepted"

    def cancel(self):
        if self.status == "commit_pending":
            raise RuntimeError("Resolve uncertain commit before cancellation")
        if self.status not in {"committed", "rejected"}:
            self.status = "cancelled"

    def commit(self, writer):
        if self.status == "committed":
            return self._commit
        if not callable(writer):
            raise ValueError("Idempotent commit writer required")
        if self.status not in {"approved", "commit_pending"} or self.displayed_end != len(
            self.candidate.text
        ):
            raise ValueError("Full approved presentation is required for commit")
        if self._commit is None:
            candidate = self.candidate
            key = hashlib.sha256(
                json.dumps(
                    [
                        candidate.turn_id,
                        candidate.attempt_id,
                        candidate.revision,
                        candidate.sha256,
                        self._review[0],
                    ]
                ).encode()
            ).hexdigest()
            self._commit = TextCommit(
                key,
                candidate.turn_id,
                candidate.attempt_id,
                candidate.revision,
                candidate.sha256,
                self._review[0],
                candidate.text,
            )
        # A failed response does not prove that the durable transaction rolled back.
        self.status = "commit_pending"
        writer(self._commit)
        self.status = "committed"
        return self._commit
