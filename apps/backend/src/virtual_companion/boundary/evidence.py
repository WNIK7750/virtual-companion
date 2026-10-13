"""Typed observed source text shared by storage and cognition."""

from dataclasses import dataclass
from hashlib import sha256

from .contracts import Scope, canonical, identifier


@dataclass(frozen=True)
class EvidenceTurn:
    scope: Scope
    source_id: str
    role: str
    text: str
    confirmed_end: int
    response_type: str | None = None

    def __post_init__(self):
        if not isinstance(self.scope, Scope) or self.role not in {"user", "assistant"}:
            raise ValueError("Typed evidence scope and explicit role required")
        identifier(self.source_id)
        if not isinstance(self.text, str) or len(self.text.encode()) > 64000:
            raise ValueError("Bounded source text required")
        if type(self.confirmed_end) is not int or not 0 <= self.confirmed_end <= len(self.text):
            raise ValueError("Confirmed source prefix required")
        if self.role == "assistant" and self.response_type is None:
            object.__setattr__(self, "response_type", "answer")
        if ((self.role == "user" and self.response_type is not None)
                or (self.role == "assistant" and self.response_type not in {"answer", "ack"})):
            raise ValueError("Response type belongs to a confirmed assistant source")


def evidence_payload(turns):
    if not isinstance(turns, tuple) or len(turns) > 32:
        raise ValueError("Bounded immutable source context required")
    if any(not isinstance(t, EvidenceTurn) for t in turns):
        raise ValueError("Typed evidence turns required")
    return [{"id": t.source_id, "role": t.role, "text": t.text[: t.confirmed_end],
             **({"response_type": "ack"} if t.response_type == "ack" else {})} for t in turns]


def evidence_digest(turns):
    return sha256(canonical(evidence_payload(turns)).encode()).hexdigest()
