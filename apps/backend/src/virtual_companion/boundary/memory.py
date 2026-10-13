"""Synthetic memory identities, provenance and versioned mutation rules."""

from dataclasses import asdict, dataclass
from datetime import datetime
from hashlib import sha256
from typing import Protocol

from virtual_companion.text.chronology import utc_microseconds

from .contracts import OwnerToken, Scope, canonical, identifier


@dataclass(frozen=True)
class FactKey:
    subject: str
    predicate: str
    scope: str

    def __post_init__(self):
        for value in (self.subject, self.predicate, self.scope):
            identifier(value)


@dataclass(frozen=True)
class MemoryMutation:
    scope: Scope
    mutation_id: str
    kind: str
    record_id: str | None
    content: str
    source_kind: str
    source_id: str
    reason: str
    occurred_at: datetime
    fact_key: FactKey | None = None
    expected_record_version: int | None = None
    expires_at: datetime | None = None
    # Optional occurrence time of the experience described by the source.
    # ``occurred_at`` remains the required append/revision audit time.
    event_time: datetime | None = None

    def __post_init__(self):
        if not isinstance(self.scope, Scope):
            raise ValueError("Memory scope required")
        identifier(self.mutation_id)
        identifier(self.source_id)
        if self.record_id is not None:
            identifier(self.record_id)
        if self.kind not in {"append_event", "assert_fact", "correct", "revoke"}:
            raise ValueError("Unknown memory mutation")
        if self.source_kind not in {"input", "assessment", "manual", "output", "background"}:
            raise ValueError("Unknown memory source; persona assets are separate")
        for value, maximum in ((self.content, 16000), (self.reason, 2000)):
            if not isinstance(value, str) or len(value.encode()) > maximum:
                raise ValueError("Invalid bounded memory text")
        if not self.reason.strip() or (self.kind != "revoke" and not self.content.strip()):
            raise ValueError("Memory content and revision reason required")
        if self.kind == "revoke" and self.content:
            raise ValueError("Revocation does not assert replacement content")
        occurred = utc_microseconds(self.occurred_at)
        if self.event_time is not None:
            utc_microseconds(self.event_time)
        if self.expires_at is not None and utc_microseconds(self.expires_at) < occurred:
            raise ValueError("Expiry precedes occurrence")
        if self.kind in {"append_event", "assert_fact"}:
            if self.record_id is None or self.expected_record_version is not None:
                raise ValueError("New record identity required; no previous version")
        elif self.record_id is not None and (
            type(self.expected_record_version) is not int or self.expected_record_version < 1
        ):
            raise ValueError("Targeted revision requires a positive expected version")
        elif self.record_id is None and self.expected_record_version is not None:
            raise ValueError("Unknown target cannot claim a version")
        if self.kind == "assert_fact":
            if not isinstance(self.fact_key, FactKey):
                raise ValueError("Explicit subject/predicate/scope required")
        elif self.fact_key is not None:
            raise ValueError("Only a new fact declares its slot; revisions retain it")

    def to_mapping(self):
        data = asdict(self)
        data["occurred_at"] = self.occurred_at.isoformat()
        data["expires_at"] = self.expires_at.isoformat() if self.expires_at else None
        data["event_time"] = self.event_time.isoformat() if self.event_time else None
        return data

    @classmethod
    def from_mapping(cls, data):
        data = dict(data)
        data["scope"] = Scope(**data["scope"])
        if data["fact_key"]:
            data["fact_key"] = FactKey(**data["fact_key"])
        data["occurred_at"] = datetime.fromisoformat(data["occurred_at"])
        if data["expires_at"]:
            data["expires_at"] = datetime.fromisoformat(data["expires_at"])
        if data.get("event_time"):
            data["event_time"] = datetime.fromisoformat(data["event_time"])
        else:
            data["event_time"] = None
        return cls(**data)

    @property
    def fingerprint(self):
        data = self.to_mapping()
        data["occurred_at"] = utc_microseconds(self.occurred_at)
        data["expires_at"] = utc_microseconds(self.expires_at) if self.expires_at else None
        return sha256(canonical(data).encode()).hexdigest()


@dataclass(frozen=True)
class MemoryEffect:
    mutation: MemoryMutation
    after_codepoints: int | None = None

    def __post_init__(self):
        if not isinstance(self.mutation, MemoryMutation):
            raise ValueError("Typed mutation required")
        if self.mutation.source_kind == "output":
            if type(self.after_codepoints) is not int or self.after_codepoints < 1:
                raise ValueError("Output effect requires a positive confirmed prefix")
        elif self.after_codepoints is not None:
            raise ValueError("Internal/input effects do not wait for output")


def evolve(mutation, before, *, slot_taken=False):
    """Keep unknown/stale proposals as explicit dispositions, never guess a target."""
    if mutation.kind in {"append_event", "assert_fact"}:
        if before is not None or slot_taken:
            return None, "conflict"
    elif before is None:
        return None, "pending_target"
    elif (
        before["version"] != mutation.expected_record_version
        or before["status"] != "active"
        or (mutation.kind == "correct" and before["kind"] != "fact")
        or (
            mutation.kind == "correct"
            and before.get("expires_at")
            and utc_microseconds(datetime.fromisoformat(before["expires_at"]))
            <= utc_microseconds(mutation.occurred_at)
        )
    ):
        return None, "conflict"
    version = before["version"] + 1 if before else 1
    return {
        "record_id": mutation.record_id,
        "version": version,
        "previous_version": before["version"] if before else None,
        "kind": before["kind"]
        if before
        else ("fact" if mutation.kind == "assert_fact" else "event"),
        "fact_key": before["fact_key"]
        if before
        else (asdict(mutation.fact_key) if mutation.fact_key else None),
        "status": "revoked" if mutation.kind == "revoke" else "active",
        "content": before["content"] if mutation.kind == "revoke" else mutation.content,
        "mutation_id": mutation.mutation_id,
        "source_kind": mutation.source_kind,
        "source_id": mutation.source_id,
        "reason": mutation.reason,
        "session_id": mutation.scope.session_id,
        "occurred_at": mutation.occurred_at.isoformat(),
        # Missing detail is not authorization to remove an existing lifetime.
        "expires_at": (
            mutation.expires_at.isoformat()
            if mutation.expires_at
            else (before.get("expires_at") if before else None)
        ),
        "event_time": (
            mutation.event_time.isoformat()
            if mutation.event_time
            else (before.get("event_time") if before else None)
        ),
    }, "applied"


class MemoryPort(Protocol):
    def memory_receipt(self, companion: str, mutation_id: str) -> dict | None: ...

    def mutate(
        self, mutation: MemoryMutation, owner: OwnerToken, *, expected_version: int
    ) -> dict: ...

    def mutate_batch(
        self,
        mutations: tuple[MemoryMutation, ...],
        owner: OwnerToken,
        *,
        expected_version: int,
        now_us: int,
    ) -> list[dict]: ...

    def recall(
        self, companion: str, *, now_us: int, candidates: tuple | None = None, limit: int = 100
    ) -> list[dict]: ...

    def history(self, companion: str, record_id: str) -> list[dict]: ...
