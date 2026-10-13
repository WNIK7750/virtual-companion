"""BR0/BR2 immutable identity, event, owner and persistent write-version contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from uuid import UUID

from virtual_companion.text.chronology import utc_microseconds


def identifier(value):
    if not isinstance(value, str) or not 1 <= len(value) <= 128 or not value.strip():
        raise ValueError("Invalid identity")


def companion_id(value):
    if not isinstance(value, str) or str(UUID(value)) != value:
        raise ValueError("Canonical companion UUID required")


def canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


class VersionConflict(ValueError):
    """Re-read and recompute before applying; never retry an old snapshot blindly."""


class SemanticConflict(ValueError):
    """A source-bound proposal contradicts current semantic state, not its lease."""


class SourceResolutionUnresolved(ValueError):
    """Confirmed source coverage cannot be completed without new evidence."""


@dataclass(frozen=True)
class Scope:
    companion_id: str
    session_id: str

    def __post_init__(self):
        companion_id(self.companion_id)
        identifier(self.session_id)


@dataclass(frozen=True)
class OwnerToken:
    companion_id: str
    owner_id: str
    epoch: int

    def __post_init__(self):
        companion_id(self.companion_id)
        identifier(self.owner_id)
        if type(self.epoch) is not int or self.epoch < 1:
            raise ValueError("Positive owner epoch required")


@dataclass(frozen=True)
class StateEvent:
    scope: Scope
    event_id: str
    kind: str
    occurred_at: datetime
    received_at: datetime
    availability: str | None = None
    values_json: str = "{}"
    business_revision: int = 0

    def __post_init__(self):
        if not isinstance(self.scope, Scope):
            raise ValueError("Scope required")
        identifier(self.event_id)
        if not isinstance(self.kind, str) or self.kind not in {"presence", "interrupt", "state"}:
            raise ValueError("Unknown state event")
        utc_microseconds(self.occurred_at)
        utc_microseconds(self.received_at)
        if type(self.business_revision) is not int or self.business_revision < 0:
            raise ValueError("Invalid business revision")
        if not isinstance(self.values_json, str) or len(self.values_json.encode()) > 64000:
            raise ValueError("State proposal exceeds synthetic contract size")
        values = json.loads(self.values_json)
        if not isinstance(values, dict):
            raise ValueError("State object required")
        object.__setattr__(self, "values_json", canonical(values))
        if self.kind == "presence":
            if not isinstance(self.availability, str) or self.availability not in {
                "available",
                "busy",
                "sleeping",
            }:
                raise ValueError("Presence availability required")
        elif self.availability is not None:
            raise ValueError("Only presence events set availability")
        if self.kind != "state" and (values or self.business_revision):
            raise ValueError("Only state proposals carry values/revision")

    @property
    def fingerprint(self):
        # Receipt receive time is an observation, not part of input identity.
        return hashlib.sha256(
            canonical(
                [
                    self.scope.companion_id,
                    self.scope.session_id,
                    self.event_id,
                    self.kind,
                    utc_microseconds(self.occurred_at),
                    self.availability,
                    self.values_json,
                    self.business_revision,
                ]
            ).encode()
        ).hexdigest()


@dataclass(frozen=True)
class StateSnapshot:
    companion_id: str
    write_version: int = 0
    sequence: int = 0
    watermark_us: int | None = None
    decay_watermark_us: int | None = None
    business_revision: int = 0
    values_json: str = "{}"
    availability: str | None = None
    period_id: str | None = None


@dataclass(frozen=True)
class StateReceipt:
    companion_id: str
    session_id: str
    event_id: str
    fingerprint: str
    sequence: int
    write_version: int
    disposition: str
    period_id: str | None
    occurred_at: str
    received_at: str
    attempt_count: int | None = None
    discovery_probability: float | None = None
    draw_value: float | None = None
    discovered: bool | None = None
    reaction_code: str | None = None

    def to_mapping(self):
        fields = asdict(self)
        for name in (
            "attempt_count",
            "discovery_probability",
            "draw_value",
            "discovered",
            "reaction_code",
        ):
            if fields[name] is None:
                del fields[name]
        return fields
