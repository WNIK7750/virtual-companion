"""RN-003 synthetic chronology candidate; not a production late-event policy.

Event time orders facts; receive order breaks ties. Late receipts are retained
without being applied. No Presence, database migration or wall-clock mutation.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from datetime import UTC, datetime

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def _utc(value):
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("An aware datetime is required")
    try:
        return value.astimezone(UTC)
    except OverflowError as error:
        raise ValueError("Time outside representable UTC range") from error


def utc_microseconds(value: datetime) -> int:
    delta = _utc(value) - _EPOCH
    return (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds


def utc_text(value: datetime) -> str:
    return _utc(value).isoformat(timespec="microseconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class TimeEvent:
    id: str
    occurred_at: datetime
    received_at: datetime
    payload_sha256: str

    def __post_init__(self):
        if not isinstance(self.id, str) or not 1 <= len(self.id) <= 128:
            raise ValueError("Invalid event identity")
        if not isinstance(self.payload_sha256, str) or not re.fullmatch(
            r"[a-f0-9]{64}", self.payload_sha256
        ):
            raise ValueError("Invalid event payload hash")
        _utc(self.occurred_at)
        _utc(self.received_at)


@dataclass(frozen=True)
class TimeReceipt:
    event: TimeEvent
    sequence: int
    status: str


class ChronologyGate:
    def __init__(self):
        self._receipts = {}
        self._watermark = None
        self._lock = threading.RLock()

    @property
    def watermark(self):
        with self._lock:
            return self._watermark

    @property
    def receipts(self):
        with self._lock:
            return tuple(self._receipts.values())

    @property
    def applied(self):
        return tuple(receipt for receipt in self.receipts if receipt.status == "accepted")

    def accept(self, event: TimeEvent) -> TimeReceipt:
        if not isinstance(event, TimeEvent):
            raise ValueError("TimeEvent required")
        moment = utc_microseconds(event.occurred_at)
        with self._lock:
            existing = self._receipts.get(event.id)
            if existing:
                if (
                    utc_microseconds(existing.event.occurred_at) != moment
                    or existing.event.payload_sha256 != event.payload_sha256
                ):
                    raise ValueError("Conflicting event identity")
                return existing
            late = self._watermark is not None and moment < self._watermark
            receipt = TimeReceipt(event, len(self._receipts) + 1, "late" if late else "accepted")
            self._receipts[event.id] = receipt
            if not late:
                self._watermark = moment
            return receipt
