"""Reliable text receipt is distinct from visibility, reading and playback."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ReceivedText:
    sequence: int
    fingerprint: str
    host_receipt: str
