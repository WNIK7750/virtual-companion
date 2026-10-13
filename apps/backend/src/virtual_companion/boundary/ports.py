"""Application-facing state operations; future turn commits must share their transaction."""

from typing import Protocol

from .contracts import OwnerToken, StateEvent, StateReceipt, StateSnapshot


class CompanionStatePort(Protocol):
    def snapshot(self, companion: str) -> StateSnapshot: ...

    def apply(
        self, event: StateEvent, owner: OwnerToken, *, expected_version: int
    ) -> StateReceipt: ...

    def receipt(self, companion: str, event_id: str) -> StateReceipt | None: ...
