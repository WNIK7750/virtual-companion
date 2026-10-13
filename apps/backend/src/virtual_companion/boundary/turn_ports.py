"""Storage commands for the opt-in single-companion turn lane.

Model/host adapters perform their I/O between calls, never inside a store
transaction. The review hash is supplied by a trusted review adapter; it is not
an authentication credential or a substitute for BR3 presentation review.
"""

from collections.abc import Callable
from typing import Protocol

from virtual_companion.text.identity import CompanionConfig

from .contracts import StateSnapshot
from .ports import CompanionStatePort
from .turns import CognitiveResult, IntakeReceipt, RunClaim


class TurnPort(Protocol):
    def accept(
        self, config: CompanionConfig, client_key: str, text: str, *, session_id: str | None = None,
        receiving_context: dict | None = None
    ) -> IntakeReceipt: ...

    def claim(
        self,
        companion: str,
        input_id: str,
        worker: str,
        *,
        now_us: int,
        lease_us: int,
        clock: Callable[[], int] | None = None,
    ) -> RunClaim: ...

    def renew(
        self,
        claim: RunClaim,
        *,
        now_us: int,
        lease_us: int,
        clock: Callable[[], int] | None = None,
        timeout: float = 5,
    ) -> RunClaim: ...

    def commit(
        self, result: CognitiveResult, *, now_us: int, clock: Callable[[], int] | None = None
    ) -> dict: ...

    def inspect(self, companion: str, input_id: str, *, timeout: float = 5) -> dict: ...

    def cancel(self, companion: str, input_id: str) -> dict: ...

    def fail(self, claim: RunClaim, *, reason: str, now_us: int) -> dict: ...

    def retry(
        self,
        companion: str,
        input_id: str,
        failed_attempt: str,
        retry_id: str,
        config: CompanionConfig,
    ) -> dict: ...

    def attempts(self, companion: str, input_id: str) -> list[dict]: ...


class RuntimePort(TurnPort, CompanionStatePort, Protocol):
    pass


class CognitiveKernelPort(Protocol):
    def generate(self, claim: RunClaim, snapshot: StateSnapshot, text: str) -> CognitiveResult: ...


class DeliveryPort(Protocol):
    def publish(self, companion: str, delivery_id: str) -> dict | None: ...

    def publication(self, companion: str, delivery_id: str) -> dict | None: ...

    def delivery(self, companion: str, delivery_id: str) -> dict: ...

    def acknowledge(
        self,
        companion: str,
        delivery_id: str,
        *,
        candidate_id: str,
        revision: int,
        sha256: str,
        sequence: int,
        displayed_end: int,
        stopped: bool = False,
        presentation_sha256: str | None = None,
    ) -> dict: ...
