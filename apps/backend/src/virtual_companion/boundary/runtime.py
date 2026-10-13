"""Single-companion application entry. Kernel I/O is outside all storage transactions.

No concrete model or orchestration policy lives here. The adapter can implement
the six cognitive duties without changing the transaction/recovery boundary.
"""

from collections.abc import Callable

from virtual_companion.text.identity import CompanionConfig

from .contracts import VersionConflict, identifier
from .turn_ports import CognitiveKernelPort, RuntimePort
from .turns import CognitiveResult


class KernelFailure(Exception):
    """An explicitly classified model failure, not an uncertain commit outcome."""

    def __init__(self, reason: str):
        identifier(reason)
        self.reason = reason
        super().__init__(reason)


class CompanionRuntime:
    def __init__(self, config: CompanionConfig, store: RuntimePort):
        if not isinstance(config, CompanionConfig):
            raise ValueError("Typed companion configuration required")
        self.config = config
        self.store = store

    def accept(self, client_key: str, text: str, *, session_id: str | None = None):
        return self.store.accept(self.config, client_key, text, session_id=session_id)

    def execute(
        self,
        input_id: str,
        worker: str,
        kernel: CognitiveKernelPort,
        *,
        clock: Callable[[], int],
        lease_us: int,
    ):
        companion = self.config.companion_id
        item = self.store.inspect(companion, input_id)
        if item["commit"] is not None:
            return item["commit"]
        claim = self.store.claim(
            companion, input_id, worker, now_us=clock(), lease_us=lease_us, clock=clock
        )
        basis = self.store.snapshot(companion)
        if basis.write_version != claim.state_version:
            raise VersionConflict("Snapshot advanced after claim; recover and recompute")
        try:
            result = kernel.generate(claim, basis, item["text"])
        except KernelFailure as failure:
            return self.store.fail(claim, reason=failure.reason, now_us=clock())
        if not isinstance(result, CognitiveResult) or result.claim != claim:
            raise ValueError("Kernel returned another run or ownership binding")
        # Exceptions here mean outcome unknown. Never classify them as model failure;
        # the next execution first queries the original durable business commit.
        return self.store.commit(result, now_us=clock(), clock=clock)

    def retry(self, input_id: str, failed_attempt: str, retry_id: str, config: CompanionConfig):
        return self.store.retry(
            self.config.companion_id, input_id, failed_attempt, retry_id, config
        )

    def cancel(self, input_id: str):
        return self.store.cancel(self.config.companion_id, input_id)
