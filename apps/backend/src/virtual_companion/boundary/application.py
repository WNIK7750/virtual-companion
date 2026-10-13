"""One companion-scoped state entry; compute proposals outside the storage transaction."""

from virtual_companion.text.identity import CompanionConfig

from .contracts import OwnerToken, StateEvent, StateSnapshot
from .ports import CompanionStatePort


class CompanionState:
    def __init__(self, config: CompanionConfig, store: CompanionStatePort):
        self.config = config
        self.store = store

    def snapshot(self):
        return self.store.snapshot(self.config.companion_id)

    def commit(self, event: StateEvent, basis: StateSnapshot, owner: OwnerToken):
        if not (
            event.scope.companion_id
            == basis.companion_id
            == owner.companion_id
            == self.config.companion_id
        ):
            raise ValueError("Companion boundary mismatch")
        return self.store.apply(event, owner, expected_version=basis.write_version)
