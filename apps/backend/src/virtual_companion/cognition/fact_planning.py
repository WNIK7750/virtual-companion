"""Prepare source-bound fact effects; the reviewed turn remains the only commit point."""

import json
from dataclasses import dataclass

from virtual_companion.boundary.contracts import Scope, VersionConflict
from virtual_companion.boundary.memory import MemoryEffect

from .fact_model import FactModel


@dataclass(frozen=True)
class FactPlan:
    effects: tuple[MemoryEffect, ...]


class FactPlanner:
    def __init__(self, store, invoke, *, evidence_clock):
        self.store, self.invoke, self.evidence_clock = store, invoke, evidence_clock

    def cancel(self):
        self.invoke.cancel()

    def references(self, claim):
        targets = self.store.fact_targets(claim.companion_id, now_us=self.evidence_clock())
        return tuple((key, value["version"]) for key, value in targets.items())

    def plan(self, context, guardian):
        claim = context.claim
        sources = self.store.exchange_sources(
            Scope(claim.companion_id, claim.session_id),
            references=(("user", claim.input_id),),
        )
        if len(sources) != 1 or sources[0].text != context.current_text:
            raise ValueError("Fact plan must use the actual current user input")
        now = self.evidence_clock()
        targets = self.store.fact_targets(claim.companion_id, now_us=now)
        recalled = {m["record_id"]: m for m in json.loads(context.payload)["memories"]}
        if any(
            key not in recalled
            or recalled[key]["version"] != value["version"]
            or recalled[key]["content"] != value["content"]
            for key, value in targets.items()
        ):
            raise ValueError("Fact targets must also be visible to reply and review")
        if self.store.snapshot(claim.companion_id).write_version != context.state.write_version:
            raise VersionConflict("Fact sources changed during business preparation")
        guardian.checkpoint()
        self.invoke.bind_turn(claim, guardian)
        commands = FactModel(self.invoke, guardian.cancelled).extract(
            scope=sources[0].scope,
            batch_id=claim.input_id,
            now_us=now,
            turn=sources[0],
            targets=targets,
        )
        guardian.checkpoint()
        return FactPlan(tuple(MemoryEffect(command) for command in commands))
