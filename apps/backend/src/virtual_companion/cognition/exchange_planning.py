"""User-turn exchange planning; effects stay provisional until reply commit."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from virtual_companion.boundary.evidence import EvidenceTurn
from virtual_companion.boundary.exchange_resolution import SourceResolution
from virtual_companion.boundary.exchanges import ExchangeCommand, ExchangeSupport

from .exchange_model import ExchangeModel
from .preparation import DisplayedTurn


@dataclass(frozen=True)
class ExchangePlan:
    sources: tuple[EvidenceTurn, ...]
    proposals: tuple[tuple[ExchangeCommand, ExchangeSupport], ...]
    resolution: SourceResolution | None = None


class ExchangePlanner:
    def __init__(self, store, invoke, *, evidence_clock, resolution_protocol=False,
                 resolution_strategy="candidate_first"):
        self.store, self.invoke, self.evidence_clock = store, invoke, evidence_clock
        self.resolution_protocol = resolution_protocol
        self.resolution_strategy = resolution_strategy

    def cancel(self):
        self.invoke.cancel()

    def history(self, claim):
        sources, _ = self.store.exchange_context(claim)
        time_source = getattr(self.store, "source_time", None)

        def occurred(source):
            stamp = (
                time_source(source.scope, source.role, source.source_id) if time_source else None
            )
            if stamp is None:
                return None
            return (datetime(1970, 1, 1, tzinfo=UTC) + timedelta(microseconds=stamp)).isoformat()

        return tuple(
            DisplayedTurn(s.scope, s.source_id, s.role, s.text, s.confirmed_end, occurred(s))
            for s in sources[:-1]
        )

    def plan(self, context, guardian):
        sources, eligible = self.store.exchange_context(context.claim)
        if not sources or sources[-1].text != context.current_text:
            raise ValueError("Exchange context differs from actual current user input")
        history = json.loads(context.payload)["history"]
        available = {
            (h["scope"]["session_id"], h["input_id"], h["role"]): h["text"] for h in history
        }
        if any(
            available.get((s.scope.session_id, s.source_id, s.role)) != s.text for s in sources[:-1]
        ):
            raise ValueError("Reply and review must receive the same confirmed exchange history")
        guardian.checkpoint()
        self.invoke.bind_turn(context.claim, guardian)
        now = self.evidence_clock()
        targets = self.store.pending_exchanges(context.claim.companion_id, now_us=now)
        guardian.checkpoint()
        model = ExchangeModel(self.invoke, guardian.cancelled)
        exchange_inputs = dict(
            scope=sources[-1].scope,
            batch_id=context.claim.input_id,
            now_us=now,
            turns=sources,
            targets={t["exchange_id"]: t for t in targets},
            eligible_source_ids=eligible,
        )
        if self.resolution_protocol:
            proposals, resolution = model.resolve_source(
                **exchange_inputs, strategy=self.resolution_strategy)
        else:
            proposals, resolution = model.extract(**exchange_inputs), None
        guardian.checkpoint()
        return ExchangePlan(sources, proposals, resolution)
