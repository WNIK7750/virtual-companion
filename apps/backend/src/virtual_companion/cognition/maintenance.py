"""Controlled post-ACK exchange extraction using the existing run lifecycle.

This executes explicitly queued maintenance; it does not create a second loop or
convert an internal event into a user message or an unreviewed assistant reply.
"""

import threading

from virtual_companion.boundary.contracts import Scope, VersionConflict

from .exchange_model import ExchangeModel
from .preparation import CallCancelled, CallControl, CallDeadline
from .turn_guardian import TurnGuardian


class ExchangeMaintenance:
    def __init__(self, store, invoke, *, clock, evidence_clock, lease_us=5_000_000,
                 resolution_protocol=False, resolution_strategy="candidate_first"):
        self.store, self.invoke = store, invoke
        self.clock, self.evidence_clock, self.lease_us = clock, evidence_clock, lease_us
        self.resolution_protocol = resolution_protocol
        self.resolution_strategy = resolution_strategy
        self._active = threading.Lock()

    def execute(self, companion, input_id, worker, *, deadline_us):
        if not self._active.acquire(blocking=False):
            raise RuntimeError("Maintenance transport already in use")
        try:
            return self._execute(companion, input_id, worker, deadline_us)
        finally:
            self._active.release()

    def _execute(self, companion, input_id, worker, deadline_us):
        item = self.store.inspect(companion, input_id)
        if (
            not isinstance(item.get("trigger"), dict)
            or item["trigger"].get("kind") != "exchange_maintenance"
        ):
            raise ValueError("Explicit exchange maintenance trigger required")
        if item["maintenance"] is not None:
            return item["maintenance"]
        claim = self.store.claim(
            companion,
            input_id,
            worker,
            now_us=self.clock(),
            lease_us=self.lease_us,
            clock=self.clock,
        )

        def cancelled():
            value = self.store.inspect(companion, input_id, timeout=0.025)
            owner, run = value["owner"], value["run"]
            if value["status"] == "cancelled":
                return True
            if (
                value["status"] != "running"
                or owner is None
                or run is None
                or (owner["owner"], owner["epoch"]) != (claim.owner.owner_id, claim.owner.epoch)
                or run["attempt"] != claim.attempt_id
            ):
                raise VersionConflict("Maintenance ownership changed; reconcile before continuing")
            return False

        control = CallControl(
            claim,
            clock=self.clock,
            cancelled=cancelled,
            renew=lambda run, now: self.store.renew(
                run,
                now_us=now,
                lease_us=self.lease_us,
                clock=self.clock,
                timeout=0.025,
            ),
            deadline_us=deadline_us,
            renewal_margin_us=min(1_000_000, self.lease_us // 2),
        )
        try:
            with TurnGuardian(control, self.invoke) as guardian:
                self.invoke.bind_turn(claim, guardian)
                scope = Scope(companion, item["session"])
                turns = self.store.exchange_sources(
                    scope,
                    references=(("assistant", item["trigger"]["source_id"]),),
                )
                now = self.evidence_clock()
                targets = self.store.pending_exchanges(companion, now_us=now)
                guardian.checkpoint()
                model = ExchangeModel(self.invoke, guardian.cancelled)
                exchange_inputs = dict(
                    scope=scope,
                    batch_id=claim.input_id,
                    now_us=now,
                    turns=turns,
                    targets={t["exchange_id"]: t for t in targets},
                )
                if self.resolution_protocol:
                    proposals, _ = model.resolve_source(
                        **exchange_inputs, eligible_source_ids=(turns[-1].source_id,),
                        strategy=self.resolution_strategy,
                    )
                else:
                    proposals = model.extract(**exchange_inputs)
                guardian.checkpoint()
        except VersionConflict:
            raise
        except (CallCancelled, InterruptedError):
            if self.store.inspect(companion, input_id)["status"] == "cancelled":
                return {"status": "cancelled", "input_id": input_id}
            return self.store.fail(claim, reason="exchange_cancelled", now_us=self.clock())
        except (ValueError, CallDeadline) as error:
            reason = (
                "exchange_deadline"
                if isinstance(error, CallDeadline)
                else "exchange_model_contract"
            )
            return self.store.fail(claim, reason=reason, now_us=self.clock())
        # Unknown storage outcomes must be reconciled from the durable receipt,
        # never converted into a model failure or reissued after a lost response.
        return self.store.finish_maintenance(
            claim, proposals, now_us=self.clock(), clock=self.clock
        )
