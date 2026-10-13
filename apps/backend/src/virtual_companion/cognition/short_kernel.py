"""Opt-in whole-response short kernel using the existing owner and commit boundary.

Admission is a trusted host Port, not a keyword router or an implicit safe default.
It must resolve tools, sensitive decisions and source conflicts before plain chat.
This first adapter leaves state and memory unchanged; RF2 owns durable maintenance.
"""

import threading
from hashlib import sha256
from uuid import uuid4

from virtual_companion.boundary.contracts import Scope, VersionConflict
from virtual_companion.boundary.presentation import (
    MessageUnit,
    Presentation,
    PresentationPolicy,
    ReviewedPresentation,
    review_by_policy,
)
from virtual_companion.boundary.runtime import KernelFailure
from virtual_companion.boundary.turns import CognitiveResult

from .preparation import CallControl
from .short_dialogue import ShortDialogue, ShortOutputError, ShortRequest
from .turn_guardian import TurnGuardian


class ShortKernel:
    def __init__(self, *, store, model_factory, admit, clock, lease_us, deadline_us,
                 renewal_margin_us, output_tokens=192):
        self.store, self.model_factory, self.admit = store, model_factory, admit
        self.clock, self.lease_us, self.deadline_us = clock, lease_us, deadline_us
        self.renewal_margin_us, self.output_tokens = renewal_margin_us, output_tokens
        self._active = threading.Lock()

    def _cancelled_or_stale(self, claim):
        item = self.store.inspect(claim.companion_id, claim.input_id, timeout=0.025)
        if item["status"] == "cancelled":
            return True
        run, owner = item["run"], item["owner"]
        if (item["status"] != "running" or run is None or run["status"] != "running"
                or owner is None
                or (owner["owner"], owner["epoch"]) != (claim.owner.owner_id, claim.owner.epoch)
                or (run["attempt"], run["owner"], run["epoch"])
                != (claim.attempt_id, claim.owner.owner_id, claim.owner.epoch)):
            raise VersionConflict("Short dialogue owner changed")
        return False

    def generate(self, claim, snapshot, text):
        if not self._active.acquire(blocking=False):
            raise ValueError("Short kernel already active")
        try:
            if (snapshot.companion_id != claim.companion_id
                    or snapshot.write_version != claim.state_version):
                raise ValueError("Short dialogue basis changed")
            item = self.store.inspect(claim.companion_id, claim.input_id)
            if item["text"] != text:
                raise ValueError("Short dialogue requires complete persisted input")
            control = CallControl(
                claim, clock=self.clock, cancelled=lambda: self._cancelled_or_stale(claim),
                renew=lambda run, now: self.store.renew(
                    run, now_us=now, lease_us=self.lease_us, clock=self.clock, timeout=0.025),
                deadline_us=self.deadline_us, renewal_margin_us=self.renewal_margin_us,
            )
            cancelled = threading.Event()
            model = self.model_factory(cancelled)
            with TurnGuardian(control, model, cancelled=cancelled) as guardian:
                request = self.admit(claim, snapshot, text)
                if not isinstance(request, ShortRequest) or request.user_text != text:
                    raise ValueError("Admission cannot replace persisted current input")
                if request.context is not None and (
                    request.context.companion != claim.companion_id
                    or request.context.input_id != claim.input_id
                    or request.context.state_version != snapshot.write_version
                ):
                    raise VersionConflict("Short context does not belong to current input basis")
                def current_context():
                    guardian.checkpoint()
                    validate_context = getattr(self.admit, "validate", None)
                    if request.context is not None and validate_context is not None:
                        validate_context(request.context)
                    if (self.store.snapshot(claim.companion_id).write_version
                            != snapshot.write_version):
                        raise VersionConflict("Short dialogue context changed before sending")

                current_context()
                try:
                    result = ShortDialogue(model, cancelled=guardian.cancelled,
                                           output_tokens=self.output_tokens,
                                           before_invoke=current_context).generate(request)
                except ShortOutputError as error:
                    raise KernelFailure("short_output_invalid") from error
                current_context()
                candidate_id, review_id = uuid4().hex, uuid4().hex
                surface = result.text or ""
                candidate = Presentation(
                    Scope(claim.companion_id, claim.session_id), claim.input_id, claim.attempt_id,
                    candidate_id, 1, surface,
                    (MessageUnit("whole", "text", surface),) if surface else (),
                    intent="speak" if result.status == "candidate" else result.status,
                )
                # UTF-8/length/action limits remain hard. Ordinary punctuation is
                # not given an invented style budget; explicit constraints ran above.
                policy = PresentationPolicy(64000, 16000, 1, False)
                verdict = review_by_policy(candidate, policy, review_id,
                                           policy_revision=result.policy)
                return CognitiveResult(
                    claim, candidate_id, 1, surface, review_id,
                    sha256(surface.encode()).hexdigest(),
                    presentation=ReviewedPresentation(candidate, policy, verdict),
                )
        finally:
            self._active.release()
