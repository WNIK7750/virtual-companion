"""Opt-in guarded streaming assembly; semantic admission/review remain explicit Ports.

Reuses the existing turn owner, cancellation observer and context read-set validator.
No default risk admission, punctuation splitting, additional loop or database migration.
"""

import threading

from virtual_companion.boundary.contracts import VersionConflict

from .preparation import CallControl
from .short_dialogue import ShortRequest
from .short_kernel import ShortKernel
from .stream_dialogue import StreamDialogue
from .turn_guardian import TurnGuardian


class StreamKernel(ShortKernel):
    """Shares ShortKernel's owner/configuration contract, returns a stream settlement."""

    def __init__(self, *, review_prefix=None, review_session_factory=None, present,
                 parallel_review=False, observe=None, **kwargs):
        super().__init__(**kwargs)
        if (not callable(present)
                or callable(review_prefix) == callable(review_session_factory)):
            raise ValueError("Explicit trusted review and presentation ports required")
        if observe is not None and not callable(observe):
            raise ValueError("Explicit observation port required")
        self.review_prefix, self.review_session_factory = review_prefix, review_session_factory
        self.present = present
        self.observe = observe
        if type(parallel_review) is not bool:
            raise ValueError("Explicit parallel review mode required")
        self.parallel_review = parallel_review

    def generate(self, claim, snapshot, text):
        if not self._active.acquire(blocking=False):
            raise ValueError("Stream kernel already active")
        guardian = None
        try:
            if (
                snapshot.companion_id != claim.companion_id
                or snapshot.write_version != claim.state_version
            ):
                raise ValueError("Stream dialogue basis changed")
            if self.store.inspect(claim.companion_id, claim.input_id)["text"] != text:
                raise ValueError("Streaming requires complete persisted input")
            if (
                self.store.stream_state(claim.companion_id, claim.input_id)["settlement"]
                is not None
            ):
                raise ValueError("Settled stream cannot be generated again")
            cancelled = threading.Event()
            receiver = None

            def receive(delta):
                if receiver is None:
                    raise ValueError("Delta outside active generation")
                receiver(delta)

            model = self.model_factory(cancelled, receive)
            control = CallControl(
                claim,
                clock=self.clock,
                cancelled=lambda: self._cancelled_or_stale(claim),
                renew=lambda run, now: self.store.renew(
                    run, now_us=now, lease_us=self.lease_us, clock=self.clock, timeout=0.025
                ),
                deadline_us=self.deadline_us,
                renewal_margin_us=self.renewal_margin_us,
            )
            guardian = TurnGuardian(control, model, cancelled=cancelled)

            def bind(callback):
                nonlocal receiver
                receiver = callback
                return model

            with guardian:
                request = self.admit(claim, snapshot, text)
                if not isinstance(request, ShortRequest) or request.user_text != text:
                    raise ValueError("Admission cannot replace persisted input")

                def checkpoint():
                    guardian.checkpoint()
                    validate = getattr(self.admit, "validate", None)
                    if request.context is not None:
                        if not callable(validate):
                            raise ValueError("Context requires an explicit read-set validator")
                        validate(request.context)
                    if (
                        self.store.snapshot(claim.companion_id).write_version
                        != snapshot.write_version
                    ):
                        raise VersionConflict("Streaming context changed")

                def settle(run, *, outcome, now_us, clock):
                    # Do not let the observer interpret our terminal queue marker
                    # as an external cancellation after successful delivery.
                    guardian.stop()
                    if outcome == "completed":
                        checkpoint()
                    return self.store.finish_stream(
                        run, outcome=outcome, now_us=now_us, clock=clock
                    )

                if self.review_session_factory is None:
                    def review(buffer, final):
                        return self.review_prefix(claim, request, buffer, final)
                else:
                    review = self.review_session_factory(claim, request, cancelled)
                    if not callable(review):
                        raise ValueError("Review session factory must return a callable")
                checkpoint()
                return StreamDialogue(
                    store=self.store,
                    model_factory=bind,
                    review_prefix=review,
                    present=self.present,
                    checkpoint=checkpoint,
                    cancelled=cancelled,
                    clock=self.clock,
                    output_tokens=self.output_tokens,
                    settle=settle,
                    parallel_review=self.parallel_review,
                    observe=self.observe,
                ).generate(claim, request)
        except Exception:
            if guardian is not None:
                guardian.stop()
                model.cancel()
                state = self.store.stream_state(claim.companion_id, claim.input_id)
                if state["settlement"] is None:
                    self.store.finish_stream(
                        claim,
                        outcome="cancelled" if cancelled.is_set() else "failed",
                        now_us=self.clock(),
                        clock=self.clock,
                    )
            raise
        finally:
            self._active.release()
