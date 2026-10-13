"""Optional streaming coordinator; no heuristic/default semantic unit approval.

The host supplies admission/context/lease checkpoints, a trusted complete-prefix
reviewer and a presenter returning an actual durable text-visible receipt ID.
Missing receipts stop generation conservatively; reconciliation uses the store.

An optional ``observe`` port records the bounded lifecycle of one attempt
(delta, observed boundary, review, approval, confirmation, terminal) so that
latency and candidate timing come from measurements. Recorded boundaries are
observations only: nothing is published without a trusted review result.

Displayed text is always a literal prefix of the model's own bytes; the only
permitted difference between the approved display and the model terminal is
trailing whitespace that carries no content.
"""

import threading
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256

from virtual_companion.boundary.presentation import ReviewedPresentation
from virtual_companion.boundary.received import ReceivedText
from virtual_companion.providers.text_deltas import UnapprovedTextDelta

from .short_dialogue import ShortDialogue, ShortOutputError, ShortRequest
from .stream_unit_review import UserConstraintViolation, first_faithful_boundary


def unit_boundary(text, start):
    """First observed unit boundary at or after ``start``; never an approval.

    Shares one definition with the trusted reviewer: ``paragraph`` (blank line),
    ``line`` (single line break, including the two-space line break) and
    ``sentence`` (sentence-final punctuation plus closing marks). The offset is the
    end of the candidate unit, so the model's own separator characters stay in the
    unit that precedes them.
    """
    return first_faithful_boundary(text, start)


def content_agrees(approved_text, final_text):
    """True when both texts carry the same content; only trailing blanks may differ."""
    if not isinstance(approved_text, str) or not isinstance(final_text, str):
        return False
    if approved_text == final_text:
        return True
    if final_text.startswith(approved_text):
        return not final_text[len(approved_text):].strip()
    if approved_text.startswith(final_text):
        return not approved_text[len(final_text):].strip()
    return False


class StreamDeliveryUnknown(RuntimeError):
    """Delivery may be visible; never automatically replay or whole-turn fallback."""


class StreamDialogue:
    def __init__(self, *, store, model_factory=None, review_prefix, present, checkpoint,
                 cancelled, clock, output_tokens=192, settle=None, parallel_review=False,
                 observe=None, response_type="answer", public_authority=None):
        if not all(callable(port) for port in
                   (review_prefix, present, checkpoint, clock)):
            raise ValueError("Explicit trusted stream ports required")
        if model_factory is not None and not callable(model_factory):
            raise ValueError("Explicit model factory required for generation")
        if observe is not None and not callable(observe):
            raise ValueError("Explicit observation port required")
        self.store, self.model_factory = store, model_factory
        self.review_prefix, self.present = review_prefix, present
        self.checkpoint, self.cancelled, self.clock = checkpoint, cancelled, clock
        self.output_tokens = output_tokens
        self.settle = self.store.finish_stream if settle is None else settle
        if type(parallel_review) is not bool:
            raise ValueError("Explicit parallel review mode required")
        self.parallel_review = parallel_review
        self.observe = observe
        if response_type not in {"answer", "ack"}:
            raise ValueError("Explicit spoken response type required")
        self.response_type = response_type
        if public_authority is not None and not callable(public_authority):
            raise ValueError('Explicit pure public authorization port required')
        self.public_authority = public_authority
        self._active = threading.Lock()

    def generate(self, claim, request):
        if self.model_factory is None:
            raise ValueError("Explicit model factory required for generation")
        model = None

        def start(receive, check):
            nonlocal model
            model = self.model_factory(receive)
            return lambda: ShortDialogue(model, cancelled=self.cancelled,
                output_tokens=self.output_tokens, before_invoke=check).generate(request).text

        def cancel():
            if model is not None:
                model.cancel()

        return self._coordinate(claim, request, start, cancel)

    def consume_answer(self, claim, request, answer, *, cancel):
        """Review decoded answer text; its producer owns the real wire and terminal."""
        text = ""

        def start(receive, check):
            def consume():
                nonlocal text
                check()
                for sequence, delta in enumerate(answer, 1):
                    text += delta
                    receive(UnapprovedTextDelta(sequence, delta))
                check()
                if (not text.strip()
                        or any(tag in text.lower() for tag in ("<think", "</think"))):
                    raise ShortOutputError("Complete plain-text answer required")
                if any(character in text for character in request.forbidden_characters):
                    raise ShortOutputError("Explicit user constraint violated")
                return text

            return consume

        self._coordinate(claim, request, start, cancel)
        return text

    def _coordinate(self, claim, request, start, cancel):
        if not self._active.acquire(blocking=False):
            raise ValueError("Stream coordinator already active")
        started = False
        try:
            if (not isinstance(request, ShortRequest) or request.action != "speak"
                    or request.risks):
                raise ValueError("Admitted plain-text speak request required")
            item = self.store.inspect(claim.companion_id, claim.input_id)
            if request.user_text != item["text"]:
                raise ValueError("Streaming request must preserve persisted input")
            if request.context is not None and (
                request.context.companion != claim.companion_id
                or request.context.input_id != claim.input_id
                or request.context.state_version != claim.state_version
            ):
                raise ValueError("Streaming context does not match current claim")
            if self.store.segments(claim.companion_id, claim.input_id):
                raise StreamDeliveryUnknown("Existing stream requires explicit reconciliation")
            buffer, sequence, approved = "", 0, None
            received_any = False
            boundary = None
            buffer_lock = threading.Lock()
            review_ready, producer_done, producer_failed = (
                threading.Event(), threading.Event(), threading.Event()
            )
            review_abort = threading.Event()
            review_errors = []
            first_boundary_length = None
            first_boundary_reviewed = threading.Event()

            def record(event, **fields):
                if self.observe is not None:
                    self.observe({"event": event, "at": self.clock(), **fields})

            def settled(outcome):
                settlement = self.settle(claim, outcome=outcome, now_us=self.clock(),
                                         clock=self.clock)
                if settlement is not None:
                    record("settled", outcome=settlement["outcome"])
                return settlement

            def check():
                if self.cancelled.is_set():
                    raise InterruptedError("Streaming dialogue cancelled")
                if producer_failed.is_set():
                    raise InterruptedError("Failed model output cannot be published")
                if review_abort.is_set():
                    raise InterruptedError("Streaming review or presentation aborted")
                self.checkpoint()

            def publish_prefix(observed, final):
                nonlocal approved, received_any
                check()
                record("review_begin", observed_bytes=len(observed.encode()), final=final)
                proposal = self.review_prefix(observed, final)
                record("review_end", approved=proposal is not None, final=final)
                check()  # Review may take time or trigger a concurrent correction.
                if proposal is None:
                    return
                if not isinstance(proposal, ReviewedPresentation):
                    raise ValueError("Trusted prefix review required")
                if proposal.candidate.response_type != self.response_type:
                    raise ValueError("Review cannot change the turn response type")
                surface = proposal.candidate.text
                if (not surface or not observed.startswith(surface)
                        or any(c in surface for c in request.forbidden_characters)
                        or any(tag in surface.lower() for tag in ("<think", "</think"))):
                    raise ShortOutputError("Approved unit differs from actual permitted output")
                if approved is not None and proposal.candidate == approved.candidate:
                    if (proposal.policy != approved.policy
                            or proposal.review.policy_revision != approved.review.policy_revision
                            or proposal.review.approval_source != approved.review.approval_source):
                        raise ValueError("Prefix policy changed")
                    return
                provenance = {}
                if request.context is not None and self.store.supports_history_readset:
                    from virtual_companion.boundary.history_readset import history_readset

                    context = request.context
                    provenance["readset"] = history_readset(
                        context.companion, context.input_id, context.state_version, context.payload,
                        **({'public_evidence': request.public_evidence,
                            'public_basis': {key: value for key, value in asdict(claim).items()
                                             if key != 'lease_until_us'}}
                           if request.public_evidence is not None else {}))
                authority = ({'public_authority': self.public_authority}
                             if provenance.get('readset', {}).get('protocol')
                             == 'history-readset.v8' else {})
                saved = self.store.approve_segment(claim, proposal, now_us=self.clock(),
                                                   clock=self.clock, **provenance, **authority)
                record("approved", sequence=saved["sequence"],
                       unit_sha256=sha256(surface.encode()).hexdigest())
                check()
                event = self.store.dispatch_segment(claim, saved["sequence"],
                                                    now_us=self.clock(), clock=self.clock,
                                                    **authority)
                if event is None:
                    raise StreamDeliveryUnknown("Segment already dispatched; reconcile receipt")
                check()
                host_receipt = self.present(deepcopy(event))
                if host_receipt is None:
                    raise StreamDeliveryUnknown("Actual visibility receipt unavailable")
                if isinstance(host_receipt, ReceivedText):
                    if self.store.received_segment(claim.companion_id, claim.input_id,
                                                  saved['sequence']) != host_receipt:
                        raise ValueError('Reliable receipt differs from persisted delivery')
                    received_any = True
                    record('received', sequence=saved['sequence'])
                else:
                    record("visible", sequence=saved["sequence"])
                    # A real receipt remains evidence even when cancellation raced rendering.
                    self.store.confirm_segment(
                        claim.companion_id, claim.input_id, saved["sequence"],
                        fingerprint=event["fingerprint"], host_receipt=host_receipt,
                        channel="text_visible")
                    record("confirmed", sequence=saved["sequence"])
                approved = proposal
                check()

            def receive(delta):
                nonlocal buffer, sequence, boundary, first_boundary_length
                check()
                with buffer_lock:
                    if (type(delta.sequence) is not int or delta.sequence != sequence + 1
                            or delta.approved is not False or not isinstance(delta.text, str)
                            or not delta.text):
                        raise ShortOutputError("Ordered unapproved text delta required")
                    sequence = delta.sequence
                    buffer += delta.text
                    if len(buffer.encode()) > 16000:
                        raise ShortOutputError("Bounded short stream required")
                    observed = buffer
                    start = 0 if approved is None else len(approved.candidate.text)
                    found = unit_boundary(observed, start)
                    fresh = found is not None and found != boundary
                    boundary = found
                    pause = self.parallel_review and fresh and first_boundary_length is None
                    if pause:
                        first_boundary_length = len(observed)
                record("delta", sequence=delta.sequence, bytes=len(delta.text.encode()))
                if fresh:
                    record("observed_boundary", kind=boundary[0], offset=boundary[1])
                if self.parallel_review:
                    review_ready.set()
                    if pause:
                        # Do not consume another provider delta while this first
                        # boundary is being reviewed, displayed and durably ACKed.
                        while not first_boundary_reviewed.wait(0.05):
                            check()
                        check()
                else:
                    publish_prefix(observed, False)

            check()
            produce = start(receive, check)
            started = True
            worker = None
            if self.parallel_review:
                def review_observed_prefixes():
                    last_seen = ""
                    try:
                        while True:
                            review_ready.wait(0.05)
                            review_ready.clear()
                            # A prefix opinion is useful only while generation is
                            # active. The complete opinion below owns any buffered
                            # text once the producer has reached its terminal.
                            if producer_done.is_set():
                                return
                            with buffer_lock:
                                observed = buffer
                            # The single early opinion is taken as soon as a candidate
                            # exists: starting it while the producer still runs is what
                            # lets it finish before the terminal. The reviewer's own
                            # bound (one early opinion) prevents one opinion per unit.
                            if observed != last_seen:
                                publish_prefix(observed, False)
                                last_seen = observed
                                with buffer_lock:
                                    if (first_boundary_length is not None
                                            and len(observed) >= first_boundary_length):
                                        first_boundary_reviewed.set()
                    except Exception as error:
                        if producer_failed.is_set():
                            return
                        review_errors.append(error)
                        review_abort.set()
                        cancel()
                        review_ready.set()

                worker = threading.Thread(target=review_observed_prefixes, daemon=True)
                worker.start()
            result, producer_error = None, None
            try:
                result = produce()
            except Exception as error:
                producer_error = error
                producer_failed.set()
            finally:
                producer_done.set()
                record("model_terminal")
                review_ready.set()
                if worker is not None:
                    worker.join(timeout=30)
                    if worker.is_alive():
                        review_abort.set()
                        cancel()
                        worker.join(timeout=1)
                        raise TimeoutError("Parallel unit review did not stop")
            if review_errors:
                raise review_errors[0] from producer_error
            if producer_error is not None:
                raise producer_error.with_traceback(producer_error.__traceback__)
            check()
            if result != buffer:
                raise ShortOutputError("Model terminal differs from observed stream")
            publish_prefix(buffer, True)
            if approved is None or not content_agrees(approved.candidate.text, result):
                raise ShortOutputError("Final output lacks complete trusted unit approval")
            # Existing visible-only ports already confirmed every published unit.
            # Only typed reliable reception needs the additional late-ACK check.
            if received_any:
                state = self.store.stream_state(claim.companion_id, claim.input_id)
                if state['confirmed_units'] != state['approved_units']:
                    return settled('received')
            return settled('completed')
        except Exception as error:
            if started:
                try:
                    cancel()
                finally:
                    # A refused answer is an explicit failure, not a user cancellation.
                    settled("failed" if isinstance(error, UserConstraintViolation)
                            else "cancelled" if self.cancelled.is_set() else "failed")
            raise
        finally:
            self._active.release()
