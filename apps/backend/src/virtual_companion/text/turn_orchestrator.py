"""One bounded turn over explicit host capabilities and checked requests.

The host owns the session, source/readset truth, cancellation, approved delivery
and settlement. This adapter chooses the answer or one memory delegation, with
one pre-approval rejection recovery or exact decoded-string carrier recovery.
"""

from dataclasses import dataclass

from virtual_companion.cognition.stream_unit_review import CandidateRejected

from .demand_memory import (
    DemandTurnResult,
    MalformedAnswerEnvelope,
    TextCandidate,
    _generate,
    run_demand_turn,
)
from .evidence_recall import EvidenceSelection
from .ports import PreparedTextRequest


@dataclass(frozen=True)
class CandidateRepair:
    """Rejected candidate data; the host freezes its replacement wire request."""

    prepared: PreparedTextRequest
    text: str
    verdict: dict


@dataclass(frozen=True)
class CarrierRepair:
    """JSON syntax recovery, independent of semantic review and replacement."""

    prepared: PreparedTextRequest
    envelope: MalformedAnswerEnvelope


class TurnOrchestrator:
    """PreparedTextRequest-compatible orchestration, without a runtime owner.

    ``prepare(claim, final=..., repair=None, recall_query=None)`` returns a checked-stage
    carrier. ``refresh(claim)`` returns the host's refreshed owner/readset basis.
    ``check(cancelled, request=None)`` owns cancellation and context validation;
    the same cancellation event remains bound for the entire turn. ``deliver``
    consumes unapproved candidate text through the host's normal approval/ACK
    path. Prefix checks and every settlement stay with that host.
    """

    def __init__(self, claim, cancelled, *, prepare, resolve_current, refresh,
                 check, has_approved_prefix, deliver, settle_failure, settle_listen,
                 delivery_state=None, recall_evidence=None, max_recall_calls=0,
                 allow_navigation=False, allow_constrained_recall=False,
                 search_public=None, max_public_calls=0, extract_public=None,
                 allow_public_extract=False):
        if (type(allow_constrained_recall) is not bool
                or (allow_constrained_recall and not allow_navigation)
                or type(max_recall_calls) is not int
                or not 0 <= max_recall_calls <= (6 if allow_navigation else 2)
                or (max_recall_calls and not callable(recall_evidence))):
            raise ValueError("Bounded readonly evidence capability required")
        self.claim, self.cancelled = claim, cancelled
        self.prepare, self.resolve_current, self.refresh = prepare, resolve_current, refresh
        self.check, self.has_approved_prefix = check, has_approved_prefix
        self.deliver, self.settle_failure = deliver, settle_failure
        self.settle_listen, self.result = settle_listen, None
        self.delivery_state = delivery_state
        self.recall_evidence, self.max_recall_calls = recall_evidence, max_recall_calls
        self.allow_navigation = allow_navigation
        self.allow_constrained_recall = allow_constrained_recall
        if (type(max_public_calls) is not int or max_public_calls not in (0, 1)
                or (max_public_calls and not callable(search_public))
                or type(allow_public_extract) is not bool
                or (allow_public_extract
                    and (not max_public_calls or not callable(extract_public)))):
            raise ValueError('Bounded optional public capability required')
        self.search_public, self.max_public_calls = search_public, max_public_calls
        self.extract_public, self.allow_public_extract = extract_public, allow_public_extract

    def close(self):
        self.settle_failure(self.cancelled)

    def check_context(self):
        try:
            self.check(self.cancelled)
        except BaseException:
            self.settle_failure(self.cancelled)
            raise

    def stream(self, cancelled):
        if cancelled is not self.cancelled:
            raise ValueError("Turn cancellation identity changed")
        claim = self.claim
        candidate_stage, candidate_text, memory_calls, delivered = None, "", 0, False
        response_type, generation_calls = "answer", 0
        recovery_used = False
        recall_calls, latest_evidence, initial_used = 0, None, False
        public_calls = 0

        def resolve(owner):
            nonlocal memory_calls
            self.resolve_current(owner)
            memory_calls += 1

        def refresh(owner):
            nonlocal claim
            claim = self.refresh(owner)
            return claim

        def deliver(stage, answer):
            nonlocal candidate_stage, candidate_text, delivered, response_type
            candidate_stage = stage
            response_type = answer.response_type
            try:
                candidate_text = self.deliver(stage, answer)
                delivered = True
            finally:
                if hasattr(answer, 'text'):
                    candidate_text = answer.text

        def final_stage(query=None):
            return self.prepare(claim, final=True,
                                **public_options(),
                                **({'recall_query': query} if query is not None else {}))

        def public_options():
            return ({'allow_public_search': public_calls < self.max_public_calls,
                     **({'allow_public_extract': public_calls < self.max_public_calls}
                        if self.allow_public_extract else {})}
                    if self.max_public_calls else {})

        def evidence_options():
            if not self.max_recall_calls:
                return public_options()
            return {**public_options(), "evidence": latest_evidence, "allow_recall": (
                recall_calls < self.max_recall_calls
                and (latest_evidence is None or latest_evidence.can_recall))}

        def recall(owner, query, *, cancelled):
            nonlocal recall_calls, latest_evidence
            if self.has_approved_prefix():
                raise ValueError("Readonly recall cannot reopen an approved prefix")
            self.check(cancelled)
            recall_calls += 1
            selected = self.recall_evidence(owner, query, cancelled=cancelled)
            self.check(cancelled)
            if not isinstance(selected, EvidenceSelection):
                raise ValueError("Bound readonly evidence selection required")
            latest_evidence = selected
            return selected

        def public_dispatch(method, owner, value, *, cancelled):
            nonlocal public_calls
            if self.has_approved_prefix():
                raise ValueError('Public query cannot reopen an approved prefix')
            self.check(cancelled)
            public_calls += 1
            result = method(owner, value, cancelled=cancelled)
            self.check(cancelled)
            return result

        def search(owner, query, *, cancelled):
            return public_dispatch(self.search_public, owner, query, cancelled=cancelled)

        def extract(owner, url, *, cancelled):
            return public_dispatch(self.extract_public, owner, url, cancelled=cancelled)

        def demand_stage(owner, *, final, recall_query=None, evidence=None, allow_recall=False,
                         allow_public_search=False, allow_public_extract=False):
            nonlocal generation_calls, initial_used, latest_evidence
            generation_calls += 1
            if not initial_used:
                initial_used = True
                prepared = initial
                if self.max_recall_calls and hasattr(prepared, 'readonly_evidence'):
                    latest_evidence = prepared.readonly_evidence
                return prepared
            if not self.max_recall_calls and not self.max_public_calls:
                return final_stage(recall_query)
            prepared = self.prepare(claim, final=final, evidence=evidence,
                                allow_recall=allow_recall,
                                **public_options(),
                                **({'recall_query': recall_query}
                                   if recall_query is not None else {}))
            if hasattr(prepared, 'readonly_evidence'):
                latest_evidence = prepared.readonly_evidence
            return prepared

        def generate(stage, *, final):
            nonlocal generation_calls, response_type, candidate_text
            generation_calls += 1
            action = _generate(stage, cancelled, final=final, on_answer=deliver)
            response_type, candidate_text = action['action'], action.get('text', '')
            return action

        def recover_carrier(stage, envelope):
            nonlocal recovery_used, generation_calls
            if recovery_used:
                raise ValueError("Turn recovery already used")
            self.check(cancelled, stage.request)
            recovery_used = True
            recovery = self.prepare(claim, final=True, repair=CarrierRepair(stage, envelope))
            generation_calls += 1
            # Buffer this carrier: its string has already entered the original
            # coordinator. A second delivery would replay confirmed content.
            action = _generate(recovery, cancelled, final=True, strict=True)
            if action != {"action": envelope.response_type, "text": envelope.text}:
                raise ValueError("Carrier recovery changed the decoded response")
            self.check(cancelled, stage.request)
            return action

        def finish_candidate():
            if response_type == 'listen':
                self.settle_listen(candidate_stage)
            elif not delivered:
                deliver(candidate_stage, TextCandidate(response_type, candidate_text))

        try:
            try:
                initial = self.prepare(claim, final=False, **evidence_options())
                result = run_demand_turn(
                    claim=claim,
                    prepare=demand_stage,
                    resolve_current=resolve, refresh=refresh,
                    cancelled=cancelled, on_answer=deliver, recover_answer=recover_carrier,
                    recall_evidence=recall if self.max_recall_calls else None,
                    max_recall_calls=self.max_recall_calls,
                    allow_navigation=self.allow_navigation,
                    allow_constrained_recall=self.allow_constrained_recall,
                    search_public=search if self.max_public_calls else None,
                    max_public_calls=self.max_public_calls,
                    extract_public=extract if self.allow_public_extract else None,
                    allow_public_extract=self.allow_public_extract,
                )
                candidate_stage, candidate_text = result.prepared, result.text
                response_type, generation_calls = result.response_type, result.generation_calls
                finish_candidate()
            except CandidateRejected as rejected:
                # Pending, unknown and confirmed approvals are all immutable.
                # Only the same zero-approved turn permits the existing repair.
                if self.has_approved_prefix() or recovery_used:
                    raise
                self.check(cancelled, candidate_stage.request)
                recovery_used = True
                memory_allowed = memory_calls == 0
                candidate_stage = self.prepare(claim, final=not memory_allowed,
                    repair=CandidateRepair(candidate_stage,
                        rejected.text if rejected.text is not None else candidate_text,
                        rejected.verdict), **evidence_options())
                if self.max_recall_calls or self.max_public_calls:
                    first_repair = True

                    def repair_stage(owner, **kwargs):
                        nonlocal first_repair, latest_evidence
                        if first_repair:
                            first_repair = False
                            prepared = candidate_stage
                        else:
                            prepared = self.prepare(claim, **kwargs)
                        if hasattr(prepared, 'readonly_evidence'):
                            latest_evidence = prepared.readonly_evidence
                        return prepared

                    result = run_demand_turn(
                        claim=claim, prepare=repair_stage, resolve_current=resolve, refresh=refresh,
                        cancelled=cancelled, on_answer=deliver, recall_evidence=recall,
                        max_recall_calls=self.max_recall_calls,
                        memory_calls_used=memory_calls, recall_calls_used=recall_calls,
                        evidence=latest_evidence,
                        allow_navigation=self.allow_navigation,
                        allow_constrained_recall=self.allow_constrained_recall,
                        search_public=search if self.max_public_calls else None,
                        max_public_calls=self.max_public_calls,
                        extract_public=extract if self.allow_public_extract else None,
                        allow_public_extract=self.allow_public_extract,
                        public_calls_used=public_calls,
                    )
                    generation_calls += result.generation_calls
                    candidate_stage, candidate_text = result.prepared, result.text
                    response_type = result.response_type
                else:
                    action = generate(candidate_stage, final=not memory_allowed)
                    if action['action'] == 'memory':
                        resolve(claim)
                        refresh(claim)
                        candidate_stage = final_stage(action.get('query'))
                        generate(candidate_stage, final=True)
                finish_candidate()
            # TextSession mirrors only text already confirmed through the host.
            state = (self.delivery_state() if self.delivery_state and response_type != 'listen'
                     else None)
            status = 'received' if state and state['outcome'] == 'received' else 'completed'
            self.result = DemandTurnResult(claim, candidate_text, candidate_stage,
                                           generation_calls, memory_calls, response_type, status,
                                           recall_calls=recall_calls, public_calls=public_calls)
            visible_text = state['confirmed_text'] if state else candidate_text
            if response_type != 'listen' and visible_text:
                yield visible_text
        except BaseException:
            self.settle_failure(cancelled)
            raise
