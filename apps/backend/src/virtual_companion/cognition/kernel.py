"""Opt-in A2 candidate: prepare, reply, then review the actual display surface.

The merged preparation call covers engagement, appraisal and direction. Two model
reviews share a call, but retain independent outcomes. This is an experimental
call arrangement, not a new default or a claim of legacy/personality equivalence.
State appraisal is projected through an explicitly supplied deterministic port;
model JSON never writes storage. No automatic recall or background cognition.
"""

import json
import threading
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256

from virtual_companion.boundary.contracts import Scope, VersionConflict, canonical
from virtual_companion.boundary.memory import FactKey
from virtual_companion.boundary.presentation import (
    MessageUnit,
    Presentation,
    ReviewedPresentation,
    review,
)
from virtual_companion.boundary.runtime import KernelFailure
from virtual_companion.boundary.turns import CognitiveResult
from virtual_companion.text.chronology import utc_microseconds

from .exchange_planning import ExchangePlan
from .fact_planning import FactPlan
from .preparation import (
    DUTIES,
    MAX_REVIEW_ATTEMPTS,
    CallControl,
    DutyAssessment,
    MemoryEvidence,
    assemble_context,
    validate_candidate,
)
from .reviews import ReviewContractError, StrictReviewGate
from .turn_guardian import TurnGuardian


@dataclass(frozen=True)
class SourceSelection:
    """Caller-selected displayed history and index references, never memory contents."""

    history: tuple = ()
    recall_refs: tuple = ()


def object_fields(value, fields):
    if not isinstance(value, dict) or set(value) != set(fields.split()):
        raise ValueError("Exact model object fields required")


def bounded_text(value, *, nonempty=True):
    if (
        not isinstance(value, str)
        or len(value.encode()) > 64000
        or (nonempty and not value.strip())
    ):
        raise ValueError("Bounded model text required")


class ControlledCalls:
    """One transport per active kernel; watch durable cancel/lease during blocked I/O.

    The transport must have bounded I/O and an interrupting cancel(), as LlamaModel
    does. Its exact check_context method owns tokenizer/template/output reservation.
    Counting has its existing two-second per-request bound; stream cancellation
    closes its socket. No inference worker is abandoned on cancellation.
    """

    def __init__(self, model, control, trace=None):
        self.model, self.control = model, control
        self.trace = [] if trace is None else trace
        self.call_count = 0

    def request(self, request):
        self.call_count += 1
        entry = {
            "stage": "transport_call",
            "request_stage": request["stage"],
            "call_id": f"call-{self.call_count}",
            "status": "entered",
            "context_sha256": request["context_sha256"],
            "request_sha256": sha256(canonical(request).encode()).hexdigest(),
        }
        self.trace.append(entry)
        try:
            result = self._request(request)
        except Exception as error:
            entry.update(status="failed", error_type=type(error).__name__)
            raise
        entry["status"] = "returned"
        return result

    def last_call(self, stage):
        return next(
            (
                r["call_id"]
                for r in reversed(self.trace)
                if r["stage"] == "transport_call" and r["request_stage"] == stage
            ),
            None,
        )

    def _request(self, request):
        messages = [{"role": "user", "content": canonical(request)}]
        self.control.checkpoint()
        stopped = threading.Event()
        whole_turn = isinstance(self.control, TurnGuardian)
        cancelled = self.control.cancelled if whole_turn else threading.Event()
        errors = []

        def observe():
            while not stopped.wait(0.025):
                try:
                    self.control.checkpoint()
                except Exception as error:
                    errors.append(error)
                    cancelled.set()
                    self.model.cancel()
                    return

        scope = getattr(self.model, "call_scope", None)
        with scope(cancelled) if scope is not None else nullcontext():
            observer = None
            if not whole_turn:
                observer = threading.Thread(target=observe, name="cognition-call-control")
                observer.start()
            try:
                self.model.check_context(messages)
                if cancelled.is_set():
                    if whole_turn:
                        self.control.raise_if_failed()
                    raise errors[0]
                chunks, size = [], 0
                for chunk in self.model.stream(messages, cancelled):
                    if not isinstance(chunk, str):
                        raise KernelFailure("model_contract")
                    size += len(chunk.encode())
                    if size > 64000:
                        self.model.cancel()
                        raise KernelFailure("model_contract")
                    chunks.append(chunk)
                result = "".join(chunks)
            finally:
                stopped.set()
                if observer is not None:
                    observer.join()
                # The durable cancellation/ownership failure takes precedence over an
                # incidental socket error caused by interrupting the same request.
                if errors:
                    raise errors[0]
                if whole_turn:
                    self.control.raise_if_failed()
        self.control.checkpoint()
        if not result.strip():
            raise KernelFailure("empty_model_result")
        try:

            def unique(pairs):
                data = {}
                for key, value in pairs:
                    if key in data:
                        raise ValueError("Duplicate model field")
                    data[key] = value
                return data

            parsed = json.loads(result, object_pairs_hook=unique)
            canonical(parsed)  # Reject NaN/Infinity, including nested values.
            if not isinstance(parsed, dict):
                raise ValueError("Model object required")
            return parsed
        except (ValueError, TypeError) as error:
            raise KernelFailure("model_contract") from error


class CognitiveKernel:
    def __init__(
        self,
        *,
        store,
        model,
        persona,
        select,
        project,
        policy,
        clock,
        lease_us,
        deadline_us,
        renewal_margin_us,
        review_gate=None,
        preparation=None,
        evidence_clock=None,
        exchange_planner=None,
        memory_planner=None,
    ):
        self.store, self.model, self.persona = store, model, persona
        self.select, self.project, self.policy, self.clock = select, project, policy, clock
        self.lease_us, self.deadline_us = lease_us, deadline_us
        self.renewal_margin_us = renewal_margin_us
        self.trace = []
        self.review_gate = StrictReviewGate() if review_gate is None else review_gate
        self.preparation = preparation
        self.exchange_planner = exchange_planner
        self.memory_planner = memory_planner
        # Lease ticks may be process monotonic; memory TTL and observation dates
        # are UTC timestamps and must never be compared with those ticks.
        self.evidence_clock = (
            evidence_clock
            if evidence_clock is not None
            else lambda: utc_microseconds(datetime.now(UTC))
        )
        self._active = threading.Lock()

    def generate(self, claim, snapshot, text):
        if not self._active.acquire(blocking=False):
            raise RuntimeError("Kernel transport already in use")
        try:
            self.trace = []
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
            self.trace.append(control.diagnostics)
            with TurnGuardian(control, self) as guardian:
                bind = getattr(self.model, "bind_turn", None)
                if bind is not None:
                    bind(claim, guardian)
                return self._select_generate(claim, snapshot, text, guardian)
        finally:
            self._active.release()

    def cancel(self):
        try:
            self.model.cancel()
        finally:
            try:
                if self.exchange_planner is not None:
                    self.exchange_planner.cancel()
            finally:
                if self.memory_planner is not None:
                    self.memory_planner.cancel()

    def _cancelled_or_stale(self, claim):
        item = self.store.inspect(claim.companion_id, claim.input_id, timeout=0.025)
        if item["status"] == "cancelled":
            return True
        run = item["run"]
        owner = item["owner"]
        if (
            item["status"] != "running"
            or run is None
            or run["status"] != "running"
            or owner is None
            or (owner["owner"], owner["epoch"]) != (claim.owner.owner_id, claim.owner.epoch)
            or (run["attempt"], run["owner"], run["epoch"])
            != (claim.attempt_id, claim.owner.owner_id, claim.owner.epoch)
        ):
            raise VersionConflict("Cognitive run ownership changed; reconcile before more work")
        return False

    def _select_generate(self, claim, snapshot, text, guardian):
        selection = self.select(claim)
        guardian.checkpoint()
        if not isinstance(selection, SourceSelection):
            raise ValueError("Typed source selection required")
        # Empty explicit references mean no recall, not the store's default scan.
        rows = self.store.recall(
            claim.companion_id, now_us=self.evidence_clock(), candidates=selection.recall_refs
        )
        memories = tuple(
            MemoryEvidence(
                claim.companion_id,
                row["record_id"],
                row["version"],
                row["status"],
                row["source_kind"],
                row["source_id"],
                row["content"],
                utc_microseconds(datetime.fromisoformat(row["expires_at"]))
                if row["expires_at"]
                else None,
                kind=row["kind"],
                fact_key=FactKey(**row["fact_key"]) if row["fact_key"] is not None else None,
            )
            for row in rows
        )
        if self.store.snapshot(claim.companion_id).write_version != snapshot.write_version:
            raise VersionConflict("Sources changed during context preparation")
        context = assemble_context(
            claim,
            snapshot,
            claim.input_id,
            text,
            self.persona(claim),
            now_us=self.evidence_clock(),
            history=selection.history,
            memories=memories,
            recall_refs=selection.recall_refs,
        )
        exchange_plan = None
        if self.exchange_planner is not None:
            try:
                exchange_plan = self.exchange_planner.plan(context, guardian)
                if not isinstance(exchange_plan, ExchangePlan):
                    raise ValueError("Typed provisional exchange plan required")
            except VersionConflict:
                raise
            except ValueError as error:
                raise KernelFailure("exchange_model_contract") from error
        memory_plan = None
        if self.memory_planner is not None:
            try:
                memory_plan = self.memory_planner.plan(context, guardian)
                if not isinstance(memory_plan, FactPlan):
                    raise ValueError("Typed provisional fact plan required")
            except VersionConflict:
                raise
            except ValueError as error:
                raise KernelFailure("fact_model_contract") from error
        return self._generate(
            context,
            ControlledCalls(self.model, guardian, self.trace),
            exchange_plan=exchange_plan,
            memory_plan=memory_plan,
        )

    def _generate(self, context, calls, *, exchange_plan=None, memory_plan=None):
        review_gate = self.review_gate
        preparation = self.preparation
        base = {
            "schema_version": 1,
            "context_sha256": context.fingerprint,
            "context": json.loads(context.payload),
        }
        if memory_plan is not None:
            base["memory_proposals"] = [
                effect.mutation.to_mapping() for effect in memory_plan.effects
            ]

        def invoke(stage, contract, **extra):
            request = {**base, "stage": stage, "response_contract": contract, **extra}
            # Preserve the fixed legacy empty-reply retry; this is independent
            # of the three candidate-review attempts, never a generic retry.
            for invocation in range(2 if stage == "reply" else 1):
                try:
                    result = calls.request(request)
                    break
                except KernelFailure as error:
                    self.trace.append({"stage": stage, "failure": error.reason})
                    if stage != "reply" or error.reason != "empty_model_result" or invocation:
                        raise
            if result.get("context_sha256") != context.fingerprint:
                raise KernelFailure("model_contract")
            self.trace.append(
                {"stage": stage, "response_sha256": sha256(canonical(result).encode()).hexdigest()}
            )
            return result

        prepared = (
            preparation.prepare(
                context,
                calls,
                **(
                    {"memory_proposals": base["memory_proposals"]}
                    if memory_plan is not None
                    else {}
                ),
            )
            if preparation is not None
            else invoke(
                "prepare",
                {
                    "context_sha256": "echo request hash",
                    "engagement": {
                        "mode": "respond|silence",
                        "reason": "nonempty reason; silence must be deliberate",
                    },
                    "appraisal": {
                        "source_input_id": "current input id",
                        "state_version": "current integer write version",
                        "observation": "grounded appraisal",
                    },
                    "director": {
                        "state_version": "current integer write version",
                        "instructions": "expression direction",
                    },
                },
            )
        )
        try:
            object_fields(prepared, "context_sha256 engagement appraisal director")
            if prepared["context_sha256"] != context.fingerprint:
                raise ValueError("Foreign preparation envelope")
            object_fields(prepared["engagement"], "mode reason")
            object_fields(prepared["appraisal"], "source_input_id state_version observation")
            object_fields(prepared["director"], "state_version instructions")
            allowed = (
                {"respond", "silence", "deferred"}
                if preparation is not None
                else {"respond", "silence"}
            )
            if prepared["engagement"]["mode"] not in allowed:
                # Deferred obligations require an explicit backend event contract;
                # never turn them into a successfully committed silent response.
                raise ValueError("Unsupported engagement disposition")
            bounded_text(prepared["engagement"]["reason"])
            bounded_text(prepared["appraisal"]["observation"])
            bounded_text(prepared["director"]["instructions"])
            for value in (
                prepared["appraisal"]["state_version"],
                prepared["director"]["state_version"],
            ):
                if type(value) is not int or value != context.state.write_version:
                    raise ValueError("Stale preparation")
            if prepared["appraisal"]["source_input_id"] != context.claim.input_id:
                raise ValueError("Foreign appraisal source")
        except (ValueError, TypeError, KeyError) as error:
            raise KernelFailure("model_contract") from error

        assessment = (
            preparation.project(context, prepared)
            if preparation is not None
            else self.project(context, prepared["appraisal"])
        )
        if prepared["engagement"]["mode"] == "deferred":
            validate_deferred = getattr(preparation, "validate_deferred", None)
            durable = (
                validate_deferred(context, assessment)
                if validate_deferred
                else (
                    assessment is not None
                    and any(
                        p.get("source_input_id") == context.claim.input_id
                        and p.get("status") == "delayed"
                        for p in json.loads(assessment.values_json)
                        .get("a2_pilot", {})
                        .get("pending_obligations", [])
                    )
                )
            )
            if durable is not True:
                raise ValueError("Deferred decision requires a durable pending obligation proposal")
        feedback = None
        run = context.claim
        for attempt in range(1, MAX_REVIEW_ATTEMPTS + 1):
            if prepared["engagement"]["mode"] in {"silence", "deferred"}:
                raw = {"semantic_text": "", "units": [], "context_sha256": context.fingerprint}
            else:
                raw = invoke(
                    "reply",
                    {
                        "context_sha256": "echo request hash",
                        "semantic_text": "semantic reply",
                        "units": [
                            {
                                "unit_id": "unique id",
                                "kind": "text|action",
                                "text": "actual display",
                            }
                        ],
                    },
                    preparation=prepared,
                    feedback=feedback,
                    review_attempt=attempt,
                )
            try:
                object_fields(raw, "context_sha256 semantic_text units")
                bounded_text(
                    raw["semantic_text"], nonempty=prepared["engagement"]["mode"] == "respond"
                )
                if not isinstance(raw["units"], list) or len(raw["units"]) > 32:
                    raise ValueError("Bounded unit array required")
                for unit in raw["units"]:
                    object_fields(unit, "unit_id kind text")
                units = tuple(MessageUnit(**unit) for unit in raw["units"])
                if prepared["engagement"]["mode"] == "respond" and not units:
                    raise ValueError("Nonempty response surface required")
                candidate = Presentation(
                    Scope(run.companion_id, run.session_id),
                    run.input_id,
                    run.attempt_id,
                    f"{run.attempt_id}-c{attempt}",
                    attempt,
                    raw["semantic_text"],
                    units,
                )
            except (ValueError, TypeError, KeyError) as error:
                raise KernelFailure("model_contract") from error
            verdict = invoke(
                "review",
                {
                    "context_sha256": "echo request hash",
                    "candidate_sha256": "echo exact actual surface hash",
                    **review_gate.response_contract,
                },
                preparation=prepared,
                candidate=candidate.to_mapping(),
                candidate_sha256=candidate.fingerprint,
                review_attempt=attempt,
            )
            try:
                adjudications = review_gate.evaluate(context, candidate, verdict)
            except ReviewContractError as error:
                raise KernelFailure("model_contract") from error
            review_id = f"{run.attempt_id}-r{attempt}"
            checked = review(
                candidate,
                self.policy,
                review_id,
                model_approved=adjudications["causal_review"]["passed"]
                and adjudications["expression_review"]["passed"],
            )
            feedback = {
                **verdict,
                "adjudications": adjudications,
                "surface_issues": list(checked.issues),
            }
            self.trace.append(
                {
                    "stage": "decision",
                    "attempt": attempt,
                    "passed": checked.passed,
                    "candidate_sha256": candidate.fingerprint,
                    "surface_issues": list(checked.issues),
                    "adjudications": adjudications,
                }
            )
            if not checked.passed:
                continue
            finalize = getattr(preparation, "finalize", None)
            if finalize is not None:
                assessment = finalize(context, assessment, candidate)
            result = CognitiveResult(
                run,
                candidate.candidate_id,
                attempt,
                candidate.text,
                review_id,
                sha256(candidate.text.encode()).hexdigest(),
                assessment=assessment,
                presentation=ReviewedPresentation(candidate, self.policy, checked),
                memories=memory_plan.effects if memory_plan else (),
                exchanges=exchange_plan.proposals if exchange_plan else (),
                exchange_sources=exchange_plan.sources if exchange_plan else (),
            )
            sources = (
                preparation.duty_sources(prepared)
                if preparation is not None
                else {
                    d: ("merged_preparation", "prepare")
                    for d in ("engagement", "state_appraisal", "expression_director")
                }
            )
            sources.update(
                main_reply=("reviewed_reply", "reply"),
                causal_review=("adjudicated_review", "review"),
                expression_review=("adjudicated_review", "review"),
            )
            duties = tuple(
                DutyAssessment(
                    duty,
                    "complete",
                    context.fingerprint,
                    candidate.fingerprint,
                    sources[duty][0],
                    calls.last_call(sources[duty][1]) if sources[duty][1] else None,
                )
                for duty in DUTIES
            )
            if prepared["engagement"]["mode"] in {"silence", "deferred"}:
                duties = tuple(
                    DutyAssessment(
                        d.duty,
                        "not_required",
                        d.context_sha256,
                        d.candidate_sha256,
                        prepared["engagement"]["reason"],
                        None,
                    )
                    if d.duty == "main_reply"
                    else d
                    for d in duties
                )
            validate_candidate(context, result, duties, review_attempt=attempt)
            self.trace.append({"stage": "accepted", "duties": [asdict(d) for d in duties]})
            calls.control.checkpoint()
            return result
        raise KernelFailure("review_exhausted")
