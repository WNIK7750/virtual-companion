"""Bounded source maintenance using the existing support-bound extractors.

The host calls run_one when foreground inference is idle. Assistant ACK text
cannot become a user fact, and user text cannot prove displayed speech.
"""

import json
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from threading import Event, Thread
from unicodedata import category

from virtual_companion.boundary.contracts import (
    Scope,
    SemanticConflict,
    SourceResolutionUnresolved,
    VersionConflict,
    canonical,
)
from virtual_companion.boundary.evidence import EvidenceTurn, evidence_digest
from virtual_companion.boundary.exchange_resolution import SourceResolution, digest
from virtual_companion.boundary.fact_resolution import FactSourceResolution
from virtual_companion.boundary.memory import FactKey, MemoryMutation
from virtual_companion.boundary.memory_management import ExplicitMemoryCommand
from virtual_companion.boundary.memory_relevance import lexical_relevance
from virtual_companion.boundary.memory_support import command_digest, target_digest

from .errors import FactSupportUnresolved
from .exchange_model import VERSION as EXCHANGE_VERSION
from .exchange_model import (
    ExchangeModel,
    _act_segments,
    _coverage_segments,
)
from .fact_model import FactModel
from .fact_target_selector import FactTargetSelector
from .joint_source_proposal import VERSION as JOINT_VERSION
from .joint_source_proposal import JointSourceProposal
from .memory_command_model import VERSION as MEMORY_COMMAND_VERSION
from .memory_command_model import MemoryCommandModel


class _StageInvoke:
    """Keep a bounded, content-free failure node for persisted diagnostics."""

    _known = frozenset({"source_read", "fact_targets", "fact_target_select",
                        "fact_extract", "fact_support", "fact_repair", "exchange_context",
                        "exchange_intent", "exchange_extract", "exchange_assess",
                        "exchange_support", "source_propose", "source_repair", "fact_assess",
                        "memory_command", "memory_command_support", "commit"})

    def __init__(self, invoke, *, store=None, claim=None):
        self.invoke, self.stage = invoke, "source_read"
        self.source = (store, claim)
        self.checkpoint = None
        self.supports_bound_evidence = bool(getattr(invoke, "supports_bound_evidence", False))

    def __call__(self, stage, messages, schema, cancelled):
        self.stage = stage if stage in self._known else "unknown"
        if self.checkpoint is not None:
            self.checkpoint()
        try:
            bind = getattr(self.invoke, "for_source", None)
            if self.source is not None and callable(bind):
                self.invoke = bind(*self.source)
            self.source = None
            return self.invoke(stage, messages, schema, cancelled)
        finally:
            if self.checkpoint is not None:
                self.checkpoint()

    def mark(self, stage):
        self.stage = stage if stage in self._known else "unknown"

    @property
    def failure_code(self):
        return f"semantic_processing_failed.{self.stage}"


class _BackgroundGuardian:
    """Observe one leased call without holding a DB lock across inference."""

    def __init__(self, store, claim, clock, external_cancelled=None):
        self.store, self.claim, self.clock = store, claim, clock
        self.external_cancelled = external_cancelled
        self.cancelled, self.stopped = Event(), Event()
        self.failure = None
        self.observer = Thread(target=self._observe, name="background-source-control")

    def checkpoint(self):
        if self.failure is not None:
            raise self.failure
        try:
            if self.external_cancelled is not None and self.external_cancelled.is_set():
                raise InterruptedError("Source access cancelled")
            self.store.background_claim_current(self.claim, now_us=self.clock())
        except Exception as error:
            self.failure = error
            self.cancelled.set()
            raise

    def _observe(self):
        while not self.stopped.wait(0.025):
            try:
                self.checkpoint()
            except Exception:
                return

    def __enter__(self):
        self.checkpoint()
        self.observer.start()
        return self

    def __exit__(self, exc_type, *_):
        self.stopped.set()
        self.observer.join()
        if exc_type is None and self.failure is not None:
            raise self.failure
        if (exc_type is None and self.external_cancelled is not None
                and self.external_cancelled.is_set()):
            raise InterruptedError("Source access cancelled")


_EXACT_REPEAT_EVALUATOR = "exact-repeat-host.v1"


def _exchange_targets(store, claim, sources, *, now_us):
    """A bounded relevant view, not a certificate about the whole catalog."""
    current = next(source.text for source in sources if source.source_id == claim.source)
    current_score = lexical_relevance(current)
    context_score = lexical_relevance("\n".join(source.text for source in sources))
    return sorted(store.ranked_pending_exchanges(
        claim.companion, now_us=now_us,
        score=lambda value: (current_score(value), context_score(value)), limit=100,
        before_source=(Scope(claim.companion, claim.session), claim.role, claim.source),
    ), key=lambda value: value["exchange_id"])


def _exact_repeat_completion(store, claim, text, catalog, *, now_us):
    """Build a no-change receipt only for one complete, byte-exact current fact."""
    matches = tuple(
        (item["record_id"], item["version"])
        for item in catalog
        if not item["content_truncated"] and item["content"] == text
    )
    if len(matches) != 1:
        return None
    targets = store.fact_targets(claim.companion, now_us=now_us, candidates=matches)
    target = targets.get(matches[0][0])
    if target is None or target["content"] != text or target["version"] != matches[0][1]:
        return None
    fact_key = target.get("fact_key")
    if not isinstance(fact_key, dict) or set(fact_key) != {"subject", "predicate", "scope"}:
        return None
    exchange_targets = sorted(
        store.pending_exchanges(claim.companion, now_us=now_us),
        key=lambda value: value["exchange_id"],
    )
    if exchange_targets:
        return None
    sources = store.background_exchange_context(claim)
    context_sha256 = evidence_digest(sources)
    scope = Scope(claim.companion, claim.session)
    expiry = datetime.fromisoformat(target["expires_at"]) if target["expires_at"] else None
    command = MemoryMutation(
        scope,
        f"{claim.source}-operation-0",
        "assert_fact",
        f"{claim.source}-record-0",
        text,
        "input",
        claim.source,
        canonical({"quote": text, "support": {}}),
        datetime(1970, 1, 1, tzinfo=UTC) + timedelta(microseconds=now_us),
        FactKey(**fact_key),
        expires_at=expiry,
    )
    stored_target = {key: value for key, value in target.items() if key != "companion_id"}
    support = {
        "verdict": "supported",
        "evaluator": _EXACT_REPEAT_EVALUATOR,
        "content_sha256": sha256(text.encode()).hexdigest(),
        "source_sha256": claim.source_hash,
        "target_sha256": target_digest(stored_target),
        "command_sha256": command_digest(command, text),
    }
    command = replace(
        command,
        reason=canonical({"quote": text, "support": support}),
    )
    candidate = canonical(
        {
            "operation": "add",
            "type": "short_term" if expiry else "long_term",
            "evidenceTurnId": claim.source,
            "evidenceQuote": text,
            "supersedesAtomId": None,
            "expiresAt": int(expiry.timestamp() * 1000) if expiry else None,
            "factKey": fact_key,
        }
    )
    fact_resolution = FactSourceResolution(
        claim.source,
        claim.source_hash,
        context_sha256,
        (text,),
        ("fact",),
        (candidate,),
        ("accepted",),
        ("fact",),
        digest([command.fingerprint]),
        evaluator=_EXACT_REPEAT_EVALUATOR,
    )
    resolution = SourceResolution(
        claim.source,
        claim.source_hash,
        context_sha256,
        digest(exchange_targets),
        digest([]),
        (),
        digest([]),
        digest([]),
        True,
        _EXACT_REPEAT_EVALUATOR,
    )
    return (command,), sources, resolution, fact_resolution


@dataclass(frozen=True)
class UncommittedSource:
    """Semantic refusal before business commit, bound to its delegated owner."""

    foreground: object
    source: str


def _fact_source_invoke(invoke, source, prepared_intents):
    """Bind continuous reading ranges without deciding whether they contain facts.

    The full source remains context for support and provenance. The compiler
    still binds mutations to its original source hash, not a sliced fake source.
    """
    text = source.text[:source.confirmed_end]
    segments = _coverage_segments(text)
    starts, cursor = [], 0
    for segment in segments:
        # The reading scaffold omits whitespace-only clauses. Align its original
        # text in order, never search for a quote that may occur more than once.
        while not text.startswith(segment["text"], cursor):
            if cursor >= len(text) or not text[cursor].isspace():
                raise ValueError("Fact scaffold differs from current source")
            cursor += 1
        starts.append(cursor)
        cursor += len(segment["text"])
    statements, other_spans, statement_segments, other_segments = [], [], set(), set()
    for act in json.loads(prepared_intents.interpretation)["acts"]:
        covered = _act_segments(act, segments)
        if not covered:
            raise ValueError("Fact range differs from current source")
        first, last = min(covered), max(covered)
        start = starts[first] + len(segments[first]["text"]) - len(
            segments[first]["text"].lstrip()
        )
        end = start + len(act["evidenceQuote"])
        if (act["evidenceTurnId"] != source.source_id
                or text[start:end] != act["evidenceQuote"]):
            raise ValueError("Fact range differs from current source")
        if act["intent"] == "statement":
            statements.append((start, end, first, last))
            statement_segments.update(covered)
        else:
            other_spans.append((start, end, first, last))
            other_segments.update(covered)
    excluded = other_segments - statement_segments
    merged = []
    for start, end, first, last in sorted(statements):
        if merged:
            previous_start, previous_end, previous_first, previous_last = merged[-1]
            gap = text[previous_end:start]
            if (not excluded.intersection(range(previous_last + 1, first))
                    and all(char.isspace() or category(char).startswith("P") for char in gap)):
                merged[-1] = (previous_start, max(previous_end, end), previous_first,
                              max(previous_last, last))
                continue
        merged.append((start, end, first, last))
    ranges = [{"start": start, "end": end, "text": text[start:end]}
              for start, end in sorted({(start, end)
                                        for start, end, _, _ in merged + other_spans})]

    def scoped(stage, messages, schema, cancelled):
        if stage not in {"fact_extract", "fact_repair"}:
            return invoke(stage, messages, schema, cancelled)
        scoped_messages = deepcopy(messages)
        system, schema_text = scoped_messages[0]["content"].split("\nJSON schema:\n", 1)
        scoped_messages[0]["content"] = (
            system + "\nfact_evidence_ranges是宿主按当前source原始索引绑定的连续阅读范围；"
            "它们不是事实批准，也不以会话意图标签判定有无事实。"
            "请求也可能明确提供事实；问题、条件、否定、他人引述和创作需结合完整source判定。"
            "evidenceQuote必须逐字包含在一个范围中，不拼接相隔范围或从历史targets补出事实。"
            "没有明确提供的可记忆事实返回空operations。"
            "\nJSON schema:\n" + schema_text
        )
        data = json.loads(scoped_messages[-1]["content"])
        data["fact_evidence_ranges"] = ranges
        scoped_messages[-1]["content"] = canonical(data)
        result = invoke(stage, scoped_messages, schema, cancelled)
        if isinstance(result, dict) and isinstance(result.get("operations"), list):
            for operation in result["operations"]:
                if isinstance(operation, dict):
                    quote = operation.get("evidenceQuote")
                    if isinstance(quote, str) and not any(quote in r["text"] for r in ranges):
                        raise ValueError("Fact proposal differs from current source ranges")
        return result
    return scoped


class UserFactWorker:
    def __init__(self, store, invoke, *, clock, lease_us=300_000_000,
                 resolution_protocol=False, resolution_strategy="candidate_first",
                 joint_proposals=False, target_selection=False, memory_commands=False):
        if joint_proposals and (not resolution_protocol or resolution_strategy != "intent_first"):
            raise ValueError("Joint proposals require resolved intent-first source processing")
        if type(target_selection) is not bool or type(memory_commands) is not bool:
            raise ValueError("Explicit target and memory command policies required")
        self.store, self.invoke, self.clock, self.lease_us = store, invoke, clock, lease_us
        self.resolution_protocol = resolution_protocol
        self.resolution_strategy = resolution_strategy
        self.joint_proposals = joint_proposals
        self.target_selection = target_selection
        self.memory_commands = memory_commands

    def run_one(self, companion, worker, *, source=None, cancelled=None, foreground=None):
        if cancelled is not None and cancelled.is_set():
            return None

        def delegated_clock():
            if cancelled is not None and cancelled.is_set():
                raise InterruptedError("Delegated source cancelled")
            return self.clock()

        claim = self.store.claim_background(
            companion, worker, now_us=self.clock(), lease_us=self.lease_us, role="user",
            **({"source": source} if source is not None else {}),
            **({"foreground": foreground, "clock": delegated_clock}
               if foreground is not None else {}),
        )
        if claim is None:
            return None
        traced = _StageInvoke(self.invoke, store=self.store, claim=claim)
        explicit_command = None
        clarification = None
        exact_repeat = None
        commands = ()
        skip_fact_processing = False
        fact_evaluator = None
        try:
            with _BackgroundGuardian(self.store, claim, self.clock, cancelled) as guardian:
                if cancelled is not None:
                    traced.checkpoint = guardian.checkpoint
                text = self.store.background_source_text(claim.companion, claim.job)
                scope = Scope(claim.companion, claim.session)
                source = EvidenceTurn(scope, claim.source, "user", text, len(text))
                memory_catalog = None
                if self.memory_commands:
                    prior = self.store.memory_source_receipt(claim.companion, claim.job)
                    skip_fact_processing = bool(
                        prior
                        and prior.get("disposition") == "memory_command_abandoned"
                    )
                if self.memory_commands and not skip_fact_processing:
                    traced.mark("fact_targets")
                    memory_catalog = self.store.fact_target_catalog(
                        claim.companion, now_us=self.clock()
                    )
                    try:
                        interpretation = MemoryCommandModel(
                            traced, guardian.cancelled
                        ).interpret(text, memory_catalog)
                    except SourceResolutionUnresolved:
                        clarification = {
                            "reason": "needs_clarification.memory_command",
                            "interpretation_sha256": sha256(
                                canonical(
                                    {
                                        "evaluator": MEMORY_COMMAND_VERSION,
                                        "source_sha256": claim.source_hash,
                                        "disposition": "catalog_pages_disagree",
                                    }
                                ).encode()
                            ).hexdigest(),
                            "action": "unclear",
                            "requested_scope": "none",
                            "unresolved": ("intent",),
                            "affected_records": (),
                        }
                        interpretation = None
                    guardian.checkpoint()
                    if (
                        interpretation is not None
                        and interpretation.source_sha256 != claim.source_hash
                    ):
                        raise ValueError("Memory interpretation belongs to another source")
                    if interpretation is not None and interpretation.executable:
                        record_id, version = interpretation.target_refs[0]
                        operation_id = "memory-" + sha256(
                            canonical(
                                {
                                    "companion": claim.companion,
                                    "job": claim.job,
                                    "source": claim.source,
                                    "source_sha256": claim.source_hash,
                                    "interpretation_sha256": interpretation.fingerprint,
                                }
                            ).encode()
                        ).hexdigest()
                        explicit_command = ExplicitMemoryCommand(
                            scope,
                            operation_id,
                            interpretation.action,
                            record_id,
                            version,
                            "conversation",
                            claim.source,
                            claim.source_hash,
                            interpretation.requested_scope,
                            interpretation.coverage,
                            interpretation.evaluator,
                            interpretation.fingerprint,
                            interpretation.evidence_quote,
                            interpretation.quote_start,
                            interpretation.quote_end,
                            interpretation.replacement,
                        )
                    elif (
                        interpretation is not None
                        and interpretation.disposition == "clarify"
                    ):
                        reason = (
                            "needs_clarification.mixed_source"
                            if "mixed_content" in interpretation.unresolved
                            else "needs_clarification.memory_scope"
                            if "scope" in interpretation.unresolved
                            else "needs_clarification.memory_command"
                        )
                        clarification = {
                            "reason": reason,
                            "interpretation_sha256": interpretation.fingerprint,
                            "action": interpretation.action,
                            "requested_scope": interpretation.requested_scope,
                            "unresolved": interpretation.unresolved,
                            "affected_records": interpretation.target_refs,
                        }
                    elif (
                        interpretation is not None
                        and interpretation.disposition == "not_command"
                        and self.resolution_protocol
                    ):
                        exact_repeat = _exact_repeat_completion(
                            self.store,
                            claim,
                            text,
                            memory_catalog,
                            now_us=self.clock(),
                        )
                        guardian.checkpoint()
                if explicit_command is not None or clarification is not None:
                    guardian.checkpoint()
                elif exact_repeat is not None:
                    pass
                elif skip_fact_processing:
                    traced.mark("exchange_context")
                    sources = self.store.background_exchange_context(claim, now_us=self.clock())
                    exchanges = _exchange_targets(
                        self.store, claim, sources, now_us=self.clock()
                    )
                    proposal_invoke = traced
                    traced.mark("exchange_extract")
                    exchange_model = ExchangeModel(traced, guardian.cancelled)
                    exchange_inputs = dict(
                        scope=scope,
                        batch_id=claim.source,
                        now_us=self.clock(),
                        turns=sources,
                        targets={t["exchange_id"]: t for t in exchanges},
                        eligible_source_ids=(claim.source,),
                    )
                    if self.resolution_protocol:
                        proposals, resolution = exchange_model.resolve_source(
                            **exchange_inputs, strategy=self.resolution_strategy
                        )
                    else:
                        proposals = exchange_model.extract(**exchange_inputs)
                    fact_evaluator = "memory-command-abandoned.v1"
                    guardian.checkpoint()
                else:
                    prepared_intents = None
                    proposal_invoke = traced
                    if (self.memory_commands and self.resolution_protocol
                            and self.resolution_strategy == "intent_first"
                            and not self.joint_proposals):
                        traced.mark("exchange_context")
                        sources = self.store.background_exchange_context(claim, now_us=self.clock())
                        exchanges = _exchange_targets(
                            self.store, claim, sources, now_us=self.clock()
                        )
                        exchange_model = ExchangeModel(traced, guardian.cancelled)
                        prepared_intents = exchange_model.prepare_source_intents(
                            turns=sources, targets={t["exchange_id"]: t for t in exchanges},
                            eligible_source_ids=(claim.source,),
                        )
                        guardian.checkpoint()
                    traced.mark("fact_targets")
                    bound_targets = ()
                    if self.target_selection:
                        catalog = (
                            memory_catalog
                            if memory_catalog is not None
                            else self.store.fact_target_catalog(
                                claim.companion, now_us=self.clock()
                            )
                        )
                        traced.mark("fact_target_select")
                        selection = FactTargetSelector(traced, guardian.cancelled).resolve(
                            text, catalog
                        )
                        guardian.checkpoint()
                        traced.mark("fact_targets")
                        if selection.intent in {"correction", "forget_request"}:
                            if len(selection.targets) != 1:
                                raise SourceResolutionUnresolved(
                                    "Correction requires one unique active fact target"
                                )
                            bound_targets = tuple(record for record, _ in selection.targets)
                            targets = self.store.fact_targets(
                                claim.companion, now_us=self.clock(),
                                candidates=selection.targets,
                            )
                        else:
                            targets = {}
                    else:
                        try:
                            targets = self.store.fact_targets(
                                claim.companion, now_us=self.clock()
                            )
                        except ValueError as error:
                            if str(error) != (
                                "Fact target capacity requires explicit retrieval selection"
                            ):
                                raise
                            catalog = (
                                memory_catalog
                                if memory_catalog is not None
                                else self.store.fact_target_catalog(
                                    claim.companion, now_us=self.clock()
                                )
                            )
                            traced.mark("fact_target_select")
                            selected = FactTargetSelector(traced, guardian.cancelled).select(
                                text, catalog
                            )
                            guardian.checkpoint()
                            traced.mark("fact_targets")
                            targets = self.store.fact_targets(
                                claim.companion, now_us=self.clock(), candidates=selected
                            )
                    if self.joint_proposals:
                        traced.mark("exchange_context")
                        sources = self.store.background_exchange_context(
                            claim, now_us=self.clock()
                        )
                        exchanges = _exchange_targets(
                            self.store, claim, sources, now_us=self.clock()
                        )
                        proposal_invoke = JointSourceProposal(
                            traced, sources=sources, source_id=claim.source,
                            targets={t["exchange_id"]: t for t in exchanges})
                    traced.mark("fact_extract")
                    fact_model = FactModel(
                        (_fact_source_invoke(proposal_invoke, source, prepared_intents)
                         if prepared_intents is not None else proposal_invoke),
                        guardian.cancelled,
                        bind_content=(self.resolution_protocol
                                      and self.resolution_strategy == "intent_first"),
                        allow_conflict_repair=claim.attempts == 1,
                        bound_target_ids=bound_targets,
                        allow_non_fact_rejection=self.joint_proposals,
                    )
                    if self.joint_proposals:
                        fact_model.version = f"{JOINT_VERSION}.{fact_model.version}"
                    commands = fact_model.extract(
                        scope=scope, batch_id=claim.source, now_us=self.clock(),
                        turn=source, targets=targets,
                    )
                    fact_evaluator = fact_model.version
                    guardian.checkpoint()
                    traced.mark("exchange_context")
                    if not self.joint_proposals and prepared_intents is None:
                        sources = self.store.background_exchange_context(claim, now_us=self.clock())
                        exchanges = _exchange_targets(
                            self.store, claim, sources, now_us=self.clock()
                        )
                    traced.mark("exchange_extract")
                    if prepared_intents is None:
                        exchange_model = ExchangeModel(proposal_invoke, guardian.cancelled)
                    exchange_inputs = dict(
                        scope=scope, batch_id=claim.source, now_us=self.clock(),
                        turns=sources, targets={t["exchange_id"]: t for t in exchanges},
                        eligible_source_ids=(claim.source,),
                    )
                    if self.resolution_protocol:
                        proposals, resolution = exchange_model.resolve_source(
                            **exchange_inputs, strategy=self.resolution_strategy,
                            _prepared_intents=prepared_intents)
                    else:
                        proposals = exchange_model.extract(**exchange_inputs)
                    guardian.checkpoint()
            traced.mark("commit")
            commit_clock = {"clock": delegated_clock} if foreground is not None else {}
            if explicit_command is not None:
                return self.store.execute_memory_command(
                    explicit_command, now_us=self.clock(), claim=claim, **commit_clock,
                )
            if clarification is not None:
                return self.store.protect_memory_source(
                    claim,
                    now_us=self.clock(),
                    **clarification,
                )
            if exact_repeat is not None:
                commands, sources, resolution, fact_resolution = exact_repeat
                return self.store.finish_resolved_background(
                    claim,
                    commands,
                    evaluator=_EXACT_REPEAT_EVALUATOR,
                    now_us=self.clock(),
                    exchanges=(),
                    sources=sources,
                    resolution=resolution,
                    fact_resolution=fact_resolution,
                    **commit_clock,
                )
            if self.resolution_protocol:
                return self.store.finish_resolved_background(
                    claim, commands, evaluator=fact_evaluator, now_us=self.clock(),
                    exchanges=proposals, sources=sources, resolution=resolution,
                    **({"fact_resolution": proposal_invoke.completed_fact_resolution(commands)}
                       if self.joint_proposals and not skip_fact_processing else {}),
                    **commit_clock,
                )
            return self.store.finish_background(
                claim, commands, evaluator=fact_evaluator, now_us=self.clock(),
                exchanges=proposals, sources=sources,
                **commit_clock,
            )
        except FactSupportUnresolved:
            # Only extraction can produce this result. A commit exception must
            # never be reported as an operation with no effects.
            if traced.stage not in {"fact_extract", "fact_support", "fact_repair"}:
                raise
            self.store.protect_background(
                claim, error="needs_clarification.source_coverage", now_us=delegated_clock()
            )
            if foreground is not None:
                return UncommittedSource(foreground, claim.source)
            raise
        except (SemanticConflict, SourceResolutionUnresolved) as error:
            reason = ("needs_clarification.fact_target_conflict"
                      if isinstance(error, SemanticConflict)
                      else "needs_clarification.source_coverage")
            try:
                if self.memory_commands:
                    affected = self.store.conflicting_fact_targets(commands)
                    interpretation_sha256 = sha256(
                        canonical(
                            {
                                "evaluator": "fact-source-recovery.v1",
                                "source_sha256": claim.source_hash,
                                "commands": [command.fingerprint for command in commands],
                                "reason": reason,
                            }
                        ).encode()
                    ).hexdigest()
                    partial_fact_commit = (
                        isinstance(error, SourceResolutionUnresolved)
                        and bool(commands)
                        and traced.stage.startswith("exchange_")
                        and not self.joint_proposals
                        and hasattr(self.store, "finish_facts_with_clarification")
                    )
                    if partial_fact_commit:
                        self.store.finish_facts_with_clarification(
                            claim,
                            commands,
                            evaluator=fact_evaluator,
                            now_us=self.clock(),
                            sources=sources,
                            reason=reason,
                            interpretation_sha256=interpretation_sha256,
                            action="unclear",
                            requested_scope=(
                                "single_record" if len(affected) == 1 else "none"
                            ),
                            unresolved=("support",),
                            affected_records=affected,
                        )
                    else:
                        self.store.protect_memory_source(
                            claim,
                            reason=reason,
                            interpretation_sha256=interpretation_sha256,
                            action="unclear",
                            requested_scope=(
                                "single_record" if len(affected) == 1 else "none"
                            ),
                            unresolved=("support",),
                            affected_records=affected,
                            now_us=self.clock(),
                        )
                else:
                    self.store.protect_background(
                        claim, error=reason, now_us=self.clock()
                    )
            except VersionConflict:
                pass
            if isinstance(error, SourceResolutionUnresolved):
                raise
            return None
        except (VersionConflict, InterruptedError):
            # A newer input or lease owner took precedence; old work cannot
            # finish or mark the replacement failed.
            try:
                self.store.requeue_background(claim, now_us=self.clock())
            except VersionConflict:
                pass
            return None
        except Exception:
            try:
                self.store.fail_background(
                    claim, error=traced.failure_code, now_us=self.clock()
                )
            except VersionConflict:
                pass
            raise


class AssistantExchangeWorker:
    """Process only actual terminal ACK sources, never model-suggested speech."""

    def __init__(self, store, invoke, *, clock, lease_us=300_000_000,
                 resolution_protocol=False, resolution_strategy="candidate_first"):
        self.store, self.invoke, self.clock, self.lease_us = store, invoke, clock, lease_us
        self.resolution_protocol = resolution_protocol
        self.resolution_strategy = resolution_strategy

    def run_one(self, companion, worker):
        claim = self.store.claim_background(
            companion, worker, now_us=self.clock(), lease_us=self.lease_us, role="assistant"
        )
        if claim is None:
            return None
        traced = _StageInvoke(self.invoke, store=self.store, claim=claim)
        try:
            invalidated = self.store.invalidate_background_source(
                claim, now_us=self.clock(), clock=self.clock,
            )
            if invalidated is not None:
                return invalidated
            with _BackgroundGuardian(self.store, claim, self.clock) as guardian:
                traced.mark("exchange_context")
                sources = self.store.background_exchange_context(claim, now_us=self.clock())
                targets = _exchange_targets(self.store, claim, sources, now_us=self.clock())
                traced.mark("exchange_extract")
                model = ExchangeModel(traced, guardian.cancelled)
                exchange_inputs = dict(
                    scope=Scope(claim.companion, claim.session), batch_id=claim.source,
                    now_us=self.clock(), turns=sources,
                    targets={t["exchange_id"]: t for t in targets},
                    eligible_source_ids=(claim.source,),
                )
                if self.resolution_protocol:
                    proposals, resolution = model.resolve_source(
                        **exchange_inputs, strategy=self.resolution_strategy)
                else:
                    proposals = model.extract(**exchange_inputs)
                guardian.checkpoint()
            traced.mark("commit")
            if self.resolution_protocol:
                return self.store.finish_resolved_assistant_background(
                    claim, proposals, sources, evaluator=resolution.evaluator,
                    now_us=self.clock(), resolution=resolution,
                )
            return self.store.finish_assistant_background(
                claim, proposals, sources, evaluator=EXCHANGE_VERSION, now_us=self.clock()
            )
        except SourceResolutionUnresolved:
            try:
                self.store.protect_background(
                    claim, error="needs_clarification.source_coverage", now_us=self.clock()
                )
            except VersionConflict:
                pass
            raise
        except (VersionConflict, InterruptedError) as error:
            try:
                if isinstance(error, VersionConflict):
                    invalidated = self.store.invalidate_background_source(
                        claim, now_us=self.clock(), clock=self.clock,
                    )
                    if invalidated is not None:
                        return invalidated
                self.store.requeue_background(claim, now_us=self.clock())
            except VersionConflict:
                pass
            return None
        except Exception:
            try:
                self.store.fail_background(
                    claim, error=traced.failure_code, now_us=self.clock()
                )
            except VersionConflict:
                pass
            raise


class BackgroundScheduler:
    """Bounded FIFO dispatch; foreground admission always wins the next claim."""

    def __init__(self, store, *, user_invoke, assistant_invoke, clock,
                 lease_us=300_000_000, resolution_protocol=False,
                 resolution_strategy="candidate_first", joint_proposals=False,
                 target_selection=False, memory_commands=False):
        self.store, self.clock = store, clock
        self.user = UserFactWorker(store, user_invoke, clock=clock, lease_us=lease_us,
                                   resolution_protocol=resolution_protocol,
                                   resolution_strategy=resolution_strategy,
                                   joint_proposals=joint_proposals,
                                   target_selection=target_selection,
                                   memory_commands=memory_commands)
        self.assistant = AssistantExchangeWorker(
            store, assistant_invoke, clock=clock, lease_us=lease_us,
            resolution_protocol=resolution_protocol,
            resolution_strategy=resolution_strategy,
        )

    def run_next(self, worker, *, companion=None):
        source = self.store.next_background_source(now_us=self.clock(), companion=companion)
        if source is None:
            return None
        processor = self.user if source["role"] == "user" else self.assistant
        return processor.run_one(source["companion"], worker)

    def drain(self, worker, *, limit=256):
        if type(limit) is not int or not 1 <= limit <= 256:
            raise ValueError("Bounded background drain required")
        count = 0
        for _ in range(limit):
            if self.run_next(worker) is None:
                break
            count += 1
        return count
