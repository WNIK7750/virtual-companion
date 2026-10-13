"""Optional short-path evidence assembly; no extraction, persistence or risk bypass.

Cyrene's context grouping is the fixed mechanism reference (see notices).
Source barriers and version binding are repository-specific host contracts.
"""

import json
from dataclasses import replace

from virtual_companion.boundary.contracts import VersionConflict, canonical
from virtual_companion.boundary.evidence import evidence_payload
from virtual_companion.boundary.history_readset import (
    context_time,
    validate_context_time,
    validate_evidence_access,
)
from virtual_companion.boundary.memory_relevance import lexical_relevance, validate_recall_query

from .short_dialogue import ShortContext, ShortRequest


def readonly_context_budget(access):
    """Reserve remaining bounded query metadata inside the same 64KB payload."""
    if access['protocol'] in ('deep-memory.v1', 'dialogue-memory.v1'):
        # The current page is already in the payload. Reserve only its possible
        # growth, plus each action logged as a JSON string (a second escaping).
        remaining = access['limits']['calls'] - access['calls']
        page_bytes = len(canonical(access['navigation']).encode('utf-8'))
        return 64000 - 4100 * remaining - max(0, 8192 - page_bytes) - 64
    return 64000 - 800 * (access["limits"]["calls"] - access["calls"]) - 64


class ShortContextAdmission:
    def __init__(self, store, admit, *, evidence_clock, semantic_retriever=None, checkpoint=None):
        self.store, self.admit, self.evidence_clock = store, admit, evidence_clock
        self.semantic_retriever = semantic_retriever
        self.checkpoint = checkpoint
        self._validated = None

    def _checkpoint(self):
        if self.checkpoint is not None:
            self.checkpoint()

    def validate(self, context):
        """Recheck time-sensitive read sets after the provider's token count."""
        self._checkpoint()
        data = json.loads(context.payload)
        if (not isinstance(data.get("memories"), list)
                or len(data["memories"]) > (24 if "evidence_access" in data else 8)):
            raise ValueError("Bounded actual short context facts required")
        if "evidence_access" in data:
            validate_evidence_access(data["evidence_access"], facts=data["memories"],
                                     retrieved_dialogue=data.get("retrieved_dialogue"))
        if 'retrieved_dialogue' in data:
            self.store.validate_dialogue_fragments(context.companion, context.input_id,
                data['retrieved_dialogue'], now_us=self.evidence_clock())
        cached = self._validated
        receiving_reader = getattr(self.store, 'receiving_context_for_input', None)
        if cached is None or cached[0] is not context:
            expected_receiving = (receiving_reader(context.companion, context.input_id)
                                  if callable(receiving_reader) else None)
            if ((expected_receiving is None and 'receiving_context' in data)
                    or data.get('receiving_context') != expected_receiving):
                raise VersionConflict(
                    'Short context receiving observation differs from original input')
        processing_reader = getattr(self.store, "source_processing_for_input", None)
        if callable(processing_reader) and "source_processing" not in data:
            raise VersionConflict("Short context current source processing missing")
        if "source_processing" in data:
            # Validate this immutable context against its persisted input, not a
            # previous admission on this instance. Runtime/approval owns the lease.
            if (not callable(processing_reader)
                    or processing_reader(context.companion, context.input_id)
                    != data["source_processing"]):
                raise VersionConflict("Short context current source processing changed")
        # Freeze the cache before observing time: a parallel checkpoint may
        # finish while this one reads received provenance from the database.
        now = self.evidence_clock()
        if 'received_output' in data:
            self.store.validate_received_output_context(
                context.companion, context.input_id, data['received_output'], now_us=now)
        if "context_time" in data:
            observed = validate_context_time(data["context_time"], data)["observed_at_us"]
            if type(now) is not int or now < observed:
                raise VersionConflict("Short context evidence clock moved backwards")
        token_reader = getattr(self.store, "context_validation_token", None)
        before = None
        if callable(token_reader):
            same = cached is not None and cached[0] is context
            before = token_reader(context.companion, now_us=now, include_deadline=not same)
            if before[0] != context.state_version:
                raise VersionConflict("Short context source version changed")
            if same:
                _, original, verified_at = cached
                if now < verified_at:
                    raise VersionConflict("Short context evidence clock moved backwards")
                if before[:-1] != original[:-1]:
                    raise VersionConflict("Short context source token changed")
                finished = self.evidence_clock()
                if ("context_time" in data and type(finished) is not int) or finished < now:
                    raise VersionConflict("Short context evidence clock moved backwards")
                if original[-1] is not None and finished >= original[-1]:
                    raise VersionConflict("Short context history expired or changed")
                self._validated = (context, original, finished)
                self._checkpoint()
                return
        invalid_reader = getattr(self.store, "ineffective_history_sources", None)
        if invalid_reader is not None:
            invalid = invalid_reader(context.companion, now_us=now)
            if any(source["id"] in invalid for source in data["history"]):
                raise VersionConflict("Short context history expired or changed")
        facts, exchanges = data["memories"], data["exchanges"]
        recall = getattr(self.store, "committed_recall", self.store.recall)
        if facts and recall(
            context.companion, now_us=now,
            limit=len(facts) if "evidence_access" in data else 8,
            candidates=tuple((r["record_id"], r["version"]) for r in facts),
        ) != facts:
            raise VersionConflict("Short context memory expired or changed")
        if exchanges:
            current = self.store.pending_exchanges(
                context.companion, now_us=now, limit=8,
                candidates=tuple((r["exchange_id"], r["revision"]) for r in exchanges),
            )
            # Lookup order is not relevance order; compare complete head values.
            if sorted(current, key=lambda r: r["exchange_id"]) != sorted(
                exchanges, key=lambda r: r["exchange_id"],
            ):
                raise VersionConflict("Short context exchange expired or changed")
        if before is not None:
            after = token_reader(context.companion, now_us=now, include_deadline=False)
            finished = self.evidence_clock()
            if before[:-1] != after[:-1] or after[0] != context.state_version:
                raise VersionConflict("Short context changed during read-set validation")
            if ("context_time" in data and type(finished) is not int) or finished < now:
                raise VersionConflict("Short context evidence clock moved backwards")
            if before[-1] is not None and finished >= before[-1]:
                raise VersionConflict("Short context history expired during validation")
            # One immutable object only: not a payload-only or cross-request cache.
            # Approval/dispatch independently recheck persisted readset references.
            self._validated = (context, before, finished)
        elif "context_time" in data:
            finished = self.evidence_clock()
            if type(finished) is not int or finished < now:
                raise VersionConflict("Short context evidence clock moved backwards")
            try:
                validate_context_time(context_time(finished), data)
            except ValueError as error:
                raise VersionConflict("Short context expired during validation") from error
        self._checkpoint()

    def _select_memories(self, claim, text, *, now, pending, include_unmatched,
                         recall_query, check, constraints=None):
        committed = getattr(self.store, "ranked_committed_recall", None)
        if pending and not callable(committed):
            return []
        if self.semantic_retriever is not None:
            return self.semantic_retriever.recall(
                self.store, claim.companion_id, now_us=now,
                query=recall_query or text, limit=8, checkpoint=check, **(constraints or {}))
        if callable(ranked := committed or getattr(self.store, "ranked_recall", None)):
            return ranked(claim.companion_id, now_us=now,
                          score=lexical_relevance(text, include_unmatched=include_unmatched,
                                                  recall_query=recall_query), limit=8)
        return self.store.recall(claim.companion_id, now_us=now, limit=8)

    def recall(self, claim, snapshot, text, *, recall_query, include_unmatched=False,
               exact_terms=(), conditions=None):
        """One readonly eight-row query over the ordinary committed retrieval port."""
        recall_query = validate_recall_query(recall_query)
        if recall_query is None:
            raise ValueError("Readonly query required")
        constraints = None
        if exact_terms or conditions is not None:
            from .hybrid_retrieval import query_constraints

            query_constraints(exact_terms, conditions)
            if getattr(self.semantic_retriever, 'query_contract', None) != 'constrained-recall.v1':
                raise ValueError('Constrained retrieval port was not selected')
            constraints = {'exact_terms': exact_terms, 'conditions': conditions}

        def check():
            self._checkpoint()
            if (snapshot.companion_id != claim.companion_id
                    or snapshot.write_version != claim.state_version
                    or self.store.snapshot(claim.companion_id).write_version
                    != snapshot.write_version):
                raise VersionConflict("Readonly context source version changed")

        check()
        now = self.evidence_clock()
        memories = self._select_memories(claim, text, now=now,
            pending=self.store.memory_status(claim.companion_id) != "current",
            include_unmatched=include_unmatched, recall_query=recall_query, check=check,
            constraints=constraints)
        check()
        finished = self.evidence_clock()
        if type(finished) is not int or finished < now:
            raise VersionConflict("Readonly context evidence clock moved backwards")
        if memories and self.store.committed_recall(claim.companion_id, now_us=finished,
                limit=len(memories), candidates=tuple(
                    (row["record_id"], row["version"]) for row in memories)) != memories:
            raise VersionConflict("Readonly originals expired or changed during lookup")
        check()
        return memories

    def __call__(self, claim, snapshot, text, *, current_only=False, include_unmatched=False,
                 recall_query=None, memory_access="not_accessed", observe_time=False,
                 evidence_candidates=None, evidence_access=None, retrieved_dialogue=None):
        self._validated = None
        self._checkpoint()
        if retrieved_dialogue is not None:
            retrieved_dialogue = json.loads(canonical(retrieved_dialogue))
        if (evidence_candidates is None) != (evidence_access is None):
            raise ValueError("Readonly originals and access must be supplied together")
        recall_query = validate_recall_query(recall_query)
        request = self.admit(claim, snapshot, text)
        if not isinstance(request, ShortRequest) or request.user_text != text or request.context:
            raise ValueError("Unmodified typed admission required before evidence assembly")
        if snapshot.companion_id != claim.companion_id:
            raise ValueError("Short context companion differs from claim")

        def check():
            self._checkpoint()
            if (snapshot.write_version != claim.state_version
                    or self.store.snapshot(claim.companion_id).write_version
                    != snapshot.write_version):
                raise VersionConflict("Short context source version changed")

        check()
        now = self.evidence_clock()
        if type(observe_time) is not bool:
            raise ValueError("Explicit context time selection required")
        observation = context_time(now) if observe_time else None
        effective_reader = getattr(self.store, "effective_exchange_context", None)
        dialogue_reader = getattr(self.store, "dialogue_context", None)
        received = []
        if callable(dialogue_reader):
            sources, received = dialogue_reader(claim, now_us=now)
        else:
            sources, _ = (effective_reader(claim, now_us=now) if callable(effective_reader)
                          else self.store.exchange_context(claim))
        invalid_reader = getattr(self.store, "ineffective_history_sources", None)
        invalid = (invalid_reader(claim.companion_id, now_us=now)
                   if invalid_reader is not None else frozenset())
        if sources and sources[-1].source_id == claim.input_id:
            if sources[-1].text != text:
                raise ValueError("Short context must end at the persisted current input")
            history = sources[:-1]
        elif claim.input_id in invalid:
            # A supported management request may quote the value it revoked.
            # Keep its exact persisted text as the current request only, never
            # re-admit that source as historical fact or waive its barrier.
            current = self.store.inspect(claim.companion_id, claim.input_id)
            if current["text"] != text or current["session"] != claim.session_id:
                raise ValueError("Current request differs from persisted source")
            history = sources
        else:
            raise ValueError("Short context must end at the persisted current input")
        status_reader = getattr(self.store, "memory_status", None)
        if status_reader is not None:
            memory_status = status_reader(claim.companion_id)
        else:
            barrier = getattr(self.store, "memory_barrier", None)
            memory_status = ("pending_source" if barrier and barrier(claim.companion_id)
                             else "current")
        pending = memory_status != "current"
        history = tuple(source for source in history if source.source_id not in invalid)
        if current_only:
            if not pending:
                raise VersionConflict("Unresolved source barrier changed")
            history = ()
        committed_reader = getattr(self.store, "ranked_committed_recall", None)
        if evidence_candidates is not None:
            if not isinstance(evidence_candidates, (list, tuple)):
                raise ValueError("Complete readonly evidence array required")
            memories = json.loads(canonical(evidence_candidates))
            validate_evidence_access(evidence_access, facts=memories,
                                     retrieved_dialogue=retrieved_dialogue)
            if len({row['record_id'] for row in memories}) != len(memories):
                raise ValueError("Unique readonly evidence identities required")
        else:
            memories = self._select_memories(claim, text, now=now, pending=pending,
                include_unmatched=include_unmatched, recall_query=recall_query, check=check)
        if pending and not callable(committed_reader):
            exchanges = []
        elif callable(ranked := getattr(self.store, "ranked_pending_exchanges", None)):
            exchanges = ranked(claim.companion_id, now_us=now,
                               score=lexical_relevance(text), limit=8)
        else:
            exchanges = self.store.pending_exchanges(claim.companion_id, now_us=now, limit=8)
        operation_reader = getattr(self.store, "current_memory_operation", None)
        operation = operation_reader(claim) if callable(operation_reader) else None
        check()
        data = {
            "history": evidence_payload(history),
            "memory_status": memory_status,
            "memories": memories, "exchanges": exchanges,
        }
        if retrieved_dialogue is not None:
            if evidence_access is None or evidence_access['protocol'] != 'dialogue-memory.v1':
                raise ValueError('Dialogue source capability not selected')
            data['retrieved_dialogue'] = json.loads(canonical(retrieved_dialogue))
            self.store.validate_dialogue_fragments(claim.companion_id, claim.input_id,
                data['retrieved_dialogue'], now_us=now)
        if evidence_access is not None:
            data["evidence_access"] = json.loads(canonical(evidence_access))
        from virtual_companion.text.character import character_from_binding

        character = character_from_binding(claim.config_json)
        if character is not None:
            data["character_context"] = character
        receiving_reader = getattr(self.store, 'receiving_context_for_input', None)
        receiving = (receiving_reader(claim.companion_id, claim.input_id)
                     if callable(receiving_reader) else None)
        if receiving is not None:
            data['receiving_context'] = receiving
        if operation is not None:
            data["current_operation"] = operation
        processing_reader = getattr(self.store, "current_source_processing", None)
        if callable(processing_reader):
            data["source_processing"] = processing_reader(claim)
            data["memory_access"] = memory_access
        if observation is not None:
            data["context_time"] = validate_context_time(observation, data)
            finished = self.evidence_clock()
            if type(finished) is not int or finished < now:
                raise VersionConflict("Short context evidence clock moved backwards")
            # The original cutoff remains absolute. A slow admission cannot
            # promise evidence still effective after crossing its TTL boundary.
            try:
                validate_context_time(context_time(finished), data)
            except ValueError as error:
                raise VersionConflict("Short context expired during admission") from error
        check()
        # Share the existing 64 KB context budget. Keep newest complete units;
        # never trim an approved unit or create a second, unbounded transcript.
        selected = []
        budget = readonly_context_budget(evidence_access) if evidence_access is not None else 64000
        for item in reversed(received):
            candidate = [item, *selected]
            if len(canonical({**data, "received_output": candidate}).encode()) > budget:
                break
            selected = candidate
        if selected:
            data["received_output"] = selected
        if (self.semantic_retriever is not None or evidence_candidates is not None) and memories:
            finished = self.evidence_clock()
            if type(finished) is not int or finished < now:
                raise VersionConflict("Short context evidence clock moved backwards")
            committed_recheck = getattr(self.store, "committed_recall", None)
            candidates = tuple((item["record_id"], item["version"]) for item in memories)
            if (not callable(committed_recheck)
                    or committed_recheck(claim.companion_id, now_us=finished,
                                         candidates=candidates, limit=len(memories)) != memories):
                raise VersionConflict("Semantic memory expired or changed during admission")
        check()
        payload = canonical(data)
        if len(payload.encode("utf-8")) > budget:
            raise ValueError("Short context exceeds reserved readonly metadata capacity")
        return replace(request, context=ShortContext(
            claim.companion_id, claim.input_id, snapshot.write_version, payload))
