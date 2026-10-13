"""Use the existing hybrid retriever over current, source-bound dialogue originals."""

import json
from bisect import bisect_right

from virtual_companion.boundary.contracts import (
    VersionConflict,
    canonical,
    companion_id,
    identifier,
)
from virtual_companion.boundary.dialogue_retrieval import fragment_row, validate_fragments
from virtual_companion.boundary.memory_relevance import validate_recall_query

from .hybrid_retrieval import HybridMemoryRetriever, query_constraints


class _DialogueCandidates:
    """One query's typed view; candidate reads do not spend tool original capacity."""

    def __init__(self, store, companion, input_id, now_us, checkpoint):
        companion_id(companion)
        identifier(input_id)
        if not callable(getattr(store, "validate_dialogue_fragments", None)):
            raise TypeError("Authoritative dialogue validation port required")
        self.store, self.companion, self.input_id = store, companion, input_id
        self.now_us, self.checkpoint = now_us, checkpoint
        self._check()
        fragments = json.loads(canonical(store.dialogue_fragments(
            companion, input_id, now_us=now_us)))
        self._check()
        self.rows = []
        for fragment in fragments:
            self._check()
            validate_fragments([fragment])
            self.rows.append(fragment_row(fragment))
        self.rows.sort(key=lambda row: row["record_id"])
        self.ids = [row["record_id"] for row in self.rows]
        self.by_ref = {(row["record_id"], row["version"]): row for row in self.rows}
        if len(self.by_ref) != len(self.rows):
            raise VersionConflict("Duplicate dialogue candidate identity")

    def _check(self):
        if self.checkpoint is not None:
            self.checkpoint()

    def _scope(self, companion, now_us):
        self._check()
        if companion != self.companion or now_us != self.now_us:
            raise VersionConflict("Dialogue candidate scope differs from its frozen query")

    def committed_catalog_page(self, companion, *, now_us, after_record_id=None, limit=8):
        self._scope(companion, now_us)
        if after_record_id is not None and after_record_id not in self.ids:
            raise ValueError("Unknown dialogue candidate cursor")
        offset = 0 if after_record_id is None else bisect_right(self.ids, after_record_id)
        rows = self.rows[offset:offset + limit]
        cursor = rows[-1]["record_id"] if rows and offset + len(rows) < len(self.rows) else None
        self._check()
        return {"records": tuple(rows), "next_after_record_id": cursor}

    def ranked_committed_recall(self, companion, *, now_us, score, limit):
        self._scope(companion, now_us)
        ranked = []
        for row in self.rows:
            self._check()
            relevance = score(row)
            if relevance > 0:
                ranked.append((relevance, row))
        ranked.sort(key=lambda pair: (-pair[0], pair[1]["record_id"]))
        self._check()
        return [row for _, row in ranked[:limit]]

    def committed_recall(self, companion, *, now_us, candidates, limit):
        self._scope(companion, now_us)
        if len(candidates) > limit or len(set(candidates)) != len(candidates):
            raise ValueError("Bounded distinct dialogue candidate references required")
        try:
            rows = [self.by_ref[pair] for pair in candidates]
        except KeyError as error:
            raise VersionConflict("Dialogue candidate is outside its frozen query") from error
        # This existing port re-reads every frozen member, its role, text hash,
        # visibility and invalidation. Late-visible additions cannot expand an
        # old confirmed subset. Its 24-fragment payload bound is not a tool limit.
        for offset in range(0, len(rows), 24):
            self._check()
            self.store.validate_dialogue_fragments(
                companion, self.input_id,
                [row["fragment"] for row in rows[offset:offset + 24]], now_us=now_us)
            self._check()
        return json.loads(canonical(rows))


class DialogueMemoryRetriever:
    """Rank dialogue candidates with the same hybrid instance, encoder and cache.

    The host combines returned candidates with facts and charges actual original
    members against its existing per-tool and turn limits. This leaf never
    admits evidence, creates another model, persists an index or grants access.
    """

    def __init__(self, retriever):
        if not isinstance(retriever, HybridMemoryRetriever):
            raise TypeError("Explicit existing hybrid retrieval instance required")
        self.retriever = retriever
        self.candidate_overfetch = retriever.candidate_overfetch

    @property
    def identity(self):
        return {"schema": "dialogue-hybrid-retriever.v1",
                "source_domain": "qualified_dialogue_fragments",
                "original_policy": "frozen_confirmed_subset",
                "hybrid": json.loads(canonical(self.retriever.identity))}

    def recall(self, store, companion, *, input_id, now_us, query, limit=None,
               exact_terms=(), conditions=None, checkpoint=None):
        query = validate_recall_query(query)
        if query is None:
            raise ValueError("Nonempty dialogue recall query required")
        if checkpoint is not None:
            checkpoint()
        terms, fields = query_constraints(exact_terms, conditions)
        if fields:
            # Positive fact keys cannot be inferred from a dialogue member's role.
            return []
        candidates = _DialogueCandidates(store, companion, input_id, now_us, checkpoint)
        return self.retriever.recall(
            candidates, companion, now_us=now_us, query=query,
            limit=self.candidate_overfetch if limit is None else limit,
            exact_terms=terms, checkpoint=checkpoint)
