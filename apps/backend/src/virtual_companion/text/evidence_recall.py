"""Turn-local readonly evidence state; store eligibility stays with the host."""

import json
from dataclasses import dataclass
from threading import Event

from virtual_companion.boundary.contracts import VersionConflict, canonical
from virtual_companion.boundary.memory_relevance import validate_recall_query

VERSION = "readonly-recall.v1"
MAX_RECALL_CALLS = 2
MAX_FACTS = 24
MAX_QUERY_FACTS = 8


@dataclass(frozen=True)
class EvidenceSelection:
    memories: tuple
    evidence_access: dict
    can_recall: bool
    retrieved_dialogue: tuple = ()


class EvidenceRecallSession:
    """One owner's bounded evidence union; never calls a worker or a provider.

    lookup(query, *, cancelled) must return up to eight current, exact fact rows
    from the host's authoritative recheck. checkpoint(cancelled) owns the same
    owner, absolute deadline and cancellation as generation. The host must seed
    initial admitted originals, then recheck the entire union on every admission.
    max_bytes bounds these originals only; the host owns the full 64KB context.
    """

    def __init__(
        self,
        *,
        lookup,
        checkpoint,
        cancelled,
        initial=(),
        max_calls=MAX_RECALL_CALLS,
        max_records=MAX_FACTS,
        max_bytes=64000,
        record_cost=None,
        evidence_size=None,
    ):
        if not callable(lookup) or not callable(checkpoint) or not isinstance(cancelled, Event):
            raise ValueError("Readonly lookup and same-turn checkpoint/cancellation required")
        if type(max_calls) is not int or not 1 <= max_calls <= MAX_RECALL_CALLS:
            raise ValueError("One or two readonly calls required")
        if type(max_records) is not int or not 1 <= max_records <= MAX_FACTS:
            raise ValueError("Readonly evidence count exceeds contract")
        if type(max_bytes) is not int or not 1 <= max_bytes <= 64000:
            raise ValueError("Readonly evidence bytes exceed context contract")
        self.lookup, self.checkpoint, self.cancelled = lookup, checkpoint, cancelled
        self.max_calls, self.max_records, self.max_bytes = max_calls, max_records, max_bytes
        self._rows, self._queries, self._calls, self._stop = [], [], 0, None
        self.record_cost = record_cost or len
        self.evidence_size = evidence_size or (lambda rows: len(canonical(rows).encode('utf-8')))
        self._seeded = False
        if initial:
            self.seed(initial)

    def seed(self, initial):
        """Bind initial originals once before any query; never replace an old hop."""
        if self._seeded or self._queries:
            raise ValueError("Readonly initial evidence already frozen")
        rows = self._validated_rows(initial, maximum=MAX_QUERY_FACTS)
        if self.record_cost(rows) > MAX_QUERY_FACTS:
            raise ValueError('Readonly seed exceeds eight actual originals')
        self._merge(rows, seed=True)
        self._seeded = True

    def refresh(self, rows):
        """Merge one host memory refresh; never consumes or resets readonly calls."""
        self._checkpoint(self.cancelled)
        if not self._seeded:
            raise ValueError("Readonly initial evidence must precede memory refresh")
        rows = self._validated_rows(rows, maximum=MAX_QUERY_FACTS)
        if self.record_cost(rows) > MAX_QUERY_FACTS:
            raise ValueError('Readonly tool exceeds eight actual originals')
        _, limited = self._merge(rows)
        if limited:
            self._stop = limited
        self._checkpoint(self.cancelled)
        return self.selection

    @staticmethod
    def _validated_rows(rows, *, maximum):
        if not isinstance(rows, (list, tuple)) or len(rows) > maximum:
            raise ValueError("Bounded complete readonly fact rows required")
        # Canonical private snapshot prevents mutable lookup/caller text becoming evidence.
        result = json.loads(canonical(rows))
        for row in result:
            if (not isinstance(row, dict) or not isinstance(row.get("record_id"), str)
                    or not row["record_id"] or type(row.get("version")) is not int
                    or row["version"] < 1 or not isinstance(row.get("content"), str)
                    or not row["content"]):
                raise ValueError("Readonly fact identity, version and original required")
        return result

    def _merge(self, rows, *, seed=False):
        merged = json.loads(canonical(self._rows))
        by_id = {row["record_id"]: row for row in merged}
        additions = 0
        limited = None
        for row in rows:
            old = by_id.get(row["record_id"])
            if old is not None:
                if old != row:
                    raise VersionConflict("Readonly evidence identity or original changed")
                continue
            if self.record_cost([*merged, row]) > self.max_records:
                limited = "record_limit"
                break
            candidate = [*merged, row]
            if self.evidence_size(candidate) > self.max_bytes:
                limited = "content_limit"
                break
            merged = candidate
            by_id[row["record_id"]] = row
            additions += 1
        if seed and limited:
            raise ValueError("Initial admitted originals exceed readonly capacity")
        self._rows = merged
        return additions, limited

    @property
    def selection(self):
        stopped = self._stop or ("call_limit" if self._calls >= self.max_calls else None)
        access = {
            "protocol": VERSION, "scope": "current_committed_facts", "coverage": "partial",
            "queries": list(self._queries), "calls": self._calls,
            "stop_reason": stopped,
            "limits": {"calls": self.max_calls, "records": self.max_records,
                       "evidence_bytes": self.max_bytes},
        }
        return EvidenceSelection(tuple(json.loads(canonical(self._rows))), access,
                                 stopped is None)

    def _checkpoint(self, cancelled):
        if cancelled.is_set():
            raise InterruptedError("Readonly recall cancelled")
        self.checkpoint(cancelled)
        if cancelled.is_set():
            raise InterruptedError("Readonly recall cancelled")

    def recall(self, query, *, cancelled):
        if cancelled is not self.cancelled:
            raise ValueError("Readonly recall cancellation identity changed")
        self._checkpoint(cancelled)
        query = validate_recall_query(query)
        if query is None:
            raise ValueError("Readonly recall query required")
        if not self.selection.can_recall:
            return self.selection
        if query in self._queries:
            self._stop = "duplicate_query"
            return self.selection
        # Consume before execution: failures must not create an uncharged retry.
        self._queries.append(query)
        self._calls += 1
        rows = self.lookup(query, cancelled=cancelled)
        self._checkpoint(cancelled)
        rows = self._validated_rows(rows, maximum=MAX_QUERY_FACTS)
        added, limited = self._merge(rows)
        self._stop = limited or ("empty" if not rows else "no_new_evidence" if not added else None)
        return self.selection
