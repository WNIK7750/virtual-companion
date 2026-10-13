"""Typed navigation and exact reads share one turn's original evidence union."""

import json

from virtual_companion.boundary.contracts import canonical
from virtual_companion.boundary.memory_relevance import validate_recall_query

from .evidence_recall import EvidenceRecallSession, EvidenceSelection

VERSION = 'deep-memory.v1'
MAX_CALLS = 6
DIALOGUE_VERSION = 'dialogue-memory.v1'


class DeepMemorySession:
    def __init__(self, *, catalog, lookup, checkpoint, cancelled, dialogue=False,
                 constrained_recall=False):
        if type(constrained_recall) is not bool:
            raise ValueError('Explicit constrained recall capability required')
        self.dialogue = dialogue
        self.constrained_recall = constrained_recall
        self.catalog, self.lookup = catalog, lookup
        self.checkpoint, self.cancelled = checkpoint, cancelled
        from virtual_companion.boundary.dialogue_retrieval import originals_size, source_count

        self._union = EvidenceRecallSession(lookup=lookup, checkpoint=checkpoint,
            cancelled=cancelled, record_cost=source_count if dialogue else None,
            evidence_size=originals_size if dialogue else None)
        self._queries, self._seen, self._stop = [], set(), None
        self._navigation = catalog.browse()

    @property
    def max_bytes(self):
        return self._union.max_bytes

    @max_bytes.setter
    def max_bytes(self, value):
        self._union.max_bytes = value

    def seed(self, rows):
        self._union.seed(rows)

    def refresh(self, rows):
        self._union.refresh(rows)
        return self.selection

    def rebase(self, catalog):
        """Source processing may change the owner basis, never the spent tool budget."""
        self.catalog = catalog
        self._navigation = catalog.browse()

    @property
    def selection(self):
        union = self._union.selection
        stop = self._stop or union.evidence_access['stop_reason']
        if len(self._queries) >= MAX_CALLS:
            stop = stop or 'call_limit'
        access = {'protocol': DIALOGUE_VERSION if self.dialogue else VERSION,
                  'scope': ('current_committed_facts_and_dialogue' if self.dialogue
                            else 'current_committed_facts'), 'coverage': 'partial',
                  'queries': list(self._queries), 'calls': len(self._queries),
                  'stop_reason': stop,
                  'limits': {'calls': MAX_CALLS, 'records': 24, 'evidence_bytes': self.max_bytes},
                  'navigation': json.loads(canonical(self._navigation))}
        if self.constrained_recall:
            access['query_contract'] = 'constrained-recall.v1'
        if self.dialogue:
            facts = tuple(row for row in union.memories if row.get('kind') != 'dialogue_fragment')
            fragments = tuple(row['fragment'] for row in union.memories
                              if row.get('kind') == 'dialogue_fragment')
            return EvidenceSelection(facts, access, stop is None, fragments)
        return EvidenceSelection(union.memories, access, stop is None)

    def recall(self, request, *, cancelled):
        if cancelled is not self.cancelled:
            raise ValueError('Memory navigation cancellation identity changed')
        self._checkpoint(cancelled)
        if not self.selection.can_recall:
            return self.selection
        if isinstance(request, str):
            query = validate_recall_query(request)
            if query is None:
                raise ValueError('Memory lookup query required')
            action = {'action': 'recall', 'query': query}
        else:
            action = json.loads(canonical(request))
        from .demand_memory import _action

        action = _action(canonical(action), final=True, allow_recall=True,
            allow_navigation=True, allow_constrained_recall=self.constrained_recall)
        if action['action'] not in {'recall', 'browse_memory', 'read_memory'}:
            raise ValueError('Known readonly memory action required')
        encoded = canonical(action)
        if len(encoded.encode('utf-8')) > 2048:
            raise ValueError('Bounded memory action required')
        key = (self.catalog.identity, encoded)
        if key in self._seen:
            self._stop = 'duplicate_query'
            return self.selection
        self._seen.add(key)
        self._queries.append(encoded)
        if action['action'] == 'browse_memory':
            self._navigation = self.catalog.browse(action['node_id'], cursor=action.get('cursor'))
        elif action['action'] == 'read_memory':
            rows = self.catalog.read(tuple((row['record_id'], row['version'])
                                          for row in action['references']))
            self._union.refresh(rows)
        elif action['action'] == 'recall':
            query = action['query'] if set(action) == {'action', 'query'} else action
            rows = self.lookup(query, cancelled=cancelled)
            self._union.refresh(rows)
        else:
            raise ValueError('Known readonly memory action required')
        # Empty navigation/lookup is information, not a reason to ban other read tools.
        self._checkpoint(cancelled)
        return self.selection

    def _checkpoint(self, cancelled):
        if cancelled.is_set():
            raise InterruptedError('Memory navigation cancelled')
        self.checkpoint(cancelled)
        if cancelled.is_set():
            raise InterruptedError('Memory navigation cancelled')
