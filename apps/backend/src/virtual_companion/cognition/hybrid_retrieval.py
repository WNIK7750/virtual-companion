"""Rank fusion with explicit literal and fact-key constraints, then authority recheck.

RRF follows Cormack, Clarke and Buettcher (SIGIR 2009); this is an independent
implementation of the formula, not copied donor code. Scores select evidence,
never establish entailment or complete coverage.
"""

import json
import re
import unicodedata
from hashlib import sha256

from virtual_companion.boundary.contracts import VersionConflict, canonical
from virtual_companion.boundary.memory_relevance import lexical_relevance, validate_recall_query


def normalized_literal(value):
    return unicodedata.normalize('NFKC', value).casefold()


def literal_match(content, phrase):
    """Unicode-normalized phrase spans; ASCII identifiers require full boundaries.

    A phrase hit is not entity disambiguation or semantic negation. Explicit
    subject/predicate/scope constraints refer only to their sourced fact keys.
    """
    content, phrase = normalized_literal(content), normalized_literal(phrase)
    if re.fullmatch(r'[a-z0-9_][a-z0-9_.:/-]*', phrase):
        return re.search(r'(?<![a-z0-9_.:/-])' + re.escape(phrase)
                         + r'(?![a-z0-9_.:/-])', content) is not None
    return phrase in content


def query_constraints(exact_terms=(), conditions=None):
    if not isinstance(exact_terms, (list, tuple)) or len(exact_terms) > 6:
        raise ValueError('At most six literal constraints required')
    terms = []
    for term in exact_terms:
        if (not isinstance(term, str) or not term.strip() or len(term) > 64
                or len(term.encode('utf-8')) > 192
                or any(ord(char) < 32 or ord(char) == 127 for char in term)):
            raise ValueError('Bounded nonempty exact phrase required')
        terms.append(normalized_literal(term.strip()))
    terms = tuple(dict.fromkeys(terms))
    conditions = {} if conditions is None else conditions
    if (not isinstance(conditions, dict)
            or not set(conditions) <= {'subject', 'predicate', 'scope'}):
        raise ValueError('Only explicit fact-key constraints supported')
    selected = {}
    for key, value in conditions.items():
        if (not isinstance(value, str) or not value.strip() or len(value) > 128
                or any(ord(char) < 32 or ord(char) == 127 for char in value)):
            raise ValueError('Bounded explicit fact-key value required')
        selected[key] = value
    return terms, selected


def fuse_rankings(rankings, *, k=60):
    """Fuse qualified rows by exact record/version; deterministic tie order."""
    if type(k) is not int or not 1 <= k <= 1000:
        raise ValueError('Positive bounded RRF constant required')
    scores, originals, positions, versions = {}, {}, {}, {}
    for ranking in rankings:
        seen = set()
        for rank, row in enumerate(ranking, 1):
            key = row['record_id'], row['version']
            if key in seen:
                continue
            seen.add(key)
            if (key[0] in versions and versions[key[0]] != key[1]) or (
                key in originals and row != originals[key]
            ):
                raise VersionConflict('Retrieval channels saw different original versions')
            versions[key[0]] = key[1]
            if key not in originals:
                originals[key] = row
                positions[key] = len(positions)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return [originals[key] for key in sorted(scores, key=lambda key: (
        -scores[key], positions[key]))]


class HybridMemoryRetriever:
    """Reuse semantic and lexical ports; hard constraints cannot be outvoted."""

    query_contract = 'constrained-recall.v1'

    def __init__(self, semantic_retriever, *, k=60):
        if not callable(getattr(semantic_retriever, 'recall', None)):
            raise TypeError('Existing semantic retrieval port required')
        if type(k) is not int or not 1 <= k <= 1000:
            raise ValueError('Positive bounded RRF constant required')
        self.semantic = semantic_retriever
        self.k = k
        self.candidate_overfetch = semantic_retriever.candidate_overfetch

    @property
    def identity(self):
        return {'schema': 'hybrid-memory-retriever.v1', 'algorithm': 'rrf.v1',
                'rank_constant': self.k, 'literal_normalization': 'nfkc-casefold.v1',
                'semantic': json.loads(canonical(self.semantic.identity))}

    @property
    def fingerprint(self):
        return sha256(canonical(self.identity).encode('utf-8')).hexdigest()

    def recall(self, store, companion, *, now_us, query, limit=8, checkpoint=None,
               exact_terms=(), conditions=None):
        query = validate_recall_query(query)
        if query is None:
            raise ValueError('Nonempty recall query required')
        if type(limit) is not int or not 1 <= limit <= self.candidate_overfetch:
            raise ValueError('Context limit must fit candidate overfetch')
        terms, fields = query_constraints(exact_terms, conditions)

        def check():
            if checkpoint is not None:
                checkpoint()

        def matches(row):
            key = row.get('fact_key') or {}
            return (all(literal_match(row['content'], term) for term in terms)
                    and all(key.get(field) == value for field, value in fields.items()))

        soft_score = lexical_relevance(query)

        def score(row):
            if not matches(row):
                return 0
            value = soft_score(row)
            return max(1, value) if terms or fields else value

        check()
        lexical = store.ranked_committed_recall(companion, now_us=now_us, score=score,
                                                limit=self.candidate_overfetch)
        check()
        semantic = self.semantic.recall(store, companion, now_us=now_us, query=query,
                                        limit=self.candidate_overfetch, checkpoint=checkpoint)
        check()
        fused = fuse_rankings([lexical, [row for row in semantic if matches(row)]], k=self.k)
        selected = fused[:limit]
        if not selected:
            return []
        pairs = tuple((row['record_id'], row['version']) for row in selected)
        current = store.committed_recall(companion, now_us=now_us,
                                         candidates=pairs, limit=len(selected))
        check()
        if current != selected:
            raise VersionConflict('Fused originals changed before admission')
        return current
