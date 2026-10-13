"""Warm full-catalog scans must not evict later page hits as they encode misses."""

import pytest
from virtual_companion.cognition.semantic_retrieval import SemanticMemoryRetriever


class Catalog:
    def __init__(self, counts):
        self.rows = {owner: [{'record_id': f'{number:04}', 'version': 1,
                              'content': f'{owner} original {number}'}
                             for number in range(count)] for owner, count in counts.items()}

    def committed_catalog_page(self, owner, *, now_us, after_record_id, limit):
        rows = [row for row in self.rows[owner]
                if after_record_id is None or row['record_id'] > after_record_id]
        page = rows[:limit]
        return {'records': page, 'next_after_record_id': (
            page[-1]['record_id'] if len(rows) > limit else None)}

    def committed_recall(self, owner, *, now_us, candidates, limit):
        originals = {(row['record_id'], row['version']): row for row in self.rows[owner]}
        return [originals[pair] for pair in candidates][:limit]


class Encoder:
    def __init__(self):
        self.document_count = 0

    def query_embed(self, query):
        return iter([[1.0, 0.0]])

    def embed(self, texts):
        self.document_count += len(texts)
        return iter([[1.0, 0.0] for _ in texts])


def selected(encoder, *, page_size=8, capacity=16):
    return SemanticMemoryRetriever(encoder, model_id='cache-regression', model_revision='1',
        synthetic_model_identity=True, page_size=page_size, max_cached_vectors=capacity,
        candidate_overfetch=8)


@pytest.mark.parametrize('count,page_size', [(17, 8), (24, 8), (24, 100)])
def test_warm_over_capacity_scan_only_encodes_rows_outside_the_cache(count, page_size):
    store, encoder = Catalog({'a': count}), Encoder()
    search = selected(encoder, page_size=page_size)
    expected = search.recall(store, 'a', now_us=100, query='original', limit=4)
    for _ in range(2):
        before = encoder.document_count
        assert search.recall(store, 'a', now_us=100, query='original', limit=4) == expected
        assert encoder.document_count - before == count - 16


def test_new_small_owner_catalog_can_enter_cache_after_a_large_scan():
    store, encoder = Catalog({'a': 20, 'b': 4}), Encoder()
    search = selected(encoder)
    search.recall(store, 'a', now_us=100, query='original', limit=4)
    expected = search.recall(store, 'b', now_us=100, query='original', limit=4)
    before = encoder.document_count
    assert search.recall(store, 'b', now_us=100, query='original', limit=4) == expected
    assert encoder.document_count == before


def test_disabled_cache_preserves_scan_results_and_encodes_each_call():
    store, encoder = Catalog({'a': 24}), Encoder()
    search = selected(encoder, capacity=0)
    expected = search.recall(store, 'a', now_us=100, query='original', limit=4)
    assert encoder.document_count == 24
    assert search.recall(store, 'a', now_us=100, query='original', limit=4) == expected
    assert encoder.document_count == 48
