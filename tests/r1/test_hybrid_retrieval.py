"""Synthetic counterexamples: candidate relevance is not answer correctness."""

import pytest
from virtual_companion.boundary.contracts import VersionConflict
from virtual_companion.cognition.hybrid_retrieval import (
    HybridMemoryRetriever,
    fuse_rankings,
    literal_match,
)


def row(name, text, **key):
    return {'record_id': name, 'version': 1, 'content': text, 'fact_key': key}


class Store:
    def __init__(self, rows):
        self.rows = rows
        self.reads = []
        self.before_read = None

    def ranked_committed_recall(self, companion, *, now_us, score, limit):
        eligible = [(score(value), index, value) for index, value in enumerate(self.rows)]
        return [value for weight, _, value in sorted(eligible, key=lambda item: (
            -item[0], item[1])) if weight > 0][:limit]

    def committed_recall(self, companion, *, now_us, candidates, limit):
        self.reads.append((companion, candidates))
        if self.before_read is not None:
            self.before_read()
        by_id = {(value['record_id'], value['version']): value for value in self.rows}
        return [by_id[key] for key in candidates if key in by_id][:limit]


class Semantic:
    candidate_overfetch = 8
    identity = {'schema': 'synthetic-semantic.v1'}

    def __init__(self, rows):
        self.rows = rows

    def recall(self, store, companion, **kwargs):
        return self.rows[:kwargs['limit']]


def test_fusion_keeps_both_complementary_channels_and_identity_deduplication():
    semantic_only = row('paraphrase', '家里的狸花叫豆包。')
    literal_only = row('literal', '备忘编号AB-12，绿色封皮。')
    shared = row('shared', 'AB-12资料。')
    assert fuse_rankings([[literal_only, shared], [semantic_only, shared]]) == [
        shared, literal_only, semantic_only]
    assert len(fuse_rankings([[shared, shared], [shared]])) == 1


@pytest.mark.parametrize(('content', 'phrase', 'expected'), [
    ('编号AB-12，绿色封皮', 'AB-12', True),
    ('编号AB-120，绿色封皮', 'AB-12', False),
    ('编号XAB-12和AB-12-3', 'AB-12', False),
    ('编号ＡＢ－１２，绿色封皮', 'ab-12', True),
    ('去明月书店，后来取消了', '明月书店', True),
    ('去明月公园旁边的书店', '明月书店', False),
])
def test_literal_phrases_and_identifier_collision(content, phrase, expected):
    assert literal_match(content, phrase) is expected


def test_hard_identifier_and_subject_cannot_be_outvoted_by_semantic_score():
    friend = row('friend', '朋友的小册子AB-12。', subject='friend')
    approximate = row('approximate', '我的小册子AB-120。', subject='user')
    exact = row('exact', '我的小册子ＡＢ－１２。', subject='user')
    store = Store([friend, approximate, exact])
    search = HybridMemoryRetriever(Semantic([friend, approximate]))
    assert search.recall(store, 'owner', now_us=1, query='那份册子的封皮',
                         exact_terms=['AB-12'], conditions={'subject': 'user'}) == [exact]
    assert store.reads[-1] == ('owner', (('exact', 1),))


def test_missing_fact_key_is_unknown_and_does_not_satisfy_positive_condition():
    unknown = row('unknown', '明月。')
    store = Store([unknown])
    assert HybridMemoryRetriever(Semantic([unknown])).recall(
        store, 'owner', now_us=1, query='明月', conditions={'subject': 'user'}) == []


def test_fusion_rechecks_exact_versions_after_both_channels():
    value = row('old', '我计划去明月书店。')
    store = Store([value])
    store.before_read = lambda: store.rows.clear()
    with pytest.raises(VersionConflict, match='changed'):
        HybridMemoryRetriever(Semantic([value])).recall(
            store, 'owner', now_us=1, query='明月')


def test_cross_channel_version_drift_fails_without_authorizing_either_version():
    one = row('same', '旧原话')
    two = {**one, 'version': 2, 'content': '更正原话'}
    with pytest.raises(VersionConflict):
        fuse_rankings([[one], [two]])


def test_checkpoint_can_cancel_before_any_source_read():
    def checkpoint():
        raise InterruptedError('cancelled')
    store = Store([])
    with pytest.raises(InterruptedError):
        HybridMemoryRetriever(Semantic([])).recall(
            store, 'owner', now_us=1, query='原话', checkpoint=checkpoint)
    assert store.reads == []


def test_unexposed_numeric_or_arbitrary_sql_conditions_are_rejected():
    with pytest.raises(ValueError, match='fact-key'):
        HybridMemoryRetriever(Semantic([])).recall(
            Store([]), 'owner', now_us=1, query='总数', conditions={'sql': 'select 1'})
