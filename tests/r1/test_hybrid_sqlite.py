"""Authority integration, using actual SQLite and semantic catalog/cache paths."""

from datetime import UTC, datetime, timedelta

from virtual_companion.boundary.contracts import Scope
from virtual_companion.boundary.memory import FactKey, MemoryMutation
from virtual_companion.cognition.hybrid_retrieval import HybridMemoryRetriever
from virtual_companion.cognition.semantic_retrieval import SemanticMemoryRetriever
from virtual_companion.text.chronology import utc_microseconds
from virtual_companion.text.memory_runtime import TextMemoryRuntime

COMPANION = '00000000-0000-0000-0000-000000000001'
WHEN = datetime(2026, 10, 11, tzinfo=UTC)


class Encoder:
    def query_embed(self, query):
        return iter(([1.0, 0.0],))

    def embed(self, texts):
        return iter([0.0, 1.0] if 'AB-12，' in text else [1.0, 0.0] for text in texts)


def setup_memory(tmp_path):
    memory = TextMemoryRuntime(tmp_path / 'authority.sqlite3', COMPANION)
    source = memory.store.accept(memory.config, 'source', '合成来源')
    memory.store.cancel(COMPANION, source.input_id)
    memory.store.claim_owner(COMPANION, 'synthetic-owner', expected_epoch=0)
    return memory, source


def mutate(memory, source, name, content, *, kind='assert_fact', version=None, expiry=None):
    item = MemoryMutation(
        Scope(COMPANION, source.session_id), f'{name}-{kind}', kind, name,
        content, 'manual', source.input_id, 'synthetic authority integration', WHEN,
        FactKey('user', name, 'profile') if kind == 'assert_fact' else None,
        expected_record_version=version, expires_at=expiry,
    )
    outcome = memory.store.mutate(item, memory.store.owner(COMPANION),
                                  expected_version=memory.store.snapshot(COMPANION).write_version)
    assert outcome['disposition'] == 'applied'


def retriever():
    return HybridMemoryRetriever(SemanticMemoryRetriever(
        Encoder(), model_id='synthetic', model_revision='1',
        candidate_overfetch=8, synthetic_model_identity=True))


def test_exact_candidate_outside_semantic_top_eight_is_recovered_from_real_authority(tmp_path):
    memory, source = setup_memory(tmp_path)
    try:
        for number in range(10):
            mutate(memory, source, f'a-{number:02}', f'编号AB-120的背景第{number}项。')
        mutate(memory, source, 'z-exact', '编号AB-12，封面绿色，仍是计划。')
        search = retriever()
        semantic = search.semantic.recall(memory.store, COMPANION,
            now_us=utc_microseconds(WHEN), query='那份小册子', limit=8)
        assert all(row['record_id'] != 'z-exact' for row in semantic)
        found = search.recall(memory.store, COMPANION, now_us=utc_microseconds(WHEN),
            query='那份小册子', exact_terms=['AB-12'], conditions={'subject': 'user'})
        assert [(row['record_id'], row['version']) for row in found] == [('z-exact', 1)]
        assert found[0]['content'] == '编号AB-12，封面绿色，仍是计划。'
    finally:
        memory.close()


def test_real_reopen_revision_revocation_and_expiry_do_not_resurrect_cached_originals(tmp_path):
    memory, source = setup_memory(tmp_path)
    search = retriever()
    try:
        mutate(memory, source, 'book', '编号AB-12，封面绿色。')
        mutate(memory, source, 'expired', '编号AB-12，临时背景。',
               expiry=WHEN + timedelta(seconds=1))
        initial = search.recall(memory.store, COMPANION, now_us=utc_microseconds(WHEN),
                                query='编号', exact_terms=['AB-12'])
        assert {row['record_id'] for row in initial} == {'book', 'expired'}
        mutate(memory, source, 'book', '编号AB-12，封面蓝色。', kind='correct', version=1)
        memory.close()
        memory = TextMemoryRuntime(tmp_path / 'authority.sqlite3', COMPANION)
        rows = search.recall(memory.store, COMPANION,
            now_us=utc_microseconds(WHEN + timedelta(seconds=2)),
            query='编号', exact_terms=['AB-12'])
        assert [(row['record_id'], row['version'], row['content']) for row in rows] == [
            ('book', 2, '编号AB-12，封面蓝色。')]
        assert search.recall(memory.store, '00000000-0000-0000-0000-000000000002',
            now_us=utc_microseconds(WHEN), query='编号', exact_terms=['AB-12']) == []
        mutate(memory, source, 'book', '', kind='revoke', version=2)
        assert search.recall(memory.store, COMPANION,
            now_us=utc_microseconds(WHEN + timedelta(seconds=2)),
            query='编号', exact_terms=['AB-12']) == []
    finally:
        memory.close()
