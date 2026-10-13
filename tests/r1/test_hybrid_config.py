import json

import pytest
from virtual_companion.cognition.hybrid_retrieval import HybridMemoryRetriever
from virtual_companion.cognition.semantic_retrieval import SemanticMemoryRetriever
from virtual_companion.text.semantic_config import (
    MODEL_ID,
    MODEL_MANIFEST_SHA256,
    MODEL_REVISION,
    load_semantic_retriever,
)


def config(schema='semantic-retrieval.v2'):
    result = {'schema': schema,
        'model': {'id': MODEL_ID, 'revision': MODEL_REVISION,
            'manifest_sha256': MODEL_MANIFEST_SHA256,
            'model_directory': 'synthetic-unused', 'manifest_path': 'synthetic-unused'},
        'preprocessing_revision': 'raw-content.v1', 'catalog_page_size': 100,
        'candidate_overfetch': 32, 'max_cached_vectors': 4096}
    if schema == 'semantic-retrieval.v2':
        result['fusion'] = {'algorithm': 'rrf.v1', 'rank_constant': 60}
    return result


@pytest.mark.parametrize('schema', ['semantic-retrieval.v1', 'semantic-retrieval.v2'])
def test_versioned_configuration_preserves_single_channel_and_exposes_explicit_fusion(
        tmp_path, monkeypatch, schema):
    from virtual_companion.text.semantic_embedding import FastEmbedEncoder

    class Encoder:
        identity = {'kind': 'synthetic-config-only.v1'}
        model_manifest_sha256 = MODEL_MANIFEST_SHA256

        def embed(self, texts):
            return iter([1.0, 0.0] for _ in texts)

        def query_embed(self, query):
            return iter(([1.0, 0.0],))

    monkeypatch.setattr(FastEmbedEncoder, 'from_local_manifest', lambda *a, **kw: Encoder())
    path = tmp_path / 'semantic-retrieval.json'
    path.write_text(json.dumps(config(schema)), encoding='utf-8')
    value = load_semantic_retriever(path)
    if schema == 'semantic-retrieval.v1':
        assert isinstance(value, SemanticMemoryRetriever)
    else:
        assert isinstance(value, HybridMemoryRetriever)
        assert value.identity['algorithm'] == 'rrf.v1'
        assert value.identity['rank_constant'] == 60
        assert value.fingerprint != value.semantic.fingerprint


@pytest.mark.parametrize('fusion', [None, {}, {'algorithm': 'unknown', 'rank_constant': 60},
    {'algorithm': 'rrf.v1', 'rank_constant': False},
    {'algorithm': 'rrf.v1', 'rank_constant': 0},
    {'algorithm': 'rrf.v1', 'rank_constant': 1001},
    {'algorithm': 'rrf.v1', 'rank_constant': 60, 'download': True}])
def test_invalid_fusion_rejects_before_optional_model_loading(tmp_path, monkeypatch, fusion):
    from virtual_companion.text.semantic_embedding import FastEmbedEncoder

    def forbidden(*args, **kwargs):
        raise AssertionError('Invalid selector must not load an embedding model')
    monkeypatch.setattr(FastEmbedEncoder, 'from_local_manifest', forbidden)
    value = config()
    value['fusion'] = fusion
    path = tmp_path / 'semantic-retrieval.json'
    path.write_text(json.dumps(value), encoding='utf-8')
    with pytest.raises(ValueError):
        load_semantic_retriever(path)
