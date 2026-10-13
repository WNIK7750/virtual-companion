"""Explicit, local-only semantic resource assembly for the existing runtime."""

import json
from pathlib import Path

from virtual_companion.cognition.semantic_retrieval import SemanticMemoryRetriever

MODEL_ID = "BAAI/bge-small-zh-v1.5"
MODEL_REVISION = "46fbe35fd4374a00fee7de77dfddaeb6dd6a2c59"
MODEL_MANIFEST_SHA256 = "3d5d7411a6a3a3f41ec7f926365229d78ef0ed30f5aa65219dcdc0aef5ef2588"


def load_semantic_retriever(path):
    path = Path(path)
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    fields = {"schema", "model", "preprocessing_revision", "catalog_page_size",
              "candidate_overfetch", "max_cached_vectors"}
    if not isinstance(value, dict):
        raise ValueError("Exact semantic retrieval configuration required")
    version = value.get('schema')
    if version not in {'semantic-retrieval.v1', 'semantic-retrieval.v2'}:
        raise ValueError("Unsupported semantic retrieval configuration")
    if set(value) != fields | ({'fusion'} if version == 'semantic-retrieval.v2' else set()):
        raise ValueError("Exact semantic retrieval configuration required")
    if version == 'semantic-retrieval.v2':
        fusion = value['fusion']
        if (not isinstance(fusion, dict) or set(fusion) != {'algorithm', 'rank_constant'}
                or fusion['algorithm'] != 'rrf.v1'
                or type(fusion['rank_constant']) is not int
                or not 1 <= fusion['rank_constant'] <= 1000):
            raise ValueError('Explicit bounded RRF configuration required')
    model = value["model"]
    if not isinstance(model, dict) or set(model) != {
        "id", "revision", "manifest_sha256", "model_directory", "manifest_path",
    }:
        raise ValueError("Explicit local semantic model required")
    if (model["id"] != MODEL_ID or model["revision"] != MODEL_REVISION
            or model["manifest_sha256"] != MODEL_MANIFEST_SHA256
            or value["preprocessing_revision"] != "raw-content.v1"):
        raise ValueError("Unsupported pinned semantic resource or preprocessing")
    # Importing the concrete optional dependency is deferred until all selector
    # fields are validated. Its loader verifies local bytes before model import.
    from .semantic_embedding import FastEmbedEncoder

    encoder = FastEmbedEncoder.from_local_manifest(
        model["model_directory"], model["manifest_path"], model_id=model["id"],
        model_revision=model["revision"], manifest_sha256=model["manifest_sha256"])
    retriever = SemanticMemoryRetriever(
        encoder, model_id=model["id"], model_revision=model["revision"],
        model_manifest_sha256=model["manifest_sha256"],
        preprocessing_revision=value["preprocessing_revision"],
        page_size=value["catalog_page_size"], candidate_overfetch=value["candidate_overfetch"],
        max_cached_vectors=value["max_cached_vectors"])
    if version == 'semantic-retrieval.v2':
        from virtual_companion.cognition.hybrid_retrieval import HybridMemoryRetriever

        return HybridMemoryRetriever(retriever, k=value['fusion']['rank_constant'])
    return retriever
