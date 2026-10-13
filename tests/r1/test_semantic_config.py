import json

import pytest
from virtual_companion.text.semantic_config import load_semantic_retriever


def test_absent_semantic_config_preserves_unconfigured_path(tmp_path):
    assert load_semantic_retriever(tmp_path / "semantic-retrieval.json") is None


@pytest.mark.parametrize("value", [{}, {"schema": "semantic-retrieval.v99"},
                                 {"schema": "semantic-retrieval.v1", "download": True}])
def test_invalid_config_cannot_trigger_loading_or_network(tmp_path, value):
    path = tmp_path / "semantic-retrieval.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        load_semantic_retriever(path)
