from copy import deepcopy
from types import SimpleNamespace

import pytest
from virtual_companion.providers.local_counter import LocalWireCounter


def test_counter_templates_exact_wire_without_baseline_prompt():
    seen = []

    def request(path, body):
        seen.append((path, deepcopy(body)))
        if path == "/apply-template":
            body["messages"].clear()
            return {"prompt": "formatted exact request"}
        return {"tokens": [1, 2, 3]}

    runtime = SimpleNamespace(process=SimpleNamespace(poll=lambda: None), _context_request=request)
    body = {
        "model": "local",
        "messages": [{"role": "user", "content": "原文"}],
        "response_format": {"type": "json_schema", "json_schema": {"schema": {}}},
    }
    original = deepcopy(body)
    assert LocalWireCounter(runtime)(body) == 3
    assert body == original
    assert seen == [
        ("/apply-template", original),
        (
            "/tokenize",
            {"content": "formatted exact request", "add_special": True, "parse_special": True},
        ),
    ]


@pytest.mark.parametrize("tokens", [[], [True], [1, -2], "123", None])
def test_counter_rejects_unverified_tokenizer_results(tokens):
    runtime = SimpleNamespace(
        process=SimpleNamespace(poll=lambda: None),
        _context_request=lambda path, body: (
            {"prompt": "text"} if path == "/apply-template" else {"tokens": tokens}
        ),
    )
    with pytest.raises(ValueError):
        LocalWireCounter(runtime)({"messages": []})


def test_counter_never_contacts_a_stopped_runtime():
    runtime = SimpleNamespace(process=SimpleNamespace(poll=lambda: 1))
    with pytest.raises(RuntimeError):
        LocalWireCounter(runtime)({"messages": []})
