import json
import threading
from hashlib import sha256
from types import SimpleNamespace

import pytest
from virtual_companion.cognition.a3_transport import SchemaTransport
from virtual_companion.text.a3_metrics import ConstrainedExactModel


def test_schema_constraint_is_frozen_copied_and_counted_with_same_payload(tmp_path):
    (tmp_path / "prompt.txt").write_text("baseline", encoding="utf-8")
    profile = {
        "prompt_path": "prompt.txt",
        "prompt_sha256": sha256(b"baseline").hexdigest(),
        "sampling": {"max_tokens": 64},
        "server_arguments": ["--ctx-size", "4096"],
    }
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    digest = sha256(
        json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    model = ConstrainedExactModel(
        profile, tmp_path, system_sha256=frozenset({sha256(b"persona").hexdigest()})
    )
    model.schema_sha256 = frozenset({digest})
    model.process = SimpleNamespace(poll=lambda: None)
    messages = [{"role": "system", "content": "persona"}, {"role": "user", "content": "hello"}]
    with pytest.raises(ValueError, match="schema"):
        model._payload(messages)
    model.bind_schema(schema)
    schema["required"].clear()
    requests = []

    def request(route, payload):
        requests.append((route, payload))
        return {"prompt": "rendered"} if route == "/apply-template" else {"tokens": [1, 2]}

    model._context_request = request
    model.check_context(messages)
    payload = model._payload(messages)
    assert payload == requests[0][1]
    assert payload["messages"] == messages
    assert payload["response_format"]["json_schema"]["schema"]["required"] == ["answer"]
    with pytest.raises(ValueError, match="schema"):
        model.bind_schema(schema)


def test_schema_transport_binds_before_capacity_and_generation():
    events = []

    class Model:
        def bind_schema(self, schema):
            events.append(("bind", schema))

        def check_context(self, messages):
            assert events == [("bind", {"type": "object"})]
            events.append(("capacity", messages))

        def stream(self, messages, cancelled):
            assert events[-1][0] == "capacity"
            yield "{}"

    call = SchemaTransport(
        {"reply": Model()}, {"reply": 64}, max_calls=1, max_output_tokens=64, observe=lambda _: None
    )
    assert call("reply", [], {"type": "object"}, threading.Event()) == {}
