import json
import threading

import pytest
from virtual_companion.cognition.a3_transport import SchemaTransport


class Model:
    def __init__(self, response):
        self.response = response
        self.checked = []

    def check_context(self, messages):
        self.checked.append(messages)

    def stream(self, messages, cancelled):
        yield self.response

    def cancel(self):
        pass


@pytest.mark.parametrize("response", ['{"x":1,"x":2}', '{"x":NaN}', "[]", "<think>x</think>{}"])
def test_schema_transport_rejects_corrupt_objects(response):
    events = []
    call = SchemaTransport(
        {"reply": Model(response)},
        {"reply": 100},
        max_calls=1,
        max_output_tokens=100,
        observe=events.append,
    )
    with pytest.raises(ValueError):
        call("reply", [], {}, threading.Event())
    assert events[-1]["status"] == "failed"
    assert events[-1]["response_content"] == response


def test_schema_transport_preserves_json_and_enforces_total_budget():
    events = []
    model = Model('{"text":"<think>literal</think>"}')
    call = SchemaTransport(
        {"reply": model}, {"reply": 100}, max_calls=2, max_output_tokens=100, observe=events.append
    )
    assert call("reply", [], {}, threading.Event()) == {"text": "<think>literal</think>"}
    with pytest.raises(ValueError, match="budget"):
        call("reply", [], {}, threading.Event())
    assert len(model.checked) == 1


def test_cancelled_schema_call_does_not_send_and_has_terminal():
    cancelled = threading.Event()
    cancelled.set()
    events = []
    model = Model("{}")
    call = SchemaTransport(
        {"reply": model}, {"reply": 100}, max_calls=1, max_output_tokens=100, observe=events.append
    )
    assert call("reply", [], {}, cancelled) is None
    assert not model.checked
    assert events[-1]["status"] == "cancelled"


def test_stream_failure_after_cancel_keeps_error():
    class Interrupted(Model):
        def stream(self, messages, cancelled):
            cancelled.set()
            raise OSError("closed")
            yield

    events = []
    call = SchemaTransport(
        {"reply": Interrupted("")},
        {"reply": 100},
        max_calls=1,
        max_output_tokens=100,
        observe=events.append,
    )
    with pytest.raises(OSError):
        call("reply", [], {}, threading.Event())
    assert events[-1]["status"] == "cancelled"
    assert events[-1]["error_type"] == "OSError"




def test_schema_transport_allows_distinct_review_stages_concurrently():
    rendezvous = threading.Barrier(2)

    class Review(Model):
        def stream(self, messages, cancelled):
            rendezvous.wait(timeout=2)
            yield "{}"

    events, failures = [], []
    call = SchemaTransport(
        {s: Review("{}") for s in ("causal", "expression")},
        {"causal": 10, "expression": 10},
        max_calls=2,
        max_output_tokens=20,
        observe=events.append,
    )

    def invoke(stage):
        try:
            call(stage, [], {}, threading.Event())
        except Exception as error:
            failures.append(error)

    workers = [threading.Thread(target=invoke, args=(s,)) for s in ("causal", "expression")]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(3)
        assert not worker.is_alive()
    assert not failures
    assert {r["call_id"] for r in events} == {1, 2}
    assert all(r["status"] == "completed" for r in events)




def test_local_session_installs_reserves_without_starting_a_process(monkeypatch, tmp_path):
    from virtual_companion.text import a3_session

    class Exact:
        def __init__(self, profile, repository, *, system_sha256):
            self.profile = profile

    monkeypatch.setattr(a3_session, "ExactLlamaModel", Exact)
    profile = {"sampling": {"max_tokens": 192}}
    session = a3_session.LocalSchemaSession(
        profile,
        tmp_path,
        {"reply": ["a" * 64]},
        {"reply": 2048},
        max_calls=1,
        max_output_tokens=2048,
        observe=lambda _: None,
    )
    assert session.models["reply"].profile["sampling"]["max_tokens"] == 2048
    assert profile["sampling"]["max_tokens"] == 192


def test_frozen_session_validates_profile_and_source_before_start(monkeypatch, tmp_path):
    from hashlib import sha256

    from virtual_companion.text import a3_session

    class Exact:
        def __init__(self, profile, repository, *, system_sha256):
            self.profile = profile

    monkeypatch.setattr(a3_session, "ExactLlamaModel", Exact)
    profile = tmp_path / "runtime.json"
    profile.write_text(json.dumps({"sampling": {"seed": 1}}), encoding="utf-8")
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    freeze = tmp_path / "freeze.json"
    config = {
        "schema_version": 1,
        "seeds": [2],
        "runtime_profile": profile.name,
        "runtime_profile_sha256": sha256(profile.read_bytes()).hexdigest(),
        "input_pins": {source.name: sha256(source.read_bytes()).hexdigest()},
        "system_sha256": {"reply": ["a" * 64]},
        "output_reservations": {"reply": 20},
        "per_seed_session_budget": {"max_calls": 1, "max_output_tokens": 20, "max_wall_seconds": 3},
    }
    freeze.write_text(json.dumps(config), encoding="utf-8")
    session = a3_session.LocalSchemaSession.from_freeze(
        freeze, tmp_path, seed=2, observe=list().append
    )
    assert session.owner.profile["sampling"] == {"seed": 2, "max_tokens": 20}
    assert session.wall_seconds == 3
    with pytest.raises(ValueError, match="seed"):
        a3_session.LocalSchemaSession.from_freeze(freeze, tmp_path, seed=3, observe=list().append)
    source.write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="input changed"):
        a3_session.LocalSchemaSession.from_freeze(freeze, tmp_path, seed=2, observe=list().append)
    source.write_text("{}", encoding="utf-8")
    profile.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="profile changed"):
        a3_session.LocalSchemaSession.from_freeze(freeze, tmp_path, seed=2, observe=list().append)


def test_schema_wall_expiry_during_capacity_prevents_generation_and_reuse():
    records = []

    class Expiring(Model):
        def check_context(self, messages):
            call.deadline_expired = True

        def stream(self, messages, cancelled):
            raise AssertionError("No generation after deadline")

    call = SchemaTransport(
        {"reply": Expiring("{}")},
        {"reply": 20},
        max_calls=2,
        max_output_tokens=40,
        observe=records.append,
    )
    with pytest.raises(TimeoutError):
        call("reply", [], {}, threading.Event())
    assert records[0]["status"] == "deadline_exceeded"
    assert not call._active
    with pytest.raises(TimeoutError):
        call("reply", [], {}, threading.Event())
    assert call.calls == 1
