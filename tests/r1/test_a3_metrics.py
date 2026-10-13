import json
import threading

from virtual_companion.cognition.a3_transport import SchemaTransport
from virtual_companion.text.a3_metrics import MeasuredExactModel
from virtual_companion.text.exact_llama import ExactLlamaModel


def test_metrics_count_received_content_and_keep_partial_failure(monkeypatch):
    def stream(self, messages, cancelled):
        yield '{"text":'
        yield '"你好"}'
        raise OSError("synthetic disconnect")

    monkeypatch.setattr(ExactLlamaModel, "stream", stream)
    model = object.__new__(MeasuredExactModel)
    model.input_tokens = 13
    payloads = []

    def tokenize(route, payload):
        payloads.append(payload)
        return {"tokens": [1, 2, 3]}

    model._context_request = tokenize
    pieces = []
    try:
        pieces.extend(model.stream([], threading.Event()))
    except OSError:
        pass
    assert json.loads("".join(pieces)) == {"text": "你好"}
    assert model.metrics["input_tokens"] == 13
    assert model.metrics["actual_output_tokens"] == 3
    assert model.metrics["provider_usage"] is None
    assert model.metrics["usage_source"] == "fixed_tokenizer_received_content"
    assert (
        model.metrics["started_monotonic"]
        <= model.metrics["first_content_monotonic"]
        <= model.metrics["generation_end_monotonic"]
    )
    assert payloads[0]["content"] == "".join(pieces)


def test_metered_calls_reserve_before_send_and_preserve_failed_attempt(tmp_path):
    from virtual_companion.text.a3_budget import ExperimentLedger
    from virtual_companion.text.a3_calls import MeteredCalls

    limits = {
        "seeds": [1],
        "total_budget": {"max_calls": 1, "max_output_tokens": 10, "max_wall_seconds": 10},
        "per_seed_session_budget": {
            "max_calls": 1,
            "max_output_tokens": 10,
            "max_wall_seconds": 10,
        },
    }

    class Model:
        def check_context(self, messages):
            assert ledger.totals()["calls"] == 1

        def stream(self, messages, cancelled):
            yield '{"text": "你好"}'

        def cancel(self):
            pass

    with ExperimentLedger(tmp_path / "ledger.sqlite3", "test", "a" * 64, limits) as ledger:
        session = ledger.start_session(1)
        transport = SchemaTransport(
            {"reply": Model()},
            {"reply": 10},
            max_calls=1,
            max_output_tokens=10,
            observe=lambda _: None,
        )
        call = MeteredCalls(transport, ledger, session, "candidate", "case", tmp_path)
        assert call("reply", [], {}, threading.Event()) == {"text": "你好"}
        call.validated("reply", True)
        assert ledger.records()[0]["status"] == "completed"
        assert call.records[0]["schema_valid"]
        assert list(tmp_path.glob("*.request.json")) and list(tmp_path.glob("*.response.json"))
        resumed = MeteredCalls(transport, ledger, session, "candidate", "case", tmp_path)
        assert resumed.counts["reply"] == 1
        other = MeteredCalls(transport, ledger, session, "control", "case", tmp_path)
        assert other.counts["reply"] == 0
