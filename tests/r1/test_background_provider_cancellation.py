"""A deliberate provider interruption must not spend semantic retry attempts."""

import json

import pytest
from test_rebuilt_presentation import prepared
from test_rebuilt_turns import A
from virtual_companion.boundary.sqlite_background_worker import SQLiteBackgroundWorkerStore
from virtual_companion.cognition.background_worker import AssistantExchangeWorker
from virtual_companion.providers.httpx_transport import ProviderFailure
from virtual_companion.text.memory_runtime import (
    _PriorityGenerationGate,
    _StructuredMemoryInvoker,
)


@pytest.mark.parametrize("interrupt", [True, False])
def test_provider_failure_only_requeues_when_host_cancelled(tmp_path, interrupt):
    store = SQLiteBackgroundWorkerStore(tmp_path / "memory.sqlite3")
    _, result = prepared(store)
    receipt = store.commit(result, now_us=101)
    store.publish(A, receipt["delivery_id"])
    store.acknowledge(
        A, receipt["delivery_id"], candidate_id=receipt["candidate_id"], revision=1,
        sha256=receipt["sha256"], presentation_sha256=receipt["presentation_sha256"],
        sequence=1, displayed_end=len(result.presentation.candidate.text),
    )

    class Model:
        failing = True

        def stream(self, messages, cancelled):
            if self.failing:
                if interrupt:
                    cancelled.set()  # Same host signal set by foreground/close.
                code = "provider_cancelled" if interrupt else "provider_read_timeout"
                raise ProviderFailure(code)
            yield '{"operations": []}'

    model = Model()
    worker = AssistantExchangeWorker(
        store, _StructuredMemoryInvoker(model, _PriorityGenerationGate()), clock=lambda: 102,
    )
    if interrupt:
        for _ in range(4):
            assert worker.run_one(A, "worker") is None
            with store.transaction(write=False) as db:
                row = db.execute("SELECT attempts,error FROM background_work").fetchone()
                assert row["attempts"] == 0 and row["error"] is None
        model.failing = False
        assert json.loads(worker.run_one(A, "worker"))["operations"] == 0
        assert worker.run_one(A, "worker") is None
    else:
        with pytest.raises(ProviderFailure, match="provider_read_timeout"):
            worker.run_one(A, "worker")
        assert store.background_sources(A)[-1]["status"] == "retryable"
        with store.transaction(write=False) as db:
            assert db.execute("SELECT attempts FROM background_work").fetchone()[0] == 1
