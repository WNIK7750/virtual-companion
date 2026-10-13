"""A selected desktop provider must never process another companion's queue."""

import json
import time
from copy import deepcopy

from virtual_companion.text.memory_runtime import TextMemoryRuntime, _fake_memory_result

A = "00000000-0000-0000-0000-000000000001"
B = "00000000-0000-0000-0000-000000000002"
SESSION_A = "00000000-0000-0000-0000-000000000011"
SESSION_B = "00000000-0000-0000-0000-000000000012"


class RecordingProvider:
    supports_stage_metadata = True

    def __init__(self):
        self.sent = []

    def cancel(self):
        pass

    def prepare_stage_request(self, messages, *, stage, schema):
        frozen = deepcopy(messages)
        owner = self

        class Request:
            def check_context(self):
                pass

            def stream(self, cancelled):
                owner.sent.append((stage, frozen))
                yield json.dumps(_fake_memory_result(stage, frozen, schema, cancelled))

        return Request()


def test_real_worker_thread_skips_foreign_fifo_head_and_only_sends_its_companion(tmp_path):
    model = RecordingProvider()
    runtime = TextMemoryRuntime(tmp_path / "shared.sqlite3", A, model=model,
                                background_idle_grace=0)
    other = TextMemoryRuntime(runtime.path, B)
    foreign = other.accept_user("foreign", "FOREIGN-B-SYNTHETIC-PRIVATE", SESSION_B)
    own = runtime.accept_user("own", "OWN-A-SYNTHETIC-CURRENT", SESSION_A)
    try:
        runtime.attach_foreground(lambda: False)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if not runtime.store.background_sources(A):
                break
            time.sleep(0.01)
        runtime.close()
        assert not runtime.store.background_sources(A), own
        wires = json.dumps(model.sent)
        assert "OWN-A-SYNTHETIC-CURRENT" in wires
        assert "FOREIGN-B-SYNTHETIC-PRIVATE" not in wires
        rows = runtime.store.background_sources(B)
        assert rows[0]["source"] == foreign.input_id
        assert rows[0]["status"] == "queued"
    finally:
        runtime.close()
        other.close()
