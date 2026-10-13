"""A larger navigation page must not strand already admitted full originals."""

import json
from copy import deepcopy
from dataclasses import replace

from jsonschema import Draft202012Validator
from test_default_text_delivery import wait_for
from test_rebuilt_memory import apply, mutation
from unit_review_recording import bound_unit_packets as bound_unit_packets
from unit_review_recording import recorded_review_packet
from virtual_companion.boundary.contracts import Scope, canonical
from virtual_companion.boundary.history_readset import history_readset
from virtual_companion.cognition.short_context import ShortContextAdmission
from virtual_companion.text.application import TextSession
from virtual_companion.text.default_runtime import DefaultTextRuntime
from virtual_companion.text.fake import synthetic_unit_opinion
from virtual_companion.text.memory_runtime import DEFAULT_COMPANION_ID, TextMemoryRuntime
from virtual_companion.text.sqlite_store import SQLiteConversationStore


class NavigationModel:
    supports_stage_metadata = True

    def __init__(self):
        self.calls = []

    def cancel(self):
        pass

    def prepare_stage_request(self, messages, *, stage, schema):
        owner = self
        messages, schema = deepcopy(messages), deepcopy(schema)

        class Prepared:
            def check_context(self):
                pass

            def stream(self, cancelled):
                owner.calls.append((stage, messages, schema))
                if stage == "stream_unit_review":
                    yield canonical(synthetic_unit_opinion(json.loads(messages[-1]["content"])))
                    return
                page = next(json.loads(message["content"])["memory_navigation"]
                            for message in messages if message["content"].startswith(
                                '{"memory_navigation":'))
                if page["node_id"] == "root":
                    action = {"action": "browse_memory", "node_id": page["entries"][0]["node_id"]}
                else:
                    action = {"action": "answer", "text": "这是合成答案。"}
                Draft202012Validator(schema).validate(action)
                yield canonical(action)

        return Prepared()


def test_directory_growth_preserves_initial_originals_and_allows_finite_answer(
    tmp_path, monkeypatch,
):
    model = NavigationModel()
    memory = TextMemoryRuntime(tmp_path / "memory.db", DEFAULT_COMPANION_ID,
                               model=model, fake=True)
    mirror = SQLiteConversationStore(tmp_path / "chat", companion_id=memory.companion)
    content = "alpha " + "原" * 160 + "x" * 4380
    source = memory.store.accept(memory.config, "capacity-seed", content)
    memory.store.cancel(memory.companion, source.input_id)
    memory.store.claim_owner(memory.companion, "capacity-seed", expected_epoch=0)
    for number in range(8):
        assert apply(memory.store, replace(mutation(
            f"capacity-{number}", record=f"large-{number:02}", content=content),
            scope=Scope(memory.companion, source.session_id), source_kind="manual",
            source_id=source.input_id))["disposition"] == "applied"
    runtime = DefaultTextRuntime(memory, mirror, model, binding={"test": "page-capacity"},
                                  readonly_recall="deep-memory.v1", delivery_timeout=1)
    text = TextSession(model, mirror, {}, prepare_turn=runtime.prepare_turn,
                       before_start=runtime.before_start)
    admitted = []
    original = ShortContextAdmission.__call__

    def observe(adapter, *args, **kwargs):
        request = original(adapter, *args, **kwargs)
        data = json.loads(request.context.payload)
        if adapter is runtime.admit and "evidence_access" in data:
            admitted.append(data)
        return request

    monkeypatch.setattr(ShortContextAdmission, "__call__", observe)
    try:
        text.send("alpha")
        wait_for(lambda: runtime.poll()["event"] or not text.snapshot()["busy"])
        event = runtime.poll()["event"]
        assert event is not None, text.snapshot()
        runtime.ack({"sequence": event["sequence"], "fingerprint": event["fingerprint"],
                     "host_receipt": "synthetic-capacity-dom"})
        wait_for(lambda: not text.snapshot()["busy"])
        assert text.snapshot()["attempt"]["status"] == "completed", text.snapshot()
        initial = next(data for data in admitted if data["evidence_access"]["calls"] == 0)
        rows = initial["memories"]
        assert rows and all(row["content"] == content for row in rows)
        # Initial selection fills the actual host budget with whole rows. A more
        # conservative metadata reserve may legitimately select fewer than eight.
        capacity = initial["evidence_access"]["limits"]["evidence_bytes"]
        assert len(rows) == 8 or len(canonical([*rows, rows[0]]).encode()) > capacity
        review = next(call for call in model.calls if call[0] == "stream_unit_review")
        packet = recorded_review_packet(review[1])
        final = json.loads(packet["context"])
        assert final["memories"] == rows
        assert final["evidence_access"]["calls"] == 1
        assert final["evidence_access"]["coverage"] == "partial"
        assert len(canonical(final["evidence_access"]["navigation"]).encode()) > len(
            canonical(initial["evidence_access"]["navigation"]).encode())
        assert len(packet["context"].encode()) <= 64000
        generated = [call for call in model.calls if call[0] != "stream_unit_review"]
        assert len(generated) == 2
        assert all(row["content"] in canonical(generated[-1][1]) for row in rows)
        assert event["readset"] == history_readset(runtime.companion, runtime.claim.input_id,
                                                   runtime.claim.state_version, packet["context"])
    finally:
        runtime.stop()
        text.stop()
        if text.worker is not None:
            text.worker.join(2)
        memory.close()
        mirror.close()
