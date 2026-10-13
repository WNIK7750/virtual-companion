"""Explicit review carriers share generation and retain existing semantic limits.

The bus-ticket text comes from live-catalog-r1's retained final packet. Supplied
opinions exercise the current binder and delivery chain, not model correctness.
"""

import json
from copy import deepcopy
from hashlib import sha256
from threading import Event

import pytest
from test_default_text_delivery import setup, wait_for
from virtual_companion.boundary.contracts import VersionConflict, canonical
from virtual_companion.cognition import stream_unit_review as review
from virtual_companion.text import default_runtime
from virtual_companion.text.application import TextSession
from virtual_companion.text.default_runtime import DefaultTextRuntime

USER = "今天翻旧相册时掉出一张公交票，想起那年临时改道去海边的一次出游。"
TEXT = (
    "一张公交票把那年临时改道去海边的那趟出游带回来，这种被旧物突然拉回去的感觉，"
    "像是时间在手里折了一下：票根还在，路线却已经拐去了别处。"
    "你想聊的，是那种“本来要去某地，结果被一张小纸片带回另一段路”的错位感吗？"
)


class CarrierRecording:
    """Local stage recording with declared opinions; never contacts a provider."""

    supports_stage_metadata = True

    def __init__(self, *, fail_first=False):
        self.calls = []
        self.fail_first = fail_first

    def prepare_stage_request(self, messages, *, stage, schema):
        owner = self
        messages, schema = deepcopy(messages), deepcopy(schema)

        class Prepared:
            def check_context(self):
                pass

            def stream(self, cancelled):
                assert not cancelled.is_set()
                owner.calls.append((stage, messages, schema))
                if stage != "stream_unit_review":
                    if owner.fail_first:
                        owner.fail_first = False
                        raise RuntimeError("Retained zero-approval generation failure")
                    yield canonical({"action": "answer", "text": TEXT})
                    return
                view = json.loads(messages[-1]["content"])
                units = [
                    {
                        "segment_id": unit["segment_id"],
                        "personal_support": "supported" if index == 0 else "none",
                        "personal_source_ids": [view["user_source_id"]] if index == 0 else [],
                        "fiction_support": "none",
                        "fiction_source_ids": [],
                    }
                    for index, unit in enumerate(view["units"])
                ]
                result = {"audit_version": view["audit_version"], "decision": "approve"}
                if view["audit_version"] == review.VERDICT_UNIT_AUDIT_VERSION:
                    for unit in units:
                        unit.update(meaning="录制供应商对完整单元的意见。", use="录制当前回应。",
                                    response_fit="fits")
                    result.update(checks=["pass"] * 4, failure=None, units=units)
                else:
                    result.update(**dict.fromkeys(review.CHECKS, True), units=units)
                yield canonical(result)

        return Prepared()

    def cancel(self):
        pass


@pytest.fixture
def factory(tmp_path):
    model = CarrierRecording()
    initial, unused_session, memory, mirror, _ = setup(tmp_path, model)
    runtimes, sessions = [initial], [unused_session]

    def create(**options):
        runtime = DefaultTextRuntime(
            memory, mirror, model, binding={"test": "synthetic"}, delivery_timeout=0.8,
            **options,
        )
        runtimes.append(runtime)
        return runtime

    def session(runtime):
        text = TextSession(
            model, mirror, runtime.config.attempt_binding({"test": "synthetic"}),
            prepare_turn=runtime.prepare_turn, before_start=runtime.before_start,
        )
        sessions.append(text)
        return text

    yield create, session, initial, memory, mirror, model
    for runtime in runtimes:
        runtime.stop()
    for text in sessions:
        text.stop()
    memory.close()
    mirror.close()


def resource(runtime):
    return dict(runtime.config.resources)["model"]


def test_default_selector_and_existing_catalog_resource_are_unchanged(factory):
    create, _, initial, *_ = factory
    runtime = create()
    assert runtime.prompt_revision == "short-dialogue.v12"
    assert runtime.review_audit_version == review.AUDIT_VERSION
    assert runtime.review_source_reference_wire == review.CATALOG_SOURCE_REFERENCE_WIRE
    assert resource(runtime) == resource(initial)
    expected = [
        {"test": "synthetic"}, runtime.control, runtime.repair_reply,
        default_runtime.OPERATION_TRUTH_SYSTEM,
        default_runtime.ACCESS_INITIAL, default_runtime.ACCESS_COMPLETE,
        default_runtime.ACCESS_UNRESOLVED, default_runtime.ACCESS_CLARIFICATION,
        review.CATALOG_SOURCE_REFERENCE_WIRE,
    ]
    assert resource(runtime).sha256 == sha256(canonical(expected).encode()).hexdigest()


@pytest.mark.parametrize("audit", review.SUPPORTED_AUDIT_VERSIONS)
def test_explicit_carrier_keeps_v12_generation_and_binds_actual_resource(factory, audit):
    create, _, initial, *_ = factory
    runtime = create(review_audit_version=audit)
    assert runtime.prompt_revision == initial.prompt_revision == "short-dialogue.v12"
    assert (runtime.control, runtime.repair_reply) == (initial.control, initial.repair_reply)
    assert runtime.review_audit_version == audit
    assert runtime.config.character == initial.config.character
    if audit == review.AUDIT_VERSION:
        assert resource(runtime) == resource(initial)
        assert runtime.review_source_reference_wire == review.CATALOG_SOURCE_REFERENCE_WIRE
    else:
        assert resource(runtime) != resource(initial)
        assert runtime.review_source_reference_wire == review.LEGACY_SOURCE_REFERENCE_WIRE
    binding = runtime.config.attempt_binding({"test": "synthetic"})
    assert binding["companion_config"]["resources"]["model"] == resource(runtime).to_mapping()
    assert binding["companion_config_sha256"] == runtime.config.snapshot_hash


@pytest.mark.parametrize("audit", ["", "candidate-unit-unknown.v99", True, []])
def test_unknown_explicit_carrier_fails_before_any_model_work(factory, audit):
    create, *_, model = factory
    with pytest.raises(ValueError, match="review"):
        create(review_audit_version=audit)
    assert model.calls == []


@pytest.mark.parametrize("audit", [
    review.UNIT_SCOPE_AUDIT_VERSION, review.VERDICT_UNIT_AUDIT_VERSION,
])
def test_original_bus_ticket_candidate_reaches_existing_carrier_and_exact_delivery(factory, audit):
    create, session, _, memory, mirror, model = factory
    runtime = create(review_audit_version=audit)
    text = session(runtime)
    text.send(USER)
    event = wait_for(lambda: runtime.poll()["event"] or not text.snapshot()["busy"])
    assert isinstance(event, dict), text.snapshot()
    assert event["display_text"] == TEXT
    assert [stage for stage, _, _ in model.calls] == ["demand_control", "stream_unit_review"]
    generation = model.calls[0][1]
    assert generation[0]["content"].startswith(runtime.control)
    stage, messages, schema = model.calls[-1]
    view = json.loads(messages[-1]["content"])
    assert view["audit_version"] == audit
    assert "".join(unit["text"] for unit in view["units"]) == TEXT
    assert schema["properties"]["audit_version"] == {"const": audit}
    assert "claim" not in schema["properties"]["units"]["items"]["properties"]
    claim_config = json.loads(runtime.claim.config_json)["companion_config"]
    assert claim_config["resources"]["model"] == resource(runtime).to_mapping()
    runtime.ack({"sequence": event["sequence"], "fingerprint": event["fingerprint"],
                 "host_receipt": "recorded-carrier-visible"})
    wait_for(lambda: not text.snapshot()["busy"])
    assert text.snapshot()["attempt"]["status"] == "completed"
    assert mirror.history()["messages"][-1]["content"] == TEXT
    assert memory.store.inspect(runtime.companion, runtime.claim.input_id)["text"] == USER


def test_old_frozen_model_resource_cannot_start_under_a_different_carrier(factory):
    create, _, initial, memory, mirror, model = factory
    runtime = create(review_audit_version=review.UNIT_SCOPE_AUDIT_VERSION)
    attempt = mirror.begin(USER, initial.config.attempt_binding({"test": "synthetic"}))
    records = mirror.history()["messages"]
    with pytest.raises(VersionConflict, match="model resource"):
        runtime.prepare_turn(attempt, mirror.session_id, records, Event())
    assert runtime.claim is None and model.calls == []
    with memory.store.transaction(write=False) as db:
        assert db.execute("SELECT COUNT(*) FROM turn_inputs").fetchone()[0] == 0


def test_explicit_zero_approval_retry_records_new_carrier_without_changing_original_intake(factory):
    create, session, initial, memory, _, model = factory
    model.fail_first = True
    first = session(initial)
    old_attempt = first.send(USER)
    wait_for(lambda: not first.snapshot()["busy"])
    assert first.snapshot()["attempt"]["status"] == "failed"
    original = memory.store.inspect(initial.companion, initial.claim.input_id)
    runtime = create(review_audit_version=review.UNIT_SCOPE_AUDIT_VERSION)
    text = session(runtime)
    new_attempt = text.retry()
    event = wait_for(lambda: runtime.poll()["event"] or not text.snapshot()["busy"])
    assert isinstance(event, dict), text.snapshot()
    assert new_attempt["turn_id"] == old_attempt["turn_id"]
    assert new_attempt["attempt_id"] != old_attempt["attempt_id"]
    current = memory.store.inspect(runtime.companion, runtime.claim.input_id)
    assert current["config"] == original["config"] and current["text"] == original["text"]
    run_config = json.loads(runtime.claim.config_json)["companion_config"]
    assert run_config["resources"]["model"] == resource(runtime).to_mapping()
    runtime.ack({"sequence": event["sequence"], "fingerprint": event["fingerprint"],
                 "host_receipt": "recorded-retry-carrier-visible"})
    wait_for(lambda: not text.snapshot()["busy"])
    assert text.snapshot()["attempt"]["status"] == "completed"
