"""Catalog identity transfer contracts; supplied opinions do not prove semantics."""

import json
from copy import deepcopy
from hashlib import sha256
from threading import Event

import pytest
from jsonschema import Draft202012Validator
from test_default_text_delivery import Model, setup, wait_for
from test_stream_unit_review import personal_ref, protocol_packet
from unit_review_recording import bound_unit_packets as bound_unit_packets
from unit_review_recording import recorded_review_packet
from virtual_companion.boundary.contracts import canonical
from virtual_companion.boundary.presentation import digest
from virtual_companion.cognition import stream_unit_review as review
from virtual_companion.text import default_runtime

WIRE = "catalog-quote.v1"
# live-default-r1/model-events.jsonl SHA:
# 315539a23ca942de87e4284726a757eda4dfce81e18221be597c591f1a2c0bc8.
# The failed final review 25e30b8c-0055-4509-923a-000e83db893a kept the id
# but copied it into source_id on the second ref. Preserve its source/claim text.
USER = (
    "今天翻旧相册时掉出一张公交票，想起那年临时改道去海边的一次出游。"
    "现在想聊聊这种突然被旧物拉回去的感觉，先别给收纳或整理建议。"
    "我暂时离开屏幕，你照常回复就好。"
)
CLAIMS = ("一张公交票把那年临时改道去海边的那天又拽了回来", "这种被旧物突然拉回去的感觉")
QUOTES = ("今天翻旧相册时掉出一张公交票，想起那年临时改道去海边的一次出游。",
          "现在想聊聊这种突然被旧物拉回去的感觉")
TEXT = (
    "一张公交票把那年临时改道去海边的那天又拽了回来，这种被旧物突然拉回去的感觉，"
    "像是时间没有按顺序走，而是被某个小东西勾了一下，整个人就站回了当时的站台。"
    "你想聊的是这种“被拉回去”本身，还是那张票带回来的海边那段？"
)


def packet():
    value = protocol_packet(USER, TEXT)
    source = value["payload"]["personal_sources"][0]
    source.update(id="current:006814a5-fe4c-4daa-a98e-bbcdcc78ce57",
                  source_id="006814a5-fe4c-4daa-a98e-bbcdcc78ce57")
    value["sha256"] = digest(value["payload"])
    return value


def opinion(payload):
    source = payload["personal_sources"][0]
    segments = [{"segment_id": item["segment_id"], "commitments": [], "questions": []}
                for item in payload["review_segments"]]
    segments[0]["commitments"] = [
        {"kind": "assertion", "claim": claim, "support": "supported",
         "sources": [personal_ref(source, quote)]}
        for claim, quote in zip(CLAIMS, QUOTES, strict=True)
    ]
    segments[1]["questions"] = [payload["review_segments"][1]["text"]]
    return {"audit_version": review.AUDIT_VERSION, "decision": "approve", "segments": segments,
            **dict.fromkeys(review.CHECKS, True)}


def compact(value):
    result = deepcopy(value)
    for segment in result["segments"]:
        for commitment in segment["commitments"]:
            commitment["sources"] = [{key: ref[key] for key in ("id", "quote")}
                                     for ref in commitment["sources"]]
    return result


def refs(value):
    return [ref for segment in value["segments"] for commitment in segment["commitments"]
            for ref in commitment["sources"]]


def assess(value, raw, *, wire=WIRE):
    return review.StructuredUnitAssessor(
        lambda *args: deepcopy(raw), Event(), source_reference_wire=wire)(value)


def test_catalog_compact_removes_actual_identity_copy_duty_without_normalized_verdict_change():
    envelope = packet()
    original = opinion(envelope["payload"])
    drifted = deepcopy(original)
    refs(drifted)[1]["source_id"] = refs(drifted)[1]["id"]
    with pytest.raises(ValueError, match="reference differs"):
        review.StructuredUnitAssessor(lambda *args: drifted, Event())(envelope)
    # A full legacy ref with a bad value must never be repaired by catalog mode.
    with pytest.raises(ValueError, match="reference differs"):
        assess(envelope, drifted)
    expected = review.StructuredUnitAssessor(lambda *args: original, Event())(envelope)
    actual = assess(envelope, compact(drifted))
    assert actual == expected
    assert actual["audit_version"] == review.AUDIT_VERSION
    assert actual["request_sha256"] == envelope["sha256"]
    review.BoundedUnitReviewer._validate_opinion(actual, envelope["sha256"], envelope["payload"])


def test_catalog_changes_only_reference_schema_and_identity_copy_sentence():
    envelope = packet()
    sent = []

    def invoke(stage, messages, schema, cancelled):
        sent.append((stage, deepcopy(messages), deepcopy(schema)))
        result = opinion(envelope["payload"])
        return result if len(sent) == 1 else compact(result)

    review.StructuredUnitAssessor(invoke, Event())(envelope)
    review.StructuredUnitAssessor(invoke, Event(), source_reference_wire=WIRE)(envelope)
    legacy, catalog = sent
    assert legacy[0] == catalog[0] == "stream_unit_review"
    assert legacy[1][1:] == catalog[1][1:]
    old = "来源的id、owner、source_id、role、revision原样对应；"
    new = ("sources每个引用只输出{id,quote}，id必须取本封包来源目录；"
           "owner、source_id、role、revision由宿主补齐；")
    assert legacy[1][0]["content"].count(old) == 1
    assert catalog[1][0]["content"] == legacy[1][0]["content"].replace(old, new)
    catalog_schema = deepcopy(catalog[2])
    sources = catalog_schema["properties"]["segments"]["items"]["properties"]["commitments"][
        "items"]["properties"]["sources"]
    assert set(sources["items"]["properties"]) == {"id", "quote"}
    assert sources["items"]["required"] == ["id", "quote"]
    assert sources["items"]["properties"]["id"]["enum"] == [
        envelope["payload"]["personal_sources"][0]["id"]]
    Draft202012Validator.check_schema(catalog_schema)
    legacy_sources = legacy[2]["properties"]["segments"]["items"]["properties"]["commitments"][
        "items"]["properties"]["sources"]
    sources["items"] = deepcopy(legacy_sources["items"])
    assert catalog_schema == legacy[2] == review._model_schema()


@pytest.mark.parametrize("change", [
    lambda raw: refs(raw)[0].update(id="current:unknown"),
    lambda raw: refs(raw)[0].update(extra="no"),
    lambda raw: refs(raw)[0].update(owner="partial-identity"),
    lambda raw: refs(raw)[0].update(quote="来源未说过的内容"),
    lambda raw: refs(raw)[0].update(quote=" "),
    lambda raw: raw["segments"][0]["commitments"][0].update(claim=QUOTES[0]),
    lambda raw: raw["segments"][0]["commitments"][0].update(sources=[]),
])
def test_catalog_keeps_unknown_reference_quote_and_source_text_as_claim_failures(change):
    envelope = packet()
    raw = compact(opinion(envelope["payload"]))
    change(raw)
    with pytest.raises(ValueError):
        assess(envelope, raw)


@pytest.mark.parametrize("field,value", [
    ("id", "current:unknown"), ("owner", "another-owner"),
    ("source_id", "current:006814a5-fe4c-4daa-a98e-bbcdcc78ce57"),
    ("role", "assistant"), ("revision", 2), ("quote", "不是来源原话"),
])
@pytest.mark.parametrize("wire", ["legacy", WIRE])
def test_full_legacy_identity_is_checked_without_overwriting_any_field(field, value, wire):
    envelope = packet()
    raw = opinion(envelope["payload"])
    refs(raw)[0][field] = value
    with pytest.raises(ValueError):
        assess(envelope, raw, wire=wire)


def test_catalog_rejects_cross_commitment_mixed_compact_and_full_refs():
    envelope = packet()
    raw = opinion(envelope["payload"])
    raw["segments"][0]["commitments"][0]["sources"] = [compact(raw)["segments"][0][
        "commitments"][0]["sources"][0]]
    with pytest.raises(ValueError):
        assess(envelope, raw)


@pytest.mark.parametrize("change", [
    lambda sources: sources.append(deepcopy(sources[0])),
    lambda sources: sources[0].update(owner="another-owner"),
    lambda sources: sources[0].update(role="assistant"),
    lambda sources: sources[0].update(revision=True),
    lambda sources: sources[0].update(revision=0),
])
def test_catalog_rejects_invalid_frozen_directory_before_call(change):
    envelope = packet()
    change(envelope["payload"]["personal_sources"])
    envelope["sha256"] = digest(envelope["payload"])
    calls = []
    assessor = review.StructuredUnitAssessor(lambda *args: calls.append(args), Event(),
                                             source_reference_wire=WIRE)
    with pytest.raises(ValueError, match="source identity"):
        assessor(envelope)
    assert calls == []


def test_catalog_uses_call_local_frozen_sources_and_rejects_previous_call_ids():
    envelope = packet()
    original = deepcopy(envelope)
    raw = compact(opinion(envelope["payload"]))
    expected = review.StructuredUnitAssessor(
        lambda *args: opinion(original["payload"]), Event())(original)
    calls = []

    def invoke(stage, messages, schema, cancelled):
        calls.append(messages)
        envelope["payload"]["personal_sources"][0].update(
            owner="mutated-owner", source_id="mutated-source", text="mutated text")
        schema["properties"].clear()
        return deepcopy(raw)

    assessor = review.StructuredUnitAssessor(invoke, Event(), source_reference_wire=WIRE)
    assert assessor(envelope) == expected
    other = deepcopy(original)
    other["payload"]["personal_sources"][0].update(id="current:next", source_id="next")
    other["sha256"] = digest(other["payload"])
    with pytest.raises(ValueError):
        assessor(other)
    assert len(calls) == 2


def test_catalog_empty_directory_has_valid_schema_and_allows_no_references():
    envelope = packet()
    envelope["payload"]["personal_sources"] = []
    envelope["sha256"] = digest(envelope["payload"])
    raw = opinion(packet()["payload"])
    for segment in raw["segments"]:
        segment["commitments"] = []

    def invoke(stage, messages, schema, cancelled):
        Draft202012Validator.check_schema(schema)
        sources = schema["properties"]["segments"]["items"]["properties"]["commitments"][
            "items"]["properties"]["sources"]
        assert sources["maxItems"] == 0
        return raw

    assert review.StructuredUnitAssessor(invoke, Event(), source_reference_wire=WIRE)(
        envelope)["decision"] == "approve"


@pytest.mark.parametrize("wire", [None, "", "catalog-quote.v2"])
def test_unknown_reference_wire_is_rejected(wire):
    with pytest.raises(ValueError):
        review.StructuredUnitAssessor(lambda *args: None, Event(), source_reference_wire=wire)


def test_catalog_cannot_silently_apply_to_whole_unit_audits():
    with pytest.raises(ValueError):
        review.StructuredUnitAssessor(lambda *args: None, Event(),
            audit_version=review.UNIT_SCOPE_AUDIT_VERSION, source_reference_wire=WIRE)


def test_default_v3_runtime_selects_catalog_on_actual_review_and_isolates_model_resource(tmp_path):
    class CatalogRecording(Model):
        def stream(self, messages, cancelled):
            self.calls.append(deepcopy(messages))
            if "你只审查候选文字" in messages[0]["content"]:
                result = compact(opinion(recorded_review_packet(messages)))
                yield json.dumps(result, ensure_ascii=False)
            else:
                yield json.dumps({"action": "answer", "text": TEXT}, ensure_ascii=False)

    runtime, session, memory, mirror, model = setup(tmp_path, CatalogRecording())
    try:
        session.send(USER)
        event = wait_for(lambda: runtime.poll()["event"] or not session.snapshot()["busy"])
        assert isinstance(event, dict), session.snapshot()
        assert runtime.review_audit_version == review.AUDIT_VERSION
        assert runtime.review_source_reference_wire == WIRE
        assert event["text"] == TEXT and len(model.calls) == 2
        base = [{"test": "synthetic"}, runtime.control, runtime.repair_reply,
                default_runtime.OPERATION_TRUTH_SYSTEM,
                default_runtime.ACCESS_INITIAL, default_runtime.ACCESS_COMPLETE,
                default_runtime.ACCESS_UNRESOLVED, default_runtime.ACCESS_CLARIFICATION]
        model_reference = dict(runtime.config.resources)["model"]
        assert model_reference.sha256 == sha256(canonical([*base, WIRE]).encode()).hexdigest()
        assert model_reference.sha256 != sha256(canonical(base).encode()).hexdigest()
        runtime.ack({"sequence": event["sequence"], "fingerprint": event["fingerprint"],
                     "host_receipt": "synthetic-catalog-visible"})
        wait_for(lambda: not session.snapshot()["busy"])
        assert session.snapshot()["attempt"]["status"] == "completed"
    finally:
        runtime.stop()
        session.stop()
        memory.close()
        mirror.close()


@pytest.mark.parametrize("revision", ["short-dialogue.v26", "short-dialogue.v28",
                                      "short-dialogue.v29", "short-dialogue.v30"])
def test_non_v3_runtime_keeps_reference_wire_and_previous_model_fingerprint(tmp_path, revision):
    runtime, session, memory, mirror, model = setup(tmp_path)
    try:
        candidate = default_runtime.DefaultTextRuntime(memory, mirror, model,
            binding={"test": "synthetic"}, prompt_revision=revision)
        assert candidate.review_source_reference_wire == "legacy"
        base = [{"test": "synthetic"}, candidate.control, candidate.repair_reply,
                default_runtime.OPERATION_TRUTH_SYSTEM,
                default_runtime.ACCESS_INITIAL, default_runtime.ACCESS_COMPLETE,
                default_runtime.ACCESS_UNRESOLVED, default_runtime.ACCESS_CLARIFICATION,
                candidate.review_audit_version]
        assert dict(candidate.config.resources)["model"].sha256 == sha256(
            canonical(base).encode()).hexdigest()
    finally:
        runtime.stop()
        session.stop()
        memory.close()
        mirror.close()
