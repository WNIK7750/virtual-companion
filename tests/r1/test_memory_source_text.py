"""CPU-only source selection and guarded fact extraction; no worker or database."""

import json
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
from threading import Event

import pytest
from virtual_companion.boundary.contracts import Scope, canonical
from virtual_companion.boundary.evidence import EvidenceTurn
from virtual_companion.cognition.background_worker import _fact_source_invoke
from virtual_companion.cognition.exchange_model import (
    ExchangeModel,
    PreparedSourceIntents,
    _coverage_segments,
)
from virtual_companion.cognition.fact_model import FactModel
from virtual_companion.text.memory_runtime import _fake_memory_result, _source_text

SCOPE = Scope("11111111-1111-4111-8111-111111111111", "source-selection")
SOURCE_ID = "3164ec3a-b686-4149-b681-0c0597abdc98"
TEXT = "我暂时喜欢安静的音乐，此偏好有效至2026-10-06T12:47:00.931Z。"
SOURCE = {"id": SOURCE_ID, "role": "user", "text": TEXT}


def messages(payload):
    return [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def source_request(text=TEXT, *, role="user"):
    return {"sources": [{"id": SOURCE_ID, "role": role, "text": text}],
            "eligible_source_ids": [SOURCE_ID], "segments": _coverage_segments(text)}


def test_exact_r3_source_reader_returns_full_source_before_scaffold():
    request = source_request()
    assert len(request["segments"]) == 2
    assert request["segments"][1]["text"] == "此偏好有效至2026-10-06T12:47:00.931Z。"
    assert _source_text(messages(request)) == TEXT


def test_exact_r3_fake_intent_to_guarded_fact_compile_preserves_source_and_expiry():
    source = EvidenceTurn(SCOPE, SOURCE_ID, "user", TEXT, len(TEXT))
    calls = []

    def recording(stage, request, schema, cancelled):
        calls.append(stage)
        payload = json.loads(request[-1]["content"])
        if stage == "exchange_intent":
            return _fake_memory_result(stage, request, schema, cancelled)
        if stage == "fact_extract":
            assert payload["source"] == SOURCE
            assert payload["fact_evidence_ranges"] == [
                {"start": 0, "end": len(TEXT), "text": TEXT}]
            return {"operations": [{"operation": "add", "type": "short_term",
                    "evidenceTurnId": SOURCE_ID, "evidenceQuote": TEXT,
                    "expiresAt": 1791290820931, "supersedesAtomId": None,
                    "factKey": {"subject": "user", "predicate": "音乐偏好"}}]}
        assert stage == "fact_support"
        assert payload["source"] == SOURCE
        return {"intent": "assertion", "verdict": "supported"}

    prepared = ExchangeModel(recording, Event()).prepare_source_intents(
        turns=(source,), targets={}, eligible_source_ids=(SOURCE_ID,))
    # Preserve the original failure site: the old helper reaches the unchanged
    # guard with a last-clause quote incorrectly bound to segmentId=0.
    scoped = _fact_source_invoke(recording, source, prepared)
    assert prepared.complete
    (command,) = FactModel(scoped, Event(), bind_content=True).extract(
        scope=SCOPE, batch_id="batch", now_us=100, turn=source, targets={})
    assert command.source_id == SOURCE_ID and command.content == TEXT
    assert command.expires_at == datetime(2026, 10, 6, 12, 47, 0, 931000, tzinfo=UTC)
    support = json.loads(command.reason)["support"]
    assert support["source_sha256"] == sha256(TEXT.encode()).hexdigest()
    assert calls == ["exchange_intent", "fact_extract", "fact_support"]


@pytest.mark.parametrize("payload", [
    {"source": SOURCE},
    {"source": SOURCE, "sources": [{"id": "foreign", "role": "assistant", "text": "别人的话"}]},
    {"source": SOURCE, "eligible_source_ids": [SOURCE_ID], "segments": [{"id": 0, "text": "尾段"}]},
    {"sources": [SOURCE]},
    source_request(),
    {**source_request(), "sources": [
        SOURCE, {"id": "foreign", "role": "assistant", "text": "最后助手"}]},
    {**source_request(), "sources": [
        {"id": "foreign", "role": "user", "text": "较早用户"}, SOURCE]},
])
def test_explicit_current_source_is_selected_without_last_turn_or_scaffold_fallback(payload):
    assert _source_text(messages(payload)) == TEXT


@pytest.mark.parametrize("payload", [
    {"sources": [SOURCE, {"id": "foreign", "role": "user", "text": "另一个用户"}]},
    {"sources": [SOURCE, SOURCE], "eligible_source_ids": [SOURCE_ID]},
    {"sources": [SOURCE], "eligible_source_ids": ["foreign"]},
    {"sources": [SOURCE], "eligible_source_ids": []},
    {"sources": [SOURCE], "eligible_source_ids": [SOURCE_ID, "foreign"]},
    {"sources": [SOURCE], "eligible_source_ids": [SOURCE_ID, SOURCE_ID]},
    {"sources": [SOURCE], "eligible_source_ids": SOURCE_ID},
    {"sources": [SOURCE], "eligible_source_ids": [None]},
    {"sources": [SOURCE], "eligible_source_ids": [""]},
    {"sources": [SOURCE], "eligible_source_ids": None},
    {"source": {**SOURCE, "id": "foreign"}, "eligible_source_ids": [SOURCE_ID],
     "sources": [SOURCE]},
    {"source": None, "sources": [SOURCE]},
    {"source": "raw-memory-command-text", "sources": [SOURCE]},
    {"source": [SOURCE], "sources": [SOURCE]},
    {"source": {**SOURCE, "text": 5}, "sources": [SOURCE]},
    {"sources": []},
    {"sources": "bad"},
    {"sources": [{"id": SOURCE_ID, "role": "user"}], "eligible_source_ids": [SOURCE_ID]},
    {"sources": [{"id": SOURCE_ID, "role": "user", "text": None}],
     "eligible_source_ids": [SOURCE_ID]},
])
def test_ambiguous_or_invalid_complete_source_never_uses_another_source_or_scaffold(payload):
    request = {**payload, "segments": [{"id": 0, "text": "可诱导的尾段"}]}
    assert _source_text(messages(request)) == ""


def test_explicit_assistant_maintenance_selects_only_its_own_confirmed_source():
    own = {"id": "own-assistant", "role": "assistant", "text": "已确认的助手原文。"}
    request = {"sources": [own, SOURCE], "eligible_source_ids": [own["id"]],
               "segments": _coverage_segments(own["text"])}
    assert _source_text(messages(request)) == own["text"]


@pytest.mark.parametrize("text", [
    "我暂时喜欢安静的音乐，\n此偏好有效至2026-10-06T12:47:00.931Z。",
    "  我暂时喜欢安静的音乐， 此偏好有效至2026-10-06T12:47:00.931Z。  ",
])
def test_complete_source_reader_preserves_whitespace_bytes(text):
    # Source selection preserves original bytes. The separate initial-target
    # artifact still records the two unowned fake/scaffold complete-act failures.
    assert _source_text(messages(source_request(text))) == text


@pytest.mark.parametrize("text", [
    "“喜欢散步”，只是引文。随后是同样的“喜欢散步”。",
    "这次选散步；你还记得吗？期限为2026-10-06T12:47:00.931Z。",
])
def test_fake_full_source_preserves_supported_punctuation_quotes_and_multiple_clauses(text):
    source = EvidenceTurn(SCOPE, SOURCE_ID, "user", text, len(text))
    prepared = ExchangeModel(_fake_memory_result, Event()).prepare_source_intents(
        turns=(source,), targets={}, eligible_source_ids=(SOURCE_ID,))
    assert prepared.complete
    assert json.loads(prepared.interpretation)["acts"][0]["evidenceQuote"] == text
    assert _fake_memory_result("fact_extract", [], {}, Event()) == {"operations": []}


def test_segments_only_legacy_reader_is_not_promoted_to_a_complete_typed_source():
    assert _source_text(messages({"segments": [{"id": 0, "text": "先段"},
                                               {"id": 1, "text": "尾段"}]})) == "尾段"
    assert _source_text(messages({"eligible_source_ids": [SOURCE_ID],
                                  "segments": [{"id": 0, "text": "尾段"}]})) == ""


@pytest.mark.parametrize("wire_messages", [[], [{"role": "assistant", "content": "{}"}],
                                       [{"role": "user", "content": "not-json"}],
                                       [{"role": "user", "content": "[]"}],
                                       [{"role": "user", "content": None}]])
def test_invalid_message_payload_cannot_supply_a_source(wire_messages):
    assert _source_text(wire_messages) == ""


def prepared_ranges(text, kinds):
    segments = _coverage_segments(text)
    assert len(segments) == len(kinds)
    return PreparedSourceIntents("0" * 64, canonical({"coverage": "complete", "acts": [
        {"segmentId": segment["id"], "evidenceTurnId": SOURCE_ID,
         "evidenceQuote": segment["text"].strip(), "intent": kind, "targetId": None}
        for segment, kind in zip(segments, kinds, strict=True)]}), True)


@pytest.mark.parametrize("other", ["question", "request", "context", "quotation"])
def test_guard_preserves_independent_other_spans_without_merging_across_them(other):
    first, middle, last = "这次选散步；", "另一个事项。", "有效至2026-10-06T12:47:00.931Z。"
    text = first + middle + last
    source = EvidenceTurn(SCOPE, SOURCE_ID, "user", text, len(text))
    request = [{"role": "system", "content": "instructions\nJSON schema:\n{}"},
               {"role": "user", "content": canonical({"source": {**SOURCE, "text": text}})}]

    def reply(stage, scoped, schema, cancelled):
        ranges = json.loads(scoped[-1]["content"])["fact_evidence_ranges"]
        assert [item["text"] for item in ranges] == [first, middle, last]
        return {"operations": [{"evidenceQuote": text}]}

    prepared = prepared_ranges(text, ["statement", other, "statement"])
    invoke = _fact_source_invoke(reply, source, prepared)
    with pytest.raises(ValueError, match="source ranges"):
        invoke("fact_extract", request, {}, Event())


@pytest.mark.parametrize("changed", [
    "wrong_quote", "wrong_source", "wrong_start", "overflow", "negative"])
def test_existing_range_guard_rejects_bad_quotes_source_refs_and_segment_bounds(changed):
    source = EvidenceTurn(SCOPE, SOURCE_ID, "user", TEXT, len(TEXT))
    prepared = prepared_ranges(TEXT, ["statement", "statement"])
    value = json.loads(prepared.interpretation)
    act = value["acts"][0]
    if changed == "wrong_quote":
        act["evidenceQuote"] = "外来原文"
    elif changed == "wrong_source":
        act["evidenceTurnId"] = "foreign"
    elif changed == "wrong_start":
        act["segmentId"] = 1
    elif changed == "overflow":
        act["segmentId"] = 100
    else:
        act["segmentId"] = -1
    with pytest.raises(ValueError, match="current source"):
        bad = replace(prepared, interpretation=canonical(value))
        _fact_source_invoke(lambda *_: None, source, bad)


def capture_ranges(source, prepared):
    captured = []

    def recording(stage, request, schema, cancelled):
        payload = json.loads(request[-1]["content"])
        assert payload["source"]["text"] == source.text
        captured.extend(payload["fact_evidence_ranges"])
        return {"operations": []}

    scoped = _fact_source_invoke(recording, source, prepared)
    scoped("fact_extract", [{"role": "system", "content": "rules\nJSON schema:\n{}"},
           {"role": "user", "content": canonical({"source": {**SOURCE, "text": source.text}})}],
           {}, Event())
    return captured


def test_legal_independent_acts_keep_original_indices_across_a_whitespace_scaffold_gap():
    text = "  我暂时喜欢安静的音乐，\n 此偏好有效至2026-10-06T12:47:00.931Z。  "
    source = EvidenceTurn(SCOPE, SOURCE_ID, "user", text, len(text))
    ranges = capture_ranges(source, prepared_ranges(text, ["statement", "statement"]))
    assert ranges == [{"start": 2, "end": len(text.rstrip()), "text": text.strip()}]
    assert source.text == text


def test_repeated_quotes_keep_the_actual_later_statement_indices():
    repeated, gap, last = "这次选散步；", "只是举例。", "有效至2026-10-06T12:47:00.931Z。"
    text = repeated + gap + repeated + last
    source = EvidenceTurn(SCOPE, SOURCE_ID, "user", text, len(text))
    ranges = capture_ranges(source, prepared_ranges(
        text, ["quotation", "context", "statement", "statement"]))
    assert ranges == [
        {"start": 0, "end": len(repeated), "text": repeated},
        {"start": len(repeated), "end": len(repeated + gap), "text": gap},
        {"start": len(repeated + gap), "end": len(text), "text": repeated + last},
    ]


def test_independent_other_act_can_span_multiple_segments_without_expanding_statement_ranges():
    first, middle, last = (
        "这次选散步；", "另一个事项，附带其条件。", "有效至2026-10-06T12:47:00.931Z。")
    text = first + middle + last
    source = EvidenceTurn(SCOPE, SOURCE_ID, "user", text, len(text))
    acts = [{"segmentId": index, "evidenceTurnId": SOURCE_ID, "evidenceQuote": quote,
             "intent": kind, "targetId": None}
            for index, quote, kind in ((0, first, "statement"), (1, middle, "request"),
                                       (3, last, "statement"))]
    prepared = PreparedSourceIntents(
        "0" * 64, canonical({"coverage": "complete", "acts": acts}), True)
    assert [item["text"] for item in capture_ranges(source, prepared)] == [first, middle, last]


def test_current_user_selection_is_already_limited_to_the_confirmed_payload_prefix():
    unconfirmed = "未确认的助手尾段不能进入当前读取。"
    source = EvidenceTurn(SCOPE, SOURCE_ID, "user", TEXT + unconfirmed, len(TEXT))
    captured = []

    def fake(stage, request, schema, cancelled):
        captured.append(_source_text(request))
        return _fake_memory_result(stage, request, schema, cancelled)

    prepared = ExchangeModel(fake, Event()).prepare_source_intents(
        turns=(source,), targets={}, eligible_source_ids=(SOURCE_ID,))
    assert prepared.complete and captured == [TEXT]
    assert unconfirmed not in prepared.interpretation


def test_history_targets_are_not_used_when_the_current_selected_source_is_missing():
    payload = {"sources": [{"id": "foreign", "role": "assistant", "text": TEXT}],
               "eligible_source_ids": [SOURCE_ID], "segments": _coverage_segments(TEXT),
               "targets": [{"content": TEXT, "source_id": SOURCE_ID}]}
    assert _source_text(messages(payload)) == ""
    assert _fake_memory_result("fact_extract", messages(payload), {}, Event()) == {"operations": []}
