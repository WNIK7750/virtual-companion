"""Actual application wire tests with synthetic-only dependencies."""


# isort: split
import json
import time
from copy import deepcopy
from dataclasses import replace
from threading import BoundedSemaphore, Event

import pytest
from test_default_text_delivery import wait_for
from virtual_companion.boundary.history_readset import validate_readset
from virtual_companion.providers.external_information import (
    ApprovedPublicQuery,
    ExternalConfiguration,
    ExternalEvidence,
    ExternalReservation,
    bind_external,
    canonical,
)
from virtual_companion.providers.public_search_ddg import HttpResponse, PiDuckDuckGoBackend
from virtual_companion.text.application import TextSession
from virtual_companion.text.default_runtime import DefaultTextRuntime
from virtual_companion.text.demand_memory import _action, action_schema
from virtual_companion.text.fake import synthetic_unit_opinion
from virtual_companion.text.memory_runtime import DEFAULT_COMPANION_ID, TextMemoryRuntime
from virtual_companion.text.public_query import PublicQueryCapability
from virtual_companion.text.sqlite_store import SQLiteConversationStore

QUERY = "public museum hours 2026-10-12"
SOURCE = "请查公开展馆今天几点开门。"
HTML = (
    b'<html><div class="result"><a class="result__a" '
    b'href="https://example.invalid/museum">Museum</a>'
    b'<a class="result__snippet">The museum opens at 09:00 today.</a></div></html>'
)


class RecordingBudget:
    def __init__(self, events):
        self.events, self.rows = events, {}

    def reserve(self, **kwargs):
        self.events.append(("reserve", kwargs["request_id"]))
        self.rows["receipt"] = {**kwargs, "allowance": repr(kwargs["allowance"]), "finish": None}
        return "receipt"

    def finish(self, receipt, **kwargs):
        self.rows[receipt]["finish"] = kwargs
        self.events.append(("finish", kwargs["usage_status"]))


class RecordingModel:
    supports_stage_metadata = True

    def __init__(self, actions=None, *, reject=False, on_review=None):
        self.actions = list(
            actions
            or [
                {"action": "search_public", "query": QUERY},
                {"action": "answer", "text": "公开摘录写的是今天九点开门。"},
            ]
        )
        self.wires, self.calls, self.reject, self.on_review = [], [], reject, on_review

    def prepare_stage_request(self, messages, *, stage, schema):
        model = self
        messages, schema = deepcopy(messages), deepcopy(schema)
        record = {"stage": stage, "messages": messages, "schema": schema, "checked": False}
        self.wires.append(record)

        class Wire:
            def check_context(self):
                record["checked"] = True
                record["counted_bytes"] = len(canonical(messages).encode("utf-8"))

            def stream(self, cancelled):
                assert record["checked"]
                model.calls.append(record)
                if stage == "stream_unit_review":
                    if model.on_review:
                        model.on_review(messages)
                    view = json.loads(messages[-1]["content"])
                    opinion = synthetic_unit_opinion(view)
                    if model.reject:
                        opinion.update(decision="reject", constraints_satisfied=False)
                    yield json.dumps(opinion, ensure_ascii=False)
                else:
                    yield json.dumps(model.actions.pop(0), ensure_ascii=False)

        return Wire()

    def cancel(self):
        pass


def environment(
    tmp_path,
    *,
    response=None,
    actions=None,
    reject=False,
    public=True,
    backend_factory=None,
    slots=None,
    http_hook=None,
    history_projection=None,
):
    events, authority = [], {"enabled": True}
    model = RecordingModel(actions, reject=reject)
    memory = TextMemoryRuntime(tmp_path / "memory.db", DEFAULT_COMPANION_ID, model=model, fake=True)
    mirror = SQLiteConversationStore(tmp_path / "chat", companion_id=DEFAULT_COMPANION_ID)
    binding = bind_external(
        ExternalConfiguration(
            "synthetic-recording-route",
            1,
            "pi-duckduckgo-html.v1",
            "synthetic-v1",
            "synthetic-credential-version",
            "synthetic-policy-version",
            True,
            True,
            2048,
            16000,
        ),
        memory.config.character.sha256,
    )

    class HTTP:
        def get(self, url, **kwargs):
            events.append(("http_get", url))
            if http_hook:
                http_hook(authority)
            return response or HttpResponse(200, {}, HTML)

    def approve(bound, raw_query):
        if not authority["enabled"]:
            raise PermissionError("external_authorization_revoked")
        events.append(("approve", raw_query))
        return ApprovedPublicQuery(QUERY, "synthetic-grant", "synthetic-task")

    def authorized(bound, approved):
        events.append(("authority",))
        return authority["enabled"] and approved.authorization_id == "synthetic-grant"

    budget = RecordingBudget(events)
    cap = PublicQueryCapability(
        binding,
        approve,
        authorized,
        budget,
        ExternalReservation(1, 1000, None, None),
        slots or BoundedSemaphore(1),
        backend_factory
        or (
            lambda bound: PiDuckDuckGoBackend(
                http=HTTP(), clock=time.monotonic, retrieved_clock=lambda: "2026-10-12T12:00:00Z"
            )
        ),
        time.monotonic,
        lambda: "2026-10-12T12:00:00Z",
    )
    runtime = DefaultTextRuntime(
        memory,
        mirror,
        model,
        binding={"test": "synthetic"},
        delivery_timeout=1.0,
        public_query=cap if public else None,
        history_projection=history_projection,
    )
    text = TextSession(
        model, mirror, {}, prepare_turn=runtime.prepare_turn, before_start=runtime.before_start
    )
    return runtime, text, memory, mirror, model, events, budget, authority


def finish(runtime, text):
    event = wait_for(lambda: runtime.poll()["event"] or not text.snapshot()["busy"])
    assert isinstance(event, dict), text.snapshot()
    runtime.ack(
        {
            "sequence": event["sequence"],
            "fingerprint": event["fingerprint"],
            "host_receipt": "synthetic-public-dom-" + event["fingerprint"][:16],
        }
    )
    wait_for(lambda: not text.snapshot()["busy"])
    return event


def close(runtime, text, memory, mirror):
    runtime.stop()
    text.stop()
    memory.close()
    mirror.close()


def test_explicit_public_schema_and_decoder_are_jointly_selected():
    default = action_schema(allow_memory=True, allow_recall=False)
    selected = action_schema(allow_memory=True, allow_recall=False, allow_public_search=True)
    assert len(selected["oneOf"]) == len(default["oneOf"]) + 1
    value = _action(
        '{"action":"search_public","query":"public exact query"}',
        final=False,
        allow_public_search=True,
    )
    assert value == {"action": "search_public", "query": "public exact query"}


def test_actual_application_search_review_manifest_and_reopen(tmp_path):
    runtime, text, memory, mirror, model, events, budget, authority = environment(tmp_path)
    try:
        assert events == []
        text.send(SOURCE)
        event = finish(runtime, text)
        readset = event["readset"]
        assert readset["protocol"] == "history-readset.v8"
        capture = readset["public_capture"]
        assert capture["source_kind"] == "public_web"
        assert capture["references"][0]["content_kind"] == "snippet"
        assert capture["references"][0]["text"] == "The museum opens at 09:00 today."
        assert capture["references"][0]["source_range"] is None
        assert "summary" not in capture and "native_events" not in capture
        review = next(row for row in model.calls if row["stage"] == "stream_unit_review")
        view = json.loads(review["messages"][-1]["content"])
        assert view["public_evidence"] == capture
        assert all("museum opens" not in item["text"] for item in view["personal_sources"])
        assert not readset["facts"]
        assert all(row["checked"] for row in model.calls)
        assert [row[0] for row in events].index("reserve") < [row[0] for row in events].index(
            "http_get"
        )
        assert sum(row[0] == "http_get" for row in events) == 1
        assert next(iter(budget.rows.values()))["finish"] is not None
        input_id, fingerprint = runtime.claim.input_id, event["fingerprint"]
        saved = memory.store.segments(memory.companion, input_id)[0]
        assert saved["fingerprint"] == fingerprint
        assert saved["readset"] == readset
        text.stop()
        memory.close()
        reopened = TextMemoryRuntime(
            tmp_path / "memory.db", DEFAULT_COMPANION_ID, model=model, fake=True
        )
        try:
            after = reopened.store.segments(reopened.companion, input_id)[0]
            assert after["readset"] == readset and after["fingerprint"] == fingerprint
            (tmp_path / "recording.json").write_text(
                canonical(
                    {
                        "generation_and_review": model.calls,
                        "public_capture": capture,
                        "manifest": after,
                        "transport_and_accounting": events,
                        "external_audit": runtime._public_session.outcome.evidence.data,
                        "budget_receipts": budget.rows,
                    }
                ),
                encoding="utf-8",
            )
        finally:
            reopened.close()
    finally:
        runtime.stop()
        text.stop()
        memory.close()
        mirror.close()


def test_host_denied_query_leaves_local_answer_available_without_dispatch(tmp_path):
    runtime, text, memory, mirror, model, events, budget, authority = environment(tmp_path)
    authority["enabled"] = False
    try:
        text.send(SOURCE)
        event = finish(runtime, text)
        assert event["readset"]["protocol"] != "history-readset.v8"
        assert not budget.rows and not any(row[0] == "http_get" for row in events)
        generation = [row for row in model.calls if row["stage"] == "demand_control"][-1]
        assert {"status": "unavailable", "reason": "external_authorization_revoked"} == next(
            json.loads(row["content"])["public_query_outcome"]
            for row in generation["messages"]
            if row["role"] == "system" and row["content"].startswith('{"public_query_outcome":')
        )
    finally:
        close(runtime, text, memory, mirror)


def test_optional_absent_or_unused_capability_keeps_existing_local_manifest(tmp_path):
    for public in (False, True):
        path = tmp_path / str(public)
        path.mkdir()
        runtime, text, memory, mirror, model, events, budget, authority = environment(
            path, public=public, actions=[{"action": "answer", "text": "可以先从本地材料说起。"}]
        )
        try:
            text.send("先聊本地材料。")
            event = finish(runtime, text)
            assert event["readset"]["protocol"] != "history-readset.v8"
            assert "public_capture" not in event["readset"]
            assert not events and not budget.rows
            if not public:
                assert all("search_public" not in canonical(row["schema"]) for row in model.calls)
        finally:
            close(runtime, text, memory, mirror)


def test_http_202_failure_has_no_sources_but_allows_local_response(tmp_path):
    runtime, text, memory, mirror, model, events, budget, authority = environment(
        tmp_path,
        response=HttpResponse(202, {}, b"bot challenge"),
        actions=[
            {"action": "search_public", "query": QUERY},
            {"action": "answer", "text": "这次公开查询没有得到可用来源，开门时间目前未知。"},
        ],
    )
    try:
        text.send(SOURCE)
        event = finish(runtime, text)
        capture = event["readset"]["public_capture"]
        assert capture["status"] == "failed" and not capture["references"]
        assert not capture["citation_map"]
        assert sum(row[0] == "http_get" for row in events) == 1
        assert next(iter(budget.rows.values()))["finish"]["status"] == "failed"
    finally:
        close(runtime, text, memory, mirror)


def test_public_sources_visible_does_not_override_semantic_rejection(tmp_path):
    runtime, text, memory, mirror, model, events, budget, authority = environment(
        tmp_path,
        reject=True,
        actions=[
            {"action": "search_public", "query": QUERY},
            {"action": "answer", "text": "我来过这个展馆，知道你每天都会去。"},
            {"action": "answer", "text": "我还是知道你每天都会去。"},
        ],
    )
    try:
        text.send(SOURCE)
        wait_for(lambda: not text.snapshot()["busy"])
        assert text.snapshot()["attempt"]["status"] == "failed"
        assert memory.store.segments(memory.companion, runtime.claim.input_id) == []
        reviews = [row for row in model.calls if row["stage"] == "stream_unit_review"]
        assert reviews and all(
            "public_evidence" in json.loads(row["messages"][-1]["content"]) for row in reviews
        )
        assert sum(row[0] == "http_get" for row in events) == 1
    finally:
        close(runtime, text, memory, mirror)


def test_live_authority_revoked_during_review_prevents_approval_and_dispatch(tmp_path):
    runtime, text, memory, mirror, model, events, budget, authority = environment(tmp_path)
    model.on_review = lambda messages: authority.update(enabled=False)
    try:
        text.send(SOURCE)
        wait_for(lambda: not text.snapshot()["busy"])
        assert text.snapshot()["attempt"]["status"] == "failed"
        assert memory.store.segments(memory.companion, runtime.claim.input_id) == []
        assert sum(row[0] == "http_get" for row in events) == 1
        assert next(iter(budget.rows.values()))["finish"] is not None
    finally:
        close(runtime, text, memory, mirror)


def test_provider_summary_reference_is_declined_locally_not_a_web_body(tmp_path):
    class SummaryBackend:
        def connect(self, **kwargs):
            pass

        def search(self, action, *, request_id, binding_id, **kwargs):
            content = "The provider infers opening hours without a captured page."
            return ExternalEvidence.create(
                request_id=request_id,
                binding_id=binding_id,
                status="partial",
                failure=None,
                coverage="partial",
                retrieved_at="2026-10-12T12:00:00Z",
                references=[
                    {
                        "source_id": "provider-summary",
                        "url": "https://example.invalid/museum",
                        "title": "Summary",
                        "published_at": None,
                        "content_kind": "provider_summary",
                        "text": content,
                        "source_range": None,
                    }
                ],
                native_events=[],
                citation_map=[],
                summary="",
                usage_status="unknown",
                usage={"tokens": None, "search": None, "cost_microusd": None, "raw": None},
            )

        def close(self):
            pass

    runtime, text, memory, mirror, model, events, budget, authority = environment(
        tmp_path,
        backend_factory=lambda binding: SummaryBackend(),
        actions=[
            {"action": "search_public", "query": QUERY},
            {"action": "answer", "text": "还没有网页原文支持，开门时间未知。"},
        ],
    )
    try:
        text.send(SOURCE)
        event = finish(runtime, text)
        assert "public_capture" not in event["readset"]
        assert all(
            "public_evidence" not in json.loads(row["messages"][-1]["content"])
            for row in model.calls
            if row["stage"] == "stream_unit_review"
        )
        assert next(iter(budget.rows.values()))["finish"] is not None
    finally:
        close(runtime, text, memory, mirror)


def test_public_capture_binding_hash_range_and_http_are_checked_independently(tmp_path):
    from virtual_companion.boundary.public_capture import (
        validate_public_capture,
        validate_public_scope,
    )

    runtime, text, memory, mirror, model, events, budget, authority = environment(tmp_path)
    try:
        text.send(SOURCE)
        event = finish(runtime, text)
        capture = deepcopy(event["readset"]["public_capture"])
        ref = capture["references"][0]
        capture["citation_map"] = [
            {
                "source_id": ref["source_id"],
                "url": ref["url"],
                "source_content_sha256": ref["content_sha256"],
                "captured_excerpt_range": {
                    "unit": "unicode_codepoints",
                    "start": 0,
                    "end": len(ref["text"]),
                },
            }
        ]
        capture["references"][0]["url"] = "http://example.invalid/museum"
        capture["citation_map"][0]["url"] = "http://example.invalid/museum"
        assert validate_public_capture(capture) == capture
        for mutate in ("text", "url", "range"):
            broken = deepcopy(event["readset"])
            broken["public_capture"] = deepcopy(capture)
            ref = broken["public_capture"]["references"][0]
            if mutate == "text":
                ref["text"] += " altered"
            elif mutate == "url":
                broken["public_capture"]["citation_map"][0]["url"] = "https://example.invalid/other"
            else:
                broken["public_capture"]["citation_map"][0]["captured_excerpt_range"]["end"] = 17.0
            with pytest.raises(ValueError):
                validate_readset(broken)
        foreign = deepcopy(event["basis"])
        foreign["attempt_id"] = "different-attempt"
        with pytest.raises(ValueError):
            validate_public_scope(event["readset"], foreign)
        row = {
            "manifest": canonical(
                {
                    key: event[key]
                    for key in (
                        "protocol",
                        "basis",
                        "sequence",
                        "presentation",
                        "readset",
                        "bubbles",
                    )
                }
            ),
            "fingerprint": event["fingerprint"],
            "status": "confirmed",
            "host_receipt": "test",
        }
        row["manifest"] = row["manifest"].replace("09:00", "10:00")
        with pytest.raises(ValueError):
            memory.store._segment_event(row)
    finally:
        close(runtime, text, memory, mirror)


def test_public_capture_survives_native_history_with_confirmed_prior_turn_and_reopen(tmp_path):
    historical = "合成历史标识：蓝色展馆地图 AB-12。"
    runtime, text, memory, mirror, model, events, budget, authority = environment(
        tmp_path,
        history_projection="native-json.v1",
        actions=[
            {"action": "answer", "text": "这张地图可以作为接下来查找的线索。"},
            {"action": "search_public", "query": QUERY},
            {"action": "answer", "text": "这次公开摘录写的是九点开门。"},
        ],
    )
    try:
        assert not events
        text.send(historical)
        prior = finish(runtime, text)
        assert "public_capture" not in prior["readset"] and not events
        text.send(SOURCE)
        event = finish(runtime, text)
        capture = event["readset"]["public_capture"]
        last_generation = [row for row in model.calls if row["stage"] == "demand_control"][-1]
        assert historical in canonical(last_generation["messages"])
        assert capture == next(
            json.loads(row["content"])["public_evidence"]
            for row in last_generation["messages"]
            if row["role"] == "user" and row["content"].startswith('{"public_evidence":')
        )
        input_id = runtime.claim.input_id
        text.stop()
        memory.close()
        reopened = TextMemoryRuntime(
            tmp_path / "memory.db", DEFAULT_COMPANION_ID, model=model, fake=True
        )
        try:
            after = reopened.store.segments(reopened.companion, input_id)[0]
            assert after["readset"] == event["readset"]
            (tmp_path / "native-history-recording.json").write_text(
                canonical(
                    {
                        "prior_confirmed_event": prior,
                        "generation_and_review": model.calls,
                        "public_capture": capture,
                        "reopened_manifest": after,
                    }
                ),
                encoding="utf-8",
            )
        finally:
            reopened.close()
    finally:
        close(runtime, text, memory, mirror)


def test_public_capability_must_match_frozen_claim_role_before_connect(tmp_path):
    runtime, text, memory, mirror, model, events, budget, authority = environment(
        tmp_path, actions=[{"action": "answer", "text": "先聊本地材料。"}]
    )
    try:
        text.send("本地题面。")
        finish(runtime, text)
        config = json.loads(runtime.claim.config_json)
        config["companion_config"]["character"]["sha256"] = "1" * 64
        forged = replace(runtime.claim, config_json=canonical(config))
        with pytest.raises(ValueError, match="frozen turn"):
            runtime.public_query.start(runtime, forged, Event())
        assert not events and not budget.rows
    finally:
        close(runtime, text, memory, mirror)


def test_authority_revoked_while_waiting_for_shared_slot_never_connects(tmp_path):
    class Slots:
        revoke = None
        released = 0

        def acquire(self, *, timeout):
            self.revoke()
            return False

        def release(self):
            self.released += 1

    slots = Slots()
    runtime, text, memory, mirror, model, events, budget, authority = environment(
        tmp_path,
        slots=slots,
        actions=[
            {"action": "search_public", "query": QUERY},
            {"action": "answer", "text": "公开查询没有继续，我先依据本地材料回答。"},
        ],
    )
    slots.revoke = lambda: authority.update(enabled=False)
    try:
        text.send(SOURCE)
        event = finish(runtime, text)
        assert "public_capture" not in event["readset"]
        assert not budget.rows and slots.released == 0
        assert not any(row[0] == "http_get" for row in events)
    finally:
        close(runtime, text, memory, mirror)


def test_late_public_result_is_settled_without_changing_local_delivery(tmp_path):
    offset = {"seconds": 0}
    runtime, text, memory, mirror, model, events, budget, authority = environment(
        tmp_path,
        http_hook=lambda authority: offset.update(seconds=1000),
        actions=[
            {"action": "search_public", "query": QUERY},
            {"action": "answer", "text": "公开查询超过期限，开门时间仍未知。"},
        ],
    )
    runtime.public_query = replace(
        runtime.public_query, clock=lambda: time.monotonic() + offset["seconds"]
    )
    try:
        text.send(SOURCE)
        event = finish(runtime, text)
        assert "public_capture" not in event["readset"]
        assert runtime._public_session.outcome.blocked_reason == "external_deadline"
        assert next(iter(budget.rows.values()))["finish"] is not None
        assert sum(row[0] == "http_get" for row in events) == 1
    finally:
        close(runtime, text, memory, mirror)
