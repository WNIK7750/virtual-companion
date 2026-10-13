"""Single-URL public extraction through the existing synthetic application host."""

import json
from dataclasses import replace
from functools import partial
from hashlib import sha256
from threading import BoundedSemaphore, Event

import pytest
import test_public_search_consumer as search_fixture
from virtual_companion.providers.external_information import (
    ApprovedPublicQuery,
    ExternalConfiguration,
    ExternalEvidence,
    ExternalQueryCall,
    ExternalReservation,
    bind_external,
    canonical,
    parse_public_action,
)
from virtual_companion.text.demand_memory import _action, action_schema, run_demand_turn
from virtual_companion.text.memory_runtime import DEFAULT_COMPANION_ID, TextMemoryRuntime
from virtual_companion.text.public_query import PublicQueryCapability

URL = "https://example.invalid/museum?date=2026-10-12"
BODY = "The museum opens at 09:00 today."
ACTION = {"action": "extract_public", "url": URL}


def configuration():
    return ExternalConfiguration("synthetic-read", 1, "recording", "v1", "none", "offline",
                                 True, True, 2048, 16000)


def test_extract_requires_an_explicit_operation_capability():
    search = bind_external(configuration(), "a" * 64)
    extract = bind_external(configuration(), "a" * 64, supports_extract=True)
    assert search.identity != extract.identity
    assert "extract_public" not in search.schema_json
    assert "extract_public" in extract.schema_json
    with pytest.raises(ValueError):
        replace(search, supports_extract=True)
    default = action_schema(allow_memory=False, allow_recall=False, allow_public_search=True)
    assert "extract_public" not in canonical(default)
    selected = action_schema(allow_memory=False, allow_recall=False, allow_public_search=True,
                             allow_public_extract=True)
    assert "extract_public" in canonical(selected)
    with pytest.raises(ValueError):
        _action(canonical(ACTION), final=False, allow_public_search=True)
    assert _action(canonical(ACTION), final=False, allow_public_search=True,
                   allow_public_extract=True) == ACTION


@pytest.mark.parametrize("invalid", [
    {"action": "extract_public", "url": [URL]},
    {"action": "extract_public", "query": URL},
    {"action": "extract_public", "url": URL, "query": "extra"},
    {"action": "extract_public", "url": "file:///private.txt"},
    {"action": "extract_public", "url": "https://user:password@example.invalid/"},
    {"action": "extract_public", "url": "https://example.invalid/ bad"},
])
def test_extract_rejects_ambiguous_or_non_http_inputs(invalid):
    with pytest.raises(ValueError):
        parse_public_action(invalid)


def _call(*, binding, approval):
    return ExternalQueryCall(
        binding=binding, request_id="synthetic-extract", approve_query=approval,
        authorized=lambda *_: True, checkpoint=lambda: None, can_admit=lambda: True,
        cancelled=Event(), clock=lambda: 1, deadline=20, slots=BoundedSemaphore(1),
        budget=search_fixture.RecordingBudget([]),
        allowance=ExternalReservation(1, 0, None, None),
        backend_factory=lambda _: pytest.fail("Invalid operation reached dispatch"),
        retrieved_clock=lambda: "2026-10-12T12:00:00Z", available_bytes=16000,
    )


def test_extract_cannot_execute_under_search_only_binding_or_mismatched_approval():
    call = _call(binding=bind_external(configuration(), "a" * 64),
                 approval=lambda *_: pytest.fail("Unsupported extract reached approval"))
    with pytest.raises(ValueError):
        call.prepare(ACTION)
    for approved in (
        ApprovedPublicQuery(URL, "grant", "task"),
        ApprovedPublicQuery(URL + "&other=1", "grant", "task", operation="extract_public"),
    ):
        call = _call(binding=bind_external(configuration(), "a" * 64, supports_extract=True),
                     approval=lambda *_, approved=approved, **__: approved)
        with pytest.raises(ValueError):
            call.prepare(ACTION)


@pytest.mark.parametrize("actions", [
    [ACTION, {"action": "search_public", "query": "museum hours"}],
    [{"action": "search_public", "query": "museum hours"}, ACTION],
])
def test_search_and_extract_share_one_public_call(actions):
    actions = iter(actions)
    dispatches = []

    class Wire:
        def check_context(self):
            pass

        def stream(self, cancelled):
            yield canonical(next(actions))

    with pytest.raises(ValueError):
        run_demand_turn(
            claim=object(), prepare=lambda *_, **__: Wire(), resolve_current=lambda _: None,
            refresh=lambda owner: owner, cancelled=Event(), max_public_calls=1,
            search_public=lambda _, query, **__: dispatches.append(("search", query)),
            extract_public=lambda _, url, **__: dispatches.append(("extract", url)),
            allow_public_extract=True,
        )
    assert len(dispatches) == 1


def extract_environment(tmp_path, monkeypatch, *, public=True, enabled=True, actions=None):
    calls = []

    class Backend:
        def connect(self, **kwargs):
            calls.append(("connect",))

        def search(self, body, *, request_id, binding_id, **kwargs):
            calls.append(("dispatch", body))
            assert body == {"url": URL}
            return ExternalEvidence.create(
                request_id=request_id, binding_id=binding_id, status="success", failure=None,
                coverage="returned_result_set", retrieved_at="2026-10-12T12:00:00Z",
                references=[{"source_id": "page-1", "url": URL, "title": "Museum",
                             "published_at": None, "content_kind": "body", "text": BODY,
                             "source_range": None}], native_events=[], citation_map=[], summary="",
                usage_status="unknown",
                usage={"tokens": None, "search": None, "cost_microusd": None, "raw": None},
            )

        def close(self):
            calls.append(("close",))

    def capability(*args):
        cap = PublicQueryCapability(*args, supports_extract=True)

        def approve(binding, raw, *, operation="search_public"):
            if not enabled:
                raise PermissionError("external_authorization_revoked")
            calls.append(("approve", raw, operation))
            return ApprovedPublicQuery(raw, "synthetic-grant", "synthetic-task",
                                       operation=operation)

        return replace(cap, approve_query=approve, supports_extract=True)

    monkeypatch.setattr(search_fixture, "bind_external", partial(bind_external,
                                                                supports_extract=True))
    monkeypatch.setattr(search_fixture, "PublicQueryCapability", capability)
    parts = search_fixture.environment(
        tmp_path, public=public, backend_factory=lambda _: Backend(),
        actions=actions or [ACTION, {"action": "answer", "text": "公开页面写的是今天九点开门。"}],
    )
    return parts, calls


def test_actual_host_extract_review_v8_ack_and_reopen(tmp_path, monkeypatch):
    parts, calls = extract_environment(tmp_path, monkeypatch)
    runtime, text, memory, mirror, model, events, budget, authority = parts
    try:
        text.send("请读这个公开页面：" + URL)
        event = search_fixture.finish(runtime, text)
        capture = event["readset"]["public_capture"]
        assert event["readset"]["protocol"] == "history-readset.v8"
        reference = capture["references"][0]
        assert reference["url"] == URL and reference["text"] == BODY
        assert reference["content_sha256"] == sha256(BODY.encode()).hexdigest()
        assert [row for row in calls if row[0] == "dispatch"] == [("dispatch", {"url": URL})]
        assert ("approve", URL, "extract_public") in calls
        review = next(row for row in model.calls if row["stage"] == "stream_unit_review")
        assert json.loads(review["messages"][-1]["content"])["public_evidence"] == capture
        generations = [row for row in model.calls if row["stage"] == "demand_control"]
        assert "extract_public" in canonical(generations[0]["schema"])
        assert "extract_public" not in canonical(generations[-1]["schema"])
        assert "search_public" not in canonical(generations[-1]["schema"])
        input_id = runtime.claim.input_id
        saved = memory.store.segments(memory.companion, input_id)[0]
        assert saved["fingerprint"] == event["fingerprint"]
        assert saved["readset"] == event["readset"]
        text.stop()
        memory.close()
        reopened = TextMemoryRuntime(tmp_path / "memory.db", DEFAULT_COMPANION_ID,
                                     model=model, fake=True)
        try:
            after = reopened.store.segments(reopened.companion, input_id)[0]
            assert after["readset"] == event["readset"]
        finally:
            reopened.close()
        (tmp_path / "extract-recording.json").write_text(canonical({
            "calls": calls, "model": model.calls, "event": event, "saved": saved,
        }), encoding="utf-8")
    finally:
        search_fixture.close(runtime, text, memory, mirror)


@pytest.mark.parametrize("public,enabled", [(False, True), (True, False)])
def test_extract_absent_or_denied_has_zero_dispatch(tmp_path, monkeypatch, public, enabled):
    actions = ([{"action": "answer", "text": "可以先从本地材料说起。"}] if not public else None)
    parts, calls = extract_environment(tmp_path, monkeypatch, public=public, enabled=enabled,
                                      actions=actions)
    runtime, text, memory, mirror, model, events, budget, authority = parts
    try:
        text.send("请读公开页面。")
        event = search_fixture.finish(runtime, text)
        assert "public_capture" not in event["readset"]
        assert not calls and not budget.rows
        if not public:
            assert all("extract_public" not in canonical(row["schema"]) for row in model.calls)
    finally:
        search_fixture.close(runtime, text, memory, mirror)
