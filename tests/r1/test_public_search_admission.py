import json
from pathlib import Path
from threading import BoundedSemaphore, Event
from urllib.parse import quote

import pytest
from test_public_search_consumer import RecordingBudget
from virtual_companion.providers import external_information as subject
from virtual_companion.providers.public_search_tavily import TavilyBasicBackend


@pytest.mark.parametrize("cause", ["actual_float_coordinate", "validator_exception"])
def test_received_invalid_evidence_does_not_grant_admission(cause, monkeypatch):
    original = Path(__file__).parent / "fixtures/public-search-invalid-native.json"
    data = json.loads(original.read_text(encoding="utf-8"))
    if cause == "validator_exception":
        data["citation_map"][0]["captured_excerpt_range"]["end"] = len(
            data["references"][0]["text"]
        )

        def broken_validator(_data):
            raise TypeError("synthetic unexpected validation failure")

        monkeypatch.setattr(subject, "_citation_admission_reason", broken_validator)
    binding = subject.bind_external(
        subject.ExternalConfiguration(
            "root-boundary-regression",
            1,
            "recording",
            "r3",
            "none",
            "offline",
            True,
            True,
            2048,
            64000,
        ),
        "a" * 64,
    )
    log, slots = [], BoundedSemaphore(1)
    budget = RecordingBudget(log)

    class Backend:
        def connect(self, **kwargs):
            log.append(("connect",))

        def search(self, body, *, request_id, binding_id, **kwargs):
            data.update(request_id=request_id, binding_id=binding_id)
            return subject.ExternalEvidence(subject.canonical(data))

        def close(self):
            log.append(("close",))

    call = subject.ExternalQueryCall(
        binding=binding,
        request_id=cause,
        approve_query=lambda _binding, query: subject.ApprovedPublicQuery(
            query, "offline", "regression"
        ),
        authorized=lambda _binding, _query: True,
        checkpoint=lambda: None,
        can_admit=lambda: True,
        cancelled=Event(),
        clock=lambda: 1,
        deadline=20,
        slots=slots,
        budget=budget,
        allowance=subject.ExternalReservation(1, 0, None, None),
        backend_factory=lambda _binding: Backend(),
        retrieved_clock=lambda: "2026-10-12T10:00:00Z",
        available_bytes=64000,
    )
    call.prepare({"action": "search_public", "query": "public institute hours"})
    outcome = call.invoke()
    assert not outcome.admissible
    assert outcome.rejected_response_sha256 is not None
    assert outcome.public_evidence is None
    assert outcome.transmission == "response_received"
    assert len(budget.rows) == 1
    assert next(iter(budget.rows.values()))["finish"] is not None
    assert ("close",) in log
    assert slots.acquire(blocking=False)
    slots.release()


def test_url_encoded_injected_credential_never_sends_or_enters_audit():
    secret = "tvly-" + "SYNTHETIC-CREDENTIAL-" + "s3cret+/only-offline"
    encoded = quote(secret, safe="")

    class Post:
        calls = 0

        def post(self, *args, **kwargs):
            self.calls += 1
            raise AssertionError("Sensitive query reached HTTP")

    post = Post()
    backend = TavilyBasicBackend(
        http_post=post,
        credential_resolver=lambda: secret,
        clock=lambda: 1.0,
        retrieved_clock=lambda: "2026-10-12T12:00:00Z",
    )
    cancel = Event()
    backend.connect(cancelled=cancel, deadline=2.0)
    try:
        outcome = backend.search(
            {"query": "public lookup " + encoded},
            request_id="synthetic-root-r2",
            binding_id="a" * 64,
            cancelled=cancel,
            deadline=2.0,
        )
        assert outcome.data["status"] == "failed"
        assert outcome.data["failure"] == "tavily_sensitive_query"
        assert post.calls == 0
        assert backend.audit_record["request_body"] is None
        assert encoded not in outcome.payload
    finally:
        backend.close()
