import asyncio
import json
import time
from hashlib import sha256
from threading import Event, Timer

import httpx
import pytest
from virtual_companion.boundary.public_capture import validate_public_capture
from virtual_companion.providers import searxng_search as leaf
from virtual_companion.providers.external_information import ExternalOutcome
from virtual_companion.providers.public_search_ddg import HttpxStreamingTransport
from virtual_companion.text import public_web as web

ENDPOINT = "http://127.0.0.1:8888/search"


def payload(*, results=None, errors=None):
    return {
        "query": "Python",
        "results": results
        if results is not None
        else [
            {
                "url": "https://www.python.org/",
                "title": "Python",
                "content": "Python official documentation.",
            }
        ],
        "answers": [],
        "corrections": [],
        "infoboxes": [],
        "suggestions": [],
        "unresponsive_engines": errors if errors is not None else [],
    }


def transport(value=None, *, raw=None, status=200, content_type="application/json", handler=None):
    requests = []

    def respond(request):
        requests.append(request)
        if handler is not None:
            return handler(request)
        body = (
            raw
            if raw is not None
            else json.dumps(value if value is not None else payload()).encode()
        )
        return httpx.Response(
            status, headers={"content-type": content_type}, stream=httpx.ByteStream(body)
        )

    http = HttpxStreamingTransport(
        clock=time.monotonic,
        client_factory=lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(respond), trust_env=False, follow_redirects=False
        ),
    )
    return http, requests


def search(value=None, **kwargs):
    http, requests = transport(value, **kwargs)
    result = leaf.search(
        http,
        ENDPOINT,
        "Python & 编程?",
        cancelled=Event(),
        deadline=time.monotonic() + 5,
        clock=time.monotonic,
    )
    return result, requests


def backend(value=None, **kwargs):
    http, requests = transport(value, **kwargs)
    page_http, page_requests = transport(
        raw=(
            "<html><title>Page</title><article><p>"
            + "Readable public Python article. " * 40
            + "</p></article></html>"
        ).encode(),
        content_type="text/html",
    )
    instance = web.SearxngPublicWebBackend(
        search_http=http,
        search_endpoint=ENDPOINT,
        http=page_http,
        clock=time.monotonic,
        retrieved_clock=lambda: "2026-10-12T13:00:00Z",
        trust_proxy_dns=True,
    )
    cancelled, deadline = Event(), time.monotonic() + 5
    instance.connect(cancelled=cancelled, deadline=deadline)
    return instance, requests, page_requests, cancelled, deadline


def invoke(instance, cancelled, deadline, body=None):
    return instance.search(
        body or {"query": "Python"},
        request_id="public-test",
        binding_id="a" * 64,
        cancelled=cancelled,
        deadline=deadline,
    ).checked()


def test_bound_query_is_one_encoded_json_get():
    result, requests = search()
    assert len(requests) == 1
    request = requests[0]
    assert request.method == "GET"
    assert str(request.url.copy_with(query=None)) == ENDPOINT
    assert dict(request.url.params) == {"q": "Python & 编程?", "format": "json"}
    assert "authorization" not in request.headers
    assert result["results"][0]["content"] == "Python official documentation."
    assert result["results"][0]["published_at"] is None


def test_partial_engine_failure_keeps_usable_original_and_v8_projection():
    instance, requests, _, cancelled, deadline = backend(payload(errors=[["engine-x", "timeout"]]))
    evidence = invoke(instance, cancelled, deadline)
    assert evidence.data["status"] == "partial"
    assert evidence.data["coverage"] == "partial"
    assert evidence.data["failure"] is None
    assert evidence.data["usage"]["raw"]["unresponsive_engines"] == [["engine-x", "timeout"]]
    projected = ExternalOutcome(evidence, True, None, None, "response_received").public_evidence
    validate_public_capture(projected)
    assert projected["status"] == "partial"
    assert len(projected["references"]) == 1
    ref = projected["references"][0]
    assert ref["content_sha256"] == sha256(ref["text"].encode()).hexdigest()
    assert ref["source_range"] == {
        "unit": "unicode_codepoints",
        "start": 0,
        "end": len(ref["text"]),
    }
    assert len(requests) == 1


def test_no_results_with_engine_failures_is_unavailable_not_empty():
    instance, _, _, cancelled, deadline = backend(
        payload(results=[], errors=[["engine-x", "blocked"]])
    )
    evidence = invoke(instance, cancelled, deadline).data
    assert evidence["status"] == "failed"
    assert evidence["failure"] == "searxng_engines_unavailable"
    assert evidence["references"] == []
    assert evidence["usage_status"] == "unknown"


def test_genuine_empty_result_is_empty_only_when_engines_report_no_failure():
    instance, _, _, cancelled, deadline = backend(payload(results=[]))
    evidence = invoke(instance, cancelled, deadline).data
    assert evidence["status"] == "empty"
    assert evidence["failure"] is None


@pytest.mark.parametrize(
    "status,content_type,raw",
    [
        (403, "application/json", b'{"results":[]}'),
        (302, "text/html", b"<html>redirect</html>"),
        (200, "text/html", b"<html>captcha</html>"),
        (200, "application/json", b"<html>challenge</html>"),
        (200, "application/json", b'{"message":"challenge"}'),
        (200, "application/json", b'{"results":[],"unresponsive_engines":"bad"}'),
        (200, "application/json", b'{"results":[null],"unresponsive_engines":[]}'),
    ],
)
def test_http_html_challenge_and_malformed_results_are_failures(status, content_type, raw):
    instance, requests, _, cancelled, deadline = backend(
        raw=raw, status=status, content_type=content_type
    )
    evidence = invoke(instance, cancelled, deadline).data
    assert evidence["status"] == "failed"
    assert evidence["failure"].startswith("searxng_")
    assert evidence["references"] == []
    assert len(requests) == 1


def test_known_published_date_is_preserved_and_text_is_byte_bounded():
    instance, _, _, cancelled, deadline = backend(
        payload(
            results=[
                {
                    "url": "https://www.python.org/",
                    "title": "Python",
                    "content": "中" * 3000,
                    "publishedDate": "2026-10-11T12:30:00+00:00",
                }
            ]
        )
    )
    evidence = invoke(instance, cancelled, deadline).data
    ref = evidence["references"][0]
    assert ref["published_at"] == "2026-10-11T12:30:00+00:00"
    assert len(ref["text"].encode()) <= 2400
    assert ref["content_kind"] == "snippet"
    assert len(json.dumps(evidence, ensure_ascii=False).encode()) < 16000


def test_results_without_captured_content_are_not_silent_empty():
    instance, _, _, cancelled, deadline = backend(
        payload(results=[{"url": "https://www.python.org/", "title": "Python"}])
    )
    evidence = invoke(instance, cancelled, deadline).data
    assert evidence["status"] == "failed"
    assert evidence["failure"] == "searxng_no_captured_content"


def test_result_credentials_still_rejected_by_existing_source_admission():
    instance, _, _, cancelled, deadline = backend(
        payload(
            results=[
                {"url": "https://user:password@example.com/", "title": "bad", "content": "bad"}
            ]
        )
    )
    with pytest.raises(ValueError, match="Credential-free"):
        invoke(instance, cancelled, deadline)


def test_extract_reuses_original_backend_and_never_calls_searxng():
    instance, requests, page_requests, cancelled, deadline = backend()
    evidence = invoke(instance, cancelled, deadline, {"url": "https://www.python.org/about/"}).data
    assert evidence["status"] == "success"
    assert evidence["usage"]["raw"]["provider"] == "pi-http-readability"
    assert requests == []
    assert len(page_requests) == 1


def test_connected_backend_allows_only_one_dispatch():
    instance, requests, _, cancelled, deadline = backend()
    invoke(instance, cancelled, deadline)
    with pytest.raises(ValueError, match="One connected"):
        invoke(instance, cancelled, deadline)
    assert len(requests) == 1


def test_query_body_cannot_override_configured_endpoint():
    instance, requests, _, cancelled, deadline = backend()
    with pytest.raises(ValueError):
        invoke(
            instance, cancelled, deadline, {"query": "Python", "endpoint": "http://127.0.0.1:22/"}
        )
    assert requests == []


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://example.com/search",
        "file:///search",
        "https://u:p@example.com/search",
        "https://example.com/search?format=html",
        "https://example.com/search#fragment",
        "https://example.com/other",
        "https://example.com\\@localhost/search",
    ],
)
def test_static_endpoint_rejects_unsafe_or_ambiguous_config(endpoint):
    with pytest.raises(ValueError):
        leaf.validate_endpoint(endpoint)


@pytest.mark.parametrize(
    "endpoint",
    [
        ENDPOINT,
        "http://localhost:8888/search",
        "http://[::1]:8888/search",
        "https://search.example.com/search",
    ],
)
def test_static_endpoint_accepts_bound_loopback_http_or_remote_https_without_io(endpoint):
    assert leaf.validate_endpoint(endpoint) == endpoint


def test_capability_assembly_is_static_and_role_endpoint_version_are_bound(monkeypatch):
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("Startup created a network client")),
    )
    kwargs = dict(proxy=None, searxng_endpoint=ENDPOINT, searxng_version=leaf.VERSION)
    cap = web.build_public_web_capability("a" * 64, **kwargs)
    assert cap.supports_extract
    assert cap.binding.configuration.credential_version == "no-service-key"
    assert cap.allowance.cost_microusd == 0
    assert (
        cap.binding.identity != web.build_public_web_capability("b" * 64, **kwargs).binding.identity
    )
    assert (
        cap.binding.identity
        != web.build_public_web_capability(
            "a" * 64, **{**kwargs, "searxng_endpoint": "http://127.0.0.1:8889/search"}
        ).binding.identity
    )
    assert (
        cap.binding.identity
        != web.build_public_web_capability(
            "a" * 64, **{**kwargs, "searxng_version": "synthetic-other-version"}
        ).binding.identity
    )
    assert (
        web.build_public_web_capability("a" * 64, proxy=None).binding.configuration.route_id
        == "pi-keyless-web"
    )


def test_pre_cancel_and_deadline_do_not_send():
    http, requests = transport()
    cancelled = Event()
    cancelled.set()
    with pytest.raises(InterruptedError):
        leaf.search(
            http,
            ENDPOINT,
            "Python",
            cancelled=cancelled,
            deadline=time.monotonic() + 5,
            clock=time.monotonic,
        )
    with pytest.raises(TimeoutError):
        leaf.search(
            http,
            ENDPOINT,
            "Python",
            cancelled=Event(),
            deadline=time.monotonic() - 1,
            clock=time.monotonic,
        )
    assert requests == []


@pytest.mark.parametrize("cancel", [False, True])
def test_stalled_first_byte_is_cancelled_or_deadlined_without_retry(cancel):
    requests = []
    cancelled = Event()

    async def stalled(request):
        requests.append(request)
        await asyncio.sleep(5)

    http = HttpxStreamingTransport(
        clock=time.monotonic,
        client_factory=lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(stalled), trust_env=False
        ),
    )
    timer = Timer(0.05, cancelled.set) if cancel else None
    if timer:
        timer.start()
    try:
        with pytest.raises(InterruptedError if cancel else TimeoutError):
            leaf.search(
                http,
                ENDPOINT,
                "Python",
                cancelled=cancelled,
                deadline=time.monotonic() + (1 if cancel else 0.05),
                clock=time.monotonic,
            )
    finally:
        if timer:
            timer.cancel()
            timer.join()
    assert len(requests) == 1


def test_response_bytes_exceeding_host_limit_are_failed_once():
    instance, requests, _, cancelled, deadline = backend(raw=b"x" * (leaf.MAX_RESPONSE_BYTES + 1))
    evidence = invoke(instance, cancelled, deadline).data
    assert evidence["status"] == "failed"
    assert evidence["failure"] == "searxng_response_too_large"
    assert len(requests) == 1
