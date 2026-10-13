import json
import time
from threading import Event

import httpx
import pytest


@pytest.mark.parametrize('sse', [False, True])
def test_exa_business_rate_limit_is_failure_even_with_http_200(sse):
    from virtual_companion.providers.pi_exa import ExaMcpError, parse_response

    value = {'jsonrpc': '2.0', 'id': 1, 'result': {
        '_meta': {'ai.exa/rateLimited': True},
        'content': [{'type': 'text', 'text': 'Provider limit notice, not a result.'}],
    }}
    raw = json.dumps(value)
    if sse:
        raw = 'event: message\ndata: ' + raw + '\n\n'
    with pytest.raises(ExaMcpError, match='^exa_rate_limited$'):
        parse_response(raw)


@pytest.mark.parametrize("content", ["Readable Python text", None])
def test_exa_port_preserves_pi_null_highlight_normalization(content):
    from virtual_companion.providers.pi_exa import parse_response

    response = {"result": {"content": [{"type": "text", "text": json.dumps({
        "results": [{"title": "Python", "url": "https://www.python.org/",
                     "text": content, "highlights": None}],
    })}]}}
    result = parse_response(json.dumps(response))
    assert result["results"][0]["url"] == "https://www.python.org/"
    assert bool(result.get("inlineContent")) == bool(content)


def test_default_web_capability_requires_no_service_key_or_startup_io(monkeypatch):
    from virtual_companion.text.public_web import build_public_web_capability

    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("EXA_API_KEY", raising=False)
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("Network client created at startup")),
    )
    capability = build_public_web_capability("a" * 64, proxy=None)
    assert capability.supports_extract
    assert capability.binding.supports_extract
    assert capability.allowance.cost_microusd == 0
    assert capability.binding.configuration.credential_version == "no-service-key"


def test_reused_search_and_page_reader_become_bounded_public_evidence():
    from virtual_companion.providers.public_search_ddg import HttpResponse
    from virtual_companion.text.public_web import PiPublicWebBackend

    page = "A readable public article about Python. " * 40
    rpc = {
        "result": {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "results": [
                                {
                                    "title": "Python",
                                    "url": "https://www.python.org/about/",
                                    "text": page,
                                }
                            ],
                        }
                    ),
                }
            ]
        }
    }

    class HTTP:
        def post(self, url, **kwargs):
            assert "Authorization" not in kwargs["headers"]
            assert not kwargs["follow_redirects"]
            return HttpResponse(200, {}, json.dumps(rpc).encode())

        def get(self, url, **kwargs):
            return HttpResponse(
                200,
                {"content-type": "text/html"},
                (
                    "<html><title>Python</title><article><p>" + page + "</p></article></html>"
                ).encode(),
            )

    for body in ({"query": "Python official website"}, {"url": "https://www.python.org/about/"}):
        backend = PiPublicWebBackend(
            http=HTTP(),
            clock=time.monotonic,
            retrieved_clock=lambda: "2026-10-12T13:00:00Z",
            trust_proxy_dns=True,
        )
        cancelled, deadline = Event(), time.monotonic() + 5
        backend.connect(cancelled=cancelled, deadline=deadline)
        evidence = (
            backend.search(
                body,
                request_id="public-test",
                binding_id="a" * 64,
                cancelled=cancelled,
                deadline=deadline,
            )
            .checked()
            .data
        )
        assert evidence["status"] == "success"
        assert len(evidence["references"]) == 1
        assert evidence["references"][0]["content_kind"] == "excerpt"
        assert "Python" in evidence["references"][0]["text"]
        assert evidence["usage"]["tokens"] is None
        backend.close()


def test_host_post_reuses_cancellable_bounded_transport():
    from virtual_companion.providers.public_search_ddg import HttpxStreamingTransport

    requests = []

    def respond(request):
        requests.append((request.method, request.content))
        return httpx.Response(200, stream=httpx.ByteStream(b'{"result":{}}'))

    transport = HttpxStreamingTransport(
        clock=time.monotonic,
        client_factory=lambda: httpx.AsyncClient(
            transport=httpx.MockTransport(respond), trust_env=False
        ),
    )
    response = transport.post(
        "https://mcp.exa.ai/mcp",
        headers={},
        body=b"{}",
        cancelled=Event(),
        deadline=time.monotonic() + 5,
        follow_redirects=False,
    )
    assert response.status_code == 200
    assert requests == [("POST", b"{}")]
