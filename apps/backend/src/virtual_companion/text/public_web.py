"""Compose reused keyless Pi web tools into the existing public-evidence Port."""

import time
from datetime import UTC, datetime
from hashlib import sha256
from threading import BoundedSemaphore
from urllib.request import getproxies

from virtual_companion.providers.external_information import (
    ApprovedPublicQuery,
    ExternalConfiguration,
    ExternalEvidence,
    ExternalReservation,
    bind_external,
    canonical,
)
from virtual_companion.providers.pi_exa import ExaMcpError, search
from virtual_companion.providers.pi_web_fetch import check, fetch_page
from virtual_companion.providers.public_search_ddg import HttpxStreamingTransport
from virtual_companion.providers.searxng_search import (
    SearxngError,
    validate_endpoint,
)
from virtual_companion.providers.searxng_search import (
    search as searxng_search,
)

from .public_query import PublicQueryCapability


class PiPublicWebBackend:
    """One search or one page read; the existing host owns turn admission."""

    def __init__(self, *, http, clock, retrieved_clock, trust_proxy_dns=False):
        self.http, self.clock, self.retrieved_clock = http, clock, retrieved_clock
        self.trust_proxy_dns = trust_proxy_dns
        self.connected, self.used = False, False

    def connect(self, *, cancelled, deadline):
        check(cancelled, self.clock, deadline)
        self.connected = True

    def close(self):
        self.connected = False

    def search(self, body, *, request_id, binding_id, cancelled, deadline):
        check(cancelled, self.clock, deadline)
        if not self.connected or self.used:
            raise ValueError("One connected public web operation required")
        self.used = True
        references, failure, metadata = [], None, {}
        try:
            if set(body) == {"query"}:
                metadata["provider"] = "pi-keyless-exa"
                result = search(
                    self.http, body["query"], cancelled=cancelled, deadline=deadline, num_results=3
                )
                content = {
                    row["url"]: row["content"]
                    for row in (result or {}).get("inlineContent", [])
                    if row.get("error") is None and isinstance(row.get("content"), str)
                }
                for row in (result or {}).get("results", [])[:3]:
                    text = content.get(row["url"], row.get("snippet", ""))
                    text = text.encode("utf-8")[:2400].decode("utf-8", errors="ignore")
                    if text.strip():
                        references.append(self._reference(row["url"], row["title"], text))
            elif set(body) == {"url"}:
                metadata["provider"] = "pi-http-readability"
                page = fetch_page(
                    body["url"],
                    http=self.http,
                    cancelled=cancelled,
                    deadline=deadline,
                    clock=self.clock,
                    trust_proxy_dns=self.trust_proxy_dns,
                )
                metadata.update(
                    http_status=page.http_status,
                    body_sha256=page.body_sha256,
                    truncated=page.truncated,
                )
                failure = page.error
                if page.text.strip() and failure is None:
                    references.append(self._reference(page.url, page.title, page.text))
            else:
                raise ValueError("Exact public query or URL required")
            check(cancelled, self.clock, deadline)
        except ExaMcpError as error:
            failure = str(error)
        return ExternalEvidence.create(
            request_id=request_id,
            binding_id=binding_id,
            status="failed" if failure else "success" if references else "empty",
            failure=failure,
            coverage="partial" if references else "none",
            retrieved_at=self.retrieved_clock(),
            references=references,
            native_events=[],
            citation_map=[],
            summary="",
            usage_status="unknown",
            usage={"tokens": None, "search": None, "cost_microusd": None, "raw": metadata},
        )

    @staticmethod
    def _reference(url, title, text):
        return {
            "source_id": "pi-web-" + sha256(canonical([url, text]).encode()).hexdigest()[:24],
            "url": url,
            "title": title[:300],
            "published_at": None,
            "content_kind": "excerpt",
            "text": text,
            "source_range": {"unit": "unicode_codepoints", "start": 0, "end": len(text)},
        }


class SearxngPublicWebBackend(PiPublicWebBackend):
    """Replace only search; original HTTP/Readability extraction stays inherited."""

    def __init__(self, *, search_http, search_endpoint, **kwargs):
        super().__init__(**kwargs)
        self.search_http = search_http
        self.search_endpoint = validate_endpoint(search_endpoint)

    def search(self, body, *, request_id, binding_id, cancelled, deadline):
        if set(body) == {"url"}:
            return super().search(body, request_id=request_id, binding_id=binding_id,
                                  cancelled=cancelled, deadline=deadline)
        if set(body) != {"query"}:
            raise ValueError("Exact public query or URL required")
        check(cancelled, self.clock, deadline)
        if not self.connected or self.used:
            raise ValueError("One connected public web operation required")
        self.used = True
        references, failure, status = [], None, "empty"
        metadata = {"provider": "searxng-json"}
        try:
            result = searxng_search(self.search_http, self.search_endpoint, body["query"],
                                   cancelled=cancelled, deadline=deadline, clock=self.clock)
            metadata.update({key: value for key, value in result.items() if key != "results"})
            references = [{**self._reference(row["url"], row["title"], row["content"]),
                           "content_kind": "snippet", "published_at": row["published_at"]}
                          for row in result["results"]]
            if references:
                status = ("partial" if result["unresponsive_engines"]
                          or result["uncaptured_results"] else "success")
            elif result["unresponsive_engines"]:
                status, failure = "failed", "searxng_engines_unavailable"
            elif result["returned_results"]:
                status, failure = "failed", "searxng_no_captured_content"
        except SearxngError as error:
            status, failure = "failed", str(error)
            metadata.update(error.metadata)
        check(cancelled, self.clock, deadline)
        return ExternalEvidence.create(
            request_id=request_id, binding_id=binding_id, status=status, failure=failure,
            coverage="partial" if references else "none", retrieved_at=self.retrieved_clock(),
            references=references, native_events=[], citation_map=[], summary="",
            usage_status="unknown",
            usage={"tokens": None, "search": None, "cost_microusd": None, "raw": metadata})


class _KeylessRequestAccounting:
    """No paid account: retain request settlement in its existing call receipt."""

    def reserve(self, **request):
        if request["allowance"] != ExternalReservation(1, 0, None, None):
            raise ValueError("Keyless web requests cannot reserve a paid model or service")
        return dict(request)

    def finish(self, receipt, **outcome):
        receipt.update(outcome)


def build_public_web_capability(character_sha256, *, proxy=...,
                                searxng_endpoint=None, searxng_version=None):
    """Static startup assembly; connection happens only on a selected Tool call.

    Uses the operating system's existing proxy settings, never another API key.
    No client, DNS request or provider probe is created during assembly.
    """
    if proxy is ...:
        proxies = getproxies()
        proxy = proxies.get("https") or proxies.get("http")
    if searxng_endpoint is not None:
        searxng_endpoint = validate_endpoint(searxng_endpoint)
        if (not isinstance(searxng_version, str) or not 1 <= len(searxng_version) <= 64
                or any(not (char.isascii() and (char.isalnum() or char in ".-_"))
                       for char in searxng_version)):
            raise ValueError("Explicit fixed SearXNG service version required")
        route_id, tool_version = "searxng-public-web", "searxng-json-" + searxng_version + ".v1"
        policy_data = ["searxng-web.v1", searxng_endpoint, searxng_version,
                       "search-direct-no-proxy", proxy, 262144, 3, 2400, 16000]
    else:
        if searxng_version is not None:
            raise ValueError("SearXNG version requires an explicit endpoint")
        route_id, tool_version = "pi-keyless-web", "pi-web-access-0.39.0.v1"
        policy_data = ["pi-keyless-web.v1", proxy, 3, 16000]
    policy = sha256(canonical(policy_data).encode()).hexdigest()
    binding = bind_external(
        ExternalConfiguration(
            route_id,
            1,
            "http",
            tool_version,
            "no-service-key",
            policy,
            True,
            True,
            2048,
            16000,
        ),
        character_sha256,
        supports_extract=True,
    )

    def approve(bound, value, *, operation="search_public"):
        if bound != binding:
            raise PermissionError("external_authorization_revoked")
        return ApprovedPublicQuery(value, binding.identity, "public-tool", operation=operation)

    def authorized(bound, query):
        return bound == binding and query is not None and query.authorization_id == binding.identity

    def backend(bound):
        if bound != binding:
            raise PermissionError("external_authorization_revoked")

        def client():
            import httpx

            return httpx.AsyncClient(proxy=proxy, trust_env=False, follow_redirects=False)

        common = dict(
            http=HttpxStreamingTransport(clock=time.monotonic, client_factory=client),
            clock=time.monotonic,
            retrieved_clock=lambda: datetime.now(UTC).isoformat(),
            trust_proxy_dns=bool(proxy),
        )
        if searxng_endpoint is not None:
            return SearxngPublicWebBackend(
                search_http=HttpxStreamingTransport(clock=time.monotonic),
                search_endpoint=searxng_endpoint, **common)
        return PiPublicWebBackend(**common)

    return PublicQueryCapability(
        binding,
        approve,
        authorized,
        _KeylessRequestAccounting(),
        ExternalReservation(1, 0, None, None),
        BoundedSemaphore(1),
        backend,
        time.monotonic,
        lambda: datetime.now(UTC).isoformat(),
        supports_extract=True,
    )
