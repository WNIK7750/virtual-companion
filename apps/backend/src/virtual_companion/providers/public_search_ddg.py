"""Bounded Python port of pi-web-access 0.39.0 DuckDuckGo HTML provider.

Copyright (c) 2025 Nico Bailon. MIT; see THIRD_PARTY_NOTICES.md.
Source: 8e812826269624406458531f9231b679547f5a3a / duckduckgo.ts.
No startup I/O, configuration discovery or provider fallback. The host owns
approval, accounting, admission and the absolute monotonic deadline.
"""

import asyncio
import contextlib
import math
import re
from dataclasses import dataclass, field
from hashlib import sha256
from html.parser import HTMLParser
from typing import Protocol
from urllib.parse import parse_qs, urlencode, urljoin, urlsplit

from .external_information import ExternalEvidence, parse_public_action

SEARCH_URL = "https://html.duckduckgo.com/html/"
VERSION = "pi-duckduckgo-html.v1"
USER_AGENT = (
    "Mozilla/5.0 (compatible; pi-web-access/1.0; +https://github.com/nicobailon/pi-web-access)"
)


@dataclass(frozen=True)
class HttpResponse:
    status_code: int
    headers: dict[str, str]
    body: bytes


class HttpPort(Protocol):
    def get(self, url, *, headers, cancelled, deadline, max_body_bytes) -> HttpResponse: ...


class SearchFailure(Exception):
    """Owned failure code plus safe metadata, never raw provider error bodies."""

    def __init__(self, code, metadata=None):
        super().__init__(code)
        self.metadata = metadata or {}


def _guard(cancelled, clock, deadline):
    if cancelled.is_set():
        raise InterruptedError("external_cancelled")
    now = clock()
    if not math.isfinite(now) or not math.isfinite(deadline) or now >= deadline:
        raise TimeoutError("external_deadline")


class HttpxStreamingTransport:
    """Lazy transport, explicitly selected in the synchronous host worker.

    No environment proxy/auth, redirects, cookie persistence, retry or logging.
    The factory is injectable for recordings or explicit host transport. Each
    request owns/closes its client, including cancellation before first bytes.
    """

    def __init__(self, *, clock, client_factory=None):
        self.clock, self.client_factory = clock, client_factory

    def post(self, url, *, headers, body, cancelled, deadline, follow_redirects=False):
        if follow_redirects or not isinstance(body, bytes):
            raise ValueError("Bounded public POST without redirects required")
        return self.get(url, headers={**headers, "Accept-Encoding": "identity"},
                        cancelled=cancelled, deadline=deadline,
                        max_body_bytes=1_000_000, method="POST", body=body)

    def get(self, url, *, headers, cancelled, deadline, max_body_bytes, method="GET", body=None):
        _guard(cancelled, self.clock, deadline)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self._get(
                    url,
                    headers=headers,
                    cancelled=cancelled,
                    deadline=deadline,
                    max_body_bytes=max_body_bytes,
                    method=method,
                    body=body,
                )
            )
        raise SearchFailure("ddg_sync_worker_required")

    async def _get(self, url, *, headers, cancelled, deadline, max_body_bytes,
                   method="GET", body=None):
        async def fetch():
            if self.client_factory is None:
                import httpx

                client = httpx.AsyncClient(trust_env=False, follow_redirects=False)
            else:
                client = self.client_factory()
            async with client:
                _guard(cancelled, self.clock, deadline)
                async with client.stream(
                    method,
                    url,
                    headers=headers,
                    follow_redirects=False,
                    timeout=max(0.001, deadline - self.clock()),
                    **({"content": body} if body is not None else {}),
                ) as response:
                    head = {key.lower(): value for key, value in response.headers.items()}
                    metadata = {"http_status": response.status_code, "body_bytes_seen": 0}
                    if head.get("content-encoding", "identity").lower() not in {"", "identity"}:
                        raise SearchFailure("ddg_unexpected_encoding", metadata)
                    length = head.get("content-length")
                    if length is not None and length.isdecimal() and int(length) > max_body_bytes:
                        raise SearchFailure("ddg_response_too_large", metadata)
                    chunks, size, digest = [], 0, sha256()
                    async for chunk in response.aiter_raw(chunk_size=8192):
                        _guard(cancelled, self.clock, deadline)
                        digest.update(chunk)
                        size += len(chunk)
                        if size > max_body_bytes:
                            raise SearchFailure(
                                "ddg_response_too_large",
                                {
                                    "http_status": response.status_code,
                                    "body_bytes_seen": size,
                                    "body_sha256": digest.hexdigest(),
                                    "body_hash_scope": "observed_prefix_bytes",
                                },
                            )
                        chunks.append(chunk)
                    _guard(cancelled, self.clock, deadline)
                    return HttpResponse(response.status_code, head, b"".join(chunks))

        # Checking only in the iterator leaves a stalled first-byte await alive.
        task = asyncio.create_task(fetch())
        try:
            while not task.done():
                _guard(cancelled, self.clock, deadline)
                await asyncio.wait({task}, timeout=min(0.025, max(0.001, deadline - self.clock())))
            _guard(cancelled, self.clock, deadline)
            return task.result()
        finally:
            if not task.done():
                task.cancel()
            # Retrieve even an already-completed exception when cancellation
            # wins the race, so asyncio never logs a raw provider exception.
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task


@dataclass
class _Node:
    tag: str
    attrs: dict[str, str]
    children: list = field(default_factory=list)

    def has_class(self, name):
        return name in self.attrs.get("class", "").split()

    def walk(self):
        yield self
        for child in self.children:
            if isinstance(child, _Node):
                yield from child.walk()

    def text(self):
        return "".join(
            child.text() if isinstance(child, _Node) else child for child in self.children
        )


class _Document(HTMLParser):
    _void = {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }

    def __init__(self, body):
        super().__init__(convert_charrefs=True)
        self.root = _Node("root", {})
        self.stack = [self.root]
        self.feed(body)
        self.close()

    def handle_starttag(self, tag, attrs):
        node = _Node(tag, {key: value or "" for key, value in attrs})
        self.stack[-1].children.append(node)
        if tag not in self._void:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in self._void:
            self.stack.pop()

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                break

    def handle_data(self, data):
        self.stack[-1].children.append(data)


def _decode_result_url(href):
    # urllib is permissive about ports, whitespace, credentials and escapes.
    # The host evidence contract is stricter than the donor WHATWG URL check.
    if (
        not href
        or any(char.isspace() or ord(char) < 32 for char in href)
        or "\\" in href
        or re.search(r"%(?![0-9a-fA-F]{2})", href)
        or not (href.startswith("//") or href.lower().startswith(("http://", "https://")))
    ):
        return None
    try:
        raw_link = urlsplit(href)
        if not raw_link.netloc:
            return None
        link = urlsplit(urljoin(SEARCH_URL, href))
        query = parse_qs(link.query, keep_blank_values=True, encoding="utf-8", errors="strict")
        destination = query["uddg"][0] if "uddg" in query else link.geturl()
        if link.hostname in {"duckduckgo.com", "html.duckduckgo.com"} and link.path == "/l/":
            if "uddg" not in query:
                return None
        result = urlsplit(destination)
        if (
            result.scheme.lower() not in {"http", "https"}
            or not result.hostname
            or result.username is not None
            or result.password is not None
            or any(char.isspace() or ord(char) < 32 for char in destination)
            or "\\" in destination
            or re.search(r"%(?![0-9a-fA-F]{2})", destination)
            or re.search(r"%(?:0[0-9a-f]|1[0-9a-f]|7f)", destination, re.IGNORECASE)
        ):
            return None
        _ = result.port
        host = result.hostname.encode("idna").decode("ascii")
        if ":" not in host and not re.fullmatch(r"[a-zA-Z0-9.-]+", host):
            return None
        if ":" not in host and any(
            not label or label.startswith("-") or label.endswith("-")
            for label in host.rstrip(".").split(".")
        ):
            return None
        return destination
    except (ValueError, UnicodeError):
        return None


def _normalize_filters(filters):
    allowed, blocked = set(), set()
    for raw in filters:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("Explicit valid static domain filters required")
        value = raw.strip().lower()
        negative = value.startswith("-")
        value = value.lstrip("-")
        normalized = _decode_result_url(value if "://" in value else "https://" + value)
        if normalized is None:
            raise ValueError("Explicit valid static domain filters required")
        host = urlsplit(normalized).hostname.rstrip(".")
        (blocked if negative else allowed).add(host)
    return allowed, blocked


def _matches_domain(url, allowed, blocked):
    host = urlsplit(url).hostname.rstrip(".").lower()

    def matches(domain):
        return host == domain or host.endswith("." + domain)

    return (not allowed or any(matches(domain) for domain in allowed)) and not any(
        matches(domain) for domain in blocked
    )


class PiDuckDuckGoBackend:
    """ExternalBackendPort: one approved query, one bounded HTTP dispatch."""

    def __init__(
        self,
        *,
        http: HttpPort,
        clock,
        retrieved_clock,
        num_results=5,
        domain_filters=(),
        max_body_bytes=524288,
    ):
        if type(max_body_bytes) is not int or not 1024 <= max_body_bytes <= 1_048_576:
            raise ValueError("Explicit response byte cap from 1024 through 1048576 required")
        self.http, self.clock, self.retrieved_clock = http, clock, retrieved_clock
        self.num_results = (
            max(1, min(20, math.floor(num_results)))
            if type(num_results) in {int, float} and math.isfinite(num_results)
            else 5
        )
        self.allowed, self.blocked = _normalize_filters(domain_filters)
        self.max_body_bytes = max_body_bytes
        self.connected, self.closed, self.used = False, False, False

    def connect(self, *, cancelled, deadline):
        _guard(cancelled, self.clock, deadline)
        if self.closed:
            raise ValueError("Closed public search backend")
        self.connected = True

    def close(self):
        self.closed = True

    def search(self, body, *, request_id, binding_id, cancelled, deadline):
        _guard(cancelled, self.clock, deadline)
        if not self.connected or self.closed or self.used:
            raise ValueError("One connected unconsumed public search backend required")
        if not isinstance(body, dict) or set(body) != {"query"}:
            raise ValueError("Host-approved exact query body required")
        query = parse_public_action({"action": "search_public", **body})["query"]
        self.used = True
        metadata = {"adapter": VERSION, "endpoint": SEARCH_URL, "requested_limit": self.num_results}
        try:
            response = self.http.get(
                SEARCH_URL + "?" + urlencode({"q": query}),
                headers={
                    "Accept": "text/html",
                    "User-Agent": USER_AGENT,
                    "Accept-Encoding": "identity",
                },
                cancelled=cancelled,
                deadline=deadline,
                max_body_bytes=self.max_body_bytes,
            )
            _guard(cancelled, self.clock, deadline)
            if not isinstance(response, HttpResponse) or not isinstance(response.body, bytes):
                raise SearchFailure("ddg_transport_protocol")
            metadata.update(
                http_status=response.status_code,
                body_bytes_seen=len(response.body),
                body_sha256=sha256(response.body).hexdigest(),
                body_hash_scope="complete_response_bytes",
            )
            if len(response.body) > self.max_body_bytes:
                raise SearchFailure("ddg_response_too_large")
            if response.status_code == 202:
                raise SearchFailure("ddg_bot_challenge")
            if response.status_code != 200:
                raise SearchFailure("ddg_http_status")
            try:
                document = _Document(response.body.decode("utf-8", errors="strict"))
                references = self._parse(document, metadata, cancelled, deadline)
            except (UnicodeError, RecursionError):
                raise SearchFailure("ddg_invalid_response") from None
            status = "success" if references else "empty"
            return self._evidence(request_id, binding_id, status, None, references, metadata)
        except (InterruptedError, TimeoutError):
            raise
        except SearchFailure as error:
            metadata.update(error.metadata)
            return self._evidence(request_id, binding_id, "failed", str(error), [], metadata)
        except Exception as error:
            # Correlation without persisting provider URLs or secret text.
            metadata["exception_type"] = type(error).__name__
            metadata["exception_sha256"] = sha256(str(error).encode("utf-8")).hexdigest()
            return self._evidence(
                request_id,
                binding_id,
                "failed",
                "ddg_transport_failure",
                [],
                metadata,
            )

    def _parse(self, document, metadata, cancelled, deadline):
        nodes = list(document.root.walk())
        if any(
            node.has_class("anomaly-modal__box") or node.attrs.get("id") == "challenge-form"
            for node in nodes
        ):
            raise SearchFailure("ddg_bot_challenge")
        containers = [node for node in nodes if node.has_class("result")]
        metadata.update(
            result_containers=len(containers),
            parseable_results=0,
            ads_skipped=0,
            invalid_results=0,
            filtered_results=0,
        )
        references = []
        for node in containers:
            _guard(cancelled, self.clock, deadline)
            if node.has_class("result--ad"):
                metadata["ads_skipped"] += 1
                continue
            anchor = next((item for item in node.walk() if item.has_class("result__a")), None)
            title = anchor.text().strip() if anchor is not None else ""
            url = _decode_result_url(anchor.attrs.get("href", "").strip()) if anchor else None
            if not title or url is None:
                metadata["invalid_results"] += 1
                continue
            metadata["parseable_results"] += 1
            if not _matches_domain(url, self.allowed, self.blocked):
                metadata["filtered_results"] += 1
                continue
            snippet = next(
                (item for item in node.walk() if item.has_class("result__snippet")), None
            )
            text = snippet.text().strip() if snippet is not None else ""
            references.append(
                {
                    "source_id": f"ddg-result-{len(references) + 1}",
                    "url": url,
                    "title": title,
                    "published_at": None,
                    "content_kind": "snippet",
                    "text": text,
                    "source_range": None,
                }
            )
            if len(references) >= self.num_results:
                break
        if metadata["parseable_results"] == 0:
            if not containers and any(node.has_class("no-results__message") for node in nodes):
                metadata["empty_reason"] = "provider_no_results"
            else:
                raise SearchFailure("ddg_invalid_response")
        elif not references:
            metadata["empty_reason"] = "domain_filter"
        metadata["returned_results"] = len(references)
        metadata["parsed_scope"] = "up_to_requested_limit"
        _guard(cancelled, self.clock, deadline)
        return references

    def _evidence(self, request_id, binding_id, status, failure, references, metadata):
        # HTTP dispatch counts are observed local units; not billable provider
        # search units. No token count or free monetary cost is inferred.
        observed = "http_status" in metadata
        return ExternalEvidence.create(
            request_id=request_id,
            binding_id=binding_id,
            status=status,
            failure=failure,
            coverage="returned_result_set" if references else "none",
            retrieved_at=self.retrieved_clock(),
            references=references,
            native_events=[metadata],
            citation_map=[],
            summary="",
            usage_status="partial" if observed else "unknown",
            usage={
                "tokens": None,
                "search": {"unit": "http_requests", "value": 1} if observed else None,
                "cost_microusd": None,
                "raw": None,
            },
        )
