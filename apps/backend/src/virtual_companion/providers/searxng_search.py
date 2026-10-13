"""Bounded JSON protocol adapter for one explicitly configured SearXNG service.

Original host adapter; no SearXNG application code is copied or imported.
Protocol checked against SearXNG ad46ebd7d87666edb5f116c5b274b47f689ac6cd.
"""

import ipaddress
import json
from hashlib import sha256
from urllib.parse import urlencode, urlsplit

import httpx

from .pi_web_fetch import check
from .public_search_ddg import HttpResponse, SearchFailure

DEFAULT_ENDPOINT = "http://127.0.0.1:8888/search"
VERSION = "2026.10.11-ad46ebd7d"
MAX_RESPONSE_BYTES = 262144
MAX_RESULTS = 3
MAX_TEXT_BYTES = 2400


class SearxngError(Exception):
    """Fixed safe failure code and bounded audit metadata, never raw error text."""

    def __init__(self, code, metadata=None):
        super().__init__(code)
        self.metadata = metadata or {}


def validate_endpoint(endpoint):
    """Validate trusted static configuration without DNS or client creation.

    Remote services use HTTPS. Plain HTTP is permitted for explicit localhost
    self-hosting only. This is never called on a model-supplied result URL.
    """
    if (
        not isinstance(endpoint, str)
        or not 1 <= len(endpoint) <= 2048
        or any(char.isspace() or ord(char) < 32 or char == "\\" for char in endpoint)
    ):
        raise ValueError("Explicit credential-free SearXNG search endpoint required")
    try:
        parsed = urlsplit(endpoint)
        host, port = parsed.hostname, parsed.port
    except ValueError:
        raise ValueError("Valid SearXNG search authority required") from None
    if (
        parsed.scheme not in {"http", "https"}
        or not host
        or parsed.username is not None
        or parsed.password is not None
        or (port is not None and not 0 < port <= 65535)
        or parsed.query
        or parsed.fragment
        or not parsed.path.endswith("/search")
    ):
        raise ValueError("Exact SearXNG /search endpoint without query or credentials required")
    if parsed.scheme == "http":
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = host.lower() == "localhost"
        if not loopback:
            raise ValueError("Plain HTTP SearXNG requires an explicit loopback host")
    return endpoint


def _parse(response):
    if not isinstance(response, HttpResponse) or not isinstance(response.body, bytes):
        raise SearxngError("searxng_response_invalid")
    metadata = {
        "http_status": response.status_code,
        "body_sha256": sha256(response.body).hexdigest(),
    }
    if not 200 <= response.status_code < 300:
        raise SearxngError("searxng_http_rejected", metadata)
    if len(response.body) > MAX_RESPONSE_BYTES:
        raise SearxngError("searxng_response_too_large", metadata)
    content_type = {key.lower(): value for key, value in response.headers.items()}.get(
        "content-type", ""
    )
    if content_type.split(";", 1)[0].strip().lower() != "application/json":
        raise SearxngError("searxng_json_required", metadata)
    try:
        value = json.loads(response.body.decode("utf-8"))
    except (ValueError, UnicodeError, RecursionError):
        raise SearxngError("searxng_response_invalid", metadata) from None
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("results"), list)
        or not isinstance(value.get("unresponsive_engines"), list)
    ):
        raise SearxngError("searxng_response_invalid", metadata)
    errors = value["unresponsive_engines"]
    if any(
        not isinstance(row, list)
        or len(row) != 2
        or any(not isinstance(item, str) or not item for item in row)
        for row in errors
    ):
        raise SearxngError("searxng_response_invalid", metadata)
    results, uncaptured = [], 0
    for index, row in enumerate(value["results"]):
        if (
            not isinstance(row, dict)
            or any(
                not isinstance(row.get(key), str) or not row[key].strip()
                for key in ("url", "title")
            )
            or (row.get("content") is not None and not isinstance(row["content"], str))
            or (
                row.get("publishedDate") is not None
                and (not isinstance(row["publishedDate"], str) or not row["publishedDate"].strip())
            )
        ):
            raise SearxngError("searxng_response_invalid", metadata)
        if index >= MAX_RESULTS:
            continue
        text = row.get("content") or ""
        text = text.encode("utf-8")[:MAX_TEXT_BYTES].decode("utf-8", errors="ignore")
        if not text.strip():
            uncaptured += 1
            continue
        results.append(
            {
                "url": row["url"],
                "title": row["title"][:300],
                "content": text,
                "published_at": row.get("publishedDate"),
            }
        )
    return {
        "results": results,
        "unresponsive_engines": errors,
        "uncaptured_results": uncaptured,
        "returned_results": len(value["results"]),
        **metadata,
    }


def search(http, endpoint, query, *, cancelled, deadline, clock):
    """One GET to the bound endpoint; no redirects, retries or provider fallback."""
    endpoint = validate_endpoint(endpoint)
    if not isinstance(query, str) or not query.strip() or len(query.encode("utf-8")) > 2048:
        raise ValueError("Host-approved bounded public query required")
    check(cancelled, clock, deadline)
    try:
        response = http.get(
            endpoint + "?" + urlencode({"q": query, "format": "json"}),
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "identity",
                "User-Agent": "VirtualCompanion-Searxng/1.0",
            },
            cancelled=cancelled,
            deadline=deadline,
            max_body_bytes=MAX_RESPONSE_BYTES,
        )
    except SearchFailure as error:
        code = (
            "searxng_response_too_large"
            if str(error) == "ddg_response_too_large"
            else "searxng_transport_failure"
        )
        raise SearxngError(code, error.metadata) from None
    except httpx.TimeoutException:
        raise SearxngError("searxng_timeout") from None
    except httpx.TransportError:
        raise SearxngError("searxng_transport_failure") from None
    check(cancelled, clock, deadline)
    result = _parse(response)
    check(cancelled, clock, deadline)
    return result
