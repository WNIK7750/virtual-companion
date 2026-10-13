"""Pi HTTP/Readability chain adapted to Python; no hosted reader or model API.

HTTP/result boundaries are adapted from nicobailon/pi-web-access 0.39.0,
8e812826269624406458531f9231b679547f5a3a / extract.ts and ssrf-protection.ts
(MIT, Copyright 2025 Nico Bailon). Actual正文 extraction delegates unchanged
to readability-lxml==0.8.4.1 (Apache-2.0), not the Node @mozilla/readability.

The host provides its existing bounded cancellable HTTP Port. DNS/IP validation
mirrors Pi preflight; neither that upstream path nor this leaf pins HTTP DNS to
the checked addresses. Redirects are not followed. No credentials, cookies,
local files, JavaScript execution, fallback provider, retry or extra model.
"""

import ipaddress
import math
import socket
from dataclasses import dataclass
from hashlib import sha256
from urllib.parse import urlsplit

VERSION = "pi-http-readability-python.v1"
USER_AGENT = "OpenAI File Downloader, XaiImageApiFetch/1.0"


def public_addresses(hostname):
    return {item[4][0] for item in socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)}


def validate_public_url(url, *, lookup=public_addresses, trust_proxy_dns=False):
    """Pi validateRemoteUrl subset; explicit host proxy may resolve public names."""
    if not isinstance(url, str) or any(
        char.isspace() or ord(char) < 32 or char == "\\" for char in url
    ):
        raise ValueError("public_web_invalid_url")
    try:
        parsed = urlsplit(url)
        hostname, port = parsed.hostname, parsed.port
    except ValueError:
        raise ValueError("public_web_invalid_url") from None
    if (
        parsed.scheme not in {"http", "https"}
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or (port is not None and not 0 < port <= 65535)
    ):
        raise ValueError("public_web_invalid_url")
    hostname = hostname.rstrip(".").lower()
    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise PermissionError("public_web_internal_address")
    try:
        addresses = {ipaddress.ip_address(hostname)}
    except ValueError:
        if "." not in hostname or "%" in hostname or hostname.endswith((".local", ".internal")):
            raise PermissionError("public_web_internal_address") from None
        # WHATWG URL in Pi canonicalizes legacy integer/short IPv4 spellings;
        # Python urlsplit does not. Reject those before trusted proxy DNS.
        try:
            socket.inet_aton(hostname)
        except OSError:
            pass
        else:
            raise PermissionError("public_web_nonstandard_ip")
        # Matches Pi's explicitly configured proxy-DNS path. It does not exempt
        # literal private IPs, localhost, credentials or grant arbitrary proxies.
        if trust_proxy_dns:
            return url
        addresses = {ipaddress.ip_address(value) for value in lookup(hostname)}
    if not addresses or any(not value.is_global or value.is_multicast for value in addresses):
        raise PermissionError("public_web_internal_address")
    return url


def check(cancelled, clock, deadline):
    if cancelled.is_set():
        raise InterruptedError("external_cancelled")
    if not math.isfinite(deadline) or clock() >= deadline:
        raise TimeoutError("external_deadline")


@dataclass(frozen=True)
class PageCapture:
    url: str
    title: str
    text: str
    content_kind: str
    mime_type: str
    http_status: int
    body_sha256: str
    truncated: bool = False
    error: str | None = None
    source_range: dict | None = None


def extract_html(body, *, url):
    """Delegate article selection; .summary() is extracted HTML, not LLM summary."""
    from lxml import html
    from readability import Document

    # The pinned library explicitly supports HtmlElement. Its bytes encoding
    # path mixes str regular expressions and bytes; do not patch the dependency.
    document = Document(html.document_fromstring(body), url=url)
    title = document.short_title() or document.title()
    extracted = html.fromstring(document.summary(html_partial=True))
    blocks = extracted.xpath(".//p|.//h1|.//h2|.//h3|.//li|.//pre|.//blockquote")
    text = "\n\n".join(" ".join(block.text_content().split()) for block in blocks)
    return title, text.strip() or extracted.text_content().strip()


def fetch_page(
    url,
    *,
    http,
    cancelled,
    deadline,
    clock,
    lookup=public_addresses,
    max_body_bytes=1_000_000,
    max_text_bytes=6_000,
    trust_proxy_dns=False,
):
    check(cancelled, clock, deadline)
    validate_public_url(url, lookup=lookup, trust_proxy_dns=trust_proxy_dns)
    check(cancelled, clock, deadline)
    response = http.get(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,text/markdown,text/plain;q=0.9",
            "Accept-Encoding": "identity",
        },
        cancelled=cancelled,
        deadline=deadline,
        max_body_bytes=max_body_bytes,
    )
    check(cancelled, clock, deadline)
    mime = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    digest = sha256(response.body).hexdigest()
    if response.status_code != 200:
        return PageCapture(
            url,
            "",
            "",
            "excerpt",
            mime,
            response.status_code,
            digest,
            error="public_web_http_" + str(response.status_code),
        )
    if mime in {"text/html", "application/xhtml+xml"}:
        title, text = extract_html(response.body, url=url)
    elif mime in {"text/plain", "text/markdown"}:
        title, text = "", response.body.decode("utf-8", errors="replace").strip()
    else:
        return PageCapture(
            url,
            "",
            "",
            "excerpt",
            mime,
            response.status_code,
            digest,
            error="public_web_unsupported_content_type",
        )
    check(cancelled, clock, deadline)
    captured = text.encode("utf-8")[:max_text_bytes].decode("utf-8", errors="ignore")
    # Range is relative to the deterministic extracted text, not HTML bytes/DOM
    # source offsets. The host retains its existing 16KB whole-attachment check.
    source_range = (
        {"unit": "unicode_codepoints", "start": 0, "end": len(captured)} if captured else None
    )
    return PageCapture(
        url,
        title,
        captured,
        "excerpt",
        mime,
        response.status_code,
        digest,
        truncated=captured != text,
        error=None if text else "public_web_no_readable_content",
        source_range=source_range,
    )
