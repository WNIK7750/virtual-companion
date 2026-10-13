"""Pi Tavily basic port: one authorized POST, captured snippets, no discovery.

Adapted from nicobailon/pi-web-access tavily.ts at
8e812826269624406458531f9231b679547f5a3a (MIT, see THIRD_PARTY_NOTICES.md).
The host owns authorization, reservation, settlement and consumer admission.
This module contains no HTTP implementation, SDK, environment or file key lookup.
"""

import json
import math
import re
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from typing import Protocol
from urllib.parse import quote, quote_plus, unquote, unquote_plus

from .external_information import ExternalEvidence, canonical

ENDPOINT = "https://api.tavily.com/search"


class HttpPostPort(Protocol):
    """One POST; enforce deadline/cancel during I/O and never follow redirects.

    Implementations must not log credentials or unsafe response/error bytes.
    Header/body ownership ends on return; there is no adapter retry or fallback.
    """

    def post(
        self, url, *, headers, body, cancelled, deadline, follow_redirects: bool
    ) -> "HttpResponse": ...


@dataclass(frozen=True)
class HttpResponse:
    status_code: int
    body: bytes


class TavilyBasicBackend:
    def __init__(
        self,
        *,
        http_post,
        credential_resolver,
        clock,
        retrieved_clock,
        max_results=5,
        response_bytes=256000,
    ):
        if (
            type(max_results) is not int
            or not 1 <= max_results <= 5
            or type(response_bytes) is not int
            or not 1 <= response_bytes <= 1_000_000
        ):
            raise ValueError("Bounded basic result count and response capacity required")
        self.http_post, self.credential_resolver = http_post, credential_resolver
        self.clock, self.retrieved_clock = clock, retrieved_clock
        self.max_results, self.response_bytes = max_results, response_bytes
        self._secret, self._connected, self._used = None, False, False
        self._audit = {
            "provider": "tavily",
            "endpoint": ENDPOINT,
            "request_body": None,
            "http_status": None,
            "provider_response_sha256": None,
            "provider_response_bytes": None,
            "sanitized_response": None,
            "redaction_count": 0,
            "failure": None,
        }

    @property
    def audit_record(self):
        """Controlled export: original hash + sanitized parsed data, never raw bytes."""
        return deepcopy(self._audit)

    def _check(self, cancelled, deadline):
        if cancelled.is_set():
            raise InterruptedError("external_cancelled")
        now = self.clock()
        if type(now) not in {int, float} or not math.isfinite(now) or now >= deadline:
            raise TimeoutError("external_deadline")

    def connect(self, *, cancelled, deadline):
        # ExternalQueryCall calls this only after reserve + current authorization.
        self._check(cancelled, deadline)
        if self._connected or self._used:
            raise ValueError("Single-use Tavily backend required")
        secret = self.credential_resolver()
        self._check(cancelled, deadline)
        if (
            not isinstance(secret, str)
            or not secret
            or any(ord(char) < 32 or ord(char) == 127 for char in secret)
        ):
            self._audit["failure"] = "tavily_credential_unavailable"
            raise PermissionError("tavily_credential_unavailable")
        self._secret, self._connected = secret, True

    def _sanitized(self, value):
        """Remove known credential fields and injected secret echoes before export.

        A redacted response cannot supply captured public references. This is a
        narrow credential boundary, not a general detector of arbitrary secrets.
        """
        variants = sorted(
            {self._secret, quote(self._secret, safe=""), quote_plus(self._secret, safe="")},
            key=len,
            reverse=True,
        )
        credential_keys = {
            "apikey",
            "authorization",
            "accesstoken",
            "token",
            "password",
            "secret",
            "credential",
            "cookie",
            "setcookie",
        }
        count = 0

        def visit(item):
            nonlocal count
            if isinstance(item, str):
                for variant in variants:
                    if variant in item:
                        count += item.count(variant)
                        item = item.replace(variant, "[REDACTED]")
                return item
            if isinstance(item, list):
                return [visit(row) for row in item]
            if isinstance(item, dict):
                result = {}
                for key, row in item.items():
                    if re.sub("[-_]", "", key).lower() in credential_keys:
                        count += 1
                        result[visit(key)] = "[REDACTED]"
                    else:
                        result[visit(key)] = visit(row)
                return result
            return item

        result = visit(value)
        return result, count

    def _usage(self, document):
        reported = document.get("usage") if isinstance(document, dict) else None
        credits = reported.get("credits") if isinstance(reported, dict) else None
        # The common contract accepts reported integer units. No basic=1 guess
        # or money/token conversion when the response omits/malforms this field.
        search = (
            {"unit": "tavily_api_credit", "value": credits}
            if type(credits) is int and credits >= 0
            else None
        )
        return ("partial" if search is not None else "unknown"), {
            "tokens": None,
            "search": search,
            "cost_microusd": None,
            "raw": {"provider": "tavily", "reported_usage": reported, "audit": self.audit_record},
        }

    def _evidence(self, request_id, binding_id, document, *, references=(), failure=None):
        usage_status, usage = self._usage(document)
        status = "failed" if failure else "success" if references else "empty"
        return ExternalEvidence.create(
            request_id=request_id,
            binding_id=binding_id,
            status=status,
            failure=failure,
            coverage="returned_result_set" if references else "none",
            retrieved_at=self.retrieved_clock(),
            references=references,
            native_events=[],
            citation_map=[],
            summary="",
            usage_status=usage_status,
            usage=usage,
        )

    def _failed(self, request_id, binding_id, document, failure):
        self._audit["failure"] = failure
        return self._evidence(request_id, binding_id, document, failure=failure)

    def search(self, body, *, request_id, binding_id, cancelled, deadline):
        self._check(cancelled, deadline)
        if not self._connected or self._used:
            raise ValueError("Connected unconsumed Tavily backend required")
        self._used = True
        if (
            not isinstance(body, dict)
            or set(body) != {"query"}
            or not isinstance(body["query"], str)
            or not body["query"].strip()
        ):
            raise ValueError("Frozen host-approved public query required")
        if any(
            self._secret in value
            for value in (body["query"], unquote(body["query"]), unquote_plus(body["query"]))
        ):
            return self._failed(request_id, binding_id, None, "tavily_sensitive_query")
        request = {
            "query": body["query"],
            "auto_parameters": False,
            "search_depth": "basic",
            "max_results": self.max_results,
            "include_answer": False,
            "include_raw_content": False,
            "include_usage": True,
        }
        self._audit["request_body"] = request
        try:
            response = self.http_post.post(
                ENDPOINT,
                headers={
                    "Authorization": "Bearer " + self._secret,
                    "Content-Type": "application/json",
                },
                body=canonical(request).encode("utf-8"),
                cancelled=cancelled,
                deadline=deadline,
                follow_redirects=False,
            )
        except (InterruptedError, TimeoutError):
            raise
        except Exception:
            # Never copy a transport exception message into evidence or logs.
            self._audit["failure"] = "tavily_transport_failure"
            # No provider response was received; the common call retains its
            # conservative may_have_been_sent state and unknown reservation.
            raise RuntimeError("tavily_transport_failure") from None
        if (
            not isinstance(response, HttpResponse)
            or type(response.status_code) is not int
            or not 100 <= response.status_code <= 599
            or not isinstance(response.body, bytes)
        ):
            return self._failed(request_id, binding_id, None, "tavily_http_port_contract")
        self._audit.update(
            http_status=response.status_code,
            provider_response_sha256=sha256(response.body).hexdigest(),
            provider_response_bytes=len(response.body),
        )
        if len(response.body) > self.response_bytes:
            return self._failed(request_id, binding_id, None, "tavily_response_capacity")
        # Capture and meter returned usage even if cancel/deadline occurred in
        # POST. The host settles first, then rejects late delivery independently.
        try:

            def invalid_constant(value):
                raise ValueError("Non-finite JSON number")

            document = json.loads(response.body, parse_constant=invalid_constant)
            sanitized, count = self._sanitized(document)
            self._audit.update(sanitized_response=sanitized, redaction_count=count)
        except (ValueError, UnicodeError, RecursionError):
            return self._failed(
                request_id,
                binding_id,
                None,
                "tavily_http_" + str(response.status_code)
                if response.status_code != 200
                else "tavily_invalid_json",
            )
        if count:
            return self._failed(
                request_id, binding_id, sanitized, "tavily_sensitive_response_redacted"
            )
        if response.status_code != 200:
            return self._failed(
                request_id, binding_id, sanitized, "tavily_http_" + str(response.status_code)
            )
        if not isinstance(sanitized, dict) or not isinstance(sanitized.get("results"), list):
            return self._failed(request_id, binding_id, sanitized, "tavily_response_schema")
        references = []
        for item in sanitized["results"]:
            if not isinstance(item, dict) or not item.get("url"):
                continue
            text = (
                re.sub(r"\s+", " ", item["content"]).strip()
                if isinstance(item.get("content"), str)
                else ""
            )
            title = item.get("title") or "Source " + str(len(references) + 1)
            published = item.get("published_date")
            identity = canonical([len(references), item["url"], text])
            references.append(
                {
                    "source_id": "tavily-" + sha256(identity.encode("utf-8")).hexdigest()[:24],
                    "url": item["url"],
                    "title": title,
                    "published_at": published if isinstance(published, str) and published else None,
                    "content_kind": "snippet",
                    "text": text,
                    "source_range": None,
                }
            )
            if len(references) >= self.max_results:
                break
        try:
            return self._evidence(request_id, binding_id, sanitized, references=references)
        except (ValueError, TypeError):
            # Invalid source URL/type cannot become a useful source; usage and
            # the original provider byte hash remain in failed audit evidence.
            return self._failed(request_id, binding_id, sanitized, "tavily_invalid_result")

    def close(self):
        # Port ownership stays with the host; erase this adapter's retained key.
        self._secret, self._connected = None, False
