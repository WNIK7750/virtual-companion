"""Isolated candidate: public-query Port, not a runtime or an HTTP provider.

The host supplies current authority, query approval, shared atomic accounting,
concurrency and the existing owner/deadline/presentation checks. No defaults
create credentials, a ledger, a connection, a provider fallback or permissions.
"""

import json
import math
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

from jsonschema import Draft202012Validator

VERSION = "external-evidence.v1"
SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"const": "search_public"},
        "query": {"type": "string", "minLength": 1, "maxLength": 512},
    },
    "required": ["action", "query"],
    "additionalProperties": False,
}
SEARCH_DESCRIPTION = (
    "Look up a missing public fact that needs current or source-backed evidence. "
    "Pass one minimal public query; resolve private references locally first. "
    "Returns external evidence with URLs, captured text scope and actual "
    "success/empty/failed/partial status. It does not search personal memory, "
    "read local files, change character facts or complete memory operations."
)
EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"const": "extract_public"},
        "url": {"type": "string", "minLength": 1, "maxLength": 2048},
    },
    "required": ["action", "url"],
    "additionalProperties": False,
    "description": (
        "Read one known public HTTP(S) page URL. Returns captured page text and its scope; "
        "it does not search for URLs, read local files or prove page completeness."
    ),
}


def canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def fingerprint(value):
    return sha256(canonical(value).encode("utf-8")).hexdigest()


def _public_url(value):
    """Validate the same URL rules at both public source binding boundaries.

    Preserve exact spelling, query and escapes; no inferred URL equivalence.
    This is not a network-address/redirect permission check.
    """
    if (
        not isinstance(value, str)
        or not value
        or any(char.isspace() or ord(char) < 32 or char == "\\" for char in value)
    ):
        raise ValueError("Unambiguous credential-free public URL required")
    try:
        url = urlsplit(value)
        port = url.port
    except ValueError as error:
        raise ValueError("Valid public URL authority required") from error
    if (
        url.scheme not in {"https", "http"}
        or not url.hostname
        or url.username is not None
        or url.password is not None
        or (port is not None and not 0 < port <= 65535)
    ):
        raise ValueError("Credential-free HTTP public URL required")
    return value


@lru_cache(maxsize=1)
def _citation_validator():
    # Frozen native-adapter contract supplied by its owner; no guessed sub-schema.
    path = Path(__file__).with_name("citation-map.schema.json")
    return Draft202012Validator(json.loads(path.read_text(encoding="utf-8-sig")))


def _citation_admission_reason(data):
    """Validate normalized source binding, not truth or semantic entailment."""
    references = {row["source_id"]: row for row in data["references"]}
    for citation in data["citation_map"]:
        if not _citation_validator().is_valid(citation):
            return "external_citation_schema_invalid"
        try:
            if citation["url"] is not None:
                _public_url(citation["url"])
            raw_url = citation["raw"].get("url")
            if raw_url is not None:
                _public_url(raw_url)
        except ValueError:
            return "external_citation_url_invalid"
        if raw_url != citation["url"]:
            return "external_citation_binding_invalid"
        source_id = citation["source_id"]
        if source_id is None:
            # A protocol-explicit unknown is legal audit data, not a binding.
            continue
        reference = references.get(source_id)
        if (
            reference is None
            or citation["url"] != reference["url"]
            or citation["source_content_sha256"] != reference["content_sha256"]
        ):
            return "external_citation_binding_invalid"
        captured = citation["captured_excerpt_range"]
        start, end = captured["start"], captured["end"]
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start < end <= len(reference["text"])
            or citation["raw"].get("cited_text") != reference["text"][start:end]
        ):
            return "external_citation_excerpt_invalid"
    return None


def parse_public_action(value):
    if Draft202012Validator(EXTRACT_SCHEMA).is_valid(value):
        return {"action": "extract_public", "url": _public_url(value["url"])}
    if not Draft202012Validator(SEARCH_SCHEMA).is_valid(value):
        raise ValueError("Exact search_public/query or extract_public/url action required")
    query = value["query"]
    if not query.strip() or any(ord(char) < 32 for char in query):
        raise ValueError("Nonempty plain public query required")
    return {"action": "search_public", "query": query}


@dataclass(frozen=True)
class ExternalConfiguration:
    """Non-secret static identity; actual protocol support must be explicit."""

    route_id: str
    revision: int
    protocol: str
    tool_version: str
    credential_version: str
    policy_version: str
    configured: bool
    enabled: bool
    query_bytes: int
    result_bytes: int

    def __post_init__(self):
        if (
            any(
                not isinstance(value, str) or not 0 < len(value) <= 128
                for value in (
                    self.route_id,
                    self.protocol,
                    self.tool_version,
                    self.credential_version,
                    self.policy_version,
                )
            )
            or type(self.revision) is not int
            or self.revision < 1
            or type(self.configured) is not bool
            or type(self.enabled) is not bool
            or type(self.query_bytes) is not int
            or not 1 <= self.query_bytes <= 2048
            or type(self.result_bytes) is not int
            or not 1 <= self.result_bytes <= 64000
        ):
            raise ValueError("Complete bounded external configuration required")


@dataclass(frozen=True)
class ExternalBinding:
    configuration: ExternalConfiguration
    character_sha256: str
    identity: str
    schema_json: str | None
    supports_extract: bool = False

    def __post_init__(self):
        if type(self.supports_extract) is not bool:
            raise ValueError("Explicit public operation capability required")
        if self.supports_extract:
            schema = {"type": "object", "oneOf": [SEARCH_SCHEMA, EXTRACT_SCHEMA]}
            exposed = self.configuration.configured and self.configuration.enabled
            if (self.identity != fingerprint([
                    VERSION, asdict(self.configuration), self.character_sha256, schema])
                    or self.schema_json != (canonical(schema) if exposed else None)):
                raise ValueError("Public extraction differs from its frozen schema identity")

    @property
    def exposed(self):
        return self.schema_json is not None


@lru_cache(maxsize=32)
def bind_external(configuration, character_sha256, *, supports_extract=False):
    """Pure static binding, reusable across model turns; performs no I/O."""
    if (
        not isinstance(configuration, ExternalConfiguration)
        or type(supports_extract) is not bool
        or not isinstance(character_sha256, str)
        or not re.fullmatch("[0-9a-f]{64}", character_sha256)
    ):
        raise ValueError("Frozen external configuration and character identity required")
    exposed = configuration.configured and configuration.enabled
    schema = ({"type": "object", "oneOf": [SEARCH_SCHEMA, EXTRACT_SCHEMA]}
              if supports_extract else SEARCH_SCHEMA)
    identity = fingerprint([VERSION, asdict(configuration), character_sha256, schema])
    return ExternalBinding(
        configuration, character_sha256, identity, canonical(schema) if exposed else None,
        supports_extract,
    )


@dataclass(frozen=True)
class ApprovedPublicQuery:
    """Host approval of this exact minimized task, not model self-attestation."""

    query: str
    authorization_id: str
    basis_id: str
    operation: str = "search_public"

    def __post_init__(self):
        field = "url" if self.operation == "extract_public" else "query"
        parse_public_action({"action": self.operation, field: self.query})
        if any(
            not isinstance(value, str) or not value
            for value in (self.authorization_id, self.basis_id)
        ):
            raise ValueError("Host authorization and public-query basis required")


@dataclass(frozen=True)
class ExternalReservation:
    """Caps are separate service units and money, never fabricated tokens."""

    search_requests: int
    cost_microusd: int
    input_tokens: int | None
    output_tokens: int | None

    def __post_init__(self):
        if (
            type(self.search_requests) is not int
            or self.search_requests != 1
            or type(self.cost_microusd) is not int
            or self.cost_microusd < 0
            or any(
                value is not None and (type(value) is not int or value < 0)
                for value in (self.input_tokens, self.output_tokens)
            )
        ):
            raise ValueError("One dispatch with explicit independent cost/token caps required")


class ExternalBudgetPort(Protocol):
    def reserve(self, *, request_id, binding_id, request_sha256, allowance): ...

    def finish(self, receipt, *, status, transmission, usage_status, usage_json): ...


@dataclass(frozen=True)
class ExternalEvidence:
    """Frozen normalized response. Native events/citations remain unmodified data."""

    payload: str

    @classmethod
    def create(
        cls,
        *,
        request_id,
        binding_id,
        status,
        failure,
        coverage,
        retrieved_at,
        references,
        native_events,
        citation_map,
        summary,
        usage_status,
        usage,
    ):
        if status not in {"success", "empty", "failed", "partial"}:
            raise ValueError("Explicit external terminal status required")
        if (
            coverage not in {"none", "partial", "returned_result_set"}
            or not isinstance(retrieved_at, str)
            or not retrieved_at
            or not isinstance(summary, str)
            or not isinstance(request_id, str)
            or not request_id
            or not isinstance(binding_id, str)
            or not re.fullmatch("[0-9a-f]{64}", binding_id)
            or (status == "failed" and (not isinstance(failure, str) or not failure))
            or (status != "failed" and failure is not None)
        ):
            raise ValueError("Explicit result identity, coverage, time and failure state required")
        if (
            not isinstance(references, (list, tuple))
            or len(references) > 32
            or not isinstance(native_events, (list, tuple))
            or not isinstance(citation_map, (list, tuple))
        ):
            raise ValueError("Bounded external references and native evidence required")
        normalized, source_ids = [], set()
        for reference in references:
            required = {
                "source_id",
                "url",
                "title",
                "published_at",
                "content_kind",
                "text",
                "source_range",
            }
            if not isinstance(reference, dict) or set(reference) != required:
                raise ValueError("Complete explicit external reference required")
            _public_url(reference["url"])
            if (
                reference["content_kind"] not in {"snippet", "excerpt", "body", "provider_summary"}
                or any(
                    not isinstance(reference[key], str) for key in ("source_id", "title", "text")
                )
                or not reference["source_id"].strip()
                or reference["source_id"] in source_ids
                or (
                    reference["published_at"] is not None
                    and not isinstance(reference["published_at"], str)
                )
            ):
                raise ValueError("Credential-free URL and scoped captured text required")
            source_ids.add(reference["source_id"])
            source_range = reference["source_range"]
            if source_range is not None and (
                not isinstance(source_range, dict)
                or set(source_range) != {"unit", "start", "end"}
                or source_range["unit"] != "unicode_codepoints"
                or type(source_range["start"]) is not int
                or type(source_range["end"]) is not int
                or not 0 <= source_range["start"] <= source_range["end"]
            ):
                raise ValueError("Explicit captured source range or unknown required")
            normalized.append(
                {
                    **reference,
                    "content_sha256": sha256(reference["text"].encode("utf-8")).hexdigest(),
                }
            )
        if (status == "empty" and normalized) or (status == "success" and not normalized):
            raise ValueError("Empty and successful source evidence must differ")
        if usage_status not in {"known", "partial", "unknown"} or not isinstance(usage, dict):
            raise ValueError("Explicit usage state required")
        if set(usage) != {"tokens", "search", "cost_microusd", "raw"}:
            raise ValueError("Token/search/cost usage must stay separate and explicit")
        tokens = usage["tokens"]
        if tokens is not None and (
            not isinstance(tokens, dict)
            or set(tokens) != {"prompt_tokens", "completion_tokens", "total_tokens"}
            or any(type(value) is not int or value < 0 for value in tokens.values())
            or tokens["prompt_tokens"] + tokens["completion_tokens"] != tokens["total_tokens"]
        ):
            raise ValueError("Reported token usage or unknown required")
        search = usage["search"]
        if search is not None and (
            not isinstance(search, dict)
            or set(search) != {"unit", "value"}
            or not isinstance(search["unit"], str)
            or not search["unit"]
            or type(search["value"]) is not int
            or search["value"] < 0
        ):
            raise ValueError("Reported service-specific search units or unknown required")
        if usage["cost_microusd"] is not None and (
            type(usage["cost_microusd"]) is not int or usage["cost_microusd"] < 0
        ):
            raise ValueError("Reported cost or explicit unknown required")
        known_units = sum(usage[key] is not None for key in ("tokens", "search", "cost_microusd"))
        expected_status = (
            "unknown" if known_units == 0 else "known" if known_units == 3 else "partial"
        )
        if usage_status != expected_status:
            raise ValueError("Usage state must preserve the reported units")
        data = dict(
            protocol=VERSION,
            source_kind="public_web",
            request_id=request_id,
            binding_id=binding_id,
            status=status,
            failure=failure,
            coverage=coverage,
            retrieved_at=retrieved_at,
            references=normalized,
            native_events=native_events,
            citation_map=citation_map,
            summary=summary,
            usage_status=usage_status,
            usage=usage,
        )
        payload = canonical(data)
        if len(payload.encode("utf-8")) > 2_000_000:
            raise ValueError("External response exceeds adapter recording capacity")
        return cls(payload)

    @property
    def data(self):
        return json.loads(self.payload)

    def checked(self):
        """Revalidate typed adapter input; a Python class is not a trust gate."""
        if not isinstance(self.payload, str) or len(self.payload.encode("utf-8")) > 2_000_000:
            raise ValueError("Bounded external response payload required")
        data = self.data
        fields = {
            "protocol",
            "source_kind",
            "request_id",
            "binding_id",
            "status",
            "failure",
            "coverage",
            "retrieved_at",
            "references",
            "native_events",
            "citation_map",
            "summary",
            "usage_status",
            "usage",
        }
        if (
            not isinstance(data, dict)
            or set(data) != fields
            or data["protocol"] != VERSION
            or data["source_kind"] != "public_web"
            or not isinstance(data["references"], list)
        ):
            raise ValueError("Exact versioned public evidence required")
        references = []
        for row in data["references"]:
            if not isinstance(row, dict) or "content_sha256" not in row:
                raise ValueError("Captured external source hash required")
            references.append({key: value for key, value in row.items() if key != "content_sha256"})
        values = {
            key: value for key, value in data.items() if key not in {"protocol", "source_kind"}
        }
        normalized = type(self).create(**{**values, "references": references})
        if normalized.data != data:
            raise ValueError("External evidence hash or normalized content differs")
        return normalized


@dataclass(frozen=True)
class ExternalOutcome:
    evidence: ExternalEvidence
    admissible: bool
    blocked_reason: str | None
    receipt: str | None
    transmission: str
    rejected_response_sha256: str | None = None

    @property
    def public_evidence(self):
        """Only reviewable public data; raw provider/native fields stay in audit.

        Admission is source-binding eligibility, never semantic support or a
        presentation receipt. Failed/partial status and unknown dates stay explicit.
        """
        if not self.admissible:
            return None
        data = self.evidence.checked().data
        if _citation_admission_reason(data) is not None:
            return None
        fields = (
            "protocol",
            "source_kind",
            "request_id",
            "binding_id",
            "status",
            "failure",
            "coverage",
            "retrieved_at",
            "references",
            "usage_status",
        )
        projection = {key: data[key] for key in fields}
        projection["references"] = [row for row in data["references"] if row["text"].strip()]
        unresolved = [row for row in data["citation_map"] if row["source_id"] is None]
        projection["citation_map"] = [
            {
                key: row[key]
                for key in ("source_id", "url", "source_content_sha256", "captured_excerpt_range")
            }
            for row in data["citation_map"]
            if row["source_id"] is not None
        ]
        projection["unresolved_citation_count"] = len(unresolved)
        projection["unresolved_citation_reasons"] = sorted(
            {row["unresolved_reason"] for row in unresolved}
        )
        projection["uncaptured_reference_count"] = len(data["references"]) - len(
            projection["references"]
        )
        if (unresolved or projection["uncaptured_reference_count"]) and data["status"] in {
            "success",
            "partial",
        }:
            projection["status"], projection["coverage"] = "partial", "partial"
        projection["audit_evidence_sha256"] = sha256(
            self.evidence.payload.encode("utf-8")
        ).hexdigest()
        return projection


class ExternalBackendPort(Protocol):
    def connect(self, *, cancelled, deadline): ...

    def search(self, body, *, request_id, binding_id, cancelled, deadline) -> ExternalEvidence: ...

    def close(self): ...


def safe_failure(error):
    """Only owned internal codes may pass through; never provider exception text."""
    owned = {
        "external_cancelled",
        "external_deadline",
        "external_authorization_revoked",
        "external_not_configured_or_disabled",
    }
    if str(error) in owned:
        return str(error)
    if isinstance(error, InterruptedError):
        return "external_cancelled"
    if isinstance(error, TimeoutError):
        return "external_deadline"
    return "external_permission_failure"


class ExternalQueryCall:
    """One bound task. Creating/preparing it never creates a client or sends."""

    def __init__(
        self,
        *,
        binding,
        request_id,
        approve_query,
        authorized,
        checkpoint,
        can_admit,
        cancelled,
        clock,
        deadline,
        slots,
        budget,
        allowance,
        backend_factory,
        retrieved_clock,
        available_bytes,
    ):
        if (
            not isinstance(binding, ExternalBinding)
            or type(deadline) not in {int, float}
            or not math.isfinite(deadline)
            or type(available_bytes) is not int
            or available_bytes < 1
            or not isinstance(allowance, ExternalReservation)
        ):
            raise ValueError("Bound capability, deadline, capacity and reservation required")
        self.binding, self.request_id = binding, request_id
        self.approve_query, self.authorized = approve_query, authorized
        self.checkpoint, self.can_admit, self.cancelled = checkpoint, can_admit, cancelled
        self.clock, self.deadline, self.retrieved_clock = clock, deadline, retrieved_clock
        self.slots, self.budget, self.allowance = slots, budget, allowance
        self.backend_factory = backend_factory
        self.available_bytes = min(available_bytes, binding.configuration.result_bytes)
        self._query, self._last, self._used = None, None, False

    def _check(self):
        self.checkpoint()
        if self.cancelled.is_set():
            raise InterruptedError("external_cancelled")
        now = self.clock()
        if (
            type(now) not in {int, float}
            or not math.isfinite(now)
            or now >= self.deadline
            or (self._last is not None and now < self._last)
        ):
            raise TimeoutError("external_deadline")
        self._last = now

    def _authorize(self):
        self._check()
        if not self.authorized(self.binding, self._query):
            raise PermissionError("external_authorization_revoked")

    def prepare(self, action):
        if self._used or self._query is not None:
            raise ValueError("External request already frozen or consumed")
        value = parse_public_action(action)
        extracting = value["action"] == "extract_public"
        if extracting and not self.binding.supports_extract:
            raise ValueError("Public extraction capability was not selected")
        self._check()
        field = "url" if extracting else "query"
        query = self.approve_query(
            self.binding, value[field], **({"operation": "extract_public"} if extracting else {})
        )
        if (
            not isinstance(query, ApprovedPublicQuery)
            or query.operation != value["action"]
            or (extracting and query.query != value["url"])
            or len(query.query.encode("utf-8")) > self.binding.configuration.query_bytes
        ):
            raise ValueError("Host-approved bounded public query required")
        self._query = query
        self._body = canonical({field: query.query})
        self._request_hash = fingerprint(
            [self.binding.identity, self._body, query.authorization_id, query.basis_id]
        )
        self._authorize()

    def _failure(self, reason):
        return ExternalEvidence.create(
            request_id=self.request_id,
            binding_id=self.binding.identity,
            status="failed",
            failure=reason,
            coverage="none",
            retrieved_at=self.retrieved_clock(),
            references=[],
            native_events=[],
            citation_map=[],
            summary="",
            usage_status="unknown",
            usage={"tokens": None, "search": None, "cost_microusd": None, "raw": None},
        )

    def invoke(self):
        if self._used or self._query is None:
            raise ValueError("Frozen unconsumed public request required")
        self._used = True
        receipt, client, acquired = None, None, False
        transmission, reason, evidence = "not_sent", None, None
        rejected_response_sha256, citation_reason = None, None
        try:
            if not self.binding.exposed:
                raise PermissionError("external_not_configured_or_disabled")
            self._authorize()
            while not acquired:
                acquired = self.slots.acquire(timeout=0.025)
                self._authorize()
            receipt = self.budget.reserve(
                request_id=self.request_id,
                binding_id=self.binding.identity,
                request_sha256=self._request_hash,
                allowance=self.allowance,
            )
            self._authorize()
            # First client creation/connection/discovery is inside the same gate.
            transmission = "may_have_been_sent"
            client = self.backend_factory(self.binding)
            client.connect(cancelled=self.cancelled, deadline=self.deadline)
            self._authorize()
            candidate = client.search(
                json.loads(self._body),
                request_id=self.request_id,
                binding_id=self.binding.identity,
                cancelled=self.cancelled,
                deadline=self.deadline,
            )
            transmission = "response_received"
            if not isinstance(candidate, ExternalEvidence):
                raise ValueError("Normalized ExternalEvidence required")
            rejected_response_sha256 = sha256(candidate.payload.encode("utf-8")).hexdigest()
            normalized = candidate.checked()
            data = normalized.data
            if data["request_id"] != self.request_id or data["binding_id"] != self.binding.identity:
                raise ValueError("External response request identity differs")
            citation_reason = _citation_admission_reason(data)
            evidence = normalized
            if citation_reason is None:
                # Clear the rejection marker only after all admission validation.
                rejected_response_sha256 = None
        except (InterruptedError, TimeoutError, PermissionError) as error:
            reason = safe_failure(error)
        except Exception:
            # Raw provider exception messages can contain credentials or URLs.
            reason = "external_transport_protocol_or_budget_failure"
        finally:
            # Accounting and resource cleanup must also run when failure-result
            # construction itself fails (for example, a host clock callback).
            data = {
                "status": "failed",
                "usage_status": "unknown",
                "usage": {"tokens": None, "search": None, "cost_microusd": None, "raw": None},
            }
            try:
                if evidence is None:
                    evidence = self._failure(reason or "external_failed")
                data = evidence.data
            finally:
                try:
                    if receipt is not None:
                        # This must join the host ledger; failure prevents release.
                        self.budget.finish(
                            receipt,
                            status=data["status"],
                            transmission=transmission,
                            usage_status=data["usage_status"],
                            usage_json=canonical(data["usage"]),
                        )
                finally:
                    try:
                        if client is not None:
                            try:
                                client.close()
                            except Exception:
                                raise RuntimeError("external_client_cleanup_failure") from None
                    finally:
                        if acquired:
                            self.slots.release()
        # Late data remains in the accounting/audit outcome, never reopens delivery.
        blocked_reason = citation_reason or (
            "external_response_evidence_invalid" if rejected_response_sha256 is not None else None
        )
        try:
            self._authorize()
            if not self.can_admit():
                blocked_reason = "external_owner_or_delivery_closed"
            elif (
                blocked_reason is None
                and len(evidence.payload.encode("utf-8")) > self.available_bytes
            ):
                blocked_reason = "external_context_capacity"
        except (InterruptedError, TimeoutError, PermissionError) as error:
            blocked_reason = safe_failure(error)
        return ExternalOutcome(
            evidence,
            blocked_reason is None,
            blocked_reason,
            receipt,
            transmission,
            rejected_response_sha256,
        )
