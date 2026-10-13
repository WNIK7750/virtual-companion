"""An injected optional public-query capability, under the existing runtime owner."""

import json
from dataclasses import asdict, dataclass

from virtual_companion.boundary.public_capture import (
    PUBLIC_MODEL_PREFIX,
    public_request_id,
    validate_public_capture,
)
from virtual_companion.providers.external_information import ExternalQueryCall, canonical


@dataclass(frozen=True)
class PublicQueryCapability:
    binding: object
    approve_query: object
    authorized: object
    budget: object
    allowance: object
    slots: object
    backend_factory: object
    clock: object
    retrieved_clock: object
    supports_extract: bool = False

    def __post_init__(self):
        if (type(self.supports_extract) is not bool
                or self.supports_extract != self.binding.supports_extract):
            raise ValueError("Public operation capability differs from its static binding")

    def start(self, runtime, claim, cancelled):
        frozen = json.loads(claim.config_json)["companion_config"]
        if (
            self.binding.character_sha256 != runtime.config.character.sha256
            or self.binding.character_sha256 != frozen["character"]["sha256"]
        ):
            raise ValueError("Public capability role differs from frozen turn")
        if frozen["resources"]["model"]["id"] != PUBLIC_MODEL_PREFIX + self.binding.identity:
            raise ValueError("Public capability identity differs from frozen turn")
        return PublicQuerySession(self, runtime, claim, cancelled)


class PublicQuerySession:
    def __init__(self, capability, runtime, claim, cancelled):
        self.capability, self.runtime, self.cancelled = capability, runtime, cancelled
        self.basis = asdict(claim)
        del self.basis["lease_until_us"]
        self.request_id = public_request_id(self.basis)
        self.deadline = capability.clock() + (runtime.deadline - runtime.clock()) / 1_000_000
        self.calls, self.outcome, self.payload, self.query = 0, None, None, None
        self.blocked_reason = None
        self.available_bytes = 0

    @property
    def can_search(self):
        return self.capability.binding.exposed and self.calls == 0

    @property
    def can_extract(self):
        return self.capability.supports_extract and self.can_search

    def validate(self, request=None):
        # An approved frozen schema is not current authority. Recheck every stage,
        # review, approve and send through the existing runtime checkpoint.
        if self.payload is not None:
            self.authorize_capture(json.loads(self.payload))
        if request is not None and request.public_evidence != self.payload:
            raise ValueError("Public attachment differs from the frozen session")

    def authorize_capture(self, capture):
        """Pure current-session check: no owner checkpoint, SQL, I/O or settlement.

        The injected ``authorized`` predicate must itself be a bounded pure
        in-process authority check. Stores invoke this inside their existing
        transaction after identity/readset checks and immediately before write.
        """
        if (
            self.runtime._public_session is not self
            or self.payload is None
            or canonical(capture) != self.payload
        ):
            raise PermissionError("external_public_session_mismatch")
        if not self.capability.authorized(self.capability.binding, self.query):
            raise PermissionError("external_authorization_revoked")
        return True

    def search(self, claim, query, *, cancelled):
        return self._dispatch(claim, {"action": "search_public", "query": query},
                              cancelled=cancelled)

    def extract(self, claim, url, *, cancelled):
        if not self.can_extract:
            raise ValueError("Public extraction capability was not selected")
        return self._dispatch(claim, {"action": "extract_public", "url": url},
                              cancelled=cancelled)

    def _dispatch(self, claim, action, *, cancelled):
        if cancelled is not self.cancelled or not self.can_search:
            raise ValueError("One same-turn public dispatch required")
        self.calls += 1
        cap, runtime = self.capability, self.runtime
        if self.available_bytes < 1:
            self.blocked_reason = "external_context_capacity"
            return None

        def approve(binding, raw, **kwargs):
            self.query = cap.approve_query(binding, raw, **kwargs)
            return self.query

        call = ExternalQueryCall(
            binding=cap.binding,
            request_id=self.request_id,
            approve_query=approve,
            authorized=cap.authorized,
            checkpoint=lambda: runtime.check(cancelled),
            can_admit=lambda: not runtime.has_approved_prefix(),
            cancelled=cancelled,
            clock=cap.clock,
            deadline=self.deadline,
            slots=cap.slots,
            budget=cap.budget,
            allowance=cap.allowance,
            backend_factory=cap.backend_factory,
            retrieved_clock=cap.retrieved_clock,
            available_bytes=self.available_bytes,
        )
        try:
            call.prepare(action)
        except PermissionError:
            # No dispatch or evidence occurred. The consumed optional query does
            # not prevent an independently reviewed local response.
            self.blocked_reason = "external_authorization_revoked"
            return None
        self.outcome = call.invoke()
        projection = self.outcome.public_evidence
        if projection is not None:
            try:
                validate_public_capture(projection)
            except ValueError:
                # r3 may carry a provider_summary reference. This consumer only
                # admits captured page text; reject the complete attachment,
                # retain the already-settled audit outcome, continue locally.
                self.blocked_reason = "external_public_capture_not_supported"
            else:
                self.payload = canonical(projection)
        else:
            self.blocked_reason = self.outcome.blocked_reason
        runtime.check(cancelled)
        return self.payload
