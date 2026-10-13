"""Versioned route capabilities are supplied, never guessed from a vendor name."""

import math
import re
from dataclasses import asdict, dataclass
from hashlib import sha256
from types import MappingProxyType
from urllib.parse import urlsplit

from virtual_companion.boundary.contracts import canonical, identifier

DEEPSEEK_FLASH_PROFILE = "deepseek-flash.v1"
DEEPSEEK_FOCUSED_PROFILE = "deepseek-flash.focused.v2"
DEEPSEEK_TEXT_PROFILES = frozenset({DEEPSEEK_FLASH_PROFILE, DEEPSEEK_FOCUSED_PROFILE})
DEEPSEEK_ENDPOINT = "https://api.deepseek.com/chat/completions"
MAX_PROVIDER_RESPONSE_BYTES = 2_000_000


class ProviderStreamCapacityError(BufferError):
    """A bounded provisional stream cannot retain another pending content event."""


DEEPSEEK_FOCUSED_STAGE_TEMPERATURES = MappingProxyType({
    "demand_control": 0.2,
    "demand_repair": 0.2,
    "stream_unit_review": 0,
    "fact_target_select": 0,
    "fact_extract": 0,
    "fact_support": 0,
    "fact_repair": 0,
    "fact_assess": 0,
    "exchange_intent": 0,
    "exchange_extract": 0,
    "exchange_assess": 0,
    "exchange_support": 0,
    "source_propose": 0,
    "source_repair": 0,
    "memory_command": 0,
    "memory_command_support": 0,
})


def focused_stage_temperature(stage):
    """Fixed per-call options for actual model stages, never host-only checkpoints."""
    if type(stage) is not str or stage not in DEEPSEEK_FOCUSED_STAGE_TEMPERATURES:
        raise ValueError("Focused DeepSeek profile requires a known model stage")
    return DEEPSEEK_FOCUSED_STAGE_TEMPERATURES[stage]


@dataclass(frozen=True)
class ProviderCapabilities:
    streaming: bool
    tools: bool
    structured_output: bool
    context_tokens: int
    output_field: str

    def __post_init__(self):
        if any(type(v) is not bool for v in (self.streaming, self.tools, self.structured_output)):
            raise ValueError("Explicit provider capabilities required")
        if type(self.context_tokens) is not int or self.context_tokens < 2:
            raise ValueError("Declared provider context capacity required")
        if self.output_field not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError("Declared output limit field required")


@dataclass(frozen=True)
class ProviderSampling:
    temperature: float
    top_p: float
    seed: int

    def __post_init__(self):
        if (
            any(
                type(v) not in {int, float} or not math.isfinite(v)
                for v in (self.temperature, self.top_p)
            )
            or not 0 <= self.temperature <= 2
            or not 0 < self.top_p <= 1
            or type(self.seed) is not int
            or not 0 <= self.seed < 2**32
        ):
            raise ValueError("Explicit bounded sampling required")


@dataclass(frozen=True)
class ProviderRoute:
    route_id: str
    revision: int
    kind: str
    endpoint: str
    model: str
    auth: str
    secret_ref: str | None
    explicit: bool
    data_scope: tuple[str, ...]
    capabilities: ProviderCapabilities
    sampling: ProviderSampling | None = None
    wire_profile: str | None = None

    def __post_init__(self):
        identifier(self.route_id)
        if type(self.revision) is not int or self.revision < 1 or type(self.explicit) is not bool:
            raise ValueError("Explicit route revision and selection required")
        if self.kind not in {"local", "api"} or self.auth not in {"none", "bearer"}:
            raise ValueError("Unsupported route or authentication")
        if not isinstance(self.model, str) or not self.model.strip() or len(self.model) > 256:
            raise ValueError("Explicit bounded model ID required")
        if not isinstance(self.endpoint, str) or len(self.endpoint) > 2048:
            raise ValueError("Bounded endpoint required")
        url = urlsplit(self.endpoint)
        if (
            url.scheme not in {"https", "http"}
            or not url.hostname
            or url.username is not None
            or url.password is not None
            or url.query
            or url.fragment
            or not url.path.endswith("/chat/completions")
            or any(c.isspace() or ord(c) < 32 for c in self.endpoint)
        ):
            raise ValueError("Explicit credential-free Chat Completions endpoint required")
        if url.port is not None and not 0 < url.port < 65536:
            raise ValueError("Invalid endpoint port")
        if self.kind == "local":
            if url.hostname not in {"127.0.0.1", "::1"}:
                raise ValueError("Local route must use a loopback IP")
        elif not self.explicit or url.scheme != "https" or self.auth != "bearer":
            raise ValueError("API route requires explicit HTTPS selection and secret reference")
        if self.auth == "none":
            if self.secret_ref is not None or self.kind != "local":
                raise ValueError("No-key authentication is local only")
        elif not isinstance(self.secret_ref, str) or not re.fullmatch(
            r"dpapi:[0-9a-f]{32}", self.secret_ref
        ):
            raise ValueError("Managed secret reference required")
        if (
            not isinstance(self.data_scope, tuple)
            or not self.data_scope
            or len(set(self.data_scope)) != len(self.data_scope)
            or set(self.data_scope) - {"current", "history", "valid_memory", "derived"}
        ):
            raise ValueError("Explicit context sending scope required")
        if not isinstance(self.capabilities, ProviderCapabilities):
            raise ValueError("Typed provider capabilities required")
        if self.sampling is not None and not isinstance(self.sampling, ProviderSampling):
            raise ValueError("Typed explicit sampling required")
        if self.wire_profile is not None and self.wire_profile not in DEEPSEEK_TEXT_PROFILES:
            raise ValueError("Unsupported explicit provider wire profile")
        if self.wire_profile in DEEPSEEK_TEXT_PROFILES and (
            self.kind != "api"
            or self.endpoint != DEEPSEEK_ENDPOINT
            or self.model != "deepseek-flash"
            or self.capabilities.output_field != "max_tokens"
            or self.capabilities.tools
            or self.sampling is not None
        ):
            # This optional profile has only text/JSON-object compatibility evidence.
            # In particular, local sampling seeds and tools are not silently dropped.
            raise ValueError("DeepSeek wire profile requires its fixed text route")

    @property
    def fingerprint(self):
        identity = asdict(self)
        if self.wire_profile is None:
            # Preserve every pre-profile frozen route and its existing budget identity.
            identity.pop("wire_profile")
        return sha256(canonical(identity).encode()).hexdigest()
