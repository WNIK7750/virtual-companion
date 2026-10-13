"""Explicit whole-source fact dispositions, bound to the supported commit batch."""

import json
from dataclasses import asdict, dataclass
from hashlib import sha256

from .contracts import canonical, identifier
from .exchange_resolution import digest


@dataclass(frozen=True)
class FactSourceResolution:
    source_id: str
    source_sha256: str
    context_sha256: str
    segments: tuple[str, ...]
    segment_kinds: tuple[str, ...]
    candidates: tuple[str, ...]
    dispositions: tuple[str, ...]
    reasons: tuple[str, ...]
    commands_sha256: str
    evaluator: str = "fact-source-assess.v1"
    protocol: str = "fact-source-resolution.v1"

    def __post_init__(self):
        identifier(self.source_id)
        if (self.evaluator not in {"fact-source-assess.v1",
                                  "fact-source-assess+support.v1",
                                  "exact-repeat-host.v1"}
                or self.protocol != "fact-source-resolution.v1"):
            raise ValueError("Known fact source resolution required")
        for value in (self.source_sha256, self.context_sha256, self.commands_sha256):
            if (not isinstance(value, str) or len(value) != 64
                    or any(c not in "0123456789abcdef" for c in value)):
                raise ValueError("Bound fact source digests required")
        arrays = (self.segments, self.segment_kinds, self.candidates,
                  self.dispositions, self.reasons)
        if (any(not isinstance(value, tuple) for value in arrays)
                or not 1 <= len(self.segments) <= 16
                or len(self.segments) != len(self.segment_kinds)
                or len(self.candidates) > 8
                or len({len(self.candidates), len(self.dispositions), len(self.reasons)}) != 1
                or any(not isinstance(s, str) or not s.strip() for s in self.segments)
                or any(k not in {"fact", "no_fact"} for k in self.segment_kinds)):
            raise ValueError("Complete bounded fact source coverage required")
        covered = set()
        for raw, disposition, reason in zip(self.candidates, self.dispositions, self.reasons,
                                            strict=True):
            if not isinstance(raw, str) or len(raw.encode()) > 16000:
                raise ValueError("Bounded canonical fact candidate required")
            candidate = json.loads(raw)
            if (not isinstance(candidate, dict) or canonical(candidate) != raw
                    or candidate.get("evidenceTurnId") != self.source_id
                    or not isinstance(candidate.get("evidenceQuote"), str)
                    or not candidate["evidenceQuote"].strip()):
                raise ValueError("Canonical source-bound fact candidate required")
            matches = {i for i, s in enumerate(self.segments) if candidate["evidenceQuote"] in s}
            if not matches:
                raise ValueError("Fact candidate lacks segment evidence")
            if disposition == "accepted" and reason == "fact":
                if any(self.segment_kinds[i] != "fact" for i in matches):
                    raise ValueError("Accepted fact contradicts whole-source coverage")
                covered.update(matches)
            elif disposition == "rejected" and reason in {"transient", "quotation", "non_factual"}:
                if any(self.segment_kinds[i] != "no_fact" for i in matches):
                    raise ValueError("Rejected candidate may not erase a source fact")
            else:
                raise ValueError("Unresolved fact candidate cannot complete source")
        if covered != {i for i, kind in enumerate(self.segment_kinds) if kind == "fact"}:
            raise ValueError("Whole-source fact coverage identifies an omitted fact")

    def validate_source(self, text, context_sha256, commands):
        if (sha256(text.encode()).hexdigest() != self.source_sha256
                or context_sha256 != self.context_sha256
                or digest([c.fingerprint for c in commands]) != self.commands_sha256):
            raise ValueError("Fact resolution differs from source, context or commands")
        end = 0
        for segment in self.segments:
            start = text.find(segment, end)
            if start < end or text[end:start].strip():
                raise ValueError("Fact source segmentation skips or changes evidence")
            end = start + len(segment)
        if text[end:].strip():
            raise ValueError("Fact source segmentation omits source tail")
        accepted = [json.loads(c) for c, d in zip(self.candidates, self.dispositions, strict=True)
                    if d == "accepted"]
        if len(accepted) != len(commands):
            raise ValueError("Fact resolution does not bind every accepted command")
        for candidate, command in zip(accepted, commands, strict=True):
            if json.loads(command.reason).get("quote") != candidate["evidenceQuote"]:
                raise ValueError("Supported fact differs from assessed candidate")

    def to_mapping(self):
        return asdict(self)

    @property
    def fingerprint(self):
        return digest(self.to_mapping())
