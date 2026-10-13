"""Identity/configuration preparation; references do not load or execute assets.

The desktop uses the canonical identity for root binding. These references do
not migrate existing user databases or import persona/media assets.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from uuid import UUID


def _identity(value):
    if not isinstance(value, str):
        raise ValueError("Invalid companion identity")
    try:
        canonical = str(UUID(value))
    except ValueError as error:
        raise ValueError("Invalid companion identity") from error
    if canonical != value:
        raise ValueError("Canonical companion UUID required")
    return canonical


@dataclass(frozen=True)
class VersionedReference:
    id: str
    version: str
    sha256: str

    def __post_init__(self):
        for value, pattern in (
            (self.id, r"[a-z0-9][a-z0-9._-]{0,127}"),
            (self.version, r"[A-Za-z0-9][A-Za-z0-9._+-]{0,63}"),
            (self.sha256, r"[a-f0-9]{64}"),
        ):
            if not isinstance(value, str) or re.fullmatch(pattern, value) is None:
                raise ValueError("Invalid versioned reference")

    def to_mapping(self):
        return {"id": self.id, "version": self.version, "sha256": self.sha256}

    @classmethod
    def from_mapping(cls, value):
        if not isinstance(value, dict) or set(value) != {"id", "version", "sha256"}:
            raise ValueError("Invalid reference fields")
        return cls(**value)


@dataclass(frozen=True)
class CompanionConfig:
    companion_id: str
    character: VersionedReference
    resources: tuple[tuple[str, VersionedReference], ...] = ()
    character_data: str | None = None

    def __post_init__(self):
        _identity(self.companion_id)
        if not isinstance(self.character, VersionedReference) or not isinstance(
            self.resources, tuple
        ):
            raise ValueError("Invalid companion configuration")
        kinds = set()
        for item in self.resources:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError("Invalid resource binding")
            kind, reference = item
            if (
                not isinstance(kind, str)
                or kind not in {"avatar", "voice", "expression", "model"}
                or kind in kinds
                or not isinstance(reference, VersionedReference)
            ):
                raise ValueError("Invalid or duplicate resource kind")
            kinds.add(kind)
        object.__setattr__(self, "resources", tuple(sorted(self.resources)))
        if self.character_data is not None:
            from .character import CharacterSnapshot

            if not isinstance(self.character_data, str):
                raise ValueError("Immutable character payload required")
            snapshot = CharacterSnapshot.from_mapping(json.loads(self.character_data))
            if (snapshot.companion_id != self.companion_id or snapshot.reference != self.character
                    or snapshot.payload != self.character_data):
                raise ValueError("Character payload differs from reference or companion")

    def to_mapping(self):
        return {
            "schema_version": 2 if self.character_data is not None else 1,
            "companion_id": self.companion_id,
            "character": self.character.to_mapping(),
            "resources": {kind: reference.to_mapping() for kind, reference in self.resources},
            **({"character_data": json.loads(self.character_data)}
               if self.character_data is not None else {}),
        }

    @classmethod
    def from_mapping(cls, value):
        if (
            not isinstance(value, dict)
            or type(value.get("schema_version")) is not int
            or (value["schema_version"], frozenset(value)) not in (
                (1, frozenset({"schema_version", "companion_id", "character", "resources"})),
                (2, frozenset({"schema_version", "companion_id", "character", "resources",
                               "character_data"})),
            )
            or not isinstance(value["resources"], dict)
        ):
            raise ValueError("Invalid companion configuration fields")
        return cls(
            value["companion_id"],
            VersionedReference.from_mapping(value["character"]),
            tuple(
                (kind, VersionedReference.from_mapping(reference))
                for kind, reference in value["resources"].items()
            ),
            (json.dumps(value["character_data"], ensure_ascii=False, sort_keys=True,
                        separators=(",", ":")) if "character_data" in value else None),
        )

    @property
    def snapshot_hash(self):
        content = json.dumps(self.to_mapping(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(content).hexdigest()

    def with_resources(self, resources):
        """Replace the complete reference map while retaining the companion identity."""
        if not isinstance(resources, dict):
            raise ValueError("Resource mapping required")
        return replace(self, resources=tuple(resources.items()))

    def attempt_binding(self, base):
        reserved = {"companion_id", "companion_config", "companion_config_sha256"}
        if not isinstance(base, dict) or reserved.intersection(base):
            raise ValueError("Companion identity fields cannot be overridden")
        return {
            **deepcopy(base),
            "companion_id": self.companion_id,
            "companion_config": self.to_mapping(),
            "companion_config_sha256": self.snapshot_hash,
        }


def companion_data_root(root: Path, companion_id: str) -> Path:
    """Resolve an opt-in data scope; never derive ownership from a resource name."""
    target = root.resolve() / "companions" / _identity(companion_id)
    if target.resolve() != target:
        raise ValueError("Companion data scope cannot alias another path")
    return target
