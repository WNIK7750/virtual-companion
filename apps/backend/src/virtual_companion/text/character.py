"""User-supplied role data, frozen independently of runtime/model selection."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .identity import VersionedReference, _identity

MAX_CHARACTER_BYTES = 32768
_FIELDS = {"protocol", "source", "companion_id", "character_id", "version",
           "personality", "background", "initial_relationship", "expression"}
_ROLE_SYSTEM = (
    "以下是用户提供并为本次来源冻结的角色设定；人格、背景、初始关系和表达偏好由它决定。"
    "虚构角色背景只描述角色，不是现实中的用户事实、共同经历、已执行操作或已呈现内容的证据；"
    "不得把设定当成用户记忆写入。角色设定不改变实际证据、来源、权限与操作结果的约束。\n"
)


@dataclass(frozen=True)
class CharacterSnapshot:
    companion_id: str
    character_id: str
    version: str
    personality: str
    background: str
    initial_relationship: str
    expression: str

    def __post_init__(self):
        _identity(self.companion_id)
        VersionedReference(self.character_id, self.version, "0" * 64)
        if any(not isinstance(value, str) or len(value.encode("utf-8")) > MAX_CHARACTER_BYTES
               for value in (self.personality, self.background,
                             self.initial_relationship, self.expression)):
            raise ValueError("Bounded character text required")
        if len(self.payload.encode("utf-8")) > MAX_CHARACTER_BYTES:
            raise ValueError("Character context exceeds byte limit")

    def to_mapping(self):
        return {"protocol": "character-context.v1", "source": "user-config",
                "companion_id": self.companion_id, "character_id": self.character_id,
                "version": self.version, "personality": self.personality,
                "background": self.background, "initial_relationship": self.initial_relationship,
                "expression": self.expression}

    @property
    def payload(self):
        return json.dumps(self.to_mapping(), ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"))

    @property
    def reference(self):
        return VersionedReference(self.character_id, self.version,
                                  hashlib.sha256(self.payload.encode("utf-8")).hexdigest())

    @classmethod
    def from_mapping(cls, data):
        if (not isinstance(data, dict) or set(data) != _FIELDS
                or data["protocol"] != "character-context.v1" or data["source"] != "user-config"):
            raise ValueError("Invalid character context fields")
        return cls(**{key: value for key, value in data.items()
                      if key not in {"protocol", "source"}})


def load_character(path: Path, companion: str):
    """Missing configuration stays absent; malformed/foreign data fails closed."""
    path = Path(path)
    if not path.exists():
        return None
    with path.open("rb") as source:
        content = source.read(MAX_CHARACTER_BYTES + 1)
    if len(content) > MAX_CHARACTER_BYTES:
        raise ValueError("Character context exceeds byte limit")
    snapshot = CharacterSnapshot.from_mapping(json.loads(content.decode("utf-8")))
    if snapshot.companion_id != companion:
        raise ValueError("Character companion differs from data scope")
    return snapshot


def character_from_binding(binding):
    """Read the original persisted input binding, never mutable global config."""
    data = json.loads(binding) if isinstance(binding, str) else binding
    config = data.get("companion_config", {})
    if "character_data" not in config:
        return None
    from .identity import CompanionConfig

    frozen = CompanionConfig.from_mapping(config)
    if (frozen.companion_id != data.get("companion_id")
            or frozen.snapshot_hash != data.get("companion_config_sha256")):
        raise ValueError("Character companion differs from input binding")
    return json.loads(frozen.character_data)


def character_messages(data):
    if data is None:
        return []
    snapshot = CharacterSnapshot.from_mapping(data)
    return [{"role": "system", "content": _ROLE_SYSTEM + snapshot.payload}]


def character_review_messages(data):
    messages = character_messages(data)
    if messages:
        messages[0]["content"] += (
            "\n审核区分角色自身的设定表达与用户现实断言：设定范围内的角色自述属于角色前提，"
            "不要求current重复授权，不能因此一律判成虚构违规或把表达统一成中性客服。"
            "它不为用户个人事实、实际共同经历或当轮作品续写提供来源；"
            "这些仍依原来源与当前授权逐项审核。"
        )
    return messages
