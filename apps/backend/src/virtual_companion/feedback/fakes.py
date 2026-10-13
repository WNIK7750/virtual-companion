from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import MessageOwnershipError, NotFoundError
from .models import ContextPolicy


@dataclass(frozen=True, slots=True)
class _Message:
    session_id: str
    turn_id: str
    assistant_message_id: str
    content: str
    committed: bool
    preceding: tuple[dict[str, Any], ...]


class InMemoryMessageCatalog:
    """Synthetic committed messages for B0/B3 tests; never a production store."""

    def __init__(self) -> None:
        self._messages: dict[str, _Message] = {}

    def register(
        self,
        *,
        session_id: str,
        turn_id: str,
        assistant_message_id: str,
        content: str,
        committed: bool = True,
        preceding: tuple[dict[str, Any], ...] = (),
    ) -> None:
        self._messages[assistant_message_id] = _Message(
            session_id=session_id,
            turn_id=turn_id,
            assistant_message_id=assistant_message_id,
            content=content,
            committed=committed,
            preceding=tuple(dict(item) for item in preceding),
        )

    def assert_owned_assistant_message(
        self,
        *,
        session_id: str,
        turn_id: str,
        assistant_message_id: str,
    ) -> None:
        message = self._messages.get(assistant_message_id)
        if (
            message is None
            or not message.committed
            or message.session_id != session_id
            or message.turn_id != turn_id
        ):
            raise MessageOwnershipError(
                "assistant message is missing, uncommitted, or outside the turn"
            )

    def context_for(
        self,
        *,
        session_id: str,
        turn_id: str,
        assistant_message_id: str,
        policy: ContextPolicy,
    ) -> dict[str, Any]:
        self.assert_owned_assistant_message(
            session_id=session_id,
            turn_id=turn_id,
            assistant_message_id=assistant_message_id,
        )
        message = self._messages[assistant_message_id]
        preceding = (
            []
            if policy.preceding_turns == 0
            else list(message.preceding[-policy.preceding_turns :])
        )
        return {
            "session_id": session_id,
            "turn_id": turn_id,
            "assistant_message_id": assistant_message_id,
            "preceding_messages": preceding,
            "assistant_message": {"role": "assistant", "content": message.content},
            "following_messages": [],
        }


class NullMessageCatalog:
    """Explicitly failing catalog for bootstraps that have no conversation adapter yet."""

    def assert_owned_assistant_message(self, **_: Any) -> None:
        raise NotFoundError("no committed message catalog is configured")

    def context_for(self, **_: Any) -> dict[str, Any]:
        raise NotFoundError("no committed message catalog is configured")
