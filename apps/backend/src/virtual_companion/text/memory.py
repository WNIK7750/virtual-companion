from __future__ import annotations

from copy import deepcopy
from uuid import uuid4

from .history import PAGE_SIZE, history_page, validate_window


class MemoryConversationStore:
    """Recording implementation of ConversationStore for deterministic tests."""

    def __init__(self):
        self.session_id = str(uuid4())
        self.messages = []
        self.attempt = None
        self.sessions = {}

    def history(self):
        return {"session_id": self.session_id, "messages": deepcopy(self.messages)}

    def history_page(self, *, before=None, after=None, session_id=None):
        validate_window(before=before, after=after, session_id=session_id)
        if session_id is not None and session_id != self.session_id:
            raise ValueError("Active session changed")
        boundary = len(self.messages)
        cursor = before or after
        if cursor:
            boundary = next((i for i, m in enumerate(self.messages) if m["id"] == cursor), -1)
            if boundary < 0:
                raise ValueError("Unknown history cursor")
        rows = (
            self.messages[boundary + 1 : boundary + 1 + PAGE_SIZE + 1]
            if after
            else list(reversed(self.messages[max(0, boundary - PAGE_SIZE - 1) : boundary]))
        )
        return history_page(deepcopy(rows), self.session_id, forward=bool(after))

    def begin(self, text, binding, *, retry=False):
        if retry:
            if not self.attempt or self.attempt["status"] not in {
                "cancelled",
                "failed",
                "interrupted",
            }:
                raise ValueError("No retryable attempt")
            turn_id = self.attempt["turn_id"]
        else:
            turn_id = str(uuid4())
            self.messages.append(
                {"id": str(uuid4()), "turn_id": turn_id, "role": "user", "content": text}
            )
        self.attempt = {
            "attempt_id": str(uuid4()),
            "turn_id": turn_id,
            "status": "running",
            "binding": deepcopy(binding),
        }
        return deepcopy(self.attempt)

    def finish(self, attempt_id, status, content):
        if not self.attempt or self.attempt["attempt_id"] != attempt_id:
            raise ValueError("Unknown attempt")
        if self.attempt["status"] != "running":
            raise ValueError("Attempt already finalized")
        if status == "completed":
            self.messages.append(
                {
                    "id": str(uuid4()),
                    "turn_id": self.attempt["turn_id"],
                    "role": "assistant",
                    "content": content,
                }
            )
        self.attempt["status"] = status

    def latest_attempt(self):
        return deepcopy(self.attempt)

    def new_session(self):
        self.sessions[self.session_id] = (deepcopy(self.messages), deepcopy(self.attempt))
        self.session_id = str(uuid4())
        self.messages = []
        self.attempt = None
        return self.history()

    def select_session(self, session_id):
        if session_id == self.session_id:
            return self.history_page()
        if session_id not in self.sessions:
            raise ValueError("Unknown session")
        self.sessions[self.session_id] = (deepcopy(self.messages), deepcopy(self.attempt))
        self.messages, self.attempt = deepcopy(self.sessions[session_id])
        self.session_id = session_id
        return self.history_page()
