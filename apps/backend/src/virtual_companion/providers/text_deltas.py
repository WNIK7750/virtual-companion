"""Provisional plain-text SSE events, never presentation approval or completion.

The existing complete decoder remains authoritative for final protocol and usage.
Pi's separate delta/terminal events are a mechanism reference, not copied source.
"""

import re
from dataclasses import dataclass, field

from .openai_compatible import strict_json


@dataclass(frozen=True)
class UnapprovedTextDelta:
    sequence: int
    text: str
    approved: bool = field(default=False, init=False)


class TextDeltas:
    def __init__(self):
        self.pending = b""
        self.size = self.text_size = self.sequence = 0
        self.finished = self.done = False

    def feed(self, chunk):
        self.size += len(chunk)
        if self.size > 2_000_000:
            raise ValueError("Bounded provisional stream required")
        records = re.split(rb"\r?\n\r?\n", self.pending + chunk)
        self.pending = records.pop()
        events = []
        for record in records:
            data = b"\n".join(line[5:].removeprefix(b" ") for line in record.splitlines()
                              if line.startswith(b"data:"))
            if not data:
                continue
            if self.done:
                raise ValueError("Data after provisional stream terminator")
            if data == b"[DONE]":
                if not self.finished:
                    raise ValueError("Missing provisional stream finish")
                self.done = True
                continue
            item = strict_json(data.decode("utf-8"))
            if not isinstance(item, dict) or not isinstance(item.get("choices"), list):
                raise ValueError("Invalid provisional event")
            choices = item["choices"]
            if not choices:
                continue
            if len(choices) != 1 or self.finished:
                raise ValueError("Multiple or late provisional choices")
            choice = choices[0]
            if not isinstance(choice, dict) or choice.get("index") != 0:
                raise ValueError("Invalid provisional choice index")
            delta = choice.get("delta")
            if (not isinstance(delta, dict) or delta.get("refusal")
                    or delta.get("tool_calls")):
                raise ValueError("Provisional text cannot contain tools or refusals")
            content = delta.get("content")
            if content is not None and not isinstance(content, str):
                raise ValueError("Invalid provisional text")
            terminal = choice.get("finish_reason")
            if terminal is not None:
                if terminal != "stop":
                    raise ValueError("Unsuccessful provisional terminal")
                self.finished = True
            if content:
                self.text_size += len(content.encode())
                if self.text_size > 64000:
                    raise ValueError("Provisional text capacity exceeded")
                self.sequence += 1
                events.append(UnapprovedTextDelta(self.sequence, content))
        return tuple(events)
