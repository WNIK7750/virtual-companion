"""Stable identity for an operation-specific semantic support receipt."""

from hashlib import sha256

from .contracts import canonical
from .memory import MemoryMutation


def command_digest(command, quote):
    if not isinstance(command, MemoryMutation) or not isinstance(quote, str) or not quote.strip():
        raise ValueError("Typed memory command and exact support quote required")
    data = command.to_mapping()
    data.pop("reason")  # The support receipt cannot hash itself.
    return sha256(canonical({"command": data, "quote": quote}).encode()).hexdigest()


def target_digest(target):
    return sha256(canonical(target).encode()).hexdigest() if target is not None else None
