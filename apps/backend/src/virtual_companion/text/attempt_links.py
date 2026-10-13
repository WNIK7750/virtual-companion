"""Immutable crosswalk of boundary and text attempts in existing mirror metadata."""

from __future__ import annotations

import json
from hashlib import sha256

from virtual_companion.boundary.contracts import canonical
from virtual_companion.boundary.sqlite_turns import _locked_time

_PROTOCOL = "text-attempt-link.v1"
_TEXT_PREFIX = "text-attempt-link:v1:text:"
_BOUNDARY_PREFIX = "text-attempt-link:v1:boundary:"
_FIELDS = {
    "protocol", "companion", "input", "attempt", "text_attempt", "session",
    "turn_id", "binding", "boundary_config_sha256",
}


def _text_key(text_attempt_id):
    digest = sha256(text_attempt_id.encode("utf-8")).hexdigest()
    return _TEXT_PREFIX + digest


def _boundary_key(companion, input_id, attempt_id):
    digest = sha256(canonical([companion, input_id, attempt_id]).encode("utf-8")).hexdigest()
    return _BOUNDARY_PREFIX + digest


def _read_metadata(mirror, key):
    row = mirror.connection.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
    return None if row is None else row[0]


def _decode_link(value, text_attempt_id):
    try:
        link = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Attempt link metadata is invalid") from exc
    if (not isinstance(link, dict) or set(link) != _FIELDS or link["protocol"] != _PROTOCOL
            or link["text_attempt"] != text_attempt_id):
        raise ValueError("Attempt link metadata is invalid")
    return link


def load_attempt_link(mirror, text_attempt_id):
    """Read both immutable mirror keys; missing history stays unknown, half-pairs fail closed."""
    if not isinstance(text_attempt_id, str) or not text_attempt_id:
        raise ValueError("Text attempt identity required")
    with mirror.lock:
        value = _read_metadata(mirror, _text_key(text_attempt_id))
        if value is None:
            return None
        link = _decode_link(value, text_attempt_id)
        if mirror.companion_id is not None and link["companion"] != mirror.companion_id:
            raise ValueError("Attempt link companion differs from mirror owner")
        reverse_key = _boundary_key(link["companion"], link["input"], link["attempt"])
        reverse = _read_metadata(mirror, reverse_key)
        expected_reverse = canonical({
            "protocol": _PROTOCOL,
            "companion": link["companion"],
            "input": link["input"],
            "attempt": link["attempt"],
            "text_attempt": text_attempt_id,
        })
        if reverse != expected_reverse:
            raise ValueError("Attempt link crosswalk is incomplete or conflicting")
        return {key: link[key] for key in _FIELDS if key != "protocol"}


def persist_attempt_link(mirror, boundary_store, claim, *, boundary_session_id, turn_id,
                         text_attempt_id, binding, now_us, clock=None):
    """Persist the crosswalk atomically in the existing conversation metadata table.

    The boundary write transaction writes no rows or DDL; it holds the verified live
    claim through the mirror commit. Both mirror indexes commit together.
    """
    if mirror.companion_id is not None and mirror.companion_id != claim.companion_id:
        raise ValueError("Attempt link mirror owner differs from boundary companion")
    if not isinstance(binding, dict):
        raise ValueError("Attempt link version binding required")
    binding_json = canonical(binding)
    config_sha256 = sha256(claim.config_json.encode("utf-8")).hexdigest()
    link = {
        "protocol": _PROTOCOL,
        "companion": claim.companion_id,
        "input": claim.input_id,
        "attempt": claim.attempt_id,
        "text_attempt": text_attempt_id,
        "session": boundary_session_id,
        "turn_id": turn_id,
        "binding": binding_json,
        "boundary_config_sha256": config_sha256,
    }
    link_json = canonical(link)
    reverse_key = _boundary_key(claim.companion_id, claim.input_id, claim.attempt_id)
    reverse_json = canonical({
        "protocol": _PROTOCOL,
        "companion": claim.companion_id,
        "input": claim.input_id,
        "attempt": claim.attempt_id,
        "text_attempt": text_attempt_id,
    })
    text_key = _text_key(text_attempt_id)

    with boundary_store.transaction() as db:
        boundary_store._segment_live(db, claim, _locked_time(now_us, clock))
        source = db.execute(
            "SELECT session,client_key FROM turn_inputs WHERE companion=? AND input=?",
            (claim.companion_id, claim.input_id),
        ).fetchone()
        if (source is None
                or (source["session"], source["client_key"])
                != (boundary_session_id, turn_id)):
            raise ValueError("Live source and attempt link required")

        with mirror.transaction():
            try:
                mirror_attempt = mirror.connection.execute(
                    "SELECT a.binding,a.status FROM attempts a JOIN turns t ON t.id=a.turn_id "
                    "WHERE a.id=? AND a.turn_id=?",
                    (text_attempt_id, turn_id),
                ).fetchone()
            except Exception as exc:
                raise ValueError("Text mirror attempt is unavailable") from exc
            if mirror_attempt is None or mirror_attempt["status"] != "running":
                raise ValueError("Live text mirror attempt required")
            try:
                mirror_binding = json.loads(mirror_attempt["binding"])
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError("Text mirror attempt binding is invalid") from exc
            if mirror_binding != binding:
                raise ValueError("Text mirror attempt binding changed")

            old_link = _read_metadata(mirror, text_key)
            old_reverse = _read_metadata(mirror, reverse_key)
            if old_link is not None or old_reverse is not None:
                if old_link == link_json and old_reverse == reverse_json:
                    return
                raise ValueError("Immutable attempt link conflict")
            mirror.connection.execute("INSERT INTO metadata(key,value) VALUES(?,?)",
                                      (text_key, link_json))
            mirror.connection.execute("INSERT INTO metadata(key,value) VALUES(?,?)",
                                      (reverse_key, reverse_json))
