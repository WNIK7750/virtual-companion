"""Authenticated, bounded loopback adapter. No general proxy or filesystem routes."""

from __future__ import annotations

import hmac
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .errors import ContextCheckError, ContextLimitError
from .history import validate_window

ALLOWED = {
    "state",
    "start",
    "history",
    "newSession",
    "send",
    "stop",
    "retry",
    "feedback",
    "sessions",
    "selectSession",
    "selectProvider",
    "memories",
    "memoryDetails",
    "memoryAction",
    "memoryAbandon",
    "streamPoll",
    "streamState",
    "streamAck",
    "streamReceive",
    "streamStop",
    "resumeInputs",
}
FEEDBACK_FIELDS = {
    "list": set(),
    "submit": {
        "session_id",
        "turn_id",
        "assistant_message_id",
        "target",
        "quick_tags",
        "freeform_reaction",
        "idempotency_key",
    },
    "revise": {"feedback_id", "expected_revision", "quick_tags", "freeform_reaction"},
    "withdraw": {"feedback_id", "expected_revision"},
    "delete": {"feedback_id"},
    "preview": {"feedback_ids"},
    "confirm": {"preview_id", "preview_revision", "payload_sha256",
                "redaction_confirmed", "external_processing_confirmed"},
}


def validate(action: str, payload: object) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("Invalid payload")
    if action == "streamPoll":
        if (set(payload) - {"wait_ms"}
                or type(payload.get("wait_ms", 0)) is not int
                or not 0 <= payload.get("wait_ms", 0) <= 1000):
            raise ValueError("Invalid bounded display wait")
        return payload
    if action in {"streamAck", "streamReceive"}:
        if (set(payload) != {"sequence", "fingerprint", "host_receipt"}
                or type(payload['sequence']) is not int or not 1 <= payload['sequence'] <= 32
                or not isinstance(payload['fingerprint'], str)
                or len(payload['fingerprint']) != 64
                or any(c not in '0123456789abcdef' for c in payload['fingerprint'])):
            raise ValueError("Invalid display receipt")
        _memory_identifier(payload['host_receipt'])
        return payload
    if action == "history":
        if set(payload) - {"before", "after", "session_id"}:
            raise ValueError("Invalid history fields")
        validate_window(**payload)
        return payload
    if action == "sessions":
        offset = payload.get("offset", 0)
        if set(payload) - {"offset"} or type(offset) is not int or not 0 <= offset <= 10**9:
            raise ValueError("Invalid session offset")
        return {"offset": offset}
    if action == "memories":
        if set(payload) - {"offset", "limit"}:
            raise ValueError("Invalid memory page fields")
        offset, limit = payload.get("offset", 0), payload.get("limit", 50)
        if (
            type(offset) is not int
            or not 0 <= offset <= 10**6
            or type(limit) is not int
            or not 1 <= limit <= 100
        ):
            raise ValueError("Invalid memory page")
        return {"offset": offset, "limit": limit}
    if action == "memoryDetails":
        if set(payload) != {"record_id"}:
            raise ValueError("Invalid memory detail fields")
        _memory_identifier(payload["record_id"])
        return payload
    if action == "memoryAction":
        required = {
            "operation_id",
            "action",
            "record_id",
            "expected_version",
            "replacement",
        }
        if not required <= set(payload) or set(payload) - required - {"resolves_job"}:
            raise ValueError("Invalid memory action fields")
        _memory_identifier(payload["operation_id"])
        _memory_identifier(payload["record_id"])
        if "resolves_job" in payload:
            _memory_identifier(payload["resolves_job"])
        if payload["action"] not in {"correct", "forget"}:
            raise ValueError("Invalid memory action")
        if type(payload["expected_version"]) is not int or payload["expected_version"] < 1:
            raise ValueError("Invalid memory version")
        replacement = payload["replacement"]
        if not isinstance(replacement, str) or len(replacement.encode("utf-8")) > 16000:
            raise ValueError("Invalid memory replacement")
        if (payload["action"] == "correct" and not replacement.strip()) or (
            payload["action"] == "forget" and replacement
        ):
            raise ValueError("Invalid memory replacement")
        return payload
    if action == "memoryAbandon":
        if set(payload) != {"job", "operation_id"}:
            raise ValueError("Invalid memory abandonment fields")
        _memory_identifier(payload["job"])
        _memory_identifier(payload["operation_id"])
        return payload
    if action == "feedback":
        operation = payload.get("operation")
        if not isinstance(operation, str) or operation not in FEEDBACK_FIELDS:
            raise ValueError("Invalid feedback operation")
        if operation == "preview" and set(payload) == {
                "operation", "preview_id", "expected_revision", "expected_sha256",
                "redacted_payload"}:
            required = {"operation", "preview_id", "expected_revision",
                        "expected_sha256", "redacted_payload"}
        else:
            required = FEEDBACK_FIELDS[operation] | {"operation"}
        allowed = required
        if operation == "submit":
            required = required - {"assistant_message_id", "target"}
            allowed = allowed | {"assistant_message_id", "target"}
        if not required <= set(payload) or set(payload) - allowed:
            raise ValueError("Invalid feedback fields")
        if operation == "preview" and "redacted_payload" in payload:
            candidate = payload["redacted_payload"]
            if (not isinstance(candidate, dict)
                    or set(candidate) != {"schema_version", "feedback", "context"}
                    or type(candidate["schema_version"]) is not int
                    or not isinstance(candidate["feedback"], list)
                    or not isinstance(candidate["context"], list)
                    or len(candidate["feedback"]) > 30 or len(candidate["context"]) > 30
                    or len(json.dumps(candidate, ensure_ascii=False).encode("utf-8")) > 40000):
                raise ValueError("Invalid redacted preview")
        if operation == "preview" and "feedback_ids" in payload:
            if (not isinstance(payload["feedback_ids"], list)
                    or not payload["feedback_ids"] or len(payload["feedback_ids"]) > 30):
                raise ValueError("Invalid preview feedback IDs")
        if operation == "submit":
            if payload.get("target") is None and "assistant_message_id" not in payload:
                raise ValueError("Message or attempt feedback target required")
            target = payload.get("target")
            if target is not None:
                if not isinstance(target, dict):
                    raise ValueError("Invalid feedback target")
                kind = target.get("kind")
                if kind == "attempt":
                    if (set(target) != {"kind", "attempt_id"}
                            or not isinstance(target["attempt_id"], str)
                            or not 1 <= len(target["attempt_id"]) <= 128):
                        raise ValueError("Invalid attempt feedback target")
                    if "assistant_message_id" in payload:
                        raise ValueError("Attempt feedback cannot name an assistant message")
                elif kind in {"reply", "bubble"}:
                    if (set(target) != {"kind", "start", "end"}
                            or type(target.get("start")) is not int
                            or type(target.get("end")) is not int
                            or target["start"] < 0 or target["end"] <= target["start"]
                            or not isinstance(payload.get("assistant_message_id"), str)):
                        raise ValueError("Invalid ranged feedback target")
                else:
                    raise ValueError("Invalid feedback target kind")
        for key, value in payload.items():
            if key.endswith("_confirmed") and type(value) is not bool:
                raise ValueError("Boolean confirmation required")
            if key in {"expected_revision", "preview_revision"} and (
                    type(value) is not int or value < 1):
                raise ValueError("Invalid revision")
            if key in {"expected_sha256", "payload_sha256"} and (
                    not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)):
                raise ValueError("Invalid preview hash")
            if key in {"feedback_ids", "quick_tags"}:
                if not isinstance(value, list) or len(value) > 30:
                    raise ValueError("Invalid list")
                if any(not isinstance(item, str) or len(item) > 128 for item in value):
                    raise ValueError("Invalid list item")
            elif key.endswith("_id") or key in {"idempotency_key", "freeform_reaction"}:
                if not isinstance(value, str) or len(value) > 8000:
                    raise ValueError("Invalid text")
        return payload
    fields = (
        ({"text", "receiving_context"} if "receiving_context" in payload else {"text"})
        if action == "send"
        else {"session_id"}
        if action == "selectSession"
        else ({"provider_id", "confirmed"} if "confirmed" in payload else {"provider_id"})
        if action == "selectProvider"
        else set()
    )
    if set(payload) != fields:
        raise ValueError("Invalid payload")
    if action == "selectSession" and (
        not isinstance(payload["session_id"], str) or len(payload["session_id"]) > 128
    ):
        raise ValueError("Invalid session")
    if action == "selectProvider" and (
        not isinstance(payload["provider_id"], str)
        or not payload["provider_id"]
        or len(payload["provider_id"]) > 64
        or ("confirmed" in payload and type(payload["confirmed"]) is not bool)
    ):
        raise ValueError("Invalid provider")
    if action == "send":
        if 'receiving_context' in payload:
            from virtual_companion.boundary.receiving_context import validate_receiving_context

            validate_receiving_context(payload['receiving_context'])
        text = payload["text"]
        if not isinstance(text, str) or not text.strip() or len(text) > 8000:
            raise ValueError("Invalid payload")
    return payload


def _memory_identifier(value):
    if not isinstance(value, str) or not value or len(value) > 128:
        raise ValueError("Invalid memory identifier")
    return value


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate field")
        result[key] = value
    return result


def create_server(
    token: str, dispatch, *, port: int = 18084, actions=None, validate_payload=validate
) -> ThreadingHTTPServer:
    if len(token) < 32 or not token.isascii():
        raise ValueError("Strong per-run token required")
    allowed = frozenset(ALLOWED if actions is None else actions)
    if not callable(validate_payload) or any(not isinstance(a, str) for a in allowed):
        raise ValueError("Explicit host routes and validator required")

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(5)

        def log_message(self, *_):
            pass  # Never log request data, headers, or token.

        def respond(self, code, payload):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

        def discard_bounded_body(self):
            raw_size = self.headers.get("Content-Length")
            try:
                size = int(raw_size) if raw_size is not None else 0
            except ValueError:
                return
            if 0 <= size <= 48000:
                self.rfile.read(size)

        def do_POST(self):  # noqa: N802
            credential = self.headers.get("Authorization", "")
            if not credential.isascii() or not hmac.compare_digest(credential, "Bearer " + token):
                self.discard_bounded_body()
                self.respond(401, {"error": "Unauthorized"})
                return
            if self.headers.get("Origin") or self.headers.get("Transfer-Encoding"):
                self.discard_bounded_body()
                self.respond(403, {"error": "Browser requests not permitted"})
                return
            action = self.path.removeprefix("/v1/")
            if self.path != "/v1/" + action or action not in allowed:
                self.discard_bounded_body()
                self.respond(404, {"error": "Unknown route"})
                return
            try:
                if self.headers.get("Content-Type") != "application/json":
                    raise ValueError("JSON required")
                size = int(self.headers.get("Content-Length", "0"))
                if not 1 <= size <= 48000:
                    raise ValueError("Invalid request size")
                body = json.loads(self.rfile.read(size), object_pairs_hook=unique_object)
                result = dispatch(action, validate_payload(action, body))
            except (ContextLimitError, ContextCheckError) as error:
                self.respond(409, {"error": str(error)})
            except (ValueError, UnicodeError):
                self.respond(400, {"error": "Invalid request"})
            except Exception:
                self.respond(409, {"error": "操作未完成，请检查状态后重试。"})
            else:
                self.respond(200, result)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server
