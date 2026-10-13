from __future__ import annotations

import threading
from dataclasses import replace
from uuid import uuid4

from virtual_companion.feedback.bootstrap import create_feedback_service
from virtual_companion.feedback.errors import MessageOwnershipError
from virtual_companion.feedback.models import VersionBinding, sha256_json

from .feedback_receipts import FeedbackReceiptCatalog


def _redactable_path(path):
    if (len(path) == 3 and path[0] == "feedback" and type(path[1]) is int
            and path[2] == "freeform_reaction"):
        return True
    if (len(path) == 3 and path[0] == "context" and type(path[1]) is int
            and path[2] == "selected_text"):
        return True
    if (len(path) == 4 and path[0] == "context" and type(path[1]) is int
            and path[2] in {"assistant_message"} and path[3] == "content"):
        return True
    return (len(path) == 5 and path[0] == "context" and type(path[1]) is int
            and path[2] in {"preceding_messages", "following_messages"}
            and type(path[3]) is int and path[4] == "content")


def _is_subsequence(candidate, source):
    source_iter = iter(source)
    return all(any(current == char for current in source_iter) for char in candidate)


def _validate_redacted_payload(source, candidate):
    """Allow only deletion or the fixed redaction marker at text-bearing paths."""
    import json

    if not isinstance(source, dict) or not isinstance(candidate, dict):
        raise ValueError("Redacted preview must retain its object shape")
    try:
        copied = json.loads(json.dumps(candidate, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("Redacted preview must be finite JSON") from exc

    def verify(old, new, path):
        if type(old) is not type(new):
            raise ValueError("Redaction cannot change preview value types")
        if isinstance(old, dict):
            if set(old) != set(new):
                raise ValueError("Redaction cannot add or remove preview fields")
            for key in old:
                verify(old[key], new[key], (*path, key))
            return
        if isinstance(old, list):
            if len(old) != len(new):
                raise ValueError("Redaction cannot add or remove preview entries")
            for index, (old_item, new_item) in enumerate(zip(old, new, strict=True)):
                verify(old_item, new_item, (*path, index))
            return
        if isinstance(old, str):
            if old == new:
                return
            if not _redactable_path(path):
                raise ValueError("Only preview text fields may be redacted")
            marker = "[已脱敏]"
            if (new.count(marker) > 16
                    or not _is_subsequence(new.replace(marker, ""), old)):
                raise ValueError(
                    "Preview edits may only remove text or insert the fixed redaction marker"
                )
            return
        if old != new:
            raise ValueError("Redaction cannot change preview anchors or metadata")

    verify(source, copied, ())
    return copied


class CommittedCatalog:
    def __init__(self, store, receipt_store_provider=None):
        self.store = store
        self.receipt_catalog = (None if receipt_store_provider is None else
                                FeedbackReceiptCatalog(store, receipt_store_provider))

    def lookup(self, session_id, turn_id, assistant_message_id):
        with self.store.lock:
            row = self.store.connection.execute(
                """SELECT m.*,t.session_id,a.binding FROM messages m
                JOIN turns t ON t.id=m.turn_id JOIN attempts a ON a.turn_id=t.id
                WHERE m.id=? AND m.turn_id=? AND t.session_id=?
                AND m.role='assistant' AND a.status='completed'""",
                (assistant_message_id, turn_id, session_id),
            ).fetchone()
        if not row:
            raise MessageOwnershipError("Committed assistant message required")
        return dict(row)

    def assert_owned_assistant_message(self, **identity):
        target_binding = identity.pop("target_binding", None)
        if target_binding is not None:
            if self.receipt_catalog is None:
                raise MessageOwnershipError("Authoritative receipt store unavailable")
            self.receipt_catalog.assert_target(**identity, target_binding=target_binding)
            return
        self.lookup(**identity)

    def context_for(self, *, policy, **identity):
        target_binding = identity.pop("target_binding", None)
        if target_binding is not None:
            if self.receipt_catalog is None:
                raise MessageOwnershipError("Authoritative receipt store unavailable")
            return self.receipt_catalog.context_for(**identity, target_binding=target_binding)
        message = self.lookup(**identity)
        with self.store.lock:
            rows = self.store.connection.execute(
                """SELECT m.role,m.content FROM messages m
                JOIN turns t ON t.id=m.turn_id WHERE t.session_id=? AND m.seq<?
                ORDER BY m.seq DESC LIMIT ?""",
                (identity["session_id"], message["seq"], policy.preceding_turns * 2),
            ).fetchall()
        return {
            **identity,
            "preceding_messages": [dict(row) for row in reversed(rows)],
            "assistant_message": {"role": "assistant", "content": message["content"]},
            "following_messages": [],
        }


class FeedbackBridge:
    """Typed adapter for the existing B0-B3 service; previews stay server-owned."""

    def __init__(self, store, receipt_store_provider=None):
        self.store = store
        self.catalog = CommittedCatalog(store, receipt_store_provider)
        self.service, self.repository = create_feedback_service(
            feedback_database_path=store.root / "feedback.sqlite3",
            message_catalog=self.catalog,
            export_root=store.root / "exports",
        )
        self.previews = {}
        self.lock = threading.RLock()

    def dispatch(self, payload):
        import json

        with self.lock:
            operation = payload["operation"]
            if operation == "list":
                return {
                    "events": [
                        event.to_mapping()
                        for event in self.repository.list_current(session_id=self.store.session_id)
                    ]
                }
            if operation == "submit":
                target_binding = None
                identity = {key: payload[key] for key in ("session_id", "turn_id")}
                if "assistant_message_id" in payload:
                    identity["assistant_message_id"] = payload["assistant_message_id"]
                if "target" in payload:
                    if self.catalog.receipt_catalog is None:
                        raise MessageOwnershipError("Authoritative receipt store unavailable")
                    resolved = self.catalog.receipt_catalog.resolve(
                        session_id=identity["session_id"],
                        turn_id=identity["turn_id"],
                        assistant_message_id=identity.get("assistant_message_id"),
                        target=payload["target"],
                    )
                    identity["assistant_message_id"] = resolved["assistant_message_id"]
                    target_binding = resolved["target_binding"]
                    binding = VersionBinding.from_mapping(resolved["version_binding"])
                else:
                    if "assistant_message_id" not in identity:
                        raise MessageOwnershipError("Committed assistant message required")
                    row = self.catalog.lookup(**identity)
                    binding_value = json.loads(row["binding"])["version_binding"]
                    binding = VersionBinding.from_mapping(binding_value)
                return self.service.submit_feedback(
                    **identity,
                    quick_tags=payload["quick_tags"],
                    freeform_reaction=payload["freeform_reaction"],
                    idempotency_key=payload["idempotency_key"],
                    version_binding=binding,
                    target_binding=target_binding,
                ).to_mapping()
            if operation == "revise":
                return self.service.revise_feedback(
                    feedback_id=payload["feedback_id"],
                    expected_revision=payload["expected_revision"],
                    quick_tags=payload["quick_tags"],
                    freeform_reaction=payload["freeform_reaction"],
                ).to_mapping()
            if operation == "withdraw":
                return self.service.withdraw_feedback(
                    feedback_id=payload["feedback_id"],
                    expected_revision=payload["expected_revision"],
                ).to_mapping()
            if operation == "delete":
                # Remove only exports owned by this store and containing this feedback.
                # On failure keep database records so deletion can be retried visibly.
                for candidate in (self.store.root / "exports").glob("*.json"):
                    export = self.repository.get_export(candidate.stem)
                    if export and any(
                        item.feedback_id == payload["feedback_id"]
                        for item in export.selected_feedback
                    ):
                        candidate.unlink()
                self.service.delete_feedback(payload["feedback_id"])
                self.previews.clear()
                return {"deleted": True}
            if operation == "preview":
                if "feedback_ids" in payload:
                    preview = self.service.preview_export(feedback_ids=payload["feedback_ids"])
                    preview_id = str(uuid4())
                    self.previews.clear()  # One bounded pending export per local application.
                    state = {
                        "preview": preview,
                        "revision": 1,
                        "payload_sha256": sha256_json(preview.payload),
                    }
                    self.previews[preview_id] = state
                    return self._preview_mapping(preview_id, state)
                preview_id = payload["preview_id"]
                state = self.previews.get(preview_id)
                if state is None:
                    raise ValueError("Export preview is no longer available")
                if (payload["expected_revision"] != state["revision"]
                        or payload["expected_sha256"] != state["payload_sha256"]
                        or sha256_json(state["preview"].payload) != state["payload_sha256"]):
                    raise ValueError("Export preview revision or hash is stale")
                redacted = _validate_redacted_payload(
                    state["preview"].payload, payload["redacted_payload"])
                state["preview"] = replace(state["preview"], payload=redacted)
                state["revision"] += 1
                state["payload_sha256"] = sha256_json(redacted)
                return self._preview_mapping(preview_id, state)
            if operation == "confirm":
                preview_id = payload["preview_id"]
                state = self.previews.get(preview_id)
                if state is None:
                    raise ValueError("Export preview is no longer available")
                if (payload["preview_revision"] != state["revision"]
                        or payload["payload_sha256"] != state["payload_sha256"]
                        or sha256_json(state["preview"].payload) != state["payload_sha256"]):
                    raise ValueError("Export preview revision or hash is stale")
                exported = self.service.confirm_export(
                    state["preview"],
                    redaction_confirmed=payload["redaction_confirmed"],
                    external_processing_confirmed=payload["external_processing_confirmed"],
                )
                del self.previews[preview_id]
                return {"export_id": exported.export_id, "saved_locally": True}
            raise ValueError("Unknown feedback operation")

    @staticmethod
    def _preview_mapping(preview_id, state):
        return {
            "preview_id": preview_id,
            "preview_revision": state["revision"],
            "payload_sha256": state["payload_sha256"],
            "payload": state["preview"].payload,
        }
