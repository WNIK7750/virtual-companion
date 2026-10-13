"""Read-only binding of feedback targets to immutable stream receipts."""

from __future__ import annotations

import hashlib
import json
import sqlite3

from virtual_companion.boundary.sqlite_segments import SQLiteSegmentStore
from virtual_companion.feedback.errors import MessageOwnershipError, ValidationError

from .attempt_links import load_attempt_link


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _digest_json(value):
    return _digest(_canonical(value))


def _table(db, name):
    return db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


class FeedbackReceiptCatalog:
    """Resolve a renderer selector against the mirror and the boundary store.

    The renderer supplies only an attempt selector or a code-point range. All
    text, attempt/version identity, delivery status and receipt hashes are
    resolved again at submit and export time.
    """

    def __init__(self, mirror, receipt_store_provider):
        self.mirror = mirror
        self.receipt_store_provider = receipt_store_provider

    def resolve(self, *, session_id, turn_id, assistant_message_id, target):
        kind = target.get("kind") if isinstance(target, dict) else None
        if kind == "attempt":
            if set(target) != {"kind", "attempt_id"} or not isinstance(target["attempt_id"], str):
                raise ValidationError("Exact attempt feedback selector required")
            return self._resolve_attempt(session_id, turn_id, target["attempt_id"])
        if kind not in {"reply", "bubble"} or set(target) != {"kind", "start", "end"}:
            raise ValidationError("Exact message feedback range required")
        if (type(target["start"]) is not int or type(target["end"]) is not int
                or target["start"] < 0 or target["end"] <= target["start"]):
            raise ValidationError("Feedback range must use ordered code-point offsets")
        message, attempts = self._mirror_message(session_id, turn_id, assistant_message_id)
        points = list(message["content"])
        start, end = target["start"], target["end"]
        if end > len(points) or (kind == "reply" and (start != 0 or end != len(points))):
            raise MessageOwnershipError("Feedback range is outside the stored reply")
        selected = "".join(points[start:end])
        matches = []
        for attempt in attempts:
            link = self._link_for_attempt(turn_id, attempt)
            if link is None:
                continue
            snapshot = self._attempt_snapshot(link)
            if snapshot["visible_prefix"] == message["content"] and snapshot["visible_prefix"]:
                matches.append((attempt, link, snapshot))
        if len(matches) > 1:
            raise MessageOwnershipError("Visible reply maps to multiple attempts")
        if matches:
            attempt, link, snapshot = matches[0]
            attempt = {**attempt, "message_id": assistant_message_id}
            return self._binding(kind, start, end, message["content"], selected,
                                 attempt, link, snapshot, visible=True)
        # Historical full replies can still receive feedback. Their exact mirror
        # attempt is retained, while receipt and visibility facts stay unknown.
        completed = [attempt for attempt in attempts if attempt["status"] == "completed"]
        if len(completed) != 1:
            raise MessageOwnershipError("Unique historical completed attempt required")
        completed[0]["message_id"] = assistant_message_id
        return self._binding(kind, start, end, message["content"], selected,
                             completed[0], None, None, visible=None)

    def _resolve_attempt(self, session_id, turn_id, text_attempt_id):
        attempts = self._mirror_attempts(session_id, turn_id)
        rows = [attempt for attempt in attempts if attempt["id"] == text_attempt_id]
        if len(rows) != 1:
            raise MessageOwnershipError("Exact mirror attempt required")
        attempt = rows[0]
        link = self._link_for_attempt(turn_id, attempt)
        if link is None:
            return self._binding("attempt", None, None, None, None,
                                 attempt, None, None, visible=None)
        snapshot = self._attempt_snapshot(link)
        return self._binding("attempt", None, None, None, None,
                             attempt, link, snapshot, visible=bool(snapshot["visible_prefix"]))

    def context_for(self, *, session_id, turn_id, assistant_message_id, target_binding):
        if not isinstance(target_binding, dict) or target_binding.get("schema_version") != 1:
            raise MessageOwnershipError("Versioned feedback target required")
        if target_binding.get("status") != "visible":
            return {
                "session_id": session_id,
                "turn_id": turn_id,
                "assistant_message_id": assistant_message_id,
                "target_binding": target_binding,
                "selected_text": None,
            }
        attempt = self._mirror_attempt_by_id(session_id, turn_id,
                                             target_binding["attempt"]["text_attempt_id"])
        link = self._link_for_attempt(turn_id, attempt)
        if (link is None or link["input"] != target_binding["attempt"]["input_id"]
                or link["attempt"] != target_binding["attempt"]["boundary_attempt_id"]):
            raise MessageOwnershipError("Feedback attempt link changed")
        snapshot = self._attempt_snapshot(link)
        sequence = target_binding.get("visible_sequence")
        source = snapshot["prefix_by_sequence"].get(sequence)
        if source is None or _digest(source) != target_binding["source_sha256"]:
            raise MessageOwnershipError("Feedback source snapshot is no longer verifiable")
        if (snapshot["receipt_refs"][:len(target_binding["receipt_refs"])]
                != target_binding["receipt_refs"]):
            raise MessageOwnershipError("Feedback receipt snapshot changed")
        selected = self._selection(source, target_binding["range"])
        if _digest(selected) != target_binding["selection_sha256"]:
            raise MessageOwnershipError("Feedback selection hash changed")
        return {
            "session_id": session_id,
            "turn_id": turn_id,
            "assistant_message_id": assistant_message_id,
            "target_binding": target_binding,
            "selected_text": selected,
        }

    def assert_target(self, *, session_id, turn_id, assistant_message_id, target_binding):
        # Reuse the exact export lookup path before accepting the target. A
        # textless attempt target is authorized solely by its immutable map.
        if target_binding.get("kind") == "attempt":
            attempt = self._mirror_attempt_by_id(
                session_id, turn_id, target_binding["attempt"]["text_attempt_id"])
            link = self._link_for_attempt(turn_id, attempt)
            expected = target_binding["attempt"]
            if link is not None and (link["input"] != expected["input_id"]
                                     or link["attempt"] != expected["boundary_attempt_id"]):
                raise MessageOwnershipError("Feedback attempt changed")
            if link is None and (expected["input_id"] is not None
                                 or expected["boundary_attempt_id"] is not None):
                raise MessageOwnershipError("Feedback attempt link is unavailable")
            if link is not None:
                snapshot = self._attempt_snapshot(link)
                if snapshot["receipt_snapshot_sha256"] != target_binding["receipt_snapshot_sha256"]:
                    raise MessageOwnershipError("Feedback receipt snapshot changed")
            return
        message, attempts = self._mirror_message(session_id, turn_id, assistant_message_id)
        expected = target_binding["attempt"]
        attempt = next(
            (item for item in attempts if item["id"] == expected["text_attempt_id"]), None
        )
        if attempt is None or _digest(message["content"]) != target_binding["source_sha256"]:
            raise MessageOwnershipError("Feedback message snapshot changed")
        selected = self._selection(message["content"], target_binding["range"])
        if _digest(selected) != target_binding["selection_sha256"]:
            raise MessageOwnershipError("Feedback message range changed")
        if target_binding.get("status") == "visible":
            link = self._link_for_attempt(turn_id, attempt)
            if (link is None or link["input"] != expected["input_id"]
                    or link["attempt"] != expected["boundary_attempt_id"]):
                raise MessageOwnershipError("Visible feedback attempt link changed")
            snapshot = self._attempt_snapshot(link)
            if snapshot["receipt_snapshot_sha256"] != target_binding["receipt_snapshot_sha256"]:
                raise MessageOwnershipError("Feedback receipt snapshot changed")
            if (snapshot["prefix_by_sequence"].get(target_binding["visible_sequence"])
                    != message["content"]):
                raise MessageOwnershipError("Visible reply prefix changed")

    def _mirror_message(self, session_id, turn_id, assistant_message_id):
        with self.mirror.lock:
            row = self.mirror.connection.execute(
                "SELECT m.id,m.content FROM messages m JOIN turns t ON t.id=m.turn_id "
                "WHERE m.id=? AND m.turn_id=? AND t.session_id=? AND m.role='assistant'",
                (assistant_message_id, turn_id, session_id),
            ).fetchone()
            if row is None:
                raise MessageOwnershipError("Owned assistant message required")
            attempts = self.mirror.connection.execute(
                "SELECT id,turn_id,status,binding,seq FROM attempts WHERE turn_id=? ORDER BY seq",
                (turn_id,),
            ).fetchall()
        return dict(row), [{**dict(item), "session_id": session_id} for item in attempts]

    def _mirror_attempts(self, session_id, turn_id):
        with self.mirror.lock:
            exists = self.mirror.connection.execute(
                "SELECT 1 FROM turns WHERE id=? AND session_id=?", (turn_id, session_id)
            ).fetchone()
            if exists is None:
                raise MessageOwnershipError("Owned conversation turn required")
            rows = self.mirror.connection.execute(
                "SELECT id,turn_id,status,binding,seq FROM attempts WHERE turn_id=? ORDER BY seq",
                (turn_id,),
            ).fetchall()
        return [{**dict(row), "session_id": session_id} for row in rows]

    def _mirror_attempt_by_id(self, session_id, turn_id, attempt_id):
        matches = [item for item in self._mirror_attempts(session_id, turn_id)
                   if item["id"] == attempt_id]
        if len(matches) != 1:
            raise MessageOwnershipError("Feedback mirror attempt is unavailable")
        return matches[0]

    def _link_for_attempt(self, turn_id, attempt):
        store = self.receipt_store_provider()
        if store is None:
            return None
        try:
            link = load_attempt_link(self.mirror, attempt["id"])
        except ValueError as exc:
            raise MessageOwnershipError("Feedback attempt crosswalk is invalid") from exc
        if link is None:
            return None
        if link["turn_id"] != turn_id:
            raise MessageOwnershipError("Feedback attempt crosswalk changed")
        if (self.mirror.companion_id is not None
                and link["companion"] != self.mirror.companion_id):
            raise MessageOwnershipError("Feedback attempt crosswalk changed")
        with store.transaction(write=False) as db:
            source = db.execute(
                "SELECT session,client_key FROM turn_inputs WHERE companion=? AND input=?",
                (link["companion"], link["input"]),
            ).fetchone()
            if (source is None or source["client_key"] != turn_id
                    or link["turn_id"] != turn_id or link["session"] != source["session"]):
                raise MessageOwnershipError("Feedback source crosswalk changed")
            try:
                binding = json.loads(link["binding"])
                attempt_binding = json.loads(attempt["binding"])
            except (TypeError, json.JSONDecodeError) as exc:
                raise MessageOwnershipError("Feedback attempt binding is invalid") from exc
            if binding != attempt_binding:
                raise MessageOwnershipError("Feedback mirror version binding changed")
            link["mirror_binding"] = binding
            link["boundary_session"] = source["session"]
            return link

    def _attempt_snapshot(self, link):
        store = self.receipt_store_provider()
        if store is None:
            raise MessageOwnershipError("Authoritative receipt store unavailable")
        with store.transaction(write=False) as db:
            source = db.execute(
                "SELECT session,client_key,config FROM turn_inputs WHERE companion=? AND input=?",
                (link["companion"], link["input"]),
            ).fetchone()
            if (source is None or source["client_key"] != link["turn_id"]
                    or source["session"] != link["session"]):
                raise MessageOwnershipError("Boundary input no longer matches turn link")
            try:
                segments = db.execute(
                    "SELECT * FROM stream_segments WHERE companion=? AND input=? ORDER BY sequence",
                    (link["companion"], link["input"]),
                ).fetchall() if _table(db, "stream_segments") else []
            except sqlite3.Error as exc:
                raise MessageOwnershipError("Boundary segment snapshot unavailable") from exc
            received_available = _table(db, "stream_received")
            history_available = _table(db, "stream_history")
            terminal_available = _table(db, "stream_terminals")
            attempt_history_available = _table(db, "turn_attempt_history")
            receipt_refs = []
            prefix_by_sequence = {}
            visible_prefix = ""
            expected_sequence = 1
            received_any = False
            terminal = None
            if terminal_available:
                terminal = db.execute(
                    "SELECT basis,outcome,receipt FROM stream_terminals "
                    "WHERE companion=? AND input=?",
                    (link["companion"], link["input"]),
                ).fetchone()
                if terminal is not None:
                    terminal_basis = json.loads(terminal["basis"])
                    if terminal_basis.get("attempt_id") == link["attempt"]:
                        self._check_basis(link, terminal_basis, source["session"])
                    else:
                        terminal = None
            attempt_history = None
            if attempt_history_available:
                attempt_history = db.execute(
                    "SELECT fingerprint,record FROM turn_attempt_history "
                    "WHERE companion=? AND input=? AND attempt=?",
                    (link["companion"], link["input"], link["attempt"]),
                ).fetchone()
                if attempt_history is not None:
                    record = json.loads(attempt_history["record"])
                    if (_digest(_canonical(record)) != attempt_history["fingerprint"]
                            or record.get("attempt") != link["attempt"]
                            or _digest(record.get("config", "")) != link["boundary_config_sha256"]):
                        raise MessageOwnershipError("Archived attempt basis changed")
            current_run = db.execute(
                "SELECT attempt FROM turn_runs WHERE companion=? AND input=?",
                (link["companion"], link["input"]),
            ).fetchone()
            if (not segments and attempt_history is None and terminal is None
                    and current_run is not None and current_run["attempt"] == link["attempt"]):
                retry_config = source["config"]
                if _table(db, "turn_retry_bindings"):
                    row = db.execute(
                        "SELECT config FROM turn_retry_bindings WHERE companion=? AND input=?",
                        (link["companion"], link["input"]),
                    ).fetchone()
                    if row is not None:
                        retry_config = row["config"]
                if _digest(retry_config) != link["boundary_config_sha256"]:
                    raise MessageOwnershipError("Current boundary config hash changed")
            for raw in segments:
                try:
                    event = SQLiteSegmentStore._segment_event(raw)
                except Exception as exc:
                    raise MessageOwnershipError("Approved segment manifest is invalid") from exc
                basis = event["basis"]
                if basis.get("attempt_id") != link["attempt"]:
                    continue
                self._check_basis(link, basis, source["session"])
                received = None
                if received_available:
                    received_row = db.execute(
                        "SELECT fingerprint,host_receipt FROM stream_received "
                        "WHERE companion=? AND input=? AND sequence=?",
                        (link["companion"], link["input"], event["sequence"]),
                    ).fetchone()
                    if received_row is not None:
                        if received_row["fingerprint"] != event["fingerprint"]:
                            raise MessageOwnershipError("Received receipt fingerprint changed")
                        received = True
                        received_any = True
                history = None
                if history_available:
                    history = db.execute(
                        "SELECT source,text FROM stream_history WHERE companion=? AND input=? "
                        "AND sequence=?",
                        (link["companion"], link["input"], event["sequence"]),
                    ).fetchone()
                visible = event["status"] == "confirmed" and history is not None
                if visible and (history["source"] != "segment-" + event["fingerprint"]
                                or history["text"] != event["text"]):
                    raise MessageOwnershipError("Visible history does not match approved unit")
                if event["status"] == "confirmed" and history_available and not visible:
                    raise MessageOwnershipError("Confirmed unit lacks matching history")
                if history is not None and not visible:
                    raise MessageOwnershipError("History contains an unconfirmed unit")
                receipt_refs.append({
                    "sequence": event["sequence"],
                    "fingerprint": event["fingerprint"],
                    "status": event["status"],
                    "received": received,
                    "visible": visible,
                    "host_receipt_sha256": (None if event["host_receipt"] is None
                                             else _digest(event["host_receipt"])),
                })
                if visible and event["sequence"] == expected_sequence:
                    visible_prefix = event["display_text"]
                    prefix_by_sequence[event["sequence"]] = visible_prefix
                    expected_sequence += 1
                else:
                    # Visible units after a gap are preserved as receipts but do
                    # not extend the contiguous readable range.
                    continue
            basis_evidence = {
                "companion": link["companion"], "input": link["input"],
                "attempt": link["attempt"], "text_attempt": link["text_attempt"],
                "turn_id": link["turn_id"], "session": link["session"],
                "boundary_session": source["session"],
                "boundary_config_sha256": link["boundary_config_sha256"],
                "mirror_binding": link["mirror_binding"],
                "receipts": receipt_refs,
                "terminal": (None if terminal is None else {
                    "outcome": terminal["outcome"], "receipt": json.loads(terminal["receipt"]),
                }),
                "attempt_history": (None if attempt_history is None
                                    else json.loads(attempt_history["record"])),
            }
            return {
                "receipt_refs": receipt_refs,
                "prefix_by_sequence": prefix_by_sequence,
                "visible_prefix": visible_prefix,
                "visible": bool(visible_prefix),
                "received": (received_any if received_available else None),
                "receipt_snapshot_sha256": _digest_json(basis_evidence),
                "visible_sequence": max(prefix_by_sequence, default=None),
            }

    @staticmethod
    def _check_basis(link, basis, boundary_session):
        if (basis.get("companion_id") != link["companion"]
                or basis.get("input_id") != link["input"]
                or basis.get("attempt_id") != link["attempt"]
                or basis.get("session_id") != boundary_session
                or not isinstance(basis.get("config_json"), str)
                or _digest(basis["config_json"]) != link["boundary_config_sha256"]):
            raise MessageOwnershipError("Boundary attempt basis does not match crosswalk")

    def _binding(self, kind, start, end, source_text, selected, attempt, link, snapshot,
                 *, visible):
        mirror_binding = json.loads(attempt["binding"])
        mapped = link is not None and snapshot is not None
        target_status = ("visible" if visible is True else
                         "attempt-only" if kind == "attempt" and mapped else "unknown")
        range_value = (None if kind == "attempt" else {
            "unit": "unicode-code-point", "start": start, "end": end,
        })
        anchor = {
            "schema_version": 1,
            "kind": kind,
            "status": target_status,
            "attempt": {
                "input_id": None if link is None else link["input"],
                "boundary_attempt_id": None if link is None else link["attempt"],
                "text_attempt_id": attempt["id"],
            },
            "range": range_value,
            "source_sha256": None if source_text is None else _digest(source_text),
            "selection_sha256": None if selected is None else _digest(selected),
            "visible_sequence": None if snapshot is None else snapshot["visible_sequence"],
            "receipt_refs": ([] if snapshot is None else
                             [ref for ref in snapshot["receipt_refs"]
                              if kind == "attempt" or not visible
                              or ref["sequence"] <= snapshot["visible_sequence"]]),
            "receipt_snapshot_sha256": (_digest_json({
                "session_id": attempt.get("session_id"), "turn_id": attempt["turn_id"],
                "text_attempt_id": attempt["id"], "mirror_binding": mirror_binding,
                "status": target_status,
            }) if snapshot is None else snapshot["receipt_snapshot_sha256"]),
            "received": None if snapshot is None else snapshot["received"],
            "visible": (None if snapshot is None and visible is None else
                        bool(snapshot and snapshot["visible"])),
            "read": "unknown",
            "audio": "unknown",
        }
        return {
            "assistant_message_id": None if kind == "attempt" else attempt.get("message_id"),
            "version_binding": mirror_binding.get("version_binding"),
            "target_binding": anchor,
        }

    @staticmethod
    def _selection(source, range_value):
        points = list(source)
        start, end = range_value["start"], range_value["end"]
        if range_value.get("unit") != "unicode-code-point" or not 0 <= start < end <= len(points):
            raise MessageOwnershipError("Stored feedback code-point range is invalid")
        return "".join(points[start:end])
