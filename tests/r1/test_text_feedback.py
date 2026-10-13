from __future__ import annotations

import json
import sqlite3

import pytest
from test_rebuilt_turns import A, claim, config
from test_segment_store import reviewed
from virtual_companion.boundary.presentation import semantic_bubble_ranges
from virtual_companion.boundary.sqlite_received import SQLiteReceivedStore
from virtual_companion.feedback.errors import IdempotencyConflictError, MessageOwnershipError
from virtual_companion.feedback.models import sha256_json
from virtual_companion.feedback.repositories.sqlite import SQLiteFeedbackRepository
from virtual_companion.text.attempt_links import persist_attempt_link
from virtual_companion.text.feedback import FeedbackBridge
from virtual_companion.text.sqlite_store import SQLiteConversationStore
from virtual_companion.text.transport import validate


def test_feedback_transport_rejects_unknown_fields_and_accepts_typed_operations():
    assert validate("feedback", {"operation": "list"}) == {"operation": "list"}
    legacy = {
        "operation": "submit", "session_id": "session", "turn_id": "turn",
        "assistant_message_id": "message", "quick_tags": [],
        "freeform_reaction": "ok", "idempotency_key": "legacy",
    }
    assert validate("feedback", legacy) == legacy
    attempt = {**legacy, "target": {"kind": "attempt", "attempt_id": "attempt"}}
    del attempt["assistant_message_id"]
    assert validate("feedback", attempt) == attempt
    with pytest.raises(ValueError):
        validate("feedback", {**legacy, "target": {"kind": "attempt", "attempt_id": "attempt"}})
    with pytest.raises(ValueError):
        validate("feedback", {**attempt, "target": {"kind": "bubble", "start": 0, "end": 2}})
    with pytest.raises(ValueError):
        validate("feedback", {**legacy, "target": {
            "kind": "bubble", "start": 0, "end": 2, "text": "leak",
        }})
    with pytest.raises(ValueError):
        validate("feedback", {"operation": "execute", "path": "private"})
    with pytest.raises(ValueError):
        validate("feedback", {"operation": "delete", "feedback_id": "id", "destination": "x"})


def test_feedback_transport_binds_preview_edit_and_confirmation_to_hash_and_revision():
    payload = {
        "operation": "preview", "preview_id": "preview", "expected_revision": 2,
        "expected_sha256": "a" * 64,
        "redacted_payload": {"schema_version": 1, "feedback": [], "context": []},
    }
    assert validate("feedback", payload) == payload
    confirm = {
        "operation": "confirm", "preview_id": "preview", "preview_revision": 3,
        "payload_sha256": "b" * 64, "redaction_confirmed": True,
        "external_processing_confirmed": True,
    }
    assert validate("feedback", confirm) == confirm
    for invalid in (
        {**payload, "expected_sha256": "A" * 64},
        {**payload, "expected_revision": True},
        {**payload, "redacted_payload": {
            "schema_version": 1, "feedback": [], "context": [], "extra": [],
        }},
        {**confirm, "payload_sha256": "short"},
        {**confirm, "preview_revision": True},
    ):
        with pytest.raises(ValueError):
            validate("feedback", invalid)


def test_feedback_repository_migrates_legacy_rows_as_unknown_scope(tmp_path):
    path = tmp_path / "feedback.sqlite3"
    binding = {
        "app_commit": "a" * 40, "schema_version": 1, "persona_sha256": "b" * 64,
        "prompt_sha256": "c" * 64, "model_sha256": "d" * 64,
        "server_sha256": "e" * 64, "sampling_profile_id": "synthetic-v1",
    }
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE feedback_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO feedback_meta VALUES ('schema_version', '1');
            CREATE TABLE feedback_events (
                feedback_id TEXT NOT NULL, feedback_revision INTEGER NOT NULL,
                session_id TEXT NOT NULL, turn_id TEXT NOT NULL,
                assistant_message_id TEXT NOT NULL, quick_tags_json TEXT NOT NULL,
                freeform_reaction TEXT, supersedes_revision INTEGER, status TEXT NOT NULL,
                created_at TEXT NOT NULL, schema_version INTEGER NOT NULL,
                version_binding_json TEXT NOT NULL,
                PRIMARY KEY (feedback_id, feedback_revision)
            );
        """)
        connection.execute(
            "INSERT INTO feedback_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("legacy", 1, "session", "turn", "message", "[]", "old feedback", None,
             "active", "2026-10-11T00:00:00+00:00", 1, json.dumps(binding)),
        )
    repository = SQLiteFeedbackRepository(path)
    try:
        event = repository.current("legacy")
        assert event is not None
        assert event.schema_version == 1
        assert event.target_binding is None
        assert event.assistant_message_id == "message"
        columns = {row["name"] for row in repository._connection.execute(
            "PRAGMA table_info(feedback_events)")}
        assert "target_binding_json" in columns
    finally:
        repository.close()


def test_committed_catalog_feedback_binding_and_preview_export(tmp_path):
    store = SQLiteConversationStore(tmp_path)
    binding = {
        "app_commit": "a" * 40,
        "schema_version": 1,
        "persona_sha256": "b" * 64,
        "prompt_sha256": "c" * 64,
        "model_sha256": "d" * 64,
        "server_sha256": "e" * 64,
        "sampling_profile_id": "synthetic-v1",
    }
    attempt = store.begin("synthetic question", {"version_binding": binding})
    bridge = FeedbackBridge(store)
    with pytest.raises(MessageOwnershipError):
        bridge.catalog.lookup(store.session_id, attempt["turn_id"], "missing")
    store.finish(attempt["attempt_id"], "completed", "synthetic answer")
    message = store.history()["messages"][-1]
    event = bridge.dispatch(
        {
            "operation": "submit",
            "session_id": store.session_id,
            "turn_id": message["turn_id"],
            "assistant_message_id": message["id"],
            "quick_tags": [],
            "freeform_reaction": "synthetic feedback",
            "idempotency_key": "one",
        }
    )
    assert event["version_binding"] == binding
    preview = bridge.dispatch({"operation": "preview", "feedback_ids": [event["feedback_id"]]})
    assert preview["preview_revision"] == 1
    export = bridge.dispatch(
        {
            "operation": "confirm",
            "preview_id": preview["preview_id"],
            "preview_revision": preview["preview_revision"],
            "payload_sha256": preview["payload_sha256"],
            "redaction_confirmed": True,
            "external_processing_confirmed": True,
        }
    )
    assert export["saved_locally"]
    assert (tmp_path / "exports" / (export["export_id"] + ".json")).is_file()
    assert "feedback" not in str(store.history())
    bridge.dispatch({"operation": "delete", "feedback_id": event["feedback_id"]})
    assert not (tmp_path / "exports" / (export["export_id"] + ".json")).exists()
    bridge.repository.close()
    store.close()


def test_feedback_preview_redaction_is_monotonic_and_final_hash_is_confirmed(tmp_path):
    store = SQLiteConversationStore(tmp_path)
    binding = {
        "app_commit": "a" * 40, "schema_version": 1, "persona_sha256": "b" * 64,
        "prompt_sha256": "c" * 64, "model_sha256": "d" * 64,
        "server_sha256": "e" * 64, "sampling_profile_id": "synthetic-v1",
    }
    attempt = store.begin("private synthetic question", {"version_binding": binding})
    store.finish(attempt["attempt_id"], "completed", "private synthetic answer")
    message = store.history()["messages"][-1]
    bridge = FeedbackBridge(store)
    try:
        event = bridge.dispatch({
            "operation": "submit", "session_id": store.session_id,
            "turn_id": message["turn_id"], "assistant_message_id": message["id"],
            "quick_tags": [], "freeform_reaction": "private reaction",
            "idempotency_key": "redaction-test",
        })
        preview = bridge.dispatch({"operation": "preview", "feedback_ids": [event["feedback_id"]]})
        assert preview["preview_revision"] == 1
        assert preview["payload_sha256"] == sha256_json(preview["payload"])
        edited = json.loads(json.dumps(preview["payload"], ensure_ascii=False))
        edited["feedback"][0]["freeform_reaction"] = "reaction"
        edited["context"][0]["preceding_messages"][0]["content"] = "synthetic question"
        edited["context"][0]["assistant_message"]["content"] = "[已脱敏]"
        updated = bridge.dispatch({
            "operation": "preview", "preview_id": preview["preview_id"],
            "expected_revision": preview["preview_revision"],
            "expected_sha256": preview["payload_sha256"], "redacted_payload": edited,
        })
        assert updated["preview_revision"] == 2
        assert updated["payload_sha256"] == sha256_json(updated["payload"])
        assert updated["payload"]["feedback"][0]["freeform_reaction"] == "reaction"
        assert (updated["payload"]["context"][0]["preceding_messages"][0]["content"]
                == "synthetic question")
        assert updated["payload"]["context"][0]["assistant_message"]["content"] == "[已脱敏]"

        with pytest.raises(ValueError):
            bridge.dispatch({
                "operation": "preview", "preview_id": preview["preview_id"],
                "expected_revision": preview["preview_revision"],
                "expected_sha256": preview["payload_sha256"], "redacted_payload": edited,
            })
        forged = json.loads(json.dumps(updated["payload"], ensure_ascii=False))
        forged["feedback"][0]["freeform_reaction"] = "new private detail"
        with pytest.raises(ValueError):
            bridge.dispatch({
                "operation": "preview", "preview_id": preview["preview_id"],
                "expected_revision": updated["preview_revision"],
                "expected_sha256": updated["payload_sha256"], "redacted_payload": forged,
            })
        with pytest.raises(ValueError):
            bridge.dispatch({
                "operation": "confirm", "preview_id": preview["preview_id"],
                "preview_revision": updated["preview_revision"],
                "payload_sha256": preview["payload_sha256"],
                "redaction_confirmed": True, "external_processing_confirmed": True,
            })
        exported = bridge.dispatch({
            "operation": "confirm", "preview_id": preview["preview_id"],
            "preview_revision": updated["preview_revision"],
            "payload_sha256": updated["payload_sha256"],
            "redaction_confirmed": True, "external_processing_confirmed": True,
        })
        export_path = tmp_path / "exports" / f"{exported['export_id']}.json"
        saved = json.loads(export_path.read_text(encoding="utf-8"))
        assert saved["payload_sha256"] == updated["payload_sha256"]
        assert saved["payload"] == updated["payload"]
    finally:
        bridge.repository.close()
        store.close()


@pytest.mark.parametrize("failed_status", ["failed", "cancelled"])
def test_old_answer_and_successful_retry_keep_their_original_prompt_bindings(
    tmp_path, failed_status,
):
    baseline = {
        "app_commit": "a" * 40, "schema_version": 1, "persona_sha256": "b" * 64,
        "prompt_sha256": "c" * 64, "model_sha256": "d" * 64,
        "server_sha256": "e" * 64, "sampling_profile_id": "synthetic-v1",
    }
    candidate = {**baseline, "prompt_sha256": "f" * 64}
    store = SQLiteConversationStore(tmp_path)
    bridge = FeedbackBridge(store)
    try:
        old = store.begin("synthetic question", {"version_binding": baseline})
        store.finish(old["attempt_id"], "completed", "synthetic baseline answer")
        old_message = store.history()["messages"][-1]
        failed = store.begin("synthetic question", {"version_binding": baseline})
        store.finish(failed["attempt_id"], failed_status, "uncommitted text")
        retry = store.begin(None, {"version_binding": candidate}, retry=True)
        assert retry["turn_id"] == failed["turn_id"]
        assert retry["attempt_id"] != failed["attempt_id"]
        store.finish(retry["attempt_id"], "completed", "synthetic candidate answer")
        retry_message = store.history()["messages"][-1]
        old_payload = dict(operation="submit", session_id=store.session_id,
                           turn_id=old_message["turn_id"], assistant_message_id=old_message["id"],
                           quick_tags=[], freeform_reaction="synthetic feedback",
                           idempotency_key="baseline-feedback")
        old_feedback = bridge.dispatch(old_payload)
        assert old_feedback["version_binding"] == baseline
        retry_payload = {
            **old_payload, "turn_id": retry_message["turn_id"],
            "assistant_message_id": retry_message["id"], "idempotency_key": "candidate-feedback",
        }
        retry_feedback = bridge.dispatch(retry_payload)
        assert retry_feedback["version_binding"] == candidate
        assert retry_feedback["feedback_id"] != old_feedback["feedback_id"]
        repeated = bridge.dispatch(old_payload)
        assert repeated["feedback_id"] == old_feedback["feedback_id"]
        assert repeated["version_binding"] == baseline
        with pytest.raises(IdempotencyConflictError):
            bridge.dispatch({**retry_payload, "idempotency_key": old_payload["idempotency_key"]})
        with pytest.raises(MessageOwnershipError):
            bridge.dispatch({**old_payload, "turn_id": retry_message["turn_id"]})
        assert len(bridge.dispatch({"operation": "list"})["events"]) == 2
    finally:
        bridge.repository.close()
        store.close()


def _linked_attempt(tmp_path, *, binding, text="same synthetic question"):
    mirror = SQLiteConversationStore(tmp_path / "mirror", companion_id=A)
    mirror_attempt = mirror.begin(text, {"version_binding": binding})
    receipts = SQLiteReceivedStore(tmp_path / "memory.sqlite3")
    receipts.register(A)
    boundary_session = "boundary-" + mirror.session_id
    receipts.ensure_session(A, boundary_session)
    item = receipts.accept(config(), mirror_attempt["turn_id"], text,
                           session_id=boundary_session)
    claim_value = claim(receipts, item)
    persist_attempt_link(mirror, receipts, claim_value, boundary_session_id=boundary_session,
                         turn_id=mirror_attempt["turn_id"],
                         text_attempt_id=mirror_attempt["attempt_id"],
                         binding=mirror_attempt["binding"], now_us=101)
    return mirror, receipts, mirror_attempt, claim_value


def test_feedback_binds_only_visible_prefix_of_failed_attempt(tmp_path):
    binding = {
        "app_commit": "a" * 40, "schema_version": 1, "persona_sha256": "b" * 64,
        "prompt_sha256": "c" * 64, "model_sha256": "d" * 64,
        "server_sha256": "e" * 64, "sampling_profile_id": "synthetic-v1",
    }
    mirror, receipts, mirror_attempt, run = _linked_attempt(tmp_path, binding=binding)
    assert run.session_id != mirror.session_id
    bridge = FeedbackBridge(mirror, receipt_store_provider=lambda: receipts)
    try:
        first = receipts.approve_segment(run, reviewed(run, ("visible bubble",)), now_us=100)
        dispatched = receipts.dispatch_segment(run, 1, now_us=101)
        receipts.receive_segment(A, run.input_id, 1, fingerprint=dispatched["fingerprint"],
                                 host_receipt="received-first")
        receipts.confirm_segment(A, run.input_id, 1, fingerprint=dispatched["fingerprint"],
                                 host_receipt="visible-first", channel="text_visible")
        # A later approved unit remains hidden when the terminal fails.
        receipts.approve_segment(run, reviewed(run, ("visible bubble", "hidden tail")), now_us=102)
        message_id = mirror.reconcile_confirmed(
            mirror_attempt["turn_id"], first["display_text"], completed=False,
            bubble_ranges=semantic_bubble_ranges(first["display_text"]),
        )["message_id"]
        receipts.finish_stream(run, outcome="failed", now_us=103)
        mirror.finish(mirror_attempt["attempt_id"], "failed", "")
        selection_start = len("visible ")
        selection_end = len("visible bubble")
        created = bridge.dispatch({
            "operation": "submit", "session_id": mirror.session_id,
            "turn_id": mirror_attempt["turn_id"], "assistant_message_id": message_id,
            "target": {"kind": "bubble", "start": selection_start, "end": selection_end},
            "quick_tags": [], "freeform_reaction": "这一句的感觉",
            "idempotency_key": "visible-prefix",
        })
        target = created["target_binding"]
        assert target["status"] == "visible"
        assert target["attempt"]["text_attempt_id"] == mirror_attempt["attempt_id"]
        assert target["attempt"]["boundary_attempt_id"] == run.attempt_id
        assert target["range"] == {"unit": "unicode-code-point", "start": selection_start,
                                   "end": selection_end}
        assert target["selection_sha256"]
        preview = bridge.dispatch({
            "operation": "preview", "feedback_ids": [created["feedback_id"]],
        })
        assert preview["payload"]["context"][0]["selected_text"] == "bubble"
        assert "hidden tail" not in str(preview["payload"])
        with pytest.raises((MessageOwnershipError, ValueError)):
            bridge.dispatch({
                "operation": "submit", "session_id": mirror.session_id,
                "turn_id": mirror_attempt["turn_id"], "assistant_message_id": message_id,
                "target": {"kind": "bubble", "start": 0, "end": len("visible bubble hidden tail")},
                "quick_tags": [], "freeform_reaction": "越界范围",
                "idempotency_key": "outside-prefix",
            })
    finally:
        bridge.repository.close()
        mirror.close()


def test_attempt_feedback_keeps_zero_unit_retry_identity_and_never_exports_text(tmp_path):
    binding = {
        "app_commit": "a" * 40, "schema_version": 1, "persona_sha256": "b" * 64,
        "prompt_sha256": "c" * 64, "model_sha256": "d" * 64,
        "server_sha256": "e" * 64, "sampling_profile_id": "synthetic-v1",
    }
    mirror, receipts, first_mirror, first_run = _linked_attempt(tmp_path, binding=binding)
    bridge = FeedbackBridge(mirror, receipt_store_provider=lambda: receipts)
    try:
        receipts.finish_stream(first_run, outcome="failed", now_us=100)
        mirror.finish(first_mirror["attempt_id"], "failed", "")
        first_event = bridge.dispatch({
            "operation": "submit", "session_id": mirror.session_id,
            "turn_id": first_mirror["turn_id"],
            "target": {"kind": "attempt", "attempt_id": first_mirror["attempt_id"]},
            "quick_tags": [], "freeform_reaction": "这次没有看到回复",
            "idempotency_key": "attempt-one",
        })
        assert first_event["assistant_message_id"] is None
        first_boundary_attempt = first_event["target_binding"]["attempt"]["boundary_attempt_id"]
        assert first_boundary_attempt == first_run.attempt_id
        assert first_event["target_binding"]["status"] == "attempt-only"

        retry_mirror = mirror.begin(None, {"version_binding": binding}, retry=True)
        assert retry_mirror["turn_id"] == first_mirror["turn_id"]
        retry_config = config()
        receipts.retry_stream(A, first_run.input_id, first_run.attempt_id,
                              retry_mirror["attempt_id"], retry_config,
                              now_us=110)
        retry_run = receipts.claim(A, first_run.input_id, "worker-retry",
                                   now_us=111, lease_us=50)
        persist_attempt_link(mirror, receipts, retry_run,
                             boundary_session_id=retry_run.session_id,
                             turn_id=retry_mirror["turn_id"],
                             text_attempt_id=retry_mirror["attempt_id"],
                             binding=retry_mirror["binding"], now_us=112)
        revised = bridge.dispatch({
            "operation": "revise", "feedback_id": first_event["feedback_id"],
            "expected_revision": first_event["feedback_revision"],
            "quick_tags": [], "freeform_reaction": "仍是第一次尝试",
        })
        assert revised["target_binding"] == first_event["target_binding"]
        second_event = bridge.dispatch({
            "operation": "submit", "session_id": mirror.session_id,
            "turn_id": retry_mirror["turn_id"],
            "target": {"kind": "attempt", "attempt_id": first_mirror["attempt_id"]},
            "quick_tags": [], "freeform_reaction": "重复提交仍指向第一次",
            "idempotency_key": "attempt-one-again",
        })
        second_boundary_attempt = second_event["target_binding"]["attempt"]["boundary_attempt_id"]
        assert second_boundary_attempt == first_run.attempt_id
        assert second_boundary_attempt != retry_run.attempt_id
        preview = bridge.dispatch({
            "operation": "preview", "feedback_ids": [first_event["feedback_id"]],
        })
        assert preview["payload"]["context"][0]["selected_text"] is None
        assert "same synthetic question" not in str(preview["payload"]["context"])
        forged_context = json.loads(json.dumps(preview["payload"], ensure_ascii=False))
        forged_context["context"][0]["selected_text"] = "same synthetic question"
        with pytest.raises(ValueError):
            bridge.dispatch({
                "operation": "preview", "preview_id": preview["preview_id"],
                "expected_revision": preview["preview_revision"],
                "expected_sha256": preview["payload_sha256"],
                "redacted_payload": forged_context,
            })
    finally:
        bridge.repository.close()
        mirror.close()


def test_legacy_message_feedback_keeps_unmapped_delivery_unknown(tmp_path):
    binding = {
        "app_commit": "a" * 40, "schema_version": 1, "persona_sha256": "b" * 64,
        "prompt_sha256": "c" * 64, "model_sha256": "d" * 64,
        "server_sha256": "e" * 64, "sampling_profile_id": "synthetic-v1",
    }
    mirror = SQLiteConversationStore(tmp_path / "mirror", companion_id=A)
    attempt = mirror.begin("legacy user input", {"version_binding": binding})
    mirror.finish(attempt["attempt_id"], "completed", "legacy assistant answer")
    message = mirror.history()["messages"][-1]
    receipts = SQLiteReceivedStore(tmp_path / "memory.sqlite3")
    receipts.register(A)
    bridge = FeedbackBridge(mirror, receipt_store_provider=lambda: receipts)
    try:
        feedback = bridge.dispatch({
            "operation": "submit", "session_id": mirror.session_id,
            "turn_id": message["turn_id"], "assistant_message_id": message["id"],
            "target": {"kind": "reply", "start": 0,
                       "end": len("legacy assistant answer")},
            "quick_tags": [], "freeform_reaction": "范围未知",
            "idempotency_key": "legacy-unknown",
        })
        target = feedback["target_binding"]
        assert target["status"] == "unknown"
        assert target["attempt"]["boundary_attempt_id"] is None
        assert target["visible"] is None
        preview = bridge.dispatch({
            "operation": "preview", "feedback_ids": [feedback["feedback_id"]],
        })
        context = preview["payload"]["context"][0]
        assert context["selected_text"] is None
        assert "legacy assistant answer" not in str(context)
        assert "legacy user input" not in str(context)
    finally:
        bridge.repository.close()
        mirror.close()


def test_received_only_feedback_records_receipt_without_hidden_text(tmp_path):
    binding = {
        "app_commit": "a" * 40, "schema_version": 1, "persona_sha256": "b" * 64,
        "prompt_sha256": "c" * 64, "model_sha256": "d" * 64,
        "server_sha256": "e" * 64, "sampling_profile_id": "synthetic-v1",
    }
    mirror, receipts, mirror_attempt, run = _linked_attempt(tmp_path, binding=binding)
    bridge = FeedbackBridge(mirror, receipt_store_provider=lambda: receipts)
    try:
        receipts.approve_segment(run, reviewed(run, ("received but unseen",)), now_us=100)
        dispatched = receipts.dispatch_segment(run, 1, now_us=101)
        receipts.receive_segment(A, run.input_id, 1, fingerprint=dispatched["fingerprint"],
                                 host_receipt="device-received")
        receipts.finish_stream(run, outcome="received", now_us=102)
        mirror.finish(mirror_attempt["attempt_id"], "received", "")
        feedback = bridge.dispatch({
            "operation": "submit", "session_id": mirror.session_id,
            "turn_id": mirror_attempt["turn_id"],
            "target": {"kind": "attempt", "attempt_id": mirror_attempt["attempt_id"]},
            "quick_tags": [], "freeform_reaction": "没有看到",
            "idempotency_key": "received-only",
        })
        target = feedback["target_binding"]
        assert target["status"] == "attempt-only"
        assert target["received"] is True and target["visible"] is False
        preview = bridge.dispatch({
            "operation": "preview", "feedback_ids": [feedback["feedback_id"]],
        })
        assert preview["payload"]["context"][0]["selected_text"] is None
        assert "received but unseen" not in str(preview["payload"])
    finally:
        bridge.repository.close()
        mirror.close()
