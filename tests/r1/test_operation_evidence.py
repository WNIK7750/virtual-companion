"""CPU projection contracts; shaped data alone never establishes a database commit."""

import json
from dataclasses import replace
from hashlib import sha256

import pytest
from unit_review_recording import bound_unit_packets as bound_unit_packets
from virtual_companion.boundary.contracts import canonical
from virtual_companion.boundary.history_readset import evidence_hash
from virtual_companion.cognition.operation_evidence import project_operation_evidence
from virtual_companion.cognition.short_dialogue import ShortContext

OWNER = "00000000-0000-0000-0000-000000000001"
OTHER = "00000000-0000-0000-0000-000000000002"
INPUT = "current-input"
USER = "我不喜欢猫。"


def source_hash(text=USER):
    return sha256(text.encode()).hexdigest()


def processing(status="completed", **changes):
    return {
        "protocol": "current-source-processing.v1", "source_id": INPUT,
        "source_sha256": source_hash(), "status": status, "error": None,
        "receipt_sha256": "a" * 64, **changes,
    }


def operation(action="forget", status="committed", **changes):
    return {
        "protocol": "memory-command-outcome.v1", "source_id": INPUT,
        "source_sha256": source_hash(), "operation_id": "command-1",
        "action": action, "status": status, "receipt_sha256": "b" * 64, **changes,
    }


def fact(**changes):
    return {
        "record_id": "record-1", "version": 1, "source_id": INPUT,
        "status": "active", "source_kind": "input", "content": USER,
        "kind": "fact", "fact_key": None, "session_id": "session-1", **changes,
    }


def context(**changes):
    return ShortContext(OWNER, INPUT, 3, canonical({
        "history": [], "memories": [], "exchanges": [], "memory_status": "current",
        **changes,
    }))


def project(value, **changes):
    return project_operation_evidence(value, **{
        "owner": OWNER, "input_id": INPUT, "state_version": 3, "user_text": USER, **changes,
    })


@pytest.mark.parametrize("text", ["请关闭提醒。", "我会替你关提醒。", "昨晚我关了提醒。"])
def test_requests_promises_history_and_quiet_are_not_execution_receipts(text):
    value = context(history=[{"id": "old-output", "role": "assistant", "text": text}],
                    receiving_context={
                        "protocol": "receiving-context.v1", "source": "receiving-client",
                        "scope": "this_client", "observation": "at_input_submission", "quiet": True,
                    })
    assert project(value, user_text=text).receipt_catalog == []
    assert project(None, user_text=text).receipt_catalog == []


def test_completed_delegation_without_a_saved_current_fact_is_not_save_success():
    value = context(source_processing=processing(), memory_access="completed")
    assert project(value).receipt_catalog == []


@pytest.mark.parametrize("action", ["correct", "forget"])
@pytest.mark.parametrize("status", ["committed", "already_applied"])
def test_command_outcome_has_a_separate_content_free_receipt(action, status):
    outcome = operation(action, status)
    evidence = project(context(current_operation=outcome,
                               source_processing=processing(), memory_access="completed"))
    [public] = evidence.receipt_catalog
    assert set(public) == {"id", "action", "status", "text"}
    assert public["action"] == {"correct": "correct_memory", "forget": "forget_memory"}[action]
    assert public["status"] == status
    assert public["id"].startswith("operation-")
    assert "command-1" not in public["text"] and USER not in public["text"]
    host = evidence.host_provenance
    assert host["receipts"][0]["provenance"]["current_operation"] == outcome
    assert host["basis"]["owner"] == OWNER and host["basis"]["input_id"] == INPUT
    assert host["basis"]["state_version"] == 3
    assert host["basis"]["source_sha256"] == source_hash()


def test_command_does_not_infer_a_second_save_from_its_replacement_record():
    evidence = project(context(current_operation=operation("correct"), memories=[fact()]))
    assert [item["action"] for item in evidence.receipt_catalog] == ["correct_memory"]


@pytest.mark.parametrize("source_kind", ["input", "manual"])
def test_save_receipt_preserves_exact_negative_content_and_current_record_scope(source_kind):
    current = fact(source_kind=source_kind)
    value = context(memories=[fact(record_id="older", source_id="old-input"), current],
                    memory_status="pending_source", source_processing=processing(),
                    memory_access="completed")
    evidence = project(value)
    [public] = evidence.receipt_catalog
    assert public["action"] == "save_memory" and public["status"] == "committed"
    assert public["text"] == USER
    assert evidence.host_provenance["receipts"][0]["provenance"]["fact"] == current
    assert json.loads(value.payload)["memories"][1] == current


@pytest.mark.parametrize("status", ["queued", "running", "retryable", "failed"])
def test_current_source_status_overrides_global_freshness_for_save(status):
    value = context(memories=[fact()], source_processing=processing(status),
                    memory_access="not_accessed")
    assert project(value).receipt_catalog == []


@pytest.mark.parametrize("status,allowed", [("current", True), ("pending_source", False)])
def test_legacy_context_retains_only_the_existing_current_fact_save_permission(status, allowed):
    evidence = project(context(memories=[fact()], memory_status=status))
    assert bool(evidence.receipt_catalog) is allowed


@pytest.mark.parametrize("changes", [{"status": "revoked"}, {"source_kind": "output"},
                                      {"source_kind": "assessment"}, {"source_id": "old-input"}])
def test_inactive_non_user_and_other_input_facts_do_not_grant_save(changes):
    assert project(context(memories=[fact(**changes)])).receipt_catalog == []


@pytest.mark.parametrize("changes", [{"companion": OTHER}, {"input_id": "other-input"},
                                      {"state_version": 4}])
def test_sealed_context_scope_cannot_be_rebound(changes):
    with pytest.raises(ValueError):
        project(replace(context(), **changes))


@pytest.mark.parametrize("field,value", [("source_id", "other-input"),
                                        ("source_sha256", "c" * 64)])
def test_command_must_match_the_exact_current_input(field, value):
    with pytest.raises(ValueError):
        project(context(current_operation=operation(**{field: value})))


def test_source_processing_hash_must_match_current_text_even_without_a_command():
    with pytest.raises(ValueError):
        project(context(memories=[fact()], source_processing=processing(source_sha256="c" * 64),
                        memory_access="completed"))


@pytest.mark.parametrize("changes", [{"action": "disable_reminder"}, {"status": "queued"},
                                      {"extra": True}])
def test_unknown_capability_incomplete_or_extended_command_is_rejected(changes):
    with pytest.raises(ValueError):
        project(context(current_operation=operation(**changes)))


def test_ambiguous_current_record_versions_are_rejected():
    with pytest.raises(ValueError):
        project(context(memories=[fact(), fact(version=2)]))


def test_fingerprint_binds_scope_seal_and_exact_record_version_without_shared_mutation():
    value = context(memories=[fact()])
    evidence = project(value)
    assert evidence.fingerprint == project(value).fingerprint
    host = evidence.host_provenance
    assert evidence.fingerprint == evidence_hash(host)
    assert host["basis"]["context_sha256"] == evidence_hash(json.loads(value.payload))
    assert evidence.receipt_catalog[0]["id"] != project(
        context(memories=[fact(version=2)])).receipt_catalog[0]["id"]
    assert evidence.receipt_catalog[0]["id"] != project(
        replace(value, state_version=4), state_version=4).receipt_catalog[0]["id"]
    public = evidence.receipt_catalog
    public[0]["action"] = "disable_reminder"
    host["receipts"][0]["provenance"]["fact"]["content"] = "我喜欢猫。"
    assert evidence.receipt_catalog[0]["action"] == "save_memory"
    assert evidence.host_provenance["receipts"][0]["provenance"]["fact"]["content"] == USER
    assert json.loads(value.payload)["memories"][0]["content"] == USER


@pytest.mark.parametrize("action", ["save_memory", "correct_memory", "forget_memory"])
def test_real_sqlite_admission_and_recording_worker_receipts(tmp_path, action):
    from test_current_command_outcome import observe
    from test_default_cross_turn_memory import CORRECT, FORGET, ORIGINAL, Scenario, build, turn

    memory, mirror, runtime, session = build(tmp_path, Scenario())
    capture = observe(runtime)
    try:
        turn(runtime, session, ORIGINAL)
        text = {"save_memory": ORIGINAL, "correct_memory": CORRECT, "forget_memory": FORGET}[action]
        if action != "save_memory":
            turn(runtime, session, text)
        admitted = capture.contexts[-1]
        evidence = project_operation_evidence(
            admitted, owner=admitted.companion, input_id=admitted.input_id,
            state_version=admitted.state_version, user_text=text,
        )
        assert {row["action"] for row in evidence.receipt_catalog} == {action}
        assert {row["status"] for row in evidence.receipt_catalog} == {"committed"}
    finally:
        memory.close()
        mirror.close()
