from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from virtual_companion.feedback.application import FeedbackService
from virtual_companion.feedback.errors import RepositoryError, StaleRevisionError
from virtual_companion.feedback.fakes import InMemoryMessageCatalog
from virtual_companion.feedback.models import ContextPolicy, VersionBinding
from virtual_companion.feedback.repositories.sqlite import SQLiteFeedbackRepository


def _binding() -> VersionBinding:
    return VersionBinding(
        app_commit="a" * 40,
        schema_version=1,
        persona_sha256="b" * 64,
        prompt_sha256="c" * 64,
        model_sha256="d" * 64,
        server_sha256="e" * 64,
        sampling_profile_id="sampling-v1",
    )


def _messages() -> InMemoryMessageCatalog:
    messages = InMemoryMessageCatalog()
    messages.register(
        session_id="session-1",
        turn_id="turn-1",
        assistant_message_id="assistant-1",
        content="合成回复",
        preceding=({"role": "user", "content": "合成问题"},),
    )
    return messages


def _service(path: Path) -> tuple[FeedbackService, SQLiteFeedbackRepository]:
    repository = SQLiteFeedbackRepository(path)
    return FeedbackService(repository, _messages()), repository


def _submit(service: FeedbackService, *, feedback_id: str = "feedback-1"):
    return service.submit_feedback(
        session_id="session-1",
        turn_id="turn-1",
        assistant_message_id="assistant-1",
        quick_tags=("too_mechanical",),
        freeform_reaction="合成反馈",
        idempotency_key=f"idem-{feedback_id}",
        version_binding=_binding(),
        feedback_id=feedback_id,
    )


def test_future_feedback_schema_is_rejected_before_schema_mutation(tmp_path):
    path = tmp_path / "future.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE feedback_meta(key TEXT PRIMARY KEY,value TEXT)")
        connection.execute("INSERT INTO feedback_meta VALUES('schema_version','99')")
        before = connection.execute("SELECT name,sql FROM sqlite_master ORDER BY name").fetchall()
    with pytest.raises(RepositoryError, match="unsupported"):
        SQLiteFeedbackRepository(path)
    with sqlite3.connect(path) as connection:
        after = connection.execute("SELECT name,sql FROM sqlite_master ORDER BY name").fetchall()
    assert after == before


def test_sqlite_feedback_survives_close_and_reopen(tmp_path: Path) -> None:
    path = tmp_path / "feedback.sqlite3"
    service, repository = _service(path)
    created = _submit(service)
    repository.close()

    reopened = SQLiteFeedbackRepository(path)
    assert reopened.current(created.feedback_id) == created
    reopened.close()


def test_sqlite_backup_is_reopenable(tmp_path: Path) -> None:
    path = tmp_path / "feedback.sqlite3"
    backup = tmp_path / "backup" / "feedback.sqlite3"
    service, repository = _service(path)
    created = _submit(service)

    repository.backup_to(backup)
    restored = SQLiteFeedbackRepository(backup)
    assert restored.current(created.feedback_id) == created
    repository.close()
    restored.close()


def test_feedback_database_cannot_be_the_production_database(tmp_path: Path) -> None:
    path = tmp_path / "feedback.sqlite3"

    with pytest.raises(ValueError, match="isolated"):
        SQLiteFeedbackRepository(path, production_database_path=path)


def test_transaction_failure_does_not_leave_a_partial_feedback(tmp_path: Path) -> None:
    service, repository = _service(tmp_path / "feedback.sqlite3")
    repository.fail_next_write = True

    with pytest.raises(RepositoryError):
        _submit(service)

    assert repository.current("feedback-1") is None


def test_expected_revision_conflict_is_visible_across_connections(tmp_path: Path) -> None:
    path = tmp_path / "feedback.sqlite3"
    service_a, repository_a = _service(path)
    service_b, repository_b = _service(path)
    created = _submit(service_a)

    service_a.revise_feedback(
        feedback_id=created.feedback_id,
        expected_revision=1,
        quick_tags=("too_verbose",),
        freeform_reaction="第一次修订",
    )

    with pytest.raises(StaleRevisionError):
        service_b.revise_feedback(
            feedback_id=created.feedback_id,
            expected_revision=1,
            quick_tags=("stale",),
            freeform_reaction="过期修订",
        )
    repository_a.close()
    repository_b.close()


def test_export_is_stale_after_revision_and_delete_removes_derived_rows(tmp_path: Path) -> None:
    path = tmp_path / "feedback.sqlite3"
    service, repository = _service(path)
    created = _submit(service)
    preview = service.preview_export(
        feedback_ids=(created.feedback_id,),
        context_policy=ContextPolicy(preceding_turns=1),
    )
    export = service.confirm_export(
        preview,
        redaction_confirmed=True,
        external_processing_confirmed=True,
        export_id="export-1",
    )
    assert repository.get_export(export.export_id) == export

    service.revise_feedback(
        feedback_id=created.feedback_id,
        expected_revision=1,
        quick_tags=("too_verbose",),
        freeform_reaction="修订反馈",
    )
    stale = repository.get_export(export.export_id)
    assert stale is not None and stale.status == "stale"

    receipt = service.delete_feedback(created.feedback_id)
    assert receipt.deleted_revisions == 2
    assert receipt.deleted_exports == 1
    assert repository.get_export(export.export_id) is None
    assert repository.list_analyses(export_id=export.export_id) == ()
    repository.close()
