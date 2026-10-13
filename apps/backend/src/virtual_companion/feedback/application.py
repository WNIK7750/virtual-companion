from __future__ import annotations

import json
import os
import tempfile
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .errors import ExportConfirmationError, NotFoundError, StaleExportError, ValidationError
from .models import (
    ContextPolicy,
    DeletionReceipt,
    ExportPreview,
    FeedbackAnalysis,
    FeedbackEvent,
    FeedbackExport,
    FeedbackSelection,
    VersionBinding,
)
from .ports import FeedbackRepository, MessageCatalog, RecordingAnalyzer


class FeedbackService:
    """Application owner for feedback lifecycle, export consent, and analysis storage."""

    def __init__(
        self,
        repository: FeedbackRepository,
        message_catalog: MessageCatalog,
        *,
        export_root: Path | None = None,
    ) -> None:
        self._repository = repository
        self._messages = message_catalog
        self._export_root = (
            None if export_root is None else Path(export_root).expanduser().resolve()
        )
        if self._export_root is not None:
            self._export_root.mkdir(parents=True, exist_ok=True)

    def submit_feedback(
        self,
        *,
        session_id: str,
        turn_id: str,
        assistant_message_id: str | None,
        quick_tags: Sequence[str] | None,
        freeform_reaction: str | None,
        idempotency_key: str,
        version_binding: VersionBinding,
        target_binding: dict[str, Any] | None = None,
        feedback_id: str | None = None,
        created_at: Any = None,
    ) -> FeedbackEvent:
        ownership = {
            "session_id": session_id,
            "turn_id": turn_id,
            "assistant_message_id": assistant_message_id,
        }
        if target_binding is not None:
            ownership["target_binding"] = target_binding
        self._messages.assert_owned_assistant_message(**ownership)
        event = FeedbackEvent.create(
            feedback_id=feedback_id or str(uuid.uuid4()),
            session_id=session_id,
            turn_id=turn_id,
            assistant_message_id=assistant_message_id,
            quick_tags=quick_tags,
            freeform_reaction=freeform_reaction,
            version_binding=version_binding,
            created_at=created_at,
            schema_version=2 if target_binding is not None else 1,
            target_binding=target_binding,
        )
        return self._repository.submit(event, idempotency_key=idempotency_key)

    def revise_feedback(
        self,
        *,
        feedback_id: str,
        expected_revision: int,
        quick_tags: Sequence[str] | None,
        freeform_reaction: str | None,
    ) -> FeedbackEvent:
        current = self._repository.current(feedback_id)
        if current is None:
            raise NotFoundError("feedback is missing or withdrawn")
        event = FeedbackEvent.create(
            feedback_id=current.feedback_id,
            session_id=current.session_id,
            turn_id=current.turn_id,
            assistant_message_id=current.assistant_message_id,
            quick_tags=quick_tags,
            freeform_reaction=freeform_reaction,
            version_binding=current.version_binding,
            feedback_revision=expected_revision + 1,
            supersedes_revision=expected_revision,
            schema_version=current.schema_version,
            target_binding=current.target_binding,
        )
        return self._repository.append_revision(event, expected_revision=expected_revision)

    def withdraw_feedback(self, *, feedback_id: str, expected_revision: int) -> FeedbackEvent:
        current = self._repository.current(feedback_id)
        if current is None:
            raise NotFoundError("feedback is missing or already withdrawn")
        event = FeedbackEvent.create(
            feedback_id=current.feedback_id,
            session_id=current.session_id,
            turn_id=current.turn_id,
            assistant_message_id=current.assistant_message_id,
            quick_tags=current.quick_tags,
            freeform_reaction=current.freeform_reaction,
            version_binding=current.version_binding,
            feedback_revision=expected_revision + 1,
            supersedes_revision=expected_revision,
            status="withdrawn",
            schema_version=current.schema_version,
            target_binding=current.target_binding,
        )
        return self._repository.append_revision(event, expected_revision=expected_revision)

    def delete_feedback(self, feedback_id: str) -> DeletionReceipt:
        return self._repository.delete_feedback(feedback_id)

    def get_current_feedback(self, feedback_id: str) -> FeedbackEvent | None:
        return self._repository.current(feedback_id)

    def preview_export(
        self,
        *,
        feedback_ids: Sequence[str],
        context_policy: ContextPolicy | None = None,
        redactor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> ExportPreview:
        policy = context_policy or ContextPolicy()
        selected: list[FeedbackSelection] = []
        feedback_payload: list[dict[str, Any]] = []
        context_payload: list[dict[str, Any]] = []
        expected_revisions: dict[str, int] = {}
        for feedback_id in feedback_ids:
            if feedback_id in expected_revisions:
                raise ValidationError("feedback_ids cannot contain duplicates")
            current = self._repository.current(feedback_id)
            if current is None:
                raise StaleExportError("only current active feedback can be exported")
            selected.append(
                FeedbackSelection(
                    feedback_id=current.feedback_id,
                    feedback_revision=current.feedback_revision,
                )
            )
            expected_revisions[current.feedback_id] = current.feedback_revision
            feedback_payload.append(current.to_mapping())
            context = {
                "session_id": current.session_id,
                "turn_id": current.turn_id,
                "assistant_message_id": current.assistant_message_id,
                "policy": policy,
            }
            if current.target_binding is not None:
                context["target_binding"] = current.target_binding
            context_payload.append(self._messages.context_for(**context))
        payload: dict[str, Any] = {
            "schema_version": 1,
            "feedback": feedback_payload,
            "context": context_payload,
        }
        if redactor is not None:
            payload = redactor(json.loads(json.dumps(payload, ensure_ascii=False)))
        return ExportPreview(
            selected_feedback=tuple(selected),
            context_policy=policy,
            payload=payload,
            expected_revisions=expected_revisions,
        )

    def cancel_export(self, _preview: ExportPreview) -> None:
        """A cancelled preview has no repository side effect."""

    def confirm_export(
        self,
        preview: ExportPreview,
        *,
        redaction_confirmed: bool,
        external_processing_confirmed: bool,
        destination: Path | None = None,
        export_id: str | None = None,
    ) -> FeedbackExport:
        if not redaction_confirmed or not external_processing_confirmed:
            raise ExportConfirmationError(
                "export requires explicit redaction and processing confirmation"
            )
        export = FeedbackExport.create(
            export_id=export_id or str(uuid.uuid4()),
            selected_feedback=preview.selected_feedback,
            context_policy=preview.context_policy,
            payload=preview.payload,
            redaction_confirmed=True,
            external_processing_confirmed=True,
        )
        artifact = json.dumps(
            export.to_mapping(),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ).encode("utf-8")
        temp_path: Path | None = None
        if destination is None and self._export_root is not None:
            destination = self._export_root / f"{export.export_id}.json"
        elif destination is not None:
            destination = Path(destination).expanduser().resolve()
            if self._export_root is not None:
                try:
                    destination.relative_to(self._export_root)
                except ValueError as exc:
                    raise ExportConfirmationError(
                        "export destination is outside the configured local root"
                    ) from exc
        if destination is not None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temp_name = tempfile.mkstemp(
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
            )
            os.close(descriptor)
            temp_path = Path(temp_name)
        try:
            if temp_path is not None:
                temp_path.write_bytes(artifact)
            stored = self._repository.create_export(
                export_id=export.export_id,
                selected_feedback=export.selected_feedback,
                context_policy=export.context_policy,
                payload=dict(export.payload),
                redaction_confirmed=True,
                external_processing_confirmed=True,
                expected_revisions=dict(preview.expected_revisions),
                created_at=export.created_at,
            )
            if temp_path is not None and destination is not None:
                try:
                    os.replace(temp_path, destination)
                except OSError:
                    self._repository.delete_export(stored.export_id)
                    raise
            return stored
        except Exception:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()
            raise

    def analyze_export(
        self,
        export_id: str,
        analyzer: RecordingAnalyzer,
    ) -> FeedbackAnalysis:
        export = self._repository.get_export(export_id)
        if export is None:
            raise NotFoundError("export does not exist")
        if export.status != "active":
            raise StaleExportError("stale export cannot be analyzed")
        analysis = analyzer.analyze(export)
        return self._repository.save_analysis(analysis)
