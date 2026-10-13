from __future__ import annotations

import threading
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from ..errors import (
    ConflictError,
    IdempotencyConflictError,
    NotFoundError,
    StaleExportError,
    StaleRevisionError,
)
from ..models import (
    ContextPolicy,
    DeletionReceipt,
    FeedbackAnalysis,
    FeedbackEvent,
    FeedbackExport,
    FeedbackSelection,
    feedback_idempotency_fingerprint,
)


class InMemoryFeedbackRepository:
    """Thread-safe fake used by B0 and recording tests."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._events: dict[str, list[FeedbackEvent]] = {}
        self._idempotency: dict[str, tuple[str, str]] = {}
        self._exports: dict[str, FeedbackExport] = {}
        self._analyses: dict[str, FeedbackAnalysis] = {}

    def submit(self, event: FeedbackEvent, *, idempotency_key: str) -> FeedbackEvent:
        if not idempotency_key.strip():
            raise ConflictError("idempotency_key is required")
        request_hash = feedback_idempotency_fingerprint(event)
        with self._lock:
            previous = self._idempotency.get(idempotency_key)
            if previous is not None:
                if previous[1] != request_hash:
                    raise IdempotencyConflictError("idempotency key was reused for another request")
                return self._events[previous[0]][0]
            if event.feedback_id in self._events:
                raise ConflictError("feedback_id already exists")
            self._events[event.feedback_id] = [event]
            self._idempotency[idempotency_key] = (event.feedback_id, request_hash)
            return event

    def current(self, feedback_id: str) -> FeedbackEvent | None:
        with self._lock:
            revisions = self._events.get(feedback_id)
            return revisions[-1] if revisions and revisions[-1].status == "active" else None

    def all_events(self, feedback_id: str) -> tuple[FeedbackEvent, ...]:
        with self._lock:
            return tuple(self._events.get(feedback_id, ()))

    def list_current(self, *, session_id: str | None = None) -> tuple[FeedbackEvent, ...]:
        with self._lock:
            values = [
                revisions[-1]
                for revisions in self._events.values()
                if revisions and revisions[-1].status == "active"
            ]
            if session_id is not None:
                values = [item for item in values if item.session_id == session_id]
            return tuple(sorted(values, key=lambda item: item.created_at))

    def append_revision(self, event: FeedbackEvent, *, expected_revision: int) -> FeedbackEvent:
        with self._lock:
            revisions = self._events.get(event.feedback_id)
            if not revisions:
                raise NotFoundError("feedback does not exist")
            current = revisions[-1]
            if current.feedback_revision != expected_revision:
                raise StaleRevisionError("feedback revision is stale")
            if event.feedback_revision != expected_revision + 1:
                raise ConflictError("new revision must increment the current revision")
            revisions.append(event)
            self._mark_exports_stale(event.feedback_id)
            return event

    def delete_feedback(self, feedback_id: str) -> DeletionReceipt:
        with self._lock:
            revisions = self._events.pop(feedback_id, None)
            if revisions is None:
                raise NotFoundError("feedback does not exist")
            export_ids = {
                export_id
                for export_id, export in self._exports.items()
                if any(item.feedback_id == feedback_id for item in export.selected_feedback)
            }
            for export_id in export_ids:
                self._exports.pop(export_id, None)
            analysis_ids = {
                analysis_id
                for analysis_id, analysis in self._analyses.items()
                if analysis.input_export_id in export_ids
            }
            for analysis_id in analysis_ids:
                self._analyses.pop(analysis_id, None)
            for key, value in tuple(self._idempotency.items()):
                if value[0] == feedback_id:
                    self._idempotency.pop(key, None)
            return DeletionReceipt(
                feedback_id=feedback_id,
                deleted_revisions=len(revisions),
                deleted_exports=len(export_ids),
                deleted_analyses=len(analysis_ids),
                deleted_at=datetime.now(UTC).isoformat(),
            )

    def create_export(
        self,
        *,
        export_id: str,
        selected_feedback: Sequence[FeedbackSelection],
        context_policy: ContextPolicy,
        payload: dict[str, Any],
        redaction_confirmed: bool,
        external_processing_confirmed: bool,
        expected_revisions: dict[str, int],
        created_at: str | None = None,
    ) -> FeedbackExport:
        with self._lock:
            self._assert_current_revisions(selected_feedback, expected_revisions)
            if export_id in self._exports:
                raise ConflictError("export_id already exists")
            export = FeedbackExport.create(
                export_id=export_id,
                selected_feedback=selected_feedback,
                context_policy=context_policy,
                payload=payload,
                redaction_confirmed=redaction_confirmed,
                external_processing_confirmed=external_processing_confirmed,
                created_at=created_at,
            )
            self._exports[export_id] = export
            return export

    def get_export(self, export_id: str) -> FeedbackExport | None:
        with self._lock:
            return self._exports.get(export_id)

    def delete_export(self, export_id: str) -> None:
        with self._lock:
            self._exports.pop(export_id, None)
            for analysis_id, analysis in tuple(self._analyses.items()):
                if analysis.input_export_id == export_id:
                    self._analyses.pop(analysis_id, None)

    def save_analysis(self, analysis: FeedbackAnalysis) -> FeedbackAnalysis:
        with self._lock:
            export = self._exports.get(analysis.input_export_id)
            if export is None:
                raise NotFoundError("analysis input export does not exist")
            if export.status != "active":
                raise StaleExportError("stale export cannot receive a new analysis")
            if analysis.input_export_sha256 != export.payload_sha256:
                raise ConflictError("analysis input export hash does not match")
            selected_ids = {item.feedback_id for item in export.selected_feedback}
            for finding in analysis.findings:
                if not set(finding.evidence_feedback_ids).issubset(selected_ids):
                    raise ConflictError("analysis cites feedback outside its input export")
            if analysis.analysis_id in self._analyses:
                raise ConflictError("analysis_id already exists")
            self._analyses[analysis.analysis_id] = analysis
            return analysis

    def get_analysis(self, analysis_id: str) -> FeedbackAnalysis | None:
        with self._lock:
            return self._analyses.get(analysis_id)

    def list_analyses(self, *, export_id: str | None = None) -> tuple[FeedbackAnalysis, ...]:
        with self._lock:
            values = tuple(self._analyses.values())
            if export_id is not None:
                values = tuple(item for item in values if item.input_export_id == export_id)
            return values

    def close(self) -> None:
        return None

    def _assert_current_revisions(
        self,
        selected_feedback: Sequence[FeedbackSelection],
        expected_revisions: dict[str, int],
    ) -> None:
        for selection in selected_feedback:
            current = self.current(selection.feedback_id)
            if current is None:
                raise StaleExportError("feedback is withdrawn or missing")
            if (
                current.feedback_revision != selection.feedback_revision
                or expected_revisions.get(selection.feedback_id) != selection.feedback_revision
            ):
                raise StaleExportError("export preview no longer matches current feedback")

    def _mark_exports_stale(self, feedback_id: str) -> None:
        for export_id, export in tuple(self._exports.items()):
            if any(item.feedback_id == feedback_id for item in export.selected_feedback):
                self._exports[export_id] = FeedbackExport(
                    schema_version=export.schema_version,
                    export_id=export.export_id,
                    created_at=export.created_at,
                    selected_feedback=export.selected_feedback,
                    context_policy=export.context_policy,
                    redaction_confirmed=export.redaction_confirmed,
                    external_processing_confirmed=export.external_processing_confirmed,
                    payload=export.payload,
                    payload_sha256=export.payload_sha256,
                    status="stale",
                )
