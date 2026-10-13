from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from .models import (
    ContextPolicy,
    DeletionReceipt,
    FeedbackAnalysis,
    FeedbackEvent,
    FeedbackExport,
    FeedbackSelection,
)


class FeedbackRepository(Protocol):
    """Persistence port owned by the application layer."""

    def submit(self, event: FeedbackEvent, *, idempotency_key: str) -> FeedbackEvent: ...

    def current(self, feedback_id: str) -> FeedbackEvent | None: ...

    def all_events(self, feedback_id: str) -> tuple[FeedbackEvent, ...]: ...

    def list_current(self, *, session_id: str | None = None) -> tuple[FeedbackEvent, ...]: ...

    def append_revision(self, event: FeedbackEvent, *, expected_revision: int) -> FeedbackEvent: ...

    def delete_feedback(self, feedback_id: str) -> DeletionReceipt: ...

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
    ) -> FeedbackExport: ...

    def get_export(self, export_id: str) -> FeedbackExport | None: ...

    def delete_export(self, export_id: str) -> None: ...

    def save_analysis(self, analysis: FeedbackAnalysis) -> FeedbackAnalysis: ...

    def get_analysis(self, analysis_id: str) -> FeedbackAnalysis | None: ...

    def list_analyses(self, *, export_id: str | None = None) -> tuple[FeedbackAnalysis, ...]: ...

    def close(self) -> None: ...


class MessageCatalog(Protocol):
    """Read-only committed assistant-message access for ownership and minimal context."""

    def assert_owned_assistant_message(
        self,
        *,
        session_id: str,
        turn_id: str,
        assistant_message_id: str,
    ) -> None: ...

    def context_for(
        self,
        *,
        session_id: str,
        turn_id: str,
        assistant_message_id: str,
        policy: ContextPolicy,
    ) -> dict[str, Any]: ...


class RecordingAnalyzer(Protocol):
    """A local recording/fake analyzer. No network or provider is implied by this port."""

    def analyze(self, export: FeedbackExport) -> FeedbackAnalysis: ...


class ExportWriter(Protocol):
    def write(self, destination: Path, export: FeedbackExport) -> None: ...
