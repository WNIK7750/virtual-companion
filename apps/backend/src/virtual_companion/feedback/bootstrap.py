from __future__ import annotations

from pathlib import Path

from .application import FeedbackService
from .fakes import NullMessageCatalog
from .repositories.sqlite import SQLiteFeedbackRepository


def create_feedback_service(
    *,
    feedback_database_path: Path,
    message_catalog,
    production_database_path: Path | None = None,
    export_root: Path | None = None,
) -> tuple[FeedbackService, SQLiteFeedbackRepository]:
    """Build the local feedback service; the renderer never supplies database paths."""
    repository = SQLiteFeedbackRepository(
        feedback_database_path,
        production_database_path=production_database_path,
    )
    return FeedbackService(repository, message_catalog, export_root=export_root), repository


def create_unconfigured_feedback_service(
    *, feedback_database_path: Path
) -> tuple[FeedbackService, SQLiteFeedbackRepository]:
    """Fail closed until a real committed message catalog is explicitly wired."""
    repository = SQLiteFeedbackRepository(feedback_database_path)
    return FeedbackService(repository, NullMessageCatalog()), repository
