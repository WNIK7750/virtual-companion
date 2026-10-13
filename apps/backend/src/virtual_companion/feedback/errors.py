from __future__ import annotations


class FeedbackError(Exception):
    """Base class for expected feedback boundary failures."""


class ValidationError(FeedbackError, ValueError):
    """An input does not satisfy a versioned feedback contract."""


class NotFoundError(FeedbackError):
    """A referenced feedback, message, export, or analysis does not exist."""


class ConflictError(FeedbackError):
    """A write conflicts with the current revision or persisted object."""


class StaleRevisionError(ConflictError):
    """The caller attempted to revise a feedback revision that is no longer current."""


class IdempotencyConflictError(ConflictError):
    """An idempotency key was reused for a different request."""


class StaleExportError(ConflictError):
    """An export preview no longer matches the current feedback revisions."""


class ExportConfirmationError(FeedbackError):
    """A local export was attempted without all explicit confirmations."""


class MessageOwnershipError(FeedbackError):
    """A feedback message is missing, uncommitted, or from another turn/session."""


class RepositoryError(FeedbackError):
    """A repository operation failed and was rolled back."""
