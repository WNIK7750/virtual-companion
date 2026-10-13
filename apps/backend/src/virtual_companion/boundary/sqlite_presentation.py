"""Opt-in v4 format gate: new commits require complete presentation review.

The JSON manifests need no new tables, but a version bump prevents an old reader
from silently skipping the new ACK binding. Restore the v3 backup to roll back.
"""

from .sqlite_memory import SQLiteMemoryStore
from .sqlite_state import _APPLICATION_ID


class SQLitePresentationStore(SQLiteMemoryStore):
    schema_version = 4
    allow_host_policy = False

    def __init__(self, path, *, upgrade_v3=False):
        self.upgrade_v3 = upgrade_v3
        super().__init__(path)

    def _upgrade(self, db, version, application, names):
        if not (
            self.upgrade_v3
            and version == 3
            and application == _APPLICATION_ID
            and names == SQLiteMemoryStore.tables
        ):
            raise ValueError("Presentation requires v4 or explicit upgrade of a v3 synthetic copy")
        db.execute("PRAGMA user_version=4")

    def _validate_result(self, result):
        if result.presentation is None:
            raise ValueError("New presentation commits require the complete display review")
        if ((result.presentation.review.approval_source == "host_policy"
             or result.presentation.candidate.intent != "speak")
                and not self.allow_host_policy):
            raise ValueError("Host policy review requires the explicit dialogue storage format")
