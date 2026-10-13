"""Opt-in v13 host-policy review receipts; no implicit upgrade of existing data.

An older binary cannot interpret a host approval as a successful model review.
Keep the previous database backup when testing this isolated format.
"""

from .sqlite_business import SQLiteBusinessStore


class SQLiteDialogueStore(SQLiteBusinessStore):
    schema_version = 13
    allow_host_policy = True

    def _upgrade(self, db, version, application, names):
        raise ValueError("Dialogue requires v13; existing databases are not migrated")
