"""Opt-in v12 receipt times; no inferred speech time or implicit database migration."""

from .contracts import Scope, identifier
from .sqlite_conversation import SQLiteConversationStore

_EXTRA = (
    "CREATE TABLE source_times (companion TEXT NOT NULL, session TEXT NOT NULL, "
    "role TEXT NOT NULL CHECK(role IN ('user','assistant')), source TEXT NOT NULL, "
    "received_us INTEGER NOT NULL CHECK(received_us>=0), "
    "PRIMARY KEY(companion,role,source), "
    "FOREIGN KEY(companion,session) REFERENCES turn_sessions(companion,session))",
)


class SQLiteTimelineStore(SQLiteConversationStore):
    schema_version = 12
    schema = SQLiteConversationStore.schema + _EXTRA
    tables = SQLiteConversationStore.tables | {"source_times"}

    def _upgrade(self, db, version, application, names):
        raise ValueError("Timeline requires v12; existing databases are not migrated")

    def _record_time(self, db, companion, session, role, source):
        now = self.evidence_clock()
        if type(now) is not int or not 0 <= now < 2**63:
            raise ValueError("Bounded UTC receipt timestamp required")
        db.execute(
            "INSERT INTO source_times VALUES(?,?,?,?,?)",
            (companion, session, role, source, now),
        )

    def _accepted(self, db, receipt, trigger):
        super()._accepted(db, receipt, trigger)
        if trigger is None:
            self._record_time(
                db,
                receipt.companion_id,
                receipt.session_id,
                "user",
                receipt.input_id,
            )

    def _ack_memories(self, db, companion, row, receipt):
        super()._ack_memories(db, companion, row, receipt)
        if (
            receipt["disposition"] == "applied"
            and receipt["status"] in {"completed", "cancelled"}
            and receipt["displayed_end"] > 0
        ):
            source = self._input(db, companion, row["input"])
            self._record_time(db, companion, source["session"], "assistant", row["delivery"])

    def source_time(self, scope, role, source_id):
        if not isinstance(scope, Scope) or role not in {"user", "assistant"}:
            raise ValueError("Typed source scope and role required")
        identifier(source_id)
        with self.transaction(write=False) as db:
            row = db.execute(
                "SELECT received_us FROM source_times "
                "WHERE companion=? AND session=? AND role=? AND source=?",
                (scope.companion_id, scope.session_id, role, source_id),
            ).fetchone()
            return row[0] if row else None
