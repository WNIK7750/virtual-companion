"""Opt-in v11: bounded actual dialogue and atomic extraction progress.

Processing receipts prevent duplicate extraction after a user interrupts ACK
maintenance. They are not user facts, and never mark a question answered.
"""

from hashlib import sha256

from .contracts import Scope, VersionConflict
from .evidence import EvidenceTurn
from .sqlite_maintenance import SQLiteMaintenanceStore
from .turns import RunClaim

_EXTRA = (
    "CREATE TABLE exchange_observations (companion TEXT NOT NULL, role TEXT NOT NULL, "
    "source TEXT NOT NULL, fingerprint TEXT NOT NULL, input TEXT NOT NULL, "
    "PRIMARY KEY(companion,role,source), "
    "FOREIGN KEY(companion,input) REFERENCES turn_inputs(companion,input))",
)


class SQLiteConversationStore(SQLiteMaintenanceStore):
    schema_version = 11
    schema = SQLiteMaintenanceStore.schema + _EXTRA
    tables = SQLiteMaintenanceStore.tables | {"exchange_observations"}

    def _upgrade(self, db, version, application, names):
        raise ValueError("Conversation context requires v11; existing databases are not migrated")

    def _source_seen(self, db, source):
        digest = sha256(source.text.encode()).hexdigest()
        row = db.execute(
            "SELECT fingerprint FROM exchange_observations "
            "WHERE companion=? AND role=? AND source=?",
            (source.scope.companion_id, source.role, source.source_id),
        ).fetchone()
        if row:
            if row[0] != digest:
                raise VersionConflict(
                    "Previously observed source changed; explicit reconciliation required"
                )
            return True
        return (
            source.role == "assistant"
            and db.execute(
                "SELECT 1 FROM turn_triggers WHERE companion=? AND receipt IS NOT NULL "
                "AND json_extract(data,'$.source_id')=? AND json_extract(data,'$.prefix_sha256')=?",
                (source.scope.companion_id, source.source_id, digest),
            ).fetchone()
            is not None
        )

    def exchange_context(self, claim, *, limit=32):
        if not isinstance(claim, RunClaim) or type(limit) is not int or not 1 <= limit <= 32:
            raise ValueError("Typed current input and bounded exchange context required")
        with self.transaction(write=False) as db:
            anchor = db.execute(
                "SELECT rowid,* FROM turn_inputs WHERE companion=? AND input=?",
                (claim.companion_id, claim.input_id),
            ).fetchone()
            if (
                anchor is None
                or anchor["session"] != claim.session_id
                or anchor["status"] != "running"
                or self._run_config(db, anchor) != claim.config_json
                or db.execute(
                    "SELECT 1 FROM turn_triggers WHERE companion=? AND input=?",
                    (claim.companion_id, claim.input_id),
                ).fetchone()
            ):
                raise ValueError("Current user input required for conversation extraction")
            binding = (claim.companion_id, claim.session_id, anchor["rowid"])
            rows = db.execute(
                "SELECT i.rowid AS ordinal,0 AS phase,i.input AS source,"
                "'user' AS role,i.text,NULL AS confirmed "
                "FROM turn_inputs i WHERE i.companion=? AND i.session=? AND i.rowid<=? "
                "AND NOT EXISTS (SELECT 1 FROM turn_triggers t "
                "WHERE t.companion=i.companion AND t.input=i.input) "
                "UNION ALL SELECT i.rowid,1,d.delivery,'assistant',d.text,d.displayed_end "
                "FROM turn_inputs i JOIN turn_deliveries d "
                "ON i.companion=d.companion AND i.input=d.input "
                "WHERE i.companion=? AND i.session=? AND i.rowid<? AND d.displayed_end>0 "
                "AND d.status IN ('completed','cancelled') "
                "ORDER BY ordinal DESC,phase DESC LIMIT ?",
                (*binding, *binding, limit),
            ).fetchall()
            scope = Scope(claim.companion_id, claim.session_id)
            sources = tuple(
                EvidenceTurn(
                    scope,
                    row["source"],
                    row["role"],
                    row["text"] if row["role"] == "user" else row["text"][: row["confirmed"]],
                    len(row["text"]) if row["role"] == "user" else row["confirmed"],
                )
                for row in reversed(rows)
            )
            return sources, tuple(s.source_id for s in sources if not self._source_seen(db, s))

    def _commit_exchanges(self, db, result, receipt):
        sources = {s.source_id: s for s in result.exchange_sources}
        for command, _ in result.exchanges:
            if sources and self._source_seen(db, sources[command.evidence.source_id]):
                raise VersionConflict(
                    "An already processed source cannot produce another exchange effect"
                )
        super()._commit_exchanges(db, result, receipt)
        for source in result.exchange_sources:
            self._source_seen(db, source)
            db.execute(
                "INSERT INTO exchange_observations VALUES(?,?,?,?,?) ON CONFLICT DO NOTHING",
                (
                    result.claim.companion_id,
                    source.role,
                    source.source_id,
                    sha256(source.text.encode()).hexdigest(),
                    result.claim.input_id,
                ),
            )
