"""Opt-in v14 durable source intake, without treating computation as conversation.

Source admission and terminal positive ACK enqueue in their original transaction.
This is the storage foundation, not a completed scheduler/extractor: no operation
here clears the conservative memory barrier without supported semantic processing.
"""

from hashlib import sha256
from uuid import uuid4

from .contracts import Scope, VersionConflict, companion_id, identifier
from .sqlite_dialogue import SQLiteDialogueStore

_EXTRA = (
    "CREATE TABLE background_sources (companion TEXT NOT NULL, session TEXT NOT NULL, "
    "job TEXT NOT NULL, role TEXT NOT NULL CHECK(role IN ('user','assistant')), "
    "source TEXT NOT NULL, source_hash TEXT NOT NULL, confirmed_end INTEGER NOT NULL, "
    "status TEXT NOT NULL CHECK(status IN ('queued','running','retryable','failed','completed')), "
    "generation INTEGER NOT NULL DEFAULT 0 CHECK(generation>=0), "
    "PRIMARY KEY(companion,job), UNIQUE(companion,role,source,source_hash), "
    "FOREIGN KEY(companion,session) REFERENCES turn_sessions(companion,session))",
    "CREATE INDEX background_pending ON background_sources(companion,status)",
)


class SQLiteBackgroundStore(SQLiteDialogueStore):
    schema_version = 14
    schema = SQLiteDialogueStore.schema + _EXTRA
    tables = SQLiteDialogueStore.tables | {"background_sources"}
    max_pending_sources = 256

    def __init__(self, path, *, evidence_clock=None):
        # Disable the old ACK -> internal turn queue; a single source queue owns
        # this responsibility. No second user input is fabricated after an ACK.
        super().__init__(path, evidence_clock=evidence_clock, automatic_maintenance=False)

    def _upgrade(self, db, version, application, names):
        raise ValueError("Background sources require v14; existing databases are not migrated")

    def _enqueue_source(self, db, scope, role, source):
        value = self._exchange_source(db, scope, role, source)
        prefix = value["text"][:value["confirmed"]]
        if not prefix:
            raise ValueError("Source task requires actual positive confirmed content")
        db.execute(
            "INSERT INTO background_sources "
            "(companion,session,job,role,source,source_hash,confirmed_end,status) "
            "VALUES(?,?,?,?,?,?,?,'queued') "
            "ON CONFLICT(companion,role,source,source_hash) DO NOTHING",
            (scope.companion_id, scope.session_id, uuid4().hex, role, source,
             sha256(prefix.encode()).hexdigest(), len(prefix)),
        )

    def _accepted(self, db, receipt, trigger):
        if trigger is not None:
            raise ValueError("Source maintenance cannot enter the foreground turn queue")
        pending = self._pending_source_count(db, receipt.companion_id)
        if pending >= self.max_pending_sources:
            raise ValueError("Source maintenance backpressure; input was not accepted")
        super()._accepted(db, receipt, trigger)
        self._preempt_background(db, receipt.companion_id)
        db.execute(
            "UPDATE background_sources SET status='queued',generation=generation+1 "
            "WHERE companion=? AND status='running'", (receipt.companion_id,),
        )
        self._enqueue_source(db, Scope(receipt.companion_id, receipt.session_id),
                             "user", receipt.input_id)

    def _pending_source_count(self, db, companion):
        return db.execute(
            "SELECT count(*) FROM background_sources WHERE companion=? AND status<>'completed'",
            (companion,),
        ).fetchone()[0]

    def _preempt_background(self, db, companion):
        """v14 has no leased worker attempt to restore."""

    def _ack_memories(self, db, companion, row, receipt):
        super()._ack_memories(db, companion, row, receipt)
        if (receipt["disposition"] == "applied"
                and receipt["status"] in {"completed", "cancelled"}
                and receipt["displayed_end"] > 0):
            source = self._input(db, companion, row["input"])
            self._enqueue_source(db, Scope(companion, source["session"]),
                                 "assistant", row["delivery"])

    def background_sources(self, companion, *, limit=100):
        companion_id(companion)
        if type(limit) is not int or not 1 <= limit <= 256:
            raise ValueError("Bounded source queue read required")
        with self.transaction(write=False) as db:
            return [dict(row) for row in db.execute(
                "SELECT * FROM background_sources WHERE companion=? AND status<>'completed' "
                "ORDER BY rowid LIMIT ?",
                (companion, limit),
            )]

    def memory_barrier(self, companion):
        """Unknown correction targets suppress memory even after failure or restart."""
        companion_id(companion)
        with self.transaction(write=False) as db:
            return db.execute(
                "SELECT 1 FROM background_sources WHERE companion=? AND role='user' "
                "AND status<>'completed' LIMIT 1", (companion,),
            ).fetchone() is not None

    def background_source_text(self, companion, job):
        companion_id(companion)
        identifier(job)
        with self.transaction(write=False) as db:
            row = db.execute("SELECT * FROM background_sources WHERE companion=? AND job=?",
                             (companion, job)).fetchone()
            if row is None:
                raise ValueError("Unknown source job in companion")
            value = self._exchange_source(db, Scope(companion, row["session"]),
                                          row["role"], row["source"])
            prefix = value["text"][:row["confirmed_end"]]
            if (value["confirmed"] < row["confirmed_end"]
                    or sha256(prefix.encode()).hexdigest() != row["source_hash"]):
                raise VersionConflict("Background source confirmation changed")
            return prefix

    def background_source_config(self, companion, job):
        """Return the input's immutable role/config binding for this source job."""
        companion_id(companion)
        identifier(job)
        with self.transaction(write=False) as db:
            source = db.execute("SELECT * FROM background_sources WHERE companion=? AND job=?",
                                (companion, job)).fetchone()
            if source is None:
                raise ValueError("Unknown source job in companion")
            if source["role"] == "user":
                item = self._input(db, companion, source["source"])
            else:
                item = db.execute(
                    "SELECT i.* FROM turn_inputs i JOIN turn_deliveries d "
                    "ON i.companion=d.companion AND i.input=d.input "
                    "WHERE d.companion=? AND d.delivery=?", (companion, source["source"]),
                ).fetchone()
                if item is None and "stream_history" in self.tables:
                    item = db.execute(
                        "SELECT i.* FROM turn_inputs i JOIN stream_history h "
                        "ON i.companion=h.companion AND i.input=h.input "
                        "WHERE h.companion=? AND h.source=?", (companion, source["source"]),
                    ).fetchone()
            if item is None or item["session"] != source["session"]:
                raise ValueError("Unknown or foreign background source config")
            return item["config"]
