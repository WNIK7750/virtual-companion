from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..errors import (
    ConflictError,
    IdempotencyConflictError,
    NotFoundError,
    RepositoryError,
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


class SQLiteFeedbackRepository:
    """Isolated SQLite adapter for feedback data only."""

    SCHEMA_VERSION = 2

    def __init__(
        self, database_path: Path, *, production_database_path: Path | None = None
    ) -> None:
        path = Path(database_path).expanduser().resolve()
        if (
            production_database_path is not None
            and path == Path(production_database_path).expanduser().resolve()
        ):
            raise ValueError("feedback database must be isolated from the production database")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.RLock()
        self._closed = False
        self.fail_next_write = False
        try:
            self._connection = sqlite3.connect(
                str(path),
                isolation_level=None,
                check_same_thread=False,
                timeout=5,
            )
            self._connection.row_factory = sqlite3.Row
            existing_meta = self._connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='feedback_meta'"
            ).fetchone()
            if existing_meta:
                version = self._connection.execute(
                    "SELECT value FROM feedback_meta WHERE key='schema_version'"
                ).fetchone()
                if version is not None and version["value"] == "1":
                    columns = {row["name"] for row in self._connection.execute(
                        "PRAGMA table_info(feedback_events)")}
                    if "target_binding_json" not in columns:
                        self._connection.execute(
                            "ALTER TABLE feedback_events ADD COLUMN target_binding_json TEXT")
                    self._connection.execute(
                        "UPDATE feedback_meta SET value=? WHERE key='schema_version'",
                        (str(self.SCHEMA_VERSION),),
                    )
                elif version is None or version["value"] != str(self.SCHEMA_VERSION):
                    raise RepositoryError("unsupported feedback database schema")
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA busy_timeout = 5000")
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._create_schema()
        except (sqlite3.Error, RepositoryError) as exc:
            if hasattr(self, "_connection"):
                self._connection.close()
            if isinstance(exc, RepositoryError):
                raise
            raise RepositoryError("unable to open isolated feedback database") from exc

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS feedback_events (
                feedback_id TEXT NOT NULL,
                feedback_revision INTEGER NOT NULL,
                session_id TEXT NOT NULL,
                turn_id TEXT NOT NULL,
                assistant_message_id TEXT NOT NULL,
                quick_tags_json TEXT NOT NULL,
                freeform_reaction TEXT,
                supersedes_revision INTEGER,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                schema_version INTEGER NOT NULL,
                version_binding_json TEXT NOT NULL,
                target_binding_json TEXT,
                PRIMARY KEY (feedback_id, feedback_revision)
            );
            CREATE INDEX IF NOT EXISTS feedback_events_current_idx
                ON feedback_events (feedback_id, feedback_revision DESC);
            CREATE TABLE IF NOT EXISTS feedback_idempotency (
                idempotency_key TEXT PRIMARY KEY,
                feedback_id TEXT NOT NULL,
                request_hash TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS feedback_exports (
                export_id TEXT PRIMARY KEY,
                schema_version INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                selected_feedback_json TEXT NOT NULL,
                context_policy_json TEXT NOT NULL,
                redaction_confirmed INTEGER NOT NULL,
                external_processing_confirmed INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL,
                status TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS feedback_analyses (
                analysis_id TEXT PRIMARY KEY,
                schema_version INTEGER NOT NULL,
                input_export_id TEXT NOT NULL,
                input_export_sha256 TEXT NOT NULL,
                analysis_model TEXT NOT NULL,
                analysis_prompt_sha256 TEXT NOT NULL,
                generated_at TEXT NOT NULL,
                findings_json TEXT NOT NULL,
                FOREIGN KEY (input_export_id) REFERENCES feedback_exports(export_id)
                    ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS feedback_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            INSERT OR IGNORE INTO feedback_meta (key, value)
                VALUES ('schema_version', '2');
            """
        )
        row = self._connection.execute(
            "SELECT value FROM feedback_meta WHERE key = 'schema_version'"
        ).fetchone()
        if row is None or row["value"] != str(self.SCHEMA_VERSION):
            raise RepositoryError("unsupported feedback database schema")

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            if self._closed:
                raise RepositoryError("feedback repository is closed")
            if self.fail_next_write:
                self.fail_next_write = False
                raise RepositoryError("injected feedback write failure")
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                yield self._connection
                self._connection.execute("COMMIT")
            except (ConflictError, NotFoundError, StaleExportError, StaleRevisionError):
                self._connection.execute("ROLLBACK")
                raise
            except sqlite3.Error as exc:
                self._connection.execute("ROLLBACK")
                raise RepositoryError("feedback transaction failed") from exc
            except Exception:
                self._connection.execute("ROLLBACK")
                raise

    def submit(self, event: FeedbackEvent, *, idempotency_key: str) -> FeedbackEvent:
        if not idempotency_key.strip():
            raise ConflictError("idempotency_key is required")
        request_hash = feedback_idempotency_fingerprint(event)
        with self._transaction() as connection:
            row = connection.execute(
                (
                    "SELECT feedback_id, request_hash "
                    "FROM feedback_idempotency WHERE idempotency_key = ?"
                ),
                (idempotency_key,),
            ).fetchone()
            if row is not None:
                if row["request_hash"] != request_hash:
                    raise IdempotencyConflictError("idempotency key was reused for another request")
                existing = connection.execute(
                    "SELECT * FROM feedback_events WHERE feedback_id = ? AND feedback_revision = 1",
                    (row["feedback_id"],),
                ).fetchone()
                if existing is None:
                    raise RepositoryError("idempotency record points to missing feedback")
                return self._event_from_row(existing)
            duplicate = connection.execute(
                "SELECT 1 FROM feedback_events WHERE feedback_id = ? LIMIT 1",
                (event.feedback_id,),
            ).fetchone()
            if duplicate is not None:
                raise ConflictError("feedback_id already exists")
            connection.execute(
                """
                INSERT INTO feedback_events (
                    feedback_id, feedback_revision, session_id, turn_id, assistant_message_id,
                    quick_tags_json, freeform_reaction, supersedes_revision, status, created_at,
                    schema_version, version_binding_json, target_binding_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self._event_values(event),
            )
            connection.execute(
                (
                    "INSERT INTO feedback_idempotency "
                    "(idempotency_key, feedback_id, request_hash) VALUES (?, ?, ?)"
                ),
                (idempotency_key, event.feedback_id, request_hash),
            )
            return event

    def current(self, feedback_id: str) -> FeedbackEvent | None:
        with self._lock:
            if self._closed:
                raise RepositoryError("feedback repository is closed")
            row = self._connection.execute(
                (
                    "SELECT * FROM feedback_events WHERE feedback_id = ? "
                    "ORDER BY feedback_revision DESC LIMIT 1"
                ),
                (feedback_id,),
            ).fetchone()
            if row is None or row["status"] != "active":
                return None
            return self._event_from_row(row)

    def all_events(self, feedback_id: str) -> tuple[FeedbackEvent, ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM feedback_events WHERE feedback_id = ? ORDER BY feedback_revision",
                (feedback_id,),
            ).fetchall()
            return tuple(self._event_from_row(row) for row in rows)

    def list_current(self, *, session_id: str | None = None) -> tuple[FeedbackEvent, ...]:
        with self._lock:
            query = """
                SELECT event.*
                FROM feedback_events AS event
                JOIN (
                    SELECT feedback_id, MAX(feedback_revision) AS max_revision
                    FROM feedback_events GROUP BY feedback_id
                ) AS latest
                  ON latest.feedback_id = event.feedback_id
                 AND latest.max_revision = event.feedback_revision
                WHERE event.status = 'active'
            """
            parameters: tuple[Any, ...] = ()
            if session_id is not None:
                query += " AND event.session_id = ?"
                parameters = (session_id,)
            query += " ORDER BY event.created_at"
            rows = self._connection.execute(query, parameters).fetchall()
            return tuple(self._event_from_row(row) for row in rows)

    def append_revision(self, event: FeedbackEvent, *, expected_revision: int) -> FeedbackEvent:
        with self._transaction() as connection:
            row = connection.execute(
                (
                    "SELECT * FROM feedback_events WHERE feedback_id = ? "
                    "ORDER BY feedback_revision DESC LIMIT 1"
                ),
                (event.feedback_id,),
            ).fetchone()
            if row is None:
                raise NotFoundError("feedback does not exist")
            current_revision = int(row["feedback_revision"])
            if current_revision != expected_revision:
                raise StaleRevisionError("feedback revision is stale")
            if event.feedback_revision != expected_revision + 1:
                raise ConflictError("new revision must increment the current revision")
            connection.execute(
                """
                INSERT INTO feedback_events (
                    feedback_id, feedback_revision, session_id, turn_id, assistant_message_id,
                    quick_tags_json, freeform_reaction, supersedes_revision, status, created_at,
                    schema_version, version_binding_json, target_binding_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self._event_values(event),
            )
            self._mark_exports_stale(connection, event.feedback_id)
            return event

    def delete_feedback(self, feedback_id: str) -> DeletionReceipt:
        with self._transaction() as connection:
            revisions = connection.execute(
                "SELECT COUNT(*) AS count FROM feedback_events WHERE feedback_id = ?",
                (feedback_id,),
            ).fetchone()["count"]
            if revisions == 0:
                raise NotFoundError("feedback does not exist")
            export_rows = connection.execute(
                "SELECT export_id, selected_feedback_json FROM feedback_exports"
            ).fetchall()
            export_ids = [
                row["export_id"]
                for row in export_rows
                if any(
                    item.get("feedback_id") == feedback_id
                    for item in json.loads(row["selected_feedback_json"])
                )
            ]
            analysis_count = 0
            for export_id in export_ids:
                analysis_count += connection.execute(
                    "SELECT COUNT(*) AS count FROM feedback_analyses WHERE input_export_id = ?",
                    (export_id,),
                ).fetchone()["count"]
                connection.execute(
                    "DELETE FROM feedback_analyses WHERE input_export_id = ?",
                    (export_id,),
                )
                connection.execute(
                    "DELETE FROM feedback_exports WHERE export_id = ?",
                    (export_id,),
                )
            connection.execute(
                "DELETE FROM feedback_idempotency WHERE feedback_id = ?", (feedback_id,)
            )
            connection.execute("DELETE FROM feedback_events WHERE feedback_id = ?", (feedback_id,))
            return DeletionReceipt(
                feedback_id=feedback_id,
                deleted_revisions=int(revisions),
                deleted_exports=len(export_ids),
                deleted_analyses=int(analysis_count),
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
        with self._transaction() as connection:
            self._assert_current_revisions(connection, selected_feedback, expected_revisions)
            try:
                export = FeedbackExport.create(
                    export_id=export_id,
                    selected_feedback=selected_feedback,
                    context_policy=context_policy,
                    payload=payload,
                    redaction_confirmed=redaction_confirmed,
                    external_processing_confirmed=external_processing_confirmed,
                    created_at=created_at,
                )
                connection.execute(
                    """
                    INSERT INTO feedback_exports (
                        export_id, schema_version, created_at, selected_feedback_json,
                        context_policy_json, redaction_confirmed, external_processing_confirmed,
                        payload_json, payload_sha256, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        export.export_id,
                        export.schema_version,
                        export.created_at,
                        json.dumps(
                            [item.to_mapping() for item in export.selected_feedback],
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        json.dumps(export.context_policy.to_mapping(), sort_keys=True),
                        int(export.redaction_confirmed),
                        int(export.external_processing_confirmed),
                        json.dumps(export.payload, ensure_ascii=False, sort_keys=True),
                        export.payload_sha256,
                        export.status,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ConflictError("export_id already exists") from exc
            return export

    def get_export(self, export_id: str) -> FeedbackExport | None:
        with self._lock:
            if self._closed:
                raise RepositoryError("feedback repository is closed")
            row = self._connection.execute(
                "SELECT * FROM feedback_exports WHERE export_id = ?",
                (export_id,),
            ).fetchone()
            return None if row is None else self._export_from_row(row)

    def delete_export(self, export_id: str) -> None:
        with self._transaction() as connection:
            connection.execute(
                "DELETE FROM feedback_analyses WHERE input_export_id = ?", (export_id,)
            )
            connection.execute("DELETE FROM feedback_exports WHERE export_id = ?", (export_id,))

    def save_analysis(self, analysis: FeedbackAnalysis) -> FeedbackAnalysis:
        with self._transaction() as connection:
            export_row = connection.execute(
                "SELECT * FROM feedback_exports WHERE export_id = ?",
                (analysis.input_export_id,),
            ).fetchone()
            if export_row is None:
                raise NotFoundError("analysis input export does not exist")
            export = self._export_from_row(export_row)
            if export.status != "active":
                raise StaleExportError("stale export cannot receive a new analysis")
            if analysis.input_export_sha256 != export.payload_sha256:
                raise ConflictError("analysis input export hash does not match")
            selected_ids = {item.feedback_id for item in export.selected_feedback}
            for finding in analysis.findings:
                if not set(finding.evidence_feedback_ids).issubset(selected_ids):
                    raise ConflictError("analysis cites feedback outside its input export")
            try:
                connection.execute(
                    """
                    INSERT INTO feedback_analyses (
                        analysis_id, schema_version, input_export_id, input_export_sha256,
                        analysis_model, analysis_prompt_sha256, generated_at, findings_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        analysis.analysis_id,
                        analysis.schema_version,
                        analysis.input_export_id,
                        analysis.input_export_sha256,
                        analysis.analysis_model,
                        analysis.analysis_prompt_sha256,
                        analysis.generated_at,
                        json.dumps(
                            [item.to_mapping() for item in analysis.findings],
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ConflictError("analysis_id already exists") from exc
            return analysis

    def get_analysis(self, analysis_id: str) -> FeedbackAnalysis | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM feedback_analyses WHERE analysis_id = ?",
                (analysis_id,),
            ).fetchone()
            return None if row is None else self._analysis_from_row(row)

    def list_analyses(self, *, export_id: str | None = None) -> tuple[FeedbackAnalysis, ...]:
        with self._lock:
            if export_id is None:
                rows = self._connection.execute(
                    "SELECT * FROM feedback_analyses ORDER BY generated_at"
                ).fetchall()
            else:
                rows = self._connection.execute(
                    (
                        "SELECT * FROM feedback_analyses WHERE input_export_id = ? "
                        "ORDER BY generated_at"
                    ),
                    (export_id,),
                ).fetchall()
            return tuple(self._analysis_from_row(row) for row in rows)

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True

    def backup_to(self, destination: Path) -> None:
        """Create a consistent SQLite backup without exposing raw data to the renderer."""
        target_path = Path(destination).expanduser().resolve()
        if target_path == self.path:
            raise ValueError("backup destination must differ from the feedback database")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if self._closed:
                raise RepositoryError("feedback repository is closed")
            try:
                target = sqlite3.connect(str(target_path))
                try:
                    self._connection.backup(target)
                finally:
                    target.close()
            except sqlite3.Error as exc:
                raise RepositoryError("feedback backup failed") from exc

    def __enter__(self) -> SQLiteFeedbackRepository:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def _event_values(event: FeedbackEvent) -> tuple[Any, ...]:
        return (
            event.feedback_id,
            event.feedback_revision,
            event.session_id,
            event.turn_id,
            event.assistant_message_id or "",
            json.dumps(list(event.quick_tags), ensure_ascii=False),
            event.freeform_reaction,
            event.supersedes_revision,
            event.status,
            event.created_at,
            event.schema_version,
            json.dumps(event.version_binding.to_mapping(), sort_keys=True),
            (None if event.target_binding is None else json.dumps(
                event.target_binding, ensure_ascii=False, sort_keys=True)),
        )

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> FeedbackEvent:
        return FeedbackEvent.from_mapping(
            {
                "schema_version": row["schema_version"],
                "feedback_id": row["feedback_id"],
                "session_id": row["session_id"],
                "turn_id": row["turn_id"],
                "assistant_message_id": row["assistant_message_id"] or None,
                "quick_tags": json.loads(row["quick_tags_json"]),
                "freeform_reaction": row["freeform_reaction"],
                "feedback_revision": row["feedback_revision"],
                "supersedes_revision": row["supersedes_revision"],
                "status": row["status"],
                "created_at": row["created_at"],
                "version_binding": json.loads(row["version_binding_json"]),
                "target_binding": (None if row["target_binding_json"] is None else
                                   json.loads(row["target_binding_json"])),
            }
        )

    @staticmethod
    def _export_from_row(row: sqlite3.Row) -> FeedbackExport:
        return FeedbackExport.from_mapping(
            {
                "schema_version": row["schema_version"],
                "export_id": row["export_id"],
                "created_at": row["created_at"],
                "selected_feedback": json.loads(row["selected_feedback_json"]),
                "context_policy": json.loads(row["context_policy_json"]),
                "redaction_confirmed": bool(row["redaction_confirmed"]),
                "external_processing_confirmed": bool(row["external_processing_confirmed"]),
                "payload": json.loads(row["payload_json"]),
                "payload_sha256": row["payload_sha256"],
                "status": row["status"],
            }
        )

    @staticmethod
    def _analysis_from_row(row: sqlite3.Row) -> FeedbackAnalysis:
        return FeedbackAnalysis.from_mapping(
            {
                "schema_version": row["schema_version"],
                "analysis_id": row["analysis_id"],
                "input_export_id": row["input_export_id"],
                "input_export_sha256": row["input_export_sha256"],
                "analysis_model": row["analysis_model"],
                "analysis_prompt_sha256": row["analysis_prompt_sha256"],
                "generated_at": row["generated_at"],
                "findings": json.loads(row["findings_json"]),
            }
        )

    @staticmethod
    def _assert_current_revisions(
        connection: sqlite3.Connection,
        selected_feedback: Sequence[FeedbackSelection],
        expected_revisions: dict[str, int],
    ) -> None:
        for selection in selected_feedback:
            row = connection.execute(
                (
                    "SELECT feedback_revision, status FROM feedback_events "
                    "WHERE feedback_id = ? ORDER BY feedback_revision DESC LIMIT 1"
                ),
                (selection.feedback_id,),
            ).fetchone()
            if row is None or row["status"] != "active":
                raise StaleExportError("feedback is withdrawn or missing")
            if (
                row["feedback_revision"] != selection.feedback_revision
                or expected_revisions.get(selection.feedback_id) != selection.feedback_revision
            ):
                raise StaleExportError("export preview no longer matches current feedback")

    @staticmethod
    def _mark_exports_stale(connection: sqlite3.Connection, feedback_id: str) -> None:
        rows = connection.execute(
            "SELECT export_id, selected_feedback_json FROM feedback_exports WHERE status = 'active'"
        ).fetchall()
        for row in rows:
            selected = json.loads(row["selected_feedback_json"])
            if any(item.get("feedback_id") == feedback_id for item in selected):
                connection.execute(
                    "UPDATE feedback_exports SET status = 'stale' WHERE export_id = ?",
                    (row["export_id"],),
                )
