"""Explicit failed-attempt retry; original intake and business commit remain immutable.

The current run is a projection. Failed attempts are archived before replacement;
lease recovery retains the same attempt. No exception handler guesses commit failure.
"""

import json
from hashlib import sha256

from virtual_companion.text.identity import CompanionConfig

from .contracts import VersionConflict, canonical, companion_id, identifier
from .sqlite_presentation import SQLitePresentationStore
from .sqlite_state import _APPLICATION_ID
from .sqlite_turns import _time
from .turns import RunClaim

_EXTRA = (
    "CREATE TABLE turn_attempt_history (companion TEXT NOT NULL, input TEXT NOT NULL, "
    "attempt TEXT NOT NULL, fingerprint TEXT NOT NULL, record TEXT NOT NULL, "
    "PRIMARY KEY(companion,attempt), FOREIGN KEY(companion,input) "
    "REFERENCES turn_inputs(companion,input))",
    "CREATE TABLE turn_retry_bindings (companion TEXT NOT NULL, input TEXT NOT NULL, "
    "config TEXT NOT NULL, PRIMARY KEY(companion,input), FOREIGN KEY(companion,input) "
    "REFERENCES turn_inputs(companion,input))",
    "CREATE TABLE turn_retry_receipts (companion TEXT NOT NULL, retry TEXT NOT NULL, "
    "fingerprint TEXT NOT NULL, receipt TEXT NOT NULL, PRIMARY KEY(companion,retry))",
)


class SQLiteLifecycleStore(SQLitePresentationStore):
    schema_version = 6
    schema = (
        tuple(
            s.replace("'committed','cancelled'", "'committed','cancelled','failed'")
            if s.startswith(("CREATE TABLE turn_inputs ", "CREATE TABLE turn_runs "))
            else s
            for s in SQLitePresentationStore.schema
        )
        + _EXTRA
    )
    tables = SQLitePresentationStore.tables | {
        "turn_attempt_history",
        "turn_retry_bindings",
        "turn_retry_receipts",
    }

    def __init__(self, path, *, upgrade_v5=False):
        self.upgrade_v5 = upgrade_v5
        super().__init__(path)

    def _upgrade(self, db, version, application, names):
        if not (
            self.upgrade_v5
            and version == 5
            and application == _APPLICATION_ID
            and names == SQLitePresentationStore.tables
        ):
            raise ValueError("Lifecycle requires v6 or explicit upgrade of a v5 synthetic copy")
        # SQLite generalized ALTER TABLE: FK checks disabled before BEGIN only for
        # this explicit upgrade connection, followed by foreign_key_check before commit.
        objects = list(
            db.execute(
                "SELECT sql FROM sqlite_schema WHERE tbl_name IN "
                "('turn_inputs','turn_runs') AND type IN ('index','trigger') "
                "AND sql IS NOT NULL"
            )
        )
        for table in ("turn_inputs", "turn_runs"):
            statement = next(s for s in self.schema if s.startswith(f"CREATE TABLE {table} "))
            db.execute(statement.replace(f"CREATE TABLE {table} ", f"CREATE TABLE {table}_new ", 1))
            db.execute(f"INSERT INTO {table}_new SELECT * FROM {table}")  # noqa: S608 - fixed names
            db.execute(f"DROP TABLE {table}")
            db.execute(f"ALTER TABLE {table}_new RENAME TO {table}")
        for row in objects:
            db.execute(row[0])
        for statement in _EXTRA:
            db.execute(statement)
        if db.execute("PRAGMA foreign_key_check").fetchone():
            raise ValueError("Invalid synthetic upgrade references")
        db.execute("PRAGMA user_version=6")

    def _prepare_connection(self, db, *, check_schema):
        if not check_schema and self.upgrade_v5:
            db.execute("PRAGMA foreign_keys=OFF")
        else:
            super()._prepare_connection(db, check_schema=check_schema)

    def _run_config(self, db, item):
        row = db.execute(
            "SELECT config FROM turn_retry_bindings WHERE companion=? AND input=?",
            (item["companion"], item["input"]),
        ).fetchone()
        return row[0] if row else item["config"]

    def fail(self, claim, *, reason, now_us):
        if not isinstance(claim, RunClaim):
            raise ValueError("Typed claim required")
        identifier(reason)
        _time(now_us)
        key = (claim.companion_id, claim.input_id)
        fingerprint = sha256(canonical([*key, claim.attempt_id, reason]).encode()).hexdigest()
        with self.transaction() as db:
            # A lost success response is always reconciled before classifying failure.
            committed = db.execute(
                "SELECT receipt FROM turn_commits WHERE companion=? AND input=?", key
            ).fetchone()
            if committed:
                return json.loads(committed[0])
            internal = self._completed_internal(db, *key)
            if internal is not None:
                return internal
            prior = db.execute(
                "SELECT * FROM turn_attempt_history WHERE companion=? AND attempt=?",
                (key[0], claim.attempt_id),
            ).fetchone()
            if prior:
                if prior["fingerprint"] != fingerprint:
                    raise ValueError("Conflicting failure reason")
                return json.loads(prior["record"])
            run = db.execute(
                "SELECT * FROM turn_runs WHERE companion=? AND input=?", key
            ).fetchone()
            item = self._input(db, *key)
            owner = db.execute(
                "SELECT owner,epoch FROM state_owners WHERE companion=?", key[:1]
            ).fetchone()
            if (
                run is None
                or item["status"] != "running"
                or run["status"] != "running"
                or run["attempt"] != claim.attempt_id
                or run["lease_until"] <= now_us
                or (run["owner"], run["epoch"]) != (claim.owner.owner_id, claim.owner.epoch)
                or tuple(owner) != (claim.owner.owner_id, claim.owner.epoch)
                or item["session"] != claim.session_id
                or self._run_config(db, item) != claim.config_json
            ):
                raise VersionConflict("Stale failure; reconcile or recover the original attempt")
            record = {
                **dict(run),
                "status": "failed",
                "reason": reason,
                "config": self._run_config(db, item),
                "failed_at_us": now_us,
            }
            db.execute(
                "INSERT INTO turn_attempt_history VALUES(?,?,?,?,?)",
                (*key, claim.attempt_id, fingerprint, canonical(record)),
            )
            for table in ("turn_inputs", "turn_runs"):
                db.execute(
                    f"UPDATE {table} SET status='failed' WHERE companion=? AND input=?",  # noqa: S608
                    key,
                )
            return record

    def retry(self, companion, input_id, failed_attempt, retry_id, config):
        companion_id(companion)
        for value in (input_id, failed_attempt, retry_id):
            identifier(value)
        if not isinstance(config, CompanionConfig) or config.companion_id != companion:
            raise ValueError("Retry configuration must belong to the same companion")
        binding = canonical(config.attempt_binding({}))
        fingerprint = sha256(canonical([input_id, failed_attempt, binding]).encode()).hexdigest()
        with self.transaction() as db:
            return self._retry_in_transaction(
                db, companion, input_id, failed_attempt, retry_id, binding, fingerprint
            )

    def _retry_in_transaction(
        self, db, companion, input_id, failed_attempt, retry_id, binding, fingerprint,
        *, details=None,
    ):
        """Share the retry commit with adapters that atomically archive their outcome."""
        key = (companion, input_id)
        original = db.execute(
            "SELECT * FROM turn_retry_receipts WHERE companion=? AND retry=?",
            (companion, retry_id),
        ).fetchone()
        if original:
            if original["fingerprint"] != fingerprint:
                raise ValueError("Conflicting retry command")
            return json.loads(original["receipt"])
        item = self._input(db, *key)
        run = db.execute(
            "SELECT * FROM turn_runs WHERE companion=? AND input=?", key
        ).fetchone()
        if (
            item["status"] != "failed"
            or run is None
            or run["status"] != "failed"
            or run["attempt"] != failed_attempt
        ):
            raise VersionConflict("Only an explicitly failed current attempt can be retried")
        self._check_retry_effects(db, companion, input_id)
        receipt = {
            **(details or {}),
            "companion_id": companion,
            "input_id": input_id,
            "retry_id": retry_id,
            "failed_attempt_id": failed_attempt,
            "status": "queued",
            "config": binding,
        }
        db.execute(
            "INSERT INTO turn_retry_bindings VALUES(?,?,?) ON CONFLICT(companion,input) "
            "DO UPDATE SET config=excluded.config",
            (*key, binding),
        )
        db.execute("DELETE FROM turn_runs WHERE companion=? AND input=?", key)
        db.execute("UPDATE turn_inputs SET status='queued' WHERE companion=? AND input=?", key)
        db.execute(
            "INSERT INTO turn_retry_receipts VALUES(?,?,?,?)",
            (companion, retry_id, fingerprint, canonical(receipt)),
        )
        return receipt

    def _check_retry_effects(self, db, companion, input_id):
        """Specialized journals may fence unresolved effects in this transaction."""

    def attempts(self, companion, input_id):
        companion_id(companion)
        identifier(input_id)
        with self.transaction(write=False) as db:
            item = self._input(db, companion, input_id)
            records = [
                json.loads(row[0])
                for row in db.execute(
                    "SELECT record FROM turn_attempt_history WHERE companion=? AND input=? "
                    "ORDER BY rowid",
                    (companion, input_id),
                )
            ]
            current = db.execute(
                "SELECT * FROM turn_runs WHERE companion=? AND input=?", (companion, input_id)
            ).fetchone()
            if current and not any(r["attempt"] == current["attempt"] for r in records):
                records.append({**dict(current), "config": self._run_config(db, item)})
            return records
