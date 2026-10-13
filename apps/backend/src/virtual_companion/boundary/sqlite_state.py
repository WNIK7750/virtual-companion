"""Isolated schema-v1 state adapter. All changes share one short SQLite transaction.

Explicit owner transfer uses epoch CAS; this is not the BR1 run-lease scheduler.
Existing foreign/forward-version stores are rejected before any DDL or journal change.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing, contextmanager
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

from virtual_companion.text.chronology import utc_microseconds

from .contracts import (
    OwnerToken,
    StateEvent,
    StateReceipt,
    StateSnapshot,
    VersionConflict,
    canonical,
    companion_id,
    identifier,
)
from .state import transition

_APPLICATION_ID = 1447252820
_SCHEMA = (
    "CREATE TABLE companion_state (companion TEXT PRIMARY KEY, "
    "version INTEGER NOT NULL, snapshot TEXT NOT NULL)",
    "CREATE TABLE state_owners (companion TEXT PRIMARY KEY REFERENCES companion_state(companion), "
    "owner TEXT, epoch INTEGER NOT NULL)",
    "CREATE TABLE state_periods (companion TEXT NOT NULL REFERENCES companion_state(companion), "
    "period_id TEXT NOT NULL, availability TEXT NOT NULL, started_us INTEGER NOT NULL, "
    "ended_us INTEGER, sequence INTEGER NOT NULL, interruptions INTEGER NOT NULL DEFAULT 0, "
    "PRIMARY KEY(companion,period_id), CHECK(ended_us IS NULL OR ended_us>=started_us))",
    "CREATE UNIQUE INDEX state_one_open_period ON state_periods(companion) WHERE ended_us IS NULL",
    "CREATE TABLE state_receipts (companion TEXT NOT NULL REFERENCES companion_state(companion), "
    "event_id TEXT NOT NULL, fingerprint TEXT NOT NULL, receipt TEXT NOT NULL, "
    "PRIMARY KEY(companion,event_id))",
)


class SQLiteStateStore:
    schema_version = 1
    schema = _SCHEMA
    tables = {"companion_state", "state_owners", "state_periods", "state_receipts"}

    def __init__(self, path: Path):
        self.path = Path(path).resolve()
        with self.transaction(check_schema=False) as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            application = db.execute("PRAGMA application_id").fetchone()[0]
            names = {r[0] for r in db.execute("SELECT name FROM sqlite_schema WHERE type='table'")}
            if version == 0 and application == 0 and not names:
                for statement in self.schema:
                    db.execute(statement)
                db.execute(f"PRAGMA application_id={_APPLICATION_ID}")
                db.execute(f"PRAGMA user_version={self.schema_version}")
            elif (
                version != self.schema_version
                or application != _APPLICATION_ID
                or names != self.tables
            ):
                self._upgrade(db, version, application, names)

    def _upgrade(self, db, version, application, names):
        raise ValueError("Unsupported or foreign state database; no migration attempted")

    @contextmanager
    def transaction(self, *, check_schema=True, write=True, timeout=5):
        db = sqlite3.connect(self.path, isolation_level=None, timeout=timeout)
        db.row_factory = sqlite3.Row
        try:
            self._prepare_connection(db, check_schema=check_schema)
            db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            if check_schema and (
                db.execute("PRAGMA user_version").fetchone()[0] != self.schema_version
                or db.execute("PRAGMA application_id").fetchone()[0] != _APPLICATION_ID
            ):
                raise ValueError("Unsupported state database")
            yield db
            db.execute("COMMIT")
        except BaseException:
            if db.in_transaction:
                db.execute("ROLLBACK")
            raise
        finally:
            db.close()

    def _prepare_connection(self, db, *, check_schema):
        db.execute("PRAGMA foreign_keys=ON")

    def backup_to(self, destination: Path) -> Path:
        """Create a stable SQLite snapshot without mutating the live store."""
        destination = Path(destination).resolve()
        if destination == self.path:
            raise ValueError("Backup destination must differ from the live database")
        if destination.exists():
            raise ValueError("Backup destination already exists")
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_uri = f"{self.path.as_uri()}?mode=ro"
        # Build privately on the same volume, then publish without replacement.
        # An exists() check alone cannot reserve a destination against another
        # writer. Never open or delete the caller's destination on failure.
        with TemporaryDirectory(prefix=".memory-backup-", dir=destination.parent) as scratch:
            snapshot = Path(scratch) / "snapshot.sqlite3"
            with closing(sqlite3.connect(source_uri, uri=True)) as source:
                with closing(sqlite3.connect(snapshot)) as target:
                    source.backup(target)
                    if target.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                        raise ValueError("Backup integrity check failed")
            os.link(snapshot, destination)
        return destination

    def register(self, companion):
        companion_id(companion)
        with self.transaction() as db:
            db.execute(
                "INSERT OR IGNORE INTO companion_state VALUES(?,0,?)",
                (companion, canonical(asdict(StateSnapshot(companion)))),
            )
            db.execute("INSERT OR IGNORE INTO state_owners VALUES(?,NULL,0)", (companion,))

    def claim_owner(self, companion, owner, *, expected_epoch):
        companion_id(companion)
        identifier(owner)
        if type(expected_epoch) is not int or expected_epoch < 0:
            raise ValueError("Expected owner epoch required")
        with self.transaction() as db:
            changed = db.execute(
                "UPDATE state_owners SET owner=?,epoch=epoch+1 WHERE companion=? AND epoch=?",
                (owner, companion, expected_epoch),
            ).rowcount
            if changed != 1:
                raise VersionConflict("Owner transfer lost epoch comparison")
        return OwnerToken(companion, owner, expected_epoch + 1)

    @staticmethod
    def _snapshot(db, companion):
        row = db.execute(
            "SELECT snapshot FROM companion_state WHERE companion=?", (companion,)
        ).fetchone()
        if row is None:
            raise ValueError("Unknown companion")
        return StateSnapshot(**json.loads(row[0]))

    def snapshot(self, companion):
        companion_id(companion)
        with self.transaction(write=False) as db:
            return self._snapshot(db, companion)

    def owner(self, companion):
        companion_id(companion)
        with self.transaction(write=False) as db:
            row = db.execute(
                "SELECT owner,epoch FROM state_owners WHERE companion=?", (companion,)
            ).fetchone()
            return OwnerToken(companion, row[0], row[1]) if row and row[0] else None

    @staticmethod
    def _receipt(db, companion, event_id):
        row = db.execute(
            "SELECT receipt FROM state_receipts WHERE companion=? AND event_id=?",
            (companion, event_id),
        ).fetchone()
        return StateReceipt(**json.loads(row[0])) if row else None

    def receipt(self, companion, event_id):
        companion_id(companion)
        identifier(event_id)
        with self.transaction(write=False) as db:
            return self._receipt(db, companion, event_id)

    def periods(self, companion):
        companion_id(companion)
        with self.transaction(write=False) as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT * FROM state_periods WHERE companion=? "
                    "ORDER BY started_us DESC,sequence DESC",
                    (companion,),
                )
            ]

    def apply(self, event: StateEvent, owner: OwnerToken, *, expected_version: int):
        if not isinstance(event, StateEvent) or not isinstance(owner, OwnerToken):
            raise ValueError("Typed event and owner required")
        companion = event.scope.companion_id
        if (
            companion != owner.companion_id
            or type(expected_version) is not int
            or expected_version < 0
        ):
            raise ValueError("Scope or expected version mismatch")
        with self.transaction() as db:
            return self._apply(db, event, owner, expected_version=expected_version)

    def _apply(self, db, event, owner, *, expected_version):
        """Adapter-only composition: caller owns this connection and transaction."""
        companion = event.scope.companion_id
        original = self._receipt(db, companion, event.event_id)
        if original is not None:
            if original.fingerprint != event.fingerprint:
                raise ValueError("Conflicting event identity")
            return original
        writer = db.execute(
            "SELECT owner,epoch FROM state_owners WHERE companion=?", (companion,)
        ).fetchone()
        if writer is None or tuple(writer) != (owner.owner_id, owner.epoch):
            raise VersionConflict("Stale state writer")
        before = self._snapshot(db, companion)
        if before.write_version != expected_version:
            raise VersionConflict("Stale state snapshot")
        after, receipt, action = transition(before, event)
        receipt = self._decorate_state_receipt(db, before, event, receipt, action)
        if action == "replace":
            db.execute(
                "UPDATE state_periods SET ended_us=? WHERE companion=? AND ended_us IS NULL",
                (utc_microseconds(event.occurred_at), companion),
            )
            db.execute(
                "INSERT INTO state_periods VALUES(?,?,?,?,NULL,?,0)",
                (
                    companion,
                    after.period_id,
                    after.availability,
                    utc_microseconds(event.occurred_at),
                    after.sequence,
                ),
            )
        elif action == "increment":
            if (
                db.execute(
                    "UPDATE state_periods SET interruptions=interruptions+1 "
                    "WHERE companion=? AND period_id=? AND ended_us IS NULL",
                    (companion, before.period_id),
                ).rowcount
                != 1
            ):
                raise VersionConflict("Current period not available")
        if (
            db.execute(
                "UPDATE companion_state SET version=?,snapshot=? WHERE companion=? AND version=?",
                (after.write_version, canonical(asdict(after)), companion, expected_version),
            ).rowcount
            != 1
        ):
            raise VersionConflict("State write lost comparison")
        db.execute(
            "INSERT INTO state_receipts VALUES(?,?,?,?)",
            (companion, event.event_id, event.fingerprint, canonical(receipt.to_mapping())),
        )
        return receipt

    def _decorate_state_receipt(self, db, before, event, receipt, action):
        return receipt
