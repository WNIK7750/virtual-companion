"""Opt-in schema-v3 memory and ACK effects in the shared companion transaction.

Only synthetic copies are upgraded explicitly. Recall filters current source
versions rather than trusting stale index hits. No automatic semantic recall.
"""

import json
from dataclasses import asdict, replace
from datetime import datetime
from hashlib import sha256

from virtual_companion.text.chronology import utc_microseconds

from .contracts import OwnerToken, VersionConflict, canonical, companion_id, identifier
from .memory import MemoryMutation, evolve
from .sqlite_state import _APPLICATION_ID
from .sqlite_turns import SQLiteTurnStore, _time

_MEMORY_SCHEMA = (
    "CREATE TABLE memory_heads (companion TEXT NOT NULL REFERENCES companion_state(companion), "
    "record TEXT NOT NULL, slot TEXT, version INTEGER NOT NULL, PRIMARY KEY(companion,record), "
    "UNIQUE(companion,slot))",
    "CREATE TABLE memory_versions (companion TEXT NOT NULL, record TEXT NOT NULL, "
    "version INTEGER NOT NULL, status TEXT NOT NULL, expires_us INTEGER, data TEXT NOT NULL, "
    "PRIMARY KEY(companion,record,version), "
    "FOREIGN KEY(companion,record) REFERENCES memory_heads(companion,record))",
    "CREATE TABLE memory_commands (companion TEXT NOT NULL REFERENCES companion_state(companion), "
    "mutation TEXT NOT NULL, fingerprint TEXT NOT NULL, payload TEXT NOT NULL, "
    "receipt TEXT NOT NULL, "
    "PRIMARY KEY(companion,mutation))",
    "CREATE TABLE memory_effects (companion TEXT NOT NULL, delivery TEXT NOT NULL, "
    "mutation TEXT NOT NULL, minimum_end INTEGER NOT NULL, payload TEXT NOT NULL, "
    "status TEXT NOT NULL DEFAULT 'pending', PRIMARY KEY(companion,mutation), "
    "FOREIGN KEY(companion,delivery) REFERENCES turn_deliveries(companion,delivery))",
)


class SQLiteMemoryStore(SQLiteTurnStore):
    schema_version = 3
    schema = SQLiteTurnStore.schema + _MEMORY_SCHEMA
    tables = SQLiteTurnStore.tables | {
        "memory_heads",
        "memory_versions",
        "memory_commands",
        "memory_effects",
    }

    def __init__(self, path, *, upgrade_v2=False):
        self.upgrade_v2 = upgrade_v2
        super().__init__(path)

    def _upgrade(self, db, version, application, names):
        if not (
            self.upgrade_v2
            and version == 2
            and application == _APPLICATION_ID
            and names == SQLiteTurnStore.tables
        ):
            raise ValueError(
                "Memory adapter requires v3 or explicit upgrade of a v2 synthetic copy"
            )
        for statement in _MEMORY_SCHEMA:
            db.execute(statement)
        db.execute("PRAGMA user_version=3")

    @staticmethod
    def _original(db, companion, mutation_id):
        return db.execute(
            "SELECT * FROM memory_commands WHERE companion=? AND mutation=?",
            (companion, mutation_id),
        ).fetchone()

    def memory_receipt(self, companion, mutation_id):
        companion_id(companion)
        identifier(mutation_id)
        with self.transaction(write=False) as db:
            original = self._original(db, companion, mutation_id)
            return json.loads(original["receipt"]) if original else None

    @staticmethod
    def _current(db, companion, record_id):
        row = db.execute(
            "SELECT v.data FROM memory_heads h JOIN memory_versions v "
            "ON h.companion=v.companion AND h.record=v.record AND h.version=v.version "
            "WHERE h.companion=? AND h.record=?",
            (companion, record_id),
        ).fetchone()
        return json.loads(row[0]) if row else None

    def mutate(self, mutation, owner, *, expected_version):
        if not isinstance(mutation, MemoryMutation) or not isinstance(owner, OwnerToken):
            raise ValueError("Typed memory command and owner required")
        companion = mutation.scope.companion_id
        if (
            companion != owner.companion_id
            or type(expected_version) is not int
            or expected_version < 0
            or mutation.source_kind == "output"
        ):
            raise ValueError("Output effects require delivery; scope and version must match")
        with self.transaction() as db:
            original = self._original(db, companion, mutation.mutation_id)
            if original:
                return self._apply_memory(db, mutation)
            if db.execute(
                "SELECT 1 FROM memory_effects WHERE companion=? AND mutation=?",
                (companion, mutation.mutation_id),
            ).fetchone():
                raise VersionConflict("Identity reserved for an output effect")
            writer = db.execute(
                "SELECT owner,epoch FROM state_owners WHERE companion=?", (companion,)
            ).fetchone()
            if writer is None or tuple(writer) != (owner.owner_id, owner.epoch):
                raise VersionConflict("Stale memory writer")
            if self._snapshot(db, companion).write_version != expected_version:
                raise VersionConflict("Stale companion state for memory mutation")
            return self._apply_memory(db, mutation)

    def mutate_batch(self, mutations, owner, *, expected_version, now_us):
        """Atomic validated extraction; one invalid/stale operation rolls back all.

        Replays must match the whole batch. A partial replay is a conflicting
        batch, not permission to combine an old operation with new side effects.
        """
        _time(now_us)
        if (
            not isinstance(mutations, tuple)
            or len(mutations) > 16
            or not isinstance(owner, OwnerToken)
            or type(expected_version) is not int
            or expected_version < 0
        ):
            raise ValueError("Typed bounded batch, owner and state version required")
        if not mutations:
            return []
        companion = owner.companion_id
        if any(
            not isinstance(m, MemoryMutation)
            or m.scope.companion_id != companion
            or m.source_kind == "output"
            for m in mutations
        ) or len({m.mutation_id for m in mutations}) != len(mutations):
            raise ValueError("Unique scoped commands required; output must wait for ACK")
        batch_sha256 = sha256(canonical([m.fingerprint for m in mutations]).encode()).hexdigest()
        with self.transaction() as db:
            originals = [self._original(db, companion, m.mutation_id) for m in mutations]
            if any(originals):
                if not all(originals) or any(
                    original["fingerprint"] != m.fingerprint
                    or json.loads(original["receipt"]).get("batch_sha256") != batch_sha256
                    for original, m in zip(originals, mutations, strict=True)
                ):
                    raise VersionConflict("Conflicting or partial batch replay")
                return [json.loads(original["receipt"]) for original in originals]
            writer = db.execute(
                "SELECT owner,epoch FROM state_owners WHERE companion=?", (companion,)
            ).fetchone()
            if (
                writer is None
                or tuple(writer) != (owner.owner_id, owner.epoch)
                or self._snapshot(db, companion).write_version != expected_version
            ):
                raise VersionConflict("Stale memory batch owner or state version")
            receipts = []
            for mutation in mutations:
                if db.execute(
                    "SELECT 1 FROM memory_effects WHERE companion=? AND mutation=?",
                    (companion, mutation.mutation_id),
                ).fetchone():
                    raise VersionConflict("Identity reserved for output effect")
                before = self._current(db, companion, mutation.record_id)
                if (mutation.expires_at and utc_microseconds(mutation.expires_at) <= now_us) or (
                    before
                    and before["expires_at"]
                    and utc_microseconds(datetime.fromisoformat(before["expires_at"])) <= now_us
                ):
                    raise VersionConflict("Memory batch targets an expired fact")
                receipt = self._apply_memory(db, mutation)
                if receipt["disposition"] != "applied":
                    raise VersionConflict("Memory batch contains a conflicting operation")
                receipt["batch_sha256"] = batch_sha256
                db.execute(
                    "UPDATE memory_commands SET receipt=? WHERE companion=? AND mutation=?",
                    (canonical(receipt), companion, mutation.mutation_id),
                )
                receipts.append(receipt)
            return receipts

    def _apply_memory(self, db, mutation, *, delivery_id=None):
        companion = mutation.scope.companion_id
        original = self._original(db, companion, mutation.mutation_id)
        if original:
            if original["fingerprint"] != mutation.fingerprint:
                raise ValueError("Conflicting memory mutation identity")
            return json.loads(original["receipt"])
        before = self._current(db, companion, mutation.record_id)
        slot = canonical(asdict(mutation.fact_key)) if mutation.fact_key else None
        occupied = (
            slot is not None
            and db.execute(
                "SELECT 1 FROM memory_heads WHERE companion=? AND slot=?", (companion, slot)
            ).fetchone()
            is not None
        )
        after, disposition = evolve(mutation, before, slot_taken=occupied)
        if after:
            if before is None:
                db.execute(
                    "INSERT INTO memory_heads VALUES(?,?,?,?)",
                    (companion, mutation.record_id, slot, after["version"]),
                )
            else:
                db.execute(
                    "UPDATE memory_heads SET version=? WHERE companion=? AND record=?",
                    (after["version"], companion, mutation.record_id),
                )
            after["delivery_id"] = delivery_id
            db.execute(
                "INSERT INTO memory_versions VALUES(?,?,?,?,?,?)",
                (
                    companion,
                    mutation.record_id,
                    after["version"],
                    after["status"],
                    utc_microseconds(datetime.fromisoformat(after["expires_at"]))
                    if after["expires_at"] else None,
                    canonical(after),
                ),
            )
        basis = self._snapshot(db, companion)
        state = replace(basis, write_version=basis.write_version + 1)
        if (
            db.execute(
                "UPDATE companion_state SET version=?,snapshot=? WHERE companion=? AND version=?",
                (state.write_version, canonical(asdict(state)), companion, basis.write_version),
            ).rowcount
            != 1
        ):
            raise VersionConflict("Memory state write lost version comparison")
        receipt = {
            "companion_id": companion,
            "mutation_id": mutation.mutation_id,
            "record_id": mutation.record_id,
            "disposition": disposition,
            "record_version": after["version"] if after else None,
            "write_version": state.write_version,
            "delivery_id": delivery_id,
        }
        db.execute(
            "INSERT INTO memory_commands VALUES(?,?,?,?,?)",
            (
                companion,
                mutation.mutation_id,
                mutation.fingerprint,
                canonical(mutation.to_mapping()),
                canonical(receipt),
            ),
        )
        return receipt

    def _commit_memories(self, db, result, receipt):
        for effect in result.memories:
            mutation = effect.mutation
            companion = mutation.scope.companion_id
            if (
                self._original(db, companion, mutation.mutation_id)
                or db.execute(
                    "SELECT 1 FROM memory_effects WHERE companion=? AND mutation=?",
                    (companion, mutation.mutation_id),
                ).fetchone()
            ):
                raise ValueError("Memory effect identity already belongs to an earlier command")
            if effect.after_codepoints is None:
                self._apply_memory(db, mutation)
            else:
                db.execute(
                    "INSERT INTO memory_effects VALUES(?,?,?,?,?,'pending')",
                    (
                        companion,
                        receipt["delivery_id"],
                        mutation.mutation_id,
                        effect.after_codepoints,
                        canonical(mutation.to_mapping()),
                    ),
                )

    def _ack_memories(self, db, companion, row, receipt):
        if receipt["disposition"] != "applied":
            return
        effects = db.execute(
            "SELECT * FROM memory_effects WHERE companion=? AND delivery=? "
            "AND status='pending' ORDER BY minimum_end,rowid",
            (companion, row["delivery"]),
        ).fetchall()
        for effect in effects:
            if effect["minimum_end"] <= receipt["displayed_end"]:
                command = MemoryMutation.from_mapping(json.loads(effect["payload"]))
                outcome = self._apply_memory(db, command, delivery_id=row["delivery"])
                db.execute(
                    "UPDATE memory_effects SET status=? WHERE companion=? AND mutation=?",
                    (outcome["disposition"], companion, command.mutation_id),
                )
        self._cancel_memories(db, companion, self._delivery(db, companion, row["delivery"]))

    def _cancel_memories(self, db, companion, row):
        if row["status"] == "cancelled":
            db.execute(
                "UPDATE memory_effects SET status='cancelled' "
                "WHERE companion=? AND delivery=? AND status='pending'",
                (companion, row["delivery"]),
            )

    def effects(self, companion, delivery_id):
        with self.transaction(write=False) as db:
            self._delivery(db, companion, delivery_id)
            return [
                dict(row)
                for row in db.execute(
                    "SELECT mutation,minimum_end,status FROM memory_effects "
                    "WHERE companion=? AND delivery=? ORDER BY rowid",
                    (companion, delivery_id),
                )
            ]

    def history(self, companion, record_id):
        companion_id(companion)
        identifier(record_id)
        with self.transaction(write=False) as db:
            return [
                json.loads(row[0])
                for row in db.execute(
                    "SELECT data FROM memory_versions WHERE companion=? AND record=? "
                    "ORDER BY version",
                    (companion, record_id),
                )
            ]

    def recall(self, companion, *, now_us, candidates=None, limit=100):
        companion_id(companion)
        _time(now_us)
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("Recall limit must be between 1 and 100")
        if candidates is not None:
            if not isinstance(candidates, tuple) or len(candidates) > 100:
                raise ValueError("At most 100 immutable index candidates")
            for candidate in candidates:
                if not isinstance(candidate, tuple) or len(candidate) != 2:
                    raise ValueError("Record/version index candidate required")
                identifier(candidate[0])
                if type(candidate[1]) is not int or candidate[1] < 1:
                    raise ValueError("Positive index source version required")
        with self.transaction(write=False) as db:
            return self._recall(db, companion, now_us=now_us,
                                candidates=candidates, limit=limit)

    def _recall(self, db, companion, *, now_us, candidates, limit):
        base = (
            "SELECT v.data FROM memory_heads h JOIN memory_versions v "
            "ON h.companion=v.companion AND h.record=v.record AND h.version=v.version "
            "WHERE h.companion=? AND v.status='active' "
            "AND (v.expires_us IS NULL OR v.expires_us>?) "
        )
        if candidates is None:
            rows = db.execute(base + "ORDER BY h.rowid LIMIT ?", (companion, now_us, limit))
            return [json.loads(row[0]) for row in rows]
        found = []
        for record, version in dict.fromkeys(candidates):
            row = db.execute(
                base + "AND h.record=? AND h.version=?", (companion, now_us, record, version)
            ).fetchone()
            if row:
                found.append(json.loads(row[0]))
            if len(found) == limit:
                break
        return found
