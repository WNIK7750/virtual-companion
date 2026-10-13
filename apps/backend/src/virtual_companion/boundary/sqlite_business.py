"""Opt-in business commit policy on the v12 schema; no implicit migration."""

import json
from datetime import datetime
from hashlib import sha256

from virtual_companion.text.chronology import utc_microseconds

from .contracts import VersionConflict, companion_id, identifier
from .memory_support import command_digest, target_digest
from .sqlite_timeline import SQLiteTimelineStore


class SQLiteBusinessStore(SQLiteTimelineStore):
    def fact_target_catalog(self, companion, *, now_us):
        """List every active fact identity up to an explicit bounded selection cap."""
        companion_id(companion)
        if type(now_us) is not int or not 0 <= now_us < 2**63:
            raise ValueError("Bounded UTC fact observation time required")
        with self.transaction(write=False) as db:
            rows = db.execute(
                "SELECT v.data FROM memory_heads h JOIN memory_versions v "
                "ON h.companion=v.companion AND h.record=v.record AND h.version=v.version "
                "WHERE h.companion=? AND v.status='active' "
                "AND (v.expires_us IS NULL OR v.expires_us>?) "
                "AND json_extract(v.data,'$.kind')='fact' ORDER BY h.rowid LIMIT 1025",
                (companion, now_us),
            ).fetchall()
        if len(rows) > 1024:
            raise ValueError("Fact target catalog capacity exceeded")
        return tuple(
            {
                "record_id": value["record_id"],
                "version": value["version"],
                "fact_key": value.get("fact_key"),
                "content": value["content"][:240],
                "content_truncated": len(value["content"]) > 240,
            }
            for row in rows
            for value in (json.loads(row[0]),)
        )

    def fact_targets(self, companion, *, now_us, candidates=None):
        # Full bounded baseline for B. Never label an arbitrary first 100 complete.
        companion_id(companion)
        if type(now_us) is not int or not 0 <= now_us < 2**63:
            raise ValueError("Bounded UTC fact observation time required")
        if candidates is not None:
            if (not isinstance(candidates, tuple) or len(candidates) > 100
                    or any(not isinstance(item, tuple) or len(item) != 2
                           for item in candidates)):
                raise ValueError("Unique bounded fact target references required")
            for record, version in candidates:
                identifier(record)
                if type(version) is not int or version < 1:
                    raise ValueError("Positive fact target version required")
            if len(set(candidates)) != len(candidates):
                raise ValueError("Duplicate fact target reference")
        with self.transaction(write=False) as db:
            if candidates is not None:
                targets = {}
                for record, version in candidates:
                    row = db.execute(
                        "SELECT v.data FROM memory_heads h JOIN memory_versions v "
                        "ON h.companion=v.companion AND h.record=v.record "
                        "AND h.version=v.version WHERE h.companion=? AND h.record=? "
                        "AND h.version=? AND v.status='active' "
                        "AND (v.expires_us IS NULL OR v.expires_us>?) "
                        "AND json_extract(v.data,'$.kind')='fact'",
                        (companion, record, version, now_us),
                    ).fetchone()
                    if row is None:
                        raise VersionConflict("Selected fact target revision is no longer active")
                    targets[record] = {**json.loads(row[0]), "companion_id": companion}
                return targets
            rows = db.execute(
                "SELECT v.data FROM memory_heads h JOIN memory_versions v "
                "ON h.companion=v.companion AND h.record=v.record AND h.version=v.version "
                "WHERE h.companion=? AND v.status='active' "
                "AND (v.expires_us IS NULL OR v.expires_us>?) "
                "AND json_extract(v.data,'$.kind')='fact' ORDER BY h.rowid LIMIT 101",
                (companion, now_us),
            ).fetchall()
            if len(rows) > 100:
                raise ValueError("Fact target capacity requires explicit retrieval selection")
            return {
                value["record_id"]: {**value, "companion_id": companion}
                for row in rows
                for value in (json.loads(row[0]),)
            }

    def _commit_memories(self, db, result, receipt):
        facts = []
        for effect in result.memories:
            command = effect.mutation
            if command.kind not in {"assert_fact", "correct", "revoke"}:
                continue
            if command.source_kind != "input" or command.source_id != result.claim.input_id:
                raise ValueError("Business fact effects require the actual current user source")
            source = self._exchange_source(db, command.scope, "user", command.source_id)
            reason = json.loads(command.reason)
            if not isinstance(reason, dict) or set(reason) != {"quote", "support"}:
                raise ValueError("Business fact requires an operation-specific support receipt")
            quote, support = reason["quote"], reason["support"]
            before = self._current(db, command.scope.companion_id, command.record_id)
            if (
                not isinstance(quote, str)
                or not quote.strip()
                or quote not in source["text"]
                or not isinstance(support, dict)
                or support.get("verdict") != "supported"
                or support.get("source_sha256") != sha256(source["text"].encode()).hexdigest()
                or support.get("content_sha256") != sha256(command.content.encode()).hexdigest()
                or support.get("command_sha256") != command_digest(command, quote)
                or support.get("target_sha256") != target_digest(before)
            ):
                raise ValueError("Business memory support differs from actual source or target")
            identifier(support.get("evaluator"))
            now = self.evidence_clock()
            if (command.expires_at and utc_microseconds(command.expires_at) <= now) or (
                before
                and before.get("expires_at")
                and utc_microseconds(datetime.fromisoformat(before["expires_at"])) <= now
            ):
                raise VersionConflict("Business memory target or proposal expired before commit")
            facts.append(command)
        super()._commit_memories(db, result, receipt)
        for command in facts:
            saved = self._original(db, command.scope.companion_id, command.mutation_id)
            if saved is None or json.loads(saved["receipt"])["disposition"] != "applied":
                raise VersionConflict("Business fact batch must apply atomically with its reply")
