"""Opt-in v19 explicit memory management on the existing state owner.

The store reuses memory mutations, source leases and the shared SQLite
transaction.  It adds operation receipts and source invalidations; no existing
database is upgraded implicitly.
"""

import json
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
from heapq import nlargest

from virtual_companion.text.chronology import utc_microseconds

from .contracts import Scope, SemanticConflict, VersionConflict, canonical, companion_id, identifier
from .history_readset import (
    evidence_hash,
    validate_current_operation,
    validate_readset,
    validate_source_processing,
)
from .memory import MemoryMutation
from .memory_management import ExplicitMemoryCommand, list_source_digest
from .sqlite_background_worker import BackgroundClaim
from .sqlite_streaming import _DIALOGUE, SQLiteStreamingStore
from .sqlite_turns import _locked_time, _time
from .turns import RunClaim


def _time_from_iso(value):
    return utc_microseconds(datetime.fromisoformat(value))

_EXTRA = (
    "CREATE TABLE memory_management_meta (companion TEXT PRIMARY KEY "
    "REFERENCES companion_state(companion), generation INTEGER NOT NULL DEFAULT 0)",
    "CREATE TABLE memory_management_operations (companion TEXT NOT NULL, "
    "operation TEXT NOT NULL, fingerprint TEXT NOT NULL, payload TEXT NOT NULL, "
    "receipt TEXT NOT NULL, PRIMARY KEY(companion,operation), "
    "FOREIGN KEY(companion) REFERENCES companion_state(companion))",
    "CREATE TABLE memory_invalidated_sources (companion TEXT NOT NULL, source TEXT NOT NULL, "
    "record TEXT NOT NULL, version INTEGER NOT NULL, operation TEXT NOT NULL, "
    "PRIMARY KEY(companion,source,record,version), "
    "FOREIGN KEY(companion,operation) "
    "REFERENCES memory_management_operations(companion,operation))",
)


class SQLiteExplicitMemoryStore(SQLiteStreamingStore):
    supports_history_readset = True
    schema_version = 19
    schema = SQLiteStreamingStore.schema + _EXTRA
    tables = SQLiteStreamingStore.tables | {
        "memory_management_meta",
        "memory_management_operations",
        "memory_invalidated_sources",
    }
    clarification_errors = SQLiteStreamingStore.clarification_errors | frozenset(
        {
            "needs_clarification.memory_command",
            "needs_clarification.memory_scope",
            "needs_clarification.memory_version_changed",
            "needs_clarification.memory_generation_changed",
            "needs_clarification.mixed_source",
        }
    )

    def _upgrade(self, db, version, application, names):
        raise ValueError("Explicit memory management requires a fresh v19 database")

    @staticmethod
    def _generation(db, companion):
        row = db.execute(
            "SELECT generation FROM memory_management_meta WHERE companion=?", (companion,)
        ).fetchone()
        return row[0] if row else 0

    @staticmethod
    def _ensure_meta(db, companion):
        db.execute(
            "INSERT OR IGNORE INTO memory_management_meta(companion,generation) VALUES(?,0)",
            (companion,),
        )

    def claim_background(self, companion, worker, *, now_us, lease_us, role=None, source=None,
                         foreground=None, clock=None):
        claim = super().claim_background(
            companion, worker, now_us=now_us, lease_us=lease_us, role=role, source=source,
            foreground=foreground, clock=clock,
        )
        if claim is None:
            return None
        with self.transaction(write=False) as db:
            generation = self._generation(db, companion)
        return replace(claim, memory_generation=generation)

    def _live(self, db, claim, now_us):
        row = super()._live(db, claim, now_us)
        if self._generation(db, claim.companion) != claim.memory_generation:
            raise VersionConflict("Explicit memory operation invalidated background basis")
        return row

    def _apply_memory(self, db, mutation, *, delivery_id=None):
        original = self._original(db, mutation.scope.companion_id, mutation.mutation_id)
        if original is None and db.execute(
            "SELECT 1 FROM memory_invalidated_sources WHERE companion=? AND source=? LIMIT 1",
            (mutation.scope.companion_id, mutation.source_id),
        ).fetchone():
            raise SemanticConflict("Invalidated source cannot recreate memory")
        return super()._apply_memory(db, mutation, delivery_id=delivery_id)

    @staticmethod
    def _suppressed(db, companion, record, version):
        return db.execute(
            "SELECT 1 FROM memory_invalidated_sources x "
            "JOIN memory_management_operations o "
            "ON x.companion=o.companion AND x.operation=o.operation "
            "WHERE x.companion=? AND x.record=? AND x.version=? "
            "AND json_extract(o.payload,'$.action')='abandon' LIMIT 1",
            (companion, record, version),
        ).fetchone() is not None

    @staticmethod
    def _recall_query(db, companion, *, committed=False):
        if not committed and db.execute(
            "SELECT 1 FROM background_sources WHERE companion=? AND role='user' "
            "AND status<>'completed' LIMIT 1",
            (companion,),
        ).fetchone():
            return None
        return (
            "SELECT v.data FROM memory_heads h JOIN memory_versions v "
            "ON h.companion=v.companion AND h.record=v.record AND h.version=v.version "
            "WHERE h.companion=? AND v.status='active' "
            "AND (v.expires_us IS NULL OR v.expires_us>?) AND NOT EXISTS ("
            "SELECT 1 FROM memory_invalidated_sources x "
            "JOIN memory_management_operations o "
            "ON x.companion=o.companion AND x.operation=o.operation "
            "WHERE x.companion=h.companion AND x.record=h.record "
            "AND x.version=h.version AND json_extract(o.payload,'$.action')='abandon') "
        )
    def ranked_recall(self, companion, *, now_us, score, limit=8, committed=False):
        """Rank the eligible catalog in one snapshot, retaining only top-k rows."""
        companion_id(companion)
        _time(now_us)
        if not callable(score) or type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("Bounded recall limit and relevance scorer required")
        with self.transaction(write=False) as db:
            base = self._recall_query(db, companion, committed=committed)
            if base is None:
                return []
            rows = db.execute(base + "ORDER BY h.rowid", (companion, now_us))

            def scored():
                for index, row in enumerate(rows):
                    value = json.loads(row[0])
                    relevance = score(value)
                    if relevance > 0:
                        yield relevance, -index, value

            return [item[2] for item in nlargest(limit, scored(), key=lambda item: item[:2])]

    def _recall(self, db, companion, *, now_us, candidates, limit, committed=False):
        base = self._recall_query(db, companion, committed=committed)
        if base is None:
            return []
        if candidates is None:
            rows = db.execute(
                base + "ORDER BY h.rowid LIMIT ?", (companion, now_us, limit)
            )
            return [json.loads(row[0]) for row in rows]
        found = []
        for record, version in dict.fromkeys(candidates):
            row = db.execute(
                base + "AND h.record=? AND h.version=?",
                (companion, now_us, record, version),
            ).fetchone()
            if row:
                found.append(json.loads(row[0]))
            if len(found) == limit:
                break
        return found

    def ranked_committed_recall(self, companion, *, now_us, score, limit=8):
        """Read valid committed heads; processing progress is a separate projection."""
        return self.ranked_recall(companion, now_us=now_us, score=score, limit=limit,
                                  committed=True)

    def committed_recall(self, companion, *, now_us, candidates=None, limit=100):
        """Recheck admitted heads with the same identity, TTL and suppression rules.

        The inherited recall port retains its historical global source barrier.
        Desktop admission explicitly uses this committed-evidence port instead.
        """
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
            return self._recall(db, companion, now_us=now_us, candidates=candidates,
                                limit=limit, committed=True)

    def committed_catalog_page(self, companion, *, now_us, after_record_id=None, limit=100):
        """Read a bounded page of current committed heads for external ranking.

        Pages are ordered by the memory head primary key. Callers must keep
        expensive ranking outside this transaction and recheck selected
        ``(record_id, version)`` pairs with ``committed_recall`` before admission.
        """
        companion_id(companion)
        _time(now_us)
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("Committed catalog page limit must be between 1 and 100")
        if after_record_id is not None:
            identifier(after_record_id)
        with self.transaction(write=False) as db:
            base = self._recall_query(db, companion, committed=True)
            if base is None:
                return {"records": (), "next_after_record_id": None}
            select = base.replace("SELECT v.data ",
                                  "SELECT h.record,h.version,v.data ", 1)
            parameters = [companion, now_us]
            if after_record_id is not None:
                select += "AND h.record>? "
                parameters.append(after_record_id)
            rows = db.execute(
                select + "ORDER BY h.record LIMIT ?",
                (*parameters, limit + 1),
            ).fetchall()
            has_more = len(rows) > limit
            rows = rows[:limit]
            records = []
            for row in rows:
                value = json.loads(row[2])
                if value.get("record_id") != row[0] or value.get("version") != row[1]:
                    raise ValueError("Committed catalog row identity differs from its head")
                records.append(value)
            cursor = records[-1]["record_id"] if records and has_more else None
            return {"records": tuple(records), "next_after_record_id": cursor}

    def dialogue_fragments(self, companion, input_id, *, now_us):
        from .dialogue_retrieval import rebuild_fragments

        companion_id(companion)
        identifier(input_id)
        _time(now_us)
        with self.transaction(write=False) as db:
            excluded = self._ineffective_history_sources(db, companion, now_us=now_us)
            return tuple(rebuild_fragments(db, companion, input_id, excluded))

    def validate_dialogue_fragments(self, companion, input_id, fragments, *, now_us):
        from .dialogue_retrieval import fragment_reference, references_current, validate_fragments

        validate_fragments(fragments)
        with self.transaction(write=False) as db:
            refs = [fragment_reference(fragment) for fragment in fragments]
            excluded = self._ineffective_history_sources(db, companion, now_us=now_us)
            if not references_current(db, companion, input_id, refs) or any(
                member["source_id"] in excluded for ref in refs for member in ref["members"]
            ):
                raise VersionConflict('Retrieved dialogue original expired or changed')

    def _filter_fact_targets(self, companion, values):
        references = tuple((value["record_id"], value["version"]) for value in values)
        with self.transaction(write=False) as db:
            suppressed = {
                reference
                for reference in references
                if self._suppressed(db, companion, *reference)
            }
        return [
            value
            for value in values
            if (value["record_id"], value["version"]) not in suppressed
        ]

    def fact_target_catalog(self, companion, *, now_us):
        return tuple(
            self._filter_fact_targets(
                companion, list(super().fact_target_catalog(companion, now_us=now_us))
            )
        )

    def fact_targets(self, companion, *, now_us, candidates=None):
        targets = super().fact_targets(companion, now_us=now_us, candidates=candidates)
        filtered = self._filter_fact_targets(companion, list(targets.values()))
        if candidates is not None and len(filtered) != len(targets):
            raise VersionConflict("Selected fact target is suppressed pending user resolution")
        return {value["record_id"]: value for value in filtered}

    def _stream_context(self, db, scope, position, limit, *, now_us=None, keep_source=None):
        now_us = self.evidence_clock() if now_us is None else now_us
        _time(now_us)
        excluded = self._ineffective_history_sources(db, scope.companion_id, now_us=now_us)
        rows = db.execute(
            f"SELECT * FROM ({_DIALOGUE}) d WHERE companion=? AND session=? "  # noqa: S608
            "AND (ordinal,phase,sequence)<=(?,?,?) "
            "ORDER BY ordinal DESC,phase DESC,sequence DESC",
            (scope.companion_id, scope.session_id, *position),
        ).fetchall()
        from .evidence import EvidenceTurn

        # An excluded user source may still be the exact current request or the
        # source being processed. It is never re-admitted as background history.
        rows = [row for row in rows if row["source"] not in excluded
                or (row["source"] == keep_source and row["role"] == "user")][:limit]
        return tuple(
            EvidenceTurn(scope, row["source"], row["role"], row["text"], len(row["text"]),
                         row["response_type"])
            for row in reversed(rows)
        )

    def effective_exchange_context(self, claim, *, now_us, limit=32):
        return self.exchange_context(claim, limit=limit, now_us=now_us)

    def context_validation_token(self, companion, *, now_us, include_deadline=True):
        """A short read fence, not a reusable global context cache.

        All memory/exchange writes and new user sources advance state.version.
        Explicit no-op management/abandon may not: its existing append-only
        operation inventory and generation therefore also participate. Source
        completion only expands evidence; checking the barrier remains conservative.
        ACK only appends confirmed history and cannot invalidate an existing read.
        """
        companion_id(companion)
        _time(now_us)
        with self.transaction(write=False) as db:
            version = self._snapshot(db, companion).write_version
            generation = self._generation(db, companion)
            operation = db.execute(
                "SELECT COALESCE(MAX(rowid),0) FROM memory_management_operations WHERE companion=?",
                (companion,),
            ).fetchone()[0]
            repeated_source = db.execute(
                "SELECT COALESCE(MAX(rowid),0) FROM memory_commands WHERE companion=?",
                (companion,),
            ).fetchone()[0]
            pending = db.execute(
                "SELECT 1 FROM background_sources WHERE companion=? AND role='user' "
                "AND status<>'completed' LIMIT 1", (companion,),
            ).fetchone() is not None
            deadline = None
            if include_deadline:
                # Conservative companion-wide heads cover history's indirect
                # dependencies, including records absent from selected memories.
                fact = db.execute(
                    "SELECT MIN(v.expires_us) FROM memory_heads h JOIN memory_versions v ON "
                    "h.companion=v.companion AND h.record=v.record AND h.version=v.version "
                    "WHERE h.companion=? AND v.status='active' AND v.expires_us>?",
                    (companion, now_us),
                ).fetchone()[0]
                exchange = db.execute(
                    "SELECT MIN(json_extract(v.data,'$.expires_us')) FROM exchange_versions v "
                    "WHERE v.companion=? AND v.revision=(SELECT MAX(p.revision) FROM "
                    "exchange_versions p WHERE p.companion=v.companion AND p.exchange=v.exchange) "
                    "AND json_extract(v.data,'$.status') IN ('pending','deferred') "
                    "AND json_extract(v.data,'$.expires_us')>?", (companion, now_us),
                ).fetchone()[0]
                deadlines = [value for value in (fact, exchange) if value is not None]
                deadline = min(deadlines) if deadlines else None
            return version, generation, operation, pending, repeated_source, deadline

    @staticmethod
    def _operation(db, companion, operation):
        return db.execute(
            "SELECT * FROM memory_management_operations WHERE companion=? AND operation=?",
            (companion, operation),
        ).fetchone()

    def ineffective_history_sources(self, companion, *, now_us):
        """Project invalid evidence identities without rewriting display history.

        A source may mix several facts. Until spans have explicit provenance,
        exclude the affected whole source and its associated confirmed output;
        unrelated sources remain usable. Background queue progress is not evidence invalidation.
        """
        companion_id(companion)
        _time(now_us)
        with self.transaction(write=False) as db:
            return self._ineffective_history_sources(db, companion, now_us=now_us)

    def _eligible_exchange_targets(self, db, companion, *, now_us):
        # Evidence invalidation applies to history and exchange cues alike;
        # background source processing has its own lifecycle projection.
        excluded = self._ineffective_history_sources(
            db, companion, now_us=now_us,
        )
        return (value for value in super()._eligible_exchange_targets(
            db, companion, now_us=now_us) if value["evidence"]["source_id"] not in excluded)

    def _history_resolved_exchange(self, db, companion, reference, head, source, *, now_us):
        """A confirmed answer may retain its own supported completion as history.

        This exception is never used to authorize a new segment. The old target
        must still match its frozen readset and the immediately following head
        must prove that this exact acknowledged source resolved it.
        """
        if source is None or head is None:
            return False
        try:
            if (head["status"] != "resolved" or head["action"] != "resolve"
                    or head["direction"] != "assistant_response"
                    or head["revision"] != reference["revision"] + 1
                    or head["expected_revision"] != reference["revision"]
                    or head["companion_id"] != companion
                    or head["scope"]["companion_id"] != companion
                    or (head["expires_us"] is not None and head["expires_us"] <= now_us)):
                return False
            previous_row = db.execute(
                "SELECT data FROM exchange_versions WHERE companion=? AND exchange=? "
                "AND revision=?", (companion, reference["exchange_id"], reference["revision"]),
            ).fetchone()
            if previous_row is None:
                return False
            previous = json.loads(previous_row[0])
            if (evidence_hash(previous) != reference["sha256"]
                    or previous["evidence"]["source_id"] != reference["source_id"]
                    or previous["status"] not in {"pending", "deferred"}
                    or any(previous[key] != head[key] for key in (
                        "exchange_id", "direction", "content", "expires_us", "scope"))):
                return False
            evidence = head["evidence"]
            if evidence["role"] != "assistant" or evidence["source_id"] != source:
                return False
            actual = self._exchange_source(db, Scope(**head["scope"]), "assistant", source)
            if (evidence["confirmed_end"] != actual["confirmed"]
                    or actual["confirmed"] != len(actual["text"])
                    or evidence["prefix_sha256"] != sha256(actual["text"].encode()).hexdigest()
                    or not evidence["quote"] or evidence["quote"] not in actual["text"]):
                return False
            command = {key: head[key] for key in (
                "scope", "command_id", "exchange_id", "action", "direction", "content",
                "evidence", "occurred_us", "expected_revision", "expires_us")}
            fingerprint = evidence_hash(command)
            stored = db.execute(
                "SELECT fingerprint FROM exchange_versions WHERE companion=? AND exchange=? "
                "AND revision=?", (companion, reference["exchange_id"], head["revision"]),
            ).fetchone()
            return bool(stored and stored[0] == fingerprint
                        and head["support"]["command_sha256"] == fingerprint)
        except (KeyError, TypeError, ValueError):
            return False

    def _received_references_current(self, db, companion, input_id, references):
        # Older stores have no reliable-reception source to authorize v4 refs.
        return not references

    def _readset_current(self, db, readset, *, now_us, cache=None, history_source=None):
        companion = readset["companion"]
        if 'receiving_sha256' in readset:
            from .receiving_context import receiving_from_binding

            row = db.execute('SELECT config FROM turn_inputs WHERE companion=? AND input=?',
                             (companion, readset['input'])).fetchone()
            observed = receiving_from_binding(row['config']) if row else None
            if observed is None or evidence_hash(observed) != readset['receiving_sha256']:
                return False
        if not self._received_references_current(
                db, companion, readset['input'], readset.get('received_output', [])):
            return False
        from .dialogue_retrieval import references_current

        if not references_current(
            db, companion, readset["input"], readset.get("retrieved_dialogue", [])
        ):
            return False
        cache = {} if cache is None else cache
        for item in readset["history"]:
            if "response_type" not in item:
                continue  # Historical answer references retain their exact contract.
            key = (companion, "history-type", canonical(item))
            if key not in cache:
                row = db.execute(
                    f"SELECT role,response_type FROM ({_DIALOGUE}) WHERE companion=? AND source=?",  # noqa: S608
                    (companion, item["id"]),
                ).fetchone()
                cache[key] = bool(row and row["role"] == "assistant"
                                  and row["response_type"] == item["response_type"])
            if not cache[key]:
                return False
        if "current_operation" in readset:
            row = db.execute("SELECT session FROM turn_inputs WHERE companion=? AND input=?",
                             (companion, readset["input"])).fetchone()
            if row is None or self._current_memory_operation(
                db, companion, row["session"], readset["input"],
            ) != readset["current_operation"]:
                return False
        for fact in readset["facts"]:
            key = (companion, "fact", canonical(fact))
            if key in cache:
                if not cache[key]:
                    return False
                continue
            cache[key] = False
            row = db.execute(
                "SELECT v.* FROM memory_heads h JOIN memory_versions v ON "
                "h.companion=v.companion AND h.record=v.record AND h.version=v.version "
                "WHERE h.companion=? AND h.record=? AND h.version=?",
                (companion, fact["record_id"], fact["version"]),
            ).fetchone()
            if row is None or row["status"] != "active" or (
                row["expires_us"] is not None and row["expires_us"] <= now_us
            ) or self._suppressed(db, companion, fact["record_id"], fact["version"]):
                return False
            data = json.loads(row["data"])
            if data["source_id"] != fact["source_id"] or evidence_hash(data) != fact["sha256"]:
                return False
            cache[key] = True
        for exchange in readset["exchanges"]:
            key = (companion, "exchange", canonical(exchange))
            head_loaded = key not in cache
            if head_loaded:
                data = self._exchange_head(db, companion, exchange["exchange_id"])
                cache[key] = bool(data is not None and data["revision"] == exchange["revision"]
                    and (data["expires_us"] is None or data["expires_us"] > now_us)
                    and data["status"] in {"pending", "deferred"}
                    and data["evidence"]["source_id"] == exchange["source_id"]
                    and evidence_hash(data) == exchange["sha256"])
            if cache[key]:
                continue
            if history_source is None:
                return False
            history_key = (companion, "exchange-history", history_source, canonical(exchange))
            if history_key not in cache:
                if not head_loaded:
                    data = self._exchange_head(db, companion, exchange["exchange_id"])
                cache[history_key] = self._history_resolved_exchange(
                    db, companion, exchange, data, history_source, now_us=now_us,
                )
            if not cache[history_key]:
                return False
        return True

    def _background_source_invalidated(self, db, claim, *, now_us):
        return claim.source in self._ineffective_history_sources(
            db, claim.companion, now_us=now_us,
        )

    def _history_provenance_rows(self, db, companion):
        return db.execute(
            'SELECT s.*,h.source FROM stream_segments s JOIN stream_history h ON '
            's.companion=h.companion AND s.input=h.input AND s.sequence=h.sequence '
            'WHERE s.companion=? ORDER BY h.rowid', (companion,))

    def _ineffective_history_sources(self, db, companion, *, now_us, read_cache=None):
        read_cache = {} if read_cache is None else read_cache
        excluded = {row[0] for row in db.execute(
            "SELECT source FROM memory_invalidated_sources WHERE companion=?", (companion,))}
        heads = db.execute(
            "SELECT h.record,h.version,v.status,v.expires_us FROM memory_heads h "
            "JOIN memory_versions v ON h.companion=v.companion AND h.record=v.record "
            "AND h.version=v.version WHERE h.companion=?", (companion,),
        ).fetchall()
        for head in heads:
            invalid = head["status"] != "active" or (
                head["expires_us"] is not None and head["expires_us"] <= now_us)
            if invalid or head["version"] > 1:
                excluded.update(source for source, version in self._related_sources(
                    db, companion, head["record"])
                    if invalid or version < head["version"])
        risky_legacy = bool(excluded)
        dependencies = {}
        modern = set()
        for row in self._history_provenance_rows(db, companion):
            manifest = json.loads(row["manifest"])
            source = row['source']
            if "readset" not in manifest:
                if risky_legacy:
                    excluded.add(source)
                continue
            modern.add(source)
            try:
                readset = validate_readset(manifest["readset"])
                if (evidence_hash(manifest) != row["fingerprint"]
                        or source != "segment-" + row["fingerprint"]
                        or manifest["basis"]["companion_id"] != companion
                        or manifest["basis"]["input_id"] != row["input"]
                        or manifest["sequence"] != row["sequence"]
                        or readset["companion"] != companion or readset["input"] != row["input"]):
                    raise ValueError("Foreign history readset")
            except (ValueError, KeyError, TypeError):
                # Unknown versions and corrupt provenance never become legacy safe.
                excluded.add(source)
                continue
            # A correction's raw wording can be unsafe as dialogue history while
            # its supported current fact is valid. _readset_current binds each
            # fact to its active head, version, TTL, source and content hash;
            # raw-source exclusion must not invalidate that checked fact again.
            dependencies[source] = {
                readset["input"],
                *(r["id"] for r in readset["history"]),
                *(r["source_id"] for r in readset["exchanges"]),
                *("segment-" + r["fingerprint"] for r in readset.get("received_output", [])),
                *(
                    member["source_id"]
                    for fragment in readset.get("retrieved_dialogue", [])
                    for member in fragment["members"]
                ),
            }
            if not self._readset_current(db, readset, now_us=now_us, cache=read_cache,
                                         history_source=source):
                excluded.add(source)
        if risky_legacy:
            excluded.update(row[0] for row in db.execute(
                f"SELECT source FROM ({_DIALOGUE}) WHERE companion=? AND role='assistant'",  # noqa: S608
                (companion,)) if row[0] not in modern)
        # Each edge is a stored source identity; linear closure, no inferred
        # semantic graph or second state owner. The memo above lasts one read only.
        dependents = {}
        for source, refs in dependencies.items():
            for ref in refs:
                dependents.setdefault(ref, set()).add(source)
        queue = list(excluded)
        while queue:
            for source in dependents.get(queue.pop(), ()):
                if source not in excluded:
                    excluded.add(source)
                    queue.append(source)
        return frozenset(excluded)

    def _validate_segment_readset(self, db, claim, readset, *, now_us):
        from virtual_companion.text.character import CharacterSnapshot, character_from_binding

        from .history_readset import validate_readset
        from .public_capture import validate_public_scope
        from .sqlite_segments import _basis

        validate_readset(readset)
        if readset["protocol"] == "history-readset.v8":
            validate_public_scope(readset, _basis(claim))

        read_cache = {}
        if (readset["companion"] != claim.companion_id or readset["input"] != claim.input_id
                or readset["state_version"] != claim.state_version
                or not self._readset_current(db, readset, now_us=now_us, cache=read_cache)):
            raise VersionConflict("Segment readset changed or belongs to another context")
        excluded = self._ineffective_history_sources(db, claim.companion_id,
                                                      now_us=now_us, read_cache=read_cache)
        if any(member['source_id'] in excluded for fragment in readset.get('retrieved_dialogue', [])
               for member in fragment['members']):
            raise VersionConflict('Segment retrieved dialogue source is ineffective')
        if any(item["id"] in excluded for item in readset["history"]):
            raise VersionConflict("Segment readset history is ineffective")
        if any(item["source_id"] in excluded for item in readset["exchanges"]):
            raise VersionConflict("Segment readset exchange source is ineffective")
        if any('segment-' + item['fingerprint'] in excluded
               for item in readset.get('received_output', [])):
            raise VersionConflict('Received output source is ineffective')
        try:
            source = self._input(db, claim.companion_id, claim.input_id)
        except ValueError as exc:
            raise VersionConflict("Segment readset input is unavailable") from exc
        from .receiving_context import receiving_from_binding

        receiving = receiving_from_binding(source['config'])
        expected_receiving = evidence_hash(receiving) if receiving is not None else None
        if readset.get('receiving_sha256') != expected_receiving:
            raise VersionConflict('Segment receiving observation differs from original input')
        character = character_from_binding(self._run_config(db, source))
        expected = (CharacterSnapshot.from_mapping(character).reference.sha256
                    if character is not None else None)
        if readset.get("character_sha256") != expected:
            raise VersionConflict("Segment role differs from original character snapshot")
        current = db.execute("SELECT rowid FROM turn_inputs WHERE companion=? AND input=?",
                             (claim.companion_id, claim.input_id)).fetchone()
        for item in readset["history"]:
            row = db.execute(
                f"SELECT * FROM ({_DIALOGUE}) WHERE companion=? AND session=? AND source=?",  # noqa: S608
                (claim.companion_id, claim.session_id, item["id"]),
            ).fetchone()
            if row is None or row["ordinal"] >= current[0] or row["role"] != item["role"] or (
                evidence_hash(row["text"]) != item["sha256"]
            ) or (
                row["role"] == "assistant"
                and row["response_type"] != item.get("response_type", "answer")
            ):
                raise VersionConflict("Segment readset original history changed")

    @staticmethod
    def _save_operation(db, command, receipt):
        db.execute(
            "INSERT INTO memory_management_operations VALUES(?,?,?,?,?)",
            (
                command.scope.companion_id,
                command.operation_id,
                command.fingerprint,
                canonical(command.to_mapping()),
                canonical(receipt),
            ),
        )

    @staticmethod
    def _validate_conversation_source(db, command, claim, text):
        if (
            not isinstance(claim, BackgroundClaim)
            or command.entry != "conversation"
            or command.scope.companion_id != claim.companion
            or command.scope.session_id != claim.session
            or command.source_id != claim.source
            or command.source_sha256 != claim.source_hash
            or text[command.quote_start : command.quote_end] != command.evidence_quote
        ):
            raise ValueError("Memory command differs from its complete dialogue source")

    @staticmethod
    def _protected(db, companion, job):
        return db.execute(
            "SELECT s.status,w.error,w.receipt FROM background_sources s "
            "JOIN background_work w "
            "ON s.companion=w.companion AND s.job=w.job "
            "WHERE s.companion=? AND s.job=?",
            (companion, job),
        ).fetchone()

    def _close_protected(self, db, command, receipt):
        if command.resolves_job is None:
            return
        row = self._protected(db, command.scope.companion_id, command.resolves_job)
        if (
            row is None
            or row["status"] != "failed"
            or row["error"] not in self.clarification_errors
        ):
            raise VersionConflict("Clarified memory source is no longer pending")
        db.execute(
            "UPDATE background_sources SET status='completed' WHERE companion=? AND job=?",
            (command.scope.companion_id, command.resolves_job),
        )
        db.execute(
            "UPDATE background_work SET receipt=?,error=NULL,lease_until=NULL "
            "WHERE companion=? AND job=?",
            (
                canonical(
                    {
                        "disposition": "clarified",
                        "operation_id": command.operation_id,
                        "memory_receipt": receipt,
                    }
                ),
                command.scope.companion_id,
                command.resolves_job,
            ),
        )

    def _validate_protected_resolution(self, db, command):
        if command.resolves_job is None:
            return
        row = self._protected(db, command.scope.companion_id, command.resolves_job)
        if (
            row is not None
            and row["status"] == "failed"
            and row["error"] == "needs_clarification.mixed_source"
        ):
            raise SemanticConflict(
                "Mixed memory action requires whole-source resolution"
            )

    @staticmethod
    def _protect_older_user_sources(db, command, claim):
        excluded = {
            item
            for item in (
                claim.job if claim is not None else None,
                command.resolves_job,
            )
            if item is not None
        }
        rows = db.execute(
            "SELECT job FROM background_sources WHERE companion=? AND role='user' "
            "AND status IN ('queued','running','retryable')",
            (command.scope.companion_id,),
        ).fetchall()
        jobs = [row["job"] for row in rows if row["job"] not in excluded]
        for job in jobs:
            receipt = canonical(
                {
                    "disposition": "needs_clarification",
                    "reason": "needs_clarification.memory_generation_changed",
                    "operation_id": command.operation_id,
                }
            )
            db.execute(
                "INSERT INTO background_work "
                "(companion,job,worker,lease_until,attempts,available_us,error,receipt) "
                "VALUES(?,?,NULL,NULL,0,0,?,?) ON CONFLICT(companion,job) DO UPDATE SET "
                "worker=NULL,lease_until=NULL,error=excluded.error,receipt=excluded.receipt",
                (
                    command.scope.companion_id,
                    job,
                    "needs_clarification.memory_generation_changed",
                    receipt,
                ),
            )
            db.execute(
                "UPDATE background_sources SET status='failed',generation=generation+1 "
                "WHERE companion=? AND job=?",
                (command.scope.companion_id, job),
            )
        return len(jobs)

    @staticmethod
    def _related_sources(db, companion, record):
        versions = db.execute(
            "SELECT version,data FROM memory_versions WHERE companion=? AND record=?",
            (companion, record),
        ).fetchall()
        related = []
        for row in versions:
            source = json.loads(row["data"])["source_id"]
            related.append((source, row["version"]))
            for item in db.execute(
                "SELECT delivery AS source FROM turn_deliveries "
                "WHERE companion=? AND input=? UNION SELECT source FROM stream_history "
                "WHERE companion=? AND input=?",
                (companion, source, companion, source),
            ):
                related.append((item["source"], row["version"]))
        for row in db.execute(
            "SELECT payload,receipt FROM memory_commands WHERE companion=? "
            "AND json_extract(receipt,'$.record_id')=?",
            (companion, record),
        ):
            payload, receipt = json.loads(row["payload"]), json.loads(row["receipt"])
            source = payload.get("source_id")
            version = receipt.get("record_version")
            if isinstance(source, str) and source and type(version) is int:
                related.append((source, version))
                for item in db.execute(
                    "SELECT delivery AS source FROM turn_deliveries "
                    "WHERE companion=? AND input=? UNION SELECT source FROM stream_history "
                    "WHERE companion=? AND input=?",
                    (companion, source, companion, source),
                ):
                    related.append((item["source"], version))
        return tuple(dict.fromkeys(related))

    def _finish_command_source(self, db, command, claim, receipt, disposition, *, now_us,
                               clock=None):
        if claim is None:
            return
        delegated_basis = self._refresh_delegated_basis(db, claim, now_us=now_us, clock=clock)
        db.execute(
            "UPDATE background_sources SET status='completed' WHERE companion=? AND job=?",
            (claim.companion, claim.job),
        )
        db.execute(
            "UPDATE background_work SET receipt=?,error=NULL,lease_until=NULL "
            "WHERE companion=? AND job=?",
            (
                canonical(
                    {
                        "disposition": disposition,
                        "operation_id": command.operation_id,
                        "memory_receipt": receipt,
                        **delegated_basis,
                    }
                ),
                claim.companion,
                claim.job,
            ),
        )

    def execute_memory_command(self, command, *, now_us, claim=None, clock=None):
        if not isinstance(command, ExplicitMemoryCommand):
            raise ValueError("Typed explicit memory command required")
        _time(now_us)
        companion = command.scope.companion_id
        with self.transaction() as db:
            now_us = _locked_time(now_us, clock)
            original = self._operation(db, companion, command.operation_id)
            if original is not None:
                if original["fingerprint"] != command.fingerprint:
                    raise ValueError("Conflicting explicit memory operation identity")
                return json.loads(original["receipt"])
            self._ensure_meta(db, companion)
            if claim is not None:
                row = self._live(db, claim, now_us)
                if row["status"] != "running":
                    raise VersionConflict("Completed background work cannot execute new commands")
                # Re-read through the transaction for identity; no model output is trusted.
                source_row = db.execute(
                    "SELECT text FROM turn_inputs WHERE companion=? AND input=?",
                    (claim.companion, claim.source),
                ).fetchone()
                if source_row is None:
                    raise VersionConflict("Memory command source disappeared")
                text = source_row[0]
                self._validate_conversation_source(db, command, claim, text)
            elif command.entry != "list":
                raise ValueError("Conversation command requires its leased source")
            elif command.source_sha256 != list_source_digest(
                action=command.action,
                record_id=command.record_id,
                expected_version=command.expected_record_version,
                replacement=command.replacement,
            ):
                raise ValueError("List command differs from the selected record and action")
            self._validate_protected_resolution(db, command)

            before = self._current(db, companion, command.record_id)
            desired = before is not None and (
                (command.action == "forget" and before["status"] == "revoked")
                or (
                    command.action == "correct"
                    and before["status"] == "active"
                    and before["content"] == command.replacement
                )
            )
            if desired:
                receipt = {
                    "status": "already_applied",
                    "operation_id": command.operation_id,
                    "action": command.action,
                    "record_id": command.record_id,
                    "expected_record_version": command.expected_record_version,
                    "record_version": before["version"],
                    "write_version": self._snapshot(db, companion).write_version,
                    "invalidated_sources": int(command.entry == "conversation"),
                }
                self._save_operation(db, command, receipt)
                if command.entry == "conversation":
                    db.execute(
                        "INSERT OR IGNORE INTO memory_invalidated_sources VALUES(?,?,?,?,?)",
                        (
                            companion,
                            command.source_id,
                            command.record_id,
                            before["version"],
                            command.operation_id,
                        ),
                    )
                    db.execute(
                        "UPDATE memory_management_meta SET generation=generation+1 "
                        "WHERE companion=?",
                        (companion,),
                    )
                    receipt["protected_sources"] = self._protect_older_user_sources(
                        db, command, claim
                    )
                    db.execute(
                        "UPDATE memory_management_operations SET receipt=? "
                        "WHERE companion=? AND operation=?",
                        (canonical(receipt), companion, command.operation_id),
                    )
                else:
                    receipt["protected_sources"] = 0
                    db.execute(
                        "UPDATE memory_management_operations SET receipt=? "
                        "WHERE companion=? AND operation=?",
                        (canonical(receipt), companion, command.operation_id),
                    )
                self._close_protected(db, command, receipt)
                self._finish_command_source(
                    db, command, claim, receipt, "memory_command_no_change",
                    now_us=now_us, clock=clock,
                )
                return receipt
            if (
                before is None
                or before["status"] != "active"
                or before["version"] != command.expected_record_version
            ):
                receipt = {
                    "status": "version_changed",
                    "operation_id": command.operation_id,
                    "action": command.action,
                    "record_id": command.record_id,
                    "expected_record_version": command.expected_record_version,
                    "current_record_version": before["version"] if before else None,
                }
                self._save_operation(db, command, receipt)
                if claim is not None:
                    db.execute(
                        "UPDATE background_sources SET status='failed' "
                        "WHERE companion=? AND job=?",
                        (claim.companion, claim.job),
                    )
                    db.execute(
                        "UPDATE background_work SET receipt=?,error=?,lease_until=NULL "
                        "WHERE companion=? AND job=?",
                        (
                            canonical(
                                {
                                    "disposition": "needs_clarification",
                                    "reason": "needs_clarification.memory_version_changed",
                                    "operation_id": command.operation_id,
                                    "memory_receipt": receipt,
                                }
                            ),
                            "needs_clarification.memory_version_changed",
                            claim.companion,
                            claim.job,
                        ),
                    )
                return receipt

            related = self._related_sources(db, companion, command.record_id)
            mutation = MemoryMutation(
                scope=command.scope,
                mutation_id=command.operation_id,
                kind="revoke" if command.action == "forget" else "correct",
                record_id=command.record_id,
                content=command.replacement,
                source_kind="input" if command.entry == "conversation" else "manual",
                source_id=command.source_id,
                reason=canonical(
                    {
                        "kind": "explicit_user_memory_management",
                        "entry": command.entry,
                        "action": command.action,
                        "quote": command.evidence_quote,
                        "interpreter": command.interpreter,
                        "interpretation_sha256": command.interpretation_sha256,
                    }
                ),
                occurred_at=datetime.fromtimestamp(now_us / 1_000_000, tz=UTC),
                expected_record_version=command.expected_record_version,
            )
            memory_receipt = self._apply_memory(db, mutation)
            if memory_receipt["disposition"] != "applied":
                raise VersionConflict("Explicit memory target changed during commit")
            if command.entry == "conversation":
                # The management request can quote the value being corrected or
                # forgotten. Keep it in user-visible history, but never offer it
                # to later dialogue/background inference as fresh fact evidence.
                related = tuple(
                    dict.fromkeys(
                        (*related, (command.source_id, memory_receipt["record_version"]))
                    )
                )
            receipt = {
                "status": "committed",
                "operation_id": command.operation_id,
                "action": command.action,
                "record_id": command.record_id,
                "previous_record_version": command.expected_record_version,
                "record_version": memory_receipt["record_version"],
                "write_version": memory_receipt["write_version"],
                "invalidated_sources": len(related),
            }
            self._save_operation(db, command, receipt)
            for source, version in related:
                db.execute(
                    "INSERT OR IGNORE INTO memory_invalidated_sources VALUES(?,?,?,?,?)",
                    (companion, source, command.record_id, version, command.operation_id),
                )
            db.execute(
                "UPDATE memory_management_meta SET generation=generation+1 WHERE companion=?",
                (companion,),
            )
            receipt["protected_sources"] = self._protect_older_user_sources(
                db, command, claim
            )
            db.execute(
                "UPDATE memory_management_operations SET receipt=? "
                "WHERE companion=? AND operation=?",
                (canonical(receipt), companion, command.operation_id),
            )
            self._close_protected(db, command, receipt)
            self._finish_command_source(
                db, command, claim, receipt, "memory_command_committed",
                now_us=now_us, clock=clock,
            )
            return receipt

    def protect_memory_source(
        self,
        claim,
        *,
        reason,
        interpretation_sha256,
        action,
        requested_scope,
        unresolved,
        affected_records=(),
        now_us,
    ):
        _time(now_us)
        if (
            reason not in self.clarification_errors
            or not isinstance(interpretation_sha256, str)
            or len(interpretation_sha256) != 64
            or any(character not in "0123456789abcdef" for character in interpretation_sha256)
            or action not in {"forget", "correct", "unsupported", "unclear"}
            or requested_scope
            not in {
                "none",
                "single_record",
                "topic",
                "category",
                "all_data",
                "physical_delete",
                "no_mention",
            }
            or not isinstance(unresolved, tuple)
            or not unresolved
            or len(set(unresolved)) != len(unresolved)
            or any(
                item
                not in {
                    "intent",
                    "target",
                    "scope",
                    "replacement",
                    "mixed_content",
                    "support",
                }
                for item in unresolved
            )
            or not isinstance(affected_records, tuple)
            or len(affected_records) > 8
            or any(
                not isinstance(item, tuple)
                or len(item) != 2
                or not isinstance(item[0], str)
                or not item[0]
                or type(item[1]) is not int
                or item[1] < 1
                for item in affected_records
            )
            or len(set(affected_records)) != len(affected_records)
        ):
            raise ValueError("Bound memory clarification result required")
        with self.transaction() as db:
            row = self._live(db, claim, now_us)
            if row["status"] == "completed":
                raise VersionConflict("Completed memory source cannot require clarification")
            for record, version in affected_records:
                current = self._current(db, claim.companion, record)
                if (
                    current is None
                    or current["status"] != "active"
                    or current["version"] != version
                ):
                    raise VersionConflict("Clarification target changed before protection")
            receipt = {
                "status": "needs_clarification",
                "job": claim.job,
                "generation": claim.generation,
                "source_sha256": claim.source_hash,
                "reason": reason,
                "interpretation_sha256": interpretation_sha256,
                "action": action,
                "requested_scope": requested_scope,
                "unresolved": list(unresolved),
                "affected_records": [list(item) for item in affected_records],
            }
            db.execute(
                "UPDATE background_sources SET status='failed' WHERE companion=? AND job=?",
                (claim.companion, claim.job),
            )
            db.execute(
                "UPDATE background_work SET receipt=?,error=?,available_us=?,lease_until=NULL "
                "WHERE companion=? AND job=?",
                (canonical(receipt), reason, now_us, claim.companion, claim.job),
            )
            return receipt

    def memory_source_receipt(self, companion, job):
        companion_id(companion)
        identifier(job)
        with self.transaction(write=False) as db:
            row = db.execute(
                "SELECT receipt FROM background_work WHERE companion=? AND job=?",
                (companion, job),
            ).fetchone()
            return json.loads(row["receipt"]) if row and row["receipt"] else None

    def _source_processing(self, db, companion, input_id, *, session=None):
        original = db.execute(
            "SELECT session,text FROM turn_inputs WHERE companion=? AND input=?",
            (companion, input_id),
        ).fetchone()
        if original is None or (session is not None and original["session"] != session):
            raise VersionConflict("Current source processing scope changed")
        row = db.execute(
            "SELECT s.*,w.error,w.receipt FROM background_sources s "
            "LEFT JOIN background_work w "
            "ON s.companion=w.companion AND s.job=w.job WHERE s.companion=? "
            "AND s.session=? AND s.source=? AND s.role='user'",
            (companion, original["session"], input_id),
        ).fetchone()
        if row is None:
            raise VersionConflict("Current source processing scope changed")
        source_hash = sha256(original["text"].encode()).hexdigest()
        if row["source_hash"] != source_hash or row["confirmed_end"] != len(original["text"]):
            raise VersionConflict("Current source processing original changed")
        return validate_source_processing({
            "protocol": "current-source-processing.v1", "source_id": input_id,
            "source_sha256": source_hash, "status": row["status"], "error": row["error"],
            "receipt_sha256": (evidence_hash(json.loads(row["receipt"]))
                               if row["receipt"] is not None else None),
        }, input_id)

    def source_processing_for_input(self, companion, input_id):
        """Read the original persisted source for immutable context validation.

        This is evidence, not an owner/attempt/lease capability. Its session comes
        from the stored input; foreground control checks the actual typed claim.
        """
        companion_id(companion)
        identifier(input_id)
        with self.transaction(write=False) as db:
            return self._source_processing(db, companion, input_id)

    def current_source_processing(self, claim):
        """Exact current input lifecycle, independent of the companion-wide queue."""
        if not isinstance(claim, RunClaim):
            raise ValueError("Typed current input claim required")
        with self.transaction(write=False) as db:
            return self._source_processing(
                db, claim.companion_id, claim.input_id, session=claim.session_id,
            )

    def current_memory_operation(self, claim):
        """Read this input's durable command result, not the companion's latest one.

        A retry may have a new owner; the original effect is already committed.
        The caller still fences the active owner/basis and revalidates the readset.
        """
        if not isinstance(claim, RunClaim):
            raise ValueError("Typed current input claim required")
        with self.transaction(write=False) as db:
            return self._current_memory_operation(
                db, claim.companion_id, claim.session_id, claim.input_id,
            )

    @staticmethod
    def _current_memory_operation(db, companion, session, input_id):
        companion_id(companion)
        identifier(session)
        identifier(input_id)
        original = db.execute("SELECT session,text FROM turn_inputs WHERE companion=? AND input=?",
                              (companion, input_id)).fetchone()
        if original is None or original["session"] != session:
            raise VersionConflict("Current memory operation input scope changed")
        rows = db.execute(
            "SELECT s.*,w.receipt FROM background_sources s LEFT JOIN background_work w "
            "ON s.companion=w.companion AND s.job=w.job "
            "WHERE s.companion=? AND s.source=? AND s.role='user'",
            (companion, input_id),
        ).fetchall()
        if not rows:
            return None
        if len(rows) != 1:
            raise ValueError("Ambiguous current memory operation source")
        source = rows[0]
        source_hash = sha256(original["text"].encode()).hexdigest()
        if (source["session"] != session or source["source_hash"] != source_hash
                or source["confirmed_end"] != len(original["text"])):
            raise ValueError("Current memory operation source binding changed")
        if source["status"] != "completed" or source["receipt"] is None:
            return None
        result = json.loads(source["receipt"])
        if not isinstance(result, dict):
            raise ValueError("Current memory source receipt must be an object")
        expected_status = {"memory_command_committed": "committed",
                           "memory_command_no_change": "already_applied"}.get(
                               result.get("disposition"))
        if expected_status is None:
            return None
        required = {"disposition", "operation_id", "memory_receipt"}
        if set(result) not in (required, required | {"foreground_basis", "foreground_binding"}):
            raise ValueError("Exact successful memory source receipt required")
        if "foreground_basis" in result and (
            type(result["foreground_basis"]) is not int or result["foreground_basis"] < 1
            or not isinstance(result["foreground_binding"], str)
            or not result["foreground_binding"].startswith("source-owner-")
        ):
            raise ValueError("Invalid original delegated command binding")
        identifier(result["operation_id"])
        operation = db.execute(
            "SELECT fingerprint,payload,receipt FROM memory_management_operations "
            "WHERE companion=? AND operation=?", (companion, result["operation_id"]),
        ).fetchone()
        if operation is None:
            raise ValueError("Current memory operation is not persisted")
        payload, receipt = json.loads(operation["payload"]), json.loads(operation["receipt"])
        try:
            command = ExplicitMemoryCommand(**{**payload, "scope": Scope(**payload["scope"])})
        except (TypeError, KeyError) as error:
            raise ValueError("Invalid persisted current memory command") from error
        if (command.fingerprint != operation["fingerprint"]
                or command.to_mapping() != payload
                or command.scope != Scope(companion, session)
                or command.entry != "conversation" or command.source_id != input_id
                or command.source_sha256 != source_hash
                or command.operation_id != result["operation_id"]
                or original["text"][command.quote_start:command.quote_end]
                != command.evidence_quote):
            raise ValueError("Current memory command belongs to another source or changed")
        version_field = ("previous_record_version" if expected_status == "committed"
                         else "expected_record_version")
        receipt_fields = {"status", "operation_id", "action", "record_id", version_field,
                          "record_version", "write_version", "invalidated_sources",
                          "protected_sources"}
        if (not isinstance(receipt, dict) or set(receipt) != receipt_fields
                or receipt != result["memory_receipt"]
                or receipt["status"] != expected_status
                or receipt["operation_id"] != command.operation_id
                or receipt["action"] != command.action or receipt["record_id"] != command.record_id
                or receipt[version_field] != command.expected_record_version
                or any(type(receipt[k]) is not int or receipt[k] < 1
                       for k in (version_field, "record_version"))
                or any(type(receipt[k]) is not int or receipt[k] < 0
                       for k in ("write_version", "invalidated_sources", "protected_sources"))
                or (expected_status == "committed"
                    and receipt["record_version"] != command.expected_record_version + 1)):
            raise ValueError("Current memory operation receipt binding changed")
        # Deliberately do not read the target's current contents/status. This is
        # an operation fact, and a successful forget leaves a revoked target.
        return validate_current_operation({
            "protocol": "memory-command-outcome.v1", "source_id": input_id,
            "source_sha256": source_hash, "operation_id": command.operation_id,
            "action": command.action, "status": expected_status,
            "receipt_sha256": evidence_hash(receipt),
        })

    def memory_records(self, companion, *, offset=0, limit=50, now_us=0):
        companion_id(companion)
        if type(offset) is not int or offset < 0 or offset > 1_000_000:
            raise ValueError("Bounded memory list offset required")
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("Bounded memory list page required")
        _time(now_us)
        with self.transaction(write=False) as db:
            base = (
                " FROM memory_heads h JOIN memory_versions v "
                "ON h.companion=v.companion AND h.record=v.record AND h.version=v.version "
                "WHERE h.companion=? AND v.status='active'"
            )
            total = db.execute("SELECT count(*)" + base, (companion,)).fetchone()[0]
            rows = db.execute(
                "SELECT v.data" + base
                + " ORDER BY json_extract(v.data,'$.occurred_at') DESC,h.rowid DESC "
                "LIMIT ? OFFSET ?",
                (companion, limit, offset),
            ).fetchall()
            values = [json.loads(row[0]) for row in rows]
            for value in values:
                if self._suppressed(
                    db, companion, value["record_id"], value["version"]
                ):
                    value["status"] = "suppressed"
                elif value.get("expires_at") and _time_from_iso(value["expires_at"]) <= now_us:
                    value["status"] = "expired"
            return {"total": total, "rows": values}

    def memory_statistics(self, companion, *, now_us):
        """Return explicit current/history counts for one companion.

        Current records are counted once at their head version. ``effective``
        excludes revoked, expired and suppressed records; history versions are
        reported separately and are never added to the effective count.
        """
        companion_id(companion)
        _time(now_us)
        with self.transaction(write=False) as db:
            rows = db.execute(
                "SELECT h.record,v.status,v.expires_us,v.data FROM memory_heads h "
                "JOIN memory_versions v ON h.companion=v.companion "
                "AND h.record=v.record AND h.version=v.version "
                "WHERE h.companion=? ORDER BY h.record",
                (companion,),
            ).fetchall()
            counts = {"active": 0, "suppressed": 0, "expired": 0, "revoked": 0}
            effective = 0
            unknown_event_time = 0
            for row in rows:
                value = json.loads(row["data"])
                if row["status"] == "revoked":
                    state = "revoked"
                elif self._suppressed(db, companion, row["record"], value["version"]):
                    state = "suppressed"
                elif row["expires_us"] is not None and row["expires_us"] <= now_us:
                    state = "expired"
                else:
                    state = "active"
                counts[state] += 1
                if state == "active":
                    effective += 1
                    if not value.get("event_time"):
                        unknown_event_time += 1
            history = db.execute(
                "SELECT count(*) FROM memory_versions WHERE companion=?", (companion,)
            ).fetchone()[0]
            return {
                "current_record_count": len(rows),
                "effective_count": effective,
                "history_version_count": history,
                "status_counts": counts,
                "unknown_event_time_count": unknown_event_time,
                "count_basis": (
                    "current_companion_records;"
                    "effective_excludes_expired_suppressed_revoked"
                ),
            }

    def memory_record(self, companion, record_id, *, now_us=0):
        companion_id(companion)
        identifier(record_id)
        _time(now_us)
        with self.transaction(write=False) as db:
            current = self._current(db, companion, record_id)
            if current is None:
                return None
            if self._suppressed(db, companion, record_id, current["version"]):
                current = {**current, "status": "suppressed"}
            elif current["status"] == "active" and current.get("expires_at"):
                if _time_from_iso(current["expires_at"]) <= now_us:
                    current = {**current, "status": "expired"}
            history = [
                json.loads(row[0])
                for row in db.execute(
                    "SELECT data FROM memory_versions WHERE companion=? AND record=? "
                    "ORDER BY version",
                    (companion, record_id),
                )
            ]
            source_id = current["source_id"]
            source = db.execute(
                "SELECT text,'user' AS role FROM turn_inputs "
                "WHERE companion=? AND input=? UNION ALL "
                "SELECT substr(text,1,displayed_end),'assistant' FROM turn_deliveries "
                "WHERE companion=? AND delivery=? UNION ALL "
                "SELECT text,'assistant' FROM stream_history "
                "WHERE companion=? AND source=? LIMIT 1",
                (companion, source_id, companion, source_id, companion, source_id),
            ).fetchone()
            return {
                "current": current,
                "history": history,
                "source": {
                    "kind": current["source_kind"],
                    "role": source["role"] if source else None,
                    "text": source["text"] if source else None,
                },
            }

    def conflicting_fact_targets(self, commands):
        if (
            not isinstance(commands, tuple)
            or len(commands) > 8
            or any(not isinstance(command, MemoryMutation) for command in commands)
        ):
            raise ValueError("Bounded fact command batch required")
        targets = []
        with self.transaction(write=False) as db:
            for command in commands:
                current = None
                if command.kind in {"correct", "revoke"}:
                    current = self._current(
                        db, command.scope.companion_id, command.record_id
                    )
                elif command.kind == "assert_fact" and command.fact_key is not None:
                    slot = canonical(
                        {
                            "predicate": command.fact_key.predicate,
                            "scope": command.fact_key.scope,
                            "subject": command.fact_key.subject,
                        }
                    )
                    row = db.execute(
                        "SELECT v.data FROM memory_heads h JOIN memory_versions v "
                        "ON h.companion=v.companion AND h.record=v.record "
                        "AND h.version=v.version WHERE h.companion=? AND h.slot=?",
                        (command.scope.companion_id, slot),
                    ).fetchone()
                    current = json.loads(row[0]) if row else None
                if current is not None and current["status"] == "active":
                    targets.append((current["record_id"], current["version"]))
        return tuple(dict.fromkeys(targets))

    def abandon_memory_source(self, companion, job, operation_id, *, now_us):
        companion_id(companion)
        identifier(job)
        identifier(operation_id)
        _time(now_us)
        payload = {"action": "abandon", "job": job, "operation_id": operation_id}
        fingerprint = sha256(canonical(payload).encode()).hexdigest()
        with self.transaction() as db:
            original = self._operation(db, companion, operation_id)
            if original is not None:
                if original["fingerprint"] != fingerprint:
                    raise ValueError("Conflicting source resolution identity")
                return json.loads(original["receipt"])
            row = self._protected(db, companion, job)
            if (
                row is None
                or row["status"] != "failed"
                or row["error"] not in self.clarification_errors
            ):
                raise VersionConflict("Memory source is not awaiting clarification")
            protected = json.loads(row["receipt"]) if row["receipt"] else {}
            if row["error"] == "needs_clarification.mixed_source":
                receipt = {
                    "status": "continued",
                    "disposition": "memory_command_abandoned",
                    "operation_id": operation_id,
                    "action": "abandon",
                    "job": job,
                    "suppressed_records": [],
                    "stale_records": [],
                }
                db.execute(
                    "INSERT INTO memory_management_operations VALUES(?,?,?,?,?)",
                    (
                        companion,
                        operation_id,
                        fingerprint,
                        canonical(payload),
                        canonical(receipt),
                    ),
                )
                db.execute(
                    "UPDATE background_sources SET status='queued',generation=generation+1 "
                    "WHERE companion=? AND job=?",
                    (companion, job),
                )
                db.execute(
                    "UPDATE background_work SET worker=NULL,lease_until=NULL,attempts=0,"
                    "available_us=?,error=NULL,receipt=? WHERE companion=? AND job=?",
                    (now_us, canonical(receipt), companion, job),
                )
                return receipt
            affected = tuple(
                (item[0], item[1])
                for item in protected.get("affected_records", ())
                if isinstance(item, list) and len(item) == 2
            )
            suppressed, stale, related = [], [], []
            for record, version in affected:
                current = self._current(db, companion, record)
                if (
                    current is None
                    or current["status"] != "active"
                    or current["version"] != version
                ):
                    stale.append((record, version))
                    continue
                suppressed.append((record, version))
                related.extend(
                    (source, record, source_version)
                    for source, source_version in self._related_sources(
                        db, companion, record
                    )
                )
            receipt = {
                "status": "committed",
                "operation_id": operation_id,
                "action": "abandon",
                "job": job,
                "suppressed_records": [list(item) for item in suppressed],
                "stale_records": [list(item) for item in stale],
            }
            db.execute(
                "INSERT INTO memory_management_operations VALUES(?,?,?,?,?)",
                (companion, operation_id, fingerprint, canonical(payload), canonical(receipt)),
            )
            for source, record, version in dict.fromkeys(related):
                db.execute(
                    "INSERT OR IGNORE INTO memory_invalidated_sources VALUES(?,?,?,?,?)",
                    (companion, source, record, version, operation_id),
                )
            db.execute(
                "UPDATE background_sources SET status='completed' WHERE companion=? AND job=?",
                (companion, job),
            )
            db.execute(
                "UPDATE background_work SET receipt=?,error=NULL,lease_until=NULL "
                "WHERE companion=? AND job=?",
                (canonical(receipt), companion, job),
            )
            return receipt

    def invalidated_sources(self, companion, *, record_id=None):
        companion_id(companion)
        if record_id is not None:
            identifier(record_id)
        with self.transaction(write=False) as db:
            query = (
                "SELECT source,record,version,operation FROM memory_invalidated_sources "
                "WHERE companion=?"
            )
            args = [companion]
            if record_id is not None:
                query += " AND record=?"
                args.append(record_id)
            query += " ORDER BY rowid"
            return [dict(row) for row in db.execute(query, args)]
