"""Opt-in v18: immutable confirmed units are actual dialogue sources.

One confirmation transaction owns the receipt, history, and background enqueue.
Stream settlement releases the run without creating a replayable whole delivery.
Device receipts, semantic unit admission, and desktop integration remain external.
"""

import json
from hashlib import sha256

from virtual_companion.text.identity import CompanionConfig

from .contracts import Scope, VersionConflict, canonical, companion_id, identifier
from .evidence import EvidenceTurn
from .sqlite_segments import SQLiteSegmentStore, _basis
from .sqlite_turns import _locked_time, _time
from .turns import RunClaim

_EXTRA = (
    "CREATE TABLE stream_history (companion TEXT NOT NULL, input TEXT NOT NULL, "
    "sequence INTEGER NOT NULL, source TEXT NOT NULL, text TEXT NOT NULL, "
    "PRIMARY KEY(companion,input,sequence), UNIQUE(companion,source), "
    "FOREIGN KEY(companion,input,sequence) "
    "REFERENCES stream_segments(companion,input,sequence))",
    "CREATE TABLE stream_terminals (companion TEXT NOT NULL, input TEXT NOT NULL, "
    "basis TEXT NOT NULL, outcome TEXT NOT NULL, receipt TEXT NOT NULL, "
    "PRIMARY KEY(companion,input), "
    "FOREIGN KEY(companion,input) REFERENCES turn_inputs(companion,input))",
)

_DIALOGUE = (
    "SELECT i.companion,i.session,i.rowid AS ordinal,0 AS phase,0 AS sequence,"
    "i.input AS source,'user' AS role,i.text,NULL AS response_type "
    "FROM turn_inputs i WHERE NOT EXISTS "
    "(SELECT 1 FROM turn_triggers t WHERE t.companion=i.companion AND t.input=i.input) "
    "UNION ALL SELECT i.companion,i.session,i.rowid,1,0,d.delivery,'assistant',"
    "substr(d.text,1,d.displayed_end),'answer' FROM turn_inputs i JOIN turn_deliveries d "
    "ON i.companion=d.companion AND i.input=d.input WHERE d.displayed_end>0 "
    "AND d.status IN ('completed','cancelled') "
    "UNION ALL SELECT i.companion,i.session,i.rowid,1,h.sequence,h.source,'assistant',"
    "h.text,coalesce(json_extract(s.manifest,'$.presentation.candidate.response_type'),'answer') "
    "FROM stream_history h JOIN turn_inputs i "
    "ON h.companion=i.companion AND h.input=i.input JOIN stream_segments s "
    "ON s.companion=h.companion AND s.input=h.input AND s.sequence=h.sequence"
)


class SQLiteStreamingStore(SQLiteSegmentStore):
    stream_outcomes = frozenset({'completed', 'failed', 'cancelled', 'listened'})
    schema_version = 18
    schema = SQLiteSegmentStore.schema + _EXTRA
    tables = SQLiteSegmentStore.tables | {"stream_history", "stream_terminals"}

    def _upgrade(self, db, version, application, names):
        raise ValueError("Streaming history requires a fresh v18 database; no migration")

    def _before_claim(self, db, companion, input_id):
        super()._before_claim(db, companion, input_id)
        if db.execute("SELECT 1 FROM stream_segments WHERE companion=? AND input=? LIMIT 1",
                      (companion, input_id)).fetchone():
            raise VersionConflict("Stream requires reconciliation before attempt replacement")

    def _segment_confirmed(self, db, event):
        basis = event["basis"]
        source = "segment-" + event["fingerprint"]
        db.execute("INSERT INTO stream_history VALUES(?,?,?,?,?)",
                   (basis["companion_id"], basis["input_id"], event["sequence"],
                    source, event["text"]))
        self._enqueue_source(db, Scope(basis["companion_id"], basis["session_id"]),
                             "assistant", source)

    @staticmethod
    def _exchange_source(db, scope, role, source_id):
        if role == "assistant":
            row = db.execute(
                "SELECT h.text,i.session,length(h.text) AS confirmed,i.rowid AS ordinal,"
                "h.sequence,s.companion,s.input,s.fingerprint,s.manifest,s.status,s.host_receipt "
                "FROM stream_history h JOIN turn_inputs i "
                "ON h.companion=i.companion AND h.input=i.input "
                "JOIN stream_segments s ON s.companion=h.companion AND s.input=h.input "
                "AND s.sequence=h.sequence "
                "WHERE h.companion=? AND h.source=?",
                (scope.companion_id, source_id)).fetchone()
            if row is not None:
                if row["session"] != scope.session_id:
                    raise ValueError("Foreign stream source session")
                event = SQLiteSegmentStore._segment_event(row)
                if (event["status"] != "confirmed" or row["text"] != event["text"]
                        or source_id != "segment-" + event["fingerprint"]):
                    raise ValueError("Assistant source differs from its immutable ACK")
                return {**dict(row), "response_type": event["presentation"]["candidate"].get(
                    "response_type", "answer")}
        return SQLiteSegmentStore._exchange_source(db, scope, role, source_id)

    @staticmethod
    def _stream_context(db, scope, position, limit, *, now_us=None, keep_source=None):
        rows = db.execute(
            f"SELECT * FROM ({_DIALOGUE}) WHERE companion=? AND session=? "  # noqa: S608 -- static SQL only
            "AND (ordinal,phase,sequence)<=(?,?,?) "
            "ORDER BY ordinal DESC,phase DESC,sequence DESC LIMIT ?",
            (scope.companion_id, scope.session_id, *position, limit)).fetchall()
        return tuple(EvidenceTurn(scope, row["source"], row["role"], row["text"],
                                  len(row["text"]), row["response_type"]) for row in reversed(rows))

    def exchange_context(self, claim, *, limit=32, now_us=None):
        if not isinstance(claim, RunClaim) or type(limit) is not int or not 1 <= limit <= 32:
            raise ValueError("Typed current input and bounded context required")
        with self.transaction(write=False) as db:
            return self._exchange_context(db, claim, limit=limit, now_us=now_us)

    def _exchange_context(self, db, claim, *, limit=32, now_us=None):
        anchor = db.execute("SELECT rowid,* FROM turn_inputs WHERE companion=? AND input=?",
                            (claim.companion_id, claim.input_id)).fetchone()
        if (anchor is None or anchor["session"] != claim.session_id
                or anchor["status"] != "running"
                or self._run_config(db, anchor) != claim.config_json
                or db.execute("SELECT 1 FROM turn_triggers WHERE companion=? AND input=?",
                              (claim.companion_id, claim.input_id)).fetchone()):
            raise ValueError("Current user input required")
        sources = self._stream_context(db, Scope(claim.companion_id, claim.session_id),
                                       (anchor["rowid"], 0, 0), limit,
                                       now_us=now_us, keep_source=claim.input_id)
        return sources, tuple(s.source_id for s in sources if not self._source_seen(db, s))

    def _background_context(self, db, claim, *, now_us=None):
        scope = Scope(claim.companion, claim.session)
        row = self._exchange_source(db, scope, claim.role, claim.source)
        seq = row["sequence"] if "sequence" in row.keys() else 0
        sources = self._stream_context(db, scope,
            (row["ordinal"], int(claim.role == "assistant"), seq), 32,
            now_us=now_us, keep_source=claim.source)
        if (not sources or sources[-1].role != claim.role
                or sources[-1].source_id != claim.source
                or sha256(sources[-1].text.encode()).hexdigest() != claim.source_hash):
            raise VersionConflict("Stream background source context changed")
        return sources

    def finish_stream(self, claim, *, outcome, now_us, clock=None):
        """Settle output, not mere model EOF. Completion needs all unit ACKs.

        Failure/cancellation can settle with unknown receipts. Matching late ACKs
        remain valid and enqueue only their own immutable unit, never the suffix.
        """
        basis = canonical(_basis(claim))
        if outcome not in self.stream_outcomes:
            raise ValueError("Explicit stream terminal required")
        key = (claim.companion_id, claim.input_id)
        with self.transaction() as db:
            old = db.execute("SELECT * FROM stream_terminals WHERE companion=? AND input=?",
                             key).fetchone()
            if old:
                if old["basis"] != basis or old["outcome"] != outcome:
                    raise ValueError("Conflicting stream terminal")
                return json.loads(old["receipt"])
            run = db.execute("SELECT * FROM turn_runs WHERE companion=? AND input=?",
                             key).fetchone()
            owner = db.execute("SELECT owner,epoch FROM state_owners WHERE companion=?",
                               key[:1]).fetchone()
            if (run is None or owner is None or run["attempt"] != claim.attempt_id
                    or run["basis"] != claim.state_version
                    or (run["owner"], run["epoch"]) != (claim.owner.owner_id, claim.owner.epoch)
                    or tuple(owner) != (claim.owner.owner_id, claim.owner.epoch)):
                raise VersionConflict("Stream terminal cannot replace another run")
            rows = self._segment_rows(db, *key)
            for row in rows:
                if self._segment_event(row)["basis"] != _basis(claim):
                    raise VersionConflict("Stream segment belongs to another run")
            if outcome == "completed":
                self._segment_live(db, claim, _locked_time(now_us, clock))
                if not rows or any(row["status"] != "confirmed" for row in rows):
                    raise ValueError("Completion requires actual confirmations for every unit")
            if outcome == "listened":
                self._segment_live(db, claim, _locked_time(now_us, clock))
                if rows:
                    raise ValueError("Listening cannot settle an approved spoken prefix")
            if outcome == 'received':
                self._validate_received_settlement(db, claim, rows, _locked_time(now_us, clock))
            receipt = {"input_id": claim.input_id, "attempt_id": claim.attempt_id,
                       "outcome": outcome, "approved_units": len(rows),
                       "confirmed_at_settlement": sum(r["status"] == "confirmed" for r in rows)}
            if outcome == "listened":
                receipt["response_type"] = "listen"
            db.execute("INSERT INTO stream_terminals VALUES(?,?,?,?,?)",
                       (*key, basis, outcome, canonical(receipt)))
            # Both states are terminal for the old queue. No whole delivery is
            # created: generic whole-turn retry/publication must not be used.
            db.execute("UPDATE turn_runs SET status='cancelled' WHERE companion=? AND input=?",
                       key)
            db.execute("UPDATE turn_inputs SET status='cancelled' WHERE companion=? AND input=?",
                       key)
            return receipt

    def _validate_received_settlement(self, db, claim, rows, now_us):
        raise ValueError('This store has no reliable reception contract')

    def _delivery_counts(self, db, companion, input_id, events):
        return {}

    def stream_state(self, companion, input_id):
        """One read snapshot; never interpret the legacy queue flag as outcome."""
        with self.transaction(write=False) as db:
            item = self._input(db, companion, input_id)
            terminal = db.execute(
                "SELECT receipt FROM stream_terminals WHERE companion=? AND input=?",
                (companion, input_id)).fetchone()
            events = [self._segment_event(row)
                      for row in self._segment_rows(db, companion, input_id)]
            receipt = json.loads(terminal[0]) if terminal else None
            confirmed = [event for event in events if event["status"] == "confirmed"]
            # Confirmations are a durable prefix, so the last confirmed unit owns
            # the whole actually displayed text for both representations.
            prefix = []
            for event in events:
                if event['status'] != 'confirmed':
                    break
                prefix.append(event)
            latest = prefix[-1] if prefix else None
            return {"input_id": input_id,
                    "outcome": receipt["outcome"] if receipt else "unsettled",
                    "queue_closed": item["status"] not in {"queued", "running"},
                    "settlement": receipt,
                    "approved_units": len(events),
                    "confirmed_units": len(confirmed),
                    "pending_units": sum(e["status"] == "pending" for e in events),
                    "unknown_units": sum(e["status"] == "unknown" for e in events),
                    **self._delivery_counts(db, companion, input_id, events),
                    "confirmed_text": "" if latest is None else latest["display_text"],
                    "display_join": "\n\n" if latest is None else (
                        "" if latest["representation"] == "spans-v1" else "\n\n"),
                    "representation": None if latest is None else latest["representation"]}

    def retry_stream(
        self, companion, input_id, failed_attempt, retry_id, config, *, now_us, clock=None,
    ):
        """Explicit retry only before any approved unit or unresolved external effect.

        The original input/source is retained. Its old attempt and any zero-unit
        settlement are archived in existing lifecycle journals before the generic
        retry commit queues a new attempt. Dispatch/ACK history is never removed.
        """
        companion_id(companion)
        for value in (input_id, failed_attempt, retry_id):
            identifier(value)
        _time(now_us)
        if not isinstance(config, CompanionConfig) or config.companion_id != companion:
            raise ValueError("Retry configuration must belong to the same companion")
        binding = canonical(config.attempt_binding({}))
        protocol = "zero-unit-stream-retry.v1"
        fingerprint = sha256(canonical(
            [protocol, input_id, failed_attempt, binding]
        ).encode()).hexdigest()
        key = (companion, input_id)
        with self.transaction() as db:
            now_us = _locked_time(now_us, clock)
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
            if run is None or run["attempt"] != failed_attempt:
                raise VersionConflict("Retry must identify the current stream attempt")
            if self._segment_rows(db, *key):
                raise VersionConflict("Approved stream requires reconciliation, not replay")
            if db.execute(
                "SELECT 1 FROM turn_commits WHERE companion=? AND input=?", key
            ).fetchone() or db.execute(
                "SELECT 1 FROM turn_deliveries WHERE companion=? AND input=?", key
            ).fetchone():
                raise VersionConflict("Committed delivery requires reconciliation, not replay")
            old = db.execute(
                "SELECT * FROM stream_terminals WHERE companion=? AND input=?", key
            ).fetchone()
            previous = None if old is None else json.loads(old["receipt"])
            if old is not None:
                basis = json.loads(old["basis"])
                if (old["outcome"] not in {"failed", "cancelled"}
                        or basis["attempt_id"] != failed_attempt
                        or previous["approved_units"] != 0
                        or previous["confirmed_at_settlement"] != 0):
                    raise VersionConflict("Stream settlement cannot be retried")
            elif not (
                item["status"] == run["status"] and (
                    run["status"] in {"failed", "cancelled"}
                    or (run["status"] == "running" and run["lease_until"] <= now_us)
                )
            ):
                raise VersionConflict("Live stream must be cancelled or expired before retry")
            self._check_retry_effects(db, companion, input_id)
            record = {
                **dict(run), "status": "failed", "previous_status": run["status"],
                "reason": protocol, "config": self._run_config(db, item),
                "failed_at_us": now_us, "previous_stream_terminal": previous,
            }
            db.execute(
                "INSERT INTO turn_attempt_history VALUES(?,?,?,?,?) "
                "ON CONFLICT(companion,attempt) DO NOTHING",
                (*key, failed_attempt, sha256(canonical(record).encode()).hexdigest(),
                 canonical(record)),
            )
            db.execute("DELETE FROM stream_terminals WHERE companion=? AND input=?", key)
            for table in ("turn_inputs", "turn_runs"):
                db.execute(
                    f"UPDATE {table} SET status='failed' WHERE companion=? AND input=?",  # noqa: S608
                    key,
                )
            return self._retry_in_transaction(
                db, companion, input_id, failed_attempt, retry_id, binding, fingerprint,
                details={"retry_protocol": protocol, "previous_stream_terminal": previous},
            )
