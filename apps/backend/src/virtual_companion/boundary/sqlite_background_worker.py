"""Opt-in v15 leased source work; model inference stays outside SQLite locks.

Trusted semantic processors certify empty extraction. Supported facts and
bilateral dialogue effects commit atomically with source completion.
"""

import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from hashlib import sha256
from itertools import islice

from virtual_companion.text.chronology import utc_microseconds

from .contracts import (
    OwnerToken,
    Scope,
    SemanticConflict,
    VersionConflict,
    canonical,
    companion_id,
    identifier,
)
from .evidence import EvidenceTurn, evidence_digest
from .exchange_resolution import NAMED_TARGETS_PROTOCOL, SourceResolution
from .exchange_resolution import digest as resolution_digest
from .exchanges import ExchangeCommand, ExchangeSupport
from .fact_resolution import FactSourceResolution
from .memory import MemoryMutation
from .memory_support import command_digest, target_digest
from .sqlite_background import SQLiteBackgroundStore
from .sqlite_turns import _locked_time, _time
from .turns import RunClaim

_EXTRA = (
    "CREATE TABLE background_work (companion TEXT NOT NULL, job TEXT NOT NULL, "
    "worker TEXT, lease_until INTEGER, attempts INTEGER NOT NULL DEFAULT 0, "
    "available_us INTEGER NOT NULL DEFAULT 0, error TEXT, receipt TEXT, "
    "PRIMARY KEY(companion,job), FOREIGN KEY(companion,job) "
    "REFERENCES background_sources(companion,job))",
)


@dataclass(frozen=True)
class BackgroundClaim:
    companion: str
    session: str
    job: str
    worker: str
    generation: int
    role: str
    source: str
    source_hash: str
    lease_until: int
    attempts: int
    memory_generation: int = 0
    foreground: RunClaim | None = None


class SQLiteBackgroundWorkerStore(SQLiteBackgroundStore):
    schema_version = 15
    schema = SQLiteBackgroundStore.schema + _EXTRA
    tables = SQLiteBackgroundStore.tables | {"background_work"}
    max_attempts = 3
    clarification_errors = frozenset({
        "needs_clarification.fact_target_conflict",
        "needs_clarification.source_coverage",
    })

    def _accepted(self, db, receipt, trigger):
        super()._accepted(db, receipt, trigger)
        # A newly pending user source invalidates any previously selected
        # fact/summary context, even when no fact has been written yet.
        before = self._snapshot(db, receipt.companion_id)
        after = replace(before, write_version=before.write_version + 1)
        if db.execute(
            "UPDATE companion_state SET version=?,snapshot=? "
            "WHERE companion=? AND version=?",
            (after.write_version, canonical(asdict(after)), receipt.companion_id,
             before.write_version),
        ).rowcount != 1:
            raise VersionConflict("User source intake lost context version comparison")

    def _preempt_background(self, db, companion):
        # An unrelated new input or foreground claim is not a failed model
        # attempt. The source itself remains queued with a new generation.
        db.execute(
            "UPDATE background_work SET attempts=MAX(0,attempts-1),"
            "worker=NULL,lease_until=NULL WHERE companion=? AND job IN "
            "(SELECT job FROM background_sources WHERE companion=? AND status='running')",
            (companion, companion),
        )

    def _recall(self, db, companion, *, now_us, candidates, limit):
        # Read the barrier and facts in one snapshot; a pending correction
        # cannot leak a stale fact through a between-query race.
        if db.execute(
            "SELECT 1 FROM background_sources WHERE companion=? AND role='user' "
            "AND status<>'completed' LIMIT 1", (companion,),
        ).fetchone():
            return []
        return super()._recall(db, companion, now_us=now_us,
                               candidates=candidates, limit=limit)

    def _upgrade(self, db, version, application, names):
        raise ValueError("Background worker requires v15; existing databases are not migrated")

    def _pending_source_count(self, db, companion):
        return db.execute(
            "SELECT count(*) FROM background_sources s WHERE s.companion=? "
            "AND s.status<>'completed' AND NOT (s.role='assistant' AND s.status='failed' "
            "AND EXISTS (SELECT 1 FROM background_work w WHERE w.companion=s.companion "
            "AND w.job=s.job AND w.error='source_invalidated'))", (companion,),
        ).fetchone()[0]

    def background_sources(self, companion, *, limit=100):
        """Expose durable lifecycle metadata without source text in the error field."""
        companion_id(companion)
        if type(limit) is not int or not 1 <= limit <= 256:
            raise ValueError("Bounded source queue read required")
        with self.transaction(write=False) as db:
            rows = [dict(row) for row in db.execute(
                "SELECT s.*,w.attempts,w.available_us,w.error "
                "FROM background_sources s LEFT JOIN background_work w "
                "ON s.companion=w.companion AND s.job=w.job "
                "WHERE s.companion=? AND s.status<>'completed' ORDER BY "
                "CASE WHEN s.role='assistant' AND s.status='failed' "
                "AND w.error='source_invalidated' THEN 1 ELSE 0 END,s.rowid LIMIT ?",
                (companion, limit),
            )]
            for row in rows:
                if (row["role"] == "assistant" and row["status"] == "failed"
                        and row["error"] == "source_invalidated"):
                    row["status"] = "invalidated"
            return rows

    def memory_status(self, companion):
        """Return the conservative, model-visible state of user-source maintenance."""
        companion_id(companion)
        errors = tuple(sorted(self.clarification_errors))
        placeholders = ",".join("?" for _ in errors)
        with self.transaction(write=False) as db:
            # Queue pagination is a display concern. A late user correction must
            # remain visible even behind hundreds of assistant maintenance jobs.
            pending, clarification = db.execute(
                "SELECT count(*),max(CASE WHEN s.status='failed' AND w.error IN ("
                + placeholders + ") THEN 1 ELSE 0 END) "
                "FROM background_sources s LEFT JOIN background_work w "
                "ON s.companion=w.companion AND s.job=w.job "
                "WHERE s.companion=? AND s.role='user' AND s.status<>'completed'",
                (*errors, companion),
            ).fetchone()
        if not pending:
            return "current"
        if clarification:
            return "needs_clarification"
        return "pending_source"

    def background_claim_current(self, claim, *, now_us):
        """Short read for cooperative inference cancellation and foreground priority."""
        _time(now_us)
        with self.transaction(write=False, timeout=0.025) as db:
            row = self._live(db, claim, now_us)
            if row["status"] != "running":
                raise VersionConflict("Background work already completed")
            if claim.foreground is None and db.execute(
                "SELECT 1 FROM turn_runs WHERE companion=? AND status='running' "
                "AND lease_until>? LIMIT 1", (claim.companion, now_us),
            ).fetchone():
                raise VersionConflict("Foreground preempted background inference")

    def next_background_source(self, *, now_us, companion=None):
        """Oldest eligible source in scope; failed jobs keep their barrier.

        Global selection remains available to explicit multi-companion experiments.
        Desktop workers bind their companion before selecting or claiming a source.
        """
        _time(now_us)
        if companion is not None:
            companion_id(companion)
        with self.transaction(write=False) as db:
            row = db.execute(
                "SELECT s.companion,s.role,s.job FROM background_sources s "
                "LEFT JOIN background_work w ON s.companion=w.companion AND s.job=w.job "
                "WHERE (s.status='queued' OR "
                "(s.status='retryable' AND w.available_us<=?) OR "
                "(s.status='running' AND w.lease_until<=?)) "
                "AND (? IS NULL OR s.companion=?) "
                "AND (s.role='user' OR NOT EXISTS (SELECT 1 FROM background_sources u "
                "WHERE u.companion=s.companion AND u.role='user' AND u.status<>'completed')) "
                "AND NOT EXISTS (SELECT 1 FROM turn_runs r WHERE r.companion=s.companion "
                "AND r.status='running' AND r.lease_until>?) "
                "AND NOT EXISTS (SELECT 1 FROM background_sources a JOIN background_work b "
                "ON a.companion=b.companion AND a.job=b.job "
                "WHERE a.companion=s.companion AND a.status='running' "
                "AND b.lease_until>?) ORDER BY s.rowid LIMIT 1",
                (now_us, now_us, companion, companion, now_us, now_us),
            ).fetchone()
            return dict(row) if row else None

    @staticmethod
    def _delegated_worker(foreground):
        if not isinstance(foreground, RunClaim):
            raise ValueError("Typed foreground source owner required")
        return "source-owner-" + sha256(canonical(asdict(foreground)).encode()).hexdigest()

    def _foreground_live(self, db, foreground, now_us, *, check_basis=True,
                         check_snapshot=True):
        if not isinstance(foreground, RunClaim):
            raise ValueError("Typed foreground source owner required")
        row = db.execute(
            "SELECT r.*,i.session,i.status AS input_status,s.owner AS state_owner,"
            "s.epoch AS state_epoch FROM turn_runs r JOIN turn_inputs i "
            "ON r.companion=i.companion AND r.input=i.input JOIN state_owners s "
            "ON r.companion=s.companion WHERE r.companion=? AND r.input=?",
            (foreground.companion_id, foreground.input_id),
        ).fetchone()
        if (row is None or row["status"] != "running" or row["input_status"] != "running"
                or row["lease_until"] <= now_us or row["session"] != foreground.session_id
                or row["attempt"] != foreground.attempt_id
                or (row["owner"], row["epoch"])
                != (foreground.owner.owner_id, foreground.owner.epoch)
                or (row["state_owner"], row["state_epoch"])
                != (foreground.owner.owner_id, foreground.owner.epoch)
                or (check_basis and row["basis"] != foreground.state_version)
                or (check_snapshot and row["basis"]
                    != self._snapshot(db, foreground.companion_id).write_version)
                or self._run_config(db, self._input(
                    db, foreground.companion_id, foreground.input_id)) != foreground.config_json):
            raise VersionConflict("Source delegation lost foreground ownership or basis")
        return row

    def _refresh_delegated_basis(self, db, claim, *, now_us, clock=None):
        if claim.foreground is None:
            return {}
        now_us = _locked_time(now_us, clock)
        # Re-read this transaction's authoritative child lease and identity.
        # Parent renewal cannot extend it, nor can a replaced claim field.
        source = self._source_live(db, claim, now_us)
        if source["status"] != "running":
            raise VersionConflict("Delegated source already completed")
        self._foreground_live(db, claim.foreground, now_us, check_snapshot=False)
        # Called only inside the already fenced source commit UOW, after its
        # effects. Even empty completion changes the pending-source projection.
        before = self._snapshot(db, claim.companion)
        after = replace(before, write_version=before.write_version + 1)
        db.execute("UPDATE companion_state SET version=?,snapshot=? WHERE companion=?",
                   (after.write_version, canonical(asdict(after)), claim.companion))
        changed = db.execute(
            "UPDATE turn_runs SET basis=? WHERE companion=? AND input=? AND attempt=? "
            "AND owner=? AND epoch=? AND basis=? AND status='running'",
            (after.write_version, claim.companion, claim.foreground.input_id,
             claim.foreground.attempt_id, claim.foreground.owner.owner_id,
             claim.foreground.owner.epoch, claim.foreground.state_version),
        ).rowcount
        if changed != 1:
            raise VersionConflict("Delegated source commit lost foreground basis")
        return {"foreground_basis": after.write_version,
                "foreground_binding": self._delegated_worker(claim.foreground)}

    def refresh_source_owner(self, foreground, *, source, now_us):
        """Recover only this owner's committed source basis; never rebase arbitrary work."""
        _time(now_us)
        worker = self._delegated_worker(foreground)
        if source != foreground.input_id:
            raise VersionConflict("Delegation source must be the current input")
        with self.transaction(write=False) as db:
            run = self._foreground_live(db, foreground, now_us, check_basis=False)
            row = db.execute(
                "SELECT w.receipt FROM background_sources s JOIN background_work w "
                "ON s.companion=w.companion AND s.job=w.job WHERE s.companion=? "
                "AND s.session=? AND s.source=? AND s.role='user' AND s.status='completed' "
                "AND w.worker=?",
                (foreground.companion_id, foreground.session_id, source, worker),
            ).fetchone()
            receipt = json.loads(row["receipt"]) if row and row["receipt"] else {}
            if (receipt.get("foreground_binding") != worker
                    or receipt.get("foreground_basis") != run["basis"]):
                raise VersionConflict("No matching delegated source completion")
            return replace(foreground, state_version=run["basis"],
                           lease_until_us=run["lease_until"])

    def claim_background(self, companion, worker, *, now_us, lease_us, role=None, source=None,
                         foreground=None, clock=None):
        companion_id(companion)
        identifier(worker)
        if source is not None:
            identifier(source)
        if role not in {None, "user", "assistant"}:
            raise ValueError("Named background source role required")
        _time(now_us)
        _time(lease_us)
        if not 0 < lease_us <= 300_000_000:
            raise ValueError("Bounded positive background lease required")
        _time(now_us + lease_us)
        if clock is not None and not callable(clock):
            raise ValueError("Trusted source clock required")
        with self.transaction() as db:
            now_us = _locked_time(now_us, clock)
            if foreground is not None:
                parent = self._foreground_live(db, foreground, now_us)
                if (companion != foreground.companion_id or role != "user"
                        or source != foreground.input_id):
                    raise VersionConflict("Delegation must name the owner's current user source")
                worker = self._delegated_worker(foreground)
                lease_us = min(lease_us, parent["lease_until"] - now_us)
            elif worker.startswith("source-owner-"):
                raise ValueError("Reserved delegated source worker identity")
            # An active foreground run owns scarce inference capacity. The
            # caller retries after it yields; no DB lock spans model work.
            if foreground is None and db.execute(
                "SELECT 1 FROM turn_runs WHERE companion=? AND status='running' "
                "AND lease_until>? LIMIT 1", (companion, now_us),
            ).fetchone():
                return None
            if db.execute(
                "SELECT 1 FROM background_sources s JOIN background_work w "
                "ON s.companion=w.companion AND s.job=w.job "
                "WHERE s.companion=? AND s.status='running' AND w.lease_until>? LIMIT 1",
                (companion, now_us),
            ).fetchone():
                return None
            row = db.execute(
                "SELECT s.*,w.lease_until,w.available_us,w.attempts FROM background_sources s "
                "LEFT JOIN background_work w ON s.companion=w.companion AND s.job=w.job "
                "WHERE s.companion=? AND (? IS NULL OR s.role=?) "
                "AND (? IS NULL OR s.source=?) AND (s.status='queued' OR "
                "(s.status='retryable' AND w.available_us<=?) OR "
                "(s.status='running' AND w.lease_until<=?)) "
                "ORDER BY s.rowid LIMIT 1",
                (companion, role, role, source, source, now_us, now_us),
            ).fetchone()
            if row is None:
                return None
            if foreground is not None and row["session"] != foreground.session_id:
                raise VersionConflict("Delegation source session differs from its owner")
            generation = row["generation"] + (row["status"] == "running")
            db.execute(
                "UPDATE background_sources SET status='running',generation=? "
                "WHERE companion=? AND job=?",
                (generation, companion, row["job"]),
            )
            db.execute(
                "INSERT INTO background_work "
                "(companion,job,worker,lease_until,attempts,available_us) "
                "VALUES(?,?,?,?,1,0) ON CONFLICT(companion,job) DO UPDATE SET "
                "worker=excluded.worker,lease_until=excluded.lease_until,"
                "attempts=background_work.attempts+1,error=NULL",
                (companion, row["job"], worker, now_us + lease_us),
            )
            attempts = (row["attempts"] or 0) + 1
            return BackgroundClaim(companion, row["session"], row["job"], worker, generation,
                                   row["role"], row["source"], row["source_hash"],
                                   now_us + lease_us, attempts, foreground=foreground)

    def _source_live(self, db, claim, now_us):
        if not isinstance(claim, BackgroundClaim):
            raise ValueError("Background claim required")
        if claim.worker.startswith("source-owner-") != (claim.foreground is not None):
            raise VersionConflict("Delegated worker cannot drop its foreground binding")
        if claim.foreground is not None and (
            claim.worker != self._delegated_worker(claim.foreground)
            or claim.companion != claim.foreground.companion_id
            or claim.session != claim.foreground.session_id
            or claim.source != claim.foreground.input_id or claim.role != "user"
        ):
            raise VersionConflict("Delegated source binding changed")
        row = db.execute(
            "SELECT s.*,w.worker,w.lease_until,w.attempts,w.receipt "
            "FROM background_sources s JOIN background_work w "
            "ON s.companion=w.companion AND s.job=w.job "
            "WHERE s.companion=? AND s.job=?", (claim.companion, claim.job),
        ).fetchone()
        if (row is None or row["generation"] != claim.generation
                or row["session"] != claim.session
                or row["worker"] != claim.worker or row["role"] != claim.role
                or row["source"] != claim.source or row["source_hash"] != claim.source_hash
                or row["attempts"] != claim.attempts):
            raise VersionConflict("Background claim generation or source changed")
        if row["status"] == "completed":
            return row
        if (row["status"] != "running" or row["lease_until"] <= now_us
                or row["lease_until"] != claim.lease_until):
            raise VersionConflict("Background lease expired or was preempted")
        return row

    def _live(self, db, claim, now_us):
        row = self._source_live(db, claim, now_us)
        if row["status"] != "completed" and claim.foreground is not None:
            self._foreground_live(db, claim.foreground, now_us)
        return row

    def _apply_fact_commands(self, db, claim, row, prefix, commands, *, evaluator,
                             now_us):
        """Apply already-supported fact commands inside the caller's UOW."""
        no_change_operations = 0
        for command in commands:
            if (command.kind not in {"assert_fact", "correct", "revoke"}
                    or command.scope != Scope(claim.companion, row["session"])
                    or command.source_kind != "input" or command.source_id != claim.source):
                raise ValueError("Fact command must bind the actual user source")
            try:
                reason = json.loads(command.reason)
            except (ValueError, TypeError) as error:
                raise ValueError("Structured fact support required") from error
            if not isinstance(reason, dict) or set(reason) != {"quote", "support"}:
                raise ValueError("Operation-specific fact support required")
            quote, support = reason["quote"], reason["support"]
            before = self._current(db, claim.companion, command.record_id)
            repeated = self._exact_repeated_fact(db, command)
            support_target = repeated or before
            if (not isinstance(quote, str) or not quote.strip() or quote not in prefix
                    or not isinstance(support, dict)
                    or support.get("verdict") != "supported"
                    or support.get("evaluator") != evaluator
                    or support.get("source_sha256") != claim.source_hash
                    or support.get("content_sha256")
                    != sha256(command.content.encode()).hexdigest()
                    or support.get("command_sha256") != command_digest(command, quote)):
                raise ValueError("Fact support differs from source or command")
            if support.get("target_sha256") != target_digest(support_target):
                raise VersionConflict("Fact support target revision changed")
            if (command.expires_at and utc_microseconds(command.expires_at) <= now_us) or (
                support_target and support_target.get("expires_at") and
                utc_microseconds(datetime.fromisoformat(support_target["expires_at"]))
                <= now_us
            ):
                raise VersionConflict("Fact target or proposal expired")
            if self._original(db, claim.companion, command.mutation_id):
                raise VersionConflict("Fact command already belongs to another batch")
            if repeated is not None:
                self._save_repeated_fact(db, command, repeated)
                no_change_operations += 1
                continue
            applied = self._apply_memory(db, command)
            if applied["disposition"] != "applied":
                raise SemanticConflict("Fact batch conflicts with current target")
        return no_change_operations

    def finish_background(self, claim, commands, *, evaluator, now_us,
                          exchanges=(), sources=(), resolution=None, fact_resolution=None,
                          clock=None):
        """Atomically apply supported user facts and bilateral dialogue effects."""
        _time(now_us)
        identifier(evaluator)
        if ((fact_resolution is not None and not isinstance(fact_resolution, FactSourceResolution))
                or (evaluator.startswith(
                    ("joint-source.v2.", "joint-source.v3.", "joint-source.v4.",
                     "joint-source.v5.", "joint-source.v6.", "joint-source.v7.",
                     "joint-source.v8.", "joint-source.v9.")
                )
                    and fact_resolution is None)):
            raise ValueError("Joint completion requires explicit fact source resolution")
        if (not isinstance(commands, tuple) or len(commands) > 8
                or any(not isinstance(c, MemoryMutation) for c in commands)
                or len({c.mutation_id for c in commands}) != len(commands)):
            raise ValueError("Unique bounded fact commands required")
        if claim.role != "user":
            raise ValueError("Assistant ACK requires exchange maintenance")
        if (not isinstance(exchanges, tuple) or len(exchanges) > 16
                or not isinstance(sources, tuple) or len(sources) > 32
                or any(not isinstance(s, EvidenceTurn) for s in sources)
                or any(not isinstance(p, tuple) or len(p) != 2
                       or not isinstance(p[0], ExchangeCommand)
                       or not isinstance(p[1], ExchangeSupport) for p in exchanges)
                or len({c.command_id for c, _ in exchanges}) != len(exchanges)
                or (exchanges and not sources)):
            raise ValueError("Typed bounded user exchange context and effects required")
        if resolution is not None and not isinstance(resolution, SourceResolution):
            raise ValueError("Typed source completion resolution required")
        command_digest_value = sha256(canonical(
            [c.fingerprint for c in commands]).encode()).hexdigest()
        exchange_digest = sha256(canonical(
            [c.fingerprint for c, _ in exchanges]).encode()).hexdigest()
        exchange_support_digest = sha256(canonical(
            [s.to_mapping() for _, s in exchanges]).encode()).hexdigest()
        supplied_context_hash = evidence_digest(sources) if sources else None
        with self.transaction() as db:
            now_us = _locked_time(now_us, clock)
            row = self._live(db, claim, now_us)
            if row["status"] == "completed":
                saved = json.loads(row["receipt"])
                if (saved["evaluator"] != evaluator
                        or saved["commands_sha256"] != command_digest_value
                        or saved.get("exchanges_sha256") != exchange_digest
                        or saved.get("exchange_support_sha256") != exchange_support_digest
                        or saved.get("context_sha256") != supplied_context_hash
                        or saved.get("resolution_sha256") != (
                            resolution.fingerprint if resolution else None)
                        or saved.get("fact_resolution_sha256") != (
                            fact_resolution.fingerprint if fact_resolution else None)):
                    raise VersionConflict("Conflicting background completion replay")
                return row["receipt"]
            value = self._exchange_source(db, Scope(claim.companion, row["session"]),
                                          claim.role, claim.source)
            prefix = value["text"][:row["confirmed_end"]]
            if (value["confirmed"] < row["confirmed_end"]
                    or sha256(prefix.encode()).hexdigest() != claim.source_hash):
                raise VersionConflict("Background source confirmation changed")
            if claim.foreground is None and (commands or exchanges) and db.execute(
                "SELECT 1 FROM turn_runs WHERE companion=? AND status='running' "
                "AND lease_until>? LIMIT 1", (claim.companion, now_us),
            ).fetchone():
                raise VersionConflict("Foreground owns the companion")
            actual = self._background_context(db, claim, now_us=now_us) if sources else ()
            if actual != sources:
                raise VersionConflict("User dialogue semantic context changed")
            context_hash = supplied_context_hash
            if fact_resolution is not None:
                if fact_resolution.source_id != claim.source:
                    raise ValueError("Fact resolution belongs to another source")
                fact_resolution.validate_source(prefix, context_hash, commands)
            if resolution is not None:
                self._check_source_resolution(
                    db, claim, row, resolution, context_hash, exchanges, now_us,
                    effects_sha256=exchange_digest, support_sha256=exchange_support_digest,
                )
            no_change_operations = self._apply_fact_commands(
                db,
                claim,
                row,
                prefix,
                commands,
                evaluator=evaluator,
                now_us=now_us,
            )
            if exchanges:
                if claim.foreground is not None:
                    owner = claim.foreground.owner
                else:
                    writer = db.execute(
                        "SELECT epoch FROM state_owners WHERE companion=?", (claim.companion,)
                    ).fetchone()
                    epoch = writer["epoch"] + 1
                    db.execute("UPDATE state_owners SET owner=?,epoch=? WHERE companion=?",
                               (claim.worker, epoch, claim.companion))
                    owner = OwnerToken(claim.companion, claim.worker, epoch)
                basis = self._snapshot(db, claim.companion).write_version
                for index, (command, support) in enumerate(exchanges):
                    if (command.scope != Scope(claim.companion, claim.session)
                            or command.evidence.role != "user"
                            or command.evidence.source_id != claim.source
                            or command.evidence.confirmed_end != row["confirmed_end"]
                            or command.evidence.prefix_sha256 != claim.source_hash
                            or support.context_sha256 != context_hash):
                        raise ValueError("User exchange lacks source and context support")
                    self._apply_exchange(
                        db, command, owner, expected_version=basis + index,
                        now_us=now_us, support=support,
                        batch_sha256=exchange_digest, context_sha256=context_hash,
                    )
            receipt = canonical({"job": claim.job, "generation": claim.generation,
                                 **self._refresh_delegated_basis(
                                     db, claim, now_us=now_us, clock=clock),
                                 "source_sha256": claim.source_hash,
                                 "evaluator": evaluator, "operations": len(commands),
                                 "no_change_operations": no_change_operations,
                                 "commands_sha256": command_digest_value,
                                 "exchanges_sha256": exchange_digest,
                                 "exchange_support_sha256": exchange_support_digest,
                                 "context_sha256": context_hash,
                                 **({"resolution": resolution.to_mapping(),
                                     "resolution_sha256": resolution.fingerprint}
                                    if resolution is not None else {}),
                                 **({"fact_resolution": fact_resolution.to_mapping(),
                                     "fact_resolution_sha256": fact_resolution.fingerprint}
                                    if fact_resolution is not None else {})})
            db.execute("UPDATE background_sources SET status='completed' "
                       "WHERE companion=? AND job=?", (claim.companion, claim.job))
            db.execute("UPDATE background_work SET receipt=?,lease_until=NULL "
                       "WHERE companion=? AND job=?", (receipt, claim.companion, claim.job))
            return receipt

    def finish_facts_with_clarification(
        self,
        claim,
        commands,
        *,
        evaluator,
        now_us,
        sources,
        reason,
        interpretation_sha256,
        action,
        requested_scope,
        unresolved,
        affected_records=(),
        fact_resolution=None,
    ):
        """Commit supported facts while retaining an unresolved source barrier.

        A user source may contain a supported fact and an unrelated request or
        question whose exchange coverage is still unresolved.  The fact has
        its own source support and must not be lost, but the source remains
        blocked until the unresolved dialogue portion is resolved.
        """
        _time(now_us)
        identifier(evaluator)
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
        if (
            not isinstance(commands, tuple)
            or not commands
            or len(commands) > 8
            or any(not isinstance(command, MemoryMutation) for command in commands)
            or len({command.mutation_id for command in commands}) != len(commands)
            or claim.role != "user"
            or not isinstance(sources, tuple)
            or len(sources) > 32
            or any(not isinstance(source, EvidenceTurn) for source in sources)
        ):
            raise ValueError("Supported fact clarification requires a bounded user source")
        if fact_resolution is not None and not isinstance(fact_resolution, FactSourceResolution):
            raise ValueError("Typed fact source resolution required")
        supplied_context_hash = evidence_digest(sources)
        command_digest_value = sha256(canonical(
            [command.fingerprint for command in commands]).encode()).hexdigest()
        with self.transaction() as db:
            row = self._live(db, claim, now_us)
            if row["status"] == "completed":
                raise VersionConflict("Completed source cannot require clarification")
            value = self._exchange_source(
                db, Scope(claim.companion, row["session"]), claim.role, claim.source
            )
            prefix = value["text"][:row["confirmed_end"]]
            if (value["confirmed"] < row["confirmed_end"]
                    or sha256(prefix.encode()).hexdigest() != claim.source_hash):
                raise VersionConflict("Background source confirmation changed")
            if db.execute(
                "SELECT 1 FROM turn_runs WHERE companion=? AND status='running' "
                "AND lease_until>? LIMIT 1", (claim.companion, now_us),
            ).fetchone():
                raise VersionConflict("Foreground owns the companion")
            actual = self._background_context(db, claim, now_us=now_us)
            if actual != sources:
                raise VersionConflict("User dialogue semantic context changed")
            context_hash = supplied_context_hash
            if fact_resolution is not None:
                if fact_resolution.source_id != claim.source:
                    raise ValueError("Fact resolution belongs to another source")
                fact_resolution.validate_source(prefix, context_hash, commands)
            no_change_operations = self._apply_fact_commands(
                db,
                claim,
                row,
                prefix,
                commands,
                evaluator=evaluator,
                now_us=now_us,
            )
            receipt = canonical({
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
                "partial_fact_operations": len(commands),
                "partial_fact_no_change_operations": no_change_operations,
                "fact_evaluator": evaluator,
                "fact_commands_sha256": command_digest_value,
                "fact_context_sha256": context_hash,
                **({
                    "fact_resolution": fact_resolution.to_mapping(),
                    "fact_resolution_sha256": fact_resolution.fingerprint,
                } if fact_resolution is not None else {}),
            })
            db.execute(
                "UPDATE background_sources SET status='failed' WHERE companion=? AND job=?",
                (claim.companion, claim.job),
            )
            db.execute(
                "UPDATE background_work SET receipt=?,error=?,available_us=?,lease_until=NULL "
                "WHERE companion=? AND job=?",
                (receipt, reason, now_us, claim.companion, claim.job),
            )
            return json.loads(receipt)

    def _exact_repeated_fact(self, db, command):
        if command.kind != "assert_fact" or command.fact_key is None:
            return None
        slot = canonical(asdict(command.fact_key))
        row = db.execute(
            "SELECT v.data FROM memory_heads h JOIN memory_versions v "
            "ON h.companion=v.companion AND h.record=v.record AND h.version=v.version "
            "WHERE h.companion=? AND h.slot=?",
            (command.scope.companion_id, slot),
        ).fetchone()
        if row is None:
            return None
        current = json.loads(row[0])
        expires_at = command.expires_at.isoformat() if command.expires_at else None
        if (
            current["kind"] != "fact"
            or current["status"] != "active"
            or current["content"] != command.content
            or current["expires_at"] != expires_at
        ):
            return None
        return current

    def _save_repeated_fact(self, db, command, current):
        state = self._snapshot(db, command.scope.companion_id)
        receipt = {
            "companion_id": command.scope.companion_id,
            "mutation_id": command.mutation_id,
            "record_id": current["record_id"],
            "requested_record_id": command.record_id,
            "disposition": "unchanged",
            "record_version": current["version"],
            "write_version": state.write_version,
            "delivery_id": None,
        }
        db.execute(
            "INSERT INTO memory_commands VALUES(?,?,?,?,?)",
            (
                command.scope.companion_id,
                command.mutation_id,
                command.fingerprint,
                canonical(command.to_mapping()),
                canonical(receipt),
            ),
        )
        return receipt

    def finish_resolved_background(self, claim, commands, *, evaluator, now_us,
                                   exchanges, sources, resolution, fact_resolution=None,
                                   clock=None):
        if not isinstance(resolution, SourceResolution):
            raise ValueError("Completion needs an explicit source resolution")
        return self.finish_background(
            claim, commands, evaluator=evaluator, now_us=now_us,
            exchanges=exchanges, sources=sources, resolution=resolution,
            fact_resolution=fact_resolution, clock=clock,
        )

    def _check_source_resolution(self, db, claim, row, resolution, context_hash,
                                 exchanges, now_us, *, effects_sha256, support_sha256):
        if (resolution.source_id != claim.source
                or resolution.source_sha256 != claim.source_hash
                or resolution.context_sha256 != context_hash
                or resolution.effects_sha256 != effects_sha256
                or resolution.support_sha256 != support_sha256
                or resolution.no_exchange != (not exchanges)
                or any(support.evaluator != resolution.evaluator
                       for _, support in exchanges)):
            raise ValueError("Resolution differs from actual source and supported effects")
        eligible = self._eligible_exchange_targets(db, claim.companion, now_us=now_us)
        if resolution.protocol == NAMED_TARGETS_PROTOCOL:
            references = set(resolution.target_refs)
            if any(command.action != "open"
                   and (command.exchange_id, command.expected_revision) not in references
                   for command, _ in exchanges):
                raise ValueError("Exchange effect target is outside the supplied snapshot")
            targets = [value for value in eligible
                       if (value["exchange_id"], value["revision"]) in references]
            if len(targets) != len(references):
                raise VersionConflict("Exchange resolution target snapshot changed")
        else:
            # Historical v1 means the complete catalog, never a truncated view.
            targets = list(islice(eligible, 101))
            if len(targets) > 100:
                raise ValueError("Exchange target capacity exceeded during resolution")
        if resolution.targets_sha256 != resolution_digest(targets):
            raise VersionConflict("Exchange resolution target snapshot changed")

    def _background_context(self, db, claim, *, now_us=None):
        if claim.role == "assistant":
            row = db.execute(
                "SELECT i.rowid AS ordinal FROM turn_deliveries d JOIN turn_inputs i "
                "ON d.companion=i.companion AND d.input=i.input "
                "WHERE d.companion=? AND d.delivery=? AND i.session=?",
                (claim.companion, claim.source, claim.session),
            ).fetchone()
        else:
            row = db.execute(
                "SELECT rowid AS ordinal FROM turn_inputs "
                "WHERE companion=? AND input=? AND session=?",
                (claim.companion, claim.source, claim.session),
            ).fetchone()
        if row is None:
            raise VersionConflict("Background dialogue source moved")
        rows = db.execute(
            "SELECT i.rowid AS ordinal,0 AS phase,i.input AS source,'user' AS role,"
            "i.text AS text,length(i.text) AS confirmed FROM turn_inputs i "
            "WHERE i.companion=? AND i.session=? AND i.rowid<=? "
            "AND NOT EXISTS (SELECT 1 FROM turn_triggers t WHERE "
            "t.companion=i.companion AND t.input=i.input) "
            "UNION ALL SELECT i.rowid,1,d.delivery,'assistant',d.text,d.displayed_end "
            "FROM turn_inputs i JOIN turn_deliveries d "
            "ON i.companion=d.companion AND i.input=d.input "
            "WHERE i.companion=? AND i.session=? AND i.rowid<=? "
            "AND d.displayed_end>0 AND d.status IN ('completed','cancelled') "
            "ORDER BY ordinal DESC,phase DESC LIMIT 32",
            (claim.companion, claim.session, row["ordinal"],
             claim.companion, claim.session,
             row["ordinal"] if claim.role == "assistant" else row["ordinal"] - 1),
        ).fetchall()
        scope = Scope(claim.companion, claim.session)
        sources = tuple(
            EvidenceTurn(scope, item["source"], item["role"],
                         item["text"][:item["confirmed"]], item["confirmed"])
            for item in reversed(rows)
        )
        if (not sources or sources[-1].role != claim.role
                or sources[-1].source_id != claim.source
                or sha256(sources[-1].text.encode()).hexdigest() != claim.source_hash):
            raise VersionConflict("Dialogue source context or confirmed prefix changed")
        return sources

    def background_exchange_context(self, claim, *, now_us=None):
        if not isinstance(claim, BackgroundClaim):
            raise ValueError("Background dialogue claim required")
        with self.transaction(write=False) as db:
            self._live(db, claim, 0 if now_us is None else now_us)
            return self._background_context(db, claim, now_us=now_us)

    def finish_assistant_background(self, claim, proposals, sources, *, evaluator, now_us,
                                    resolution=None):
        """One ACK source, semantic context and exchange batch in a short UOW."""
        _time(now_us)
        identifier(evaluator)
        if (not isinstance(claim, BackgroundClaim) or claim.role != "assistant"
                or not isinstance(proposals, tuple) or len(proposals) > 16
                or not isinstance(sources, tuple) or len(sources) > 32
                or any(not isinstance(s, EvidenceTurn) for s in sources)
                or any(not isinstance(p, tuple) or len(p) != 2
                       or not isinstance(p[0], ExchangeCommand)
                       or not isinstance(p[1], ExchangeSupport) for p in proposals)
                or len({c.command_id for c, _ in proposals}) != len(proposals)):
            raise ValueError("Typed assistant source, context and bounded effects required")
        if resolution is not None and not isinstance(resolution, SourceResolution):
            raise ValueError("Typed assistant source resolution required")
        digest = sha256(canonical([c.fingerprint for c, _ in proposals]).encode()).hexdigest()
        support_digest = sha256(canonical(
            [s.to_mapping() for _, s in proposals]).encode()).hexdigest()
        supplied_context_hash = evidence_digest(sources) if sources else None
        with self.transaction() as db:
            row = self._live(db, claim, now_us)
            if row["status"] == "completed":
                saved = json.loads(row["receipt"])
                if (saved["evaluator"] != evaluator or saved["commands_sha256"] != digest
                        or saved.get("support_sha256") != support_digest
                        or saved.get("context_sha256") != supplied_context_hash
                        or saved.get("resolution_sha256") != (
                            resolution.fingerprint if resolution else None)):
                    raise VersionConflict("Conflicting assistant maintenance replay")
                return row["receipt"]
            actual = self._background_context(db, claim, now_us=now_us)
            if actual != sources:
                raise VersionConflict("Assistant semantic context changed")
            if (actual[-1].response_type == "ack" and any(
                command.direction == "assistant_response" and command.action in {"resolve", "close"}
                for command, _ in proposals
            )):
                raise ValueError("Acknowledgement cannot settle a response obligation")
            context_hash = supplied_context_hash
            if resolution is not None:
                self._check_source_resolution(
                    db, claim, row, resolution, context_hash, proposals, now_us,
                    effects_sha256=digest, support_sha256=support_digest,
                )
            if proposals:
                if db.execute(
                    "SELECT 1 FROM turn_runs WHERE companion=? AND status='running' "
                    "AND lease_until>? LIMIT 1", (claim.companion, now_us),
                ).fetchone():
                    raise VersionConflict("Foreground owns the companion")
                writer = db.execute(
                    "SELECT epoch FROM state_owners WHERE companion=?", (claim.companion,)
                ).fetchone()
                epoch = writer["epoch"] + 1
                db.execute("UPDATE state_owners SET owner=?,epoch=? WHERE companion=?",
                           (claim.worker, epoch, claim.companion))
                owner = OwnerToken(claim.companion, claim.worker, epoch)
                basis = self._snapshot(db, claim.companion).write_version
                for index, (command, support) in enumerate(proposals):
                    if (command.scope != Scope(claim.companion, claim.session)
                            or command.evidence.role != "assistant"
                            or command.evidence.source_id != claim.source
                            or command.evidence.confirmed_end != row["confirmed_end"]
                            or command.evidence.prefix_sha256 != claim.source_hash
                            or support.evaluator != evaluator
                            or support.context_sha256 != context_hash):
                        raise ValueError("Exchange effect lacks actual ACK and context support")
                    self._apply_exchange(
                        db, command, owner, expected_version=basis + index,
                        now_us=now_us, support=support,
                        batch_sha256=digest, context_sha256=context_hash,
                    )
            receipt = canonical({"job": claim.job, "generation": claim.generation,
                                 "source_sha256": claim.source_hash,
                                 "evaluator": evaluator, "operations": len(proposals),
                                 "commands_sha256": digest,
                                 "support_sha256": support_digest,
                                 "context_sha256": context_hash,
                                 **({"response_type": "ack"}
                                    if actual[-1].response_type == "ack" else {}),
                                 **({"resolution": resolution.to_mapping(),
                                     "resolution_sha256": resolution.fingerprint}
                                    if resolution is not None else {})})
            db.execute("UPDATE background_sources SET status='completed' "
                       "WHERE companion=? AND job=?", (claim.companion, claim.job))
            db.execute("UPDATE background_work SET receipt=?,lease_until=NULL "
                       "WHERE companion=? AND job=?", (receipt, claim.companion, claim.job))
            return receipt

    def finish_resolved_assistant_background(self, claim, proposals, sources, *,
                                             evaluator, now_us, resolution):
        if not isinstance(resolution, SourceResolution):
            raise ValueError("Assistant completion needs explicit source resolution")
        return self.finish_assistant_background(
            claim, proposals, sources, evaluator=evaluator, now_us=now_us,
            resolution=resolution,
        )

    def fail_background(self, claim, *, error, now_us):
        _time(now_us)
        identifier(error)
        if error == "source_invalidated":
            raise ValueError("Invalidation requires authoritative source validation")
        with self.transaction() as db:
            row = self._live(db, claim, now_us)
            if row["status"] == "completed":
                raise VersionConflict("Completed background work cannot fail")
            terminal = row["attempts"] >= self.max_attempts
            delay = min(60_000_000, 1_000_000 * 2 ** (row["attempts"] - 1))
            db.execute("UPDATE background_sources SET status=? WHERE companion=? AND job=?",
                       ("failed" if terminal else "retryable", claim.companion, claim.job))
            db.execute("UPDATE background_work SET error=?,available_us=?,lease_until=NULL "
                       "WHERE companion=? AND job=?",
                       (error, now_us + delay, claim.companion, claim.job))

    def protect_background(self, claim, *, error, now_us):
        """Stop automatic retries while retaining the source and global barrier."""
        _time(now_us)
        if error not in self.clarification_errors:
            raise ValueError("Known content-free clarification reason required")
        with self.transaction() as db:
            row = self._live(db, claim, now_us)
            if row["status"] == "completed":
                raise VersionConflict("Completed background work cannot require clarification")
            db.execute(
                "UPDATE background_sources SET status='failed' WHERE companion=? AND job=?",
                (claim.companion, claim.job),
            )
            db.execute(
                "UPDATE background_work SET error=?,available_us=?,lease_until=NULL "
                "WHERE companion=? AND job=?",
                (error, now_us, claim.companion, claim.job),
            )

    def requeue_background(self, claim, *, now_us):
        """Yield a still-owned job after a related conflict without losing it."""
        _time(now_us)
        with self.transaction() as db:
            row = self._live(db, claim, now_us)
            if row["status"] != "running":
                raise VersionConflict("Completed background work cannot be requeued")
            db.execute(
                "UPDATE background_sources SET status='queued',generation=generation+1 "
                "WHERE companion=? AND job=?", (claim.companion, claim.job),
            )
            db.execute(
                "UPDATE background_work SET attempts=MAX(0,attempts-1),"
                "worker=NULL,lease_until=NULL,available_us=? "
                "WHERE companion=? AND job=?", (now_us, claim.companion, claim.job),
            )

    def _background_source_invalidated(self, db, claim, *, now_us):
        return False

    def invalidate_background_source(self, claim, *, now_us, clock=None):
        """Dispose only a proven ineffective ACK source; never certify semantics."""
        _time(now_us)
        with self.transaction() as db:
            now_us = _locked_time(now_us, clock)
            row = self._live(db, claim, now_us)
            if row["status"] != "running" or claim.role != "assistant":
                raise VersionConflict("Invalidation requires a live assistant source")
            source = self._exchange_source(
                db, Scope(claim.companion, claim.session), claim.role, claim.source,
            )
            prefix = source["text"][:row["confirmed_end"]]
            if (source["confirmed"] != row["confirmed_end"]
                    or sha256(prefix.encode()).hexdigest() != claim.source_hash):
                raise VersionConflict("Background source confirmation changed")
            if not self._background_source_invalidated(db, claim, now_us=now_us):
                return None
            self._live(db, claim, _locked_time(now_us, clock))
            receipt = canonical({
                "disposition": "source_invalidated", "role": "assistant",
                "job": claim.job, "source_id": claim.source,
                "source_sha256": claim.source_hash, "generation": claim.generation,
            })
            db.execute("UPDATE background_sources SET status='failed' "
                       "WHERE companion=? AND job=?", (claim.companion, claim.job))
            db.execute("UPDATE background_work SET receipt=?,error='source_invalidated',"
                       "lease_until=NULL WHERE companion=? AND job=?",
                       (receipt, claim.companion, claim.job))
            return receipt


class SQLiteResolvedBackgroundStore(SQLiteBackgroundWorkerStore):
    """Isolated v16 opt-in: every completion needs a source-resolution receipt."""

    schema_version = 16

    def finish_background(self, claim, commands, *, evaluator, now_us,
                          exchanges=(), sources=(), resolution=None, fact_resolution=None,
                          clock=None):
        if not isinstance(resolution, SourceResolution):
            raise ValueError("v16 source completion requires a resolution receipt")
        return super().finish_background(
            claim, commands, evaluator=evaluator, now_us=now_us,
            exchanges=exchanges, sources=sources, resolution=resolution,
            fact_resolution=fact_resolution, clock=clock,
        )

    def finish_assistant_background(self, claim, proposals, sources, *, evaluator, now_us,
                                    resolution=None):
        if not isinstance(resolution, SourceResolution):
            raise ValueError("v16 assistant completion requires a resolution receipt")
        return super().finish_assistant_background(
            claim, proposals, sources, evaluator=evaluator, now_us=now_us,
            resolution=resolution,
        )
