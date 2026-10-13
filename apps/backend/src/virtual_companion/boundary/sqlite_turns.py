"""Opt-in schema-v2 turn adapter sharing the state adapter's short UnitOfWork.

No default/private database is opened. A v1 synthetic copy requires explicit
upgrade_v1=True; rollback uses its v1 backup and the matching application.
"""

import json
from contextlib import nullcontext
from dataclasses import asdict, replace
from hashlib import sha256
from uuid import uuid4

from virtual_companion.text.identity import CompanionConfig

from .contracts import (
    OwnerToken,
    StateSnapshot,
    VersionConflict,
    canonical,
    companion_id,
    identifier,
)
from .receiving_context import receiving_from_binding, validate_receiving_context
from .sqlite_delivery import DeliveryOperations
from .sqlite_state import _APPLICATION_ID, SQLiteStateStore
from .turns import CognitiveResult, IntakeReceipt, RunClaim

_TURN_SCHEMA = (
    "CREATE TABLE turn_sessions (companion TEXT NOT NULL REFERENCES companion_state(companion), "
    "session TEXT NOT NULL, PRIMARY KEY(companion,session))",
    "CREATE TABLE turn_inputs (companion TEXT NOT NULL, input TEXT NOT NULL, session TEXT NOT "
    "NULL, "
    "client_key TEXT NOT NULL, request_hash TEXT NOT NULL, text TEXT NOT NULL, config TEXT NOT "
    "NULL, "
    "status TEXT NOT NULL CHECK(status IN ('queued','running','committed','cancelled')), "
    "PRIMARY KEY(companion,input), UNIQUE(companion,client_key), "
    "FOREIGN KEY(companion,session) REFERENCES turn_sessions(companion,session))",
    "CREATE TABLE turn_runs (companion TEXT NOT NULL, input TEXT NOT NULL, attempt TEXT NOT NULL, "
    "owner TEXT NOT NULL, epoch INTEGER NOT NULL, basis INTEGER NOT NULL, lease_until INTEGER "
    "NOT NULL, "
    "status TEXT NOT NULL CHECK(status IN ('running','committed','cancelled')), "
    "PRIMARY KEY(companion,input), UNIQUE(companion,attempt), "
    "FOREIGN KEY(companion,input) REFERENCES turn_inputs(companion,input))",
    "CREATE UNIQUE INDEX turn_one_running ON turn_runs(companion) WHERE status='running'",
    "CREATE TABLE turn_commits (companion TEXT NOT NULL, input TEXT NOT NULL, fingerprint TEXT "
    "NOT NULL, "
    "receipt TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(companion,input), "
    "FOREIGN KEY(companion,input) REFERENCES turn_inputs(companion,input))",
    "CREATE TABLE turn_deliveries (companion TEXT NOT NULL, input TEXT NOT NULL, delivery TEXT "
    "NOT NULL, "
    "text TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN "
    "('pending','published','stopping','completed','cancelled')), "
    "displayed_end INTEGER NOT NULL DEFAULT 0, sequence INTEGER NOT NULL DEFAULT 0, "
    "PRIMARY KEY(companion,input), UNIQUE(companion,delivery), "
    "FOREIGN KEY(companion,input) REFERENCES turn_inputs(companion,input))",
    "CREATE TABLE turn_publications (companion TEXT NOT NULL, delivery TEXT NOT NULL, event "
    "TEXT NOT NULL, "
    "PRIMARY KEY(companion,delivery), FOREIGN KEY(companion,delivery) REFERENCES "
    "turn_deliveries(companion,delivery))",
    "CREATE TABLE turn_acks (companion TEXT NOT NULL, delivery TEXT NOT NULL, sequence INTEGER "
    "NOT NULL, "
    "fingerprint TEXT NOT NULL, receipt TEXT NOT NULL, PRIMARY KEY(companion,delivery,sequence), "
    "FOREIGN KEY(companion,delivery) REFERENCES turn_deliveries(companion,delivery))",
)


def _time(value):
    if type(value) is not int or not 0 <= value < 2**63:
        raise ValueError("Trusted nonnegative UTC microseconds required")


def _locked_time(now_us, clock):
    if clock is None:
        return now_us
    acquired_us = clock()
    _time(acquired_us)
    if acquired_us < now_us:
        raise VersionConflict("Transaction clock moved backwards")
    return acquired_us


class SQLiteTurnStore(DeliveryOperations, SQLiteStateStore):
    schema_version = 2
    schema = SQLiteStateStore.schema + _TURN_SCHEMA
    tables = SQLiteStateStore.tables | {
        "turn_sessions",
        "turn_inputs",
        "turn_runs",
        "turn_commits",
        "turn_deliveries",
        "turn_publications",
        "turn_acks",
    }

    def __init__(self, path, *, upgrade_v1=False):
        self.upgrade_v1 = upgrade_v1
        super().__init__(path)

    def _upgrade(self, db, version, application, names):
        if not (
            self.upgrade_v1
            and version == 1
            and application == _APPLICATION_ID
            and names == SQLiteStateStore.tables
        ):
            return super()._upgrade(db, version, application, names)
        for statement in _TURN_SCHEMA:
            db.execute(statement)
        db.execute("PRAGMA user_version=2")

    @staticmethod
    def _intake(row):
        return IntakeReceipt(
            row["companion"],
            row["session"],
            row["input"],
            row["client_key"],
            row["request_hash"],
            row["config"],
        )

    def accept(self, config, client_key, text, *, session_id=None, receiving_context=None):
        if not isinstance(config, CompanionConfig):
            raise ValueError("Companion configuration required")
        identifier(client_key)
        if not isinstance(text, str) or not text.strip() or len(text.encode()) > 64000:
            raise ValueError("Nonempty bounded UTF-8 input required")
        if session_id is not None:
            identifier(session_id)
        return self._accept(config, client_key, text, session_id=session_id,
                            receiving_context=receiving_context)

    def ensure_session(self, companion, session_id):
        """Create an empty session in a registered companion scope.

        Adapters that mirror an existing conversation store use this to keep
        the same session identity without inventing a second session mapping.
        It does not accept a turn or change the state version.
        """
        companion_id(companion)
        identifier(session_id)
        with self.transaction() as db:
            if not db.execute(
                "SELECT 1 FROM companion_state WHERE companion=?", (companion,)
            ).fetchone():
                raise ValueError("Unknown companion")
            db.execute(
                "INSERT OR IGNORE INTO turn_sessions(companion,session) VALUES(?,?)",
                (companion, session_id),
            )

    def _accept(self, config, client_key, text, *, session_id, trigger=None, db=None,
                receiving_context=None):
        """Adapter-only intake; extra typed origins share the same queue transaction."""
        companion = config.companion_id
        identity = [companion, session_id, text]
        if trigger is not None:
            identity.append(trigger)
        if receiving_context is not None:
            receiving_context = validate_receiving_context(receiving_context)
            identity.append({'receiving_context': receiving_context})
        fingerprint = sha256(canonical(identity).encode()).hexdigest()
        with (self.transaction() if db is None else nullcontext(db)) as db:
            original = db.execute(
                "SELECT * FROM turn_inputs WHERE companion=? AND client_key=?",
                (companion, client_key),
            ).fetchone()
            if original:
                if original["request_hash"] != fingerprint:
                    raise ValueError("Conflicting client request identity")
                return self._intake(original)
            if (
                session_id is not None
                and not db.execute(
                    "SELECT 1 FROM turn_sessions WHERE companion=? AND session=?",
                    (companion, session_id),
                ).fetchone()
            ):
                raise ValueError("Unknown session in companion scope")
            db.execute(
                "INSERT OR IGNORE INTO companion_state VALUES(?,0,?)",
                (companion, canonical(asdict(StateSnapshot(companion)))),
            )
            db.execute("INSERT OR IGNORE INTO state_owners VALUES(?,NULL,0)", (companion,))
            session = session_id or str(uuid4())
            if session_id is None:
                db.execute("INSERT INTO turn_sessions VALUES(?,?)", (companion, session))
            input_id = str(uuid4())
            binding = canonical(config.attempt_binding(
                {} if receiving_context is None else {'receiving_context': receiving_context}))
            db.execute(
                "INSERT INTO turn_inputs VALUES(?,?,?,?,?,?,?,'queued')",
                (companion, input_id, session, client_key, fingerprint, text, binding),
            )
            receipt = IntakeReceipt(companion, session, input_id, client_key, fingerprint, binding)
            self._accepted(db, receipt, trigger)
            return receipt

    def _accepted(self, db, receipt, trigger):
        if trigger is not None:
            raise ValueError("Internal origins require an explicit adapter")

    @staticmethod
    def _input(db, companion, input_id):
        row = db.execute(
            "SELECT * FROM turn_inputs WHERE companion=? AND input=?", (companion, input_id)
        ).fetchone()
        if row is None:
            raise ValueError("Unknown input in companion scope")
        return row

    def receiving_context_for_input(self, companion, input_id):
        """Read the original input, even when an explicit retry changes run config."""
        companion_id(companion)
        identifier(input_id)
        with self.transaction(write=False) as db:
            return receiving_from_binding(self._input(db, companion, input_id)['config'])

    def inspect(self, companion, input_id, *, timeout=5):
        companion_id(companion)
        identifier(input_id)
        with self.transaction(write=False, timeout=timeout) as db:
            item = dict(self._input(db, companion, input_id))
            receipt = db.execute(
                "SELECT receipt FROM turn_commits WHERE companion=? AND input=?",
                (companion, input_id),
            ).fetchone()
            item["commit"] = json.loads(receipt[0]) if receipt else None
            run = db.execute(
                "SELECT * FROM turn_runs WHERE companion=? AND input=?", (companion, input_id)
            ).fetchone()
            item["run"] = dict(run) if run else None
            owner = db.execute(
                "SELECT owner,epoch FROM state_owners WHERE companion=?", (companion,)
            ).fetchone()
            item["owner"] = dict(owner) if owner else None
            delivery = db.execute(
                "SELECT * FROM turn_deliveries WHERE companion=? AND input=?", (companion, input_id)
            ).fetchone()
            item["delivery"] = self._display(delivery) if delivery else None
            return item

    def claim(self, companion, input_id, worker, *, now_us, lease_us, clock=None):
        companion_id(companion)
        identifier(input_id)
        identifier(worker)
        _time(now_us)
        _time(lease_us)
        if not 0 < lease_us <= 300_000_000:
            raise ValueError("Lease must be positive and at most five minutes")
        _time(now_us + lease_us)
        with self.transaction() as db:
            now_us = _locked_time(now_us, clock)
            _time(now_us + lease_us)
            item = self._input(db, companion, input_id)
            if item["status"] not in {"queued", "running"}:
                raise VersionConflict(f"Input already {item['status']}; reconcile instead")
            self._before_claim(db, companion, input_id)
            if db.execute(
                "SELECT 1 FROM turn_deliveries WHERE companion=? AND status IN "
                "('pending','published','stopping')",
                (companion,),
            ).fetchone():
                raise VersionConflict("Awaiting delivery reconciliation")
            active = db.execute(
                "SELECT * FROM turn_runs WHERE companion=? AND status='running'", (companion,)
            ).fetchone()
            if active and (active["input"] != input_id or active["lease_until"] > now_us):
                raise VersionConflict("Companion already has an active run")
            earlier = db.execute(
                "SELECT input FROM turn_inputs WHERE companion=? "
                "AND status IN ('queued','running') ORDER BY rowid LIMIT 1",
                (companion,),
            ).fetchone()
            if earlier and earlier["input"] != input_id:
                raise VersionConflict("An earlier input must be processed or explicitly cancelled")
            epoch = (
                db.execute(
                    "SELECT epoch FROM state_owners WHERE companion=?", (companion,)
                ).fetchone()[0]
                + 1
            )
            db.execute(
                "UPDATE state_owners SET owner=?,epoch=? WHERE companion=?",
                (worker, epoch, companion),
            )
            basis = self._snapshot(db, companion).write_version
            attempt = active["attempt"] if active else str(uuid4())
            if active:
                db.execute(
                    "UPDATE turn_runs SET owner=?,epoch=?,basis=?,lease_until=? WHERE "
                    "companion=? AND input=?",
                    (worker, epoch, basis, now_us + lease_us, companion, input_id),
                )
            else:
                db.execute(
                    "INSERT INTO turn_runs VALUES(?,?,?,?,?,?,?,'running')",
                    (companion, input_id, attempt, worker, epoch, basis, now_us + lease_us),
                )
            db.execute(
                "UPDATE turn_inputs SET status='running' WHERE companion=? AND input=?",
                (companion, input_id),
            )
            return RunClaim(
                companion,
                item["session"],
                input_id,
                attempt,
                OwnerToken(companion, worker, epoch),
                basis,
                self._run_config(db, item),
                now_us + lease_us,
            )

    def _before_claim(self, db, companion, input_id):
        """Optional formats can fence unresolved external effects in this UOW."""

    def renew(self, claim, *, now_us, lease_us, clock=None, timeout=5):
        if not isinstance(claim, RunClaim):
            raise ValueError("Run claim required")
        _time(now_us)
        _time(lease_us)
        _time(now_us + lease_us)
        if not 0 < lease_us <= 300_000_000:
            raise ValueError("Lease must be positive and at most five minutes")
        with self.transaction(timeout=timeout) as db:
            now_us = _locked_time(now_us, clock)
            _time(now_us + lease_us)
            row = db.execute(
                "SELECT r.* FROM turn_runs r JOIN state_owners s ON r.companion=s.companion "
                "WHERE r.companion=? AND r.input=? AND r.attempt=? AND r.owner=? AND r.epoch=? "
                "AND r.status='running' AND r.lease_until>? AND r.owner=s.owner AND "
                "r.epoch=s.epoch",
                (
                    claim.companion_id,
                    claim.input_id,
                    claim.attempt_id,
                    claim.owner.owner_id,
                    claim.owner.epoch,
                    now_us,
                ),
            ).fetchone()
            if row is None or claim.owner.companion_id != claim.companion_id:
                raise VersionConflict("Stale or expired lease renewal")
            until = max(row["lease_until"], now_us + lease_us)
            db.execute(
                "UPDATE turn_runs SET lease_until=? WHERE companion=? AND input=?",
                (until, claim.companion_id, claim.input_id),
            )
            return replace(claim, lease_until_us=until)

    def commit(self, result, *, now_us, clock=None):
        if not isinstance(result, CognitiveResult):
            raise ValueError("Typed cognitive result required")
        _time(now_us)
        claim = result.claim
        key = (claim.companion_id, claim.input_id)
        with self.transaction() as db:
            original = db.execute(
                "SELECT * FROM turn_commits WHERE companion=? AND input=?", key
            ).fetchone()
            fingerprint = result.fingerprint
            if original:
                if original["fingerprint"] != fingerprint:
                    raise ValueError("Conflicting cognitive result for completed input")
                return json.loads(original["receipt"])
            now_us = _locked_time(now_us, clock)
            self._validate_result(result)
            item = self._input(db, *key)
            run = db.execute(
                "SELECT * FROM turn_runs WHERE companion=? AND input=?", key
            ).fetchone()
            writer = db.execute(
                "SELECT owner,epoch FROM state_owners WHERE companion=?", key[:1]
            ).fetchone()
            if (
                item["status"] != "running"
                or run is None
                or run["status"] != "running"
                or item["session"] != claim.session_id
                or self._run_config(db, item) != claim.config_json
                or run["attempt"] != claim.attempt_id
                or run["basis"] != claim.state_version
                or (run["owner"], run["epoch"]) != (claim.owner.owner_id, claim.owner.epoch)
                or claim.owner.companion_id != claim.companion_id
                or tuple(writer) != (claim.owner.owner_id, claim.owner.epoch)
            ):
                raise VersionConflict("Stale or cancelled run ownership")
            if run["lease_until"] <= now_us:
                raise VersionConflict("Run lease expired; reconcile and recover")
            if self._snapshot(db, claim.companion_id).write_version != claim.state_version:
                raise VersionConflict("Stale state snapshot before delivery")
            if result.assessment:
                # Do not borrow a previously committed event as this attempt's effect.
                if self._receipt(db, claim.companion_id, result.assessment.event_id):
                    raise VersionConflict("Assessment event already belongs to a prior commit")
                self._apply(
                    db, result.assessment, claim.owner, expected_version=claim.state_version
                )
            delivery = str(uuid4())
            receipt = {
                "status": "committed",
                "companion_id": claim.companion_id,
                "input_id": claim.input_id,
                "attempt_id": claim.attempt_id,
                "fingerprint": fingerprint,
                "candidate_id": result.candidate_id,
                "revision": result.revision,
                "review_id": result.review_id,
                "sha256": result.reviewed_sha256,
                "delivery_id": delivery,
                "delivery_status": "pending" if result.text else "completed",
                "displayed_end": 0,
                **(
                    {
                        "presentation": result.presentation.to_mapping(),
                        "presentation_sha256": result.presentation.candidate.fingerprint,
                    }
                    if result.presentation
                    else {}
                ),
            }
            db.execute(
                "INSERT INTO turn_commits VALUES(?,?,?,?,?)",
                (*key, fingerprint, canonical(receipt), result.payload),
            )
            db.execute(
                "INSERT INTO turn_deliveries VALUES(?,?,?,?,?,0,0)",
                (*key, delivery, result.text, receipt["delivery_status"]),
            )
            db.execute(
                "UPDATE turn_inputs SET status='committed' WHERE companion=? AND input=?", key
            )
            db.execute("UPDATE turn_runs SET status='committed' WHERE companion=? AND input=?", key)
            self._commit_memories(db, result, receipt)
            self._commit_exchanges(db, result, receipt)
            if run["lease_until"] <= _locked_time(now_us, clock):
                raise VersionConflict("Run lease expired during commit effects")
            return receipt

    def _commit_exchanges(self, db, result, receipt):
        if result.exchanges or result.exchange_sources:
            raise ValueError("Exchange effects require the opt-in exchange adapter")

    def _commit_memories(self, db, result, receipt):
        if result.memories:
            raise ValueError("Memory effects require the opt-in memory adapter")

    def _run_config(self, db, item):
        return item["config"]

    def _validate_result(self, result):
        if result.presentation is not None:
            raise ValueError("Presentation manifests require the opt-in v4 adapter")

    def _cancel_memories(self, db, companion, row):
        pass

    def _completed_internal(self, db, companion, input_id):
        return None

    def cancel(self, companion, input_id):
        companion_id(companion)
        identifier(input_id)
        with self.transaction() as db:
            item = self._input(db, companion, input_id)
            key = (companion, input_id)
            completed = self._completed_internal(db, companion, input_id)
            if completed is not None:
                return completed
            if item["status"] != "committed":
                db.execute(
                    "UPDATE turn_inputs SET status='cancelled' WHERE companion=? AND input=?", key
                )
                db.execute(
                    "UPDATE turn_runs SET status='cancelled' WHERE companion=? AND input=?", key
                )
                return {"status": "cancelled", "input_id": input_id}
            # A publish winner may already be visible. Stop request is not a host ACK.
            db.execute(
                "UPDATE turn_deliveries SET status=CASE status "
                "WHEN 'pending' THEN 'cancelled' WHEN 'published' THEN 'stopping' ELSE status END "
                "WHERE companion=? AND input=?",
                key,
            )
            row = db.execute(
                "SELECT * FROM turn_deliveries WHERE companion=? AND input=?", key
            ).fetchone()
            self._cancel_memories(db, companion, row)
            return self._display(row)
