"""Opt-in v8 Harness journal inside the existing owner's database.

Append-only execution evidence is not a second business state owner. New synthetic
databases only: existing v7/private stores are rejected, never upgraded implicitly.
Checkpoint CAS and live attempt/owner/lease validation share one short transaction.
"""

import json
from hashlib import sha256

from .contracts import VersionConflict, canonical, companion_id, identifier
from .sqlite_lifecycle import SQLiteLifecycleStore
from .sqlite_turns import _locked_time, _time
from .turns import RunClaim

_RETIRED_SCHEMA = (
    "CREATE TABLE relationship_commands (companion TEXT NOT NULL "
    "REFERENCES companion_state(companion), effect TEXT NOT NULL, fingerprint TEXT NOT NULL, "
    "payload TEXT NOT NULL, receipt TEXT NOT NULL, "
    "PRIMARY KEY(companion,effect))",
    "CREATE TABLE relationship_effects (companion TEXT NOT NULL, delivery TEXT NOT NULL, "
    "effect TEXT NOT NULL, minimum_end INTEGER NOT NULL, payload TEXT NOT NULL, "
    "status TEXT NOT NULL, "
    "PRIMARY KEY(companion,effect), FOREIGN KEY(companion,delivery) "
    "REFERENCES turn_deliveries(companion,delivery))",
)

_EXTRA = (
    "CREATE TABLE harness_checkpoints (companion TEXT NOT NULL, input TEXT NOT NULL, "
    "attempt TEXT NOT NULL, subrun TEXT NOT NULL, revision INTEGER NOT NULL, "
    "owner TEXT NOT NULL, epoch INTEGER NOT NULL, fingerprint TEXT NOT NULL, "
    "payload TEXT NOT NULL, PRIMARY KEY(companion,attempt,subrun,revision), "
    "FOREIGN KEY(companion,input) REFERENCES turn_inputs(companion,input))",
)


def harness_binding(claim):
    if not isinstance(claim, RunClaim):
        raise ValueError("Typed Harness run claim required")
    return {
        "companion": claim.companion_id,
        "session": claim.session_id,
        "input": claim.input_id,
        "attempt": claim.attempt_id,
        "owner": claim.owner.owner_id,
        "epoch": claim.owner.epoch,
    }


class SQLiteHarnessStore(SQLiteLifecycleStore):
    schema_version = 8
    # Keep existing schema recognition and opaque historical evidence. Retired
    # projections are never replayed, including pending effects during an ACK.
    schema = SQLiteLifecycleStore.schema + _RETIRED_SCHEMA + _EXTRA
    tables = SQLiteLifecycleStore.tables | {
        "relationship_commands", "relationship_effects", "harness_checkpoints",
    }

    def mutate_relationship(self, effect, owner, *, expected_version):
        raise ValueError("Retired relationship effects are unsupported")

    def relationship_receipt(self, companion, effect_id):
        companion_id(companion)
        identifier(effect_id)
        with self.transaction(write=False) as db:
            row = db.execute(
                "SELECT receipt FROM relationship_commands WHERE companion=? AND effect=?",
                (companion, effect_id),
            ).fetchone()
            return json.loads(row[0]) if row else None

    def relationship_effects(self, companion, delivery_id):
        with self.transaction(write=False) as db:
            self._delivery(db, companion, delivery_id)
            return [dict(row) for row in db.execute(
                "SELECT effect,minimum_end,status FROM relationship_effects "
                "WHERE companion=? AND delivery=? ORDER BY rowid", (companion, delivery_id),
            )]

    def _upgrade(self, db, version, application, names):
        raise ValueError("Harness requires v8; existing databases are not migrated")

    def harness_journal(self, claim, subrun, *, clock):
        return HarnessJournal(self, claim, subrun, clock)

    def _check_retry_effects(self, db, companion, input_id):
        # Inspect each append-only stream's head, including older failed attempts.
        # New attempt IDs must never make an uncertain external effect disappear.
        rows = db.execute(
            "SELECT h.* FROM harness_checkpoints h WHERE companion=? AND input=? "
            "AND revision=(SELECT MAX(p.revision) FROM harness_checkpoints p "
            "WHERE p.companion=h.companion AND p.attempt=h.attempt AND p.subrun=h.subrun)",
            (companion, input_id),
        )
        for row in rows:
            snapshot = _snapshot(row)
            if _unresolved(snapshot):
                raise VersionConflict("Retry requires reconciliation of unresolved Harness effects")


def _snapshot(row):
    if sha256(row["payload"].encode()).hexdigest() != row["fingerprint"]:
        raise ValueError("Corrupt Harness checkpoint")
    snapshot = json.loads(row["payload"])
    required = {"binding", "messages", "status", "rounds", "cache_epoch", "pending", "uncertain"}
    if (
        not isinstance(snapshot, dict) or not required <= snapshot.keys()
        or not isinstance(snapshot["messages"], list)
        or not isinstance(snapshot["uncertain"], list)
    ):
        raise ValueError("Incomplete Harness execution checkpoint")
    binding = snapshot.get("binding", {})
    if any(binding.get(k) != row[k] for k in ("companion", "input", "attempt", "owner", "epoch")):
        raise ValueError("Corrupt Harness owner metadata")
    return snapshot


def _unresolved(snapshot):
    if snapshot["pending"] is not None or snapshot["uncertain"]:
        return True
    results = {}
    calls = set()
    for message in snapshot["messages"]:
        calls.update(call["id"] for call in message.get("tool_calls", []))
        if message.get("role") == "tool":
            outcome = json.loads(message["content"])["outcome"]
            if outcome not in {"success", "failure", "not_executed"}:
                return True
            results[message["tool_call_id"]] = outcome
    # Missing result/pending evidence is not proof that execution never started.
    return bool(calls - results.keys())


class HarnessJournal:
    def __init__(self, store, claim, subrun, clock):
        self.binding = harness_binding(claim)
        identifier(subrun)
        if not callable(clock):
            raise ValueError("Trusted current clock required")
        self.store, self.claim, self.subrun, self.clock = store, claim, subrun, clock
        self.revision = 0

    @property
    def key(self):
        return self.claim.companion_id, self.claim.attempt_id, self.subrun

    def _live(self, db, began):
        now = _locked_time(began, self.clock)
        c = self.claim
        row = db.execute(
            "SELECT r.*,i.session,i.status AS input_status,s.owner AS state_owner, "
            "s.epoch AS state_epoch FROM turn_runs r JOIN turn_inputs i "
            "ON r.companion=i.companion AND r.input=i.input JOIN state_owners s "
            "ON r.companion=s.companion WHERE r.companion=? AND r.input=?",
            (c.companion_id, c.input_id),
        ).fetchone()
        now = _locked_time(now, self.clock)
        if (
            row is None
            or row["status"] != "running"
            or row["input_status"] != "running"
            or row["lease_until"] <= now
            or row["session"] != c.session_id
            or row["attempt"] != c.attempt_id
            or row["basis"] != c.state_version
            or (row["owner"], row["epoch"]) != (c.owner.owner_id, c.owner.epoch)
            or (row["state_owner"], row["state_epoch"]) != (c.owner.owner_id, c.owner.epoch)
            or c.owner.companion_id != c.companion_id
        ):
            raise VersionConflict("Stale, cancelled or expired Harness owner")

    def _head(self, db):
        return db.execute(
            "SELECT * FROM harness_checkpoints WHERE companion=? AND attempt=? AND subrun=? "
            "ORDER BY revision DESC LIMIT 1",
            self.key,
        ).fetchone()

    def save(self, snapshot):
        if not isinstance(snapshot, dict) or snapshot.get("binding") != self.binding:
            raise VersionConflict("Harness checkpoint binding mismatch")
        payload = canonical(snapshot)
        if len(payload.encode()) > 2_000_000:
            raise ValueError("Harness checkpoint capacity exceeded")
        fingerprint = sha256(payload.encode()).hexdigest()
        began = self.clock()
        _time(began)
        with self.store.transaction(timeout=0.025) as db:
            self._live(db, began)
            head = self._head(db)
            revision = head["revision"] if head else 0
            if revision != self.revision:
                raise VersionConflict("Stale Harness checkpoint revision")
            db.execute(
                "INSERT INTO harness_checkpoints VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    self.claim.companion_id,
                    self.claim.input_id,
                    self.claim.attempt_id,
                    self.subrun,
                    revision + 1,
                    self.claim.owner.owner_id,
                    self.claim.owner.epoch,
                    fingerprint,
                    payload,
                ),
            )
        self.revision = revision + 1

    def load(self):
        return self._load(reconcile=False)

    def load_for_execution(self):
        """Restore effects under live ownership; no implicit tool replay permission.

        Same-input effects survive explicit retry. Only an explicitly new pure
        text attempt starts afresh; reopening the same attempt retains its spent
        rounds and candidate. The provider ledger owns cumulative billing.
        The snapshot is re-bound only here, inside the existing owner transaction.
        """
        began = self.clock()
        _time(began)
        with self.store.transaction(write=False, timeout=0.025) as db:
            self._live(db, began)
            # A renamed subrun is not a new authority to bypass another stream's
            # unresolved effect for this input. Reading/reconciling that stream
            # remains available through its original journal identity.
            others = db.execute(
                "SELECT h.* FROM harness_checkpoints h WHERE companion=? AND input=? "
                "AND subrun<>? AND revision=(SELECT MAX(p.revision) "
                "FROM harness_checkpoints p WHERE p.companion=h.companion "
                "AND p.attempt=h.attempt AND p.subrun=h.subrun)",
                (self.claim.companion_id, self.claim.input_id, self.subrun),
            )
            for other in others:
                previous = _snapshot(other)
                if _unresolved(previous):
                    raise VersionConflict("Another subrun has unresolved Harness effects")
            row = self._head(db)
            self.revision = row["revision"] if row else 0
            if row is None:
                row = db.execute(
                    "SELECT * FROM harness_checkpoints WHERE companion=? AND input=? "
                    "AND subrun=? ORDER BY rowid DESC LIMIT 1",
                    (self.claim.companion_id, self.claim.input_id, self.subrun),
                ).fetchone()
            if row is None:
                return None
            snapshot = _snapshot(row)
            previous = snapshot["binding"]
            if any(previous.get(k) != self.binding[k] for k in ("companion", "session", "input")):
                raise VersionConflict("Harness execution recovery binding mismatch")
            if (previous["attempt"] != self.binding["attempt"]
                    and not any(m.get("tool_calls") for m in snapshot["messages"])):
                return None
            if previous != self.binding:
                snapshot["binding"] = dict(self.binding)
                # A candidate bypasses reconciliation, but still invalidates caches
                # when the owner/attempt changes. Interrupted recovery increments it.
                if snapshot["status"] == "candidate":
                    snapshot["cache_epoch"] += 1
            self._live(db, began)
            return snapshot

    def load_for_reconciliation(self):
        """Current owner may inspect this same attempt's old binding, never run it."""
        return self._load(reconcile=True)

    def _load(self, *, reconcile):
        began = self.clock()
        _time(began)
        with self.store.transaction(write=False, timeout=0.025) as db:
            self._live(db, began)
            row = self._head(db)
            if row is None:
                return None
            if sha256(row["payload"].encode()).hexdigest() != row["fingerprint"]:
                raise ValueError("Corrupt Harness checkpoint")
            snapshot = json.loads(row["payload"])
            binding = snapshot.get("binding", {})
            if binding != self.binding:
                if (
                    not reconcile
                    or any(
                        binding.get(k) != self.binding[k]
                        for k in ("companion", "session", "input", "attempt")
                    )
                    or type(binding.get("epoch")) is not int
                    or binding["epoch"] >= self.binding["epoch"]
                ):
                    raise VersionConflict("Harness checkpoint binding requires reconciliation")
            if (binding.get("owner"), binding.get("epoch")) != (row["owner"], row["epoch"]):
                raise ValueError("Corrupt Harness owner metadata")
            self._live(db, began)
            self.revision = row["revision"]
            return snapshot
