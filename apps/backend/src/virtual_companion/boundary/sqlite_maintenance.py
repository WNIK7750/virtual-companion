"""Opt-in v10: ACK maintenance shares the turn queue, owner and Harness lease.

No fabricated user message, extra scheduler or implicit database upgrade. This
adapts Cyrene's post-turn social extraction to actual terminal presentation ACKs.
"""

import json
from dataclasses import asdict
from hashlib import sha256

from virtual_companion.text.identity import CompanionConfig

from .contracts import Scope, VersionConflict, canonical, companion_id, identifier
from .exchanges import ExchangeCommand, ExchangeEvidence, ExchangeSupport
from .sqlite_exchanges import SQLiteExchangeStore
from .sqlite_harness import HarnessJournal
from .sqlite_turns import _locked_time, _time
from .turns import RunClaim

_EXTRA = (
    "CREATE TABLE turn_triggers (companion TEXT NOT NULL, input TEXT NOT NULL, "
    "data TEXT NOT NULL, fingerprint TEXT, receipt TEXT, PRIMARY KEY(companion,input), "
    "FOREIGN KEY(companion,input) REFERENCES turn_inputs(companion,input))",
)


class SQLiteMaintenanceStore(SQLiteExchangeStore):
    schema_version = 10
    schema = SQLiteExchangeStore.schema + _EXTRA
    tables = SQLiteExchangeStore.tables | {"turn_triggers"}

    def __init__(self, path, *, evidence_clock=None, automatic_maintenance=False):
        if type(automatic_maintenance) is not bool:
            raise ValueError("Explicit automatic maintenance setting required")
        self.automatic_maintenance = automatic_maintenance
        super().__init__(path, evidence_clock=evidence_clock)

    def _ack_memories(self, db, companion, row, receipt):
        super()._ack_memories(db, companion, row, receipt)
        if (
            not self.automatic_maintenance
            or receipt["disposition"] != "applied"
            or receipt["status"] not in {"completed", "cancelled"}
            or receipt["displayed_end"] <= 0
            or self._later_user(db, companion, row["delivery"])
        ):
            return
        source = self._input(db, companion, row["input"])
        binding = json.loads(self._run_config(db, source))
        config = CompanionConfig.from_mapping(binding["companion_config"])
        if config.snapshot_hash != binding["companion_config_sha256"]:
            raise ValueError("Acknowledged source configuration binding mismatch")
        prefix = row["text"][: receipt["displayed_end"]]
        trigger = {
            "kind": "exchange_maintenance",
            "role": "assistant",
            "source_id": row["delivery"],
            "quote": prefix,
            "confirmed_end": len(prefix),
            "prefix_sha256": sha256(prefix.encode()).hexdigest(),
        }
        self._accept(
            config,
            "ack-" + row["delivery"],
            "",
            session_id=source["session"],
            trigger=trigger,
            db=db,
        )

    @staticmethod
    def _later_user(db, companion, delivery):
        return (
            db.execute(
                "SELECT 1 FROM turn_inputs i WHERE i.companion=? AND i.rowid>"
                "(SELECT s.rowid FROM turn_inputs s JOIN turn_deliveries d "
                "ON s.companion=d.companion AND s.input=d.input WHERE d.companion=? "
                "AND d.delivery=?) AND NOT EXISTS (SELECT 1 FROM turn_triggers t "
                "WHERE t.companion=i.companion AND t.input=i.input) LIMIT 1",
                (companion, companion, delivery),
            ).fetchone()
            is not None
        )

    def queued_maintenance(self, companion, *, limit=20):
        companion_id(companion)
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("Bounded maintenance queue read required")
        with self.transaction(write=False) as db:
            return [
                r[0]
                for r in db.execute(
                    "SELECT i.input FROM turn_inputs i JOIN turn_triggers t "
                    "ON i.companion=t.companion AND i.input=t.input "
                    "WHERE i.companion=? AND i.status='queued' ORDER BY i.rowid LIMIT ?",
                    (companion, limit),
                )
            ]

    def _upgrade(self, db, version, application, names):
        raise ValueError("Maintenance requires v10; existing databases are not migrated")

    def accept_maintenance(self, config, client_key, scope, evidence):
        if (
            not isinstance(config, CompanionConfig)
            or not isinstance(scope, Scope)
            or scope.companion_id != config.companion_id
            or not isinstance(evidence, ExchangeEvidence)
            or evidence.role != "assistant"
        ):
            raise ValueError("Scoped acknowledged assistant evidence required")
        identifier(client_key)
        return self._accept(
            config,
            client_key,
            "",
            session_id=scope.session_id,
            trigger={"kind": "exchange_maintenance", **asdict(evidence)},
        )

    def _accepted(self, db, receipt, trigger):
        if trigger is None:
            # A genuine new user request takes priority. Idempotent retries never
            # reach this hook and therefore cannot cancel unrelated later work.
            for statement in (
                "UPDATE turn_inputs SET status='cancelled' WHERE companion=? "
                "AND status IN ('queued','running') AND input IN "
                "(SELECT input FROM turn_triggers WHERE companion=?)",
                "UPDATE turn_runs SET status='cancelled' WHERE companion=? "
                "AND status='running' AND input IN "
                "(SELECT input FROM turn_triggers WHERE companion=?)",
            ):
                db.execute(statement, (receipt.companion_id, receipt.companion_id))
            return
        db.execute(
            "INSERT INTO turn_triggers VALUES(?,?,?,NULL,NULL)",
            (receipt.companion_id, receipt.input_id, canonical(trigger)),
        )
        self._validate_source(db, receipt.companion_id, receipt.session_id, trigger)
        if db.execute(
            "SELECT 1 FROM turn_triggers t JOIN turn_inputs i "
            "ON t.companion=i.companion AND t.input=i.input WHERE t.companion=? "
            "AND json_extract(t.data,'$.source_id')=? "
            "AND json_extract(t.data,'$.prefix_sha256')=? "
            "AND (t.receipt IS NOT NULL OR i.status IN ('queued','running')) AND t.input<>?",
            (
                receipt.companion_id,
                trigger["source_id"],
                trigger["prefix_sha256"],
                receipt.input_id,
            ),
        ).fetchone():
            raise VersionConflict("Acknowledged source already queued or maintained")

    def _validate_source(self, db, companion, session, trigger):
        if self._later_user(db, companion, trigger["source_id"]):
            raise ValueError("A subsequent user input requires context reconciliation")
        row = self._exchange_source(
            db, Scope(companion, session), "assistant", trigger["source_id"]
        )
        delivery = db.execute(
            "SELECT status FROM turn_deliveries WHERE companion=? AND delivery=?",
            (companion, trigger["source_id"]),
        ).fetchone()
        if (
            delivery[0] not in {"completed", "cancelled"}
            or row["confirmed"] <= 0
            or row["confirmed"] != trigger["confirmed_end"]
            or sha256(row["text"][: row["confirmed"]].encode()).hexdigest()
            != trigger["prefix_sha256"]
            or trigger["quote"] not in row["text"][: row["confirmed"]]
        ):
            raise ValueError("Maintenance requires the final actual acknowledged prefix")

    @staticmethod
    def _exchange_source(db, scope, role, source_id):
        if (
            role == "user"
            and db.execute(
                "SELECT 1 FROM turn_triggers WHERE companion=? AND input=?",
                (scope.companion_id, source_id),
            ).fetchone()
        ):
            raise ValueError("An internal trigger is not user evidence")
        return SQLiteExchangeStore._exchange_source(db, scope, role, source_id)

    def inspect(self, companion, input_id, *, timeout=5):
        result = super().inspect(companion, input_id, timeout=timeout)
        with self.transaction(write=False, timeout=timeout) as db:
            row = db.execute(
                "SELECT data,receipt FROM turn_triggers WHERE companion=? AND input=?",
                (companion, input_id),
            ).fetchone()
        result["trigger"] = json.loads(row[0]) if row else None
        result["maintenance"] = json.loads(row[1]) if row and row[1] else None
        return result

    def _completed_internal(self, db, companion, input_id):
        row = db.execute(
            "SELECT receipt FROM turn_triggers WHERE companion=? AND input=?",
            (companion, input_id),
        ).fetchone()
        return json.loads(row[0]) if row and row[0] else None

    def commit(self, result, *, now_us, clock=None):
        with self.transaction(write=False) as db:
            if db.execute(
                "SELECT 1 FROM turn_triggers WHERE companion=? AND input=?",
                (result.claim.companion_id, result.claim.input_id),
            ).fetchone():
                raise ValueError("Use maintenance completion for an internal trigger")
        return super().commit(result, now_us=now_us, clock=clock)

    def finish_maintenance(self, claim, proposals, *, now_us, clock=None):
        if (
            not isinstance(claim, RunClaim)
            or not isinstance(proposals, tuple)
            or len(proposals) > 16
        ):
            raise ValueError("Typed claim and bounded maintenance proposals required")
        _time(now_us)
        for pair in proposals:
            if (
                not isinstance(pair, tuple)
                or len(pair) != 2
                or not isinstance(pair[0], ExchangeCommand)
                or not isinstance(pair[1], ExchangeSupport)
            ):
                raise ValueError("Typed exchange command/support pairs required")
        if len({c.command_id for c, _ in proposals}) != len(proposals):
            raise ValueError("Duplicate maintenance command")
        fingerprint = sha256(
            canonical(
                {
                    "claim": asdict(claim),
                    "effects": [asdict(c) for c, _ in proposals],
                    "support": [s.to_mapping() for _, s in proposals],
                }
            ).encode()
        ).hexdigest()
        key = (claim.companion_id, claim.input_id)
        with self.transaction() as db:
            row = db.execute(
                "SELECT * FROM turn_triggers WHERE companion=? AND input=?", key
            ).fetchone()
            if row is None:
                raise ValueError("Unknown maintenance trigger")
            if row["receipt"] is not None:
                if row["fingerprint"] != fingerprint:
                    raise ValueError("Conflicting maintenance completion")
                return json.loads(row["receipt"])

            def current():
                return _locked_time(now_us, clock)

            live = HarnessJournal(self, claim, "maintenance-commit", current)
            live._live(db, now_us)
            if self._run_config(db, self._input(db, *key)) != claim.config_json:
                raise VersionConflict("Maintenance configuration differs from its claim")
            if self._snapshot(db, claim.companion_id).write_version != claim.state_version:
                raise VersionConflict("Maintenance basis has changed")
            trigger = json.loads(row["data"])
            self._validate_source(db, claim.companion_id, claim.session_id, trigger)
            batch = sha256(canonical([c.fingerprint for c, _ in proposals]).encode()).hexdigest()
            receipts = []
            for command, support in proposals:
                if (
                    command.scope != Scope(claim.companion_id, claim.session_id)
                    or command.evidence.role != "assistant"
                    or command.evidence.source_id != trigger["source_id"]
                    or command.evidence.confirmed_end != trigger["confirmed_end"]
                    or command.evidence.prefix_sha256 != trigger["prefix_sha256"]
                ):
                    raise ValueError("Maintenance effect is outside its actual source")
                if db.execute(
                    "SELECT 1 FROM exchange_versions WHERE companion=? AND command=?",
                    (claim.companion_id, command.command_id),
                ).fetchone():
                    raise VersionConflict("Maintenance effect belongs to another transaction")
                receipts.append(
                    self._apply_exchange(
                        db,
                        command,
                        claim.owner,
                        expected_version=claim.state_version + len(receipts),
                        now_us=self.evidence_clock(),
                        support=support,
                        batch_sha256=batch,
                    )
                )
            live._live(db, now_us)
            receipt = {
                "kind": trigger["kind"],
                "input_id": claim.input_id,
                "source_id": trigger["source_id"],
                "effects": receipts,
            }
            db.execute(
                "UPDATE turn_triggers SET fingerprint=?,receipt=? WHERE companion=? AND input=?",
                (fingerprint, canonical(receipt), *key),
            )
            db.execute("UPDATE turn_runs SET status='committed' WHERE companion=? AND input=?", key)
            db.execute(
                "UPDATE turn_inputs SET status='committed' WHERE companion=? AND input=?", key
            )
            return receipt
