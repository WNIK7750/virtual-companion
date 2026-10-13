"""Opt-in v9 shared-owner exchange history; no private/older store migration.

Adaptation of Cyrene's open_loop/resolve semantics (1ce59f2, MIT) adds the
separate assistant-response direction, ACK proof, revisions and explicit closure.
There is no automatic scheduling, model extraction or second state owner here.
"""

import json
import time
from dataclasses import asdict, replace
from hashlib import sha256
from heapq import nlargest
from itertools import islice

from .contracts import OwnerToken, Scope, VersionConflict, canonical, companion_id, identifier
from .evidence import EvidenceTurn, evidence_digest
from .exchanges import ExchangeCommand, ExchangeSupport
from .sqlite_harness import SQLiteHarnessStore
from .sqlite_turns import _time

_EXTRA = (
    "CREATE TABLE exchange_versions (companion TEXT NOT NULL "
    "REFERENCES companion_state(companion), exchange TEXT NOT NULL, revision INTEGER NOT NULL, "
    "command TEXT NOT NULL, fingerprint TEXT NOT NULL, data TEXT NOT NULL, receipt TEXT NOT NULL, "
    "PRIMARY KEY(companion,exchange,revision), UNIQUE(companion,command))",
)


class SQLiteExchangeStore(SQLiteHarnessStore):
    schema_version = 9
    schema = SQLiteHarnessStore.schema + _EXTRA
    tables = SQLiteHarnessStore.tables | {"exchange_versions"}

    def __init__(self, path, *, evidence_clock=None):
        if evidence_clock is not None and not callable(evidence_clock):
            raise ValueError("Trusted UTC evidence clock required")
        self.evidence_clock = evidence_clock or (lambda: time.time_ns() // 1000)
        super().__init__(path)

    def _commit_exchanges(self, db, result, receipt):
        context_sha256 = self._verify_exchange_context(db, result)
        if not result.exchanges:
            return
        batch = sha256(canonical([c.fingerprint for c, _ in result.exchanges]).encode()).hexdigest()
        for command, support in result.exchanges:
            original = db.execute(
                "SELECT 1 FROM exchange_versions WHERE companion=? AND command=?",
                (result.claim.companion_id, command.command_id),
            ).fetchone()
            if original:
                raise VersionConflict("Exchange effect already belongs to another commit")
            self._apply_exchange(
                db,
                command,
                result.claim.owner,
                expected_version=self._snapshot(db, result.claim.companion_id).write_version,
                now_us=self.evidence_clock(),
                support=support,
                batch_sha256=batch,
                context_sha256=context_sha256,
            )

    def _verify_exchange_context(self, db, result):
        if not result.exchange_sources:
            return None
        if self.schema_version < 11:
            raise ValueError("Contextual exchanges require the opt-in conversation adapter")
        anchor = db.execute(
            "SELECT rowid FROM turn_inputs WHERE companion=? AND input=?",
            (result.claim.companion_id, result.claim.input_id),
        ).fetchone()[0]
        previous = (-1, -1)
        for source in result.exchange_sources:
            row = self._exchange_source(db, source.scope, source.role, source.source_id)
            end = len(row["text"]) if source.role == "user" else row["confirmed"]
            position = (row["ordinal"], int(source.role == "assistant"))
            if (
                source.confirmed_end != end
                or source.text != row["text"][:end]
                or position <= previous
                or position > (anchor, 0)
            ):
                raise ValueError("Exchange context differs from actual ordered confirmed sources")
            previous = position
        return evidence_digest(result.exchange_sources)

    def _upgrade(self, db, version, application, names):
        raise ValueError("Exchanges require v9; existing databases are not migrated")

    def exchange_sources(self, scope, *, references):
        """Return actual input or ACK prefix, never an unshown candidate tail."""
        if (
            not isinstance(scope, Scope)
            or not isinstance(references, tuple)
            or len(references) > 32
        ):
            raise ValueError("Typed scope and bounded source references required")
        seen = set()
        for reference in references:
            if (
                not isinstance(reference, tuple)
                or len(reference) != 2
                or reference[0] not in {"user", "assistant"}
            ):
                raise ValueError("Explicit source role and identity required")
            identifier(reference[1])
            if reference[1] in seen:
                raise ValueError("Duplicate evidence identity")
            seen.add(reference[1])
        with self.transaction(write=False) as db:
            sources = []
            for role, source_id in references:
                row = self._exchange_source(db, scope, role, source_id)
                end = len(row["text"]) if role == "user" else row["confirmed"]
                if end:
                    sources.append(EvidenceTurn(scope, source_id, role, row["text"][:end], end,
                        row["response_type"] if "response_type" in row.keys() else None))
            return tuple(sources)

    @staticmethod
    def _exchange_head(db, companion, key):
        row = db.execute(
            "SELECT data FROM exchange_versions WHERE companion=? AND exchange=? "
            "ORDER BY revision DESC LIMIT 1",
            (companion, key),
        ).fetchone()
        return json.loads(row[0]) if row else None

    @staticmethod
    def _exchange_source(db, scope, role, source_id):
        if role == "user":
            row = db.execute(
                "SELECT text,session,length(text) AS confirmed,rowid AS ordinal FROM turn_inputs "
                "WHERE companion=? AND input=?",
                (scope.companion_id, source_id),
            ).fetchone()
        else:
            row = db.execute(
                "SELECT d.text,i.session,d.displayed_end AS confirmed,i.rowid AS ordinal "
                "FROM turn_deliveries d "
                "JOIN turn_inputs i ON d.companion=i.companion AND d.input=i.input "
                "WHERE d.companion=? AND d.delivery=?",
                (scope.companion_id, source_id),
            ).fetchone()
        if row is None or row["session"] != scope.session_id:
            raise ValueError("Unknown or foreign exchange source")
        return row

    def _verify_exchange_source(self, db, command):
        evidence = command.evidence
        row = self._exchange_source(db, command.scope, evidence.role, evidence.source_id)
        confirmed = len(row["text"]) if evidence.role == "user" else row["confirmed"]
        if evidence.confirmed_end > confirmed:
            raise ValueError("Exchange source has not been confirmed")
        if evidence.confirmed_end != confirmed:
            raise ValueError("Exchange judgment must cover the current confirmed source")
        prefix = row["text"][: evidence.confirmed_end]
        if (
            evidence.quote not in prefix
            or sha256(prefix.encode()).hexdigest() != evidence.prefix_sha256
        ):
            raise ValueError("Exchange source quote or digest mismatch")
        return EvidenceTurn(command.scope, evidence.source_id, evidence.role, prefix, confirmed,
                            row["response_type"] if "response_type" in row.keys() else None)

    def exchange(self, command, owner, *, expected_version, now_us, support):
        """support is issued only by the caller's trusted semantic evaluator.

        It binds the entire action, target revision, quote and confirmed source.
        No model-supplied boolean can bypass the independent source checks below.
        """
        with self.transaction() as db:
            return self._apply_exchange(
                db,
                command,
                owner,
                expected_version=expected_version,
                now_us=now_us,
                support=support,
            )

    def _apply_exchange(
        self,
        db,
        command,
        owner,
        *,
        expected_version,
        now_us,
        support,
        batch_sha256=None,
        context_sha256=None,
    ):
        _time(now_us)
        if not isinstance(command, ExchangeCommand) or not isinstance(owner, OwnerToken):
            raise ValueError("Typed exchange command and owner required")
        if owner.companion_id != command.scope.companion_id:
            raise ValueError("Exchange owner scope mismatch")
        if type(expected_version) is not int or expected_version < 0:
            raise ValueError("Companion state version required")
        if (
            not isinstance(support, ExchangeSupport)
            or support.command_sha256 != command.fingerprint
        ):
            raise ValueError("Missing or mismatched semantic support")
        companion = command.scope.companion_id
        writer = db.execute(
            "SELECT owner,epoch FROM state_owners WHERE companion=?",
            (companion,),
        ).fetchone()
        if writer is None or tuple(writer) != (owner.owner_id, owner.epoch):
            raise VersionConflict("Stale exchange owner")
        original = db.execute(
            "SELECT fingerprint,receipt FROM exchange_versions WHERE companion=? AND command=?",
            (companion, command.command_id),
        ).fetchone()
        if original:
            if original["fingerprint"] != command.fingerprint:
                raise VersionConflict("Conflicting exchange replay")
            receipt = json.loads(original["receipt"])
            if receipt.get("batch_sha256") != batch_sha256:
                raise VersionConflict("Conflicting or partial exchange batch replay")
            return receipt
        basis = self._snapshot(db, companion)
        if basis.write_version != expected_version:
            raise VersionConflict("Stale exchange state")
        if command.occurred_us > now_us:
            raise ValueError("Exchange occurrence is in the future")
        source = self._verify_exchange_source(db, command)
        if (source.response_type == "ack" and command.direction == "assistant_response"
                and command.action in {"resolve", "close"}):
            raise ValueError("Acknowledgement cannot settle a response obligation")
        if support.context_sha256 is not None and support.context_sha256 != (
            context_sha256 or evidence_digest((source,))
        ):
            raise ValueError("Semantic support belongs to another source context")
        previous = self._exchange_head(db, companion, command.exchange_id)
        if previous is not None and not self._exchange_precedes_source(
            db, previous, command.scope, command.evidence.role, command.evidence.source_id
        ):
            raise ValueError("Exchange target originates after its responding source")
        data = self._evolve_exchange(command, previous, now_us)
        data["support"] = support.to_mapping()
        state = replace(basis, write_version=basis.write_version + 1)
        receipt = {
            "exchange_id": command.exchange_id,
            "revision": data["revision"],
            "status": data["status"],
            "write_version": state.write_version,
        }
        if batch_sha256 is not None:
            receipt["batch_sha256"] = batch_sha256
        db.execute(
            "INSERT INTO exchange_versions VALUES(?,?,?,?,?,?,?)",
            (
                companion,
                command.exchange_id,
                data["revision"],
                command.command_id,
                command.fingerprint,
                canonical(data),
                canonical(receipt),
            ),
        )
        if (
            db.execute(
                "UPDATE companion_state SET version=?,snapshot=? WHERE companion=? AND version=?",
                (state.write_version, canonical(asdict(state)), companion, basis.write_version),
            ).rowcount
            != 1
        ):
            raise VersionConflict("Exchange state CAS lost")
        return receipt

    @staticmethod
    def _evolve_exchange(command, previous, now_us):
        if command.action == "open":
            if previous is not None:
                raise VersionConflict("Exchange identity already exists")
            expires = command.expires_us
        else:
            if (
                previous is None
                or previous["revision"] != command.expected_revision
                or previous["status"] not in {"pending", "deferred"}
                or previous["direction"] != command.direction
                or previous["content"] != command.content
                or previous["expires_us"] != command.expires_us
                or previous["occurred_us"] > command.occurred_us
            ):
                raise VersionConflict("Stale or altered exchange target")
            expires = previous["expires_us"]
        if expires is not None and expires <= now_us:
            raise VersionConflict("Expired exchange cannot be revived")
        if command.action == "resolve":
            responding = "user" if command.direction == "user_question" else "assistant"
            if command.evidence.role != responding:
                raise ValueError("Resolution needs the responding party's evidence")
        if command.action == "resume" and previous["status"] != "deferred":
            raise VersionConflict("Only a deferred exchange can resume")
        return {
            **asdict(command),
            "companion_id": command.scope.companion_id,
            "revision": previous["revision"] + 1 if previous else 1,
            "initiator": previous["initiator"] if previous else command.evidence.role,
            "status": {
                "open": "pending",
                "resume": "pending",
                "defer": "deferred",
                "resolve": "resolved",
                "close": "closed",
            }[command.action],
        }

    def exchange_batch(self, proposals, owner, *, expected_version, now_us):
        """One extraction commits wholly; retries bind the original ordered batch."""
        _time(now_us)
        if (
            not isinstance(proposals, tuple)
            or len(proposals) > 16
            or not isinstance(owner, OwnerToken)
            or type(expected_version) is not int
            or expected_version < 0
        ):
            raise ValueError("Typed bounded exchange batch and owner required")
        if any(
            not isinstance(p, tuple)
            or len(p) != 2
            or not isinstance(p[0], ExchangeCommand)
            or not isinstance(p[1], ExchangeSupport)
            for p in proposals
        ):
            raise ValueError("Typed exchange proposals and support required")
        commands = [p[0] for p in proposals]
        if len({c.command_id for c in commands}) != len(commands):
            raise ValueError("Duplicate exchange batch command")
        batch = sha256(canonical([c.fingerprint for c in commands]).encode()).hexdigest()
        with self.transaction() as db:
            receipts = []
            for command, support in proposals:
                receipt = self._apply_exchange(
                    db,
                    command,
                    owner,
                    expected_version=expected_version + len(receipts),
                    now_us=now_us,
                    support=support,
                    batch_sha256=batch,
                )
                receipts.append(receipt)
            return receipts

    def _eligible_exchange_targets(self, db, companion, *, now_us):
        """One ordered live catalog for reads and source-resolution receipts."""
        rows = db.execute(
            "SELECT v.data FROM exchange_versions v WHERE companion=? AND revision="
            "(SELECT max(h.revision) FROM exchange_versions h WHERE "
            "h.companion=v.companion AND h.exchange=v.exchange) "
            "AND json_extract(v.data,'$.status') IN ('pending','deferred') "
            "AND (json_extract(v.data,'$.expires_us') IS NULL OR "
            "json_extract(v.data,'$.expires_us')>?) ORDER BY exchange", (companion, now_us),
        )
        return (json.loads(row[0]) for row in rows)

    def _exchange_precedes_source(self, db, target, scope, role, source_id):
        """Use conversation order, never asynchronous processing timestamps."""
        def position(item_scope, item_role, item_source):
            row = self._exchange_source(db, item_scope, item_role, item_source)
            sequence = row["sequence"] if "sequence" in row.keys() else 0
            return row["ordinal"], int(item_role == "assistant"), sequence

        current = position(scope, role, source_id)
        origin = db.execute(
            "SELECT data FROM exchange_versions WHERE companion=? AND exchange=? "
            "AND revision=1", (scope.companion_id, target["exchange_id"]),
        ).fetchone()
        if origin is None:
            raise ValueError("Exchange origin is missing")
        # Check both original request and current revision: older persisted
        # revisions may predate this guard, and transitions must not run backwards.
        for value in (json.loads(origin[0]), target):
            evidence = value["evidence"]
            target_scope = Scope(**value["scope"])
            if target_scope.companion_id != scope.companion_id or position(
                target_scope, evidence["role"], evidence["source_id"]
            ) > current:
                return False
        return True

    def ranked_pending_exchanges(self, companion, *, now_us, score, limit=8,
                                 before_source=None):
        """Select cues from all live heads; ranking never resolves an obligation.

        Reuse the caller's lexical baseline. No match keeps recent cues rather
        than treating a vague continuation as proof that no obligation matters.
        """
        companion_id(companion)
        _time(now_us)
        if not callable(score) or type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("Bounded exchange limit and relevance scorer required")
        with self.transaction(write=False) as db:
            def scored():
                for value in self._eligible_exchange_targets(db, companion, now_us=now_us):
                    if before_source is not None and not self._exchange_precedes_source(
                        db, value, *before_source
                    ):
                        continue
                    yield score(value), value["occurred_us"], value["exchange_id"], value

            return [item[3] for item in nlargest(limit, scored(), key=lambda item: item[:3])]

    def pending_exchanges(self, companion, *, now_us, limit=100, candidates=None):
        companion_id(companion)
        _time(now_us)
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("Bounded pending exchange limit required")
        if candidates is not None:
            if not isinstance(candidates, tuple) or len(candidates) > 100:
                raise ValueError("Bounded exchange references required")
            for reference in candidates:
                if (
                    not isinstance(reference, tuple)
                    or len(reference) != 2
                    or type(reference[1]) is not int
                    or reference[1] < 1
                ):
                    raise ValueError("Typed exchange identity/revision required")
                identifier(reference[0])
        with self.transaction(write=False) as db:
            references = None if candidates is None else set(candidates)
            values = (value for value in self._eligible_exchange_targets(
                db, companion, now_us=now_us) if references is None
                or (value["exchange_id"], value["revision"]) in references)
            return list(islice(values, limit))

    def exchange_history(self, companion, exchange_id):
        companion_id(companion)
        identifier(exchange_id)
        with self.transaction(write=False) as db:
            return [
                json.loads(row[0])
                for row in db.execute(
                    "SELECT data FROM exchange_versions WHERE companion=? AND exchange=? "
                    "ORDER BY revision",
                    (companion, exchange_id),
                )
            ]
