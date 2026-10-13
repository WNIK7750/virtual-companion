"""Isolated v17 segment outbox pilot in the existing owner's SQLite database.

Trusted approval covers the entire immutable prefix. This adapter does not judge
semantic closure, project receipts into dialogue history, or enable the desktop.
Dispatch is at most once: a lost return/ACK remains unknown until a host receipt.
"""

import json
from dataclasses import asdict

from .contracts import VersionConflict, canonical, companion_id, identifier
from .history_readset import validate_readset
from .presentation import SPANS, ReviewedPresentation, digest, semantic_bubble_ranges
from .sqlite_background_worker import SQLiteResolvedBackgroundStore
from .sqlite_turns import _locked_time, _time
from .turns import RunClaim

_EXTRA = (
    "CREATE TABLE stream_segments (companion TEXT NOT NULL, input TEXT NOT NULL, "
    "sequence INTEGER NOT NULL CHECK(sequence BETWEEN 1 AND 32), "
    "fingerprint TEXT NOT NULL, manifest TEXT NOT NULL, "
    "status TEXT NOT NULL CHECK(status IN ('pending','unknown','confirmed')), "
    "host_receipt TEXT, PRIMARY KEY(companion,input,sequence), "
    "UNIQUE(companion,host_receipt), "
    "FOREIGN KEY(companion,input) REFERENCES turn_inputs(companion,input))",
)


def _sequence(value):
    if type(value) is not int or not 1 <= value <= 32:
        raise ValueError("Bounded segment sequence required")


def _basis(claim):
    if not isinstance(claim, RunClaim):
        raise ValueError("Typed run claim required")
    value = asdict(claim)
    del value["lease_until_us"]  # Renewals do not change an approved identity.
    return value


class SQLiteSegmentStore(SQLiteResolvedBackgroundStore):
    supports_history_readset = False
    schema_version = 17
    schema = SQLiteResolvedBackgroundStore.schema + _EXTRA
    tables = SQLiteResolvedBackgroundStore.tables | {"stream_segments"}

    def _upgrade(self, db, version, application, names):
        raise ValueError("Segment pilot requires a fresh v17 database; no migration")

    def commit(self, result, *, now_us, clock=None):
        # Until prefix projection/terminal reconciliation is connected, allowing
        # the old whole-turn commit could re-publish an already shown prefix.
        raise ValueError("Segment pilot cannot use whole-turn commit or fallback")

    def _segment_live(self, db, claim, now_us):
        _basis(claim)
        _time(now_us)
        item = self._input(db, claim.companion_id, claim.input_id)
        run = db.execute("SELECT * FROM turn_runs WHERE companion=? AND input=?",
                         (claim.companion_id, claim.input_id)).fetchone()
        owner = db.execute("SELECT owner,epoch FROM state_owners WHERE companion=?",
                           (claim.companion_id,)).fetchone()
        if (run is None or owner is None or item["status"] != "running"
                or run["status"] != "running" or run["lease_until"] <= now_us
                or item["session"] != claim.session_id
                or self._run_config(db, item) != claim.config_json
                or run["attempt"] != claim.attempt_id
                or run["basis"] != claim.state_version
                or (run["owner"], run["epoch"]) != (claim.owner.owner_id, claim.owner.epoch)
                or tuple(owner) != (claim.owner.owner_id, claim.owner.epoch)
                or self._snapshot(db, claim.companion_id).write_version != claim.state_version):
            raise VersionConflict("Segment owner, context, cancellation or lease changed")

    @staticmethod
    def _segment_rows(db, companion, input_id):
        companion_id(companion)
        identifier(input_id)
        return db.execute("SELECT * FROM stream_segments WHERE companion=? AND input=? "
                          "ORDER BY sequence", (companion, input_id)).fetchall()

    @staticmethod
    def _segment_event(row):
        manifest = json.loads(row["manifest"])
        if "readset" in manifest:
            validate_readset(manifest["readset"])
            from .public_capture import validate_public_scope

            validate_public_scope(manifest['readset'], manifest['basis'])
        bound = ReviewedPresentation.from_mapping(manifest["presentation"])
        candidate = bound.candidate
        if "bubbles" in manifest and (
                candidate.representation != SPANS
                or manifest["bubbles"] != semantic_bubble_ranges(candidate.text)):
            raise ValueError("Persisted semantic bubble ranges changed")
        if (digest(manifest) != row["fingerprint"]
                or manifest["sequence"] != row["sequence"]
                or len(candidate.units) != row["sequence"]
                or candidate.scope.companion_id != row["companion"]
                or candidate.input_id != row["input"]):
            raise ValueError("Persisted segment binding changed")
        return {**manifest, "fingerprint": row["fingerprint"], "status": row["status"],
                "host_receipt": row["host_receipt"], "text": candidate.units[-1].text,
                "display_text": candidate.text,
                "representation": candidate.representation}

    def approve_segment(self, claim, presentation, *, now_us, clock=None, readset=None,
                        public_authority=None):
        basis = _basis(claim)
        if not isinstance(presentation, ReviewedPresentation):
            raise ValueError("A trusted complete prefix approval is required")
        candidate = presentation.candidate
        sequence = len(candidate.units)
        _sequence(sequence)
        if (candidate.intent != "speak"
                or any(unit.kind != "text" for unit in candidate.units)
                or candidate.scope.companion_id != claim.companion_id
                or candidate.scope.session_id != claim.session_id
                or candidate.input_id != claim.input_id
                or candidate.attempt_id != claim.attempt_id):
            raise ValueError("Approved prefix does not belong to this text run")
        manifest = {"protocol": "approved-prefix.v1", "basis": basis,
                    "sequence": sequence, "presentation": presentation.to_mapping()}
        if candidate.representation == SPANS:
            manifest["bubbles"] = semantic_bubble_ranges(candidate.text)
        if readset is not None:
            # Detach caller state before hashing or entering a callback/transaction.
            manifest["readset"] = json.loads(canonical(validate_readset(readset)))
        fingerprint = digest(manifest)
        with self.transaction() as db:
            self._segment_live(db, claim, _locked_time(now_us, clock))
            if readset is not None:
                self._validate_segment_readset(db, claim, manifest["readset"],
                                               now_us=_locked_time(now_us, clock))
            rows = self._segment_rows(db, claim.companion_id, claim.input_id)
            if sequence <= len(rows):
                event = self._segment_event(rows[sequence - 1])
                if event["fingerprint"] != fingerprint:
                    raise ValueError("Conflicting segment retry")
                return event
            if sequence != len(rows) + 1:
                raise ValueError("Exactly one additional approved unit required")
            if rows:
                previous = self._segment_event(rows[-1])
                old = ReviewedPresentation.from_mapping(previous["presentation"])
                if (previous["basis"] != basis
                        or previous.get("readset") != manifest.get("readset")
                        or candidate.units[:-1] != old.candidate.units
                        or candidate.candidate_id != old.candidate.candidate_id
                        or candidate.revision != old.candidate.revision
                        or candidate.response_type != old.candidate.response_type
                        or presentation.policy != old.policy
                        or presentation.review.approval_source != old.review.approval_source
                        or presentation.review.policy_revision != old.review.policy_revision):
                    raise ValueError("Approved prefix or policy cannot be rewritten")
            from .public_capture import require_public_authority

            require_public_authority(manifest.get('readset'), public_authority)
            db.execute("INSERT INTO stream_segments VALUES(?,?,?,?,?,'pending',NULL)",
                       (claim.companion_id, claim.input_id, sequence, fingerprint,
                        canonical(manifest)))
            return self._segment_event(self._segment_rows(
                db, claim.companion_id, claim.input_id)[-1])

    def _validate_segment_readset(self, db, claim, readset, *, now_us):
        raise ValueError("This store cannot verify an effective history readset")

    def dispatch_segment(self, claim, sequence, *, now_us, clock=None, public_authority=None):
        _sequence(sequence)
        with self.transaction() as db:
            self._segment_live(db, claim, _locked_time(now_us, clock))
            rows = self._segment_rows(db, claim.companion_id, claim.input_id)
            if sequence > len(rows):
                raise ValueError("Segment has not been approved")
            row = rows[sequence - 1]
            event = self._segment_event(row)
            if event["basis"] != _basis(claim):
                raise VersionConflict("Segment belongs to another run")
            if row["status"] != "pending":
                return None  # Unknown must be reconciled, never automatically replayed.
            if "readset" in event:
                self._validate_segment_readset(db, claim, event["readset"],
                                               now_us=_locked_time(now_us, clock))
            if any(not self._segment_delivered(db, r) for r in rows[:sequence - 1]):
                raise ValueError("Predecessor requires a durable confirmation")
            from .public_capture import require_public_authority

            require_public_authority(event.get('readset'), public_authority)
            db.execute("UPDATE stream_segments SET status='unknown' "
                       "WHERE companion=? AND input=? AND sequence=?",
                       (claim.companion_id, claim.input_id, sequence))
            return {**event, "status": "unknown"}

    def _segment_delivered(self, db, row):
        return row['status'] == 'confirmed'

    def confirm_segment(self, companion, input_id, sequence, *, fingerprint,
                        host_receipt, channel):
        """Trusted host reports a complete text unit; not audio or a transport ACK.

        Cancellation does not erase an actual earlier display. Identity includes
        the original run epoch; a late receipt cannot confirm another attempt.
        """
        _sequence(sequence)
        identifier(host_receipt)
        if channel != "text_visible":
            raise ValueError("Text visibility cannot attest audio playback")
        with self.transaction() as db:
            rows = self._segment_rows(db, companion, input_id)
            if sequence > len(rows):
                raise ValueError("Unknown segment receipt")
            row = rows[sequence - 1]
            event = self._segment_event(row)
            if fingerprint != event["fingerprint"] or row["status"] == "pending":
                raise ValueError("Receipt does not match a dispatched approved segment")
            if row["status"] == "confirmed":
                if host_receipt != row["host_receipt"]:
                    raise ValueError("Conflicting host receipt")
                return event
            db.execute("UPDATE stream_segments SET status='confirmed',host_receipt=? "
                       "WHERE companion=? AND input=? AND sequence=?",
                       (host_receipt, companion, input_id, sequence))
            confirmed = {**event, "status": "confirmed", "host_receipt": host_receipt}
            self._segment_confirmed(db, confirmed)
            return confirmed

    def _segment_confirmed(self, db, event):
        """v17 has no history projection; successors share this exact transaction."""

    def segments(self, companion, input_id):
        with self.transaction(write=False) as db:
            return [self._segment_event(row)
                    for row in self._segment_rows(db, companion, input_id)]
