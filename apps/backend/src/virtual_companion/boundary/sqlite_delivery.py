"""Delivery operations composed into the owning SQLite turn UnitOfWork.

This adapter mixin never opens a second domain connection. Publication records
are durable authorizations; actual host acknowledgement is a separate fact.
"""

import json

from .contracts import canonical, companion_id, identifier
from .presentation import verify_delivery


class DeliveryOperations:
    def _ack_memories(self, db, companion, row, receipt):
        pass

    @staticmethod
    def _delivery(db, companion, delivery_id):
        companion_id(companion)
        identifier(delivery_id)
        row = db.execute(
            "SELECT * FROM turn_deliveries WHERE companion=? AND delivery=?",
            (companion, delivery_id),
        ).fetchone()
        if row is None:
            raise ValueError("Unknown delivery in companion scope")
        return row

    @staticmethod
    def _display(row):
        return {
            "delivery_id": row["delivery"],
            "status": row["status"],
            "displayed_end": row["displayed_end"],
            "displayed_text": row["text"][: row["displayed_end"]],
            "sequence": row["sequence"],
        }

    def delivery(self, companion, delivery_id):
        with self.transaction(write=False) as db:
            return self._display(self._delivery(db, companion, delivery_id))

    def publish(self, companion, delivery_id):
        """Persist publication authorization, not a claim that a device rendered text.

        Only the winning transition returns a new event. Recovery may retrieve the
        persisted event with publication(); a host must deduplicate its delivery ID.
        """
        with self.transaction() as db:
            row = self._delivery(db, companion, delivery_id)
            changed = db.execute(
                "UPDATE turn_deliveries SET status='published' "
                "WHERE companion=? AND delivery=? AND status='pending'",
                (companion, delivery_id),
            ).rowcount
            if changed != 1:
                return None
            commit = json.loads(
                db.execute(
                    "SELECT receipt FROM turn_commits WHERE companion=? AND input=?",
                    (companion, row["input"]),
                ).fetchone()[0]
            )
            verify_delivery(row["text"], commit)
            event = {**commit, "text": row["text"], "delivery_status": "published"}
            db.execute(
                "INSERT INTO turn_publications VALUES(?,?,?)",
                (companion, delivery_id, canonical(event)),
            )
            return event

    def publication(self, companion, delivery_id):
        with self.transaction(write=False) as db:
            delivery = self._delivery(db, companion, delivery_id)
            row = db.execute(
                "SELECT event FROM turn_publications WHERE companion=? AND delivery=?",
                (companion, delivery_id),
            ).fetchone()
            if not row:
                return None
            event = json.loads(row[0])
            receipt = json.loads(
                db.execute(
                    "SELECT receipt FROM turn_commits WHERE companion=? AND input=?",
                    (companion, delivery["input"]),
                ).fetchone()[0]
            )
            verify_delivery(delivery["text"], receipt)
            if event != {
                **receipt,
                "text": delivery["text"],
                "delivery_status": "published",
            }:
                raise ValueError("Publication no longer matches committed presentation")
            return event

    def acknowledge(
        self,
        companion,
        delivery_id,
        *,
        candidate_id,
        revision,
        sha256,
        sequence,
        displayed_end,
        stopped=False,
        presentation_sha256=None,
    ):
        identifier(candidate_id)
        if (
            type(sequence) is not int
            or sequence < 1
            or sequence >= 2**63
            or type(displayed_end) is not int
            or displayed_end < 0
            or type(revision) is not int
            or revision < 1
            or type(stopped) is not bool
        ):
            raise ValueError("Invalid presentation acknowledgement")
        with self.transaction() as db:
            row = self._delivery(db, companion, delivery_id)
            commit = json.loads(
                db.execute(
                    "SELECT receipt FROM turn_commits WHERE companion=? AND input=?",
                    (companion, row["input"]),
                ).fetchone()[0]
            )
            if (candidate_id, revision, sha256) != (
                commit["candidate_id"],
                commit["revision"],
                commit["sha256"],
            ) or displayed_end > len(row["text"]):
                raise ValueError("Acknowledgement candidate or codepoint range mismatch")
            verify_delivery(row["text"], commit)
            if presentation_sha256 != commit.get("presentation_sha256"):
                raise ValueError("Acknowledgement presentation manifest mismatch")
            fingerprint = canonical([candidate_id, revision, sha256, displayed_end, stopped])
            key = (companion, delivery_id, sequence)
            original = db.execute(
                "SELECT fingerprint,receipt FROM turn_acks "
                "WHERE companion=? AND delivery=? AND sequence=?",
                key,
            ).fetchone()
            if original:
                if original["fingerprint"] != fingerprint:
                    raise ValueError("Conflicting acknowledgement identity")
                return json.loads(original["receipt"])
            if sequence < row["sequence"]:
                receipt = {**self._display(row), "disposition": "late"}
            else:
                if (
                    row["status"] not in {"published", "stopping"}
                    or displayed_end < row["displayed_end"]
                ):
                    raise ValueError(
                        "Acknowledgement cannot expand terminal or rewind displayed range"
                    )
                if stopped and row["status"] != "stopping":
                    raise ValueError("Stop receipt requires a stop request")
                status = (
                    "completed"
                    if displayed_end == len(row["text"])
                    else ("cancelled" if stopped else row["status"])
                )
                db.execute(
                    "UPDATE turn_deliveries SET status=?,displayed_end=?,sequence=? "
                    "WHERE companion=? AND delivery=?",
                    (status, displayed_end, sequence, companion, delivery_id),
                )
                receipt = {
                    **self._display(self._delivery(db, companion, delivery_id)),
                    "disposition": "applied",
                }
            db.execute(
                "INSERT INTO turn_acks VALUES(?,?,?,?,?)", (*key, fingerprint, canonical(receipt))
            )
            self._ack_memories(db, companion, row, receipt)
            return receipt
