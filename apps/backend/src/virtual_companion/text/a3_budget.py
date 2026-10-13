"""Local experiment ledger. Reservations survive failures and process restarts.

An exclusive OS file lock owns recovery; SQLite transactions serialize provider
threads. Interrupted sessions conservatively consume their full wall reservation.
This database belongs to an experiment, never to a companion's private state.
"""

import json
import math
import os
import sqlite3
from contextlib import contextmanager
from uuid import uuid4


class ExperimentLedger:
    def __init__(self, path, experiment, freeze_hash, limits, *, previous_freeze_hash=None):
        self.path = path
        self.identity = json.dumps([experiment, freeze_hash, limits], sort_keys=True)
        self.limits = limits
        self.freeze_hash = freeze_hash
        self.previous_freeze_hash = previous_freeze_hash
        for budget in (limits["total_budget"], limits["per_seed_session_budget"]):
            if any(
                type(budget[k]) is not int or budget[k] <= 0
                for k in ("max_calls", "max_output_tokens", "max_wall_seconds")
            ):
                raise ValueError("Positive explicit budget required")
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.with_suffix(".lock").open("a+b")
        if self.handle.tell() == 0:
            self.handle.write(b"0")
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.handle.close()
            self.handle = None
            raise RuntimeError("Experiment already owned") from error
        try:
            with self._connect() as db:
                db.executescript("""
                    CREATE TABLE IF NOT EXISTS identity(value TEXT NOT NULL);
                    CREATE TABLE IF NOT EXISTS freeze_history(
                        hash TEXT PRIMARY KEY, identity_json TEXT);
                    CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY, seed INTEGER,
                        wall REAL NOT NULL, status TEXT NOT NULL, freeze_hash TEXT);
                    CREATE TABLE IF NOT EXISTS calls(id TEXT PRIMARY KEY,
                        session TEXT, seed INTEGER,
                        route TEXT, case_id TEXT, attempt INTEGER, stage TEXT, reserved INTEGER,
                        status TEXT, metrics TEXT, UNIQUE(seed,route,case_id,attempt,stage));
                """)
                db.execute("BEGIN IMMEDIATE")
                old = db.execute("SELECT value FROM identity").fetchone()
                previous = json.loads(old[0]) if old is not None else None
                if "freeze_hash" not in {
                    row[1] for row in db.execute("PRAGMA table_info(sessions)")
                }:
                    db.execute("ALTER TABLE sessions ADD COLUMN freeze_hash TEXT")
                    db.execute("UPDATE sessions SET freeze_hash=?", (previous[1],))
                if old is None:
                    db.execute("INSERT INTO identity VALUES(?)", (self.identity,))
                elif old[0] != self.identity:
                    current = json.loads(self.identity)
                    if (
                        self.previous_freeze_hash != previous[1]
                        or current[0] != previous[0]
                        or current[1] == previous[1]
                        or current[2]["seeds"] != previous[2]["seeds"]
                    ):
                        raise ValueError("Experiment identity or limits changed")
                    for key in ("total_budget", "per_seed_session_budget"):
                        if any(
                            current[2][key][limit] > previous[2][key][limit]
                            for limit in ("max_calls", "max_output_tokens", "max_wall_seconds")
                        ):
                            raise ValueError("Freeze transition cannot increase budget")
                    db.execute(
                        "INSERT OR IGNORE INTO freeze_history VALUES(?,?)", (previous[1], old[0])
                    )
                    db.execute("UPDATE identity SET value=?", (self.identity,))
                db.execute(
                    "INSERT OR IGNORE INTO freeze_history VALUES(?,?)",
                    (self.freeze_hash, self.identity),
                )
                db.execute("UPDATE calls SET status='interrupted' WHERE status='reserved'")
                db.execute("UPDATE sessions SET status='interrupted' WHERE status='running'")
            return self
        except BaseException:
            self.__exit__()
            raise

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.close()  # OS releases the exclusive lock even after a crash.
            self.handle = None

    @contextmanager
    def _connect(self):
        if self.handle is None:
            raise RuntimeError("Ledger owner required")
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _totals(db, seed=None):
        row = db.execute(
            "SELECT COUNT(*),COALESCE(SUM(reserved),0) FROM calls WHERE (? IS NULL OR seed=?)",
            (seed, seed),
        ).fetchone()
        wall = db.execute(
            "SELECT COALESCE(SUM(wall),0) FROM sessions WHERE (? IS NULL OR seed=?)", (seed, seed)
        ).fetchone()[0]
        return {"calls": row[0], "output_tokens": row[1], "wall_seconds": float(wall)}

    def totals(self):
        with self._connect() as db:
            return self._totals(db)

    def start_session(self, seed, *, wall_cap=None):
        if seed not in self.limits["seeds"]:
            raise ValueError("Unfrozen seed")
        if wall_cap is not None and (
            type(wall_cap) not in {int, float} or not math.isfinite(wall_cap) or wall_cap <= 0
        ):
            raise ValueError("Positive finite pilot wall cap required")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM sessions WHERE status='running'").fetchone():
                raise ValueError("An experiment session is still running")
            remaining = min(
                self.limits["total_budget"]["max_wall_seconds"] - self._totals(db)["wall_seconds"],
                self.limits["per_seed_session_budget"]["max_wall_seconds"]
                - self._totals(db, seed)["wall_seconds"],
            )
            if remaining <= 0:
                raise ValueError("A3 wall budget exhausted")
            if wall_cap is not None:
                remaining = min(remaining, wall_cap)
            identity = uuid4().hex
            db.execute(
                "INSERT INTO sessions(id,seed,wall,status,freeze_hash) VALUES(?,?,?,'running',?)",
                (identity, seed, remaining, self.freeze_hash),
            )
            return identity

    def wall_remaining(self, session):
        with self._connect() as db:
            return db.execute("SELECT wall FROM sessions WHERE id=?", (session,)).fetchone()[0]

    def session_seed(self, session):
        with self._connect() as db:
            return db.execute("SELECT seed FROM sessions WHERE id=?", (session,)).fetchone()[0]

    def end_session(self, session, elapsed):
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Invalid elapsed time")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT wall,status FROM sessions WHERE id=?", (session,)).fetchone()
            if row is None or row["status"] != "running":
                raise ValueError("Session already terminal")
            db.execute(
                "UPDATE sessions SET wall=?,status='completed' WHERE id=?",
                (min(elapsed, row["wall"]), session),
            )

    def reserve(self, session, route, case, attempt, stage, tokens):
        if (
            route not in {"candidate", "control"}
            or type(attempt) is not int
            or attempt < 1
            or type(tokens) is not int
            or tokens <= 0
        ):
            raise ValueError("Explicit call identity and positive reservation required")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            owner = db.execute(
                "SELECT seed FROM sessions WHERE id=? AND status='running'", (session,)
            ).fetchone()
            if owner is None:
                raise ValueError("Active session required")
            seed = owner["seed"]
            if db.execute(
                "SELECT 1 FROM calls WHERE seed=? AND route=? AND case_id=? "
                "AND attempt=? AND stage=?",
                (seed, route, case, attempt, stage),
            ).fetchone():
                raise ValueError("A3 duplicate attempt; do not resend uncertain work")
            for scope, budget in (
                (None, self.limits["total_budget"]),
                (seed, self.limits["per_seed_session_budget"]),
            ):
                used = self._totals(db, scope)
                if (
                    used["calls"] + 1 > budget["max_calls"]
                    or used["output_tokens"] + tokens > budget["max_output_tokens"]
                ):
                    raise ValueError("A3 global or seed budget exhausted")
            identity = uuid4().hex
            db.execute(
                "INSERT INTO calls VALUES(?,?,?,?,?,?,?,?,?,?)",
                (identity, session, seed, route, case, attempt, stage, tokens, "reserved", "{}"),
            )
            return identity

    def finish(self, identity, status, metrics):
        if status not in {
            "completed",
            "failed",
            "cancelled",
            "empty",
            "deadline_exceeded",
            "protocol_invalid",
        }:
            raise ValueError("Terminal status required")
        with self._connect() as db:
            cursor = db.execute(
                "UPDATE calls SET status=?,metrics=? WHERE id=? AND status='reserved'",
                (status, json.dumps(metrics, ensure_ascii=False), identity),
            )
            if cursor.rowcount != 1:
                raise ValueError("Call already terminal or unknown")

    def records(self):
        with self._connect() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT calls.*,sessions.freeze_hash FROM calls JOIN sessions "
                    "ON calls.session=sessions.id ORDER BY calls.rowid"
                )
            ]
