from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from ..boundary.presentation import semantic_bubble_ranges
from .history import PAGE_SIZE, history_page, validate_window
from .identity import _identity


class SQLiteConversationStore:
    def __init__(self, root: Path, *, companion_id: str | None = None):
        # None is historical unbound mode, never an ownership bypass.
        if companion_id is not None:
            _identity(companion_id)
        self.companion_id = companion_id
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.connection = sqlite3.connect(
            self.root / "conversations.sqlite3", isolation_level=None, check_same_thread=False
        )
        self.connection.row_factory = sqlite3.Row
        try:
            self._initialize()
        except BaseException:
            self.connection.close()
            raise

    def _check_companion(self):
        tables = {row[0] for row in self.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        row = self.connection.execute(
            "SELECT value FROM metadata WHERE key='companion_id'"
        ).fetchone() if "metadata" in tables else None
        if row is not None:
            if row[0] != self.companion_id:
                raise ValueError("Conversation companion identity mismatch")
            return
        if self.companion_id is not None:
            for table, query in (
                ("turns", "SELECT 1 FROM turns LIMIT 1"),
                ("attempts", "SELECT 1 FROM attempts LIMIT 1"),
                ("messages", "SELECT 1 FROM messages LIMIT 1"),
            ):
                if table in tables and self.connection.execute(query).fetchone() is not None:
                    raise ValueError("Unbound conversation history requires explicit ownership")

    def _initialize(self):
        version = self.connection.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, 1):
            raise ValueError("Unsupported conversation schema")
        # Reject existing ownership before WAL/schema/recovery can mutate it.
        self._check_companion()
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA synchronous=FULL")
        schema = """
            CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS turns(
                id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id));
            CREATE TABLE IF NOT EXISTS attempts(
                seq INTEGER PRIMARY KEY, id TEXT NOT NULL UNIQUE,
                turn_id TEXT NOT NULL REFERENCES turns(id), status TEXT NOT NULL,
                binding TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS messages(
                seq INTEGER PRIMARY KEY, id TEXT NOT NULL UNIQUE,
                turn_id TEXT NOT NULL REFERENCES turns(id), role TEXT NOT NULL,
                content TEXT NOT NULL, UNIQUE(turn_id, role));
            PRAGMA user_version=1;
        """
        with self.transaction():
            # A competing opener may have bound the previously empty root.
            self._check_companion()
            for statement in schema.split(";"):
                if statement.strip():
                    self.connection.execute(statement)
            if self.companion_id is not None:
                self.connection.execute(
                    "INSERT OR IGNORE INTO metadata(key,value) VALUES('companion_id',?)",
                    (self.companion_id,),
                )
            self.connection.execute(
                "UPDATE attempts SET status='interrupted' WHERE status='running'"
            )
            row = self.connection.execute(
                "SELECT value FROM metadata WHERE key='active'"
            ).fetchone()
            if row:
                self.session_id = row[0]
            else:
                self.session_id = self._new()
        self.connection.execute("PRAGMA journal_mode=WAL")

    @contextmanager
    def transaction(self):
        with self.lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                yield
                self.connection.execute("COMMIT")
            except BaseException:
                self.connection.execute("ROLLBACK")
                raise

    def _new(self):
        session_id = str(uuid4())
        self.connection.execute("INSERT INTO sessions VALUES(?)", (session_id,))
        self.connection.execute("INSERT OR REPLACE INTO metadata VALUES('active',?)", (session_id,))
        return session_id

    def new_session(self):
        with self.transaction():
            session_id = self._new()
        self.session_id = session_id
        return self.history()

    def history(self):
        with self.lock:
            rows = self.connection.execute(
                """SELECT m.id,m.turn_id,m.role,m.content,
                COALESCE(k.value,'answer') AS response_type,b.value AS bubble_ranges,
                o.value AS bubble_orders,CASE WHEN EXISTS(SELECT 1 FROM metadata
                WHERE key='semantic-timeline') THEN m.seq END AS timeline_anchor FROM messages m
                JOIN turns t ON t.id=m.turn_id
                LEFT JOIN metadata k ON k.key='response-type:'||m.id
                LEFT JOIN metadata b ON b.key='bubble-ranges:'||m.id
                LEFT JOIN metadata o ON o.key='bubble-timeline:'||m.turn_id
                WHERE t.session_id=? ORDER BY m.seq""",
                (self.session_id,),
            ).fetchall()
            return {"session_id": self.session_id, "messages": [self._history_message(row)
                                                                for row in rows]}

    @staticmethod
    def _history_message(row):
        value = dict(row)
        bubbles = value.pop("bubble_ranges")
        orders = value.pop("bubble_orders")
        anchor = value.pop("timeline_anchor")
        if anchor is not None:
            value['timeline_order'] = [anchor, 0]
        if bubbles is not None:
            ranges = json.loads(bubbles)
            if value["role"] != "assistant" or ranges != semantic_bubble_ranges(value["content"]):
                raise ValueError("Invalid persisted semantic bubbles")
            value["bubble_ranges"] = ranges
            if orders is not None:
                by_start = {item['start']: item for item in json.loads(orders)}
                value['bubble_orders'] = [by_start[item['start']] for item in ranges]
        if value["role"] == "user" or value["response_type"] == "answer":
            del value["response_type"]
        elif value["response_type"] not in {"answer", "ack"}:
            raise ValueError("Unknown persisted assistant response type")
        return value

    def history_page(self, *, before=None, after=None, session_id=None):
        validate_window(before=before, after=after, session_id=session_id)
        with self.lock:
            if session_id is not None and session_id != self.session_id:
                raise ValueError("Active session changed")
            cursor = before or after
            boundary = 0 if after else 2**63 - 1
            if cursor:
                row = self.connection.execute(
                    """SELECT m.seq FROM messages m JOIN turns t ON t.id=m.turn_id
                    WHERE m.id=? AND t.session_id=?""",
                    (cursor, self.session_id),
                ).fetchone()
                if row is None:
                    raise ValueError("Unknown history cursor")
                boundary = row[0]
            query = (
                """SELECT m.id,m.turn_id,m.role,m.content,
                COALESCE(k.value,'answer') AS response_type,b.value AS bubble_ranges,
                o.value AS bubble_orders,CASE WHEN EXISTS(SELECT 1 FROM metadata
                WHERE key='semantic-timeline') THEN m.seq END AS timeline_anchor FROM messages m
                JOIN turns t ON t.id=m.turn_id
                LEFT JOIN metadata k ON k.key='response-type:'||m.id
                LEFT JOIN metadata b ON b.key='bubble-ranges:'||m.id
                LEFT JOIN metadata o ON o.key='bubble-timeline:'||m.turn_id
                WHERE t.session_id=? AND m.seq>?
                ORDER BY m.seq ASC LIMIT ?"""
                if after
                else """SELECT m.id,m.turn_id,m.role,m.content,
                COALESCE(k.value,'answer') AS response_type,b.value AS bubble_ranges,
                o.value AS bubble_orders,CASE WHEN EXISTS(SELECT 1 FROM metadata
                WHERE key='semantic-timeline') THEN m.seq END AS timeline_anchor FROM messages m
                JOIN turns t ON t.id=m.turn_id
                LEFT JOIN metadata k ON k.key='response-type:'||m.id
                LEFT JOIN metadata b ON b.key='bubble-ranges:'||m.id
                LEFT JOIN metadata o ON o.key='bubble-timeline:'||m.turn_id
                WHERE t.session_id=? AND m.seq<?
                ORDER BY m.seq DESC LIMIT ?"""
            )
            rows = self.connection.execute(
                query, (self.session_id, boundary, PAGE_SIZE + 1)
            ).fetchall()
            return history_page([self._history_message(row) for row in rows], self.session_id,
                                forward=bool(after))

    def sessions(self, *, offset=0, limit=100):
        with self.lock:
            rows = self.connection.execute(
                """SELECT s.id,COALESCE((SELECT substr(m.content,1,36)
                FROM messages m JOIN turns t ON t.id=m.turn_id WHERE t.session_id=s.id
                AND m.role='user' ORDER BY m.seq LIMIT 1),'新的对话') AS title
                FROM sessions s ORDER BY s.rowid DESC LIMIT ? OFFSET ?""",
                (limit, offset),
            ).fetchall()
            return [dict(row) for row in rows]

    def select_session(self, session_id):
        with self.transaction():
            if not self.connection.execute(
                "SELECT 1 FROM sessions WHERE id=?", (session_id,)
            ).fetchone():
                raise ValueError("Unknown session")
            self.connection.execute("UPDATE metadata SET value=? WHERE key='active'", (session_id,))
        self.session_id = session_id
        return self.history_page()

    def latest_attempt(self):
        with self.lock:
            row = self.connection.execute(
                """SELECT a.id AS attempt_id,a.turn_id,a.status,a.binding
                FROM attempts a JOIN turns t ON t.id=a.turn_id
                WHERE t.session_id=? ORDER BY a.seq DESC LIMIT 1""",
                (self.session_id,),
            ).fetchone()
            if not row:
                return None
            return {**dict(row), "binding": json.loads(row["binding"])}

    def turn_records(self, *, turn_id=None):
        """Stable mirror identities for canonical ACK reconciliation across sessions."""
        with self.lock:
            query = (
                "SELECT t.id AS turn_id,t.session_id,a.id AS attempt_id,a.status,a.binding "
                "FROM turns t JOIN attempts a ON a.turn_id=t.id "
                "WHERE a.seq=(SELECT MAX(p.seq) FROM attempts p WHERE p.turn_id=t.id)")
            arguments = ()
            if turn_id is not None:
                query += " AND t.id=?"
                arguments = (turn_id,)
            rows = self.connection.execute(query + " ORDER BY t.rowid", arguments).fetchall()
            return [
                {"turn_id": row["turn_id"], "session_id": row["session_id"],
                 "latest_attempt": {"attempt_id": row["attempt_id"],
                                    "turn_id": row["turn_id"], "status": row["status"],
                                    "binding": json.loads(row["binding"])}}
                for row in rows
            ]

    def queued_inputs(self):
        """Durable received user text, not yet admitted to a cognitive turn."""
        with self.lock:
            rows = self.connection.execute(
                "SELECT t.id,m.content,k.value FROM turns t JOIN messages m "
                "ON m.turn_id=t.id AND m.role='user' JOIN metadata k "
                "ON k.key='queued-input:'||t.id WHERE t.session_id=? ORDER BY t.rowid",
                (self.session_id,),
            ).fetchall()
            result = []
            for row in rows:
                value = json.loads(row['value'])
                if (value.get('protocol') != 'text-input-queue.v1'
                        or value.get('turn_id') != row['id']
                        or value.get('session_id') != self.session_id
                        or not isinstance(value.get('binding'), dict)):
                    raise ValueError('Invalid persisted input queue binding')
                result.append({**value, 'text': row['content'], 'status': 'queued'})
            return result

    def enqueue_input(self, text, binding):
        if (not isinstance(text, str) or not text.strip() or len(text) > 8000
                or not isinstance(binding, dict)):
            raise ValueError('Invalid queued input')
        with self.transaction():
            if len(self.queued_inputs()) >= 256:
                raise ValueError('Input queue capacity reached')
            value = {'protocol': 'text-input-queue.v1', 'session_id': self.session_id,
                     'turn_id': str(uuid4()), 'attempt_id': str(uuid4()), 'binding': binding}
            encoded = json.dumps(value)
            if len(encoded.encode('utf-8')) > 131072:
                raise ValueError('Input queue binding exceeds capacity')
            self.connection.execute('INSERT INTO turns VALUES(?,?)',
                                    (value['turn_id'], self.session_id))
            self.connection.execute(
                "INSERT INTO messages(id,turn_id,role,content) VALUES(?,?,'user',?)",
                (str(uuid4()), value['turn_id'], text.strip()))
            self.connection.execute('INSERT INTO metadata VALUES(?,?)',
                                    ('queued-input:' + value['turn_id'], encoded))
        return {**json.loads(encoded), 'text': text.strip(), 'status': 'queued'}

    def begin_queued(self, turn_id):
        with self.transaction():
            pending = self.queued_inputs()
            if not pending or pending[0]['turn_id'] != turn_id:
                raise ValueError('Oldest received input required')
            if self.connection.execute(
                "SELECT 1 FROM attempts WHERE status='running' LIMIT 1"
            ).fetchone():
                raise ValueError('Unfinalized attempt')
            value = pending[0]
            self.connection.execute(
                "INSERT INTO attempts(id,turn_id,status,binding) VALUES(?,?,'running',?)",
                (value['attempt_id'], turn_id, json.dumps(value['binding'])))
            self.connection.execute('DELETE FROM metadata WHERE key=?',
                                    ('queued-input:' + turn_id,))
        return self.latest_attempt()

    def input_queue_paused(self):
        with self.lock:
            return self.connection.execute('SELECT 1 FROM metadata WHERE key=?',
                ('input-queue-paused:' + self.session_id,)).fetchone() is not None

    def pause_input_queue(self, paused):
        if type(paused) is not bool:
            raise ValueError('Explicit queue preference required')
        with self.transaction():
            key = 'input-queue-paused:' + self.session_id
            if paused:
                self.connection.execute('INSERT OR REPLACE INTO metadata VALUES(?,?)', (key, '1'))
            else:
                self.connection.execute('DELETE FROM metadata WHERE key=?', (key,))

    def _mirror_assistant(self, turn_id, content, *, confirmed_prefix=False, response_type=None):
        """Transaction-only confirmed display mirror, never an ACK issuer."""
        old = self.connection.execute(
            "SELECT id,content FROM messages WHERE turn_id=? AND role='assistant'", (turn_id,)
        ).fetchone()
        if old is not None:
            recorded = self.connection.execute(
                "SELECT value FROM metadata WHERE key=?", ("response-type:" + old["id"],)
            ).fetchone()
            recorded_type = "answer" if recorded is None else recorded[0]
            if response_type is not None and response_type != recorded_type:
                raise ValueError("Conflicting confirmed conversation response type")
            if old["content"] != content:
                if not confirmed_prefix or not content.startswith(old["content"]):
                    raise ValueError("Conflicting confirmed conversation text")
                self.connection.execute("UPDATE messages SET content=? WHERE id=?",
                                        (content, old["id"]))
            return old["id"]
        message_id = str(uuid4())
        self.connection.execute(
            "INSERT INTO messages(id,turn_id,role,content) VALUES(?,?,'assistant',?)",
            (message_id, turn_id, content),
        )
        if response_type == "ack":
            # Extend the existing v1 metadata contract in the same UOW. Missing
            # markers remain historical answers; no SQLite DDL/backfill is needed.
            self.connection.execute("INSERT INTO metadata(key,value) VALUES(?,?)",
                                    ("response-type:" + message_id, "ack"))
        return message_id

    def reserve_bubbles(self, turn_id, ranges):
        """Record arrival order only, without putting unseen text into history.

        User message sequence anchors the position; the receive clock orders
        multiple bubbles arriving between two user messages. Retries keep the
        first arrival, including after reopening the mirror.
        """
        with self.transaction():
            if not self.connection.execute("SELECT 1 FROM turns WHERE id=?", (turn_id,)).fetchone():
                raise ValueError('Known conversation turn required')
            key = 'bubble-timeline:' + turn_id
            row = self.connection.execute('SELECT value FROM metadata WHERE key=?',
                                          (key,)).fetchone()
            orders = [] if row is None else json.loads(row[0])
            existing = {item['start'] for item in orders}
            clock_row = self.connection.execute(
                "SELECT value FROM metadata WHERE key='semantic-timeline'").fetchone()
            clock = 0 if clock_row is None else int(clock_row[0])
            anchor = self.connection.execute(
                'SELECT COALESCE(MAX(seq),0) FROM messages').fetchone()[0]
            for item in ranges:
                if item['start'] not in existing:
                    clock += 1
                    orders.append({'start': item['start'], 'order': [anchor, clock]})
                    existing.add(item['start'])
            self.connection.execute('INSERT OR REPLACE INTO metadata VALUES(?,?)',
                                    (key, json.dumps(orders)))
            self.connection.execute("INSERT OR REPLACE INTO metadata VALUES('semantic-timeline',?)",
                                    (str(clock),))
            return [item for item in orders if item['start'] in {r['start'] for r in ranges}]

    def reconcile_confirmed(self, turn_id, content, *, completed, response_type="answer",
                            bubble_ranges=None):
        """Mirror trusted v19 confirmed text; the caller must read the canonical ACK.

        `completed` must come from v19's actual stream terminal, not model EOF.
        An active TextSession still owns finalization, while repaired interrupted
        attempts may be completed without regenerating or creating another input.
        Successive ACKs may extend one immutable cumulative prefix on the same
        assistant message; this mirror never accepts a replacement or shortening.
        """
        if (not isinstance(turn_id, str) or not turn_id or len(turn_id) > 128
                or not isinstance(content, str) or not content.strip() or len(content) > 64000
                or type(completed) is not bool or response_type not in {"answer", "ack"}):
            raise ValueError("Known turn, confirmed text and explicit outcome required")
        if bubble_ranges is not None and bubble_ranges != semantic_bubble_ranges(content):
            raise ValueError("Semantic bubbles must preserve the confirmed source paragraphs")
        with self.transaction():
            user = self.connection.execute(
                "SELECT 1 FROM messages WHERE turn_id=? AND role='user'", (turn_id,)
            ).fetchone()
            attempt = self.connection.execute(
                "SELECT id,status FROM attempts WHERE turn_id=? ORDER BY seq DESC LIMIT 1",
                (turn_id,),
            ).fetchone()
            if user is None or attempt is None:
                raise ValueError("Unknown conversation turn for confirmed mirror")
            message_id = self._mirror_assistant(turn_id, content, confirmed_prefix=True,
                                                response_type=response_type)
            if bubble_ranges is not None:
                self.connection.execute("INSERT OR REPLACE INTO metadata VALUES(?,?)",
                    ("bubble-ranges:" + message_id, json.dumps(bubble_ranges)))
            status = attempt["status"]
            if completed and status != "running":
                status = "completed"
                self.connection.execute(
                    "UPDATE attempts SET status='completed' WHERE id=?", (attempt["id"],)
                )
            return {"turn_id": turn_id, "message_id": message_id,
                    "attempt_id": attempt["id"], "status": status}

    def reconcile_listened(self, turn_id):
        """Restore the canonical silent terminal after a process/mirror failure."""
        with self.transaction():
            attempt = self.connection.execute(
                "SELECT id,status FROM attempts WHERE turn_id=? ORDER BY seq DESC LIMIT 1",
                (turn_id,),
            ).fetchone()
            if attempt is None or self.connection.execute(
                "SELECT 1 FROM messages WHERE turn_id=? AND role='assistant'", (turn_id,)
            ).fetchone():
                raise ValueError("Listening mirror requires a known silent turn")
            # An active TextSession owns its finish; an opener changes old running
            # attempts to interrupted before canonical reconciliation gets here.
            if attempt["status"] != "running":
                self.connection.execute("UPDATE attempts SET status='listened' WHERE id=?",
                                        (attempt["id"],))

    def begin(self, text, binding, *, retry=False):
        with self.transaction():
            latest = self.latest_attempt()
            if latest and latest["status"] == "running":
                raise ValueError("Unfinalized attempt")
            if retry:
                if not latest or latest["status"] not in {"failed", "cancelled", "interrupted"}:
                    raise ValueError("No retryable attempt")
                turn_id = latest["turn_id"]
            else:
                turn_id = str(uuid4())
                self.connection.execute("INSERT INTO turns VALUES(?,?)", (turn_id, self.session_id))
                self.connection.execute(
                    """INSERT INTO messages(id,turn_id,role,content)
                    VALUES(?,?,'user',?)""",
                    (str(uuid4()), turn_id, text),
                )
            attempt_id = str(uuid4())
            self.connection.execute(
                """INSERT INTO attempts(id,turn_id,status,binding)
                VALUES(?,?,'running',?)""",
                (attempt_id, turn_id, json.dumps(binding)),
            )
        return self.latest_attempt()

    def finish(self, attempt_id, status, content):
        if status not in {"completed", "cancelled", "failed", "listened", "received"}:
            raise ValueError("Invalid terminal status")
        if status == "listened" and content:
            raise ValueError("Listening cannot create an assistant message")
        with self.transaction():
            row = self.connection.execute(
                "SELECT * FROM attempts WHERE id=?", (attempt_id,)
            ).fetchone()
            if not row or row["status"] != "running":
                raise ValueError("Attempt missing or already finalized")
            if status == "completed":
                self._mirror_assistant(row["turn_id"], content)
            self.connection.execute("UPDATE attempts SET status=? WHERE id=?", (status, attempt_id))

    def backup(self):
        destination = self.root / ("backup-" + str(uuid4()) + ".sqlite3")
        with self.lock, sqlite3.connect(destination) as target:
            self.connection.backup(target)
        return destination

    def close(self):
        with self.lock:
            self.connection.close()
