"""v20: received units release delivery; only actual visibility creates history."""

from uuid import uuid4

from .contracts import VersionConflict, identifier
from .history_readset import evidence_hash, received_output_references
from .received import ReceivedText
from .sqlite_memory_management import SQLiteExplicitMemoryStore
from .sqlite_state import _APPLICATION_ID
from .turns import RunClaim

_EXTRA = (
    'CREATE TABLE stream_received (companion TEXT NOT NULL, input TEXT NOT NULL, '
    'sequence INTEGER NOT NULL, fingerprint TEXT NOT NULL, host_receipt TEXT NOT NULL, '
    'PRIMARY KEY(companion,input,sequence), '
    'FOREIGN KEY(companion,input,sequence) '
    'REFERENCES stream_segments(companion,input,sequence))',
)


class SQLiteReceivedStore(SQLiteExplicitMemoryStore):
    schema_version = 20
    schema = SQLiteExplicitMemoryStore.schema + _EXTRA
    tables = SQLiteExplicitMemoryStore.tables | {'stream_received'}
    supports_received_text = True
    stream_outcomes = SQLiteExplicitMemoryStore.stream_outcomes | {'received'}
    upgrade_backup = None

    def _history_provenance_rows(self, db, companion):
        # Own sent text retains the same revocation/expiry dependency graph even
        # before it becomes visible history. This does not enqueue a source.
        return db.execute(
            "SELECT s.*,'segment-' || s.fingerprint AS source FROM stream_segments s "
            'JOIN turn_inputs i ON i.companion=s.companion AND i.input=s.input '
            'LEFT JOIN stream_history h ON h.companion=s.companion AND h.input=s.input '
            'AND h.sequence=s.sequence LEFT JOIN stream_received r '
            'ON r.companion=s.companion AND r.input=s.input AND r.sequence=s.sequence '
            'WHERE s.companion=? AND (h.source IS NOT NULL OR r.input IS NOT NULL) '
            'ORDER BY i.rowid,s.sequence', (companion,))

    def _received_references_current(self, db, companion, input_id, references):
        for reference in references:
            row = db.execute(
                'SELECT s.* FROM stream_segments s JOIN stream_received r '
                'ON r.companion=s.companion AND r.input=s.input AND r.sequence=s.sequence '
                'AND r.fingerprint=s.fingerprint JOIN turn_inputs origin '
                'ON origin.companion=s.companion AND origin.input=s.input '
                'JOIN turn_inputs current ON current.companion=origin.companion '
                'AND current.session=origin.session AND origin.rowid<current.rowid '
                'WHERE s.companion=? AND current.input=? AND s.input=? AND s.sequence=?',
                (companion, input_id, reference['input_id'], reference['sequence']),
            ).fetchone()
            if row is None:
                return False
            event = self._segment_event(row)
            if (event['fingerprint'] != reference['fingerprint']
                    or evidence_hash(event['text']) != reference['sha256']
                    or event['presentation']['candidate'].get('response_type', 'answer')
                    != reference['response_type']):
                return False
        return True

    def dialogue_context(self, claim, *, now_us):
        """One SQLite snapshot: a racing visible ACK cannot lose/double own text."""
        if not isinstance(claim, RunClaim):
            raise ValueError('Typed input required for received context')
        with self.transaction(write=False) as db:
            sources, _ = self._exchange_context(db, claim, now_us=now_us)
            return sources, self._received_output_context(db, claim, now_us=now_us)

    def _received_output_context(self, db, claim, *, now_us):
        current = self._input(db, claim.companion_id, claim.input_id)
        if (current['session'] != claim.session_id or current['status'] != 'running'
                or self._run_config(db, current) != claim.config_json):
            raise VersionConflict('Received context requires the admitted current input')
        excluded = self._ineffective_history_sources(db, claim.companion_id, now_us=now_us)
        rows = db.execute(
            'SELECT s.* FROM stream_segments s JOIN stream_received r '
            'ON r.companion=s.companion AND r.input=s.input AND r.sequence=s.sequence '
            'AND r.fingerprint=s.fingerprint JOIN turn_inputs origin '
            'ON origin.companion=s.companion AND origin.input=s.input '
            'JOIN turn_inputs current ON current.companion=origin.companion '
            'AND current.session=origin.session AND origin.rowid<current.rowid '
            "WHERE s.companion=? AND current.input=? AND s.status!='confirmed' "
            'ORDER BY origin.rowid DESC,s.sequence DESC LIMIT 32',
            (claim.companion_id, claim.input_id)).fetchall()
        result = []
        for row in reversed(rows):
            if row['input'] in excluded or 'segment-' + row['fingerprint'] in excluded:
                continue
            event = self._segment_event(row)
            result.append({
                'input_id': row['input'], 'sequence': row['sequence'],
                'fingerprint': event['fingerprint'], 'text': event['text'],
                'response_type': event['presentation']['candidate'].get('response_type', 'answer'),
                'visibility': 'not_confirmed_at_admission',
            })
        return result

    def validate_received_output_context(self, companion, input_id, items, *, now_us):
        references = received_output_references(items)
        with self.transaction(write=False) as db:
            excluded = self._ineffective_history_sources(db, companion, now_us=now_us)
            if (not self._received_references_current(db, companion, input_id, references)
                    or any(item['input_id'] in excluded
                           or 'segment-' + item['fingerprint'] in excluded
                           for item in references)):
                raise VersionConflict('Received output context expired or changed')

    def _upgrade(self, db, version, application, names):
        if (version != 19 or application != _APPLICATION_ID
                or names != SQLiteExplicitMemoryStore.tables):
            raise ValueError('Reliable reception accepts only the exact v19 schema')
        # BEGIN IMMEDIATE already excludes competing writers. Backup reads via
        # a separate read-only connection before any DDL changes the source.
        self.upgrade_backup = self.backup_to(self.path.with_name(
            self.path.name + '.before-v20-' + str(uuid4()) + '.sqlite3'))
        for statement in _EXTRA:
            db.execute(statement)
        db.execute('PRAGMA user_version=20')

    @staticmethod
    def _received(db, companion, input_id, sequence):
        row = db.execute('SELECT sequence,fingerprint,host_receipt FROM stream_received '
                         'WHERE companion=? AND input=? AND sequence=?',
                         (companion, input_id, sequence)).fetchone()
        return None if row is None else ReceivedText(*row)

    def received_segment(self, companion, input_id, sequence):
        with self.transaction(write=False) as db:
            return self._received(db, companion, input_id, sequence)

    def receive_segment(self, companion, input_id, sequence, *, fingerprint, host_receipt):
        identifier(host_receipt)
        with self.transaction() as db:
            rows = self._segment_rows(db, companion, input_id)
            if type(sequence) is not int or not 1 <= sequence <= len(rows):
                raise ValueError('Unknown received unit')
            event = self._segment_event(rows[sequence - 1])
            if event['status'] == 'pending' or event['fingerprint'] != fingerprint:
                raise ValueError('Receipt must identify the exact dispatched approval')
            receipt = ReceivedText(sequence, fingerprint, host_receipt)
            old = self._received(db, companion, input_id, sequence)
            if old:
                if receipt != old:
                    raise ValueError('Conflicting reliable text receipt')
                return old
            db.execute('INSERT INTO stream_received VALUES(?,?,?,?,?)',
                       (companion, input_id, sequence, fingerprint, host_receipt))
            return receipt

    def _segment_delivered(self, db, row):
        return (super()._segment_delivered(db, row)
                or self._received(db, row['companion'], row['input'], row['sequence']) is not None)

    def _validate_received_settlement(self, db, claim, rows, now_us):
        self._segment_live(db, claim, now_us)
        if not rows or any(not self._segment_delivered(db, r) for r in rows):
            raise ValueError('Reliable settlement requires every unit to be received')

    def _delivery_counts(self, db, companion, input_id, events):
        received = [e for e in events if self._received(db, companion, input_id, e['sequence'])]
        return {'received_units': len(received),
                'unknown_units': sum(e['status'] == 'unknown' and not self._received(
                    db, companion, input_id, e['sequence']) for e in events)}
