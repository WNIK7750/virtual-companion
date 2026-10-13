"""Rebuildable, source-bound dialogue views; no derived persistence or model call."""

import json

from .contracts import Scope, VersionConflict, canonical, identifier
from .history_readset import evidence_hash
from .sqlite_streaming import _DIALOGUE, SQLiteStreamingStore

VERSION = 'dialogue-memory.v1'
FRAGMENT_PROTOCOL = 'dialogue-fragment.v1'


def fragment_reference(fragment):
    """Content-free dependency edges for exactly the originals actually admitted."""
    validate_fragments([fragment])
    return {key: fragment[key] for key in (
        'fragment_id', 'version', 'origin_session', 'input_id', 'config_sha256',
        'part', 'response_completeness', 'membership_policy')
    } | {'source_type': 'dialogue_fragment', 'sha256': evidence_hash(fragment), 'members': [
        {key: member[key] for key in (
            'source_id', 'role', 'sha256', 'response_type', 'visibility')}
        for member in fragment['members']]}


def source_count(rows):
    """Count every transmitted original, including a repeated partition anchor."""
    return sum(len(row['fragment']['members']) if row.get('kind') == 'dialogue_fragment'
               else 1 for row in rows)


def originals_size(rows):
    facts = [row for row in rows if row.get('kind') != 'dialogue_fragment']
    fragments = [row['fragment'] for row in rows if row.get('kind') == 'dialogue_fragment']
    return len(canonical(facts).encode('utf-8')) + len(canonical(fragments).encode('utf-8')) - 2


def validate_fragments(fragments):
    if not isinstance(fragments, list) or len(fragments) > 24:
        raise ValueError('Bounded dialogue fragment array required')
    seen = set()
    for fragment in fragments:
        if not isinstance(fragment, dict) or set(fragment) != {
            'protocol', 'fragment_id', 'version', 'origin_session', 'input_id',
            'config_sha256', 'character_context', 'event_time', 'event_time_status',
            'part', 'response_completeness', 'membership_policy', 'members',
        } or fragment['protocol'] != FRAGMENT_PROTOCOL or fragment['version'] != 1:
            raise ValueError('Exact source-bound dialogue fragment required')
        for key in ('fragment_id', 'origin_session', 'input_id'):
            identifier(fragment[key])
        if fragment['fragment_id'] in seen:
            raise ValueError('Duplicate dialogue fragment')
        seen.add(fragment['fragment_id'])
        if (fragment['event_time'] is not None or fragment['event_time_status'] != 'unknown'
                or type(fragment['part']) is not int or fragment['part'] < 0
                or fragment['response_completeness'] != 'unknown'
                or fragment['membership_policy'] != 'frozen_confirmed_subset'):
            raise ValueError('Dialogue source time is unknown; local order is not UTC')
        members = fragment['members']
        if not isinstance(members, list) or not 2 <= len(members) <= 8:
            raise ValueError('Dialogue fragment requires a user and visible assistant originals')
        member_ids = set()
        for index, member in enumerate(members):
            if not isinstance(member, dict) or set(member) != {
                'source_id', 'role', 'text', 'sha256', 'response_type', 'visibility',
            }:
                raise ValueError('Exact dialogue original member required')
            identifier(member['source_id'])
            if member['source_id'] in member_ids:
                raise ValueError('Duplicate original member')
            member_ids.add(member['source_id'])
            if (
                not isinstance(member["text"], str)
                or not member["text"]
                or member["sha256"] != evidence_hash(member["text"])
                or member["role"] != ("user" if index == 0 else "assistant")
                or member["response_type"] not in ((None,) if index == 0 else ("answer", "ack"))
                or member["visibility"] != ("submitted" if index == 0 else "confirmed_visible")
            ):
                raise ValueError('Dialogue original role, hash or visibility differs')
        if members[0]['source_id'] != fragment['input_id']:
            raise ValueError('Dialogue input anchor differs')
        content = {key: value for key, value in fragment.items() if key != 'fragment_id'}
        if fragment['fragment_id'] != 'dialogue-fragment-' + evidence_hash(content):
            raise ValueError('Dialogue fragment identity differs from complete originals')
    return fragments


def rebuild_fragments(db, companion, input_id, excluded, *, max_inputs=1000):
    """Read prior authority rows in one owner transaction, across sessions.

    A mixed invalidated turn is suppressed as a whole. Confirmed source units
    are never truncated; long assistant turns are explicitly partitioned.
    """
    anchor = db.execute('SELECT rowid FROM turn_inputs WHERE companion=? AND input=?',
                        (companion, input_id)).fetchone()
    if anchor is None:
        raise VersionConflict('Dialogue retrieval current input missing')
    inputs = db.execute(
        'SELECT rowid AS ordinal,input,session,config FROM turn_inputs '
        'WHERE companion=? AND rowid<? ORDER BY rowid DESC LIMIT ?',
        (companion, anchor[0], max_inputs)).fetchall()
    result = []
    from virtual_companion.text.character import character_from_binding

    for source in inputs:
        rows = db.execute(
            f'SELECT * FROM ({_DIALOGUE}) WHERE companion=? AND ordinal=? '  # noqa: S608
            'ORDER BY phase,sequence', (companion, source['ordinal'])).fetchall()
        if (len(rows) < 2 or rows[0]['role'] != 'user'
                or any(row['source'] in excluded for row in rows)):
            continue
        members = [{'source_id': row['source'], 'role': row['role'], 'text': row['text'],
                    'sha256': evidence_hash(row['text']), 'response_type': row['response_type'],
                    'visibility': 'submitted' if row['role'] == 'user' else 'confirmed_visible'}
                   for row in rows]
        for offset in range(1, len(members), 7):
            fragment = {'protocol': FRAGMENT_PROTOCOL, 'version': 1,
                'origin_session': source['session'], 'input_id': source['input'],
                'config_sha256': evidence_hash(json.loads(source['config'])),
                'character_context': character_from_binding(source['config']),
                'event_time': None, 'event_time_status': 'unknown',
                'part': (offset - 1) // 7, 'response_completeness': 'unknown',
                'membership_policy': 'frozen_confirmed_subset',
                'members': [members[0], *members[offset:offset + 7]]}
            fragment['fragment_id'] = 'dialogue-fragment-' + evidence_hash(fragment)
            result.append(fragment)
    return validate_fragments(result) if len(result) <= 24 else result


def references_current(db, companion, input_id, references):
    """Recheck frozen confirmed subsets; later visible additions do not expand them.

    Every selected original and its receipt must still match. Membership is
    explicitly partial, never a promise of the full response or all visible
    units. Invalidation closure remains owned by the store.
    """
    if not references:
        # Older fact/exchange readsets have no dialogue dependency to recheck.
        return True
    anchor = db.execute('SELECT rowid FROM turn_inputs WHERE companion=? AND input=?',
                        (companion, input_id)).fetchone()
    if anchor is None:
        return False
    for ref in references:
        source = db.execute('SELECT rowid,session,config FROM turn_inputs '
                            'WHERE companion=? AND input=?',
                            (companion, ref['input_id'])).fetchone()
        if (source is None or source[0] >= anchor[0] or source['session'] != ref['origin_session']
                or evidence_hash(json.loads(source['config'])) != ref['config_sha256']):
            return False
        members = []
        for member in ref['members']:
            row = db.execute(
                f'SELECT * FROM ({_DIALOGUE}) WHERE companion=? AND source=?',  # noqa: S608
                (companion, member['source_id'])).fetchone()
            if (row is None or row['ordinal'] != source[0] or row['role'] != member['role']
                    or row['response_type'] != member['response_type']
                    or evidence_hash(row['text']) != member['sha256']):
                return False
            if member['role'] == 'assistant':
                try:
                    original = SQLiteStreamingStore._exchange_source(
                        db,
                        Scope(companion, ref["origin_session"]),
                        "assistant",
                        member["source_id"],
                    )
                except (ValueError, KeyError, TypeError):
                    return False
                if original is None or original['text'] != row['text']:
                    return False
            members.append({**member, 'text': row['text']})
        from virtual_companion.text.character import character_from_binding

        fragment = {'protocol': FRAGMENT_PROTOCOL, 'version': ref['version'],
            'origin_session': ref['origin_session'], 'input_id': ref['input_id'],
            'config_sha256': ref['config_sha256'],
            'character_context': character_from_binding(source['config']),
            'event_time': None, 'event_time_status': 'unknown', 'part': ref['part'],
            'response_completeness': ref['response_completeness'],
            'membership_policy': ref['membership_policy'], 'members': members}
        if ref['fragment_id'] != 'dialogue-fragment-' + evidence_hash(fragment):
            return False
        fragment['fragment_id'] = ref['fragment_id']
        if ref['sha256'] != evidence_hash(fragment):
            return False
    return True


def fragment_row(fragment):
    original_text = '\n'.join(f"{member['role']}: {member['text']}"
                              for member in fragment['members'])
    return {'record_id': fragment['fragment_id'], 'version': 1,
            'kind': 'dialogue_fragment', 'source_kind': 'dialogue', 'status': 'active',
            'expires_at': None, 'fact_key': None, 'fragment': fragment,
            'content': original_text, 'member_count': len(fragment['members'])}


class DialogueCatalogStore:
    """Typed union port over the existing owner; originals are read again."""

    def __init__(self, store, input_id):
        self.store, self.input_id = store, input_id
        self._types = {}

    def snapshot(self, companion):
        return self.store.snapshot(companion)

    def committed_catalog_page(self, companion, *, now_us, after_record_id=None, limit=8):
        facts = self.store.committed_catalog_page(companion, now_us=now_us,
            after_record_id=after_record_id, limit=limit)
        fragments = self.store.dialogue_fragments(companion, self.input_id, now_us=now_us)
        # ID strings are opaque. A collision in either domain is rejected even
        # when the fact would land outside the current paginated metadata page.
        with self.store.transaction(write=False) as db:
            anchor = db.execute('SELECT rowid FROM turn_inputs WHERE companion=? AND input=?',
                                (companion, self.input_id)).fetchone()
            count = db.execute('SELECT count(*) FROM turn_inputs WHERE companion=? AND rowid<?',
                               (companion, anchor[0])).fetchone()[0]
            self.catalog_source_coverage = 'partial' if count > 1000 else 'complete'
            if any(db.execute('SELECT 1 FROM memory_heads WHERE companion=? AND record=?',
                              (companion, row['fragment_id'])).fetchone() for row in fragments):
                raise VersionConflict('Fact and dialogue source identity collision')
        rows = [*facts['records'], *(fragment_row(row) for row in fragments
                if after_record_id is None or row['fragment_id'] > after_record_id)]
        if len({row['record_id'] for row in rows}) != len(rows):
            raise VersionConflict('Fact and dialogue source identity collision')
        rows.sort(key=lambda row: row['record_id'])
        more = len(rows) > limit or facts['next_after_record_id'] is not None
        selected = rows[:limit]
        for row in selected:
            kind = row['kind']
            if row['record_id'] in self._types and self._types[row['record_id']] != kind:
                raise VersionConflict('Fact and dialogue source identity collision')
            self._types[row['record_id']] = kind
        return {'records': tuple(selected),
                'next_after_record_id': selected[-1]['record_id'] if selected and more else None}

    def committed_recall(self, companion, *, now_us, candidates, limit):
        rows = []
        fragments = {row['fragment_id']: row for row in self.store.dialogue_fragments(
            companion, self.input_id, now_us=now_us)}
        for record_id, version in candidates:
            source_type = self._types.get(record_id)
            if source_type == 'dialogue_fragment':
                if record_id in fragments and version == 1:
                    rows.append(fragment_row(fragments[record_id]))
            elif source_type in {'fact', 'event'}:
                found = self.store.committed_recall(companion, now_us=now_us,
                    candidates=((record_id, version),), limit=1)
                rows.extend(found)
            else:
                raise ValueError('Original read requires a frozen typed catalog reference')
        if source_count(rows) > 8:
            raise ValueError(
                "A read tool may return at most eight original members including facts"
            )
        return rows
