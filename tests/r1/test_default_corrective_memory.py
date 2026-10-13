"""A rejected initial answer can spend the one existing memory delegation."""

import json
from copy import deepcopy

import pytest
from test_default_cross_turn_memory import ORIGINAL, QUERY, Scenario, build, turn
from test_default_text_delivery import wait_for
from unit_review_recording import bound_unit_packets as bound_unit_packets
from unit_review_recording import recorded_review_packet
from virtual_companion.text.default_runtime import ACCESS_COMPLETE, ACCESS_INITIAL, REPAIR_REPLY
from virtual_companion.text.fake import synthetic_unit_opinion


class CorrectiveMemory(Scenario):
    def __init__(self, *, already_accessed=False, reject_again=False):
        super().__init__()
        self.already_accessed, self.reject_again = already_accessed, reject_again
        self.query_reviews = 0

    def stream(self, messages, cancelled):
        if '你只审查候选文字' in messages[0]['content']:
            payload = recorded_review_packet(messages)
            if payload['user_text'] == QUERY:
                self.calls.append(deepcopy(messages))
                self.query_reviews += 1
                rejected = self.query_reviews == 1 or self.reject_again
                yield json.dumps(synthetic_unit_opinion(payload,
                    decision='reject' if rejected else 'approve',
                    constraints_satisfied=not rejected))
                return
        elif messages[-1]['content'] == QUERY:
            repair = any(message['content'].startswith(REPAIR_REPLY) for message in messages)
            if repair or (ACCESS_INITIAL in messages[0]['content'] and not self.already_accessed):
                self.calls.append(deepcopy(messages))
                self.primary.append(deepcopy(messages))
                yield json.dumps({'action': 'memory'} if repair else {
                    'action': 'answer', 'text': '相关记忆这次没有成功取用。'}, ensure_ascii=False)
                return
        yield from super().stream(messages, cancelled)


@pytest.mark.parametrize('reject_again', [False, True])
def test_corrective_memory_refreshes_once_and_final_review_uses_new_context(tmp_path, reject_again):
    model = CorrectiveMemory(reject_again=reject_again)
    memory, mirror, runtime, session = build(tmp_path, model)
    try:
        turn(runtime, session, ORIGINAL)
        model.calls.clear()
        resolutions = []
        original_resolve = runtime.resolve

        def resolve(claim):
            resolutions.append(claim)
            return original_resolve(claim)

        runtime.resolve = resolve
        session.send(QUERY)
        wait_for(lambda: runtime.poll()['event'] or not session.snapshot()['busy'])
        assert len(model.calls) == 5  # answer, reject, corrective plan, final answer, review
        assert len(resolutions) == 1
        initial, first_review, repair, final, last_review = model.calls
        assert all(ACCESS_INITIAL in call[0]['content']
                   for call in (initial, first_review, repair))
        assert all(ACCESS_COMPLETE in call[0]['content'] for call in (final, last_review))
        before = recorded_review_packet(first_review)
        after = recorded_review_packet(last_review)
        assert before['basis']['input_id'] == after['basis']['input_id']
        assert before['context'] != after['context']
        assert json.loads(after['context'])['memories']
        if reject_again:
            assert session.snapshot()['attempt']['status'] == 'failed'
            assert not memory.store.segments(memory.companion, runtime.claim.input_id)
        else:
            event = runtime.poll()['event']
            assert event['display_text'] == '你喜欢安静的音乐。'
            assert event['readset']['state_version'] == after['basis']['state_version']
            runtime.ack(dict(sequence=event['sequence'], fingerprint=event['fingerprint'],
                             host_receipt='synthetic-corrective-memory-visible'))
            wait_for(lambda: not session.snapshot()['busy'])
    finally:
        memory.close()
        mirror.close()


def test_corrective_plan_cannot_delegate_memory_twice(tmp_path):
    model = CorrectiveMemory(already_accessed=True)
    memory, mirror, runtime, session = build(tmp_path, model)
    try:
        turn(runtime, session, ORIGINAL)
        model.calls.clear()
        resolutions = []
        original_resolve = runtime.resolve

        def resolve(claim):
            resolutions.append(claim)
            return original_resolve(claim)

        runtime.resolve = resolve
        session.send(QUERY)
        wait_for(lambda: not session.snapshot()['busy'])
        assert session.snapshot()['attempt']['status'] == 'failed'
        assert len(resolutions) == 1
        assert len(model.calls) == 4  # initial memory, final answer, reject, invalid repair
        assert model.query_reviews == 1
        assert not memory.store.segments(memory.companion, runtime.claim.input_id)
    finally:
        memory.close()
        mirror.close()


@pytest.mark.parametrize('mode', ['correct', 'forget'])
def test_explicit_management_processes_its_source_before_a_wrong_initial_answer(tmp_path, mode):
    from test_default_demand_integration import DemandModel, ack_and_finish, demand_setup

    class WrongInitialAnswer(DemandModel):
        def stream(self, messages, cancelled):
            if '你只审查候选文字' in messages[0]['content']:
                self.calls.append(deepcopy(messages))
                payload = recorded_review_packet(messages)
                context = json.loads(payload['context'])
                supported = context.get('current_operation', {}).get('action') == mode
                yield json.dumps(synthetic_unit_opinion(payload,
                    decision='approve' if supported else 'reject',
                    constraints_satisfied=supported))
                return
            self.calls.append(deepcopy(messages))
            self.primary.append(deepcopy(messages))
            final = '本次只能返回answer' in messages[0]['content']
            if any(message['content'].startswith(REPAIR_REPLY) for message in messages):
                yield json.dumps({'action': 'memory'})
                return
            yield json.dumps({'action': 'answer', 'text': '已按本次请求处理记忆。' if final
                              else '我已经记下来了。'}, ensure_ascii=False)

    source = '更正，我喜欢摇滚。' if mode == 'correct' else '请忘记我喜欢安静音乐这件事。'
    runtime, text, memory, mirror, model, _, stages = demand_setup(
        tmp_path, mode, source=source, model=WrongInitialAnswer(first_action='answer'),
    )
    try:
        text.send(source)
        event = ack_and_finish(runtime, text)
        assert stages == ['memory_command', 'memory_command_support']
        assert len(model.primary) == 3 and len(model.calls) == 5
        initial, rejected, repair, final, approved = model.calls
        assert recorded_review_packet(rejected)['candidate_prefix'] == '我已经记下来了。'
        assert 'current_operation' not in json.loads(recorded_review_packet(rejected)['context'])
        assert any(message['content'].startswith(REPAIR_REPLY) for message in repair)
        assert '本次只能返回answer' in final[0]['content']
        assert recorded_review_packet(approved)['candidate_prefix'] == event['display_text']
        context = json.loads(recorded_review_packet(model.calls[-1])['context'])
        assert context['current_operation']['action'] == mode
        assert context['current_operation']['source_id'] == event['basis']['input_id']
        assert event['display_text'] == '已按本次请求处理记忆。'
        assert mirror.history()['messages'][-1]['content'] == event['display_text']
        history = memory.store.history(memory.companion, 'music')
        assert len(history) == 2 and history[-1]['version'] == 2
        assert history[-1]['source_id'] == event['basis']['input_id']
        assert history[-1]['status'] == ('active' if mode == 'correct' else 'revoked')
    finally:
        memory.close()
        mirror.close()


def test_explicit_save_only_answers_after_the_current_source_fact_is_persisted(tmp_path):
    original = '请记住，我喜欢画画'
    model = Scenario(original=original, recalled='已记住你喜欢画画。', predicate='hobby')
    memory, mirror, runtime, session = build(tmp_path, model)
    try:
        event = turn(runtime, session, original)
        assert len(model.primary) == 2
        assert '本次只能返回answer' not in model.primary[0][0]['content']
        assert '本次只能返回answer' in model.primary[1][0]['content']
        fact, = event['readset']['facts']
        assert fact['source_id'] == event['basis']['input_id']
        assert memory.store.history(memory.companion, fact['record_id'])[0]['content'] == original
        assert event['display_text'] == '已记住你喜欢画画。'
    finally:
        memory.close()
        mirror.close()


@pytest.mark.parametrize('change', ['correct', 'forget', 'expiry'])
def test_default_query_after_post_change_backup_restore_keeps_effective_memory(tmp_path, change):
    from datetime import datetime, timedelta
    from hashlib import sha256

    from virtual_companion.boundary.sqlite_received import SQLiteReceivedStore
    from virtual_companion.text.chronology import utc_microseconds

    original, updated, query = '我喜欢画画', '我偏爱散步', '空闲时能怎样放松？'
    assert not set(original) & set(query)

    class RestoredMemory(Scenario):
        def __init__(self):
            super().__init__(expires=change == 'expiry', original=original, updated=updated,
                query=query, predicate='hobby', recalled='你喜欢画画，可以画一片叶子放松。',
                updated_reply='你现在偏爱散步，可以慢慢走一会儿放松。',
                missing='没有可核验的兴趣偏好记录，可以先短暂休息。')
            self.review_packets, self.query_actions = [], []

        def stream(self, messages, cancelled):
            if '你只审查候选文字' in messages[0]['content']:
                self.review_packets.append(recorded_review_packet(messages))
            elif messages[-1]['content'] == query and (
                '本次只能返回answer' not in messages[0]['content']
            ):
                self.calls.append(deepcopy(messages))
                self.primary.append(deepcopy(messages))
                expression = '绘画 画画 散步 兴趣 爱好'
                self.query_actions.append(expression)
                yield json.dumps({'action': 'memory', 'query': expression}, ensure_ascii=False)
                return
            # The answer depends on the admitted evidence, not the test's expected state.
            yield from super().stream(messages, cancelled)

    def assert_confirmed(event):
        input_id = event['basis']['input_id']
        segment, = memory.store.segments(memory.companion, input_id)
        assert segment['status'] == 'confirmed'
        assert segment['fingerprint'] == event['fingerprint']
        assert segment['basis'] == event['basis'] and segment['readset'] == event['readset']
        assert segment['display_text'] == event['display_text']
        assert segment['host_receipt'].startswith('recorded-visible-')
        assert segment['presentation']['review']['model_approved'] is True
        state = memory.store.stream_state(memory.companion, input_id)
        assert state['outcome'] == 'completed'
        assert state['approved_units'] == state['confirmed_units'] == 1
        assert state['pending_units'] == state['unknown_units'] == 0
        assert state['confirmed_text'] == event['display_text']

    def completed_user_source(event):
        with memory.store.transaction(write=False) as db:
            row = db.execute(
                "SELECT s.status,w.receipt FROM background_sources s JOIN background_work w "
                "ON s.companion=w.companion AND s.job=w.job WHERE s.companion=? "
                "AND s.source=? AND s.role='user'",
                (memory.companion, event['basis']['input_id']),
            ).fetchone()
        assert row['status'] == 'completed'
        return json.loads(row['receipt'])

    def memory_audit(now_us):
        # Read the real committed heads/revisions/receipts, never seed them by mutation.
        with memory.store.transaction(write=False) as db:
            tables = {
                table: [dict(row) for row in db.execute(sql, (memory.companion,))]
                for table, sql in (
                    ('heads', 'SELECT * FROM memory_heads WHERE companion=? ORDER BY record'),
                    ('versions', 'SELECT * FROM memory_versions WHERE companion=? '
                                 'ORDER BY record,version'),
                    ('operations', 'SELECT * FROM memory_management_operations '
                                   'WHERE companion=? ORDER BY operation'),
                    ('invalidations', 'SELECT * FROM memory_invalidated_sources '
                                      'WHERE companion=? ORDER BY source,record,version,operation'),
                )
            }
        return dict(tables=tables,
            versions=memory.store.history(memory.companion, saved['record_id']),
            eligible=memory.store.recall(memory.companion, now_us=now_us),
            excluded=memory.store.ineffective_history_sources(memory.companion, now_us=now_us),
            old_ack=memory.store.segments(memory.companion, recalled['basis']['input_id']),
            old_terminal=memory.store.stream_state(memory.companion, recalled['basis']['input_id']))

    model = RestoredMemory()
    (tmp_path / 'authority').mkdir()
    memory, mirror, runtime, session = build(tmp_path / 'authority', model)
    try:
        authority = memory.path
        first = turn(runtime, session, original)
        assert_confirmed(first)
        receipt = completed_user_source(first)
        assert receipt['operations'] == 1
        assert receipt['resolution']['source_id'] == first['basis']['input_id']
        assert [stage for stage, _ in model.worker_calls] == [
            'memory_command', 'exchange_intent', 'fact_extract', 'fact_support', 'exchange_extract']
        saved, = memory.store.recall(memory.companion, now_us=model.clock())
        assert saved['content'] == original and saved['status'] == 'active'
        assert saved['version'] == 1 and saved['source_id'] == first['basis']['input_id']
        assert saved['fact_key'] == {'subject': 'user', 'predicate': 'hobby', 'scope': 'profile'}
        assert json.loads(saved['reason'])['support']['verdict'] == 'supported'
        assert memory.store.history(memory.companion, saved['record_id']) == [saved]

        recalled = turn(runtime, session, query)
        assert_confirmed(recalled)
        assert recalled['display_text'] == model.recalled
        old_fact, = recalled['readset']['facts']
        assert old_fact['record_id'] == saved['record_id'] and old_fact['version'] == 1
        assert old_fact['source_id'] == first['basis']['input_id']
        old_ack_source = 'segment-' + recalled['fingerprint']
        pre_change = memory.store.backup_to(tmp_path / 'before-change.sqlite3')
        if change == 'expiry':
            version = memory.store.snapshot(memory.companion).write_version
            expires_at = datetime.fromisoformat(saved['expires_at'])
            model.now = expires_at - timedelta(microseconds=1)
            assert memory.store.recall(memory.companion, now_us=model.clock()) == [saved]
            model.now = expires_at
            assert memory.store.recall(memory.companion, now_us=model.clock()) == []
            assert memory.store.snapshot(memory.companion).write_version == version
        else:
            before_calls = len(model.worker_calls)
            changed = turn(runtime, session, model.correct if change == 'correct' else model.forget)
            assert_confirmed(changed)
            completed_user_source(changed)
            assert [stage for stage, _ in model.worker_calls[before_calls:]] == [
                'memory_command', 'memory_command_support']
            operation = changed['readset']['current_operation']
            assert operation['status'] == 'committed' and operation['action'] == change
            assert operation['source_id'] == changed['basis']['input_id']
            versions = memory.store.history(memory.companion, saved['record_id'])
            assert [(row['version'], row['status']) for row in versions] == [
                (1, 'active'), (2, 'active' if change == 'correct' else 'revoked')]
            assert versions[-1]['source_id'] == changed['basis']['input_id']
            if change == 'correct':
                assert versions[-1]['content'] == updated
                assert memory.store.recall(memory.companion, now_us=model.clock()) == [versions[-1]]
            else:
                assert memory.store.recall(memory.companion, now_us=model.clock()) == []

        reopened_at = model.now + timedelta(days=2)
        reopened_us = utc_microseconds(reopened_at)
        assert reopened_us - model.clock() >= 24 * 60 * 60 * 1_000_000
        post_state = memory_audit(reopened_us)
        assert first['basis']['input_id'] in post_state['excluded']
        assert old_ack_source in post_state['excluded']
        assert post_state['old_ack'][0]['text'] == model.recalled
        assert post_state['old_terminal']['outcome'] == 'completed'
        if change == 'expiry':
            assert post_state['versions'] == [saved]
            assert post_state['tables']['invalidations'] == []
        else:
            assert any(row['source'] == first['basis']['input_id']
                       for row in post_state['tables']['invalidations'])
            assert json.loads(
                post_state['tables']['operations'][0]['receipt'],
            )['status'] == 'committed'
        post_change = memory.store.backup_to(tmp_path / 'after-change.sqlite3')
        old_session, authority_hash = mirror.session_id, sha256(authority.read_bytes()).hexdigest()
    finally:
        memory.close()
        mirror.close()

    # A pre-change business snapshot must never replace the authoritative database.
    # The existing API forbids overwrite; no invented rollback/restore platform is used.
    stale = SQLiteReceivedStore(pre_change)
    with pytest.raises(ValueError, match='Backup destination already exists'):
        stale.backup_to(authority)
    assert sha256(authority.read_bytes()).hexdigest() == authority_hash

    restore_source = post_change  # Restore only a snapshot that includes the committed change.
    snapshot_hash = sha256(restore_source.read_bytes()).hexdigest()
    restored_path = SQLiteReceivedStore(restore_source).backup_to(
        tmp_path / 'restored' / 'memory.sqlite3')
    assert sha256(restore_source.read_bytes()).hexdigest() == snapshot_hash
    restored_model = RestoredMemory()
    restored_model.now = reopened_at
    memory, mirror, runtime, session = build(tmp_path / 'restored', restored_model)
    try:
        assert memory.path == restored_path and memory.path != authority
        session.new_session()
        assert mirror.session_id != old_session and mirror.history()['messages'] == []
        restored_state = memory_audit(reopened_us)
        resolutions = []
        resolve = runtime.resolve

        def resolve_once(claim):
            resolutions.append(claim.input_id)
            return resolve(claim)

        runtime.resolve = resolve_once
        event = turn(runtime, session, query)
        # These are actual default worker, review, display and persistent ACK proofs.
        assert resolutions == [event['basis']['input_id']]
        assert restored_model.query_actions == ['绘画 画画 散步 兴趣 爱好']
        assert len(restored_model.primary) == 2 and len(restored_model.review_packets) == 1
        completed_user_source(event)
        assert_confirmed(event)
        expected = restored_model.updated_reply if change == 'correct' else restored_model.missing
        assert event['display_text'] == expected
        assert original not in event['display_text'] and '画画' not in event['display_text']
        assert mirror.history()['messages'][-1]['content'] == expected
        facts = event['readset']['facts']
        assert len(facts) == (1 if change == 'correct' else 0)
        if change == 'correct':
            current, = post_state['eligible']
            assert (facts[0]['record_id'], facts[0]['version'], facts[0]['source_id']) == (
                current['record_id'], 2, current['source_id'])
            assert current['content'] == updated
        assert saved['source_id'] not in {fact['source_id'] for fact in facts}
        # New session isolation alone is not revocation proof.
        assert event['readset']['history'] == []
        evidence = '\n'.join(item['content'] for item in restored_model.primary[-1][1:-1])
        assert original not in evidence and model.recalled not in evidence
        review = restored_model.review_packets[-1]
        assert review['user_text'] == query and review['final']
        assert review['candidate_prefix'] == expected
        assert review['basis']['input_id'] == event['basis']['input_id']
        reviewed = json.loads(review['context'])['memories']
        assert [(fact['record_id'], fact['version'], fact['source_id']) for fact in facts] == [
            (fact['record_id'], fact['version'], fact['source_id']) for fact in reviewed]
        # The full source/version/invalidation and ACK audit survived the snapshot copy.
        assert restored_state == post_state
        after_query = memory_audit(reopened_us)
        assert after_query['tables'] == post_state['tables']
        assert after_query['versions'] == post_state['versions']
        assert after_query['eligible'] == post_state['eligible']
        assert first['basis']['input_id'] in after_query['excluded']
        assert old_ack_source in after_query['excluded']
        assert after_query['old_ack'] == post_state['old_ack']
        assert after_query['old_terminal'] == post_state['old_terminal']
        with memory.store.transaction(write=False) as db:
            assert memory.store._readset_current(db, event['readset'], now_us=reopened_us)
        assert (
            memory.store.snapshot(memory.companion).write_version
            == event['readset']['state_version']
        )
        assert sha256(authority.read_bytes()).hexdigest() == authority_hash
    finally:
        memory.close()
        mirror.close()
