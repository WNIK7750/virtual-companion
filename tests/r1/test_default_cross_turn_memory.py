"""Current default multi-turn evidence flow; all model decisions are recordings."""

import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from test_default_text_delivery import Model, wait_for
from unit_review_recording import bound_unit_packets as bound_unit_packets
from unit_review_recording import recorded_review_packet
from virtual_companion.cognition.background_worker import BackgroundScheduler
from virtual_companion.text.application import TextSession
from virtual_companion.text.chronology import utc_microseconds
from virtual_companion.text.default_runtime import DefaultTextRuntime
from virtual_companion.text.memory_management import MemoryManagementService
from virtual_companion.text.memory_runtime import (
    DEFAULT_COMPANION_ID,
    TextMemoryRuntime,
    _fake_memory_result,
)
from virtual_companion.text.sqlite_store import SQLiteConversationStore

ORIGINAL = '我喜欢安静的音乐'
UPDATED = '我喜欢摇滚音乐'
QUERY = '我喜欢什么音乐？'
CORRECT = f'把{ORIGINAL}改成{UPDATED}。'
FORGET = f'请忘记{ORIGINAL}这件事。'


class Scenario(Model):
    def __init__(self, *, expires=False, original=ORIGINAL, query=QUERY,
                 recalled='你喜欢安静的音乐。', predicate='music', updated=UPDATED,
                 updated_reply='你喜欢摇滚音乐。', missing='没有有效的音乐偏好记录。'):
        super().__init__()
        self.original, self.query = original, query
        self.recalled, self.predicate = recalled, predicate
        self.updated, self.updated_reply, self.missing = updated, updated_reply, missing
        self.correct = f'把{original}改成{updated}。'
        self.forget = f'请忘记{original}这件事。'
        self.now = datetime(2026, 9, 27, 1, tzinfo=UTC)
        self.expiry = self.now + timedelta(days=1) if expires else None
        self.primary, self.worker_calls = [], []

    def clock(self):
        return utc_microseconds(self.now)

    def stream(self, messages, cancelled):
        if '你只审查候选文字' in messages[0]['content']:
            yield from super().stream(messages, cancelled)
            return
        self.calls.append(deepcopy(messages))
        self.primary.append(deepcopy(messages))
        if '本次只能返回answer' not in messages[0]['content']:
            yield '{"action":"memory"}'
            return
        evidence = '\n'.join(m['content'] for m in messages[1:-1])
        text = (self.updated_reply if self.updated in evidence else
                self.recalled if self.original in evidence else self.missing)
        yield json.dumps({'action': 'answer', 'text': text}, ensure_ascii=False)

    def invoke(self, stage, messages, schema, cancelled):
        payload = recorded_review_packet(messages)
        self.worker_calls.append((stage, deepcopy(payload)))
        source = payload.get('source', '')
        text = source.get('text', '') if isinstance(source, dict) else source
        if stage == 'memory_command' and text in {self.correct, self.forget}:
            target = schema['properties']['targetIds']['items']['enum'][0]
            replacement = self.updated if text == self.correct else ''
            start = text.index(replacement) if replacement else None
            return dict(action='correct' if replacement else 'forget',
                        requestedScope='single_record', targetIds=[target],
                        evidenceQuote=text, quoteStart=0, quoteEnd=len(text),
                        replacementQuote=replacement, replacementStart=start,
                        replacementEnd=start + len(replacement) if replacement else None,
                        coverage='pure_command', unresolved=[])
        if stage == 'memory_command_support':
            return dict(meaning='explicit_single_correction' if text == self.correct
                        else 'explicit_single_forget', verdict='supported')
        if stage == 'fact_extract' and text == self.original:
            value = dict(operation='add', type='short_term' if self.expiry else 'long_term',
                         content=text, evidenceTurnId=source['id'], evidenceQuote=text,
                         supersedesAtomId=None, expiresAt=int(self.expiry.timestamp() * 1000)
                         if self.expiry else None,
                         factKey=dict(subject='user', predicate=self.predicate))
            # Both host-bound and verbatim older fact schemas have the same source contract.
            variants = schema['properties']['operations']['items']['oneOf']
            if not any('content' in v['properties'] for v in variants):
                value.pop('content')
            return {'operations': [value]}
        if stage == 'fact_support':
            return dict(intent='assertion', verdict='supported')
        if stage == 'exchange_intent':
            result = _fake_memory_result(stage, messages, schema, cancelled)
            if result['acts'][0]['evidenceQuote'] == self.query:
                result['acts'][0]['intent'] = 'question'
            return result
        if stage == 'exchange_extract':
            sources = payload['sources']
            current = sources[-1]
            if current['text'] == self.query and current['role'] == 'user':
                return {'operations': [dict(operation='add', type='response_obligation',
                    evidenceQuote=self.query,
                    supersedesAtomId=None, expiresAt=None)]}
        if stage == 'exchange_support':
            return dict(intent='question', verdict='supported')
        return _fake_memory_result(stage, messages, schema, cancelled)


def build(tmp_path, model, *, companion=DEFAULT_COMPANION_ID):
    memory = TextMemoryRuntime(tmp_path / 'memory.sqlite3', companion, model=model, fake=True)
    memory.scheduler = BackgroundScheduler(memory.store, user_invoke=model.invoke,
        assistant_invoke=model.invoke, clock=model.clock, resolution_protocol=True,
        resolution_strategy='intent_first', memory_commands=True)
    mirror = SQLiteConversationStore(tmp_path / companion, companion_id=companion)
    runtime = DefaultTextRuntime(memory, mirror, model, binding={'test': 'cross-turn'},
                                 clock=model.clock, delivery_timeout=0.8)
    session = TextSession(model, mirror, {}, prepare_turn=runtime.prepare_turn,
                          before_start=runtime.before_start)
    return memory, mirror, runtime, session


def turn(runtime, session, text):
    session.send(text)
    wait_for(lambda: runtime.poll()['event'] or not session.snapshot()['busy'])
    event = runtime.poll()['event']
    assert event is not None, session.snapshot()
    runtime.ack(dict(sequence=event['sequence'], fingerprint=event['fingerprint'],
                     host_receipt='recorded-visible-' + str(uuid4())))
    wait_for(lambda: not session.snapshot()['busy'])
    assert session.snapshot()['attempt']['status'] == 'completed', session.snapshot()
    return event


@pytest.mark.parametrize('change', [None, 'correct', 'forget'])
def test_supported_normal_statement_completes_background_and_reopens_with_current_readset(
    tmp_path, change,
):
    original, query, updated = '我喜欢画画', '空闲时能怎样放松？', '我偏爱散步'
    reply = '可以随手画一片叶子。你喜欢画画，先勾轮廓，慢慢放松。'
    updated_reply = '可以出门慢慢走一会儿。你现在偏爱散步，不用赶时间。'
    missing = '没有可核验的兴趣偏好记录，可以先尝试短暂休息。'

    class StatementScenario(Scenario):
        def stream(self, messages, cancelled):
            if messages[-1]['content'] == self.original:
                self.calls.append(deepcopy(messages))
                self.primary.append(deepcopy(messages))
                yield json.dumps({'action': 'answer', 'text': '可以先随手勾几笔。'},
                                 ensure_ascii=False)
            else:
                yield from super().stream(messages, cancelled)

    model = StatementScenario(original=original, query=query, recalled=reply, predicate='hobby',
                              updated=updated, updated_reply=updated_reply, missing=missing)
    memory, mirror, runtime, session = build(tmp_path, model)
    try:
        first = turn(runtime, session, original)
        assert first['display_text'] == '可以先随手勾几笔。'
        assert not model.worker_calls  # Ordinary response precedes background processing.
        result = memory.scheduler.user.run_one(memory.companion, 'statement-background')
        receipt = json.loads(result)
        assert receipt['operations'] == 1 and receipt['resolution']['no_exchange']
        assert receipt['resolution']['source_id'] == first['basis']['input_id']
        assert [stage for stage, _ in model.worker_calls] == [
            'memory_command', 'exchange_intent', 'fact_extract', 'fact_support',
            'exchange_extract']
        assert memory.store.memory_status(memory.companion) == 'current'
        assert all(source['role'] == 'assistant' for source in
                   memory.store.background_sources(memory.companion))
        saved, = memory.store.recall(memory.companion, now_us=model.clock())
        assert saved['content'] == original and saved['source_id'] == first['basis']['input_id']
        assert saved['version'] == 1 and saved['event_time'] is None
        assert json.loads(saved['reason'])['support']['verdict'] == 'supported'
        assert memory.store.history(memory.companion, saved['record_id']) == [saved]
        page = MemoryManagementService(memory.store, memory.companion,
                                       clock=model.clock).list_memories()
        assert page['items'][0]['content'] == original and page['items'][0]['version'] == 1
        assert page['items'][0]['event_time'] is None and not page['attention']
        if change is not None:
            turn(runtime, session, model.correct if change == 'correct' else model.forget)
        old_session = mirror.session_id
    finally:
        memory.close()
        mirror.close()

    model.now += timedelta(days=1)
    memory, mirror, runtime, session = build(tmp_path, model)
    try:
        session.new_session()
        assert mirror.session_id != old_session and not mirror.history()['messages']
        event = turn(runtime, session, query)
        facts = event['readset']['facts']
        evidence = '\n'.join(item['content'] for item in model.primary[-1][1:-1])
        assert event['display_text'] == (reply if change is None else
                                        updated_reply if change == 'correct' else missing)
        assert len(facts) == (0 if change == 'forget' else 1)
        if change is None:
            assert facts[0]['source_id'] == saved['source_id'] and facts[0]['version'] == 1
        else:
            assert original not in evidence
            assert saved['source_id'] not in {fact['source_id'] for fact in facts}
            assert saved['source_id'] in memory.store.ineffective_history_sources(
                memory.companion, now_us=model.clock())
        if change == 'correct':
            assert facts[0]['record_id'] == saved['record_id'] and facts[0]['version'] == 2
        if facts:
            assert '发生时间：不详' in evidence
            expected_content = original if change is None else updated
            assert expected_content in evidence
        assert '2026-09-27' not in evidence
    finally:
        memory.close()
        mirror.close()


def test_default_recalls_a_rephrased_old_preference_after_reopen_in_a_new_session(tmp_path):
    original, query = '我喜欢画画', '画点什么比较好？'
    reply = '既然你喜欢画画，可以试试画一片叶子，先从轮廓开始。'
    model = Scenario(original=original, query=query, recalled=reply, predicate='drawing')
    model.now = datetime(2025, 1, 1, tzinfo=UTC)
    memory, mirror, runtime, session = build(tmp_path, model)
    first = turn(runtime, session, original)
    saved = memory.store.recall(memory.companion, now_us=model.clock())[0]
    assert saved['content'] == original and saved['event_time'] is None
    old_session = mirror.session_id
    memory.close()
    mirror.close()

    model.now += timedelta(days=600)
    memory, mirror, runtime, session = build(tmp_path, model)
    try:
        session.new_session()
        assert mirror.session_id != old_session
        event = turn(runtime, session, query)
        memories = event['readset']['facts']
        assert len(memories) == 1
        assert memories[0]['source_id'] == first['basis']['input_id']
        evidence = '\n'.join(item['content'] for item in model.primary[-1][1:-1])
        assert original in evidence
        assert '发生时间：不详' in evidence
        assert '2025-01-01' not in evidence  # Recording time is not the experience time.
        assert event['display_text'] == reply
    finally:
        memory.close()
        mirror.close()


@pytest.mark.parametrize('change', [None, 'correct', 'forget'])
def test_default_semantic_query_uses_only_current_candidates_after_cross_day_reopen(
    tmp_path, change,
):
    original, query, updated = '我喜欢画画', '空闲时能怎样放松？', '我偏爱散步'
    assert not set(original) & set(query)
    reply = '可以随手画一片叶子。你喜欢画画，先勾轮廓，慢慢放松。'
    updated_reply = '可以出门慢慢走一会儿。你现在偏爱散步，不用赶时间。'
    missing = '没有可核验的兴趣偏好记录，可以先尝试短暂散步。'
    model = Scenario(original=original, query=query, recalled=reply, predicate='hobby',
                     updated=updated, updated_reply=updated_reply, missing=missing)
    model.now = datetime(2025, 1, 1, tzinfo=UTC)
    memory, mirror, runtime, session = build(tmp_path, model)
    first = turn(runtime, session, original)
    old_record = memory.store.recall(memory.companion, now_us=model.clock())[0]
    assert old_record['content'] == original and old_record['event_time'] is None
    if change is not None:
        turn(runtime, session, model.correct if change == 'correct' else model.forget)
    old_session = mirror.session_id
    memory.close()
    mirror.close()

    model.now += timedelta(days=600)
    memory, mirror, runtime, session = build(tmp_path, model)
    try:
        session.new_session()
        assert mirror.session_id != old_session
        before_calls = len(model.primary)
        event = turn(runtime, session, query)
        assert len(model.primary) - before_calls == 2  # Existing demand + final answer.
        initial_evidence = '\n'.join(item['content'] for item in model.primary[-2][1:-1])
        assert original not in initial_evidence and updated not in initial_evidence
        facts = event['readset']['facts']
        evidence = '\n'.join(item['content'] for item in model.primary[-1][1:-1])
        expected = reply if change is None else updated_reply if change == 'correct' else missing
        assert event['display_text'] == expected
        assert len(facts) == (0 if change == 'forget' else 1)
        if change is None:
            assert facts[0]['source_id'] == first['basis']['input_id']
        else:
            assert original not in evidence
            assert old_record['source_id'] not in {fact['source_id'] for fact in facts}
        if facts:
            assert '发生时间：不详' in evidence
        assert '2025-01-01' not in evidence
    finally:
        memory.close()
        mirror.close()


@pytest.mark.parametrize('change', ['correct', 'forget', 'expiry'])
def test_current_default_memory_and_derived_history_stay_invalid_after_reopen(tmp_path, change):
    model = Scenario(expires=change == 'expiry')
    memory, mirror, runtime, session = build(tmp_path, model)
    first = turn(runtime, session, ORIGINAL)
    fact = memory.store.recall(memory.companion, now_us=model.clock())[0]
    assert fact['content'] == ORIGINAL and fact.get('event_time') is None
    assert fact['source_id'] == first['basis']['input_id']
    recalled = turn(runtime, session, QUERY)
    assert recalled['display_text'] == '你喜欢安静的音乐。'
    # Source-aware readset is obtained from the actual default request/approval.
    assert recalled['readset']['facts']
    if change == 'expiry':
        version = memory.store.snapshot(memory.companion).write_version
        expires_at = datetime.fromisoformat(fact['expires_at'])
        expires_us = utc_microseconds(expires_at)
        model.now = expires_at - timedelta(microseconds=1)
        assert model.clock() == expires_us - 1
        assert memory.store.recall(memory.companion, now_us=model.clock())
        model.now = expires_at
        assert model.clock() == expires_us
        assert memory.store.recall(memory.companion, now_us=model.clock()) == []
        boundary_excluded = memory.store.ineffective_history_sources(
            memory.companion, now_us=model.clock())
        assert first['basis']['input_id'] in boundary_excluded
        assert 'segment-' + recalled['fingerprint'] in boundary_excluded
        assert memory.store.snapshot(memory.companion).write_version == version
        model.now += timedelta(days=2)
        assert memory.store.recall(memory.companion, now_us=model.clock()) == []
        excluded = memory.store.ineffective_history_sources(memory.companion, now_us=model.clock())
        assert first['basis']['input_id'] in excluded
        assert 'segment-' + recalled['fingerprint'] in excluded
        assert memory.store.snapshot(memory.companion).write_version == version
    else:
        turn(runtime, session, CORRECT if change == 'correct' else FORGET)
    # Process actual pending ACK sources through the production scheduler before
    # reopening. The spoken old fact must never become a fresh user assertion.
    before_background = len(model.worker_calls)
    assert memory.scheduler.drain('cross-turn-background', limit=16) > 0
    # All existing assistant units depend on the now-invalid fact: deterministic
    # retirement needs no model judgment and must not certify semantic coverage.
    assert len(model.worker_calls) == before_background
    discarded = memory.store.background_sources(memory.companion)
    assert discarded and all(row['status'] == 'invalidated' for row in discarded)
    assert memory.store.next_background_source(now_us=model.clock()) is None
    assert all(row['content'] != ORIGINAL for row in
               memory.store.recall(memory.companion, now_us=model.clock()))
    memory.close()
    mirror.close()
    memory, mirror, runtime, session = build(tmp_path, model)
    latest = turn(runtime, session, QUERY)
    evidence = '\n'.join(item['content'] for item in model.primary[-1][1:-1])
    assert ORIGINAL not in evidence and '你喜欢安静的音乐' not in evidence
    assert latest['display_text'] == ('你喜欢摇滚音乐。' if change == 'correct'
                                      else '没有有效的音乐偏好记录。')
    if change == 'forget':
        versions = memory.store.history(memory.companion, fact['record_id'])
        assert [(row['version'], row['status']) for row in versions] == [
            (1, 'active'), (2, 'revoked')]
        assert latest['readset']['facts'] == []
    # Display history and audit evidence survive correction, forgetting and TTL.
    assert any(m['content'] == '你喜欢安静的音乐。' for m in mirror.history()['messages'])
    assert memory.store.segments(memory.companion, recalled['basis']['input_id'])[0]['text'] \
        == '你喜欢安静的音乐。'
    mirror.close()
    memory.close()
    other, other_mirror, other_runtime, other_session = build(
        tmp_path, model, companion=str(uuid4()))
    isolated = turn(other_runtime, other_session, QUERY)
    assert isolated['display_text'] == '没有有效的音乐偏好记录。'
    other.close()
    other_mirror.close()


def test_default_query_at_exact_expiry_excludes_fact_and_derived_ack_without_ttl_write(tmp_path):
    model = Scenario(expires=True)
    memory, mirror, runtime, session = build(tmp_path, model)
    try:
        first = turn(runtime, session, ORIGINAL)
        fact, = memory.store.recall(memory.companion, now_us=model.clock())
        assert fact['source_id'] == first['basis']['input_id']
        recalled = turn(runtime, session, QUERY)
        old_ack_source = 'segment-' + recalled['fingerprint']
        old_reply = recalled['display_text']

        expires_at = datetime.fromisoformat(fact['expires_at'])
        expires_us = utc_microseconds(expires_at)
        model.now = expires_at
        assert model.clock() == expires_us
        write_version = memory.store.snapshot(memory.companion).write_version
        assert memory.store.recall(memory.companion, now_us=model.clock()) == []
        assert memory.store.snapshot(memory.companion).write_version == write_version

        # The source version and acknowledged answer remain available for audit.
        assert memory.store.history(memory.companion, fact['record_id']) == [fact]
        assert memory.store.segments(
            memory.companion, recalled['basis']['input_id'],
        )[0]['text'] == old_reply
        assert any(item['content'] == old_reply for item in mirror.history()['messages'])

        event = turn(runtime, session, QUERY)
        assert model.clock() == expires_us
        evidence = '\n'.join(item['content'] for item in model.primary[-1][1:-1])
        assert ORIGINAL not in evidence
        assert old_reply not in evidence
        assert event['display_text'] == model.missing
        assert event['readset']['facts'] == []
        history_sources = {item['id'] for item in event['readset']['history']}
        assert fact['source_id'] not in history_sources
        assert old_ack_source not in history_sources

        excluded = memory.store.ineffective_history_sources(
            memory.companion, now_us=expires_us,
        )
        assert fact['source_id'] in excluded
        assert old_ack_source in excluded
        assert memory.store.recall(memory.companion, now_us=expires_us) == []
        assert memory.store.history(memory.companion, fact['record_id']) == [fact]
        assert (
            memory.store.snapshot(memory.companion).write_version
            == event['readset']['state_version']
        )
        assert memory.store.segments(
            memory.companion, recalled['basis']['input_id'],
        )[0]['text'] == old_reply
    finally:
        mirror.close()
        memory.close()


def test_default_ninth_fact_is_ingested_and_recalled_by_one_rephrased_query_after_reopen(
    tmp_path,
):
    original, query = '我喜欢画画', '空闲时能怎样放松？'
    distractors = {
        '空闲时间我会整理书柜': 'bookcase_order',
        '空闲时间我会擦窗户': 'window_cleaning',
        '空闲时间我会给盆栽浇水': 'plant_watering',
        '空闲时间我会叠衣服': 'clothes_folding',
        '空闲时间我会清理鞋柜': 'shoe_cabinet_order',
        '空闲时间我会打扫阳台': 'balcony_cleaning',
        '空闲时间我会整理抽屉': 'drawer_order',
        '空闲时间我会收拾工具盒': 'tool_box_order',
    }
    assert len(set(distractors.values()) | {'hobby'}) == 9
    assert not set(original) & set(query)

    class IngestedCatalog(Scenario):
        def __init__(self):
            super().__init__(original=original, query=query, predicate='hobby',
                             recalled='你喜欢画画，可以画一片叶子放松。')
            self.review_packets, self.query_actions = [], []

        def stream(self, messages, cancelled):
            if '你只审查候选文字' in messages[0]['content']:
                self.review_packets.append(recorded_review_packet(messages))
                yield from super().stream(messages, cancelled)
                return
            text = messages[-1]['content']
            final = '本次只能返回answer' in messages[0]['content']
            if text == query and not final:
                self.calls.append(deepcopy(messages))
                self.primary.append(deepcopy(messages))
                expression = '绘画 画画 兴趣 爱好'
                self.query_actions.append(expression)
                yield json.dumps({'action': 'memory', 'query': expression}, ensure_ascii=False)
                return
            if text in {*distractors, original} and final:
                self.calls.append(deepcopy(messages))
                self.primary.append(deepcopy(messages))
                yield json.dumps({'action': 'answer', 'text': '这条已保存。'}, ensure_ascii=False)
                return
            yield from super().stream(messages, cancelled)

        def invoke(self, stage, messages, schema, cancelled):
            payload = recorded_review_packet(messages)
            source = payload.get('source', '')
            text = source.get('text', '') if isinstance(source, dict) else source
            if stage == 'fact_extract' and text in distractors:
                self.worker_calls.append((stage, deepcopy(payload)))
                operation = dict(operation='add', type='long_term', content=text,
                    evidenceTurnId=source['id'], evidenceQuote=text,
                    supersedesAtomId=None, expiresAt=None,
                    factKey=dict(subject='user', predicate=distractors[text]))
                variants = schema['properties']['operations']['items']['oneOf']
                if not any('content' in variant['properties'] for variant in variants):
                    operation.pop('content')
                return {'operations': [operation]}
            return super().invoke(stage, messages, schema, cancelled)

    model = IngestedCatalog()
    model.now = datetime(2025, 1, 1, tzinfo=UTC)
    memory, mirror, runtime, session = build(tmp_path, model)

    def assert_confirmed(event):
        input_id = event['basis']['input_id']
        segment, = memory.store.segments(memory.companion, input_id)
        assert segment['status'] == 'confirmed'
        assert segment['fingerprint'] == event['fingerprint']
        assert segment['host_receipt'].startswith('recorded-visible-')
        assert segment['presentation']['review']['model_approved'] is True
        state = memory.store.stream_state(memory.companion, input_id)
        assert state['outcome'] == 'completed'
        assert state['approved_units'] == state['confirmed_units'] == 1
        assert state['pending_units'] == state['unknown_units'] == 0
        assert state['confirmed_text'] == event['display_text']

    events, saved = [], []
    try:
        database = memory.path
        assert database == (tmp_path / 'memory.sqlite3').resolve()
        for index, (text, predicate) in enumerate(
            [*distractors.items(), (original, 'hobby')], 1,
        ):
            worker_before = len(model.worker_calls)
            event = turn(runtime, session, text)
            events.append(event)
            assert_confirmed(event)
            input_id = event['basis']['input_id']
            assert memory.store.inspect(memory.companion, input_id)['text'] == text
            catalog = memory.store.recall(memory.companion, now_us=model.clock(), limit=100)
            assert len(catalog) == index
            fact, = [row for row in catalog if row['source_id'] == input_id]
            assert fact['content'] == text and fact['status'] == 'active'
            assert fact['version'] == 1 and fact['expires_at'] is None
            assert fact['fact_key']['subject'] == 'user'
            assert fact['fact_key']['predicate'] == predicate
            assert memory.store.history(memory.companion, fact['record_id']) == [fact]
            assert json.loads(fact['reason'])['support']['verdict'] == 'supported'
            with memory.store.transaction(write=False) as db:
                source = db.execute(
                    "SELECT s.status,w.receipt FROM background_sources s JOIN background_work w "
                    "ON s.companion=w.companion AND s.job=w.job WHERE s.companion=? "
                    "AND s.source=? AND s.role='user'",
                    (memory.companion, input_id),
                ).fetchone()
            assert source['status'] == 'completed'
            receipt = json.loads(source['receipt'])
            assert receipt['operations'] == 1
            assert receipt['resolution']['source_id'] == input_id
            assert [stage for stage, _ in model.worker_calls[worker_before:]] == [
                'memory_command', 'exchange_intent', 'fact_extract', 'fact_support',
                'exchange_extract']
            assert memory.store.memory_status(memory.companion) == 'current'
            saved.append(fact)
        assert len({row['record_id'] for row in saved}) == 9
        assert len({tuple(row['fact_key'].values()) for row in saved}) == 9
        target = saved[-1]
        assert catalog[8] == target and target['content'] == original
        assert all(row['record_id'] != target['record_id'] for row in
                   memory.store.recall(memory.companion, now_us=model.clock(), limit=8))
        old_session, old_clock = mirror.session_id, model.clock()
    finally:
        memory.close()
        mirror.close()

    model.now += timedelta(days=600)
    assert model.clock() - old_clock >= 24 * 60 * 60 * 1_000_000
    memory, mirror, runtime, session = build(tmp_path, model)
    try:
        assert memory.path == database
        session.new_session()
        assert mirror.session_id != old_session and mirror.history()['messages'] == []
        assert memory.store.recall(memory.companion, now_us=model.clock(), limit=100) == saved
        for event in events:
            assert_confirmed(event)
        resolutions = []
        resolve = runtime.resolve

        def resolve_once(claim):
            resolutions.append(claim.input_id)
            return resolve(claim)

        runtime.resolve = resolve_once
        primary_before, review_before = len(model.primary), len(model.review_packets)
        event = turn(runtime, session, query)
        assert resolutions == [event['basis']['input_id']]
        assert len(model.query_actions) == 1
        assert len(model.primary) - primary_before == 2
        assert len(model.review_packets) - review_before == 1
        assert model.primary[-1][-1]['content'] == query
        assert event['readset']['history'] == []
        facts = event['readset']['facts']
        # The default adds lower-ranked eligible candidates within its eight-row cap.
        assert len(facts) == 8 and facts[0]['record_id'] == target['record_id']
        assert facts[0]['source_id'] == events[-1]['basis']['input_id']
        assert facts[0]['version'] == target['version'] == 1
        assert sum(row['record_id'] == target['record_id'] for row in facts) == 1
        assert {row['record_id'] for row in facts} <= {row['record_id'] for row in saved}
        evidence = '\n'.join(item['content'] for item in model.primary[-1][1:-1])
        assert original in evidence
        assert event['display_text'] == model.recalled
        assert all(text not in event['display_text'] for text in distractors)
        review = model.review_packets[-1]
        assert review['user_text'] == query and review['final']
        assert review['candidate_prefix'] == event['display_text']
        assert review['basis']['input_id'] == event['basis']['input_id']
        reviewed_facts = json.loads(review['context'])['memories']
        assert reviewed_facts[0]['record_id'] == target['record_id']
        assert reviewed_facts[0]['source_id'] == target['source_id']
        assert_confirmed(event)
        assert mirror.history()['messages'][-1]['content'] == model.recalled
        assert not memory.store.memory_barrier(memory.companion)
        assert memory.store.recall(memory.companion, now_us=model.clock(), limit=100) == saved
    finally:
        memory.close()
        mirror.close()


@pytest.mark.parametrize('explicit_past', [False, True])
@pytest.mark.parametrize('change', [None, 'correct', 'forget', 'expiry'])
def test_default_reopened_tail_fact_is_selected_by_the_one_memory_query(
    tmp_path, monkeypatch, explicit_past, change,
):
    from virtual_companion.boundary.contracts import Scope
    from virtual_companion.boundary.memory import FactKey, MemoryMutation

    query = ('按以前的兴趣怎样利用空闲时间放松？' if explicit_past else '空闲时能怎样放松？')
    original, updated = '我喜欢画画', '我偏爱素描'
    assert not set(query) & set(original)

    class QueryScenario(Scenario):
        def stream(self, messages, cancelled):
            query_stage = ('你只审查候选文字' not in messages[0]['content']
                           and messages[-1]['content'] == self.query)
            final = '本次只能返回answer' in messages[0]['content']
            if query_stage and final and getattr(self, 'fail_final_once', False):
                self.calls.append(deepcopy(messages))
                self.primary.append(deepcopy(messages))
                self.fail_final_once = False
                raise OSError('synthetic final transport failure after source completion')
            if query_stage and not final:
                self.calls.append(deepcopy(messages))
                self.primary.append(deepcopy(messages))
                yield json.dumps({'action': 'memory', 'query': '绘画 画画 素描 爱好'},
                                 ensure_ascii=False)
                return
            yield from super().stream(messages, cancelled)

        def invoke(self, stage, messages, schema, cancelled):
            payload = recorded_review_packet(messages)
            source = payload.get('source', '')
            text = source.get('text', '') if isinstance(source, dict) else source
            if stage == 'memory_command' and text in {self.correct, self.forget}:
                self.worker_calls.append((stage, deepcopy(payload)))
                replacement = self.updated if text == self.correct else ''
                start = text.index(replacement) if replacement else None
                return dict(action='correct' if replacement else 'forget',
                    requestedScope='single_record', targetIds=[self.target_record],
                    evidenceQuote=text, quoteStart=0, quoteEnd=len(text),
                    replacementQuote=replacement, replacementStart=start,
                    replacementEnd=start + len(replacement) if replacement else None,
                    coverage='pure_command', unresolved=[])
            return super().invoke(stage, messages, schema, cancelled)

    model = QueryScenario(original=original, query=query, recalled='可以画一片叶子。',
                          predicate='hobby', updated=updated, updated_reply='可以练习一张素描。',
                          missing='没有有效的兴趣偏好记录。')
    model.now = datetime(2025, 1, 1, tzinfo=UTC)
    model.expiry = model.now + timedelta(days=1) if change == 'expiry' else None
    memory, mirror, runtime, session = build(tmp_path, model)
    try:
        owner = memory.store.claim_owner(memory.companion, 'synthetic-catalog', expected_epoch=0)
        for index in range(8):
            memory.store.mutate(MemoryMutation(
                Scope(memory.companion, mirror.session_id), f'seed-{index}', 'assert_fact',
                f'other-{index}', f'空闲时间合成事项 {index}', 'manual', f'noise-source-{index}',
                'synthetic distractor', model.now, FactKey('user', f'other-{index}', 'profile'),
            ), owner, expected_version=memory.store.snapshot(memory.companion).write_version)
        original_event = turn(runtime, session, original)
        catalog = memory.store.recall(memory.companion, now_us=model.clock(), limit=100)
        saved = catalog[8]
        assert saved['content'] == original
        model.target_record = saved['record_id']
        if change in {'correct', 'forget'}:
            turn(runtime, session, model.correct if change == 'correct' else model.forget)
        old_session = mirror.session_id
    finally:
        memory.close()
        mirror.close()

    model.now += timedelta(days=600)
    memory, mirror, runtime, session = build(tmp_path, model)
    try:
        session.new_session()
        assert mirror.session_id != old_session and mirror.history()['messages'] == []
        calls_before = len(model.primary)
        source_calls = []
        original_resolve = runtime.resolve

        def resolve(claim):
            source_calls.append(claim.input_id)
            return original_resolve(claim)

        runtime.resolve = resolve
        if explicit_past and change is None:
            routes = []
            original_prepare = runtime.prepare

            def observe_prepare(*args, **kwargs):
                stage = original_prepare(*args, **kwargs)
                routes.append((json.loads(stage.request.context.payload), kwargs['final']))
                return stage

            monkeypatch.setattr(runtime, 'prepare', observe_prepare)
            model.fail_final_once = True
            session.send(query)
            wait_for(lambda: not session.snapshot()['busy'])
            assert session.snapshot()['attempt']['status'] == 'failed'
            assert len(model.primary) - calls_before == 2
            assert not memory.store.segments(memory.companion, runtime.claim.input_id)
            assert memory.store.memory_status(memory.companion) == 'current'
            calls_before, worker_before = len(model.primary), len(model.worker_calls)
            source_calls.clear()
            session.retry()
            wait_for(lambda: runtime.poll()['event'] or not session.snapshot()['busy'])
            event = runtime.poll()['event']
            assert event is not None, session.snapshot()
            runtime.ack(dict(sequence=event['sequence'], fingerprint=event['fingerprint'],
                             host_receipt='synthetic-reopened-tail-retry-visible'))
            wait_for(lambda: not session.snapshot()['busy'])
            assert session.snapshot()['attempt']['status'] == 'completed'
            initial, final_stage = routes[-2]
            assert initial['memory_status'] == 'current' and len(initial['memories']) == 8
            assert all(row['record_id'] != model.target_record for row in initial['memories'])
            assert not final_stage  # The generator requests the one focused recall.
            assert initial['memory_access'] == 'not_accessed'
            assert len(model.worker_calls) == worker_before  # Completed source, no repeat effects.
        else:
            event = turn(runtime, session, query)
        assert source_calls == [event['basis']['input_id']]
        facts = event['readset']['facts']
        assert len(facts) == 8 and event['readset']['history'] == []
        target = [fact for fact in facts if fact['record_id'] == model.target_record]
        if change in {'forget', 'expiry'}:
            assert target == [] and event['display_text'] == model.missing
        else:
            assert len(target) == 1 and target[0]['version'] == (2 if change == 'correct' else 1)
            assert event['display_text'] == (
                model.updated_reply if change == 'correct' else model.recalled)
            if change is None:
                assert target[0]['source_id'] == original_event['basis']['input_id']
        assert len(model.primary) - calls_before == 2  # One query action and one final answer.
        assert model.primary[-1][-1]['content'] == query
        assert not memory.store.memory_barrier(memory.companion)
        assert mirror.history()['messages'][-1]['content'] == event['display_text']
    finally:
        memory.close()
        mirror.close()


def test_g8_current_query_reads_corrected_head_and_settles_only_from_acknowledged_answer(tmp_path):
    query = '按我现在的音乐偏好，有什么适合听着放松的？'
    previous = '依据以前的偏好，有什么适合听着放松的？'
    updated = '我现在更偏爱轻快的音乐'

    class RecordedQuery(Scenario):
        def stream(self, messages, cancelled):
            if ('你只审查候选文字' not in messages[0]['content']
                    and messages[-1]['content'] == query
                    and '本次只能返回answer' not in messages[0]['content']):
                self.calls.append(deepcopy(messages))
                self.primary.append(deepcopy(messages))
                yield json.dumps({'action': 'memory',
                    'query': '音乐偏好 放松 听歌 歌单 曲风 轻音乐 习惯'}, ensure_ascii=False)
                return
            yield from super().stream(messages, cancelled)

        def invoke(self, stage, messages, schema, cancelled):
            data = recorded_review_packet(messages)
            sources = data.get('sources', [])
            eligible = data.get('eligible_source_ids', [])
            source = next((item for item in sources if item['id'] in eligible), {})
            text, role = source.get('text'), source.get('role')
            if stage in {'exchange_intent', 'exchange_extract'} and role == 'user' and text in {
                previous, query,
            }:
                self.worker_calls.append((stage, deepcopy(data)))
                if stage == 'exchange_intent':
                    ids = {target['exchange_id'] for target in data['targets']}
                    named = self.previous_target if text == query and self.previous_target in ids \
                        else None
                    return {'acts': [{'segmentId': 0, 'evidenceQuote': text,
                                      'intent': 'question', 'targetId': named}],
                            'coverage': 'complete'}
                return {'operations': [{'operation': 'add', 'type': 'response_obligation',
                                        'evidenceQuote': text, 'supersedesAtomId': None,
                                        'expiresAt': None}]}
            if stage in {'exchange_intent', 'exchange_extract'} and role == 'assistant' and (
                text == self.updated_reply
            ):
                self.worker_calls.append((stage, deepcopy(data)))
                if stage == 'exchange_intent':
                    return {'acts': [{'segmentId': 0, 'evidenceQuote': text, 'intent': 'answer',
                                      'targetId': target['exchange_id']}
                                     for target in data['targets']
                                     if target['content'] in {previous, query}],
                            'coverage': 'complete'}
                return {'operations': [{'operation': 'resolve', 'type': None,
                                        'evidenceQuote': act['evidenceQuote'],
                                        'supersedesAtomId': act['targetId'], 'expiresAt': None}
                                       for act in data['independent_acts']]}
            if stage == 'exchange_support':
                self.worker_calls.append((stage, deepcopy(data)))
                return {'intent': 'answer' if data['action'] == 'resolve' else 'question',
                        'verdict': 'supported'}
            return super().invoke(stage, messages, schema, cancelled)

    model = RecordedQuery(query=query, updated=updated,
                          updated_reply='按你现在的偏好，可以听一段轻快、柔和的音乐。')
    model.correct = '更正，我现在更偏爱轻快的音乐。'
    model.forget = '请忘记我现在更偏爱轻快的音乐这件事。'
    memory, mirror, runtime, session = build(tmp_path, model)
    try:
        turn(runtime, session, ORIGINAL)
        turn(runtime, session, previous)
        old, = memory.store.pending_exchanges(memory.companion, now_us=model.clock())
        model.previous_target = old['exchange_id']
        assert old['direction'] == 'assistant_response' and old['status'] == 'pending'
        correction = turn(runtime, session, model.correct)
        current, = memory.store.recall(memory.companion, now_us=model.clock())
        assert current['version'] == 2 and current['content'] == updated
        assert current['source_id'] == correction['basis']['input_id']
        assert correction['readset']['current_operation']['status'] == 'committed'
        assert correction['readset']['current_operation']['action'] == 'correct'
        before = memory.store.history(memory.companion, current['record_id'])
        calls_before, generations_before = len(model.worker_calls), len(model.primary)
        event = turn(runtime, session, query)
        assert len(model.primary) - generations_before == 2  # Existing action + final answer.
        assert [stage for stage, _ in model.worker_calls[calls_before:]] == [
            'memory_command', 'exchange_intent', 'fact_extract',
            'exchange_extract', 'exchange_support']
        with memory.store.transaction(write=False) as db:
            source = dict(db.execute(
                "SELECT s.status,w.receipt FROM background_sources s JOIN background_work w "
                "ON s.companion=w.companion AND s.job=w.job WHERE s.companion=? "
                "AND s.source=? AND s.role='user'",
                (memory.companion, event['basis']['input_id']),
            ).fetchone())
        assert source['status'] == 'completed' and memory.store.memory_status(
            memory.companion) == 'current'
        resolution = json.loads(source['receipt'])['resolution']
        assert resolution['source_id'] == event['basis']['input_id']
        assert not resolution['no_exchange'] and len(resolution['assessments']) == 1
        assert event['readset']['facts'][0]['record_id'] == current['record_id']
        assert event['readset']['facts'][0]['version'] == 2
        assert event['readset']['facts'][0]['source_id'] == current['source_id']
        assert event['display_text'] == model.updated_reply
        assert memory.store.history(memory.companion, current['record_id']) == before
        assert current['source_id'] not in {row['id'] for row in event['readset']['history']}
        assert ORIGINAL not in '\n'.join(row['content'] for row in model.primary[-1][1:-1])
        stream = memory.store.stream_state(memory.companion, event['basis']['input_id'])
        assert stream['outcome'] == 'completed' and stream['confirmed_units'] == 1
        assert stream['confirmed_text'] == event['display_text']
        pending = memory.store.pending_exchanges(memory.companion, now_us=model.clock())
        assert len(pending) == 2 and old in pending
        assert {item['content'] for item in pending} == {previous, query}
        with memory.store.transaction(write=False) as db:
            excluded = memory.store._ineffective_history_sources(
                db, memory.companion, now_us=model.clock())
            assert current['source_id'] in excluded  # Raw correction stays excluded from history.
            assert memory.store._readset_current(db, event['readset'], now_us=model.clock())
            assert 'segment-' + event['fingerprint'] not in excluded
        # ACK confirms display. Only the existing assistant worker's independently
        # supported answer may resolve the prior and current user requests.
        maintenance = [memory.scheduler.assistant.run_one(
            memory.companion, 'synthetic-ack-settlement') for _ in range(4)]
        assert all(maintenance)
        receipts = [json.loads(row) for row in maintenance]
        assert [row.get('disposition') for row in receipts[:3]] == ['source_invalidated'] * 3
        acknowledged_source = 'segment-' + event['fingerprint']
        assert receipts[-1]['resolution']['source_id'] == acknowledged_source
        assert len(receipts[-1]['resolution']['assessments']) == 2
        assert memory.store.pending_exchanges(memory.companion, now_us=model.clock()) == [], (
            maintenance, model.worker_calls[calls_before:])
        for target in pending:
            head = memory.store.exchange_history(memory.companion, target['exchange_id'])[-1]
            assert head['status'] == 'resolved' and head['revision'] == target['revision'] + 1
            assert head['content'] == target['content']
            assert head['evidence']['role'] == 'assistant'
            assert head['evidence']['source_id'] == acknowledged_source
            assert head['evidence']['quote'] == event['text']
        assert memory.store.history(memory.companion, current['record_id']) == before
        turn(runtime, session, model.forget)
        assert memory.store.recall(memory.companion, now_us=model.clock()) == []
        record = current['record_id']
    finally:
        memory.close()
        mirror.close()

    memory, mirror, runtime, session = build(tmp_path, model)
    try:
        session.new_session()
        reopened = turn(runtime, session, query)
        assert reopened['display_text'] == model.missing and reopened['readset']['facts'] == []
        assert mirror.history()['messages'][-1]['content'] == model.missing
        assert memory.store.history(memory.companion, record)[-1]['status'] == 'revoked'
        assert len(memory.store.history(memory.companion, record)) == 3
    finally:
        memory.close()
        mirror.close()
