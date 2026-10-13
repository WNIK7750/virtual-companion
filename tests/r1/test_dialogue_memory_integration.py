"""DefaultRuntime synthetic provider chain, using actual owner and ACK sources."""

import json
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from test_default_text_delivery import Model, setup, wait_for
from virtual_companion.boundary.contracts import canonical
from virtual_companion.text.application import TextSession
from virtual_companion.text.character import CharacterSnapshot
from virtual_companion.text.default_runtime import DefaultTextRuntime
from virtual_companion.text.fake import synthetic_unit_opinion


def close(runtime, text, memory, mirror):
    runtime.stop()
    text.stop()
    if text.worker is not None:
        text.worker.join(2)
    memory.close()
    mirror.close()


def send_visible(runtime, text, value, receipt):
    text.send(value)
    event = wait_for(lambda: runtime.poll()['event'] or not text.snapshot()['busy'])
    assert isinstance(event, dict), text.snapshot()
    runtime.ack({'sequence': event['sequence'], 'fingerprint': event['fingerprint'],
                 'host_receipt': receipt})
    wait_for(lambda: not text.snapshot()['busy'])
    assert text.snapshot()['attempt']['status'] == 'completed', text.snapshot()
    return event


class DialogueNavigator:
    supports_stage_metadata = True

    def __init__(self, *, reject_once=False):
        self.calls, self.read = [], False
        self.reject_once = reject_once

    def cancel(self):
        pass

    def prepare_stage_request(self, messages, *, stage, schema):
        owner = self
        messages, schema = deepcopy(messages), deepcopy(schema)

        class Prepared:
            def check_context(self):
                pass

            def stream(self, cancelled):
                owner.calls.append((stage, messages))
                if stage == 'stream_unit_review':
                    opinion = synthetic_unit_opinion(json.loads(messages[-1]['content']))
                    if owner.reject_once:
                        owner.reject_once = False
                        opinion.update(decision='reject', constraints_satisfied=False)
                    yield canonical(opinion)
                    return
                page = next(
                    json.loads(message["content"])["memory_navigation"]
                    for message in messages
                    if message["content"].startswith('{"memory_navigation":')
                )
                if owner.read:
                    action = {'action': 'answer', 'text': '那次我们确实聊过这个话题。'}
                elif page['node_id'] == 'root':
                    target = next(row for row in page['entries'] if row.get('label') == 'dialogue')
                    action = {'action': 'browse_memory', 'node_id': target['node_id']}
                else:
                    target = next(row for row in page['entries']
                                  if row.get('source_type') == 'dialogue_fragment')
                    action = {'action': 'read_memory', 'references': [
                        {'record_id': target['record_id'], 'version': target['version']}]}
                    owner.read = True
                yield canonical(action)

        return Prepared()


@pytest.mark.parametrize('outside_recent_window,change,repair', [
    (False, 'forget', False), (True, 'correct', False),
    (False, 'expiry', False), (True, 'forget', True),
])
def test_real_runtime_cross_session_dialogue_generation_review_and_readset(
    tmp_path, outside_recent_window, change, repair,
):
    old, old_text, memory, mirror, _ = setup(tmp_path, Model())
    past_role = CharacterSnapshot(memory.companion, 'past-role', 'v1', '过去角色', '', '', '')
    current_role = CharacterSnapshot(memory.companion, 'current-role', 'v1', '当前角色', '', '', '')
    old.config = replace(
        old.config, character=past_role.reference, character_data=past_role.payload
    )
    first = send_visible(old, old_text, '我不是每天都去蓝桥，只是偶尔想去。', 'old-dom')
    old.stop()
    old_text.stop()
    if outside_recent_window:
        for index in range(34):
            filler = memory.store.accept(old.config, f'filler-{index}', f'间隔输入{index}。',
                                         session_id=first['basis']['session_id'])
            memory.store.cancel(memory.companion, filler.input_id)
    else:
        old_text.new_session()
    memory.config = replace(memory.config, character=current_role.reference,
                            character_data=current_role.payload)
    model = DialogueNavigator(reject_once=repair)
    runtime = DefaultTextRuntime(memory, mirror, model, binding={'test': 'dialogue'},
        readonly_recall='dialogue-memory.v1', delivery_timeout=1)
    text = TextSession(model, mirror, {}, prepare_turn=runtime.prepare_turn,
                       before_start=runtime.before_start)
    try:
        send_visible(runtime, text, '那次蓝桥我们说过什么？', 'new-dom')
        review = next(json.loads(messages[-1]['content']) for stage, messages in model.calls
                      if stage == 'stream_unit_review')
        data = review['context']
        if isinstance(data, str):
            data = json.loads(data)
        data = deepcopy(data)
        review_sources = {row['id']: row for row in review['personal_sources']}
        for fragment in data['retrieved_dialogue']:
            for member in fragment['members']:
                if 'personal_source_id' in member:
                    member['text'] = review_sources[member.pop('personal_source_id')]['text']
        assert len(data['history']) == (31 if outside_recent_window else 0)
        assert first['basis']['input_id'] not in {row['id'] for row in data['history']}
        assert data['memories'] == []
        fragments = data['retrieved_dialogue']
        assert len(fragments) == 1
        fragment = fragments[0]
        assert [row['text'] for row in fragment['members']] == [
            '我不是每天都去蓝桥，只是偶尔想去。', '这是合成答案。']
        assert fragment['origin_session'] == first['basis']['session_id']
        assert fragment['event_time_status'] == 'unknown'
        assert fragment['character_context'] == past_role.to_mapping()
        assert data['character_context'] == current_role.to_mapping()
        assert all(row['visibility'] == 'confirmed_visible'
                   for row in fragment['members'] if row['role'] == 'assistant')
        review_context = review['context']
        if isinstance(review_context, str):
            review_context = json.loads(review_context)
        assert tuple(fragments) == runtime._evidence_session.selection.retrieved_dialogue
        assert all(member['role'] != 'user' or 'personal_source_id' in member
                   for fragment in review_context['retrieved_dialogue']
                   for member in fragment['members'])
        reviews = [json.loads(messages[-1]['content'])['context']
                   for stage, messages in model.calls if stage == 'stream_unit_review']
        assert len(reviews) == (2 if repair else 1)
        assert all(value == reviews[0] for value in reviews)
        generation = [messages for stage, messages in model.calls if stage != 'stream_unit_review']
        assert any('我不是每天都去蓝桥，只是偶尔想去。' in message['content']
                   and '这是合成答案。' in message['content'] for message in generation[-1])
        assert runtime._evidence_session.selection.evidence_access['calls'] == 2
        with memory.store.transaction(write=False) as db:
            rows = db.execute('SELECT manifest FROM stream_segments WHERE companion=? AND input=?',
                              (memory.companion, runtime.claim.input_id)).fetchall()
        readset = json.loads(rows[-1][0])['readset']
        assert readset['protocol'] == 'history-readset.v7'
        assert (
            readset["retrieved_dialogue"][0]["members"][0]["source_id"]
            == first["basis"]["input_id"]
        )
        # Revoke a fact derived from the original mixed turn. The already shown
        # answer must become ineffective through its dialogue-member readset.
        from test_explicit_memory_management import list_command
        from test_rebuilt_memory import mutation
        from virtual_companion.boundary.contracts import Scope
        from virtual_companion.boundary.memory import FactKey

        probe_us = runtime.clock()
        expires = datetime.fromtimestamp(probe_us / 1_000_000, UTC) + timedelta(seconds=1)
        seed = replace(
            mutation(
                "dependency-seed",
                kind="assert_fact",
                record="music",
                content="我偶尔去蓝桥。",
                fact_key=FactKey("user", "bridge", "profile"),
            ),
            scope=Scope(memory.companion, first["basis"]["session_id"]),
            source_id=first["basis"]["input_id"],
            expires_at=expires if change == "expiry" else None,
        )
        memory.store.mutate(seed, memory.store.owner(memory.companion),
                            expected_version=memory.store.snapshot(memory.companion).write_version)
        if change == 'expiry':
            probe_us += 2_000_000
        else:
            memory.store.execute_memory_command(list_command(companion=memory.companion,
                action=change, replacement='我现在常去红桥。' if change == 'correct' else ''),
                now_us=probe_us)
        invalid = memory.store.ineffective_history_sources(memory.companion, now_us=probe_us)
        assert first['basis']['input_id'] in invalid
        with memory.store.transaction(write=False) as db:
            dependent = db.execute(
                "SELECT source FROM stream_history WHERE companion=? AND input=?",
                (memory.companion, runtime.claim.input_id),
            ).fetchone()[0]
        assert dependent in invalid
        later = memory.store.accept(runtime.config, 'post-forget', '还记得吗？')
        assert memory.store.dialogue_fragments(memory.companion, later.input_id,
                                               now_us=probe_us) == ()
        if change == 'correct':
            assert (
                memory.store.committed_recall(memory.companion, now_us=probe_us)[0]["content"]
                == "我现在常去红桥。"
            )
    finally:
        close(runtime, text, memory, mirror)
