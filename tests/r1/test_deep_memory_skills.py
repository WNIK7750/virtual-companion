"""Runtime skill loading and typed readonly actions, with synthetic model receipts."""

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from threading import Event

import pytest
from jsonschema import Draft202012Validator
from test_default_text_delivery import setup, wait_for
from test_readonly_evidence_recall import fact
from test_rebuilt_memory import apply, mutation
from virtual_companion.boundary.contracts import Scope, canonical
from virtual_companion.boundary.history_readset import validate_evidence_access
from virtual_companion.text.application import TextSession
from virtual_companion.text.deep_memory import DeepMemorySession
from virtual_companion.text.default_runtime import DefaultTextRuntime
from virtual_companion.text.demand_memory import _action, action_schema
from virtual_companion.text.fake import synthetic_unit_opinion
from virtual_companion.text.runtime_skills import load_memory_skills


def test_real_runtime_navigates_pages_reads_original_and_loads_role_skill(tmp_path):
    class Navigator:
        supports_stage_metadata = True

        def __init__(self):
            self.calls, self.read_done = [], False

        def cancel(self):
            pass

        def prepare_stage_request(self, messages, *, stage, schema):
            owner = self
            messages, schema = deepcopy(messages), deepcopy(schema)

            class Prepared:
                def check_context(self):
                    pass

                def stream(self, cancelled):
                    owner.calls.append((stage, messages, schema))
                    if stage == 'stream_unit_review':
                        yield canonical(synthetic_unit_opinion(json.loads(messages[-1]['content'])))
                        return
                    page = next(json.loads(msg['content'])['memory_navigation']
                                for msg in messages if msg['content'].startswith(
                                    '{"memory_navigation":'))
                    if owner.read_done:
                        action = {'action': 'answer', 'text': '已找到那条原文。'}
                    elif page['entries'] and 'node_id' in page['entries'][0]:
                        action = {'action': 'browse_memory',
                                  'node_id': page['entries'][0]['node_id']}
                    elif page['next_cursor']:
                        action = {'action': 'browse_memory', 'node_id': page['node_id'],
                                  'cursor': page['next_cursor']}
                    else:
                        target = next(row for row in page['entries']
                                      if row['record_id'] == 'fact-13')
                        action = {'action': 'read_memory', 'references': [
                            {'record_id': target['record_id'], 'version': target['version']}]}
                        owner.read_done = True
                    Draft202012Validator(schema).validate(action)
                    yield canonical(action)

            return Prepared()

    model = Navigator()
    _, _, memory, mirror, _ = setup(tmp_path, model)
    source = memory.store.accept(memory.config, 'directory-seed', '合成目录来源。')
    memory.store.cancel(memory.companion, source.input_id)
    memory.store.claim_owner(memory.companion, 'directory-seed', expected_epoch=0)
    for number in range(14):
        assert apply(memory.store, replace(mutation(
            f'nav-seed-{number}', record=f'fact-{number:02}',
            content=f'只在某些时候选择第{number}项，不代表每天。'),
            scope=Scope(memory.companion, source.session_id), source_kind='manual',
            source_id=source.input_id))['disposition'] == 'applied'
    runtime = DefaultTextRuntime(memory, mirror, model, binding={'test': 'deep-memory'},
                                  readonly_recall='deep-memory.v1', delivery_timeout=1)
    text = TextSession(model, mirror, {}, prepare_turn=runtime.prepare_turn,
                       before_start=runtime.before_start)
    try:
        text.send('请查找那条隐藏的偏好记录。')
        event = wait_for(lambda: runtime.poll()['event'] or not text.snapshot()['busy'])
        assert isinstance(event, dict), text.snapshot()
        runtime.ack({'sequence': event['sequence'], 'fingerprint': event['fingerprint'],
                     'host_receipt': 'synthetic-deep-memory-dom'})
        wait_for(lambda: not text.snapshot()['busy'])
        assert text.snapshot()['attempt']['status'] == 'completed', text.snapshot()
        selected = runtime._evidence_session.selection
        assert selected.evidence_access['calls'] == 3
        assert any(row['record_id'] == 'fact-13' and '不代表每天' in row['content']
                   for row in selected.memories)
        generated = [call for call in model.calls if call[0] != 'stream_unit_review']
        assert len(generated) == 4
        assert all(len([m for m in call[1] if m['role'] == 'system']) >= 3
                   for call in generated)
        assert selected.evidence_access['navigation']['catalog_coverage'] == 'complete'
        assert selected.evidence_access['coverage'] == 'partial'
    finally:
        runtime.stop()
        text.stop()
        if text.worker is not None:
            text.worker.join(2)
        memory.close()
        mirror.close()


class Catalog:
    identity = 'a' * 64

    def browse(self, node_id='root', *, cursor=None):
        return {'protocol': 'memory-catalog.v1', 'catalog_sha256': self.identity,
                'node_id': node_id, 'entries': [], 'next_cursor': None,
                'catalog_coverage': 'complete', 'scope': 'current_committed_records'}

    def read(self, references):
        return tuple(fact(int(record.removeprefix('fact-'))) for record, _ in references)


def test_bundled_skills_are_instructions_and_have_stable_content_identity():
    bundle = load_memory_skills()
    assert set(bundle.names) == {'vc-memory-retrieval', 'vc-context-composition'}
    assert bundle.fingerprint == load_memory_skills().fingerprint
    assert all(len(item['sha256']) == 64 for item in bundle.identity)
    assert len(bundle.messages(navigation=True)) == 2
    assert len(bundle.messages(navigation=False)) == 1
    assert all(message['role'] == 'system' for message in bundle.messages(navigation=True))


@pytest.mark.parametrize('header', [
    'name: wrong\n# name: vc-memory-retrieval',
    'name: wrong\nname: vc-memory-retrieval',
    'name: vc-memory-retrieval-extra',
])
def test_packaged_skill_name_is_an_exact_unambiguous_field(monkeypatch, header):
    monkeypatch.setattr(Path, 'read_bytes', lambda _: (
        f'---\n{header}\ndescription: synthetic\n---\nsynthetic body').encode())
    with pytest.raises(ValueError, match='named SKILL'):
        load_memory_skills()


@pytest.mark.parametrize('action', [
    {'action': 'browse_memory', 'node_id': 'root'},
    {'action': 'browse_memory', 'node_id': 'root', 'cursor': None},
    {'action': 'read_memory', 'references': [{'record_id': 'fact-13', 'version': 1}]},
])
def test_navigation_actions_require_selected_capability(action):
    raw = json.dumps(action)
    with pytest.raises(ValueError):
        _action(raw, final=False, allow_recall=True)
    assert _action(raw, final=False, allow_recall=True, allow_navigation=True) == action
    schema = action_schema(allow_memory=True, allow_recall=True, allow_navigation=True)
    Draft202012Validator(schema).validate(action)
    assert not Draft202012Validator(action_schema(
        allow_memory=True, allow_recall=True)).is_valid(action)


def test_navigation_is_not_fact_evidence_and_does_not_stop_before_read():
    cancelled = Event()
    state = DeepMemorySession(catalog=Catalog(), lookup=lambda *_args, **_kw: [],
                              checkpoint=lambda event: None, cancelled=cancelled)
    state.seed([])
    browsed = state.recall({'action': 'browse_memory', 'node_id': 'root'}, cancelled=cancelled)
    assert browsed.memories == () and browsed.can_recall
    assert browsed.evidence_access['calls'] == 1
    read = state.recall({'action': 'read_memory', 'references': [
        {'record_id': 'fact-13', 'version': 1}]}, cancelled=cancelled)
    assert read.memories == (fact(13),)
    validate_evidence_access(read.evidence_access, facts=list(read.memories))
    assert read.evidence_access['coverage'] == 'partial'


def test_all_read_tools_share_one_budget_and_cancellation_identity():
    cancelled = Event()
    state = DeepMemorySession(catalog=Catalog(), lookup=lambda *_args, **_kw: [],
                              checkpoint=lambda event: None, cancelled=cancelled)
    state.seed([])
    for number in range(6):
        result = state.recall(f'query-{number}', cancelled=cancelled)
    assert result.evidence_access['calls'] == 6 and not result.can_recall
    assert result.evidence_access['stop_reason'] == 'call_limit'
    assert state.recall('excess', cancelled=cancelled) == result
    with pytest.raises(ValueError):
        state.recall('other', cancelled=Event())
