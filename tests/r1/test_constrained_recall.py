"""Explicit constrained recall must survive the actual host/action boundaries."""

import json
from threading import Event

import pytest
from jsonschema import Draft202012Validator
from virtual_companion.boundary.contracts import canonical
from virtual_companion.text import demand_memory
from virtual_companion.text.deep_memory import DeepMemorySession

ACTION = {'action': 'recall', 'query': '那份小册子', 'exact_terms': ['AB-12'],
          'conditions': {'subject': 'user', 'scope': 'profile'}}


def test_constrained_recall_requires_explicit_schema_and_decoder_capability():
    legacy = demand_memory.action_schema(allow_memory=False, allow_recall=True)
    assert not Draft202012Validator(legacy).is_valid(ACTION)
    with pytest.raises(ValueError):
        demand_memory._action(canonical(ACTION), final=True, allow_recall=True)
    selected = demand_memory.action_schema(
        allow_memory=False, allow_recall=True, allow_constrained_recall=True)
    assert Draft202012Validator(selected).is_valid(ACTION)
    assert demand_memory._action(canonical(ACTION), final=True, allow_recall=True,
        allow_constrained_recall=True) == ACTION
    assert selected != legacy


@pytest.mark.parametrize('extra', [
    {'conditions': {'sql': 'select *'}}, {'conditions': {'subject': None}},
    {'conditions': {'predicate': 'x' * 129}}, {'exact_terms': ['AB-12'] * 7},
    {'exact_terms': ['']}, {'exact_terms': ['a\nb']}, {'limit': 24},
])
def test_constrained_parameters_are_bounded_before_lookup(extra):
    action = {'action': 'recall', 'query': '小册子', **extra}
    with pytest.raises(ValueError):
        demand_memory._action(canonical(action), final=True, allow_recall=True,
                              allow_constrained_recall=True)


class Prepared:
    def __init__(self, action):
        self.action = action

    def check_context(self):
        pass

    def stream(self, cancelled):
        yield canonical(self.action)


def test_demand_loop_forwards_constraints_without_changing_plain_query_port():
    for first, expected in [(ACTION, ACTION), ({'action': 'recall', 'query': '小册子'}, '小册子')]:
        actions = iter([first, {'action': 'answer', 'text': '按原文回应。'}])
        calls = []
        result = demand_memory.run_demand_turn(
            claim=object(), prepare=lambda *a, actions=actions, **kw: Prepared(next(actions)),
            resolve_current=lambda c: pytest.fail('Readonly recall cannot process sources'),
            refresh=lambda c: c, cancelled=Event(), max_recall_calls=6,
            recall_evidence=lambda c, q, calls=calls, **kw: calls.append(q),
            allow_navigation=True, allow_constrained_recall=True)
        assert calls == [expected]
        assert result.recall_calls == 1 and result.memory_calls == 0


class Catalog:
    identity = 'frozen-catalog'

    def browse(self, *args, **kwargs):
        return {'node_id': 'root'}


def test_deep_session_validates_and_counts_distinct_constraints_once():
    calls, cancelled = [], Event()
    session = DeepMemorySession(catalog=Catalog(), lookup=lambda q, **kw: calls.append(q) or [],
        checkpoint=lambda c: None, cancelled=cancelled, constrained_recall=True)
    session.seed([])
    session.recall(ACTION, cancelled=cancelled)
    second = {**ACTION, 'conditions': {'subject': 'coworker'}}
    session.recall(second, cancelled=cancelled)
    assert calls == [ACTION, second]
    assert session.selection.evidence_access['calls'] == 2
    assert session.selection.evidence_access['query_contract'] == 'constrained-recall.v1'
    session.recall(second, cancelled=cancelled)
    assert session.selection.evidence_access['calls'] == 2
    assert session.selection.evidence_access['stop_reason'] == 'duplicate_query'


def test_deep_session_cannot_bypass_unselected_or_malformed_action_contract():
    for selected, action in [(False, ACTION), (True, {**ACTION, 'limit': 24})]:
        calls, cancelled = [], Event()
        session = DeepMemorySession(catalog=Catalog(),
            lookup=lambda q, calls=calls, **kw: calls.append(q) or [], checkpoint=lambda c: None,
            cancelled=cancelled, constrained_recall=selected)
        with pytest.raises(ValueError):
            session.recall(action, cancelled=cancelled)
        assert calls == [] and session.selection.evidence_access['calls'] == 0


@pytest.mark.parametrize('readonly', ['deep-memory.v1', 'dialogue-memory.v1'])
def test_selected_runtime_schema_and_actual_sqlite_lookup_keep_hard_constraints(tmp_path, readonly):
    from test_default_text_delivery import Model, setup
    from test_hybrid_sqlite import mutate, retriever
    from virtual_companion.text.default_runtime import DefaultTextRuntime

    class StageModel(Model):
        supports_stage_metadata = True

        def prepare_stage_request(self, messages, *, stage, schema):
            self.schema = schema

    old, text, memory, mirror, model = setup(tmp_path, StageModel())
    source = memory.store.accept(memory.config, 'seed-source', '合成来源')
    memory.store.cancel(memory.companion, source.input_id)
    memory.store.owner(memory.companion) or memory.store.claim_owner(
        memory.companion, 'seed-owner', expected_epoch=0)
    for number in range(10):
        mutate(memory, source, f'a-{number:02}', f'编号AB-120的背景第{number}项。')
    mutate(memory, source, 'z-exact', '编号AB-12，封面绿色，仍是计划。')
    runtime = DefaultTextRuntime(memory, mirror, model, binding={'test': 'constrained'},
        semantic_retriever=retriever(), readonly_recall=readonly)
    cancelled = Event()
    try:
        attempt = mirror.begin('那份小册子的原话？', {})
        runtime.prepare_turn(attempt, mirror.session_id, mirror.history()['messages'], cancelled)
        prepared = runtime.prepare(runtime.claim, final=False, allow_recall=True)
        assert Draft202012Validator(model.schema).is_valid(ACTION)
        assert 'constrained-recall.v1' in repr(prepared.messages)
        selection = runtime.recall_evidence(runtime.claim, ACTION, cancelled=cancelled)
        target = next(row for row in selection.memories if row['record_id'] == 'z-exact')
        assert target['content'] == '编号AB-12，封面绿色，仍是计划。'
        query = json.loads(selection.evidence_access['queries'][-1])
        assert query == ACTION
        # The evidence union retains earlier originals; the constrained lookup
        # itself admits only the exact identifier and sourced positive fact keys.
        found = runtime._lookup_evidence(ACTION, cancelled=cancelled)
        assert [row['record_id'] for row in found] == ['z-exact']
        if readonly == 'dialogue-memory.v1':
            # Selected semantic fact candidates must survive the additional
            # dialogue lexical branch even when no query word overlaps.
            assert len(runtime._lookup_evidence('换说法', cancelled=cancelled)) == 8
    finally:
        runtime.stop()
        old.stop()
        text.stop()
        memory.close()
        mirror.close()
