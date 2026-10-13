"""Actual runtime routing and capability identity; no inference or network."""

import json
from copy import deepcopy
from threading import Event

import pytest
from jsonschema import Draft202012Validator
from test_default_text_delivery import Model, setup
from virtual_companion.text import default_runtime as runtime_module
from virtual_companion.text.demand_memory import _action


class CaptureModel(Model):
    supports_stage_metadata = True

    def prepare_stage_request(self, messages, *, stage, schema):
        self.prepared_messages = deepcopy(messages)
        self.prepared_schema = deepcopy(schema)
        return None

    def stream(self, messages, cancelled):
        raise AssertionError('This route contract must not invoke inference')


@pytest.fixture
def build_runtime(tmp_path):
    allocated = []

    def build(capability=None):
        case_dir = tmp_path / f'case-{len(allocated)}'
        case_dir.mkdir()
        model = CaptureModel()
        old, old_text, memory, mirror, _ = setup(case_dir, model)
        runtime = runtime_module.DefaultTextRuntime(
            memory, mirror, model, binding={'test': 'memory-route'},
            readonly_recall=capability, memory_skills='memory-skills.v2')
        allocated.append((runtime, old, old_text, memory, mirror))
        attempt = mirror.begin('请回顾以前提到的安排。', {})
        runtime.prepare_turn(attempt, mirror.session_id, mirror.history()['messages'], Event())
        return runtime, model

    yield build
    for runtime, old, old_text, memory, mirror in reversed(allocated):
        runtime.stop()
        old.stop()
        old_text.stop()
        memory.close()
        mirror.close()


@pytest.mark.parametrize('capability', ['deep-memory.v1', 'dialogue-memory.v1'])
def test_initial_history_gap_has_one_readonly_route_with_loaded_skill(build_runtime, capability):
    runtime, model = build_runtime(capability)
    runtime.prepare(runtime.claim, final=False, allow_recall=True)
    system = model.prepared_messages[0]['content']
    assert '现有背景未提供所需历史时，应先memory' not in system
    assert '{"action":"memory","query":"兴趣 爱好 绘画 运动"}' not in system
    assert 'memory是本条用户来源的整理委托' in system
    assert any('Do not use it as a read fallback' in msg['content']
               for msg in model.prepared_messages if msg['role'] == 'system')
    for action in ({'action': 'recall', 'query': '以前的安排'},
                   {'action': 'browse_memory', 'node_id': 'root'}):
        Draft202012Validator(model.prepared_schema).validate(action)


def test_readonly_keeps_current_source_memory_legal_but_closes_spent_actions(build_runtime):
    runtime, model = build_runtime('dialogue-memory.v1')
    runtime.prepare(runtime.claim, final=False, allow_recall=False)
    action = {'action': 'memory', 'query': '本条来源的有效事实'}
    Draft202012Validator(model.prepared_schema).validate(action)
    assert _action(json.dumps(action), final=False) == action
    assert not Draft202012Validator(model.prepared_schema).is_valid(
        {'action': 'recall', 'query': '以前的安排'})
    assert '不以memory绕过' in model.prepared_messages[0]['content']
    runtime.prepare(runtime.claim, final=True, allow_recall=False)
    assert not Draft202012Validator(model.prepared_schema).is_valid(action)
    assert '本次不得再次memory' in model.prepared_messages[0]['content']


def test_legacy_control_and_initial_access_remain_byte_identical(build_runtime):
    runtime, model = build_runtime()
    stage = runtime.prepare(runtime.claim, final=False)
    assert runtime.control == runtime_module.CONTROL
    assert stage.access_status == runtime_module.ACCESS_INITIAL
    assert '现有背景未提供所需历史时，应先memory' in model.prepared_messages[0]['content']
    Draft202012Validator(model.prepared_schema).validate({'action': 'memory'})


def test_selected_readonly_access_text_participates_in_resource_identity(
    build_runtime, monkeypatch,
):
    before, _ = build_runtime('dialogue-memory.v1')
    first = dict(before.config.resources)['model']
    monkeypatch.setattr(runtime_module, 'READONLY_ACCESS_INITIAL',
                        runtime_module.READONLY_ACCESS_INITIAL + '合成身份变化。')
    after, _ = build_runtime('dialogue-memory.v1')
    second = dict(after.config.resources)['model']
    assert first != second, 'Changed effective routing authority retained an old resource identity'
