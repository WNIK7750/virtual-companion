"""The development entry must exercise the current desktop wire protocol."""

import json

from jsonschema import Draft202012Validator
from virtual_companion.cognition.short_dialogue import SYSTEM
from virtual_companion.text.default_runtime import CONTROL
from virtual_companion.text.demand_memory import ACTION_SCHEMA
from virtual_companion.text.fake import FakeModel


class ImmediateSignal:
    def wait(self, _seconds):
        return False


def test_fake_accepts_the_actual_default_control_request():
    output = ''.join(FakeModel().stream([
        {'role': 'system', 'content': CONTROL},
        {'role': 'user', 'content': '合成本地切回验证'},
    ], ImmediateSignal()))
    action = json.loads(output)
    Draft202012Validator(ACTION_SCHEMA).validate(action)
    assert action['action'] == 'answer'
    assert '合成回复' in action['text']


def test_fake_plain_text_port_remains_plain_text():
    output = ''.join(FakeModel().stream([
        {'role': 'system', 'content': SYSTEM},
    ], ImmediateSignal()))
    assert '合成回复' in output
    assert not output.startswith('{')
