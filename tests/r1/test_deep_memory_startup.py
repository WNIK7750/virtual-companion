"""Selected skill resources must be verified before exposing a send-ready runtime."""

import json

import pytest
from test_default_flash_route import boot as boot
from virtual_companion.text import __main__ as bootstrap
from virtual_companion.text.runtime_skills import load_memory_skills


def test_selected_skills_are_preloaded_once_and_bound_before_ready(boot, monkeypatch):
    bundle = load_memory_skills()
    calls = []
    monkeypatch.setenv('VC_TEXT_DIAGNOSTIC_READONLY_RECALL', 'deep-memory.v1')
    monkeypatch.setattr(bootstrap, 'load_memory_skills', lambda: calls.append(True) or bundle)
    dispatch, api, _ = boot()
    startup = dispatch('state', {})['startup']
    assert startup['can_send']
    assert startup['components']['runtime_skills'] == {
        'status': 'configured', 'protocol': 'memory-skills.v1',
        'bundle_sha256': bundle.fingerprint, 'skills': bundle.identity,
    }
    assert calls == [True] and api.sent == []


def test_missing_selected_skill_blocks_send_before_any_model_request(boot, monkeypatch):
    monkeypatch.setenv('VC_TEXT_DIAGNOSTIC_READONLY_RECALL', 'deep-memory.v1')
    sentinel = 'private-skill-path-does-not-escape'

    def unavailable():
        raise OSError(sentinel)

    monkeypatch.setattr(bootstrap, 'load_memory_skills', unavailable)
    dispatch, api, _ = boot()
    state = dispatch('state', {})
    assert state['state'] == 'configuration_unavailable'
    assert not state['startup']['can_send']
    assert 'runtime_skills' in state['startup']['blocked_reasons']
    assert sentinel not in json.dumps(state)
    with pytest.raises(RuntimeError, match='skill'):
        dispatch('send', {'text': 'must remain unsent'})
    assert api.sent == []


def test_default_runtime_does_not_read_unselected_skills(boot, monkeypatch):
    monkeypatch.delenv('VC_TEXT_DIAGNOSTIC_READONLY_RECALL', raising=False)
    monkeypatch.delenv('VC_TEXT_RUNTIME_MEMORY_SKILLS', raising=False)

    def unexpected():
        raise AssertionError('Unselected skill must not be read')

    monkeypatch.setattr(bootstrap, 'load_memory_skills', unexpected)
    dispatch, api, _ = boot()
    assert dispatch('state', {})['startup']['can_send']
    assert 'runtime_skills' not in dispatch('state', {})['startup']['components']
    assert api.sent == []
