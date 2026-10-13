"""An explicit generation-only experiment; recordings do not prove naturalness."""

import json
from dataclasses import replace
from hashlib import sha256
from threading import Event

import pytest
from test_character_context import character
from test_trusted_context_time import Model, admitted, runtime
from virtual_companion.boundary.contracts import canonical
from virtual_companion.cognition.short_dialogue import PROMPTS, ShortDialogue
from virtual_companion.text import default_runtime

BASELINE = 'short-dialogue.v35'
CANDIDATE = 'short-dialogue.v36'


@pytest.mark.parametrize('final', [False, True])
def test_candidate_changes_generation_policy_only_on_existing_runtime(monkeypatch, final):
    baseline, _, _, baseline_model = runtime(monkeypatch, BASELINE)
    candidate, _, _, candidate_model = runtime(monkeypatch, CANDIDATE)
    old = baseline.prepare(baseline.claim, final=final)
    new = candidate.prepare(candidate.claim, final=final)

    assert new.request == old.request
    assert new.messages[1:] == old.messages[1:]
    assert new.messages[0]['content'].replace(
        PROMPTS[CANDIDATE], PROMPTS[BASELINE], 1
    ) == old.messages[0]['content']
    assert candidate_model.prepared[-1]['schema'] == baseline_model.prepared[-1]['schema']
    assert candidate.review_audit_version == baseline.review_audit_version
    assert candidate.repair_reply == baseline.repair_reply
    assert default_runtime.DIAGNOSTIC_PROMPT_REVISIONS.count(CANDIDATE) == 1
    assert default_runtime.DEFAULT_PROMPT_REVISION == 'short-dialogue.v12'
    assert sha256(default_runtime.CONTROL.encode()).hexdigest() == (
        '91819f92b04d66fc002eb703a9a7cffff282e6949164625e23ccf123152290a5')


def test_candidate_keeps_role_and_received_context_out_of_personal_sources():
    _, _, _, request = admitted()
    data = json.loads(request.context.payload)
    data['character_context'] = json.loads(character().payload)
    own_text = '这只是我先前提出的一种看法。'
    data['received_output'] = [{
        'input_id': 'prior-input', 'sequence': 1, 'fingerprint': 'a' * 64,
        'text': own_text, 'response_type': 'answer',
        'visibility': 'not_confirmed_at_admission',
    }]
    request = replace(request, context=replace(
        request.context, payload=canonical(data), rendering='prose'))
    baseline, candidate = Model(), Model()
    ShortDialogue(baseline, cancelled=Event(), policy=BASELINE).generate(request)
    ShortDialogue(candidate, cancelled=Event(), policy=CANDIDATE).generate(request)

    old, new = baseline.sent[0], candidate.sent[0]
    assert candidate.counted == candidate.sent and len(candidate.sent) == 1
    assert {key: value for key, value in new.items() if key != 'messages'} == {
        key: value for key, value in old.items() if key != 'messages'}
    assert new['messages'][1:] == old['messages'][1:]
    assert new['messages'][0]['content'].replace(
        PROMPTS[CANDIDATE], PROMPTS[BASELINE], 1
    ) == old['messages'][0]['content']
    assert any(own_text in message['content'] for message in new['messages'])
    assert any(canonical(data['character_context']) in message['content']
               for message in new['messages'])
    sources = request.personal_sources(request.context.companion,
                                       request.context.input_id,
                                       request.context.state_version)
    assert all(own_text not in source['text'] and '旧书档案员' not in source['text']
               for source in sources)
