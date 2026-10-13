"""An independent reviewer keeps its own repair contract across generation prompts."""

import json

import pytest
from test_default_text_delivery import setup, wait_for
from virtual_companion.cognition.stream_unit_review import (
    AUDIT_VERSION,
    VERDICT_UNIT_AUDIT_VERSION,
    restore_repair_review,
)
from virtual_companion.text.application import TextSession
from virtual_companion.text.default_runtime import DefaultTextRuntime
from virtual_companion.text.fake import synthetic_unit_opinion


class RejectedOnce:
    def __init__(self):
        self.calls = []
        self.reviews = 0
        self.generations = 0

    def check_context(self, messages):
        pass

    def cancel(self):
        pass

    def stream(self, messages, cancelled):
        self.calls.append(messages)
        if '你只审查候选文字' in messages[0]['content']:
            self.reviews += 1
            view = json.loads(messages[-1]['content'])
            opinion = synthetic_unit_opinion(view)
            if self.reviews == 1:
                opinion['decision'] = 'reject'
                if opinion['audit_version'] == VERDICT_UNIT_AUDIT_VERSION:
                    opinion['checks'][3] = 'fail'
                    opinion['failure'] = {'scope': 'accuracy', 'segment_id': None,
                                          'basis': '供应商录制的首次拒绝。'}
                else:
                    opinion['constraints_satisfied'] = False
            yield json.dumps(opinion, ensure_ascii=False)
        else:
            self.generations += 1
            yield json.dumps({'action': 'ack', 'text': (
                '第一条录制候选。' if self.generations == 1 else '第二条录制候选。')},
                ensure_ascii=False)


@pytest.mark.parametrize('prompt,audit', [
    ('short-dialogue.v30', AUDIT_VERSION),
    ('short-dialogue.v12', VERDICT_UNIT_AUDIT_VERSION),
])
def test_repair_uses_selected_auditor_contract_not_generation_revision(tmp_path, prompt, audit):
    model = RejectedOnce()
    _, _, memory, mirror, _ = setup(tmp_path, model)
    runtime = DefaultTextRuntime(memory, mirror, model, binding={'test': 'selector-repair'},
        prompt_revision=prompt, review_audit_version=audit, delivery_timeout=0.8)
    session = TextSession(model, mirror, {}, prepare_turn=runtime.prepare_turn,
                          before_start=runtime.before_start)
    try:
        session.send('合成接话请求。')
        event = wait_for(lambda: runtime.poll()['event'] or not session.snapshot()['busy'])
        assert isinstance(event, dict), session.snapshot()['attempt']
        runtime.ack(dict(sequence=event['sequence'], fingerprint=event['fingerprint'],
                         host_receipt='synthetic-selector-repair-dom'))
        wait_for(lambda: not session.snapshot()['busy'])
        assert session.snapshot()['attempt']['status'] == 'completed'
        assert model.generations == model.reviews == 2
        repair = json.loads(model.calls[2][-2]['content'].removeprefix(runtime.repair_reply))
        assert repair['rejected_candidate'] == '第一条录制候选。'
        review = repair['review']
        if audit == VERDICT_UNIT_AUDIT_VERSION:
            assert review['projection_version'] == 'repair-unit-verdict-projection.v1'
            original = restore_repair_review(repair['rejected_candidate'], review)
            assert original['audit_version'] == audit and original['decision'] == 'reject'
        else:
            assert review['audit_version'] == audit and review['decision'] == 'reject'
            assert 'projection_version' not in review
        assert mirror.history()['messages'][-1]['content'] == '第二条录制候选。'
    finally:
        memory.close()
        mirror.close()
