"""Only a parsed, explicit rejection permits one pre-publication rewrite."""

import json
from copy import deepcopy

import pytest
from test_default_text_delivery import Model, StreamingModel, setup, wait_for
from unit_review_recording import bound_unit_packets as bound_unit_packets
from unit_review_recording import recorded_review_packet
from virtual_companion.text.default_runtime import ACCESS_INITIAL
from virtual_companion.text.fake import synthetic_unit_opinion


@pytest.mark.parametrize('user_text,prefix,tail,support,raw_decision', [
    ('最近总觉得房间很安静，想和你聊聊这种感觉。',
     '安静得久了，房间里的声音会慢慢显出来：冰箱的嗡嗡声、水管里偶尔的一下、窗外远远的车声。',
     '想聊这种感觉的话，你留意到的安静是什么样的——是空落落的，还是那种让人松下来的静？',
     'unsupported', 'reject'),
    ('刚整理完桌上的一堆旧票根。',
     '旧票根攒着攒着就是一小叠，翻起来像把一段时间压成了几张纸片。',
     '整理的时候，大概也是在决定哪些留、哪些可以放下了。',
     'uncertain', 'approve'),
])
def test_v24_rejected_prefix_stops_before_append_and_uses_one_bound_repair(
    tmp_path, monkeypatch, user_text, prefix, tail, support, raw_decision,
):
    from virtual_companion.text.default_runtime import REPAIR_REPLY

    repaired = '小小的物件或一处安静，也能成为话题的起点。'

    class RecordedModel(Model):
        appended = False
        cancelled_original = False

        def review_peer(self):
            owner = self

            class Peer(Model):
                def stream(self, messages, cancelled):
                    owner.calls.append(deepcopy(messages))
                    packet = recorded_review_packet(messages)
                    raw = synthetic_unit_opinion(packet)
                    if not packet['final'] and packet['candidate_prefix'] == prefix:
                        raw.update(decision=raw_decision,
                                   constraints_satisfied=raw_decision != 'reject')
                        raw['segments'][0]['commitments'] = [dict(
                            kind='assertion', claim=prefix, support=support, sources=[])]
                    yield json.dumps(raw, ensure_ascii=False)

            return Peer()

        def stream(self, messages, cancelled):
            self.calls.append(deepcopy(messages))
            if any(message['content'].startswith(REPAIR_REPLY) for message in messages):
                yield json.dumps(dict(action='answer', text=repaired), ensure_ascii=False)
                return
            yield '{"action":"answer","text":' + json.dumps(prefix, ensure_ascii=False)[:-1]
            self.appended = True  # v24 resumed the rejected candidate at this point.
            yield json.dumps(tail, ensure_ascii=False)[1:-1] + '"}'

        def cancel(self):
            self.cancelled_original = True

    model = RecordedModel()
    runtime, session, memory, mirror, _ = setup(tmp_path, model)
    prepares = []
    original_prepare = runtime._prepare_turn_stage

    def prepare(claim, *, final, repair=None):
        prepares.append(repair)
        return original_prepare(claim, final=final, repair=repair)

    monkeypatch.setattr(runtime, '_prepare_turn_stage', prepare)
    try:
        session.send(user_text)
        event = wait_for(lambda: runtime.poll()['event'])
        assert event['display_text'] == repaired
        assert not model.appended and model.cancelled_original
        assert len(model.calls) == 4 and len(prepares) == 2
        assert prepares[1].text == prefix
        assert prepares[1].verdict['decision'] == 'reject'
        assert prepares[1].verdict['model_opinion']['decision'] == raw_decision
        initial, rejected_review, repair, replacement_review = model.calls
        before, after = (recorded_review_packet(call) for call in (
            rejected_review, replacement_review))
        assert before['final'] is False and before['displayed_prefix'] == ''
        assert before['candidate_prefix'] == prefix
        assert after['candidate_prefix'] == repaired
        assert after['basis'] == before['basis'] and after['context'] == before['context']
        payload = next(message['content'][len(REPAIR_REPLY):] for message in repair
                       if message['content'].startswith(REPAIR_REPLY))
        assert json.loads(payload) == dict(rejected_candidate=prefix, review=prepares[1].verdict)
        assert [message for message in repair
                if not message['content'].startswith(REPAIR_REPLY)] == initial
        runtime.ack(dict(sequence=event['sequence'], fingerprint=event['fingerprint'],
                         host_receipt='synthetic-v24-rejected-prefix-repair'))
        wait_for(lambda: not session.snapshot()['busy'])
        assert session.snapshot()['attempt']['status'] == 'completed'
        assert mirror.history()['messages'][-1]['content'] == repaired
        assert memory.store.stream_state(
            memory.companion, runtime.claim.input_id)['confirmed_text'] == repaired
    finally:
        memory.close()
        mirror.close()


class RejectThenRepair(Model):
    def __init__(self, *, repeat=False, interrupt=None):
        super().__init__()
        self.repeat, self.interrupt = repeat, interrupt
        self.reviews = 0

    def stream(self, messages, cancelled):
        self.calls.append(deepcopy(messages))
        if '你只审查候选文字' in messages[0]['content']:
            self.reviews += 1
            reject = self.reviews == 1 or self.repeat
            if self.interrupt:
                self.interrupt()
            yield json.dumps(synthetic_unit_opinion(recorded_review_packet(messages),
                decision='reject' if reject else 'approve',
                constraints_satisfied=not reject))
        else:
            text = '我已经记住了。' if self.reviews == 0 else '听起来你很喜欢这本作品。'
            yield json.dumps({'action': 'answer', 'text': text}, ensure_ascii=False)


def test_zero_approved_rejection_reenters_only_host_prepare_callback(tmp_path, monkeypatch):
    from virtual_companion.text.turn_orchestrator import CandidateRepair

    runtime, session, memory, mirror, model = setup(tmp_path, RejectThenRepair())
    prepares, fences = [], []
    original_prepare, original_fence = runtime._prepare_turn_stage, runtime.has_approved_prefix

    def prepare(claim, *, final, repair=None):
        stage = original_prepare(claim, final=final, repair=repair)
        prepares.append((final, repair, stage))
        return stage

    def fence():
        result = original_fence()
        fences.append(result)
        return result

    monkeypatch.setattr(runtime, "_prepare_turn_stage", prepare)
    monkeypatch.setattr(runtime, "has_approved_prefix", fence)
    try:
        session.send("我喜欢这本作品。")
        event = wait_for(lambda: runtime.poll()["event"])
        assert fences == [False] and len(prepares) == 2
        _, repair, repaired_stage = prepares[-1]
        assert isinstance(repair, CandidateRepair)
        assert repair.prepared is prepares[0][2]
        assert repair.text == "我已经记住了。"
        assert repair.verdict["decision"] == "reject"
        assert repaired_stage.request is repair.prepared.request
        assert len(model.calls) == 4
        runtime.ack(dict(sequence=event["sequence"], fingerprint=event["fingerprint"],
                         host_receipt="synthetic-host-repair-callback"))
        wait_for(lambda: not session.snapshot()["busy"])
        assert session.snapshot()["attempt"]["status"] == "completed"
    finally:
        memory.close()
        mirror.close()


def test_approved_prefix_rejection_fences_orchestrator_rewrite(tmp_path, monkeypatch):
    runtime, session, memory, mirror, model = setup(tmp_path, StreamingModel(ending="reject"))
    fences, repairs = [], []
    original_prepare, original_fence = runtime._prepare_turn_stage, runtime.has_approved_prefix

    def prepare(claim, *, final, repair=None):
        if repair is not None:
            repairs.append(repair)
        return original_prepare(claim, final=final, repair=repair)

    def fence():
        result = original_fence()
        fences.append(result)
        return result

    monkeypatch.setattr(runtime, "_prepare_turn_stage", prepare)
    monkeypatch.setattr(runtime, "has_approved_prefix", fence)
    try:
        session.send("已有批准前缀的合成输入")
        event = wait_for(lambda: runtime.poll()["event"])
        runtime.ack(dict(sequence=event["sequence"], fingerprint=event["fingerprint"],
                         host_receipt="synthetic-immutable-prefix"))
        model.release.set()
        wait_for(lambda: not session.snapshot()["busy"])
        state = memory.store.stream_state(memory.companion, runtime.claim.input_id)
        assert fences == [True] and repairs == []
        assert state["outcome"] == "failed" and state["confirmed_units"] == 1
        assert state["confirmed_text"] == event["text"]
        assert mirror.history()["messages"][-1]["content"] == event["text"]
        assert sum("你只审查候选文字" not in call[0]["content"] for call in model.calls) == 1
    finally:
        model.release.set()
        memory.close()
        mirror.close()


def test_rejected_candidate_rewritten_once_under_same_context_then_delivered(tmp_path):
    model = RejectThenRepair()
    runtime, session, memory, mirror, _ = setup(tmp_path, model)
    try:
        session.send('我喜欢这本作品。')
        event = wait_for(lambda: runtime.poll()['event'])
        assert len(model.calls) == 4
        initial, first_review, repair, final_review = model.calls
        assert repair[:-2] == initial[:-1]  # Original system and evidence retained.
        assert repair[-1] == initial[-1]  # No fabricated user feedback.
        assert '我已经记住了。' in repair[-2]['content']
        before = recorded_review_packet(first_review)
        after = recorded_review_packet(final_review)
        assert before['context'] == after['context']
        assert before['basis'] == after['basis']
        assert all(ACCESS_INITIAL in call[0]['content'] for call in model.calls)
        assert after['displayed_prefix'] == ''
        assert event['display_text'] == after['candidate_prefix'] == '听起来你很喜欢这本作品。'
        assert len(memory.store.segments(memory.companion, runtime.claim.input_id)) == 1
        runtime.ack(dict(sequence=event['sequence'], fingerprint=event['fingerprint'],
                         host_receipt='synthetic-repaired-visible'))
        wait_for(lambda: not session.snapshot()['busy'])
        assert mirror.history()['messages'][-1]['content'] == event['display_text']
        assert session.snapshot()['attempt']['status'] == 'completed'
    finally:
        memory.close()
        mirror.close()


def test_second_rejection_fails_without_publication_or_more_calls(tmp_path):
    model = RejectThenRepair(repeat=True)
    runtime, session, memory, mirror, _ = setup(tmp_path, model)
    try:
        session.send('我喜欢这本作品。')
        wait_for(lambda: not session.snapshot()['busy'])
        assert session.snapshot()['attempt']['status'] == 'failed'
        assert len(model.calls) == 4
        assert not memory.store.segments(memory.companion, runtime.claim.input_id)
    finally:
        memory.close()
        mirror.close()


@pytest.mark.parametrize('interruption', ['cancel', 'context'])
def test_invalidated_review_never_starts_rewrite(tmp_path, interruption):
    model = RejectThenRepair()
    runtime, session, memory, mirror, _ = setup(tmp_path, model)
    try:
        def interrupt():
            if interruption == 'cancel':
                runtime.cancelled.set()
            else:
                memory.store.accept(runtime.config, 'newer-input', '另一条输入',
                                    session_id=runtime.claim.session_id)

        model.interrupt = interrupt
        session.send('我喜欢这本作品。')
        wait_for(lambda: not session.snapshot()['busy'])
        assert session.snapshot()['attempt']['status'] in {'failed', 'cancelled'}
        assert len(model.calls) == 2
        assert not memory.store.segments(memory.companion, runtime.claim.input_id)
    finally:
        memory.close()
        mirror.close()


@pytest.mark.parametrize('failure', ['hold', 'malformed', 'transport'])
def test_nonrejection_review_failure_is_not_rewritten(tmp_path, failure):
    class NonRejection(RejectThenRepair):
        def stream(self, messages, cancelled):
            if '你只审查候选文字' not in messages[0]['content']:
                yield from super().stream(messages, cancelled)
                return
            self.calls.append(deepcopy(messages))
            if failure == 'transport':
                raise OSError('synthetic transport error')
            if failure == 'malformed':
                yield '{"decision":"reject"}'  # No bound/validated verdict.
                return
            yield json.dumps(synthetic_unit_opinion(recorded_review_packet(messages),
                decision='hold',
                self_contained=False, no_open_dependency=False,
                consistent_with_displayed=True, constraints_satisfied=False))

    model = NonRejection()
    runtime, session, memory, mirror, _ = setup(tmp_path, model)
    try:
        session.send('我喜欢这本作品。')
        wait_for(lambda: not session.snapshot()['busy'])
        assert session.snapshot()['attempt']['status'] == 'failed'
        assert len(model.calls) == 2
        assert not memory.store.segments(memory.companion, runtime.claim.input_id)
    finally:
        memory.close()
        mirror.close()
@pytest.mark.parametrize("final_closed", [True, False])
def test_s08_valid_approval_with_open_dependency_holds_without_ordinary_exception(
    tmp_path, final_closed,
):
    from test_default_text_delivery import Model, wait_for
    from test_default_text_delivery import setup as default_setup
    from virtual_companion.boundary.presentation import digest
    from virtual_companion.cognition.stream_unit_review import _bind_audit
    from virtual_companion.text.fake import synthetic_unit_opinion

    first = "相册能把不同时候的画面放在一起。"
    reply = first + "一页一页翻也有自己的节奏。"
    opinions = []

    class OpenDependencyModel(Model):
        def review_peer(self):
            owner = self

            class Peer(Model):
                def stream(self, messages, cancelled):
                    owner.calls.append(deepcopy(messages))
                    packet = recorded_review_packet(messages)
                    raw = synthetic_unit_opinion(packet)
                    raw["no_open_dependency"] = packet["final"] and final_closed
                    normalized = _bind_audit(packet, raw, digest(packet))
                    opinions.append((deepcopy(packet), deepcopy(raw), normalized))
                    yield json.dumps(raw)

            return Peer()

        def stream(self, messages, cancelled):
            self.calls.append(deepcopy(messages))
            yield '{"action":"answer","text":' + json.dumps(first, ensure_ascii=False)[:-1]
            yield reply[len(first):] + '"}'

    model = OpenDependencyModel()
    runtime, session, memory, mirror, _ = default_setup(tmp_path, model)
    try:
        session.send("刚翻到一本旧相册。")
        if final_closed:
            event = wait_for(lambda: runtime.poll()["event"])
            assert event["display_text"] == reply
            runtime.ack(dict(sequence=event["sequence"], fingerprint=event["fingerprint"],
                             host_receipt="synthetic-open-dependency-complete"))
        wait_for(lambda: not session.snapshot()["busy"])
        assert session.snapshot()["attempt"]["status"] == (
            "completed" if final_closed else "failed")
        assert len(model.calls) == 3 and len(opinions) == 2
        packet, raw, early = opinions[0]
        assert packet["final"] is False and packet["candidate_prefix"] == first
        assert raw["decision"] == early["model_opinion"]["decision"] == "approve"
        assert early["decision"] == "hold" and early["no_open_dependency"] is False
        assert opinions[-1][0]["final"] is True
        assert opinions[-1][2]["decision"] == ("approve" if final_closed else "hold")
        assert not any("rejected_candidate" in row["content"]
                       for call in model.calls for row in call)
        segments = memory.store.segments(memory.companion, runtime.claim.input_id)
        assert len(segments) == int(final_closed)
        if not final_closed:
            assert runtime.poll()["event"] is None
            assert not any(row["role"] == "assistant" for row in mirror.history()["messages"])
    finally:
        memory.close()
        mirror.close()
