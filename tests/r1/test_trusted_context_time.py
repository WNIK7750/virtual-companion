"""Sourced host time wiring; recordings do not prove a model's time reasoning."""

import json
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from threading import Event
from types import SimpleNamespace

import pytest
from virtual_companion.boundary import history_readset as readsets
from virtual_companion.boundary.contracts import OwnerToken, Scope, VersionConflict, canonical
from virtual_companion.boundary.evidence import EvidenceTurn
from virtual_companion.boundary.turns import RunClaim
from virtual_companion.cognition import short_dialogue as prompts
from virtual_companion.cognition import stream_unit_review as audits
from virtual_companion.cognition.short_context import ShortContextAdmission
from virtual_companion.cognition.short_dialogue import ShortDialogue, ShortRequest
from virtual_companion.cognition.stream_dialogue import StreamDialogue
from virtual_companion.providers.text_deltas import UnapprovedTextDelta
from virtual_companion.text import default_runtime
from virtual_companion.text.default_runtime import DefaultTextRuntime
from virtual_companion.text.fake import synthetic_unit_opinion
from virtual_companion.text.identity import CompanionConfig, VersionedReference
from virtual_companion.text.turn_orchestrator import TurnOrchestrator

A = '00000000-0000-0000-0000-000000000001'
NOW = 1791285068977900
EXPIRY = 1791305394928000
EXPIRY_TEXT = '2026-10-06T16:49:54.928000+00:00'
CURRENT = '帮我回顾一下：这次放松安排做什么？'
VALUE = '请记住：这次放松选散步；仅在 2026-10-06T16:49:54.928Z 之前有效。'
REVISION = 'short-dialogue.v31'
CANDIDATE = 'short-dialogue.v32'
SPARSE_CANDIDATE = 'short-dialogue.v33'


class Store:
    """In-memory effective ports only; no SQLite import or connection."""

    def __init__(self):
        self.version = 1
        self.fact = {'record_id': 'fact-original', 'version': 1, 'source_id': 'seed-original',
                     'content': VALUE, 'status': 'active', 'expires_at': EXPIRY_TEXT}
        self.selected_at = []
        self.approved = []

    def snapshot(self, _):
        return SimpleNamespace(companion_id=A, write_version=self.version)

    def inspect(self, *_):
        return {'text': CURRENT}

    def effective_exchange_context(self, claim, *, now_us):
        self.selected_at.append(now_us)
        turn = EvidenceTurn(Scope(A, claim.session_id), claim.input_id, 'user',
                            CURRENT, len(CURRENT))
        return (turn,), None

    def recall(self, _, *, now_us, limit, candidates=None):
        self.selected_at.append(now_us)
        assert limit == 8
        return [deepcopy(self.fact)] if now_us < EXPIRY and self.fact['status'] == 'active' else []

    def pending_exchanges(self, _, *, now_us, limit, candidates=None):
        self.selected_at.append(now_us)
        return []

    def transaction(self, *, write):
        assert write is False
        return nullcontext(None)

    def _segment_live(self, _, claim, now):
        assert claim.companion_id == A
        if claim.state_version != self.version:
            raise VersionConflict('Fake current state changed')

    def segments(self, *_):
        return deepcopy(self.approved)


def claim():
    return RunClaim(A, 'session-original', 'current', 'attempt', OwnerToken(A, 'owner', 1),
                    1, '{}', NOW + 120_000_000)


def admitted(store=None, now=None, *, timed=True):
    store = store or Store()
    now = now or [NOW]
    admission = ShortContextAdmission(store, lambda *args: ShortRequest(args[-1]),
                                      evidence_clock=lambda: now[0])
    kwargs = {'observe_time': True} if timed else {}
    request = admission(claim(), store.snapshot(A), CURRENT, **kwargs)
    return store, now, admission, request


class Model:
    supports_stage_metadata = True

    def __init__(self):
        self.counted, self.sent, self.prepared = [], [], []
        self.after_count = lambda: None

    def count(self, body):
        self.counted.append(deepcopy(body))
        self.after_count()

    def invoke(self, body):
        self.sent.append(deepcopy(body))
        return dict(text='这次放松选散步。', tool_calls=[], finish_reason='stop')

    def prepare_stage_request(self, messages, **metadata):
        model = self
        snapshot = deepcopy(messages)
        self.prepared.append({'messages': snapshot, **deepcopy(metadata)})

        class Wire:
            def check_context(self):
                model.counted.append(deepcopy(snapshot))
                model.after_count()

            def stream(self, _):
                model.sent.append(deepcopy(snapshot))
                yield '{"action":"answer","text":"这次放松选散步。"}'

        return Wire()


def runtime(monkeypatch, revision=REVISION):
    store, now, model = Store(), [NOW], Model()
    config = CompanionConfig(A, VersionedReference('synthetic', 'v1', 'a' * 64))
    memory = SimpleNamespace(store=store, companion=A, config=config,
                             generation_gate=SimpleNamespace(foreground=lambda _: nullcontext()))
    monkeypatch.setattr(DefaultTextRuntime, 'reconcile', lambda _: None)
    root = DefaultTextRuntime(memory, None, model, binding='synthetic-cpu',
                              clock=lambda: now[0], prompt_revision=revision)
    root.claim, root.cancelled, root.deadline = claim(), Event(), NOW + 120_000_000
    root.guard = lambda *args, **kwargs: nullcontext()
    return root, store, now, model


def context_time(request):
    return json.loads(request.context.payload)['context_time']


def test_host_time_is_the_exact_same_now_used_by_effective_ports_and_context_hash():
    store, _, _, request = admitted()
    value = context_time(request)
    assert value == {'protocol': 'host-context-time.v1', 'source': 'admission.evidence_clock',
                     'clock': 'utc_epoch_microseconds', 'observed_at_us': NOW,
                     'observed_at_utc': '2026-10-06T11:11:08.977900Z',
                     'selection': 'effective_at_observation'}
    assert store.selected_at and set(store.selected_at) == {NOW}
    readset = readsets.history_readset(A, 'current', 1, request.context.payload)
    assert readset['protocol'] == 'history-readset.v1'
    assert readset['context_sha256'] == readsets.evidence_hash(json.loads(request.context.payload))
    assert readset['facts'][0]['sha256'] == readsets.evidence_hash(store.fact)
    assert json.loads(request.context.payload)['history'] == []


def test_old_admission_payload_and_default_request_have_no_time_field():
    _, _, _, request = admitted(timed=False)
    assert 'context_time' not in json.loads(request.context.payload)
    body = request.model_request(192)
    assert all('host-context-time.v1' not in message['content'] for message in body['messages'])


def test_v31_generation_and_existing_v8_reviewer_receive_identical_host_observation():
    _, _, _, request = admitted()
    model = Model()
    ShortDialogue(model, cancelled=Event(), policy=REVISION).generate(request)
    assert model.counted == model.sent and len(model.sent) == 1
    time_messages = [message for message in model.sent[0]['messages']
                     if message['content'].startswith('{"context_time":')]
    assert time_messages == [{'role': 'system', 'content': canonical({
        'context_time': context_time(request)})}]
    seen = []

    packet = []

    def assess(envelope):
        packet[:] = [envelope]
        return assessor(envelope)

    # Keep the exact original sealed envelope; the fixture returns an opinion, not truth.
    def bound_invoke(stage, messages, schema, cancelled):
        envelope = packet[0]
        assert (
            json.loads(messages[-1]['content'])['context']['context_time'] == context_time(request)
        )
        seen.append(deepcopy(json.loads(messages[-1]['content'])))
        assert prompts.CONTEXT_TIME_SYSTEM in messages[0]['content']
        assert stage == 'stream_unit_review' and not cancelled.is_set()
        return synthetic_unit_opinion(envelope['payload'])

    assessor = audits.StructuredUnitAssessor(bound_invoke, Event(),
                                             audit_version=audits.VERDICT_UNIT_AUDIT_VERSION)
    reviewer = audits.BoundedUnitReviewer(claim(), request, assess, early_limit=0,
                                          audit_version=audits.VERDICT_UNIT_AUDIT_VERSION)
    assert reviewer('这次放松选散步。', True) is not None
    assert len(seen) == 1
    assert packet[0]['payload']['context'] == request.context.payload


def test_v32_candidate_changes_only_generation_content_selection(monkeypatch):
    baseline, _, _, _ = runtime(monkeypatch, REVISION)
    candidate, _, _, _ = runtime(monkeypatch, CANDIDATE)
    old_stage = baseline.prepare(baseline.claim, final=True)
    new_stage = candidate.prepare(candidate.claim, final=True)

    assert context_time(old_stage.request) == context_time(new_stage.request)
    assert old_stage.messages[1:] == new_stage.messages[1:]
    assert candidate.control.replace(prompts.PROMPTS[CANDIDATE],
                                     prompts.PROMPTS[REVISION], 1) == baseline.control
    assert new_stage.messages[0]['content'].replace(candidate.control,
                                                     baseline.control, 1) == (
        old_stage.messages[0]['content'])
    assert candidate.review_audit_version == baseline.review_audit_version
    assert candidate.repair_reply == baseline.repair_reply
    assert prompts.PROMPTS[CANDIDATE].startswith(prompts.PROMPTS[REVISION])
    assert prompts.PROMPTS[CANDIDATE] != prompts.PROMPTS[REVISION]

    _, _, _, frozen = admitted()
    old_model, new_model = Model(), Model()
    ShortDialogue(old_model, cancelled=Event(), policy=REVISION).generate(frozen)
    ShortDialogue(new_model, cancelled=Event(), policy=CANDIDATE).generate(frozen)
    assert old_model.sent[0]['messages'][1:] == new_model.sent[0]['messages'][1:]
    assert new_model.sent[0]['messages'][0]['content'].replace(
        prompts.PROMPTS[CANDIDATE], prompts.PROMPTS[REVISION], 1
    ) == old_model.sent[0]['messages'][0]['content']


@pytest.mark.parametrize('new_revision', ['short-dialogue.v33', 'short-dialogue.v34',
                                        'short-dialogue.v35'])
def test_v33_sparse_sharing_candidate_is_independent_of_v32_and_keeps_v31_wiring(
    monkeypatch, new_revision,
):
    baseline, _, _, _ = runtime(monkeypatch, REVISION)
    candidate, _, _, _ = runtime(monkeypatch, new_revision)
    old_stage = baseline.prepare(baseline.claim, final=True)
    new_stage = candidate.prepare(candidate.claim, final=True)

    assert prompts.PROMPTS[new_revision].startswith(prompts.PROMPTS[REVISION])
    assert not prompts.PROMPTS[new_revision].startswith(prompts.PROMPTS[CANDIDATE])
    assert candidate.control.replace(prompts.PROMPTS[new_revision],
                                     prompts.PROMPTS[REVISION], 1) == baseline.control
    assert context_time(old_stage.request) == context_time(new_stage.request)
    assert old_stage.messages[1:] == new_stage.messages[1:]
    assert candidate.review_audit_version == baseline.review_audit_version
    assert candidate.repair_reply == baseline.repair_reply
    assert new_revision in default_runtime.DIAGNOSTIC_PROMPT_REVISIONS

    _, _, _, frozen = admitted()
    old_model, new_model = Model(), Model()
    ShortDialogue(old_model, cancelled=Event(), policy=REVISION).generate(frozen)
    ShortDialogue(new_model, cancelled=Event(), policy=new_revision).generate(frozen)
    assert old_model.sent[0]['messages'][1:] == new_model.sent[0]['messages'][1:]
    assert new_model.sent[0]['messages'][0]['content'].replace(
        prompts.PROMPTS[new_revision], prompts.PROMPTS[REVISION], 1
    ) == old_model.sent[0]['messages'][0]['content']


@pytest.mark.parametrize('change', ['missing', 'bool', 'negative', 'utc', 'source', 'extra'])
def test_required_time_missing_malformed_or_conflicting_fails_before_count(change):
    _, _, _, request = admitted()
    data = json.loads(request.context.payload)
    if change == 'missing':
        del data['context_time']
    elif change == 'bool':
        data['context_time']['observed_at_us'] = True
    elif change == 'negative':
        data['context_time']['observed_at_us'] = -1
    elif change == 'utc':
        data['context_time']['observed_at_utc'] = EXPIRY_TEXT
    elif change == 'source':
        data['context_time']['source'] = 'user_text'
    else:
        data['context_time']['guess'] = 'today'
    forged = replace(request, context=replace(request.context, payload=canonical(data)))
    model = Model()
    with pytest.raises(ValueError, match='time|Time'):
        ShortDialogue(model, cancelled=Event(), policy=REVISION).generate(forged)
    assert model.counted == model.sent == []


def test_first_validation_clock_rollback_and_exact_expiry_fail_closed():
    _, now, gate, request = admitted()
    now[0] = NOW - 1
    with pytest.raises(VersionConflict, match='backwards'):
        gate.validate(request.context)
    now[0] = EXPIRY
    with pytest.raises(VersionConflict, match='expired'):
        gate.validate(request.context)


def test_runtime_v31_selected_only_old_revision_wire_is_byte_identical(monkeypatch):
    new, _, _, _ = runtime(monkeypatch)
    stage = new.prepare(new.claim, final=True)
    assert context_time(stage.request)['observed_at_us'] == NOW
    assert new.review_audit_version == audits.VERDICT_UNIT_AUDIT_VERSION
    from virtual_companion.text.default_runtime import NATURAL_REPAIR_REPLY

    assert new.repair_reply == NATURAL_REPAIR_REPLY
    for revision in ('short-dialogue.v12', 'short-dialogue.v19', 'short-dialogue.v30'):
        old, _, _, _ = runtime(monkeypatch, revision)
        previous = old.prepare(old.claim, final=True)
        assert 'context_time' not in json.loads(previous.request.context.payload)
        assert all('host-context-time.v1' not in item['content'] for item in previous.messages)


@pytest.mark.parametrize('failure', ['cancel', 'expiry', 'rollback', 'revision'])
def test_runtime_rechecks_after_count_and_never_sends_stale_timed_context(monkeypatch, failure):
    root, store, now, model = runtime(monkeypatch)
    stage = root.prepare(root.claim, final=True)

    def change():
        if failure == 'cancel':
            root.cancelled.set()
        elif failure == 'expiry':
            now[0] = EXPIRY
            root.deadline = EXPIRY + 1_000_000
        elif failure == 'rollback':
            now[0] = NOW - 1
        else:
            store.version += 1

    model.after_count = change
    with pytest.raises((InterruptedError, VersionConflict)):
        stage.check_context()
    assert model.sent == [] and stage.checked is False


def test_runtime_healthy_timed_prepared_messages_are_counted_and_sent_without_reclock(monkeypatch):
    root, _, now, model = runtime(monkeypatch)
    stage = root.prepare(root.claim, final=True)
    stage.check_context()
    now[0] += 1000
    assert list(stage.stream(root.cancelled))
    assert model.sent == model.counted
    assert context_time(stage.request)['observed_at_us'] == NOW
    assert sha256(canonical(model.sent[0]).encode()).hexdigest()


@pytest.mark.parametrize('boundary', [EXPIRY - 1, EXPIRY, EXPIRY + 1])
def test_absolute_cutoff_never_moves_when_context_is_rebuilt(boundary):
    store, _, _, request = admitted(now=[boundary])
    data = json.loads(request.context.payload)
    assert data['context_time']['observed_at_us'] == boundary
    assert len(data['memories']) == (1 if boundary < EXPIRY else 0)
    assert store.fact['expires_at'] == EXPIRY_TEXT and store.fact['version'] == 1


@pytest.mark.parametrize('drift', [EXPIRY, NOW - 1])
def test_slow_non_token_validation_cannot_cross_expiry_or_roll_clock_back(drift):
    store, now, gate, request = admitted()
    original = store.recall

    def delayed(*args, **kwargs):
        result = original(*args, **kwargs)
        now[0] = drift
        return result

    store.recall = delayed
    with pytest.raises(VersionConflict, match='expired|backwards'):
        gate.validate(request.context)


def test_cached_context_clock_and_original_deadline_are_not_extended():
    store, now = Store(), [NOW]

    def token(_, *, now_us, include_deadline):
        return (1, 0, 0, False, 0, EXPIRY if include_deadline else None)

    store.context_validation_token = token
    _, _, gate, request = admitted(store, now)
    gate.validate(request.context)
    now[0] += 1
    gate.validate(request.context)
    now[0] = EXPIRY
    with pytest.raises(VersionConflict, match='expired'):
        gate.validate(request.context)
    now[0] = NOW
    with pytest.raises(VersionConflict, match='backwards'):
        gate.validate(request.context)


@pytest.mark.parametrize('bad_clock', [True, float(NOW), None, str(NOW)])
@pytest.mark.parametrize('cached', [False, True])
def test_timed_token_validation_rejects_wrong_clock_type_at_its_final_read(bad_clock, cached):
    store, now = Store(), [NOW]
    fail = [False]

    def token(_, *, now_us, include_deadline):
        if fail[0]:
            now[0] = bad_clock
        return (1, 0, 0, False, 0, EXPIRY if include_deadline else None)

    store.context_validation_token = token
    _, _, gate, request = admitted(store, now)
    if cached:
        gate.validate(request.context)
    fail[0] = True
    with pytest.raises(VersionConflict, match='backwards'):
        gate.validate(request.context)


@pytest.mark.parametrize('expiry', ['2026-10-06T16:49:54.928', 1791305394928000,
                                   'not-a-date', '2026-10-06T11:11:08.977900Z'])
def test_malformed_or_already_expired_selected_record_cannot_be_marked_effective(expiry):
    store = Store()
    store.fact['expires_at'] = expiry
    with pytest.raises(ValueError, match='time|expiry|expired'):
        admitted(store)


def test_timezone_offset_represents_the_same_original_absolute_cutoff():
    store = Store()
    store.fact['expires_at'] = '2026-10-07T00:49:54.928000+08:00'
    _, _, _, request = admitted(store)
    assert (
        json.loads(request.context.payload)['memories'][0]['expires_at'] == store.fact['expires_at']
    )
    assert readsets.validate_context_time(
        context_time(request), json.loads(request.context.payload)
    )


def test_same_v31_review_preserves_original_time_context_in_zero_approved_repair(monkeypatch):
    root, store, _, _ = runtime(monkeypatch)
    stage = root.prepare(root.claim, final=True)
    packet = []

    def reject(envelope):
        packet[:] = [deepcopy(envelope)]
        opinion = synthetic_unit_opinion(envelope['payload'])
        opinion.update(decision='reject', checks=['pass', 'pass', 'pass', 'fail'],
                       failure={'scope': 'accuracy', 'segment_id': None,
                                'basis': 'synthetic temporal claim is unverified'})
        return audits._bind_audit(envelope['payload'], opinion, envelope['sha256'])

    reviewer = audits.BoundedUnitReviewer(claim(), stage.request, reject, early_limit=0,
                                          audit_version=audits.VERDICT_UNIT_AUDIT_VERSION)
    candidate = '这条安排现在已到期。'
    with pytest.raises(audits.CandidateRejected) as error:
        reviewer(candidate, True)
    verdict = error.value.verdict
    repair = SimpleNamespace(prepared=stage, text=candidate, verdict=verdict)
    fixed = root._prepare_turn_stage(root.claim, final=True, repair=repair)
    assert fixed.request is stage.request
    assert context_time(fixed.request) == context_time(stage.request)
    feedback = next(item['content'] for item in fixed.messages
                    if item['content'].startswith(root.repair_reply))
    payload = json.loads(feedback[len(root.repair_reply):])
    assert audits.restore_repair_review(payload['rejected_candidate'], payload['review']) == verdict
    assert packet[0]['payload']['context'] == stage.request.context.payload
    store.approved.append({'sequence': 1, 'status': 'confirmed', 'text': '原已显示句。'})
    with pytest.raises(ValueError, match='approved prefix'):
        root._prepare_turn_stage(root.claim, final=True, repair=repair)
    assert store.approved == [{'sequence': 1, 'status': 'confirmed', 'text': '原已显示句。'}]








def test_finite_eight_unit_review_keeps_one_time_record_and_existing_output_schema():
    _, _, _, request = admitted()
    recorded = []

    def assess(envelope):
        opinion = synthetic_unit_opinion(envelope['payload'])
        recorded.append((envelope, opinion))
        return audits._bind_audit(envelope['payload'], opinion, envelope['sha256'])

    reviewer = audits.BoundedUnitReviewer(claim(), request, assess, early_limit=0,
                                          audit_version=audits.VERDICT_UNIT_AUDIT_VERSION)
    candidate = ''.join(f'这是第{index}条录制短句。' for index in range(1, 9))
    assert reviewer(candidate, True) is not None
    envelope, opinion = recorded[0]
    assert len(opinion['units']) == 8
    assert json.loads(envelope['payload']['context'])['context_time'] == context_time(request)
    assert 'context_time' not in opinion
    assert audits._verdict_unit_model_schema()['properties']['units']['maxItems'] == 8


@pytest.mark.parametrize('second_reject', [True, False])
def test_actual_orchestrator_keeps_one_repair_and_the_original_time_bound_context(monkeypatch,
                                                                              second_reject):
    root, store, _, model = runtime(monkeypatch)
    stages, settlements = [], []

    def deliver(stage, answer):
        text = ''.join(answer)
        stages.append(stage)

        def assess(envelope):
            opinion = synthetic_unit_opinion(envelope['payload'])
            if len(stages) == 1 or second_reject:
                opinion.update(decision='reject', checks=['pass', 'pass', 'pass', 'fail'],
                               failure={'scope': 'accuracy', 'segment_id': None,
                                        'basis': 'synthetic unresolved time statement'})
            return audits._bind_audit(envelope['payload'], opinion, envelope['sha256'])

        reviewer = audits.BoundedUnitReviewer(claim(), stage.request, assess, early_limit=0,
                                              audit_version=audits.VERDICT_UNIT_AUDIT_VERSION)
        assert reviewer(text, True) is not None
        store.approved.append({'sequence': 1, 'status': 'confirmed', 'text': text})
        return text

    turn = TurnOrchestrator(root.claim, root.cancelled, prepare=root._prepare_turn_stage,
        resolve_current=lambda _: None, refresh=lambda owner: owner, check=root.check,
        has_approved_prefix=root.has_approved_prefix, deliver=deliver,
        settle_failure=lambda _: settlements.append('failure'), settle_listen=lambda _: None)
    if second_reject:
        with pytest.raises(audits.CandidateRejected):
            list(turn.stream(root.cancelled))
        assert store.approved == [] and settlements == ['failure']
    else:
        assert list(turn.stream(root.cancelled)) == ['这次放松选散步。']
        assert len(store.approved) == 1 and settlements == []
    assert len(model.sent) == 2 and len(stages) == 2
    assert stages[0].request is stages[1].request
    assert context_time(stages[0].request) == context_time(stages[1].request)


@pytest.mark.parametrize('crossing', ['healthy', 'after_review', 'after_visible'])
def test_actual_stream_ports_recheck_ttl_and_never_erase_the_supplied_visible_ack(crossing):
    store, now, gate, request = admitted()
    calls = []
    store.supports_history_readset = True
    text = '这次放松选散步。'

    def approve(owner, proposal, *, now_us, clock, readset):
        assert owner == claim() and now_us == clock() < EXPIRY
        assert readset == readsets.history_readset(A, 'current', 1, request.context.payload)
        calls.append('approve')
        store.approved.append({'sequence': 1, 'status': 'approved', 'text': text})
        return {'sequence': 1}

    def dispatch(owner, sequence, *, now_us, clock):
        assert owner == claim() and sequence == 1 and now_us == clock() < EXPIRY
        calls.append('dispatch')
        store.approved[0]['status'] = 'unknown'
        return {'sequence': sequence, 'text': text, 'fingerprint': 'synthetic-fingerprint'}

    def confirm(companion, input_id, sequence, **receipt):
        assert (companion, input_id, sequence) == (A, 'current', 1)
        assert receipt == {'fingerprint': 'synthetic-fingerprint',
                           'host_receipt': 'supplied-visible-receipt', 'channel': 'text_visible'}
        calls.append('confirm')
        store.approved[0]['status'] = 'confirmed'

    store.approve_segment, store.dispatch_segment, store.confirm_segment = (
        approve, dispatch, confirm
    )
    store.finish_stream = lambda *args, **kwargs: {'outcome': kwargs['outcome']}

    def assess(envelope):
        opinion = synthetic_unit_opinion(envelope['payload'])
        if crossing == 'after_review':
            now[0] = EXPIRY
        return audits._bind_audit(envelope['payload'], opinion, envelope['sha256'])

    reviewer = audits.BoundedUnitReviewer(claim(), request, assess, early_limit=1,
                                          audit_version=audits.VERDICT_UNIT_AUDIT_VERSION)

    def present(event):
        calls.append('present')
        assert event['text'] == text and store.approved[0]['status'] == 'unknown'
        if crossing == 'after_visible':
            now[0] = EXPIRY
        return 'supplied-visible-receipt'

    class DeltaModel:
        def count(self, _):
            return 10

        def invoke(self, _):
            self.receive(UnapprovedTextDelta(1, text))
            return {'text': text, 'tool_calls': [], 'finish_reason': 'stop'}

        def cancel(self):
            calls.append('cancel')

    def factory(receive):
        model = DeltaModel()
        model.receive = receive
        return model

    coordinator = StreamDialogue(store=store, model_factory=factory, review_prefix=reviewer,
        present=present, checkpoint=lambda: gate.validate(request.context), cancelled=Event(),
        clock=lambda: now[0])
    if crossing == 'healthy':
        assert coordinator.generate(claim(), request)['outcome'] == 'completed'
        assert calls == ['approve', 'dispatch', 'present', 'confirm']
    else:
        with pytest.raises(VersionConflict, match='expired'):
            coordinator.generate(claim(), request)
        assert calls == (['cancel'] if crossing == 'after_review'
                         else ['approve', 'dispatch', 'present', 'confirm', 'cancel'])
    assert store.approved == ([] if crossing == 'after_review' else [
        {'sequence': 1, 'status': 'confirmed', 'text': text}])
