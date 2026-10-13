import json
from copy import deepcopy
from dataclasses import replace
from threading import Event

import pytest
from test_default_text_delivery import Model, setup
from test_dialogue_memory_integration import close, send_visible
from test_explicit_memory_management import list_command
from test_rebuilt_memory import mutation
from virtual_companion.boundary.contracts import Scope, VersionConflict


def test_fragment_rebuild_reopen_cross_owner_and_invalidation(tmp_path):
    from virtual_companion.boundary.dialogue_retrieval import fragment_reference

    runtime, text, memory, mirror, _ = setup(tmp_path, Model())
    try:
        event = send_visible(runtime, text, '旧话题：我偶尔喝热茶，不是每天。', 'old-visible')
        next_input = memory.store.accept(runtime.config, 'later', '再想起热茶。')
        fragments = memory.store.dialogue_fragments(memory.companion, next_input.input_id,
                                                   now_us=runtime.clock())
        assert len(fragments) == 1
        reopened = type(memory.store)(memory.store.path)
        assert reopened.dialogue_fragments(memory.companion, next_input.input_id,
                                           now_us=runtime.clock()) == fragments
        from test_rebuilt_turns import B, config
        other = memory.store.accept(config(B), 'other-owner', '同词热茶')
        assert reopened.dialogue_fragments(B, other.input_id, now_us=runtime.clock()) == ()
        assert (
            fragment_reference(fragments[0])["members"][0]["source_id"]
            == event["basis"]["input_id"]
        )
        owner = memory.store.owner(memory.companion) or memory.store.claim_owner(
            memory.companion, 'seed-owner', expected_epoch=0)
        command = replace(
            mutation("dialogue-memory-seed", record="music", content="我偶尔喝热茶。"),
            scope=Scope(memory.companion, event["basis"]["session_id"]),
            source_id=event["basis"]["input_id"],
        )
        memory.store.mutate(command, owner, expected_version=memory.store.snapshot(
            memory.companion).write_version)
        memory.store.execute_memory_command(list_command(companion=memory.companion),
                                            now_us=runtime.clock())
        assert memory.store.dialogue_fragments(memory.companion, next_input.input_id,
                                               now_us=runtime.clock()) == ()
    finally:
        close(runtime, text, memory, mirror)


def test_received_only_stays_independent_and_late_visible_rebuilds(tmp_path):
    from test_rebuilt_turns import A
    from test_received_context import receive_then_continue

    store, first, event, run, admission, request = receive_then_continue(tmp_path)
    assert json.loads(request.context.payload)['received_output'][0]['text'] == '我在这里。'
    assert store.dialogue_fragments(A, run.input_id, now_us=200) == ()
    store.confirm_segment(A, first.input_id, 1, fingerprint=event['fingerprint'],
                          host_receipt='late-visible', channel='text_visible')
    fragments = store.dialogue_fragments(A, run.input_id, now_us=200)
    assert len(fragments) == 1
    assert fragments[0]['members'][-1]['text'] == '我在这里。'
    assert 'received_output' not in json.loads(admission(
        run, store.snapshot(A), '接着刚才说的继续').context.payload)


def test_typed_catalog_rejects_fact_fragment_collision_even_when_fact_page_is_short(tmp_path):
    from test_rebuilt_turns import A
    from test_received_context import receive_then_continue
    from virtual_companion.boundary.dialogue_retrieval import DialogueCatalogStore

    store, first, event, run, _, _ = receive_then_continue(tmp_path)
    store.confirm_segment(A, first.input_id, 1, fingerprint=event['fingerprint'],
                          host_receipt='visible', channel='text_visible')
    fragment = store.dialogue_fragments(A, run.input_id, now_us=200)[0]
    command = replace(mutation('collision', record=fragment['fragment_id'], content='独立事件'),
                      source_kind='manual')
    store.mutate(command, store.owner(A), expected_version=store.snapshot(A).write_version)
    with pytest.raises(VersionConflict, match='identity collision'):
        DialogueCatalogStore(store, run.input_id).committed_catalog_page(A, now_us=200, limit=1)


def test_dialogue_catalog_preview_exposes_original_topic_with_typed_member_count(tmp_path):
    from test_rebuilt_turns import A
    from test_received_context import receive_then_continue
    from virtual_companion.boundary.dialogue_retrieval import DialogueCatalogStore
    from virtual_companion.text.memory_navigation import MemoryCatalog

    store, first, event, run, _, _ = receive_then_continue(tmp_path)
    store.confirm_segment(A, first.input_id, 1, fingerprint=event['fingerprint'],
                          host_receipt='preview-visible', channel='text_visible')
    catalog = MemoryCatalog(DialogueCatalogStore(store, run.input_id), A,
        state_version=run.state_version, clock=lambda: 200, checkpoint=None, dialogue=True)
    node = next(row['node_id'] for row in catalog.browse()['entries'] if row['label'] == 'dialogue')
    entry = catalog.browse(node)['entries'][0]
    assert entry['source_type'] == 'dialogue_fragment' and entry['member_count'] == 2
    assert '你好' in entry['summary'] and '我在这里。' in entry['summary']
    from virtual_companion.boundary.memory_relevance import lexical_relevance

    row = catalog.read(((entry['record_id'], entry['version']),))[0]
    assert row['content'] == 'user: 你好\nassistant: 我在这里。'
    assert lexical_relevance('你好')(row) > 0
    assert lexical_relevance('config_sha256')(row) == 0
    assert row['fragment']['members'][0]['text'] == '你好'


def make_fragment(index, *, text=None):
    from virtual_companion.boundary.history_readset import evidence_hash
    fragment = {
        "protocol": "dialogue-fragment.v1",
        "version": 1,
        "origin_session": "synthetic-old-session",
        "input_id": f"input-{index}",
        "config_sha256": "a" * 64,
        "character_context": None,
        "event_time": None,
        "event_time_status": "unknown",
        "part": 0,
        "response_completeness": "unknown",
        "membership_policy": "frozen_confirmed_subset",
        "members": [
            {
                "source_id": f"input-{index}",
                "role": "user",
                "text": text or f"原话{index}：并非每天。",
                "response_type": None,
                "visibility": "submitted",
            },
            {
                "source_id": f"answer-{index}",
                "role": "assistant",
                "text": f"当时回应{index}。",
                "response_type": "answer",
                "visibility": "confirmed_visible",
            },
        ],
    }
    for member in fragment['members']:
        member['sha256'] = evidence_hash(member['text'])
    fragment['fragment_id'] = 'dialogue-fragment-' + evidence_hash(fragment)
    return fragment


def mixed_session():
    from test_deep_memory_skills import Catalog
    from virtual_companion.boundary.dialogue_retrieval import fragment_row
    from virtual_companion.text.deep_memory import DeepMemorySession

    class Fragments(Catalog):
        def read(self, references):
            return tuple(fragment_row(make_fragment(int(record))) for record, _ in references)

    cancel = Event()
    return DeepMemorySession(catalog=Fragments(), lookup=lambda *args, **kwargs: (),
        checkpoint=lambda signal: None, cancelled=cancel, dialogue=True), cancel


def test_actual_members_share_eight_per_tool_twentyfour_union_and_six_calls():
    from test_readonly_evidence_recall import fact
    from virtual_companion.boundary.history_readset import validate_evidence_access

    session, cancel = mixed_session()
    session.seed([fact(index) for index in range(8)])
    for start in (0, 4):
        session.recall({'action': 'read_memory', 'references': [
            {'record_id': str(index), 'version': 1} for index in range(start, start + 4)]},
            cancelled=cancel)
    selected = session.selection
    assert len(selected.memories) == 8 and len(selected.retrieved_dialogue) == 8
    validate_evidence_access(selected.evidence_access, facts=list(selected.memories),
                             retrieved_dialogue=list(selected.retrieved_dialogue))
    limited = session.recall({'action': 'read_memory', 'references': [
        {'record_id': '8', 'version': 1}]}, cancelled=cancel)
    assert len(limited.retrieved_dialogue) == 8
    assert limited.evidence_access['stop_reason'] == 'record_limit'
    assert limited.evidence_access['calls'] == 3
    # A fresh turn still cannot treat five two-member fragments as five records.
    fresh, token = mixed_session()
    fresh.seed([])
    with pytest.raises(ValueError, match='eight actual originals'):
        fresh.recall({'action': 'read_memory', 'references': [
            {'record_id': str(index), 'version': 1} for index in range(5)]}, cancelled=token)
    assert fresh.selection.evidence_access['calls'] == 1
    tools, signal = mixed_session()
    tools.seed([])
    for index in range(6):
        tools.recall(f'no evidence {index}', cancelled=signal)
    assert tools.selection.evidence_access['calls'] == 6
    assert not tools.selection.can_recall


def test_byte_budget_counts_actual_complete_originals_and_never_truncates():
    from virtual_companion.boundary.dialogue_retrieval import fragment_row, originals_size
    from virtual_companion.boundary.history_readset import validate_evidence_access

    session, cancel = mixed_session()
    session.seed([])
    fragment = make_fragment(0)
    size = originals_size([fragment_row(fragment)])
    session.max_bytes = size
    exact = session.recall({'action': 'read_memory', 'references': [
        {'record_id': '0', 'version': 1}]}, cancelled=cancel)
    assert exact.retrieved_dialogue == (fragment,)
    validate_evidence_access(exact.evidence_access, facts=[], retrieved_dialogue=[fragment])
    under, token = mixed_session()
    under.seed([])
    under.max_bytes = size - 1
    limited = under.recall({'action': 'read_memory', 'references': [
        {'record_id': '0', 'version': 1}]}, cancelled=token)
    assert limited.retrieved_dialogue == ()
    assert limited.evidence_access['stop_reason'] == 'content_limit'


def test_forged_fragment_is_rejected_at_admission_and_approval_from_exact_originals(tmp_path):
    from test_rebuilt_turns import A
    from test_received_context import receive_then_continue
    from test_segment_store import reviewed
    from virtual_companion.boundary.contracts import canonical
    from virtual_companion.boundary.dialogue_retrieval import DialogueCatalogStore
    from virtual_companion.boundary.history_readset import evidence_hash, history_readset
    from virtual_companion.text.deep_memory import DeepMemorySession
    from virtual_companion.text.memory_navigation import MemoryCatalog

    store, first, event, run, admission, _ = receive_then_continue(tmp_path)
    store.confirm_segment(A, first.input_id, 1, fingerprint=event['fingerprint'],
                          host_receipt='shown', channel='text_visible')
    catalog = MemoryCatalog(DialogueCatalogStore(store, run.input_id), A,
        state_version=run.state_version, clock=lambda: 200, checkpoint=None, dialogue=True)
    signal = Event()
    session = DeepMemorySession(catalog=catalog, lookup=lambda *args, **kwargs: (),
                               checkpoint=lambda signal: None, cancelled=signal, dialogue=True)
    session.seed([])
    fragment = store.dialogue_fragments(A, run.input_id, now_us=200)[0]
    session.recall({'action': 'read_memory', 'references': [
        {'record_id': fragment['fragment_id'], 'version': 1}]}, cancelled=signal)
    selected = session.selection
    request = admission(run, store.snapshot(A), '接着刚才说的继续',
        evidence_candidates=[], evidence_access=selected.evidence_access,
        retrieved_dialogue=selected.retrieved_dialogue)
    data = json.loads(request.context.payload)
    forged = deepcopy(data['retrieved_dialogue'][0])
    forged['members'][0]['text'] = '伪造的用户原话。'
    forged['members'][0]['sha256'] = evidence_hash(forged['members'][0]['text'])
    del forged['fragment_id']
    forged['fragment_id'] = 'dialogue-fragment-' + evidence_hash(forged)
    data['retrieved_dialogue'] = [forged]
    fake_context = replace(request.context, payload=canonical(data))
    with pytest.raises(VersionConflict, match='dialogue original'):
        admission.validate(fake_context)
    readset = history_readset(A, run.input_id, run.state_version, fake_context.payload)
    with pytest.raises(VersionConflict):
        store.approve_segment(run, reviewed(run), now_us=200, readset=readset)
    original = history_readset(A, run.input_id, run.state_version, request.context.payload)
    assert store.approve_segment(run, reviewed(run), now_us=200, readset=original)


@pytest.mark.parametrize('terminal', [False, True])
def test_visible_prefix_never_claims_complete_reply_and_late_members_use_frozen_subset(
    tmp_path, terminal,
):
    from test_rebuilt_turns import A, claim, config
    from test_received_delivery import setup_received
    from test_segment_store import reviewed
    from virtual_companion.boundary.dialogue_retrieval import DialogueCatalogStore
    from virtual_companion.boundary.history_readset import history_readset
    from virtual_companion.cognition.short_context import ShortContextAdmission
    from virtual_companion.cognition.short_dialogue import ShortRequest, prose_evidence
    from virtual_companion.text.deep_memory import DeepMemorySession
    from virtual_companion.text.memory_navigation import MemoryCatalog

    store, first, event = setup_received(tmp_path)
    store.confirm_segment(A, first.input_id, 1, fingerprint=event['fingerprint'],
                          host_receipt='first-visible', channel='text_visible')
    store.approve_segment(first, reviewed(first, ('我在这里。', '仍在继续。')), now_us=100)
    second = store.dispatch_segment(first, 2, now_us=100)
    store.receive_segment(A, first.input_id, 2, fingerprint=second['fingerprint'],
                          host_receipt='second-received-only')
    if terminal:
        store.finish_stream(first, outcome='received', now_us=100)
    next_input = store.accept(config(), 'next-prefix', '继续', session_id=first.session_id)
    fragment = store.dialogue_fragments(A, next_input.input_id, now_us=200)[0]
    assert [member['text'] for member in fragment['members']] == ['你好', '我在这里。']
    assert 'complete_response' not in fragment
    assert fragment['response_completeness'] == 'unknown'
    assert fragment['membership_policy'] == 'frozen_confirmed_subset'
    if not terminal:
        return
    run = claim(store, next_input, now=200)
    catalog = MemoryCatalog(DialogueCatalogStore(store, run.input_id), A,
        state_version=run.state_version, clock=lambda: 200, checkpoint=None, dialogue=True)
    signal = Event()
    session = DeepMemorySession(catalog=catalog, lookup=lambda *args, **kwargs: (),
                               checkpoint=lambda signal: None, cancelled=signal, dialogue=True)
    session.seed([])
    session.recall({'action': 'read_memory', 'references': [
        {'record_id': fragment['fragment_id'], 'version': 1}]}, cancelled=signal)
    admission = ShortContextAdmission(store, lambda *args: ShortRequest(args[-1]),
                                      evidence_clock=lambda: 200)
    request = admission(run, store.snapshot(A), '继续', evidence_candidates=[],
        evidence_access=session.selection.evidence_access,
        retrieved_dialogue=session.selection.retrieved_dialogue)
    assert '整体回复完整性未知' in prose_evidence(request.context.payload)
    assert json.loads(request.context.payload)['received_output'][0]['text'] == '仍在继续。'
    # Late visibility adds a new reconstruction, never rewrites this admitted subset.
    store.confirm_segment(A, first.input_id, 2, fingerprint=second['fingerprint'],
                          host_receipt='second-late-visible', channel='text_visible')
    fresh = store.dialogue_fragments(A, run.input_id, now_us=200)[0]
    assert len(fresh['members']) == 3 and fresh['fragment_id'] != fragment['fragment_id']
    admission.validate(request.context)
    readset = history_readset(A, run.input_id, run.state_version, request.context.payload)
    assert readset['retrieved_dialogue'][0]['membership_policy'] == 'frozen_confirmed_subset'
    assert store.approve_segment(run, reviewed(run), now_us=200, readset=readset)
