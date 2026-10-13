"""Paraphrased old dialogue through the actual text host and authority readset."""

import json
from threading import Event

import pytest
from test_default_text_delivery import Model, setup
from test_dialogue_memory_integration import close, send_visible
from test_public_search_consumer import RecordingModel
from virtual_companion.boundary.dialogue_retrieval import fragment_row, source_count
from virtual_companion.cognition.hybrid_retrieval import HybridMemoryRetriever
from virtual_companion.cognition.semantic_retrieval import SemanticMemoryRetriever
from virtual_companion.text.application import TextSession
from virtual_companion.text.default_runtime import DefaultTextRuntime

QUERY = '避开人潮的安排'
TARGET = '周六黎明沿江骑车'


class Encoder:
    def __init__(self):
        self.documents = []

    def query_embed(self, query):
        return iter(([1.0, 0.0],))

    def embed(self, texts):
        self.documents.extend(texts)
        return iter([1.0, 0.0] if TARGET in text else [0.0, 1.0] for text in texts)


def seed(tmp_path):
    old, text, memory, mirror, _ = setup(tmp_path, Model())
    target = None
    for number, value in enumerate((TARGET, '避开人潮预约展馆',
                                    '避开人潮改走小路', '避开人潮错峰游园')):
        event = send_visible(old, text, value, f'past-visible-{number}')
        if target is None:
            target = event['basis']['input_id']
    old.stop()
    text.stop()
    for number in range(34):
        filler = memory.store.accept(old.config, f'filler-{number}', f'无关间隔输入{number}',
                                     session_id=event['basis']['session_id'])
        memory.store.cancel(memory.companion, filler.input_id)
    return old, text, memory, mirror, target


def retriever(encoder):
    return HybridMemoryRetriever(SemanticMemoryRetriever(
        encoder, model_id='synthetic', model_revision='1', synthetic_model_identity=True,
        candidate_overfetch=8, page_size=2))


def test_paraphrased_dialogue_enters_generation_review_ack_and_reopened_readset(tmp_path):
    old, old_text, memory, mirror, target = seed(tmp_path)
    encoder = Encoder()
    model = RecordingModel([
        {'action': 'recall', 'query': QUERY},
        {'action': 'answer', 'text': '可以接着聊这个安排。'},
    ])
    runtime = DefaultTextRuntime(memory, mirror, model, binding={'test': 'dialogue-hybrid'},
        semantic_retriever=retriever(encoder), readonly_recall='dialogue-memory.v1')
    text = TextSession(model, mirror, {}, prepare_turn=runtime.prepare_turn,
                       before_start=runtime.before_start)
    try:
        event = send_visible(runtime, text, QUERY, 'current-visible')
        generations = [call for call in model.calls if call['stage'] != 'stream_unit_review']
        assert TARGET not in repr(generations[0]['messages'])
        fragments = runtime._evidence_session.selection.retrieved_dialogue
        selected = next((row for row in fragments if row['input_id'] == target), None)
        assert selected is not None, 'Semantic dialogue target was omitted after lexical miss'
        assert selected['members'][0]['text'] == TARGET
        assert any(TARGET in document for document in encoder.documents)
        assert TARGET in repr(generations[-1]['messages'])
        assert runtime._evidence_session.selection.evidence_access['calls'] == 1
        selection = runtime._evidence_session.selection
        assert source_count([*selection.memories,
                             *(fragment_row(row) for row in selection.retrieved_dialogue)]) <= 8
        review = next(call for call in model.calls if call['stage'] == 'stream_unit_review')
        packet = json.loads(review['messages'][-1]['content'])
        context = packet['context']
        if isinstance(context, str):
            context = json.loads(context)
        assert any(row['input_id'] == target for row in context['retrieved_dialogue'])
        stored = memory.store.segments(memory.companion, runtime.claim.input_id)[0]
        assert stored['readset']['protocol'] == 'history-readset.v7'
        assert any(row['input_id'] == target for row in stored['readset']['retrieved_dialogue'])
        reopened = type(memory.store)(memory.path)
        persisted = reopened.segments(memory.companion, runtime.claim.input_id)[0]
        assert persisted['readset'] == stored['readset']
        assert event['text'] == '可以接着聊这个安排。'
    finally:
        close(runtime, text, memory, mirror)


@pytest.mark.parametrize('hybrid', [False, True])
def test_dialogue_semantic_selection_preserves_explicit_selector_and_exact_constraints(
        tmp_path, hybrid):
    old, text, memory, mirror, target = seed(tmp_path)
    encoder = Encoder()
    selected = retriever(encoder)
    runtime = DefaultTextRuntime(memory, mirror, Model(), binding={'test': 'selectors'},
        semantic_retriever=selected if hybrid else selected.semantic,
        readonly_recall='dialogue-memory.v1')
    cancelled = Event()
    try:
        attempt = mirror.begin(QUERY, {})
        runtime.prepare_turn(attempt, mirror.session_id, mirror.history()['messages'], cancelled)
        result = runtime._lookup_evidence(QUERY, cancelled=cancelled)
        assert any(row.get('fragment', {}).get('input_id') == target for row in result) is hybrid
        if hybrid:
            exact = runtime._lookup_evidence(
                {'action': 'recall', 'query': QUERY, 'exact_terms': [TARGET]},
                cancelled=cancelled)
            assert len(exact) == 1 and exact[0]['fragment']['input_id'] == target
            facts_only = runtime._lookup_evidence(
                {'action': 'recall', 'query': QUERY, 'conditions': {'subject': 'user'}},
                cancelled=cancelled)
            assert all(row['kind'] != 'dialogue_fragment' for row in facts_only)
    finally:
        close(runtime, text, memory, mirror)
