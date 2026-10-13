"""Dialogue ranking reuses hybrid ports; tool capacity remains the host's decision."""

from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from threading import Event

import pytest
from test_dialogue_retrieval import make_fragment
from test_rebuilt_turns import A, claim, config
from test_segment_store import reviewed
from virtual_companion.boundary.contracts import Scope, VersionConflict
from virtual_companion.boundary.dialogue_retrieval import source_count
from virtual_companion.boundary.memory import FactKey, MemoryMutation
from virtual_companion.boundary.sqlite_received import SQLiteReceivedStore
from virtual_companion.cognition.dialogue_memory_retrieval import DialogueMemoryRetriever
from virtual_companion.cognition.hybrid_retrieval import HybridMemoryRetriever
from virtual_companion.cognition.semantic_retrieval import SemanticMemoryRetriever

QUERY = "避开人潮的安排"
TARGET = "周六黎明沿江骑车"


class Encoder:
    def __init__(self, after_encode=None):
        self.queries, self.documents = [], []
        self.after_encode = after_encode

    def query_embed(self, query):
        self.queries.append(query)
        return iter(([1.0, 0.0],))

    def embed(self, texts):
        self.documents.extend(texts)
        vectors = [[1.0, 0.0] if TARGET in text else [0.0, 1.0] for text in texts]
        if self.after_encode is not None:
            callback, self.after_encode = self.after_encode, None
            callback()
        return iter(vectors)


def retriever(encoder=None, *, page_size=2, candidate_overfetch=32):
    semantic = SemanticMemoryRetriever(
        encoder or Encoder(), model_id="synthetic", model_revision="1",
        synthetic_model_identity=True, page_size=page_size,
        candidate_overfetch=candidate_overfetch)
    return HybridMemoryRetriever(semantic)


class Store:
    def __init__(self, count=6):
        self.fragments = [make_fragment(index, text=TARGET if index == 0 else "无关展馆")
                          for index in range(count)]
        self.rebuilds, self.validations = 0, []

    def dialogue_fragments(self, companion, input_id, *, now_us):
        assert (companion, input_id, now_us) == (A, "current", 200)
        self.rebuilds += 1
        return deepcopy(self.fragments)

    def validate_dialogue_fragments(self, companion, input_id, fragments, *, now_us):
        assert (companion, input_id, now_us) == (A, "current", 200)
        self.validations.append(len(fragments))
        for fragment in fragments:
            if fragment not in self.fragments:
                raise VersionConflict("Synthetic original changed or was revoked")


def lookup(leaf, store, query=QUERY, **kwargs):
    return leaf.recall(store, A, input_id="current", now_us=200, query=query, **kwargs)


def test_paraphrase_is_recovered_with_existing_encoder_cache_and_one_paged_snapshot():
    encoder, store = Encoder(), Store(count=30)
    hybrid = retriever(encoder)
    leaf = DialogueMemoryRetriever(hybrid)
    rows = lookup(leaf, store)
    assert rows[0]["fragment"]["members"][0]["text"] == TARGET
    assert source_count(rows) == 60  # Candidate ranking is not an admitted tool response.
    assert store.rebuilds == 1 and max(store.validations) <= 24
    assert len(encoder.documents) == 30
    assert leaf.retriever is hybrid and leaf.retriever.semantic.embedder is encoder
    again = lookup(leaf, store)
    assert again == rows and len(encoder.documents) == 30


def test_literal_identifier_is_hard_and_fact_conditions_do_not_guess_dialogue_subjects():
    store = Store(count=2)
    store.fragments = [make_fragment(0, text="编号AB-12，"),
                       make_fragment(1, text=TARGET + "编号AB-120，")]
    encoder = Encoder()
    leaf = DialogueMemoryRetriever(retriever(encoder))
    rows = lookup(leaf, store, exact_terms=["AB-12"])
    assert len(rows) == 1 and "AB-120" not in rows[0]["content"]
    before = (store.rebuilds, len(encoder.queries))
    assert lookup(leaf, store, conditions={"subject": "user"}) == []
    assert (store.rebuilds, len(encoder.queries)) == before


def test_lexical_signal_is_fused_and_does_not_require_a_semantic_top_hit():
    store = Store(count=2)
    store.fragments = [make_fragment(0, text="避开人潮的安排：预约展馆"),
                       make_fragment(1, text=TARGET)]
    rows = lookup(DialogueMemoryRetriever(retriever()), store)
    assert {row["fragment"]["members"][0]["text"] for row in rows} == {
        "避开人潮的安排：预约展馆", TARGET}


@pytest.mark.parametrize("change", ["revoke", "original_bytes"])
def test_changed_original_is_rejected_after_encoding_before_return(change):
    store = Store()

    def change_source():
        if change == "revoke":
            store.fragments = []
        else:
            store.fragments[0]["members"][0]["text"] = "被改写的原文"

    leaf = DialogueMemoryRetriever(retriever(Encoder(change_source)))
    with pytest.raises(VersionConflict):
        lookup(leaf, store)


def test_cancel_after_encode_prevents_validation_or_evidence_return():
    store, cancelled = Store(), Event()
    leaf = DialogueMemoryRetriever(retriever(Encoder(cancelled.set)))

    def checkpoint():
        if cancelled.is_set():
            raise InterruptedError("Synthetic cancellation")

    with pytest.raises(InterruptedError):
        lookup(leaf, store, checkpoint=checkpoint)
    assert not store.validations


def test_actual_sqlite_correction_invalidates_cached_dialogue_candidates(tmp_path):
    store = SQLiteReceivedStore(tmp_path / "authority.sqlite3")
    item = store.accept(config(), "old", TARGET)
    old = claim(store, item, now=100)
    store.approve_segment(old, reviewed(old), now_us=100)
    event = store.dispatch_segment(old, 1, now_us=100)
    store.confirm_segment(A, old.input_id, 1, fingerprint=event["fingerprint"],
                          host_receipt="synthetic-visible", channel="text_visible")
    store.finish_stream(old, outcome="completed", now_us=100)
    current = store.accept(config(), "current", QUERY)
    run = claim(store, current, now=200)
    seed = MemoryMutation(
        Scope(A, old.session_id), "seed", "assert_fact", "plan", TARGET, "manual", old.input_id,
        "synthetic source", datetime(2026, 10, 12, tzinfo=UTC), FactKey("user", "plan", "profile"))
    store.mutate(seed, store.owner(A), expected_version=store.snapshot(A).write_version)

    def correct():
        store.mutate(replace(seed, mutation_id="correct", kind="correct", content="安排取消了",
                             source_id=run.input_id, expected_record_version=1, fact_key=None),
                     store.owner(A), expected_version=store.snapshot(A).write_version)

    leaf = DialogueMemoryRetriever(retriever(Encoder(correct)))
    with pytest.raises(VersionConflict, match="expired or changed"):
        leaf.recall(store, A, input_id=run.input_id, now_us=200, query=QUERY)
