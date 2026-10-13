"""Current-topic selection must not be a permanent exchange-ID prefix."""

import json

import pytest
from test_effective_history import NOW, complete
from test_rebuilt_turns import A, claim, config
from virtual_companion.boundary.sqlite_memory_management import SQLiteExplicitMemoryStore
from virtual_companion.cognition.background_worker import UserFactWorker
from virtual_companion.cognition.short_context import ShortContextAdmission
from virtual_companion.cognition.short_dialogue import ShortRequest


def pending_catalog(tmp_path, count, *, expires=False):
    store = SQLiteExplicitMemoryStore(tmp_path / "pending.sqlite3")

    def recorded(stage, messages, schema, cancelled):
        if stage == "fact_extract":
            return {"operations": []}
        if stage == "exchange_support":
            return {"intent": "question", "verdict": "supported"}
        data = json.loads(messages[-1]["content"])
        source = data["sources"][-1]
        if stage == "exchange_intent":
            return {"coverage": "complete", "acts": [{
                "segmentId": data["segments"][-1]["id"],
                "evidenceQuote": source["text"], "intent": "question", "targetId": None}]}
        assert stage == "exchange_extract"
        return {"operations": [{"operation": "add", "type": "response_obligation",
                 "evidenceQuote": source["text"],
                "supersedesAtomId": None,
                "expiresAt": (NOW + 1_000_000) / 1000 if expires else None}]}

    worker = UserFactWorker(store, recorded, clock=lambda: NOW, resolution_protocol=True,
                            resolution_strategy="intent_first")
    for index in range(count):
        item = store.accept(config(), f"question-{index}", f"topic_{index} 怎么处理？")
        worker.run_one(A, "setup", source=item.input_id)
        store.cancel(A, item.input_id)
    return store


@pytest.mark.parametrize("count", [9, 99])
def test_current_question_retrieves_target_beyond_id_prefix(tmp_path, count):
    store = pending_catalog(tmp_path, count)
    with store.transaction(write=False) as db:
        rows = [json.loads(row[0]) for row in db.execute(
            "SELECT data FROM exchange_versions WHERE companion=? ORDER BY exchange", (A,))]
    target = rows[-1]
    assert target not in store.pending_exchanges(A, now_us=NOW, limit=8)
    current = store.accept(config(), "current", target["content"])
    complete(store, current)
    run = claim(store, current)
    admission = ShortContextAdmission(store, lambda *args: ShortRequest(args[-1]),
                                     evidence_clock=lambda: NOW)
    request = admission(run, store.snapshot(A), target["content"])
    selected = json.loads(request.context.payload)["exchanges"]
    assert target in selected
    assert selected[0] == target
    assert len(selected) == 8
    admission.validate(request.context)
    # Ranking is a read-only selection: it neither resolves nor discards a duty.
    assert store.exchange_history(A, target["exchange_id"]) == [target]


def test_pending_ranking_keeps_unmatched_cues_and_excludes_other_companion(tmp_path):
    from test_rebuilt_turns import B
    from virtual_companion.boundary.memory_relevance import lexical_relevance

    store = pending_catalog(tmp_path, 9)
    selected = store.ranked_pending_exchanges(A, now_us=NOW, score=lexical_relevance("继续"))
    assert len(selected) == 8  # No lexical match is not proof of irrelevance.
    assert store.ranked_pending_exchanges(B, now_us=NOW, score=lexical_relevance("topic_8")) == []
    assert not store.memory_barrier(A)


def test_expired_pending_never_wins_ranking_or_becomes_live_again(tmp_path):
    from virtual_companion.boundary.memory_relevance import lexical_relevance

    store = pending_catalog(tmp_path, 9, expires=True)
    assert store.ranked_pending_exchanges(A, now_us=NOW, score=lexical_relevance("topic_8"))
    assert store.ranked_pending_exchanges(
        A, now_us=NOW + 1_000_001, score=lexical_relevance("topic_8"),
    ) == []
