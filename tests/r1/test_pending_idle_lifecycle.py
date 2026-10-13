"""Actual ACK queue lifecycle; recorded semantics do not assert model quality."""

import json
import threading
import time

import pytest
from test_effective_history import NOW, complete, emit, seeded
from test_invalidated_background_source import invalidate
from test_rebuilt_turns import A, config
from virtual_companion.boundary.sqlite_memory_management import SQLiteExplicitMemoryStore
from virtual_companion.cognition.background_worker import BackgroundScheduler, UserFactWorker
from virtual_companion.text.memory_runtime import TextMemoryRuntime, _fake_memory_result


def _idle_lifecycle(tmp_path, *, invalidate_answer=False, expires=False, fact_expires=False):
    store, original, _, seed = seeded(tmp_path, expires=fact_expires)
    question = "我的饮食偏好是什么？"
    answer = "你偏好偏辣的口味。"
    item = store.accept(config(), "question", question, session_id=original.session_id)
    calls = []

    def recorded(stage, messages, schema, cancelled):
        data = json.loads(messages[-1]["content"])
        calls.append(stage)
        if stage not in {"exchange_intent", "exchange_extract", "exchange_support"}:
            return _fake_memory_result(stage, messages, schema, cancelled)
        if stage == "exchange_support":
            return {"intent": "question" if not store.pending_exchanges(A, now_us=NOW)
                    else "answer", "verdict": "supported"}
        source = data["sources"][-1]
        is_question = source["text"] == question
        is_answer = source["text"] == answer
        if not is_question and not is_answer:
            return _fake_memory_result(stage, messages, schema, cancelled)
        target = (store.pending_exchanges(A, now_us=NOW)[0]["exchange_id"]
                  if is_answer else None)
        if stage == "exchange_intent":
            segment = data["segments"][-1]
            return {"coverage": "complete", "acts": [{"segmentId": segment["id"],
                     "evidenceQuote": source["text"],
                    "intent": "question" if is_question else "answer", "targetId": target}]}
        return {"operations": [{"operation": "add" if is_question else "resolve",
                "type": "response_obligation" if is_question else None,
                 "evidenceQuote": source["text"],
                "supersedesAtomId": target,
                "expiresAt": (NOW + 1_000_000) // 1000 if expires and is_question else None}]}

    UserFactWorker(store, recorded, clock=lambda: NOW, resolution_protocol=True,
                   resolution_strategy="intent_first").run_one(A, "setup", source=item.input_id)
    assert len(store.pending_exchanges(A, now_us=NOW)) == 1
    # Use a real fact-dependent admission/readset and durable actual ACK.
    emit(store, item, answer)
    answer_source = next(row["source"] for row in store.background_sources(A)
                         if row["role"] == "assistant"
                         and store.background_source_text(A, row["job"]) == answer)
    if invalidate_answer:
        invalidate(store, seed, "forget")
    newest = store.accept(config(), "new-topic", "我喜欢爵士音乐。",
                          session_id=original.session_id)
    complete(store, newest)
    store.cancel(A, newest.input_id)
    before = store.pending_exchanges(A, now_us=NOW)
    assert len(before) == 1  # New topic and ACK alone are not semantic completion.
    busy = threading.Event()
    busy.set()
    runtime = TextMemoryRuntime(store.path, A, background_idle_grace=0.02)
    store = runtime.store
    runtime.scheduler = BackgroundScheduler(runtime.store, user_invoke=recorded,
        assistant_invoke=recorded, clock=lambda: NOW + 20,
        resolution_protocol=True, resolution_strategy="intent_first")
    runtime.attach_foreground(busy.is_set)
    try:
        time.sleep(0.04)
        assert store.pending_exchanges(A, now_us=NOW) == before
        busy.clear()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if all(row["status"] == "invalidated" for row in store.background_sources(A)):
                break
            time.sleep(0.02)
        pending = store.pending_exchanges(A, now_us=NOW)
        assert pending == (before if invalidate_answer else [])
        assert all(row["status"] == "invalidated" for row in store.background_sources(A))
        assert runtime.last_error == ""
        assert not store.memory_barrier(A)
        assert (answer_source in store.ineffective_history_sources(A, now_us=NOW + 20)
                ) == invalidate_answer
    finally:
        runtime.close()
    return store, before[0], answer_source, seed


@pytest.mark.parametrize("invalidate_answer", [False, True])
def test_idle_worker_distinguishes_answer_semantics_from_invalidated_ack(tmp_path,
                                                                     invalidate_answer):
    _idle_lifecycle(tmp_path, invalidate_answer=invalidate_answer)


def test_own_resolution_is_history_only_and_cache_is_source_specific(tmp_path):
    from types import SimpleNamespace

    from virtual_companion.boundary.contracts import VersionConflict

    store, _, source, _ = _idle_lifecycle(tmp_path)
    with store.transaction(write=False) as db:
        manifest = json.loads(db.execute(
            "SELECT s.manifest FROM stream_segments s JOIN stream_history h ON "
            "s.companion=h.companion AND s.input=h.input AND s.sequence=h.sequence "
            "WHERE h.companion=? AND h.source=?", (A, source),
        ).fetchone()[0])
        readset = manifest["readset"]
        for history_first in (False, True):
            cache = {}
            for historical in (history_first, not history_first):
                assert store._readset_current(db, readset, now_us=NOW + 20, cache=cache,
                    history_source=source if historical else None) == historical
            assert not store._readset_current(db, readset, now_us=NOW + 20, cache=cache,
                                               history_source="different-ack")
        claim = SimpleNamespace(companion_id=A, input_id=readset["input"],
                                state_version=readset["state_version"])
        with pytest.raises(VersionConflict, match="Segment readset changed"):
            store._validate_segment_readset(db, claim, readset, now_us=NOW + 20)


@pytest.mark.parametrize("change", ["source", "session", "companion", "revision", "original",
                                    "support", "prefix", "partial", "defer", "close"])
def test_history_exception_rejects_altered_or_foreign_resolution(tmp_path, change):
    from virtual_companion.boundary.history_readset import evidence_hash

    store, original, source, _ = _idle_lifecycle(tmp_path)
    with store.transaction() as db:
        head = store._exchange_head(db, A, original["exchange_id"])
        if change == "original":
            old = dict(original, content="tampered original target")
            db.execute("UPDATE exchange_versions SET data=? WHERE companion=? AND exchange=? "
                       "AND revision=1", (json.dumps(old), A, original["exchange_id"]))
        else:
            if change == "source":
                head["evidence"]["source_id"] = "other-ack"
            elif change == "session":
                head["scope"]["session_id"] = "other-session"
            elif change == "companion":
                head["scope"]["companion_id"] = "00000000-0000-0000-0000-000000000002"
            elif change == "revision":
                head["revision"] = 3
            elif change == "support":
                head["support"]["command_sha256"] = "0" * 64
            elif change == "prefix":
                head["evidence"]["prefix_sha256"] = "0" * 64
            elif change == "partial":
                head["evidence"]["confirmed_end"] -= 1
            elif change in {"defer", "close"}:
                head["action"] = change
                head["status"] = "deferred" if change == "defer" else "closed"
            # Even a self-consistent command hash must not grant this exception.
            command = {key: head[key] for key in (
                "scope", "command_id", "exchange_id", "action", "direction", "content",
                "evidence", "occurred_us", "expected_revision", "expires_us")}
            fingerprint = evidence_hash(command)
            if change != "support":
                head["support"]["command_sha256"] = fingerprint
            db.execute("UPDATE exchange_versions SET data=?,fingerprint=? WHERE companion=? "
                       "AND exchange=? AND revision=2",
                       (json.dumps(head), fingerprint, A, original["exchange_id"]))
    assert source in store.ineffective_history_sources(A, now_us=NOW + 20)


@pytest.mark.parametrize("change", ["forget", "correct", "expiry", "fact_expiry"])
def test_own_resolution_does_not_hide_later_fact_or_exchange_invalidation(tmp_path, change):
    store, _, source, seed = _idle_lifecycle(tmp_path, expires=change == "expiry",
                                            fact_expires=change == "fact_expiry")
    if change in {"forget", "correct"}:
        invalidate(store, seed, change)
    assert source in store.ineffective_history_sources(A, now_us=NOW + 2_000_000)


def _pending_catalog(tmp_path):
    store = SQLiteExplicitMemoryStore(tmp_path / "prefix.sqlite3")

    def recorded(stage, messages, schema, cancelled):
        if stage == "fact_extract":
            return {"operations": []}
        if stage == "exchange_support":
            return {"intent": "question", "verdict": "supported"}
        data = json.loads(messages[-1]["content"])
        source = data["sources"][-1]
        if stage == "exchange_intent":
            return {"coverage": "complete", "acts": [{"segmentId": data["segments"][-1]["id"],
                     "evidenceQuote": source["text"],
                    "intent": "question", "targetId": None}]}
        assert stage == "exchange_extract"
        return {"operations": [{"operation": "add", "type": "response_obligation",
                 "evidenceQuote": source["text"],
                "supersedesAtomId": None, "expiresAt": None}]}

    worker = UserFactWorker(store, recorded, clock=lambda: NOW, resolution_protocol=True,
                            resolution_strategy="intent_first")
    for index in range(9):
        item = store.accept(config(), f"question-{index}", f"请介绍编号{index}的不同方案？")
        worker.run_one(A, "setup", source=item.input_id)
        store.cancel(A, item.input_id)
    all_pending = store.pending_exchanges(A, now_us=NOW)
    return store, all_pending


def test_pending_projection_is_fixed_id_prefix_not_recent_or_current_relevance(tmp_path):
    store, all_pending = _pending_catalog(tmp_path)
    assert len(all_pending) == 9
    projected = store.pending_exchanges(A, now_us=NOW, limit=8)
    assert projected == sorted(all_pending, key=lambda item: item["exchange_id"])[:8]
    assert store.pending_exchanges(A, now_us=NOW + 1000, limit=8) == projected
    assert not store.memory_barrier(A)


def test_live_exchange_readset_cache_shares_one_head_query_across_history_sources(tmp_path):
    from virtual_companion.boundary.history_readset import history_readset

    store, pending = _pending_catalog(tmp_path)
    readset = history_readset(A, "synthetic-input", 1, json.dumps({
        "memories": [], "history": [], "exchanges": pending[:1], "memory_status": "current",
    }))
    statements = []
    with store.transaction(write=False) as db:
        db.set_trace_callback(statements.append)
        cache = {}
        for source in (None, "ack-one", "ack-two", None):
            assert store._readset_current(db, readset, now_us=NOW, cache=cache,
                                          history_source=source)
        db.set_trace_callback(None)
    head_queries = [sql for sql in statements if "FROM exchange_versions" in sql]
    assert len(head_queries) == 1
    assert all(type(value) is bool for value in cache.values())
