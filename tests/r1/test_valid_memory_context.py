from dataclasses import replace
from hashlib import sha256

import pytest
from test_rebuilt_memory import WHEN, A, B, apply, mutation
from virtual_companion.boundary.memory import FactKey
from virtual_companion.boundary.sqlite_background_worker import SQLiteBackgroundWorkerStore
from virtual_companion.boundary.sqlite_memory import SQLiteMemoryStore
from virtual_companion.cognition.valid_context import (
    DerivedContext,
    MemoryRef,
    select_valid_context,
)
from virtual_companion.text.chronology import utc_microseconds
from virtual_companion.text.context_selection import Turn


def setup(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "context.sqlite3")
    store.register(A)
    store.register(B)
    store.claim_owner(A, "worker", expected_epoch=0)
    apply(
        store,
        mutation(
            kind="assert_fact",
            record="drink",
            content="喜欢茶",
            fact_key=FactKey("user", "drink", "profile"),
        ),
    )
    return store


def select(store, **changes):
    args = dict(
        companion=A,
        history=(Turn("turn", "昨天的聊天", "嗯"),),
        current="今天喝什么🙂",
        refs=(MemoryRef(A, "drink", 1),),
        summaries=(),
        now_us=utc_microseconds(WHEN),
        budget=1000,
        count=lambda messages: sum(len(m["content"]) for m in messages),
    )
    args.update(changes)
    return select_valid_context(store, **args)


def test_recall_is_loaded_from_current_store_and_source_versions_are_exposed(tmp_path):
    store = setup(tmp_path)
    result = select(store)
    assert result.memory_refs == (MemoryRef(A, "drink", 1),)
    assert result.messages[-1] == {"role": "user", "content": "今天喝什么🙂"}
    assert "喜欢茶" in str(result.messages)
    assert result.tokens <= result.budget
    assert result.state_version == store.snapshot(A).write_version


def test_correction_rejects_old_summary_and_old_index_after_reopen(tmp_path):
    store = setup(tmp_path)
    summary = DerivedContext(
        "summary",
        "旧结论：喜欢茶",
        (MemoryRef(A, "drink", 1),),
        (("drink", sha256("喜欢茶".encode()).hexdigest()),),
    )
    assert "summary" in select(store, summaries=(summary,)).summary_ids
    apply(
        store,
        mutation(
            "correct", kind="correct", record="drink", content="喜欢咖啡", expected_record_version=1
        ),
    )
    reopened = SQLiteMemoryStore(store.path)
    result = select(
        reopened, summaries=(summary,), refs=(MemoryRef(A, "drink", 1), MemoryRef(A, "drink", 2))
    )
    assert result.summary_ids == ()
    assert result.rejected_summary_ids == ("summary",)
    assert result.memory_refs == (MemoryRef(A, "drink", 2),)
    assert "喜欢茶" not in str(result.messages)
    assert "喜欢咖啡" in str(result.messages)


def test_foreign_scope_cannot_use_same_named_record(tmp_path):
    store = setup(tmp_path)
    with pytest.raises(ValueError, match="scope"):
        select(store, refs=(MemoryRef(B, "drink", 1),))


def test_current_input_never_truncated_and_counter_errors_propagate(tmp_path):
    store = setup(tmp_path)
    with pytest.raises(ValueError, match="exceeds"):
        select(store, current="长" * 100, budget=10)

    def fail(_):
        raise OSError("tokenizer unavailable")

    with pytest.raises(OSError):
        select(store, count=fail)
    assert select(store, budget=15).messages[-1]["content"] == "今天喝什么🙂"


def test_mutation_during_counting_invalidates_selection(tmp_path):
    store = setup(tmp_path)
    changed = []

    def count(messages):
        if not changed:
            changed.append(True)
            apply(
                store,
                mutation(
                    "revoke", kind="revoke", record="drink", content="", expected_record_version=1
                ),
            )
        return len(str(messages))

    with pytest.raises(ValueError, match="changed"):
        select(store, count=count)


def test_new_user_source_during_counting_invalidates_old_memory_selection(tmp_path):
    from test_rebuilt_turns import config

    store = SQLiteBackgroundWorkerStore(tmp_path / "barrier.sqlite3")
    store.register(A)
    store.claim_owner(A, "worker", expected_epoch=0)
    apply(store, mutation(kind="assert_fact", record="drink", content="喜欢茶",
                          fact_key=FactKey("user", "drink", "profile")))
    changed = []

    def count(messages):
        if not changed:
            changed.append(True)
            store.accept(config(), "correction", "更正，我不喜欢茶")
        return len(str(messages))

    with pytest.raises(ValueError, match="changed"):
        select(store, count=count)
    result = select(store)
    assert result.memory_refs == ()
    assert "喜欢茶" not in str(result.messages)


def test_derived_context_requires_matching_content_and_every_reference(tmp_path):
    store = setup(tmp_path)
    summary = DerivedContext(
        "summary", "录制摘要", (MemoryRef(A, "drink", 1),), (("drink", "0" * 64),)
    )
    assert select(store, summaries=(summary,)).rejected_summary_ids == ("summary",)
    summary = replace(summary, refs=())
    assert select(store, summaries=(summary,)).rejected_summary_ids == ("summary",)
