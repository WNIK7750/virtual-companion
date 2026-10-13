"""Invalid evidence cannot re-enter model context through a pending exchange."""

import json
from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace

import pytest
from test_effective_history import NOW, complete, seeded
from test_invalidated_background_source import invalidate
from test_pending_exchanges import apply
from test_rebuilt_turns import A, config
from virtual_companion.boundary.contracts import Scope, VersionConflict
from virtual_companion.boundary.exchanges import ExchangeCommand, ExchangeEvidence
from virtual_companion.boundary.history_readset import history_readset
from virtual_companion.cognition.background_worker import UserFactWorker
from virtual_companion.text.memory_runtime import _fake_memory_result


def open_exchange(store, item, key):
    text = store.inspect(A, item.input_id)["text"]
    command = ExchangeCommand(Scope(A, item.session_id), key, key, "open",
        "assistant_response", text,
        ExchangeEvidence("user", item.input_id, text, len(text), sha256(text.encode()).hexdigest()),
        NOW, None, None)
    apply(store, command, now_us=NOW)
    return command


@pytest.mark.parametrize("change", ["correct", "forget", "expiry"])
def test_invalid_source_is_filtered_before_limit_rank_and_worker_target_hash(tmp_path, change):
    store, original, _, seed = seeded(tmp_path, expires=change == "expiry")
    old = open_exchange(store, original, "a-old")
    valid = store.accept(config(), "safe", "请聊爵士音乐。", session_id=original.session_id)
    complete(store, valid)
    store.cancel(A, valid.input_id)
    fresh = open_exchange(store, valid, "z-valid")
    original_history = store.exchange_history(A, old.exchange_id)
    frozen = history_readset(A, "frozen", 0, json.dumps({
        "history": [], "memories": [], "memory_status": "current",
        "exchanges": store.pending_exchanges(A, now_us=NOW),
    }))
    invalidate(store, seed, change)
    now = NOW + 2_000_000
    assert [row["exchange_id"] for row in store.pending_exchanges(A, now_us=now, limit=1)
            ] == [fresh.exchange_id]
    assert [row["exchange_id"] for row in store.ranked_pending_exchanges(
        A, now_us=now, score=lambda value: 100 if value["exchange_id"] == old.exchange_id else 0,
        limit=1)] == [fresh.exchange_id]
    assert store.pending_exchanges(A, now_us=now, candidates=((old.exchange_id, 1),)) == []
    assert store.exchange_history(A, old.exchange_id) == original_history
    assert original_history[-1]["status"] == "pending"
    # A later synthetic supported head citing the same invalid source stays
    # auditable but must not revive it in either read projection.
    copied = replace(old, command_id="copy-old", exchange_id="copy-old")
    apply(store, copied, now_us=now)
    assert {row["exchange_id"] for row in store.pending_exchanges(A, now_us=now)} == {"z-valid"}
    with store.transaction(write=False) as db:
        claim = SimpleNamespace(companion_id=A, input_id="frozen", state_version=0)
        with pytest.raises(VersionConflict, match="readset"):
            store._validate_segment_readset(db, claim, frozen, now_us=now)
    current = store.accept(config(), "current", "今天聊什么？", session_id=original.session_id)
    seen = []

    def model(stage, messages, schema, cancelled):
        data = json.loads(messages[-1]["content"])
        if stage.startswith("exchange_"):
            seen.extend(row["exchange_id"] for row in data.get("targets", []))
        return _fake_memory_result(stage, messages, schema, cancelled)

    worker = UserFactWorker(store, model, clock=lambda: now, resolution_protocol=True,
                            resolution_strategy="intent_first")
    assert worker.run_one(A, "worker", source=current.input_id)
    assert set(seen) == {"z-valid"}
    assert not store.memory_barrier(A)
    assert store.exchange_history(A, old.exchange_id) == original_history


def test_temporary_pending_user_barrier_does_not_erase_worker_exchange_catalog(tmp_path):
    store, original, _, _ = seeded(tmp_path)
    old = open_exchange(store, original, "valid-request")
    store.accept(config(), "pending", "新输入尚未处理。", session_id=original.session_id)
    assert store.memory_barrier(A)
    assert [row["exchange_id"] for row in store.pending_exchanges(A, now_us=NOW)
            ] == [old.exchange_id]
