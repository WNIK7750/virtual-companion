import json
from dataclasses import replace
from hashlib import sha256

import pytest
from test_rebuilt_memory import WHEN
from test_rebuilt_presentation import approved
from test_rebuilt_turns import A, claim, config
from test_rebuilt_turns import result as bare_result
from virtual_companion.boundary.contracts import Scope, VersionConflict, canonical
from virtual_companion.boundary.memory import FactKey, MemoryEffect, MemoryMutation
from virtual_companion.boundary.memory_support import command_digest
from virtual_companion.boundary.presentation import Presentation
from virtual_companion.boundary.sqlite_business import SQLiteBusinessStore


def result(run):
    display = Presentation(
        Scope(A, run.session_id),
        run.input_id,
        run.attempt_id,
        "candidate",
        1,
        "",
        (),
    )
    return replace(bare_result(run, ""), presentation=approved(display))


def prepared(tmp_path):
    store = SQLiteBusinessStore(tmp_path / "business.sqlite3", evidence_clock=lambda: 100)
    item = store.accept(config(), "first", "我喜欢茶")
    run = claim(store, item)
    command = MemoryMutation(
        Scope(A, item.session_id),
        "fact-op",
        "assert_fact",
        "fact",
        "喜欢茶",
        "input",
        item.input_id,
        "unvalidated",
        WHEN,
        fact_key=FactKey("user", "drink", "profile"),
    )
    proof = {
        "verdict": "supported",
        "evaluator": "synthetic-operation-support.v1",
        "source_sha256": sha256("我喜欢茶".encode()).hexdigest(),
        "content_sha256": sha256(command.content.encode()).hexdigest(),
        "command_sha256": command_digest(command, "我喜欢茶"),
        "target_sha256": None,
    }
    command = replace(command, reason=canonical({"quote": "我喜欢茶", "support": proof}))
    return store, run, command


def test_supported_current_fact_commits_with_reply_and_survives_reopen(tmp_path):
    store, run, command = prepared(tmp_path)
    store.commit(replace(result(run), memories=(MemoryEffect(command),)), now_us=101)
    assert SQLiteBusinessStore(store.path).recall(A, now_us=100)[0]["content"] == "喜欢茶"


@pytest.mark.parametrize("change", ["content", "source", "target", "quote"])
def test_changed_support_binding_cannot_commit_reply_or_memory(tmp_path, change):
    store, run, command = prepared(tmp_path)
    if change == "content":
        command = replace(command, content="喜欢咖啡")
    elif change == "source":
        command = replace(command, source_id="invented")
    else:
        reason = json.loads(command.reason)
        if change == "quote":
            reason["quote"] = "我喜欢咖啡"
        else:
            reason["support"]["target_sha256"] = "0" * 64
        command = replace(command, reason=canonical(reason))
    with pytest.raises(ValueError):
        store.commit(replace(result(run), memories=(MemoryEffect(command),)), now_us=101)
    assert store.inspect(A, run.input_id)["commit"] is None
    assert store.recall(A, now_us=100) == []


def test_fact_slot_conflict_rolls_back_entire_reply_and_batch(tmp_path):
    store, run, command = prepared(tmp_path)
    second = replace(command, mutation_id="second-op", record_id="second")
    reason = json.loads(second.reason)
    reason["support"]["command_sha256"] = command_digest(second, reason["quote"])
    second = replace(second, reason=canonical(reason))
    with pytest.raises(VersionConflict):
        store.commit(
            replace(result(run), memories=(MemoryEffect(command), MemoryEffect(second))),
            now_us=101,
        )
    assert store.inspect(A, run.input_id)["commit"] is None
    assert store.recall(A, now_us=100) == []
    assert store.memory_receipt(A, command.mutation_id) is None


def test_fact_target_capacity_never_silently_omits_an_active_fact(tmp_path):
    store, _, command = prepared(tmp_path)
    for index in range(101):
        value = replace(
            command,
            mutation_id=f"seed-{index}",
            record_id=f"fact-{index}",
            fact_key=FactKey("user", f"property-{index}", "profile"),
            reason="synthetic seed",
        )
        store.mutate(value, store.owner(A), expected_version=store.snapshot(A).write_version)
    with pytest.raises(ValueError, match="capacity"):
        store.fact_targets(A, now_us=100)
