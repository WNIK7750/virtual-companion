from datetime import UTC, datetime

from test_rebuilt_turns import A, config
from virtual_companion.boundary.contracts import Scope
from virtual_companion.boundary.memory import FactKey, MemoryMutation
from virtual_companion.boundary.memory_projection import clean_memory_text
from virtual_companion.boundary.sqlite_memory_management import SQLiteExplicitMemoryStore
from virtual_companion.text.memory_management import MemoryManagementService

WHEN = datetime(2026, 9, 21, tzinfo=UTC)


def add_fact(store, owner, record, content, *, source=None):
    return store.mutate(
        MemoryMutation(
            Scope(A, "seed-session"),
            f"seed-{record}",
            "assert_fact",
            record,
            content,
            "manual",
            source or f"source-{record}",
            "seed",
            WHEN,
            FactKey("user", f"predicate-{record}", "profile"),
        ),
        owner,
        expected_version=store.snapshot(A).write_version,
    )


def seeded(tmp_path):
    store = SQLiteExplicitMemoryStore(tmp_path / "memory.sqlite3")
    store.register(A)
    owner = store.claim_owner(A, "seed-owner", expected_epoch=0)
    add_fact(store, owner, "negative", "  我不喜欢茶\n但今天可能会喝一点  ")
    add_fact(
        store,
        owner,
        "wrapped",
        '{"content":"我今天可能不去北京","record_id":"internal",'
        '"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}',
    )
    add_fact(store, owner, "similar", "我明天可能不去北京")
    return store, owner


def test_display_cleaning_is_deterministic_and_does_not_erase_qualifiers(tmp_path):
    store, _ = seeded(tmp_path)
    service = MemoryManagementService(store, A, clock=lambda: 10)

    page = service.list_memories(offset=0, limit=50)

    assert page["total"] == 3
    assert page["next_offset"] is None
    assert len({item["record_id"] for item in page["items"]}) == 3
    by_id = {item["record_id"]: item for item in page["items"]}
    assert by_id["negative"]["content"] == "我不喜欢茶 但今天可能会喝一点"
    assert by_id["wrapped"]["content"] == "我今天可能不去北京"
    assert by_id["similar"]["content"] == "我明天可能不去北京"
    assert "{" not in by_id["wrapped"]["content"]
    assert "sha256" not in by_id["wrapped"]["content"]
    assert by_id["wrapped"]["content"] != by_id["similar"]["content"]
    assert store.history(A, "wrapped")[0]["content"].startswith('{"content"')


def test_unknown_structured_wrapper_never_uses_raw_json_as_list_body():
    value = '{"subject":"我","predicate":"不喜欢","value":"茶",'
    value += '"time":"今天","uncertainty":"可能"}'

    cleaned = clean_memory_text(value)

    assert all(part in cleaned for part in ("我", "不喜欢", "茶", "今天", "可能"))
    assert "{" not in cleaned and '"subject"' not in cleaned


def test_details_expand_source_and_version_history_without_hash_body(tmp_path):
    store, owner = seeded(tmp_path)
    store.mutate(
        MemoryMutation(
            Scope(A, "seed-session"),
            "correct-negative",
            "correct",
            "negative",
            "我现在不喝茶",
            "manual",
            "correction-source",
            "explicit correction",
            WHEN,
            expected_record_version=1,
        ),
        owner,
        expected_version=store.snapshot(A).write_version,
    )
    service = MemoryManagementService(store, A, clock=lambda: 10)

    details = service.memory_details("negative")

    assert details["item"]["content"] == "我现在不喝茶"
    assert [item["version"] for item in details["history"]] == [1, 2]
    assert [item["content"] for item in details["history"]] == [
        "我不喜欢茶 但今天可能会喝一点",
        "我现在不喝茶",
    ]
    assert all("sha256" not in item for item in details["history"])
    assert details["source"]["label"] == "手动管理"


def test_list_action_returns_feedback_only_after_real_commit_and_uses_no_model(tmp_path):
    store, _ = seeded(tmp_path)
    service = MemoryManagementService(store, A, clock=lambda: 10)

    result = service.apply_action(
        {
            "operation_id": "forget-negative",
            "action": "forget",
            "record_id": "negative",
            "expected_version": 1,
            "replacement": "",
        }
    )

    assert result["status"] == "committed"
    assert result["review_calls"] == 0
    assert "停止在后续回答中使用" in result["message"]
    assert "彻底删除" not in result["message"]
    assert {row["record_id"] for row in store.recall(A, now_us=11)} == {
        "wrapped",
        "similar",
    }
    assert store.history(A, "negative")[-1]["status"] == "revoked"


def test_memory_stats_separate_effective_records_from_versions_and_unknown_time(tmp_path):
    store, _ = seeded(tmp_path)
    service = MemoryManagementService(store, A, clock=lambda: 10)

    before = service.list_memories()
    assert before["stats"]["effective_count"] == 3
    assert before["stats"]["current_record_count"] == 3
    assert before["stats"]["history_version_count"] == 3
    assert before["stats"]["status_counts"]["active"] == 3
    assert before["stats"]["unknown_event_time_count"] == 3

    service.apply_action({
        "operation_id": "stats-correct",
        "action": "correct",
        "record_id": "negative",
        "expected_version": 1,
        "replacement": "我现在不喝茶",
    })
    service.apply_action({
        "operation_id": "stats-forget",
        "action": "forget",
        "record_id": "wrapped",
        "expected_version": 1,
        "replacement": "",
    })

    after = service.list_memories()
    assert after["stats"]["effective_count"] == 2
    assert after["stats"]["current_record_count"] == 3
    assert after["stats"]["history_version_count"] == 5
    assert after["stats"]["status_counts"] == {
        "active": 2,
        "suppressed": 0,
        "expired": 0,
        "revoked": 1,
    }
    assert after["stats"]["unknown_event_time_count"] == 2
    assert after["stats"]["count_basis"] == (
        "current_companion_records;effective_excludes_expired_suppressed_revoked"
    )


def test_stale_list_action_refreshes_instead_of_overwriting_current_version(tmp_path):
    store, owner = seeded(tmp_path)
    service = MemoryManagementService(store, A, clock=lambda: 10)
    store.mutate(
        MemoryMutation(
            Scope(A, "other-session"),
            "concurrent-negative",
            "correct",
            "negative",
            "我现在只喝水",
            "manual",
            "other-source",
            "concurrent",
            WHEN,
            expected_record_version=1,
        ),
        owner,
        expected_version=store.snapshot(A).write_version,
    )

    result = service.apply_action(
        {
            "operation_id": "stale-forget",
            "action": "forget",
            "record_id": "negative",
            "expected_version": 1,
            "replacement": "",
        }
    )

    assert result["status"] == "version_changed"
    assert "已经变化" in result["message"]
    assert result["current"]["version"] == 2
    assert store.history(A, "negative")[-1]["content"] == "我现在只喝水"


def test_large_list_is_stably_paginated_without_semantic_merging(tmp_path):
    store = SQLiteExplicitMemoryStore(tmp_path / "memory.sqlite3")
    store.register(A)
    owner = store.claim_owner(A, "seed-owner", expected_epoch=0)
    for index in range(55):
        add_fact(store, owner, f"record-{index:02d}", f"第 {index:02d} 条独立记忆")
    service = MemoryManagementService(store, A, clock=lambda: 10)

    first = service.list_memories(offset=0, limit=50)
    second = service.list_memories(offset=first["next_offset"], limit=50)

    assert len(first["items"]) == 50
    assert len(second["items"]) == 5
    assert first["next_offset"] == 50 and second["next_offset"] is None
    assert len(
        {item["record_id"] for item in [*first["items"], *second["items"]]}
    ) == 55


def test_pending_clarification_is_discoverable_and_can_be_abandoned(tmp_path):
    store, _ = seeded(tmp_path)
    item = store.accept(config(), "ambiguous", "忘掉那个。")
    claim = store.claim_background(A, "worker", now_us=10, lease_us=100, role="user")
    store.protect_memory_source(
        claim,
        reason="needs_clarification.memory_command",
        interpretation_sha256="b" * 64,
        action="unclear",
        requested_scope="none",
        unresolved=("target",),
        now_us=11,
    )
    service = MemoryManagementService(store, A, clock=lambda: 12)

    page = service.list_memories(offset=0, limit=50)

    assert page["attention"][0]["job"] == claim.job
    assert page["attention"][0]["source"] == "忘掉那个。"
    assert page["attention"][0]["resolution_mode"] == "record_selection"
    result = service.abandon(
        {"job": claim.job, "operation_id": "abandon-ambiguous"}
    )
    assert result["status"] == "committed"
    assert store.memory_status(A) == "current"
    assert store.inspect(A, item.input_id)["text"] == "忘掉那个。"


def test_mixed_request_attention_cannot_offer_record_selection(tmp_path):
    store, _ = seeded(tmp_path)
    store.accept(config(), "mixed", "忘记我的音乐偏好，再推荐一首歌。")
    claim = store.claim_background(A, "worker", now_us=10, lease_us=100, role="user")
    store.protect_memory_source(
        claim,
        reason="needs_clarification.mixed_source",
        interpretation_sha256="c" * 64,
        action="forget",
        requested_scope="single_record",
        unresolved=("mixed_content",),
        now_us=11,
    )

    item = MemoryManagementService(store, A, clock=lambda: 12).list_memories()[
        "attention"
    ][0]

    assert item["resolution_mode"] == "whole_source"
    assert "其他请求" in item["message"]
