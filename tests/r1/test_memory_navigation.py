from datetime import UTC, datetime, timedelta

import pytest
from virtual_companion.boundary.contracts import Scope, VersionConflict, canonical
from virtual_companion.boundary.memory import FactKey, MemoryMutation
from virtual_companion.boundary.memory_management import (
    ExplicitMemoryCommand,
    list_source_digest,
)
from virtual_companion.boundary.sqlite_memory_management import SQLiteExplicitMemoryStore
from virtual_companion.text.chronology import utc_microseconds
from virtual_companion.text.memory_navigation import MemoryCatalog

A = "00000000-0000-0000-0000-000000000001"
B = "00000000-0000-0000-0000-000000000002"
WHEN = datetime(2026, 10, 12, tzinfo=UTC)
NOW = utc_microseconds(WHEN)


def _store(path):
    store = SQLiteExplicitMemoryStore(path)
    for companion in (A, B):
        store.register(companion)
        store.claim_owner(companion, f"seed-{companion[-4:]}", expected_epoch=0)
    return store


def _seed(store, companion, record_id, content, *, predicate=None, expires_at=None,
          kind="assert_fact", source_kind="manual", subject="user", scope="profile"):
    fact_key = FactKey(subject, predicate, scope) if predicate else None
    mutation = MemoryMutation(
        Scope(companion, "catalog-test"), f"seed-{record_id}", kind, record_id,
        content, source_kind, f"source-{record_id}", "synthetic catalog fixture",
        WHEN, fact_key=fact_key, expires_at=expires_at,
    )
    return store.mutate(
        mutation, store.owner(companion),
        expected_version=store.snapshot(companion).write_version,
    )


def _catalog(store, companion=A, *, now=None, checkpoint=None, max_records=1000, page_size=8):
    clock = now if now is not None else lambda: NOW
    return MemoryCatalog(
        store, companion, state_version=store.snapshot(companion).write_version,
        clock=clock, checkpoint=checkpoint or (lambda: None), max_records=max_records,
        page_size=page_size,
    )


def _forget_command(record_id, version=1):
    operation_id = f"forget-{record_id}"
    digest = list_source_digest(
        action="forget", record_id=record_id, expected_version=version,
    )
    return ExplicitMemoryCommand(
        Scope(A, "memory-list"), operation_id, "forget", record_id, version,
        "list", operation_id, digest, "single_record", "explicit_selection",
        "host-list.v1", digest,
    )


def test_catalog_pages_find_complete_original_beyond_eight_without_retaining_bodies(tmp_path):
    store = _store(tmp_path / "catalog.sqlite3")
    for index in range(10):
        content = ("目标原文：地铁站旁的旧书摊，周三傍晚会摆出旧地图。\n" * 22
                   if index == 9 else f"合成事实 {index}。")
        _seed(store, A, f"fact-{index:02d}", content, predicate=f"topic-{index:02d}")
    _seed(store, A, "event-walk", "上周一起散步。", kind="append_event",
          source_kind="input")
    _seed(store, B, "foreign-record", "另一位用户的合成原文。", predicate="private-note")

    catalog = _catalog(store, page_size=8)
    root = catalog.browse()
    subject = next(entry for entry in root["entries"] if entry["entry_type"] == "subject")
    event_kind = next(entry for entry in root["entries"]
                      if entry["entry_type"] == "source_kind")

    first_predicates = catalog.browse(subject["node_id"])
    assert len(first_predicates["entries"]) == 8
    assert first_predicates["next_cursor"]
    second_predicates = catalog.browse(subject["node_id"], cursor=first_predicates["next_cursor"])
    target = next(entry for entry in second_predicates["entries"]
                  if entry["label"] == "topic-09")
    leaves = catalog.browse(target["node_id"])
    leaf = next(entry for entry in leaves["entries"] if entry["record_id"] == "fact-09")
    (row,) = catalog.read(((leaf["record_id"], leaf["version"]),))

    expected = "目标原文：地铁站旁的旧书摊，周三傍晚会摆出旧地图。\n" * 22
    assert row["content"] == expected
    assert len(row["content"]) > 500
    assert catalog.identity == root["catalog_sha256"]
    assert root["catalog_coverage"] == "complete"
    assert len(catalog.identity) == 64
    for page in (root, first_predicates, second_predicates, leaves, event_leaves :=
                 catalog.browse(event_kind["node_id"])):
        assert len(page["entries"]) <= 8
        assert len(canonical(page).encode("utf-8")) <= 8192
        assert all(len(entry["summary"]) <= 240 for entry in page["entries"])
    assert "foreign-record" not in repr(root)
    assert expected not in repr(catalog)
    assert "content" not in catalog._metadata[0]

    assert [entry["record_id"] for entry in event_leaves["entries"]] == ["event-walk"]
    assert event_leaves["entries"][0]["navigation_only"] is True
    assert event_leaves["entries"][0]["summary"].startswith(
        "导航预览，需 read_memory 回查原文：上周一起散步。"
    )
    assert event_leaves["entries"][0]["preview_truncated"] is False
    with pytest.raises(ValueError, match="catalog"):
        catalog.read((("foreign-record", 1),))


def test_partial_coverage_and_cursors_are_bound_to_catalog_and_node(tmp_path):
    store = _store(tmp_path / "partial.sqlite3")
    for index in range(5):
        _seed(store, A, f"partial-{index}", f"原文 {index}", predicate=f"p-{index}")
    _seed(store, B, "other-0", "其他owner", predicate="other")

    partial = _catalog(store, max_records=3, page_size=2)
    root = partial.browse()
    subject = next(entry for entry in root["entries"] if entry["entry_type"] == "subject")
    first = partial.browse(subject["node_id"])
    assert first["catalog_coverage"] == "partial"
    assert subject["record_count"] == 3
    assert first["next_cursor"] is not None
    with pytest.raises(ValueError, match="cursor"):
        partial.browse(subject["node_id"], cursor="!")

    other_node = "root"
    with pytest.raises(ValueError, match="[Cc]ursor"):
        partial.browse(other_node, cursor=first["next_cursor"])

    complete = _catalog(store, max_records=10, page_size=2)
    complete_root = complete.browse()
    complete_subject = next(entry for entry in complete_root["entries"]
                            if entry["entry_type"] == "subject")
    with pytest.raises(ValueError, match="[Cc]ursor"):
        complete.browse(complete_subject["node_id"], cursor=first["next_cursor"])

    foreign = _catalog(store, companion=B, max_records=10, page_size=2)
    foreign_root = foreign.browse()
    foreign_subject = next(entry for entry in foreign_root["entries"]
                           if entry["entry_type"] == "subject")
    with pytest.raises(ValueError, match="[Cc]ursor"):
        foreign.browse(foreign_subject["node_id"], cursor=first["next_cursor"])


def test_read_rechecks_correction_and_explicit_forget_instead_of_using_old_catalog(tmp_path):
    store = _store(tmp_path / "changes.sqlite3")
    _seed(store, A, "correctable", "旧内容", predicate="home")
    _seed(store, A, "forgettable", "需要忘记的内容", predicate="music")
    def clock():
        return NOW

    catalog = _catalog(store, now=clock)
    (old_row,) = catalog.read((("correctable", 1),))
    assert old_row["content"] == "旧内容"

    correction = MemoryMutation(
        Scope(A, "catalog-test"), "correct-correctable", "correct", "correctable",
        "新内容", "manual", "source-correctable-new", "synthetic correction", WHEN,
        expected_record_version=1,
    )
    store.mutate(correction, store.owner(A), expected_version=store.snapshot(A).write_version)
    with pytest.raises(VersionConflict):
        catalog.read((("correctable", 1),))

    corrected = _catalog(store, now=clock)
    with pytest.raises(ValueError, match="catalog"):
        corrected.read((("correctable", 1),))
    (new_row,) = corrected.read((("correctable", 2),))
    assert new_row["content"] == "新内容"

    before_forget = _catalog(store, now=clock)
    receipt = store.execute_memory_command(_forget_command("forgettable"), now_us=NOW)
    assert receipt["status"] == "committed"
    with pytest.raises(VersionConflict):
        before_forget.read((("forgettable", 1),))
    after_forget = _catalog(store, now=clock)
    with pytest.raises(ValueError, match="catalog"):
        after_forget.read((("forgettable", 1),))
    assert store.committed_recall(A, now_us=NOW, candidates=(("forgettable", 1),)) == []


def test_expiry_during_exact_original_read_rejects_result(tmp_path):
    store = _store(tmp_path / "expiry.sqlite3")
    expires = WHEN + timedelta(minutes=1)
    _seed(store, A, "short-lived", "TTL 原文", predicate="temporary", expires_at=expires)
    now = [NOW]
    catalog = _catalog(store, now=lambda: now[0])
    original_read = store.committed_recall

    def expire_after_read(*args, **kwargs):
        rows = original_read(*args, **kwargs)
        now[0] = utc_microseconds(expires)
        return rows

    store.committed_recall = expire_after_read
    with pytest.raises(VersionConflict):
        catalog.read((("short-lived", 1),))


def test_checkpoint_and_clock_are_checked_around_navigation_and_original_read(tmp_path):
    store = _store(tmp_path / "checkpoint.sqlite3")
    _seed(store, A, "checkpoint-record", "原文", predicate="topic")
    cancelled = [False]

    def checkpoint():
        if cancelled[0]:
            raise RuntimeError("catalog cancelled")

    catalog = _catalog(store, checkpoint=checkpoint)
    original_read = store.committed_recall

    def cancel_after_read(*args, **kwargs):
        rows = original_read(*args, **kwargs)
        cancelled[0] = True
        return rows

    store.committed_recall = cancel_after_read
    with pytest.raises(RuntimeError, match="catalog cancelled"):
        catalog.read((("checkpoint-record", 1),))

    now = [NOW]
    clock_checked = _catalog(store, now=lambda: now[0])
    now[0] -= 1
    with pytest.raises(VersionConflict, match="clock moved backwards"):
        clock_checked.browse()


def test_utf8_page_budget_cursor_accepts_short_page_offsets(tmp_path):
    store = _store(tmp_path / "utf8-budget.sqlite3")
    subject = "主" * 128
    scope = "域" * 128
    for index in range(10):
        predicate = "谓" * 127 + str(index)
        _seed(store, A, f"long-label-{index}", f"记录 {index}",
              predicate=predicate, subject=subject, scope=scope)

    catalog = _catalog(store, page_size=8)
    root = catalog.browse()
    subject_entry, = root["entries"]
    first = catalog.browse(subject_entry["node_id"])
    assert 0 < len(first["entries"]) < 8
    assert first["next_cursor"] is not None
    assert len(first["next_cursor"]) <= 512
    assert len(canonical(first).encode("utf-8")) <= 8192

    pages = [first]
    cursor = first["next_cursor"]
    while cursor is not None:
        page = catalog.browse(subject_entry["node_id"], cursor=cursor)
        assert len(page["entries"]) <= 8
        assert len(canonical(page).encode("utf-8")) <= 8192
        pages.append(page)
        cursor = page["next_cursor"]
    predicates = [entry["predicate"] for page in pages for entry in page["entries"]]
    assert len(predicates) == 10
    assert len(set(predicates)) == 10


def test_same_category_records_have_distinct_bounded_navigation_previews(tmp_path):
    store = _store(tmp_path / "navigation-previews.sqlite3")
    first_original = "公交站牌旁的书店周三会摆旧地图。" + ("甲" * 230)
    second_original = "河边的修车摊周五才营业。" + ("乙" * 230)
    _seed(store, A, "event-first", first_original, kind="append_event",
          source_kind="input")
    _seed(store, A, "event-second", second_original, kind="append_event",
          source_kind="input")

    catalog = _catalog(store)
    root = catalog.browse()
    event_kind = next(entry for entry in root["entries"]
                      if entry["entry_type"] == "source_kind")
    page = catalog.browse(event_kind["node_id"])
    entries = {entry["record_id"]: entry for entry in page["entries"]}

    first = entries["event-first"]
    second = entries["event-second"]
    assert first["summary"].startswith("导航预览，需 read_memory 回查原文：")
    assert second["summary"].startswith("导航预览，需 read_memory 回查原文：")
    first_preview = first["summary"].split("：", 1)[1]
    second_preview = second["summary"].split("：", 1)[1]
    assert first_preview != second_preview
    assert len(first_preview) == len(second_preview) == 160
    assert first["preview_truncated"] is True
    assert second["preview_truncated"] is True
    assert first_original not in repr(page)
    assert second_original not in repr(page)
    assert all(entry["navigation_only"] is True for entry in entries.values())
