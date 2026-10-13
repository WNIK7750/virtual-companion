"""A single immutable request may reuse validation only until its live token/TTL changes."""

import json
from dataclasses import replace

import pytest
from test_effective_history import NOW, admission, complete, seeded
from test_explicit_memory_management import list_command
from test_rebuilt_turns import A, config
from virtual_companion.boundary.contracts import VersionConflict, canonical
from virtual_companion.cognition.short_context import ShortContextAdmission
from virtual_companion.cognition.short_dialogue import ShortRequest


def prepared(tmp_path, *, expires=False):
    store, original, _, _ = seeded(tmp_path, expires=expires)
    item = store.accept(config(), "next", "hello", session_id=original.session_id)
    complete(store, item)
    _, request = admission(store, item)
    clock = [NOW + 20]
    admit = ShortContextAdmission(store, lambda *args: ShortRequest(args[-1]),
                                  evidence_clock=lambda: clock[0])
    scans = []
    original_reader = store.ineffective_history_sources

    def reader(*args, **kwargs):
        scans.append(1)
        return original_reader(*args, **kwargs)

    store.ineffective_history_sources = reader
    return store, request.context, admit, clock, scans


def test_same_context_deltas_do_not_repeat_full_projection(tmp_path):
    _, context, admit, _, scans = prepared(tmp_path)
    for _ in range(20):
        admit.validate(context)
    assert len(scans) == 1
    admit.validate(replace(context))
    assert len(scans) == 2  # A different request object is not a cache capability.


@pytest.mark.parametrize("field,value", [
    ("source_id", "foreign-input"), ("source_sha256", "0" * 64), ("status", "queued"),
    ("missing", None),
])
def test_cross_instance_validation_requires_exact_persisted_processing(tmp_path, field, value):
    store, context, admit, _, _ = prepared(tmp_path)
    data = json.loads(context.payload)
    assert store.source_processing_for_input(A, context.input_id) == data["source_processing"]
    if field == "missing":
        del data["source_processing"]
        del data["memory_access"]
    else:
        data["source_processing"][field] = value
    with pytest.raises(VersionConflict, match="processing"):
        admit.validate(replace(context, payload=canonical(data)))


def test_source_completion_changes_processing_even_without_state_version_change(tmp_path):
    store, original, _, _ = seeded(tmp_path)
    item = store.accept(config(), "queued", "hello", session_id=original.session_id)
    _, request = admission(store, item)
    admit = ShortContextAdmission(store, lambda *args: ShortRequest(args[-1]),
                                  evidence_clock=lambda: NOW + 20)
    admit.validate(request.context)
    version = store.snapshot(A).write_version
    store.cancel(A, item.input_id)
    complete(store, item)
    assert store.snapshot(A).write_version == version
    assert store.source_processing_for_input(A, item.input_id)["status"] == "completed"
    with pytest.raises(VersionConflict, match="processing"):
        admit.validate(request.context)


def test_source_arrival_rejects_reused_context(tmp_path):
    store, context, admit, _, _ = prepared(tmp_path)
    admit.validate(context)
    store.accept(config(), "correction", "刚才偏好有误")
    with pytest.raises(VersionConflict):
        admit.validate(context)


def test_indirect_history_ttl_expires_without_a_write_or_selected_memory(tmp_path):
    import json

    store, context, admit, clock, _ = prepared(tmp_path, expires=True)
    assert json.loads(context.payload)["memories"] == []
    admit.validate(context)
    before = store.snapshot(A).write_version
    clock[0] = NOW + 2_000_000
    with pytest.raises(VersionConflict):
        admit.validate(context)
    assert store.snapshot(A).write_version == before


def test_clock_reversal_cannot_extend_cached_deadline(tmp_path):
    _, context, admit, clock, _ = prepared(tmp_path)
    admit.validate(context)
    clock[0] -= 1
    with pytest.raises(VersionConflict, match="clock"):
        admit.validate(context)


def test_projection_token_race_is_not_cached(tmp_path):
    store, context, admit, _, _ = prepared(tmp_path)
    original = store.ineffective_history_sources

    def raced(*args, **kwargs):
        result = original(*args, **kwargs)
        store.accept(config(), "racing-source", "新来源")
        return result

    store.ineffective_history_sources = raced
    with pytest.raises(VersionConflict):
        admit.validate(context)


def test_noop_management_command_without_version_change_invalidates_token(tmp_path):
    store, _, _, _ = seeded(tmp_path)
    store.execute_memory_command(list_command(), now_us=NOW + 20)
    item = store.accept(config(), "next", "hello")
    complete(store, item)
    _, request = admission(store, item)
    admit = ShortContextAdmission(store, lambda *args: ShortRequest(args[-1]),
                                  evidence_clock=lambda: NOW + 21)
    admit.validate(request.context)
    before = store.snapshot(A).write_version
    token = store.context_validation_token(A, now_us=NOW + 21)
    store.execute_memory_command(list_command(operation="noop", version=2), now_us=NOW + 22)
    assert store.snapshot(A).write_version == before
    assert store.context_validation_token(A, now_us=NOW + 23) != token


def test_actual_repeat_completion_invalidates_cached_source_alias_without_version_bump(tmp_path):
    import json

    from virtual_companion.cognition.background_worker import UserFactWorker
    from virtual_companion.text.memory_runtime import _fake_memory_result

    store, original, _, _ = seeded(tmp_path)
    repeat = store.accept(config(), "repeat", "我喜欢安静的音乐", session_id=original.session_id)
    store.cancel(A, repeat.input_id)  # Turn cancellation does not complete its background source.
    current = store.accept(config(), "pending", "hello", session_id=original.session_id)
    _, request = admission(store, current)
    assert repeat.input_id in {r["id"] for r in json.loads(request.context.payload)["history"]}
    admit = ShortContextAdmission(store, lambda *args: ShortRequest(args[-1]),
                                  evidence_clock=lambda: NOW + 20)
    admit.validate(request.context)
    version = store.snapshot(A).write_version
    # Release foreground; evidence validation is tested separately.
    store.cancel(A, current.input_id)
    result = UserFactWorker(store, _fake_memory_result, clock=lambda: NOW + 21,
                            resolution_protocol=True, resolution_strategy="intent_first",
                            memory_commands=True).run_one(A, "worker", source=repeat.input_id)
    assert json.loads(result)["no_change_operations"] == 1
    assert store.snapshot(A).write_version == version
    assert store.memory_barrier(A)  # The other user source remains unresolved.
    # Completion of a byte-exact repeat remains valid evidence even while a
    # different user source is pending. The changed source inventory still fences
    # this cached context below, without treating the repeated fact as revoked.
    assert repeat.input_id not in store.ineffective_history_sources(A, now_us=NOW + 22)
    with pytest.raises(VersionConflict, match="token"):
        admit.validate(request.context)


@pytest.mark.parametrize("during_read", ["expiry", "backwards"])
def test_cached_token_read_cannot_cross_deadline_or_reverse_clock(tmp_path, during_read):
    store, context, admit, clock, _ = prepared(tmp_path, expires=True)
    admit.validate(context)
    original = store.context_validation_token

    def delayed(*args, **kwargs):
        token = original(*args, **kwargs)
        clock[0] = NOW + 2_000_000 if during_read == "expiry" else clock[0] - 1
        return token

    store.context_validation_token = delayed
    with pytest.raises(VersionConflict):
        admit.validate(context)


def test_clock_reversal_is_checked_against_latest_cache_hit(tmp_path):
    _, context, admit, clock, _ = prepared(tmp_path)
    admit.validate(context)
    clock[0] += 10
    admit.validate(context)
    clock[0] -= 1
    with pytest.raises(VersionConflict, match="clock"):
        admit.validate(context)
