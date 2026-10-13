import json
import threading
from dataclasses import replace

import pytest
from test_rebuilt_turns import A, claim
from virtual_companion.boundary.contracts import canonical
from virtual_companion.boundary.sqlite_memory_management import SQLiteExplicitMemoryStore
from virtual_companion.cognition.short_context import ShortContextAdmission
from virtual_companion.cognition.short_dialogue import ShortRequest
from virtual_companion.text.character import CharacterSnapshot, load_character
from virtual_companion.text.identity import CompanionConfig
from virtual_companion.text.memory_runtime import TextMemoryRuntime, _StructuredMemoryInvoker


def character(**changes):
    return CharacterSnapshot.from_mapping({
        "protocol": "character-context.v1", "source": "user-config",
        "companion_id": A, "character_id": "synthetic-archivist", "version": "1",
        "personality": "谨慎而好奇", "background": "虚构角色：旧书档案员",
        "initial_relationship": "认识多年的朋友", "expression": "轻快、简洁",
        **changes,
    })


def config(snapshot):
    return CompanionConfig(A, snapshot.reference, character_data=snapshot.payload)


def test_character_loader_is_scoped_bounded_and_does_not_invent_a_default(tmp_path):
    path = tmp_path / "character.json"
    assert load_character(path, A) is None
    snapshot = character()
    path.write_text(snapshot.payload, encoding="utf-8")
    assert load_character(path, A) == snapshot
    with pytest.raises(ValueError, match="companion"):
        load_character(path, "00000000-0000-0000-0000-000000000099")
    with pytest.raises(ValueError):
        character(expression="x" * 32769)
    path.write_text('{"protocol":"wrong"}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_character(path, A)


def test_character_config_freezes_and_validates_content_without_changing_legacy_shape():
    snapshot = character()
    original = config(snapshot)
    assert CompanionConfig.from_mapping(original.to_mapping()) == original
    assert original.with_resources({}).character_data == snapshot.payload
    with pytest.raises(ValueError):
        replace(original, character_data=character(personality="冷静").payload)
    legacy = replace(original, character_data=None)
    assert "character_data" not in legacy.to_mapping()
    assert CompanionConfig.from_mapping(legacy.to_mapping()) == legacy


def test_admission_and_prose_request_share_original_role_but_role_is_not_user_fact(tmp_path):
    store = SQLiteExplicitMemoryStore(tmp_path / "memory.db")
    original = config(character())
    item = store.accept(original, "first", "今天想聊旧书。")
    run = claim(store, item)
    admission = ShortContextAdmission(store, lambda *args: ShortRequest(args[-1]),
                                      evidence_clock=lambda: 101)
    request = admission(run, store.snapshot(A), "今天想聊旧书。")
    expected = json.loads(original.character_data)
    assert json.loads(request.context.payload)["character_context"] == expected
    from virtual_companion.boundary.history_readset import history_readset

    history_readset(A, run.input_id, run.state_version, request.context.payload)
    request = replace(request, context=replace(request.context, rendering="prose"))
    messages = request.model_request(192)["messages"]
    role = [m for m in messages if m["role"] == "system" and "character-context.v1" in m["content"]]
    assert len(role) == 1 and canonical(expected) in role[0]["content"]
    sources = request.personal_sources(A, run.input_id, run.state_version)
    assert all("档案员" not in source["text"] for source in sources)
    admission.validate(request.context)


def test_worker_binds_each_stage_to_original_source_character(tmp_path):
    from virtual_companion.cognition.background_worker import UserFactWorker
    from virtual_companion.text.memory_runtime import _fake_memory_result

    runtime = TextMemoryRuntime(tmp_path / "worker.db", A, character=character())
    runtime.accept_user("first", "今天想聊旧书。", "00000000-0000-0000-0000-000000000010")
    seen = []

    class Recorder(_StructuredMemoryInvoker):
        def __call__(self, stage, messages, schema, cancelled):
            seen.append((stage, self.character))
            return _fake_memory_result(stage, messages, schema, cancelled)

    invoke = Recorder(None, runtime.generation_gate)
    worker = UserFactWorker(runtime.store, invoke, clock=lambda: 100,
                            resolution_protocol=True, resolution_strategy="intent_first",
                            memory_commands=True)
    assert worker.run_one(A, "worker")
    assert seen and all(role == character().to_mapping() for _, role in seen)
    runtime.close()


def test_review_uses_same_role_without_adding_it_to_personal_sources(tmp_path):
    from virtual_companion.cognition.stream_unit_review import (
        BoundedUnitReviewer,
        StructuredUnitAssessor,
    )

    store = SQLiteExplicitMemoryStore(tmp_path / "review.db")
    run = claim(store, store.accept(config(character()), "first", "今天想聊旧书。"))
    admission = ShortContextAdmission(store, lambda *args: ShortRequest(args[-1]),
                                      evidence_clock=lambda: 101)
    request = admission(run, store.snapshot(A), "今天想聊旧书。")

    def capture(stage, messages, schema, cancelled):
        view = json.loads(messages[-1]["content"])
        assert view["context"]["character_context"] == character().to_mapping()
        role_messages = [m for m in messages if m["role"] == "system"
                         and character().payload in m["content"]]
        assert len(role_messages) == 1
        assert all("档案员" not in source["text"] for source in view["personal_sources"])
        raise RuntimeError("captured role review")

    reviewer = BoundedUnitReviewer(run, request, StructuredUnitAssessor(capture, threading.Event()))
    with pytest.raises(RuntimeError, match="captured role review"):
        reviewer("作为旧书档案员，这个话题很合我心意。", True)
@pytest.mark.parametrize("change", ["omit", "replace"])
def test_segment_cannot_approve_with_missing_or_changed_role(tmp_path, change):
    from test_segment_store import reviewed
    from virtual_companion.boundary.contracts import VersionConflict
    from virtual_companion.boundary.history_readset import history_readset

    store = SQLiteExplicitMemoryStore(tmp_path / "segment.db")
    run = claim(store, store.accept(config(character()), "first", "今天想聊旧书。"))
    admission = ShortContextAdmission(store, lambda *args: ShortRequest(args[-1]),
                                      evidence_clock=lambda: 101)
    context = admission(run, store.snapshot(A), "今天想聊旧书。").context
    readset = history_readset(A, run.input_id, run.state_version, context.payload)
    assert readset["protocol"] == "history-readset.v3"
    if change == "replace":
        readset["character_sha256"] = "a" * 64
    else:
        del readset["character_sha256"]
        readset["protocol"] = "history-readset.v1"
    with pytest.raises(VersionConflict, match="character snapshot"):
        store.approve_segment(run, reviewed(run), now_us=102, readset=readset)


def test_memory_source_binding_survives_changed_character_on_restart(tmp_path):
    first = character()
    runtime = TextMemoryRuntime(tmp_path / "memory.db", A, character=first)
    item = runtime.accept_user("first", "今天想聊旧书。", "00000000-0000-0000-0000-000000000010")
    runtime.close()
    runtime = TextMemoryRuntime(tmp_path / "memory.db", A, character=character(personality="沉稳"))
    job = runtime.store.claim_background(A, "worker", now_us=100, lease_us=10000, role="user")
    captured = []

    class Model:
        def stream(self, messages, cancelled):
            captured.append(messages)
            yield '{"operations":[]}'

    invoke = _StructuredMemoryInvoker(Model(), runtime.generation_gate).for_source(
        runtime.store, job)
    invoke("fact_extract", [{"role": "user", "content": "今天想聊旧书。"}], {}, threading.Event())
    role = captured[0][0]
    assert role["role"] == "system"
    assert canonical(first.to_mapping()) in role["content"]
    assert "沉稳" not in role["content"]
    persisted = json.loads(runtime.store.inspect(A, item.input_id)["config"])["companion_config"]
    assert persisted["character_data"] == first.to_mapping()
    runtime.close()
