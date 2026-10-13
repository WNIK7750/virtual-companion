from dataclasses import replace
from uuid import uuid4

import pytest
from virtual_companion.text.identity import CompanionConfig, VersionedReference, companion_data_root
from virtual_companion.text.sqlite_store import SQLiteConversationStore


def config(identity=None, avatar="avatar-a"):
    return CompanionConfig(
        companion_id=identity or str(uuid4()),
        character=VersionedReference("synthetic-character", "v1", "a" * 64),
        resources=(("avatar", VersionedReference(avatar, "v1", "b" * 64)),),
    )


def test_resource_changes_preserve_identity_and_character_but_change_snapshot():
    original = config()
    changed = original.with_resources({"avatar": VersionedReference("avatar-b", "v2", "c" * 64)})
    assert changed.companion_id == original.companion_id
    assert changed.character == original.character
    assert changed.snapshot_hash != original.snapshot_hash
    assert original.resources[0][1].id == "avatar-a"
    assert CompanionConfig.from_mapping(changed.to_mapping()) == changed


def test_two_configs_use_separate_sqlite_history_and_session_ownership(tmp_path):
    first, second = config(), config()
    one = SQLiteConversationStore(companion_data_root(tmp_path, first.companion_id))
    two = SQLiteConversationStore(companion_data_root(tmp_path, second.companion_id))
    try:
        attempt = one.begin("first companion only", first.attempt_binding({}))
        one.finish(attempt["attempt_id"], "completed", "first reply")
        assert two.history()["messages"] == []
        with pytest.raises(ValueError):
            two.select_session(one.session_id)
        old_session = one.session_id
        one.new_session()
        assert one.history()["messages"] == []
        one.select_session(old_session)
        assert one.history()["messages"][-1]["content"] == "first reply"
    finally:
        one.close()
        two.close()


def test_resource_change_and_config_roundtrip_reopen_same_history(tmp_path):
    original = config()
    store = SQLiteConversationStore(companion_data_root(tmp_path, original.companion_id))
    attempt = store.begin("retained", original.attempt_binding({"version_binding": {"test": True}}))
    store.finish(attempt["attempt_id"], "completed", "reply")
    store.close()
    changed = original.with_resources({"voice": VersionedReference("voice-a", "v3", "d" * 64)})
    restored = CompanionConfig.from_mapping(changed.to_mapping())
    reopened = SQLiteConversationStore(companion_data_root(tmp_path, restored.companion_id))
    try:
        assert len(reopened.history()["messages"]) == 2
        assert (
            reopened.latest_attempt()["binding"]["companion_config_sha256"]
            == original.snapshot_hash
        )
        next_attempt = reopened.begin("new resources", restored.attempt_binding({}))
        assert next_attempt["binding"]["companion_config_sha256"] == restored.snapshot_hash
    finally:
        reopened.close()


@pytest.mark.parametrize("identity", ["../escape", "D:/elsewhere", "not-a-uuid", ""])
def test_invalid_identity_cannot_become_a_storage_path(tmp_path, identity):
    with pytest.raises(ValueError):
        companion_data_root(tmp_path, identity)


def test_unknown_fields_and_duplicate_resource_kinds_are_rejected():
    original = config()
    with pytest.raises(ValueError):
        CompanionConfig.from_mapping({**original.to_mapping(), "execute": "anything"})
    with pytest.raises(ValueError):
        replace(original, resources=original.resources * 2)


def test_attempt_binding_is_a_detached_snapshot_and_cannot_rebind_another_owner():
    original = config()
    base = {"version_binding": {"model": "synthetic"}}
    binding = original.attempt_binding(base)
    binding["version_binding"]["model"] = "changed"
    assert base["version_binding"]["model"] == "synthetic"
    with pytest.raises(ValueError):
        original.attempt_binding({"companion_id": str(uuid4())})


def test_config_hash_is_independent_of_resource_mapping_order():
    original = config()
    avatar = VersionedReference("avatar-a", "v1", "b" * 64)
    voice = VersionedReference("voice-a", "v2", "c" * 64)
    one = original.with_resources({"avatar": avatar, "voice": voice})
    two = original.with_resources({"voice": voice, "avatar": avatar})
    assert one == two
    assert one.snapshot_hash == two.snapshot_hash


@pytest.mark.parametrize(
    "values",
    [
        ("../external", "v1", "a" * 64),
        ("valid", "", "a" * 64),
        ("valid", "v1", "not-a-hash"),
    ],
)
def test_resource_references_reject_paths_and_unpinned_versions(values):
    with pytest.raises(ValueError):
        VersionedReference(*values)


def test_feedback_database_follows_companion_scope(tmp_path):
    from virtual_companion.text.feedback import FeedbackBridge

    one_config, two_config = config(), config()
    one = SQLiteConversationStore(companion_data_root(tmp_path, one_config.companion_id))
    two = SQLiteConversationStore(companion_data_root(tmp_path, two_config.companion_id))
    binding = {
        "app_commit": "a" * 40,
        "schema_version": 1,
        "persona_sha256": "b" * 64,
        "prompt_sha256": "c" * 64,
        "model_sha256": "d" * 64,
        "server_sha256": "e" * 64,
        "sampling_profile_id": "synthetic-identity",
    }
    first_bridge, second_bridge = FeedbackBridge(one), FeedbackBridge(two)
    try:
        attempt = one.begin("synthetic", one_config.attempt_binding({"version_binding": binding}))
        one.finish(attempt["attempt_id"], "completed", "reply")
        message = one.history()["messages"][-1]
        first_bridge.dispatch(
            {
                "operation": "submit",
                "session_id": one.session_id,
                "turn_id": message["turn_id"],
                "assistant_message_id": message["id"],
                "quick_tags": [],
                "freeform_reaction": "synthetic feedback",
                "idempotency_key": "identity-fixture",
            }
        )
        assert len(first_bridge.dispatch({"operation": "list"})["events"]) == 1
        assert second_bridge.dispatch({"operation": "list"})["events"] == []
    finally:
        first_bridge.repository.close()
        second_bridge.repository.close()
        one.close()
        two.close()
