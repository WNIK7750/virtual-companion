"""Remote feedback records cannot masquerade as known local model artifacts."""

import json
from hashlib import sha256
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from test_default_text_delivery import wait_for
from test_natural_intent_candidate import PreparedRecordingModel
from test_text_provider_bootstrap import application  # noqa: F401
from test_text_provider_bootstrap import runtime_observer as runtime_observer
from virtual_companion.feedback.errors import ValidationError
from virtual_companion.feedback.models import VersionBinding
from virtual_companion.text import __main__ as bootstrap
from virtual_companion.text import default_runtime
from virtual_companion.text.feedback import FeedbackBridge
from virtual_companion.text.sqlite_store import SQLiteConversationStore

API_BINDING = dict(
    app_commit='a' * 40, schema_version=2, persona_sha256='b' * 64,
    prompt_sha256='c' * 64, model_sha256=None, server_sha256=None,
    sampling_profile_id='deepseek-flash.v1', provider_route='d' * 64,
    provider_config='e' * 64, provider_revision=1, provider_model='deepseek-flash',
    provider_wire_profile='deepseek-flash.v1',
)


@pytest.fixture
def feedback_application(monkeypatch, request):
    revision = request.param
    if revision is None:
        monkeypatch.delenv("VC_TEXT_DIAGNOSTIC_PROMPT_REVISION", raising=False)
    else:
        monkeypatch.setenv("VC_TEXT_DIAGNOSTIC_PROMPT_REVISION", revision)
    stores, bridges = [], []
    store_factory, bridge_factory = bootstrap.SQLiteConversationStore, bootstrap.FeedbackBridge

    def store(*args, **kwargs):
        value = store_factory(*args, **kwargs)
        stores.append(value)
        return value

    def bridge(*args, **kwargs):
        value = bridge_factory(*args, **kwargs)
        bridges.append(value)
        return value

    class API(PreparedRecordingModel):
        config_revision = API_BINDING["provider_revision"]
        config_fingerprint = API_BINDING["provider_config"]
        route = type("Route", (), {
            "fingerprint": API_BINDING["provider_route"],
            "model": API_BINDING["provider_model"],
            "wire_profile": API_BINDING["provider_wire_profile"],
        })()

    monkeypatch.setattr(bootstrap, "FakeModel", PreparedRecordingModel)
    monkeypatch.setattr(bootstrap, "SQLiteConversationStore", store)
    monkeypatch.setattr(bootstrap, "FeedbackBridge", bridge)
    dispatch, memories = request.getfixturevalue("application")
    local, api = memories[0].model, API()
    monkeypatch.setattr(bootstrap, "load_provider_model", lambda *a, **k: api)
    yield dispatch, {"local": local, "deepseek-api": api}
    dispatch("stop", {})
    for memory in memories:
        memory.close()
    for value in bridges:
        value.repository.close()
    for value in stores:
        value.close()


@pytest.mark.parametrize("feedback_application", [
    None, *default_runtime.DIAGNOSTIC_PROMPT_REVISIONS,
], indirect=True, ids=["default", *default_runtime.DIAGNOSTIC_PROMPT_REVISIONS])
@pytest.mark.parametrize("provider_id", ["local", "deepseek-api"])
def test_feedback_prompt_hash_matches_checked_and_sent_runtime_control(
    feedback_application, provider_id,
):
    dispatch, models = feedback_application
    if provider_id != "local":
        assert dispatch("selectProvider", {
            "provider_id": provider_id, "confirmed": True,
        })["status"] == "selected"
    initial = dispatch("state", {})
    dispatch("send", {"text": "给我一个雨天散步的简单安排。"})
    event = wait_for(lambda: dispatch("streamPoll", {})["event"])
    dispatch("streamAck", {
        "sequence": event["sequence"], "fingerprint": event["fingerprint"],
        "host_receipt": "synthetic-feedback-visible",
    })
    wait_for(lambda: not dispatch("state", {})["busy"])
    finished = dispatch("state", {})
    assert finished["attempt"]["status"] == "completed"
    generation = next(record for record in models[provider_id].prepared
                      if record["stage"] == "demand_control")
    assert generation["checked"] == generation["sent"]
    system = generation["sent"][0]["content"]
    control = default_runtime.control_for_prompt_revision(initial["request_prompt_revision"])
    assert system.startswith(control + default_runtime.ACCESS_INITIAL)
    actual_control = system[:len(control)]
    actual_hash = sha256(actual_control.encode("utf-8")).hexdigest()
    assert initial["request_control_sha256"] == actual_hash
    message = dispatch("history", {})["messages"][-1]
    payload = dict(operation="submit", session_id=finished["session_id"],
                   turn_id=message["turn_id"], assistant_message_id=message["id"],
                   quick_tags=[], freeform_reaction="synthetic feedback",
                   idempotency_key="synthetic-control-feedback")
    feedback = dispatch("feedback", payload)
    version = feedback["version_binding"]
    assert version == finished["attempt"]["binding"]["version_binding"]
    assert version["prompt_sha256"] == actual_hash
    if provider_id == "local":
        assert version["schema_version"] == 1
        profile = bootstrap.load_profile(Path(__file__).resolve().parents[2]
                                         / "configs/text-baseline-runtime.json")
        assert version["model_sha256"] == profile["artifacts"][1]["sha256"]
        assert version["server_sha256"] == profile["artifacts"][0]["sha256"]
        assert not any(key.startswith("provider_") for key in version)
    else:
        assert version["schema_version"] == 2
        assert version["model_sha256"] is version["server_sha256"] is None
        assert {key: value for key, value in version.items() if key.startswith("provider_")} == {
            key: value for key, value in API_BINDING.items() if key.startswith("provider_")
        }
    target = "deepseek-api" if provider_id == "local" else "local"
    assert dispatch("selectProvider", {
        "provider_id": target, "confirmed": True,
    })["status"] == "selected"
    repeated = dispatch("feedback", payload)
    assert repeated["feedback_id"] == feedback["feedback_id"]
    assert repeated["version_binding"] == version
    revised = dispatch("feedback", dict(operation="revise", feedback_id=feedback["feedback_id"],
                       expected_revision=1, quick_tags=["相关"], freeform_reaction="revised"))
    assert revised["version_binding"] == version


def test_api_feedback_survives_reopen_revision_and_export_with_unknown_model_hash(tmp_path):
    store = SQLiteConversationStore(tmp_path)
    attempt = store.begin('synthetic input', {'version_binding': API_BINDING})
    store.finish(attempt['attempt_id'], 'completed', 'synthetic answer')
    message = store.history()['messages'][-1]
    bridge = FeedbackBridge(store)
    event = bridge.dispatch(dict(operation='submit', session_id=store.session_id,
        turn_id=message['turn_id'], assistant_message_id=message['id'], quick_tags=[],
        freeform_reaction='synthetic feedback', idempotency_key='synthetic-api-feedback'))
    assert event['version_binding'] == API_BINDING
    schema = json.loads((Path(__file__).resolve().parents[2] /
        'packages/contracts/feedback/feedback-event.v1.schema.json').read_text(encoding='utf-8'))
    Draft202012Validator(schema).validate(event)
    bridge.repository.close()
    bridge = FeedbackBridge(store)
    revised = bridge.dispatch(dict(operation='revise', feedback_id=event['feedback_id'],
        expected_revision=1, quick_tags=['相关'], freeform_reaction='revised synthetic feedback'))
    assert revised['version_binding'] == API_BINDING
    preview = bridge.dispatch(dict(operation='preview', feedback_ids=[event['feedback_id']]))
    assert 'provider_wire_profile' in json.dumps(preview['payload'])
    assert bridge.dispatch(dict(operation='confirm', preview_id=preview['preview_id'],
        preview_revision=preview['preview_revision'],
        payload_sha256=preview['payload_sha256'], redaction_confirmed=True,
        external_processing_confirmed=True))['saved_locally']
    bridge.repository.close()
    store.close()


@pytest.mark.parametrize('change', [
    {'model_sha256': 'f' * 64}, {'server_sha256': 'f' * 64}, {'provider_config': ''},
    {'provider_revision': True}, {'provider_revision': 0}, {'provider_model': ''},
    {'provider_route': 'not-a-hash'}, {'schema_version': 3}, {'unknown': 'discard-me'},
])
def test_api_binding_rejects_forged_or_incomplete_remote_identity(change):
    with pytest.raises(ValidationError):
        VersionBinding.from_mapping({**API_BINDING, **change})


def test_local_v1_mapping_remains_exact_and_cannot_silently_drop_api_fields():
    local = {key: value for key, value in API_BINDING.items() if not key.startswith('provider_')}
    local.update(schema_version=1, model_sha256='d' * 64, server_sha256='e' * 64)
    assert VersionBinding.from_mapping(local).to_mapping() == local
    with pytest.raises(ValidationError):
        VersionBinding.from_mapping({**local, 'provider_route': 'd' * 64})
