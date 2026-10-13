"""Product Flash default: bind before the first authority or session is built.

All models, credentials, transport, server, and thread entry points are CPU
recordings. These checks never load a model, decrypt a key, or send an API call.
"""

import io
import json
import sqlite3
import threading
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_default_text_delivery import wait_for
from test_natural_intent_candidate import PreparedRecordingModel
from virtual_companion.text import __main__ as bootstrap
from virtual_companion.text.fake import FakeModel
from virtual_companion.text.memory_runtime import _fake_memory_result
from virtual_companion.text.provider_selection import ProviderSelection

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "configs/text-baseline-provider.json"
API_ID = "deepseek-api"


def flash_catalog():
    value = json.loads(CATALOG.read_text(encoding="utf-8"))
    value["default_provider"] = API_ID
    return value


def test_product_catalog_defaults_to_flash():
    value = json.loads(CATALOG.read_text(encoding="utf-8"))
    assert value["default_provider"] == API_ID
    assert value["api"]["provider_id"] == API_ID
    assert "Flash" in value["api"]["model_label"]


def test_catalog_leaves_live_readiness_and_network_state_to_the_binding():
    value = json.loads(CATALOG.read_text(encoding="utf-8"))
    assert "status" not in value["api"]
    assert "network_enabled" not in value["api"]
    selected = ProviderSelection(CATALOG, {}, configured=True, switch=lambda _: {})
    assert selected.state()["options"][1]["network_enabled"] is False
    selected.select(API_ID, confirmed=True)
    assert selected.state()["options"][1]["network_enabled"] is True


def test_catalog_accepts_its_named_api_as_the_default(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(flash_catalog()), encoding="utf-8")
    selected = ProviderSelection(path, {"profile_id": "synthetic-local"})
    assert selected.state()["default_provider"] == API_ID
    assert selected.selected == API_ID


def test_selecting_default_api_must_bind_it_before_reporting_selected(monkeypatch):
    monkeypatch.setattr(ProviderSelection, "_load", staticmethod(lambda _: flash_catalog()))
    calls = []
    identity = {"config_revision": 7, "model_profile": "deepseek-flash.focused.v2"}

    def switch(provider_id):
        calls.append(provider_id)
        return identity

    selected = ProviderSelection(CATALOG, {}, configured=True, switch=switch)
    result = selected.select(API_ID, confirmed=True)
    assert calls == [API_ID], "a catalog selection is not an actual provider binding"
    assert result["status"] == "selected"
    assert selected.state()["active"]["config_revision"] == 7
    assert selected.state()["active"]["model_profile"] == "deepseek-flash.focused.v2"


def test_unconfigured_default_api_cannot_report_an_active_binding(monkeypatch):
    monkeypatch.setattr(ProviderSelection, "_load", staticmethod(lambda _: flash_catalog()))
    selected = ProviderSelection(CATALOG, {}, configured=False)
    result = selected.select(API_ID, confirmed=True)
    assert result["status"] == "unavailable"
    assert result["provider"]["active"]["status"] != "active"
    assert result["provider"]["selected_provider"] == API_ID


@pytest.fixture
def boot(monkeypatch, tmp_path):
    """Run the real bootstrap with recording model and inert process boundaries."""
    events, resources = [], []
    factory_memory = bootstrap.TextMemoryRuntime
    factory_runtime = bootstrap.DefaultTextRuntime
    factory_session = bootstrap.TextSession
    factory_store = bootstrap.SQLiteConversationStore
    factory_feedback = bootstrap.FeedbackBridge
    find_spec = bootstrap.importlib.util.find_spec
    factory_thread = threading.Thread
    attach_foreground = factory_memory.attach_foreground

    class CPUModel(FakeModel):
        process = None

        def __init__(self):
            self.sent = []
            self.started = False

        def start(self):
            self.started = True
            raise AssertionError("the default API route must never start local inference")

        def stream(self, *_):
            self.sent.append("unexpected generation")
            raise AssertionError("bootstrap must not send any model request")

    class API(CPUModel):
        config_revision = 7
        config_fingerprint = "b" * 64
        route = SimpleNamespace(
            fingerprint="a" * 64,
            model="deepseek-flash",
            wire_profile="deepseek-flash.focused.v2",
        )

        def review_peer(self):
            return self

    class RecordingAPI(PreparedRecordingModel):
        config_revision = API.config_revision
        config_fingerprint = API.config_fingerprint
        route = API.route
        supports_stage_output_override = True

        def __init__(self):
            super().__init__()
            self.sent = []

        def review_peer(self):
            return self

        def prepare_stage_request(self, messages, *, stage, schema, output_tokens=None):
            request = super().prepare_stage_request(messages, stage=stage, schema=schema)
            original_stream = request.stream

            def stream(cancelled):
                self.sent.append(stage)
                if stage in {"demand_control", "stream_unit_review"}:
                    yield from original_stream(cancelled)
                else:
                    yield json.dumps(_fake_memory_result(stage, messages, schema, cancelled))

            request.stream = stream
            return request

    def memory(*args, **kwargs):
        value = factory_memory(*args, **kwargs)
        events.append(("memory", value.model, value.fake))
        resources.append(value)
        return value

    def runtime(*args, **kwargs):
        value = factory_runtime(*args, **kwargs)
        events.append(("runtime", value.model, deepcopy(kwargs["binding"])))
        return value

    def session(*args, **kwargs):
        value = factory_session(*args, **kwargs)
        events.append(("session", value.model, deepcopy(value.binding)))
        return value

    def store(*args, **kwargs):
        value = factory_store(*args, **kwargs)
        resources.append(value)
        return value

    def feedback(*args, **kwargs):
        value = factory_feedback(*args, **kwargs)
        resources.append(value.repository)
        return value

    class NoThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

    monkeypatch.setenv("VC_DATA_ROOT", str(tmp_path))
    for variable in ("VC_MEMORY_MANAGEMENT_DB", "VC_MEMORY_COMPANION_ID",
                     "VC_TEXT_DIAGNOSTIC_PROMPT_REVISION"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(ProviderSelection, "_load", staticmethod(lambda _: flash_catalog()))
    monkeypatch.setattr(bootstrap, "LlamaModel", lambda *_: CPUModel())
    monkeypatch.setattr(bootstrap, "TextMemoryRuntime", memory)
    monkeypatch.setattr(bootstrap, "DefaultTextRuntime", runtime)
    monkeypatch.setattr(bootstrap, "TextSession", session)
    monkeypatch.setattr(bootstrap, "SQLiteConversationStore", store)
    monkeypatch.setattr(bootstrap, "FeedbackBridge", feedback)
    monkeypatch.setattr(bootstrap.threading, "Thread", NoThread)
    monkeypatch.setattr(factory_memory, "attach_foreground", lambda *_: None)
    monkeypatch.setattr(bootstrap, "contain_current_process", lambda: True)
    monkeypatch.setattr(bootstrap.importlib.util, "find_spec",
                        lambda name: True if name == "tokenizers" else find_spec(name))

    def launch(*, fake=False, configured=True, invalid=False, missing_dependency=False,
               workers=False, wrong_model=False, local_default=False,
               module_entry=False, server_error=None, serve_error=None):
        events.clear()
        catalog = flash_catalog()
        if local_default:
            catalog["default_provider"] = "local"
        monkeypatch.setattr(ProviderSelection, "_load", staticmethod(lambda _: catalog))
        provider_path = tmp_path / "provider.json"
        if configured:
            provider_path.write_text("{}", encoding="utf-8")
        elif provider_path.exists():
            provider_path.unlink()
        captured = {}
        api = RecordingAPI() if workers else API()
        if wrong_model:
            api.route = SimpleNamespace(**{**vars(api.route), "model": "another-model"})

        def load(path, *, data_root):
            events.append(("load", Path(path), Path(data_root)))
            if not configured or invalid:
                raise ValueError("Synthetic managed provider configuration missing")
            return api

        def server(_, dispatch):
            if server_error is not None:
                raise server_error
            captured["dispatch"] = dispatch

            def serve_forever(**_):
                if serve_error is not None:
                    raise serve_error

            return SimpleNamespace(serve_forever=serve_forever)

        monkeypatch.setattr(bootstrap, "load_provider_model", load)
        monkeypatch.setattr(bootstrap, "create_server", server)
        monkeypatch.setattr(bootstrap.importlib.util, "find_spec", lambda name: (
            None if missing_dependency else True) if name == "tokenizers" else find_spec(name))
        if workers:
            def thread(*args, **kwargs):
                if getattr(kwargs.get("target"), "__name__", "") == "watch_parent":
                    return NoThread()
                return factory_thread(*args, **kwargs)

            monkeypatch.setattr(bootstrap.threading, "Thread", thread)
            monkeypatch.setattr(factory_memory, "attach_foreground", attach_foreground)
        monkeypatch.setattr(bootstrap.sys, "argv", ["text", "--fake"] if fake else ["text"])
        monkeypatch.setattr(bootstrap.sys, "stdin", io.StringIO("x" * 64 + "\n"))
        (bootstrap._run_module if module_entry else bootstrap.main)()
        return captured["dispatch"], api, list(events)

    yield launch
    for resource in reversed(resources):
        resource.close()


def assert_flash_authorities(events, api):
    assert events[0][0] == "load", "bind the route before creating any background owner"
    assert [event[0] for event in events] == ["load", "memory", "runtime", "session"]
    memory, runtime, session = events[1:]
    assert memory[1] is runtime[1] is session[1] is api
    assert memory[2] is False
    assert api.review_peer().route is api.route
    binding = session[2]["version_binding"]
    assert runtime[2] == binding
    assert binding["schema_version"] == 2
    assert binding["model_sha256"] is binding["server_sha256"] is None
    assert binding["provider_route"] == api.route.fingerprint
    assert binding["provider_config"] == api.config_fingerprint
    assert binding["provider_revision"] == api.config_revision
    assert binding["provider_model"] == "deepseek-flash"
    assert binding["provider_wire_profile"] == "deepseek-flash.focused.v2"


def test_first_memory_runtime_and_session_use_bound_flash_identity(boot):
    dispatch, api, events = boot()
    assert_flash_authorities(events, api)
    state = dispatch("state", {})
    assert state["mode"] == "api"
    assert state["state"] == "ready"
    assert state["startup"]["status"] == "configured"
    assert state["startup"]["can_send"] is True
    assert len(state["startup"]["identity_sha256"]) == 64
    assert state["startup"]["components"]["provider_route"]["status"] == "bound"
    assert state["provider"]["active"]["model_profile"] == api.route.wire_profile
    assert state["provider"]["active"]["config_revision"] == api.config_revision
    assert api.sent == []


def test_reopen_rebinds_flash_to_the_same_selected_data_root(boot):
    first, api, events = boot()
    assert_flash_authorities(events, api)
    session_id = first("state", {})["session_id"]
    first_load = events[0]
    reopened, next_api, next_events = boot()
    assert_flash_authorities(next_events, next_api)
    assert next_events[0] == first_load
    assert reopened("state", {})["session_id"] == session_id
    assert api.sent == next_api.sent == []


@pytest.mark.parametrize("options", [
    {"configured": False}, {"invalid": True}, {"missing_dependency": True},
    {"wrong_model": True},
], ids=["missing-configuration", "invalid-configuration", "missing-dependency", "wrong-model"])
def test_missing_default_flash_configuration_blocks_without_local_fallback(boot, options):
    dispatch, api, events = boot(**options)
    state = dispatch("state", {})
    assert state["mode"] == "api"
    assert state["state"] != "ready"
    assert state["startup"]["can_send"] is False
    assert "provider_binding" in state["startup"]["blocked_reasons"]
    assert state["provider"]["selected_provider"] == API_ID
    assert state["provider"]["active"]["status"] != "active"
    assert all(event[1] is None for event in events if event[0] == "memory")
    dispatch("start", {})
    with pytest.raises(RuntimeError, match="not ready"):
        dispatch("send", {"text": "synthetic input must stay unsent"})
    assert api.sent == []


@pytest.mark.parametrize(
    ("component", "blocked_reason"),
    [
        ("role", "role_configuration"),
        ("semantic", "semantic_configuration"),
        ("memory", "memory_authority"),
    ],
)
def test_invalid_startup_configuration_blocks_send_and_is_safely_reported(
    boot, monkeypatch, tmp_path, component, blocked_reason,
):
    private_sentinel = "private-config-value-must-not-escape"
    if component == "role":
        monkeypatch.setattr(
            bootstrap,
            "load_character",
            lambda *_: (_ for _ in ()).throw(ValueError(private_sentinel)),
        )
    elif component == "semantic":
        monkeypatch.setattr(
            bootstrap,
            "load_semantic_retriever",
            lambda *_: (_ for _ in ()).throw(RuntimeError(private_sentinel)),
        )
    else:
        monkeypatch.setattr(
            bootstrap,
            "TextMemoryRuntime",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError(private_sentinel)),
        )

    dispatch, api, _ = boot()
    state = dispatch("state", {})
    assert state["state"] == "configuration_unavailable"
    startup = state["startup"]
    assert startup["status"] == "configuration_unavailable"
    assert startup["can_send"] is False
    assert blocked_reason in startup["blocked_reasons"]
    assert len(startup["identity_sha256"]) == 64
    assert startup["components"]["runtime_profile"]["status"] == "verified"
    assert private_sentinel not in json.dumps(startup, ensure_ascii=False)
    assert str(tmp_path) not in json.dumps(startup, ensure_ascii=False)
    assert private_sentinel not in state["error"]
    assert state["provider"]["active"]["status"] == "active"

    dispatch("start", {})
    assert dispatch("state", {})["state"] == "configuration_unavailable"
    with pytest.raises(RuntimeError, match="配置|记忆|语义"):
        dispatch("send", {"text": "synthetic blocked input"})
    with pytest.raises(RuntimeError, match="配置|记忆|语义"):
        dispatch("retry", {})
    assert dispatch("state", {})["queued_inputs"] == []
    assert api.sent == []


def test_invalid_configuration_cannot_be_recovered_by_starting_local_model(boot, monkeypatch):
    monkeypatch.setattr(
        bootstrap,
        "load_semantic_retriever",
        lambda *_: (_ for _ in ()).throw(ValueError("private semantic config")),
    )
    dispatch, api, _ = boot(local_default=True)
    assert dispatch("state", {})["state"] == "configuration_unavailable"
    dispatch("start", {})
    state = dispatch("state", {})
    assert state["state"] == "configuration_unavailable"
    assert state["startup"]["can_send"] is False
    with pytest.raises(RuntimeError, match="语义"):
        dispatch("send", {"text": "synthetic local blocked input"})
    assert api.sent == []


def test_startup_identity_binds_scope_and_request_selection(boot, monkeypatch, tmp_path):
    for variable in (
        "VC_TEXT_DIAGNOSTIC_REVIEW_AUDIT_VERSION",
        "VC_TEXT_DIAGNOSTIC_HISTORY_PROJECTION",
        "VC_TEXT_DIAGNOSTIC_READONLY_RECALL",
    ):
        monkeypatch.delenv(variable, raising=False)

    first_dispatch, _, _ = boot()
    first = first_dispatch("state", {})["startup"]

    second_root = tmp_path / "second-data-root"
    second_root.mkdir()
    (second_root / "provider.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("VC_DATA_ROOT", str(second_root))
    second_dispatch, _, _ = boot()
    second = second_dispatch("state", {})["startup"]

    first_scope = first["components"]["scope"]
    second_scope = second["components"]["scope"]
    assert first_scope["companion_id"] == second_scope["companion_id"]
    assert first_scope["data_root_sha256"] != second_scope["data_root_sha256"]
    assert first_scope["memory_path_sha256"] != second_scope["memory_path_sha256"]
    assert first["components"]["request_contract"] == second["components"]["request_contract"]
    assert first["identity_sha256"] != second["identity_sha256"]

    from virtual_companion.text.native_history import NATIVE_HISTORY_PROJECTION

    monkeypatch.setenv("VC_TEXT_DIAGNOSTIC_HISTORY_PROJECTION", NATIVE_HISTORY_PROJECTION)
    third_dispatch, _, _ = boot()
    third = third_dispatch("state", {})["startup"]
    assert third["components"]["scope"] == second_scope
    assert (third["components"]["request_contract"]["selection_sha256"]
            != second["components"]["request_contract"]["selection_sha256"])
    assert (third["components"]["request_contract"]["prompt_sha256"]
            != second["components"]["request_contract"]["prompt_sha256"])
    assert third["identity_sha256"] != second["identity_sha256"]


@pytest.mark.parametrize(
    "error_type",
    [ValueError, OSError, ImportError, RuntimeError, sqlite3.OperationalError],
    ids=["value", "os", "import", "runtime", "sqlite"],
)
def test_module_entry_emits_safe_configuration_failure_without_exception_text(
    monkeypatch, capsys, error_type,
):
    private_sentinel = "private-init-error-must-not-escape"

    def fail_profile(*_):
        raise error_type(private_sentinel)

    monkeypatch.setattr(bootstrap, "diagnostic_options", lambda _: {})
    monkeypatch.setattr(bootstrap, "load_profile", fail_profile)
    monkeypatch.setattr(bootstrap, "contain_current_process", lambda: True)
    monkeypatch.setattr(bootstrap.sys, "argv", ["text", "--fake"])
    monkeypatch.setattr(bootstrap.sys, "stdin", io.StringIO("x" * 64 + "\n"))
    with pytest.raises(SystemExit) as failure:
        bootstrap._run_module()
    assert failure.value.code != 0
    captured = capsys.readouterr()
    assert captured.out == '{"event": "startup_failed", "category": "configuration"}\n'
    assert captured.err == ""
    assert private_sentinel not in captured.out + captured.err


def test_module_entry_does_not_reclassify_errors_after_ready(monkeypatch, capsys):
    private_sentinel = "post-ready-runtime-error"

    def fail_after_ready():
        bootstrap._READY_HANDSHAKE_SENT = True
        raise RuntimeError(private_sentinel)

    monkeypatch.setattr(bootstrap, "main", fail_after_ready)
    with pytest.raises(RuntimeError, match=private_sentinel):
        bootstrap._run_module()
    assert capsys.readouterr().out == ""


def test_module_entry_does_not_classify_http_bind_failure_as_configuration(
    boot, capsys,
):
    private_sentinel = "synthetic-http-bind-failure"
    with pytest.raises(OSError, match=private_sentinel):
        boot(module_entry=True, server_error=OSError(private_sentinel))
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_module_entry_propagates_server_runtime_failure_after_ready(boot, capsys):
    private_sentinel = "synthetic-server-runtime-failure"
    with pytest.raises(OSError, match=private_sentinel):
        boot(module_entry=True, serve_error=OSError(private_sentinel))
    captured = capsys.readouterr()
    assert [json.loads(line)["event"] for line in captured.out.splitlines()] == ["ready"]
    assert private_sentinel not in captured.out + captured.err
    assert captured.err == ""


def test_direct_main_call_still_raises_initialization_error(monkeypatch, capsys):
    private_sentinel = "direct-main-test-error"

    def fail_profile(*_):
        raise ValueError(private_sentinel)

    monkeypatch.setattr(bootstrap, "diagnostic_options", lambda _: {})
    monkeypatch.setattr(bootstrap, "load_profile", fail_profile)
    monkeypatch.setattr(bootstrap, "contain_current_process", lambda: True)
    monkeypatch.setattr(bootstrap.sys, "argv", ["text", "--fake"])
    monkeypatch.setattr(bootstrap.sys, "stdin", io.StringIO("x" * 64 + "\n"))
    with pytest.raises(ValueError, match=private_sentinel):
        bootstrap.main()
    assert capsys.readouterr().out == ""


def test_unavailable_default_can_bind_the_same_route_after_configuration_is_fixed(
    boot, monkeypatch,
):
    dispatch, api, _ = boot(invalid=True)
    assert dispatch("state", {})["state"] == "provider_unavailable"
    monkeypatch.setattr(bootstrap, "load_provider_model", lambda *args, **kwargs: api)
    result = dispatch("selectProvider", {"provider_id": API_ID, "confirmed": True})
    assert result["status"] == "selected"
    state = dispatch("state", {})
    assert state["state"] == "ready"
    assert state["mode"] == "api"
    assert state["error"] == ""
    assert state["provider"]["active"]["config_revision"] == api.config_revision
    assert api.sent == []


def test_unavailable_default_allows_an_explicit_local_selection_and_reopens_as_flash(boot):
    dispatch, api, _ = boot(configured=False)
    result = dispatch("selectProvider", {"provider_id": "local"})
    assert result["status"] == "selected"
    assert dispatch("state", {})["mode"] == "local"
    reopened, next_api, _ = boot(configured=False)
    assert reopened("state", {})["provider"]["selected_provider"] == API_ID
    assert reopened("state", {})["state"] == "provider_unavailable"
    assert api.sent == next_api.sent == []


def test_default_flash_foreground_review_and_background_share_one_route(boot):
    dispatch, api, events = boot(workers=True)
    assert_flash_authorities(events, api)
    state = dispatch("state", {})
    dispatch("send", {"text": "请给我一个合成雨天安排。"})
    event = wait_for(lambda: dispatch("streamPoll", {})["event"])
    dispatch("streamAck", {"sequence": event["sequence"],
                           "fingerprint": event["fingerprint"],
                           "host_receipt": "synthetic-default-flash-dom"})
    wait_for(lambda: not dispatch("state", {})["busy"])
    finished = dispatch("state", {})
    assert finished["attempt"]["status"] == "completed"
    assert finished["attempt"]["binding"]["version_binding"]["schema_version"] == 2
    assert finished["provider"] == state["provider"]
    # The real same-store worker consumes the input and confirmed assistant source.
    wait_for(lambda: any(stage == "fact_extract" for stage in api.sent))
    assert "demand_control" in api.sent
    assert "stream_unit_review" in api.sent
    assert dispatch("history", {})["messages"][-1]["content"] == "这是合成答案。"
    dispatch("stop", {})


def test_fake_bootstrap_stays_cpu_and_never_loads_provider_configuration(boot):
    dispatch, api, events = boot(fake=True)
    assert not any(event[0] == "load" for event in events)
    assert dispatch("state", {})["mode"] == "fake"
    memory = next(event for event in events if event[0] == "memory")
    session = next(event for event in events if event[0] == "session")
    assert isinstance(memory[1], FakeModel)
    assert memory[2] is True
    assert session[2]["version_binding"]["schema_version"] == 1
    assert api.sent == []


def test_fake_mode_refuses_manual_api_selection_even_with_managed_configuration(boot, monkeypatch):
    dispatch, api, events = boot(fake=True, configured=True)
    loads = []

    def forbidden(*args, **kwargs):
        loads.append(args)
        raise AssertionError("Fake mode cannot load the API route")

    monkeypatch.setattr(bootstrap, "load_provider_model", forbidden)
    result = dispatch("selectProvider", {"provider_id": API_ID, "confirmed": True})
    assert result["status"] == "unavailable"
    state = dispatch("state", {})
    assert state["mode"] == "fake"
    assert state["provider"]["selected_provider"] == "local"
    assert state["provider"]["options"][1]["selectable"] is False
    assert not any(event[0] == "load" for event in events)
    assert loads == []
    assert api.sent == []
