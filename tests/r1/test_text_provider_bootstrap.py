import hashlib
import io
import threading
import time
from types import SimpleNamespace

import pytest
from virtual_companion.text import __main__ as bootstrap
from virtual_companion.text.fake import FakeModel


@pytest.fixture
def runtime_observer(monkeypatch):
    factory, runtimes = bootstrap.DefaultTextRuntime, []

    def create(*args, **kwargs):
        runtime = factory(*args, **kwargs)
        runtime._bootstrap_binding = kwargs.get("binding")
        runtimes.append(runtime)
        return runtime

    monkeypatch.setattr(bootstrap, "DefaultTextRuntime", create)
    return runtimes


@pytest.fixture
def application(monkeypatch, tmp_path, runtime_observer, request):
    captured, memories = {}, []
    memory_class = bootstrap.TextMemoryRuntime
    thread_class = threading.Thread
    find_spec = bootstrap.importlib.util.find_spec
    load_catalog = bootstrap.ProviderSelection._load

    def memory(*args, **kwargs):
        # Route lifecycle is real; only the memory model protocol is CPU synthetic.
        kwargs["fake"] = True
        value = memory_class(*args, **kwargs)
        memories.append(value)
        return value

    def thread(*args, **kwargs):
        if getattr(kwargs.get("target"), "__name__", "") == "watch_parent":
            return type("NoWatcher", (), {"start": lambda self: None})()
        return thread_class(*args, **kwargs)

    class Server:
        def serve_forever(self, **kwargs):
            pass

    def server(token, dispatch):
        captured["dispatch"] = dispatch
        return Server()

    class API(FakeModel):
        config_revision = 2
        config_fingerprint = "b" * 64
        route = type(
            "Route",
            (),
            {
                "fingerprint": "a" * 64,
                "model": "deepseek-flash",
                "wire_profile": "deepseek-flash.v1",
            },
        )()

    def local_catalog(path):
        value = load_catalog(path)
        value["default_provider"] = "local"
        return value

    def local_model(*args):
        value = bootstrap.FakeModel()
        value.process = SimpleNamespace(poll=lambda: None, pid=None)
        value.start = value.close = lambda: None
        return value

    monkeypatch.setenv("VC_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("VC_MEMORY_MANAGEMENT_DB", raising=False)
    monkeypatch.delenv("VC_MEMORY_COMPANION_ID", raising=False)
    monkeypatch.delenv("VC_SEARXNG_ENDPOINT", raising=False)
    monkeypatch.delenv("VC_SEARXNG_VERSION", raising=False)
    monkeypatch.delenv("VC_PUBLIC_WEB_BACKEND", raising=False)
    invalid_search = getattr(request, "param", None) == "invalid-search"
    if invalid_search or getattr(request, "param", None) == "searxng":
        monkeypatch.setenv("VC_PUBLIC_WEB_BACKEND", "searxng")
    if invalid_search:
        monkeypatch.setenv("VC_SEARXNG_ENDPOINT", "http://remote.example/search")
    (tmp_path / "provider.json").write_text("{}")
    monkeypatch.setattr(bootstrap, "TextMemoryRuntime", memory)
    monkeypatch.setattr(bootstrap.threading, "Thread", thread)
    monkeypatch.setattr(bootstrap, "contain_current_process", lambda: True)
    monkeypatch.setattr(bootstrap, "create_server", server)
    monkeypatch.setattr(bootstrap, "load_provider_model", lambda *a, **k: API())
    monkeypatch.setattr(bootstrap, "LlamaModel", local_model)
    monkeypatch.setattr(bootstrap.ProviderSelection, "_load", staticmethod(local_catalog))
    monkeypatch.setattr(
        bootstrap.importlib.util,
        "find_spec",
        lambda name: True if name == "tokenizers" else find_spec(name),
    )
    monkeypatch.setattr(bootstrap.sys, "argv", ["text"])
    monkeypatch.setattr(bootstrap.sys, "stdin", io.StringIO("x" * 64 + "\n"))
    bootstrap.main()
    if invalid_search:
        captured["dispatch"]("selectProvider", {"provider_id": "deepseek-api", "confirmed": True})
    else:
        captured["dispatch"]("start", {})
    deadline = time.monotonic() + 5
    while captured["dispatch"]("state", {})["state"] != "ready" and time.monotonic() < deadline:
        time.sleep(0.01)
    assert captured["dispatch"]("state", {})["state"] == "ready"
    yield captured["dispatch"], memories
    for value in memories:
        value.close()


@pytest.mark.parametrize("application", ["searxng"], indirect=True)
def test_selected_self_hosted_search_binds_without_search_credentials(
    application, runtime_observer,
):
    dispatch, _ = application
    public = runtime_observer[0].public_query
    assert "searxng" in public.binding.configuration.tool_version.lower()
    assert public.binding.configuration.credential_version == "no-service-key"
    assert public.supports_extract
    assert dispatch("state", {})["startup"]["components"]["public_web"]["status"] == "configured"


def test_normal_startup_keeps_existing_route_until_live_search_quality_passes(
    application, runtime_observer,
):
    assert runtime_observer[0].public_query.binding.configuration.route_id == "pi-keyless-web"


@pytest.mark.parametrize("application", ["invalid-search"], indirect=True)
def test_invalid_search_configuration_preserves_queued_input_after_provider_switch(
    application, runtime_observer, monkeypatch,
):
    dispatch, _ = application
    runtime = runtime_observer[-1]
    state = dispatch("state", {})
    assert not state["startup"]["can_send"]
    assert "public_web_configuration" in state["startup"]["blocked_reasons"]
    frozen = runtime.config.attempt_binding({
        "runtime_profile_sha256": bootstrap.PROFILE_SHA256,
        "version_binding": runtime._bootstrap_binding,
    })
    frozen["request_control_sha256"] = hashlib.sha256(runtime.control.encode()).hexdigest()
    queued = runtime.mirror.enqueue_input("已收到但不能在无效配置下执行", frozen)

    def unexpected_start(*args, **kwargs):
        raise AssertionError("Invalid startup configuration drained the queue")

    monkeypatch.setattr(bootstrap.TextSession, "start_queued", unexpected_start)
    dispatch("resumeInputs", {})
    assert runtime.mirror.queued_inputs() == [queued]
    assert runtime.mirror.latest_attempt() is None


def test_idle_route_switch_rebuilds_same_authority_and_returns_local(application, runtime_observer):
    dispatch, memories = application
    public = runtime_observer[0].public_query
    assert public is not None and public.supports_extract
    before = dispatch("state", {})
    assert dispatch("selectProvider", {"provider_id": "deepseek-api"})["status"] == "unavailable"
    assert len(memories) == 1
    result = dispatch("selectProvider", {"provider_id": "deepseek-api", "confirmed": True})
    assert result["status"] == "selected"
    assert dispatch("state", {})["mode"] == "api"
    assert memories[0]._thread.is_alive() is False
    assert memories[0].path == memories[1].path
    assert dispatch("state", {})["session_id"] == before["session_id"]
    assert dispatch("selectProvider", {"provider_id": "local"})["status"] == "selected"
    assert dispatch("state", {})["mode"] == "local"
    assert len(memories) == 3
    assert all(runtime.public_query is public for runtime in runtime_observer)


def test_active_or_unacknowledged_reply_refuses_switch(application):
    dispatch, memories = application
    dispatch("send", {"text": "synthetic"})
    result = dispatch("selectProvider", {"provider_id": "deepseek-api", "confirmed": True})
    assert result["status"] == "unavailable"
    dispatch("stop", {})
    deadline = time.monotonic() + 5
    while dispatch("state", {})["busy"] and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(memories) == 1


def test_missing_configuration_keeps_worker_and_route(application, monkeypatch):
    dispatch, memories = application

    def missing(*args, **kwargs):
        raise ValueError("Managed provider credential missing")

    monkeypatch.setattr(bootstrap, "load_provider_model", missing)
    result = dispatch("selectProvider", {"provider_id": "deepseek-api", "confirmed": True})
    assert result["status"] == "unavailable"
    assert dispatch("state", {})["mode"] == "local"
    assert len(memories) == 1
    assert memories[0]._thread.is_alive()


def test_unknown_delivery_after_stop_blocks_provider_switch(application):
    dispatch, memories = application
    dispatch("send", {"text": "synthetic"})
    event = None
    deadline = time.monotonic() + 8
    while not event and time.monotonic() < deadline:
        event = dispatch("streamPoll", {})["event"]
        time.sleep(0.01)
    assert event
    dispatch("stop", {})
    deadline = time.monotonic() + 5
    while dispatch("state", {})["busy"] and time.monotonic() < deadline:
        time.sleep(0.01)
    result = dispatch("selectProvider", {"provider_id": "deepseek-api", "confirmed": True})
    assert result["status"] == "unavailable"
    assert len(memories) == 1


def test_failed_worker_close_requires_explicit_recovery_without_second_worker(application):
    dispatch, memories = application
    original = memories[0].close

    def delayed_close():
        memories[0]._stopped.set()
        memories[0].generation_gate.close()
        raise RuntimeError("worker has not exited")

    memories[0].close = delayed_close
    result = dispatch("selectProvider", {"provider_id": "deepseek-api", "confirmed": True})
    assert result["status"] == "unavailable"
    assert result["provider"]["recovery_required"] is True
    assert dispatch("state", {})["state"] == "provider_recovery_required"
    assert len(memories) == 1
    with pytest.raises(RuntimeError, match="not ready"):
        dispatch("send", {"text": "must not send"})
    memories[0].close = original
    result = dispatch("selectProvider", {"provider_id": "local"})
    assert result["status"] == "selected"
    assert result["provider"]["recovery_required"] is False
    assert dispatch("state", {})["state"] == "ready"
    assert len(memories) == 2


def test_rebuild_failure_restores_original_worker_without_switch(application, monkeypatch):
    dispatch, memories = application
    factory = bootstrap.TextMemoryRuntime
    calls = 0

    def fail_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("synthetic construction failure")
        return factory(*args, **kwargs)

    monkeypatch.setattr(bootstrap, "TextMemoryRuntime", fail_once)
    result = dispatch("selectProvider", {"provider_id": "deepseek-api", "confirmed": True})
    assert result["status"] == "unavailable"
    assert result["provider"]["selected_provider"] == "local"
    assert result["provider"]["recovery_required"] is False
    assert memories[0]._thread.is_alive() is False
    assert memories[1]._thread.is_alive()
    assert dispatch("state", {})["state"] == "ready"
