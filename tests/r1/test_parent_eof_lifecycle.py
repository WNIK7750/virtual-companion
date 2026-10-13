"""In-process parent EOF ordering; real Windows child/Job checks stay separate."""

import io
from types import SimpleNamespace

import pytest
from virtual_companion.text import __main__ as bootstrap
from virtual_companion.text.fake import FakeModel


class ParentExit(BaseException):
    def __init__(self, code):
        self.code = code


@pytest.fixture
def launch(monkeypatch, tmp_path):
    events, watchers, resources = [], [], []
    memory_class = bootstrap.TextMemoryRuntime
    store_class = bootstrap.SQLiteConversationStore
    runtime_class = bootstrap.DefaultTextRuntime
    thread_class = bootstrap.threading.Thread

    class Model(FakeModel):
        def cancel(self):
            events.append("cancel_model")

    def memory(*args, **kwargs):
        value = memory_class(*args, **kwargs)
        close = value.close

        def tracked_close():
            events.append("close_memory")
            close()

        value.close = tracked_close
        resources.append(value)
        return value

    def store(*args, **kwargs):
        value = store_class(*args, **kwargs)
        resources.append(value)
        return value

    def run(*, semantic_error=None, eof_at=None):
        def semantic(_):
            events.append("semantic_enter")
            if watchers:
                events.append("reader_during_semantic")
            events.append("semantic_finished")
            if semantic_error:
                raise semantic_error("Synthetic optional configuration failure")
            return None

        def thread(*args, **kwargs):
            if getattr(kwargs.get("target"), "__name__", "") != "watch_parent":
                return thread_class(*args, **kwargs)
            target = kwargs["target"]

            def start():
                events.append("watch_parent_started")
                watchers.append(target)
                if eof_at == "watch_start":
                    target()

            return SimpleNamespace(start=start)

        def runtime(*args, **kwargs):
            if eof_at == "before_session":
                watchers[0]()
            return runtime_class(*args, **kwargs)

        monkeypatch.setattr(bootstrap, "load_semantic_retriever", semantic)
        monkeypatch.setattr(bootstrap.threading, "Thread", thread)
        monkeypatch.setattr(bootstrap, "DefaultTextRuntime", runtime)
        bootstrap.main()

    def exit_process(code):
        raise ParentExit(code)

    monkeypatch.setenv("VC_DATA_ROOT", str(tmp_path / "synthetic"))
    for key in (
        "VC_MEMORY_MANAGEMENT_DB", "VC_MEMORY_COMPANION_ID",
        "VC_TEXT_DIAGNOSTIC_PROMPT_REVISION", "VC_TEXT_DIAGNOSTIC_READONLY_RECALL",
        "VC_TEXT_DIAGNOSTIC_REVIEW_AUDIT_VERSION", "VC_TEXT_DIAGNOSTIC_HISTORY_PROJECTION",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(bootstrap.sys, "stdin", io.StringIO("x" * 64 + "\n"))
    monkeypatch.setattr(bootstrap.sys, "argv", ["text", "--fake"])
    monkeypatch.setattr(bootstrap, "contain_current_process", lambda: True)
    monkeypatch.setattr(bootstrap.os, "_exit", exit_process)
    monkeypatch.setattr(bootstrap, "FakeModel", Model)
    monkeypatch.setattr(bootstrap, "TextMemoryRuntime", memory)
    monkeypatch.setattr(memory_class, "attach_foreground", lambda *_: None)
    monkeypatch.setattr(bootstrap, "SQLiteConversationStore", store)
    monkeypatch.setattr(bootstrap, "create_server", lambda *_: SimpleNamespace(
        serve_forever=lambda **_: events.append("ready_server")))
    yield run, events, watchers
    for resource in reversed(resources):
        resource.close()


@pytest.mark.parametrize("semantic_error", [None, ValueError, ImportError, RuntimeError])
def test_optional_semantic_initialization_finishes_before_parent_reader_starts(
    launch, semantic_error,
):
    run, events, watchers = launch
    run(semantic_error=semantic_error)
    assert events[:3] == ["semantic_enter", "semantic_finished", "watch_parent_started"]
    assert len(watchers) == 1
    assert "ready_server" in events


def test_eof_before_resources_exist_exits_without_unbound_cleanup(launch, capsys):
    run, events, _ = launch
    with pytest.raises(ParentExit) as exited:
        run(eof_at="watch_start")
    assert exited.value.code == 0
    assert events == ["semantic_enter", "semantic_finished", "watch_parent_started"]
    assert "shutdown_incomplete" not in capsys.readouterr().err


def test_eof_before_session_exists_cleans_constructed_model_and_memory(launch, capsys):
    run, events, _ = launch
    with pytest.raises(ParentExit) as exited:
        run(eof_at="before_session")
    assert exited.value.code == 0
    assert events[-2:] == ["cancel_model", "close_memory"]
    assert "ready_server" not in events
    assert "shutdown_incomplete" not in capsys.readouterr().err
