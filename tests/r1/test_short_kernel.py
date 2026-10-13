from dataclasses import replace
from threading import Event

import pytest
from test_rebuilt_turns import A, claim, config
from test_short_dialogue import Model
from virtual_companion.boundary.contracts import VersionConflict
from virtual_companion.boundary.runtime import CompanionRuntime
from virtual_companion.boundary.sqlite_background_worker import SQLiteBackgroundWorkerStore
from virtual_companion.boundary.sqlite_dialogue import SQLiteDialogueStore
from virtual_companion.boundary.sqlite_timeline import SQLiteTimelineStore
from virtual_companion.cognition.preparation import CallCancelled
from virtual_companion.cognition.short_dialogue import ShortRequest
from virtual_companion.cognition.short_kernel import ShortKernel


def setup(tmp_path):
    store = SQLiteDialogueStore(tmp_path / "short.sqlite3")
    item = store.accept(config(), "send", "先做什么？")
    run = claim(store, item)
    model = Model()
    model.cancel = lambda: None
    kernel = ShortKernel(store=store, model_factory=lambda cancelled: model,
                         admit=lambda run, state, text: ShortRequest(text),
                         clock=lambda: 100, lease_us=50, deadline_us=1000,
                         renewal_margin_us=10)
    return store, item, run, model, kernel


def test_one_call_commits_reviewed_text_without_state_or_unshown_memory(tmp_path):
    store, item, run, model, kernel = setup(tmp_path)
    result = kernel.generate(run, store.snapshot(A), "先做什么？")
    assert result.presentation.review.approval_source == "host_policy"
    assert result.presentation.review.model_approved is False
    receipt = store.commit(result, now_us=100)
    assert receipt["status"] == "committed" and len(model.sent) == 1
    reopened = SQLiteDialogueStore(store.path)
    assert reopened.inspect(A, item.input_id)["delivery"]["displayed_end"] == 0
    assert reopened.recall(A, now_us=100) == []
    assert reopened.snapshot(A).values_json == "{}"


def test_old_reader_refuses_new_policy_format_and_no_implicit_upgrade(tmp_path):
    store, _, _, _, _ = setup(tmp_path)
    with pytest.raises(ValueError):
        SQLiteTimelineStore(store.path)
    old = SQLiteTimelineStore(tmp_path / "old.sqlite3")
    with pytest.raises(ValueError):
        SQLiteDialogueStore(old.path)


def test_cancel_during_model_does_not_commit_or_release_candidate(tmp_path):
    store, item, run, model, kernel = setup(tmp_path)
    original = model.invoke

    def invoke(value):
        store.cancel(A, item.input_id)
        return original(value)

    model.invoke = invoke
    with pytest.raises(CallCancelled):
        kernel.generate(run, store.snapshot(A), "先做什么？")
    assert store.inspect(A, item.input_id)["commit"] is None


def test_new_user_source_fences_old_short_candidate_before_commit(tmp_path):
    store = SQLiteBackgroundWorkerStore(tmp_path / "short-background.sqlite3")
    item = store.accept(config(), "first", "我喜欢茶")
    run = claim(store, item)
    model = Model()
    model.cancel = lambda: None
    original = model.invoke

    def invoke(value):
        store.accept(config(), "second", "更正，我不喜欢茶")
        return original(value)

    model.invoke = invoke
    kernel = ShortKernel(store=store, model_factory=lambda cancelled: model,
                         admit=lambda run, state, text: ShortRequest(text),
                         clock=lambda: 100, lease_us=50, deadline_us=1000,
                         renewal_margin_us=10)
    with pytest.raises(VersionConflict, match="context"):
        kernel.generate(run, store.snapshot(A), "我喜欢茶")
    assert store.inspect(A, item.input_id)["delivery"] is None


def test_admission_cannot_replace_current_input_and_stale_basis_rejected(tmp_path):
    store, _, run, model, kernel = setup(tmp_path)
    kernel.admit = lambda run, state, text: ShortRequest("伪造用户输入")
    with pytest.raises(ValueError):
        kernel.generate(run, store.snapshot(A), "先做什么？")
    assert model.sent == []
    with pytest.raises(ValueError):
        kernel.generate(run, replace(store.snapshot(A), write_version=999), "先做什么？")


@pytest.mark.parametrize("arrival", ["before_admission", "during_admission", "during_count"])
def test_pending_correction_does_not_send_previously_admitted_context(tmp_path, arrival):
    store = SQLiteBackgroundWorkerStore(tmp_path / "admission-barrier.sqlite3")
    item = store.accept(config(), "first", "我喜欢茶")
    run = claim(store, item)
    basis = store.snapshot(A)
    model = Model()
    model.cancel = lambda: None

    def correction():
        store.accept(config(), "second", "更正，我不喜欢茶")

    if arrival == "during_count":
        original_count = model.count

        def count(request):
            correction()
            return original_count(request)

        model.count = count

    def admit(run, state, text):
        if arrival == "during_admission":
            correction()
        return ShortRequest(text)

    if arrival == "before_admission":
        correction()
    kernel = ShortKernel(store=store, model_factory=lambda cancelled: model,
                         admit=admit, clock=lambda: 100, lease_us=50, deadline_us=1000,
                         renewal_margin_us=10)
    with pytest.raises(VersionConflict, match="context"):
        kernel.generate(run, basis, "我喜欢茶")
    assert model.sent == []
    assert len(model.counted) == int(arrival == "during_count")
    assert store.inspect(A, item.input_id)["delivery"] is None


def test_pre_cancelled_short_path_has_no_model_io():
    from virtual_companion.cognition.short_dialogue import ShortDialogue

    cancelled = Event()
    cancelled.set()
    model = Model()
    with pytest.raises(InterruptedError):
        ShortDialogue(model, cancelled=cancelled).generate(ShortRequest("你好"))
    assert model.counted == model.sent == []


def test_invalid_output_is_durable_failure_without_delivery(tmp_path):
    store, item, _, model, kernel = setup(tmp_path)
    store.cancel(A, item.input_id)
    other = store.accept(config(), "second-send", "先做什么？")
    model.finish = "length"
    receipt = CompanionRuntime(config(), store).execute(
        other.input_id, "worker", kernel, clock=lambda: 100, lease_us=50,
    )
    assert receipt["status"] == "failed"
    assert store.inspect(A, other.input_id)["delivery"] is None


@pytest.mark.parametrize("action", ["silent", "defer"])
def test_explicit_terminal_survives_commit_as_distinct_intent(tmp_path, action):
    store, item, run, model, kernel = setup(tmp_path)
    kernel.admit = lambda claim, state, text: ShortRequest(text, action=action)
    result = kernel.generate(run, store.snapshot(A), "先做什么？")
    assert result.presentation.candidate.intent == action
    receipt = store.commit(result, now_us=101)
    assert receipt["presentation"]["candidate"]["intent"] == action
    delivery = store.inspect(A, item.input_id)["delivery"]
    assert delivery["status"] == "completed" and delivery["displayed_end"] == 0
    assert store.publish(A, receipt["delivery_id"]) is None
    assert model.sent == []
