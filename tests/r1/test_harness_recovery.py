"""Execution recovery uses synthetic effects and fresh owner-checked journals."""

import copy

import pytest
from test_harness_journal import setup_store
from test_harness_port import Journal, Model, execute, response, tool_call
from test_rebuilt_turns import A, config
from virtual_companion.boundary.contracts import VersionConflict
from virtual_companion.boundary.sqlite_harness import harness_binding
from virtual_companion.cognition import harness


def crash(_):
    raise KeyboardInterrupt("synthetic lost receipt")


def interrupted(journal=None, **kwargs):
    journal = journal or Journal()
    with pytest.raises(KeyboardInterrupt):
        execute(Model([response("", [tool_call("original", key="fixture")])]),
                journal, dispatch=crash, **kwargs)
    return journal


def test_message_only_restart_cannot_drop_unknown_effect():
    journal = interrupted()
    saved = journal.entries[-1]
    recovered = harness.recover_transcript(saved, binding=saved["binding"])
    sent = []
    with pytest.raises(ValueError, match="execution restore"):
        execute(Model([response("", [tool_call("changed", key="fixture")]), response()]),
                Journal(), messages=recovered["messages"], dispatch=lambda call: sent.append(call))
    assert not sent


def test_typed_restore_retains_unknown_rounds_and_epoch():
    journal = interrupted()
    saved = journal.entries[-1]
    restored = harness.HarnessRestore(saved)
    output = Journal()
    sent = []
    result = execute(
        Model([response("", [tool_call("changed", key="fixture")]), response()]), output,
        restore=restored, dispatch=lambda call: sent.append(call))
    assert not sent
    assert result["uncertain"]
    assert output.entries[-1]["rounds"] == 3
    assert output.entries[-1]["cache_epoch"] == saved["cache_epoch"] + 1


def test_completed_execution_returns_original_without_model_or_effect():
    journal = Journal()
    original = execute(Model([response("", [tool_call()]), response("original")]), journal)
    model = Model([])
    restored = execute(model, Journal(), restore=harness.HarnessRestore(journal.entries[-1]))
    assert restored == original
    assert not model.requests


def test_success_does_not_swallow_a_new_intent_with_same_arguments():
    sent = []
    execute(Model([response("", [tool_call("one")]), response("", [tool_call("two")]),
                   response()]), Journal(),
            dispatch=lambda call: sent.append(call) or {"outcome": "success", "content": "ok"})
    assert [call["id"] for call in sent] == ["one", "two"]


def test_restore_round_limit_does_not_reset():
    saved = interrupted().entries[-1]
    model = Model([])
    with pytest.raises(ValueError, match="round limit"):
        execute(model, Journal(), restore=harness.HarnessRestore(saved),
                limits=harness.HarnessLimits(1, 16000, 256))
    assert not model.requests


def test_restore_rejects_binding_and_changed_tool_contract():
    saved = interrupted().entries[-1]
    foreign = copy.deepcopy(saved)
    foreign["binding"]["input"] = "other"
    with pytest.raises(ValueError, match="binding"):
        execute(Model([]), Journal(), restore=harness.HarnessRestore(foreign))
    with pytest.raises(ValueError, match="tool contract"):
        execute(Model([]), Journal(), restore=harness.HarnessRestore(saved), tools=[])


def test_explicit_retry_blocks_pending_effect_but_text_retry_remains_usable(tmp_path):
    store, item, run = setup_store(tmp_path)
    interrupted(store.harness_journal(run, "tool", clock=lambda: 101),
                binding=harness_binding(run))
    store.fail(run, reason="synthetic_failure", now_us=102)
    with pytest.raises(VersionConflict, match="unresolved Harness"):
        store.retry(A, item.input_id, run.attempt_id, "retry", config())
    assert store.inspect(A, item.input_id)["status"] == "failed"


def test_text_retry_does_not_inherit_exhausted_single_round(tmp_path):
    store, item, run = setup_store(tmp_path)
    execute(Model([response()]), store.harness_journal(run, "text", clock=lambda: 101),
            binding=harness_binding(run), tools=[], limits=harness.HarnessLimits(1, 16000, 256))
    store.fail(run, reason="synthetic_failure", now_us=102)
    store.retry(A, item.input_id, run.attempt_id, "retry", config())
    new = store.claim(A, item.input_id, "new", now_us=103, lease_us=50)
    result = execute(Model([response("new")]),
                     store.harness_journal(new, "text", clock=lambda: 104),
                     binding=harness_binding(new), tools=[],
                     limits=harness.HarnessLimits(1, 16000, 256))
    assert result["text"] == "new"


def test_new_epoch_automatically_restores_pending_without_replaying(tmp_path):
    store, item, old = setup_store(tmp_path)
    interrupted(store.harness_journal(old, "tool", clock=lambda: 101),
                binding=harness_binding(old))
    new = store.claim(A, item.input_id, "new", now_us=150, lease_us=50)
    sent = []
    result = execute(
        Model([response("", [tool_call("changed", key="fixture")]), response()]),
        store.harness_journal(new, "tool", clock=lambda: 151),
        binding=harness_binding(new), dispatch=lambda call: sent.append(call))
    assert not sent
    assert result["uncertain"]


def test_new_subrun_name_cannot_erase_pending_effect(tmp_path):
    store, _, run = setup_store(tmp_path)
    interrupted(store.harness_journal(run, "tool", clock=lambda: 101),
                binding=harness_binding(run))
    model = Model([])
    with pytest.raises(VersionConflict, match="unresolved Harness"):
        execute(model, store.harness_journal(run, "renamed", clock=lambda: 102),
                binding=harness_binding(run))
    assert not model.requests


def test_cancelled_owner_cannot_restore_completed_candidate(tmp_path):
    store, item, run = setup_store(tmp_path)
    journal = store.harness_journal(run, "tool", clock=lambda: 101)
    execute(Model([response("", [tool_call()]), response()]), journal,
            binding=harness_binding(run))
    store.cancel(A, item.input_id)
    with pytest.raises(VersionConflict):
        execute(Model([]), journal, binding=harness_binding(run))


def test_idempotent_declaration_does_not_authorize_unknown_replay():
    tools = [{"name": "lookup", "parameters": {"type": "object"}, "effect": "idempotent"}]
    saved = interrupted(tools=tools).entries[-1]
    sent = []
    execute(Model([response("", [tool_call("new", key="fixture")]), response()]), Journal(),
            tools=tools, restore=harness.HarnessRestore(saved),
            dispatch=lambda call: sent.append(call))
    assert not sent
    tools[0]["effect"] = "non_idempotent"
    with pytest.raises(ValueError, match="tool contract"):
        execute(Model([]), Journal(), tools=tools, restore=harness.HarnessRestore(saved))


def test_confirmed_not_executed_call_may_be_requested_after_restore():
    journal = Journal()
    # The second call was never dispatched after the first returned failure.
    execute(Model([response("", [tool_call("failed"), tool_call("unstarted", key="later")]),
                   response()]), journal,
            dispatch=lambda _: {"outcome": "failure", "content": "not available"})
    saved = journal.entries[-1]
    saved["status"] = "interrupted"
    sent = []
    execute(Model([response("", [tool_call("new", key="later")]), response()]), Journal(),
            restore=harness.HarnessRestore(saved), limits=harness.HarnessLimits(5, 16000, 256),
            dispatch=lambda call: sent.append(call) or {"outcome": "success", "content": "ok"})
    assert len(sent) == 1


def test_completed_effect_survives_new_attempt_without_redispatch(tmp_path):
    store, item, run = setup_store(tmp_path)
    journal = store.harness_journal(run, "tool", clock=lambda: 101)
    original = execute(Model([response("", [tool_call()]), response("original")]), journal,
                       binding=harness_binding(run))
    store.fail(run, reason="synthetic_failure", now_us=102)
    store.retry(A, item.input_id, run.attempt_id, "retry", config())
    new = store.claim(A, item.input_id, "new", now_us=103, lease_us=50)
    fresh = store.harness_journal(new, "tool", clock=lambda: 104)
    restored = execute(Model([]), fresh,
                       binding=harness_binding(new))
    assert restored == original
    assert fresh.load()["operation_origin"] == harness_binding(run)
    assert fresh.load()["binding"] == harness_binding(new)


def test_new_same_argument_intent_after_recovery_is_not_a_success_cache_hit():
    journal = Journal()
    with pytest.raises(StopIteration):
        execute(Model([response("", [tool_call("old")])]), journal)
    sent = []
    execute(Model([response("", [tool_call("new")]), response()]), Journal(),
            restore=harness.HarnessRestore(journal.entries[-1]),
            dispatch=lambda call: sent.append(call) or {"outcome": "success", "content": "new"})
    assert [call["id"] for call in sent] == ["new"]


@pytest.mark.parametrize("omitted", ["pending", "uncertain", "messages"])
def test_partial_latest_checkpoint_cannot_erase_retry_fence(tmp_path, omitted):
    store, item, run = setup_store(tmp_path)
    journal = store.harness_journal(run, "tool", clock=lambda: 101)
    interrupted(journal, binding=harness_binding(run))
    partial = journal.load()
    del partial[omitted]
    journal.save(partial)
    store.fail(run, reason="synthetic_failure", now_us=102)
    with pytest.raises(ValueError, match="Incomplete Harness"):
        store.retry(A, item.input_id, run.attempt_id, "retry", config())


def test_unknown_tool_result_also_fences_retry_if_uncertain_list_is_missing_data(tmp_path):
    store, item, run = setup_store(tmp_path)
    journal = store.harness_journal(run, "tool", clock=lambda: 101)
    execute(Model([response("", [tool_call()]), response()]), journal,
            binding=harness_binding(run),
            dispatch=lambda _: {"outcome": "unknown", "content": "lost receipt"})
    partial = journal.load()
    partial["uncertain"] = []
    journal.save(partial)
    store.fail(run, reason="synthetic_failure", now_us=102)
    with pytest.raises(VersionConflict, match="unresolved Harness"):
        store.retry(A, item.input_id, run.attempt_id, "retry", config())


def test_restore_rejects_changed_initial_request():
    saved = interrupted().entries[-1]
    with pytest.raises(ValueError, match="input transcript"):
        execute(Model([]), Journal(), restore=harness.HarnessRestore(saved),
                messages=[{"role": "user", "content": "different logical operation"}])


@pytest.mark.parametrize("new_epoch", [False, True])
def test_same_text_attempt_preserves_candidate_after_reopen(tmp_path, new_epoch):
    store, item, run = setup_store(tmp_path)
    original = execute(Model([response("original text")]),
                       store.harness_journal(run, "text", clock=lambda: 101),
                       binding=harness_binding(run), tools=[],
                       limits=harness.HarnessLimits(1, 16000, 256))
    if new_epoch:
        run = store.claim(A, item.input_id, "replacement", now_us=150, lease_us=50)
    fresh = store.harness_journal(run, "text", clock=lambda: 151 if new_epoch else 102)
    model = Model([response("should not generate again")])
    recovered = execute(model, fresh, binding=harness_binding(run), tools=[],
                        limits=harness.HarnessLimits(1, 16000, 256))
    assert recovered == original
    assert model.requests == []


def test_same_text_attempt_keeps_prepared_round_after_crash(tmp_path):
    store, _, run = setup_store(tmp_path)
    journal = store.harness_journal(run, "text", clock=lambda: 101)
    with pytest.raises(StopIteration):
        execute(Model([]), journal, binding=harness_binding(run), tools=[],
                limits=harness.HarnessLimits(1, 16000, 256))
    model = Model([response("should not reset spent round")])
    with pytest.raises(ValueError, match="round limit"):
        execute(model, store.harness_journal(run, "text", clock=lambda: 102),
                binding=harness_binding(run), tools=[],
                limits=harness.HarnessLimits(1, 16000, 256))
    assert model.requests == []
