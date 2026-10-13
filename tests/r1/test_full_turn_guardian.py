import threading

import pytest
from test_cognition_preparation import basis
from test_cognitive_kernel import RecordedModel, kernel
from test_rebuilt_turns import A
from virtual_companion.boundary.contracts import VersionConflict
from virtual_companion.cognition.kernel import SourceSelection
from virtual_companion.cognition.preparation import CallCancelled


def test_keeper_renews_during_non_model_context_selection(tmp_path):
    store, _, run, persona = basis(tmp_path)
    renewed = threading.Event()
    original = store.renew

    def renew(*args, **kwargs):
        result = original(*args, **kwargs)
        renewed.set()
        return result

    store.renew = renew
    now = [100]
    candidate = kernel(store, persona, RecordedModel())
    candidate.clock = lambda: now[0]

    def select(_):
        now[0] = 140
        assert renewed.wait(1), "No lease keeper exists between model calls"
        return SourceSelection()

    candidate.select = select
    result = candidate.generate(run, store.snapshot(A), "input")
    assert result.text
    assert store.inspect(A, run.input_id)["run"]["lease_until"] == 190
    assert not any(t.name == "cognition-turn-control" for t in threading.enumerate())


def test_one_keeper_covers_rewrites_and_stops_on_failure(tmp_path):
    store, _, run, persona = basis(tmp_path)
    model = RecordedModel(rejects=1)
    thread_ids = set()

    def observe(stage, result):
        keepers = [t for t in threading.enumerate() if t.name.startswith("cognition-")]
        assert len(keepers) == 1
        assert keepers[0].name == "cognition-turn-control"
        thread_ids.add(keepers[0].ident)
        return result

    model.mutate = observe
    candidate = kernel(store, persona, model)
    assert candidate.generate(run, store.snapshot(A), "input").revision == 2
    assert len(thread_ids) == 1
    assert not any(t.name.startswith("cognition-") for t in threading.enumerate())


def test_cancel_during_selection_prevents_next_request(tmp_path):
    store, item, run, persona = basis(tmp_path)
    model = RecordedModel()
    cancelled = threading.Event()
    model.cancel = cancelled.set
    candidate = kernel(store, persona, model)

    def select(_):
        store.cancel(A, item.input_id)
        assert cancelled.wait(1)
        return SourceSelection()

    candidate.select = select
    with pytest.raises(CallCancelled):
        candidate.generate(run, store.snapshot(A), "input")
    assert model.requests == []
    assert not any(t.name.startswith("cognition-") for t in threading.enumerate())


@pytest.mark.parametrize("transfer", ["run", "state_owner"])
def test_taken_over_epoch_interrupts_before_another_model_request(tmp_path, transfer):
    store, item, run, persona = basis(tmp_path)
    model = RecordedModel()
    interrupted = threading.Event()
    model.cancel = interrupted.set
    candidate = kernel(store, persona, model)

    def select(_):
        if transfer == "run":
            store.claim(A, item.input_id, "replacement", now_us=151, lease_us=50)
        else:
            store.claim_owner(A, "replacement", expected_epoch=run.owner.epoch)
        assert interrupted.wait(1), "Keeper did not observe a replaced owner epoch"
        return SourceSelection()

    candidate.select = select
    with pytest.raises(VersionConflict, match="ownership"):
        candidate.generate(run, store.snapshot(A), "input")
    assert model.requests == []
