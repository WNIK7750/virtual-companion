import json

import pytest
from test_cognition_preparation import basis
from virtual_companion.boundary.contracts import VersionConflict
from virtual_companion.cognition.preparation import CallCancelled, CallControl


def control_for(run, **kwargs):
    return CallControl(
        run,
        clock=kwargs.pop("clock", lambda: 100),
        cancelled=kwargs.pop("cancelled", lambda: False),
        renew=kwargs.pop("renew", lambda *_: pytest.fail("unexpected renewal")),
        deadline_us=300,
        renewal_margin_us=15,
        **kwargs,
    )


def test_expired_checkpoint_retains_timing_and_never_renews(tmp_path):
    store, _, run, _ = basis(tmp_path)
    control = control_for(run, clock=lambda: 150)
    with pytest.raises(VersionConflict, match="Expired owner"):
        control.checkpoint()
    trace = control.diagnostics
    assert trace["operations"]["checkpoint"]["failed"] == 1
    assert trace["operations"]["inspect"]["count"] == 1
    assert "renew" not in trace["operations"]
    last = trace["recent"][-1]
    assert last["error"] == "VersionConflict"
    assert last["lease_before_us"] == last["lease_after_us"] == run.lease_until_us
    assert last["observed_now_us"] == 150
    assert last["end_monotonic"] >= last["start_monotonic"]
    assert store.inspect(run.companion_id, run.input_id)["commit"] is None


def test_diagnostics_are_bounded_and_separate_renewal_from_inspection(tmp_path):
    store, _, run, _ = basis(tmp_path)
    control = control_for(
        run, clock=lambda: 140, renew=lambda r, t: store.renew(r, now_us=t, lease_us=50)
    )
    for _ in range(200):
        control.checkpoint()
    trace = control.diagnostics
    assert trace["operations"]["checkpoint"]["count"] == 200
    assert trace["operations"]["inspect"]["count"] == 400
    assert trace["operations"]["renew"]["count"] == 1
    assert len(trace["recent"]) <= 16
    assert len(json.dumps(trace)) < 12000
    assert "完整当前输入" not in json.dumps(trace, ensure_ascii=False)


@pytest.mark.parametrize("phase", ["inspect", "renew"])
def test_failed_io_keeps_original_error_and_phase(tmp_path, phase):
    _, _, run, _ = basis(tmp_path)
    failure = OSError("synthetic blocked operation")

    def fail(*_):
        raise failure

    control = control_for(
        run, clock=lambda: 140, **{"cancelled" if phase == "inspect" else "renew": fail}
    )
    with pytest.raises(OSError) as caught:
        control.checkpoint()
    assert caught.value is failure
    assert control.diagnostics["operations"][phase]["failed"] == 1
    assert control.diagnostics["recent"][-1]["error"] == "OSError"


def test_cancelled_checkpoint_records_terminal_without_renewal(tmp_path):
    _, _, run, _ = basis(tmp_path)
    control = control_for(run, cancelled=lambda: True)
    with pytest.raises(CallCancelled):
        control.checkpoint()
    assert control.diagnostics["recent"][-1]["error"] == "CallCancelled"
    assert "renew" not in control.diagnostics["operations"]
