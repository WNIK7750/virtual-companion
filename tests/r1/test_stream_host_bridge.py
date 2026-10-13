from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from test_rebuilt_turns import A
from test_stream_history import setup
from test_text_continuous_input import backend as backend
from test_text_continuous_input import send
from virtual_companion.text.transport import validate


def test_host_ack_is_durable_before_publish_waiter_returns(tmp_path):
    from virtual_companion.text.stream_host import StreamHostBridge

    store, run, event = setup(tmp_path)
    bridge = StreamHostBridge(store=store, claim=run, cancelled=Event(), timeout=2)
    with ThreadPoolExecutor(max_workers=1) as pool:
        waiting = pool.submit(bridge.present, event)
        assert bridge.ready.wait(1)
        assert bridge.poll()["event"] == event
        bridge.ack({"sequence": 1, "fingerprint": event["fingerprint"], "host_receipt": "host"})
        assert waiting.result(1) == "host"
    assert store.stream_state(A, run.input_id)["confirmed_units"] == 1
    assert bridge.poll()["event"] is None


def test_cancel_stops_offer_but_late_actual_ack_can_reconcile(tmp_path):
    from virtual_companion.text.stream_host import StreamHostBridge

    store, run, event = setup(tmp_path)
    cancelled = Event()
    bridge = StreamHostBridge(store=store, claim=run, cancelled=cancelled, timeout=2)
    with ThreadPoolExecutor(max_workers=1) as pool:
        waiting = pool.submit(bridge.present, event)
        assert bridge.ready.wait(1)
        cancelled.set()
        assert bridge.poll()["event"] is None
        assert waiting.result(1) is None
    bridge.ack({"sequence": 1, "fingerprint": event["fingerprint"], "host_receipt": "late-host"})
    assert store.stream_state(A, run.input_id)["confirmed_units"] == 1
    with pytest.raises(ValueError):
        bridge.ack({"sequence": 1, "fingerprint": "0" * 64, "host_receipt": "bad"})


def test_explicit_host_stop_cancels_persisted_run_but_keeps_unknown_for_reconciliation(tmp_path):
    from virtual_companion.text.stream_host import StreamHostBridge

    store, run, event = setup(tmp_path)
    bridge = StreamHostBridge(store=store, claim=run, cancelled=Event(), timeout=2)
    with ThreadPoolExecutor(max_workers=1) as pool:
        waiting = pool.submit(bridge.present, event)
        assert bridge.ready.wait(1)
        result = bridge.stop()
        assert waiting.result(1) is None
    assert result["state"]["unknown_units"] == 1
    assert bridge.cancelled.is_set()
    assert store.inspect(A, run.input_id)["status"] == "cancelled"
    assert bridge.poll()["event"] is None
    assert len(bridge.stop()["reconcile"]) == 1



@pytest.mark.parametrize("payload", [{}, {"wait_ms": 0}, {"wait_ms": 1000}])
def test_stream_poll_validator_keeps_legacy_and_bounded_wait(payload):
    assert validate("streamPoll", payload) == payload


@pytest.mark.parametrize("payload", [
    {"wait_ms": -1}, {"wait_ms": 1001}, {"wait_ms": True},
    {"wait_ms": 1.5}, {"wait_ms": "1"}, {"wait_ms": 1, "scope": {}},
])
def test_stream_poll_validator_rejects_unbounded_or_unscoped_fields(payload):
    with pytest.raises(ValueError):
        validate("streamPoll", payload)


def test_waiting_poll_wakes_on_exact_offer_without_early_ack(tmp_path):
    from virtual_companion.text.stream_host import StreamHostBridge

    store, run, event = setup(tmp_path)
    bridge = StreamHostBridge(store=store, claim=run, cancelled=Event(), timeout=2)
    assert bridge.poll() == {"event": None}
    with ThreadPoolExecutor(max_workers=2) as pool:
        poll = pool.submit(bridge.poll, 1000)
        delivery = pool.submit(bridge.present, event)
        assert poll.result(1)["event"] == event
        assert bridge.ready.is_set()
        assert not delivery.done()
        assert store.stream_state(A, run.input_id)["confirmed_units"] == 0
        bridge.ack({"sequence": 1, "fingerprint": event["fingerprint"], "host_receipt": "host"})
        assert delivery.result(1) == "host"
    assert not bridge.ready.is_set()
    assert bridge.poll() == {"event": None}


def test_waiting_poll_yields_when_capture_is_no_longer_active(tmp_path):
    from virtual_companion.text.stream_host import StreamHostBridge

    store, run, _ = setup(tmp_path)
    bridge = StreamHostBridge(store=store, claim=run, cancelled=Event(), timeout=2)
    entered, active = Event(), Event()
    active.set()

    def current():
        entered.set()
        return active.is_set()

    with ThreadPoolExecutor(max_workers=1) as pool:
        poll = pool.submit(bridge.poll, 1000, active=current)
        assert entered.wait(1)
        active.clear()
        assert poll.result(0.5) == {"event": None}


def test_bounded_poll_releases_lifecycle_lock_for_state_and_stop(backend, monkeypatch):
    backend.supplier.plans.append({"blocked_prefix": ""})
    send(backend, "synthetic pending generation")
    assert backend.supplier.entered.wait(1)
    entered = Event()
    bridge = backend.runtime.bridge
    original = bridge.poll

    def observe(*args, **kwargs):
        entered.set()
        return original(*args, **kwargs)

    monkeypatch.setattr(bridge, "poll", observe)
    with ThreadPoolExecutor(max_workers=3) as pool:
        poll = pool.submit(backend.api.call, "streamPoll", {"wait_ms": 1000})
        assert entered.wait(1)
        assert pool.submit(backend.api.call, "state").result(0.5)["busy"] is True
        assert not poll.done(), "state must not wait for the bounded poll"
        pool.submit(backend.api.call, "stop").result(0.5)
        assert poll.result(0.5) == {"event": None}


def test_bounded_poll_discards_offer_when_actual_runtime_is_replaced(backend, monkeypatch):
    from types import SimpleNamespace

    backend.supplier.plans.append({"blocked_prefix": ""})
    send(backend, "synthetic old runtime")
    assert backend.supplier.entered.wait(1)
    entered, release = Event(), Event()

    def bind():
        def poll(wait_ms=0):
            if wait_ms:
                entered.set()
                assert release.wait(1)
            return {"event": {"old": True}}
        return poll

    monkeypatch.setattr(backend.runtime, "bind_poll", bind)
    # Replace the actual dispatch closure's runtime, as a provider selection
    # does. The HTTP adapter remains a CPU stub and no provider is loaded.
    dispatch = dict(zip(backend.api.dispatch.__code__.co_freevars,
                        backend.api.dispatch.__closure__, strict=True))["dispatch"].cell_contents
    cell = dict(zip(dispatch.__code__.co_freevars, dispatch.__closure__, strict=True))["runtime"]
    original = cell.cell_contents
    with ThreadPoolExecutor(max_workers=1) as pool:
        poll = pool.submit(backend.api.call, "streamPoll", {"wait_ms": 1000})
        assert entered.wait(1)
        cell.cell_contents = SimpleNamespace()
        release.set()
        try:
            assert poll.result(1) == {"event": None}
        finally:
            cell.cell_contents = original


def test_bound_poll_does_not_retarget_a_later_bridge_or_selected_session(backend):
    backend.supplier.plans.append({"blocked_prefix": ""})
    send(backend, "synthetic captured scope")
    assert backend.supplier.entered.wait(1)
    captured = backend.runtime.bind_poll()
    original_bridge = backend.runtime.bridge
    backend.runtime.bridge = None
    assert captured(0) == {"event": None}
    backend.runtime.bridge = original_bridge
    original_session = backend.mirror.session_id
    backend.mirror.session_id = "different-selected-session"
    try:
        assert captured(0) == {"event": None}
    finally:
        backend.mirror.session_id = original_session



def test_poll_deadline_and_failed_display_do_not_reoffer_reopened_unknown(tmp_path):
    from virtual_companion.text.stream_host import StreamHostBridge

    store, run, event = setup(tmp_path)
    bridge = StreamHostBridge(store=store, claim=run, cancelled=Event(), timeout=0.03)
    assert bridge.poll(1) == {"event": None}
    assert bridge.present(event) is None
    assert store.stream_state(A, run.input_id)["unknown_units"] == 1
    assert not bridge.ready.is_set() and bridge.poll(1) == {"event": None}
    reopened = StreamHostBridge(store=store, claim=run, cancelled=Event())
    assert reopened.poll(1) == {"event": None}
    assert reopened.state()["reconcile"] == [event]


def test_next_suffix_wakes_after_durable_prefix_ack(tmp_path):
    from test_segment_store import reviewed
    from virtual_companion.text.stream_host import StreamHostBridge

    store, run, first = setup(tmp_path)
    bridge = StreamHostBridge(store=store, claim=run, cancelled=Event(), timeout=2)
    with ThreadPoolExecutor(max_workers=2) as pool:
        delivered = pool.submit(bridge.present, first)
        assert bridge.ready.wait(1)
        bridge.ack({"sequence": 1, "fingerprint": first["fingerprint"], "host_receipt": "prefix"})
        assert delivered.result(1) == "prefix"
        assert not bridge.ready.is_set()
        store.approve_segment(run, reviewed(run, ("我在这里。", "继续🌿")), now_us=100)
        second = store.dispatch_segment(run, 2, now_us=100)
        waiting = pool.submit(bridge.poll, 1000)
        delivered = pool.submit(bridge.present, second)
        assert waiting.result(1)["event"] == second
        assert store.stream_state(A, run.input_id)["confirmed_units"] == 1
        bridge.ack({"sequence": 2, "fingerprint": second["fingerprint"], "host_receipt": "suffix"})
        assert delivered.result(1) == "suffix"
    assert store.stream_state(A, run.input_id)["confirmed_units"] == 2
    assert bridge.poll() == {"event": None}


def test_captured_poll_allows_lease_renewal_but_not_foreign_claim(backend):
    from dataclasses import replace

    backend.supplier.plans.append({"blocked_prefix": ""})
    send(backend, "synthetic identity")
    assert backend.supplier.entered.wait(1)
    entered = Event()
    captured = backend.runtime.bind_poll()
    bridge_poll = backend.runtime.bridge.poll

    def observe(wait_ms=0, *, active=None):
        assert active()
        entered.set()
        return bridge_poll(wait_ms, active=active)

    backend.runtime.bridge.poll = observe
    original = backend.runtime.claim
    backend.runtime.claim = replace(original, lease_until_us=original.lease_until_us + 1)
    assert captured(0) == {"event": None} and entered.is_set()
    backend.runtime.claim = replace(original, attempt_id="foreign-attempt")
    entered.clear()
    try:
        assert captured(0) == {"event": None} and not entered.is_set()
    finally:
        backend.runtime.claim = original
