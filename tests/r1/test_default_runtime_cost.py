"""Synthetic history cost probes; timings are diagnostics, not latency acceptance."""

import json
import time
from contextlib import contextmanager

from test_default_text_delivery import setup
from test_segment_store import reviewed


def append_confirmed(runtime, count):
    """Use real public boundaries; synthetic review is not model quality evidence."""
    store, mirror = runtime.store, runtime.mirror
    store.ensure_session(runtime.companion, mirror.session_id)
    last_input = None
    for index in range(count):
        attempt = mirror.begin(f"synthetic cost history {index}", {})
        item = store.accept(runtime.config, attempt["turn_id"], f"synthetic cost history {index}",
                            session_id=mirror.session_id)
        now = runtime.clock()
        claim = store.claim(runtime.companion, item.input_id, "cost-seed",
                            now_us=now, lease_us=120_000_000)
        store.approve_segment(claim, reviewed(claim), now_us=now)
        event = store.dispatch_segment(claim, 1, now_us=now)
        store.confirm_segment(runtime.companion, item.input_id, 1,
                              fingerprint=event["fingerprint"], host_receipt=item.input_id,
                              channel="text_visible")
        store.finish_stream(claim, outcome="completed", now_us=now)
        # An interrupted local mirror is the realistic cold recovery case.
        mirror.finish(attempt["attempt_id"], "failed", "")
        assert runtime.memory.scheduler.run_next("cost-maintenance") is not None
        assert runtime.memory.scheduler.run_next("cost-maintenance") is not None
        last_input = item.input_id
    return last_input


def measure(runtime, operation):
    authority = runtime.store.transaction
    records = runtime.mirror.turn_records
    metrics = {"authority_transactions": 0, "authority_writes": 0,
               "mirror_statements": 0, "mirror_writes": 0, "mirror_rows_materialized": 0}

    @contextmanager
    def counted(**kwargs):
        metrics["authority_transactions"] += 1
        metrics["authority_writes"] += int(kwargs.get("write", True))
        with authority(**kwargs) as db:
            yield db

    def count_records(**kwargs):
        rows = records(**kwargs)
        metrics["mirror_rows_materialized"] += len(rows)
        return rows

    def trace(statement):
        metrics["mirror_statements"] += 1
        metrics["mirror_writes"] += int(statement.upper().startswith("BEGIN IMMEDIATE"))

    runtime.store.transaction = counted
    runtime.mirror.turn_records = count_records
    runtime.mirror.connection.set_trace_callback(trace)
    started = time.perf_counter_ns()
    try:
        result = operation()
    finally:
        metrics["elapsed_ms"] = round((time.perf_counter_ns() - started) / 1_000_000, 3)
        runtime.store.transaction = authority
        runtime.mirror.turn_records = records
        runtime.mirror.connection.set_trace_callback(None)
    return metrics, result


def test_default_state_and_scoped_reconciliation_transaction_cost_with_large_history(tmp_path):
    runtime, _, memory, mirror, model = setup(tmp_path)
    rows, total, last_input = [], 0, None
    # The standalone lab run also measured 1024. The ordinary regression needs
    # only growth across these denominators, not minutes of repeated seeding.
    for target in (0, 16, 64):
        if target > total:
            last_input = append_confirmed(runtime, target - total)
        total = target
        state, result = measure(runtime, runtime.state)
        assert result == {"reconcile": []}
        assert state["authority_transactions"] <= 2
        assert state["authority_writes"] == state["mirror_writes"] == 0
        scoped = None
        if last_input is not None:
            scoped, _ = measure(
                runtime, lambda current=last_input: runtime.reconcile(input_id=current)
            )
            assert scoped["authority_transactions"] <= 2
            assert scoped["mirror_writes"] <= 1
            assert scoped["mirror_rows_materialized"] <= 1
        cold, _ = measure(runtime, runtime.reconcile)
        warm, _ = measure(runtime, runtime.reconcile)
        assert cold["authority_transactions"] <= 2
        assert warm["authority_transactions"] <= 2
        assert warm["mirror_writes"] == 0
        rows.append({"history_turns": total, "state": state, "scoped": scoped,
                     "cold_reconcile": cold, "warm_reconcile": warm})
    print("DEFAULT_RUNTIME_COST=" + json.dumps(rows, sort_keys=True))
    assert not model.calls
    memory.close()
    mirror.connection.close()
