from concurrent.futures import ThreadPoolExecutor

import pytest
from virtual_companion.providers.budget import ProviderBudgetLedger

ROUTE = "a" * 64


def policy(**changes):
    return {
        "id": "synthetic-budget.v1",
        "max_calls": 2,
        "max_input_tokens": 1000,
        "max_output_tokens": 1000,
        "max_cost_microusd": 0,
        "routes": {ROUTE: {"kind": "local", "input_rate": 0, "output_rate": 0}},
        **changes,
    }


def reserve(ledger, **changes):
    return ledger.reserve(
        route=ROUTE,
        stage="reply",
        request_sha256="b" * 64,
        input_tokens=20,
        output_tokens=32,
        **changes,
    )


def test_failed_and_unfinished_reservations_survive_reopen(tmp_path):
    ledger = ProviderBudgetLedger(tmp_path / "calls.sqlite3", policy())
    one = reserve(ledger)
    ledger.finish(one, status="failed", transmission="may_have_been_sent", usage=None)
    reserve(ledger)  # Simulate interruption without a terminal response.
    fresh = ProviderBudgetLedger(ledger.path, policy())
    with pytest.raises(ValueError, match="budget"):
        reserve(fresh)
    records = fresh.records()
    assert len(records) == 2
    assert records[0]["usage"] is None
    assert records[1]["status"] == "reserved"
    assert fresh.summary()["reserved_output_tokens"] == 64
    assert fresh.summary()["unknown_usage_calls"] == 2


def test_parallel_reservations_cannot_both_spend_last_slot(tmp_path):
    ledger = ProviderBudgetLedger(tmp_path / "calls.sqlite3", policy(max_calls=1))

    def attempt(_):
        try:
            return reserve(ledger)
        except ValueError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, range(2)))
    assert sum(result is not None for result in results) == 1


def test_contended_parallel_reservations_stay_bounded_and_exactly_counted(tmp_path):
    """Concurrent model and review calls share one worst-case reservation ledger."""

    ledger = ProviderBudgetLedger(tmp_path / "calls.sqlite3", policy(max_calls=4))

    def attempt(index):
        try:
            return ledger.reserve(
                route=ROUTE,
                stage="reply" if index % 2 else "stream_unit_review",
                request_sha256=f"{index:064x}",
                input_tokens=20,
                output_tokens=32,
            )
        except ValueError as error:
            return str(error)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(attempt, range(8)))
    accepted = [value for value in results if not value.startswith("Provider budget")]
    rejected = [value for value in results if value.startswith("Provider budget")]
    assert len(accepted) == 4 and len(rejected) == 4, results
    summary = ledger.summary()
    assert summary["reserved_calls"] == 4
    assert summary["reserved_output_tokens"] == 4 * 32
    assert summary["unknown_usage_calls"] == 4
    reopened = ProviderBudgetLedger(ledger.path, policy(max_calls=4))
    stages = [row["stage"] for row in reopened.records()]
    assert set(stages) <= {"reply", "stream_unit_review"}, stages
    assert len(stages) == 4 and reopened.summary()["reserved_calls"] == 4


def test_actual_usage_above_reservation_is_recorded_and_stops_further_work(tmp_path):
    ledger = ProviderBudgetLedger(tmp_path / "calls.sqlite3", policy())
    receipt = reserve(ledger)
    with pytest.raises(ValueError, match="exceeded"):
        ledger.finish(
            receipt,
            status="completed",
            transmission="response_received",
            usage={"prompt_tokens": 20, "completion_tokens": 100, "total_tokens": 120},
        )
    assert ledger.records()[0]["usage"]["completion_tokens"] == 100
    with pytest.raises(ValueError, match="exceeded"):
        reserve(ledger)


def test_route_budget_identity_and_terminal_replay_are_bound(tmp_path):
    ledger = ProviderBudgetLedger(tmp_path / "calls.sqlite3", policy())
    receipt = reserve(ledger)
    fields = dict(status="cancelled", transmission="not_sent", usage=None)
    ledger.finish(receipt, **fields)
    ledger.finish(receipt, **fields)
    with pytest.raises(ValueError, match="terminal"):
        ledger.finish(receipt, status="completed", transmission="response_received", usage=None)
    with pytest.raises(ValueError, match="identity"):
        ProviderBudgetLedger(ledger.path, policy(max_calls=3))
    with pytest.raises(ValueError, match="route"):
        ledger.reserve(
            route="c" * 64,
            stage="reply",
            request_sha256="b" * 64,
            input_tokens=20,
            output_tokens=32,
        )


def test_zero_remote_allowance_cannot_send_even_with_configured_route(tmp_path):
    ledger = ProviderBudgetLedger(
        tmp_path / "calls.sqlite3",
        policy(routes={ROUTE: {"kind": "api", "input_rate": 1000, "output_rate": 2000}}),
    )
    with pytest.raises(ValueError, match="budget"):
        reserve(ledger)
    assert ledger.records() == []


def test_explicit_priced_synthetic_allowance_reserves_worst_case_before_send(tmp_path):
    ledger = ProviderBudgetLedger(
        tmp_path / "calls.sqlite3",
        policy(
            max_cost_microusd=1,
            routes={ROUTE: {"kind": "api", "input_rate": 1000, "output_rate": 2000}},
        ),
    )
    receipt = reserve(ledger)
    assert ledger.records()[0]["reserved_cost_microusd"] == 1
    with pytest.raises(ValueError, match="budget"):
        reserve(ledger)
    ledger.finish(receipt, status="cancelled", transmission="not_sent", usage=None)
    assert ledger.records()[0]["estimated_cost_microusd"] == 0
