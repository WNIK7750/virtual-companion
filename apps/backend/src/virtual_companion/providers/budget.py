"""Durable explicit provider sub-budget; never resets or refunds unknown requests.

This ledger stores request hashes and accounting only, not companion state or
credentials. Price-derived estimates are not invoices. Existing A3 experiments
must still inherit their historical ExperimentLedger; this is not a fresh grant.
"""

import json
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from virtual_companion.boundary.contracts import canonical, identifier


class ProviderBudgetLedger:
    def __init__(self, path, policy):
        self.path = Path(path).resolve()
        if any((p / ".git").exists() for p in (self.path.parent, *self.path.parents)):
            raise ValueError("Provider usage must be outside Git workspaces")
        required = {
            "id",
            "max_calls",
            "max_input_tokens",
            "max_output_tokens",
            "max_cost_microusd",
            "routes",
        }
        if not isinstance(policy, dict) or set(policy) != required:
            raise ValueError("Explicit complete provider budget required")
        identifier(policy["id"])
        for key in ("max_calls", "max_input_tokens", "max_output_tokens", "max_cost_microusd"):
            if type(policy[key]) is not int or not 0 <= policy[key] <= 10**12:
                raise ValueError("Bounded nonnegative provider budget required")
        if not isinstance(policy["routes"], dict) or not 0 < len(policy["routes"]) <= 32:
            raise ValueError("Explicit budgeted routes required")
        for route, rates in policy["routes"].items():
            if not isinstance(route, str) or not re.fullmatch(r"[0-9a-f]{64}", route):
                raise ValueError("Frozen route fingerprint required")
            if (
                not isinstance(rates, dict)
                or set(rates) != {"kind", "input_rate", "output_rate"}
                or rates["kind"] not in {"local", "api"}
                or any(
                    type(rates[k]) is not int or not 0 <= rates[k] <= 10**12
                    for k in ("input_rate", "output_rate")
                )
                or (rates["kind"] == "local" and (rates["input_rate"] or rates["output_rate"]))
            ):
                raise ValueError("Frozen route cost rates required")
        self.identity = canonical(policy)
        self.policy = json.loads(self.identity)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._transaction() as db:
            names = {r[0] for r in db.execute("SELECT name FROM sqlite_schema WHERE type='table'")}
            if not names:
                db.execute("CREATE TABLE provider_budget(identity TEXT NOT NULL)")
                db.execute("INSERT INTO provider_budget VALUES(?)", (self.identity,))
                db.execute(
                    "CREATE TABLE provider_calls(receipt TEXT PRIMARY KEY, route TEXT NOT NULL, "
                    "stage TEXT NOT NULL, request_sha256 TEXT NOT NULL, "
                    "input_tokens INTEGER NOT NULL, "
                    "output_tokens INTEGER NOT NULL, reserved_cost_microusd INTEGER NOT NULL, "
                    "status TEXT NOT NULL, transmission TEXT NOT NULL, usage TEXT, "
                    "estimated_cost_microusd INTEGER, overrun INTEGER NOT NULL, terminal TEXT)"
                )
            elif names != {"provider_budget", "provider_calls"}:
                raise ValueError("Foreign provider ledger; no migration attempted")
            rows = list(db.execute("SELECT identity FROM provider_budget"))
            if len(rows) != 1 or rows[0][0] != self.identity:
                raise ValueError("Provider budget identity changed")

    @contextmanager
    def _transaction(self):
        db = sqlite3.connect(self.path, isolation_level=None, timeout=0.1)
        db.row_factory = sqlite3.Row
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.execute("COMMIT")
        except BaseException:
            if db.in_transaction:
                db.execute("ROLLBACK")
            raise
        finally:
            db.close()

    def _cost(self, route, input_tokens, output_tokens):
        rates = self.policy["routes"][route]
        return (
            input_tokens * rates["input_rate"] + output_tokens * rates["output_rate"] + 999999
        ) // 1000000

    def reserve(self, *, route, stage, request_sha256, input_tokens, output_tokens):
        identifier(stage)
        if route not in self.policy["routes"]:
            raise ValueError("Unbudgeted model route")
        if not isinstance(request_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", request_sha256):
            raise ValueError("Frozen final request hash required")
        if (
            any(type(n) is not int or not 0 <= n <= 10**9 for n in (input_tokens, output_tokens))
            or output_tokens < 1
        ):
            raise ValueError("Bounded input/output reservation required")
        cost = self._cost(route, input_tokens, output_tokens)
        with self._transaction() as db:
            if db.execute("SELECT 1 FROM provider_calls WHERE overrun=1").fetchone():
                raise ValueError("Provider exceeded reservation; reconciliation required")
            totals = db.execute(
                "SELECT count(*),coalesce(sum(input_tokens),0), "
                "coalesce(sum(output_tokens),0),coalesce(sum(reserved_cost_microusd),0) "
                "FROM provider_calls"
            ).fetchone()
            if (
                totals[0] + 1 > self.policy["max_calls"]
                or totals[1] + input_tokens > self.policy["max_input_tokens"]
                or totals[2] + output_tokens > self.policy["max_output_tokens"]
                or totals[3] + cost > self.policy["max_cost_microusd"]
                or (
                    self.policy["routes"][route]["kind"] == "api"
                    and self.policy["max_cost_microusd"] == 0
                )
            ):
                raise ValueError("Provider budget exhausted or remote allowance absent")
            receipt = uuid4().hex
            db.execute(
                "INSERT INTO provider_calls VALUES(?,?,?,?,?,?,?,'reserved','not_sent',"
                "NULL,NULL,0,NULL)",
                (receipt, route, stage, request_sha256, input_tokens, output_tokens, cost),
            )
        return receipt

    def finish(self, receipt, *, status, transmission, usage):
        if status not in {
            "completed",
            "failed",
            "cancelled",
            "deadline_exceeded",
        } or transmission not in {"not_sent", "may_have_been_sent", "response_received"}:
            raise ValueError("Explicit terminal provider outcome required")
        if usage is not None:
            keys = ("prompt_tokens", "completion_tokens", "total_tokens")
            if (
                not isinstance(usage, dict)
                or any(type(usage.get(k)) is not int or not 0 <= usage[k] <= 10**9 for k in keys)
                or usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]
                or transmission != "response_received"
            ):
                raise ValueError("Valid provider usage or explicit unknown required")
        terminal = canonical([status, transmission, usage])
        overrun = False
        with self._transaction() as db:
            row = db.execute("SELECT * FROM provider_calls WHERE receipt=?", (receipt,)).fetchone()
            if row is None:
                raise ValueError("Unknown provider reservation")
            if row["terminal"] is not None:
                if row["terminal"] != terminal:
                    raise ValueError("Conflicting provider terminal replay")
                if row["overrun"]:
                    raise ValueError("Provider exceeded reservation")
                return
            estimate = 0 if transmission == "not_sent" else None
            if usage is not None:
                estimate = self._cost(
                    row["route"], usage["prompt_tokens"], usage["completion_tokens"]
                )
                overrun = (
                    usage["prompt_tokens"] > row["input_tokens"]
                    or usage["completion_tokens"] > row["output_tokens"]
                    or estimate > row["reserved_cost_microusd"]
                )
            db.execute(
                "UPDATE provider_calls SET status=?,transmission=?,usage=?, "
                "estimated_cost_microusd=?,overrun=?,terminal=? WHERE receipt=?",
                (
                    status,
                    transmission,
                    canonical(usage) if usage is not None else None,
                    estimate,
                    int(overrun),
                    terminal,
                    receipt,
                ),
            )
        if overrun:
            # Persist what the provider reported before rejecting the candidate.
            raise ValueError("Provider exceeded reservation")

    def records(self, *, offset=0, limit=100):
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 0 < limit <= 1000:
            raise ValueError("Bounded ledger page required")
        with self._transaction() as db:
            return [
                {**dict(row), "usage": json.loads(row["usage"]) if row["usage"] else None}
                for row in db.execute(
                    "SELECT * FROM provider_calls ORDER BY rowid LIMIT ? OFFSET ?", (limit, offset)
                )
            ]

    def summary(self):
        with self._transaction() as db:
            row = db.execute(
                "SELECT count(*),coalesce(sum(output_tokens),0), "
                "coalesce(sum(usage IS NULL),0),coalesce(sum(reserved_cost_microusd),0) "
                "FROM provider_calls"
            ).fetchone()
            return {
                "reserved_calls": row[0],
                "reserved_output_tokens": row[1],
                "unknown_usage_calls": row[2],
                "reserved_cost_microusd": row[3],
            }
