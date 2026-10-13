"""Coordinator-owned metering shared by historical and independent A3 runners."""

import json
import threading
import time
from collections import Counter
from hashlib import sha256


class MeteredCalls:
    def __init__(self, transport, ledger, session, route, case, output):
        self.transport, self.ledger, self.session = transport, ledger, session
        self.route, self.case, self.output = route, case, output
        self.records, self.transport_records = [], []
        self.counts = Counter()
        seed = ledger.session_seed(session)
        for row in ledger.records():
            if row["seed"] == seed and row["route"] == route and row["case_id"] == case:
                self.counts[row["stage"]] = max(self.counts[row["stage"]], row["attempt"])
        self.local = threading.local()
        self.lock = threading.Lock()
        transport.observe = self.observe

    def cancel(self):
        self.transport.cancel()

    def observe(self, record):
        identity = self.local.identity
        enriched = {**record, "ledger_call_id": identity, "route": self.route, "case": self.case}
        path = self.output / f"{identity}.response.json"
        path.write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8")
        self.ledger.finish(identity, record["status"], enriched)
        self.local.finished = True
        with self.lock:
            self.transport_records.append(enriched)

    def validated(self, stage, valid):
        with self.lock:
            row = next(r for r in reversed(self.records) if r["stage"] == stage)
            row["schema_valid"] = valid
            row["schema_validated_monotonic"] = time.monotonic()
            (self.output / f"{row['ledger_call_id']}.validation.json").write_text(
                json.dumps(row), encoding="utf-8"
            )

    def __call__(self, stage, messages, schema, cancelled):
        with self.lock:
            self.counts[stage] += 1
            attempt = self.counts[stage]
        identity = self.ledger.reserve(
            self.session, self.route, self.case, attempt, stage, self.transport.reservations[stage]
        )
        self.local.identity, self.local.finished = identity, False
        row = {
            "stage": stage,
            "attempt": attempt,
            "messages": messages,
            "schema": schema,
            "ledger_call_id": identity,
            "response_source": "local_real",
            "started_monotonic": time.monotonic(),
        }
        (self.output / f"{identity}.request.json").write_text(
            json.dumps(row, ensure_ascii=False), encoding="utf-8"
        )
        with self.lock:
            self.records.append(row)
        try:
            row["response"] = self.transport(stage, messages, schema, cancelled)
            return row["response"]
        finally:
            if not self.local.finished:
                self.ledger.finish(
                    identity, "failed", {"scope": "before transport terminal", "request": row}
                )

    def validated_plan(self, plan):
        with self.lock:
            row = next(r for r in reversed(self.records) if r["stage"] == "reply")
            row["plan_validated_monotonic"] = time.monotonic()
            row["plan_sha256"] = sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
            (self.output / f"{row['ledger_call_id']}.plan.json").write_text(
                json.dumps({"plan": plan, "metrics": row}, ensure_ascii=False), encoding="utf-8"
            )
