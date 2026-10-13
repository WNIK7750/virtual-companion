"""Exact approval/dispatch entry races; all dependencies are synthetic recordings."""

import pytest
from test_default_text_delivery import wait_for
from test_public_search_consumer import SOURCE, close, environment
from virtual_companion.providers.external_information import canonical


@pytest.mark.parametrize("operation", ["approve_segment", "dispatch_segment"])
@pytest.mark.parametrize("change", ["revoke", "omit_callback"])
def test_v8_mutation_requires_live_authority_inside_store_boundary(tmp_path, operation, change):
    runtime, text, memory, mirror, model, events, budget, authority = environment(tmp_path)
    original = getattr(memory.store, operation)
    trace = []

    def intercepted(*args, **kwargs):
        trace.append(
            {
                "phase": operation + "_entry",
                "callback_present": callable(kwargs.get("public_authority")),
            }
        )
        if change == "revoke":
            authority["enabled"] = False
        else:
            kwargs.pop("public_authority", None)
        try:
            result = original(*args, **kwargs)
            trace.append({"phase": "returned", "event": result})
            return result
        except Exception as error:
            trace.append({"phase": "rejected", "type": type(error).__name__, "reason": str(error)})
            raise

    setattr(memory.store, operation, intercepted)
    try:
        text.send(SOURCE)
        wait_for(lambda: not text.snapshot()["busy"])
        segments = memory.store.segments(memory.companion, runtime.claim.input_id)
        (tmp_path / "actual.json").write_text(
            canonical(
                {
                    "operation": operation,
                    "change": change,
                    "trace": trace,
                    "snapshot": text.snapshot(),
                    "segments": segments,
                    "events": events,
                    "budget": budget.rows,
                    "generation_and_review": model.calls,
                }
            ),
            encoding="utf-8",
        )
        if operation == "approve_segment":
            assert segments == []
        else:
            assert len(segments) == 1 and segments[0]["status"] == "pending"
        assert trace[0]["callback_present"] is True
        assert text.snapshot()["attempt"]["status"] == "failed"
        assert trace[-1]["phase"] == "rejected"
        assert isinstance(trace[-1]["reason"], str)
        assert next(iter(budget.rows.values()))["finish"] is not None
        assert [row["role"] for row in mirror.history()["messages"]] == ["user"]
    finally:
        close(runtime, text, memory, mirror)


def test_public_authority_callback_is_pure_and_runs_after_readset_inside_transaction(tmp_path):
    runtime, text, memory, mirror, model, events, budget, authority = environment(tmp_path)
    from contextlib import contextmanager
    from threading import local

    from test_public_search_consumer import finish

    transaction = memory.store.transaction
    validate_readset = memory.store._validate_segment_readset
    authorized = runtime.public_query.authorized
    state = {"authority_in_transaction": []}
    thread = local()

    @contextmanager
    def tracked_transaction(*args, **kwargs):
        thread.depth = getattr(thread, "depth", 0) + 1
        if thread.depth == 1:
            thread.readset_checked = False
        try:
            with transaction(*args, **kwargs) as db:
                yield db
        finally:
            thread.depth -= 1

    def tracked_readset(*args, **kwargs):
        result = validate_readset(*args, **kwargs)
        thread.readset_checked = True
        return result

    def tracked_authority(binding, query):
        if getattr(thread, "depth", 0):
            assert thread.depth == 1 and thread.readset_checked
            state["authority_in_transaction"].append(thread.depth)
        return authorized(binding, query)

    from dataclasses import replace

    runtime.public_query = replace(runtime.public_query, authorized=tracked_authority)
    memory.store.transaction = tracked_transaction
    memory.store._validate_segment_readset = tracked_readset
    try:
        text.send(SOURCE)
        event = finish(runtime, text)
        assert event["readset"]["protocol"] == "history-readset.v8"
        assert state["authority_in_transaction"] == [1, 1]
        assert next(iter(budget.rows.values()))["finish"] is not None
    finally:
        close(runtime, text, memory, mirror)
