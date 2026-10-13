"""The model's bounded target view must not deadlock durable source completion."""
import json
from dataclasses import replace

import pytest
from test_pending_context_ranking import NOW, pending_catalog
from test_rebuilt_turns import A, config
from virtual_companion.cognition.background_worker import UserFactWorker


def recording(*, target=None, intent=None):
    def invoke(stage, messages, schema, cancelled):
        if stage == "fact_extract":
            return {"operations": []}
        if stage == "exchange_support":
            return {"intent": intent or "end", "verdict": "supported"}
        data = json.loads(messages[-1]["content"])
        source = data["sources"][-1]
        if stage == "exchange_intent":
            if target:
                assert target["exchange_id"] in {t["exchange_id"] for t in data["targets"]}
            return {"coverage": "complete", "acts": [{
                "segmentId": data["segments"][-1]["id"],
                "evidenceQuote": source["text"],
                "intent": intent or ("end" if target else "statement"),
                "targetId": target["exchange_id"] if target else None}]}
        assert stage == "exchange_extract"
        if target is None:
            return {"operations": []}
        return {"operations": [{
            "operation": "resolve" if intent == "answer" else "revoke", "type": None,
             "evidenceQuote": source["text"],
            "supersedesAtomId": target["exchange_id"], "expiresAt": None}]}
    return invoke


def run(store, item, invoke):
    worker = UserFactWorker(store, invoke, clock=lambda: NOW, resolution_protocol=True,
                            resolution_strategy="intent_first")
    return worker.run_one(A, "worker", source=item.input_id)


def test_101_targets_do_not_block_unrelated_source_or_create_false_global_receipt(tmp_path):
    store = pending_catalog(tmp_path, 101)
    current = store.accept(config(), "unrelated", "窗边很安静。")
    receipt = json.loads(run(store, current, recording()))
    resolution = receipt["resolution"]
    assert resolution["protocol"] == "source-resolution.v2"
    assert len(resolution["target_refs"]) == 100
    assert not store.memory_barrier(A)


def test_current_source_can_close_target_outside_old_first_100(tmp_path):
    store = pending_catalog(tmp_path, 101)
    with store.transaction(write=False) as db:
        targets = list(store._eligible_exchange_targets(db, A, now_us=NOW))
    target = targets[-1]
    assert target not in store.pending_exchanges(A, now_us=NOW)
    current = store.accept(config(), "close", "不用再回答" + target["content"].rstrip("？") + "。")
    run(store, current, recording(target=target))
    assert store.exchange_history(A, target["exchange_id"])[-1]["status"] == "closed"
    assert len(store.pending_exchanges(A, now_us=NOW)) == 100
    assert not store.memory_barrier(A)


def test_named_snapshot_rejects_target_changed_during_inference(tmp_path, monkeypatch):
    store = pending_catalog(tmp_path, 1, expires=True)
    current = store.accept(config(), "current", "窗边很安静。")
    finish = store.finish_resolved_background

    def expired(*args, **kwargs):
        # The exact same target becomes ineligible between selection and the
        # commit transaction. A named scope must still fail closed.
        return finish(*args, **{**kwargs, "now_us": NOW + 1_000_001})

    monkeypatch.setattr(store, "finish_resolved_background", expired)
    assert run(store, current, recording()) is None
    assert store.memory_barrier(A)


def test_ack_resolves_the_101st_target_without_catalog_capacity_failure(tmp_path):
    from test_effective_history import emit
    from virtual_companion.cognition.background_worker import AssistantExchangeWorker

    store = pending_catalog(tmp_path, 101)
    with store.transaction(write=False) as db:
        target = list(store._eligible_exchange_targets(db, A, now_us=NOW))[-1]
    current = store.accept(config(), "repeat", target["content"])
    run(store, current, recording(intent="question"))
    emit(store, current, "直接拆成小步骤处理。")
    worker = AssistantExchangeWorker(store, recording(target=target, intent="answer"),
        clock=lambda: NOW + 20, resolution_protocol=True, resolution_strategy="intent_first")
    assert worker.run_one(A, "assistant")
    assert store.exchange_history(A, target["exchange_id"])[-1]["status"] == "resolved"
    assert not store.memory_barrier(A)


def test_target_ranking_prioritizes_current_source_and_uses_context_for_vague_reply(tmp_path):
    from types import SimpleNamespace

    from virtual_companion.cognition.background_worker import _exchange_targets

    store = pending_catalog(tmp_path, 101)
    with store.transaction(write=False) as db:
        catalog = list(store._eligible_exchange_targets(db, A, now_us=NOW))
    target = catalog[-1]
    # Older unrelated requests can contain many words; they cannot outweigh an
    # explicit current reference. The same context remains useful for "好".
    current = store.accept(config(), "current", target["content"])
    source = SimpleNamespace(source_id=current.input_id, text=target["content"])
    history = SimpleNamespace(source_id="prior", text="\n".join(t["content"] for t in catalog[:-1]))
    claim = SimpleNamespace(companion=A, source=current.input_id,
                            session=current.session_id, role="user")
    assert target in _exchange_targets(store, claim, (history, source), now_us=NOW)
    source.text = "好"
    history.text = target["content"]
    assert target in _exchange_targets(store, claim, (history, source), now_us=NOW)


def test_v1_mapping_and_digest_stay_unchanged_and_do_not_claim_named_scope(tmp_path):
    from hashlib import sha256

    from virtual_companion.boundary.exchange_resolution import SourceResolution, digest

    empty = digest([])
    old = SourceResolution("source", sha256(b"source").hexdigest(), empty, empty,
                           empty, (), empty, empty, True, "recording")
    assert old.protocol == "source-resolution.v1"
    assert "target_refs" not in old.to_mapping()
    assert old.fingerprint == digest(old.to_mapping())
    with pytest.raises(ValueError):
        replace(old, protocol="source-resolution.v2")
    with pytest.raises(ValueError):
        replace(old, protocol="source-resolution.v2", target_refs=(("target", 0),))
