import json
from hashlib import sha256
from threading import Event

import pytest
from test_rebuilt_presentation import prepared
from test_rebuilt_turns import A, claim, config
from test_segment_store import reviewed
from virtual_companion.boundary.contracts import Scope, SourceResolutionUnresolved, canonical
from virtual_companion.boundary.evidence import EvidenceTurn
from virtual_companion.boundary.sqlite_background_worker import SQLiteBackgroundWorkerStore
from virtual_companion.boundary.sqlite_memory_management import SQLiteExplicitMemoryStore
from virtual_companion.cognition.background_worker import AssistantExchangeWorker, UserFactWorker
from virtual_companion.cognition.exchange_model import ExchangeModel


def test_displayed_statement_completes_once_without_redundant_extraction(tmp_path):
    store = SQLiteBackgroundWorkerStore(tmp_path / "work.sqlite3")
    _, result = prepared(store)
    receipt = store.commit(result, now_us=101)
    store.publish(A, receipt["delivery_id"])
    store.acknowledge(
        A, receipt["delivery_id"], candidate_id=receipt["candidate_id"], revision=1,
        sha256=receipt["sha256"], presentation_sha256=receipt["presentation_sha256"],
        sequence=1, displayed_end=len(result.presentation.candidate.text),
    )
    calls = []

    def invoke(stage, messages, schema, cancelled):
        calls.append(stage)
        assert stage == "exchange_intent"
        data = json.loads(messages[-1]["content"])
        return {"coverage": "complete", "acts": [{"segmentId": segment["id"],
                "evidenceQuote": segment["text"].strip(),
                "intent": "statement", "targetId": None} for segment in data["segments"]]}

    worker = AssistantExchangeWorker(store, invoke, clock=lambda: 102,
                                    resolution_protocol=True, resolution_strategy="intent_first")
    result = json.loads(worker.run_one(A, "worker"))
    assert result["operations"] == 0 and result["resolution"]["no_exchange"]
    assert worker.run_one(A, "worker") is None
    assert calls == ["exchange_intent"]
    assert not store.pending_exchanges(A, now_us=103)


@pytest.mark.parametrize("intent,targets,coverage,expected", [
    ("statement", False, "complete", ["exchange_intent"]),
    ("statement", True, "complete", ["exchange_intent", "exchange_extract"]),
    ("question", False, "complete", ["exchange_intent", "exchange_extract"]),
    ("promise", False, "complete", ["exchange_intent", "exchange_extract"]),
    ("answer", True, "complete", ["exchange_intent", "exchange_extract"]),
    ("unclear", False, "complete", ["exchange_intent"]),
    ("statement", False, "unknown", ["exchange_intent"]),
])
def test_only_complete_targetless_statements_skip_extraction(intent, targets, coverage, expected):
    scope, calls = Scope(A, "session"), []
    target = {"companion_id": A, "exchange_id": "target", "revision": 1,
              "direction": "assistant_response", "content": "问题", "status": "pending",
              "expires_us": None}

    def invoke(stage, messages, schema, cancelled):
        calls.append(stage)
        if stage == "exchange_intent":
            return {"coverage": coverage, "acts": [{"segmentId": 0,
                    "evidenceQuote": "原文。", "intent": intent,
                    "targetId": "target" if intent == "answer" else None}]}
        raise RuntimeError("extraction still required")

    params = dict(scope=scope, batch_id="batch", now_us=100,
                  turns=(EvidenceTurn(scope, "source", "assistant", "原文。", 3),),
                  targets={"target": target} if targets else {},
                  eligible_source_ids=("source",), strategy="intent_first")
    if intent == "statement" and not targets and coverage == "complete":
        effects, receipt = ExchangeModel(invoke, Event()).resolve_source(**params)
        assert not effects and receipt.no_exchange
    else:
        with pytest.raises((RuntimeError, SourceResolutionUnresolved)):
            ExchangeModel(invoke, Event()).resolve_source(**params)
    assert calls == expected


def test_user_statement_conflicting_with_a_second_proposal_keeps_source_unresolved():
    # The live seed reached supported fact storage, then source_coverage failed.
    # Its raw exchange response was not retained. This synthetic conflicting
    # proposal isolates the unnecessary second-stage failure path, not a replay
    # of an undocumented provider decision.
    text, calls, scope = "我喜欢画画", [], Scope(A, "session")

    def invoke(stage, messages, schema, cancelled):
        calls.append(stage)
        if stage == "exchange_intent":
            assert "evidenceTurnId" not in schema["properties"]["acts"]["items"]["properties"]
            return {"coverage": "complete", "acts": [{"segmentId": 0,
                    "evidenceQuote": text, "intent": "statement", "targetId": None}]}
        if stage == "exchange_support":
            return {"intent": "unclear", "verdict": "unknown"}
        assert stage == "exchange_extract"
        return {"operations": [{"operation": "add", "type": "response_obligation",
                "evidenceQuote": text, "supersedesAtomId": None, "expiresAt": None}]}

    with pytest.raises(SourceResolutionUnresolved, match="contradictory.*coverage"):
        ExchangeModel(invoke, Event()).resolve_source(
            scope=scope, batch_id="batch", now_us=100,
            turns=(EvidenceTurn(scope, "source", "user", text, len(text)),), targets={},
            eligible_source_ids=("source",), strategy="intent_first")
    assert calls == ["exchange_intent", "exchange_extract", "exchange_support"]


@pytest.mark.parametrize("text,intent,kind", [
    ("告诉我你希望怎么称呼你。", "request", "open_loop"),
    ("你希望我怎么称呼你？", "question", "open_loop"),
])
def test_genuine_assistant_request_can_open_after_independent_support(text, intent, kind):
    calls = []

    def invoke(stage, messages, schema, cancelled):
        calls.append(stage)
        if stage == "exchange_intent":
            return {"coverage": "complete", "acts": [{"segmentId": 0,
                    "evidenceQuote": text,
                    "intent": intent, "targetId": None}]}
        if stage == "exchange_extract":
            return {"operations": [{"operation": "add", "type": kind,
                    "evidenceQuote": text,
                    "supersedesAtomId": None, "expiresAt": None}]}
        return {"intent": intent, "verdict": "supported"}

    scope = Scope(A, "session")
    effects, receipt = ExchangeModel(invoke, Event()).resolve_source(
        scope=scope, batch_id="batch", now_us=100,
        turns=(EvidenceTurn(scope, "source", "assistant", text, len(text)),), targets={},
        eligible_source_ids=("source",), strategy="intent_first")
    assert len(effects) == 1 and effects[0][0].direction == "user_question"
    assert not receipt.no_exchange
    assert calls == ["exchange_intent", "exchange_extract", "exchange_support"]


def test_optional_invitation_classified_as_statement_creates_no_obligation():
    text = "你也可以稍后再聊。"
    scope = Scope(A, "session")
    calls = []

    def invoke(stage, messages, schema, cancelled):
        calls.append(stage)
        assert stage == "exchange_intent"
        return {"coverage": "complete", "acts": [{"segmentId": 0,
                "evidenceQuote": text,
                "intent": "statement", "targetId": None}]}

    effects, receipt = ExchangeModel(invoke, Event()).resolve_source(
        scope=scope, batch_id="batch", now_us=100,
        turns=(EvidenceTurn(scope, "source", "assistant", text, len(text)),), targets={},
        eligible_source_ids=("source",), strategy="intent_first")
    assert not effects and receipt.no_exchange and calls == ["exchange_intent"]


@pytest.mark.parametrize("first_intent", ["answer", "statement"])
def test_recorded_partial_resolve_cannot_erase_request_or_override_statement(first_intent):
    # Natural-02 calls 049/051: a nonempty proposal must not hide a missing
    # independently identified act, even when the first operation looks useful.
    first, second = "目前没有相关记录。", "你可以直接告诉我。"
    text, calls, scope = first + second, [], Scope(A, "session")
    target = {"companion_id": A, "exchange_id": "target", "revision": 1,
              "direction": "assistant_response", "content": "我的饮食偏好是什么？",
              "status": "pending", "expires_us": None}

    def invoke(stage, messages, schema, cancelled):
        calls.append(stage)
        if stage == "exchange_intent":
            return {"coverage": "complete", "acts": [
                {"segmentId": 0, "evidenceQuote": first,
                 "intent": first_intent,
                 "targetId": "target" if first_intent == "answer" else None},
                {"segmentId": 1, "evidenceQuote": second,
                 "intent": "request", "targetId": None}]}
        assert stage == "exchange_extract"
        return {"operations": [{"operation": "resolve", "type": None,
                "evidenceQuote": first,
                "supersedesAtomId": "target", "expiresAt": None}]}

    with pytest.raises(SourceResolutionUnresolved):
        ExchangeModel(invoke, Event()).resolve_source(
            scope=scope, batch_id="batch", now_us=100,
            turns=(EvidenceTurn(scope, "source", "assistant", text, len(text)),),
            targets={"target": target}, eligible_source_ids=("source",), strategy="intent_first")
    assert calls == ["exchange_intent", "exchange_extract"]


@pytest.mark.parametrize("support_intent,verdict,accepted", [
    ("answer", "supported", True),
    ("answer", "unknown", False),
    ("decline", "supported", False),
    ("decline", "unsupported", False),
    ("promise", "supported", False),
])
def test_negative_query_result_requires_answer_support(support_intent, verdict, accepted):
    scope, calls = Scope(A, "session"), []
    text = "我查过了，目前没有关于你饮食偏好的记录。"
    target = {"companion_id": A, "exchange_id": "target", "revision": 1,
              "direction": "assistant_response", "content": "我的饮食偏好是什么？",
              "status": "pending", "expires_us": None}

    def invoke(stage, messages, schema, cancelled):
        calls.append(stage)
        if stage == "exchange_intent":
            return {"coverage": "complete", "acts": [{"segmentId": 0,
                    "evidenceQuote": text,
                    "intent": "answer", "targetId": "target"}]}
        if stage == "exchange_extract":
            return {"operations": [{"operation": "resolve", "type": None,
                    "evidenceQuote": text,
                    "supersedesAtomId": "target", "expiresAt": None}]}
        return {"intent": support_intent, "verdict": verdict}

    params = dict(scope=scope, batch_id="batch", now_us=100,
                  turns=(EvidenceTurn(scope, "source", "assistant", text, len(text)),),
                  targets={"target": target}, eligible_source_ids=("source",),
                  strategy="intent_first")
    if accepted:
        effects, receipt = ExchangeModel(invoke, Event()).resolve_source(**params)
        assert effects[0][0].action == "resolve" and not receipt.no_exchange
    else:
        with pytest.raises(ValueError):
            ExchangeModel(invoke, Event()).resolve_source(**params)
    assert calls == ["exchange_intent", "exchange_extract", "exchange_support"]


def _two_confirmed_answers(tmp_path, *, current_request=False):
    """Use the real v19 user-source, approval, dispatch and ACK boundaries."""
    store = SQLiteExplicitMemoryStore(tmp_path / "two-answers.sqlite3")
    questions = ("我现在还有已保存的音乐偏好吗？", "如果有，是什么？")
    item = store.accept(config(), "two-questions", "".join(questions))

    def user_invoke(stage, messages, schema, cancelled):
        if stage == "fact_extract":
            return {"operations": []}
        if stage == "exchange_intent":
            return {"coverage": "complete", "acts": [
                {"segmentId": index, "evidenceQuote": quote,
                 "intent": "question", "targetId": None}
                for index, quote in enumerate(questions)]}
        if stage == "exchange_extract":
            return {"operations": [
                {"operation": "add", "type": "response_obligation",
                 "evidenceQuote": quote, "supersedesAtomId": None, "expiresAt": None}
                for quote in questions]}
        assert stage == "exchange_support"
        return {"intent": "question", "verdict": "supported"}

    UserFactWorker(store, user_invoke, clock=lambda: 100, resolution_protocol=True,
                   resolution_strategy="intent_first").run_one(A, "user-worker")
    targets = {row["content"]: row for row in store.pending_exchanges(A, now_us=101)}
    assert set(targets) == set(questions)
    target_ids = tuple(targets[quote]["exchange_id"] for quote in questions)
    assert all(targets[quote]["evidence"]["source_id"] == item.input_id for quote in questions)
    assert all(targets[quote]["revision"] == 1 for quote in questions)

    texts = ("没有已保存的音乐偏好记录。" + ("你愿意重新告诉我吗？" if current_request else ""),
             "没有可提供的具体偏好内容。")
    owner = claim(store, item, now=101)
    events = []
    for sequence in (1, 2):
        store.approve_segment(owner, reviewed(owner, texts[:sequence]), now_us=102)
        event = store.dispatch_segment(owner, sequence, now_us=103)
        confirmed = store.confirm_segment(
            A, item.input_id, sequence, fingerprint=event["fingerprint"],
            host_receipt=f"visible-unit-{sequence}", channel="text_visible")
        assert confirmed["status"] == "confirmed"
        events.append(confirmed)
    store.finish_stream(owner, outcome="completed", now_us=104)
    return store, item, target_ids, texts, events


def _assistant_source_rows(store):
    with store.transaction(write=False) as db:
        return [dict(row) for row in db.execute(
            "SELECT s.*,w.error,w.receipt FROM background_sources s "
            "LEFT JOIN background_work w ON s.companion=w.companion AND s.job=w.job "
            "WHERE s.companion=? AND s.role='assistant' ORDER BY s.rowid", (A,))]


def _bound_answer_invoke(target_ids, texts, calls, contexts, *, compact=False, wrong_target=False):
    def invoke(stage, messages, schema, cancelled):
        calls.append(stage)
        data = json.loads(messages[-1]["content"])
        if stage == "exchange_support":
            assert data["evidence_quote"] in data["confirmed_text"]
            assert data["pending_for"] == "assistant"
            return {"intent": "answer", "verdict": "supported"}
        source = data["sources"][-1]
        index = texts.index(source["text"])
        target_id = target_ids[index]
        target_field = "targetId" if stage == "exchange_intent" else "supersedesAtomId"

        def target_choice(value):
            if compact:
                return {"targetIndex": next((i for i, target in enumerate(data["targets"])
                                             if target["exchange_id"] == value), None)}
            return {target_field: value}

        if stage == "exchange_intent":
            contexts.append(data["sources"])
            assert "evidenceTurnId" not in schema["properties"]["acts"]["items"]["properties"]
            acts = []
            for segment in data["segments"]:
                first = segment["id"] == 0
                acts.append({"segmentId": segment["id"],
                             **({"endSegmentId": segment["id"]} if compact else
                                {"evidenceQuote": segment["text"]}),
                             "intent": "answer" if first else "question",
                             **target_choice(target_id if first else None)})
            return {"coverage": "complete", "acts": acts}
        assert stage == "exchange_extract"
        assert "evidenceTurnId" not in schema["properties"]["operations"]["items"]["properties"]
        return {"operations": [{"operation": "resolve", "type": None,
                **({"evidenceQuoteIndex": 0} if compact else
                   {"evidenceQuote": data["independent_acts"][0]["evidenceQuote"]}),
                **target_choice(target_ids[1] if wrong_target else target_id), "expiresAt": None}]}

    invoke.supports_bound_evidence = compact
    return invoke


@pytest.mark.parametrize("compact", [False, True])
def test_same_input_ack_units_resolve_only_their_answered_targets(tmp_path, compact):
    # The first source must complete even while another target awaits a later ACK.
    # This is a legal synthetic contract probe, not an undocumented model replay.
    store, item, target_ids, texts, events = _two_confirmed_answers(tmp_path)
    user_text = store.exchange_sources(
        Scope(A, item.session_id), references=(("user", item.input_id),))[0].text
    rows = _assistant_source_rows(store)
    assert [row["status"] for row in rows] == ["queued", "queued"]
    for row, text, event in zip(rows, texts, events, strict=True):
        assert event["basis"]["input_id"] == item.input_id
        assert event["basis"]["session_id"] == item.session_id
        assert event["host_receipt"] == f"visible-unit-{event['sequence']}"
        assert row["source"] == "segment-" + event["fingerprint"]
        assert row["source_hash"] == sha256(text.encode()).hexdigest()
        assert row["confirmed_end"] == len(text)
        manifest = {key: event[key] for key in ("protocol", "basis", "sequence", "presentation")}
        assert event["fingerprint"] == sha256(canonical(manifest).encode()).hexdigest()

    calls, contexts = [], []
    worker = AssistantExchangeWorker(
        store, _bound_answer_invoke(target_ids, texts, calls, contexts, compact=compact),
        clock=lambda: 105, resolution_protocol=True, resolution_strategy="intent_first")
    first = json.loads(worker.run_one(A, "assistant-worker"))
    assert first["operations"] == 1 and not first["resolution"]["no_exchange"]
    assert first["resolution"]["target_refs"] == [[target_id, 1] for target_id in target_ids]
    assert [row["status"] for row in _assistant_source_rows(store)] == ["completed", "queued"]
    assert [row["exchange_id"] for row in store.pending_exchanges(A, now_us=105)] == [target_ids[1]]
    assert store.exchange_history(A, target_ids[0])[-1]["revision"] == 2
    assert [source["text"] for source in contexts[0]] == [user_text, texts[0]]

    # Persistence and the current target snapshot also apply to the later source.
    store = SQLiteExplicitMemoryStore(store.path)
    worker.store = store
    second = json.loads(worker.run_one(A, "assistant-worker"))
    assert second["operations"] == 1 and not second["resolution"]["no_exchange"]
    assert second["resolution"]["target_refs"] == [[target_ids[1], 1]]
    assert [source["text"] for source in contexts[1]] == [user_text, *texts]
    assert not store.pending_exchanges(A, now_us=106)
    assert worker.run_one(A, "assistant-worker") is None
    assert calls == ["exchange_intent", "exchange_extract", "exchange_support"] * 2
    rows = _assistant_source_rows(store)
    assert [row["status"] for row in rows] == ["completed", "completed"]
    for index, receipt in enumerate((first, second)):
        assert rows[index]["error"] is None and json.loads(rows[index]["receipt"]) == receipt
        resolution = receipt["resolution"]
        assert resolution["status"] == "complete_with_effects"
        assert resolution["coverage"] == "complete" and resolution["uncovered"] == []
        assert resolution["source_id"] == rows[index]["source"]
        assert resolution["source_sha256"] == receipt["source_sha256"] == rows[index]["source_hash"]
        assert resolution["context_sha256"] == sha256(
            canonical(contexts[index]).encode()).hexdigest()
        history = store.exchange_history(A, target_ids[index])
        assert [version["status"] for version in history] == ["pending", "resolved"]
        assert history[-1]["revision"] == 2
        assert history[-1]["evidence"]["source_id"] == rows[index]["source"]
        assert history[-1]["evidence"]["prefix_sha256"] == rows[index]["source_hash"]
        assert history[-1]["evidence"]["quote"] == texts[index]
        assert history[-1]["support"]["context_sha256"] == resolution["context_sha256"]


@pytest.mark.parametrize("fault", ["omit_current_request", "wrong_target"])
def test_segmented_source_coverage_and_target_conflicts_remain_protected(tmp_path, fault):
    store, _, target_ids, texts, events = _two_confirmed_answers(
        tmp_path, current_request=fault == "omit_current_request")
    before = store.snapshot(A).write_version
    calls, contexts = [], []
    worker = AssistantExchangeWorker(
        store, _bound_answer_invoke(target_ids, texts, calls, contexts, compact=True,
                                   wrong_target=fault == "wrong_target"),
        clock=lambda: 105, resolution_protocol=True, resolution_strategy="intent_first")
    with pytest.raises(SourceResolutionUnresolved, match="source coverage"):
        worker.run_one(A, "assistant-worker")
    rows = _assistant_source_rows(store)
    assert calls == ["exchange_intent", "exchange_extract"]
    assert [row["status"] for row in rows] == ["failed", "queued"]
    assert rows[0]["source"] == "segment-" + events[0]["fingerprint"]
    assert rows[0]["source_hash"] == sha256(texts[0].encode()).hexdigest()
    assert rows[0]["error"] == "needs_clarification.source_coverage" and rows[0]["receipt"] is None
    assert rows[1]["error"] is None and rows[1]["receipt"] is None
    assert store.snapshot(A).write_version == before
    assert {row["exchange_id"] for row in store.pending_exchanges(A, now_us=106)} == set(target_ids)
    assert all(
        [version["status"] for version in store.exchange_history(A, target_id)] == ["pending"]
        for target_id in target_ids)
