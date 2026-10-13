import json
from hashlib import sha256

import pytest
from test_rebuilt_presentation import approved, prepared
from test_rebuilt_turns import A, config
from virtual_companion.boundary.contracts import Scope, VersionConflict
from virtual_companion.boundary.presentation import MessageUnit, Presentation
from virtual_companion.boundary.sqlite_background_worker import SQLiteBackgroundWorkerStore
from virtual_companion.boundary.turns import CognitiveResult
from virtual_companion.cognition.background_worker import AssistantExchangeWorker, UserFactWorker


def test_confirmed_assistant_question_becomes_waiting_user_once(tmp_path):
    store = SQLiteBackgroundWorkerStore(tmp_path / "work.sqlite3")
    item = store.accept(config(), "u1", "你好")
    run = store.claim(A, item.input_id, "turn", now_us=10, lease_us=100)
    said = "你今天想聊什么？"
    candidate = Presentation(
        Scope(A, item.session_id), item.input_id, run.attempt_id, "c1", 1, said,
        (MessageUnit("one", "text", said),),
    )
    receipt = store.commit(CognitiveResult(
        run, "c1", 1, said, "review", sha256(said.encode()).hexdigest(),
        presentation=approved(candidate),
    ), now_us=11)
    store.publish(A, receipt["delivery_id"])
    store.acknowledge(A, receipt["delivery_id"], candidate_id="c1", revision=1,
                      sha256=receipt["sha256"],
                      presentation_sha256=receipt["presentation_sha256"],
                      sequence=1, displayed_end=len(said))

    def model(stage, messages, schema, cancelled):
        data = json.loads(messages[-1]["content"])
        if stage == "exchange_extract":
            source = data["sources"][-1]
            assert source["role"] == "assistant" and source["text"] == said
            return {"operations": [{"operation": "add", "type": "open_loop",
                                    "evidenceTurnId": source["id"], "evidenceQuote": said,
                                    "supersedesAtomId": None, "expiresAt": None}]}
        return {"intent": "question", "verdict": "supported"}

    worker = AssistantExchangeWorker(store, model, clock=lambda: 12)
    assert worker.run_one(A, "background")
    pending = store.pending_exchanges(A, now_us=13)
    assert len(pending) == 1
    assert pending[0]["direction"] == "user_question"
    assert worker.run_one(A, "background") is None
    assert len(SQLiteBackgroundWorkerStore(store.path).pending_exchanges(A, now_us=13)) == 1

    # The next user input can resolve the question using the same actual
    # conversation and target revision, without a fabricated user turn.
    store.accept(config(), "u2", "聊旅行吧", session_id=item.session_id)

    def answer_model(stage, messages, schema, cancelled):
        data = json.loads(messages[-1]["content"])
        if stage == "fact_extract":
            return {"operations": []}
        if stage == "exchange_extract":
            assert data["sources"][-1]["text"] == "聊旅行吧"
            target = data["targets"][0]["exchange_id"]
            return {"operations": [{"operation": "resolve", "type": None,
                                    "evidenceTurnId": data["sources"][-1]["id"],
                                    "evidenceQuote": "聊旅行吧",
                                    "supersedesAtomId": target, "expiresAt": None}]}
        return {"intent": "answer", "verdict": "supported"}

    # First queued user source is the earlier greeting. Its maintenance is
    # independent of the already confirmed assistant question.
    UserFactWorker(store, lambda stage, messages, schema, cancelled: {"operations": []},
                   clock=lambda: 14).run_one(A, "user-worker")
    assert UserFactWorker(store, answer_model, clock=lambda: 15).run_one(A, "user-worker")
    assert store.pending_exchanges(A, now_us=16) == []


def test_assistant_maintenance_sees_only_confirmed_prefix(tmp_path):
    store = SQLiteBackgroundWorkerStore(tmp_path / "work.sqlite3")
    item, result = prepared(store)
    receipt = store.commit(result, now_us=101)
    store.publish(A, receipt["delivery_id"])
    store.cancel(A, item.input_id)
    store.acknowledge(
        A, receipt["delivery_id"], candidate_id=receipt["candidate_id"],
        revision=1, sha256=receipt["sha256"],
        presentation_sha256=receipt["presentation_sha256"],
        sequence=1, displayed_end=2, stopped=True,
    )
    claim = store.claim_background(A, "background", now_us=102,
                                   lease_us=100, role="assistant")
    sources = store.background_exchange_context(claim)
    assert sources[-1].text == "你🙂"
    assert "好" not in sources[-1].text

    # Requeued after a new input would invalidate this claim. Here the
    # already claimed source is completed directly with its bound context.
    assert store.finish_assistant_background(
        claim, (), sources, evaluator="exchange-model.v7", now_us=103
    )
    with pytest.raises(VersionConflict, match="replay"):
        store.finish_assistant_background(
            claim, (), (), evaluator="exchange-model.v7", now_us=104
        )
