import json
from dataclasses import replace
from hashlib import sha256

import pytest
from test_cognitive_kernel import RecordedModel, kernel
from test_rebuilt_turns import A, claim, config
from virtual_companion.boundary.contracts import VersionConflict
from virtual_companion.boundary.runtime import KernelFailure
from virtual_companion.boundary.sqlite_conversation import SQLiteConversationStore
from virtual_companion.cognition.exchange_planning import ExchangePlanner
from virtual_companion.cognition.kernel import SourceSelection


class Extraction:
    def __init__(self, verdict="supported"):
        self.calls, self.verdict = [], verdict

    def bind_turn(self, claim, guardian):
        self.guardian = guardian

    def cancel(self):
        pass

    def __call__(self, stage, messages, schema, cancelled):
        self.guardian.checkpoint()
        self.calls.append(stage)
        if stage == "exchange_support":
            return {"intent": "request", "verdict": self.verdict}
        source = json.loads(messages[-1]["content"])["sources"][-1]
        return {
            "operations": [
                {
                    "operation": "add",
                    "type": "response_obligation",
                    "evidenceQuote": source["text"],
                    "evidenceTurnId": source["id"],
                    "supersedesAtomId": None,
                    "expiresAt": None,
                }
            ]
        }


def prepared(tmp_path, *, rejects=0, verdict="supported"):
    store = SQLiteConversationStore(tmp_path / "kernel.sqlite3", evidence_clock=lambda: 100)
    system = "中性合成测试，不附加角色设定。"
    cfg = config()
    cfg = replace(cfg, character=replace(cfg.character, sha256=sha256(system.encode()).hexdigest()))
    text = "帮我想想周末安排"
    item = store.accept(cfg, "input", text)
    run = claim(store, item)
    extraction, reply = Extraction(verdict), RecordedModel(rejects=rejects)
    planner = ExchangePlanner(store, extraction, evidence_clock=lambda: 100)
    candidate = kernel(store, system, reply, exchange_planner=planner, evidence_clock=lambda: 100)
    return store, item, run, text, candidate, extraction, reply


def test_exchange_plan_commits_only_with_the_reviewed_reply(tmp_path):
    store, _, run, text, candidate, extraction, reply = prepared(tmp_path)
    value = candidate.generate(run, store.snapshot(A), text)
    assert extraction.calls == ["exchange_extract", "exchange_support"]
    assert [r["stage"] for r in reply.requests] == ["prepare", "reply", "review"]
    assert store.pending_exchanges(A, now_us=100) == []
    assert len(value.exchanges) == 1
    store.commit(value, now_us=101)
    assert store.pending_exchanges(A, now_us=100)[0]["content"] == text
    with store.transaction(write=False) as db:
        assert db.execute("SELECT count(*) FROM exchange_observations").fetchone()[0] == 1


def test_failed_reply_review_does_not_consume_sources_or_write_obligations(tmp_path):
    store, _, run, text, candidate, _, _ = prepared(tmp_path, rejects=3)
    with pytest.raises(KernelFailure, match="review_exhausted"):
        candidate.generate(run, store.snapshot(A), text)
    assert store.exchange_context(run)[1] == (run.input_id,)
    assert store.pending_exchanges(A, now_us=100) == []


def test_unknown_extraction_fails_before_generating_a_success_reply(tmp_path):
    store, _, run, text, candidate, _, reply = prepared(tmp_path, verdict="unknown")
    with pytest.raises(KernelFailure, match="exchange_model_contract"):
        candidate.generate(run, store.snapshot(A), text)
    assert reply.requests == []
    assert store.pending_exchanges(A, now_us=100) == []


def test_context_text_must_equal_actual_input_before_any_extraction(tmp_path):
    store, _, run, _, candidate, extraction, _ = prepared(tmp_path)
    with pytest.raises(KernelFailure, match="exchange_model_contract"):
        candidate.generate(run, store.snapshot(A), "替换过的输入")
    assert extraction.calls == []


def test_source_conflict_is_not_misclassified_as_a_model_failure(tmp_path, monkeypatch):
    store, _, run, text, candidate, _, reply = prepared(tmp_path)

    def changed(*args):
        raise VersionConflict("synthetic changed source")

    monkeypatch.setattr(candidate.exchange_planner, "plan", changed)
    with pytest.raises(VersionConflict):
        candidate.generate(run, store.snapshot(A), text)
    assert reply.requests == []


def test_user_answer_catches_up_cancelled_ack_work_without_reopening_question(tmp_path):
    from test_ack_maintenance_queue import acknowledge
    from test_pending_exchanges import question

    store = SQLiteConversationStore(
        tmp_path / "catchup.sqlite3", evidence_clock=lambda: 100, automatic_maintenance=True
    )
    system = "中性合成测试。"
    cfg = config()
    cfg = replace(cfg, character=replace(cfg.character, sha256=sha256(system.encode()).hexdigest()))
    first = store.accept(cfg, "first", "你好")
    question_command = question(store, item=first, displayed=7)
    background = store.queued_maintenance(A)[0]
    item = store.accept(cfg, "answer", "想聊旅行", session_id=first.session_id)
    assert store.inspect(A, background)["status"] == "cancelled"
    run = claim(store, item, now=200)

    class Answered(Extraction):
        def __call__(self, stage, messages, schema, cancelled):
            self.guardian.checkpoint()
            self.calls.append(stage)
            assert stage == "exchange_extract"
            source_text = [s["text"] for s in json.loads(messages[-1]["content"])["sources"]]
            assert source_text == ["你好", question_command.content, "想聊旅行"]
            return {"operations": []}  # Recorded judgment: this question was answered.

    extraction, reply = Answered(), RecordedModel()
    planner = ExchangePlanner(store, extraction, evidence_clock=lambda: 100)
    candidate = kernel(store, system, reply, exchange_planner=planner, evidence_clock=lambda: 100)
    candidate.clock = lambda: 200
    candidate.select = lambda c: SourceSelection(history=planner.history(c))
    value = candidate.generate(run, store.snapshot(A), "想聊旅行")
    receipt = store.commit(value, now_us=201)
    assert store.pending_exchanges(A, now_us=100) == []
    assert [h["text"] for h in reply.requests[-1]["context"]["history"]] == [
        "你好",
        question_command.content,
    ]
    event = store.publish(A, receipt["delivery_id"])
    acknowledge(store, event, 1, len(event["text"]))
    reopened = SQLiteConversationStore(store.path)
    next_item = reopened.accept(cfg, "next", "继续聊旅行", session_id=item.session_id)
    next_run = claim(reopened, next_item, now=300)
    _, eligible = reopened.exchange_context(next_run)
    assert question_command.evidence.source_id not in eligible
    assert next_item.input_id in eligible
