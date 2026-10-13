import json
from dataclasses import replace
from hashlib import sha256

import pytest
from test_cognitive_kernel import RecordedModel, kernel
from test_rebuilt_turns import A, claim, config
from virtual_companion.boundary.runtime import KernelFailure
from virtual_companion.boundary.sqlite_business import SQLiteBusinessStore
from virtual_companion.cognition.fact_planning import FactPlanner


class Recording:
    def bind_turn(self, claim, guardian):
        self.guardian = guardian

    def cancel(self):
        pass

    def __call__(self, stage, messages, schema, cancelled):
        self.guardian.checkpoint()
        data = json.loads(messages[-1]["content"])
        if stage == "fact_support":
            return {"intent": "assertion", "verdict": "supported"}
        return {
            "operations": [
                {
                    "operation": "add",
                    "type": "long_term",
                    "content": "我喜欢茶",
                    "evidenceTurnId": data["source"]["id"],
                    "evidenceQuote": "我喜欢茶",
                    "supersedesAtomId": None,
                    "expiresAt": None,
                    "factKey": {"subject": "user", "predicate": "drink"},
                }
            ]
        }


@pytest.mark.parametrize("rejects", [0, 3])
def test_fact_plan_is_visible_to_reply_review_and_only_commits_with_success(tmp_path, rejects):
    store = SQLiteBusinessStore(tmp_path / "facts.sqlite3", evidence_clock=lambda: 100)
    persona = "中性合成测试。"
    cfg = config()
    cfg = replace(
        cfg, character=replace(cfg.character, sha256=sha256(persona.encode()).hexdigest())
    )
    item = store.accept(cfg, "first", "我喜欢茶")
    run = claim(store, item)
    model = RecordedModel(rejects=rejects)
    planner = FactPlanner(store, Recording(), evidence_clock=lambda: 100)
    candidate = kernel(store, persona, model, memory_planner=planner, evidence_clock=lambda: 100)
    if rejects:
        with pytest.raises(KernelFailure, match="review_exhausted"):
            candidate.generate(run, store.snapshot(A), "我喜欢茶")
        assert store.recall(A, now_us=100) == []
    else:
        value = candidate.generate(run, store.snapshot(A), "我喜欢茶")
        assert store.recall(A, now_us=100) == []
        assert len(value.memories) == 1
        store.commit(value, now_us=101)
        assert store.recall(A, now_us=100)[0]["content"] == "我喜欢茶"
    for request in model.requests:
        assert request["memory_proposals"][0]["content"] == "我喜欢茶"
