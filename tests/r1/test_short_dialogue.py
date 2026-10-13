"""RF0 short-path controls; a generated candidate is never an acknowledged reply."""

from copy import deepcopy
from threading import Event

import pytest
from virtual_companion.boundary.contracts import canonical
from virtual_companion.cognition.short_dialogue import ShortContext, ShortDialogue, ShortRequest


class Model:
    def __init__(self, text="先把最急的一件事做完，再休息一下。", finish="stop"):
        self.text, self.finish = text, finish
        self.counted, self.sent = [], []

    def count(self, request):
        self.counted.append(deepcopy(request))
        return 100

    def invoke(self, request):
        self.sent.append(deepcopy(request))
        return {"text": self.text, "tool_calls": [], "finish_reason": self.finish}


def request(**changes):
    return ShortRequest("事情太多了，先做什么？", **changes)


def test_one_generation_exact_counted_request_no_planner_and_host_metadata():
    model = Model()
    result = ShortDialogue(model, cancelled=Event()).generate(request())
    assert len(model.sent) == 1
    assert model.counted == model.sent
    assert model.sent[0]["tools"] == []
    assert result.text == model.text
    assert result.policy.startswith("short-dialogue.")
    assert result.status == "candidate"
    assert not hasattr(result, "memory_effects")


@pytest.mark.parametrize("risk", ["unknown", "side_effect", "source_conflict", "sensitive"])
def test_named_unresolved_risk_never_calls_plain_path(risk):
    model = Model()
    with pytest.raises(ValueError, match="controlled"):
        ShortDialogue(model, cancelled=Event()).generate(request(risks=(risk,)))
    assert model.counted == model.sent == []


@pytest.mark.parametrize("text,finish", [("", "stop"), ("半句话", "length"),
                                        ("<think>秘密</think>你好", "stop")])
def test_incomplete_or_reasoning_output_is_not_a_candidate(text, finish):
    model = Model(text, finish)
    with pytest.raises(ValueError):
        ShortDialogue(model, cancelled=Event()).generate(request())
    assert len(model.sent) == 1  # No hidden repair or fallback.


def test_cancel_after_model_prevents_candidate_release():
    cancelled = Event()
    model = Model()
    original = model.invoke

    def invoke(value):
        result = original(value)
        cancelled.set()
        return result

    model.invoke = invoke
    with pytest.raises(InterruptedError):
        ShortDialogue(model, cancelled=cancelled).generate(request())


def test_user_constraint_has_source_and_is_checked_but_no_default_zero_punctuation():
    model = Model("可以！先喝一杯无糖茶。")
    assert ShortDialogue(model, cancelled=Event()).generate(request()).text == model.text
    with pytest.raises(ValueError, match="constraint"):
        ShortDialogue(model, cancelled=Event()).generate(
            request(forbidden_characters="！!", constraint_source="current-user")
        )


def test_current_input_never_truncated_and_unknown_constraint_provenance_rejected():
    with pytest.raises(ValueError):
        request(forbidden_characters="!", constraint_source="")
    model = Model()
    large = "完整输入" * 1200
    ShortDialogue(model, cancelled=Event()).generate(ShortRequest(large))
    assert model.sent[0]["messages"][-1]["content"] == large


@pytest.mark.parametrize("action", ["silent", "defer"])
def test_explicit_host_terminal_without_fake_reply_or_call(action):
    model = Model()
    result = ShortDialogue(model, cancelled=Event()).generate(request(action=action))
    assert result.status == action and result.text is None
    assert model.sent == []


def test_empty_current_memory_projection_keeps_its_lookup_scope_in_the_evidence():
    payload = canonical({"history": [], "memory_status": "current",
                         "memories": [], "exchanges": []})
    context = ShortContext("00000000-0000-0000-0000-000000000001", "query", 0, payload)
    query = ShortRequest("我的饮食偏好是什么？", context=context)
    model = Model("目前的有效记忆里没有相关偏好，无法确认。")

    result = ShortDialogue(model, cancelled=Event(), policy="short-dialogue.v5").generate(query)

    evidence = model.sent[0]["messages"][1]["content"]
    related_memory = next(line for line in evidence.splitlines() if line.startswith("相关长期记忆"))
    assert "本次" in related_memory, "An empty related projection is not a global absence claim"
    assert context.payload == payload
    assert model.sent[0]["messages"][-1] == {"role": "user", "content": query.user_text}
    assert model.counted == model.sent
    assert len(model.sent) == 1
    assert result.text == model.text
