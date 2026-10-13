"""Synthetic source classification and opt-in raw-stage capture fixtures.

The grammar fixture is not a semantic model. It exposes the old wire contract's
legal empty act list. StageCapture records an explicitly supplied synthetic model
in tests or a bounded local probe; production does not import or log through it.
"""

import json
from copy import deepcopy
from threading import Event

import pytest
from virtual_companion.boundary.contracts import Scope, SourceResolutionUnresolved
from virtual_companion.boundary.evidence import EvidenceTurn
from virtual_companion.cognition.exchange_model import INTENT, ExchangeModel

SCOPE = Scope("11111111-1111-4111-8111-111111111111", "synthetic-preference")


class StageCapture:
    supports_stage_metadata = True

    def __init__(self, model, *, max_calls=5):
        self.model, self.max_calls, self.calls = model, max_calls, []
        self.supports_stage_output_override = getattr(
            model, "supports_stage_output_override", False
        )

    def prepare_stage_request(self, messages, *, stage, schema, output_tokens=None):
        if len(self.calls) >= self.max_calls:
            raise RuntimeError("Synthetic probe stage budget exhausted")
        options = {"stage": stage, "schema": schema}
        if output_tokens is not None:
            options["output_tokens"] = output_tokens
        request = self.model.prepare_stage_request(messages, **options)
        reserve = (request.reserve if hasattr(request, "reserve") else self.model.output_tokens)
        entry = {"stage": stage, "messages": deepcopy(messages), "schema": deepcopy(schema),
                 "reserve": reserve, "started": False, "raw_text": "",
                 "completed": False}
        self.calls.append(entry)

        class CapturedRequest:
            def check_context(self):
                return request.check_context()

            def stream(self, cancelled):
                entry["started"] = True
                stream = request.stream(cancelled)
                try:
                    for part in stream:
                        entry["raw_text"] += part
                        yield part
                    entry["completed"] = not cancelled.is_set()
                except Exception as error:
                    entry["error"] = type(error).__name__ + ": " + str(error)
                    raise
                finally:
                    stream.close()

        return CapturedRequest()

    def cancel(self):
        return self.model.cancel()


def resolve(text, invoke):
    return ExchangeModel(invoke, Event()).resolve_source(
        scope=SCOPE, batch_id="current", now_us=100,
        turns=(EvidenceTurn(SCOPE, "current", "user", text, len(text)),), targets={},
        eligible_source_ids=("current",), strategy="intent_first")


@pytest.mark.parametrize("text", ["我喜欢画画", "我偏爱散步", "更正，我现在更偏爱散步。"])
def test_nonempty_declaration_grammar_cannot_choose_an_empty_complete_act_list(text):
    calls = []

    def shortest_statement_output(stage, messages, schema, cancelled):
        calls.append(stage)
        if stage == "exchange_intent":
            # Pick the shortest list the supplied wire grammar permits. The
            # statement label is synthetic gold, not inferred model evidence.
            minimum = schema["properties"]["acts"].get("minItems", 0)
            return {"coverage": "complete", "acts": [{"segmentId": 0,
                    "evidenceQuote": text, "intent": "statement", "targetId": None}]
                    if minimum else []}
        assert stage == "exchange_extract"
        return {"operations": []}

    effects, receipt = resolve(text, shortest_statement_output)
    assert effects == () and receipt.no_exchange
    assert calls == ["exchange_intent", "exchange_extract"]


def test_declaration_prompt_distinguishes_social_response_from_request_and_coverage():
    assert "正常接话的期待不等于请求行动" in INTENT
    assert "没有待续事项不等于没有话语行为" in INTENT
    assert "普通陈述也须逐字列出statement并标coverage=complete" in INTENT


@pytest.mark.parametrize("text,intent,verdict,unresolved", [
    ("我喜欢画画", "statement", "unsupported", False),
    ("我喜欢画画。推荐个练习题材吧。", "request", "supported", False),
    ("我不喜欢茶了，推荐别的吧。", "request", "supported", False),
    ("我喜欢画画。怎样练习呢？", "question", "supported", False),
    ("我喜欢画画", "statement", "supported", True),
    ("我喜欢画画", "statement", "unknown", True),
    ("我喜欢画画。推荐个练习题材吧。", "unclear", "unknown", True),
    ("我喜欢画画。推荐个练习题材吧。", "request", "unsupported", True),
])
def test_proposal_conflict_is_adjudicated_once_and_reuses_supported_verdict(
        text, intent, verdict, unresolved):
    calls = []

    def invoke(stage, messages, schema, cancelled):
        calls.append(stage)
        if stage == "exchange_intent":
            # Deliberately label the entire mixed source statement: extraction
            # and adjudication must still preserve its actual request.
            return {"coverage": "complete", "acts": [{"segmentId": 0,
                    "evidenceQuote": text, "intent": "statement", "targetId": None}]}
        if stage == "exchange_extract":
            return {"operations": [{"operation": "add", "type": "response_obligation",
                    "evidenceQuote": text, "supersedesAtomId": None, "expiresAt": None}]}
        assert stage == "exchange_support"
        data = json.loads(messages[-1]["content"])
        assert data["action"] == "open" and data["pending_for"] == "assistant"
        assert data["evidence_quote"] == text and data["confirmed_text"] == text
        assert data["conversation"] == [{"id": "current", "role": "user", "text": text}]
        if "repair" in data:
            return {"intent": intent, "verdict": verdict, "supporting_quote": text}
        return {"intent": intent, "verdict": verdict}

    if unresolved:
        with pytest.raises(SourceResolutionUnresolved, match="contradictory.*coverage"):
            resolve(text, invoke)
    else:
        effects, receipt = resolve(text, invoke)
        assert receipt.no_exchange == (verdict == "unsupported")
        assert receipt.assessments[0].disposition == (
            "rejected" if verdict == "unsupported" else "accepted")
        if verdict == "supported":
            assert len(effects) == 1
            assert effects[0][0].action == "open"
            assert effects[0][0].evidence.quote == text
        else:
            assert effects == () and receipt.assessments[0].reason == "statement"
    assert calls == ["exchange_intent", "exchange_extract", "exchange_support"] + (
        ["exchange_support"] if intent == "statement" and verdict == "supported" else [])


def test_multiple_conflicting_proposals_do_not_spend_multiple_adjudications():
    first, second = "我喜欢画画。", "我偏爱散步。"
    calls = []

    def invoke(stage, messages, schema, cancelled):
        calls.append(stage)
        if stage == "exchange_intent":
            return {"coverage": "complete", "acts": [
                {"segmentId": i, "evidenceQuote": quote, "intent": "statement", "targetId": None}
                for i, quote in enumerate((first, second))]}
        assert stage == "exchange_extract"
        return {"operations": [{"operation": "add", "type": "response_obligation",
                "evidenceQuote": quote, "supersedesAtomId": None, "expiresAt": None}
                for quote in (first, second)]}

    with pytest.raises(SourceResolutionUnresolved, match="contradictory.*coverage"):
        resolve(first + second, invoke)
    assert calls == ["exchange_intent", "exchange_extract"]


@pytest.mark.parametrize("case", ["unknown", "unclear", "incomplete", "conflicting_request"])
def test_declaration_candidates_still_close_on_uncertainty_or_conflicting_request(case):
    text = "我喜欢画画。推荐个练习题材吧。" if case != "unclear" else "我喜欢画画"

    def invoke(stage, messages, schema, cancelled):
        if stage == "exchange_intent":
            quote = "我喜欢画画。" if case == "incomplete" else text
            return {"coverage": "unknown" if case == "unknown" else "complete", "acts": [{
                "segmentId": 0, "evidenceQuote": quote,
                "intent": "unclear" if case == "unclear" else "statement", "targetId": None}]}
        if stage == "exchange_support":
            assert case == "conflicting_request"
            return {"intent": "unclear", "verdict": "unknown"}
        assert stage == "exchange_extract" and case == "conflicting_request"
        return {"operations": [{"operation": "add", "type": "response_obligation",
                "evidenceQuote": text, "supersedesAtomId": None, "expiresAt": None}]}

    with pytest.raises(SourceResolutionUnresolved, match="coverage"):
        resolve(text, invoke)


def test_raw_stage_capture_preserves_wire_text_and_bounds_prepared_calls():
    class SyntheticModel:
        def prepare_stage_request(self, messages, *, stage, schema):
            class Request:
                reserve = 192

                def check_context(self):
                    return 12

                def stream(self, cancelled):
                    yield '{"acts":'
                    yield '[],"coverage":"complete"}'

            return Request()

    capture = StageCapture(SyntheticModel(), max_calls=1)
    messages, schema = [{"role": "user", "content": "synthetic"}], {"type": "object"}
    request = capture.prepare_stage_request(messages, stage="exchange_intent", schema=schema)
    assert request.check_context() == 12
    result = "".join(request.stream(Event()))
    assert json.loads(result) == {"acts": [], "coverage": "complete"}
    assert capture.calls[0]["raw_text"] == result and capture.calls[0]["completed"]
    assert capture.calls[0]["messages"] == messages and capture.calls[0]["schema"] == schema
    with pytest.raises(RuntimeError, match="stage budget"):
        capture.prepare_stage_request(messages, stage="exchange_intent", schema=schema)


@pytest.mark.parametrize("conflicting_proposal,adjudication,unresolved", [
    (False, None, False),
    (True, {"intent": "unclear", "verdict": "unknown"}, True),
    (True, {"intent": "statement", "verdict": "unsupported"}, False),
    (True, [{"intent": "statement", "verdict": "supported"},
            {"intent": "statement", "verdict": "supported",
             "supporting_quote": "我喜欢画画"}], True),
])
def test_synthetic_preference_worker_uses_deepseek_contract_without_network(
        tmp_path, conflicting_proposal, adjudication, unresolved):
    """Wire replay only: the good output is synthetic, the conflict was observed locally."""
    from datetime import UTC, datetime

    from test_text_provider_model import model
    from virtual_companion.text.chronology import utc_microseconds
    from virtual_companion.text.memory_runtime import TextMemoryRuntime

    value, transport = model()  # Existing fake counter/transport; no credential loader.
    capture = StageCapture(value, max_calls=7)
    text = "我喜欢画画"

    def invoke(body, cancelled, **options):
        transport.calls.append((body, options))
        stage, payload = options["stage"], json.loads(body["messages"][-1]["content"])
        assert body["model"] == "deepseek-flash"
        assert body["thinking"] == {"type": "disabled"}
        assert body["response_format"] == {"type": "json_object"}
        if stage == "memory_command":
            assert payload["source"] == text and payload["catalog"] == []
            result = {"action": "none", "requestedScope": "none", "targetIds": [],
                      "evidenceQuote": text, "quoteStart": 0, "quoteEnd": len(text),
                      "replacementQuote": "", "replacementStart": None,
                      "replacementEnd": None, "coverage": "not_command", "unresolved": []}
        elif stage == "fact_extract":
            assert payload["source"]["text"] == text and payload["targets"] == []
            result = {"operations": [{"type": "long_term", "evidenceTurnId":
                      payload["source"]["id"], "evidenceQuote": text, "expiresAt": None,
                      "operation": "add", "supersedesAtomId": None,
                      "factKey": {"subject": "user", "predicate": "喜好"}}]}
        elif stage == "fact_support":
            assert payload["source"]["text"] == text
            result = {"intent": "assertion", "verdict": "supported"}
        elif stage == "exchange_intent":
            assert payload["segments"][0]["text"] == text and payload["targets"] == []
            result = {"coverage": "complete", "acts": [{"segmentId": 0,
                      "endSegmentId": 0, "intent": "statement", "targetIndex": None}]}
        elif stage == "exchange_support":
            assert conflicting_proposal and payload["confirmed_text"] == text
            assert payload["action"] == "open" and payload["pending_for"] == "assistant"
            result = (adjudication[int("repair" in payload)]
                      if isinstance(adjudication, list) else adjudication)
        else:
            assert stage == "exchange_extract"
            assert payload["independent_acts"][0]["intent"] == "statement"
            result = {"operations": [{"operation": "add", "type": "response_obligation",
                      "evidenceQuoteIndex": 0, "targetIndex": None,
                      "supersedesAtomId": None, "expiresAt": None}]
                      if conflicting_proposal else []}
        return {"text": json.dumps(result, ensure_ascii=False), "tool_calls": [],
                "finish_reason": "stop", "usage": {"prompt_tokens": 17,
                "completion_tokens": 5, "total_tokens": 22}}  # Fake usage, not billing evidence.

    transport.invoke = invoke
    memory = TextMemoryRuntime(tmp_path / "memory.sqlite3", SCOPE.companion_id,
                               model=capture, fake=False)
    try:
        intake = memory.accept_user("synthetic-interest", text, SCOPE.session_id)
        if unresolved:
            with pytest.raises(SourceResolutionUnresolved, match="contradictory"):
                memory.scheduler.user.run_one(SCOPE.companion_id, "synthetic",
                                               source=intake.input_id)
        else:
            memory.scheduler.user.run_one(SCOPE.companion_id, "synthetic",
                                           source=intake.input_id)
        status = memory.store.memory_status(SCOPE.companion_id)
        assert status == ("needs_clarification" if unresolved else "current")
        eligible = memory.store.recall(
            SCOPE.companion_id, now_us=utc_microseconds(datetime.now(UTC)))
        if unresolved:
            assert eligible == []
        else:
            assert len(eligible) == 1 and eligible[0]["content"] == text
            assert eligible[0]["source_id"] == intake.input_id
            assert eligible[0]["event_time"] is None
        support_count = (len(adjudication) if isinstance(adjudication, list)
                         else int(conflicting_proposal))
        assert [call["stage"] for call in capture.calls] == [
            "memory_command", "exchange_intent", "fact_extract", "fact_support",
            "exchange_extract"
        ] + ["exchange_support"] * support_count
        assert all(call["completed"] for call in capture.calls)
        assert all(call["reserve"] == (
            1024 if call["stage"] in {"exchange_intent", "exchange_extract"}
            else value.output_tokens
        ) for call in capture.calls)
        assert len(transport.calls) == 5 + support_count
        if not unresolved:
            memory.close()
            memory = TextMemoryRuntime(tmp_path / "memory.sqlite3", SCOPE.companion_id,
                                       model=capture, fake=False)
            reopened = memory.store.recall(
                SCOPE.companion_id, now_us=utc_microseconds(datetime.now(UTC)))
            assert len(reopened) == 1 and reopened[0]["source_id"] == intake.input_id
            assert reopened[0]["version"] == 1 and reopened[0]["event_time"] is None
    finally:
        memory.close()
        value.close()


@pytest.mark.parametrize("text,corrected,unresolved", [
    ("我喜欢画画", {"intent": "statement", "verdict": "unsupported"}, False),
    ("我喜欢画画。推荐个练习题材吧。",
     {"intent": "request", "verdict": "supported"}, False),
    ("我喜欢画画", {"intent": "statement", "verdict": "supported"}, True),
    ("我喜欢画画", {"intent": "unclear", "verdict": "unknown"}, True),
    ("我喜欢画画。推荐个练习题材吧。",
     {"intent": "request", "verdict": "unsupported"}, True),
])
def test_conflicting_support_uses_one_bound_correction(text, corrected, unresolved):
    calls, support_inputs = [], []
    first = {"intent": "statement", "verdict": "supported"}

    def invoke(stage, messages, schema, cancelled):
        data = json.loads(messages[-1]["content"])
        calls.append(stage)
        if stage == "exchange_intent":
            return {"coverage": "complete", "acts": [{"segmentId": 0,
                    "evidenceQuote": text, "intent": "statement", "targetId": None}]}
        if stage == "exchange_extract":
            return {"operations": [{"operation": "add", "type": "response_obligation",
                    "evidenceQuote": text, "supersedesAtomId": None, "expiresAt": None}]}
        assert stage == "exchange_support"
        support_inputs.append(deepcopy(data))
        assert data["action"] == "open" and data["pending_for"] == "assistant"
        assert data["evidence_quote"] == data["confirmed_text"] == text
        assert data["conversation"] == [{"id": "current", "role": "user", "text": text}]
        if len(support_inputs) == 1:
            assert "repair" not in data
            return first
        assert len(support_inputs) == 2
        assert {key: value for key, value in data.items() if key != "repair"} == support_inputs[0]
        assert data["repair"]["previous"] == first
        assert "supporting_quote" in schema["required"]
        return {**corrected, "supporting_quote": text}

    if unresolved:
        with pytest.raises(SourceResolutionUnresolved, match="contradictory.*coverage"):
            resolve(text, invoke)
    else:
        effects, receipt = resolve(text, invoke)
        assert receipt.no_exchange == (corrected["verdict"] == "unsupported")
        if receipt.no_exchange:
            assert effects == () and receipt.assessments[0].disposition == "rejected"
        else:
            command, support = effects[0]
            assert command.action == "open" and command.evidence.quote == text
            assert command.evidence.source_id == "current"
            assert support.command_sha256 == command.fingerprint
    assert calls == ["exchange_intent", "exchange_extract", "exchange_support", "exchange_support"]


@pytest.mark.parametrize("changed", [
    {"supporting_quote": "编造的原文"},
    {"evidenceTurnId": "foreign-source"},
    {"targetId": "foreign-target"},
])
def test_support_correction_cannot_change_grounded_quote_or_identity(changed):
    calls = []
    text = "我喜欢画画"

    def invoke(stage, messages, schema, cancelled):
        calls.append(stage)
        if stage == "exchange_intent":
            return {"coverage": "complete", "acts": [{"segmentId": 0,
                    "evidenceQuote": text, "intent": "statement", "targetId": None}]}
        if stage == "exchange_extract":
            return {"operations": [{"operation": "add", "type": "response_obligation",
                    "evidenceQuote": text, "supersedesAtomId": None, "expiresAt": None}]}
        assert stage == "exchange_support"
        if calls.count(stage) == 1:
            return {"intent": "statement", "verdict": "supported"}
        return {"intent": "statement", "verdict": "unsupported",
                "supporting_quote": text, **changed}

    with pytest.raises(ValueError, match="grounded evidence|bound schema"):
        resolve(text, invoke)
    assert calls == ["exchange_intent", "exchange_extract", "exchange_support", "exchange_support"]


def test_corrected_statement_support_commits_original_fact_and_reopens_default_readset(tmp_path):
    from datetime import timedelta

    from test_default_cross_turn_memory import Scenario, build, turn

    original, query = "我喜欢画画", "按你记得的爱好，空闲时能怎样放松？"
    reply = "你喜欢画画，可以随手画一片叶子，慢慢放松。"
    assert not set(original) & set(query)

    class CorrectedStatementScenario(Scenario):
        def stream(self, messages, cancelled):
            if messages[-1]["content"] == self.original:
                self.calls.append(deepcopy(messages))
                self.primary.append(deepcopy(messages))
                yield json.dumps({"action": "answer", "text": "可以先随手勾几笔。"},
                                 ensure_ascii=False)
            else:
                yield from super().stream(messages, cancelled)

        def invoke(self, stage, messages, schema, cancelled):
            payload = json.loads(messages[-1]["content"])
            if (stage == "exchange_intent" and payload["sources"][-1]["text"] == query
                    and payload["sources"][-1]["role"] == "user"):
                self.worker_calls.append((stage, deepcopy(payload)))
                return {"coverage": "complete", "acts": [{"segmentId": 0,
                        "evidenceQuote": query, "intent": "question", "targetId": None}]}
            if (stage == "exchange_extract" and payload["sources"][-1]["text"] == original
                    and payload["sources"][-1]["role"] == "user"):
                self.worker_calls.append((stage, deepcopy(payload)))
                return {"operations": [{"operation": "add", "type": "response_obligation",
                        "evidenceQuote": original, "supersedesAtomId": None, "expiresAt": None}]}
            if stage == "exchange_support" and payload["confirmed_text"] == original:
                self.worker_calls.append((stage, deepcopy(payload)))
                if "repair" not in payload:
                    return {"intent": "statement", "verdict": "supported"}
                return {"intent": "statement", "verdict": "unsupported",
                        "supporting_quote": original}
            return super().invoke(stage, messages, schema, cancelled)

    model = CorrectedStatementScenario(original=original, query=query, recalled=reply,
                                       predicate="hobby")
    memory, mirror, runtime, session = build(tmp_path, model)
    try:
        first = turn(runtime, session, original)
        assert first["display_text"] == "可以先随手勾几笔。"
        result = json.loads(memory.scheduler.user.run_one(memory.companion, "statement-background"))
        resolution = result["resolution"]
        assert result["operations"] == 1 and resolution["no_exchange"]
        assert resolution["source_id"] == first["basis"]["input_id"]
        assert resolution["assessments"][0]["disposition"] == "rejected"
        assert [stage for stage, _ in model.worker_calls] == [
            "memory_command", "exchange_intent", "fact_extract", "fact_support",
            "exchange_extract", "exchange_support", "exchange_support"]
        initial, corrected = [payload for stage, payload in model.worker_calls
                              if stage == "exchange_support"]
        assert {key: value for key, value in corrected.items() if key != "repair"} == initial
        assert memory.store.memory_status(memory.companion) == "current"
        assert not memory.store.pending_exchanges(memory.companion, now_us=model.clock())
        saved, = memory.store.recall(memory.companion, now_us=model.clock())
        assert saved["content"] == original and saved["source_id"] == first["basis"]["input_id"]
        assert saved["version"] == 1 and saved["event_time"] is None
        assert memory.store.history(memory.companion, saved["record_id"]) == [saved]
        old_session = mirror.session_id
    finally:
        memory.close()
        mirror.close()

    model.now += timedelta(days=1)
    memory, mirror, runtime, session = build(tmp_path, model)
    try:
        session.new_session()
        assert mirror.session_id != old_session and not mirror.history()["messages"]
        event = turn(runtime, session, query)
        assert event["display_text"] == reply
        fact, = event["readset"]["facts"]
        assert fact["record_id"] == saved["record_id"] and fact["version"] == 1
        assert fact["source_id"] == saved["source_id"]
        assert original in "\n".join(item["content"] for item in model.primary[-1][1:-1])
        reopened, = memory.store.recall(memory.companion, now_us=model.clock())
        assert reopened["version"] == 1 and reopened["event_time"] is None
    finally:
        memory.close()
        mirror.close()
