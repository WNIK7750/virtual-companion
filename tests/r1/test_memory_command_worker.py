import json
from datetime import UTC, datetime
from threading import Lock

import pytest
from test_evidence_memory import operation
from test_rebuilt_turns import A, config
from virtual_companion.boundary.contracts import Scope, SemanticConflict
from virtual_companion.boundary.memory import FactKey, MemoryMutation
from virtual_companion.boundary.memory_management import (
    ExplicitMemoryCommand,
    list_source_digest,
)
from virtual_companion.boundary.sqlite_memory_management import SQLiteExplicitMemoryStore
from virtual_companion.cognition.background_worker import UserFactWorker
from virtual_companion.cognition.errors import FactSupportUnresolved
from virtual_companion.text.memory_runtime import _StructuredMemoryInvoker

# M07 receipt b7bd5236989f117b0ae003798fa03e68d4d8d54fa38d965650f3216d43074156.
# These are the actual first-chain intent/fact fields. Only fresh SQLite source
# identity is host-bound below; later support/exchange replies are semantic fakes.
M07_TEXT = "请记住：这次放松选散步；仅在 2026-10-08T16:06:49.593Z 之前有效。"
M07_NOW_US = 1791174580000000
M07_EXPIRY_MS = 1791475609593
M07_WRONG_EXPIRY_MS = 1791542809593


def _no_memory_command():
    return {
        "action": "none", "requestedScope": "none", "targetIds": [],
        "evidenceQuote": "", "quoteStart": None, "quoteEnd": None,
        "replacementQuote": "", "replacementStart": None, "replacementEnd": None,
        "coverage": "not_command", "unresolved": [],
    }


class _StageTextModel:
    supports_stage_metadata = True

    def __init__(self, respond):
        self.respond, self.calls = respond, []

    def prepare_stage_request(self, messages, *, stage, schema):
        data = json.loads(messages[-1]["content"])
        self.calls.append((stage, data))
        respond = self.respond

        class Request:
            def check_context(self):
                pass

            def stream(self, cancelled):
                yield json.dumps(respond(stage, data), ensure_ascii=False)

        return Request()


def _m07_worker(store):
    item = store.accept(config(), "m07", M07_TEXT)

    def respond(stage, data):
        if stage == "memory_command":
            return {**_no_memory_command(), "requestedScope": "no_mention"}
        if stage == "exchange_intent":
            return {"coverage": "complete", "acts": [
                {"segmentId": 0, "intent": "request", "targetIndex": None,
                 "endSegmentId": 0},
                {"segmentId": 0, "intent": "statement", "targetIndex": None,
                 "endSegmentId": 0},
                {"segmentId": 1, "intent": "statement", "targetIndex": None,
                 "endSegmentId": 1},
            ]}
        if stage == "fact_extract":
            return {"operations": [{
                "type": "short_term", "evidenceTurnId": item.input_id,
                "evidenceQuote": M07_TEXT, "expiresAt": M07_WRONG_EXPIRY_MS,
                "operation": "add", "supersedesAtomId": None,
                "factKey": {"subject": "user", "predicate": "本次放松方式"},
            }]}
        if stage == "fact_support":
            return {"intent": "assertion", "verdict": "supported"}
        if stage == "exchange_extract":
            return {"operations": [{
                "operation": "add", "type": "response_obligation", "expiresAt": None,
                "targetIndex": None, "evidenceQuoteIndex": 0,
            }]}
        if stage == "exchange_support":
            return {"intent": "request", "verdict": "supported"}
        raise AssertionError(stage)

    model = _StageTextModel(respond)
    worker = UserFactWorker(
        store, _StructuredMemoryInvoker(model, Lock()), clock=lambda: M07_NOW_US,
        memory_commands=True, resolution_protocol=True, resolution_strategy="intent_first",
    )
    receipt = worker.run_one(A, "m07-worker", source=item.input_id)
    return item, model, receipt


def test_m07_two_adjacent_statements_complete_source_with_literal_utc_ttl(tmp_path):
    store = SQLiteExplicitMemoryStore(tmp_path / "m07.sqlite3")
    item, model, receipt = _m07_worker(store)
    assert receipt and store.memory_status(A) == "current"
    assert store.background_sources(A) == []
    assert [stage for stage, _ in model.calls] == [
        "memory_command", "exchange_intent", "fact_extract", "fact_support",
        "exchange_extract", "exchange_support",
    ]
    (record,) = store.recall(A, now_us=M07_NOW_US)
    assert record["content"] == M07_TEXT
    assert record["expires_at"] == "2026-10-08T16:06:49.593000+00:00"
    support = next(data for stage, data in model.calls if stage == "fact_support")
    assert support["source"]["text"] == M07_TEXT
    assert support["proposal"]["expiresAt"] == M07_EXPIRY_MS
    history = store.history(A, f"{item.input_id}-record-0")
    assert len(history) == 1
    assert json.loads(history[0]["reason"])["quote"] == M07_TEXT
    reopened = SQLiteExplicitMemoryStore(store.path)
    assert reopened.recall(A, now_us=M07_EXPIRY_MS * 1000 - 1) == [record]
    assert reopened.recall(A, now_us=M07_EXPIRY_MS * 1000) == []


# Lifecycle receipt 404fe9f6c61f6bbf66641048aed17952fe65bf9f809391b983b07cd9db468c41.
# The none/request/exchange replies below are its actual wire fields. Fact
# extraction and support were never called there; those two replies are CPU fakes.
REQUEST_FACT_TEXT = "请记住：这次放松选散步；仅在 2026-10-06T16:49:55.259Z 之前有效。"
REQUEST_FACT_NOW_US = 1791198723539000
REQUEST_FACT_EXPIRY_MS = 1791305395259
REQUEST_FACT_OBLIGATION_MS = 1791199795259


def _request_fact_worker(store, *, text=REQUEST_FACT_TEXT, intent="request"):
    item = store.accept(config(), "request-fact", text)

    def respond(stage, data):
        if stage == "memory_command":
            return _no_memory_command()
        if stage == "exchange_intent":
            return {"acts": [{"segmentId": 0, "intent": intent, "targetIndex": None,
                              "endSegmentId": len(data["segments"]) - 1}],
                    "coverage": "complete"}
        if stage == "fact_extract":
            assert data["source"]["text"] == text
            return {"operations": [{
                "operation": "add", "type": "short_term", "evidenceTurnId": item.input_id,
                "evidenceQuote": text, "expiresAt": REQUEST_FACT_EXPIRY_MS,
                "supersedesAtomId": None,
                "factKey": {"subject": "user", "predicate": "本次放松方式"},
            }]}
        if stage == "fact_support":
            assert data["source"]["text"] == text
            return {"intent": "assertion", "verdict": "supported"}
        if stage == "exchange_extract":
            assert data["independent_acts"][0]["intent"] == intent
            return {"operations": []} if intent == "statement" else {"operations": [{
                "operation": "add", "type": "response_obligation",
                "expiresAt": REQUEST_FACT_OBLIGATION_MS,
                "targetIndex": None, "evidenceQuoteIndex": 0,
            }]}
        assert stage == "exchange_support"
        return {"intent": "request", "verdict": "supported"}

    model = _StageTextModel(respond)
    receipt = UserFactWorker(
        store, _StructuredMemoryInvoker(model, Lock()), clock=lambda: REQUEST_FACT_NOW_US,
        memory_commands=True, resolution_protocol=True, resolution_strategy="intent_first",
    ).run_one(A, "worker", source=item.input_id)
    return item, model, receipt


@pytest.mark.parametrize("text,intent", [
    (REQUEST_FACT_TEXT, "request"),
    (REQUEST_FACT_TEXT, "statement"),
    ("这次放松选散步；仅在 2026-10-06T16:49:55.259Z 之前有效；按这个给我建议。", "request"),
])
def test_request_fact_is_extracted_supported_and_committed_independently(tmp_path, text, intent):
    store = SQLiteExplicitMemoryStore(tmp_path / "request-fact.sqlite3")
    item, model, receipt = _request_fact_worker(store, text=text, intent=intent)
    assert receipt and store.background_sources(A) == []
    stages = [stage for stage, _ in model.calls]
    assert stages[:4] == ["memory_command", "exchange_intent", "fact_extract", "fact_support"]
    record, = store.recall(A, now_us=REQUEST_FACT_NOW_US)
    assert record["version"] == 1 and record["content"] == text
    assert record["expires_at"] == "2026-10-06T16:49:55.259000+00:00"
    history = store.history(A, f"{item.input_id}-record-0")
    assert len(history) == 1 and json.loads(history[0]["reason"])["quote"] == text
    pending = store.pending_exchanges(A, now_us=REQUEST_FACT_NOW_US)
    assert [row["content"] for row in pending] == ([] if intent == "statement" else [text])
    if pending:
        assert pending[0]["expires_us"] == REQUEST_FACT_OBLIGATION_MS * 1000
    reopened = SQLiteExplicitMemoryStore(store.path)
    assert reopened.recall(A, now_us=REQUEST_FACT_EXPIRY_MS * 1000 - 1) == [record]
    assert reopened.recall(A, now_us=REQUEST_FACT_EXPIRY_MS * 1000) == []


@pytest.mark.parametrize("text,quote,intent,support_intent", [
    ("我以前选择散步吗？", "选择散步", "question", "question"),
    ("如果天气好，我才考虑选择散步。", "选择散步", "context", "hypothetical"),
    ("我不喜欢茶，请别据此说我喜欢茶。", "喜欢茶", "request", "assertion"),
    ("朋友说：‘我选择散步。’", "我选择散步", "context", "quotation"),
    ("帮我写个故事：我选择散步。", "我选择散步", "request", "quotation"),
    ("这次应该选哪种方式放松？", "这次应该选哪种方式放松？", "question", "question"),
])
def test_non_asserted_data_does_not_become_a_user_fact(
    tmp_path, text, quote, intent, support_intent,
):
    store = SQLiteExplicitMemoryStore(tmp_path / "unsupported.sqlite3")
    item = store.accept(config(), "unsupported", text)

    def respond(stage, data):
        if stage == "memory_command":
            return _no_memory_command()
        if stage == "exchange_intent":
            return {"coverage": "complete", "acts": [{
                "segmentId": 0, "endSegmentId": len(data["segments"]) - 1,
                "intent": intent, "targetIndex": None,
            }]}
        if stage == "fact_extract":
            return {"operations": [{
                "operation": "add", "type": "long_term", "evidenceTurnId": item.input_id,
                "evidenceQuote": quote, "supersedesAtomId": None, "expiresAt": None,
                "factKey": {"subject": "user", "predicate": "选择"},
            }]}
        assert stage == "fact_support"
        assert data["source"]["text"] == text
        return {"intent": support_intent, "verdict": "unsupported"}

    model = _StageTextModel(respond)
    with pytest.raises(FactSupportUnresolved):
        UserFactWorker(
            store, _StructuredMemoryInvoker(model, Lock()), clock=lambda: REQUEST_FACT_NOW_US,
            memory_commands=True, resolution_protocol=True, resolution_strategy="intent_first",
        ).run_one(A, "worker", source=item.input_id)
    assert [stage for stage, _ in model.calls] == [
        "memory_command", "exchange_intent", "fact_extract", "fact_support",
    ]
    assert store.recall(A, now_us=REQUEST_FACT_NOW_US) == []
    assert store.history(A, f"{item.input_id}-record-0") == []
    assert store.memory_barrier(A)


@pytest.mark.parametrize("gap,intent", [
    ("你还记得吗？", "question"),
    ("请推荐一个办法。", "request"),
    ("今天有点累。", "context"),
])
def test_fact_statement_ranges_do_not_merge_across_another_intent(tmp_path, gap, intent):
    store = SQLiteExplicitMemoryStore(tmp_path / "gapped.sqlite3")
    first, last = "这次放松选散步；", "仅在 2026-10-08T16:06:49.593Z 之前有效。"
    source = first + gap + last
    item = store.accept(config(), "gapped", source)

    def respond(stage, data):
        if stage == "memory_command":
            return _no_memory_command()
        if stage == "exchange_intent":
            return {"coverage": "complete", "acts": [
                {"segmentId": i, "endSegmentId": i, "intent": kind, "targetIndex": None}
                for i, kind in enumerate(("statement", intent, "statement"))
            ]}
        assert stage == "fact_extract"
        assert [r["text"] for r in data["fact_evidence_ranges"]] == [first, gap, last]
        return {"operations": [{
            "type": "short_term", "evidenceTurnId": item.input_id,
            "evidenceQuote": source, "expiresAt": M07_WRONG_EXPIRY_MS,
            "operation": "add", "supersedesAtomId": None,
            "factKey": {"subject": "user", "predicate": "本次放松方式"},
        }]}

    model = _StageTextModel(respond)
    with pytest.raises(ValueError, match="source ranges"):
        UserFactWorker(
            store, _StructuredMemoryInvoker(model, Lock()), clock=lambda: M07_NOW_US,
            memory_commands=True, resolution_protocol=True, resolution_strategy="intent_first",
        ).run_one(A, "worker", source=item.input_id)
    assert [stage for stage, _ in model.calls] == [
        "memory_command", "exchange_intent", "fact_extract",
    ]
    assert store.memory_barrier(A)
    assert store.history(A, f"{item.input_id}-record-0") == []


def test_repeated_statement_text_is_bound_to_its_actual_later_segment(tmp_path):
    store = SQLiteExplicitMemoryStore(tmp_path / "repeat.sqlite3")
    repeated = "这次放松选散步；"
    gap = "只是举例。"
    last = "仅在 2026-10-08T16:06:49.593Z 之前有效。"
    source = repeated + gap + repeated + last
    item = store.accept(config(), "repeat", source)

    def respond(stage, data):
        if stage == "memory_command":
            return _no_memory_command()
        if stage == "exchange_intent":
            return {"coverage": "complete", "acts": [
                {"segmentId": i, "endSegmentId": i, "intent": kind, "targetIndex": None}
                for i, kind in enumerate(("context", "context", "statement", "statement"))
            ]}
        if stage == "fact_extract":
            assert data["fact_evidence_ranges"] == [
                {"start": 0, "end": len(repeated), "text": repeated},
                {"start": len(repeated), "end": len(repeated + gap), "text": gap},
                {"start": len(repeated + gap), "end": len(source), "text": repeated + last},
            ]
            return {"operations": [{
                "type": "short_term", "evidenceTurnId": item.input_id,
                "evidenceQuote": repeated + last, "expiresAt": M07_WRONG_EXPIRY_MS,
                "operation": "add", "supersedesAtomId": None,
                "factKey": {"subject": "user", "predicate": "本次放松方式"},
            }]}
        if stage == "exchange_extract":
            return {"operations": []}
        assert stage == "fact_support"
        return {"intent": "assertion", "verdict": "supported"}

    model = _StageTextModel(respond)
    assert UserFactWorker(
        store, _StructuredMemoryInvoker(model, Lock()), clock=lambda: M07_NOW_US,
        memory_commands=True, resolution_protocol=True, resolution_strategy="intent_first",
    ).run_one(A, "worker", source=item.input_id)
    assert store.recall(A, now_us=M07_NOW_US)[0]["content"] == repeated + last


def _m07_reopen_lifecycle(store, *, initial_worker=_m07_worker, now_us=M07_NOW_US):
    """Real worker/SQLite/approval readsets, with CPU semantic replies only."""
    from test_segment_store import reviewed
    from virtual_companion.boundary.history_readset import history_readset
    from virtual_companion.cognition.short_context import ShortContextAdmission
    from virtual_companion.cognition.short_dialogue import ShortRequest
    from virtual_companion.text.memory_runtime import _fake_memory_result

    original, model, receipt = initial_worker(store)
    record_id = f"{original.input_id}-record-0"
    store.cancel(A, original.input_id)
    contexts, commands, queries = [], [], []

    def admitted(store, key):
        nonlocal now_us
        now_us += 100
        item = store.accept(config(), key, "这次放松选什么？", session_id=original.session_id)
        UserFactWorker(
            store, _fake_memory_result, clock=lambda: now_us,
            memory_commands=True, resolution_protocol=True, resolution_strategy="intent_first",
        ).run_one(A, "worker", source=item.input_id)
        run = store.claim(A, item.input_id, key, now_us=now_us + 10, lease_us=1000)
        request = ShortContextAdmission(
            store, lambda *args: ShortRequest(args[-1]), evidence_clock=lambda: now_us + 11,
        )(run, store.snapshot(A), store.inspect(A, item.input_id)["text"], include_unmatched=True)
        context = request.context
        contexts.append(json.loads(context.payload))
        queries.append(item.input_id)
        return item, run, context

    def ack(store, item, run, context, text):
        readset = history_readset(A, item.input_id, run.state_version, context.payload)
        store.approve_segment(run, reviewed(run, (text,)), now_us=now_us + 12,
                              readset=readset)
        event = store.dispatch_segment(run, 1, now_us=now_us + 13)
        store.confirm_segment(A, item.input_id, 1, fingerprint=event["fingerprint"],
                              host_receipt=item.input_id, channel="text_visible")
        store.finish_stream(run, outcome="completed", now_us=now_us + 14)

    def manage(store, action, text, replacement=""):
        nonlocal now_us
        now_us += 100
        item = store.accept(config(), action, text, session_id=original.session_id)

        def respond(stage, data):
            if stage == "memory_command":
                return {**command(text, targets=(record_id,)), "action": action,
                        "replacementQuote": replacement,
                        "replacementStart": text.index(replacement) if replacement else None,
                        "replacementEnd": text.index(replacement) + len(replacement)
                        if replacement else None}
            assert stage == "memory_command_support"
            return {"meaning": "explicit_single_correction" if action == "correct"
                    else "explicit_single_forget", "verdict": "supported"}

        semantic = _StageTextModel(respond)
        result = UserFactWorker(
            store, _StructuredMemoryInvoker(semantic, Lock()), clock=lambda: now_us,
            memory_commands=True, resolution_protocol=True, resolution_strategy="intent_first",
        ).run_one(A, "worker", source=item.input_id)
        assert result["status"] == "committed"
        assert [stage for stage, _ in semantic.calls] == [
            "memory_command", "memory_command_support",
        ]
        store.cancel(A, item.input_id)
        commands.append({"source": item.input_id, "text": text,
                         "receipt": result, "calls": semantic.calls})

    first, run, context = admitted(store, "query-v1")
    ack(store, first, run, context, "这次放松你选散步。")
    manage(store, "correct", "更正：这次放松选拉伸，散步那条改掉。", "这次放松选拉伸")
    store = SQLiteExplicitMemoryStore(store.path)
    second, run, context = admitted(store, "query-v2")
    ack(store, second, run, context, "现在这次放松选拉伸。")
    manage(store, "forget", "请忘记本次放松方式里拉伸这条。")
    store = SQLiteExplicitMemoryStore(store.path)
    current, run, _ = admitted(store, "query-revoked")
    store.cancel(A, current.input_id)
    return {"store": store, "original": original, "initial_receipt": receipt,
            "initial_calls": model.calls, "record_id": record_id,
            "contexts": contexts, "commands": commands, "queries": queries}


def _assert_reopen_readsets(proof, initial_text, expiry):
    store, contexts, queries = proof["store"], proof["contexts"], proof["queries"]
    assert [(row["version"], row["content"]) for row in contexts[0]["memories"]] == [
        (1, initial_text),
    ]
    assert [(row["version"], row["content"]) for row in contexts[1]["memories"]] == [
        (2, "这次放松选拉伸"),
    ]
    assert contexts[2]["memories"] == []
    old_ack = store.segments(A, queries[0])[0]
    new_ack = store.segments(A, queries[1])[0]
    assert old_ack["status"] == new_ack["status"] == "confirmed"
    assert old_ack["text"] == "这次放松你选散步。"
    assert new_ack["text"] == "现在这次放松选拉伸。"
    assert not any(row["text"] == old_ack["text"] for row in contexts[1]["history"])
    assert not any(row["text"] in {old_ack["text"], new_ack["text"]}
                   for row in contexts[2]["history"])
    assert contexts[1]["memories"][0]["expires_at"] == expiry
    assert store.recall(A, now_us=REQUEST_FACT_NOW_US) == []
    assert [row["status"] for row in store.history(A, proof["record_id"])] == [
        "active", "active", "revoked",
    ]
    assert store.inspect(A, proof["commands"][0]["source"])["text"] == (
        "更正：这次放松选拉伸，散步那条改掉。"
    )


def test_m07_correction_forget_and_same_authority_reopen_filter_effective_readsets(tmp_path):
    proof = _m07_reopen_lifecycle(SQLiteExplicitMemoryStore(tmp_path / "lifecycle.sqlite3"))
    _assert_reopen_readsets(proof, M07_TEXT, "2026-10-08T16:06:49.593000+00:00")


def test_request_fact_correction_forget_and_reopen_preserve_effective_scope(tmp_path):
    proof = _m07_reopen_lifecycle(
        SQLiteExplicitMemoryStore(tmp_path / "request-lifecycle.sqlite3"),
        initial_worker=_request_fact_worker, now_us=REQUEST_FACT_NOW_US,
    )
    assert [stage for stage, _ in proof["initial_calls"]][:4] == [
        "memory_command", "exchange_intent", "fact_extract", "fact_support",
    ]
    _assert_reopen_readsets(proof, REQUEST_FACT_TEXT, "2026-10-06T16:49:55.259000+00:00")


def seed(store):
    store.register(A)
    owner = store.claim_owner(A, "seed-owner", expected_epoch=0)
    store.mutate(
        MemoryMutation(
            Scope(A, "seed-session"),
            "seed-music",
            "assert_fact",
            "music",
            "我喜欢安静的音乐",
            "manual",
            "seed-source",
            "seed",
            datetime(2026, 9, 21, tzinfo=UTC),
            FactKey("user", "music", "profile"),
        ),
        owner,
        expected_version=store.snapshot(A).write_version,
    )
    return owner


def command(source, *, coverage="pure_command", targets=("music",)):
    return {
        "action": "forget",
        "requestedScope": "single_record",
        "targetIds": list(targets),
        "evidenceQuote": source,
        "quoteStart": 0,
        "quoteEnd": len(source),
        "replacementQuote": "",
        "replacementStart": None,
        "replacementEnd": None,
        "coverage": coverage,
        "unresolved": [],
    }


def ordinary_command():
    # The g3 local-model response carried old catalog fields despite action=none.
    return {
        "action": "none",
        "requestedScope": "none",
        "targetIds": ["music"],
        "evidenceQuote": "我喜欢安静的音乐。",
        "quoteStart": 0,
        "quoteEnd": 15,
        "replacementQuote": "我喜欢安静的音乐。",
        "replacementStart": 0,
        "replacementEnd": 15,
        "coverage": "not_command",
        "unresolved": [],
    }


@pytest.mark.parametrize("source", [
    "忙完一整天了，依据以前的偏好，有什么适合听着放松的？",
    "以前我喜欢听哪种音乐？",
])
def test_memory_query_does_not_reextract_a_historical_fact(tmp_path, source):
    store = SQLiteExplicitMemoryStore(tmp_path / "memory.sqlite3")
    seed(store)
    store.accept(config(), "aged-memory-query", source)
    source_id = store.background_sources(A)[0]["source"]
    stages = []

    def invoke(stage, messages, schema, cancelled):
        stages.append(stage)
        if stage == "memory_command":
            return ordinary_command()
        if stage == "fact_extract":
            # This is the observed failure: an old fact quoted as this new source.
            return {"operations": [{
                "operation": "add", "type": "long_term",
                "evidenceTurnId": source_id,
                "evidenceQuote": "我喜欢安静的音乐。",
                "supersedesAtomId": None, "expiresAt": None,
                "factKey": {"subject": "user", "predicate": "music"},
            }]}
        if stage == "exchange_intent":
            return {"coverage": "complete", "acts": [{
                "segmentId": 0, "evidenceQuote": source,
                "intent": "question", "targetId": None,
            }]}
        if stage == "exchange_extract":
            data = json.loads(messages[-1]["content"])
            assert data["independent_acts"][0]["evidenceQuote"] == source
            return {"operations": [{
                "operation": "add", "type": "response_obligation",
                "evidenceQuote": source, "supersedesAtomId": None, "expiresAt": None,
            }]}
        if stage == "exchange_support":
            return {"intent": "question", "verdict": "supported"}
        raise AssertionError(stage)

    with pytest.raises(ValueError, match="source ranges"):
        UserFactWorker(
            store, invoke, clock=lambda: 10, memory_commands=True,
            resolution_protocol=True, resolution_strategy="intent_first",
        ).run_one(A, "worker")
    assert stages == ["memory_command", "exchange_intent", "fact_extract"]
    assert store.memory_barrier(A)
    assert store.pending_exchanges(A, now_us=10) == []
    # The invalid new source keeps the current read barrier closed; the old
    # durable revision is still intact rather than copied into this source.
    assert store.recall(A, now_us=10) == []
    assert len(store.history(A, "music")) == 1


@pytest.mark.parametrize("separator", ["。", "，"])
def test_memory_query_with_a_new_fact_still_extracts_it(tmp_path, separator):
    store = SQLiteExplicitMemoryStore(tmp_path / "memory.sqlite3")
    seed(store)
    fact, question = "我后来搬到杭州了", "你还记得吗？"
    source = fact + separator + question
    store.accept(config(), "fact-and-query", source)
    source_id = store.background_sources(A)[0]["source"]
    stages = []

    def invoke(stage, messages, schema, cancelled):
        stages.append(stage)
        if stage == "memory_command":
            return ordinary_command()
        if stage == "exchange_intent":
            return {"coverage": "complete", "acts": [
                {"segmentId": 0, "evidenceQuote": fact + separator,
                 "intent": "statement", "targetId": None},
                {"segmentId": 1, "evidenceQuote": question,
                 "intent": "question", "targetId": None},
            ]}
        if stage == "fact_extract":
            data = json.loads(messages[-1]["content"])
            assert [r["text"] for r in data["fact_evidence_ranges"]] == [fact + separator, question]
            assert data["source"]["text"] == source
            return {"operations": [{
                "operation": "add", "type": "long_term",
                "evidenceTurnId": source_id, "evidenceQuote": fact,
                "supersedesAtomId": None, "expiresAt": None,
                "factKey": {"subject": "user", "predicate": "residence"},
            }]}
        if stage == "fact_support":
            return {"intent": "assertion", "verdict": "supported"}
        if stage == "exchange_extract":
            return {"operations": [{
                "operation": "add", "type": "response_obligation",
                "evidenceQuote": question, "supersedesAtomId": None, "expiresAt": None,
            }]}
        if stage == "exchange_support":
            return {"intent": "question", "verdict": "supported"}
        raise AssertionError(stage)

    assert UserFactWorker(
        store, invoke, clock=lambda: 10, memory_commands=True,
        resolution_protocol=True, resolution_strategy="intent_first",
    ).run_one(A, "worker")

    assert stages == [
        "memory_command", "exchange_intent", "fact_extract", "fact_support",
        "exchange_extract", "exchange_support",
    ]
    assert store.history(A, f"{source_id}-record-0")[-1]["content"] == fact
    assert store.memory_status(A) == "current"
    assert [item["content"] for item in store.pending_exchanges(A, now_us=10)] == [question]


@pytest.mark.parametrize("source,context,pending_text", [
    ("忙完一整天了，依据以前的偏好，有什么适合听着放松的？",
     ("忙完一整天了，", "依据以前的偏好，"), "有什么适合听着放松的？"),
    ("今天有点累，按以前的兴趣，推荐一个放松办法。",
     ("今天有点累，", "按以前的兴趣，"), "推荐一个放松办法。"),
])
def test_context_with_memory_request_does_not_create_a_fact(
    tmp_path, source, context, pending_text,
):
    store = SQLiteExplicitMemoryStore(tmp_path / "memory.sqlite3")
    seed(store)
    store.accept(config(), "context-and-query", source)
    stages = []

    def invoke(stage, messages, schema, cancelled):
        stages.append(stage)
        if stage == "memory_command":
            return ordinary_command()
        if stage == "exchange_intent":
            return {"coverage": "complete", "acts": [
                *({"segmentId": i, "evidenceQuote": quote,
                   "intent": "context", "targetId": None}
                  for i, quote in enumerate(context)),
                {"segmentId": len(context), "evidenceQuote": pending_text,
                 "intent": "request", "targetId": None},
            ]}
        if stage == "fact_extract":
            return {"operations": []}
        if stage == "exchange_extract":
            data = json.loads(messages[-1]["content"])
            assert [act["evidenceQuote"] for act in data["independent_acts"]] == [pending_text]
            quote_schema = schema["properties"]["operations"]["items"]["properties"][
                "evidenceQuote"
            ]
            assert quote_schema == {"enum": [pending_text]}
            return {"operations": [{
                "operation": "add", "type": "response_obligation",
                "evidenceQuote": pending_text, "supersedesAtomId": None, "expiresAt": None,
            }]}
        if stage == "exchange_support":
            return {"intent": "request", "verdict": "supported"}
        raise AssertionError(stage)

    assert UserFactWorker(
        store, invoke, clock=lambda: 10, memory_commands=True,
        resolution_protocol=True, resolution_strategy="intent_first",
    ).run_one(A, "worker")
    assert stages == [
        "memory_command", "exchange_intent", "fact_extract", "exchange_extract", "exchange_support",
    ]
    assert store.memory_status(A) == "current"
    assert [row["content"] for row in store.pending_exchanges(A, now_us=10)] == [pending_text]
    assert [row["content"] for row in store.recall(A, now_us=10)] == ["我喜欢安静的音乐"]
    assert len(store.history(A, "music")) == 1


def test_independent_fact_support_rejects_the_question_span_of_a_mixed_source(tmp_path):
    store = SQLiteExplicitMemoryStore(tmp_path / "memory.sqlite3")
    seed(store)
    source = "我后来搬到杭州了，你还记得吗？"
    store.accept(config(), "bad-mixed-fact", source)
    source_id = store.background_sources(A)[0]["source"]
    stages = []

    def invoke(stage, messages, schema, cancelled):
        stages.append(stage)
        if stage == "memory_command":
            return ordinary_command()
        if stage == "exchange_intent":
            return {"coverage": "complete", "acts": [
                {"segmentId": 0, "evidenceQuote": "我后来搬到杭州了，",
                 "intent": "statement", "targetId": None},
                {"segmentId": 1, "evidenceQuote": "你还记得吗？",
                 "intent": "question", "targetId": None},
            ]}
        if stage == "fact_extract":
            return {"operations": [{
                "operation": "add", "type": "long_term", "evidenceTurnId": source_id,
                "evidenceQuote": "你还记得吗？", "supersedesAtomId": None, "expiresAt": None,
                "factKey": {"subject": "user", "predicate": "memory_question"},
            }]}
        if stage == "fact_support":
            return {"intent": "question", "verdict": "supported"}
        if stage == "exchange_extract":
            return {"operations": [{
                "operation": "add", "type": "response_obligation",
                "evidenceQuote": "你还记得吗？", "supersedesAtomId": None, "expiresAt": None,
            }]}
        if stage == "exchange_support":
            return {"intent": "question", "verdict": "supported"}
        raise AssertionError(stage)

    with pytest.raises(FactSupportUnresolved):
        UserFactWorker(
            store, invoke, clock=lambda: 10, memory_commands=True,
            resolution_protocol=True, resolution_strategy="intent_first",
        ).run_one(A, "worker")
    assert "fact_support" in stages
    assert store.memory_barrier(A)
    assert store.history(A, f"{source_id}-record-0") == []
    assert len(store.history(A, "music")) == 1


def test_context_only_source_creates_neither_fact_nor_obligation(tmp_path):
    store = SQLiteExplicitMemoryStore(tmp_path / "memory.sqlite3")
    seed(store)
    text = "今天有点累。"
    store.accept(config(), "context-only", text)
    stages = []

    def invoke(stage, messages, schema, cancelled):
        stages.append(stage)
        if stage == "memory_command":
            return ordinary_command()
        if stage == "fact_extract":
            return {"operations": []}
        assert stage == "exchange_intent"
        return {"coverage": "complete", "acts": [{
            "segmentId": 0, "evidenceQuote": text, "intent": "context", "targetId": None,
        }]}

    assert UserFactWorker(
        store, invoke, clock=lambda: 10, memory_commands=True,
        resolution_protocol=True, resolution_strategy="intent_first",
    ).run_one(A, "worker")
    assert stages == ["memory_command", "exchange_intent", "fact_extract"]
    assert store.memory_status(A) == "current"
    assert store.pending_exchanges(A, now_us=10) == []
    assert [row["content"] for row in store.recall(A, now_us=10)] == ["我喜欢安静的音乐"]
    assert len(store.history(A, "music")) == 1


def test_cancellation_after_context_classification_keeps_the_source_retryable(tmp_path):
    from threading import Event

    store = SQLiteExplicitMemoryStore(tmp_path / "memory.sqlite3")
    seed(store)
    text = "今天有点累。"
    store.accept(config(), "cancel-context", text)
    cancelled = Event()
    stages = []

    def invoke(stage, messages, schema, call_cancelled):
        stages.append(stage)
        if stage == "memory_command":
            return ordinary_command()
        assert stage == "exchange_intent"
        cancelled.set()
        return {"coverage": "complete", "acts": [{
            "segmentId": 0, "evidenceQuote": text, "intent": "context", "targetId": None,
        }]}

    assert UserFactWorker(
        store, invoke, clock=lambda: 10, memory_commands=True,
        resolution_protocol=True, resolution_strategy="intent_first",
    ).run_one(A, "worker", cancelled=cancelled) is None
    assert stages == ["memory_command", "exchange_intent"]
    source = store.background_sources(A)[0]
    assert source["status"] == "queued" and source["attempts"] == 0
    assert store.memory_barrier(A)
    assert store.pending_exchanges(A, now_us=10) == []
    assert len(store.history(A, "music")) == 1


def test_worker_executes_supported_pure_command_without_joint_fact_pipeline(tmp_path):
    store = SQLiteExplicitMemoryStore(tmp_path / "memory.sqlite3")
    seed(store)
    source = "请忘记我喜欢安静音乐这件事。"
    store.accept(config(), "forget", source)
    job = store.background_sources(A)[0]["job"]
    stages = []

    def invoke(stage, messages, schema, cancelled):
        stages.append(stage)
        if stage == "memory_command":
            return command(source)
        if stage == "memory_command_support":
            return {"meaning": "explicit_single_forget", "verdict": "supported"}
        raise AssertionError(stage)

    receipt = UserFactWorker(
        store, invoke, clock=lambda: 10, memory_commands=True
    ).run_one(A, "worker")

    assert receipt["status"] == "committed"
    assert receipt["action"] == "forget"
    assert stages == ["memory_command", "memory_command_support"]
    assert store.recall(A, now_us=11) == []
    assert store.memory_status(A) == "current"
    source_receipt = store.memory_source_receipt(A, job)
    assert source_receipt["disposition"] == "memory_command_committed"
    assert source_receipt["operation_id"] == receipt["operation_id"]
    assert len(store.history(A, "music")) == 2


def test_worker_routes_an_ordinary_memory_question_past_command_barrier(tmp_path):
    store = SQLiteExplicitMemoryStore(tmp_path / "memory.sqlite3")
    seed(store)
    source = "我后来搬去哪了？"
    store.accept(config(), "ordinary-memory-question", source)
    stages = []

    def invoke(stage, messages, schema, cancelled):
        stages.append(stage)
        if stage == "memory_command":
            return {
                "action": "unclear",
                "requestedScope": "none",
                "targetIds": [],
                "evidenceQuote": "",
                "quoteStart": None,
                "quoteEnd": None,
                "replacementQuote": "",
                "replacementStart": None,
                "replacementEnd": None,
                "coverage": "unclear",
                "unresolved": ["intent"],
            }
        if stage == "fact_extract":
            return {"operations": []}
        if stage == "exchange_intent":
            return {
                "acts": [{
                    "segmentId": 0,

                    "evidenceQuote": source,
                    "intent": "question",
                    "targetId": None,
                }],
                "coverage": "complete",
            }
        if stage == "exchange_extract":
            return {"operations": [{
                "operation": "add",
                "type": "response_obligation",

                "evidenceQuote": source,
                "supersedesAtomId": None,
                "expiresAt": None,
            }]}
        if stage == "exchange_support":
            return {"intent": "question", "verdict": "supported"}
        raise AssertionError(stage)

    receipt = UserFactWorker(
        store,
        invoke,
        clock=lambda: 10,
        memory_commands=True,
        resolution_protocol=True,
        resolution_strategy="intent_first",
    ).run_one(A, "worker")

    assert receipt
    assert stages == [
        "memory_command",
        "exchange_intent",
        "fact_extract",
        "exchange_extract",
        "exchange_support",
    ]
    assert store.memory_status(A) == "current"
    assert store.background_sources(A) == []
    assert [item["content"] for item in store.pending_exchanges(A, now_us=10)] == [source]


def test_worker_protects_mixed_command_without_partial_memory_commit(tmp_path):
    store = SQLiteExplicitMemoryStore(tmp_path / "memory.sqlite3")
    seed(store)
    source = "忘记我的音乐偏好，顺便推荐一首歌。"
    store.accept(config(), "mixed", source)

    def invoke(stage, messages, schema, cancelled):
        assert stage == "memory_command"
        return command(source, coverage="mixed")

    receipt = UserFactWorker(
        store, invoke, clock=lambda: 10, memory_commands=True
    ).run_one(A, "worker")

    assert receipt["status"] == "needs_clarification"
    assert receipt["reason"] == "needs_clarification.mixed_source"
    assert receipt["unresolved"] == ["mixed_content"]
    assert store.history(A, "music")[-1]["status"] == "active"
    assert store.memory_status(A) == "needs_clarification"
    assert store.recall(A, now_us=11) == []


@pytest.mark.parametrize("strategy", ["candidate_first", "intent_first"])
def test_supported_fact_is_kept_when_unrelated_exchange_coverage_is_unresolved(
    tmp_path, strategy,
):
    """An unresolved dialogue tail must not discard a supported user fact."""
    store = SQLiteExplicitMemoryStore(tmp_path / "memory.sqlite3")
    seed(store)
    source = "我后来搬到杭州了；我喜欢喝茶。"
    store.accept(config(), "compound", source)
    source_id = store.background_sources(A)[0]["source"]

    def invoke(stage, messages, schema, cancelled):
        if stage == "memory_command":
            return {
                "action": "none",
                "requestedScope": "none",
                "targetIds": [],
                "evidenceQuote": "",
                "quoteStart": None,
                "quoteEnd": None,
                "replacementQuote": "",
                "replacementStart": None,
                "replacementEnd": None,
                "coverage": "not_command",
                "unresolved": [],
            }
        if stage == "fact_extract":
            candidate = {
                **operation(
                    content="我后来搬到杭州了",
                    evidenceTurnId=source_id,
                    evidenceQuote="我后来搬到杭州了",
                ),
                "factKey": {"subject": "user", "predicate": "residence"},
            }
            if strategy == "intent_first":
                del candidate["content"]
            return {"operations": [candidate]}
        if stage == "fact_support":
            return {"intent": "assertion", "verdict": "supported"}
        if stage == "exchange_intent":
            return {"coverage": "unknown", "acts": [{
                "segmentId": 0, "evidenceQuote": source,
                "intent": "statement", "targetId": None,
            }]}
        if stage == "exchange_extract":
            return {"operations": []}
        if stage == "exchange_assess":
            return {
                "assessments": [],
                "coverage": "incomplete",
                "uncovered": [
                    {"evidenceTurnId": source_id, "evidenceQuote": "我喜欢喝茶。"}
                ],
                "no_exchange": False,
            }
        raise AssertionError(stage)

    with pytest.raises(ValueError, match="coverage"):
        UserFactWorker(
            store,
            invoke,
            clock=lambda: 10,
            memory_commands=True,
            resolution_protocol=True,
            resolution_strategy=strategy,
        ).run_one(A, "worker")

    source_row = store.background_sources(A)[0]
    assert source_row["status"] == "failed"
    assert source_row["error"] == "needs_clarification.source_coverage"
    assert store.history(A, f"{source_id}-record-0")[-1]["content"] == "我后来搬到杭州了"
    assert store.memory_status(A) == "needs_clarification"
    assert store.recall(A, now_us=11) == []

    abandoned = store.abandon_memory_source(A, source_row["job"], "abandon-exchange", now_us=12)
    assert abandoned["status"] == "committed"
    assert "我后来搬到杭州了" in [
        item["content"] for item in store.recall(A, now_us=13)
    ]
    assert store.memory_statistics(A, now_us=13)["effective_count"] == 2
    assert len(store.history(A, f"{source_id}-record-0")) == 1


def test_worker_reports_version_change_instead_of_overwriting_concurrent_target(tmp_path):
    store = SQLiteExplicitMemoryStore(tmp_path / "memory.sqlite3")
    owner = seed(store)
    source = "请忘记我喜欢安静音乐这件事。"
    store.accept(config(), "forget", source)

    def invoke(stage, messages, schema, cancelled):
        if stage == "memory_command":
            return command(source)
        if stage == "memory_command_support":
            store.mutate(
                MemoryMutation(
                    Scope(A, "other-session"),
                    "concurrent-correction",
                    "correct",
                    "music",
                    "我现在喜欢摇滚",
                    "manual",
                    "concurrent-source",
                    "concurrent correction",
                    datetime(2026, 9, 21, tzinfo=UTC),
                    expected_record_version=1,
                ),
                owner,
                expected_version=store.snapshot(A).write_version,
            )
            return {"meaning": "explicit_single_forget", "verdict": "supported"}
        raise AssertionError(stage)

    receipt = UserFactWorker(
        store, invoke, clock=lambda: 10, memory_commands=True
    ).run_one(A, "worker")

    assert receipt["status"] == "version_changed"
    assert receipt["current_record_version"] == 2
    assert store.history(A, "music")[-1]["content"] == "我现在喜欢摇滚"
    assert store.memory_status(A) == "needs_clarification"


def test_mixed_source_cannot_close_on_memory_action_and_abandon_continues_request(
    tmp_path,
):
    store = SQLiteExplicitMemoryStore(tmp_path / "memory.sqlite3")
    seed(store)
    source = "忘记我的音乐偏好。请推荐一首歌。"
    store.accept(config(), "mixed", source)
    job = store.background_sources(A)[0]["job"]

    def interpret(stage, messages, schema, cancelled):
        assert stage == "memory_command"
        return command(source, coverage="mixed")

    protected = UserFactWorker(
        store, interpret, clock=lambda: 10, memory_commands=True
    ).run_one(A, "worker")
    assert protected["reason"] == "needs_clarification.mixed_source"

    digest = list_source_digest(
        action="forget", record_id="music", expected_version=1, replacement=""
    )
    partial = ExplicitMemoryCommand(
        Scope(A, "memory-list"),
        "partial-mixed-forget",
        "forget",
        "music",
        1,
        "list",
        "partial-mixed-forget",
        digest,
        "single_record",
        "explicit_selection",
        "host-list.v1",
        digest,
        resolves_job=job,
    )
    with pytest.raises(SemanticConflict, match="whole-source"):
        store.execute_memory_command(partial, now_us=11)
    assert store.history(A, "music")[-1]["status"] == "active"
    assert store.background_sources(A)[0]["status"] == "failed"

    abandoned = store.abandon_memory_source(A, job, "abandon-mixed-memory", now_us=12)
    assert abandoned["status"] == "continued"
    assert store.background_sources(A)[0]["status"] == "queued"
    stages = []

    def finish_request(stage, messages, schema, cancelled):
        stages.append(stage)
        if stage == "exchange_intent":
            return {
                "acts": [
                    {
                        "segmentId": 0,

                        "evidenceQuote": "忘记我的音乐偏好。",
                        "intent": "statement",
                        "targetId": None,
                    },
                    {
                        "segmentId": 1,

                        "evidenceQuote": "请推荐一首歌。",
                        "intent": "request",
                        "targetId": None,
                    },
                ],
                "coverage": "complete",
            }
        if stage == "exchange_extract":
            return {
                "operations": [
                    {
                        "operation": "add",
                        "type": "response_obligation",

                        "evidenceQuote": "请推荐一首歌。",
                        "supersedesAtomId": None,
                        "expiresAt": None,
                    }
                ]
            }
        if stage == "exchange_support":
            return {"intent": "request", "verdict": "supported"}
        raise AssertionError(stage)

    receipt = UserFactWorker(
        store,
        finish_request,
        clock=lambda: 13,
        memory_commands=True,
        resolution_protocol=True,
        resolution_strategy="intent_first",
    ).run_one(A, "worker")

    assert receipt
    assert stages == ["exchange_intent", "exchange_extract", "exchange_support"]
    assert store.memory_status(A) == "current"
    assert store.recall(A, now_us=14)[0]["record_id"] == "music"
    assert len(store.pending_exchanges(A, now_us=14)) == 1
