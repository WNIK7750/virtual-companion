import json
from dataclasses import replace
from datetime import UTC, datetime
from threading import Event

import pytest
from jsonschema import Draft202012Validator
from test_business_memory_commit import result
from test_evidence_memory import operation
from test_rebuilt_turns import A, claim, config
from virtual_companion.boundary.contracts import Scope
from virtual_companion.boundary.evidence import EvidenceTurn
from virtual_companion.boundary.memory import MemoryEffect
from virtual_companion.boundary.memory_support import command_digest
from virtual_companion.boundary.sqlite_business import SQLiteBusinessStore
from virtual_companion.cognition.errors import FactSupportUnresolved
from virtual_companion.cognition.fact_model import FactModel, extraction_schema


def _literal_ttl_extract(source, *, lifetime="short_term", verdict="supported",
                         expiry_ms=1791542809593, expected_support_expiry=None):
    calls = []

    def model(stage, messages, schema, cancelled):
        data = json.loads(messages[-1]["content"])
        calls.append((stage, data))
        if stage == "fact_extract":
            return {"operations": [{
                "operation": "add", "type": lifetime,
                "evidenceTurnId": "user-1", "evidenceQuote": source,
                "expiresAt": expiry_ms if lifetime == "short_term" else None,
                "supersedesAtomId": None,
                "factKey": {"subject": "user", "predicate": "本次放松方式"},
            }]}
        assert stage == "fact_support"
        if expected_support_expiry is not None:
            assert data["proposal"]["expiresAt"] == expected_support_expiry
        return {"intent": "assertion", "verdict": verdict}

    scope = Scope(A, "session")
    result = FactModel(model, Event(), bind_content=True).extract(
        scope=scope, batch_id="batch", now_us=1791174580000000,
        turn=EvidenceTurn(scope, "user-1", "user", source, len(source)), targets={},
    )
    return result, calls


def test_explicit_validity_deadline_is_bound_without_using_the_event_timestamp():
    source = ("2026-10-02T08:00:00Z那天我去散步；这次放松选散步；"
              "仅在 2026-10-08T16:06:49.593Z 之前有效。")
    (command,), calls = _literal_ttl_extract(source)
    assert command.expires_at == datetime(2026, 10, 8, 16, 6, 49, 593000, tzinfo=UTC)
    assert calls[-1][1]["proposal"]["expiresAt"] == 1791475609593
    assert [stage for stage, _ in calls] == ["fact_extract", "fact_support"]


def test_event_timestamp_alone_does_not_create_a_host_expiry():
    source = "2026-10-02T08:00:00Z那天我买了雨伞。"
    (command,), calls = _literal_ttl_extract(source, lifetime="long_term")
    assert command.expires_at is None
    assert calls[-1][1]["proposal"]["expiresAt"] is None
    with pytest.raises(FactSupportUnresolved, match="affirmative semantic support"):
        _literal_ttl_extract(source, verdict="unsupported", expected_support_expiry=1791542809593)


def test_ambiguous_explicit_validity_deadlines_remain_unresolved():
    source = ("这次放松选散步；仅在 2026-10-08T16:06:49.593Z 之前有效；"
              "有效至 2026-10-09T16:06:49.593Z。")
    with pytest.raises(FactSupportUnresolved, match="ambiguous"):
        _literal_ttl_extract(source)


def test_existing_relative_lifetime_keeps_its_independent_support_contract():
    source = "这次放松选散步，只在接下来五分钟有效。"
    relative_expiry_ms = 1791174580000 + 300000
    (command,), calls = _literal_ttl_extract(source, expiry_ms=relative_expiry_ms)
    assert command.expires_at == datetime.fromtimestamp(relative_expiry_ms / 1000, UTC)
    assert [stage for stage, _ in calls] == ["fact_extract", "fact_support"]


def test_extraction_schema_discriminates_add_correction_and_forget_targets():
    schema = extraction_schema("source", ("record-10", "record-100"))
    invalid = {"operations": [{**operation(
        operation="revoke", evidenceTurnId="source", evidenceQuote="更正，我不喜欢饮品100",
        content="更正，我不喜欢饮品100", supersedesAtomId=None,
    ), "factKey": {"subject": "user", "predicate": "drink-100"}}]}
    assert not Draft202012Validator(schema).is_valid(invalid)
    valid = {"operations": [{**operation(
        operation="supersede", evidenceTurnId="source", evidenceQuote="更正，我不喜欢饮品100",
        content="更正，我不喜欢饮品100", supersedesAtomId="record-100",
    ), "factKey": None}]}
    assert Draft202012Validator(schema).is_valid(valid)


class Recording:
    def __init__(self, *, intent="assertion", verdict="supported"):
        self.calls, self.intent, self.verdict = [], intent, verdict

    def __call__(self, stage, messages, schema, cancelled):
        self.calls.append((stage, json.loads(messages[-1]["content"])))
        if stage == "fact_extract":
            return {
                "operations": [
                    {
                        **operation(content="我喜欢茶"),
                        "factKey": {"subject": "user", "predicate": "drink"},
                    },
                ]
            }
        return {"intent": self.intent, "verdict": self.verdict}


def extract(recording):
    return FactModel(recording, Event()).extract(
        scope=Scope(A, "session"),
        batch_id="batch",
        now_us=100,
        turn=EvidenceTurn(Scope(A, "session"), "user-1", "user", "我喜欢茶", 4),
        targets={},
    )


def test_fact_model_support_binds_full_source_slot_and_operation():
    recording = Recording()
    (command,) = extract(recording)
    reason = json.loads(command.reason)
    assert command.fact_key.subject == "user"
    assert reason["support"]["command_sha256"] == command_digest(command, reason["quote"])
    assert recording.calls[1][1]["source"]["text"] == "我喜欢茶"
    assert recording.calls[1][1]["proposal"]["factKey"]["predicate"] == "drink"
    assert recording.calls[1][1]["target"] is None


def test_model_reported_speech_is_not_stored_in_place_of_original_fact():
    class Reported(Recording):
        def __call__(self, stage, messages, schema, cancelled):
            raw = super().__call__(stage, messages, schema, cancelled)
            if stage == "fact_extract":
                raw["operations"][0]["content"] = "用户提到喜欢茶"
            return raw

    recording = Reported()
    with pytest.raises(ValueError, match="verbatim"):
        extract(recording)
    assert [stage for stage, _ in recording.calls] == ["fact_extract"]


def test_bound_content_schema_lets_host_derive_only_verbatim_fact_text():
    schema = extraction_schema("user-1", ("old-drink",), bind_content=True)
    source = "更正，我现在不喜欢茶。请推荐别的饮品。"
    quote = "更正，我现在不喜欢茶"
    candidate = {**operation(operation="supersede", evidenceTurnId="user-1",
                             evidenceQuote=quote, supersedesAtomId="old-drink"),
                 "factKey": None}
    candidate.pop("content")
    assert Draft202012Validator(schema).is_valid({"operations": [candidate]})
    assert not Draft202012Validator(schema).is_valid(
        {"operations": [{**candidate, "content": "我喜欢茶"}]})

    target = {"record_id": "old-drink", "companion_id": A, "kind": "fact",
              "status": "active", "version": 1, "expires_at": None,
              "fact_key": {"subject": "user", "predicate": "drink", "scope": "profile"},
              "content": "我喜欢茶"}
    stages = []

    def model(stage, messages, supplied_schema, cancelled):
        stages.append(stage)
        if stage == "fact_extract":
            return {"operations": [candidate]}
        assert stage == "fact_support"
        data = json.loads(messages[-1]["content"])
        assert data["proposal"]["content"] == quote
        assert data["version"] == "fact-model.v7"
        return {"intent": "correction", "verdict": "supported"}

    (command,) = FactModel(model, Event(), bind_content=True).extract(
        scope=Scope(A, "session"), batch_id="batch", now_us=100,
        turn=EvidenceTurn(Scope(A, "session"), "user-1", "user", source, len(source)),
        targets={"old-drink": target},
    )
    assert command.kind == "correct" and command.content == quote
    assert json.loads(command.reason)["support"]["evaluator"] == "fact-model.v7"
    assert stages == ["fact_extract", "fact_support"]


def test_bound_fact_accepts_complete_contiguous_quote_across_clause_punctuation():
    source = "前天我提到的是《城南旧事》，作者是林海音。"
    candidate = {**operation(evidenceTurnId="user-1", evidenceQuote=source,
                             content=source),
                 "factKey": {"subject": "user", "predicate": "mentioned_book"}}
    candidate.pop("content")
    stages = []

    def model(stage, messages, schema, cancelled):
        stages.append(stage)
        if stage == "fact_extract":
            assert Draft202012Validator(schema).is_valid({"operations": [candidate]})
            return {"operations": [candidate]}
        assert stage == "fact_support"
        data = json.loads(messages[-1]["content"])
        assert data["quote"] == source
        assert data["proposal"]["content"] == source
        return {"intent": "assertion", "verdict": "supported"}

    (command,) = FactModel(model, Event(), bind_content=True).extract(
        scope=Scope(A, "session"), batch_id="batch", now_us=100,
        turn=EvidenceTurn(Scope(A, "session"), "user-1", "user", source, len(source)),
        targets={},
    )
    assert command.content == source
    assert json.loads(command.reason)["quote"] == source
    assert stages == ["fact_extract", "fact_support"]


def test_bound_fact_rejects_negation_stripped_quote_with_full_source_support():
    source = "我不喜欢茶"
    quote = "喜欢茶"
    candidate = {**operation(evidenceTurnId="user-1", evidenceQuote=quote,
                             content=quote),
                 "factKey": {"subject": "user", "predicate": "drink"}}
    candidate.pop("content")
    stages = []

    def model(stage, messages, schema, cancelled):
        stages.append(stage)
        if stage == "fact_extract":
            assert Draft202012Validator(schema).is_valid({"operations": [candidate]})
            return {"operations": [candidate]}
        assert stage == "fact_support"
        evidence = json.loads(messages[-1]["content"])
        assert evidence["quote"] == quote
        assert evidence["source"]["text"] == source
        return {"intent": "assertion", "verdict": "unsupported"}

    with pytest.raises(ValueError, match="affirmative semantic support"):
        FactModel(model, Event(), bind_content=True).extract(
            scope=Scope(A, "session"), batch_id="batch", now_us=100,
            turn=EvidenceTurn(Scope(A, "session"), "user-1", "user", source, len(source)),
            targets={},
        )
    assert stages == ["fact_extract", "fact_support"]


def test_bound_fact_correction_repair_preserves_host_quote_and_target():
    source = "更正，我现在不喜欢茶。"
    quote = "更正，我现在不喜欢茶"
    target = {"record_id": "old-drink", "companion_id": A, "kind": "fact",
              "status": "active", "version": 1, "expires_at": None,
              "fact_key": {"subject": "user", "predicate": "drink", "scope": "profile"},
              "content": "我喜欢茶"}
    stages = []

    def model(stage, messages, schema, cancelled):
        stages.append(stage)
        if stage in {"fact_extract", "fact_repair"}:
            action = "add" if stage == "fact_extract" else "supersede"
            candidate = {**operation(operation=action, evidenceTurnId="user-1",
                                     evidenceQuote=quote,
                                     supersedesAtomId=(None if action == "add"
                                                       else "old-drink")),
                         "factKey": ({"subject": "user", "predicate": "drink"}
                                     if action == "add" else None)}
            candidate.pop("content")
            assert Draft202012Validator(schema).is_valid({"operations": [candidate]})
            return {"operations": [candidate]}
        assert stage == "fact_support"
        proposal = json.loads(messages[-1]["content"])["proposal"]
        assert proposal["content"] == quote
        return {"intent": "correction", "verdict": "supported"}

    (command,) = FactModel(model, Event(), bind_content=True).extract(
        scope=Scope(A, "session"), batch_id="batch", now_us=100,
        turn=EvidenceTurn(Scope(A, "session"), "user-1", "user", source, len(source)),
        targets={"old-drink": target},
    )
    assert command.kind == "correct" and command.record_id == "old-drink"
    assert command.content == quote
    assert stages == ["fact_extract", "fact_support", "fact_repair", "fact_support"]


def test_bound_fact_schema_couples_lifetime_and_expiry():
    schema = extraction_schema("source", ("old-drink",), bind_content=True)
    base = {**operation(operation="supersede", evidenceTurnId="source",
                        evidenceQuote="我喜欢茶", supersedesAtomId="old-drink"),
            "factKey": None}
    base.pop("content")
    valid_long = {**base, "type": "long_term", "expiresAt": None}
    invalid_long = {**base, "type": "long_term", "expiresAt": 1000}
    valid_short = {**base, "type": "short_term", "expiresAt": 1000}
    invalid_short = {**base, "type": "short_term", "expiresAt": None}
    validator = Draft202012Validator(schema)
    assert validator.is_valid({"operations": [valid_long]})
    assert not validator.is_valid({"operations": [invalid_long]})
    assert validator.is_valid({"operations": [valid_short]})
    assert not validator.is_valid({"operations": [invalid_short]})


@pytest.mark.parametrize("intent,verdict", [("question", "supported"), ("assertion", "unknown")])
def test_non_assertion_or_unknown_support_cannot_create_fact(intent, verdict):
    with pytest.raises(ValueError):
        extract(Recording(intent=intent, verdict=verdict))


def test_correction_support_cannot_approve_an_add_operation():
    recording = Recording(intent="correction")
    with pytest.raises(ValueError, match="Correction cannot be added"):
        extract(recording)
    assert [stage for stage, _ in recording.calls] == ["fact_extract", "fact_support"]


def test_single_correction_add_gets_one_target_bound_repair_and_rejects_bad_repairs():
    source = "更正，我现在不喜欢茶"
    target = {"record_id": "old-drink", "companion_id": A, "kind": "fact",
              "status": "active", "version": 1, "expires_at": None,
              "fact_key": {"subject": "user", "predicate": "drink", "scope": "profile"},
              "content": "我喜欢茶"}

    class Repair:
        def __init__(self, mode="valid"):
            self.calls, self.mode = [], mode

        def __call__(self, stage, messages, schema, cancelled):
            self.calls.append(stage)
            if stage == "fact_extract":
                return {"operations": [{**operation(content=source, evidenceQuote=source,
                                                      evidenceTurnId="user-1"),
                                        "factKey": {"subject": "user", "predicate": "drink"}}]}
            if stage == "fact_repair":
                item_schema = schema["properties"]["operations"]["items"]
                assert item_schema["properties"]["operation"]["const"] == "supersede"
                if self.mode == "empty":
                    return {"operations": []}
                quote = "我养猫" if self.mode == "wrong_quote" else source
                return {"operations": [{**operation(
                    operation="supersede", content=quote, evidenceQuote=quote,
                    evidenceTurnId="user-1", supersedesAtomId="old-drink",
                ), "factKey": None,
                    **({"type": "short_term", "expiresAt": 10_000_000}
                       if self.mode == "new_expiry" else {})}]}
            return {"intent": "correction", "verdict": "supported"}

    model = Repair()
    (command,) = FactModel(model, Event()).extract(
        scope=Scope(A, "session"), batch_id="batch", now_us=100,
        turn=EvidenceTurn(Scope(A, "session"), "user-1", "user", source, len(source)),
        targets={"old-drink": target},
    )
    assert command.kind == "correct" and command.record_id == "old-drink"
    assert command.expected_record_version == 1
    assert command.content == source
    assert model.calls == ["fact_extract", "fact_support", "fact_repair", "fact_support"]
    for mode in ("empty", "wrong_quote", "wrong_slot", "new_expiry"):
        broken = Repair(mode)
        candidate_target = {**target}
        if mode == "wrong_slot":
            candidate_target["fact_key"] = {"subject": "user", "predicate": "pet",
                                            "scope": "profile"}
        with pytest.raises(ValueError):
            FactModel(broken, Event()).extract(
                scope=Scope(A, "session"), batch_id="batch", now_us=100,
                turn=EvidenceTurn(Scope(A, "session"), "user-1", "user", source,
                                  len(source)),
                targets={"old-drink": candidate_target},
            )
        assert broken.calls == ["fact_extract", "fact_support", "fact_repair"]


def test_truncated_user_source_rejected_before_model_call():
    recording = Recording()
    with pytest.raises(ValueError):
        FactModel(recording, Event()).extract(
            scope=Scope(A, "session"),
            batch_id="batch",
            now_us=100,
            turn=EvidenceTurn(Scope(A, "session"), "user-1", "user", "我喜欢茶是举例", 4),
            targets={},
        )
    assert recording.calls == []


def test_cancellation_after_extraction_prevents_support_calls():
    cancelled = Event()

    class Cancel(Recording):
        def __call__(self, stage, messages, schema, signal):
            raw = super().__call__(stage, messages, schema, signal)
            cancelled.set()
            return raw

    recording = Cancel()
    with pytest.raises(InterruptedError):
        FactModel(recording, cancelled).extract(
            scope=Scope(A, "session"),
            batch_id="batch",
            now_us=100,
            turn=EvidenceTurn(Scope(A, "session"), "user-1", "user", "我喜欢茶", 4),
            targets={},
        )
    assert [stage for stage, _ in recording.calls] == ["fact_extract"]


def test_model_correction_and_revocation_bind_the_actual_target_revision(tmp_path):
    class Dynamic(Recording):
        action = "add"

        def __call__(self, stage, messages, schema, cancelled):
            data = json.loads(messages[-1]["content"])
            self.calls.append((stage, data))
            if stage == "fact_extract":
                source = data["source"]
                target = data["targets"][0]["record_id"] if data["targets"] else None
                return {
                    "operations": [
                        {
                            **operation(
                                operation=self.action,
                                evidenceTurnId=source["id"],
                                evidenceQuote=source["text"],
                                supersedesAtomId=target,
                                content="" if self.action == "revoke" else source["text"],
                            ),
                            "factKey": {"subject": "user", "predicate": "drink"}
                            if self.action == "add"
                            else None,
                        }
                    ]
                }
            return {
                "verdict": "supported",
                "intent": "forget_request" if self.action == "revoke" else "assertion",
            }

    store = SQLiteBusinessStore(tmp_path / "facts.sqlite3", evidence_clock=lambda: 100)
    recording = Dynamic()
    session = None
    for index, (action, text) in enumerate(
        [
            ("add", "我喜欢茶"),
            ("supersede", "现在我喜欢咖啡"),
            ("revoke", "忘记我的饮品偏好"),
        ]
    ):
        recording.action = action
        item = store.accept(config(), f"input-{index}", text, session_id=session)
        session = item.session_id
        run = claim(store, item, now=100 + 100 * index)
        rows = store.recall(A, now_us=100)
        commands = FactModel(recording, Event()).extract(
            scope=Scope(A, session),
            batch_id=item.input_id,
            now_us=100,
            turn=EvidenceTurn(Scope(A, session), item.input_id, "user", text, len(text)),
            targets={row["record_id"]: {**row, "companion_id": A} for row in rows},
        )
        store.commit(
            replace(result(run), memories=tuple(MemoryEffect(c) for c in commands)),
            now_us=101 + 100 * index,
        )
        if action != "add":
            support = recording.calls[-1][1]
            assert support["target"]["version"] == index
            assert support["proposal"]["operation"] == action
    assert SQLiteBusinessStore(store.path).recall(A, now_us=100) == []
