from dataclasses import replace
from hashlib import sha256

import pytest
from test_rebuilt_presentation import approved
from test_rebuilt_turns import A, B, claim, config, result
from virtual_companion.boundary.contracts import Scope, VersionConflict
from virtual_companion.boundary.exchanges import (
    ExchangeCommand,
    ExchangeEvidence,
    ExchangeSupport,
    evaluate_exchange,
)
from virtual_companion.boundary.presentation import MessageUnit, Presentation
from virtual_companion.boundary.sqlite_exchanges import SQLiteExchangeStore
from virtual_companion.boundary.sqlite_harness import SQLiteHarnessStore


@pytest.fixture
def store(tmp_path):
    return SQLiteExchangeStore(tmp_path / "exchanges.sqlite3")


def user_command(store, key="request", text="能帮我想想周末怎么安排吗？"):
    item = store.accept(config(), key, text)
    if store.owner(A) is None:
        store.claim_owner(A, "writer", expected_epoch=0)
    return ExchangeCommand(
        Scope(A, item.session_id),
        key,
        key,
        "open",
        "assistant_response",
        text,
        ExchangeEvidence("user", item.input_id, text, len(text), sha256(text.encode()).hexdigest()),
        100,
        None,
        1000,
    )


def apply(store, command, **kwargs):
    return store.exchange(
        command,
        store.owner(A),
        expected_version=store.snapshot(A).write_version,
        now_us=kwargs.get("now_us", 100),
        support=ExchangeSupport(command.fingerprint, "synthetic-evaluator"),
    )


def question(store, displayed=0, *, item=None, text=None, quote="晚上想聊什么？"):
    item = item or store.accept(config(), "start", "你好")
    run = claim(store, item)
    text = text or "晚上想聊什么？也可以先休息。"
    value = Presentation(
        Scope(A, item.session_id),
        item.input_id,
        run.attempt_id,
        "candidate",
        1,
        text,
        (MessageUnit("one", "text", text),),
    )
    receipt = store.commit(replace(result(run, text), presentation=approved(value)), now_us=101)
    store.publish(A, receipt["delivery_id"])
    if displayed:
        store.cancel(A, item.input_id)
        store.acknowledge(
            A,
            receipt["delivery_id"],
            candidate_id=receipt["candidate_id"],
            revision=1,
            sha256=receipt["sha256"],
            sequence=1,
            displayed_end=displayed,
            stopped=True,
            presentation_sha256=receipt["presentation_sha256"],
        )
    return ExchangeCommand(
        Scope(A, item.session_id),
        "ask",
        "question",
        "open",
        "user_question",
        quote,
        ExchangeEvidence(
            "assistant",
            receipt["delivery_id"],
            quote,
            len(quote),
            sha256(text[: len(quote)].encode()).hexdigest(),
        ),
        100,
        None,
        1000,
    )


def test_unshown_question_cannot_become_pending(store):
    command = question(store)
    with pytest.raises(ValueError, match="confirmed"):
        apply(store, command)
    assert store.pending_exchanges(A, now_us=100) == []


def test_acknowledged_question_survives_stopped_tail_and_reopen(store):
    command = question(store, displayed=len("晚上想聊什么？"))
    apply(store, command)
    reopened = SQLiteExchangeStore(store.path)
    rows = reopened.pending_exchanges(A, now_us=100)
    assert [(r["direction"], r["content"]) for r in rows] == [("user_question", command.content)]
    assert reopened.recall(A, now_us=100) == []  # A question is not a user fact.


def test_both_directions_preserve_initiator_and_no_automatic_resolution(store):
    apply(store, question(store, displayed=len("晚上想聊什么？")))
    apply(store, user_command(store))
    rows = store.pending_exchanges(A, now_us=500)
    assert {r["direction"] for r in rows} == {"user_question", "assistant_response"}
    assert {r["initiator"] for r in rows} == {"assistant", "user"}
    assert all(r["status"] == "pending" for r in rows)
    assert store.pending_exchanges(B, now_us=100) == []


def test_user_can_defer_then_end_question_without_answer(store):
    original = question(store, displayed=len("晚上想聊什么？"))
    apply(store, original)
    source = user_command(store, "later", "先不聊这个了")
    deferred = replace(
        source,
        exchange_id="question",
        action="defer",
        direction="user_question",
        content=original.content,
        expected_revision=1,
    )
    apply(store, deferred)
    assert store.pending_exchanges(A, now_us=100)[0]["status"] == "deferred"
    ended = replace(deferred, command_id="end", action="close", expected_revision=2)
    apply(store, ended)
    assert store.pending_exchanges(A, now_us=100) == []
    assert [r["status"] for r in store.exchange_history(A, "question")] == [
        "pending",
        "deferred",
        "closed",
    ]


def test_wrong_actor_cannot_claim_question_answered(store):
    original = question(store, displayed=len("晚上想聊什么？"))
    apply(store, original)
    with pytest.raises(ValueError, match="responding"):
        apply(store, replace(original, command_id="wrong", action="resolve", expected_revision=1))


def test_resolution_requires_current_revision_and_bound_support(store):
    original = question(store, displayed=len("晚上想聊什么？"))
    apply(store, original)
    source = user_command(store, "answer", "想聊旅行")
    answer = replace(
        source,
        exchange_id="question",
        direction="user_question",
        action="resolve",
        content=original.content,
        expected_revision=1,
    )
    with pytest.raises(ValueError, match="support"):
        store.exchange(
            answer,
            store.owner(A),
            expected_version=store.snapshot(A).write_version,
            now_us=100,
            support=ExchangeSupport("0" * 64, "synthetic-evaluator"),
        )
    apply(store, answer)
    with pytest.raises(VersionConflict):
        apply(store, replace(answer, command_id="stale"))
    assert store.pending_exchanges(A, now_us=100) == []


def test_replay_is_idempotent_but_changed_payload_is_rejected(store):
    command = user_command(store)
    receipt = apply(store, command)
    version = store.snapshot(A).write_version
    assert apply(SQLiteExchangeStore(store.path), command) == receipt
    assert store.snapshot(A).write_version == version
    with pytest.raises(VersionConflict):
        apply(store, replace(command, content="另一件事"))


def test_shared_version_owner_expiry_and_source_fences(store):
    command = user_command(store)
    owner, version = store.owner(A), store.snapshot(A).write_version
    store.claim_owner(A, "next", expected_epoch=owner.epoch)
    with pytest.raises(VersionConflict):
        store.exchange(
            command,
            owner,
            expected_version=version,
            now_us=100,
            support=ExchangeSupport(command.fingerprint, "synthetic-evaluator"),
        )
    with pytest.raises(ValueError, match="source"):
        apply(store, replace(command, evidence=replace(command.evidence, source_id="missing")))
    apply(store, command)
    with pytest.raises(VersionConflict):
        store.exchange(
            replace(command, command_id="another", exchange_id="another"),
            store.owner(A),
            expected_version=version,
            now_us=100,
            support=ExchangeSupport(
                replace(command, command_id="another", exchange_id="another").fingerprint,
                "synthetic-evaluator",
            ),
        )
    assert store.pending_exchanges(A, now_us=1000) == []
    with pytest.raises(VersionConflict):
        apply(
            store,
            replace(command, command_id="expired", action="defer", expected_revision=1),
            now_us=1000,
        )


def test_source_session_and_hash_cannot_be_forged(store):
    command = user_command(store)
    for forged in (
        replace(command, scope=Scope(A, "foreign")),
        replace(command, evidence=replace(command.evidence, prefix_sha256="0" * 64)),
    ):
        with pytest.raises(ValueError, match="source"):
            apply(store, forged)


def test_existing_v8_database_is_not_implicitly_migrated(tmp_path):
    path = tmp_path / "old.sqlite3"
    SQLiteHarnessStore(path)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="not migrated"):
        SQLiteExchangeStore(path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("verdict", ["unknown", "unsupported", True])
def test_quote_alone_does_not_establish_semantic_support(store, verdict):
    command = user_command(store)
    with pytest.raises(ValueError, match="support"):
        evaluate_exchange(
            command,
            confirmed_text=command.evidence.quote,
            support=lambda _: {
                "verdict": verdict,
                "command_sha256": command.fingerprint,
                "evaluator": "synthetic-evaluator",
            },
        )


def test_semantic_support_is_bound_to_action_and_target(store):
    command = user_command(store)
    proof = evaluate_exchange(
        command,
        confirmed_text=command.evidence.quote,
        support=lambda proposal: {
            "verdict": "supported",
            "command_sha256": command.fingerprint,
            "evaluator": "synthetic-evaluator",
        },
    )
    assert proof.command_sha256 == command.fingerprint
    with pytest.raises(ValueError, match="support"):
        store.exchange(
            replace(command, content="unrelated"),
            store.owner(A),
            expected_version=store.snapshot(A).write_version,
            now_us=100,
            support=proof,
        )
