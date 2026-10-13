import json
from dataclasses import replace
from hashlib import sha256

import pytest
from test_pending_exchanges import apply, question, user_command
from test_rebuilt_turns import A, B, config
from virtual_companion.boundary.contracts import Scope, canonical
from virtual_companion.boundary.sqlite_exchanges import SQLiteExchangeStore
from virtual_companion.cognition.evidence_memory import EvidenceTurn
from virtual_companion.cognition.exchange_extraction import compile_exchanges


@pytest.fixture
def store(tmp_path):
    return SQLiteExchangeStore(tmp_path / "extraction.sqlite3")


def operation(**changes):
    return {
        "operation": "add",
        "type": "open_loop",
        "content": "周末想聊什么？",
        "evidenceTurnId": "assistant",
        "evidenceQuote": "周末想聊什么？",
        "supersedesAtomId": None,
        "expiresAt": None,
        **changes,
    }


def compile_batch(operations, *, turns=None, targets=None, support=None):
    return compile_exchanges(
        json.dumps({"operations": operations}, ensure_ascii=False),
        scope=Scope(A, "session"),
        batch_id="batch",
        now_us=100,
        turns=turns
        if turns is not None
        else [
            EvidenceTurn(
                Scope(A, "session"), "assistant", "assistant", "周末想聊什么？未展示尾巴", 7
            )
        ],
        targets=targets or {},
        support=support
        or (
            lambda command: {
                "verdict": "supported",
                "evaluator": "synthetic",
                "command_sha256": sha256(canonical(command["command"]).encode()).hexdigest(),
            }
        ),
    )


def test_both_directions_compile_without_invented_ttl_or_unshown_tail():
    seen = []

    def support(command):
        seen.append(command)
        return {
            "verdict": "supported",
            "evaluator": "synthetic",
            "command_sha256": sha256(canonical(command["command"]).encode()).hexdigest(),
        }

    commands = compile_batch([operation()], support=support)
    command, proof = commands[0]
    assert command.direction == "user_question" and command.expires_us is None
    assert "未展示尾巴" not in canonical(seen)
    assert command.fingerprint == proof.command_sha256
    user = EvidenceTurn(Scope(A, "session"), "user", "user", "帮我分析一下", 6)
    response = compile_batch(
        [
            operation(
                type="response_obligation",
                evidenceTurnId="user",
                evidenceQuote=user.text,
                content=user.text,
            )
        ],
        turns=[user],
    )
    assert response[0][0].direction == "assistant_response"


@pytest.mark.parametrize(
    "change",
    [
        {"evidenceQuote": "未展示尾巴"},
        {"extra": True},
        {"type": "long_term"},
        {"expiresAt": True},
        {"operation": "resolve"},
        {"content": ""},
    ],
)
def test_invalid_operation_rejects_whole_batch(change):
    with pytest.raises(ValueError):
        compile_batch([operation(), operation(**change)])


def test_foreign_or_duplicate_source_is_rejected():
    turn = EvidenceTurn(Scope(B, "session"), "assistant", "assistant", "周末想聊什么？", 7)
    for turns in ([turn], [replace(turn, scope=Scope(A, "session"))] * 2):
        with pytest.raises(ValueError):
            compile_batch([operation()], turns=turns)


def test_resolution_binds_current_target_and_support_verdict():
    user = EvidenceTurn(Scope(A, "session"), "user", "user", "聊旅行吧", 4)
    target = {
        "companion_id": A,
        "exchange_id": "old",
        "revision": 3,
        "direction": "user_question",
        "status": "pending",
        "content": "周末想聊什么？",
        "expires_us": None,
    }
    ops = [
        operation(
            operation="resolve",
            supersedesAtomId="old",
            content=None,
            type=None,
            evidenceTurnId="user",
            evidenceQuote=user.text,
        )
    ]
    command, _ = compile_batch(ops, turns=[user], targets={"old": target})[0]
    assert command.expected_revision == 3 and command.content == target["content"]
    for bad in (
        {**target, "companion_id": B},
        {**target, "status": "closed"},
        {**target, "expires_us": 100},
    ):
        with pytest.raises(ValueError):
            compile_batch(ops, turns=[user], targets={"old": bad})
    with pytest.raises(ValueError, match="support"):
        compile_batch(
            ops, turns=[user], targets={"old": target}, support=lambda _: {"verdict": "unknown"}
        )


def test_assistant_cannot_resolve_its_own_question():
    target = {
        "companion_id": A,
        "exchange_id": "old",
        "revision": 1,
        "direction": "user_question",
        "status": "pending",
        "content": "周末想聊什么？",
        "expires_us": None,
    }
    with pytest.raises(ValueError, match="responding"):
        compile_batch(
            [operation(operation="resolve", type=None, content=None, supersedesAtomId="old")],
            targets={"old": target},
        )


def test_actual_store_response_requires_ack_and_can_be_completed(store):
    request = user_command(store)
    apply(store, request)
    item = store.accept(config(), "request", request.content)
    text = "周六可以散步，周日留给休息。"
    reply = question(store, displayed=len(text), item=item, text=text, quote=text)
    resolution = replace(
        reply,
        command_id="reply",
        exchange_id=request.exchange_id,
        action="resolve",
        direction="assistant_response",
        content=request.content,
        expected_revision=1,
    )
    apply(store, resolution)
    assert store.pending_exchanges(A, now_us=100) == []
    assert store.exchange_history(A, request.exchange_id)[-1]["status"] == "resolved"


def test_acknowledged_assistant_promise_preserves_assistant_initiator(store):
    text = "我先整理几个周末安排，稍后接着聊。"
    promise = replace(
        question(store, displayed=len(text), text=text, quote=text), direction="assistant_response"
    )
    apply(store, promise)
    assert store.pending_exchanges(A, now_us=100)[0]["initiator"] == "assistant"


def test_compiled_batch_reopens_and_replays_as_one_unit(store):
    from virtual_companion.boundary.exchanges import ExchangeSupport

    command = user_command(store)
    other = replace(command, command_id="other", exchange_id="other")
    batch = tuple((c, ExchangeSupport(c.fingerprint, "synthetic")) for c in (command, other))
    owner, version = store.owner(A), store.snapshot(A).write_version
    receipts = store.exchange_batch(batch, owner, expected_version=version, now_us=100)
    assert len(receipts) == 2
    assert store.snapshot(A).write_version == version + 2
    reopened = type(store)(store.path)
    assert reopened.exchange_batch(batch, owner, expected_version=version, now_us=100) == receipts
    assert len(reopened.pending_exchanges(A, now_us=100)) == 2


def test_batch_rolls_back_first_write_if_second_source_is_invalid(store):
    from virtual_companion.boundary.exchanges import ExchangeSupport

    command = user_command(store)
    bad = replace(
        command,
        command_id="bad",
        exchange_id="bad",
        evidence=replace(command.evidence, source_id="missing"),
    )
    batch = tuple((c, ExchangeSupport(c.fingerprint, "synthetic")) for c in (command, bad))
    version = store.snapshot(A).write_version
    with pytest.raises(ValueError, match="source"):
        store.exchange_batch(batch, store.owner(A), expected_version=version, now_us=100)
    assert store.snapshot(A).write_version == version
    assert store.exchange_history(A, command.exchange_id) == []


def test_batch_replay_cannot_mix_an_old_command_with_new_effect(store):
    from virtual_companion.boundary.contracts import VersionConflict
    from virtual_companion.boundary.exchanges import ExchangeSupport

    command = user_command(store)
    proof = ExchangeSupport(command.fingerprint, "synthetic")
    store.exchange_batch(((command, proof),), store.owner(A), expected_version=0, now_us=100)
    another = replace(command, command_id="new", exchange_id="new")
    with pytest.raises(VersionConflict, match="batch"):
        store.exchange_batch(
            ((command, proof), (another, ExchangeSupport(another.fingerprint, "synthetic"))),
            store.owner(A),
            expected_version=1,
            now_us=100,
        )
    assert len(store.pending_exchanges(A, now_us=100)) == 1


def test_stale_selected_revision_cannot_restore_closed_exchange(store):
    command = user_command(store)
    apply(store, command)
    refs = ((command.exchange_id, 1),)
    assert len(store.pending_exchanges(A, now_us=100, candidates=refs)) == 1
    apply(store, replace(command, action="close", command_id="withdraw", expected_revision=1))
    assert store.pending_exchanges(A, now_us=100, candidates=refs) == []


def test_two_concurrent_writers_cannot_commit_from_same_basis(store):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    from virtual_companion.boundary.contracts import VersionConflict
    from virtual_companion.boundary.exchanges import ExchangeSupport

    command = user_command(store)
    owner, version, barrier = store.owner(A), store.snapshot(A).write_version, Barrier(2)

    def write(key):
        c = replace(command, command_id=key, exchange_id=key)
        barrier.wait()
        try:
            type(store)(store.path).exchange(
                c,
                owner,
                expected_version=version,
                now_us=100,
                support=ExchangeSupport(c.fingerprint, "synthetic"),
            )
            return "applied"
        except VersionConflict:
            return "stale"

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(write, ["one", "two"])) == ["applied", "stale"]
    assert store.snapshot(A).write_version == version + 1


def test_semantic_support_sees_negation_outside_extracted_quote():
    text = "我不会说‘稍后帮你整理’，因为我现在就能回答。"
    turn = EvidenceTurn(Scope(A, "session"), "assistant", "assistant", text, len(text))
    seen = []

    def reject(request):
        seen.append(request)
        return {"verdict": "unsupported"}

    with pytest.raises(ValueError, match="support"):
        compile_batch(
            [
                operation(
                    type="response_obligation", content="稍后帮你整理", evidenceQuote="稍后帮你整理"
                )
            ],
            turns=[turn],
            support=reject,
        )
    assert seen[0]["confirmed_text"] == text


def test_pending_content_cannot_invent_an_answer_even_if_support_would_approve():
    with pytest.raises(ValueError, match="verbatim"):
        compile_batch([operation(content="去公园和博物馆吧")])
