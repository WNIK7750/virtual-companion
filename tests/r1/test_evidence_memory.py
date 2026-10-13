import json
from dataclasses import replace
from datetime import timedelta
from hashlib import sha256

import pytest
from test_rebuilt_memory import WHEN, A, B, apply, mutation
from virtual_companion.boundary.contracts import Scope, VersionConflict
from virtual_companion.boundary.memory import FactKey
from virtual_companion.boundary.sqlite_memory import SQLiteMemoryStore
from virtual_companion.cognition.evidence_memory import EvidenceTurn, compile_operations
from virtual_companion.text.chronology import utc_microseconds


@pytest.fixture
def store(tmp_path):
    adapter = SQLiteMemoryStore(tmp_path / "evidence.sqlite3")
    adapter.register(A)
    adapter.claim_owner(A, "worker", expected_epoch=0)
    return adapter


def operation(**changes):
    return {
        "operation": "add",
        "type": "long_term",
        "content": "喜欢茶",
        "evidenceTurnId": "user-1",
        "evidenceQuote": "我喜欢茶",
        "supersedesAtomId": None,
        "expiresAt": None,
        **changes,
    }


def compile_batch(operations, *, turns=None, targets=None, verdict="supported", scope=None):
    turns = turns or [EvidenceTurn(Scope(A, "session"), "user-1", "user", "我喜欢茶", 4)]
    checks = []

    def support(content, turn, quote):
        checks.append((content, turn, quote))
        return {
            "verdict": verdict,
            "evaluator": "recorded-synthetic.v1",
            "content_sha256": sha256(content.encode()).hexdigest(),
            "source_sha256": sha256(turn.text.encode()).hexdigest(),
        }

    result = compile_operations(
        json.dumps({"operations": operations}, ensure_ascii=False),
        scope=scope or Scope(A, "session"),
        batch_id="batch",
        now=WHEN,
        turns=turns,
        targets=targets or {},
        support=support,
        fact_keys={0: FactKey("user", "drink", "profile")},
    )
    return result


def test_supported_source_is_persisted_then_corrected_and_reopened(store):
    first = compile_batch([operation()])
    receipt = store.mutate_batch(
        first, store.owner(A), expected_version=0, now_us=utc_microseconds(WHEN)
    )
    assert receipt[0]["disposition"] == "applied"
    head = store.recall(A, now_us=utc_microseconds(WHEN))[0]
    assert json.loads(head["reason"])["quote"] == "我喜欢茶"
    new_turn = EvidenceTurn(Scope(A, "new-session"), "user-2", "user", "改了，我喜欢咖啡", 8)
    corrected = compile_batch(
        [
            operation(
                operation="supersede",
                content="喜欢咖啡",
                evidenceTurnId="user-2",
                evidenceQuote="我喜欢咖啡",
                supersedesAtomId=head["record_id"],
            )
        ],
        turns=[new_turn],
        targets={head["record_id"]: {**head, "companion_id": A}},
        scope=new_turn.scope,
    )
    # A different source batch must have its own durable idempotency identity.
    corrected = tuple(replace(m, mutation_id="correction-" + m.mutation_id) for m in corrected)
    store.mutate_batch(corrected, store.owner(A), expected_version=1, now_us=utc_microseconds(WHEN))
    reopened = SQLiteMemoryStore(store.path)
    assert (
        reopened.recall(A, now_us=utc_microseconds(WHEN), candidates=((head["record_id"], 1),))
        == []
    )
    assert reopened.recall(A, now_us=utc_microseconds(WHEN))[0]["content"] == "喜欢咖啡"
    assert len(reopened.history(A, head["record_id"])) == 2
    assert reopened.mutate_batch(
        corrected, reopened.owner(A), expected_version=1, now_us=utc_microseconds(WHEN)
    )
    assert len(reopened.history(A, head["record_id"])) == 2


@pytest.mark.parametrize("verdict", ["unsupported", "unknown"])
def test_present_quote_does_not_prove_the_proposed_fact(verdict):
    with pytest.raises(ValueError, match="support"):
        compile_batch([operation(content="讨厌茶")], verdict=verdict)


@pytest.mark.parametrize(
    "turn",
    [
        EvidenceTurn(Scope(B, "session"), "user-1", "user", "我喜欢茶", 4),
        EvidenceTurn(Scope(A, "session"), "user-1", "assistant", "我喜欢茶", 4),
        EvidenceTurn(Scope(A, "session"), "user-1", "user", "我喜欢茶", 1),
    ],
)
def test_wrong_scope_role_or_unconfirmed_quote_is_rejected(turn):
    with pytest.raises(ValueError):
        compile_batch([operation()], turns=[turn])


def test_batch_rejects_invalid_quote_and_duplicate_keys():
    with pytest.raises(ValueError):
        compile_batch([operation(), operation(evidenceQuote="虚构引文")])
    with pytest.raises(ValueError):
        compile_operations(
            '{"operations":[],"operations":[]}',
            scope=Scope(A, "session"),
            batch_id="x",
            now=WHEN,
            turns=[],
            targets={},
            support=None,
            fact_keys={},
        )
    assert compile_batch([]) == ()


def test_partial_database_batch_rolls_back_including_state_version(store):
    first = mutation(
        kind="assert_fact", record="drink", fact_key=FactKey("user", "drink", "profile")
    )
    conflict = replace(first, mutation_id="different", record_id="other")
    with pytest.raises(VersionConflict):
        store.mutate_batch(
            (first, conflict), store.owner(A), expected_version=0, now_us=utc_microseconds(WHEN)
        )
    assert store.recall(A, now_us=utc_microseconds(WHEN)) == []
    assert store.snapshot(A).write_version == 0
    assert store.memory_receipt(A, first.mutation_id) is None


def test_expired_correction_cannot_resurrect_fact_in_atomic_batch(store):
    first = mutation(
        kind="assert_fact",
        record="drink",
        fact_key=FactKey("user", "drink", "profile"),
        expires_at=WHEN + timedelta(seconds=1),
    )
    apply(store, first)
    correction = mutation("correction", kind="correct", record="drink", expected_record_version=1)
    with pytest.raises(VersionConflict, match="expired"):
        store.mutate_batch(
            (correction,),
            store.owner(A),
            expected_version=1,
            now_us=utc_microseconds(WHEN + timedelta(seconds=2)),
        )
    assert len(store.history(A, "drink")) == 1


def test_unknown_target_and_expiry_are_rejected_before_write():
    with pytest.raises(ValueError):
        compile_batch([operation(operation="supersede", supersedesAtomId="missing")])
    with pytest.raises(ValueError):
        compile_batch([operation(type="short_term", expiresAt=0)])


def test_subset_of_a_committed_batch_cannot_be_replayed_as_a_different_batch(store):
    first, second = mutation(), mutation("two", record="event-two")
    store.mutate_batch(
        (first, second), store.owner(A), expected_version=0, now_us=utc_microseconds(WHEN)
    )
    with pytest.raises(VersionConflict, match="batch replay"):
        store.mutate_batch(
            (first,), store.owner(A), expected_version=0, now_us=utc_microseconds(WHEN)
        )


def test_revoke_filters_reopened_and_stale_references(store):
    first = compile_batch([operation()])
    store.mutate_batch(first, store.owner(A), expected_version=0, now_us=utc_microseconds(WHEN))
    head = store.recall(A, now_us=utc_microseconds(WHEN))[0]
    source = EvidenceTurn(Scope(A, "session"), "revoke-source", "user", "忘掉这个偏好", 6)
    revoked = compile_batch(
        [
            operation(
                operation="revoke",
                content="",
                evidenceTurnId="revoke-source",
                evidenceQuote=source.text,
                supersedesAtomId=head["record_id"],
            )
        ],
        turns=[source],
        targets={head["record_id"]: {**head, "companion_id": A}},
    )
    revoked = tuple(replace(m, mutation_id="revoke-" + m.mutation_id) for m in revoked)
    store.mutate_batch(revoked, store.owner(A), expected_version=1, now_us=utc_microseconds(WHEN))
    fresh = SQLiteMemoryStore(store.path)
    assert fresh.recall(A, now_us=utc_microseconds(WHEN)) == []
    assert (
        fresh.recall(A, now_us=utc_microseconds(WHEN), candidates=((head["record_id"], 1),)) == []
    )
    assert fresh.history(A, head["record_id"])[-1]["status"] == "revoked"
