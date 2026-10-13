import json
from dataclasses import replace
from hashlib import sha256

import pytest
from test_rebuilt_turns import A, B, claim, config
from virtual_companion.boundary.contracts import Scope, VersionConflict
from virtual_companion.boundary.memory import FactKey
from virtual_companion.boundary.sqlite_lifecycle import SQLiteLifecycleStore
from virtual_companion.cognition.preparation import (
    CallCancelled,
    CallControl,
    CallDeadline,
    DisplayedTurn,
    MemoryEvidence,
    assemble_context,
    classify_difference,
)


def basis(tmp_path):
    store = SQLiteLifecycleStore(tmp_path / "context.db")
    persona = "合成人格：明确区分角色背景与共同经历。"
    cfg = config()
    cfg = replace(
        cfg, character=replace(cfg.character, sha256=sha256(persona.encode()).hexdigest())
    )
    item = store.accept(cfg, "input", "完整当前输入🙂")
    run = claim(store, item)
    return store, item, run, persona


def prepare(tmp_path, **options):
    store, item, run, persona = basis(tmp_path)
    return assemble_context(
        run, store.snapshot(A), item.input_id, "完整当前输入🙂", persona, now_us=100, **options
    )


def test_context_pins_identity_persona_state_and_current_input(tmp_path):
    context = prepare(tmp_path)
    assert context.current_text == "完整当前输入🙂"
    assert context.claim.companion_id == A
    assert context.fingerprint == sha256(context.payload.encode()).hexdigest()
    with pytest.raises(ValueError):
        replace(context, current_text="替换的当前输入")


def test_memory_context_preserves_fact_identity_without_promoting_events(tmp_path):
    fact = MemoryEvidence(
        A,
        "preference",
        1,
        "active",
        "input",
        "source",
        "喜欢茶",
        None,
        kind="fact",
        fact_key=FactKey("user", "drink_preference", "user"),
    )
    event = MemoryEvidence(
        A,
        "event",
        1,
        "active",
        "output",
        "reply",
        "助手提过茶",
        None,
        kind="event",
    )
    context = prepare(
        tmp_path,
        memories=(fact, event),
        recall_refs=(("preference", 1), ("event", 1)),
    )
    records = {row["record_id"]: row for row in json.loads(context.payload)["memories"]}
    assert records["preference"]["fact_key"] == {
        "subject": "user",
        "predicate": "drink_preference",
        "scope": "user",
    }
    assert records["event"]["kind"] == "event"
    assert records["event"]["fact_key"] is None
    with pytest.raises(ValueError):
        replace(event, kind="fact")
    with pytest.raises(ValueError):
        replace(fact, kind="event")
    with pytest.raises(ValueError):
        replace(fact, fact_key={"subject": "user"})


def test_persona_and_current_identity_mismatch_are_rejected(tmp_path):
    store, item, run, persona = basis(tmp_path)
    for input_id, text in [("other", persona), (item.input_id, "modified persona")]:
        with pytest.raises(ValueError):
            assemble_context(run, store.snapshot(A), input_id, "input", text, now_us=100)
    with pytest.raises(VersionConflict):
        assemble_context(
            run,
            replace(store.snapshot(A), write_version=1),
            item.input_id,
            "input",
            persona,
            now_us=100,
        )


def test_context_includes_only_confirmed_prefix_and_current_valid_memory(tmp_path):
    store, item, run, persona = basis(tmp_path)
    history = (
        DisplayedTurn(Scope(A, "old-session"), "old", "assistant", "你🙂未展示", 2),
        DisplayedTurn(Scope(A, "old-session"), "unknown", "assistant", "未知", None),
    )
    memories = (
        MemoryEvidence(A, "fact", 2, "active", "input", "source", "新事实", None),
        MemoryEvidence(A, "revoked", 1, "revoked", "input", "source", "撤销", None),
        MemoryEvidence(A, "expired", 1, "active", "input", "source", "过期", 100),
    )
    context = assemble_context(
        run,
        store.snapshot(A),
        item.input_id,
        "完整输入",
        persona,
        history=history,
        memories=memories,
        recall_refs=(("fact", 1), ("fact", 2), ("revoked", 1), ("expired", 1)),
        now_us=100,
    )
    payload = json.loads(context.payload)
    assert payload["history"][0]["text"] == "你🙂"
    assert "未展示" not in context.payload and "未知" not in context.payload
    assert payload["memories"][0]["content"] == "新事实"
    assert len(payload["memories"]) == 1
    assert ("fact", 1, "stale_version") in context.excluded
    assert ("unknown", 0, "unconfirmed_display") in context.excluded


def test_cross_companion_and_duplicate_history_rejected(tmp_path):
    for index, history in enumerate(
        [
            (DisplayedTurn(Scope(B, "session"), "old", "user", "other", 5),),
            (DisplayedTurn(Scope(A, "session"), "old", "user", "hi", 2),) * 2,
        ]
    ):
        path = tmp_path / str(index)
        path.mkdir()
        with pytest.raises(ValueError):
            prepare(path, history=history)


def test_lease_checkpoint_renews_original_attempt_and_cancellation_wins(tmp_path):
    store, _, run, _ = basis(tmp_path)
    now = [140]
    cancelled = [False]
    control = CallControl(
        run,
        clock=lambda: now[0],
        cancelled=lambda: cancelled[0],
        renew=lambda r, t: store.renew(r, now_us=t, lease_us=50),
        deadline_us=300,
        renewal_margin_us=15,
    )
    renewed = control.checkpoint()
    assert renewed.lease_until_us == 190
    assert renewed.attempt_id == run.attempt_id
    cancelled[0] = True
    with pytest.raises(CallCancelled):
        control.checkpoint()
    assert store.inspect(A, run.input_id)["status"] == "running"


@pytest.mark.parametrize(
    "now,deadline,error", [(150, 300, VersionConflict), (140, 140, CallDeadline)]
)
def test_expired_lease_and_deadline_never_renew(tmp_path, now, deadline, error):
    _, _, run, _ = basis(tmp_path)

    def forbidden(*args):
        pytest.fail("expired/cancelled calls must not renew")

    control = CallControl(
        run,
        clock=lambda: now,
        cancelled=lambda: False,
        renew=forbidden,
        deadline_us=deadline,
        renewal_margin_us=15,
    )
    with pytest.raises(error):
        control.checkpoint()


def test_allowed_difference_requires_exact_before_and_after():
    rule = {"id": "LA-06", "before": True, "after": False, "reason": "Actual surface rejected"}
    assert classify_difference(True, False, rule) == "intentional_fix"
    assert classify_difference(True, True, rule) == "equivalent"
    assert classify_difference(True, "different", rule) == "unjudged"
    assert classify_difference(1, False, rule) == "unjudged"


def test_six_duties_bind_context_and_actual_candidate_without_six_calls(tmp_path):
    from test_rebuilt_lifecycle import proposal
    from virtual_companion.cognition.preparation import DUTIES, DutyAssessment, validate_candidate

    store, item, run, persona = basis(tmp_path)
    context = assemble_context(
        run, store.snapshot(A), item.input_id, "当前输入", persona, now_us=100
    )
    result = proposal(run)
    assessments = tuple(
        DutyAssessment(
            duty,
            "complete",
            context.fingerprint,
            result.presentation.candidate.fingerprint,
            "recording",
            "shared-call",
        )
        for duty in DUTIES
    )
    assert validate_candidate(context, result, assessments, review_attempt=3) == result
    for records, attempt in [
        (assessments[:-1], 1),
        (assessments, 4),
        ((replace(assessments[0], context_sha256="0" * 64), *assessments[1:]), 1),
        ((replace(assessments[0], disposition="not_required"), *assessments[1:]), 1),
    ]:
        with pytest.raises(ValueError):
            validate_candidate(context, result, records, review_attempt=attempt)


def test_renewal_cannot_change_attempt_or_hide_cancellation(tmp_path):
    _, _, run, _ = basis(tmp_path)
    control = CallControl(
        run,
        clock=lambda: 140,
        cancelled=lambda: False,
        renew=lambda r, t: replace(r, attempt_id="other", lease_until_us=200),
        deadline_us=300,
        renewal_margin_us=15,
    )
    with pytest.raises(VersionConflict):
        control.checkpoint()
    stopped = [False]

    def renew(r, t):
        stopped[0] = True
        return replace(r, lease_until_us=200)

    control = CallControl(
        run,
        clock=lambda: 140,
        cancelled=lambda: stopped[0],
        renew=renew,
        deadline_us=300,
        renewal_margin_us=15,
    )
    with pytest.raises(CallCancelled):
        control.checkpoint()


def test_history_capacity_failure_does_not_silently_drop_current_input(tmp_path):
    history = (DisplayedTurn(Scope(A, "session"), "old", "user", "hi", 2),) * 257
    with pytest.raises(ValueError, match="capacity"):
        prepare(tmp_path, history=history)


def test_conflicting_config_and_memory_identity_fail_before_context(tmp_path):
    store, item, run, persona = basis(tmp_path)
    other = config(B)
    with pytest.raises(ValueError, match="configuration"):
        assemble_context(
            replace(run, config_json=json.dumps(other.attempt_binding({}))),
            store.snapshot(A),
            item.input_id,
            "input",
            persona,
            now_us=100,
        )
    memory = MemoryEvidence(B, "foreign", 1, "active", "input", "source", "other", None)
    with pytest.raises(ValueError, match="Cross-companion"):
        assemble_context(
            run, store.snapshot(A), item.input_id, "input", persona, memories=(memory,), now_us=100
        )
