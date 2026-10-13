import json
import sqlite3
import subprocess
import sys
from contextlib import closing
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest
from virtual_companion.boundary.contracts import Scope
from virtual_companion.boundary.presentation import (
    MessageUnit,
    Presentation,
    PresentationPolicy,
    ReviewedPresentation,
    review,
)
from virtual_companion.boundary.sqlite_presentation import SQLitePresentationStore
from virtual_companion.boundary.turns import CognitiveResult
from virtual_companion.text.identity import CompanionConfig, VersionedReference

A = "00000000-0000-0000-0000-000000000001"


def candidate(units=None):
    return Presentation(
        Scope(A, "session"),
        "input",
        "attempt",
        "candidate",
        1,
        "你好。",
        units if units is not None else (MessageUnit("one", "text", "你好。"),),
    )


def approved(value, policy=None):
    policy = policy or PresentationPolicy(0, 100, 8, False)
    verdict = review(value, policy, "review", model_approved=True)
    return ReviewedPresentation(value, policy, verdict)


def test_actual_display_rejected_even_when_semantic_reply_and_model_approve():
    value = candidate((MessageUnit("one", "text", "你好！！！"),))
    policy = PresentationPolicy(0, 100, 8, False)
    verdict = review(value, policy, "review", model_approved=True)
    assert not verdict.passed
    assert "exclamation_budget" in verdict.issues
    with pytest.raises(ValueError):
        ReviewedPresentation(value, policy, verdict)


def test_forged_pass_cannot_bypass_hard_checks_or_model_rejection():
    value = candidate((MessageUnit("one", "text", "你好！！！"),))
    policy = PresentationPolicy(0, 100, 8, False)
    verdict = review(value, policy, "review", model_approved=True)
    with pytest.raises(ValueError):
        ReviewedPresentation(value, policy, replace(verdict, issues=()))
    clean = candidate()
    verdict = review(clean, policy, "review", model_approved=False)
    with pytest.raises(ValueError):
        ReviewedPresentation(clean, policy, verdict)


def test_resegment_same_flat_text_or_change_action_invalidates_review():
    original = candidate((MessageUnit("one", "text", "你好。\n\n再见。"),))
    bound = approved(original)
    split = replace(
        original, units=(MessageUnit("one", "text", "你好。"), MessageUnit("two", "text", "再见。"))
    )
    assert original.text == split.text
    assert original.fingerprint != split.fingerprint
    with pytest.raises(ValueError):
        replace(bound, candidate=split)
    with pytest.raises(ValueError):
        replace(
            bound, candidate=replace(original, units=(MessageUnit("one", "action", original.text),))
        )


def test_unit_boundaries_unicode_and_review_policy_are_bound():
    value = candidate((MessageUnit("one", "text", "你🙂"), MessageUnit("two", "text", "好")))
    assert value.text == "你🙂\n\n好"
    assert value.unit_end("one") == 2
    assert value.unit_end("two") == 5
    bound = approved(value)
    with pytest.raises(ValueError):
        replace(bound, policy=replace(bound.policy, max_exclamations=10))
    assert ReviewedPresentation.from_mapping(bound.to_mapping()) == bound


@pytest.mark.parametrize(
    "units,policy,issue",
    [
        (
            (MessageUnit("one", "action", "*点头*"),),
            PresentationPolicy(0, 100, 8, False),
            "actions_not_allowed",
        ),
        (
            (MessageUnit("one", "text", "太长的内容"),),
            PresentationPolicy(0, 3, 8, False),
            "unit_length",
        ),
        (
            (MessageUnit("one", "text", "好"), MessageUnit("two", "text", "啊")),
            PresentationPolicy(0, 100, 1, False),
            "unit_count",
        ),
    ],
)
def test_explicit_surface_policy_checks_every_unit(units, policy, issue):
    verdict = review(candidate(units), policy, "review", model_approved=True)
    assert issue in verdict.issues


@pytest.fixture
def store(tmp_path):
    return SQLitePresentationStore(tmp_path / "presentation.sqlite3")


def prepared(store):
    config = CompanionConfig(A, VersionedReference("persona", "1", "a" * 64))
    item = store.accept(config, "first", "你好")
    run = store.claim(A, item.input_id, "worker", now_us=100, lease_us=100)
    value = Presentation(
        Scope(A, item.session_id),
        item.input_id,
        run.attempt_id,
        "candidate",
        1,
        "你好。",
        (MessageUnit("one", "text", "你🙂"), MessageUnit("two", "text", "好")),
    )
    bound = approved(value)
    result = CognitiveResult(
        run,
        "candidate",
        1,
        value.text,
        "review",
        sha256(value.text.encode()).hexdigest(),
        presentation=bound,
    )
    return item, result


def acknowledge(store, receipt, **changes):
    args = dict(
        candidate_id=receipt["candidate_id"],
        revision=receipt["revision"],
        sha256=receipt["sha256"],
        presentation_sha256=receipt["presentation_sha256"],
        sequence=1,
        displayed_end=5,
    )
    args.update(changes)
    return store.acknowledge(A, receipt["delivery_id"], **args)


def test_commit_publish_ack_bind_exact_manifest_and_recover(store):
    item, result = prepared(store)
    receipt = store.commit(result, now_us=101)
    event = store.publish(A, receipt["delivery_id"])
    assert event["presentation"]["candidate"]["units"][1]["unit_id"] == "two"
    assert receipt["presentation_sha256"] == result.presentation.candidate.fingerprint
    with pytest.raises(ValueError):
        acknowledge(store, receipt, presentation_sha256=None)
    with pytest.raises(ValueError):
        acknowledge(store, receipt, presentation_sha256="0" * 64)
    assert acknowledge(store, receipt)["status"] == "completed"
    reopened = SQLitePresentationStore(store.path)
    assert reopened.commit(result, now_us=1000) == receipt
    assert acknowledge(reopened, receipt)["status"] == "completed"
    assert reopened.inspect(A, item.input_id)["status"] == "committed"


def test_new_adapter_does_not_allow_legacy_unreviewed_fallback(store):
    item, result = prepared(store)
    with pytest.raises(ValueError):
        store.commit(replace(result, presentation=None), now_us=101)
    assert store.inspect(A, item.input_id)["commit"] is None


def test_scoped_binding_rejects_altered_text_candidate_and_attempt(store):
    _, result = prepared(store)
    for changes in [
        dict(candidate_id="other"),
        dict(revision=2),
        dict(review_id="other"),
        dict(claim=replace(result.claim, attempt_id="other")),
    ]:
        with pytest.raises(ValueError):
            replace(result, **changes)
    with pytest.raises(ValueError):
        replace(result, text="different", reviewed_sha256=sha256(b"different").hexdigest())


def test_reviewed_units_cancel_and_memory_effects_compose(store):
    from test_rebuilt_memory import output_effect

    item, result = prepared(store)
    candidate = result.presentation.candidate
    effects = tuple(
        replace(
            output_effect(unit.unit_id, candidate.unit_end(unit.unit_id)),
            mutation=replace(output_effect(unit.unit_id).mutation, scope=candidate.scope),
        )
        for unit in candidate.units
    )
    result = replace(result, memories=effects)
    receipt = store.commit(result, now_us=101)
    assert store.recall(A, now_us=0) == []
    store.publish(A, receipt["delivery_id"])
    acknowledge(store, receipt, displayed_end=2)
    assert [r["record_id"] for r in store.recall(A, now_us=0)] == ["one"]
    store.cancel(A, item.input_id)
    acknowledge(store, receipt, displayed_end=2, sequence=2, stopped=True)
    assert store.effects(A, receipt["delivery_id"])[1]["status"] == "cancelled"
    assert store.delivery(A, receipt["delivery_id"])["displayed_text"] == "你🙂"


def test_zero_retry_budget_has_no_unreviewed_escape_and_ordinary_interjections_pass():
    # No retry loop or fallback is introduced by the boundary.
    value = candidate((MessageUnit("one", "text", "哈，哈，好的。"),))
    assert approved(value).review.passed
    invalid = replace(value, units=(MessageUnit("one", "text", "你好！！！"),))
    with pytest.raises(ValueError):
        approved(invalid)


def test_punctuation_budget_is_shared_across_units_and_actions():
    value = candidate((MessageUnit("one", "text", "好!"), MessageUnit("two", "action", "*点头！*")))
    verdict = review(value, PresentationPolicy(1, 100, 8, True), "review", model_approved=True)
    assert verdict.issues == ("exclamation_budget",)


def test_tampered_manifest_cannot_publish_or_emit_event(store):
    _, result = prepared(store)
    receipt = store.commit(result, now_us=101)
    with store.transaction() as db:
        changed = dict(receipt)
        changed["presentation"] = json.loads(json.dumps(receipt["presentation"]))
        changed["presentation"]["candidate"]["units"][0]["text"] = "坏！"
        db.execute("UPDATE turn_commits SET receipt=?", (json.dumps(changed),))
    with pytest.raises(ValueError):
        store.publish(A, receipt["delivery_id"])
    assert store.delivery(A, receipt["delivery_id"])["status"] == "pending"
    with store.transaction(write=False) as db:
        assert db.execute("SELECT count(*) FROM turn_publications").fetchone()[0] == 0


def test_recovered_publication_must_match_committed_manifest(store):
    _, result = prepared(store)
    receipt = store.commit(result, now_us=101)
    event = store.publish(A, receipt["delivery_id"])
    with store.transaction() as db:
        event["text"] = "被替换的输出"
        db.execute("UPDATE turn_publications SET event=?", (json.dumps(event),))
    with pytest.raises(ValueError):
        store.publication(A, receipt["delivery_id"])


def test_v3_explicit_upgrade_preserves_legacy_receipt_and_old_app_refuses_new_format(tmp_path):
    from test_rebuilt_memory import turn
    from virtual_companion.boundary.sqlite_memory import SQLiteMemoryStore

    path = tmp_path / "v3.sqlite3"
    old = SQLiteMemoryStore(path)
    _, proposal, receipt = turn(old)
    original_hash = sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        SQLitePresentationStore(path)
    assert sha256(path.read_bytes()).hexdigest() == original_hash
    backup = tmp_path / "backup.sqlite3"
    with closing(sqlite3.connect(path)) as source, closing(sqlite3.connect(backup)) as destination:
        source.backup(destination)
    new = SQLitePresentationStore(path, upgrade_v3=True)
    assert new.commit(proposal, now_us=999) == receipt
    assert new.publish(A, receipt["delivery_id"])["text"] == proposal.text
    with pytest.raises(ValueError):
        old.snapshot(A)
    assert SQLiteMemoryStore(backup).commit(proposal, now_us=999) == receipt


@pytest.mark.parametrize("version", [1, 2, 99])
def test_other_versions_rejected_without_changes(store, version):
    with closing(sqlite3.connect(store.path)) as db:
        db.execute(f"PRAGMA user_version={version}")
    before = sha256(store.path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        SQLitePresentationStore(store.path, upgrade_v3=True)
    assert sha256(store.path.read_bytes()).hexdigest() == before


def test_failed_version_upgrade_rolls_back_format_marker(tmp_path):
    from virtual_companion.boundary.sqlite_memory import SQLiteMemoryStore

    path = tmp_path / "v3.sqlite3"
    SQLiteMemoryStore(path).register(A)
    before = sha256(path.read_bytes()).hexdigest()

    class FailUpgrade(SQLitePresentationStore):
        def _upgrade(self, *args):
            super()._upgrade(*args)
            raise RuntimeError("injected after format marker update")

    with pytest.raises(RuntimeError):
        FailUpgrade(path, upgrade_v3=True)
    assert sha256(path.read_bytes()).hexdigest() == before
    assert SQLiteMemoryStore(path).snapshot(A).write_version == 0


def test_ack_rejects_inconsistent_text_digest_even_if_host_echoes_it(store):
    _, result = prepared(store)
    receipt = store.commit(result, now_us=101)
    store.publish(A, receipt["delivery_id"])
    changed = dict(receipt, sha256="0" * 64)
    with store.transaction() as db:
        db.execute("UPDATE turn_commits SET receipt=?", (json.dumps(changed),))
    with pytest.raises(ValueError):
        acknowledge(store, changed)


@pytest.mark.parametrize(
    "changes",
    [
        {"units": [MessageUnit("one", "text", "你好")]},
        {"units": (MessageUnit("same", "text", "一"), MessageUnit("same", "text", "二"))},
        {"revision": True},
        {"semantic_text": "好" * 22000},
    ],
)
def test_invalid_presentation_rejected(changes):
    with pytest.raises(ValueError):
        replace(candidate(), **changes)


_CHILD = """
import sys, os
sys.path.insert(0, sys.argv[1])
from contextlib import contextmanager
from virtual_companion.boundary.sqlite_presentation import SQLitePresentationStore
class CrashStore(SQLitePresentationStore):
    @contextmanager
    def transaction(self, **kwargs):
        with super().transaction(**kwargs) as db:
            db.create_function('crash', 0, lambda: os._exit(23))
            yield db
store = CrashStore(sys.argv[2])
store.publish(sys.argv[3], sys.argv[4])
os._exit(23)
"""


@pytest.mark.parametrize("point", ["before_event", "after_publication"])
def test_review_manifest_survives_publication_process_crash(store, point):
    _, result = prepared(store)
    receipt = store.commit(result, now_us=101)
    if point == "before_event":
        with store.transaction() as db:
            db.execute(
                "CREATE TRIGGER crash_publish BEFORE INSERT ON turn_publications "
                "BEGIN SELECT crash(); END"
            )
    child = subprocess.run(  # noqa: S603 - fixed isolated child with synthetic database
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            _CHILD,
            str(Path(__file__).resolve().parents[2] / "apps/backend/src"),
            str(store.path),
            A,
            receipt["delivery_id"],
        ],
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert child.returncode == 23, child.stderr.decode(errors="replace")
    reopened = SQLitePresentationStore(store.path)
    if point == "before_event":
        assert reopened.delivery(A, receipt["delivery_id"])["status"] == "pending"
        assert reopened.publication(A, receipt["delivery_id"]) is None
        with reopened.transaction() as db:
            db.execute("DROP TRIGGER crash_publish")
        reopened.publish(A, receipt["delivery_id"])
    assert (
        reopened.publication(A, receipt["delivery_id"])["presentation"]
        == result.presentation.to_mapping()
    )
    assert reopened.publish(A, receipt["delivery_id"]) is None
    assert acknowledge(reopened, receipt)["status"] == "completed"
