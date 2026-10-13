import json
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256

import pytest
from test_evidence_memory import operation
from test_rebuilt_turns import A, B, config
from virtual_companion.boundary.contracts import Scope, SemanticConflict, VersionConflict
from virtual_companion.boundary.memory import FactKey, MemoryMutation
from virtual_companion.boundary.memory_management import (
    ExplicitMemoryCommand,
    list_source_digest,
)
from virtual_companion.boundary.sqlite_background_worker import BackgroundClaim
from virtual_companion.boundary.sqlite_memory_management import SQLiteExplicitMemoryStore
from virtual_companion.cognition.background_worker import (
    UserFactWorker,
    _exact_repeat_completion,
)


def seed(store, companion=A, *, record="music", content="我喜欢安静的音乐", source="seed"):
    store.register(companion)
    owner = store.owner(companion) or store.claim_owner(companion, "seed-owner", expected_epoch=0)
    mutation = MemoryMutation(
        Scope(companion, "seed-session"),
        f"seed-{companion[-4:]}-{record}",
        "assert_fact",
        record,
        content,
        "manual",
        source,
        "synthetic seed",
        datetime(2026, 9, 21, tzinfo=UTC),
        FactKey("user", "music_preference", "profile"),
    )
    return store.mutate(
        mutation, owner, expected_version=store.snapshot(companion).write_version
    )


def list_command(
    *,
    operation="list-op",
    companion=A,
    action="forget",
    version=1,
    replacement="",
    resolves_job=None,
):
    source_digest = list_source_digest(
        action=action,
        record_id="music",
        expected_version=version,
        replacement=replacement,
    )
    return ExplicitMemoryCommand(
        Scope(companion, "memory-list"),
        operation,
        action,
        "music",
        version,
        "list",
        operation,
        source_digest,
        "single_record",
        "explicit_selection",
        "host-list.v1",
        source_digest,
        replacement=replacement,
        resolves_job=resolves_job,
    )


def test_list_forget_is_idempotent_cas_bound_and_persists_revocation(tmp_path):
    store = SQLiteExplicitMemoryStore(tmp_path / "memory.sqlite3")
    seed(store)
    command = list_command()

    receipt = store.execute_memory_command(command, now_us=10)

    assert receipt == store.execute_memory_command(command, now_us=11)
    assert receipt["status"] == "committed"
    assert receipt["action"] == "forget"
    assert store.recall(A, now_us=11) == []
    assert [row["status"] for row in store.history(A, "music")] == ["active", "revoked"]
    assert SQLiteExplicitMemoryStore(store.path).recall(A, now_us=12) == []

    changed = replace(command, record_id="different")
    with pytest.raises(ValueError, match="identity"):
        store.execute_memory_command(changed, now_us=12)


def test_stale_list_row_returns_version_changed_without_overwriting_new_value(tmp_path):
    store = SQLiteExplicitMemoryStore(tmp_path / "memory.sqlite3")
    seed(store)
    correction = list_command(
        operation="correct-op",
        action="correct",
        replacement="我现在更喜欢摇滚",
    )
    assert store.execute_memory_command(correction, now_us=10)["record_version"] == 2

    stale = list_command(operation="stale-forget", version=1)
    receipt = store.execute_memory_command(stale, now_us=11)

    assert receipt["status"] == "version_changed"
    assert receipt["current_record_version"] == 2
    assert store.execute_memory_command(stale, now_us=12) == receipt
    assert store.recall(A, now_us=12)[0]["content"] == "我现在更喜欢摇滚"


def test_repeated_desired_state_returns_no_change_without_another_revision(tmp_path):
    store = SQLiteExplicitMemoryStore(tmp_path / "memory.sqlite3")
    seed(store)
    replacement = "我现在更喜欢摇滚"
    first = list_command(
        operation="correct-first", action="correct", replacement=replacement
    )
    assert store.execute_memory_command(first, now_us=10)["record_version"] == 2

    repeated = list_command(
        operation="correct-again",
        action="correct",
        version=1,
        replacement=replacement,
    )
    receipt = store.execute_memory_command(repeated, now_us=11)

    assert receipt["status"] == "already_applied"
    assert receipt["record_version"] == 2
    assert len(store.history(A, "music")) == 2

    store.execute_memory_command(list_command(operation="forget", version=2), now_us=12)
    repeated_forget = list_command(operation="forget-again", version=2)
    assert store.execute_memory_command(repeated_forget, now_us=13)["status"] == (
        "already_applied"
    )
    assert len(store.history(A, "music")) == 3


def test_background_exact_repeat_closes_source_and_forget_invalidates_its_provenance(
    tmp_path,
):
    store = SQLiteExplicitMemoryStore(tmp_path / "memory.sqlite3")
    seed(store, source="original-source")
    source = "我喜欢安静的音乐"
    item = store.accept(config(), "repeat", source)
    stages = []

    def model(stage, messages, schema, cancelled):
        stages.append(stage)
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
            proposal = {
                **operation(
                    content=source,
                    evidenceQuote=source,
                    evidenceTurnId=item.input_id,
                ),
                "factKey": {"subject": "user", "predicate": "music_preference"},
            }
            proposal.pop("content")
            return {
                "operations": [proposal]
            }
        if stage == "fact_support":
            return {"intent": "assertion", "verdict": "supported"}
        if stage == "exchange_intent":
            return {
                "acts": [
                    {
                        "segmentId": 0,

                        "evidenceQuote": source,
                        "intent": "statement",
                        "targetId": None,
                    }
                ],
                "coverage": "complete",
            }
        if stage == "exchange_extract":
            return {"operations": []}
        raise AssertionError(stage)

    receipt = json.loads(UserFactWorker(
        store,
        model,
        clock=lambda: 10,
        memory_commands=True,
        resolution_protocol=True,
        resolution_strategy="intent_first",
    ).run_one(A, "worker"))

    assert receipt["no_change_operations"] == 1
    assert receipt["evaluator"] == "exact-repeat-host.v1"
    assert receipt["resolution"]["status"] == "complete_no_exchange"
    assert receipt["fact_resolution"]["evaluator"] == "exact-repeat-host.v1"
    assert receipt["fact_resolution"]["dispositions"] == ["accepted"]
    assert stages == ["memory_command"]
    assert len(store.history(A, "music")) == 1
    assert store.memory_receipt(A, f"{item.input_id}-operation-0") == {
        "companion_id": A,
        "mutation_id": f"{item.input_id}-operation-0",
        "record_id": "music",
        "requested_record_id": f"{item.input_id}-record-0",
        "disposition": "unchanged",
        "record_version": 1,
        "write_version": store.snapshot(A).write_version,
        "delivery_id": None,
    }
    assert store.memory_status(A) == "current"
    store.cancel(A, item.input_id)

    store.execute_memory_command(list_command(operation="forget-after-repeat"), now_us=11)

    invalidated = {row["source"] for row in store.invalidated_sources(A, record_id="music")}
    assert invalidated == {"original-source", item.input_id}
    assert store.recall(A, now_us=12) == []
    next_item = store.accept(
        config(), "next", "我们聊点别的。", session_id=item.session_id
    )
    run = store.claim(A, next_item.input_id, "foreground", now_us=13, lease_us=100)
    sources, _ = store.exchange_context(run)
    assert item.input_id not in {turn.source_id for turn in sources}


def test_exact_repeat_host_path_defers_to_semantic_processing_for_pending_exchange():
    source = "我喜欢安静的音乐"
    fact_key = {"subject": "user", "predicate": "music_preference", "scope": "profile"}
    target = {
        "companion_id": A,
        "record_id": "music",
        "version": 1,
        "content": source,
        "fact_key": fact_key,
    }

    class Store:
        @staticmethod
        def fact_targets(companion, *, now_us, candidates):
            assert (companion, now_us, candidates) == (A, 10, (("music", 1),))
            return {"music": target}

        @staticmethod
        def pending_exchanges(companion, *, now_us):
            assert (companion, now_us) == (A, 10)
            return [{"exchange_id": "pending-question"}]

        @staticmethod
        def background_exchange_context(claim):
            raise AssertionError("pending exchange must preserve semantic processing")

    claim = BackgroundClaim(
        A,
        "session",
        "job",
        "worker",
        0,
        "user",
        "repeat-source",
        sha256(source.encode()).hexdigest(),
        100,
        1,
    )
    catalog = ({
        "record_id": "music",
        "version": 1,
        "fact_key": fact_key,
        "content": source,
        "content_truncated": False,
    },)

    assert _exact_repeat_completion(Store(), claim, source, catalog, now_us=10) is None


def test_exact_repeat_host_path_rejects_ambiguous_or_truncated_catalog_matches():
    source = "我喜欢安静的音乐"
    claim = BackgroundClaim(
        A,
        "session",
        "job",
        "worker",
        0,
        "user",
        "repeat-source",
        sha256(source.encode()).hexdigest(),
        100,
        1,
    )

    class Store:
        @staticmethod
        def fact_targets(*_args, **_kwargs):
            raise AssertionError("ambiguous catalog must preserve semantic processing")

    common = {"version": 1, "content": source, "content_truncated": False}
    ambiguous = (
        {**common, "record_id": "music-a"},
        {**common, "record_id": "music-b"},
    )
    truncated = ({**common, "record_id": "music", "content_truncated": True},)

    assert _exact_repeat_completion(Store(), claim, source, ambiguous, now_us=10) is None
    assert _exact_repeat_completion(Store(), claim, source, truncated, now_us=10) is None


def test_list_command_rejects_tampered_selection_digest(tmp_path):
    store = SQLiteExplicitMemoryStore(tmp_path / "memory.sqlite3")
    seed(store)
    command = replace(list_command(), source_sha256="0" * 64)

    with pytest.raises(ValueError, match="selected record and action"):
        store.execute_memory_command(command, now_us=10)

    assert store.history(A, "music")[-1]["status"] == "active"


def test_conversation_forget_validates_exact_source_and_completes_barrier(tmp_path):
    store = SQLiteExplicitMemoryStore(tmp_path / "memory.sqlite3")
    seed(store)
    source = "请忘记我喜欢安静音乐这件事。"
    item = store.accept(config(), "forget-request", source)
    claim = store.claim_background(A, "worker", now_us=10, lease_us=100, role="user")
    command = ExplicitMemoryCommand(
        Scope(A, item.session_id),
        "conversation-forget",
        "forget",
        "music",
        1,
        "conversation",
        item.input_id,
        sha256(source.encode()).hexdigest(),
        "single_record",
        "pure_command",
        "memory-command.v1",
        sha256(b"conversation-forget").hexdigest(),
        source,
        0,
        len(source),
    )

    receipt = store.execute_memory_command(command, now_us=11, claim=claim)

    assert receipt["status"] == "committed"
    assert store.background_sources(A) == []
    assert store.memory_status(A) == "current"
    assert store.recall(A, now_us=12) == []
    assert {row["source"] for row in store.invalidated_sources(A)} == {
        "seed",
        item.input_id,
    }


def test_conversation_command_rejects_fabricated_quote_without_partial_commit(tmp_path):
    store = SQLiteExplicitMemoryStore(tmp_path / "memory.sqlite3")
    seed(store)
    source = "请忘记这条记忆。"
    item = store.accept(config(), "forget-request", source)
    claim = store.claim_background(A, "worker", now_us=10, lease_us=100, role="user")
    command = ExplicitMemoryCommand(
        Scope(A, item.session_id),
        "fabricated",
        "forget",
        "music",
        1,
        "conversation",
        item.input_id,
        sha256(source.encode()).hexdigest(),
        "single_record",
        "pure_command",
        "memory-command.v1",
        sha256(b"fabricated").hexdigest(),
        "忘记另一条",
        2,
        7,
    )

    with pytest.raises(ValueError, match="source"):
        store.execute_memory_command(command, now_us=11, claim=claim)
    assert store.history(A, "music")[-1]["status"] == "active"
    assert store.memory_barrier(A)


def test_explicit_operation_fences_late_worker_and_invalidated_source_recreation(tmp_path):
    store = SQLiteExplicitMemoryStore(tmp_path / "memory.sqlite3")
    seed(store, source="old-source")
    store.accept(config(), "late-source", "我还是喜欢安静音乐")
    claim = store.claim_background(A, "worker", now_us=10, lease_us=100, role="user")

    receipt = store.execute_memory_command(list_command(), now_us=11)

    assert receipt["protected_sources"] == 1
    with pytest.raises(VersionConflict, match="generation or source changed"):
        store.background_claim_current(claim, now_us=12)
    assert store.next_background_source(now_us=12) is None
    pending = store.background_sources(A)
    assert [(row["status"], row["error"]) for row in pending] == [
        ("failed", "needs_clarification.memory_generation_changed")
    ]
    reopened = SQLiteExplicitMemoryStore(store.path)
    assert reopened.memory_status(A) == "needs_clarification"
    assert reopened.next_background_source(now_us=13) is None
    assert reopened.memory_source_receipt(A, claim.job)["reason"] == (
        "needs_clarification.memory_generation_changed"
    )
    assert {row["source"] for row in store.invalidated_sources(A)} == {"old-source"}
    owner = store.owner(A)
    recreate = MemoryMutation(
        Scope(A, "new-session"),
        "late-recreate",
        "assert_fact",
        "music-again",
        "我喜欢安静的音乐",
        "background",
        "old-source",
        "late extraction",
        datetime(2026, 9, 21, tzinfo=UTC),
        FactKey("user", "music_preference_again", "profile"),
    )
    with pytest.raises(SemanticConflict, match="Invalidated source"):
        store.mutate(
            recreate, owner, expected_version=store.snapshot(A).write_version
        )

    abandoned = reopened.abandon_memory_source(A, claim.job, "abandon-late", now_us=14)
    assert abandoned["status"] == "committed"
    assert reopened.memory_status(A) == "current"
    assert reopened.recall(A, now_us=15) == []


def test_clarification_selection_or_abandon_legally_closes_protected_source(tmp_path):
    selected = SQLiteExplicitMemoryStore(tmp_path / "selected.sqlite3")
    seed(selected)
    selected.accept(config(), "ambiguous", "忘掉那个。")
    claim = selected.claim_background(A, "worker", now_us=10, lease_us=100, role="user")
    selected.protect_background(
        claim, error="needs_clarification.memory_command", now_us=11
    )
    command = list_command(operation="clarify-target", resolves_job=claim.job)
    assert selected.execute_memory_command(command, now_us=12)["status"] == "committed"
    assert selected.background_sources(A) == []
    assert selected.recall(A, now_us=13) == []
    with selected.transaction(write=False) as db:
        saved = db.execute(
            "SELECT receipt,error FROM background_work WHERE companion=? AND job=?",
            (A, claim.job),
        ).fetchone()
    assert saved["error"] is None
    assert '"disposition":"clarified"' in saved["receipt"]

    abandoned = SQLiteExplicitMemoryStore(tmp_path / "abandoned.sqlite3")
    seed(abandoned)
    abandoned.accept(config(), "ambiguous", "忘掉那个。")
    claim = abandoned.claim_background(A, "worker", now_us=10, lease_us=100, role="user")
    abandoned.protect_background(
        claim, error="needs_clarification.memory_command", now_us=11
    )
    receipt = abandoned.abandon_memory_source(A, claim.job, "abandon-op", now_us=12)
    assert receipt == abandoned.abandon_memory_source(
        A, claim.job, "abandon-op", now_us=13
    )
    assert receipt["action"] == "abandon"
    assert abandoned.background_sources(A) == []
    assert abandoned.recall(A, now_us=13)[0]["status"] == "active"
    assert SQLiteExplicitMemoryStore(abandoned.path).memory_status(A) == "current"


def test_abandoning_denial_quarantines_only_bound_old_fact_until_user_resolves_it(
    tmp_path,
):
    store = SQLiteExplicitMemoryStore(tmp_path / "memory.sqlite3")
    seed(store, source="old-music-source")
    owner = store.owner(A)
    store.mutate(
        MemoryMutation(
            Scope(A, "seed-session"),
            "seed-drink",
            "assert_fact",
            "drink",
            "我喜欢水",
            "manual",
            "drink-source",
            "synthetic seed",
            datetime(2026, 9, 21, tzinfo=UTC),
            FactKey("user", "drink_preference", "profile"),
        ),
        owner,
        expected_version=store.snapshot(A).write_version,
    )
    store.accept(config(), "denial", "我已经不喜欢安静的音乐了。")
    claim = store.claim_background(A, "worker", now_us=10, lease_us=100, role="user")
    protected = store.protect_memory_source(
        claim,
        reason="needs_clarification.fact_target_conflict",
        interpretation_sha256=sha256(b"denial-conflict").hexdigest(),
        action="unclear",
        requested_scope="single_record",
        unresolved=("support",),
        affected_records=(("music", 1),),
        now_us=11,
    )
    assert protected["affected_records"] == [["music", 1]]
    assert store.recall(A, now_us=12) == []

    receipt = store.abandon_memory_source(A, claim.job, "abandon-denial", now_us=13)

    assert receipt["suppressed_records"] == [["music", 1]]
    assert store.memory_status(A) == "current"
    assert [row["record_id"] for row in store.recall(A, now_us=14)] == ["drink"]
    assert [row["record_id"] for row in store.recall(A, now_us=14, limit=1)] == [
        "drink"
    ]
    memories = {row["record_id"]: row for row in store.memory_records(A)["rows"]}
    assert memories["music"]["status"] == "suppressed"
    assert memories["drink"]["status"] == "active"
    reopened = SQLiteExplicitMemoryStore(store.path)
    assert [row["record_id"] for row in reopened.recall(A, now_us=15)] == ["drink"]

    correction = list_command(
        operation="resolve-suppressed",
        action="correct",
        replacement="我现在喜欢摇滚",
    )
    assert reopened.execute_memory_command(correction, now_us=16)["status"] == "committed"
    assert {row["record_id"]: row["content"] for row in reopened.recall(A, now_us=17)} == {
        "music": "我现在喜欢摇滚",
        "drink": "我喜欢水",
    }


def test_target_revision_change_after_support_requeues_instead_of_becoming_semantic_failure(
    tmp_path,
):
    store = SQLiteExplicitMemoryStore(tmp_path / "memory.sqlite3")
    seed(store)
    source = "更正，我现在喜欢摇滚"
    item = store.accept(config(), "correction", source)

    def model(stage, messages, schema, cancelled):
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
        if stage == "fact_target_select":
            return {"selected": ["music"], "uncertain": False, "intent": "correction"}
        if stage == "fact_extract":
            proposal = {
                **operation(
                    operation="supersede",
                    evidenceTurnId=item.input_id,
                    evidenceQuote=source,
                    supersedesAtomId="music",
                ),
                "factKey": None,
            }
            proposal.pop("content")
            return {"operations": [proposal]}
        if stage == "fact_support":
            return {"intent": "correction", "verdict": "supported"}
        if stage == "exchange_intent":
            return {
                "acts": [
                    {
                        "segmentId": 0,

                        "evidenceQuote": source,
                        "intent": "statement",
                        "targetId": None,
                    }
                ],
                "coverage": "complete",
            }
        if stage == "exchange_extract":
            store.mutate(
                MemoryMutation(
                    Scope(A, "other-session"),
                    "concurrent-correction",
                    "correct",
                    "music",
                    "并发改成喜欢水",
                    "manual",
                    "concurrent-source",
                    "concurrent",
                    datetime(2026, 9, 21, tzinfo=UTC),
                    expected_record_version=1,
                ),
                store.owner(A),
                expected_version=store.snapshot(A).write_version,
            )
            return {"operations": []}
        raise AssertionError(stage)

    result = UserFactWorker(
        store,
        model,
        clock=lambda: 10,
        memory_commands=True,
        target_selection=True,
        resolution_protocol=True,
        resolution_strategy="intent_first",
    ).run_one(A, "worker")

    assert result is None
    pending = store.background_sources(A)[0]
    assert pending["status"] == "queued" and pending["attempts"] == 0
    assert store.history(A, "music")[-1]["content"] == "并发改成喜欢水"


def test_list_action_does_not_overwrite_unrelated_clarification_receipt(tmp_path):
    store = SQLiteExplicitMemoryStore(tmp_path / "memory.sqlite3")
    seed(store)
    store.accept(config(), "ambiguous", "忘掉那个。")
    claim = store.claim_background(A, "worker", now_us=10, lease_us=100, role="user")
    original = store.protect_memory_source(
        claim,
        reason="needs_clarification.memory_command",
        interpretation_sha256=sha256(b"ambiguous").hexdigest(),
        action="unclear",
        requested_scope="none",
        unresolved=("target",),
        now_us=11,
    )

    store.execute_memory_command(list_command(), now_us=12)

    assert store.memory_source_receipt(A, claim.job) == original
    assert store.background_sources(A)[0]["error"] == (
        "needs_clarification.memory_command"
    )


def test_memory_command_clarification_receipt_survives_reopen(tmp_path):
    store = SQLiteExplicitMemoryStore(tmp_path / "memory.sqlite3")
    store.accept(config(), "mixed", "忘记音乐偏好，顺便推荐一首歌。")
    claim = store.claim_background(A, "worker", now_us=10, lease_us=100, role="user")
    digest = sha256(b"mixed-interpretation").hexdigest()

    receipt = store.protect_memory_source(
        claim,
        reason="needs_clarification.mixed_source",
        interpretation_sha256=digest,
        action="forget",
        requested_scope="single_record",
        unresolved=("mixed_content",),
        now_us=11,
    )

    assert receipt["status"] == "needs_clarification"
    assert receipt["interpretation_sha256"] == digest
    reopened = SQLiteExplicitMemoryStore(store.path)
    assert reopened.memory_source_receipt(A, claim.job) == receipt
    assert reopened.memory_status(A) == "needs_clarification"
    assert reopened.next_background_source(now_us=12) is None


def test_forget_removes_source_from_model_projection_but_keeps_visible_history(tmp_path):
    store = SQLiteExplicitMemoryStore(tmp_path / "memory.sqlite3")
    text = "我喜欢安静的音乐"
    first = store.accept(config(), "fact-source", text)
    claim = store.claim_background(A, "worker", now_us=10, lease_us=100, role="user")
    store.protect_background(
        claim, error="needs_clarification.memory_command", now_us=11
    )
    store.abandon_memory_source(A, claim.job, "seed-source-resolved", now_us=12)
    store.cancel(A, first.input_id)
    seed(store, source=first.input_id)

    store.execute_memory_command(list_command(), now_us=13)
    current = store.accept(
        config(), "next-input", "我们聊点别的。", session_id=first.session_id
    )
    run = store.claim(A, current.input_id, "foreground", now_us=14, lease_us=100)
    sources, _ = store.exchange_context(run)

    assert [source.source_id for source in sources] == [current.input_id]
    assert store.inspect(A, first.input_id)["text"] == text
    assert store.recall(A, now_us=15) == []


def test_memory_command_never_crosses_companion_scope(tmp_path):
    store = SQLiteExplicitMemoryStore(tmp_path / "memory.sqlite3")
    seed(store, A)
    seed(store, B)

    store.execute_memory_command(list_command(companion=A), now_us=10)

    assert store.recall(A, now_us=11) == []
    assert store.recall(B, now_us=11)[0]["content"] == "我喜欢安静的音乐"


def test_effective_history_exclusions_are_scoped_and_keep_current_revision(tmp_path):
    store = SQLiteExplicitMemoryStore(tmp_path / "memory.sqlite3")
    seed(store, A, source="original-source")
    seed(store, B, source="original-source")
    owner = store.owner(A)
    store.mutate(
        MemoryMutation(
            Scope(A, "seed-session"), "correction", "correct", "music",
            "我更喜欢摇滚", "manual", "corrected-source", "synthetic correction",
            datetime(2026, 9, 22, tzinfo=UTC), expected_record_version=1,
        ),
        owner, expected_version=store.snapshot(A).write_version,
    )
    assert store.ineffective_history_sources(A, now_us=12) == {"original-source"}
    assert store.ineffective_history_sources(B, now_us=12) == set()
    reopened = SQLiteExplicitMemoryStore(store.path)
    assert reopened.ineffective_history_sources(A, now_us=12) == {"original-source"}


def test_invalid_fact_filters_associated_ack_without_erasing_confirmed_prefix(tmp_path):
    from test_rebuilt_turns import claim
    from test_segment_store import reviewed
    from test_stream_history import confirm

    store = SQLiteExplicitMemoryStore(tmp_path / "memory.sqlite3")
    item = store.accept(config(), "first", "我喜欢安静的音乐")
    run = claim(store, item)
    store.approve_segment(run, reviewed(run), now_us=100)
    event = store.dispatch_segment(run, 1, now_us=100)
    confirm(store, run, event)
    store.finish_stream(run, outcome="failed", now_us=100)
    acknowledged = next(s["source"] for s in store.background_sources(A)
                        if s["role"] == "assistant")
    seed(store, source=item.input_id)
    store.mutate(
        MemoryMutation(
            Scope(A, item.session_id), "correction", "correct", "music",
            "我更喜欢摇滚", "manual", "corrected-source", "synthetic",
            datetime(2026, 9, 22, tzinfo=UTC), expected_record_version=1,
        ),
        store.owner(A), expected_version=store.snapshot(A).write_version,
    )
    reopened = SQLiteExplicitMemoryStore(store.path)
    excluded = reopened.ineffective_history_sources(A, now_us=101)
    assert {item.input_id, acknowledged} <= excluded
    evidence = reopened.exchange_sources(
        Scope(A, item.session_id), references=(("assistant", acknowledged),)
    )
    assert evidence[0].text == "我在这里。"
    assert reopened.segments(A, item.input_id)[0]["status"] == "confirmed"
