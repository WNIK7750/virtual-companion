import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from test_joint_source_proposal import question_store
from test_rebuilt_memory import WHEN, apply, mutation
from test_rebuilt_turns import A, claim, config
from test_short_dialogue import Model
from virtual_companion.boundary.contracts import VersionConflict
from virtual_companion.boundary.memory import FactKey
from virtual_companion.cognition.background_worker import UserFactWorker
from virtual_companion.cognition.short_context import ShortContextAdmission
from virtual_companion.cognition.short_dialogue import ShortRequest
from virtual_companion.cognition.short_kernel import ShortKernel


@pytest.mark.parametrize("change", ["none", "expiry", "correct", "revoke"])
def test_effective_history_does_not_restore_invalid_fact_sources(tmp_path, change):
    from virtual_companion.boundary.contracts import Scope
    from virtual_companion.boundary.memory import MemoryMutation
    from virtual_companion.boundary.sqlite_memory_management import SQLiteExplicitMemoryStore
    from virtual_companion.text.chronology import utc_microseconds

    store = SQLiteExplicitMemoryStore(tmp_path / "memory.db")
    first = store.accept(config(), "original", "我这周住在上海")
    store.cancel(A, first.input_id)
    unrelated = store.accept(config(), "unrelated", "聊聊电影吧", session_id=first.session_id)
    store.cancel(A, unrelated.input_id)
    owner = store.owner(A) or store.claim_owner(A, "seed", expected_epoch=0)
    when = datetime(2026, 9, 27, tzinfo=UTC)
    seed = MemoryMutation(
        Scope(A, first.session_id), "seed", "assert_fact", "home", "我这周住在上海",
        "manual", first.input_id, "synthetic", when,
        FactKey("user", "home", "profile"), expires_at=when + timedelta(minutes=1),
    )
    store.mutate(seed, owner, expected_version=store.snapshot(A).write_version)
    if change in {"correct", "revoke"}:
        store.mutate(
            replace(seed, mutation_id="change", kind=change,
                    content="我这周住在杭州" if change == "correct" else "",
                    source_id="new-source", expected_record_version=1, fact_key=None),
            owner, expected_version=store.snapshot(A).write_version,
        )
    current = store.accept(config(), "current", "我们刚才聊了什么？", session_id=first.session_id)
    # This test isolates completed-source expiry/revision. Pending user sources
    # now deliberately suppress associated fact history (covered separately).
    from virtual_companion.text.memory_runtime import _fake_memory_result

    worker = UserFactWorker(store, _fake_memory_result, clock=lambda: 10,
                            resolution_protocol=True, resolution_strategy="intent_first",
                            memory_commands=True)
    for item in (first, unrelated, current):
        worker.run_one(A, "source-worker", source=item.input_id)
    run = claim(store, current)
    clock = [utc_microseconds(when + timedelta(minutes=2 if change == "expiry" else 0))]
    admission = ShortContextAdmission(store, lambda *args: ShortRequest(args[-1]),
                                      evidence_clock=lambda: clock[0])
    request = admission(run, store.snapshot(A), "我们刚才聊了什么？")
    history = json.loads(request.context.payload)["history"]
    assert (first.input_id in {s["id"] for s in history}) == (change == "none")
    assert unrelated.input_id in {s["id"] for s in history}
    assert store.inspect(A, first.input_id)["text"] == "我这周住在上海"
    admission.validate(request.context)
    if change == "none":
        version = store.snapshot(A).write_version
        clock[0] = utc_microseconds(when + timedelta(minutes=2))
        with pytest.raises(VersionConflict, match="history expired"):
            admission.validate(request.context)
        assert store.snapshot(A).write_version == version


def test_supported_current_forget_is_request_not_replayed_history(tmp_path):
    from test_memory_command_worker import command, seed
    from virtual_companion.boundary.sqlite_memory_management import SQLiteExplicitMemoryStore

    store = SQLiteExplicitMemoryStore(tmp_path / "memory.db")
    seed(store)
    source = "请忘记我喜欢安静音乐这件事。"
    current = store.accept(config(), "forget", source)

    def invoke(stage, messages, schema, cancelled):
        if stage == "memory_command":
            return command(source)
        if stage == "memory_command_support":
            return {"meaning": "explicit_single_forget", "verdict": "supported"}
        raise AssertionError(stage)

    receipt = UserFactWorker(store, invoke, clock=lambda: 10, memory_commands=True).run_one(
        A, "worker"
    )
    assert receipt["status"] == "committed"
    run = claim(store, current)
    admission = ShortContextAdmission(store, lambda *args: ShortRequest(args[-1]),
                                      evidence_clock=lambda: 100)
    request = admission(run, store.snapshot(A), source)
    context = json.loads(request.context.payload)
    assert context["memories"] == context["history"] == []
    messages = request.model_request(192)["messages"]
    assert messages[-1] == {"role": "user", "content": source}
    assert sum(source in item["content"] for item in messages) == 1
    admission.validate(request.context)


def test_actual_confirmed_history_and_barrier_reach_short_model(tmp_path):
    store, session = question_store(tmp_path)
    item = store.accept(config(), "answer", "聊旅行吧", session_id=session)
    run = claim(store, item)
    admission = ShortContextAdmission(store, lambda *args: ShortRequest(args[-1]),
                                      evidence_clock=lambda: 100)
    assembled = admission(run, store.snapshot(A), "聊旅行吧")
    context = json.loads(assembled.context.payload)
    assert context["history"][-1]["text"] == "你今天想聊什么？"
    assert context["memory_status"] == "pending_source"
    assert context["memories"] == context["exchanges"] == []
    # The adopted default renders the same facts as prose; the stored payload above
    # stays canonical JSON for the read-set recheck.
    assert assembled.context.rendering == "prose"
    model = Model()
    model.cancel = lambda: None
    kernel = ShortKernel(store=store, model_factory=lambda cancelled: model, admit=admission,
                         clock=lambda: 100, lease_us=50, deadline_us=1000,
                         renewal_margin_us=10)
    result = kernel.generate(run, store.snapshot(A), "聊旅行吧")
    body = model.sent[0]
    evidence = body["messages"][1]["content"]
    assert evidence.startswith("以下是已确认的背景资料")
    assert "你今天想聊什么？" in evidence and "尚未完成" in evidence
    assert "{" not in evidence and '"' not in evidence
    assert body["messages"][-1] == {"role": "user", "content": "聊旅行吧"}
    assert body == model.counted[0]
    assert store.commit(result, now_us=101)["status"] == "committed"


def test_protected_source_exposes_clarification_status_to_the_actual_model_request(tmp_path):
    store, session = question_store(tmp_path)
    unresolved = store.accept(config(), "unclear", "更正，是另一条", session_id=session)
    background = store.claim_background(A, "worker", now_us=10, lease_us=100)
    assert background is not None and background.source == unresolved.input_id
    store.protect_background(
        background, error="needs_clarification.source_coverage", now_us=11
    )
    run = claim(store, unresolved)
    admission = ShortContextAdmission(
        store, lambda *args: ShortRequest(args[-1]), evidence_clock=lambda: 100
    )
    request = admission(run, store.snapshot(A), "更正，是另一条")
    assert json.loads(request.context.payload)["memory_status"] == "needs_clarification"
    evidence = request.model_request(192)["messages"][1]["content"]
    assert "本次用户事实整理需要用户澄清，尚未保存" in evidence
    assert "只有当前消息确在请求更正或遗忘时，才询问目标" in evidence
    assert "已经保存" not in evidence


def test_source_change_during_context_read_rejected(tmp_path):
    store, session = question_store(tmp_path)
    item = store.accept(config(), "answer", "聊旅行吧", session_id=session)
    run = claim(store, item)
    original = store.exchange_context

    def read(*args, **kwargs):
        value = original(*args, **kwargs)
        store.accept(config(), "newer", "先等等", session_id=session)
        return value

    store.exchange_context = read
    admission = ShortContextAdmission(store, lambda *args: ShortRequest(args[-1]),
                                      evidence_clock=lambda: 100)
    with pytest.raises(VersionConflict):
        admission(run, store.snapshot(A), "聊旅行吧")


@pytest.mark.parametrize("failure", ["foreign", "expired", "expired_during_generation"])
def test_completed_source_reads_persisted_fact_and_pending_question_after_reopen(tmp_path, failure):
    store, session = question_store(tmp_path)
    apply(store, mutation(kind="assert_fact", record="home", content="住在北京", expires_at=WHEN,
                          fact_key=FactKey("user", "home", "profile")))
    item = store.accept(config(), "hello", "你好", session_id=session)

    def model(stage, messages, schema, cancelled):
        if stage == "exchange_intent":
            return {"coverage": "complete", "acts": [{"segmentId": 0,
                     "evidenceQuote": "你好",
                    "intent": "statement", "targetId": None}]}
        return {"operations": []}

    UserFactWorker(store, model, clock=lambda: 14, resolution_protocol=True,
                   resolution_strategy="intent_first").run_one(A, "seed")
    store = type(store)(store.path)
    run = claim(store, item)
    clock = [100]
    admission = ShortContextAdmission(store, lambda *args: ShortRequest(args[-1]),
                                      evidence_clock=lambda: clock[0])
    request = admission(run, store.snapshot(A), "你好")
    context = json.loads(request.context.payload)
    assert context["memory_status"] == "current"
    assert context["memories"][0]["content"] == "住在北京"
    assert context["exchanges"][0]["content"] == "你今天想聊什么？"
    assert context["exchanges"][0]["direction"] == "user_question"
    rendered = request.model_request(192)["messages"][1]["content"]
    assert "本次用户事实整理已完成" in rendered
    assert "住在北京" in rendered
    assert "你今天想聊什么？" in rendered
    assert "助手曾提问，需结合后续对话核验是否仍待回应" in rendered
    model = Model()
    model.cancel = lambda: None
    original_count = model.count

    def count(body):
        if failure == "expired":
            clock[0] = 10**18
        return original_count(body)

    model.count = count
    original_invoke = model.invoke

    def invoke(body):
        if failure == "expired_during_generation":
            clock[0] = 10**18
        return original_invoke(body)

    model.invoke = invoke
    kernel = ShortKernel(store=store, model_factory=lambda cancelled: model,
                         admit=admission if failure != "foreign" else lambda *args: replace(
                             request, context=replace(request.context, input_id="foreign-input")),
                         clock=lambda: 100, lease_us=50, deadline_us=1000,
                         renewal_margin_us=10)
    with pytest.raises(VersionConflict, match="context"):
        kernel.generate(run, store.snapshot(A), "你好")
    assert len(model.sent) == int(failure == "expired_during_generation")


def test_real_sqlite_admission_renders_corrected_memory_and_both_pending_directions(tmp_path):
    store, session = question_store(tmp_path)
    apply(store, mutation(kind="assert_fact", record="home", content="住在北京",
                          fact_key=FactKey("user", "home", "profile")))
    apply(store, mutation("correct-home", kind="correct", record="home", content="住在上海",
                          expected_record_version=1))
    request_text = "帮我想想周末怎么安排。"
    item = store.accept(config(), "weekend-request", request_text, session_id=session)

    def resolve_request(stage, messages, schema, cancelled):
        if stage == "fact_extract":
            return {"operations": []}
        if stage == "exchange_intent":
            return {"coverage": "complete", "acts": [{
                "segmentId": 0,
                "evidenceQuote": request_text, "intent": "request", "targetId": None,
            }]}
        if stage == "exchange_extract":
            return {"operations": [{
                "operation": "add", "type": "response_obligation",
                 "evidenceQuote": request_text,
                "supersedesAtomId": None, "expiresAt": None,
            }]}
        if stage == "exchange_support":
            return {"intent": "request", "verdict": "supported"}
        raise AssertionError(stage)

    UserFactWorker(store, resolve_request, clock=lambda: 14, resolution_protocol=True,
                   resolution_strategy="intent_first").run_one(A, "seed")
    store = type(store)(store.path)
    run = claim(store, item)
    admission = ShortContextAdmission(store, lambda *args: ShortRequest(args[-1]),
                                      evidence_clock=lambda: 100)
    request = admission(run, store.snapshot(A), request_text)
    payload = json.loads(request.context.payload)
    assert [row["content"] for row in payload["memories"]] == ["住在上海"]
    assert {row["direction"] for row in payload["exchanges"]} == {
        "user_question", "assistant_response",
    }
    rendered = request.model_request(192)["messages"][1]["content"]
    assert "住在上海" in rendered and "住在北京" not in rendered
    assert "你今天想聊什么？" in rendered
    assert request_text in rendered
    assert "助手曾提问，需结合后续对话核验是否仍待回应" in rendered
    assert "用户曾请求，需结合已确认对话核验是否已回应" in rendered


def test_committed_catalog_pages_use_current_qualified_versions_in_primary_key_order(tmp_path):
    from virtual_companion.boundary.sqlite_memory_management import SQLiteExplicitMemoryStore
    from virtual_companion.text.chronology import utc_microseconds

    store = SQLiteExplicitMemoryStore(tmp_path / "semantic-catalog.sqlite3")
    store.register(A)
    store.claim_owner(A, "semantic-catalog-seed", expected_epoch=0)
    apply(store, mutation(
        "seed-home", kind="assert_fact", record="z-home", content="住在北京",
        fact_key=FactKey("user", "home-city", "profile"),
    ))
    apply(store, mutation(
        "correct-home", kind="correct", record="z-home", content="住在上海",
        expected_record_version=1,
    ))
    apply(store, mutation(
        "seed-expired", kind="assert_fact", record="a-expired", content="曾经住在南京",
        fact_key=FactKey("user", "old-home-city", "profile"),
        expires_at=WHEN + timedelta(days=1),
    ))
    apply(store, mutation(
        "seed-current", kind="assert_fact", record="b-current", content="喜欢散步",
        fact_key=FactKey("user", "walking", "profile"),
    ))
    now_us = utc_microseconds(WHEN + timedelta(days=2))

    first = store.committed_catalog_page(A, now_us=now_us, limit=1)
    second = store.committed_catalog_page(
        A, now_us=now_us, after_record_id=first["next_after_record_id"], limit=1,
    )

    assert [(item["record_id"], item["version"], item["content"])
            for item in first["records"] + second["records"]] == [
                ("b-current", 1, "喜欢散步"), ("z-home", 2, "住在上海"),
            ]
    assert first["next_after_record_id"] == "b-current"
    assert second["next_after_record_id"] is None
    assert store.committed_recall(
        A, now_us=now_us, candidates=(("z-home", 1),),
    ) == []


def test_semantic_vector_cache_does_not_cross_companion_owners():
    from virtual_companion.cognition.semantic_retrieval import SemanticMemoryRetriever

    class Store:
        row = {"record_id": "shared-record", "version": 1, "content": "shared text"}

        def committed_catalog_page(self, companion, *, now_us, after_record_id=None, limit=100):
            return {"records": (self.row,), "next_after_record_id": None}

        def committed_recall(self, companion, *, now_us, candidates=None, limit=100):
            return ([self.row] if candidates == (("shared-record", 1),) else [])

    class Embedder:
        def __init__(self):
            self.documents = []

        def query_embed(self, query):
            return iter(([1.0, 0.0],))

        def embed(self, texts):
            texts = list(texts)
            self.documents.extend(texts)
            return iter([1.0, 0.0] for _ in texts)

    embedder = Embedder()
    retriever = SemanticMemoryRetriever(
        embedder, model_id="cache-scope-test", model_revision="fixed",
        candidate_overfetch=1, synthetic_model_identity=True,
    )
    store = Store()
    retriever.recall(store, "companion-one", now_us=1, query="question", limit=1)
    retriever.recall(store, "companion-two", now_us=1, query="question", limit=1)

    assert embedder.documents == ["shared text", "shared text"]


def test_semantic_retriever_identity_is_stable_and_binds_artifact_and_ranking_config():
    from virtual_companion.cognition.semantic_retrieval import SemanticMemoryRetriever

    class Embedder:
        identity = {"kind": "synthetic-identity-test.v1"}

        def query_embed(self, query):
            return iter(([1.0, 0.0],))

        def embed(self, texts):
            return iter([1.0, 0.0] for _ in texts)

    common = dict(
        model_id="BAAI/bge-small-zh-v1.5",
        model_revision="46fbe35fd4374a00fee7de77dfddaeb6dd6a2c59",
        preprocessing_revision="raw-content.v1", candidate_overfetch=32,
        synthetic_model_identity=True,
    )
    first = SemanticMemoryRetriever(Embedder(), **common)
    same = SemanticMemoryRetriever(Embedder(), **common)
    different = SemanticMemoryRetriever(Embedder(), **{**common, "candidate_overfetch": 48})

    assert first.identity == same.identity
    assert first.fingerprint == same.fingerprint
    assert first.fingerprint != different.fingerprint
    assert first.identity["model"]["manifest_sha256"] is None
    assert first.identity["model"]["identity_kind"] == "synthetic"
    assert "object at" not in repr(first.identity)
    assert all("path" not in key.lower() for key in first.identity)


def test_fastembed_adapter_verifies_pinned_local_manifest_and_disables_download(
        tmp_path, monkeypatch):
    import hashlib
    import sys
    from types import SimpleNamespace

    import virtual_companion.text.semantic_embedding as semantic_embedding
    from virtual_companion.cognition.semantic_retrieval import SemanticMemoryRetriever

    model_directory = tmp_path / "model"
    model_directory.mkdir()
    model_bytes = b"synthetic local model artifact"
    (model_directory / "model_optimized.onnx").write_bytes(model_bytes)
    manifest = {
        "model": "synthetic/model",
        "revision": "fixed-test-revision",
        "files": [{
            "file": "model_optimized.onnx",
            "bytes": len(model_bytes),
            "sha256": hashlib.sha256(model_bytes).hexdigest(),
        }],
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest_path = tmp_path / "model-manifest.json"
    manifest_path.write_bytes(manifest_bytes)
    observed = []

    class FakeTextEmbedding:
        def __init__(self, **kwargs):
            observed.append(kwargs)

        def query_embed(self, query):
            return iter(([1.0, 0.0],))

        def embed(self, texts):
            return iter([1.0, 0.0] for _ in texts)

    monkeypatch.setitem(sys.modules, "fastembed", SimpleNamespace(TextEmbedding=FakeTextEmbedding))
    monkeypatch.setattr(semantic_embedding, "package_version", lambda _: "0.9.0")
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    encoder = semantic_embedding.FastEmbedEncoder.from_local_manifest(
        model_directory, manifest_path, model_id="synthetic/model",
        model_revision="fixed-test-revision", manifest_sha256=manifest_sha256,
    )
    retriever = SemanticMemoryRetriever(
        encoder, model_id="synthetic/model", model_revision="fixed-test-revision",
        candidate_overfetch=8,
    )

    assert observed == [{
        "model_name": "synthetic/model",
        "specific_model_path": str(model_directory.resolve()),
        "local_files_only": True,
        "threads": 2,
        "providers": ["CPUExecutionProvider"],
    }]
    assert retriever.identity["embedding_backend"] == {
        "adapter": "fastembed-text.v1",
        "distribution": "fastembed",
        "version": "0.9.0",
        "execution_provider": "CPUExecutionProvider",
    }
    assert retriever.identity["model"]["manifest_sha256"] == manifest_sha256
    with pytest.raises(ValueError, match="differs from verified encoder"):
        SemanticMemoryRetriever(
            encoder, model_id="synthetic/model", model_revision="fixed-test-revision",
            model_manifest_sha256="f" * 64, candidate_overfetch=8,
        )

    (model_directory / "model_optimized.onnx").write_bytes(b"changed")
    with pytest.raises(ValueError, match="failed manifest verification"):
        semantic_embedding.FastEmbedEncoder.from_local_manifest(
            model_directory, manifest_path, model_id="synthetic/model",
            model_revision="fixed-test-revision", manifest_sha256=manifest_sha256,
        )
    assert len(observed) == 1


def test_semantic_recall_checkpoints_around_pages_embeddings_and_exact_recheck():
    from virtual_companion.cognition.semantic_retrieval import SemanticMemoryRetriever

    events = []
    row = {"record_id": "fact", "version": 1, "content": "喜欢散步"}

    class Store:
        def committed_catalog_page(self, companion, *, now_us, after_record_id=None, limit=100):
            events.append("page")
            return {"records": (row,), "next_after_record_id": None}

        def committed_recall(self, companion, *, now_us, candidates=None, limit=100):
            events.append("recheck")
            return [row] if candidates == (("fact", 1),) else []

    class Embedder:
        def query_embed(self, query):
            events.append("query")
            return iter(([1.0, 0.0],))

        def embed(self, texts):
            events.append("documents")
            return iter([1.0, 0.0] for _ in texts)

    retriever = SemanticMemoryRetriever(
        Embedder(), model_id="test", model_revision="fixed", candidate_overfetch=1,
        synthetic_model_identity=True,
    )
    result = retriever.recall(
        Store(), "owner", now_us=1, query="散步", limit=1,
        checkpoint=lambda: events.append("checkpoint"),
    )

    assert result == [row]
    assert events == [
        "checkpoint", "page", "checkpoint", "checkpoint", "query", "checkpoint",
        "checkpoint", "documents", "checkpoint", "checkpoint", "recheck",
    ]


def test_semantic_recall_stops_before_recheck_when_checkpoint_cancels():
    from virtual_companion.cognition.semantic_retrieval import SemanticMemoryRetriever

    calls = 0

    class Store:
        def committed_catalog_page(self, companion, *, now_us, after_record_id=None, limit=100):
            return {"records": ({"record_id": "fact", "version": 1,
                                 "content": "喜欢散步"},), "next_after_record_id": None}

        def committed_recall(self, companion, *, now_us, candidates=None, limit=100):
            raise AssertionError("checkpoint cancellation must precede final recheck")

    class Embedder:
        def query_embed(self, query):
            return iter(([1.0, 0.0],))

        def embed(self, texts):
            return iter([1.0, 0.0] for _ in texts)

    def checkpoint():
        nonlocal calls
        calls += 1
        if calls == 6:
            raise RuntimeError("synthetic cancellation")

    retriever = SemanticMemoryRetriever(
        Embedder(), model_id="test", model_revision="fixed", candidate_overfetch=1,
        synthetic_model_identity=True,
    )
    with pytest.raises(RuntimeError, match="synthetic cancellation"):
        retriever.recall(
            Store(), "owner", now_us=1, query="散步", limit=1, checkpoint=checkpoint,
        )
