"""Stage-one memory adapter for the text desktop.

The text application keeps its conversation history store, while this adapter
owns the already approved evidence-memory store and its background source
worker.  The two stores share the product companion/session identity; neither
store is allowed to infer the other's state from display text.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

from virtual_companion.boundary.contracts import VersionConflict
from virtual_companion.boundary.memory_relevance import lexical_relevance
from virtual_companion.boundary.sqlite_received import SQLiteReceivedStore
from virtual_companion.cognition.background_worker import BackgroundScheduler
from virtual_companion.text.chronology import utc_microseconds
from virtual_companion.text.identity import CompanionConfig, VersionedReference
from virtual_companion.text.memory_management import MemoryManagementService
from virtual_companion.text.runtime_skills import (
    SUPPORTED_VERSIONS,
    VERSION,
    MemorySkills,
    load_memory_skills,
)

DEFAULT_COMPANION_ID = "00000000-0000-0000-0000-000000000001"
_MEMORY_CHARACTER = VersionedReference("neutral", "stage1", "0" * 64)
_LOGGER = logging.getLogger(__name__)
_EXCHANGE_OUTPUT_TOKENS = 1024


def _memory_config(companion: str) -> CompanionConfig:
    return CompanionConfig(companion, _MEMORY_CHARACTER)


def _json_object(text: str):
    value = text.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            value = "\n".join(lines[1:-1]).strip()
    result = json.loads(value)
    if not isinstance(result, dict):
        raise ValueError("Memory model must return one JSON object")
    return result


def _normalize_structured_result(stage, result, schema):
    """Apply only host-owned, source-preserving defaults to local JSON.

    The fixed local model sometimes omits schema fields whose value is already
    determined by the evidence contract.  We may fill a schema-owned content
    field from the model's exact evidence quote and treat an omitted
    long-term expiry as ``None``.  Host-bound fact schemas deliberately omit
    content and let ``FactModel`` bind it later.  We never invent a short-term
    expiry, target, identity, or management action; all other schema failures
    remain failures and keep the source barrier active.
    """
    if stage != "fact_extract" or not isinstance(result, dict):
        if stage == "exchange_extract" and isinstance(result, dict):
            operations = result.get("operations")
            if isinstance(operations, list):
                # The exchange schema permits null type for a candidate the
                # model cannot bind to either direction. It is not a valid
                # operation. Drop only that candidate; intent-first coverage
                # still rejects a real request that was left uncovered.
                return {
                    **result,
                    "operations": [
                        item
                        for item in operations
                        if not (
                            isinstance(item, dict)
                            and item.get("operation") == "add"
                            and item.get("type") is None
                        )
                    ],
                }
        return result
    operations = result.get("operations")
    if not isinstance(operations, list):
        return result
    branches = schema.get("properties", {}).get("operations", {}).get("items", {}).get("oneOf", ())
    required_fields = {
        field
        for branch in branches
        if isinstance(branch, dict)
        for field in branch.get("required", ())
    }
    content_is_schema_owned = "content" in required_fields
    normalized = []
    for operation in operations:
        if not isinstance(operation, dict):
            normalized.append(operation)
            continue
        item = dict(operation)
        if (
            content_is_schema_owned
            and item.get("operation") in {"add", "supersede"}
            and "content" not in item
            and isinstance(item.get("evidenceQuote"), str)
            and item["evidenceQuote"]
        ):
            item["content"] = item["evidenceQuote"]
        if item.get("type") == "long_term" and "expiresAt" not in item:
            item["expiresAt"] = None
        normalized.append(item)
    return {**result, "operations": normalized}


def _source_text(messages):
    """Read the selected complete source before its structural scaffold."""
    for item in reversed(messages):
        if isinstance(item, dict) and item.get("role") == "user":
            try:
                payload = json.loads(item.get("content", ""))
            except (TypeError, ValueError):
                return ""
            if not isinstance(payload, dict):
                return ""
            eligible = payload.get("eligible_source_ids")
            if "eligible_source_ids" in payload and (
                    not isinstance(eligible, list) or len(eligible) != 1
                    or not isinstance(eligible[0], str) or not eligible[0]):
                return ""
            if "source" in payload:
                source = payload["source"]
            elif "sources" in payload:
                sources = payload["sources"]
                if not isinstance(sources, list):
                    return ""
                if eligible is not None:
                    sources = [source for source in sources
                               if isinstance(source, dict) and source.get("id") == eligible[0]]
                if len(sources) != 1:
                    return ""
                source = sources[0]
            else:
                # Retain the legacy source-less scaffold reader. It cannot
                # satisfy an explicit current-source selection.
                segments = payload.get("segments")
                if eligible is not None or not isinstance(segments, list) or not segments:
                    return ""
                source = segments[-1]
            if not isinstance(source, dict):
                return ""
            if eligible is not None and source.get("id") != eligible[0]:
                return ""
            return source.get("text", "") if isinstance(source.get("text"), str) else ""
    return ""


def _fake_memory_result(stage, messages, schema, cancelled):
    """Complete fake-mode sources without pretending to infer a memory."""
    if cancelled.is_set():
        raise InterruptedError("Memory processing cancelled")
    if stage == "memory_command":
        return {
            "action": "none", "requestedScope": "none", "targetIds": [],
            "evidenceQuote": "", "quoteStart": None, "quoteEnd": None,
            "replacementQuote": "", "replacementStart": None,
            "replacementEnd": None, "coverage": "not_command", "unresolved": [],
        }
    if stage == "fact_extract":
        return {"operations": []}
    if stage == "exchange_intent":
        payload = json.loads(messages[-1]["content"])
        source = _source_text(messages)
        segments = payload.get("segments") or []
        item = segments[0] if segments else {}
        properties = schema["properties"]["acts"]["items"]["properties"]
        if "endSegmentId" in properties:
            return {
                "acts": [{
                    "segmentId": item["id"],
                    "endSegmentId": segments[-1]["id"],
                    "intent": "statement",
                    "targetIndex": None,
                }],
                "coverage": "complete",
            }
        return {
            "acts": [{
                "segmentId": item.get("id", 0),
                "evidenceQuote": source,
                "intent": "statement",
                "targetId": None,
            }],
            "coverage": "complete",
        }
    if stage == "exchange_extract":
        return {"operations": []}
    raise ValueError(f"Fake memory stage is not supported: {stage}")


class _PriorityGenerationGate:
    """Give a waiting foreground turn priority over background extraction.

    A foreground-busy check prevents new background work from starting while a
    turn is already running, but it cannot stop a background generation that
    began in the small gap between two turns.  The gate records foreground
    waiters and asks the in-flight background request to cancel before the next
    foreground request acquires the shared local model.
    """

    def __init__(self):
        self._condition = threading.Condition()
        self._active = None
        self._foreground_waiters = 0
        self._background_cancellation = None
        self._closed = False

    def close(self):
        with self._condition:
            self._closed = True
            if self._background_cancellation is not None:
                cancel, signal = self._background_cancellation
                if signal is not None:
                    signal.set()
                cancel()
            self._condition.notify_all()

    @contextmanager
    def foreground(self, cancelled=None):
        with self._condition:
            self._check_cancelled(cancelled)
            self._foreground_waiters += 1
            cancellation = (
                self._background_cancellation if self._active == "background" else None
            )
            if cancellation is not None:
                cancel, background_cancelled = cancellation
                if background_cancelled is not None:
                    background_cancelled.set()
                try:
                    cancel()
                except Exception:
                    _LOGGER.debug(
                        "Background generation cancellation request failed", exc_info=True
                    )
        with self._condition:
            try:
                while self._active is not None:
                    self._check_cancelled(cancelled)
                    self._condition.wait(0.025)
                self._check_cancelled(cancelled)
                self._active = "foreground"
            finally:
                self._foreground_waiters -= 1
                self._condition.notify_all()
        try:
            yield
        finally:
            with self._condition:
                self._active = None
                self._condition.notify_all()

    @contextmanager
    def background(self, cancel, cancelled=None):
        with self._condition:
            while self._active is not None or self._foreground_waiters:
                if self._closed:
                    raise InterruptedError("Memory worker closed")
                self._check_cancelled(cancelled)
                self._condition.wait(0.025)
            if self._closed:
                raise InterruptedError("Memory worker closed")
            self._check_cancelled(cancelled)
            self._active = "background"
            self._background_cancellation = (cancel, cancelled)
        try:
            yield
        except Exception:
            # Provider/local adapters use different cancellation exceptions.
            # The host signal decides whether this is a cooperative yield.
            self._check_cancelled(cancelled)
            raise
        finally:
            with self._condition:
                self._active = None
                self._background_cancellation = None
                self._condition.notify_all()

    @staticmethod
    def _check_cancelled(cancelled):
        if cancelled is not None and cancelled.is_set():
            raise InterruptedError("Generation gate wait cancelled")


class _StructuredMemoryInvoker:
    supports_bound_evidence = True

    def __init__(self, model, generation_gate, character=None, *, memory_skills=None,
                 source_role=None):
        self.model, self.generation_gate = model, generation_gate
        self.character = character
        self.memory_skills, self.source_role = memory_skills, source_role

    def for_source(self, store, claim):
        from .character import character_from_binding

        return type(self)(self.model, self.generation_gate,
                          character_from_binding(store.background_source_config(
                              claim.companion, claim.job)),
                          memory_skills=self.memory_skills, source_role=claim.role)

    def __call__(self, stage, messages, schema, cancelled):
        from .character import character_messages

        policy = (self.memory_skills.maintenance_messages(stage=stage, source_role=self.source_role)
                  if self.memory_skills is not None else [])
        messages = [*character_messages(self.character), *policy, *messages]
        gate = self.generation_gate
        context = (
            gate.background(getattr(self.model, "cancel", lambda: None), cancelled)
            if hasattr(gate, "background")
            else nullcontext(gate)
        )
        with context:
            if cancelled.is_set():
                raise InterruptedError("Memory processing cancelled")
            prepare = getattr(self.model, "prepare_stage_request", None)
            if callable(prepare):
                metadata = ({"stage": stage, "schema": schema}
                            if getattr(self.model, "supports_stage_metadata", False) else {})
                if (
                    metadata
                    and stage in {"exchange_intent", "exchange_extract"}
                    and getattr(self.model, "supports_stage_output_override", False)
                ):
                    # The host binds full quotes from compact choices; reserve
                    # enough for the bounded multi-act envelope in either route.
                    metadata["output_tokens"] = _EXCHANGE_OUTPUT_TOKENS
                request = prepare(messages, **metadata)
                request.check_context()
                if cancelled.is_set():
                    raise InterruptedError("Memory processing cancelled")
                stream = request.stream(cancelled)
            else:
                # Recording/fake stage models retain their existing text Port.
                stream = self.model.stream(messages, cancelled)
            try:
                output = "".join(stream)
            finally:
                close = getattr(stream, "close", None)
                if callable(close):
                    close()
        if cancelled.is_set():
            raise InterruptedError("Memory processing cancelled")
        return _normalize_structured_result(stage, _json_object(output), schema)


class _PreparedMemoryRequest:
    """One-shot pilot request; validation never replaces the frozen payload.

    This is not a presentation authorization. The default session must still
    acquire source-aware admission and persistent delivery receipts.
    """

    def __init__(self, owner, messages, validate, cancelled=None):
        self.owner = owner
        self.cancelled = cancelled
        self._messages = deepcopy(messages)
        prepare = getattr(owner.model, "prepare_request", None)
        self._wire_request = prepare(deepcopy(messages)) if callable(prepare) else None
        self.validate = validate
        self.checked = False
        self.used = False
        self.lock = threading.Lock()

    def check_context(self):
        with self.lock, self.owner._foreground(self.cancelled):
            if self.used:
                raise ValueError("Prepared request already consumed")
            self.checked = False
            self.validate()
            result = (self._wire_request.check_context() if self._wire_request is not None
                      else self.owner.model.check_context(deepcopy(self._messages)))
            _PriorityGenerationGate._check_cancelled(self.cancelled)
            self.validate()
            self.checked = True
            return result

    def stream(self, cancelled):
        with self.lock:
            if self.cancelled is not None and cancelled is not self.cancelled:
                raise ValueError("Prepared request cancellation identity changed")
            if self.used or not self.checked:
                raise ValueError("Prepared request requires unused successful preflight")
            self.used = True
        if cancelled.is_set():
            return
        with self.owner._foreground(cancelled):
            self.validate()
            stream = (self._wire_request.stream(cancelled) if self._wire_request is not None
                      else self.owner.model.stream(deepcopy(self._messages), cancelled))
            try:
                for delta in stream:
                    if cancelled.is_set():
                        return
                    self.validate()
                    yield delta
                if not cancelled.is_set():
                    self.validate()
            finally:
                stream.close()


class MemoryAwareModel:
    """Add only selected, valid local memory to the existing text model."""

    def __init__(self, model, memory, generation_gate):
        self.model, self.memory, self.generation_gate = model, memory, generation_gate

    def _messages(self, messages):
        return self.memory.with_context(messages)

    def prepare_request(self, messages, *, cancelled=None):
        projected, validate = self.memory.capture_context(deepcopy(messages))
        return _PreparedMemoryRequest(self, projected, validate, cancelled)

    def _foreground(self, cancelled=None):
        return (
            self.generation_gate.foreground(cancelled)
            if hasattr(self.generation_gate, "foreground")
            else nullcontext(self.generation_gate)
        )

    def check_context(self, messages):
        with self._foreground():
            return self.model.check_context(self._messages(messages))

    def stream(self, messages, cancelled):
        with self._foreground(cancelled):
            yield from self.model.stream(self._messages(messages), cancelled)

    def cancel(self):
        self.model.cancel()


class TextMemoryRuntime:
    def __init__(
        self,
        path: Path,
        companion: str,
        *,
        model=None,
        fake=False,
        background_idle_grace=2.0,
        character=None,
        memory_skills=None,
        loaded_memory_skills=None,
    ):
        if type(background_idle_grace) not in (int, float) or not 0 <= background_idle_grace <= 60:
            raise ValueError("Background idle grace must be bounded")
        if memory_skills not in (None, *SUPPORTED_VERSIONS):
            raise ValueError('Unsupported runtime memory skills')
        selected_version = memory_skills or VERSION
        if loaded_memory_skills is not None and (
                not isinstance(loaded_memory_skills, MemorySkills)
                or loaded_memory_skills.version != selected_version):
            raise ValueError('Preloaded memory skills differ from the selected capability')
        self.memory_skills = (loaded_memory_skills or load_memory_skills(selected_version)
                              if memory_skills is not None else loaded_memory_skills)
        self.path = Path(path).resolve()
        self.companion = companion
        self.store = SQLiteReceivedStore(self.path)
        self.store.register(companion)
        self.service = MemoryManagementService(
            self.store,
            companion,
            clock=lambda: utc_microseconds(datetime.now(UTC)),
        )
        if character is not None:
            from .character import CharacterSnapshot

            if not isinstance(character, CharacterSnapshot) or character.companion_id != companion:
                raise ValueError("Character companion differs from memory scope")
        self.config = (_memory_config(companion) if character is None else
                       CompanionConfig(companion, character.reference,
                                       character_data=character.payload))
        self.fake = fake
        self.model = model
        self.generation_gate = _PriorityGenerationGate()
        self._foreground_busy = lambda: False
        self._wake = threading.Event()
        self._stopped = threading.Event()
        self._thread = None
        self._background_idle_grace = float(background_idle_grace)
        self._background_not_before = 0.0
        self.last_error = ""
        self.scheduler = None
        if model is not None:
            invoke = (
                _fake_memory_result
                if fake
                else _StructuredMemoryInvoker(model, self.generation_gate,
                                               memory_skills=self.memory_skills)
            )
            self.scheduler = BackgroundScheduler(
                self.store,
                user_invoke=invoke,
                assistant_invoke=invoke,
                clock=lambda: utc_microseconds(datetime.now(UTC)),
                resolution_protocol=True,
                resolution_strategy="intent_first",
                memory_commands=True,
            )

    def attach_foreground(self, busy):
        if not callable(busy):
            raise ValueError("Foreground busy callback required")
        self._foreground_busy = busy
        if self.scheduler is not None and self._thread is None:
            self._defer_background()
            self._thread = threading.Thread(
                target=self._run, name="text-memory-worker", daemon=True
            )
            self._thread.start()

    def _defer_background(self):
        self._background_not_before = max(
            self._background_not_before,
            time.monotonic() + self._background_idle_grace,
        )

    def _run(self):
        while not self._stopped.is_set():
            if self._foreground_busy():
                self._defer_background()
                self._wake.wait(0.15)
                self._wake.clear()
                continue
            remaining = self._background_not_before - time.monotonic()
            if remaining > 0:
                self._wake.wait(min(remaining, 0.15))
                self._wake.clear()
                continue
            try:
                result = self.scheduler.run_next("text-memory-worker", companion=self.companion)
                self.last_error = "" if result is not None else self.last_error
            except Exception:
                # The source store retains a bounded retry/clarification state;
                # never take down the foreground conversation for a failed model
                # extraction or a stale background claim.
                self.last_error = "记忆整理暂时未完成；可在记忆面板查看状态。"
                self._defer_background()
                self._wake.wait(0.5)
                self._wake.clear()
                continue
            if result is not None:
                self._defer_background()
            else:
                self._background_not_before = max(
                    self._background_not_before, time.monotonic() + 0.4
                )
            self._wake.wait(0.15 if result is not None else 0.4)
            self._wake.clear()

    def accept_user(self, client_key, text, session_id):
        if not isinstance(client_key, str) or not client_key:
            raise ValueError("Text source identity required")
        self.store.ensure_session(self.companion, session_id)
        receipt = self.store.accept(
            self.config,
            client_key,
            text,
            session_id=session_id,
        )
        self._wake.set()
        return receipt

    def resolve_for_recall(self, input_id, cancelled):
        """Opt-in memory access pilot, not a mandatory per-turn preprocessing chain.

        Intake must already exist. Process at most this one user source using
        the existing leased worker. Never drain older inputs or assistant work,
        waive unresolved coverage, or equate a partial fact receipt with safety.
        The caller must still acquire a fresh bound context and delivery claim.
        """
        self.store.inspect(self.companion, input_id)
        if cancelled.is_set():
            return "cancelled"
        if not self.store.memory_barrier(self.companion):
            return "current"
        pending = self.store.background_sources(self.companion, limit=256)
        if any(row["role"] == "user" and row["source"] != input_id for row in pending):
            return "pending_source"
        if self.scheduler is None:
            return "pending_source"
        self.scheduler.user.run_one(
            self.companion, "text-demand-memory", source=input_id, cancelled=cancelled
        )
        if cancelled.is_set():
            return "cancelled"
        if not self.store.memory_barrier(self.companion):
            return "current"
        status = self.store.memory_status(self.companion)
        return status if status != "current" else "pending_source"

    def capture_context(self, messages):
        """Conservative pilot guard, using the existing store version authority.

        The selected records' deadline detects TTL expiry without a write. Broad version
        invalidation is deliberately not a final default-path latency solution.
        Failures propagate here; the legacy optional-context fallback is not an
        admission proof. No source is exempted or resolved by this method.
        """
        version = self.store.snapshot(self.companion).write_version
        projected, valid_until = self._project_context(deepcopy(messages))

        def validate():
            if (
                self.store.snapshot(self.companion).write_version != version
                or (valid_until is not None
                    and utc_microseconds(datetime.now(UTC)) >= valid_until)
            ):
                raise VersionConflict("Prepared memory context changed")

        validate()
        return projected, validate

    def with_context(self, messages):
        try:
            return self._project_context(messages)[0]
        except Exception:
            # Memory is an optional context source. A stale/locked memory DB
            # must not make public conversation unavailable.
            return messages

    def _project_context(self, messages):
        current = next(
            (item["content"] for item in reversed(messages) if item.get("role") == "user"),
            "",
        )
        if not current:
            return messages, None
        selected = self.store.ranked_recall(
            self.companion,
            now_us=utc_microseconds(datetime.now(UTC)),
            score=lexical_relevance(current),
            limit=8,
        )
        if not selected:
            return messages, None
        context = [
            "以下是与当前输入有词面关联的本地有效记忆。只在确实相关时使用；"
            "发生时间未知时不要补造日期，也不要把记忆当成用户刚刚说过的话。",
        ]
        deadlines = []
        for value in selected:
            event_time = value.get("event_time") or "不详"
            context.append(f"- 内容：{value['content']}（发生时间：{event_time}）")
            if value.get("expires_at"):
                deadlines.append(utc_microseconds(datetime.fromisoformat(value["expires_at"])))
        return (
            [{"role": "system", "content": "\n".join(context)}, *messages],
            min(deadlines, default=None),
        )

    def state(self):
        return {
            "memory_management": True,
            "memory_status": self.store.memory_status(self.companion),
            "memory_error": self.last_error,
        }

    def list_memories(self, **payload):
        return self.service.list_memories(**payload)

    def memory_details(self, record_id):
        return self.service.memory_details(record_id)

    def apply_action(self, payload):
        return self.service.apply_action(payload)

    def abandon(self, payload):
        return self.service.abandon(payload)

    def close(self):
        self._stopped.set()
        self._wake.set()
        self.generation_gate.close()
        if self._thread is not None:
            self._thread.join(5)
            if self._thread.is_alive():
                raise RuntimeError("Memory worker did not stop; provider switch rejected")
