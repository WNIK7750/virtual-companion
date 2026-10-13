"""Turn/attempt coordinator. No transport, database, or model implementation imports."""

from __future__ import annotations

import threading
from collections.abc import Callable
from copy import deepcopy

from virtual_companion.boundary.receiving_context import (
    receiving_from_binding,
    validate_receiving_context,
)

from .demand_memory import DemandTurnResult
from .ports import ConversationStore, PreparedTextRequest, TextGenerationPort


class TextSession:
    def __init__(
        self,
        model: TextGenerationPort,
        store: ConversationStore,
        binding: dict,
        *,
        preflight: Callable[[list[dict]], None] | None = None,
        before_start: Callable[[bool], None] | None = None,
        prepare_turn: Callable[[dict, str, list[dict], threading.Event],
                               PreparedTextRequest] | None = None,
    ):
        self.model, self.store, self.binding = model, store, binding
        self.preflight = preflight
        self.before_start = before_start
        if prepare_turn is not None and preflight is not None:
            raise ValueError("Bound preparation owns preflight; do not preflight another request")
        self.prepare_turn = prepare_turn
        self.lock = threading.RLock()
        self.cancelled = threading.Event()
        self.worker = None
        self.partial = ""
        self.sequence = 0
        self.error = ""

    def _busy(self):
        return self.worker is not None and self.worker.is_alive()

    def snapshot(self):
        with self.lock:
            return {
                "session_id": self.store.session_id,
                "busy": self._busy(),
                "attempt": self.store.latest_attempt(),
                "partial": self.partial,
                "sequence": self.sequence,
                "error": self.error,
            }

    def history(self, **window):
        with self.lock:
            return self.store.history_page(**window)

    def new_session(self):
        with self.lock:
            if self._busy():
                raise ValueError("Generation in progress")
            self.partial, self.error = "", ""
            return self.store.new_session()

    def select_session(self, session_id):
        with self.lock:
            if self._busy():
                raise ValueError("Generation in progress")
            self.partial, self.error = "", ""
            return self.store.select_session(session_id)

    def send(self, text, *, receiving_context=None):
        if not isinstance(text, str) or not text.strip() or len(text) > 8000:
            raise ValueError("Invalid message")
        return self._start(text.strip(), retry=False, receiving_context=receiving_context)

    def retry(self):
        return self._start(None, retry=True)

    def start_queued(self, received):
        """Start the already persisted source; never submit its text twice."""
        return self._start(received['text'], retry=False, received=received)

    def _start(self, text, *, retry, received=None, receiving_context=None):
        with self.lock:
            if self._busy():
                raise ValueError("Generation in progress")
            if self.before_start is not None:
                self.before_start(retry)
            messages = None
            if self.prepare_turn is None:
                messages = [
                    {"role": m["role"], "content": m["content"]}
                    for m in self.store.history()["messages"]
                ]
                if not retry:
                    messages.append({"role": "user", "content": text})
                if self.preflight is not None:
                    self.preflight(messages)
            binding = dict(self.binding)
            if retry:
                original = self.store.latest_attempt()
                receiving_context = (receiving_from_binding(original['binding'])
                                     if original else None)
            if receiving_context is not None:
                binding['receiving_context'] = validate_receiving_context(receiving_context)
            attempt = (self.store.begin_queued(received['turn_id']) if received is not None
                       else self.store.begin(text, binding, retry=retry))
            self.cancelled = threading.Event()
            self.partial, self.sequence, self.error = "", 0, ""
            try:
                self.worker = threading.Thread(
                    target=self._generate,
                    args=(deepcopy(attempt), messages, self.store.session_id), daemon=True
                )
                self.worker.start()
            except Exception:
                self.worker = None
                self.store.finish(attempt["attempt_id"], "failed", "")
                self.error = "回复未能启动，可以重试。"
                raise
            return attempt

    def _generate(self, attempt, messages, session_id):
        attempt_id = attempt["attempt_id"]
        status = "completed"
        stream = None
        request = None
        result = None
        try:
            if self.prepare_turn is not None:
                # The persisted turn ID survives a failed intake and process reopen.
                # Pass source-bearing records, never the legacy role/content replay.
                records = deepcopy(self.store.history()["messages"])
                if self.cancelled.is_set():
                    raise InterruptedError("Turn preparation cancelled")
                request = self.prepare_turn(attempt, session_id, records, self.cancelled)
                if self.cancelled.is_set():
                    raise InterruptedError("Turn preparation cancelled")
                request.check_context()
                if self.cancelled.is_set():
                    raise InterruptedError("Turn preflight cancelled")
                stream = request.stream(self.cancelled)
            else:
                stream = self.model.stream(messages, self.cancelled)
            for delta in stream:
                with self.lock:
                    if self.cancelled.is_set():
                        break
                    if not isinstance(delta, str) or len(self.partial) + len(delta) > 64000:
                        raise ValueError("Invalid generation")
                    self.partial += delta
                    self.sequence += 1
            result = getattr(request, "result", None)
            if isinstance(result, DemandTurnResult) and result.delivery_status == 'received':
                status = 'received'
            if isinstance(result, DemandTurnResult) and result.response_type == "listen":
                if self.partial:
                    raise ValueError("Listening turn cannot contain assistant text")
                status = "listened"
        except Exception:
            status = "failed"
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    status = "failed"
            close_request = getattr(request, "close", None)
            if callable(close_request):
                try:
                    close_request()
                except Exception:
                    status = "failed"
        with self.lock:
            if self.cancelled.is_set():
                status = "cancelled"
            if status == "completed" and not self.partial.strip():
                status = "failed"
            try:
                self.store.finish(attempt_id, status, self.partial)
            except Exception:
                self.error = "回复未能保存，请重启恢复后重试。"
                return
            if status == "failed":
                self.error = "回复中断，未保存为完整消息。"

    def stop(self):
        with self.lock:
            if not self._busy():
                return self.snapshot()
            self.cancelled.set()
            worker = self.worker
        self.model.cancel()
        worker.join(3)
        if worker.is_alive():
            raise RuntimeError("Cancellation not yet confirmed")
        return self.snapshot()
