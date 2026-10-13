"""Optional native-request to exact-message bridge, with one prepared call owner.

Systems must be explicitly assembled/frozen by the experiment. This module does
not translate legacy semantics or certify persona equivalence. The optional
observer receives raw local evidence; there is no default logging or persistence.
"""

import json
import threading
from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
from types import MappingProxyType

from virtual_companion.boundary.runtime import KernelFailure


def fingerprint(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class StageMessageModel:
    def __init__(self, transport, *, systems, observe=None):
        if (
            not isinstance(systems, dict)
            or not systems
            or any(
                not isinstance(k, str) or not k or not isinstance(v, str) or not v.strip()
                for k, v in systems.items()
            )
        ):
            raise ValueError("Explicit stage system texts required")
        if observe is not None and not callable(observe):
            raise ValueError("Optional local observer must be callable")
        self.transport = transport
        self.systems = MappingProxyType(dict(systems))
        self.observe = observe
        self._lock = threading.Lock()
        self._operation = None
        self._sequence = 0
        self._cancel_count = 0
        self._local = threading.local()

    def _assemble(self, messages):
        if (
            not isinstance(messages, list)
            or len(messages) != 1
            or not isinstance(messages[0], dict)
            or set(messages[0]) != {"role", "content"}
            or messages[0]["role"] != "user"
            or not isinstance(messages[0]["content"], str)
        ):
            raise ValueError("One native JSON user request required")
        request = json.loads(messages[0]["content"])
        if not isinstance(request, dict) or not isinstance(request.get("stage"), str):
            raise ValueError("Native request stage required")
        system = self.systems.get(request["stage"])
        if system is None:
            raise ValueError("Unconfigured stage")
        final = [{"role": "system", "content": system}, *deepcopy(messages)]
        return request, final

    def _emit(self, record):
        if self.observe is not None:
            self.observe(deepcopy(record))

    @contextmanager
    def call_scope(self, cancelled):
        """Bind cancellation before ControlledCalls starts its observer or assembly.

        A per-thread signal belongs only to this invocation; cancelling an idle
        model cannot poison the next call. Prepared ownership is also cleaned up
        if the caller exits without ever starting its generator.
        """
        if getattr(self._local, "cancelled", None) is not None:
            raise RuntimeError("Nested model call scope")
        self._local.cancelled = cancelled
        try:
            yield
        finally:
            del self._local.cancelled
            with self._lock:
                operation = self._operation
                ready = (
                    operation is not None
                    and operation["signal"] is cancelled
                    and operation["phase"] == "ready"
                )
            if ready:
                self._finish(operation, {"status": "failed"})

    def _is_cancelled(self, operation):
        signal = operation["signal"]
        return operation["cancelled"] or (signal is not None and signal.is_set())

    def _finish(self, operation, record):
        with self._lock:
            if operation["finished"]:
                return
            operation["finished"] = True
            if self._is_cancelled(operation):
                record = {**record, "status": "cancelled"}
            if self._operation is operation:
                self._operation = None
        self._emit({**operation["record"], **record})

    def check_context(self, messages):
        # The signal was bound before call entry, including overridden assembly.
        signal = getattr(self._local, "cancelled", None)
        request, final = self._assemble(messages)
        with self._lock:
            if self._operation is not None or self._cancel_count:
                raise RuntimeError("Prepared or active call already owns this transport")
            self._sequence += 1
            operation = {
                "phase": "checking",
                "cancelled": False,
                "signal": signal,
                "finished": False,
                "input_sha256": fingerprint(messages),
                "messages": final,
                "record": {
                    "call_id": self._sequence,
                    "stage": request["stage"],
                    "context_sha256": request.get("context_sha256"),
                    "candidate_sha256": request.get("candidate_sha256"),
                    "request_sha256": fingerprint(final),
                    "messages": final,
                    "status": "checking_capacity",
                },
            }
            self._operation = operation
        try:
            self._emit(operation["record"])
            if self._is_cancelled(operation):
                raise ValueError("Call cancelled before capacity check")
            self.transport.check_context(deepcopy(final))
            with self._lock:
                if self._is_cancelled(operation):
                    raise ValueError("Call cancelled during capacity check")
                operation["phase"] = "ready"
        except BaseException as error:
            self._finish(operation, {"status": "failed", "error_type": type(error).__name__})
            raise

    def stream(self, messages, cancelled):
        with self._lock:
            operation = self._operation
            if (
                operation is None
                or operation["phase"] != "ready"
                or operation["input_sha256"] != fingerprint(messages)
            ):
                raise ValueError("Generation must match its successful capacity check")
            operation["phase"] = "streaming"
        record = {**operation["record"], "status": "failed"}
        chunks, size, iterator = [], 0, None
        try:
            if cancelled.is_set() or operation["cancelled"]:
                record["status"] = "cancelled"
                return
            iterator = iter(self.transport.stream(deepcopy(operation["messages"]), cancelled))
            for chunk in iterator:
                if cancelled.is_set() or operation["cancelled"]:
                    record["status"] = "cancelled"
                    return
                if not isinstance(chunk, str):
                    raise KernelFailure("model_contract")
                size += len(chunk.encode())
                if size > 64000:
                    self.transport.cancel()
                    raise KernelFailure("model_contract")
                chunks.append(chunk)
                yield chunk
            record["status"] = (
                "cancelled" if cancelled.is_set() or operation["cancelled"] else "completed"
            )
        except BaseException as error:
            record["error_type"] = type(error).__name__
            raise
        finally:
            try:
                if iterator is not None and hasattr(iterator, "close"):
                    iterator.close()
            except BaseException as error:
                record.update(status="failed", cleanup_error_type=type(error).__name__)
                raise
            finally:
                if cancelled.is_set():
                    record["status"] = "cancelled"
                raw = "".join(chunks)
                record.update(
                    response_content=raw, response_sha256=sha256(raw.encode()).hexdigest()
                )
                self._finish(operation, record)

    def cancel(self):
        with self._lock:
            operation = self._operation
            self._cancel_count += 1
            ready = operation is not None and operation["phase"] == "ready"
            if operation is not None:
                operation["cancelled"] = True
                if ready:
                    operation["phase"] = "cancelling"
        try:
            self.transport.cancel()
        finally:
            with self._lock:
                self._cancel_count -= 1
            if ready:
                self._finish(operation, {"status": "cancelled"})
