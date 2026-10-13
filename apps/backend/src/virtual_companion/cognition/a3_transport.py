"""Explicit bounded schema calls for the isolated A3 experiment."""

import json
import threading
import time
from copy import deepcopy
from hashlib import sha256

from virtual_companion.boundary.contracts import canonical

from .errors import ProtocolInvalid


class SchemaTransport:
    def __init__(
        self,
        transports,
        reservations,
        *,
        max_calls,
        max_output_tokens,
        observe,
        validate_response=None,
    ):
        if (
            not transports
            or set(transports) != set(reservations)
            or any(type(n) is not int or n <= 0 for n in reservations.values())
            or type(max_calls) is not int
            or max_calls <= 0
            or type(max_output_tokens) is not int
            or max_output_tokens <= 0
        ):
            raise ValueError("Explicit stage reservations and total budget required")
        self.transports = dict(transports)
        self.reservations = dict(reservations)
        self.max_calls, self.max_output_tokens = max_calls, max_output_tokens
        self.observe = observe
        self.validate_response = validate_response
        self.calls = self.reserved = 0
        self.deadline_expired = False
        self._lock = threading.Lock()
        self._active = {}

    def cancel(self):
        with self._lock:
            # Hold the ownership lock through cancel so an old cancellation
            # cannot reach a new call sharing the same transport.
            errors = []
            for model in self._active.values():
                try:
                    model.cancel()
                except Exception as error:
                    errors.append(error)
            if errors:
                raise errors[0]

    def __call__(self, stage, messages, schema, cancelled):
        with self._lock:
            if self.deadline_expired:
                raise TimeoutError("A3 session wall budget exhausted")
            if self.transports[stage] in self._active.values():
                raise RuntimeError("A3 stage invocation already active")
            reservation = self.reservations[stage]
            if self.calls >= self.max_calls or self.reserved + reservation > self.max_output_tokens:
                raise ValueError("A3 invocation budget exhausted")
            self.calls += 1
            self.reserved += reservation
            model = self.transports[stage]
            call_id = self.calls
            self._active[call_id] = model
        record = {
            "call_id": call_id,
            "stage": stage,
            "messages": deepcopy(messages),
            "schema_sha256": sha256(canonical(schema).encode()).hexdigest(),
            "request_sha256": sha256(canonical(messages).encode()).hexdigest(),
            "output_reservation": reservation,
            "status": "failed",
        }
        start = time.monotonic()
        chunks, size = [], 0
        try:
            if cancelled.is_set():
                return None
            messages = deepcopy(messages)
            bind_schema = getattr(model, "bind_schema", None)
            if bind_schema is not None:
                bind_schema(deepcopy(schema))
            model.check_context(deepcopy(messages))
            if self.deadline_expired:
                raise TimeoutError("A3 session wall budget exhausted")
            if cancelled.is_set():
                return None
            iterator = model.stream(deepcopy(messages), cancelled)
            try:
                for chunk in iterator:
                    if self.deadline_expired:
                        raise TimeoutError("A3 session wall budget exhausted")
                    if cancelled.is_set():
                        return None
                    if not isinstance(chunk, str):
                        raise ValueError("Nontext schema response")
                    size += len(chunk.encode())
                    if size > 64000:
                        model.cancel()
                        raise ValueError("A3 response byte budget exceeded")
                    chunks.append(chunk)
            finally:
                if hasattr(iterator, "close"):
                    iterator.close()
            text = "".join(chunks)
            if self.deadline_expired:
                raise TimeoutError("A3 session wall budget exhausted")
            if not text.strip():
                record["status"] = "empty"
                return None

            def unique(pairs):
                result = {}
                for key, value in pairs:
                    if key in result:
                        raise ValueError("Duplicate response key")
                    result[key] = value
                return result

            result = json.loads(text, object_pairs_hook=unique)
            canonical(result)
            if not isinstance(result, dict):
                raise ValueError("JSON object required")
            if self.validate_response is not None:
                result = self.validate_response(stage, result, deepcopy(messages))
            record["status"] = "completed"
            return result
        except ProtocolInvalid as error:
            record["status"] = "protocol_invalid"
            record["semantic_status"] = "unknown"
            record["error_type"] = type(error).__name__
            raise
        except BaseException as error:
            record["error_type"] = type(error).__name__
            raise
        finally:
            record.update(
                response_content="".join(chunks), elapsed_seconds=time.monotonic() - start
            )
            record["model_metrics"] = deepcopy(getattr(model, "metrics", {}))
            if cancelled.is_set():
                record["status"] = "cancelled"
            elif self.deadline_expired:
                record["status"] = "deadline_exceeded"
            with self._lock:
                self._active.pop(call_id, None)
            self.observe(record)
