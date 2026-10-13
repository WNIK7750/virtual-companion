"""Bounded cancellable provider I/O with mandatory pre-send ledger reservation.

No retry, proxy environment, redirect, model substitution or error-body logging.
The caller supplies a verified exact counter and a durable budget ledger. These
are Ports: synthetic tests do not establish remote tokenizer or billing accuracy.
"""

import asyncio
import math
import threading
import time
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256

import httpx

from virtual_companion.boundary.contracts import canonical, identifier

from .contracts import (
    DEEPSEEK_FOCUSED_PROFILE,
    DEEPSEEK_TEXT_PROFILES,
    MAX_PROVIDER_RESPONSE_BYTES,
    ProviderRoute,
    ProviderStreamCapacityError,
)
from .openai_compatible import (
    ContentObservation,
    StreamFailure,
    decode_stream,
    validate_wire_profile,
)
from .text_deltas import TextDeltas


class ProviderFailure(RuntimeError):
    """Only fixed safe codes cross the application boundary."""


def _stream_failure_code(reason):
    return {
        "output_limit_reached": "provider_output_limit_reached",
        "stream_incomplete": "provider_stream_incomplete",
        "response_refused": "provider_response_refused",
        "finish_reason_rejected": "provider_finish_reason_rejected",
    }.get(reason, "provider_protocol_rejected")


class HttpxProviderTransport:
    def __init__(self, route, *, secrets, ledger):
        if not isinstance(route, ProviderRoute) or ledger is None:
            raise ValueError("Explicit provider route and budget ledger required")
        self.route, self.secrets, self.ledger = route, secrets, ledger
        self._active, self._state = threading.Lock(), threading.Lock()
        self._loop, self._task = None, None
        self._metrics = {}

    @property
    def metrics(self):
        with self._state:
            return dict(self._metrics)

    def _mark(self, **values):
        with self._state:
            self._metrics.update(values)

    def cancel(self):
        with self._state:
            if self._loop is not None and self._task is not None:
                self._loop.call_soon_threadsafe(self._task.cancel)

    def review_peer(self):
        fork = getattr(self.ledger, 'review_peer', None)
        return HttpxProviderTransport(self.route, secrets=self.secrets,
                                      ledger=fork() if callable(fork) else self.ledger)

    def invoke(self, body, cancelled, *, stage, data_scope, input_tokens, timeout_seconds,
               on_content=None, structured_content=False):
        began = time.monotonic()
        identifier(stage)
        if type(structured_content) is not bool or (structured_content and (
            on_content is None or not isinstance(body, dict)
            or body.get('response_format', {}).get('type') not in {'json_object', 'json_schema'}
        )):
            raise ValueError("Explicit structured candidate stream required")
        if on_content is not None and (
            not callable(on_content) or not isinstance(body, dict)
            or body.get("tools") or (body.get("response_format") and not structured_content)
        ):
            raise ValueError("Provisional content requires an explicit plain-text consumer")
        if (
            not isinstance(data_scope, tuple)
            or not data_scope
            or set(data_scope) - set(self.route.data_scope)
        ):
            raise ValueError("Request exceeds selected context sending scope")
        if type(input_tokens) is not int or input_tokens < 0:
            raise ValueError("Exact input token count required")
        if (
            type(timeout_seconds) not in {int, float}
            or not math.isfinite(timeout_seconds)
            or not 0 < timeout_seconds <= 300
        ):
            raise ValueError("Bounded remaining provider deadline required")
        if (
            not isinstance(body, dict)
            or body.get("model") != self.route.model
            or body.get("stream") is not True
            or set(body)
            - ({
                "model",
                "messages",
                "stream",
                "stream_options",
                "tools",
                "tool_choice",
                "response_format",
                "temperature",
                "top_p",
                "seed",
                self.route.capabilities.output_field,
            } | ({"thinking"} if self.route.wire_profile in DEEPSEEK_TEXT_PROFILES else set()))
        ):
            raise ValueError("Provider request differs from the explicit route")
        validate_wire_profile(self.route, body, stage=stage)
        sampling = asdict(self.route.sampling) if self.route.sampling is not None else {}
        supplied_sampling = {k: body[k] for k in ("temperature", "top_p", "seed") if k in body}
        if (self.route.wire_profile != DEEPSEEK_FOCUSED_PROFILE
                and supplied_sampling != sampling):
            raise ValueError("Provider sampling differs from the explicit route")
        output_tokens = body.get(self.route.capabilities.output_field)
        if (
            type(output_tokens) is not int
            or output_tokens < 1
            or input_tokens + output_tokens + 1 > self.route.capabilities.context_tokens
        ):
            raise ValueError("Provider context capacity exceeded")
        payload = canonical(deepcopy(body)).encode()
        if len(payload) > 2_000_000:
            raise ValueError("Provider request byte limit exceeded")
        if cancelled.is_set():
            raise ProviderFailure("provider_cancelled")
        if not self._active.acquire(blocking=False):
            raise ProviderFailure("provider_already_active")
        self._mark(started=began, headers=None, first_content=None, finished=None)
        receipt, outcome = None, {"status": "failed", "usage": None, "transmission": "not_sent"}
        try:
            headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
            if self.route.auth == "bearer":
                if self.secrets is None:
                    raise ProviderFailure("provider_secret_unavailable")
                try:
                    credential = self.secrets.get(self.route.secret_ref, self.route.endpoint)
                except Exception:
                    raise ProviderFailure("provider_secret_unavailable") from None
                if (
                    not isinstance(credential, str)
                    or not 1 <= len(credential) <= 8192
                    or any(not 33 <= ord(c) <= 126 for c in credential)
                ):
                    raise ProviderFailure("provider_secret_invalid")
                headers["Authorization"] = "Bearer " + credential
            if cancelled.is_set():
                raise ProviderFailure("provider_cancelled")
            receipt = self.ledger.reserve(
                route=self.route.fingerprint,
                stage=stage,
                request_sha256=sha256(payload).hexdigest(),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            remaining = timeout_seconds - (time.monotonic() - began)
            if remaining <= 0:
                outcome["status"] = "deadline_exceeded"
                raise ProviderFailure("provider_deadline_exceeded")
            try:
                result = asyncio.run(
                    self._exchange(payload, headers, cancelled, remaining, outcome, on_content)
                )
            except asyncio.CancelledError:
                outcome["status"] = "cancelled"
                raise ProviderFailure("provider_cancelled") from None
            except TimeoutError:
                outcome["status"] = "deadline_exceeded"
                raise ProviderFailure("provider_deadline_exceeded") from None
            except httpx.ConnectTimeout:
                raise ProviderFailure("provider_connect_timeout") from None
            except httpx.ConnectError:
                raise ProviderFailure("provider_connect_error") from None
            except httpx.ReadTimeout:
                raise ProviderFailure("provider_read_timeout") from None
            except ProviderFailure:
                raise
            except ProviderStreamCapacityError:
                raise ProviderFailure("provider_stream_buffer_capacity_exceeded") from None
            except Exception:
                raise ProviderFailure("provider_transport_or_protocol_failure") from None
            if cancelled.is_set():
                outcome["status"] = "cancelled"
                raise ProviderFailure("provider_cancelled")
            outcome.update(status="completed", usage=result["usage"])
            return result
        finally:
            try:
                if receipt is not None:
                    # Failure to persist the terminal usage prevents candidate release.
                    self.ledger.finish(receipt, **outcome)
            finally:
                self._mark(finished=time.monotonic())
                self._active.release()

    async def _exchange(self, payload, headers, cancelled, timeout, outcome, on_content=None):
        task = asyncio.current_task()
        with self._state:
            self._loop, self._task = asyncio.get_running_loop(), task

        async def watch():
            while True:
                if cancelled.is_set():
                    task.cancel()
                    return
                await asyncio.sleep(0.025)

        watcher = asyncio.create_task(watch())
        try:
            async with asyncio.timeout(timeout):
                if cancelled.is_set():
                    raise asyncio.CancelledError
                # One scoped client per bounded call avoids sharing cancellation across
                # routes. Connection-pooling performance is not claimed by this adapter.
                async with httpx.AsyncClient(
                    transport=httpx.AsyncHTTPTransport(retries=0, trust_env=False),
                    trust_env=False,
                    follow_redirects=False,
                    timeout=httpx.Timeout(30, connect=1, pool=1),
                ) as client:
                    outcome["transmission"] = "may_have_been_sent"
                    async with client.stream(
                        "POST", self.route.endpoint, content=payload, headers=headers
                    ) as response:
                        self._mark(headers=time.monotonic())
                        outcome["transmission"] = "response_received"
                        if response.status_code != 200:
                            raise ProviderFailure("provider_http_rejected")
                        if (
                            not response.headers.get("Content-Type", "")
                            .lower()
                            .startswith("text/event-stream")
                        ):
                            raise ProviderFailure("provider_stream_unsupported")
                        chunks, size, observation = [], 0, ContentObservation()
                        deltas = TextDeltas() if on_content is not None else None
                        delta_rejected = False
                        async for chunk in response.aiter_bytes():
                            size += len(chunk)
                            if size > MAX_PROVIDER_RESPONSE_BYTES:
                                raise ProviderFailure("provider_response_too_large")
                            if observation.feed(chunk):
                                self._mark(first_content=time.monotonic())
                            chunks.append(chunk)
                            if deltas is not None:
                                try:
                                    pending_deltas = deltas.feed(chunk)
                                except ValueError:
                                    # Stop publication, but retain the bounded response for
                                    # the complete decoder's terminal and usage validation.
                                    deltas, delta_rejected, pending_deltas = None, True, ()
                                for delta in pending_deltas:
                                    if cancelled.is_set():
                                        raise asyncio.CancelledError
                                    on_content(delta)
                                if cancelled.is_set():
                                    raise asyncio.CancelledError
                        try:
                            result = decode_stream(b"".join(chunks))
                        except StreamFailure as error:
                            outcome["usage"] = error.usage
                            raise ProviderFailure(_stream_failure_code(error.reason)) from None
                        outcome["usage"] = result["usage"]
                        if delta_rejected:
                            raise ProviderFailure("provider_protocol_rejected")
                        return result
        finally:
            # I/O has ended. Retire public cancellation handles before a second
            # cancellation can interrupt the watcher's cleanup await.
            with self._state:
                self._loop, self._task = None, None
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
