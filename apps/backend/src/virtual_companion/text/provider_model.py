"""Explicit provider binding for checked desktop stages; no route fallback."""

import asyncio
import threading
import time
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from queue import Empty

from jsonschema import Draft202012Validator

from virtual_companion.boundary.contracts import canonical, identifier
from virtual_companion.providers.contracts import (
    MAX_PROVIDER_RESPONSE_BYTES,
    ProviderStreamCapacityError,
)
from virtual_companion.providers.openai_compatible import build_request

MAX_PENDING_STREAM_EVENTS = 4096


class ProviderModel:
    supports_stage_metadata = True
    supports_stage_output_override = True

    def __init__(self, route, *, counter, transport, output_tokens=512, timeout_seconds=45):
        if not route.capabilities.streaming or not route.capabilities.structured_output:
            raise ValueError("Provider requires streaming and structured output")
        if not callable(counter) or not callable(getattr(counter, "local_byte_bound", None)):
            raise ValueError("Pinned provider counter required")
        if type(output_tokens) is not int or not 1 <= output_tokens <= 4096:
            raise ValueError("Bounded provider output required")
        if type(timeout_seconds) not in {int, float} or not 0 < timeout_seconds <= 120:
            raise ValueError("Bounded provider deadline required")
        self.route, self.counter, self.transport = route, counter, transport
        self.output_tokens, self.timeout_seconds = output_tokens, timeout_seconds
        self.lock = threading.Lock()
        self._failed = threading.Event()
        self._streams_lock = threading.Lock()
        self._streams = set()

    @property
    def failed(self):
        return self._failed.is_set()

    def review_peer(self):
        fork = getattr(self.transport, 'review_peer', None)
        if not callable(fork):
            return self
        peer = ProviderModel(self.route, counter=self.counter, transport=fork(),
                             output_tokens=self.output_tokens,
                             timeout_seconds=self.timeout_seconds)
        peer._failed = self._failed
        return peer

    def prepare_stage_request(self, messages, *, stage=None, schema=None, output_tokens=None):
        if self.failed:
            raise ValueError("Provider counter compatibility failed; route disabled")
        identifier(stage)
        if not isinstance(schema, dict) or schema.get("type") != "object":
            raise ValueError("Explicit host stage schema required")
        Draft202012Validator.check_schema(schema)
        reserve = self.output_tokens if output_tokens is None else output_tokens
        if type(reserve) is not int or not 1 <= reserve <= 4096:
            raise ValueError("Bounded provider output required")
        return _Prepared(self, messages, stage, schema, reserve)

    def cancel(self):
        with self._streams_lock:
            for stopped in self._streams:
                stopped.set()
        self.transport.cancel()

    def close(self):
        self.cancel()


class _StreamCancellation:
    def __init__(self, cancelled, stopped):
        self.cancelled, self.stopped = cancelled, stopped

    def is_set(self):
        return self.cancelled.is_set() or self.stopped.is_set()


@dataclass(frozen=True, slots=True)
class _ProducerTerminal:
    """One independently completed producer result, never a queue item."""

    text: str | None = None
    error: Exception | None = None


class _ContentBuffer:
    """Ordered pending text bounded by bytes and events; overflow never pauses I/O."""

    def __init__(self, max_bytes, max_events):
        self.max_bytes, self.max_events = max_bytes, max_events
        self._condition = threading.Condition()
        self._events = deque()
        self._bytes = 0
        self._terminal = None
        self._closed = False

    def append(self, text):
        size = len(text.encode())
        with self._condition:
            if self._closed or self._terminal is not None:
                raise InterruptedError("Provider content buffer closed")
            if self._bytes + size > self.max_bytes or len(self._events) >= self.max_events:
                raise ProviderStreamCapacityError("Provider stream buffer capacity exceeded")
            self._events.append((text, size))
            self._bytes += size
            self._condition.notify()

    def finish(self, terminal):
        with self._condition:
            if self._terminal is not None:
                raise ValueError("Provider producer already finished")
            self._terminal = terminal
            if terminal.error is not None or self._closed:
                self._events.clear()
                self._bytes = 0
            self._condition.notify_all()

    def take(self, timeout):
        with self._condition:
            if not self._events and self._terminal is None and not self._closed:
                self._condition.wait(timeout)
            if self._closed:
                raise InterruptedError("Provider content buffer closed")
            if self._terminal is not None and self._terminal.error is not None:
                return "terminal", self._terminal
            if self._events:
                text, size = self._events.popleft()
                self._bytes -= size
                return "delta", text
            if self._terminal is not None:
                return "terminal", self._terminal
            raise Empty

    @property
    def terminal(self):
        with self._condition:
            return self._terminal

    def close(self):
        with self._condition:
            self._closed = True
            self._events.clear()
            self._bytes = 0
            self._condition.notify_all()


class _Prepared:
    def __init__(self, model, messages, stage, schema, output_tokens):
        self.model, self.stage = model, stage
        self._body = build_request(
            model.route,
            deepcopy(messages),
            output_tokens=output_tokens,
            schema=deepcopy(schema),
            stage=stage,
        )
        self.reserve = self._body[model.route.capabilities.output_field]
        self._fingerprint = sha256(canonical(self._body).encode()).hexdigest()
        self.checked = self.used = False

    def _check(self):
        if (
            self.used
            or self.model.failed
            or (sha256(canonical(self._body).encode()).hexdigest() != self._fingerprint)
        ):
            raise ValueError("Frozen provider request changed, failed or consumed")

    def check_context(self):
        self._check()
        self.count = self.model.counter(deepcopy(self._body))
        self.bound = self.model.counter.local_byte_bound(deepcopy(self._body))
        if (
            type(self.count) is not int
            or type(self.bound) is not int
            or not 0 < self.count <= self.bound
            or self.count + self.reserve >= self.model.route.capabilities.context_tokens
        ):
            raise ValueError("Provider context count rejected")
        self.checked = True

    def stream(self, cancelled):
        self._check()
        if not self.checked:
            raise ValueError("Provider request preflight required")
        if cancelled.is_set():
            raise InterruptedError("Provider request cancelled")
        self.used = True
        deadline = time.monotonic() + self.model.timeout_seconds
        if self.stage not in {'demand_control', 'demand_repair'}:
            yield self._invoke(cancelled, deadline)['text']
            return

        # I/O owns the producer terminal. Semantic review, presentation and ACK
        # consume these original ordered deltas without pausing network reads.
        observations = _ContentBuffer(MAX_PROVIDER_RESPONSE_BYTES, MAX_PENDING_STREAM_EVENTS)
        stopped = threading.Event()
        signal = _StreamCancellation(cancelled, stopped)

        def check_delivery():
            if cancelled.is_set():
                raise asyncio.CancelledError
            if time.monotonic() >= deadline:
                raise TimeoutError("Provider request deadline")
            if stopped.is_set():
                raise asyncio.CancelledError

        def deliver(delta):
            check_delivery()
            observations.append(delta.text)

        def produce():
            try:
                response = self._invoke(signal, deadline, on_content=deliver)
                terminal = _ProducerTerminal(text=response['text'])
            except Exception as error:
                terminal = _ProducerTerminal(error=error)
            observations.finish(terminal)

        def check_consumption():
            if cancelled.is_set():
                raise InterruptedError("Provider request cancelled")
            if time.monotonic() >= deadline:
                terminal = observations.terminal
                if terminal is not None and terminal.error is not None:
                    raise terminal.error
                raise TimeoutError("Provider request deadline")

        worker = threading.Thread(target=produce, name='desktop-provider-stream', daemon=True)
        with self.model._streams_lock:
            self.model._streams.add(stopped)
        text = ""
        try:
            worker.start()
            while True:
                check_consumption()
                try:
                    kind, value = observations.take(timeout=0.025)
                except Empty:
                    continue
                if kind == 'terminal':
                    if value.error is not None:
                        raise value.error
                    check_consumption()
                    if stopped.is_set():
                        raise InterruptedError("Provider request cancelled")
                    if text and text != value.text:
                        raise ValueError("Provider terminal differs from observed content")
                    if not text:
                        yield value.text
                    return
                check_consumption()
                if stopped.is_set():
                    raise InterruptedError("Provider request cancelled")
                text += value
                yield value
        finally:
            stopped.set()
            observations.close()
            if worker.is_alive():
                self.model.transport.cancel()
            if worker.ident is not None:
                worker.join(timeout=2)
            with self.model._streams_lock:
                self.model._streams.discard(stopped)

    def _invoke(self, cancelled, deadline, *, on_content=None):
        with self.model.lock:
            if cancelled.is_set():
                raise InterruptedError("Provider request cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Provider request deadline")
            ledger = getattr(self.model.transport, "ledger", None)
            if isinstance(ledger, ConservativeReservation):
                ledger.bind(self._fingerprint, self.count, self.bound)
            response = self.model.transport.invoke(
                deepcopy(self._body),
                cancelled,
                stage=self.stage,
                data_scope=self.model.route.data_scope,
                input_tokens=self.count,
                timeout_seconds=remaining,
                **({'on_content': on_content, 'structured_content': True}
                   if on_content is not None else {}),
            )
            usage = response.get("usage")
            if not isinstance(usage, dict) or usage.get("prompt_tokens") != self.count:
                self.model._failed.set()
                raise ValueError("Provider hosted input count differs; route disabled")
            if response.get("finish_reason") != "stop" or response.get("tool_calls"):
                raise ValueError("Incomplete provider terminal")
            if cancelled.is_set():
                raise InterruptedError("Provider request cancelled")
            return response


class ConservativeReservation:
    """Keep byte-bound reservations until hosted equivalence is proven per call."""

    def __init__(self, ledger):
        self.ledger = ledger
        self.pending = None

    def bind(self, fingerprint, count, bound):
        self.pending = (fingerprint, count, bound)

    def review_peer(self):
        return ConservativeReservation(self.ledger)

    def reserve(self, **values):
        expected, self.pending = self.pending, None
        if expected is None or (values["request_sha256"], values["input_tokens"]) != expected[:2]:
            raise ValueError("Provider count/reservation binding changed")
        return self.ledger.reserve(**{**values, "input_tokens": expected[2]})

    def finish(self, receipt, **values):
        return self.ledger.finish(receipt, **values)


def load_provider_model(config_path, *, data_root):
    """Load only explicit local configuration; never load executable counter plugins."""
    import json
    from pathlib import Path

    from virtual_companion.providers.budget import ProviderBudgetLedger
    from virtual_companion.providers.contracts import (
        DEEPSEEK_FOCUSED_PROFILE,
        DEEPSEEK_TEXT_PROFILES,
        ProviderCapabilities,
        ProviderRoute,
    )
    from virtual_companion.providers.httpx_transport import HttpxProviderTransport
    from virtual_companion.providers.windows_secrets import WindowsSecretStore

    from .provider_counter import DeepSeekCounter

    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    required = {
        "revision",
        "data_root",
        "route",
        "secret_store",
        "tokenizer",
        "ledger",
        "policy",
        "prior_ledgers",
        "cumulative_token_limit",
        "output_tokens",
        "timeout_seconds",
    }
    if set(config) != required or type(config["revision"]) is not int or config["revision"] < 1:
        raise ValueError("Complete versioned provider configuration required")
    if Path(config["data_root"]).resolve() != Path(data_root).resolve():
        raise ValueError("Provider configuration belongs to another data root")
    values = deepcopy(config["route"])
    values["capabilities"] = ProviderCapabilities(**values["capabilities"])
    values["data_scope"] = tuple(values["data_scope"])
    route = ProviderRoute(**values)
    if route.wire_profile not in DEEPSEEK_TEXT_PROFILES or set(route.data_scope) != {
        "current",
        "history",
        "valid_memory",
        "derived",
    }:
        raise ValueError("Desktop requires explicit full companion processing scope")
    secrets = WindowsSecretStore(config["secret_store"])
    # Check only managed ciphertext presence. Decryption remains inside transport.
    if not secrets._path(route.secret_ref).is_file():
        raise ValueError("Managed provider credential missing")
    counter = (DeepSeekCounter(config["tokenizer"], wire_profile=route.wire_profile)
               if route.wire_profile == DEEPSEEK_FOCUSED_PROFILE
               else DeepSeekCounter(config["tokenizer"]))
    ledger = ProviderBudgetLedger(config["ledger"], config["policy"])
    if route.fingerprint not in ledger.policy["routes"]:
        raise ValueError("Provider route is not covered by this budget")
    budget = ledger.summary()
    if (
        budget["reserved_calls"] >= ledger.policy["max_calls"]
        or budget["reserved_output_tokens"] + config["output_tokens"]
        > ledger.policy["max_output_tokens"]
        or budget["reserved_cost_microusd"] >= ledger.policy["max_cost_microusd"]
    ):
        raise ValueError("Provider budget exhausted")
    inherited = _InheritedBudget(ledger, config["prior_ledgers"], config["cumulative_token_limit"])
    inherited.check(0)
    transport = HttpxProviderTransport(
        route, secrets=secrets, ledger=ConservativeReservation(inherited)
    )
    model = ProviderModel(
        route,
        counter=counter,
        transport=transport,
        output_tokens=config["output_tokens"],
        timeout_seconds=config["timeout_seconds"],
    )
    model.config_revision = config["revision"]
    model.config_fingerprint = sha256(canonical(config).encode()).hexdigest()
    return model


class _InheritedBudget:
    """Explicit historical accounts remain part of the cumulative allowance."""

    def __init__(self, ledger, prior_paths, limit):
        from pathlib import Path

        if (
            not isinstance(prior_paths, list)
            or not prior_paths
            or type(limit) is not int
            or not 0 < limit <= 100_000_000
        ):
            raise ValueError("Inherited provider accounting and bounded allowance required")
        self.ledger, self.limit = ledger, limit
        self.paths = [Path(p).resolve() for p in prior_paths] + [ledger.path]
        self._lock = threading.Lock()
        if len(set(self.paths)) != len(self.paths):
            raise ValueError("Duplicate provider accounting path")

    def check(self, additional):
        import sqlite3

        total = 0
        for path in self.paths:
            with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
                # Reservations include unknown outcomes and are never refunded.
                total += db.execute(
                    "SELECT coalesce(sum(input_tokens+output_tokens),0) FROM provider_calls"
                ).fetchone()[0]
        if total + additional > self.limit:
            raise ValueError("Cumulative provider allowance exhausted")

    def reserve(self, **values):
        with self._lock:
            self.check(values["input_tokens"] + values["output_tokens"])
            return self.ledger.reserve(**values)

    def finish(self, receipt, **values):
        return self.ledger.finish(receipt, **values)
