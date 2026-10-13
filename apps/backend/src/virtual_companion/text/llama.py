from __future__ import annotations

import hashlib
import http.client
import io
import json
import os
import secrets
import select
import socket
import subprocess
import threading
import time
from copy import deepcopy
from pathlib import Path

from virtual_companion.providers.openai_compatible import strict_json

from .errors import ContextCheckError, ContextLimitError


def verify_file(path: Path, digest: str):
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != digest:
        raise ValueError("Artifact hash mismatch")


class VisibleText:
    def __init__(self):
        self.buffer, self.hidden = "", False

    def feed(self, text):
        self.buffer += text
        output = ""
        while self.buffer:
            tag = "</think>" if self.hidden else "<think>"
            index = self.buffer.find(tag)
            if index >= 0:
                if not self.hidden:
                    output += self.buffer[:index]
                self.buffer = self.buffer[index + len(tag) :]
                self.hidden = not self.hidden
                continue
            keep = 0
            for size in range(1, min(len(tag), len(self.buffer) + 1)):
                if self.buffer.endswith(tag[:size]):
                    keep = size
            safe = self.buffer[:-keep] if keep else self.buffer
            if not self.hidden:
                output += safe
            self.buffer = self.buffer[-keep:] if keep else ""
            break
        return output

    def finish(self):
        return ""  # A partial delimiter is never user-facing content.


def _wait_socket(stream, cancelled, deadline, *, writing=False):
    while True:
        if cancelled.is_set():
            raise OSError("Model request cancelled")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Model I/O timeout")
        readable, writable, _ = select.select(
            [] if writing else [stream], [stream] if writing else [], [], min(0.1, remaining)
        )
        if readable or writable:
            return


class _ResponseReader(io.RawIOBase):
    def __init__(self, stream, cancelled):
        super().__init__()
        # HTTP/1.0 may close the connection owner while its response still reads.
        self.stream = stream.dup()
        self.cancelled = cancelled

    def readable(self):
        return True

    def readinto(self, buffer):
        deadline = time.monotonic() + 30
        while True:
            _wait_socket(self.stream, self.cancelled, deadline)
            try:
                return self.stream.recv_into(buffer)
            except BlockingIOError:
                continue

    def close(self):
        if not self.closed:
            self.stream.close()
        super().close()


class _CancellableSocket:
    def __init__(self, stream, cancelled):
        self.stream, self.cancelled = stream, cancelled
        stream.setblocking(False)

    def sendall(self, data):
        pending = memoryview(data)
        deadline = time.monotonic() + 30
        while pending:
            _wait_socket(self.stream, self.cancelled, deadline, writing=True)
            try:
                written = self.stream.send(pending)
            except BlockingIOError:
                continue
            if written == 0:
                raise ConnectionError("Model socket closed")
            pending = pending[written:]

    def makefile(self, mode):
        if mode != "rb":
            raise ValueError("Read-only HTTP response required")
        return io.BufferedReader(_ResponseReader(self.stream, self.cancelled))

    def close(self):
        self.stream.close()


def _slot_capacity(profile):
    arguments = profile["server_arguments"]
    slots = int(arguments[arguments.index("--parallel") + 1]) if "--parallel" in arguments else 1
    return int(arguments[arguments.index("--ctx-size") + 1]) // slots


class _PreparedLlamaRequest:
    """Freeze actual wire fields and capacity for a single checked transmission."""

    def __init__(self, model, messages, *, stage=False, stage_name=None, schema=None,
                 output_tokens=None):
        self.model = model
        self.stage = stage
        self._payload = deepcopy(
            model._stage_payload(messages) if stage else model._payload(messages)
        )
        if output_tokens is not None:
            self._payload["max_tokens"] = output_tokens
        if schema is not None:
            self._payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": stage_name, "strict": True, "schema": deepcopy(schema)},
            }
        self.process = model.process
        self.capacity = _slot_capacity(model.profile)
        self.reserve = self._payload["max_tokens"]
        if type(self.reserve) is not int or self.reserve < 1 or self.capacity < 1:
            raise ContextCheckError()
        self.checked = self.used = False
        self.lock = threading.Lock()

    def _current(self):
        if (self.model.process is not self.process or self.process is None
                or self.process.poll() is not None):
            raise RuntimeError("Prepared model process changed")

    def check_context(self):
        with self.lock:
            if self.used:
                raise ValueError("Prepared request already consumed")
            self.checked = False
            self._current()
            tokens = self.model._count_payload(deepcopy(self._payload))
            self._current()
            if tokens + self.reserve >= self.capacity:
                raise ContextLimitError()
            self.checked = True

    def stream(self, cancelled):
        with self.lock:
            if self.used or not self.checked:
                raise ValueError("Unused checked request required")
            self.used = True
        if cancelled.is_set():
            return
        self._current()
        stream = self.model._stream_payload(
            deepcopy(self._payload), cancelled, **({"stage": True} if self.stage else {})
        )
        try:
            for delta in stream:
                if cancelled.is_set():
                    return
                self._current()
                yield delta
            if not cancelled.is_set():
                self._current()
        finally:
            stream.close()


class LlamaModel:
    supports_stage_metadata = True
    supports_stage_output_override = True

    def __init__(self, profile, repository):
        self.profile, self.repository = profile, repository
        self.process = None
        self.connection = None
        self.socket = None
        self.lock = threading.Lock()
        self.token = secrets.token_hex(32)

    def review_peer(self):
        """Independent I/O to the same loaded server; no process or route change."""
        peer = LlamaModel(self.profile, self.repository)
        peer.process, peer.token = self.process, self.token
        return peer

    def start(self):
        profile = self.profile
        for item in profile["artifacts"]:
            verify_file(Path(item["path"]), item["sha256"])
        verify_file(self.repository / profile["prompt_path"], profile["prompt_sha256"])
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", 18083)) == 0:
                raise RuntimeError("Model port occupied")
        server = Path(profile["server"])
        arguments = [
            str(server),
            "--model",
            profile["model"],
            *profile["server_arguments"],
            "--api-key",
            self.token,
        ]
        environment = {
            key: value for key, value in os.environ.items() if not key.startswith("LLAMA_ARG_")
        }
        environment["PATH"] = (
            str(server.parent)
            + os.pathsep
            + profile["cuda_bin"]
            + os.pathsep
            + os.environ.get("PATH", "")
        )
        self.process = subprocess.Popen(  # noqa: S603 - verified fixed executable and arguments
            arguments,
            cwd=server.parent,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError("Model exited during load")
            connection = http.client.HTTPConnection("127.0.0.1", 18083, timeout=1)
            try:
                connection.request(
                    "GET", "/health", headers={"Authorization": "Bearer " + self.token}
                )
                response = connection.getresponse()
                response.read()
                if response.status == 200:
                    return
            except OSError:
                pass
            finally:
                connection.close()
            time.sleep(0.2)
        self.close()
        raise TimeoutError("Model loading timeout")

    def _payload(self, messages):
        if not self.process or self.process.poll() is not None:
            raise RuntimeError("Model not running")
        prompt = (self.repository / self.profile["prompt_path"]).read_text(encoding="utf-8")
        if hashlib.sha256(prompt.encode()).hexdigest() != self.profile["prompt_sha256"]:
            raise ValueError("Prompt hash mismatch")
        return {
            **self.profile["sampling"],
            "model": "Qwen3-4B-Q4_K_M.gguf",
            "messages": [{"role": "system", "content": prompt}, *messages],
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }

    def _context_request(self, route, payload):
        connection = http.client.HTTPConnection("127.0.0.1", 18083, timeout=2)
        try:
            connection.request(
                "POST",
                route,
                json.dumps(payload),
                {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + self.token,
                },
            )
            response = connection.getresponse()
            if response.status != 200:
                raise ContextCheckError()
            data = response.read(2_000_001)
            if len(data) > 2_000_000:
                raise ContextCheckError()
            result = json.loads(data)
            if not isinstance(result, dict):
                raise ContextCheckError()
            return result
        finally:
            connection.close()

    def count_context(self, messages):
        """Count the exact chat template, including the fixed system prompt."""
        try:
            payload = self._payload(messages)
            return self._count_payload(payload)
        except Exception as error:
            raise ContextCheckError() from error

    def _count_payload(self, payload):
        try:
            template = self._context_request("/apply-template", payload)["prompt"]
            if not isinstance(template, str):
                raise ContextCheckError()
            tokens = self._context_request(
                "/tokenize",
                {
                    "content": template,
                    "add_special": True,
                    "parse_special": True,
                },
            )["tokens"]
            if (
                not isinstance(tokens, list)
                or not tokens
                or any(type(token) is not int for token in tokens)
            ):
                raise ContextCheckError()
            return len(tokens)
        except Exception as error:
            raise ContextCheckError() from error

    def check_context(self, messages):
        """Measure the exact chat template before the application writes an attempt."""
        try:
            tokens = self.count_context(messages)
            capacity = _slot_capacity(self.profile)
            reserve = self.profile["sampling"]["max_tokens"]
            if type(reserve) is not int or reserve < 1 or capacity < 1:
                raise ContextCheckError()
        except Exception as error:
            raise ContextCheckError() from error
        # Leave one position free, as required by the fixed llama.cpp slot boundary.
        if tokens + reserve >= capacity:
            raise ContextLimitError()

    def _filter_content(self, text, display_filter):
        return display_filter.feed(text)

    def prepare_request(self, messages):
        return _PreparedLlamaRequest(self, messages)

    def prepare_stage_request(self, messages, *, stage=None, schema=None, output_tokens=None):
        """Prepare complete internal stage messages without a second system prompt.

        The coordinator binds the stage prompt/version. This shares the existing
        frozen capacity, transport and process checks. Supplied stage schemas use
        the same fixed llama.cpp JSON schema wire contract as the A3 adapter.
        """
        if (stage is None) != (schema is None):
            raise ValueError("Stage and schema must be bound together")
        if output_tokens is not None and (
            type(output_tokens) is not int or not 1 <= output_tokens <= 4096
        ):
            raise ValueError("Bounded stage output required")
        return _PreparedLlamaRequest(self, messages, stage=True, stage_name=stage, schema=schema,
                                     output_tokens=output_tokens)

    def _stage_payload(self, messages):
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError("Explicit system and user messages required")
        for index, message in enumerate(messages):
            if (
                not isinstance(message, dict)
                or set(message) != {"role", "content"}
                or not isinstance(message["content"], str)
                or not message["content"].strip()
                or message["role"] not in ({"system"} if index == 0 else {"user", "assistant"})
            ):
                raise ValueError("One initial system and explicit text messages required")
        if messages[-1]["role"] != "user":
            raise ValueError("Final user input required")
        # Keep baseline artifact verification and all sampling/route fields.
        payload = LlamaModel._payload(self, [])
        payload["messages"] = deepcopy(messages)
        return payload

    def stream(self, messages, cancelled):
        payload = self._payload(messages)
        yield from self._stream_payload(payload, cancelled)

    def _stream_payload(self, payload, cancelled, *, stage=False):
        connection = http.client.HTTPConnection("127.0.0.1", 18083, timeout=1)
        visible = VisibleText()
        response = None
        try:
            with self.lock:
                if cancelled.is_set():
                    return
                self.connection = connection
            # Connect is bounded separately; register the socket before a request
            # write can block. Never let request() reconnect after cancel closes it.
            connection.connect()
            with self.lock:
                if cancelled.is_set():
                    return
                connection.auto_open = 0
                self.socket = connection.sock
                connection.sock = _CancellableSocket(self.socket, cancelled)
            connection.request(
                "POST",
                "/v1/chat/completions",
                json.dumps(payload),
                {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + self.token,
                },
            )
            if cancelled.is_set():
                return
            response = connection.getresponse()
            if response.status != 200:
                raise RuntimeError("Model rejected request")
            done = False
            stopped = False
            usage = None
            while not cancelled.is_set():
                line = response.readline(128000)
                if not line:
                    break
                if not line.startswith(b"data: "):
                    continue
                if line.strip() == b"data: [DONE]":
                    done = True
                    break
                item = strict_json(line[6:])
                if not isinstance(item, dict) or not isinstance(item.get("choices"), list):
                    raise RuntimeError("Invalid model event")
                if "error" in item:
                    raise RuntimeError("Model stream failed")
                # Match the strict API wire contract without buffering the whole
                # local transcript. b10516 may emit usage after its stop choice.
                reported = item.get("usage")
                if reported is not None:
                    keys = ("prompt_tokens", "completion_tokens", "total_tokens")
                    if (
                        not isinstance(reported, dict)
                        or any(type(reported.get(k)) is not int or reported[k] < 0 for k in keys)
                        or reported["total_tokens"]
                        != reported["prompt_tokens"] + reported["completion_tokens"]
                        or (usage is not None and usage != reported)
                    ):
                        raise RuntimeError("Invalid or conflicting model usage")
                    usage = reported
                choices = item["choices"]
                if choices:
                    if (
                        len(choices) != 1 or not isinstance(choices[0], dict)
                        or type(choices[0].get("index")) is not int
                        or choices[0]["index"] != 0 or stopped
                    ):
                        raise RuntimeError("Multiple or late model choices")
                    choice = choices[0]
                    delta = choice.get("delta")
                    if not isinstance(delta, dict) or delta.get("refusal"):
                        raise RuntimeError("Invalid or refused model response")
                    if delta.get("tool_calls") or delta.get("function_call"):
                        raise RuntimeError("Model did not complete a text response")
                    text = delta.get("content")
                    if text is not None and not isinstance(text, str):
                        raise RuntimeError("Invalid model content")
                    # [DONE] closes SSE transport; it does not establish that
                    # generation completed. Never commit length/filter/tool
                    # termination as a complete text answer.
                    finish_reason = choice.get("finish_reason")
                    if finish_reason is not None and finish_reason != "stop":
                        raise RuntimeError("Model did not complete a text response")
                    if finish_reason == "stop":
                        stopped = True
                    # Internal JSON may contain literal <think> delimiters.
                    # Separate reasoning deltas remain excluded by the decoder.
                    output = (text or "") if stage else self._filter_content(text or "", visible)
                    if output:
                        yield output
            if not (done and stopped) and not cancelled.is_set():
                raise RuntimeError("Incomplete model stream")
        finally:
            if response is not None:
                response.close()
            connection.close()
            with self.lock:
                self.connection = self.socket = None

    def cancel(self):
        with self.lock:
            if self.socket:
                try:
                    self.socket.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            if self.connection:
                self.connection.close()

    def close(self):
        self.cancel()
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
