import http.client
import json
import threading
import time

import pytest
from test_cognition_preparation import basis
from test_cognitive_kernel import kernel
from test_rebuilt_turns import A
from test_text_cancel_boundaries import model_fixture
from virtual_companion.boundary.runtime import CompanionRuntime
from virtual_companion.cognition.preparation import CallCancelled, CallDeadline
from virtual_companion.text.identity import CompanionConfig


@pytest.mark.parametrize("stop", ["cancel", "deadline"])
def test_real_socket_stall_renews_lease_and_interrupts_without_commit(tmp_path, monkeypatch, stop):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    store, item, _, persona = basis(tmp_path)
    model = model_fixture(tmp_path)
    model.profile["server_arguments"] = ["--ctx-size", "4096"]
    model.profile["sampling"] = {"max_tokens": 192}
    entered, release, renewed = threading.Event(), threading.Event(), threading.Event()
    routes, errors = [], []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):  # noqa: N802
            self.rfile.read(int(self.headers["Content-Length"]))
            routes.append(self.path)
            if self.path == "/v1/chat/completions":
                entered.set()
                release.wait(4)
                return
            payload = (
                {"prompt": "synthetic exact template"}
                if self.path == "/apply-template"
                else {"tokens": [1, 2, 3]}
            )
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    listener = threading.Thread(target=server.serve_forever)
    listener.start()
    original_connection = http.client.HTTPConnection
    monkeypatch.setattr(
        "virtual_companion.text.llama.http.client.HTTPConnection",
        lambda host, port, **kw: original_connection("127.0.0.1", server.server_port, **kw),
    )
    original_renew = store.renew

    def renew(*args, **kwargs):
        result = original_renew(*args, **kwargs)
        renewed.set()
        return result

    store.renew = renew

    def clock():
        return time.time_ns() // 1000

    candidate = kernel(store, persona, model)
    candidate.clock = clock
    candidate.lease_us = 1_000_000
    candidate.renewal_margin_us = 700_000
    candidate.deadline_us = clock() + (1_500_000 if stop == "deadline" else 10_000_000)
    config = CompanionConfig.from_mapping(json.loads(item.config_json)["companion_config"])
    runtime = CompanionRuntime(config, store)

    def run():
        try:
            runtime.execute(item.input_id, "worker", candidate, clock=clock, lease_us=1_000_000)
        except Exception as error:
            errors.append(error)

    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert entered.wait(2)
        assert renewed.wait(1), "blocked inference did not renew its durable owner"
        if stop == "cancel":
            store.cancel(A, item.input_id)
        worker.join(2)
        assert not worker.is_alive(), "socket remained blocked after cancellation/deadline"
        assert len(errors) == 1
        assert isinstance(errors[0], CallCancelled if stop == "cancel" else CallDeadline)
        assert store.inspect(A, item.input_id)["commit"] is None
        assert routes == ["/apply-template", "/tokenize", "/v1/chat/completions"]
        assert model.socket is None and model.connection is None
    finally:
        release.set()
        model.cancel()
        worker.join(3)
        server.shutdown()
        server.server_close()
        listener.join(2)
