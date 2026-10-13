from __future__ import annotations

import http.client
import json
import threading

import pytest
from virtual_companion.text.transport import create_server


@pytest.fixture
def server():
    calls = []

    def dispatch(action, payload):
        calls.append((action, payload))
        return {"state": "ready"}

    instance = create_server("a" * 48, dispatch, port=0)
    thread = threading.Thread(target=instance.serve_forever, daemon=True)
    thread.start()
    yield instance.server_address[1], calls
    instance.shutdown()
    instance.server_close()
    thread.join(2)


def request(server, path="/v1/state", body=None, **headers):
    connection = http.client.HTTPConnection("127.0.0.1", server[0], timeout=3)
    data = json.dumps(body or {})
    connection.request("POST", path, data, headers)
    response = connection.getresponse()
    result = response.status, json.loads(response.read())
    connection.close()
    return result


def test_transport_requires_token_and_rejects_browser_origins(server):
    assert request(server)[0] == 401
    assert request(server, Authorization="Bearer " + "b" * 48)[0] == 401
    assert request(server, Authorization="Bearer " + "a" * 48, Origin="https://evil.test")[0] == 403
    assert server[1] == []


def test_non_ascii_authorization_is_rejected_without_handler_crash(server):
    assert request(server, Authorization="Bearer café")[0] == 401
    assert server[1] == []


def test_transport_versioned_schema_and_unknown_routes_fail_before_dispatch(server):
    headers = {"Authorization": "Bearer " + "a" * 48, "Content-Type": "application/json"}
    assert request(server, **headers)[0] == 200
    assert request(server, "/v1/execute", **headers)[0] == 404
    assert request(server, body={"path": "private"}, **headers)[0] == 400
    assert request(server, "/v1/send", {"text": " "}, **headers)[0] == 400
    assert len(server[1]) == 1


def test_transport_cannot_bind_remote_or_use_weak_token():
    with pytest.raises(ValueError):
        create_server("short", lambda *_: {}, port=0)


@pytest.mark.parametrize(
    ("action", "payload"),
    [
        ("memories", {"offset": 0, "limit": 50}),
        ("memoryDetails", {"record_id": "record-1"}),
        (
            "memoryAction",
            {
                "operation_id": "operation-1",
                "action": "correct",
                "record_id": "record-1",
                "expected_version": 1,
                "replacement": "更正后的事实",
            },
        ),
        (
            "memoryAbandon",
            {"job": "background-job-1", "operation_id": "operation-2"},
        ),
    ],
)
def test_transport_accepts_only_bounded_memory_management_routes(server, action, payload):
    headers = {"Authorization": "Bearer " + "a" * 48, "Content-Type": "application/json"}
    assert request(server, f"/v1/{action}", payload, **headers)[0] == 200
    assert server[1][-1] == (action, payload)


@pytest.mark.parametrize(
    ("action", "payload"),
    [
        ("memories", {"offset": -1, "limit": 50}),
        ("memories", {"offset": 0, "limit": 101}),
        ("memoryDetails", {"record_id": ""}),
        (
            "memoryAction",
            {
                "operation_id": "operation-1",
                "action": "forget",
                "record_id": "record-1",
                "expected_version": 1,
                "replacement": "must-be-empty",
            },
        ),
        (
            "memoryAction",
            {
                "operation_id": "operation-1",
                "action": "correct",
                "record_id": "record-1",
                "expected_version": 1,
                "replacement": "事实",
                "unexpected": True,
            },
        ),
        ("memoryAbandon", {"job": "background-job-1", "operation_id": ""}),
    ],
)
def test_transport_rejects_invalid_memory_management_payloads(server, action, payload):
    headers = {"Authorization": "Bearer " + "a" * 48, "Content-Type": "application/json"}
    assert request(server, f"/v1/{action}", payload, **headers)[0] == 400
    assert server[1] == []


def test_transport_rejects_duplicate_json_fields_and_oversize(server):
    for body in ['{"text":"a","text":"b"}', json.dumps({"text": "a" * 8001})]:
        connection = http.client.HTTPConnection("127.0.0.1", server[0], timeout=3)
        connection.request(
            "POST",
            "/v1/send",
            body,
            {
                "Authorization": "Bearer " + "a" * 48,
                "Content-Type": "application/json",
            },
        )
        response = connection.getresponse()
        assert response.status == 400
        response.read()
        connection.close()
    assert not server[1]
