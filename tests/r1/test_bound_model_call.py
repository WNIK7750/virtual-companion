import threading

import pytest
from test_model_provider_contract import route
from virtual_companion.providers.bound_call import BoundProviderCall


class Transport:
    def __init__(self):
        self.calls = []

    def cancel(self):
        pass

    def invoke(self, body, cancelled, **kwargs):
        self.calls.append((body, kwargs))
        return {"text": "candidate", "tool_calls": [], "finish_reason": "stop", "usage": None}


def test_count_and_send_bind_identical_final_schema_body():
    transport, counted = Transport(), []

    def count(body):
        counted.append(body)
        return 100

    call = BoundProviderCall(
        route(),
        transport=transport,
        counter=count,
        stage="reply",
        cancelled=threading.Event(),
        clock=lambda: 1,
        deadline=10,
        output_tokens=256,
        data_scope=("current",),
        schema={"type": "object"},
    )
    request = {
        "messages": [{"role": "user", "content": "x"}],
        "tools": [],
        "max_output_tokens": 256,
    }
    assert call.count(request) == 100
    assert call.invoke(request)["text"] == "candidate"
    assert counted[0] == transport.calls[0][0]
    assert counted[0]["response_format"]["json_schema"]["schema"] == {"type": "object"}
    request["messages"][0]["content"] = "different"
    with pytest.raises(ValueError, match="binding"):
        call.invoke(request)
    assert len(transport.calls) == 1


def test_counter_failure_deadline_and_backwards_clock_never_send():
    now = [1]
    transport = Transport()
    call = BoundProviderCall(
        route(),
        transport=transport,
        counter=lambda _: 100,
        stage="reply",
        cancelled=threading.Event(),
        clock=lambda: now[0],
        deadline=10,
        output_tokens=256,
        data_scope=("current",),
    )
    request = {
        "messages": [{"role": "user", "content": "x"}],
        "tools": [],
        "max_output_tokens": 256,
    }
    call.count(request)
    now[0] = 0
    with pytest.raises(TimeoutError):
        call.invoke(request)
    now[0] = 11
    with pytest.raises(TimeoutError):
        call.count(request)
    assert transport.calls == []
