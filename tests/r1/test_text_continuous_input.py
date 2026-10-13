"""Actual default dispatch with explicit CPU replies, never semantic scores.

Only the HTTP server and provider transport are replaced. There is no socket,
Electron, hosted model, credential, or private data in this fixture. Its explicit
opinions establish wiring and source scope; root must judge real model effects.
"""

import io
import json
import os
import threading
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator
from test_default_text_delivery import wait_for
from unit_review_recording import bound_unit_packets as bound_unit_packets
from unit_review_recording import recorded_review_packet
from virtual_companion.boundary.contracts import canonical
from virtual_companion.cognition.stream_unit_review import semantic_review_view
from virtual_companion.providers.contracts import ProviderCapabilities, ProviderRoute
from virtual_companion.providers.text_deltas import UnapprovedTextDelta
from virtual_companion.text import __main__ as entry
from virtual_companion.text.fake import synthetic_unit_opinion
from virtual_companion.text.provider_model import ProviderModel
from virtual_companion.text.transport import validate


class Counter:
    def __call__(self, body):
        return 17

    def local_byte_bound(self, body):
        return len(canonical(body).encode())


class Supplier:
    def __init__(self, shared=None):
        self.shared = shared or SimpleNamespace(
            plans=[],
            records=[],
            order=[],
            lock=threading.RLock(),
            entered=threading.Event(),
            release=threading.Event(),
            closed=threading.Event(),
        )

    def review_peer(self):
        return Supplier(self.shared)

    def cancel(self):
        self.shared.release.set()

    def invoke(self, body, cancelled, **options):
        assert options["input_tokens"] == 17
        value = {
            "stage": options["stage"],
            "body": deepcopy(body),
            "body_sha256": sha256(canonical(body).encode()).hexdigest(),
        }
        with self.shared.lock:
            self.shared.records.append(value)
        if options["stage"] == "stream_unit_review":
            packet = recorded_review_packet(body["messages"])
            assert semantic_review_view(packet) == json.loads(body["messages"][-1]["content"])
            value["sealed_packet"] = packet
            value["sealed_sha256"] = sha256(canonical(packet).encode()).hexdigest()
            response = synthetic_unit_opinion(json.loads(body["messages"][-1]["content"]))
        else:
            assert options["stage"] == "demand_control", "No additional repair/memory/style call"
            with self.shared.lock:
                assert self.shared.plans, "Only the explicit CPU generation sequence is allowed"
                plan = self.shared.plans.pop(0)
            if "blocked_prefix" in plan:
                prefix = plan["blocked_prefix"]
                if prefix:
                    fragment = (
                        '{"action":"answer","text":' + json.dumps(prefix, ensure_ascii=False)[:-1]
                    )
                    options["on_content"](UnapprovedTextDelta(1, fragment))
                self.shared.entered.set()
                try:
                    while not self.shared.release.wait(0.01):
                        if cancelled.is_set():
                            break
                    if plan.get('complete_on_release') and not cancelled.is_set():
                        options['on_content'](UnapprovedTextDelta(2, '"}'))
                        return {'text': canonical({'action': 'answer', 'text': prefix}),
                                'tool_calls': [], 'finish_reason': 'stop',
                                'usage': {'prompt_tokens': 17, 'completion_tokens': 5,
                                          'total_tokens': 22}}
                    raise InterruptedError("Synthetic old producer cancelled before its suffix")
                finally:
                    self.shared.order.append("old-producer-closed")
                    self.shared.closed.set()
            response = plan
        Draft202012Validator(body["response_format"]["json_schema"]["schema"]).validate(response)
        return {
            "text": canonical(response),
            "tool_calls": [],
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 17, "completion_tokens": 5, "total_tokens": 22},
        }


@pytest.fixture
def backend(tmp_path, monkeypatch):
    """Run main() and its serialized dispatch, replacing only external suppliers."""
    supplier = Supplier()
    route = ProviderRoute(
        "synthetic-continuous",
        1,
        "local",
        "http://127.0.0.1:9/v1/chat/completions",
        "synthetic-recording-only",
        "none",
        None,
        True,
        ("current", "history", "valid_memory", "derived"),
        ProviderCapabilities(True, False, True, 1_048_576, "max_tokens"),
    )
    model = ProviderModel(route, counter=Counter(), transport=supplier)
    objects = {}
    old_thread = threading.Thread

    def thread(*args, **kwargs):
        if getattr(kwargs.get("target"), "__name__", None) == "watch_parent":
            return SimpleNamespace(start=lambda: None)
        return old_thread(*args, **kwargs)

    for name in (
        "TextSession",
        "DefaultTextRuntime",
        "SQLiteConversationStore",
        "TextMemoryRuntime",
    ):
        original = getattr(entry, name)

        def construct(*args, _name=name, _original=original, **kwargs):
            if _name == "TextMemoryRuntime":
                kwargs["background_idle_grace"] = 60
            result = _original(*args, **kwargs)
            objects[_name] = result
            return result

        monkeypatch.setattr(entry, name, construct)
    monkeypatch.setattr(entry, "contain_current_process", lambda: None)
    monkeypatch.setattr(entry.threading, "Thread", thread)
    monkeypatch.setattr(entry, "FakeModel", lambda: model)
    monkeypatch.setattr(
        entry.sys, "stdin", io.StringIO("synthetic-no-credential-" + "0" * 32 + "\n")
    )
    monkeypatch.setattr(entry.sys, "argv", ["virtual-companion-text", "--fake"])
    monkeypatch.setenv("VC_DATA_ROOT", str(tmp_path / "owned-synthetic-data"))
    monkeypatch.setenv("VC_APP_COMMIT", "0" * 40)
    monkeypatch.setenv("VC_TEXT_DIAGNOSTIC_PROMPT_REVISION", "short-dialogue.v25")
    monkeypatch.delenv("VC_MEMORY_MANAGEMENT_DB", raising=False)
    monkeypatch.delenv("VC_MEMORY_COMPANION_ID", raising=False)

    class Server:
        def __init__(self, token, dispatch):
            assert token.startswith("synthetic-no-credential-")
            self.dispatch = dispatch

        def serve_forever(self, **kwargs):
            pass

        def call(self, action, payload=None):
            return self.dispatch(action, validate(action, {} if payload is None else payload))

    def create_server(token, dispatch):
        objects["server"] = Server(token, dispatch)
        return objects["server"]

    monkeypatch.setattr(entry, "create_server", create_server)
    entry.main()
    mirror = objects["SQLiteConversationStore"]
    original_begin = mirror.begin

    def begin(*args, **kwargs):
        supplier.shared.order.append("begin")
        return original_begin(*args, **kwargs)

    monkeypatch.setattr(mirror, "begin", begin)
    result = SimpleNamespace(
        api=objects["server"],
        supplier=supplier.shared,
        text=objects["TextSession"],
        runtime=objects["DefaultTextRuntime"],
        memory=objects["TextMemoryRuntime"],
        mirror=mirror,
    )
    try:
        yield result
    finally:
        supplier.cancel()
        result.runtime.stop()
        result.text.__class__.stop(result.text)
        result.memory.close()
        result.mirror.close()


def send(backend, text):
    return backend.api.call("send", {"text": text})


def complete(backend, *, listen=False):
    if listen:
        wait_for(lambda: not backend.text.snapshot()["busy"])
        assert backend.api.call("streamPoll")["event"] is None
        assert backend.text.snapshot()["attempt"]["status"] == "listened"
        assert backend.text.snapshot()["partial"] == "" and not backend.text.snapshot()["error"]
        return None
    events = []
    while True:
        event = wait_for(
            lambda: backend.api.call("streamPoll")["event"] or not backend.text.snapshot()["busy"]
        )
        if event is True:
            break
        assert backend.text.snapshot()["partial"] == "", "Unacknowledged text is not legacy output"
        backend.api.call(
            "streamAck",
            {
                "sequence": event["sequence"],
                "fingerprint": event["fingerprint"],
                "host_receipt": "synthetic-visible-" + event["fingerprint"],
            },
        )
        events.append(event)
    assert backend.text.snapshot()["attempt"]["status"] == "completed"
    assert events
    return events[-1]


def record(label, backend, extra=None):
    sink = os.environ.get("VC_CONTINUOUS_RECORD_DIR")
    if sink:
        target = Path(sink)
        target.mkdir(parents=True, exist_ok=True)
        data = {
            "schema": "cpu-default-continuous-input.v1",
            "label": label,
            "records": backend.supplier.records,
            "order": backend.supplier.order,
            "history": backend.mirror.history()["messages"],
            "extra": extra,
            "api_calls": 0,
            "semantic_effect_proven": False,
        }
        (target / f"{label}.json").write_text(
            canonical(data) + "\n", encoding="utf-8", newline="\n"
        )


def test_received_only_reply_reaches_queued_generation_and_review_without_visible_history(backend):
    first_text = '我更愿意先听听他的解释，再决定怎么回应。'
    backend.supplier.plans.extend([
        {'blocked_prefix': first_text, 'complete_on_release': True},
        {'action': 'answer', 'text': '解释里最需要弄清楚的是他改动的原因。'},
    ])
    send(backend, '他没问我就改了方案，你怎么看？')
    assert backend.supplier.entered.wait(2)
    event = wait_for(lambda: backend.api.call('streamPoll')['event'])
    first_claim = backend.runtime.claim
    send(backend, '接着刚才的看法说。')
    backend.api.call('streamReceive', {
        'sequence': event['sequence'], 'fingerprint': event['fingerprint'],
        'host_receipt': 'synthetic-durable-not-visible',
    })
    backend.supplier.release.set()
    wait_for(lambda: backend.api.call('state') and len([
        r for r in backend.supplier.records if r['stage'] == 'demand_control']) == 2)
    try:
        complete(backend)
    except AssertionError:
        record('received-only-failure', backend, backend.text.snapshot())
        raise
    generation = [r for r in backend.supplier.records if r['stage'] == 'demand_control'][1]
    assert any(first_text in m['content'] for m in generation['body']['messages'])
    reviews = [r for r in backend.supplier.records if r['stage'] == 'stream_unit_review']
    view = json.loads(reviews[-1]['body']['messages'][-1]['content'])
    assert view['context']['received_output'][0]['text'] == first_text
    assert not [s for s in view['personal_sources'] if s['text'] == first_text]
    state = backend.memory.store.stream_state(first_claim.companion_id, first_claim.input_id)
    assert state['received_units'] == 1 and state['confirmed_units'] == 0
    assert not [s for s in backend.memory.store.background_sources(first_claim.companion_id)
                if s['role'] == 'assistant' and s['source'] == 'segment-' + event['fingerprint']]
    record('received-only-continuation', backend, {'first_stream_state': state})


@pytest.mark.parametrize("continuing", [True, False])
def test_actual_busy_send_persists_input_without_cancelling_current_expression(
    backend, continuing
):
    old_text = "他把方案改了，也没问我。你帮我看看怎么跟他说。"
    prefix = "你在意他改之前没问，这一点可以直接说。"
    new_text = "嗯，继续。" if continuing else "补充一下，他后来解释过原因，我也想听听你的看法。"
    reply = (
        "可以从你想确认的改动说起，再问他这样改的考虑。"
        if continuing
        else "那可以把他的解释也放在一起看，再决定怎么沟通。"
    )
    backend.supplier.plans.extend(
        [
            {"blocked_prefix": prefix, "complete_on_release": True},
            {"action": "answer" if continuing else "ack", "text": reply},
        ]
    )
    old = send(backend, old_text)
    assert backend.supplier.entered.wait(2)
    prefix_event = wait_for(lambda: backend.api.call("streamPoll")["event"])
    backend.api.call(
        "streamAck",
        {
            "sequence": prefix_event["sequence"],
            "fingerprint": prefix_event["fingerprint"],
            "host_receipt": "synthetic-prefix-visible",
        },
    )
    assert backend.text.snapshot()["busy"]
    old_source = prefix_event["basis"]["input_id"]
    new = send(backend, new_text)
    assert new["turn_id"] != old["turn_id"]
    assert new['status'] == 'queued'
    assert not backend.supplier.closed.is_set()
    assert not backend.text.cancelled.is_set()
    assert backend.api.call('state')['attempt']['attempt_id'] == old['attempt_id']
    assert backend.runtime.claim.input_id == old_source
    backend.supplier.release.set()
    wait_for(lambda: not backend.text.snapshot()['busy'])
    assert backend.memory.store.stream_state(backend.memory.companion, old_source)[
        'outcome'] == 'completed'
    backend.api.call('state')
    event = complete(backend)
    assert event["basis"]["input_id"] == backend.runtime.claim.input_id
    assert event["presentation"]["candidate"].get("response_type", "answer") == (
        "answer" if continuing else "ack"
    )
    history = backend.mirror.history()["messages"]
    assert [m["content"] for m in history if m["role"] == "user"] == [old_text, new_text]
    assert [m["content"] for m in history if m["role"] == "assistant"] == [prefix, reply]
    assert backend.memory.store.inspect(backend.memory.companion, old_source)["text"] == old_text
    assert (
        backend.memory.store.inspect(backend.memory.companion, old_source)["status"] == "cancelled"
    )
    assert len(backend.memory.store.segments(backend.memory.companion, old_source)) == 1
    review = backend.supplier.records[-1]
    assert review["stage"] == "stream_unit_review"
    sources = review["sealed_packet"]["personal_sources"]
    assert any(s["source_id"] == old_source and s["text"] == old_text for s in sources)
    assert any(s["text"] == new_text for s in sources)
    generation = next(
        r for r in reversed(backend.supplier.records) if r["stage"] == "demand_control"
    )
    assert old_text in generation["body"]["messages"][1]["content"]
    assert generation["body"]["messages"][-1]["content"] == new_text
    record(
        "continue" if continuing else "correction",
        backend,
        {
            "old_source": old_source,
            "old_prefix_fingerprint": prefix_event["fingerprint"],
            "new_attempt": new,
        },
    )


def test_invalid_new_text_is_validated_before_current_run_is_stopped(backend):
    backend.supplier.plans.append({"blocked_prefix": ""})
    send(backend, "仍有效的原请求。")
    assert backend.supplier.entered.wait(2)
    with pytest.raises(ValueError, match="Invalid payload"):
        send(backend, " ")
    assert backend.text.snapshot()["busy"]
    assert not backend.text.cancelled.is_set()
    assert not backend.supplier.closed.is_set()
    assert backend.supplier.order == ["begin"]


def test_ordinary_input_does_not_depend_on_stopping_the_current_producer(backend, monkeypatch):
    backend.supplier.plans.append({"blocked_prefix": ""})
    send(backend, "仍有效的原请求。")
    assert backend.supplier.entered.wait(2)
    original = backend.text.stop

    def not_confirmed():
        raise RuntimeError("Cancellation not yet confirmed")

    monkeypatch.setattr(backend.text, "stop", not_confirmed)
    queued = send(backend, "新文字应可靠保存。")
    assert queued['status'] == 'queued'
    assert not backend.text.cancelled.is_set()
    assert [m["content"] for m in backend.mirror.history()["messages"] if m["role"] == "user"] == [
        "仍有效的原请求。", "新文字应可靠保存。"
    ]
    assert backend.supplier.order.count("begin") == 1
    monkeypatch.setattr(backend.text, "stop", original)
    record("stop-unconfirmed", backend)


def test_unknown_display_still_blocks_new_begin_until_same_real_receipt_is_reconciled(backend):
    backend.supplier.plans.extend(
        [
            {"blocked_prefix": "这是旧已派发范围。"},
            {"action": "answer", "text": "这是后来合成回应。"},
        ]
    )
    old = send(backend, "原请求。")
    unknown = wait_for(lambda: backend.api.call("streamPoll")["event"])
    queued = send(backend, "后来输入。")
    assert queued['status'] == 'queued'
    assert backend.supplier.order.count("begin") == 1
    assert [m["content"] for m in backend.mirror.history()["messages"]] == [
        "原请求。", "后来输入。"]
    assert backend.api.call("streamState")["reconcile"][0]["fingerprint"] == unknown["fingerprint"]
    backend.api.call(
        "streamAck",
        {
            "sequence": unknown["sequence"],
            "fingerprint": unknown["fingerprint"],
            "host_receipt": "synthetic-original-device-receipt",
        },
    )
    backend.supplier.release.set()
    wait_for(lambda: not backend.text.snapshot()['busy'])
    backend.api.call('state')
    assert backend.text.snapshot()['attempt']['turn_id'] == queued['turn_id'] != old['turn_id']
    complete(backend)
    assert [m['content'] for m in backend.mirror.history()['messages'] if m['role'] == 'user'
            ] == ['原请求。', '后来输入。']
    record("unknown-blocked", backend, {"original_fingerprint": unknown["fingerprint"]})


SEQUENCES = {
    "withdraw-draft": (
        ["帮我想一句请假的话。", "不对，不请假了。", "我只想跟同事说今天晚点到，别替我编原因。"],
        "我今天会晚一点到，跟你说一声。",
    ),
    "gathering-handoff": (
        [
            "上次桌游我负责订场地、收钱、教规则，比上班还累；但我喜欢和他们玩。",
            "周六那场桌游我想直接取消，懒得再折腾。",
            "不过我还是想见他们，就是又问我订哪儿、吃什么，规则也等我教。",
            "说完了。你觉得我该怎么办？",
        ],
        "聚会先留着，这次只负责来玩，场地和规则请他们分一下。没人接，再决定还聚不聚。",
    ),
}


def test_explicit_stop_holds_received_inputs_until_the_user_resumes(backend):
    backend.supplier.plans.extend([
        {'blocked_prefix': ''}, {'action': 'listen'},
        {'action': 'answer', 'text': '这两条补充我都接着看了。'}])
    send(backend, '当前请求。')
    assert backend.supplier.entered.wait(2)
    first = send(backend, '补充一。')
    second = send(backend, '补充二。')
    backend.api.call('stop')
    state = backend.api.call('state')
    assert state['input_queue_paused']
    assert [value['turn_id'] for value in state['queued_inputs']] == [
        first['turn_id'], second['turn_id']]
    assert not state['busy']
    backend.api.call('resumeInputs')
    complete(backend, listen=True)
    assert backend.text.snapshot()['attempt']['turn_id'] == first['turn_id']
    backend.api.call('state')
    complete(backend)
    assert backend.text.snapshot()['attempt']['turn_id'] == second['turn_id']
    assert [value['content'] for value in backend.mirror.history()['messages']
            if value['role'] == 'user'] == ['当前请求。', '补充一。', '补充二。']


def test_received_input_uses_its_frozen_character_after_current_configuration_changes(backend):
    from dataclasses import replace

    from test_character_context import character

    prefix = '先把当前这句话讲完。'
    backend.supplier.plans.extend([
        {'blocked_prefix': prefix, 'complete_on_release': True},
        {'action': 'answer', 'text': '我们接着聊旧书。'}])
    send(backend, '当前请求。')
    event = wait_for(lambda: backend.api.call('streamPoll')['event'])
    backend.api.call('streamAck', {'sequence': event['sequence'],
        'fingerprint': event['fingerprint'], 'host_receipt': 'synthetic-original-visible'})
    first = character()
    backend.runtime.config = replace(backend.runtime.config, character=first.reference,
                                     character_data=first.payload)
    received = send(backend, '补充的旧书话题。')
    changed = character(personality='沉稳')
    backend.runtime.config = replace(backend.runtime.config, character=changed.reference,
                                     character_data=changed.payload)
    backend.supplier.release.set()
    wait_for(lambda: not backend.text.snapshot()['busy'])
    backend.api.call('state')
    complete(backend)
    assert backend.text.snapshot()['attempt']['turn_id'] == received['turn_id']
    saved = json.loads(backend.memory.store.inspect(backend.memory.companion,
                       backend.runtime.claim.input_id)['config'])
    assert saved['companion_config']['character_data'] == first.to_mapping()
    generation = next(value for value in reversed(backend.supplier.records)
                      if value['stage'] == 'demand_control')
    assert any(first.payload in message['content']
               for message in generation['body']['messages'] if message['role'] == 'system')


def test_received_input_waits_when_the_request_control_changes(backend):
    backend.supplier.plans.extend([
        {'blocked_prefix': ''}, {'action': 'listen'}])
    send(backend, '当前请求。')
    assert backend.supplier.entered.wait(2)
    received = send(backend, '补充。')
    backend.supplier.release.set()
    wait_for(lambda: not backend.text.snapshot()['busy'])
    original = backend.runtime.control
    backend.runtime.control += '\nsynthetic changed control'
    state = backend.api.call('state')
    assert not state['busy'] and state['queued_inputs'][0]['turn_id'] == received['turn_id']
    assert '版本已变化' in state['input_queue_waiting']
    backend.runtime.control = original
    backend.api.call('state')
    complete(backend, listen=True)


def test_zero_unit_retry_retains_the_received_sources_original_character(backend):
    from dataclasses import replace

    from test_character_context import character

    backend.supplier.plans.extend([
        {'blocked_prefix': ''}, {'blocked_prefix': ''}, {'action': 'listen'}])
    send(backend, '当前请求。')
    assert backend.supplier.entered.wait(2)
    original = character()
    backend.runtime.config = replace(backend.runtime.config, character=original.reference,
                                     character_data=original.payload)
    received = send(backend, '补充的旧书话题。')
    backend.supplier.release.set()
    wait_for(lambda: not backend.text.snapshot()['busy'])
    backend.api.call('state')
    wait_for(lambda: not backend.text.snapshot()['busy'])
    assert backend.text.snapshot()['attempt']['turn_id'] == received['turn_id']
    assert backend.text.snapshot()['attempt']['status'] == 'failed'
    source = backend.runtime.claim.input_id
    changed = character(personality='沉稳')
    backend.runtime.config = replace(backend.runtime.config, character=changed.reference,
                                     character_data=changed.payload)
    backend.api.call('retry')
    complete(backend, listen=True)
    assert backend.runtime.claim.input_id == source
    effective = json.loads(backend.runtime.claim.config_json)
    assert effective['companion_config']['character_data'] == original.to_mapping()


@pytest.mark.parametrize("label", list(SEQUENCES))
def test_separate_sourced_listen_turns_have_zero_assistant_until_full_material_handoff(
    backend, label
):
    inputs, reply = SEQUENCES[label]
    backend.supplier.plans.extend(
        [*({"action": "listen"} for _ in inputs[:-1]), {"action": "answer", "text": reply}]
    )
    source_ids = []
    for text in inputs[:-1]:
        send(backend, text)
        complete(backend, listen=True)
        source_ids.append(backend.runtime.claim.input_id)
        assert all(m["role"] == "user" for m in backend.mirror.history()["messages"])
        assert not any(r["stage"] == "stream_unit_review" for r in backend.supplier.records)
    send(backend, inputs[-1])
    final_event = complete(backend)
    source_ids.append(final_event["basis"]["input_id"])
    generation = next(
        r for r in reversed(backend.supplier.records) if r["stage"] == "demand_control"
    )
    review = backend.supplier.records[-1]
    context = generation["body"]["messages"][1]["content"]
    assert all(text in context for text in inputs[:-1])
    assert generation["body"]["messages"][-1]["content"] == inputs[-1]
    actual = review["sealed_packet"]["personal_sources"]
    assert {(s["source_id"], s["text"]) for s in actual} == set(
        zip(source_ids, inputs, strict=True)
    )
    assert [
        m["content"] for m in backend.mirror.history()["messages"] if m["role"] == "assistant"
    ] == [reply]
    stages = [r["stage"] for r in backend.supplier.records]
    assert stages[: len(inputs)] == ["demand_control"] * len(inputs)
    assert stages[len(inputs) :] and set(stages[len(inputs) :]) == {"stream_unit_review"}
    record(
        label,
        backend,
        {"separate_source_ids": source_ids, "online_aggregation_window_proven": False},
    )


def test_ack_then_listen_and_handoff_keep_unanswered_material_and_typed_ack(backend):
    inputs = ["帮我选个读书时间。我先讲完情况。", "早上不方便，晚上能留半小时。", "好了，你说。"]
    reply = "那就晚上留半小时读书，早上这次不安排。"
    backend.supplier.plans.extend(
        [
            {"action": "ack", "text": "嗯，我在听。"},
            {"action": "listen"},
            {"action": "answer", "text": reply},
        ]
    )
    send(backend, inputs[0])
    ack = complete(backend)
    first_id = ack["basis"]["input_id"]
    assert backend.mirror.history()["messages"][-1]["response_type"] == "ack"
    send(backend, inputs[1])
    complete(backend, listen=True)
    send(backend, inputs[2])
    complete(backend)
    generation = next(
        r for r in reversed(backend.supplier.records) if r["stage"] == "demand_control"
    )
    review = backend.supplier.records[-1]
    assert inputs[0] in generation["body"]["messages"][1]["content"]
    assert inputs[1] in generation["body"]["messages"][1]["content"]
    assert "待答未解决" in generation["body"]["messages"][1]["content"]
    payload = review["sealed_packet"]
    assert any(
        s["source_id"] == first_id and s["text"] == inputs[0] for s in payload["personal_sources"]
    )
    history = json.loads(payload["context"])["history"]
    assert any(item.get("response_type") == "ack" for item in history)
    assert backend.mirror.history()["messages"][-1]["content"] == reply
    record("ack-handoff", backend, {"unanswered_source": first_id})
