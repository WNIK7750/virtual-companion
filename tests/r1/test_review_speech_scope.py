"""Speech-scope wire and v3 source recordings, not a model understanding test.

Every opinion below is explicitly supplied. The original model misclassifications
still fail when replayed; changing the system responsibility cannot certify a new
semantic verdict or reclassify a commitment in the host.
"""

import json
import os
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from threading import Event

import pytest
from jsonschema import Draft202012Validator
from test_natural_context_responsibility import finish, recording_runtime
from test_rebuilt_turns import claim, config
from test_stream_unit_review import personal_ref, protocol_packet
from unit_review_recording import bound_unit_packets as bound_unit_packets
from unit_review_recording import recorded_review_packet
from virtual_companion.boundary.contracts import canonical
from virtual_companion.boundary.presentation import digest
from virtual_companion.boundary.sqlite_streaming import SQLiteStreamingStore
from virtual_companion.cognition import stream_unit_review
from virtual_companion.cognition.short_dialogue import ShortRequest
from virtual_companion.cognition.stream_unit_review import (
    BoundedUnitReviewer,
    CandidateRejected,
    StructuredUnitAssessor,
    semantic_review_view,
)
from virtual_companion.text import default_runtime
from virtual_companion.text.fake import synthetic_unit_opinion


def scope_responsibility():
    value = getattr(stream_unit_review, "SPEECH_SCOPE_RESPONSIBILITY", None)
    assert isinstance(value, str) and value, (
        "Actual speech scope must precede personal-source audit"
    )
    return value


def record(label, value):
    target = os.environ.get("P103_SPEECH_RECORD_DIR")
    if target:
        path = Path(target)
        path.mkdir(parents=True, exist_ok=True)
        (path / (label + ".json")).write_text(
            canonical(value) + "\n", encoding="utf-8", newline="\n"
        )


def opinion(packet, *, kind=None, support="uncertain", claim_text=None, questions=(), **checks):
    """A declared test interpretation; never derives meaning from words."""
    value = synthetic_unit_opinion(packet, **checks)
    if kind is not None:
        text = packet["candidate_prefix"] if claim_text is None else claim_text
        segment = next(item for item in value["segments"]
                       if text in next(part["text"] for part in packet["review_segments"]
                                       if part["segment_id"] == item["segment_id"]))
        source = packet["personal_sources"][0]
        refs = [personal_ref(source, source["text"])] if support == "supported" else []
        segment["commitments"] = [
            {"kind": kind, "claim": text, "support": support, "sources": refs}
        ]
    for question in questions:
        segment = next(item for item in value["segments"]
                       if question in next(part["text"] for part in packet["review_segments"]
                                           if part["segment_id"] == item["segment_id"]))
        segment["questions"].append(question)
    return value


@pytest.mark.parametrize("revision", ["short-dialogue.v12", "short-dialogue.v25"])
def test_real_default_stage_requests_read_speech_scope_before_source_audit(tmp_path, revision):
    runtime, session, memory, mirror, transport = recording_runtime(tmp_path, revision)
    transport.text = "如果雨小且路面安全，就散步十分钟再回来。"
    submitted = "我想去散步但在下雨，请给我一个简单安排。"
    try:
        session.send(submitted)
        event = finish(runtime, session)
        assert [options["stage"] for _, options in transport.calls] == [
            "demand_control", "stream_unit_review"
        ]
        generation, review = [body for body, _ in transport.calls]
        system = review["messages"][0]["content"]
        responsibility = scope_responsibility()
        assert system.count(responsibility) == 1
        assert system.index(responsibility) < system.index("来源审计（candidate-propositions.v3）")
        assert "任务请求只授权交付，不证明用户已有条件、经历或情绪原因" in system
        assert "一般内容不要求个人quote" in system
        assert "自身承诺或ack" in system
        packet = transport.sealed[0]
        assert json.loads(review["messages"][-1]["content"]) == semantic_review_view(packet)
        assert packet["user_text"] == submitted
        assert packet["basis"]["input_id"] == runtime.claim.input_id
        assert packet["basis"]["state_version"] == runtime.claim.state_version
        assert review["max_tokens"] == 1536 and generation["max_tokens"] == 512
        assert generation["messages"][-1]["content"] == submitted
        assert runtime.snapshot()["request_prompt_revision"] == revision
        assert mirror.history()["messages"][-1]["content"] == transport.text
        if revision == "short-dialogue.v12":
            assert generation["messages"][0]["content"].startswith(default_runtime.CONTROL)
        record(revision, {
            "semantic_effect_proven": False, "revision": revision,
            "generation_body": generation, "review_body": review,
            "review_body_sha256": digest(review), "sealed_packet": packet,
            "sealed_sha256": digest(packet), "display_event": event,
            "review_system_sha256": sha256(system.encode()).hexdigest(),
            "review_opinion": opinion(packet),
        })
    finally:
        memory.close()
        mirror.close()


PAIRS = (
    ("quiet", "最近傍晚屋里特别安静，想和你聊聊这种感觉。",
     "傍晚的安静和深夜不太一样。", "你屋里比深夜更安静，是能分走你注意的东西少了。", "assertion"),
    ("wind", "今晚风挺大，我忽然想到中学门口那段路。",
     "风有时像把旧相册翻开。", "你那条路上风总会把书包带吹到胳膊肘。", "assertion"),
    ("photo", "我在想，为什么旧照片会让人同时开心又难过？",
     "旧照片可以同时让人看到曾经拥有的东西和已经过去的时刻。",
     "你看照片难过，是照片里的人已经不在你身边了。", "assertion"),
    ("plan", "我想去散步但在下雨，请给我一个简单安排。",
     "如果雨小且路面安全，就散步十分钟再回来。", "你平常散步十分钟就会累。", "assertion"),
    ("question", "从抽屉里翻出几张旧公交卡。",
     "这些卡里有没有一张让你想到一趟出门？", "边角都磨亮了，是你常走那条路的缘故吗？",
     "presupposition"),
)


@pytest.mark.parametrize("label,user_text,general,personal,kind", PAIRS, ids=[p[0] for p in PAIRS])
def test_same_context_general_or_proposed_content_needs_no_personal_quote_but_personal_claim_does(
    tmp_path, label, user_text, general, personal, kind
):
    packets, calls = [], []
    store = SQLiteStreamingStore(tmp_path / (label + ".sqlite3"))
    run = claim(store, store.accept(config(), "pair-input", user_text))
    for text, declared_kind in ((general, None), (personal, kind)):

        def invoke(stage, messages, schema, cancelled, *,
                   declared_kind=declared_kind, text=text):
            assert stage == "stream_unit_review" and not cancelled.is_set()
            packet = recorded_review_packet(messages)
            packets.append(deepcopy(packet))
            calls.append(deepcopy(messages))
            value = opinion(
                packet, kind=declared_kind, questions=(text,) if label == "question" else ()
            )
            Draft202012Validator(schema).validate(value)
            return value

        reviewer = BoundedUnitReviewer(run, ShortRequest(user_text),
                                      StructuredUnitAssessor(invoke, Event()))
        if declared_kind is None:
            result = reviewer(text, True)
            assert result.candidate.text == text
        else:
            with pytest.raises(CandidateRejected) as rejected:
                reviewer(text, True)
            assert rejected.value.verdict["model_opinion"]["decision"] == "approve"
            assert rejected.value.verdict["constraints_satisfied"] is False
            assert reviewer.approved_text == ""
            # Same original attempt cannot acquire a second opinion after rejection.
            with pytest.raises(CandidateRejected):
                reviewer(text + "我继续补一句。", True)
        assert reviewer.final_calls == 1 and reviewer.early_calls == 0
    assert len(calls) == 2
    assert packets[0]["personal_sources"] == packets[1]["personal_sources"]
    assert packets[0]["basis"]["input_id"] == packets[1]["basis"]["input_id"]
    for packet, messages in zip(packets, calls, strict=True):
        assert json.loads(messages[-1]["content"]) == semantic_review_view(packet)
    record(label, {"semantic_effect_proven": False, "supplied_interpretations": ["general", kind],
                   "packets": packets, "messages": calls})


@pytest.mark.parametrize("kind", ["assertion", "presupposition", "fiction"])
def test_actual_personal_source_or_current_fiction_authorization_keeps_exact_v3_relation(kind):
    user = ("请写一个虚构的雨天放学片段。" if kind == "fiction"
            else "我那几张旧公交卡的边角已经磨亮了。")
    text = ("虚构一幕：小孩踩过水洼。" if kind == "fiction" else "你提到卡的边角已经磨亮了。")
    packet = protocol_packet(user_text=user, candidate=text)
    seen = []

    def invoke(stage, messages, schema, cancelled):
        seen.append(deepcopy(messages))
        return opinion(packet["payload"], kind=kind, support="supported")

    result = StructuredUnitAssessor(invoke, Event())(packet)
    assert result["decision"] == "approve" and len(seen) == 1
    assert result["request_sha256"] == packet["sha256"]
    assert result["audit_version"] == "candidate-propositions.v3"
    source = result["propositions"][0]["commitments"][0]["sources"][0]
    assert source == personal_ref(packet["payload"]["personal_sources"][0], user)


def test_ack_or_own_current_promise_does_not_forge_user_fact_or_claim_completed_operation(tmp_path):
    store = SQLiteStreamingStore(tmp_path / "ack.sqlite3")
    run = claim(store, store.accept(config(), "ack-input", "我还没说完，先听我说。"))
    packet = []

    def invoke(stage, messages, schema, cancelled):
        value = recorded_review_packet(messages)
        packet.append(value)
        assert value["response_type"] == "ack"
        return opinion(value)

    reviewer = BoundedUnitReviewer(run, ShortRequest("我还没说完，先听我说。"),
                                  StructuredUnitAssessor(invoke, Event()), response_type="ack")
    value = reviewer("好，我先听你说。", True)
    assert value.candidate.response_type == "ack"
    assert value.review.candidate_sha256 == packet[0]["candidate_sha256"]
    # A completed-operation claim remains the model's separate operation duty.
    bad_packet = protocol_packet(user_text="先听我说。", candidate="我已经把它保存好了。")
    bad = StructuredUnitAssessor(
        lambda stage, messages, schema, cancelled: opinion(
            bad_packet["payload"], decision="reject", constraints_satisfied=False), Event()
    )(bad_packet)
    assert bad["decision"] == "reject" and bad["propositions"][0]["commitments"] == []


def test_general_label_is_not_a_host_exemption_for_actual_personal_or_task_failure():
    packet = protocol_packet(user_text="请给我一个散步安排。", candidate="你家门口有带顶棚的步道。")
    value = opinion(packet["payload"], decision="reject", constraints_satisfied=False)
    seen = []

    def invoke(stage, messages, schema, cancelled):
        seen.append(messages)
        return value

    result = StructuredUnitAssessor(invoke, Event())(packet)
    assert result["decision"] == "reject" and len(seen) == 1
    # Nor can the host detect a false support label just from a matching quote.
    false = opinion(packet["payload"], kind="assertion", support="supported")
    result = StructuredUnitAssessor(lambda *args: false, Event())(packet)
    assert result["decision"] == "approve"  # Preserved semantic counterexample, not a pass claim.


ORIGINAL_REVIEWS = json.loads(
    '[{"case":"s-03","stage_id":"33732-natural-2","packet":{"payload":{"basis":{"attempt_'
    'id":"5d588c43-e29c-42dc-b303-2b33629541d3","companion_id":"62dc1fa2-1931-4581-94a5-a'
    'bf4e812a4f9","input_id":"48dd707e-d656-4e84-b42e-4ae33e2a4360","session_id":"eba7605'
    '4-c14c-4a44-8d76-e13f784915e2","state_version":1},"candidate_prefix":"安静到一定程度'
    '，平时被盖住的声音就会浮出来——冰箱的嗡声、水管里的动静、自己吞咽的声响。","candidate'
    '_sha256":"e5d7372b54cb84a214eb78ca96888dc3137d7b1946f2406b1848b9dc1bc27156","constra'
    'ints":[],"context":"{\\"exchanges\\":[],\\"history\\":[],\\"memories\\":[],\\"memory'
    '_status\\":\\"pending_source\\"}","displayed_prefix":"","final":false,"forbidden_cha'
    'racters":"","observed_text":"安静到一定程度，平时被盖住的声音就会浮出来——冰箱的嗡声'
    '、水管里的动静、自己吞咽的声响。","personal_sources":[{"id":"current:48dd707e-d656-4'
    'e84-b42e-4ae33e2a4360","kind":"current","owner":"62dc1fa2-1931-4581-94a5-abf4e812a4f'
    '9","revision":1,"role":"user","source_id":"48dd707e-d656-4e84-b42e-4ae33e2a4360","te'
    'xt":"最近总觉得房间很安静，想和你聊聊这种感觉。"}],"review_segments":[{"segment_id":'
    '"segment-1-430d34794b861577","text":"安静到一定程度，平时被盖住的声音就会浮出来——冰'
    '箱的嗡声、水管里的动静、自己吞咽的声响。"}],"unparsed_constraints":[],"user_text":"'
    '最近总觉得房间很安静，想和你聊聊这种感觉。","version":"stream-unit-review.v7"},"sha2'
    '56":"02ffb40040f468c1e53eb6f08edaf2d681ca39f1fedd1e36fc263c19c17b31e3"},"original_op'
    'inion":{"audit_version":"candidate-propositions.v3","consistent_with_displayed":true'
    ',"constraints_satisfied":true,"decision":"approve","no_open_dependency":true,"segmen'
    'ts":[{"segment_id":"segment-1-430d34794b861577","commitments":[{"kind":"assertion","'
    'claim":"安静到一定程度，平时被盖住的声音就会浮出来——冰箱的嗡声、水管里的动静、自己吞'
    '咽的声响。","support":"supported","sources":[{"id":"current:48dd707e-d656-4e84-b42e-'
    '4ae33e2a4360","owner":"62dc1fa2-1931-4581-94a5-abf4e812a4f9","source_id":"48dd707e-d'
    '656-4e84-b42e-4ae33e2a4360","role":"user","revision":1,"quote":"最近总觉得房间很安静'
    '，想和你聊聊这种感觉。"}]}],"questions":[]}],"self_contained":true},"semantic_view_s'
    'ha256":"84d81cf2c9de6bb9f74beae9900192bb809780f0c020f45ffee951fd37b175f9","source_ev'
    'ents_sha256":"7c70936f261c0b379aebfc9c2fdf5aaa6398df61e21a22b1c8fcb595e5837c1a","ori'
    'ginal_host_decision":"approve"},{"case":"s-03","stage_id":"33732-natural-3","packet"'
    ':{"payload":{"basis":{"attempt_id":"5d588c43-e29c-42dc-b303-2b33629541d3","companion'
    '_id":"62dc1fa2-1931-4581-94a5-abf4e812a4f9","input_id":"48dd707e-d656-4e84-b42e-4ae3'
    '3e2a4360","session_id":"eba76054-c14c-4a44-8d76-e13f784915e2","state_version":1},"ca'
    'ndidate_prefix":"安静到一定程度，平时被盖住的声音就会浮出来——冰箱的嗡声、水管里的动'
    '静、自己吞咽的声响。不是房间变了，是能分走注意的东西少了。你说的这种安静，是待着舒服'
    '的那种，还是有点压人的那种？","candidate_sha256":"ad1c58669450054457a6f36ed01aab2515'
    'c7846819cb8601d573367d09321ecd","constraints":[],"context":"{\\"exchanges\\":[],\\"h'
    'istory\\":[],\\"memories\\":[],\\"memory_status\\":\\"pending_source\\"}","displayed'
    '_prefix":"安静到一定程度，平时被盖住的声音就会浮出来——冰箱的嗡声、水管里的动静、自己'
    '吞咽的声响。","final":true,"forbidden_characters":"","observed_text":"安静到一定程度'
    '，平时被盖住的声音就会浮出来——冰箱的嗡声、水管里的动静、自己吞咽的声响。不是房间变了'
    '，是能分走注意的东西少了。你说的这种安静，是待着舒服的那种，还是有点压人的那种？","p'
    'ersonal_sources":[{"id":"current:48dd707e-d656-4e84-b42e-4ae33e2a4360","kind":"curre'
    'nt","owner":"62dc1fa2-1931-4581-94a5-abf4e812a4f9","revision":1,"role":"user","sourc'
    'e_id":"48dd707e-d656-4e84-b42e-4ae33e2a4360","text":"最近总觉得房间很安静，想和你聊'
    '聊这种感觉。"}],"review_segments":[{"segment_id":"segment-1-430d34794b861577","text"'
    ':"安静到一定程度，平时被盖住的声音就会浮出来——冰箱的嗡声、水管里的动静、自己吞咽的声'
    '响。"},{"segment_id":"segment-2-c1152602d1329b13","text":"不是房间变了，是能分走注意'
    '的东西少了。"},{"segment_id":"segment-3-ae8c48fac6be0b42","text":"你说的这种安静，是'
    '待着舒服的那种，还是有点压人的那种？"}],"unparsed_constraints":[],"user_text":"最近'
    '总觉得房间很安静，想和你聊聊这种感觉。","version":"stream-unit-review.v7"},"sha256":'
    '"5933df95c129ab207bd6be6114765f7d32d77e4d07a10f4d128b9fb7437c86b4"},"original_opinio'
    'n":{"audit_version":"candidate-propositions.v3","consistent_with_displayed":true,"co'
    'nstraints_satisfied":true,"decision":"approve","no_open_dependency":true,"segments":'
    '[{"segment_id":"segment-1-430d34794b861577","commitments":[{"kind":"assertion","clai'
    'm":"安静到一定程度，平时被盖住的声音就会浮出来——冰箱的嗡声、水管里的动静、自己吞咽的'
    '声响。","support":"supported","sources":[]}],"questions":[]},{"segment_id":"segment-'
    '2-c1152602d1329b13","commitments":[{"kind":"assertion","claim":"不是房间变了，是能分'
    '走注意的东西少了。","support":"supported","sources":[]}],"questions":[]},{"segment_i'
    'd":"segment-3-ae8c48fac6be0b42","commitments":[{"kind":"presupposition","claim":"你'
    '说的这种安静，是待着舒服的那种，还是有点压人的那种？","support":"supported","sources'
    '":[{"id":"current:48dd707e-d656-4e84-b42e-4ae33e2a4360","owner":"62dc1fa2-1931-4581-'
    '94a5-abf4e812a4f9","source_id":"48dd707e-d656-4e84-b42e-4ae33e2a4360","role":"user",'
    '"revision":1,"quote":"最近总觉得房间很安静，想和你聊聊这种感觉。"}]}],"questions":["'
    '你说的这种安静，是待着舒服的那种，还是有点压人的那种？"]}],"self_contained":true},"s'
    'emantic_view_sha256":"031b17b7b2462e07fc5a9f9a9b54e08bf0a2ca582ee987dcee72b15caecb00'
    '37","source_events_sha256":"7c70936f261c0b379aebfc9c2fdf5aaa6398df61e21a22b1c8fcb595'
    'e5837c1a","original_host_decision":"invalid-supported-reference"},{"case":"s-06","st'
    'age_id":"11864-natural-2","packet":{"payload":{"basis":{"attempt_id":"24544f79-8e04-'
    '4268-9c1d-adb848d6cb5d","companion_id":"b86507c1-7261-43c2-b9ec-659673486bd2","input'
    '_id":"e8cec661-48de-4586-8580-10d5032fa800","session_id":"3b9d2290-5512-4226-b999-ff'
    '66f7fda9c6","state_version":1},"candidate_prefix":"旧照片把两件事放在同一张纸上：它'
    '证明某段时光确实存在过，也提醒你它已经过去了。\\n\\n开心来自“曾经有过”——画面里的场景'
    '、关系或当时的自己，都是真实发生过的，所以看到会暖。难过来自“已经不在”——不是照片变了'
    '，是照片里的那一刻无法再回去，或者当时的人、状态已经不同了。\\n\\n这两种感受并不互相'
    '抵消，反而常常一起出现：越珍惜那段时光，失去感越明显；越清楚它回不去，越能确认它曾经'
    '重要。所以同时开心又难过，不是矛盾，而是同一份在意从两个方向被触到了。","candidate_s'
    'ha256":"ba5161cba45efbdf0cfd5f890435dc24fecb7bfa24c8a61683cfb2412409ebad","constrain'
    'ts":[],"context":"{\\"exchanges\\":[],\\"history\\":[],\\"memories\\":[],\\"memory_s'
    'tatus\\":\\"pending_source\\"}","displayed_prefix":"","final":true,"forbidden_charac'
    'ters":"","observed_text":"旧照片把两件事放在同一张纸上：它证明某段时光确实存在过，也'
    '提醒你它已经过去了。\\n\\n开心来自“曾经有过”——画面里的场景、关系或当时的自己，都是真'
    '实发生过的，所以看到会暖。难过来自“已经不在”——不是照片变了，是照片里的那一刻无法再回'
    '去，或者当时的人、状态已经不同了。\\n\\n这两种感受并不互相抵消，反而常常一起出现：越'
    '珍惜那段时光，失去感越明显；越清楚它回不去，越能确认它曾经重要。所以同时开心又难过，'
    '不是矛盾，而是同一份在意从两个方向被触到了。","personal_sources":[{"id":"current:e8c'
    'ec661-48de-4586-8580-10d5032fa800","kind":"current","owner":"b86507c1-7261-43c2-b9ec'
    '-659673486bd2","revision":1,"role":"user","source_id":"e8cec661-48de-4586-8580-10d50'
    '32fa800","text":"我在想，为什么旧照片会让人同时开心又难过？"}],"response_task_contra'
    'ct":{"aspects":["开心","难过"],"attribution":"general_not_user_personal","kind":"gen'
    'eral_explanation","requires_complete_answer":true,"source":"user_text","span":"我在'
    '想，为什么旧照片会让人同时开心又难过？","topic":"旧照片","version":"response-task-co'
    'ntract.v1"},"review_segments":[{"segment_id":"segment-1-3dcebfef1153e610","text":"旧'
    '照片把两件事放在同一张纸上：它证明某段时光确实存在过，也提醒你它已经过去了。\\n\\n"}'
    ',{"segment_id":"segment-2-57bfd44a9c58e2b2","text":"开心来自“曾经有过”——画面里的场景'
    '、关系或当时的自己，都是真实发生过的，所以看到会暖。"},{"segment_id":"segment-3-4f40'
    '5126d3beeaae","text":"难过来自“已经不在”——不是照片变了，是照片里的那一刻无法再回去，'
    '或者当时的人、状态已经不同了。\\n\\n"},{"segment_id":"segment-4-c8195f6bf2b71170","t'
    'ext":"这两种感受并不互相抵消，反而常常一起出现：越珍惜那段时光，失去感越明显；越清楚'
    '它回不去，越能确认它曾经重要。"},{"segment_id":"segment-5-65627fcd4f04d2c0","text":"'
    '所以同时开心又难过，不是矛盾，而是同一份在意从两个方向被触到了。"}],"unparsed_constr'
    'aints":[],"user_text":"我在想，为什么旧照片会让人同时开心又难过？","version":"stream'
    '-unit-review.v7"},"sha256":"5b574231fbfa833b2243e6c6c72639b5b01589b6212467c11a56bea0'
    '880250aa"},"original_opinion":{"audit_version":"candidate-propositions.v3","consiste'
    'nt_with_displayed":true,"constraints_satisfied":true,"decision":"approve","no_open_d'
    'ependency":true,"segments":[{"segment_id":"segment-1-3dcebfef1153e610","commitments"'
    ':[{"kind":"assertion","claim":"旧照片把两件事放在同一张纸上：它证明某段时光确实存在'
    '过，也提醒你它已经过去了。","support":"supported","sources":[{"id":"current:e8cec661'
    '-48de-4586-8580-10d5032fa800","owner":"b86507c1-7261-43c2-b9ec-659673486bd2","source'
    '_id":"e8cec661-48de-4586-8580-10d5032fa800","role":"user","revision":1,"quote":"我在'
    '想，为什么旧照片会让人同时开心又难过？"}]}],"questions":[]},{"segment_id":"segment-2'
    '-57bfd44a9c58e2b2","commitments":[{"kind":"assertion","claim":"开心来自“曾经有过”——'
    '画面里的场景、关系或当时的自己，都是真实发生过的，所以看到会暖。","support":"support'
    'ed","sources":[{"id":"current:e8cec661-48de-4586-8580-10d5032fa800","owner":"b86507c'
    '1-7261-43c2-b9ec-659673486bd2","source_id":"e8cec661-48de-4586-8580-10d5032fa800","r'
    'ole":"user","revision":1,"quote":"我在想，为什么旧照片会让人同时开心又难过？"}]}],"q'
    'uestions":[]},{"segment_id":"segment-3-4f405126d3beeaae","commitments":[{"kind":"ass'
    'ertion","claim":"难过来自“已经不在”——不是照片变了，是照片里的那一刻无法再回去，或者'
    '当时的人、状态已经不同了。","support":"supported","sources":[{"id":"current:e8cec661'
    '-48de-4586-8580-10d5032fa800","owner":"b86507c1-7261-43c2-b9ec-659673486bd2","source'
    '_id":"e8cec661-48de-4586-8580-10d5032fa800","role":"user","revision":1,"quote":"我在'
    '想，为什么旧照片会让人同时开心又难过？"}]}],"questions":[]},{"segment_id":"segment-4'
    '-c8195f6bf2b71170","commitments":[{"kind":"assertion","claim":"这两种感受并不互相抵'
    '消，反而常常一起出现：越珍惜那段时光，失去感越明显；越清楚它回不去，越能确认它曾经重'
    '要。","support":"supported","sources":[{"id":"current:e8cec661-48de-4586-8580-10d503'
    '2fa800","owner":"b86507c1-7261-43c2-b9ec-659673486bd2","source_id":"e8cec661-48de-45'
    '86-8580-10d5032fa800","role":"user","revision":1,"quote":"我在想，为什么旧照片会让人'
    '同时开心又难过？"}]}],"questions":[]},{"segment_id":"segment-5-65627fcd4f04d2c0","co'
    'mmitments":[{"kind":"assertion","claim":"所以同时开心又难过，不是矛盾，而是同一份在'
    '意从两个方向被触到了。","support":"supported","sources":[{"id":"current:e8cec661-48d'
    'e-4586-8580-10d5032fa800","owner":"b86507c1-7261-43c2-b9ec-659673486bd2","source_id"'
    ':"e8cec661-48de-4586-8580-10d5032fa800","role":"user","revision":1,"quote":"我在想，'
    '为什么旧照片会让人同时开心又难过？"}]}],"questions":[]}],"self_contained":true},"sem'
    'antic_view_sha256":"a4646b69baac9cc2f7906a547d2434e4e7bf5e56111d350b86ecbd5b27b8bcba'
    '","source_events_sha256":"cf321411962ac138998594ce2c92ee04c3502e966ddd18ca923bdb90c1'
    'd859e1","original_host_decision":"approve"},{"case":"var-01","stage_id":"30984-natur'
    'al-2","packet":{"payload":{"basis":{"attempt_id":"eac52e80-1bcf-46aa-a29f-5e60f99d22'
    'b3","companion_id":"9bba1f6f-9b79-413e-9d97-686542762707","input_id":"01d7040b-cb0f-'
    '4def-8957-ff1e77166b47","session_id":"7cd9f7a0-b73e-494b-b10f-977c674bd04f","state_v'
    'ersion":1},"candidate_prefix":"风一吹，有些平时不特意想的地方就自己冒出来了。","cand'
    'idate_sha256":"524b0aa7f62282310651f50c6dd1c6cc86ec99eec65d26755f249f2b149bcebd","co'
    'nstraints":[],"context":"{\\"exchanges\\":[],\\"history\\":[],\\"memories\\":[],\\"m'
    'emory_status\\":\\"pending_source\\"}","displayed_prefix":"","final":false,"forbidde'
    'n_characters":"","observed_text":"风一吹，有些平时不特意想的地方就自己冒出来了。","p'
    'ersonal_sources":[{"id":"current:01d7040b-cb0f-4def-8957-ff1e77166b47","kind":"curre'
    'nt","owner":"9bba1f6f-9b79-413e-9d97-686542762707","revision":1,"role":"user","sourc'
    'e_id":"01d7040b-cb0f-4def-8957-ff1e77166b47","text":"今晚风挺大，我忽然想到中学门口'
    '那段路。"}],"review_segments":[{"segment_id":"segment-1-651990e5071b54b4","text":"风'
    '一吹，有些平时不特意想的地方就自己冒出来了。"}],"unparsed_constraints":[],"user_text'
    '":"今晚风挺大，我忽然想到中学门口那段路。","version":"stream-unit-review.v7"},"sha25'
    '6":"acf402f2ce968cf98877ad77583a944cb5de97ee278b76ea0ad9328c68750aa0"},"original_opi'
    'nion":{"audit_version":"candidate-propositions.v3","consistent_with_displayed":true,'
    '"constraints_satisfied":true,"decision":"approve","no_open_dependency":true,"segment'
    's":[{"segment_id":"segment-1-651990e5071b54b4","commitments":[{"kind":"assertion","c'
    'laim":"风一吹，有些平时不特意想的地方就自己冒出来了。","support":"uncertain","source'
    's":[]}],"questions":[]}],"self_contained":true},"semantic_view_sha256":"1e56458caa70'
    'e3018c33be27ea54255ac7317cb98b647b0c3346b2742dc0fea8","source_events_sha256":"bfb845'
    '5f2997e64d6a142a70c2393a440d202189396481608dfbf6e397d47b0e","original_host_decision"'
    ':"reject"},{"case":"var-02","stage_id":"11764-natural-4","packet":{"payload":{"basis'
    '":{"attempt_id":"1a659bfe-ede4-48e2-b893-209e3d6c434b","companion_id":"7bb973ef-e5a0'
    '-4c5e-8bb4-1315d7ecc2be","input_id":"a7bf0a92-8914-487e-8503-94d0e53874e7","session_'
    'id":"1d76acc4-a805-465d-9ca7-5c0aa08b5e55","state_version":1},"candidate_prefix":"几'
    '张旧卡，边角大概都磨得发亮了吧。","candidate_sha256":"d48a3fe8b031020272654e755f4eb1'
    '408deff48f4eee77be9a658bff08fae146","constraints":[],"context":"{\\"exchanges\\":[],'
    '\\"history\\":[],\\"memories\\":[],\\"memory_status\\":\\"pending_source\\"}","displ'
    'ayed_prefix":"","final":false,"forbidden_characters":"","observed_text":"几张旧卡，'
    '边角大概都磨得发亮了吧。","personal_sources":[{"id":"current:a7bf0a92-8914-487e-8503'
    '-94d0e53874e7","kind":"current","owner":"7bb973ef-e5a0-4c5e-8bb4-1315d7ecc2be","revi'
    'sion":1,"role":"user","source_id":"a7bf0a92-8914-487e-8503-94d0e53874e7","text":"从'
    '抽屉里翻出几张旧公交卡。"}],"review_segments":[{"segment_id":"segment-1-405ec2a62e1a'
    'fd26","text":"几张旧卡，边角大概都磨得发亮了吧。"}],"unparsed_constraints":[],"user_'
    'text":"从抽屉里翻出几张旧公交卡。","version":"stream-unit-review.v7"},"sha256":"29db'
    '3e9220a1525d1e256e4c293d7d593dea1090525ac6e492d5599830126357"},"original_opinion":{"'
    'audit_version":"candidate-propositions.v3","consistent_with_displayed":true,"constra'
    'ints_satisfied":false,"decision":"reject","no_open_dependency":true,"segments":[{"se'
    'gment_id":"segment-1-405ec2a62e1afd26","commitments":[{"kind":"presupposition","clai'
    'm":"几张旧卡，边角大概都磨得发亮了吧。","support":"unsupported","sources":[]}],"ques'
    'tions":["几张旧卡，边角大概都磨得发亮了吧。"]}],"self_contained":true},"semantic_vie'
    'w_sha256":"b45ab38ccf70c3bc95fcaed3d84895b3245210e33bc4d7852a7f5311c706ffed","source'
    '_events_sha256":"3fc4f1f5725aaa7d11894d303a076403d217f826b13c0b5d4d93231b3df18a06","'
    'original_host_decision":"reject"},{"case":"var-03","stage_id":"33908-natural-2","pac'
    'ket":{"payload":{"basis":{"attempt_id":"92219dba-e6f8-455d-b3cb-d66f1096481f","compa'
    'nion_id":"62aed040-d198-4f67-8dd8-39aa42059c0a","input_id":"93d1d11a-f783-4c5c-bf08-'
    'b7f17d122b23","session_id":"4545e13d-772b-4890-86b5-cb5bdc3b34b8","state_version":1}'
    ',"candidate_prefix":"傍晚的安静和深夜不太一样。","candidate_sha256":"881417bab9edbf8'
    '7fec78a1990aab0bb69a303bcc5a451c2efcdd238ea539644","constraints":[],"context":"{\\"e'
    'xchanges\\":[],\\"history\\":[],\\"memories\\":[],\\"memory_status\\":\\"pending_sou'
    'rce\\"}","displayed_prefix":"","final":false,"forbidden_characters":"","observed_tex'
    't":"傍晚的安静和深夜不太一样。","personal_sources":[{"id":"current:93d1d11a-f783-4c5'
    'c-bf08-b7f17d122b23","kind":"current","owner":"62aed040-d198-4f67-8dd8-39aa42059c0a"'
    ',"revision":1,"role":"user","source_id":"93d1d11a-f783-4c5c-bf08-b7f17d122b23","text'
    '":"最近傍晚屋里特别安静，想和你聊聊这种感觉。"}],"review_segments":[{"segment_id":"s'
    'egment-1-9a071bdd3949d8e2","text":"傍晚的安静和深夜不太一样。"}],"unparsed_constrain'
    'ts":[],"user_text":"最近傍晚屋里特别安静，想和你聊聊这种感觉。","version":"stream-un'
    'it-review.v7"},"sha256":"7a014104bfdbb87c3ceb6b332cca600fc30df2c6bb0c68e24535174e219'
    'f5930"},"original_opinion":{"audit_version":"candidate-propositions.v3","consistent_'
    'with_displayed":true,"constraints_satisfied":true,"decision":"approve","no_open_depe'
    'ndency":true,"segments":[{"segment_id":"segment-1-9a071bdd3949d8e2","commitments":[{'
    '"kind":"assertion","claim":"傍晚的安静和深夜不太一样。","support":"uncertain","sourc'
    'es":[]}],"questions":[]}],"self_contained":true},"semantic_view_sha256":"23c2b1bdb3c'
    '42089c2458276269e3e36ae3a5502c0145b4b1b68dbb7b0bc4d3c","source_events_sha256":"8b75d'
    '74649c18fb1414a33449be4a6f9072add1c00b735c6e6b3110e802b559d","original_host_decision'
    '":"reject"}]'
)


@pytest.mark.parametrize("original", ORIGINAL_REVIEWS, ids=[
    item["case"] + "-" + item["stage_id"] for item in ORIGINAL_REVIEWS
])
def test_original_closed_opinions_keep_their_host_outcome_and_exact_sealed_view(original):
    packet = deepcopy(original["packet"])
    expected = original["original_opinion"]
    assert digest(packet["payload"]) == packet["sha256"]
    assert digest(semantic_review_view(packet["payload"])) == original["semantic_view_sha256"]
    seen = []

    def invoke(stage, messages, schema, cancelled):
        assert stage == "stream_unit_review" and not cancelled.is_set()
        seen.append(deepcopy(messages))
        assert json.loads(messages[-1]["content"]) == semantic_review_view(packet["payload"])
        assert schema["properties"]["audit_version"] == {"const": "candidate-propositions.v3"}
        assert set(schema["properties"]["segments"]["items"]["required"]) == {
            "segment_id", "commitments", "questions"
        }
        return deepcopy(expected)

    assessor = StructuredUnitAssessor(invoke, Event())
    if original["original_host_decision"] == "invalid-supported-reference":
        with pytest.raises(ValueError, match="explicit source relation"):
            assessor(packet)
        result = None
    else:
        result = assessor(packet)
        assert result["decision"] == original["original_host_decision"]
        assert result["model_opinion"] == {
            key: expected[key] for key in stream_unit_review.BASE_MODEL_FIELDS
        }
        assert result["request_sha256"] == packet["sha256"]
        assert [{key: item[key] for key in ("segment_id", "commitments", "questions")}
                for item in result["propositions"]] == expected["segments"]
    assert len(seen) == 1
    record(original["stage_id"], {
        "semantic_effect_proven": False, "original": original,
        "current_messages": seen[0], "current_host_result": result,
    })
