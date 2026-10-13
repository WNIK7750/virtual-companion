import threading

import pytest
from virtual_companion.text.demand_memory import run_demand_turn


def test_answer_carrier_decodes_escapes_without_exposing_control_or_partial_surrogate():
    closed, observed = threading.Event(), []

    class Streaming(Request):
        def stream(self, cancelled):
            try:
                yield '{"action":"answer","text":"原文\\uD83C'
                yield '\\uDF3F\\n'
                assert observed == ['原文', '🌿\n']
                assert not closed.is_set()
                yield '后句。"}'
            finally:
                closed.set()

    def consume(prepared, answer):
        observed.extend(answer)
        assert answer.action['text'] == '原文🌿\n后句。'

    result = run_demand_turn(claim=object(), prepare=lambda c, final: Streaming('', []),
        resolve_current=lambda c: pytest.fail('Unexpected memory delegation'),
        refresh=lambda c: c, cancelled=threading.Event(), on_answer=consume)
    assert result.text == ''.join(observed) == '原文🌿\n后句。'
    assert closed.is_set()


class Request:
    def __init__(self, body, events, *, on_stream=None):
        self.body, self.events, self.on_stream = body, events, on_stream

    def check_context(self):
        self.events.append("checked")

    def stream(self, cancelled):
        self.events.append("opened")
        try:
            if self.on_stream:
                self.on_stream()
            yield self.body
        finally:
            self.events.append("closed")


# Complete recorded r9 demand_control carriers; both text fields are identical.
_RECORDED_IDENTICAL_TEXT = (
    pytest.param(
        (
            '{"action":"answer","text":"能，这句话确实有歧义，关键不在“简单”这个词本身'
            '，而在同事说它时的语气、场合和前后文。\\n\\n可以拆成两层看：\\n\\n1. **偏'
            '夸的可能**\\n   “简单”在这里可能指你把复杂的东西讲得清楚、好懂、不绕。比如'
            '你原本要讲一个很难的内容，但同事听完觉得“居然这么容易就听明白了”，那“真简'
            '单”更像是在夸你表达得好、抓重点准。\\n\\n2. **偏嫌浅的可能**\\n   如果你们'
            '本来期待的是深入分析、细节或难点，而同事觉得你只讲了表面，那“真简单”就可能'
            '是在说“内容不够深”“没讲到关键”。这时重点不是表达清楚，而是信息量或深度不够'
            '。\\n\\n3. **也可能是中性甚至随口一说**\\n   有些人说“简单”只是描述感受，'
            '不一定带评价。比如他本来以为会很难，结果发现不难，就顺口说了这么一句。\\n'
            '\\n要判断到底是哪种，可以看几个信号：\\n\\n- **语气**：轻松、带笑，偏夸；'
            '平淡、拖长音、带“就这？”的感觉，偏嫌浅。\\n- **场合**：公开场合顺口夸一句'
            '，和私下认真反馈，分量不一样。\\n- **前后文**：如果前面有“你讲得清楚”“我一'
            '下就懂了”，偏夸；如果后面接“不过是不是还可以再展开”“感觉没讲到难点”，偏嫌'
            '浅。\\n- **对方期待**：他本来想听概述，还是想听深入分析？期待不同，“简单”'
            '的含义就不同。\\n\\n如果你拿不准，最直接的办法不是猜，而是轻轻确认一句，比'
            '如：\\n\\n> “你刚说简单，是指我讲得比较好懂，还是觉得内容还可以再深一点？”'
            '\\n\\n这样既不会把夸当成批评，也不会把批评当成夸。\\n\\n如果你愿意，可以把'
            '当时的原话、语气和前后聊的内容再给我一点，我可以帮你更具体地判断。","text"'
            ':"能，这句话确实有歧义，关键不在“简单”这个词本身，而在同事说它时的语气、场'
            '合和前后文。\\n\\n可以拆成两层看：\\n\\n1. **偏夸的可能**\\n   “简单”在这'
            '里可能指你把复杂的东西讲得清楚、好懂、不绕。比如你原本要讲一个很难的内容，'
            '但同事听完觉得“居然这么容易就听明白了”，那“真简单”更像是在夸你表达得好、抓'
            '重点准。\\n\\n2. **偏嫌浅的可能**\\n   如果你们本来期待的是深入分析、细节'
            '或难点，而同事觉得你只讲了表面，那“真简单”就可能是在说“内容不够深”“没讲到'
            '关键”。这时重点不是表达清楚，而是信息量或深度不够。\\n\\n3. **也可能是中性'
            '甚至随口一说**\\n   有些人说“简单”只是描述感受，不一定带评价。比如他本来以'
            '为会很难，结果发现不难，就顺口说了这么一句。\\n\\n要判断到底是哪种，可以看'
            '几个信号：\\n\\n- **语气**：轻松、带笑，偏夸；平淡、拖长音、带“就这？”的感'
            '觉，偏嫌浅。\\n- **场合**：公开场合顺口夸一句，和私下认真反馈，分量不一样'
            '。\\n- **前后文**：如果前面有“你讲得清楚”“我一下就懂了”，偏夸；如果后面接“'
            '不过是不是还可以再展开”“感觉没讲到难点”，偏嫌浅。\\n- **对方期待**：他本来'
            '想听概述，还是想听深入分析？期待不同，“简单”的含义就不同。\\n\\n如果你拿不'
            '准，最直接的办法不是猜，而是轻轻确认一句，比如：\\n\\n> “你刚说简单，是指'
            '我讲得比较好懂，还是觉得内容还可以再深一点？”\\n\\n这样既不会把夸当成批评'
            '，也不会把批评当成夸。\\n\\n如果你愿意，可以把当时的原话、语气和前后聊的内'
            '容再给我一点，我可以帮你更具体地判断。"}'
        ),
        id='S-01a',
    ),
    pytest.param(
        (
            '{"action":"answer","text":"可以这样分开讲：先明确“我真的没有偏好”，再补一'
            '句“不是敷衍，你选哪个我都愿意配合”。\\n\\n比如：\\n\\n- “我是真的都行，不'
            '是客气。你定就好，我跟着来。”\\n- “我没偏好，所以你说哪个我都接受；如果一'
            '定要我选，我也可以选，只是没有特别倾向。”\\n- “我怕你觉得我在敷衍，所以说'
            '明一下：我是真没意见，不是不想参与。”\\n\\n关键是把两层拆开：\\n\\n1. **偏'
            '好层**：我没有更想要的那个。\\n2. **态度层**：我在认真回应，也愿意配合，不'
            '是把决定推给你。\\n\\n如果对方容易多想，还可以加一句具体的参与感，比如“你'
            '选完告诉我，我来配合安排”，这样“都行”就不像结束对话，而像继续参与。","text'
            '":"可以这样分开讲：先明确“我真的没有偏好”，再补一句“不是敷衍，你选哪个我都'
            '愿意配合”。\\n\\n比如：\\n\\n- “我是真的都行，不是客气。你定就好，我跟着来'
            '。”\\n- “我没偏好，所以你说哪个我都接受；如果一定要我选，我也可以选，只是'
            '没有特别倾向。”\\n- “我怕你觉得我在敷衍，所以说明一下：我是真没意见，不是'
            '不想参与。”\\n\\n关键是把两层拆开：\\n\\n1. **偏好层**：我没有更想要的那个'
            '。\\n2. **态度层**：我在认真回应，也愿意配合，不是把决定推给你。\\n\\n如果'
            '对方容易多想，还可以加一句具体的参与感，比如“你选完告诉我，我来配合安排”，'
            '这样“都行”就不像结束对话，而像继续参与。"}'
        ),
        id='S-05a',
    ),
    pytest.param(
        (
            '{"action":"answer","text":"介绍时可以按“合作完成”来组织，把每个人的贡献分'
            '别说清楚，而不是只归到一个人名下。\\n\\n比如可以这样表述：\\n\\n“这张海报'
            '由我和同桌合作完成。画面部分我们各画了一半，最后的标题由同桌拟定。”\\n\\n'
            '如果还想更具体，可以再补一句各自负责的内容，例如：\\n\\n“我负责了其中一半'
            '画面，同桌负责另一半画面，并完成了标题的构思和定稿。”\\n\\n这样既保留了“共'
            '同完成”的整体，也没有把标题的功劳含糊掉。关键是把动作和对象对应清楚：谁画'
            '了哪部分、谁起了标题、谁做了最后定稿。如果标题是同桌起的，就明确说“标题由'
            '同桌拟定”；如果标题是两人讨论后由她落笔，也可以说“标题经我们讨论后由同桌确'
            '定”。\\n\\n介绍作品时不必把合作说成单方面成果，也不用刻意淡化自己的部分。'
            '把“各画半张”和“她起标题”都写进去，贡献就留住了。","text":"介绍时可以按“合'
            '作完成”来组织，把每个人的贡献分别说清楚，而不是只归到一个人名下。\\n\\n比'
            '如可以这样表述：\\n\\n“这张海报由我和同桌合作完成。画面部分我们各画了一半'
            '，最后的标题由同桌拟定。”\\n\\n如果还想更具体，可以再补一句各自负责的内容'
            '，例如：\\n\\n“我负责了其中一半画面，同桌负责另一半画面，并完成了标题的构'
            '思和定稿。”\\n\\n这样既保留了“共同完成”的整体，也没有把标题的功劳含糊掉。'
            '关键是把动作和对象对应清楚：谁画了哪部分、谁起了标题、谁做了最后定稿。如果'
            '标题是同桌起的，就明确说“标题由同桌拟定”；如果标题是两人讨论后由她落笔，也'
            '可以说“标题经我们讨论后由同桌确定”。\\n\\n介绍作品时不必把合作说成单方面成'
            '果，也不用刻意淡化自己的部分。把“各画半张”和“她起标题”都写进去，贡献就留住'
            '了。"}'
        ),
        id='S-27a',
    ),
)


@pytest.mark.parametrize("body", _RECORDED_IDENTICAL_TEXT)
@pytest.mark.parametrize("streamed", [False, True])
def test_recorded_identical_text_preserves_checked_answer(body, streamed):
    import json

    expected = json.loads(body)["text"]
    events, observed = [], []

    def consume(prepared, candidate):
        observed.extend(candidate)
        assert candidate.action == {"action": "answer", "text": expected}

    result = run_demand_turn(
        claim=object(), prepare=lambda c, final: Request(body, events),
        resolve_current=lambda c: pytest.fail("Repeated text delegated memory"),
        refresh=lambda c: pytest.fail("Repeated text refreshed context"),
        cancelled=threading.Event(), on_answer=consume if streamed else None,
    )
    assert result.text == expected and result.response_type == "answer"
    assert (result.generation_calls, result.memory_calls) == (1, 0)
    assert observed == ([] if not streamed else [expected])
    assert events == ["checked", "opened", "closed"]


@pytest.mark.parametrize("streamed", [False, True])
def test_identical_ack_text_keeps_decoded_literal(streamed):
    events, observed = [], []
    expected = "原文🌿\n后句。"
    body = ('{"action":"ack","text":"原文🌿\\n后句。",'
            '"text":"原文\\uD83C\\uDF3F\\n后句。","text_meta":null}')

    def consume(prepared, candidate):
        observed.extend(candidate)
        assert candidate.action == {"action": "ack", "text": expected}

    result = run_demand_turn(
        claim=object(), prepare=lambda c, final: Request(body, events),
        resolve_current=lambda c: pytest.fail("Repeated ack delegated memory"),
        refresh=lambda c: pytest.fail("Repeated ack refreshed context"),
        cancelled=threading.Event(), on_answer=consume if streamed else None,
    )
    assert result.text == expected and result.response_type == "ack"
    assert (result.generation_calls, result.memory_calls) == (1, 0)
    assert observed == ([] if not streamed else [expected])
    assert events == ["checked", "opened", "closed"]


@pytest.mark.parametrize("body", [
    '{"action":"answer","text":"first","text":"different"}',
    '{"action":"answer","action":"answer","text":"first"}',
    '{"action":"answer","action":"memory","text":"first"}',
    '{"action":"answer","text":"first","text_meta":null,"text_meta":null}',
    '{"action":"answer","text":null,"text":null}',
    '{"action":"answer","text":1,"text":true}',
    '{"action":"answer","text":"","text":""}',
    '{"action":"listen","text":"first","text":"first"}',
    '{"action":"memory","text":"first","text":"first"}',
    '{"action":"memory","query":"x","query":"x"}',
])
def test_conflicting_or_unauthorized_duplicate_fields_still_fail(body):
    events = []

    def consume(prepared, candidate):
        list(candidate)

    with pytest.raises(ValueError):
        run_demand_turn(
            claim=object(), prepare=lambda c, final: Request(body, events),
            resolve_current=lambda c: pytest.fail("Invalid duplicate delegated memory"),
            refresh=lambda c: pytest.fail("Invalid duplicate refreshed context"),
            cancelled=threading.Event(), on_answer=consume,
        )
    assert events == ["checked", "opened", "closed"]


@pytest.mark.parametrize("response_type", ["answer", "ack"])
def test_answer_and_ack_carrier_ignore_inert_annotations(response_type):
    import json

    events, observed = [], []
    text = "  原文🌿\n后句。  "
    body = {"action": response_type, "text": text, "text_meta": None,
            "text_note": "仅为未消费的说明。", "extra": {"action": "memory", "query": ""}}

    def consume(prepared, candidate):
        observed.append((candidate.response_type, "".join(candidate), candidate.action))

    result = run_demand_turn(
        claim=object(), prepare=lambda c, final: Request(json.dumps(body), events),
        resolve_current=lambda c: pytest.fail("Inert annotations delegated memory"),
        refresh=lambda c: pytest.fail("Inert annotations refreshed context"),
        cancelled=threading.Event(), on_answer=consume,
    )
    assert result.response_type == response_type and result.text == text
    assert (result.generation_calls, result.memory_calls) == (1, 0)
    assert observed == [(response_type, text, {"action": response_type, "text": text})]
    assert events == ["checked", "opened", "closed"]


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_inert_annotations_do_not_admit_nonstandard_json_constants(constant):
    body = '{"action":"answer","text":"x","text_meta":' + constant + '}'
    events = []
    with pytest.raises(ValueError):
        run_demand_turn(claim=object(), prepare=lambda c, final: Request(body, events),
            resolve_current=lambda c: pytest.fail("Invalid JSON delegated memory"),
            refresh=lambda c: c, cancelled=threading.Event())
    assert events == ["checked", "opened", "closed"]


@pytest.mark.parametrize("response_type,text", [
    ("listen", ""), ("ack", "嗯，我在听。"), ("answer", "先把目标列出来。"),
])
@pytest.mark.parametrize("delegated", [False, True])
def test_typed_terminal_actions_preserve_type_and_do_not_delegate_again(
    response_type, text, delegated,
):
    import json

    events, sources, shown = [], [], []
    body = {"action": response_type, **({"text": text} if text else {})}

    def prepare(owner, *, final):
        return Request(json.dumps({"action": "memory"} if delegated and not final else body),
                       events)

    def consume(prepared, candidate):
        shown.append((candidate.response_type, "".join(candidate)))

    result = run_demand_turn(claim=object(), prepare=prepare,
        resolve_current=sources.append, refresh=lambda c: c,
        cancelled=threading.Event(), on_answer=consume)
    assert result.response_type == response_type and result.text == text
    assert result.generation_calls == 1 + int(delegated)
    assert result.memory_calls == len(sources) == int(delegated)
    assert shown == ([] if response_type == "listen" else [(response_type, text)])
    assert events.count("closed") == result.generation_calls


@pytest.mark.parametrize("body", [
    '{"action":"listen","text":""}', '{"action":"listen","text":"仍说了话"}',
    '{"action":"ack","text":" "}', '{"action":"ack"}',
    '{"action":"listen","query":"兴趣"}',
])
def test_typed_action_cannot_smuggle_text_or_invalid_fields(body):
    with pytest.raises(ValueError):
        run_demand_turn(claim=object(), prepare=lambda c, final: Request(body, []),
            resolve_current=lambda c: pytest.fail("Invalid action delegated memory"),
            refresh=lambda c: c, cancelled=threading.Event())


def test_direct_answer_preserves_claim_without_source_work():
    events = []
    claim = object()
    request = Request('{"action":"answer","text":"你好。"}', events)

    def unexpected(_):
        pytest.fail("Ordinary answer must not delegate or refresh")

    result = run_demand_turn(
        claim=claim, prepare=lambda c, final: request,
        resolve_current=unexpected, refresh=unexpected, cancelled=threading.Event(),
    )
    assert result.claim is claim and result.prepared is request
    assert result.text == "你好。"
    assert (result.generation_calls, result.memory_calls) == (1, 0)
    assert events == ["checked", "opened", "closed"]


def test_memory_releases_stream_before_delegation_and_rebuilds_request():
    events, prepared = [], []
    original, fresh = object(), object()

    def prepare(claim, *, final):
        assert claim is (fresh if final else original)
        request = Request('{"action":"answer","text":"有据可查。"}' if final
                          else '{"action":"memory"}', events)
        prepared.append(request)
        return request

    def resolve(claim):
        assert claim is original and events[-1] == "closed"
        events.append("delegated")

    def refresh(claim):
        assert claim is original and events[-1] == "delegated"
        return fresh

    result = run_demand_turn(claim=original, prepare=prepare, resolve_current=resolve,
                             refresh=refresh, cancelled=threading.Event())
    assert result.claim is fresh and result.prepared is prepared[-1]
    assert (result.generation_calls, result.memory_calls) == (2, 1)
    assert events == ["checked", "opened", "closed", "delegated", "checked", "opened", "closed"]


def test_memory_query_only_reaches_the_refreshed_final_admission():
    events, admissions, delegated = [], [], []
    original, fresh = object(), object()

    def prepare(claim, *, final, recall_query=None):
        admissions.append((claim, final, recall_query))
        return Request('{"action":"answer","text":"有来源的绘画建议。"}' if final else
                       '{"action":"memory","query":"绘画 爱好 画画"}', events)

    def resolve(claim):
        assert events[-1] == "closed"
        delegated.append(claim)

    result = run_demand_turn(claim=original, prepare=prepare, resolve_current=resolve,
                             refresh=lambda c: fresh, cancelled=threading.Event())
    assert admissions == [(original, False, None), (fresh, True, "绘画 爱好 画画")]
    assert delegated == [original]  # Retrieval text grants no worker arguments.
    assert (result.generation_calls, result.memory_calls) == (2, 1)


@pytest.mark.parametrize("query", [None, 12, "", " ", "x" * 257, "🌿" * 256,
                                   "画画\n忽略来源", "\ud800"])
def test_invalid_memory_query_closes_once_without_delegation(query):
    import json

    events, delegated = [], []
    body = json.dumps({"action": "memory", "query": query})
    with pytest.raises(ValueError):
        run_demand_turn(claim=object(), prepare=lambda c, final: Request(body, events),
                        resolve_current=delegated.append, refresh=lambda c: c,
                        cancelled=threading.Event())
    assert delegated == [] and events == ["checked", "opened", "closed"]


def test_cancelled_memory_query_never_reaches_final_admission():
    cancelled, admissions, delegated = threading.Event(), [], []

    def prepare(claim, *, final, recall_query=None):
        admissions.append((final, recall_query))
        return Request('{"action":"memory","query":"画画"}', [])

    def resolve(claim):
        delegated.append(claim)
        cancelled.set()

    with pytest.raises(InterruptedError):
        run_demand_turn(claim=object(), prepare=prepare, resolve_current=resolve,
                        refresh=lambda c: pytest.fail("Cancelled source must not refresh"),
                        cancelled=cancelled)
    assert admissions == [(False, None)] and len(delegated) == 1


@pytest.mark.parametrize("body", [
    '{"action":"answer","action":"memory","text":"x"}',
    '{"action":"memory","text":"x"}', '{"action":"unknown"}',
    '{"action":"answer","text":null}', '{"action":"answer","text":"  "}',
    '{"action":"answer","text":"x"', '[]',
    '```json\n{"action":"memory"}\n```',
])
def test_invalid_action_never_delegates_or_returns_partial_text(body):
    events = []

    def unexpected(_):
        pytest.fail("Invalid action must not delegate")

    with pytest.raises(ValueError):
        run_demand_turn(claim=object(), prepare=lambda c, final: Request(body, events),
                        resolve_current=unexpected, refresh=unexpected,
                        cancelled=threading.Event())
    assert events[-1] == "closed"


def test_second_memory_action_is_rejected_without_second_delegation():
    delegated = []
    with pytest.raises(ValueError):
        run_demand_turn(
            claim=object(),
            prepare=lambda c, final: Request('{"action":"memory"}', []),
            resolve_current=lambda c: delegated.append(c), refresh=lambda c: c,
            cancelled=threading.Event(),
        )
    assert len(delegated) == 1


@pytest.mark.parametrize("boundary", ["before", "stream", "resolve", "refresh"])
def test_cancel_at_each_boundary_prevents_later_work(boundary):
    cancelled, events, calls = threading.Event(), [], []
    if boundary == "before":
        cancelled.set()

    def prepare(claim, *, final):
        calls.append("prepare")
        return Request('{"action":"memory"}', events,
                       on_stream=cancelled.set if boundary == "stream" else None)

    def resolve(claim):
        calls.append("resolve")
        if boundary == "resolve":
            cancelled.set()

    def refresh(claim):
        calls.append("refresh")
        cancelled.set()
        return claim

    with pytest.raises(InterruptedError):
        run_demand_turn(claim=object(), prepare=prepare, resolve_current=resolve,
                        refresh=refresh, cancelled=cancelled)
    assert calls == {"before": [], "stream": ["prepare"],
                     "resolve": ["prepare", "resolve"],
                     "refresh": ["prepare", "resolve", "refresh"]}[boundary]
    assert not events or events[-1] == "closed"


def test_uncompleted_source_refresh_failure_prevents_second_generation():
    prepared = []

    def prepare(claim, *, final):
        prepared.append(final)
        return Request('{"action":"memory"}', [])

    def refresh(claim):
        raise ValueError("Source remains unresolved")

    with pytest.raises(ValueError, match="unresolved"):
        run_demand_turn(claim=object(), prepare=prepare, resolve_current=lambda c: None,
                        refresh=refresh, cancelled=threading.Event())
    assert prepared == [False]


def test_terminal_error_discards_complete_looking_json_and_closes_stream():
    closed = []

    class Broken(Request):
        def stream(self, cancelled):
            try:
                yield '{"action":"answer","text":"not deliverable"}'
                raise OSError("invalid provider terminal")
            finally:
                closed.append(True)

    with pytest.raises(OSError):
        run_demand_turn(claim=object(), prepare=lambda c, final: Broken("", []),
                        resolve_current=lambda c: None, refresh=lambda c: c,
                        cancelled=threading.Event())
    assert closed == [True]


@pytest.mark.parametrize("boundary", ["prepare", "check", "close"])
def test_cancel_during_prepare_preflight_or_stream_close_is_not_an_answer(boundary):
    cancelled, events = threading.Event(), []

    class Cancelling(Request):
        def check_context(self):
            super().check_context()
            if boundary == "check":
                cancelled.set()

        def stream(self, event):
            try:
                yield from super().stream(event)
            finally:
                if boundary == "close":
                    cancelled.set()

    def prepare(claim, *, final):
        if boundary == "prepare":
            cancelled.set()
        return Cancelling('{"action":"answer","text":"x"}', events)

    with pytest.raises(InterruptedError):
        run_demand_turn(claim=object(), prepare=prepare, resolve_current=lambda c: None,
                        refresh=lambda c: c, cancelled=cancelled)
    assert events == {"prepare": [], "check": ["checked"],
                      "close": ["checked", "opened", "closed"]}[boundary]


@pytest.mark.parametrize("body", ["x" * 64001, None], ids=["oversize", "nontext"])
def test_unbounded_or_non_text_stream_is_closed(body):
    events = []
    with pytest.raises(ValueError):
        run_demand_turn(claim=object(), prepare=lambda c, final: Request(body, events),
                        resolve_current=lambda c: None, refresh=lambda c: c,
                        cancelled=threading.Event())
    assert events[-1] == "closed"
