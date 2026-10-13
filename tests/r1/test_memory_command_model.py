from threading import Event

import pytest
from virtual_companion.boundary.contracts import SourceResolutionUnresolved
from virtual_companion.cognition.memory_command_model import MemoryCommandModel

CATALOG = (
    {
        "record_id": "music",
        "version": 3,
        "fact_key": {"subject": "user", "predicate": "music", "scope": "profile"},
        "content": "我喜欢安静的音乐",
        "content_truncated": False,
    },
    {
        "record_id": "drink",
        "version": 2,
        "fact_key": {"subject": "user", "predicate": "drink", "scope": "profile"},
        "content": "我喜欢茶",
        "content_truncated": False,
    },
)


def raw_command(
    source,
    *,
    action="forget",
    scope="single_record",
    targets=("music",),
    coverage="pure_command",
    unresolved=(),
    quote=None,
    replacement="",
):
    quote = source if quote is None else quote
    quote_start = source.index(quote) if quote else None
    quote_end = quote_start + len(quote) if quote else None
    replacement_start = source.index(replacement) if replacement else None
    replacement_end = replacement_start + len(replacement) if replacement else None
    return {
        "action": action,
        "requestedScope": scope,
        "targetIds": list(targets),
        "evidenceQuote": quote,
        "quoteStart": quote_start,
        "quoteEnd": quote_end,
        "replacementQuote": replacement,
        "replacementStart": replacement_start,
        "replacementEnd": replacement_end,
        "coverage": coverage,
        "unresolved": list(unresolved),
    }


def interpreting(source, command, support=None):
    stages = []

    def invoke(stage, messages, schema, cancelled):
        assert not cancelled.is_set()
        stages.append(stage)
        if stage == "memory_command":
            return command
        if stage == "memory_command_support":
            return support
        raise AssertionError(stage)

    result = MemoryCommandModel(invoke, Event()).interpret(source, CATALOG)
    return result, stages


def test_explicit_forget_is_source_bound_and_host_versions_the_selected_target():
    source = "请忘记我喜欢安静音乐这件事。"
    command = raw_command(source)
    command["replacementStart"] = 0
    command["replacementEnd"] = 0
    result, stages = interpreting(
        source,
        command,
        {"meaning": "explicit_single_forget", "verdict": "supported"},
    )

    assert result.disposition == "execute"
    assert result.executable
    assert result.action == "forget"
    assert result.target_refs == (("music", 3),)
    assert result.evidence_quote == source
    assert result.source_sha256
    assert result.fingerprint
    assert stages == ["memory_command", "memory_command_support"]


def test_explicit_forget_ignores_inert_replacement_field_noise():
    source = "请忘记我喜欢安静音乐这件事。"
    command = raw_command(source)
    command.update(
        replacementQuote=",",
        replacementStart=None,
        replacementEnd=None,
    )

    result, stages = interpreting(
        source,
        command,
        {"meaning": "explicit_single_forget", "verdict": "supported"},
    )

    assert result.executable
    assert result.evaluator == "memory-command.v9"
    assert result.replacement == ""
    assert result.replacement_start is None and result.replacement_end is None
    assert stages == ["memory_command", "memory_command_support"]


def test_semantic_support_schema_cannot_rewrite_the_bound_command_action():
    source = "请忘记我喜欢安静音乐这件事。"
    stages = []

    def invoke(stage, messages, schema, cancelled):
        stages.append(stage)
        if stage == "memory_command":
            return raw_command(source)
        if stage == "memory_command_support":
            meanings = schema["properties"]["meaning"]["enum"]
            assert "explicit_single_forget" in meanings
            assert "explicit_single_correction" not in meanings
            return {"meaning": "explicit_single_forget", "verdict": "supported"}
        raise AssertionError(stage)

    result = MemoryCommandModel(invoke, Event()).interpret(source, CATALOG)

    assert result.executable
    assert stages == ["memory_command", "memory_command_support"]


def test_explicit_correction_preserves_exact_replacement_and_target_version():
    source = "刚才说错了，其实我更喜欢摇滚。"
    replacement = "我更喜欢摇滚"
    result, _ = interpreting(
        source,
        raw_command(
            source,
            action="correct",
            replacement=replacement,
            unresolved=("intent", "scope", "replacement"),
        ),
        {"meaning": "explicit_single_correction", "verdict": "supported"},
    )

    assert result.executable
    assert result.action == "correct"
    assert result.replacement == replacement
    assert source[result.replacement_start : result.replacement_end] == replacement


@pytest.mark.parametrize(
    ("source", "meaning"),
    [
        ("她说‘请忘记我喜欢音乐’。", "quotation"),
        ("如果我让你忘记音乐偏好，会怎样？", "hypothetical"),
        ("不要忘记我喜欢安静的音乐。", "negated"),
    ],
)
def test_semantic_support_rejects_quoted_hypothetical_and_negated_false_commands(
    source, meaning
):
    result, _ = interpreting(
        source,
        raw_command(source),
        {"meaning": meaning, "verdict": "unsupported"},
    )

    assert result.disposition == "not_command"
    assert not result.executable
    assert result.meaning == meaning


@pytest.mark.parametrize(
    ("command", "issue"),
    [
        (raw_command("忘掉那个。", targets=()), "target"),
        (raw_command("把音乐和饮品都忘掉。", targets=("music", "drink")), "target"),
        (raw_command("忘掉音乐，顺便推荐一首歌。", coverage="mixed"), "mixed_content"),
        (
            raw_command(
                "以后任何饮品都不要记。",
                action="unsupported",
                scope="category",
                targets=(),
                unresolved=("scope",),
            ),
            "scope",
        ),
    ],
)
def test_ambiguous_mixed_and_unsupported_scope_never_become_commands(command, issue):
    result, stages = interpreting(command["evidenceQuote"], command)

    assert result.disposition == "clarify"
    assert not result.executable
    assert issue in result.unresolved
    assert stages == ["memory_command"]


def test_unsupported_scope_cannot_attach_an_advisory_record_to_clarification():
    source = "请彻底删除我的全部数据。"
    result, stages = interpreting(
        source,
        raw_command(
            source,
            action="forget",
            scope="physical_delete",
            targets=("music",),
        ),
    )

    assert result.disposition == "clarify"
    assert result.unresolved == ("scope",)
    assert result.target_refs == ()
    assert stages == ["memory_command"]


def test_host_recovers_unique_quote_position_instead_of_trusting_model_arithmetic():
    source = "请忘记音乐偏好。"
    command = raw_command(source)
    command["quoteStart"] = 1

    result, _ = interpreting(
        source,
        command,
        {"meaning": "explicit_single_forget", "verdict": "supported"},
    )

    assert result.quote_start == 0
    assert result.quote_end == len(source)


def test_host_rejects_an_ambiguous_quote_when_the_supplied_position_is_wrong():
    source = "忘记音乐偏好，再说一次：忘记音乐偏好。"
    command = raw_command(source, quote="忘记音乐偏好")
    command["quoteStart"] = 1

    with pytest.raises(ValueError, match="exact source span"):
        interpreting(source, command)


def test_host_binds_catalog_copied_evidence_to_one_unique_source_overlap():
    source = "请忘记我喜欢喝茶这件事。"
    catalog = (
        {
            "record_id": "drink",
            "version": 2,
            "fact_key": {"subject": "user", "predicate": "drink", "scope": "profile"},
            "content": "我喜欢喝茶。",
            "content_truncated": False,
        },
    )
    command = raw_command(source)
    command.update(
        targetIds=["drink"],
        evidenceQuote="我喜欢喝茶。",
        quoteStart=0,
        quoteEnd=len("我喜欢喝茶。"),
    )

    def invoke(stage, messages, schema, cancelled):
        if stage == "memory_command":
            return command
        if stage == "memory_command_support":
            return {"meaning": "explicit_single_forget", "verdict": "supported"}
        raise AssertionError(stage)

    result = MemoryCommandModel(invoke, Event()).interpret(source, catalog)

    assert result.executable
    assert result.evidence_quote == "我喜欢喝茶"
    assert source[result.quote_start : result.quote_end] == result.evidence_quote


def test_host_does_not_bind_a_short_or_ambiguous_overlap():
    source = "请忘记音乐偏好，再提一次音乐偏好。"
    catalog = (
        {
            "record_id": "music",
            "version": 1,
            "fact_key": {"subject": "user", "predicate": "music", "scope": "profile"},
            "content": "我喜欢音乐偏好",
            "content_truncated": False,
        },
    )
    command = raw_command(source)
    command.update(
        targetIds=["music"],
        evidenceQuote="我喜欢音乐偏好",
        quoteStart=0,
        quoteEnd=len("我喜欢音乐偏好"),
    )

    def invoke(stage, messages, schema, cancelled):
        if stage == "memory_command":
            return command
        raise AssertionError(stage)

    with pytest.raises(ValueError, match="exact source span"):
        MemoryCommandModel(invoke, Event()).interpret(source, catalog)


def test_host_rejects_target_identity_absent_from_its_catalog_page():
    source = "请忘记音乐偏好。"
    command = raw_command(source)
    command["targetIds"] = ["invented"]

    with pytest.raises(ValueError, match="bound schema"):
        interpreting(source, command)


def test_no_command_ignores_non_authoritative_quote_and_position_noise():
    source = "我喜欢安静的音乐。"
    command = raw_command(
        source,
        action="none",
        scope="none",
        targets=("music", "drink"),
        coverage="pure_command",
        unresolved=("intent", "scope", "replacement"),
    )
    command.update(
        quoteStart=0,
        quoteEnd=len(source.encode()),
        replacementStart=0,
        replacementEnd=0,
    )

    result, stages = interpreting(source, command)

    assert result.disposition == "not_command"
    assert result.evidence_quote == ""
    assert result.quote_start is None and result.quote_end is None
    assert result.replacement == ""
    assert result.replacement_start is None and result.replacement_end is None
    assert stages == ["memory_command"]


def test_unclear_without_management_evidence_is_a_normal_memory_query():
    source = "我后来搬去哪了？"
    result, stages = interpreting(
        source,
        raw_command(
            source,
            action="unclear",
            scope="none",
            targets=(),
            coverage="unclear",
        ),
    )

    assert result.disposition == "not_command"
    assert result.action == "none"
    assert result.coverage == "not_command"
    assert result.unresolved == ()
    assert stages == ["memory_command"]


def test_unclear_explicit_management_marker_stays_protected():
    source = "忘掉那个。"
    result, stages = interpreting(
        source,
        raw_command(
            source,
            action="unclear",
            scope="none",
            targets=(),
            coverage="unclear",
        ),
    )

    assert result.disposition == "clarify"
    assert result.action == "unclear"
    assert result.unresolved == ("intent",)
    assert stages == ["memory_command"]


def test_paged_catalog_cannot_change_source_meaning_between_pages():
    source = "请忘记音乐偏好。"
    catalog = tuple(
        {
            "record_id": f"record-{index}",
            "version": 1,
            "fact_key": None,
            "content": str(index),
            "content_truncated": False,
        }
        for index in range(101)
    )
    calls = 0

    def invoke(stage, messages, schema, cancelled):
        nonlocal calls
        assert stage == "memory_command"
        calls += 1
        return raw_command(
            source,
            action="forget" if calls == 1 else "none",
            scope="single_record" if calls == 1 else "none",
            targets=(),
            coverage="pure_command" if calls == 1 else "not_command",
            quote=source if calls == 1 else "",
        )

    with pytest.raises(SourceResolutionUnresolved, match="pages disagree"):
        MemoryCommandModel(invoke, Event()).interpret(source, catalog)
