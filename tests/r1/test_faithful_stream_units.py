"""RF3-S1 faithful display units: the model's own bytes, never an inserted separator.

A display unit is a contiguous range of the host-bound immutable source text, so
reassembling the units reproduces the original text exactly. Candidate boundaries
(`\n\n`, `\n`, two-space line break, sentence end) only schedule a review; they
never approve content.
"""

from threading import Event

import pytest
from test_rebuilt_turns import A, claim, config
from virtual_companion.boundary.presentation import (
    MessageUnit,
    Presentation,
    PresentationPolicy,
    PresentationReview,
    ReviewedPresentation,
    digest,
    review_by_policy,
)
from virtual_companion.boundary.sqlite_segments import SQLiteSegmentStore
from virtual_companion.boundary.sqlite_streaming import SQLiteStreamingStore
from virtual_companion.cognition.short_dialogue import ShortRequest
from virtual_companion.cognition.stream_dialogue import StreamDialogue
from virtual_companion.cognition.stream_unit_review import BoundedUnitReviewer
from virtual_companion.providers.text_deltas import UnapprovedTextDelta

SOURCE = "甲句。乙句。\n\n丙句。"
CUTS = (3, 8, len(SOURCE))


def test_response_type_is_part_of_review_identity_and_old_answer_hash_is_unchanged():
    from dataclasses import replace

    candidate = Presentation(complex_scope(claim_store()), "input", "attempt", "candidate", 1,
                             SOURCE, (MessageUnit("one", "text", SOURCE),))
    assert candidate.response_type == "answer"
    assert "response_type" not in candidate.to_mapping()
    ack = replace(candidate, response_type="ack")
    assert ack.fingerprint != candidate.fingerprint
    policy = PresentationPolicy(10, 1000, 32, False)
    old_review = review_by_policy(candidate, policy, "answer-review", policy_revision="recording")
    with pytest.raises(ValueError, match="binding changed"):
        ReviewedPresentation(ack, policy, old_review)
    bound = ReviewedPresentation(ack, policy,
        review_by_policy(ack, policy, "ack-review", policy_revision="recording"))
    assert ReviewedPresentation.from_mapping(bound.to_mapping()) == bound


def test_approved_prefix_cannot_change_ack_type_to_answer(tmp_path):
    from dataclasses import replace

    from test_segment_store import reviewed

    store = SQLiteSegmentStore(tmp_path / "typed-prefix.sqlite3")
    run = claim(store, store.accept(config(), "typed", "我还没说完。"))

    def typed(texts, response_type):
        previous = reviewed(run, texts)
        candidate = replace(previous.candidate, response_type=response_type)
        return ReviewedPresentation(candidate, previous.policy, review_by_policy(
            candidate, previous.policy, "typed-review", policy_revision="synthetic-unit-policy-v1"))

    first = store.approve_segment(run, typed(("我在听。",), "ack"), now_us=100)
    with pytest.raises(ValueError, match="cannot be rewritten"):
        store.approve_segment(run, typed(("我在听。", "继续。"), "answer"), now_us=100)
    assert store.segments(A, run.input_id)[0]["fingerprint"] == first["fingerprint"]


def spanned(run, source, cuts, texts=None):
    from virtual_companion.boundary.presentation import span_units

    candidate = Presentation(complex_scope(run), run.input_id, run.attempt_id, "candidate", 1,
                             source, span_units(source, cuts), "speak", "spans-v1", source)
    policy = PresentationPolicy(10, 1000, 32, False)
    return ReviewedPresentation(candidate, policy,
        review_by_policy(candidate, policy, "review" + str(len(cuts)),
                         policy_revision="synthetic-span-policy-v1"))


def complex_scope(run):
    from virtual_companion.boundary.contracts import Scope

    return Scope(A, run.session_id)


def test_span_units_reproduce_the_original_text_without_inserting_characters():
    from virtual_companion.boundary.presentation import span_units

    units = span_units(SOURCE, CUTS)
    assert [unit.text for unit in units] == ["甲句。", "乙句。\n\n", "丙句。"]
    assert [unit.unit_id for unit in units] == ["unit-1", "unit-2", "unit-3"]
    assert [(unit.start, unit.end) for unit in units] == [(0, 3), (3, 8), (8, len(SOURCE))]
    assert "".join(unit.text for unit in units) == SOURCE
    assert "\n\n" not in "".join(unit.text for unit in units if unit.unit_id != "unit-2")


def test_spans_representation_round_trips_and_keeps_its_own_fingerprint():
    from virtual_companion.boundary.presentation import span_units

    candidate = Presentation(complex_scope(claim_store()), "input-1", "attempt-1", "candidate", 1,
                             SOURCE, span_units(SOURCE, CUTS), "speak", "spans-v1", SOURCE)
    mapping = candidate.to_mapping()
    assert mapping["representation"] == "spans-v1" and mapping["source_text"] == SOURCE
    assert mapping["units"][1] == {"unit_id": "unit-2", "kind": "text", "text": "乙句。\n\n",
                                   "start": 3, "end": 8}
    assert candidate.text == SOURCE
    restored = Presentation(**{**mapping, "scope": candidate.scope,
                               "units": tuple(MessageUnit(**unit) for unit in mapping["units"])})
    assert restored == candidate and restored.fingerprint == candidate.fingerprint


def claim_store():
    import tempfile
    from pathlib import Path

    from test_rebuilt_turns import SQLiteTurnStore

    directory = Path(tempfile.mkdtemp())
    store = SQLiteTurnStore(directory / "scope.sqlite3")
    return claim(store, store.accept(config(), "first", "你好"))


def test_legacy_joined_mapping_and_hash_are_untouched():
    scope = claim_store()
    from virtual_companion.boundary.contracts import Scope

    units = (MessageUnit("unit1", "text", "甲句。"), MessageUnit("unit2", "text", "乙句。"))
    candidate = Presentation(Scope(A, scope.session_id), "input-1", "attempt-1", "candidate", 1,
                             "甲句。\n\n乙句。", units)
    mapping = candidate.to_mapping()
    assert "representation" not in mapping and "source_text" not in mapping
    assert mapping["units"] == [{"unit_id": "unit1", "kind": "text", "text": "甲句。"},
                                {"unit_id": "unit2", "kind": "text", "text": "乙句。"}]
    assert "intent" not in mapping
    assert digest(mapping) == candidate.fingerprint
    assert candidate.text == "甲句。\n\n乙句。"


@pytest.mark.parametrize("fault", ["gap", "overlap", "text_mismatch", "truncated", "bad_version"])
def test_span_validation_rejects_broken_ranges(fault):
    from virtual_companion.boundary.presentation import span_units

    scope = claim_store()
    from virtual_companion.boundary.contracts import Scope

    units = list(span_units(SOURCE, CUTS))
    representation, source = "spans-v1", SOURCE
    if fault == "gap":
        units[1] = MessageUnit("unit-2", "text", SOURCE[5:8], 5, 8)
    elif fault == "overlap":
        units[1] = MessageUnit("unit-2", "text", SOURCE[2:8], 2, 8)
    elif fault == "text_mismatch":
        units[1] = MessageUnit("unit-2", "text", "改写", 4, 8)
    elif fault == "truncated":
        units = units[:-1]
    else:
        representation = "joined-v3"
    with pytest.raises(ValueError):
        Presentation(Scope(A, scope.session_id), "input-1", "attempt-1", "candidate", 1,
                     SOURCE, tuple(units), "speak", representation, source)


def test_span_units_survive_astral_characters_and_utf8_boundaries():
    from virtual_companion.boundary.presentation import span_units

    source = "你好🌿。\n下一句。"
    units = span_units(source, (len("你好🌿。\n"), len(source)))
    assert [unit.text for unit in units] == ["你好🌿。\n", "下一句。"]
    assert "".join(unit.text for unit in units) == source
    assert units[0].end == 5 and units[1].start == 5


def test_reviewer_offers_actual_line_breaks_and_short_sentences(tmp_path):
    store = SQLiteStreamingStore(tmp_path / "faithful.sqlite3")
    run = claim(store, store.accept(config(), "first", "你好"))
    calls = []

    def assess(packet):
        calls.append(packet["payload"]["candidate_prefix"])
        return {"request_sha256": packet["sha256"], "decision": "approve",
                "self_contained": True, "no_open_dependency": True,
                "consistent_with_displayed": True, "constraints_satisfied": True}

    reviewer = BoundedUnitReviewer(run, ShortRequest("你好"), assess, early_limit=2)
    first = reviewer("甲句。", False)
    assert first is not None and first.candidate.text == "甲句。"
    assert first.candidate.representation == "spans-v1"
    assert first.candidate.source_text == "甲句。"
    assert "".join(unit.text for unit in first.candidate.units) == "甲句。"
    # The earliest boundary wins: the second sentence end precedes the blank line,
    # so the separator stays with the following unit and is never inserted twice.
    second = reviewer("甲句。乙句。\n\n", False)
    assert second is not None and second.candidate.text == "甲句。乙句。"
    assert [unit.text for unit in second.candidate.units] == ["甲句。", "乙句。"]
    final = reviewer("甲句。乙句。\n\n丙句。", True)
    assert final is not None and final.candidate.text == "甲句。乙句。\n\n丙句。"
    assert "".join(unit.text for unit in final.candidate.units) == "甲句。乙句。\n\n丙句。"
    assert calls == ["甲句。", "甲句。乙句。", "甲句。乙句。\n\n丙句。"]


def test_two_space_line_break_is_a_boundary_and_kept_verbatim(tmp_path):
    store = SQLiteStreamingStore(tmp_path / "twospace.sqlite3")
    run = claim(store, store.accept(config(), "first", "你好"))
    seen = []

    def assess(packet):
        seen.append(packet["payload"]["candidate_prefix"])
        return {"request_sha256": packet["sha256"], "decision": "approve",
                "self_contained": True, "no_open_dependency": True,
                "consistent_with_displayed": True, "constraints_satisfied": True}

    reviewer = BoundedUnitReviewer(run, ShortRequest("你好"), assess)
    result = reviewer("第一行  \n第二行。", False)
    assert result is not None
    assert result.candidate.text == "第一行  \n"
    assert result.candidate.units[-1].text == "第一行  \n"
    assert seen == ["第一行  \n"]


@pytest.mark.parametrize("observed", ["条件如下：\n", "他说：“甲句。\n", "甲句，\n"])
def test_structural_gates_still_withhold_new_boundary_kinds(tmp_path, observed):
    store = SQLiteStreamingStore(tmp_path / "gates.sqlite3")
    run = claim(store, store.accept(config(), "first", "你好"))
    calls = []

    def assess(packet):  # pragma: no cover - must never be consulted
        calls.append(packet)
        raise AssertionError("structural gate must decide without a model opinion")

    reviewer = BoundedUnitReviewer(run, ShortRequest("你好"), assess)
    assert reviewer(observed, False) is None
    assert calls == []
    assert reviewer.approved_text == ""


def test_one_early_opinion_is_enough_for_an_ordinary_turn(tmp_path):
    """An active producer gets one early opinion and one complete opinion."""
    store = SQLiteStreamingStore(tmp_path / "done-first.sqlite3")
    run = claim(store, store.accept(config(), "first", "你好"))
    calls, seen = [], []
    visible = Event()

    def assess(packet):
        calls.append(packet["payload"]["final"])
        return {"request_sha256": packet["sha256"], "decision": "approve",
                "self_contained": True, "no_open_dependency": True,
                "consistent_with_displayed": True, "constraints_satisfied": True}

    class Model:
        def __init__(self):
            self.receive = None

        def count(self, body):
            return 10

        def cancel(self):
            pass

        def invoke(self, body):
            self.receive(UnapprovedTextDelta(1, "甲句。乙句。"))
            assert visible.wait(2), "First unit was not presented during generation"
            return {"text": "甲句。乙句。", "tool_calls": [], "finish_reason": "stop"}

    def present(event):
        seen.append(event)
        visible.set()
        return f"host-{len(seen)}"

    model = Model()
    reviewer = BoundedUnitReviewer(run, ShortRequest("你好"), assess, early_limit=1)
    dialogue = StreamDialogue(
        store=store, model_factory=lambda receive: (setattr(model, "receive", receive), model)[1],
        review_prefix=reviewer,
        present=present,
        checkpoint=lambda: None, cancelled=Event(), clock=lambda: 100, parallel_review=True)
    assert dialogue.generate(run, ShortRequest("你好"))["outcome"] == "completed"
    state = store.stream_state(A, run.input_id)
    assert state["confirmed_text"] == "甲句。乙句。"
    # One early opinion plus the terminal opinion that owns the remainder: an
    # ordinary two-sentence reply costs two opinions, never one per unit.
    assert calls == [False, True], f"bounded opinions per turn, got {calls}"
    assert state["confirmed_units"] == 2


def test_long_answer_still_reviews_its_first_unit_while_generating(tmp_path):
    """Early presentation costs exactly one early opinion while generation continues."""
    store = SQLiteStreamingStore(tmp_path / "early-one.sqlite3")
    run = claim(store, store.accept(config(), "first", "你好"))
    calls, seen = [], []
    first_unit = Event()

    def assess(packet):
        calls.append(packet["payload"]["final"])
        return {"request_sha256": packet["sha256"], "decision": "approve",
                "self_contained": True, "no_open_dependency": True,
                "consistent_with_displayed": True, "constraints_satisfied": True}

    def present(event):
        seen.append(event)
        first_unit.set()
        return f"host-{len(seen)}"

    class Model:
        def __init__(self):
            self.receive = None

        def count(self, body):
            return 10

        def cancel(self):
            pass

        def invoke(self, body):
            self.receive(UnapprovedTextDelta(1, "甲句。"))
            assert first_unit.wait(2), "The first unit must be displayed while generating"
            self.receive(UnapprovedTextDelta(2, "乙句。"))
            self.receive(UnapprovedTextDelta(3, "丙句。"))
            return {"text": "甲句。乙句。丙句。", "tool_calls": [], "finish_reason": "stop"}

    model = Model()
    reviewer = BoundedUnitReviewer(run, ShortRequest("你好"), assess, early_limit=1)
    dialogue = StreamDialogue(
        store=store, model_factory=lambda receive: (setattr(model, "receive", receive), model)[1],
        review_prefix=reviewer, present=present,
        checkpoint=lambda: None, cancelled=Event(), clock=lambda: 100, parallel_review=True)
    assert dialogue.generate(run, ShortRequest("你好"))["outcome"] == "completed"
    assert [event["text"] for event in seen] == ["甲句。", "乙句。丙句。"]
    assert calls == [False, True]


def test_coordinator_confirms_faithful_units_without_inserted_blank_lines(tmp_path):
    store = SQLiteStreamingStore(tmp_path / "faithful-end.sqlite3")
    run = claim(store, store.accept(config(), "first", "你好"))
    displayed, seen = Event(), []
    source = "甲句。乙句。\n\n丙句。"

    def approve(packet):
        return {"request_sha256": packet["sha256"], "decision": "approve",
                "self_contained": True, "no_open_dependency": True,
                "consistent_with_displayed": True, "constraints_satisfied": True}

    class Model:
        def __init__(self):
            self.receive = None

        def count(self, body):
            return 10

        def cancel(self):
            pass

        def invoke(self, body):
            self.receive(UnapprovedTextDelta(1, "甲句。"))
            assert displayed.wait(2), "First sentence was not visible before terminal"
            self.receive(UnapprovedTextDelta(2, "乙句。\n\n丙句。"))
            return {"text": source, "tool_calls": [], "finish_reason": "stop"}

    model = Model()
    reviewer = BoundedUnitReviewer(run, ShortRequest("你好"), approve)

    def present(event):
        seen.append(event)
        displayed.set()
        return f"host-{len(seen)}"

    dialogue = StreamDialogue(
        store=store, model_factory=lambda receive: (setattr(model, "receive", receive), model)[1],
        review_prefix=reviewer, present=present,
        checkpoint=lambda: None, cancelled=Event(), clock=lambda: 100, parallel_review=True)
    assert dialogue.generate(run, ShortRequest("你好"))["outcome"] == "completed"
    state = store.stream_state(A, run.input_id)
    assert state["confirmed_text"] == source
    assert "".join(event["text"] for event in seen) == source
    assert state["display_join"] == ""
    assert state["representation"] == "spans-v1"
    final_units = seen[-1]["presentation"]["candidate"]["units"]
    assert [unit["start"] for unit in final_units] == sorted(unit["start"] for unit in final_units)
    assert "".join(unit["text"] for unit in final_units) == source


def test_trailing_whitespace_needs_no_new_unit_and_stays_faithful(tmp_path):
    store = SQLiteStreamingStore(tmp_path / "trailing.sqlite3")
    run = claim(store, store.accept(config(), "first", "你好"))
    seen = []
    visible = Event()

    def approve(packet):
        return {"request_sha256": packet["sha256"], "decision": "approve",
                "self_contained": True, "no_open_dependency": True,
                "consistent_with_displayed": True, "constraints_satisfied": True}

    reviewer = BoundedUnitReviewer(run, ShortRequest("你好"), approve)
    assert reviewer("甲句。", False) is not None
    # A whitespace-only remainder adds no unit and is never trimmed into content.
    assert reviewer("甲句。\n", True) is reviewer.approved
    assert reviewer.approved.candidate.text == "甲句。"
    assert reviewer.approved.candidate.source_text == "甲句。"

    class Model:
        def __init__(self):
            self.receive = None

        def count(self, body):
            return 10

        def cancel(self):
            pass

        def invoke(self, body):
            self.receive(UnapprovedTextDelta(1, "甲句。\n"))
            assert visible.wait(2), "First unit was not presented during generation"
            return {"text": "甲句。\n", "tool_calls": [], "finish_reason": "stop"}

    def present(event):
        seen.append(event)
        visible.set()
        return f"host-{len(seen)}"

    model = Model()
    instance = BoundedUnitReviewer(run, ShortRequest("你好"), approve)
    dialogue = StreamDialogue(
        store=store, model_factory=lambda receive: (setattr(model, "receive", receive), model)[1],
        review_prefix=instance, present=present,
        checkpoint=lambda: None, cancelled=Event(), clock=lambda: 100, parallel_review=True)
    assert dialogue.generate(run, ShortRequest("你好"))["outcome"] == "completed"
    state = store.stream_state(A, run.input_id)
    # The sentence is the earliest boundary, so the invisible trailing newline
    # never becomes its own unit; the raw model terminal still carries it.
    assert state["confirmed_text"] == "甲句。"
    assert "".join(event["text"] for event in seen) == "甲句。"
    from virtual_companion.cognition.stream_dialogue import content_agrees

    assert content_agrees("甲句。", "甲句。\n") and not content_agrees("甲句。", "甲句。乙")


def test_reopened_store_keeps_faithful_units_and_legacy_join(tmp_path):
    store = SQLiteStreamingStore(tmp_path / "reopen.sqlite3")
    run = claim(store, store.accept(config(), "first", "你好"))
    store.approve_segment(run, spanned(run, "甲句。", (3,)), now_us=100)
    event = store.dispatch_segment(run, 1, now_us=101)
    store.confirm_segment(A, run.input_id, 1, fingerprint=event["fingerprint"],
                          host_receipt="device-1", channel="text_visible")
    reopened = SQLiteStreamingStore(store.path)
    state = reopened.stream_state(A, run.input_id)
    assert state["confirmed_text"] == "甲句。"
    assert state["display_join"] == ""
    assert state["representation"] == "spans-v1"
    assert state["confirmed_units"] == 1
    assert event["display_text"] == "甲句。" and event["text"] == "甲句。"
    assert [unit["start"] for unit in event["presentation"]["candidate"]["units"]] == [0]

    legacy = SQLiteSegmentStore(tmp_path / "legacy.sqlite3")
    legacy_run = claim(legacy, legacy.accept(config(), "first", "你好"))
    from virtual_companion.boundary.contracts import Scope
    from virtual_companion.boundary.presentation import MessageUnit as Unit

    policy = PresentationPolicy(10, 1000, 32, False)

    def legacy_bound(texts):
        units = tuple(Unit(f"unit{index}", "text", text)
                      for index, text in enumerate(texts, 1))
        candidate = Presentation(Scope(A, legacy_run.session_id), legacy_run.input_id,
                                 legacy_run.attempt_id, "candidate", 1,
                                 "\n\n".join(texts), units)
        return ReviewedPresentation(candidate, policy,
            PresentationReview(f"review{len(texts)}", candidate.fingerprint,
                               policy.fingerprint, True, ()))

    first_legacy = legacy.approve_segment(legacy_run, legacy_bound(("甲句。",)), now_us=100)
    assert first_legacy["sequence"] == 1
    first = legacy.dispatch_segment(legacy_run, 1, now_us=101)
    legacy.confirm_segment(A, legacy_run.input_id, 1, fingerprint=first["fingerprint"],
                           host_receipt="device-2", channel="text_visible")
    second = legacy.approve_segment(legacy_run, legacy_bound(("甲句。", "乙句。")), now_us=102)
    assert second["sequence"] == 2 and second["text"] == "乙句。"
    assert second["display_text"] == "甲句。\n\n乙句。"
    assert SQLiteSegmentStore(legacy.path).segments(A, legacy_run.input_id)[1]["text"] == "乙句。"
