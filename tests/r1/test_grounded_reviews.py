from dataclasses import replace

import pytest
from test_cognition_preparation import basis
from test_cognitive_kernel import RecordedModel, kernel
from test_rebuilt_turns import A
from virtual_companion.boundary.contracts import Scope
from virtual_companion.boundary.presentation import MessageUnit, Presentation
from virtual_companion.boundary.runtime import KernelFailure
from virtual_companion.cognition.reviews import GroundedReviewGate, ReviewOpinion, adjudicate


def surface():
    return Presentation(
        Scope(A, "session"),
        "input",
        "attempt",
        "candidate",
        1,
        "隐藏语义",
        (MessageUnit("one", "text", "合成回答"),),
    )


@pytest.mark.parametrize(
    "positive,foreign,quotes,hard,expected",
    [
        (False, False, ("合成",), True, (False, "rejected")),
        (False, True, ("合成",), True, (True, "ignored_ungrounded")),
        (False, False, ("不存在",), True, (True, "ignored_ungrounded")),
        (False, False, ("隐藏语义",), True, (True, "ignored_ungrounded")),
        (False, False, ("合成", ""), True, (True, "ignored_ungrounded")),
        (False, True, ("合成",), False, (False, "rejected")),
        (True, False, (), False, (False, "accepted")),
        (True, True, (), True, (True, "ignored_ungrounded")),
    ],
)
def test_grounding_truth_table_preserves_actual_surface_and_hard_failure(
    positive, foreign, quotes, hard, expected
):
    candidate = surface()
    opinion = ReviewOpinion(
        "0" * 64 if foreign else candidate.fingerprint, positive, quotes, "recorded"
    )
    result = adjudicate(candidate, hard, opinion)
    assert (result.passed, result.disposition) == expected
    assert result.candidate_sha256 == candidate.fingerprint


def grounded_model(quote, *, foreign=False):
    model = RecordedModel()

    def transform(stage, row):
        if stage == "review":
            for duty in ("causal_review", "expression_review"):
                row[duty].update(
                    candidate_sha256="0" * 64 if foreign else row["candidate_sha256"],
                    evidence_quotes=[quote],
                )
            row["causal_review"]["passed"] = False
        return row

    model.mutate = transform
    return model


@pytest.mark.parametrize("quote,foreign", [("fabricated", False), ("合成", True)])
def test_kernel_ignores_ungrounded_opinion_but_binds_final_duties(tmp_path, quote, foreign):
    store, _, run, persona = basis(tmp_path)
    model = grounded_model(quote, foreign=foreign)
    candidate = kernel(store, persona, model)
    candidate.review_gate = GroundedReviewGate(lambda context, presentation: (True, True))
    result = candidate.generate(run, store.snapshot(A), "input")
    assert result.revision == 1
    decisions = [r for r in candidate.trace if r["stage"] == "decision"]
    assert decisions[0]["adjudications"]["causal_review"]["disposition"] == "ignored_ungrounded"
    store.commit(result, now_us=100)


def test_grounded_negative_rewrites_but_cannot_override_deterministic_failure(tmp_path):
    store, _, run, persona = basis(tmp_path)
    model = grounded_model("合成")
    candidate = kernel(store, persona, model)
    candidate.review_gate = GroundedReviewGate(lambda context, presentation: (True, True))
    with pytest.raises(KernelFailure, match="review_exhausted"):
        candidate.generate(run, store.snapshot(A), "input")
    assert len([r for r in model.requests if r["stage"] == "reply"]) == 3
    model = grounded_model("fabricated")
    candidate.model = model
    candidate.review_gate = GroundedReviewGate(lambda context, presentation: (False, True))
    with pytest.raises(KernelFailure, match="review_exhausted"):
        candidate.generate(run, store.snapshot(A), "input")


def test_gate_rejects_missing_deterministic_evidence_and_bad_quote_types():
    candidate = surface()
    with pytest.raises(ValueError):
        ReviewOpinion(candidate.fingerprint, False, ["quote"], "reason")
    with pytest.raises(ValueError):
        adjudicate(candidate, "true", None)
    with pytest.raises(ValueError):
        replace(
            ReviewOpinion(candidate.fingerprint, True, (), "reason"), candidate_sha256="foreign"
        )


def test_internal_judge_failure_is_not_mislabeled_as_bad_model_output(tmp_path):
    store, _, run, persona = basis(tmp_path)
    candidate = kernel(store, persona, grounded_model("合成"))

    def broken(context, presentation):
        raise ValueError("internal_judge_failure")

    candidate.review_gate = GroundedReviewGate(broken)
    with pytest.raises(ValueError, match="internal_judge_failure"):
        candidate.generate(run, store.snapshot(A), "input")
    assert store.inspect(A, run.input_id)["status"] == "running"
