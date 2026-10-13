"""A host decision must not be persisted as a model review that never happened."""

from dataclasses import replace

import pytest
from test_rebuilt_presentation import approved, candidate
from virtual_companion.boundary.presentation import (
    PresentationPolicy,
    ReviewedPresentation,
    review_by_policy,
)


def test_host_approval_roundtrips_without_fabricating_model_approval():
    value = candidate()
    policy = PresentationPolicy(64000, 4000, 1, False)
    result = review_by_policy(value, policy, "review", policy_revision="short-dialogue.v1")
    assert result.passed and result.model_approved is False
    bound = ReviewedPresentation(value, policy, result)
    assert bound.to_mapping()["review"]["approval_source"] == "host_policy"
    assert ReviewedPresentation.from_mapping(bound.to_mapping()) == bound


def test_host_policy_still_checks_actual_surface_and_requires_revision():
    value = candidate()
    policy = PresentationPolicy(0, 1, 1, False)
    result = review_by_policy(value, policy, "review", policy_revision="short-dialogue.v1")
    assert not result.passed
    with pytest.raises(ValueError):
        ReviewedPresentation(value, policy, replace(result, issues=()))
    with pytest.raises(ValueError):
        review_by_policy(value, policy, "review", policy_revision="")
    with pytest.raises(ValueError):
        replace(result, model_approved=True)


def test_legacy_serialization_stays_unchanged_and_old_store_rejects_host_policy(tmp_path):
    from test_rebuilt_presentation import prepared
    from virtual_companion.boundary.sqlite_presentation import SQLitePresentationStore

    assert set(approved(candidate()).to_mapping()["review"]) == {
        "review_id", "candidate_sha256", "policy_sha256", "model_approved", "issues"
    }
    store = SQLitePresentationStore(tmp_path / "v4.sqlite3")
    item, result = prepared(store)
    bound = result.presentation
    decision = review_by_policy(bound.candidate, bound.policy, bound.review.review_id,
                                policy_revision="short-dialogue.v2")
    changed = replace(result, presentation=replace(bound, review=decision))
    with pytest.raises(ValueError, match="storage format"):
        store.commit(changed, now_us=101)
    assert store.inspect(item.companion_id, item.input_id)["commit"] is None
