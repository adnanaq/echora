import pytest
from work_matching.doubled_provider_decision import (
    DoubledProviderDecision,
    Outcome,
    ReviewReason,
)


@pytest.mark.parametrize(
    "outcome",
    [
        Outcome.DUPLICATE_PAGES,
        Outcome.PART_PAGES,
        Outcome.SEPARATE_WORKS,
        Outcome.UNDECIDED,
    ],
)
def test_doubled_provider_decision_needs_review_always_reviewed_outcome_true(
    outcome: Outcome,
) -> None:
    assert DoubledProviderDecision("mal", (), outcome).needs_review


@pytest.mark.parametrize("outcome", [Outcome.SAME_PAGE, Outcome.PAGES_OF_OTHER_WORKS])
def test_doubled_provider_decision_needs_review_applied_outcome_without_reasons_false(
    outcome: Outcome,
) -> None:
    assert not DoubledProviderDecision("mal", (), outcome).needs_review


@pytest.mark.parametrize("review_reason", list(ReviewReason))
def test_doubled_provider_decision_needs_review_any_review_reason_true(
    review_reason: ReviewReason,
) -> None:
    decision = DoubledProviderDecision(
        "mal", (), Outcome.PAGES_OF_OTHER_WORKS, (review_reason,)
    )
    assert decision.needs_review
