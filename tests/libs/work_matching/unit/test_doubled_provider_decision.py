import pytest
from work_matching.doubled_provider_decision import (
    DoubledProviderDecision,
    DoubledProviderOutcome,
    DoubledProviderReviewReason,
)


@pytest.mark.parametrize(
    "outcome",
    [
        DoubledProviderOutcome.DUPLICATE_PAGES,
        DoubledProviderOutcome.PART_PAGES,
        DoubledProviderOutcome.SEPARATE_WORKS,
        DoubledProviderOutcome.UNDECIDED,
    ],
)
def test_doubled_provider_decision_needs_review_always_reviewed_outcome_true(
    outcome: DoubledProviderOutcome,
) -> None:
    assert DoubledProviderDecision("mal", (), outcome).needs_review


@pytest.mark.parametrize(
    "outcome",
    [DoubledProviderOutcome.SAME_PAGE, DoubledProviderOutcome.PAGES_OF_OTHER_WORKS],
)
def test_doubled_provider_decision_needs_review_applied_outcome_without_reasons_false(
    outcome: DoubledProviderOutcome,
) -> None:
    assert not DoubledProviderDecision("mal", (), outcome).needs_review


@pytest.mark.parametrize("review_reason", list(DoubledProviderReviewReason))
def test_doubled_provider_decision_needs_review_any_review_reason_true(
    review_reason: DoubledProviderReviewReason,
) -> None:
    decision = DoubledProviderDecision(
        "mal", (), DoubledProviderOutcome.PAGES_OF_OTHER_WORKS, (review_reason,)
    )
    assert decision.needs_review
