import pytest
from work_matching.models import (
    DoubledProviderDecision,
    DoubledProviderOutcome,
    DoubledProviderReviewReason,
    LinkedPageReason,
    read_member_by_name,
    write_member_name,
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
def test_doubled_provider_decision_is_review_needed_always_reviewed_outcome_true(
    outcome: DoubledProviderOutcome,
) -> None:
    assert DoubledProviderDecision("mal", (), outcome).is_review_needed


@pytest.mark.parametrize(
    "outcome",
    [DoubledProviderOutcome.SAME_PAGE, DoubledProviderOutcome.PAGES_OF_OTHER_WORKS],
)
def test_doubled_provider_decision_is_review_needed_applied_outcome_without_reasons_false(
    outcome: DoubledProviderOutcome,
) -> None:
    assert not DoubledProviderDecision("mal", (), outcome).is_review_needed


@pytest.mark.parametrize("review_reason", list(DoubledProviderReviewReason))
def test_doubled_provider_decision_is_review_needed_any_review_reason_true(
    review_reason: DoubledProviderReviewReason,
) -> None:
    decision = DoubledProviderDecision(
        "mal", (), DoubledProviderOutcome.PAGES_OF_OTHER_WORKS, (review_reason,)
    )
    assert decision.is_review_needed


def test_write_member_name_gives_member_name() -> None:
    assert write_member_name(DoubledProviderOutcome.SAME_PAGE) == "SAME_PAGE"


@pytest.mark.parametrize(
    "stored_value", ["SAME_PAGE", DoubledProviderOutcome.SAME_PAGE]
)
def test_read_member_by_name_name_or_member_gives_member(stored_value) -> None:
    read_outcome = read_member_by_name(DoubledProviderOutcome)
    assert read_outcome(stored_value) is DoubledProviderOutcome.SAME_PAGE


def test_read_member_by_name_sentence_raises_value_error() -> None:
    read_outcome = read_member_by_name(DoubledProviderOutcome)
    with pytest.raises(ValueError, match="SAME_PAGE"):
        read_outcome(DoubledProviderOutcome.SAME_PAGE.value + " (reworded)")


@pytest.mark.parametrize(
    ("stored_enum", "member_names"),
    [
        (
            DoubledProviderOutcome,
            [
                "SAME_PAGE",
                "DUPLICATE_PAGES",
                "PART_PAGES",
                "PAGES_OF_OTHER_WORKS",
                "SEPARATE_WORKS",
                "UNDECIDED",
            ],
        ),
        (
            DoubledProviderReviewReason,
            [
                "CURRENT_ADDRESS_MISSING",
                "NO_PAGE_LINKED_WITH_ENTRY",
                "MOVE_NOT_SURE",
                "DETAILS_DIFFER",
                "CONTRADICTING_LINKS",
                "UNLINKED_PAGES",
                "OVERLAPPING_LINKS",
                "MULTI_WORK_LINK",
                "DEAD_PAGE",
            ],
        ),
        (
            LinkedPageReason,
            [
                "CHECKS_PASSED",
                "NOT_FOUND",
                "HELD_UNDER_OTHER_ADDRESS",
                "SEVERAL_ENTRIES",
                "PROVIDER_IN_ENTRY",
                "SEVERAL_PAGES_OF_PROVIDER",
                "DETAILS_DIFFER",
                "EPISODE_COUNTS_DIFFER",
            ],
        ),
    ],
)
def test_read_member_by_name_stored_enum_names_unchanged(
    stored_enum, member_names
) -> None:
    assert [member.name for member in stored_enum] == member_names
