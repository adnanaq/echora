from pathlib import Path

import pytest
from work_matching.errors import (
    DuplicateCaseError,
    NoLinkingEntryError,
    ReviewFileLockedError,
    SavedAnswerChangedError,
    UnknownCaseError,
    UnknownMemberNameError,
    WorkMatchingError,
)
from work_matching.models import DoubledProviderOutcome


@pytest.mark.parametrize(
    ("error_type", "expected_prefix"),
    [
        (DuplicateCaseError, "Review cases given more than once"),
        (UnknownCaseError, "Review cases not in the review file"),
        (
            SavedAnswerChangedError,
            "Saved answers changed since their edits began",
        ),
    ],
)
def test_review_case_names_error_stores_case_names_and_formats_message(
    error_type, expected_prefix
) -> None:
    error = error_type(["split work: mal:anime:1", "linked page: mal:anime:2"])
    assert isinstance(error, WorkMatchingError)
    assert error.case_names == (
        "split work: mal:anime:1",
        "linked page: mal:anime:2",
    )
    assert (
        str(error)
        == f"{expected_prefix}: split work: mal:anime:1, linked page: mal:anime:2"
    )


def test_review_file_locked_error_names_review_file_path() -> None:
    error = ReviewFileLockedError(Path("data/seed_cleaning/review.jsonl"))
    assert isinstance(error, WorkMatchingError)
    assert (
        str(error)
        == "Another process is writing the review file: data/seed_cleaning/review.jsonl"
    )


def test_no_linking_entry_error_names_page_key() -> None:
    error = NoLinkingEntryError("mal:anime:65144")
    assert isinstance(error, WorkMatchingError)
    assert str(error) == "No entry links the page to decide: mal:anime:65144"


def test_unknown_member_name_error_lists_expected_member_names() -> None:
    error = UnknownMemberNameError("WRONG_OUTCOME", DoubledProviderOutcome)
    assert isinstance(error, (WorkMatchingError, ValueError))
    assert str(error) == (
        "'WRONG_OUTCOME' is not a DoubledProviderOutcome name; "
        "expected one of SAME_PAGE, DUPLICATE_PAGES, PART_PAGES, "
        "PAGES_OF_OTHER_WORKS, SEPARATE_WORKS, UNDECIDED"
    )
