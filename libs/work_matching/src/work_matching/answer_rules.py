"""Define the validation rules for review answers and their user-facing messages.

The checks in ``page_placement``, ``entry_group`` and ``linked_page_check`` report
every rule an answer breaks, not only whether the answer is valid. A run and the
review page read answers with the same checks, so the page never accepts an
answer the run rejects, and shows the reviewer why an answer is refused.

Each rule's value is the sentence shown to the reviewer. The names are kept
in the run results, so the names stay stable when the wording changes.
"""

from enum import StrEnum


class PlacementRule(StrEnum):
    """Define a validation rule that an entry group's placement answer must satisfy."""

    INVALID_PLACEMENT_MAP = "the answer must map each page key to a placement"
    INVALID_PAGE_PLACEMENT = (
        "a placement must be 'work N', 'work N, main', 'work N, part', "
        "'other work', or 'other work <page_key>'"
    )
    PAGE_NOT_IN_CASE = "the answer includes a page that does not belong to this case"
    MISSING_REQUIRED_PLACEMENT = "every required page must be assigned to a work"
    WORK_WITH_TWO_MAL_PAGES = "each work can include at most one MAL page"
    SINGLE_PAGE_MARKED_MAIN = "a provider's only page in a work cannot be marked 'main'"
    INVALID_PROVIDER_PAGE_MARKS = (
        "when a work has multiple pages from one provider, mark all of them "
        "'part' or mark one 'main' and leave the rest unmarked"
    )
    NAMED_ENTRY_IN_CASE = (
        "the target entry specified after 'other work' is already part of "
        "this case; assign the page to 'work N' instead"
    )
    UNKNOWN_NAMED_ENTRY = (
        "no seed entry holds the page key specified after 'other work'"
    )
    WORK_WITHOUT_REFERENCE_PAGE = (
        "a work combining pages from multiple entries must include at least "
        "one entry's reference page"
    )


class LinkedPageAnswerRule(StrEnum):
    """Define a validation rule that a linked-page review answer must satisfy."""

    WRONG_LINKED_PAGE_KEY = (
        "the answer must map only the linked page's key to a decision"
    )
    INVALID_LINKED_PAGE_ANSWER = (
        "the decision must be 'add', 'add as part', 'add to <entry>', "
        "'add as part to <entry>', or 'skip'"
    )
    MISSING_TARGET_ENTRY = (
        "multiple entries link to this page; specify the target entry with 'to <entry>'"
    )
    TARGET_ENTRY_NOT_LINKING = "the specified target entry does not link to this page"
    PROVIDER_ALREADY_IN_ENTRY = (
        "the target entry already has a page from this provider, so this page "
        "can only be added with 'add as part'"
    )
