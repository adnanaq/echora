"""The rules a review answer must follow, named so a reviewer is told which one broke.

The checks in ``page_placement``, ``entry_group`` and ``linked_page_check`` report
every rule an answer breaks, not only whether it is valid. A run and the
review page read answers with the same checks, so the page never accepts an
answer the run rejects, and it shows the reviewer why an answer is refused.

Each rule's value is the sentence shown to the reviewer. The names are kept
in the run results, so they stay when the wording changes.
"""

from enum import StrEnum


class PlacementRule(StrEnum):
    """A rule an entry group's answer, placing each page in a work, must follow."""

    ANSWER_NOT_PAGE_PLACES = "the answer must map each page key to its place"
    PLACE_NOT_READABLE = (
        "a place must be 'work N', 'work N, main', 'work N, part' or 'other work'"
    )
    PAGE_OUTSIDE_CASE = "the answer places a page this case does not hold"
    PAGE_WITHOUT_PLACE = "a page this answer must place has no place"
    WORK_WITH_TWO_MAL_PAGES = (
        "a work holds one MAL page, because a work is what MAL lists"
    )
    SINGLE_PAGE_MARKED_MAIN = (
        "a provider's only page in a work cannot be 'main', since there is "
        "nothing to choose between"
    )
    PROVIDER_PAGES_NOT_MARKED_ONCE = (
        "a provider's pages in one work must be all 'part', or one 'main' and "
        "the rest unmarked"
    )
    WORK_WITHOUT_REFERENCE_PAGE = (
        "a work holding pages of two or more entries must hold one of those "
        "entries' reference pages, so one entry survives as the work"
    )


class LinkedPageAnswerRule(StrEnum):
    """A rule an answer about a linked page must follow."""

    ANSWER_NOT_PAGE_ANSWER = (
        "the answer must map the linked page's key, alone, to its answer"
    )
    ANSWER_NOT_READABLE = (
        "an answer must be 'add', 'add as part', either one with 'to <entry>', "
        "or 'skip'"
    )
    ENTRY_NOT_NAMED = (
        "two or more entries link the page, so the answer must name one with "
        "'to <entry>'"
    )
    ENTRY_NOT_LINKING = "the entry the answer names does not link the page"
    PROVIDER_IN_ENTRY = (
        "the entry already holds a page of this provider, so the page can only "
        "be added as a part"
    )
