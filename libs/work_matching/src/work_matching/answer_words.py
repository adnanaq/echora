"""Name the words a review answer uses, and say what each one does.

A reviewer picks these words from the review page's dropdowns. The page shows
each word's meaning beside it, so the reviewer knows what a run will do with
the answer. The meanings repeat the rules of ``page_placement`` and
``linked_page_check`` in a reviewer's terms; those modules decide what a word
does. Each item reads ``<word>: <meaning>``.

- ``work 1``: the page stays in the entry that survives as the case's first
  work.
- ``work 2`` and later: the page belongs to another work in the case; every
  page then needs a place.
- ``other work``: the page belongs to a work outside the case.
- no mark, ``main``, ``part``: how the page stands beside the provider's other
  pages in its work.
- ``add``, ``add as part``, ``add to <reference page>``, ``add as part to
  <reference page>``, ``skip``: whether a linked page joins a linking entry.
"""

import re
from typing import Final

from work_matching.linked_page_check import (
    PART_PAGE_ANSWER,
    SKIP_ANSWER,
    WHOLE_PAGE_ANSWER,
)
from work_matching.page_placement import OTHER_WORK

# The marks a page's place can carry after ``work N``; no mark is written as
# nothing.
NO_MARK: Final = ""
MAIN_MARK: Final = "main"
PART_MARK: Final = "part"

_FIRST_WORK: Final = "work 1"
_LATER_WORK: Final = re.compile(r"work ([2-9]|[1-9]\d+)")
_NAMED_ENTRY_ANSWER: Final = re.compile(
    rf"(?P<addition>{re.escape(PART_PAGE_ANSWER)}|{re.escape(WHOLE_PAGE_ANSWER)})"
    r" to (?P<reference_page>.+)"
)

# One meaning for work 2 and every later work, so a legend can list them once.
LATER_WORK_MEANING: Final = (
    "The page belongs to another work in the case, not the first. Pages with "
    "the same number are one work, and an answer using work 2 or later places "
    "every page."
)

_MEANING_BY_WORD: Final[dict[str, str]] = {
    _FIRST_WORK: (
        "The page belongs to the case's first work. It stays in the entry that "
        "survives as that work."
    ),
    OTHER_WORK: (
        "The page belongs to a work outside this case. It joins that work's "
        "seed entry when the move is sure, and otherwise starts a new entry."
    ),
    NO_MARK: (
        "No mark: the page is the provider's only page in its work, or another "
        "of the provider's pages there is 'main'."
    ),
    MAIN_MARK: (
        "The work keeps two or more whole pages of this provider, and this is "
        "the one enrichment fetches. The provider's other pages there stay "
        "unmarked."
    ),
    PART_MARK: (
        "The page covers only part of the work, such as one of two OVAs. "
        "Enrichment combines it with the work's other part pages."
    ),
    WHOLE_PAGE_ANSWER: (
        "Add the page to the entry linking it. The entry must hold no page of "
        "the page's provider."
    ),
    PART_PAGE_ANSWER: (
        "Add the page as a part page of the linking entry's work. The entry may "
        "already hold pages of the provider."
    ),
    SKIP_ANSWER: "Leave the page out. The linking entry stays as it is.",
}


def describe_answer_word(answer_word: str) -> str:
    """Return what an answer word does, as the review page shows it.

    Args:
        answer_word: A work, a mark or a linked-page answer, as written in an
            answer: ``work 2``, ``part``, ``add as part to mal:anime:1``.

    Returns:
        The word's meaning, or an empty string for a word no rule knows.
    """
    if answer_word in _MEANING_BY_WORD:
        return _MEANING_BY_WORD[answer_word]
    if _LATER_WORK.fullmatch(answer_word):
        return LATER_WORK_MEANING
    if named_entry_answer := _NAMED_ENTRY_ANSWER.fullmatch(answer_word):
        what_is_added = (
            "as a part page of the work of"
            if named_entry_answer["addition"] == PART_PAGE_ANSWER
            else "to"
        )
        return (
            f"Add the page {what_is_added} the entry whose reference page is "
            f"{named_entry_answer['reference_page']}. Several entries link the "
            "page, so the answer names one."
        )
    return ""


# A page the answer leaves out has no place; the review page shows it as
# "no place".
NO_PLACE_MEANING: Final = (
    "The page has no place yet. A page marked * needs one before the answer "
    "can be saved."
)

# Each placement word and its meaning, in the order a reviewer meets them, so
# a page can list them once under an entry group's pages.
PLACEMENT_WORD_MEANINGS: Final[tuple[tuple[str, str], ...]] = (
    (_FIRST_WORK, _MEANING_BY_WORD[_FIRST_WORK]),
    ("work 2, work 3, …", LATER_WORK_MEANING),
    (OTHER_WORK, _MEANING_BY_WORD[OTHER_WORK]),
    ("no place", NO_PLACE_MEANING),
    ("— (no mark)", _MEANING_BY_WORD[NO_MARK]),
    (MAIN_MARK, _MEANING_BY_WORD[MAIN_MARK]),
    (PART_MARK, _MEANING_BY_WORD[PART_MARK]),
)
