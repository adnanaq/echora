"""Name the words a review answer uses, and say what each one does.

A reviewer picks these words from the review page's dropdowns. The page shows
each word's meaning beside it, so the reviewer knows what a run will do with
the answer. The meanings repeat the rules of ``page_placement`` and
``linked_page_check`` in a reviewer's terms. Those modules decide what a
word does. Each item reads ``<word>: <meaning>``.

- ``work 1``: the page stays in the entry that survives as the case's first
  work.
- ``work 2`` and later: the page belongs to another work in the case. Every
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
from work_matching.page_placement import (
    FIRST_WORK,
    MAIN_MARK,
    NO_MARK,
    OTHER_WORK,
    PART_MARK,
)

_LATER_WORK: Final = re.compile(r"work ([2-9]|[1-9]\d+)")
_NAMED_ENTRY_ANSWER: Final = re.compile(
    rf"(?P<addition>{re.escape(PART_PAGE_ANSWER)}|{re.escape(WHOLE_PAGE_ANSWER)})"
    r" to (?P<reference_page>.+)"
)

# One meaning for work 2 and every later work, so a legend can list them once.
LATER_WORK_MEANING: Final = (
    "Assign the page to a separate work within this case. "
    "Pages with the same work number are grouped into one work, and using "
    "work 2 or higher requires placing every page in the case."
)

_MEANING_BY_WORD: Final[dict[str, str]] = {
    FIRST_WORK: (
        "Assign the page to the first work in this case. Pages in work 1 stay "
        "in the surviving entry for that work."
    ),
    OTHER_WORK: (
        "Move the page out of this case because it belongs to a different "
        "work. The page joins an existing seed entry if linked or selected, "
        "or otherwise starts a new entry."
    ),
    NO_MARK: (
        "Treat the page as a full-work page. Leave pages unmarked when a "
        "provider has only one page in the work, or when another page from "
        "the same provider is marked 'main'."
    ),
    MAIN_MARK: (
        "Mark this page as the primary page to fetch when a work keeps "
        "multiple full pages from the same provider. Leave the provider's "
        "other pages in the work unmarked."
    ),
    PART_MARK: (
        "Mark this page as covering only part of the work (such as one season "
        "or OVA). Enrichment combines all part pages for the work."
    ),
    WHOLE_PAGE_ANSWER: (
        "Add the linked page to the entry as a full-work page. Only allowed "
        "when the entry does not already have a page from this provider."
    ),
    PART_PAGE_ANSWER: (
        "Add the linked page as a partial page of the entry's work, even if "
        "the entry already has a page from this provider."
    ),
    SKIP_ANSWER: "Do not add the linked page. Leave the linking entry unchanged.",
}


# A page the answer leaves out has no place; the review page shows it as
# "no place".
NO_PLACE_MEANING: Final = (
    "Leave the page unassigned for now. Pages marked with * must be assigned "
    "before saving."
)

# Each placement word and its meaning, in the order a reviewer meets them, so
# a page can list them once under an entry group's pages.
PLACEMENT_WORD_MEANINGS: Final[tuple[tuple[str, str], ...]] = (
    (FIRST_WORK, _MEANING_BY_WORD[FIRST_WORK]),
    ("work 2, work 3, …", LATER_WORK_MEANING),
    (OTHER_WORK, _MEANING_BY_WORD[OTHER_WORK]),
    ("no place", NO_PLACE_MEANING),
    ("— (no mark)", _MEANING_BY_WORD[NO_MARK]),
    (MAIN_MARK, _MEANING_BY_WORD[MAIN_MARK]),
    (PART_MARK, _MEANING_BY_WORD[PART_MARK]),
)


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
        addition_phrase = (
            "as a partial page to"
            if named_entry_answer["addition"] == PART_PAGE_ANSWER
            else "to"
        )
        return (
            f"Add the linked page {addition_phrase} entry "
            f"{named_entry_answer['reference_page']} (chosen from the "
            "multiple entries that link this page)."
        )
    return ""
