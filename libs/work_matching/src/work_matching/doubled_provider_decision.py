"""What a doubled provider's pages turned out to be, and whether a person must check it.

A decision covers one doubled provider: that provider's pages in one entry. A
decision holds two separate answers:

| Field | Answers |
| -- | -- |
| ``outcome`` | What the provider's pages are, relative to the entry. |
| ``review_reasons`` | Why a person must check the decision; empty when nothing stops the rules. |

The rules apply a decision without a person only when the outcome is
``SAME_PAGE`` or ``PAGES_OF_OTHER_WORKS`` and no review reason applies. Every
other outcome needs a person even without a review reason:

| Outcome | Why a person is needed |
| -- | -- |
| ``DUPLICATE_PAGES`` | One of the pages has to be named the main page. |
| ``PART_PAGES`` | Air dates and episode counts only suggest parts; they never decide. |
| ``SEPARATE_WORKS`` | Every page of the entry has to be placed in one of the works. |
| ``UNDECIDED`` | The links do not place every page. |
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Annotated

from pydantic import PlainSerializer
from work_matching.work_entry import WorkEntry


class Outcome(StrEnum):
    """Name what a doubled provider's pages are, relative to the entry holding them."""

    # Every address resolves to one current key: an old address leads to the
    # page's current address, so the provider is not really doubled.
    SAME_PAGE = (
        "the addresses are one page: an old address leads to the page's current address"
    )
    # Every page is linked with the entry's work, and their air dates and
    # episode counts agree: the provider lists the whole work more than once.
    DUPLICATE_PAGES = (
        "every page belongs to the entry's work and covers all of it: the provider "
        "lists the work more than once"
    )
    # Every page is linked with the entry's work, but their air dates or
    # episode counts differ: each page covers part of the work, such as one of
    # two OVAs or one season.
    PART_PAGES = (
        "every page belongs to the entry's work, but their air dates or episode "
        "counts differ, so each page covers part of the work"
    )
    # At most one page is linked with the entry's work; every other page links
    # a work outside the entry and moves out to that work.
    PAGES_OF_OTHER_WORKS = (
        "at most one page belongs to the entry's work; every other page belongs to "
        "a work outside the entry"
    )
    # Each page is linked with its own, separate set of the entry's pages: the
    # entry holds several works, and is split into one entry per work.
    SEPARATE_WORKS = (
        "each page is linked with a separate set of the entry's pages, so the entry "
        "holds several works"
    )
    # The links do not tell which work every page belongs to.
    UNDECIDED = "the links do not tell which work every page belongs to"


class ReviewReason(StrEnum):
    """Name why a person must check a doubled-provider decision."""

    CURRENT_ADDRESS_MISSING = (
        "the entry lists only old addresses of the page, so dropping them would "
        "lose the page"
    )
    NO_PAGE_LINKED_WITH_ENTRY = (
        "no page is linked with the entry, so the entry would lose every page of "
        "the provider"
    )
    MOVE_NOT_SURE = (
        "a page of another work has no sure place to go: its links name no seed "
        "entry or several, that entry already holds a page of the provider, or "
        "their year or type differ"
    )
    DETAILS_DIFFER = "the pages belong to one work, but their years or types differ"
    CONTRADICTING_LINKS = (
        "a page is linked with the entry and also links a different work"
    )
    UNLINKED_PAGES = "a page has no link to the entry and no link to another work"
    OVERLAPPING_LINKS = (
        "the pages share some linked pages, but no page is linked with all of them"
    )
    MULTI_WORK_LINK = "a page naming several pages of one provider links these pages"
    DEAD_PAGE = "a page no longer exists on the provider's site"


# Outcomes a person always checks, whatever the review reasons say.
_ALWAYS_REVIEWED_OUTCOMES = frozenset(
    {
        Outcome.DUPLICATE_PAGES,
        Outcome.PART_PAGES,
        Outcome.SEPARATE_WORKS,
        Outcome.UNDECIDED,
    }
)

# Page keys held as a set, written sorted: a set's order changes from one process
# to the next, and a review file written in a fixed order lets two runs be
# compared line by line.
PageKeySet = Annotated[
    frozenset[str], PlainSerializer(sorted, return_type=list[str], when_used="json")
]


@dataclass(frozen=True, slots=True)
class UnrelatedPage:
    """Hold what a doubled page's links say about the work it belongs to instead."""

    # Keys of the linked pages that show another work: pages of providers
    # already in the entry, or pages of known work entries.
    linked_page_keys: PageKeySet
    # The known work entries holding those linked pages.
    other_work_entries: tuple[WorkEntry, ...]
    # The seed entry the page moves to, when the move is sure; None otherwise.
    move_target_entry: WorkEntry | None = None


@dataclass(frozen=True, slots=True)
class DoubledProviderDecision:
    """Hold what one doubled provider's pages are, and why a person must check them."""

    # The provider decided; "simkl" for Simkl and AnimeCountdown together.
    provider: str
    # The provider's page keys, with renamed pages resolved to their current key.
    page_keys: tuple[str, ...]
    outcome: Outcome
    # Every reason a person must check the decision, in ``ReviewReason`` order.
    review_reasons: tuple[ReviewReason, ...] = ()
    # Each belonging page's key, mapped to the keys of the entry's pages linked
    # with it.
    belonging_page_keys: Mapping[str, PageKeySet] = field(default_factory=dict)
    # Each unrelated page's key, mapped to what its links say about other works.
    unrelated_pages: Mapping[str, UnrelatedPage] = field(default_factory=dict)
    # Keys of pages linked with the entry that also link a different work.
    contradicting_page_keys: tuple[str, ...] = ()
    # Keys of pages with no links either to the entry or to another work.
    unlinked_page_keys: tuple[str, ...] = ()
    # On review only: keys of pages matching the entry's title, year and type, or
    # season.
    title_matches: tuple[str, ...] = ()
    year_type_matches: tuple[str, ...] = ()
    season_matches: tuple[str, ...] = ()

    @property
    def needs_review(self) -> bool:
        """Report whether a person must check the decision before it is applied.

        Returns:
            True when a review reason applies, or when the outcome is one a
            person always checks.
        """
        return bool(self.review_reasons) or self.outcome in _ALWAYS_REVIEWED_OUTCOMES
