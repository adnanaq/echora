"""Decide whether separate seed entries are one split work.

Entries are one work only when links tie every entry to the others and their
types and years agree. A shared title alone is never enough: remakes, music
videos and short films reuse titles.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import chain

from work_matching.provider_page import ProviderPage
from work_matching.provider_page_links import ProviderPageLinks
from work_matching.work_entry import WorkEntry, is_type_matching, is_year_matching


class SplitWorkReason(StrEnum):
    """Why candidate entries are or are not one work."""

    LINKED = "links tie every entry together and type and year agree"
    TOO_FEW_ENTRIES = "fewer than two entries, so there is nothing to join"
    NOT_LINKED = "links do not tie every entry together"
    TYPE_DIFFERS = "the entries' types differ"
    YEAR_DIFFERS = "the entries' years differ"


@dataclass(frozen=True, slots=True)
class SplitWorkDecision:
    """Whether candidate entries are one work, and why."""

    entries: tuple[WorkEntry, ...]
    reason: SplitWorkReason

    @property
    def is_one_work(self) -> bool:
        """Whether the entries are one work; otherwise they go to review."""
        return self.reason is SplitWorkReason.LINKED


def decide_split_work(
    entries: Sequence[WorkEntry], pages: Iterable[ProviderPage]
) -> SplitWorkDecision:
    """Decide whether candidate entries are one split work.

    Args:
        entries: The candidate entries, such as entries sharing a title.
        pages: Details of the entries' fetched pages.

    Returns:
        The decision, with the first check that failed as its reason: links
        first, then type, then year.
    """
    entries = tuple(entries)
    if len(entries) < 2:
        return SplitWorkDecision(entries, SplitWorkReason.TOO_FEW_ENTRIES)
    links = ProviderPageLinks(
        pages, chain.from_iterable(entry.pages for entry in entries)
    )
    if not _is_tied_together(
        [frozenset(map(links.same_page, entry.pages)) for entry in entries], links
    ):
        return SplitWorkDecision(entries, SplitWorkReason.NOT_LINKED)
    if not is_type_matching(*(entry.type for entry in entries)):
        return SplitWorkDecision(entries, SplitWorkReason.TYPE_DIFFERS)
    if not is_year_matching(*(entry.year for entry in entries)):
        return SplitWorkDecision(entries, SplitWorkReason.YEAR_DIFFERS)
    return SplitWorkDecision(entries, SplitWorkReason.LINKED)


def _is_tied_together(
    page_sets: list[frozenset[str]], links: ProviderPageLinks
) -> bool:
    """Report whether links reach every page set from the first one.

    Args:
        page_sets: Each entry's page keys.
        links: The links between all the entries' pages.
    """
    reached = {0}
    to_visit = [0]
    while to_visit:
        linked = frozenset().union(*map(links.linked_with, page_sets[to_visit.pop()]))
        for index, pages in enumerate(page_sets):
            if index not in reached and linked & pages:
                reached.add(index)
                to_visit.append(index)
    return len(reached) == len(page_sets)
