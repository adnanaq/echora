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

    LINKED = "links connect every entry, and their types and years agree"
    TOO_FEW_ENTRIES = "fewer than two entries were given, so there is nothing to join"
    NOT_LINKED = "links do not connect every entry"
    TYPE_DIFFERS = "links connect every entry, but their types differ"
    YEAR_DIFFERS = (
        "links connect every entry and their types agree, but their years differ"
    )


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
        [frozenset(map(links.get_current_key, entry.pages)) for entry in entries], links
    ):
        return SplitWorkDecision(entries, SplitWorkReason.NOT_LINKED)
    if not is_type_matching(*(entry.type for entry in entries)):
        return SplitWorkDecision(entries, SplitWorkReason.TYPE_DIFFERS)
    if not is_year_matching(*(entry.year for entry in entries)):
        return SplitWorkDecision(entries, SplitWorkReason.YEAR_DIFFERS)
    return SplitWorkDecision(entries, SplitWorkReason.LINKED)


def _is_tied_together(
    entry_page_keys: list[frozenset[str]], links: ProviderPageLinks
) -> bool:
    """Report whether links connect every candidate entry, starting from the first.

    The walk moves from an entry to every entry whose pages are linked with its
    pages, until no new entry is reached. A page both entries list does not tie
    them: it may be a page wrongly added to one of them, so only a link counts.

    Args:
        entry_page_keys: Each candidate entry's current page keys.
        links: The links between all the candidate entries' pages.
    """
    reached_entries = {0}
    entries_to_visit = [0]
    while entries_to_visit:
        linked_keys = frozenset().union(
            *map(links.get_linked_pages, entry_page_keys[entries_to_visit.pop()])
        )
        for entry_index, page_keys in enumerate(entry_page_keys):
            if entry_index not in reached_entries and linked_keys & page_keys:
                reached_entries.add(entry_index)
                entries_to_visit.append(entry_index)
    return len(reached_entries) == len(entry_page_keys)
