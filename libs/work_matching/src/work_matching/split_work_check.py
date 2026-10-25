"""Check what speaks against separate seed entries being one split work.

A split work is one work the seed lists as two or more entries. Every group of
such candidate entries goes to a person, because a shared title alone never
proves one work: remakes, music videos and short films reuse titles. The checks
only tell the reviewer what speaks against merging the group.

Each item reads ``<review reason>: <what it means>``.

- ``NOT_TIED_BY_LINKS``: links between the entries' pages do not connect every
  entry. A page two entries both list does not tie them, since it may be a page
  wrongly added to one of them.
- ``TYPE_DIFFERS``: the entries' types differ.
- ``YEAR_DIFFERS``: the entries' years differ.

Every reason that applies is listed, not only the first, so a reviewer sees
every problem at once.
"""

from collections.abc import Iterable, Sequence
from enum import StrEnum
from itertools import chain

from work_matching.provider_page import ProviderPage
from work_matching.provider_page_links import ProviderPageLinks
from work_matching.work_entry import WorkEntry, is_type_matching, is_year_matching


class SplitWorkReviewReason(StrEnum):
    """What speaks against candidate entries being one work."""

    NOT_TIED_BY_LINKS = "links do not connect every entry"
    TYPE_DIFFERS = "the entries' types differ"
    YEAR_DIFFERS = "the entries' years differ"


def list_split_work_review_reasons(
    entries: Sequence[WorkEntry], pages: Iterable[ProviderPage]
) -> tuple[SplitWorkReviewReason, ...]:
    """List what speaks against candidate entries being one split work.

    Args:
        entries: The candidate entries, such as entries sharing a title.
        pages: Details of the entries' fetched pages.

    Returns:
        Every review reason that applies, in ``SplitWorkReviewReason`` order;
        empty when nothing does or fewer than two entries are given.
    """
    if len(entries) < 2:
        return ()
    links = ProviderPageLinks(
        pages, chain.from_iterable(entry.addresses_by_page_key for entry in entries)
    )
    reason_applies = {
        SplitWorkReviewReason.NOT_TIED_BY_LINKS: not _is_tied_together(
            [
                frozenset(map(links.get_current_key, entry.addresses_by_page_key))
                for entry in entries
            ],
            links,
        ),
        SplitWorkReviewReason.TYPE_DIFFERS: not is_type_matching(
            *(entry.type for entry in entries)
        ),
        SplitWorkReviewReason.YEAR_DIFFERS: not is_year_matching(
            *(entry.year for entry in entries)
        ),
    }
    return tuple(reason for reason in SplitWorkReviewReason if reason_applies[reason])


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

    Returns:
        True when the walk reaches every entry.
    """
    reached_entries = {0}
    entries_to_visit = [0]
    while entries_to_visit:
        linked_keys = frozenset().union(
            *map(links.get_linked_page_keys, entry_page_keys[entries_to_visit.pop()])
        )
        for entry_index, page_keys in enumerate(entry_page_keys):
            if entry_index not in reached_entries and linked_keys & page_keys:
                reached_entries.add(entry_index)
                entries_to_visit.append(entry_index)
    return len(reached_entries) == len(entry_page_keys)
