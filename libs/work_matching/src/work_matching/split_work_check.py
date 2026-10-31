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
- ``NO_PAGE_CHECKED``: no page of the entries could be fetched, so only what
  the seed holds is known.
- ``LINK_SHARED_WITH_OTHER_ENTRIES``: a page tying the entries is also linked
  from entries outside them, so it may cover several works and its link is
  weaker evidence.
- ``LINKS_MAL_PAGE_OUTSIDE_GROUP``: a page of the entries links a MAL page no
  entry of the group holds. The work the entries belong to may be missing from
  the case, such as a series whose per-season MAL pages were deleted and
  merged into one entry the links no longer reach.

Every reason that applies is listed, not only the first, so a reviewer sees
every problem at once.
"""

from collections.abc import Iterable, Sequence
from enum import StrEnum
from itertools import chain

from work_matching.provider_page import ProviderPage
from work_matching.provider_page_key import get_page_provider
from work_matching.provider_page_links import ProviderPageLinks
from work_matching.work_entry import WorkEntry, is_type_matching, is_year_matching

# A work is what MAL lists, so a linked MAL page outside the group may be the
# entries' work.
_MAL_PROVIDER = "mal"


class SplitWorkReviewReason(StrEnum):
    """What speaks against candidate entries being one work, or weakens the evidence."""

    NOT_TIED_BY_LINKS = "links do not connect every entry"
    TYPE_DIFFERS = "the entries' types differ"
    YEAR_DIFFERS = "the entries' years differ"
    NO_PAGE_CHECKED = (
        "no page of the entries could be fetched, so only what the seed holds is known"
    )
    LINK_SHARED_WITH_OTHER_ENTRIES = (
        "a page tying the entries is also linked from entries outside the group, "
        "so it may cover several works"
    )
    LINKS_MAL_PAGE_OUTSIDE_GROUP = (
        "a page links a MAL page no entry of the group holds, so the work may be "
        "missing from the case"
    )


def list_split_work_review_reasons(
    entries: Sequence[WorkEntry],
    pages: Iterable[ProviderPage],
    *,
    has_shared_tying_link: bool = False,
) -> tuple[SplitWorkReviewReason, ...]:
    """List what speaks against candidate entries being one split work.

    Args:
        entries: The candidate entries, such as entries sharing a title.
        pages: Details of the entries' fetched pages.
        has_shared_tying_link: Whether a page tying the entries is also linked
            from entries outside them. Only the caller sees every entry's links,
            so the caller tells.

    Returns:
        Every review reason that applies, in ``SplitWorkReviewReason`` order;
        empty when nothing does or fewer than two entries are given.
    """
    if len(entries) < 2:
        return ()
    pages = tuple(pages)
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
        SplitWorkReviewReason.NO_PAGE_CHECKED: all(page.is_dead for page in pages),
        SplitWorkReviewReason.LINK_SHARED_WITH_OTHER_ENTRIES: has_shared_tying_link,
        SplitWorkReviewReason.LINKS_MAL_PAGE_OUTSIDE_GROUP: _has_mal_link_outside_group(
            entries, pages
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


def _has_mal_link_outside_group(
    entries: Sequence[WorkEntry], pages: Sequence[ProviderPage]
) -> bool:
    """Report whether a page links a MAL page none of the entries holds."""
    held_page_keys = {key for entry in entries for key in entry.addresses_by_page_key}
    return any(
        get_page_provider(linked_page_key) == _MAL_PROVIDER
        and linked_page_key not in held_page_keys
        for page in pages
        for linked_page_key in page.linked_page_keys
    )
