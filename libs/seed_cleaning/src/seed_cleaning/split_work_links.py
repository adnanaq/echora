"""Find seed entries tied by links, so a split work with different titles is found.

Seasons and parts often carry titles of their own ("Aware! Meisaku-kun 2"), so
the title lookup misses them. A page of such an entry usually links the page
of the work's main entry, so links find what titles do not.

Each item gives the rule, then the reason for it.

- Two entries are tied when a fetched page of one, or a page searched by one of
  its pages, links a page the other holds. Searched pages count because their
  links, such as AnimeSchedule's, name the anime they were found for.
- Entries that each hold their own MAL page are never tied. A work is what MAL
  lists, so two MAL pages are two works, and a review would always answer
  "separate".
- A page linking two or more pages of one provider ties nothing. Such a page
  covers several works, such as an anthology, so its links do not show one work.
- A page linked from two or more entries that each hold their own MAL page ties
  nothing. Such a page covers several works, such as an AniDB page holding a
  series and its specials, so tying through it would join separate works.
- A dead page ties nothing. Its links cannot be read.

Which entries link each page is kept as well, so a review case can warn when a
page tying its entries is also linked from entries outside it. Entries holding
their own MAL page are recorded too, though they tie nothing: a page linked
from another work's entry may cover several works.
"""

from collections import Counter, defaultdict
from typing import NamedTuple

from work_matching import (
    KnownWorkEntriesFromSeed,
    ProviderPageStore,
    WorkEntry,
    get_page_provider,
)

# Entries that each hold a page of this provider are never one work.
_MAL_PROVIDER = "mal"


class SplitWorkLink(NamedTuple):
    """One link tying two entries: a page of one links a page the other holds."""

    linking_page_key: str
    linked_page_key: str


class SplitWorkLinks(NamedTuple):
    """The links tying seed entries, by the entries' places in the seed."""

    # Each tied pair of entries, by their places in the seed (the lower first),
    # mapped to every link tying them.
    links_by_entry_pair: dict[tuple[int, int], tuple[SplitWorkLink, ...]]
    # Each linked page that ties entries, mapped to the places of every entry
    # whose pages link it.
    linking_entries_by_page_key: dict[str, frozenset[int]]
    # Each page linked from two or more entries holding their own MAL page,
    # which covers several works, mapped to the places of those entries.
    mal_entries_by_several_works_page_key: dict[str, frozenset[int]] = {}


def find_split_work_links(
    known_work_entries: KnownWorkEntriesFromSeed, store: ProviderPageStore
) -> SplitWorkLinks:
    """Find every pair of entries tied by a link, by the rules in the module docstring.

    Args:
        known_work_entries: The entries to tie, such as seed entries as the
            corrections leave them; places are their iteration order.
        store: The fetched pages, including pages searched by them.

    Returns:
        The tied pairs with their links, the entries linking each page that
        ties a pair, and the MAL entries linking each page that covers
        several works.
    """
    work_entries = list(known_work_entries)
    holds_mal_page = [_has_mal_page(work_entry) for work_entry in work_entries]
    entry_links = _find_entry_links(known_work_entries, work_entries, store)
    linking_entries_by_page_key: defaultdict[str, set[int]] = defaultdict(set)
    for position, _, link in entry_links:
        linking_entries_by_page_key[link.linked_page_key].add(position)
    mal_entries_by_several_works_page_key = _find_several_works_pages(
        linking_entries_by_page_key, holds_mal_page
    )
    links_by_entry_pair: defaultdict[tuple[int, int], list[SplitWorkLink]] = (
        defaultdict(list)
    )
    for position, linked_position, link in entry_links:
        if link.linked_page_key in mal_entries_by_several_works_page_key or (
            holds_mal_page[position] and holds_mal_page[linked_position]
        ):
            continue
        pair = (min(position, linked_position), max(position, linked_position))
        links_by_entry_pair[pair].append(link)
    return SplitWorkLinks(
        {
            pair: tuple(dict.fromkeys(links))
            for pair, links in links_by_entry_pair.items()
        },
        {
            key: frozenset(positions)
            for key, positions in linking_entries_by_page_key.items()
        },
        mal_entries_by_several_works_page_key,
    )


def _find_entry_links(
    known_work_entries: KnownWorkEntriesFromSeed,
    work_entries: list[WorkEntry],
    store: ProviderPageStore,
) -> list[tuple[int, int, SplitWorkLink]]:
    """List every link from an entry's live pages to a page another entry holds.

    Args:
        known_work_entries: The lookup of entries by page.
        work_entries: The entries, by place.
        store: The fetched pages.

    Returns:
        Each link as the linking entry's place, the linked entry's place and
        the link, before any rule on MAL pages is applied.
    """
    seed_position_by_entry_id = {
        id(work_entry): position for position, work_entry in enumerate(work_entries)
    }
    entry_links: list[tuple[int, int, SplitWorkLink]] = []
    for position, work_entry in enumerate(work_entries):
        for page in store.get_entry_pages(work_entry):
            if page.is_dead:
                continue
            for linked_page_key in _read_single_work_links(page.linked_page_keys):
                linked_entry = known_work_entries.find_by_page_key(linked_page_key)
                if linked_entry is None or linked_entry is work_entry:
                    continue
                entry_links.append(
                    (
                        position,
                        seed_position_by_entry_id[id(linked_entry)],
                        SplitWorkLink(page.current_key, linked_page_key),
                    )
                )
    return entry_links


def _find_several_works_pages(
    linking_entries_by_page_key: dict[str, set[int]], holds_mal_page: list[bool]
) -> dict[str, frozenset[int]]:
    """Return each page two or more entries holding a MAL page link, with those entries."""
    mal_entries_by_page_key = {
        page_key: frozenset(
            position for position in positions if holds_mal_page[position]
        )
        for page_key, positions in linking_entries_by_page_key.items()
    }
    return {
        page_key: positions
        for page_key, positions in mal_entries_by_page_key.items()
        if len(positions) > 1
    }


def _read_single_work_links(linked_page_keys: tuple[str, ...]) -> list[str]:
    """Return the linked pages a page names one per provider.

    A provider with two or more linked pages means the page covers several
    works, so none of that provider's links is kept.
    """
    linked_pages_per_provider = Counter(map(get_page_provider, linked_page_keys))
    return [
        key
        for key in linked_page_keys
        if linked_pages_per_provider[get_page_provider(key)] == 1
    ]


def _has_mal_page(work_entry: WorkEntry) -> bool:
    """Report whether an entry holds a MAL page."""
    return any(
        get_page_provider(key) == _MAL_PROVIDER
        for key in work_entry.addresses_by_page_key
    )
