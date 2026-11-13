"""Group linked pages by the page each one is, with the entries linking it.

Links spell one page several ways, such as a Kitsu slug and the Kitsu number
the seed uses. Only the fetched page tells which spellings are one page, so
the linked pages are grouped by the current key their fetch reports before any
is decided.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from operator import itemgetter

from seed_cleaning.models import LinkingSeedEntry
from work_matching import ProviderPage, ProviderPageStore

# One linked page's details and every entry linking it, under any spelling.
type LinkedPageGroup = tuple[ProviderPage, tuple[LinkingSeedEntry, ...]]


def group_linked_pages_by_current_key(
    linking_seed_entries_by_page_key: Mapping[str, Sequence[LinkingSeedEntry]],
    store: ProviderPageStore,
) -> tuple[dict[str, LinkedPageGroup], tuple[str, ...]]:
    """Group the fetched linked pages by the current key their fetch reports.

    Links spell one page several ways, such as a Kitsu slug and the Kitsu
    number the seed uses. Only the fetched page tells which spellings are one
    page. Grouping applies two rules:

    - The spellings of one page are decided once, with every entry linking any
      of them.
    - Each entry's other linked pages are named by their current key too, so
      two spellings of one page never count as two pages of one provider.

    Args:
        linking_seed_entries_by_page_key: Each linked page's key mapped to the
            entries linking it.
        store: The fetched pages.

    Returns:
        Each current key, sorted, mapped to the page's details (stored at the
        current key when fetched there) and the entries linking it; and the
        sorted keys of linked pages not fetched yet.
    """
    fetched_pages = {
        page_key: page
        for page_key in linking_seed_entries_by_page_key
        if (page := store.get_page(page_key)) is not None
    }
    current_page_key_by_linked_page_key = {
        page_key: page.current_key for page_key, page in fetched_pages.items()
    }
    linking_seed_entries_by_current_key: defaultdict[
        str, dict[str, LinkingSeedEntry]
    ] = defaultdict(dict)
    for page_key, current_page_key in current_page_key_by_linked_page_key.items():
        for linking_seed_entry in linking_seed_entries_by_page_key[page_key]:
            linking_seed_entries_by_current_key[current_page_key].setdefault(
                linking_seed_entry.seed_entry_reference_page,
                rename_linked_pages_to_current_keys(
                    linking_seed_entry, current_page_key_by_linked_page_key
                ),
            )
    linked_page_groups_by_current_key = {
        current_page_key: (
            store.get_page(current_page_key) or fetched_pages[linked_page_key],
            tuple(linking_seed_entries_by_current_key[current_page_key].values()),
        )
        for linked_page_key, current_page_key in sorted(
            current_page_key_by_linked_page_key.items(), key=itemgetter(1)
        )
    }
    waiting_page_keys = tuple(
        sorted(
            page_key
            for page_key in linking_seed_entries_by_page_key
            if page_key not in fetched_pages
        )
    )
    return linked_page_groups_by_current_key, waiting_page_keys


def rename_linked_pages_to_current_keys(
    linking_seed_entry: LinkingSeedEntry,
    current_page_key_by_linked_page_key: Mapping[str, str],
) -> LinkingSeedEntry:
    """Rename a linking entry's other linked pages to their fetched current keys.

    Args:
        linking_seed_entry: An entry linking a page, with the linked pages as
            its links spell them.
        current_page_key_by_linked_page_key: Each fetched linked page's current
            key, by the key its link spells; pages not fetched keep their
            spelling.

    Returns:
        The same linking entry, its linked pages named by current key.
    """
    return replace(
        linking_seed_entry,
        entry_details=replace(
            linking_seed_entry.entry_details,
            linked_page_keys=frozenset(
                current_page_key_by_linked_page_key.get(
                    linked_page_key, linked_page_key
                )
                for linked_page_key in linking_seed_entry.entry_details.linked_page_keys
            ),
        ),
    )
