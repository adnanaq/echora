"""Find which pages of a doubled provider match the entry, as hints for a reviewer.

A decision a person must check carries three lists of matching pages, so the
reviewer sees at once which pages agree with the entry. The matches never
decide the outcome, because providers listing one work twice often vary the
titles, and details only tell duplicate pages from part pages.

Each item reads ``<match>: <what agrees>``.

- Title: the page's main title equals the entry's title or the main title of
  one of the entry's pages from another provider, once folded for comparison.
- Year and type: the page's year and seed type agree with the entry's.
- Season: the page airs in the entry's season. An entry without a season has
  no season matches.
"""

from collections.abc import Collection, Iterable, Mapping

from work_matching.comparable_title import fold_title_for_comparison
from work_matching.doubled_provider_pages import resolve_decided_provider
from work_matching.page_details_comparison import is_page_details_matching
from work_matching.provider_page import ProviderPage
from work_matching.work_entry import WorkEntry


def find_fetched_pages(
    page_keys: Iterable[str], pages_by_current_page_key: Mapping[str, ProviderPage]
) -> dict[str, ProviderPage]:
    """Find the fetched details of these pages, leaving out pages never fetched.

    Args:
        page_keys: The current keys of the pages to look up.
        pages_by_current_page_key: The fetched pages, by their current keys.

    Returns:
        Each fetched page by its key, in the order given.
    """
    return {
        page_key: page
        for page_key in page_keys
        if (page := pages_by_current_page_key.get(page_key))
    }


def find_title_matches(
    entry: WorkEntry,
    entry_page_keys: Collection[str],
    provider: str,
    page_keys: tuple[str, ...],
    pages_by_current_page_key: Mapping[str, ProviderPage],
) -> tuple[str, ...]:
    """Return the keys of the pages whose main title matches the entry's titles.

    The entry's titles are its own title and the main titles of its pages from
    other providers.

    Args:
        entry: The work entry.
        entry_page_keys: The current keys of the entry's pages.
        provider: The doubled provider being decided.
        page_keys: The current keys of the provider's pages.
        pages_by_current_page_key: The fetched pages, by their current keys.

    Returns:
        The keys of the fetched pages whose main title matches the entry.
    """
    other_provider_page_keys = [
        page_key
        for page_key in entry_page_keys
        if resolve_decided_provider(page_key) != provider
    ]
    entry_titles = {fold_title_for_comparison(entry.title)} | {
        fold_title_for_comparison(page.titles[0])
        for page in find_fetched_pages(
            other_provider_page_keys, pages_by_current_page_key
        ).values()
        if page.titles
    }
    return tuple(
        page_key
        for page_key, page in find_fetched_pages(
            page_keys, pages_by_current_page_key
        ).items()
        if page.titles and fold_title_for_comparison(page.titles[0]) in entry_titles
    )


def find_year_type_matches(
    entry: WorkEntry,
    page_keys: tuple[str, ...],
    pages_by_current_page_key: Mapping[str, ProviderPage],
) -> tuple[str, ...]:
    """Return the keys of the pages whose year and type agree with the entry's.

    Args:
        entry: The work entry.
        page_keys: The current keys of one doubled provider's pages.
        pages_by_current_page_key: The fetched pages, by their current keys.

    Returns:
        The keys of the fetched pages that agree, in the order given.
    """
    return tuple(
        page_key
        for page_key, page in find_fetched_pages(
            page_keys, pages_by_current_page_key
        ).items()
        if is_page_details_matching(page, entry)
    )


def find_season_matches(
    entry: WorkEntry,
    page_keys: tuple[str, ...],
    pages_by_current_page_key: Mapping[str, ProviderPage],
) -> tuple[str, ...]:
    """Return the keys of the pages in the entry's season, or empty without a season.

    Args:
        entry: The work entry.
        page_keys: The current keys of one doubled provider's pages.
        pages_by_current_page_key: The fetched pages, by their current keys.

    Returns:
        The keys of the fetched pages matching the entry's season.
    """
    if not entry.season:
        return ()
    return tuple(
        page_key
        for page_key, page in find_fetched_pages(
            page_keys, pages_by_current_page_key
        ).items()
        if page.season == entry.season
    )
