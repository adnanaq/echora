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

from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping

from work_matching.comparable_title import fold_title_for_comparison
from work_matching.page_details_comparison import is_page_details_matching
from work_matching.provider_page import ProviderPage
from work_matching.provider_page_key import read_page_id, read_page_provider
from work_matching.work_entry import WorkEntry

# Simkl and AnimeCountdown number a work the same, so they are decided together,
# under Simkl's name: a Simkl and an AnimeCountdown page with different numbers
# are two works, just as two Simkl pages are.
_PROVIDER_DECIDED_UNDER = {"animecountdown": "simkl"}


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


def find_doubled_provider_page_keys(
    page_keys: Iterable[str],
) -> dict[str, tuple[str, ...]]:
    """Return the page keys of each doubled provider among these page keys.

    A provider is doubled when its pages carry more than one identifier. Pages
    of providers decided together count as one provider under
    ``resolve_decided_provider``. A Simkl page and an AnimeCountdown page with
    the same number therefore count as a single page rather than a doubled
    provider. Seed cleaning uses this function so it checks the exact pages
    that ``decide_doubled_providers`` evaluates.

    Args:
        page_keys: Page keys, such as the keys of an entry's pages.

    Returns:
        Each doubled provider, by the name its decision falls under, with its
        page keys in the order given.
    """
    page_keys_by_provider: defaultdict[str, list[str]] = defaultdict(list)
    for page_key in page_keys:
        if provider := resolve_decided_provider(page_key):
            page_keys_by_provider[provider].append(page_key)
    return {
        provider: tuple(provider_page_keys)
        for provider, provider_page_keys in page_keys_by_provider.items()
        if has_several_page_ids(provider_page_keys)
    }


def resolve_decided_provider(page_key: str) -> str | None:
    """Resolve the provider name under which a page key is decided.

    AnimeCountdown numbers works identically to Simkl, so AnimeCountdown pages
    are grouped under ``simkl``.

    Args:
        page_key: A page key, as ``resolve_page_key`` gives it.

    Returns:
        The provider name, or None when the key is not a known provider's page.
    """
    provider = read_page_provider(page_key)
    return _PROVIDER_DECIDED_UNDER.get(provider, provider) if provider else None


def has_several_page_ids(page_keys: Iterable[str]) -> bool:
    """Report whether a decision group's page keys carry more than one identifier.

    Pages of one provider with different identifiers are distinct pages. Pages
    of providers decided together follow the same rule because they share a
    numbering scheme.

    Args:
        page_keys: Page keys belonging to one decided provider group.

    Returns:
        True when the page keys contain at least two distinct identifiers.
    """
    return len(set(map(read_page_id, page_keys))) > 1
