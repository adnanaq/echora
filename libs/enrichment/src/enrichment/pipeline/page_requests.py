"""Build the requests that fetch a work entry's pages and its AnimeSchedule searches.

Each item gives the rule, then the reason for it.

- A page is requested through the enrichment helper of its provider; ANN,
  LiveChart, Simkl and AnimeCountdown have none, so their pages are skipped.
- AnimeSchedule is not in the seed: its page is found by searching AnimeSchedule
  with the entry's MAL pages, else AniList, else AniDB, the ids its helper
  finds an anime by.
- A linked page is known only by its key, so its address is built back from
  the key.
"""

from dataclasses import dataclass
from typing import NamedTuple

from enrichment.pipeline.fetch_summary import FetchOutcome
from work_matching import (
    WorkEntry,
    build_page_address,
    read_page_provider,
    resolve_page_key,
)


class _EnrichmentHelper(NamedTuple):
    """An enrichment helper: its ``ApiFetcher`` service name and the id key it reads."""

    service: str
    id_field: str


# The helper for each provider that has one, by the provider's page key name.
_HELPER_BY_PROVIDER = {
    "mal": _EnrichmentHelper("mal", "mal_url"),
    "anilist": _EnrichmentHelper("anilist", "anilist_url"),
    "kitsu": _EnrichmentHelper("kitsu", "kitsu_url"),
    "anidb": _EnrichmentHelper("anidb", "anidb_url"),
    "animeplanet": _EnrichmentHelper("anime_planet", "anime_planet_url"),
    "anisearch": _EnrichmentHelper("anisearch", "anisearch_url"),
}
# The providers whose pages are fetched, so a page of theirs can be checked live.
FETCHED_PROVIDERS: frozenset[str] = frozenset(_HELPER_BY_PROVIDER)
# The providers AnimeSchedule is searched by, in the order tried: the
# AnimeSchedule helper finds an anime by these ids.
_SEARCHED_BY_PROVIDERS = ("mal", "anilist", "anidb")
_SEARCHED_PAGE_SERVICE = "animeschedule"


@dataclass(frozen=True)
class PageRequest:
    """One request to make: a seed page, or an AnimeSchedule search by one."""

    service: str
    id_field: str
    address: str
    title: str
    is_page_search: bool = False

    @property
    def label(self) -> str:
        """Name the request in the summary: its page key, or its search."""
        key = resolve_page_key(self.address)
        return f"{self.service} via {key}" if self.is_page_search else key


@dataclass(frozen=True)
class RequestResult:
    """What one request came to, with the error when it failed."""

    outcome: FetchOutcome
    request: PageRequest
    error: str | None = None


def build_page_request(
    key: str, title: str, address: str | None = None
) -> PageRequest | None:
    """Return the request fetching one page.

    Args:
        key: The page's key.
        title: The title of the entry the page belongs to or is linked from,
            which some helpers search by.
        address: The page's address as the seed lists it. A linked page is
            known only by its key, so without an address the address is built
            from the key.

    Returns:
        The request, or None when the page's provider has no enrichment helper.
    """
    helper = _HELPER_BY_PROVIDER.get(read_page_provider(key) or "")
    request_address = address or build_page_address(key)
    if helper is None or request_address is None:
        return None
    return PageRequest(helper.service, helper.id_field, request_address, title)


def list_entry_page_requests(entry: WorkEntry) -> list[PageRequest]:
    """List the requests for every page an entry lists, whether done already or not.

    Args:
        entry: The work entry whose pages to request.

    Returns:
        One request per page of a provider with an enrichment helper.
    """
    return [
        request
        for key, address in entry.addresses_by_page_key.items()
        if (request := build_page_request(key, entry.title, address))
    ]


def list_animeschedule_search_requests(entry: WorkEntry) -> list[PageRequest]:
    """List the AnimeSchedule searches an entry needs, whether done already or not.

    AnimeSchedule is searched by the first provider in the order that has pages
    in the entry: MAL, else AniList, else AniDB, once per page of it.

    Args:
        entry: The work entry to search AnimeSchedule for.

    Returns:
        One search per page of that provider; none when the entry has no page
        of any of them.
    """
    for provider in _SEARCHED_BY_PROVIDERS:
        searched_by_addresses = [
            address
            for key, address in entry.addresses_by_page_key.items()
            if read_page_provider(key) == provider
        ]
        if searched_by_addresses:
            return [
                PageRequest(
                    _SEARCHED_PAGE_SERVICE,
                    _HELPER_BY_PROVIDER[provider].id_field,
                    address,
                    entry.title,
                    is_page_search=True,
                )
                for address in searched_by_addresses
            ]
    return []
