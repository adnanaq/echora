"""Fetch every page of the seed entries that need checking into the provider page store.

The work matching rules decide from each page's details and links, so every
page of each entry to check is fetched once through the enrichment helpers and kept in
the ``ProviderPageStore``. The same fetch fills the cache enrichment reads.

Rules:

- One request per page, with characters and episodes off, so links from two
  pages of one provider never mix.
- Pages are fetched for the providers enrichment has a helper for; ANN,
  LiveChart, Simkl and AnimeCountdown are not.
- AnimeSchedule is not in the seed: its page is found through the entry's MAL
  pages, else AniList, else AniDB, and records that page as ``found_by``.
- A page already in the store is not fetched again, and an AnimeSchedule
  lookup already done through a page is not repeated, so a run picks up where
  the last one stopped.
- A page is stored as dead only when its provider said it does not exist. A
  failed fetch is not stored, so the next run tries it again.
- Each provider runs in its own queue, one page at a time, so each site's own
  rate limits hold while different sites run side by side.
"""

import asyncio
import logging
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum

from enrichment.pipeline.api_fetcher import ApiFetcher
from work_matching import (
    ProviderPage,
    ProviderPageStore,
    WorkEntry,
    provider_of_key,
    provider_page_key,
)

logger = logging.getLogger(__name__)

# The enrichment id key for each provider with a helper; the helper's service
# name is the id key without "_url".
_ID_KEY_BY_PROVIDER = {
    "mal": "mal_url",
    "anilist": "anilist_url",
    "kitsu": "kitsu_url",
    "anidb": "anidb_url",
    "animeplanet": "anime_planet_url",
    "anisearch": "anisearch_url",
}
# The providers AnimeSchedule is looked up through, in the order tried: the
# AnimeSchedule helper finds an anime by these ids.
_FOUND_PAGE_LOOKUP_PROVIDERS = ("mal", "anilist", "anidb")
_FOUND_PAGE_SERVICE = "animeschedule"


class FetchOutcome(StrEnum):
    """What fetching one page, or one AnimeSchedule lookup, came to."""

    FETCHED = "fetched"
    # The provider said the page does not exist.
    DEAD = "dead"
    # AnimeSchedule has no page for the lookup page: nothing to store.
    NOT_LISTED = "not listed"
    FAILED = "failed"
    ALREADY_STORED = "already stored"


@dataclass(frozen=True)
class FetchSummary:
    """How a fetch run went: a count per outcome, and why each failure failed."""

    outcomes: dict[FetchOutcome, int]
    # Each failed page by its page key, or "animeschedule via <lookup page>".
    failed_pages: dict[str, str]


@dataclass(frozen=True)
class _PageRequest:
    """One request to make: a seed page, or an AnimeSchedule lookup through one."""

    service: str
    id_key: str
    address: str
    title: str
    is_found_page_lookup: bool = False

    @property
    def label(self) -> str:
        """Name the request in the summary: its page key, or its lookup."""
        key = provider_page_key(self.address)
        return f"{self.service} via {key}" if self.is_found_page_lookup else key


class ProviderPageFetcher:
    """Fetch the pages of seed entries into a provider page store."""

    def __init__(
        self,
        new_api_fetcher: Callable[[], ApiFetcher],
        store: ProviderPageStore,
        *,
        seconds_between_found_page_lookups: float = 3.0,
    ) -> None:
        """Set up the fetcher.

        Args:
            new_api_fetcher: Makes the ``ApiFetcher`` for one provider queue.
                Each queue needs its own, because an ``ApiFetcher`` keeps the
                last fetch's errors on itself.
            store: Where fetched pages are kept.
            seconds_between_found_page_lookups: The wait between AnimeSchedule
                lookups. Requests without an app token get AnimeSchedule's
                stricter, undocumented limit, and on 2026-10-17 lookups at full
                speed got HTTP 429 within 24 lookups.
        """
        self._new_api_fetcher = new_api_fetcher
        self._store = store
        self._seconds_between_found_page_lookups = seconds_between_found_page_lookups

    async def fetch_entry_pages(self, entries: Iterable[WorkEntry]) -> FetchSummary:
        """Fetch every page of these entries that the store does not hold yet.

        Args:
            entries: The entries to fetch, such as the seed entries to check.

        Returns:
            A count per outcome, and the error of each failed page.
        """
        outcomes: Counter[FetchOutcome] = Counter()
        requests_by_service: defaultdict[str, dict[str, _PageRequest]] = defaultdict(
            dict
        )
        for entry in entries:
            for request in self._requests_for(entry):
                if request is None:
                    outcomes[FetchOutcome.ALREADY_STORED] += 1
                else:
                    requests_by_service[request.service][request.address] = request
        failed_pages: dict[str, str] = {}
        for queue_results in await asyncio.gather(
            *(
                self._run_queue(service, requests.values())
                for service, requests in requests_by_service.items()
            )
        ):
            for outcome, request, error in queue_results:
                outcomes[outcome] += 1
                if error is not None:
                    failed_pages[request.label] = error
        return FetchSummary(outcomes=dict(outcomes), failed_pages=failed_pages)

    def _requests_for(self, entry: WorkEntry) -> list[_PageRequest | None]:
        """List the requests one entry needs; None for each one already done."""
        requests: list[_PageRequest | None] = []
        for key, address in entry.pages.items():
            id_key = _ID_KEY_BY_PROVIDER.get(provider_of_key(key) or "")
            if id_key is not None:
                requests.append(
                    None
                    if key in self._store
                    else _PageRequest(
                        id_key.removesuffix("_url"), id_key, address, entry.title
                    )
                )
        for provider in _FOUND_PAGE_LOOKUP_PROVIDERS:
            lookup_pages = {
                key: address
                for key, address in entry.pages.items()
                if provider_of_key(key) == provider
            }
            if lookup_pages:
                requests.extend(
                    None
                    if self._store.has_page_found_by(key)
                    else _PageRequest(
                        _FOUND_PAGE_SERVICE,
                        _ID_KEY_BY_PROVIDER[provider],
                        address,
                        entry.title,
                        is_found_page_lookup=True,
                    )
                    for key, address in lookup_pages.items()
                )
                break
        return requests

    async def _run_queue(
        self, service: str, requests: Iterable[_PageRequest]
    ) -> list[tuple[FetchOutcome, _PageRequest, str | None]]:
        """Make one provider's requests one at a time, storing each page at once."""
        api_fetcher = self._new_api_fetcher()
        results: list[tuple[FetchOutcome, _PageRequest, str | None]] = []
        for position, request in enumerate(requests):
            if request.is_found_page_lookup and position:
                await asyncio.sleep(self._seconds_between_found_page_lookups)
            fetched = await api_fetcher.fetch_all_data(
                {request.id_key: request.address},
                {"title": request.title, "sources": [request.address]},
                only_services=[service],
                fetch_characters=False,
                fetch_episodes=False,
            )
            if service in api_fetcher.api_errors:
                error = api_fetcher.api_errors[service]
                logger.warning(f"Page fetch failed, retried next run: {request.label}")
                results.append((FetchOutcome.FAILED, request, error))
                continue
            record = (fetched.get(service) or {}).get("anime")
            outcome = self._store_page(request, record)
            results.append((outcome, request, None))
        return results

    def _store_page(self, request: _PageRequest, record: dict | None) -> FetchOutcome:
        """Store what one request returned, and say what it came to."""
        if request.is_found_page_lookup:
            if not record:
                return FetchOutcome.NOT_LISTED
            # The AnimeSchedule mapper lists the page's own address first.
            self._store.add_pages(
                ProviderPage.from_canonical_record(
                    record["sources"][0], record, found_by=request.address
                )
            )
            return FetchOutcome.FETCHED
        if not record:
            self._store.add_pages(ProviderPage.dead_page(request.address))
            return FetchOutcome.DEAD
        self._store.add_pages(
            ProviderPage.from_canonical_record(request.address, record)
        )
        return FetchOutcome.FETCHED
