"""Fetch every page of the seed entries that need checking into the provider page store.

The work matching rules decide from each page's details and links, so every
page of each entry to check is fetched once through the enrichment helpers and kept in
the ``ProviderPageStore``. The same fetch fills the cache enrichment reads.

Rules:

- One request per page, with characters and episodes off, so links from two
  pages of one provider never mix.
- Pages are fetched for the providers enrichment has a helper for; ANN,
  LiveChart, Simkl and AnimeCountdown are not. A fetch can be limited to some
  providers, such as only those whose pages carry links (AniList, Kitsu,
  AniDB), when the pages are wanted for their links alone.
- Linked pages, which another page names by page key, are fetched at the
  address built back from the key, so the store holds their details too.
- AnimeSchedule is not in the seed: its page is found through the entry's MAL
  pages, else AniList, else AniDB, and records that page as ``found_by``.
- A page already in the store is not fetched again, and an AnimeSchedule
  lookup that found a page is not repeated, so a run picks up where the last
  one stopped. A lookup that found nothing is repeated on every run, so a page
  AnimeSchedule adds later is still found. When stored details gain a field,
  one run fetching stored pages again fills it in.
- A page is stored as dead only when its provider said it does not exist: its
  helper answered None. A failed fetch, a service that did not run or a
  payload without an anime record is not stored, so the next run tries it
  again.
- Each provider runs in its own queue, one page at a time, so each site's own
  rate limits hold while different sites run side by side.
"""

import asyncio
import logging
from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, NamedTuple

from enrichment.pipeline.api_fetcher import ApiFetcher
from work_matching import (
    ProviderPage,
    ProviderPageStore,
    WorkEntry,
    build_page_address,
    get_page_key,
    get_page_provider,
)

logger = logging.getLogger(__name__)


class _EnrichmentHelper(NamedTuple):
    """An enrichment helper: its ``ApiFetcher`` service name and the id key it reads."""

    service: str
    id_key: str


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

    outcomes: Mapping[FetchOutcome, int]
    # Each failed page by its page key, or "animeschedule via <lookup page>".
    failed_pages: Mapping[str, str]


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
        key = get_page_key(self.address)
        return f"{self.service} via {key}" if self.is_found_page_lookup else key


@dataclass(frozen=True)
class _RequestResult:
    """What one request came to, with the error when it failed."""

    outcome: FetchOutcome
    request: _PageRequest
    error: str | None = None


def _build_page_request(key: str, address: str, title: str) -> _PageRequest | None:
    """Return the request fetching one page, or None when its provider has no helper.

    Args:
        key: The page's key.
        address: The page's address, as the request sends it.
        title: The title of the entry the page belongs to or is linked from,
            which some helpers search by.
    """
    helper = _HELPER_BY_PROVIDER.get(get_page_provider(key) or "")
    return (
        _PageRequest(helper.service, helper.id_key, address, title) if helper else None
    )


class ProviderPageFetcher:
    """Fetch the pages of seed entries into a provider page store."""

    def __init__(
        self,
        new_api_fetcher: Callable[[], ApiFetcher],
        store: ProviderPageStore,
        *,
        seconds_between_found_page_lookups: float = 3.0,
        is_stored_page_fetched_again: bool = False,
    ) -> None:
        """Set up the fetcher.

        Args:
            new_api_fetcher: Makes the ``ApiFetcher`` for one provider queue.
                Each queue needs its own, because an ``ApiFetcher`` keeps the
                last fetch's errors on itself.
            store: Where fetched pages are kept.
            seconds_between_found_page_lookups: The wait between AnimeSchedule
                lookups. Requests without an app token get AnimeSchedule's
                stricter, undocumented limit, and lookups at full speed get
                HTTP 429.
            is_stored_page_fetched_again: Whether pages and lookups the store
                already holds are fetched again, replacing their stored
                details. Used once after the stored details gain a field; the
                cache serves pages fetched in the last week.
        """
        self._new_api_fetcher = new_api_fetcher
        self._store = store
        self._seconds_between_found_page_lookups = seconds_between_found_page_lookups
        self._is_stored_page_fetched_again = is_stored_page_fetched_again

    async def fetch_entry_pages(
        self,
        entries: Iterable[WorkEntry],
        *,
        providers: Collection[str] | None = None,
    ) -> FetchSummary:
        """Fetch every page of these entries that the store does not hold yet.

        Args:
            entries: The entries to fetch, such as the seed entries to check.
            providers: The providers whose pages are fetched. None fetches every
                provider with a helper and looks up AnimeSchedule; a given set
                fetches only those providers' pages, without lookups, such as
                only the providers whose pages carry links.

        Returns:
            A count per outcome, and the error of each failed page.
        """
        return await self._fetch_requests(
            request
            for entry in entries
            for request in self._list_requests(entry, providers)
        )

    async def fetch_linked_pages(
        self, titles_by_page_key: Mapping[str, str]
    ) -> FetchSummary:
        """Fetch linked pages, named by page key, that the store does not hold yet.

        Links are kept as page keys, so each page's address is built back from
        its key. Pages of providers without a helper are skipped.

        Args:
            titles_by_page_key: Each linked page's key, mapped to the title of
                the entry linking it, which some helpers search by.

        Returns:
            A count per outcome, and the error of each failed page.
        """
        return await self._fetch_requests(
            request
            for key, title in titles_by_page_key.items()
            if (address := build_page_address(key))
            if (request := _build_page_request(key, address, title))
        )

    async def _fetch_requests(self, requests: Iterable[_PageRequest]) -> FetchSummary:
        """Make every request the store does not answer yet, each once, and sum up.

        A request is made once even when several entries need it, and each
        provider's requests run in a queue of their own.
        """
        distinct_requests = {
            (request.service, request.address): request for request in requests
        }
        outcomes: Counter[FetchOutcome] = Counter()
        requests_by_service: defaultdict[str, list[_PageRequest]] = defaultdict(list)
        for request in distinct_requests.values():
            if self._is_done(request):
                outcomes[FetchOutcome.ALREADY_STORED] += 1
            else:
                requests_by_service[request.service].append(request)
        async with asyncio.TaskGroup() as task_group:
            queue_tasks = [
                task_group.create_task(self._run_queue(service, requests))
                for service, requests in requests_by_service.items()
            ]
        request_results = [
            request_result
            for queue_task in queue_tasks
            for request_result in queue_task.result()
        ]
        outcomes.update(request_result.outcome for request_result in request_results)
        return FetchSummary(
            outcomes=dict(outcomes),
            failed_pages={
                request_result.request.label: request_result.error
                for request_result in request_results
                if request_result.error is not None
            },
        )

    def _list_requests(
        self, entry: WorkEntry, providers: Collection[str] | None
    ) -> list[_PageRequest]:
        """List every request one entry needs, whether done already or not.

        With ``providers`` given, only those providers' pages are requested and
        AnimeSchedule is not looked up.
        """
        requests = [
            request
            for key, address in entry.pages.items()
            if providers is None or get_page_provider(key) in providers
            if (request := _build_page_request(key, address, entry.title))
        ]
        if providers is not None:
            return requests
        # AnimeSchedule is looked up through the first provider in the order that
        # has pages in the entry: MAL, else AniList, else AniDB.
        for provider in _FOUND_PAGE_LOOKUP_PROVIDERS:
            lookup_addresses = [
                address
                for key, address in entry.pages.items()
                if get_page_provider(key) == provider
            ]
            if lookup_addresses:
                requests.extend(
                    _PageRequest(
                        _FOUND_PAGE_SERVICE,
                        _HELPER_BY_PROVIDER[provider].id_key,
                        address,
                        entry.title,
                        is_found_page_lookup=True,
                    )
                    for address in lookup_addresses
                )
                break
        return requests

    def _is_done(self, request: _PageRequest) -> bool:
        """Report whether the store already holds what a request would fetch.

        A page request is done when the page is stored; a lookup is done when a
        found page was found through its lookup page. Nothing is done while
        stored pages are being fetched again.
        """
        if self._is_stored_page_fetched_again:
            return False
        key = get_page_key(request.address)
        if request.is_found_page_lookup:
            return self._store.has_page_found_by(key)
        return key in self._store

    async def _run_queue(
        self, service: str, requests: Iterable[_PageRequest]
    ) -> list[_RequestResult]:
        """Make one provider's requests one at a time, storing each page at once.

        An unexpected error in one request, such as a failed store write, fails
        that request and the queue goes on, so a long run keeps the rest of its
        work and its summary.
        """
        api_fetcher = self._new_api_fetcher()
        request_results: list[_RequestResult] = []
        for position, request in enumerate(requests):
            if request.is_found_page_lookup and position:
                await asyncio.sleep(self._seconds_between_found_page_lookups)
            try:
                request_results.append(await self._fetch_request(api_fetcher, request))
            except Exception as error:
                logger.exception(f"Page fetch failed unexpectedly: {request.label}")
                request_results.append(
                    _RequestResult(
                        FetchOutcome.FAILED, request, f"{type(error).__name__}: {error}"
                    )
                )
        return request_results

    async def _fetch_request(
        self, api_fetcher: ApiFetcher, request: _PageRequest
    ) -> _RequestResult:
        """Make one request and store what it returned.

        Only a helper answering None means the page does not exist. A service
        that did not run, or a payload without an anime record, is a failed
        fetch: storing it as dead would keep a live page from ever being
        fetched again.
        """
        records_by_service = await api_fetcher.fetch_all_data(
            {request.id_key: request.address},
            {"title": request.title, "sources": [request.address]},
            only_services=[request.service],
            fetch_characters=False,
            fetch_episodes=False,
        )
        if request.service in api_fetcher.api_errors:
            failure = api_fetcher.api_errors[request.service]
        elif request.service not in records_by_service:
            failure = "the service did not run"
        elif records_by_service[request.service] is None:
            return _RequestResult(self._store_page(request, None), request)
        elif (anime_record := records_by_service[request.service].get("anime")) is None:
            failure = "the helper returned no anime record"
        else:
            return _RequestResult(self._store_page(request, anime_record), request)
        logger.warning(f"Page fetch failed, retried next run: {request.label}")
        return _RequestResult(FetchOutcome.FAILED, request, failure)

    def _store_page(
        self, request: _PageRequest, anime_record: dict[str, Any] | None
    ) -> FetchOutcome:
        """Store what one request returned, and say what it came to."""
        if request.is_found_page_lookup:
            if not anime_record:
                return FetchOutcome.NOT_LISTED
            # The AnimeSchedule mapper lists the page's own address first.
            self._store.add_pages(
                ProviderPage.from_canonical_record(
                    anime_record["sources"][0], anime_record, found_by=request.address
                )
            )
            return FetchOutcome.FETCHED
        if not anime_record:
            self._store.add_pages(ProviderPage.from_dead_address(request.address))
            return FetchOutcome.DEAD
        self._store.add_pages(
            ProviderPage.from_canonical_record(request.address, anime_record)
        )
        return FetchOutcome.FETCHED
