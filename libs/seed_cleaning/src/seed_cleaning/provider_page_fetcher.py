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
- AnimeSchedule is not in the seed: its page is a searched page, found by
  searching AnimeSchedule with the entry's MAL pages, else AniList, else AniDB.
  The page records the page it was searched by as ``searched_by``.
- A page already in the store is not fetched again, and an AnimeSchedule
  search that found a page is not repeated, so a run picks up where the last
  one stopped. A search that found nothing is repeated on every run, so a page
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


class FetchOutcome(StrEnum):
    """What fetching one page, or one AnimeSchedule search, came to."""

    FETCHED = "fetched"
    # The provider said the page does not exist.
    DEAD = "dead"
    # AnimeSchedule has no page for the page searched by: nothing to store.
    NOT_LISTED = "not listed"
    FAILED = "failed"
    ALREADY_STORED = "already stored"


@dataclass(frozen=True)
class FetchSummary:
    """How a fetch run went: a count per outcome, and why each failure failed."""

    outcomes: Mapping[FetchOutcome, int]
    # Each failed request's error, by its label: the page key, or
    # "animeschedule via <page key searched by>".
    errors_by_request_label: Mapping[str, str]


@dataclass(frozen=True)
class _PageRequest:
    """One request to make: a seed page, or an AnimeSchedule search by one."""

    service: str
    id_field: str
    address: str
    title: str
    is_page_search: bool = False

    @property
    def label(self) -> str:
        """Name the request in the summary: its page key, or its search."""
        key = get_page_key(self.address)
        return f"{self.service} via {key}" if self.is_page_search else key


@dataclass(frozen=True)
class _RequestResult:
    """What one request came to, with the error when it failed."""

    outcome: FetchOutcome
    request: _PageRequest
    error: str | None = None


def _build_page_request(
    key: str, title: str, address: str | None = None
) -> _PageRequest | None:
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
    helper = _HELPER_BY_PROVIDER.get(get_page_provider(key) or "")
    request_address = address or build_page_address(key)
    if helper is None or request_address is None:
        return None
    return _PageRequest(helper.service, helper.id_field, request_address, title)


def _combine_fetch_summaries(*summaries: FetchSummary) -> FetchSummary:
    """Add up the outcomes and failures of several fetches into one summary.

    Args:
        *summaries: The summaries of the fetches.

    Returns:
        One summary with every count added and every failure kept.
    """
    outcomes: Counter[FetchOutcome] = Counter()
    errors_by_request_label: dict[str, str] = {}
    for summary in summaries:
        outcomes.update(summary.outcomes)
        errors_by_request_label |= summary.errors_by_request_label
    return FetchSummary(
        outcomes=dict(outcomes), errors_by_request_label=errors_by_request_label
    )


class ProviderPageFetcher:
    """Fetch the pages of seed entries into a provider page store."""

    def __init__(
        self,
        new_api_fetcher: Callable[[], ApiFetcher],
        store: ProviderPageStore,
        *,
        seconds_between_page_searches: float = 3.0,
        is_stored_page_fetched_again: bool = False,
    ) -> None:
        """Set up the fetcher.

        Args:
            new_api_fetcher: Makes the ``ApiFetcher`` for one provider queue.
                Each queue needs its own, because an ``ApiFetcher`` keeps the
                last fetch's errors on itself.
            store: Where fetched pages are kept.
            seconds_between_page_searches: The wait between AnimeSchedule
                searches. Requests without an app token get AnimeSchedule's
                stricter, undocumented limit, and searches at full speed get
                HTTP 429.
            is_stored_page_fetched_again: Whether pages and searches the store
                already holds are fetched again, replacing their stored
                details. Used once after the stored details gain a field; the
                cache serves pages fetched in the last week.
        """
        self._new_api_fetcher = new_api_fetcher
        self._store = store
        self._seconds_between_page_searches = seconds_between_page_searches
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
                provider with a helper and searches AnimeSchedule; a given set
                fetches only those providers' pages, without searches, such as
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
        its key. Pages of providers without a helper are skipped. A link can
        spell a page another way than the seed does, such as a Kitsu slug for a
        page the seed lists by number; when a fetched page reports another
        current key, that key is fetched too, so the page's details are stored
        under the key an entry would hold it by.

        Args:
            titles_by_page_key: Each linked page's key, mapped to the title of
                the entry linking it, which some helpers search by.

        Returns:
            A count per outcome, and the error of each failed page.
        """
        linked_page_summary = await self._fetch_requests(
            request
            for key, title in titles_by_page_key.items()
            if (request := _build_page_request(key, title))
        )
        current_key_summary = await self._fetch_requests(
            request
            for key, title in titles_by_page_key.items()
            if (page := self._store.get_page(key)) is not None
            and page.renamed_to is not None
            and page.renamed_to not in self._store
            and (request := _build_page_request(page.renamed_to, title))
        )
        return _combine_fetch_summaries(linked_page_summary, current_key_summary)

    async def _fetch_requests(self, requests: Iterable[_PageRequest]) -> FetchSummary:
        """Make every request the store does not answer yet, and sum up the fetch.

        - A request is made once, even when several entries need it.
        - Each provider's requests run in a queue of their own.

        Args:
            requests: The requests to make, repeats included.

        Returns:
            A count per outcome, and the error of each failed page.
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
            errors_by_request_label={
                request_result.request.label: request_result.error
                for request_result in request_results
                if request_result.error is not None
            },
        )

    def _list_requests(
        self, entry: WorkEntry, providers: Collection[str] | None
    ) -> list[_PageRequest]:
        """List every request one entry needs, whether done already or not.

        Args:
            entry: The work entry whose pages to request.
            providers: The providers whose pages are requested. With a set
                given, AnimeSchedule is not searched. None requests every
                provider with a helper and searches AnimeSchedule.

        Returns:
            The page requests, then the AnimeSchedule searches.
        """
        requests = [
            request
            for key, address in entry.addresses_by_page_key.items()
            if (providers is None or get_page_provider(key) in providers)
            and (request := _build_page_request(key, entry.title, address))
        ]
        if providers is not None:
            return requests
        # AnimeSchedule is searched by the first provider in the order that
        # has pages in the entry: MAL, else AniList, else AniDB.
        for provider in _SEARCHED_BY_PROVIDERS:
            searched_by_addresses = [
                address
                for key, address in entry.addresses_by_page_key.items()
                if get_page_provider(key) == provider
            ]
            if searched_by_addresses:
                requests.extend(
                    _PageRequest(
                        _SEARCHED_PAGE_SERVICE,
                        _HELPER_BY_PROVIDER[provider].id_field,
                        address,
                        entry.title,
                        is_page_search=True,
                    )
                    for address in searched_by_addresses
                )
                break
        return requests

    def _is_done(self, request: _PageRequest) -> bool:
        """Report whether the store already holds what a request would fetch.

        - A page request is done when the page is stored.
        - A search is done when the store holds a page searched by its page.
        - No request is done while stored pages are being fetched again.

        Args:
            request: The page request or AnimeSchedule search.

        Returns:
            True when the request needs no fetch.
        """
        if self._is_stored_page_fetched_again:
            return False
        key = get_page_key(request.address)
        if request.is_page_search:
            return self._store.has_page_searched_by(key)
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
            if request.is_page_search and position:
                await asyncio.sleep(self._seconds_between_page_searches)
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
            {request.id_field: request.address},
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
        if request.is_page_search:
            if not anime_record:
                return FetchOutcome.NOT_LISTED
            # The AnimeSchedule mapper lists the page's own address first.
            self._store.add_pages(
                ProviderPage.from_canonical_record(
                    anime_record["sources"][0],
                    anime_record,
                    searched_by_address=request.address,
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
