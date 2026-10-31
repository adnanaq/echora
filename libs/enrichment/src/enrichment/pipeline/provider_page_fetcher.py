"""Fetch work entries' provider pages, and pages linked by key, into the provider page store.

The work matching rules decide from each page's details and links, so each
page is fetched once through the enrichment helpers and kept in the
``ProviderPageStore``. The same fetch fills the cache enrichment reads. Seed
cleaning fetches a whole seed release through it, and discovery will fetch
newly found pages the same way.

Rules:

- One request per page, with characters and episodes off, so links from two
  pages of one provider never mix.
- Pages are fetched for the providers enrichment has a helper for; ANN,
  LiveChart, Simkl and AnimeCountdown are not.
- Linked pages, which another page names by page key, are fetched at the
  address built back from the key, so the store holds their details too.
- AnimeSchedule is not in the seed: its page is a searched page, found by
  searching AnimeSchedule with the entry's MAL pages, else AniList, else AniDB.
  The page records the page it was searched by as ``searched_by``.
- A page already in the store is not fetched again, and an AnimeSchedule
  search that found a page is not repeated, so a run picks up where the last
  one stopped. A search that found nothing is repeated, so a page AnimeSchedule
  adds later is still found, unless the caller keeps a record of such
  searches (``NotListedSearchRecord``) to skip them. When stored details gain a field,
  one run fetching stored pages again fills it in.
- A page is stored as dead only when its provider said it does not exist: its
  helper answered None. A failed fetch, a service that did not run or a
  payload without an anime record is not stored.
- A failed page is retried once, after the rest of its provider's queue, so a
  provider that failed for a while, such as during a block, gets a second
  chance in the same run. A page failing again is retried on the next run.
- Each provider runs in its own queue, one page at a time, so each site's own
  rate limits hold while different sites run side by side.
"""

import asyncio
import logging
from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from typing import Any, Protocol

from enrichment.pipeline.api_fetcher import ApiFetcher
from enrichment.pipeline.fetch_summary import (
    FetchOutcome,
    FetchSummary,
    combine_fetch_summaries,
)
from enrichment.pipeline.page_requests import (
    PageRequest,
    RequestResult,
    build_page_request,
    list_entry_requests,
)
from work_matching import (
    ProviderPage,
    ProviderPageStore,
    WorkEntry,
    get_page_key,
)

logger = logging.getLogger(__name__)


# Requests between two progress lines of one provider queue.
_REQUESTS_PER_PROGRESS_LINE = 100


def _log_queue_progress(
    request: PageRequest,
    made_request_count: int,
    queue_request_count: int,
    is_retry: bool,
) -> None:
    """Log how many of a queue's requests are made, every so many and after the last.

    Args:
        request: The request just made, naming the queue's service.
        made_request_count: The queue's requests made so far, this one included.
        queue_request_count: The queue's requests in all.
        is_retry: Whether the queue is retrying its failed requests.
    """
    if (
        made_request_count % _REQUESTS_PER_PROGRESS_LINE
        and made_request_count != queue_request_count
    ):
        return
    queue_name = f"{request.service} retries" if is_retry else request.service
    logger.info(
        f"{queue_name}: {made_request_count}/{queue_request_count} requests made"
    )


class NotListedSearchRecord(Protocol):
    """A record of the page keys whose AnimeSchedule search found nothing.

    The fetcher skips a search whose page key the record holds, and adds each
    search that finds nothing. The caller decides how long a search stays
    recorded, such as seed cleaning for one seed release.
    """

    def __contains__(self, searched_by_key: object) -> bool:
        """Report whether the search by this page key found nothing before."""
        ...

    def add(self, searched_by_key: str) -> None:
        """Record that the search by this page key found nothing."""
        ...


class ProviderPageFetcher:
    """Fetch the pages of seed entries into a provider page store."""

    def __init__(
        self,
        new_api_fetcher: Callable[[], ApiFetcher],
        store: ProviderPageStore,
        *,
        is_stored_page_fetched_again: bool = False,
        not_listed_searches: NotListedSearchRecord | None = None,
    ) -> None:
        """Set up the fetcher.

        Args:
            new_api_fetcher: Makes the ``ApiFetcher`` for one provider queue.
                Each queue needs its own, because an ``ApiFetcher`` keeps the
                last fetch's errors on itself.
            store: Where fetched pages are kept.
            is_stored_page_fetched_again: Whether pages and searches the store
                already holds are fetched again, replacing their stored
                details. Used once after the stored details gain a field; the
                cache serves pages fetched in the last week.
            not_listed_searches: The page keys whose AnimeSchedule search found
                nothing and is not to be repeated, such as seed cleaning's
                record for one seed release; None repeats every such search.
        """
        self._new_api_fetcher = new_api_fetcher
        self._store = store
        self._is_stored_page_fetched_again = is_stored_page_fetched_again
        self._not_listed_searches = not_listed_searches

    async def fetch_entry_pages(
        self,
        entries: Iterable[WorkEntry],
    ) -> FetchSummary:
        """Fetch every page of these entries, and AnimeSchedule's, not stored yet.

        Args:
            entries: The entries to fetch, such as every seed entry.

        Returns:
            A count per outcome, and the error of each failed page.
        """
        return await self._fetch_requests(
            request for entry in entries for request in list_entry_requests(entry)
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
            if (request := build_page_request(key, title))
        )
        current_key_summary = await self._fetch_requests(
            request
            for key, title in titles_by_page_key.items()
            if (page := self._store.get_page(key)) is not None
            and page.renamed_to is not None
            and page.renamed_to not in self._store
            and (request := build_page_request(page.renamed_to, title))
        )
        return combine_fetch_summaries(linked_page_summary, current_key_summary)

    async def _fetch_requests(self, requests: Iterable[PageRequest]) -> FetchSummary:
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
        requests_by_service: defaultdict[str, list[PageRequest]] = defaultdict(list)
        done_results_by_service: defaultdict[str, list[RequestResult]] = defaultdict(
            list
        )
        for request in distinct_requests.values():
            if done_outcome := self._find_done_outcome(request):
                outcomes[done_outcome] += 1
                done_results_by_service[request.service].append(
                    RequestResult(done_outcome, request)
                )
            else:
                requests_by_service[request.service].append(request)
        self._log_fetch_plan(requests_by_service, done_results_by_service)
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

    def _log_fetch_plan(
        self,
        requests_by_service: Mapping[str, Collection[PageRequest]],
        done_results_by_service: Mapping[str, Collection[RequestResult]],
    ) -> None:
        """Log, per provider, the requests needed, already done, and to fetch.

        A run over a new seed release then shows at its start how much of the
        release is new, rather than only the store's total.

        Each line reads ``<service>: <needed> needed, <done> done (<dead> dead,
        <not listed> not listed), <to fetch> to fetch``. Dead counts the stored
        pages the provider said do not exist; not listed counts the
        AnimeSchedule searches that found nothing earlier in this release.

        Args:
            requests_by_service: The requests still to make, by service.
            done_results_by_service: The requests needing no fetch, with why,
                by service.
        """
        for service in sorted(requests_by_service.keys() | done_results_by_service):
            to_fetch_count = len(requests_by_service.get(service, ()))
            done_results = done_results_by_service.get(service, ())
            dead_count = sum(
                not done_result.request.is_page_search
                and (
                    page := self._store.get_page(
                        get_page_key(done_result.request.address)
                    )
                )
                is not None
                and page.is_dead
                for done_result in done_results
            )
            not_listed_count = sum(
                done_result.outcome is FetchOutcome.NOT_LISTED_EARLIER
                for done_result in done_results
            )
            logger.info(
                f"{service}: {to_fetch_count + len(done_results)} needed, "
                f"{len(done_results)} done ({dead_count} dead, "
                f"{not_listed_count} not listed), {to_fetch_count} to fetch"
            )

    def _find_done_outcome(self, request: PageRequest) -> FetchOutcome | None:
        """Report whether a request needs no fetch, and why.

        - A page request is done when the page is stored.
        - A search is done when the store holds a page searched by its page,
          or when it found nothing earlier in this seed release.
        - No request is done while stored pages are being fetched again.

        Args:
            request: The page request or AnimeSchedule search.

        Returns:
            ``ALREADY_STORED`` or ``NOT_LISTED_EARLIER`` when no fetch is
            needed; None when the request must be made.
        """
        if self._is_stored_page_fetched_again:
            return None
        key = get_page_key(request.address)
        if not request.is_page_search:
            return FetchOutcome.ALREADY_STORED if key in self._store else None
        if self._store.has_page_searched_by(key):
            return FetchOutcome.ALREADY_STORED
        if self._not_listed_searches is not None and key in self._not_listed_searches:
            return FetchOutcome.NOT_LISTED_EARLIER
        return None

    async def _run_queue(
        self, service: str, requests: Sequence[PageRequest]
    ) -> list[RequestResult]:
        """Make one provider's requests one at a time, then retry the failed ones once.

        - Each page is stored as soon as it is fetched.
        - A failed request is made once more after the rest of the queue, with
          the same fetcher, so the provider's pacing and pauses still hold. A
          provider that failed for a while, such as during a block, often
          answers again by then. A request failing again waits for the next run.

        Returns:
            One result per request: its retry's result when it failed at first.
        """
        api_fetcher = self._new_api_fetcher()
        first_results = await self._make_requests(api_fetcher, requests, is_retry=False)
        retried_results = await self._make_requests(
            api_fetcher,
            [
                request_result.request
                for request_result in first_results
                if request_result.outcome is FetchOutcome.FAILED
            ],
            is_retry=True,
        )
        return [
            request_result
            for request_result in first_results
            if request_result.outcome is not FetchOutcome.FAILED
        ] + retried_results

    async def _make_requests(
        self,
        api_fetcher: ApiFetcher,
        requests: Sequence[PageRequest],
        *,
        is_retry: bool,
    ) -> list[RequestResult]:
        """Make requests one at a time, logging each failure and the queue's progress.

        - An unexpected error in one request, such as a failed store write,
          fails that request and the queue goes on, so a long run keeps the
          rest of its work and its summary.
        - A progress line is logged every ``_REQUESTS_PER_PROGRESS_LINE``
          requests and after the last, so a run lasting days shows how far
          each provider got.

        Args:
            api_fetcher: The queue's fetcher.
            requests: The requests to make, in order.
            is_retry: Whether these requests already failed once this run,
                which names the queue's progress lines.

        Returns:
            One result per request, in order.
        """
        request_results: list[RequestResult] = []
        for request in requests:
            try:
                request_result = await self._fetch_request(api_fetcher, request)
            except Exception as error:
                logger.exception(f"Page fetch failed unexpectedly: {request.label}")
                request_result = RequestResult(
                    FetchOutcome.FAILED, request, f"{type(error).__name__}: {error}"
                )
            if request_result.outcome is FetchOutcome.FAILED:
                retry_time = "next run" if is_retry else "at the end of the queue"
                logger.warning(
                    f"Page fetch failed, retried {retry_time}: {request.label}"
                )
            request_results.append(request_result)
            _log_queue_progress(request, len(request_results), len(requests), is_retry)
        return request_results

    async def _fetch_request(
        self, api_fetcher: ApiFetcher, request: PageRequest
    ) -> RequestResult:
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
            return RequestResult(self._store_page(request, None), request)
        elif (anime_record := records_by_service[request.service].get("anime")) is None:
            failure = "the helper returned no anime record"
        else:
            return RequestResult(self._store_page(request, anime_record), request)
        return RequestResult(FetchOutcome.FAILED, request, failure)

    def _store_page(
        self, request: PageRequest, anime_record: dict[str, Any] | None
    ) -> FetchOutcome:
        """Store what one request returned, and say what it came to."""
        if request.is_page_search:
            if not anime_record:
                if self._not_listed_searches is not None:
                    self._not_listed_searches.add(get_page_key(request.address))
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
