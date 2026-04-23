"""
Parallel API fetcher for anime enrichment.
Fetches data from all APIs concurrently using existing helpers.
Reduces API fetching from 30-60s sequential to 5-10s parallel.
"""

import asyncio
import logging
import time
from types import TracebackType
from typing import Any, ClassVar

from enrichment.sources.anidb.anidb_helper import AniDBHelper
from enrichment.sources.anilist.anilist_helper import AniListHelper
from enrichment.sources.anime_planet.anime_planet_helper import AnimePlanetHelper
from enrichment.sources.animeschedule.animeschedule_helper import AnimescheduleHelper
from enrichment.sources.anisearch.anisearch_helper import AniSearchHelper
from enrichment.sources.base.base_helper import normalize_enrichment_payload
from enrichment.sources.kitsu.kitsu_helper import KitsuHelper
from enrichment.sources.mal.mal_helper import MalHelper

from .config import EnrichmentConfig

logger = logging.getLogger(__name__)


class ApiFetcher:
    """Fetch data from all anime APIs in parallel.

    Implements graceful degradation — continues with partial data if individual APIs fail.
    """

    _REGISTRY: ClassVar[dict[str, type]] = {
        "anilist": AniListHelper,
        "kitsu": KitsuHelper,
        "anidb": AniDBHelper,
        "anime_planet": AnimePlanetHelper,
        "anisearch": AniSearchHelper,
        "animeschedule": AnimescheduleHelper,
        "mal": MalHelper,
    }

    def __init__(self, config: EnrichmentConfig | None = None):
        """Initialize the parallel API fetcher."""
        self.config = config or EnrichmentConfig()

        # Track API performance
        self.api_timings: dict[str, float] = {}
        self.api_errors: dict[str, str] = {}

    @staticmethod
    def _should_include(
        service: str,
        only: list[str] | None,
        skip: list[str] | None,
    ) -> bool:
        """Return True if service should be fetched given only/skip filter lists."""
        if only:
            return service in only
        if skip:
            return service not in skip
        return True

    def _build_helpers(
        self,
        skip_services: list[str] | None = None,
        only_services: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a fresh helper instance per service for a single fetch run."""
        return {
            name: cls()
            for name, cls in self._REGISTRY.items()
            if self._should_include(name, only_services, skip_services)
        }

    async def fetch_all_data(
        self,
        ids: dict[str, str],
        offline_data: dict[str, Any],
        temp_dir: str | None = None,
        skip_services: list[str] | None = None,
        only_services: list[str] | None = None,
    ) -> dict[str, Any]:
        """Fetch data from multiple anime APIs concurrently with optional service filtering.

        Args:
            ids: Platform-key → ID/URL map (e.g. ``"mal_url"``, ``"anilist_url"``, ``"kitsu_id"``).
            offline_data: Original offline anime metadata; used as input/fallback by some fetchers.
            temp_dir: Directory where per-service JSONL files are written when provided.
            skip_services: Services to omit. Ignored when ``only_services`` is set.
            only_services: Exclusive allowlist; takes precedence over ``skip_services``.

        Returns:
            Mapping of service name → fetched result dict, or ``None`` for failed/timed-out services.
        """
        self.api_timings = {}
        self.api_errors = {}

        helpers = self._build_helpers(
            skip_services=skip_services, only_services=only_services
        )

        if only_services:
            logger.info(f"Only fetching services: {only_services}")
        elif skip_services:
            logger.info(f"Skipping services: {skip_services}")

        start_time = time.time()
        tasks: list[tuple[str, Any]] = [
            (name, self._fetch_service(name, helper, ids, offline_data, temp_dir))
            for name, helper in helpers.items()
        ]

        try:
            results = await self._gather(tasks)
        finally:
            for helper in helpers.values():
                if hasattr(helper, "close"):
                    await helper.close()

        results = {
            name: normalize_enrichment_payload(result) if result is not None else None
            for name, result in results.items()
        }

        elapsed = time.time() - start_time
        logger.info(f"Fetched all API data in {elapsed:.2f} seconds")
        self._log_performance_metrics(elapsed)

        return results

    async def _fetch_service(
        self,
        name: str,
        helper: Any,
        ids: dict[str, str],
        offline_data: dict[str, Any],
        temp_dir: str | None,
    ) -> dict[str, Any] | None:
        """Generic fetcher for any registered service."""
        try:
            start = time.time()
            result = await helper.fetch_all(ids, offline_data, temp_dir)
            self.api_timings[name] = time.time() - start
        except Exception as e:
            logger.exception(f"API {name} fetch failed")
            self.api_errors[name] = str(e)
            return None
        else:
            return result

    async def _gather(self, tasks: list[tuple[str, Any]]) -> dict[str, Any]:
        """Run named coroutines concurrently and collect results with graceful degradation."""
        results: dict[str, Any] = {}
        task_names = [name for name, _ in tasks]
        coroutines = [coro for _, coro in tasks]

        task_results = await asyncio.gather(*coroutines, return_exceptions=True)

        for i, result in enumerate(task_results):
            name = task_names[i]
            if isinstance(result, Exception):
                logger.error(f"API {name} failed with error: {result}")
                results[name] = None
                self.api_errors[name] = str(result)
            else:
                results[name] = result
                if result:
                    logger.debug(f"API {name} completed successfully")
                else:
                    logger.warning(f"API {name} returned empty result")

        return results

    def _log_performance_metrics(self, total_time: float) -> None:
        """Log per-service timings, errors, and overall success rate.

        Args:
            total_time: Total elapsed seconds for the parallel fetch.
        """
        logger.info("API Performance Metrics:")
        logger.info(f"  Total Time: {total_time:.2f}s")

        for api, timing in self.api_timings.items():
            logger.info(f"  {api}: {timing:.2f}s")

        if self.api_errors:
            logger.warning("API Errors:")
            for api, error in self.api_errors.items():
                logger.warning(f"  {api}: {error}")

        # Calculate success rate
        total_apis = len(self.api_timings) + len(self.api_errors)
        success_rate = (
            (len(self.api_timings) / total_apis * 100) if total_apis > 0 else 0
        )
        logger.info(f"  Success Rate: {success_rate:.1f}%")

    async def __aenter__(self) -> "ApiFetcher":
        """Enter async context."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        return False
