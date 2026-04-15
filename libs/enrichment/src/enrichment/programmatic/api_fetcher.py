"""
Parallel API fetcher for anime enrichment.
Fetches data from all APIs concurrently using existing helpers.
Reduces API fetching from 30-60s sequential to 5-10s parallel.
"""

import asyncio
import logging
import os
import time
from types import TracebackType
from typing import Any, ClassVar

from enrichment.api_helpers.anidb_helper import AniDBEnrichmentHelper
from enrichment.api_helpers.anilist_helper import AniListHelper
from enrichment.api_helpers.anime_planet_helper import AnimePlanetEnrichmentHelper
from enrichment.api_helpers.animeschedule_helper import AnimescheduleHelper
from enrichment.api_helpers.anisearch_helper import AniSearchEnrichmentHelper
from enrichment.api_helpers.kitsu_helper import KitsuEnrichmentHelper
from enrichment.api_helpers.mal_helper import MalHelper
from enrichment.exceptions import (
    AniListGraphQLError,
    ServiceBlockedError,
    ServiceNetworkError,
    ServiceRateLimitedError,
)

from .config import EnrichmentConfig

logger = logging.getLogger(__name__)


class ApiFetcher:
    """Fetch data from all anime APIs in parallel.

    Implements graceful degradation — continues with partial data if individual APIs fail.
    """

    _REGISTRY: ClassVar[dict[str, type]] = {
        "anilist": AniListHelper,
        "kitsu": KitsuEnrichmentHelper,
        "anidb": AniDBEnrichmentHelper,
        "anime_planet": AnimePlanetEnrichmentHelper,
        "anisearch": AniSearchEnrichmentHelper,
        "animeschedule": AnimescheduleHelper,
        "mal": MalHelper,
    }

    def __init__(self, config: EnrichmentConfig | None = None):
        """Initialize the parallel API fetcher."""
        self.config = config or EnrichmentConfig()
        self._helpers: dict[str, Any] = {}

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

    async def initialize_helpers(
        self,
        skip_services: list[str] | None = None,
        only_services: list[str] | None = None,
    ) -> None:
        """Lazily initialize only the helpers for services that will actually be used."""
        for name, cls in self._REGISTRY.items():
            if name not in self._helpers and self._should_include(
                name, only_services, skip_services
            ):
                self._helpers[name] = cls()

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
        await self.initialize_helpers(
            skip_services=skip_services, only_services=only_services
        )

        start_time = time.time()
        tasks: list[tuple[str, Any]] = []

        # Log filtering info
        if only_services:
            logger.info(f"Only fetching services: {only_services}")
        elif skip_services:
            logger.info(f"Skipping services: {skip_services}")

        # Create parallel tasks for each registered helper
        for name, helper in self._helpers.items():
            tasks.append((name, self._fetch_service(name, helper, ids, offline_data, temp_dir)))

        # Execute all tasks in parallel
        results = await self._gather(tasks)

        elapsed = time.time() - start_time
        logger.info(f"Fetched all API data in {elapsed:.2f} seconds")

        # Log performance metrics (context-rich logging)
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
            
            # Special handling for AniList to run in executor if needed
            # (Keeping the original pattern for AniList as it was in a separate thread)
            if name == "anilist":
                result = await self._fetch_anilist_via_executor(helper, ids, offline_data, temp_dir)
            else:
                result = await helper.fetch_all(ids, offline_data, temp_dir)
                
            self.api_timings[name] = time.time() - start
            return result
        except Exception as e:
            logger.error(f"API {name} fetch failed: {e}")
            self.api_errors[name] = str(e)
            return None

    def _anilist_sync_wrapper(
        self, helper: Any, ids: dict[str, str], offline_data: dict[str, Any], temp_dir: str | None
    ) -> dict[str, Any] | None:
        """Synchronous AniList fetch wrapper for executor."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(helper.fetch_all(ids, offline_data, temp_dir))
        finally:
            loop.close()

    async def _fetch_anilist_via_executor(
        self, helper: Any, ids: dict[str, str], offline_data: dict[str, Any], temp_dir: str | None
    ) -> dict[str, Any] | None:
        """Run AniList fetch in a separate thread to prevent event loop blocking."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._anilist_sync_wrapper, helper, ids, offline_data, temp_dir
        )

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
        """Close all helpers."""
        for helper in self._helpers.values():
            if hasattr(helper, "close"):
                await helper.close()
        self._helpers.clear()
        return False
