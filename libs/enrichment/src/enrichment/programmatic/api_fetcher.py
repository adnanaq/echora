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
from enrichment.api_helpers.anilist_helper import AniListEnrichmentHelper
from enrichment.api_helpers.animeplanet_helper import AnimePlanetEnrichmentHelper
from enrichment.api_helpers.animeschedule_helper import AnimescheduleEnrichmentHelper
from enrichment.api_helpers.anisearch_helper import AniSearchEnrichmentHelper
from enrichment.api_helpers.kitsu_helper import KitsuEnrichmentHelper
from enrichment.api_helpers.mal_helper import MalEnrichmentHelper
from enrichment.exceptions import (
    AniListGraphQLError,
    ServiceBlockedError,
    ServiceNetworkError,
    ServiceRateLimitedError,
)

from .config import EnrichmentConfig

logger = logging.getLogger(__name__)


class ParallelAPIFetcher:
    """
    Fetches data from all anime APIs in parallel.
    Implements graceful degradation - continues with partial data if APIs fail.
    """

    _REGISTRY: ClassVar[dict[str, type]] = {
        "anilist": AniListEnrichmentHelper,
        "kitsu": KitsuEnrichmentHelper,
        "anidb": AniDBEnrichmentHelper,
        "anime_planet": AnimePlanetEnrichmentHelper,
        "anisearch": AniSearchEnrichmentHelper,
        "animeschedule": AnimescheduleEnrichmentHelper,
    }

    def __init__(self, config: EnrichmentConfig | None = None):
        self.config = config or EnrichmentConfig()
        self._helpers: dict[str, Any] = {}
        self.mal_session: Any | None = None

        # Track API performance
        self.api_timings: dict[str, float] = {}
        self.api_errors: dict[str, str] = {}

    @staticmethod
    def _should_include(
        service: str,
        only: list[str] | None,
        skip: list[str] | None,
    ) -> bool:
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
        if (
            self._should_include("mal", only_services, skip_services)
            and not self.mal_session
        ):
            from http_cache.instance import http_cache_manager as cache_manager

            self.mal_session = cache_manager.get_aiohttp_session("mal")

    async def fetch_all_data(
        self,
        ids: dict[str, str],
        offline_data: dict[str, Any],
        temp_dir: str | None = None,
        skip_services: list[str] | None = None,
        only_services: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Fetch data from multiple anime APIs concurrently, allowing optional service filtering.

        Parameters:
            ids (Dict[str, str]): Mapping of platform keys to their IDs or slugs (e.g., "mal_id", "anilist_id", "kitsu_id", "anidb_id", "anime_planet_slug", "anisearch_id").
            offline_data (Dict): Original offline anime metadata used as input or fallback by some fetchers (e.g., title, episode count).
            temp_dir (Optional[str]): Directory path where per-service JSON responses will be saved when provided.
            skip_services (Optional[List[str]]): Services to omit from fetching. Ignored if `only_services` is provided.
            only_services (Optional[List[str]]): If provided, only these services will be fetched; takes precedence over `skip_services`.

        Returns:
            Dict[str, Any]: Mapping of service names to their fetched result object, or `None` for services that failed or timed out.
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

        # Create parallel tasks for each API (only if not filtered out)
        if ids.get("mal_id") and self._should_include(
            "mal", only_services, skip_services
        ):
            tasks.append(("mal", self._fetch_mal(ids["mal_id"], temp_dir)))

        if ids.get("anilist_id") and self._should_include(
            "anilist", only_services, skip_services
        ):
            tasks.append(("anilist", self._fetch_anilist(ids["anilist_id"], temp_dir)))

        if ids.get("kitsu_id") and self._should_include(
            "kitsu", only_services, skip_services
        ):
            tasks.append(("kitsu", self._fetch_kitsu(ids["kitsu_id"], temp_dir)))

        if ids.get("anidb_id") and self._should_include(
            "anidb", only_services, skip_services
        ):
            tasks.append(("anidb", self._fetch_anidb(ids["anidb_id"], temp_dir)))

        if ids.get("anime_planet_slug") and self._should_include(
            "anime_planet", only_services, skip_services
        ):
            tasks.append(
                ("anime_planet", self._fetch_anime_planet(offline_data, temp_dir))
            )

        if ids.get("anisearch_id") and self._should_include(
            "anisearch", only_services, skip_services
        ):
            tasks.append(
                ("anisearch", self._fetch_anisearch(ids["anisearch_id"], temp_dir))
            )

        # Always try AnimSchedule with title search (unless explicitly filtered)
        if self._should_include("animeschedule", only_services, skip_services):
            tasks.append(
                ("animeschedule", self._fetch_animeschedule(offline_data, temp_dir))
            )

        # Execute all tasks in parallel
        results = await self._gather(tasks)

        elapsed = time.time() - start_time
        logger.info(f"Fetched all API data in {elapsed:.2f} seconds")

        # Log performance metrics (context-rich logging)
        self._log_performance_metrics(elapsed)

        return results

    async def _fetch_mal(
        self, mal_id: str, temp_dir: str | None = None
    ) -> dict[str, Any] | None:
        """
        Fetch comprehensive MAL data for a given MyAnimeList ID, including full anime info, episodes, and characters, writing intermediate files to temp_dir when provided.

        Parameters:
            mal_id (str): The MyAnimeList identifier for the anime.
            temp_dir (Optional[str]): Directory path to write/read temporary JSON files for intermediate results.

        Returns:
            Optional[Dict[str, Any]]: A dictionary with keys "anime" (anime info), "episodes" (list of episode dicts), and "characters" (list of character dicts); returns `None` on failure.
        """
        try:
            start = time.time()

            # Build streaming output paths (None when no temp_dir)
            anime_path = os.path.join(temp_dir, "mal_anime.jsonl") if temp_dir else None
            episodes_path = (
                os.path.join(temp_dir, "mal_episodes.jsonl") if temp_dir else None
            )
            characters_path = (
                os.path.join(temp_dir, "mal_characters.jsonl") if temp_dir else None
            )

            async with MalEnrichmentHelper(mal_id, session=self.mal_session) as helper:
                result = await helper.fetch_all(
                    anime_output_path=anime_path,
                    episodes_output_path=episodes_path,
                    characters_output_path=characters_path,
                )
                if not result:
                    logger.warning(
                        f"Failed to fetch MAL anime data for MAL ID {mal_id}"
                    )
                    return None

            self.api_timings["mal"] = time.time() - start
            return result

        except Exception as e:
            logger.error(f"MAL fetch failed for MAL ID {mal_id}: {e}")
            self.api_errors["mal"] = str(e)
            return None

    def _fetch_anilist_sync(
        self, anilist_id: str, temp_dir: str | None = None
    ) -> dict[str, Any] | None:
        """Synchronous wrapper for AniList fetch - runs in executor to avoid cancellation."""
        try:
            start = time.time()

            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                # Create fresh AniList helper instance for this thread
                from enrichment.api_helpers.anilist_helper import (
                    AniListEnrichmentHelper,
                )

                anilist_helper = AniListEnrichmentHelper()

                result, chars_list = loop.run_until_complete(
                    anilist_helper.fetch_all(int(anilist_id), temp_dir)
                )

                # Close the helper's session
                loop.run_until_complete(anilist_helper.close())
            finally:
                loop.close()

            elapsed = time.time() - start
            self.api_timings["anilist"] = elapsed

            if result:
                logger.info(
                    f"AniList fetched: {len(chars_list)} characters in {elapsed:.2f}s"
                )
            else:
                logger.warning(f"AniList returned no data for ID {anilist_id}")

            return result
        except ServiceRateLimitedError as e:
            logger.error(f"AniList rate limit exhausted for ID {anilist_id}: {e}")  # noqa: TRY400
            self.api_errors["anilist"] = str(e)
            return None
        except ServiceBlockedError as e:
            logger.warning(f"AniList blocked/disabled for ID {anilist_id}: {e}")
            self.api_errors["anilist"] = str(e)
            # TODO: publish a deferred retry event via NATS JetStream / Temporal
            #   so this enrichment job is retried once AniList recovers.
            #   ServiceBlockedError signals "try again later" — not a permanent failure.
            return None
        except AniListGraphQLError as e:
            logger.error(f"AniList GraphQL error for ID {anilist_id}: {e}")  # noqa: TRY400
            self.api_errors["anilist"] = str(e)
            return None
        except ServiceNetworkError as e:
            logger.error(f"AniList network error for ID {anilist_id}: {e}")  # noqa: TRY400
            self.api_errors["anilist"] = str(e)
            return None
        except Exception as e:
            logger.error(f"AniList fetch failed for ID {anilist_id}: {e}")
            self.api_errors["anilist"] = str(e)
            return None

    async def _fetch_anilist(
        self, anilist_id: str, temp_dir: str | None = None
    ) -> dict[str, Any] | None:
        """Fetch ALL AniList data in executor to prevent timeout cancellation."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._fetch_anilist_sync, anilist_id, temp_dir
        )

    async def _fetch_kitsu(
        self, kitsu_id: str, temp_dir: str | None = None
    ) -> dict[str, Any] | None:
        """Fetch canonical Kitsu data (anime + episodes + characters)."""
        try:
            start = time.time()
            helper = self._helpers.get("kitsu")
            if helper is None:
                raise RuntimeError("Kitsu helper not initialized")

            try:
                numeric_id = int(kitsu_id)
            except ValueError:
                # Slug — resolve to numeric ID first
                logger.info(f"Resolving Kitsu slug '{kitsu_id}' to numeric ID...")
                import aiohttp as _aiohttp

                async with _aiohttp.ClientSession() as slug_session:
                    url = f"https://kitsu.io/api/edge/anime?filter[slug]={kitsu_id}"
                    async with slug_session.get(
                        url, timeout=_aiohttp.ClientTimeout(total=5)
                    ) as response:
                        if response.status != 200:
                            logger.warning(
                                f"Failed to resolve Kitsu slug: {response.status}"
                            )
                            return None
                        data = await response.json()
                        if not data.get("data"):
                            logger.warning(f"No Kitsu anime found for slug: {kitsu_id}")
                            return None
                        numeric_id = int(data["data"][0]["id"])
                        logger.info(f"Resolved slug '{kitsu_id}' to ID {numeric_id}")

            result = await helper.fetch_all(numeric_id, output_dir=temp_dir)
            self.api_timings["kitsu"] = time.time() - start
            return result
        except Exception as e:
            logger.error(f"Kitsu fetch failed for ID {kitsu_id}: {e}")
            self.api_errors["kitsu"] = str(e)
            return None

    async def _fetch_anidb(
        self, anidb_id: str, temp_dir: str | None = None
    ) -> dict[str, Any] | None:
        """Fetch AniDB data using async helper."""
        try:
            start = time.time()
            helper = self._helpers.get("anidb")
            if helper is None:
                raise RuntimeError("AniDB helper not initialized")
            output_path = os.path.join(temp_dir, "anidb.jsonl") if temp_dir else None
            result = await helper.fetch_all(int(anidb_id), output_path=output_path)
            self.api_timings["anidb"] = time.time() - start
            return result
        except Exception as e:
            logger.error(f"AniDB fetch failed for ID {anidb_id}: {e}")
            self.api_errors["anidb"] = str(e)
            return None

    async def _fetch_anime_planet(
        self, offline_data: dict[str, Any], temp_dir: str | None = None
    ) -> dict[str, Any] | None:
        """Fetch Anime-Planet data using scraper."""
        try:
            start = time.time()
            helper = self._helpers.get("anime_planet")
            if helper is None:
                raise RuntimeError("Anime Planet helper not initialized")
            output_path = (
                os.path.join(temp_dir, "anime_planet.jsonl") if temp_dir else None
            )
            result = await helper.fetch_all(offline_data, output_path=output_path)
            self.api_timings["anime_planet"] = time.time() - start
            return result
        except Exception as e:
            logger.error(f"Anime-Planet fetch failed: {e}")
            self.api_errors["anime_planet"] = str(e)
            return None

    async def _fetch_anisearch(
        self, anisearch_id: str, temp_dir: str | None = None
    ) -> dict[str, Any] | None:
        """Fetch AniSearch data using scraper."""
        try:
            start = time.time()
            helper = self._helpers.get("anisearch")
            if helper is None:
                raise RuntimeError("AniSearch helper not initialized")
            output_path = (
                os.path.join(temp_dir, "anisearch.jsonl") if temp_dir else None
            )
            result = await helper.fetch_all(int(anisearch_id), output_path=output_path)
            self.api_timings["anisearch"] = time.time() - start
            return result
        except Exception as e:
            logger.error(f"AniSearch fetch failed for ID {anisearch_id}: {e}")
            self.api_errors["anisearch"] = str(e)
            return None

    async def _fetch_animeschedule(
        self, offline_data: dict[str, Any], temp_dir: str | None = None
    ) -> dict[str, Any] | None:
        """Fetch AnimSchedule data for the anime in offline_data.

        Passes known source URLs for cross-source validation so the correct
        result is selected even when the search returns multiple candidates.
        Writes the mapped canonical dict as JSONL directly (like MAL) when
        temp_dir is provided, so _save_temp_files can skip it.
        """
        try:
            start = time.time()

            helper = self._helpers.get("animeschedule")
            if helper is None:
                raise RuntimeError("AnimSchedule helper not initialized")

            search_term = offline_data.get("title", "")
            if not search_term:
                return None

            output_path = (
                os.path.join(temp_dir, "animeschedule.jsonl") if temp_dir else None
            )
            sources: list[str] = offline_data.get("sources", [])
            result = await helper.fetch_all(
                search_term, sources=sources or None, output_path=output_path
            )

            self.api_timings["animeschedule"] = time.time() - start
            return result

        except Exception as e:
            logger.error(f"AnimSchedule fetch failed: {e}")
            self.api_errors["animeschedule"] = str(e)
            return None

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
        """
        Log API performance metrics including total runtime, per-service timings, recorded errors, and a computed success rate.

        Parameters:
            total_time (float): Total elapsed time in seconds for the overall parallel fetch operation.

        Description:
            - Writes an informational header with the total elapsed time.
            - Writes per-API timing entries from `self.api_timings`.
            - Writes warnings for any entries in `self.api_errors`.
            - Computes and logs a simple success rate based on the number of timed APIs versus total recorded APIs (timings + errors).
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

    async def __aenter__(self) -> "ParallelAPIFetcher":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        for helper in self._helpers.values():
            await helper.close()
        self._helpers.clear()
        if self.mal_session:
            await self.mal_session.close()
            self.mal_session = None
        return False
