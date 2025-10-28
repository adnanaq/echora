"""
Parallel API fetcher for anime enrichment.
Fetches data from all APIs concurrently using existing helpers.
Reduces API fetching from 30-60s sequential to 5-10s parallel.
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, cast

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.enrichment.api_helpers.anidb_helper import AniDBEnrichmentHelper
from src.enrichment.api_helpers.anilist_helper import AniListEnrichmentHelper
from src.enrichment.api_helpers.animeplanet_helper import AnimePlanetEnrichmentHelper
from src.enrichment.api_helpers.animeschedule_fetcher import fetch_animeschedule_data
from src.enrichment.api_helpers.anisearch_helper import AniSearchEnrichmentHelper
from src.enrichment.api_helpers.jikan_helper import JikanDetailedFetcher
from src.enrichment.api_helpers.kitsu_helper import KitsuEnrichmentHelper

from .config import EnrichmentConfig

logger = logging.getLogger(__name__)


class ParallelAPIFetcher:
    """
    Fetches data from all anime APIs in parallel.
    Implements graceful degradation - continues with partial data if APIs fail.
    """

    def __init__(self, config: EnrichmentConfig | None = None):
        """
        Initialize with configuration and API helpers.

        Args:
            config: Enrichment configuration (uses defaults if not provided)
        """
        self.config = config or EnrichmentConfig()

        # Initialize async helpers
        self.anilist_helper: AniListEnrichmentHelper | None = (
            None  # Lazy init in async context
        )
        self.kitsu_helper: KitsuEnrichmentHelper | None = None
        self.anidb_helper: AniDBEnrichmentHelper | None = None
        self.anime_planet_helper: AnimePlanetEnrichmentHelper | None = None
        self.anisearch_helper: AniSearchEnrichmentHelper | None = None

        # Track API performance
        self.api_timings: dict[str, float] = {}
        self.api_errors: dict[str, str] = {}

    async def initialize_helpers(self) -> None:
        """Initialize async API helpers."""
        if not self.anilist_helper:
            self.anilist_helper = AniListEnrichmentHelper()
        if not self.kitsu_helper:
            self.kitsu_helper = KitsuEnrichmentHelper()
        if not self.anidb_helper:
            self.anidb_helper = AniDBEnrichmentHelper()
        if not self.anime_planet_helper:
            self.anime_planet_helper = AnimePlanetEnrichmentHelper()
        if not self.anisearch_helper:
            self.anisearch_helper = AniSearchEnrichmentHelper()

    async def fetch_all_data(
        self,
        ids: dict[str, str],
        offline_data: dict[str, Any],
        temp_dir: str | None = None,
        skip_services: list[str] | None = None,
        only_services: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Fetch data from all APIs in parallel with optional service filtering.

        Args:
            ids: Dictionary of platform IDs
            offline_data: Original offline anime data
            temp_dir: Optional temp directory for saving responses
            skip_services: Optional list of service names to skip (e.g., ["jikan", "anidb"])
            only_services: Optional list of service names to fetch exclusively (e.g., ["anime_planet"])

        Returns:
            Dictionary with API responses

        Performance: 5-10 seconds (vs 30-60 seconds sequential)

        Note: skip_services and only_services are mutually exclusive.
              If both provided, only_services takes precedence.

        Available services: jikan, anilist, kitsu, anidb, anime_planet, anisearch, animeschedule
        """
        await self.initialize_helpers()

        start_time = time.time()
        tasks: list[tuple[str, Any]] = []

        # Helper to check if service should be included
        def should_include(service_name: str) -> bool:
            if only_services:
                return service_name in only_services
            if skip_services:
                return service_name not in skip_services
            return True

        # Log filtering info
        if only_services:
            logger.info(f"Only fetching services: {only_services}")
        elif skip_services:
            logger.info(f"Skipping services: {skip_services}")

        # Create parallel tasks for each API (only if not filtered out)
        if ids.get("mal_id") and should_include("jikan"):
            tasks.append(
                (
                    "jikan",
                    self._fetch_jikan_complete(ids["mal_id"], offline_data, temp_dir),
                )
            )

        if ids.get("anilist_id") and should_include("anilist"):
            tasks.append(("anilist", self._fetch_anilist(ids["anilist_id"], temp_dir)))

        if ids.get("kitsu_id") and should_include("kitsu"):
            tasks.append(("kitsu", self._fetch_kitsu(ids["kitsu_id"])))

        if ids.get("anidb_id") and should_include("anidb"):
            tasks.append(("anidb", self._fetch_anidb(ids["anidb_id"])))

        if ids.get("anime_planet_slug") and should_include("anime_planet"):
            tasks.append(("anime_planet", self._fetch_anime_planet(offline_data)))

        if ids.get("anisearch_id") and should_include("anisearch"):
            tasks.append(("anisearch", self._fetch_anisearch(ids["anisearch_id"])))

        # Always try AnimSchedule with title search (unless explicitly filtered)
        if should_include("animeschedule"):
            tasks.append(("animeschedule", self._fetch_animeschedule(offline_data)))

        # Execute all tasks in parallel with timeout
        results = await self._gather_with_timeout(
            tasks, timeout=self.config.api_timeout
        )

        # Save to temp files if directory provided
        if temp_dir:
            await self._save_temp_files(results, temp_dir)

        elapsed = time.time() - start_time
        logger.info(f"Fetched all API data in {elapsed:.2f} seconds")

        # Log performance metrics (context-rich logging)
        self._log_performance_metrics(elapsed)

        return results

    async def _fetch_jikan_complete(
        self, mal_id: str, offline_data: dict[str, Any], temp_dir: str | None = None
    ) -> dict[str, Any] | None:
        """
        Fetch ALL Jikan data using the JikanDetailedFetcher helper.
        This properly handles rate limiting and batch processing for large series.
        """
        try:
            logger.info(
                f"🔍 [JIKAN DEBUG] Starting _fetch_jikan_complete for MAL ID {mal_id}, temp_dir={temp_dir}"
            )
            start = time.time()
            loop = asyncio.get_event_loop()

            # First, fetch anime full data
            logger.info("🔍 [JIKAN DEBUG] Fetching anime full data...")
            anime_url = f"https://api.jikan.moe/v4/anime/{mal_id}/full"
            anime_data = await loop.run_in_executor(
                None, self._fetch_jikan_sync, anime_url
            )

            if not anime_data or not anime_data.get("data"):
                logger.warning(f"Failed to fetch Jikan anime data for MAL ID {mal_id}")
                return None

            logger.info("🔍 [JIKAN DEBUG] Anime full data fetched successfully")
            anime_info = anime_data["data"]

            # Save jikan.json immediately with the anime full data
            # This ensures we have the main anime info even if episode/character fetching times out
            if temp_dir:
                jikan_file = os.path.join(temp_dir, "jikan.json")
                with open(jikan_file, "w", encoding="utf-8") as f:
                    json.dump(anime_data, f, ensure_ascii=False, indent=2)
                logger.info(
                    f"🔍 [JIKAN DEBUG] ✓ Saved jikan.json with anime full data to {jikan_file}"
                )

            # For ongoing series, episodes might be None - use offline data as fallback
            episode_count = anime_info.get("episodes")
            if episode_count is None:
                episode_count = offline_data.get("episodes", 0)
            logger.info(f"🔍 [JIKAN DEBUG] Episode count: {episode_count}")

            # Fetch character list first (before starting detailed fetches)
            logger.info("🔍 [JIKAN DEBUG] Fetching character list...")
            characters_url = f"https://api.jikan.moe/v4/anime/{mal_id}/characters"
            characters_basic = await loop.run_in_executor(
                None, self._fetch_jikan_sync, characters_url
            )
            char_count = (
                len(characters_basic.get("data", [])) if characters_basic else 0
            )
            logger.info(
                f"🔍 [JIKAN DEBUG] Character list fetched: {char_count} characters"
            )

            # Prepare file paths
            if temp_dir is not None:
                episodes_input = os.path.join(temp_dir, "episodes.json")
                episodes_output = os.path.join(temp_dir, "episodes_detailed.json")
                characters_input = os.path.join(temp_dir, "characters.json")
                characters_output = os.path.join(temp_dir, "characters_detailed.json")

            # Create tasks for parallel execution
            tasks = []

            # Task 1: Fetch detailed episodes (if any)
            if episode_count and episode_count > 0:
                logger.info(
                    f"🔍 [JIKAN DEBUG] Preparing episode task for {episode_count} episodes..."
                )
                with open(episodes_input, "w") as f:
                    json.dump({"episodes": episode_count}, f)

                episode_fetcher = JikanDetailedFetcher(mal_id, "episodes")
                tasks.append(
                    (
                        "episodes",
                        loop.run_in_executor(
                            None,
                            episode_fetcher.fetch_detailed_data,
                            episodes_input,
                            episodes_output,
                        ),
                    )
                )

            # Task 2: Fetch detailed characters (if any)
            if characters_basic and characters_basic.get("data"):
                logger.info(
                    f"🔍 [JIKAN DEBUG] Preparing character task for {char_count} characters..."
                )
                with open(characters_input, "w") as f:
                    json.dump(characters_basic, f)

                character_fetcher = JikanDetailedFetcher(mal_id, "characters")
                tasks.append(
                    (
                        "characters",
                        loop.run_in_executor(
                            None,
                            character_fetcher.fetch_detailed_data,
                            characters_input,
                            characters_output,
                        ),
                    )
                )

            # Run both tasks in parallel
            if tasks:
                logger.info(
                    f"🔍 [JIKAN DEBUG] Running {len(tasks)} detailed fetch tasks in parallel..."
                )
                await asyncio.gather(*[task for _, task in tasks])
                logger.info("🔍 [JIKAN DEBUG] All detailed fetch tasks completed")

            # Load results
            episodes_data = []
            if os.path.exists(episodes_output):
                with open(episodes_output) as f:
                    episodes_data = json.load(f)
                logger.info(f"🔍 [JIKAN DEBUG] Loaded {len(episodes_data)} episodes")

            characters_data = []
            if os.path.exists(characters_output):
                with open(characters_output) as f:
                    characters_data = json.load(f)
                logger.info(
                    f"🔍 [JIKAN DEBUG] Loaded {len(characters_data)} characters"
                )
            elif characters_basic and characters_basic.get("data"):
                characters_data = characters_basic.get("data", [])
                logger.info(
                    f"🔍 [JIKAN DEBUG] Using basic character data: {len(characters_data)} characters"
                )

            result = {
                "anime": anime_info,
                "episodes": episodes_data if isinstance(episodes_data, list) else [],
                "characters": characters_data,
            }

            self.api_timings["jikan"] = time.time() - start
            logger.info(
                f"🔍 [JIKAN DEBUG] ✓ Jikan fetch complete: {len(result['episodes'])} episodes, {len(result['characters'])} characters in {self.api_timings['jikan']:.2f}s"
            )
            return result

        except Exception as e:
            logger.error(f"Jikan fetch failed for MAL ID {mal_id}: {e}")
            self.api_errors["jikan"] = str(e)
            return None

    def _fetch_jikan_sync(self, url: str) -> dict[str, Any] | None:
        """Synchronous Jikan API fetch with rate limiting."""
        import time

        import requests

        # Respect Jikan rate limits (3 req/sec)
        time.sleep(0.35)  # ~3 requests per second

        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return cast(dict[Any, Any], response.json())
            else:
                logger.warning(
                    f"Jikan API returned status {response.status_code} for {url}"
                )
                return None
        except Exception as e:
            logger.error(f"Jikan API request failed: {e}")
            return None

    def _fetch_anilist_sync(
        self, anilist_id: str, temp_dir: str | None = None
    ) -> dict[str, Any] | None:
        """Synchronous wrapper for AniList fetch - runs in executor to avoid cancellation."""
        try:
            start = time.time()

            logger.info(
                f"Fetching AniList data for ID {anilist_id} (will fetch ALL characters, staff, episodes)..."
            )

            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                # Create fresh AniList helper instance for this thread
                from src.enrichment.api_helpers.anilist_helper import (
                    AniListEnrichmentHelper,
                )

                anilist_helper = AniListEnrichmentHelper()

                result = loop.run_until_complete(
                    anilist_helper.fetch_all_data_by_anilist_id(int(anilist_id))
                )

                # Close the helper's session
                loop.run_until_complete(anilist_helper.close())
            finally:
                loop.close()

            elapsed = time.time() - start
            self.api_timings["anilist"] = elapsed

            if result:
                # Save anilist.json directly to ensure it persists even if timeout occurs
                if temp_dir:
                    anilist_file = os.path.join(temp_dir, "anilist.json")
                    with open(anilist_file, "w", encoding="utf-8") as f:
                        json.dump(result, f, ensure_ascii=False, indent=2)
                    logger.info(f"Saved anilist data to {anilist_file}")

                # Log what we got
                chars = len(result.get("characters", {}).get("edges", []))
                staff = len(result.get("staff", {}).get("edges", []))
                eps = len(result.get("airingSchedule", {}).get("edges", []))
                logger.info(
                    f"AniList fetched: {chars} characters, {staff} staff, {eps} episodes in {elapsed:.2f}s"
                )
            else:
                logger.warning(f"AniList returned no data for ID {anilist_id}")

            return result
        except Exception as e:
            logger.error(f"AniList fetch failed for ID {anilist_id}: {e}")
            self.api_errors["anilist"] = str(e)
            return None

    async def _fetch_anilist(
        self, anilist_id: str, temp_dir: str | None = None
    ) -> dict[str, Any] | None:
        """Fetch ALL AniList data in executor to prevent timeout cancellation."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._fetch_anilist_sync, anilist_id, temp_dir
        )

    async def _fetch_kitsu(self, kitsu_id: str) -> dict[str, Any] | None:
        """Fetch Kitsu data using async helper."""
        try:
            start = time.time()

            # Check if it's numeric or slug
            try:
                # Try as numeric ID first
                numeric_id = int(kitsu_id)
                if self.kitsu_helper is None:
                    raise RuntimeError("Kitsu helper not initialized")
                result = await self.kitsu_helper.fetch_all_data(numeric_id)
            except ValueError:
                # If not numeric, it's a slug - need to resolve to ID first
                logger.info(f"Resolving Kitsu slug '{kitsu_id}' to numeric ID...")

                # Use Kitsu API to search by slug
                import aiohttp

                async with aiohttp.ClientSession() as session:
                    url = f"https://kitsu.io/api/edge/anime?filter[slug]={kitsu_id}"
                    async with session.get(
                        url, timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data.get("data"):
                                numeric_id = int(data["data"][0]["id"])
                                logger.info(
                                    f"Resolved slug '{kitsu_id}' to ID {numeric_id}"
                                )
                                if self.kitsu_helper is None:
                                    raise RuntimeError("Kitsu helper not initialized")
                                result = await self.kitsu_helper.fetch_all_data(
                                    numeric_id
                                )
                            else:
                                logger.warning(
                                    f"No Kitsu anime found for slug: {kitsu_id}"
                                )
                                result = None
                        else:
                            logger.warning(
                                f"Failed to resolve Kitsu slug: {response.status}"
                            )
                            result = None

            self.api_timings["kitsu"] = time.time() - start
            return result
        except Exception as e:
            logger.error(f"Kitsu fetch failed for ID {kitsu_id}: {e}")
            self.api_errors["kitsu"] = str(e)
            return None

    async def _fetch_anidb(self, anidb_id: str) -> dict[str, Any] | None:
        """Fetch AniDB data using async helper."""
        try:
            start = time.time()
            if self.anidb_helper is None:
                raise RuntimeError("AniDB helper not initialized")
            result = await self.anidb_helper.fetch_all_data(int(anidb_id))
            self.api_timings["anidb"] = time.time() - start
            return result
        except Exception as e:
            logger.error(f"AniDB fetch failed for ID {anidb_id}: {e}")
            self.api_errors["anidb"] = str(e)
            return None

    async def _fetch_anime_planet(
        self, offline_data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Fetch Anime-Planet data using scraper."""
        try:
            start = time.time()
            if self.anime_planet_helper is None:
                raise RuntimeError("Anime Planet helper not initialized")
            result = await self.anime_planet_helper.fetch_all_data(offline_data)
            self.api_timings["anime_planet"] = time.time() - start
            return result
        except Exception as e:
            logger.error(f"Anime-Planet fetch failed: {e}")
            self.api_errors["anime_planet"] = str(e)
            return None

    async def _fetch_anisearch(self, anisearch_id: str) -> dict[str, Any] | None:
        """Fetch AniSearch data using scraper."""
        try:
            start = time.time()
            if self.anisearch_helper is None:
                raise RuntimeError("AniSearch helper not initialized")
            result = await self.anisearch_helper.fetch_all_data(int(anisearch_id))
            self.api_timings["anisearch"] = time.time() - start
            return result
        except Exception as e:
            logger.error(f"AniSearch fetch failed for ID {anisearch_id}: {e}")
            self.api_errors["anisearch"] = str(e)
            return None

    async def _fetch_animeschedule(
        self, offline_data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """
        Fetch AnimSchedule data using sync helper.
        Note: AnimSchedule helper is sync, so we run in executor.
        """
        try:
            start = time.time()

            # Get search term from offline data
            search_term = offline_data.get("title", "")
            if not search_term:
                return None

            # Run sync function in executor
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, fetch_animeschedule_data, search_term
            )

            self.api_timings["animeschedule"] = time.time() - start
            return result

        except Exception as e:
            logger.error(f"AnimSchedule fetch failed: {e}")
            self.api_errors["animeschedule"] = str(e)
            return None

    async def _gather_with_timeout(
        self, tasks: list[tuple[str, Any]], timeout: int
    ) -> dict[str, Any]:
        """
        Execute tasks in parallel. In no-timeout mode, wait for ALL data.
        Implements graceful degradation - doesn't fail if one API is down.

        Args:
            tasks: List of (name, coroutine) tuples
            timeout: Timeout in seconds (ignored if no_timeout_mode is True)

        Returns:
            Dictionary of results
        """
        results = {}

        # Create tasks with names
        named_tasks = []
        for name, coro in tasks:
            task = asyncio.create_task(coro)
            named_tasks.append((name, task))

        # Check if we're in no-timeout mode
        if self.config.no_timeout_mode:
            logger.info(
                "Running in NO TIMEOUT mode - will fetch ALL data from all APIs"
            )
            # Just gather all results without timeout
            for name, task in named_tasks:
                try:
                    result = await task
                    results[name] = result

                    if result:
                        logger.debug(f"API {name} completed successfully")
                    else:
                        logger.warning(f"API {name} returned empty result")

                except Exception as e:
                    logger.error(f"API {name} failed with error: {e}")
                    results[name] = None
                    self.api_errors[name] = str(e)
        else:
            # Normal mode with timeouts
            for name, task in named_tasks:
                try:
                    # Each API gets its own timeout
                    result = await asyncio.wait_for(task, timeout=timeout)
                    results[name] = result

                    if result:
                        logger.debug(f"API {name} completed successfully")
                    else:
                        logger.warning(f"API {name} returned empty result")

                except TimeoutError:
                    logger.warning(f"API {name} timed out after {timeout}s")
                    results[name] = None
                    task.cancel()

                except Exception as e:
                    logger.error(f"API {name} failed with error: {e}")
                    results[name] = None
                    self.api_errors[name] = str(e)

        return results

    async def _save_temp_files(self, results: dict[str, Any], temp_dir: str) -> None:
        """Save API responses to temp files for debugging/caching."""
        os.makedirs(temp_dir, exist_ok=True)

        for api_name, data in results.items():
            if data:
                file_path = os.path.join(temp_dir, f"{api_name}.json")
                try:
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    logger.debug(f"Saved {api_name} response to {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to save {api_name} response: {e}")

    def _log_performance_metrics(self, total_time: float) -> None:
        """Log detailed performance metrics for optimization."""
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

    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.anilist_helper:
            await self.anilist_helper.close()
        if self.anisearch_helper:
            await self.anisearch_helper.close()
        # Add cleanup for other helpers as needed
