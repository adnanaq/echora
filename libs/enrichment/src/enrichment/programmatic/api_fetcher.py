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
from types import TracebackType
from typing import Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from enrichment.api_helpers.anidb_helper import AniDBEnrichmentHelper
from enrichment.api_helpers.anilist_helper import AniListEnrichmentHelper
from enrichment.api_helpers.animeplanet_helper import AnimePlanetEnrichmentHelper
from enrichment.api_helpers.animeschedule_fetcher import fetch_animeschedule_data
from enrichment.api_helpers.anisearch_helper import AniSearchEnrichmentHelper
from enrichment.api_helpers.jikan_helper import JikanDetailedFetcher
from enrichment.api_helpers.kitsu_helper import KitsuEnrichmentHelper

from .config import EnrichmentConfig

logger = logging.getLogger(__name__)


class ParallelAPIFetcher:
    """
    Fetches data from all anime APIs in parallel.
    Implements graceful degradation - continues with partial data if APIs fail.
    """

    def __init__(self, config: EnrichmentConfig | None = None):
        """
        Initialize the ParallelAPIFetcher with an optional configuration and prepare lazy helper placeholders and performance trackers.

        Parameters:
            config (Optional[EnrichmentConfig]): Enrichment configuration to use; if not provided, a default EnrichmentConfig() is created.
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
        self.jikan_session: Any | None = None

        # Track API performance
        self.api_timings: dict[str, float] = {}
        self.api_errors: dict[str, str] = {}

    async def initialize_helpers(self) -> None:
        """
        Lazily initialize enrichment helpers and a shared cached aiohttp session for Jikan.

        This prepares helper instances for AniList, Kitsu, AniDB, Anime-Planet, and AniSearch only if they are not already created, and obtains a shared cached aiohttp session for Jikan requests to enable connection pooling and reuse.
        """
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
        if not self.jikan_session:
            from http_cache.instance import http_cache_manager as cache_manager

            self.jikan_session = cache_manager.get_aiohttp_session("jikan")

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
        Fetch comprehensive Jikan data for a given MyAnimeList ID, including full anime info, episodes, and characters, writing intermediate files to temp_dir when provided.

        Fetches the anime "full" endpoint, attempts to retrieve detailed episodes and character data (respecting rate limits), falls back to offline_data for missing episode counts, and saves/loads temporary JSON files if temp_dir is supplied.

        Parameters:
            mal_id (str): The MyAnimeList identifier for the anime.
            offline_data (Dict): Local/offline metadata used as fallback (e.g., episode count).
            temp_dir (Optional[str]): Directory path to write/read temporary JSON files for intermediate results.

        Returns:
            Optional[Dict[str, Any]]: A dictionary with keys "anime" (anime info), "episodes" (list of episode dicts), and "characters" (list of character dicts); returns `None` on failure.
        """
        try:
            logger.info(
                f"🔍 [JIKAN DEBUG] Starting _fetch_jikan_complete for MAL ID {mal_id}, temp_dir={temp_dir}"
            )
            start = time.time()

            # First, fetch anime full data
            logger.info("🔍 [JIKAN DEBUG] Fetching anime full data...")
            anime_url = f"https://api.jikan.moe/v4/anime/{mal_id}/full"
            anime_data = await self._fetch_jikan_async(anime_url)

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
            characters_basic = await self._fetch_jikan_async(characters_url)
            char_count = (
                len(characters_basic.get("data", [])) if characters_basic else 0
            )
            logger.info(
                f"🔍 [JIKAN DEBUG] Character list fetched: {char_count} characters"
            )

            # Prepare file path variables (only set when temp_dir is provided)
            episodes_input: str | None = None
            episodes_output: str | None = None
            characters_input: str | None = None
            characters_output: str | None = None
            if temp_dir:
                episodes_input = os.path.join(temp_dir, "episodes.json")
                episodes_output = os.path.join(temp_dir, "episodes_detailed.json")
                characters_input = os.path.join(temp_dir, "characters.json")
                characters_output = os.path.join(temp_dir, "characters_detailed.json")

            # Create tasks for parallel execution
            tasks = []

            # Task 1: Fetch detailed episodes (if any)
            if (
                episode_count
                and episode_count > 0
                and episodes_input is not None
                and episodes_output is not None
            ):
                logger.info(
                    f"🔍 [JIKAN DEBUG] Preparing episode task for {episode_count} episodes..."
                )
                with open(episodes_input, "w") as f:
                    json.dump({"episodes": episode_count}, f)

                # Reuse the shared jikan session for caching and connection pooling
                episode_fetcher = JikanDetailedFetcher(
                    mal_id, "episodes", session=self.jikan_session
                )
                tasks.append(
                    (
                        "episodes",
                        episode_fetcher.fetch_detailed_data(
                            episodes_input,
                            episodes_output,
                        ),
                    )
                )

            # Task 2: Fetch detailed characters (if any)
            if (
                characters_basic
                and characters_basic.get("data")
                and characters_input is not None
                and characters_output is not None
            ):
                logger.info(
                    f"🔍 [JIKAN DEBUG] Preparing character task for {char_count} characters..."
                )
                with open(characters_input, "w") as f:
                    json.dump(characters_basic, f)

                # Reuse the shared jikan session for caching and connection pooling
                character_fetcher = JikanDetailedFetcher(
                    mal_id, "characters", session=self.jikan_session
                )
                tasks.append(
                    (
                        "characters",
                        character_fetcher.fetch_detailed_data(
                            characters_input,
                            characters_output,
                        ),
                    )
                )

            # Run tasks SEQUENTIALLY to respect Jikan's 3 req/sec limit
            # Running in parallel would double the request rate and trigger 429 errors
            if tasks:
                logger.info(
                    f"🔍 [JIKAN DEBUG] Running {len(tasks)} detailed fetch tasks SEQUENTIALLY (to avoid rate limiting)..."
                )
                for task_name, task_coro in tasks:
                    logger.info(f"🔍 [JIKAN DEBUG] Starting {task_name} fetch...")
                    await task_coro
                    logger.info(f"🔍 [JIKAN DEBUG] Completed {task_name} fetch")
                logger.info("🔍 [JIKAN DEBUG] All detailed fetch tasks completed")

            # Load results
            episodes_data = []
            if episodes_output and os.path.exists(episodes_output):
                with open(episodes_output) as f:
                    episodes_data = json.load(f)
                logger.info(f"🔍 [JIKAN DEBUG] Loaded {len(episodes_data)} episodes")

            characters_data = []
            if characters_output and os.path.exists(characters_output):
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
            logger.exception(f"Jikan fetch failed for MAL ID {mal_id}")
            self.api_errors["jikan"] = str(e)
            return None

    async def _fetch_all_jikan_episodes(
        self, mal_id: str, episode_count: int
    ) -> list[dict[str, Any]]:
        """
        Retrieve all episodes for a Jikan anime entry by following the paginated episodes endpoint until no more pages remain.

        Parameters:
                mal_id (str): MyAnimeList ID for the anime used to build the Jikan endpoint URL.
                episode_count (int): Expected total number of episodes; used for progress logging and may be 0.

        Returns:
                episodes (List[Dict]): Aggregated list of episode objects as returned by the Jikan episodes endpoint.
        """
        if episode_count == 0:
            return []

        all_episodes = []
        page = 1

        while True:
            url = f"https://api.jikan.moe/v4/anime/{mal_id}/episodes?page={page}"
            data = await self._fetch_jikan_async(url)

            if not data or not data.get("data"):
                break

            all_episodes.extend(data["data"])

            # Check if there are more pages
            pagination = data.get("pagination", {})
            if not pagination.get("has_next_page", False):
                break

            page += 1

            # Log progress for long-running series
            if len(all_episodes) % 100 == 0:
                logger.debug(f"Fetched {len(all_episodes)}/{episode_count} episodes...")

        return all_episodes

    async def _fetch_all_jikan_characters(self, mal_id: str) -> list[dict[str, Any]]:
        """
        Retrieve the complete list of characters for a MyAnimeList anime from the Jikan API.

        Parameters:
            mal_id (str): The MyAnimeList anime identifier.

        Returns:
            List[Dict]: A list of character objects as returned by Jikan; returns an empty list if no character data is available.
        """
        all_characters = []

        # First, get initial page to see how many there are
        url = f"https://api.jikan.moe/v4/anime/{mal_id}/characters"
        data = await self._fetch_jikan_async(url)

        if data and data.get("data"):
            all_characters.extend(data["data"])

            # Jikan v4 doesn't paginate characters endpoint directly
            # It returns all characters in one response
            # But we should verify we got them all
            logger.debug(f"Fetched {len(all_characters)} characters from Jikan")

        return all_characters

    async def _fetch_jikan_async(self, url: str) -> dict[str, Any] | None:
        """
        Fetches JSON data from a Jikan API URL using a cached aiohttp session.

        Initializes and reuses an internal cached HTTP session if needed. On HTTP 429 the request is retried once after a 2 second delay. Any non-200 response or exceptions result in `None`.

        Returns:
            dict: Parsed JSON response on success.
            None: If the request fails, is rate-limited beyond the single retry, or returns a non-200 status.
        """
        try:
            # Use the reusable cached session (initialized in initialize_helpers)
            if not self.jikan_session:
                from http_cache.instance import (
                    http_cache_manager as cache_manager,
                )

                self.jikan_session = cache_manager.get_aiohttp_session("jikan")

            async with self.jikan_session.get(url, timeout=10) as response:
                if response.status == 200:
                    result: dict[str, Any] = await response.json()
                    return result
                elif response.status == 429:
                    # Rate limited - wait and retry once
                    logger.warning(
                        f"Jikan rate limited for {url}, waiting 2s and retrying..."
                    )
                    await asyncio.sleep(2)
                    async with self.jikan_session.get(
                        url, timeout=10
                    ) as retry_response:
                        if retry_response.status == 200:
                            retry_result: dict[str, Any] = await retry_response.json()
                            return retry_result
                        else:
                            logger.error(f"Retry failed: HTTP {retry_response.status}")
                            return None
                else:
                    logger.warning(
                        f"Jikan API returned status {response.status} for {url}"
                    )
                    return None
        except Exception:
            logger.exception("Jikan API request failed")
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
                from enrichment.api_helpers.anilist_helper import (
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
            logger.exception(f"AniList fetch failed for ID {anilist_id}")
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
            logger.exception(f"Kitsu fetch failed for ID {kitsu_id}")
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
            logger.exception(f"AniDB fetch failed for ID {anidb_id}")
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
            logger.exception("Anime-Planet fetch failed: ")
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
            logger.exception(f"AniSearch fetch failed for ID {anisearch_id}")
            self.api_errors["anisearch"] = str(e)
            return None

    async def _fetch_animeschedule(
        self, offline_data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """
        Fetches anime schedule information based on the title found in the provided offline data.

        Parameters:
            offline_data (Dict): A dictionary containing at least a "title" key whose value is used as the search term.

        Returns:
            Dict: Schedule data returned by the AnimSchedule lookup, or `None` if no title is available or the lookup fails.
        """
        try:
            start = time.time()

            # Get search term from offline data
            search_term = offline_data.get("title", "")
            if not search_term:
                return None

            # Call async function directly
            result = await fetch_animeschedule_data(search_term)

            self.api_timings["animeschedule"] = time.time() - start
            return result

        except Exception as e:
            logger.exception("AnimSchedule fetch failed: ")
            self.api_errors["animeschedule"] = str(e)
            return None

    async def _gather_with_timeout(
        self, tasks: list[tuple[str, Any]], timeout: int
    ) -> dict[str, Any]:
        """
        Run multiple named coroutines concurrently and collect their results with optional per-task timeouts and graceful degradation.

        If a coroutine raises an exception or times out, its entry in the returned mapping will be None and the error string will be recorded in self.api_errors. When self.config.no_timeout_mode is True, the timeout parameter is ignored and all coroutines are allowed to complete.

        Parameters:
            tasks (List[Tuple[str, Any]]): List of (name, coroutine) pairs identifying each task.
            timeout (int): Per-task timeout in seconds (ignored when no_timeout_mode is enabled).

        Returns:
            Dict[str, Any]: Mapping from task name to the coroutine result, or None for timed-out/failed tasks.
        """
        results: dict[str, Any] = {}
        task_names = [name for name, _ in tasks]
        coroutines = [coro for _, coro in tasks]

        if self.config.no_timeout_mode:
            logger.info(
                "Running in NO TIMEOUT mode - will fetch ALL data from all APIs"
            )
            # Gather all results without timeout, returning exceptions for failed tasks
            task_results = await asyncio.gather(*coroutines, return_exceptions=True)

        else:
            # Normal mode with individual timeouts for each task
            # Wrap each coroutine in wait_for to apply the timeout
            timed_coroutines = [
                asyncio.wait_for(coro, timeout=timeout) for coro in coroutines
            ]
            task_results = await asyncio.gather(
                *timed_coroutines, return_exceptions=True
            )

        # Process the results from gather
        for i, result in enumerate(task_results):
            name = task_names[i]
            if isinstance(result, asyncio.TimeoutError):
                logger.warning(f"API {name} timed out after {timeout}s")
                results[name] = None
                self.api_errors[name] = f"Timeout after {timeout}s"
                # The task is already cancelled by wait_for, so no need to call cancel()
            elif isinstance(result, Exception):
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
        """
        Initialize internal helpers and return the fetcher instance for use as an async context manager.

        Returns:
            ParallelAPIFetcher: The same fetcher instance with helpers initialized.
        """
        await self.initialize_helpers()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        """
        Cleanup and close all initialized helper resources when exiting the async context.

        Closes AniList, AniDB, AniSearch, Kitsu, Anime-Planet helpers and the shared Jikan session if they were created.
        Resets all attributes to None for safe reusability of the fetcher instance.

        Returns:
            bool: `False` to indicate exceptions (if any) should not be suppressed.
        """
        # Close all helpers and reset to None for safe reusability
        if self.anilist_helper:
            await self.anilist_helper.close()
            self.anilist_helper = None
        if self.anidb_helper:
            await self.anidb_helper.close()
            self.anidb_helper = None
        if self.anisearch_helper:
            await self.anisearch_helper.close()
            self.anisearch_helper = None
        if self.kitsu_helper:
            await self.kitsu_helper.close()
            self.kitsu_helper = None
        if self.anime_planet_helper:
            await self.anime_planet_helper.close()
            self.anime_planet_helper = None
        if self.jikan_session:
            await self.jikan_session.close()
            self.jikan_session = None
        return False
