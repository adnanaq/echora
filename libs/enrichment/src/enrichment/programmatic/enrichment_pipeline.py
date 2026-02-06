"""
Main programmatic enrichment pipeline.
Orchestrates ID extraction, parallel API fetching, and episode processing.
Target performance: 10-30 seconds per anime (vs 5-15 minutes with AI).
"""

import asyncio
import json
import logging
import os
import sys
import time
from types import TracebackType
from typing import Any, cast

from .api_fetcher import ParallelAPIFetcher
from .config import EnrichmentConfig
from .id_extractor import PlatformIDExtractor

logger = logging.getLogger(__name__)


class ProgrammaticEnrichmentPipeline:
    """
    Main orchestrator for programmatic enrichment.
    Implements Steps 1-3 of enrichment process programmatically.
    """

    def __init__(self, config: EnrichmentConfig | None = None):
        """
        Create a ProgrammaticEnrichmentPipeline configured for enrichment runs.

        Initializes internal components used by the pipeline (ID extractor, parallel API fetcher)
        and a timing breakdown store. When `config` is omitted, a default EnrichmentConfig is used;
        if `config.verbose_logging` is true the configuration will be logged.

        Parameters:
            config (Optional[EnrichmentConfig]): Pipeline configuration; defaults to a new EnrichmentConfig().
        """
        self.config = config or EnrichmentConfig()

        # Initialize components
        self.id_extractor = PlatformIDExtractor()
        self.api_fetcher = ParallelAPIFetcher(config)

        # Performance tracking
        self.timing_breakdown: dict[str, float] = {}

        # Log configuration if verbose
        if self.config.verbose_logging:
            self.config.log_configuration()

    async def enrich_anime(
        self,
        offline_data: dict[str, Any],
        agent_dir: str | None = None,
        skip_services: list[str] | None = None,
        only_services: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Enrich a single anime record with data fetched from configured APIs.

        Parameters:
            offline_data (Dict): Existing anime data used as the basis for enrichment.
            agent_dir (Optional[str]): Optional agent directory name to use for temporary processing (e.g., "Dandadan_agent1"). If omitted, a new directory is created with gap-filled agent ID.
            skip_services (Optional[List[str]]): Optional list of service names to skip when fetching API data.
            only_services (Optional[List[str]]): Optional list of service names to fetch exclusively; if provided, other services are ignored.

        Returns:
            Dict[str, Any]: A dictionary with the following keys:
                - offline_data: The original input `offline_data`.
                - extracted_ids: Validated platform IDs extracted from `offline_data`.
                - api_data: Raw responses from the fetched APIs keyed by service name.
                - processed_episodes: Episode data extracted from API responses.
                - enrichment_metadata: Metadata about the enrichment run containing:
                    - method: The enrichment method used ("programmatic").
                    - total_time: Total elapsed time in seconds for the enrichment.
                    - timing_breakdown: Per-step timing information.
                    - successful_apis: Count of successful API responses.
                    - temp_directory: Path to the temporary agent directory used.

        Notes:
            If an error occurs during enrichment and the pipeline is configured to skip failed APIs, the function returns a partial result containing `offline_data`, an `error` string, and `partial_data: True`.
        """
        start_time = time.time()
        anime_title = offline_data.get("title", "Unknown")

        logger.info(f"Starting programmatic enrichment for: {anime_title}")

        try:
            # Step 1: Extract platform IDs (instant)
            step1_start = time.time()
            ids = self.id_extractor.extract_all_ids(offline_data)
            valid_ids = self.id_extractor.validate_ids(ids)
            self.timing_breakdown["id_extraction"] = time.time() - step1_start

            logger.info(
                f"Step 1 complete: Extracted {len(valid_ids)} platform IDs in {self.timing_breakdown['id_extraction']:.3f}s"
            )

            # Create or use specified temp directory for this anime
            if agent_dir:
                # Use provided agent directory
                temp_dir = os.path.join(self.config.temp_dir, agent_dir)
                os.makedirs(temp_dir, exist_ok=True)
                logger.info(f"Using specified agent directory: {temp_dir}")
            else:
                # Auto-generate agent directory with gap filling
                temp_dir = self._create_temp_dir(anime_title)

            # Save current anime entry for stage scripts
            current_anime_path = os.path.join(temp_dir, "current_anime.json")
            with open(current_anime_path, "w", encoding="utf-8") as f:
                json.dump(offline_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved current anime entry to {current_anime_path}")

            # Step 2: Parallel API fetching (5-10 seconds)
            step2_start = time.time()
            api_data = await self.api_fetcher.fetch_all_data(
                valid_ids, offline_data, temp_dir, skip_services, only_services
            )
            self.timing_breakdown["api_fetching"] = time.time() - step2_start

            # Count successful API responses
            successful_apis = sum(1 for v in api_data.values() if v)
            logger.info(
                f"Step 2 complete: Fetched data from {successful_apis} APIs in {self.timing_breakdown['api_fetching']:.2f}s"
            )

            # Step 3: Process episodes (instant)
            step3_start = time.time()
            processed_episodes = self._process_episodes(api_data)
            self.timing_breakdown["episode_processing"] = time.time() - step3_start

            logger.info(
                f"Step 3 complete: Processed {len(processed_episodes)} episodes in {self.timing_breakdown['episode_processing']:.3f}s"
            )

            # Compile results
            result = {
                "offline_data": offline_data,
                "extracted_ids": valid_ids,
                "api_data": api_data,
                "processed_episodes": processed_episodes,
                "enrichment_metadata": {
                    "method": "programmatic",
                    "total_time": time.time() - start_time,
                    "timing_breakdown": self.timing_breakdown.copy(),
                    "successful_apis": successful_apis,
                    "temp_directory": temp_dir,
                },
            }

            total_time = time.time() - start_time
            logger.info(f"✓ Enrichment complete for {anime_title} in {total_time:.2f}s")

            return result

        except Exception as e:
            logger.exception(f"Enrichment failed for {anime_title}: {e}")
            if self.config.skip_failed_apis:
                # Return partial data on failure (graceful degradation)
                return {
                    "offline_data": offline_data,
                    "error": str(e),
                    "partial_data": True,
                }
            raise

    async def enrich_batch(self, anime_list: list[dict]) -> list[dict]:
        """
        Enrich multiple anime in parallel.

        Args:
            anime_list: List of offline anime data

        Returns:
            List of enriched anime data

        Performance: Processes batch_size anime concurrently
        """
        logger.info(f"Starting batch enrichment for {len(anime_list)} anime")

        # Create semaphore to limit concurrent processing
        semaphore = asyncio.Semaphore(self.config.batch_size)

        async def enrich_with_limit(anime):
            async with semaphore:
                return await self.enrich_anime(anime)

        # Process all anime concurrently (limited by semaphore)
        tasks = [enrich_with_limit(anime) for anime in anime_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle results and exceptions
        successful: list[dict[str, Any]] = []
        failed = []

        for anime, result in zip(anime_list, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to enrich {anime.get('title')}: {result}")
                failed.append(anime)
            else:
                # result is guaranteed to be Dict[str, Any] here due to the isinstance check above
                successful.append(cast(dict[str, Any], result))

        logger.info(
            f"Batch complete: {len(successful)} successful, {len(failed)} failed"
        )

        return successful

    def _find_next_agent_id(self, anime_name: str) -> int:
        """
        Determine the next available global agent ID, preferring the lowest missing positive integer.

        Parameters:
            anime_name (str): Unused; retained for backward compatibility.

        Returns:
            int: The next available agent ID (fills gaps first, otherwise returns one greater than the current maximum).
        """
        # Check if temp directory exists
        temp_base = self.config.temp_dir
        if not os.path.exists(temp_base):
            return 1  # First agent

        # Scan for ALL agent directories (any anime)
        # Pattern: *_agent<N>
        existing_ids = []

        try:
            for item in os.listdir(temp_base):
                if "_agent" in item:
                    try:
                        # Split on "_agent" and get the part after it
                        after_agent = item.split("_agent")[1]
                        # Get first segment (number part before any additional "_")
                        num_str = (
                            after_agent.split("_")[0]
                            if "_" in after_agent
                            else after_agent
                        )
                        if num_str.isdigit():
                            existing_ids.append(int(num_str))
                    except (IndexError, ValueError):
                        # Skip directories that don't match expected pattern
                        continue
        except Exception as e:
            logger.warning(f"Error scanning temp directory: {e}")
            return 1

        if not existing_ids:
            return 1  # No existing agents

        # Sort existing IDs
        existing_ids.sort()

        # Find first missing ID (gap filling)
        for i in range(1, existing_ids[-1] + 1):
            if i not in existing_ids:
                logger.info(
                    f"Gap-filling agent ID: Using {i} (existing: {existing_ids})"
                )
                return i

        # No gaps found, return next sequential
        next_id = existing_ids[-1] + 1
        logger.info(
            f"No gaps: Using next agent ID {next_id} (existing: {existing_ids})"
        )
        return next_id

    def _create_temp_dir(self, anime_title: str) -> str:
        """
        Create a temporary processing directory for an anime and return its path.

        The directory is created under the configured temp_dir and named "<FirstWord>_agent<N>",
        where FirstWord is the sanitized first token of `anime_title` and N is the next available agent ID.

        Parameters:
            anime_title (str): Anime title from offline data; the first word is used for the directory name.

        Returns:
            str: Full path to the created temporary directory.
        """
        # Get first word from title for directory name
        first_word = anime_title.split()[0] if anime_title else "unknown"
        # Remove special characters
        clean_word = "".join(c for c in first_word if c.isalnum() or c in "-_")

        # Find next available agent ID
        agent_id = self._find_next_agent_id(clean_word)
        dir_name = f"{clean_word}_agent{agent_id}"

        temp_dir = os.path.join(self.config.temp_dir, dir_name)
        os.makedirs(temp_dir, exist_ok=True)

        logger.info(f"Created temp directory: {temp_dir}")

        return temp_dir

    def _process_episodes(self, api_data: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Extract episode entries from the Jikan response present in `api_data`.

        Returns:
            A list of episode dictionaries from Jikan's `episodes` field (each typically includes title, synopsis, aired date, etc.); returns an empty list if no Jikan data or episodes are present.
        """
        # Only use Jikan episodes - they have full details (title, synopsis, aired, etc.)
        # AniList episodes only have episode number and air time, not useful
        if jikan_data := api_data.get("jikan"):
            episodes = jikan_data.get("episodes", [])
            logger.debug(f"Extracted {len(episodes)} episodes from Jikan")
            return episodes

        return []

    async def __aenter__(self) -> "ProgrammaticEnrichmentPipeline":
        """
        Enter the asynchronous context for the pipeline.

        Returns:
            ProgrammaticEnrichmentPipeline: The pipeline instance.
        """
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        """
        Async context manager exit that delegates cleanup to the pipeline's API fetcher.

        Delegates async cleanup to the internal `api_fetcher` and does not suppress exceptions raised in the context.

        Returns:
            bool: `False` to indicate exceptions should be propagated.
        """
        await self.api_fetcher.__aexit__(exc_type, exc_val, exc_tb)
        return False

    def get_performance_report(self) -> str:
        """
        Produce a human-readable performance report for the pipeline.

        The report includes the configured maximum concurrent APIs and batch size. If available, it also lists a timing breakdown for pipeline steps and per-API response times.

        Returns:
            report (str): A multi-line string containing the assembled performance report.
        """
        report = ["Performance Report:"]
        report.append(f"  Total APIs configured: {self.config.max_concurrent_apis}")
        report.append(f"  Batch size: {self.config.batch_size}")

        if self.timing_breakdown:
            report.append("\nTiming Breakdown:")
            for step, time_taken in self.timing_breakdown.items():
                report.append(f"  {step}: {time_taken:.3f}s")

        if self.api_fetcher.api_timings:
            report.append("\nAPI Response Times:")
            for api, time_taken in self.api_fetcher.api_timings.items():
                report.append(f"  {api}: {time_taken:.2f}s")

        return "\n".join(report)


async def main() -> int:
    """
    Run a test enrichment of the pipeline using sample anime data and save the results.

    Performs a single enrichment with a built-in sample anime, prints a short summary and a performance report to stdout, writes the full enrichment result to "programmatic_enrichment_test.json", and returns an exit code.

    Returns:
        int: 0 on success.
    """

    # Sample offline data
    sample_anime = {
        "sources": [
            "https://myanimelist.net/anime/21/One_Piece",
            "https://anilist.co/anime/21",
            "https://kitsu.io/anime/one-piece",
        ],
        "title": "One Piece",
        "episodes": 1000,
        "type": "TV",
        "status": "Currently Airing",
    }

    async with ProgrammaticEnrichmentPipeline() as pipeline:
        # Run enrichment
        result = await pipeline.enrich_anime(sample_anime)

        # Print results
        print("\n" + "=" * 60)
        print("ENRICHMENT RESULTS")
        print("=" * 60)

        print(f"\nExtracted IDs: {result['extracted_ids']}")
        print(f"\nSuccessful APIs: {result['enrichment_metadata']['successful_apis']}")
        print(f"Total Time: {result['enrichment_metadata']['total_time']:.2f}s")

        print("\n" + pipeline.get_performance_report())

        # Save result
        output_file = "programmatic_enrichment_test.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"\nResults saved to {output_file}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    sys.exit(asyncio.run(main()))
