#!/usr/bin/env python3
"""
Reusable script for fetching detailed data from Jikan API with proper rate limiting.

Usage:
    python scripts/fetch_detailed_jikan_data.py episodes <anime_id> <input_file> <output_file>
    python scripts/fetch_detailed_jikan_data.py characters <anime_id> <input_file> <output_file>

Examples:
    python scripts/fetch_detailed_jikan_data.py episodes 21 temp/episodes.json temp/episodes_detailed.json
    python scripts/fetch_detailed_jikan_data.py characters 21 temp/characters.json temp/characters_detailed.json
"""

import argparse
import asyncio
import json
import os
import sys
import time
from types import TracebackType
from typing import Any, Dict, List, Optional, Type

from src.cache_manager.instance import http_cache_manager as _cache_manager


class JikanDetailedFetcher:
    """
    Fetches detailed data from Jikan API with proper rate limiting.
    Supports episodes and characters endpoints.
    Uses async/await with aiohttp for improved performance.
    """

    def __init__(self, anime_id: str, data_type: str, session: Optional[Any] = None):
        """
        Initialize the fetcher for detailed Jikan API data for a specific anime.
        
        Parameters:
            anime_id (str): MyAnimeList anime ID used to build API requests.
            data_type (str): Type of data to fetch; expected values are 'episodes' or 'characters'.
            session (Optional[Any]): Optional aiohttp-style session to use for HTTP requests. If omitted, a cached session is obtained internally.
        
        Notes:
            - Sets default rate-limit configuration to 3 requests/second and 60 requests/minute.
            - Tracks whether the instance owns the provided session so it can close it when appropriate.
        """
        self.anime_id = anime_id
        self.data_type = data_type  # 'episodes' or 'characters'
        self.request_count = 0
        self.start_time = time.time()
        self.batch_size = 50

        # Jikan API rate limits: 3 requests per second, 60 per minute
        self.max_requests_per_second = 3
        self.max_requests_per_minute = 60

        # Reuse provided session or create new one
        self._owns_session = session is None
        self.session = session or _cache_manager.get_aiohttp_session("jikan")

    async def respect_rate_limits(self) -> None:
        """
        Enforce Jikan API pacing constraints to avoid exceeding 3 requests/second and 60 requests/minute.
        
        Resets the per-minute counter when a minute has elapsed or the system clock moves backwards, pauses until the minute boundary if 60 requests have been made, and enforces a 0.5‑second delay between consecutive network requests.
        """
        current_time = time.time()
        elapsed = current_time - self.start_time

        # Handle time going backwards (clock adjustments, NTP sync, etc.)
        if elapsed < 0:
            self.request_count = 0
            self.start_time = current_time
            elapsed = 0

        # Reset counter every minute
        if elapsed >= 60:
            self.request_count = 0
            self.start_time = current_time
            elapsed = 0

        # If we've made 60 requests in current minute, wait until minute resets
        if self.request_count >= self.max_requests_per_minute:
            wait_time = 60 - elapsed
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                self.request_count = 0
                self.start_time = time.time()

        # Always wait 0.5s between requests (ensures 2 requests/second << 3/sec limit)
        # Being more conservative to avoid 429 errors
        if self.request_count > 0:  # Don't wait before first request
            await asyncio.sleep(0.5)

    async def _record_network_request(self, from_cache: bool) -> None:
        """
        Increment the internal network request counter and enforce pacing when a response was not served from cache.
        
        Parameters:
            from_cache (bool): True if the response was served from cache; when False the request counter is incremented and rate-limiting/pacing is applied.
        """
        if not from_cache:
            self.request_count += 1
            await self.respect_rate_limits()

    async def fetch_episode_detail(
        self, episode_id: int, retry_count: int = 0
    ) -> Optional[Dict[str, Any]]:
        """
        Fetches detailed information for a specific episode from the Jikan API.
        
        Parameters:
            episode_id (int): Episode number to fetch.
            retry_count (int): Internal retry attempt count; used to retry up to 3 times when the API returns HTTP 429.
        
        Returns:
            dict: Episode detail containing keys such as `episode_number`, `url`, `title`, `title_japanese`, `title_romaji`, `aired`, `score`, `filler`, `recap`, `duration`, and `synopsis` on success.
            None: If the request ultimately fails or a non-recoverable HTTP error occurs.
        """
        try:
            url = (
                f"https://api.jikan.moe/v4/anime/{self.anime_id}/episodes/{episode_id}"
            )
            async with self.session.get(url) as response:
                # Check if response was served from cache
                # Explicitly check for boolean type to handle mocks properly
                from_cache = (
                    isinstance(getattr(response, "from_cache", None), bool)
                    and response.from_cache
                )

                if response.status == 200:
                    data = await response.json()
                    episode_detail = data["data"]

                    # Only rate limit for network requests, not cache hits
                    await self._record_network_request(from_cache)

                    return {
                        "episode_number": episode_id,
                        "url": episode_detail.get("url"),
                        "title": episode_detail.get("title"),
                        "title_japanese": episode_detail.get("title_japanese"),
                        "title_romaji": episode_detail.get("title_romaji"),
                        "aired": episode_detail.get("aired"),
                        "score": episode_detail.get("score"),
                        "filler": episode_detail.get("filler", False),
                        "recap": episode_detail.get("recap", False),
                        "duration": episode_detail.get("duration"),
                        "synopsis": episode_detail.get("synopsis"),
                    }

                elif response.status == 429:
                    if retry_count >= 3:
                        print(
                            f"Max retries reached for episode {episode_id}, giving up"
                        )
                        return None

                    print(
                        f"Rate limit hit for episode {episode_id}. Waiting and retrying (attempt {retry_count + 1}/3)..."
                    )
                    await asyncio.sleep(5)
                    await self._record_network_request(from_cache)
                    return await self.fetch_episode_detail(episode_id, retry_count + 1)

                else:
                    print(
                        f"Error fetching episode {episode_id}: HTTP {response.status}"
                    )
                    await self._record_network_request(from_cache)
                    return None

        except Exception as e:
            print(f"Error fetching episode {episode_id}: {e}")
            return None

    async def fetch_character_detail(
        self, character_data: Dict[str, Any], retry_count: int = 0
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch detailed information for a character from the Jikan API.
        
        Parameters:
            character_data (Dict[str, Any]): Input record containing at least
                `character["mal_id"]`. May include additional fields such as
                `role` and `voice_actors` which are preserved in the result.
            retry_count (int): Internal retry counter; not required by callers.
        
        Returns:
            Optional[Dict[str, Any]]: A dictionary with selected character fields:
            `character_id`, `url`, `name`, `name_kanji`, `nicknames`, `about`,
            `images`, `favorites`, `role`, and `voice_actors`. Returns `None` if the
            fetch ultimately fails.
        """
        character_id = character_data["character"]["mal_id"]

        try:
            url = f"https://api.jikan.moe/v4/characters/{character_id}"
            async with self.session.get(url) as response:
                # Check if response was served from cache
                # Explicitly check for boolean type to handle mocks properly
                from_cache = (
                    isinstance(getattr(response, "from_cache", None), bool)
                    and response.from_cache
                )

                if response.status == 200:
                    data = await response.json()
                    character_detail = data["data"]

                    # Only rate limit for network requests, not cache hits
                    await self._record_network_request(from_cache)

                    return {
                        "character_id": character_id,
                        "url": character_detail.get("url"),
                        "name": character_detail.get("name"),
                        "name_kanji": character_detail.get("name_kanji"),
                        "nicknames": character_detail.get("nicknames", []),
                        "about": character_detail.get("about"),
                        "images": character_detail.get("images", {}),
                        "favorites": character_detail.get("favorites"),
                        "role": character_data.get("role"),
                        "voice_actors": character_data.get("voice_actors", []),
                    }

                elif response.status == 429:
                    if retry_count >= 3:
                        print(
                            f"Max retries reached for character {character_id}, giving up"
                        )
                        return None

                    print(
                        f"Rate limit hit for character {character_id}. Waiting and retrying (attempt {retry_count + 1}/3)..."
                    )
                    await asyncio.sleep(5)
                    await self._record_network_request(from_cache)
                    return await self.fetch_character_detail(
                        character_data, retry_count + 1
                    )

                else:
                    print(
                        f"Error fetching character {character_id}: HTTP {response.status}"
                    )
                    await self._record_network_request(from_cache)
                    return None

        except Exception as e:
            print(f"Error fetching character {character_id}: {e}")
            return None

    def append_batch_to_file(
        self, batch_data: List[Dict[str, Any]], progress_file: str
    ) -> int:
        """
        Append a batch of items into a JSON progress file.
        
        If the file does not exist it will be created containing the batch; if it exists the batch items are appended to the existing JSON array.
        
        Parameters:
            batch_data (List[Dict[str, Any]]): Items to append to the progress file.
            progress_file (str): Path to the JSON progress file.
        
        Returns:
            total_items (int): Total number of items present in the progress file after appending.
        """
        # Load existing data
        if os.path.exists(progress_file):
            with open(progress_file, "r", encoding="utf-8") as f:
                all_data = json.load(f)
        else:
            all_data = []

        # Append new batch
        all_data.extend(batch_data)

        # Save updated data
        with open(progress_file, "w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)

        return len(all_data)

    async def close(self) -> None:
        """Close the underlying HTTP session if we created it."""
        if self._owns_session and self.session:
            await self.session.close()

    async def __aenter__(self) -> "JikanDetailedFetcher":
        """
        Enter the asynchronous context and yield the fetcher instance.
        
        Returns:
            JikanDetailedFetcher: The fetcher instance to be used as the async context manager.
        """
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        """
        Exit the async context and close the owned HTTP session.
        
        Returns:
            bool: `False` to indicate any exception should be propagated.
        """
        await self.close()
        return False

    async def fetch_detailed_data(self, input_file: str, output_file: str) -> None:
        """
        Fetch detailed episode or character data from input_file and write aggregated results to output_file.
        
        Reads input_file (supports episode lists, paginated "data" shapes, or anime-level episode counts; for characters accepts list or "data" wrapped list), resumes work from a progress file named "<output_file>.progress", fetches details in batches (flushed to the progress file every self.batch_size items), prints periodic progress, and finally writes the assembled and sorted detailed items to output_file. If no items are present, writes an empty array to output_file. Removes the progress file after successful completion.
        
        Parameters:
        	input_file (str): Path to the JSON input describing episodes or characters.
        	output_file (str): Path where the final detailed JSON array will be written.
        """
        # Load input data
        with open(input_file, "r", encoding="utf-8") as f:
            input_data = json.load(f)

        if self.data_type == "episodes":
            # For episodes, we can use the episode count directly from anime data
            # or process existing episode list
            if isinstance(input_data, list):
                total_items = len(input_data)
                episode_ids = [ep["mal_id"] for ep in input_data]
            elif "data" in input_data:
                # Handle paginated episode data
                input_data = input_data["data"]
                total_items = len(input_data)
                episode_ids = [ep["mal_id"] for ep in input_data]
            else:
                # If input is anime data with episode count
                total_items = input_data.get("episodes", 0)
                episode_ids = list(range(1, total_items + 1))
            print(f"Fetching detailed data for {total_items} episodes...")
        else:  # characters
            if "data" in input_data:
                input_data = input_data["data"]
            total_items = len(input_data)
            print(f"Fetching detailed data for {total_items} characters...")

        # Handle empty input
        if total_items == 0:
            print("No items to process, creating empty output file")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)
            return

        # Progress tracking
        progress_file = f"{output_file}.progress"
        if os.path.exists(progress_file):
            with open(progress_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            print(
                f"Found existing progress: {len(existing_data)} items already fetched"
            )
            start_index = len(existing_data)
        else:
            start_index = 0

        batch_data = []

        # Determine item type for logging
        item_type = "episode" if self.data_type == "episodes" else "character"

        # Process items starting from where we left off
        for i in range(start_index, total_items):
            if self.data_type == "episodes":
                item_id = episode_ids[i]
                detailed_item = await self.fetch_episode_detail(item_id)
            else:  # characters
                item = input_data[i]
                detailed_item = await self.fetch_character_detail(item)
                item_id = item["character"]["mal_id"]

            if detailed_item:
                batch_data.append(detailed_item)

            # Progress update
            if (i + 1) % 10 == 0:
                print(f"Progress: {i+1}/{total_items} {item_type}s fetched")

            # Save progress every batch_size items
            if len(batch_data) >= self.batch_size:
                total_count = self.append_batch_to_file(batch_data, progress_file)
                print(
                    f"Appended batch: {len(batch_data)} {item_type}s (total: {total_count})"
                )
                batch_data = []  # Clear batch

        # Save any remaining items in the final batch
        if batch_data:
            total_count = self.append_batch_to_file(batch_data, progress_file)
            print(
                f"Appended final batch: {len(batch_data)} {item_type}s (total: {total_count})"
            )

        # Load final data and create final file
        if os.path.exists(progress_file):
            with open(progress_file, "r", encoding="utf-8") as f:
                all_detailed_data = json.load(f)
        else:
            # No items were successfully fetched
            all_detailed_data = []

        print(f"\\nCompleted fetching {len(all_detailed_data)} detailed {item_type}s")

        # Sort data by ID
        if self.data_type == "episodes":
            all_detailed_data.sort(key=lambda x: x.get("episode_number", 0))
            synopsis_count = sum(1 for ep in all_detailed_data if ep.get("synopsis"))
            print(f"Episodes with synopsis: {synopsis_count}/{len(all_detailed_data)}")
        else:  # characters
            all_detailed_data.sort(key=lambda x: x.get("character_id", 0))
            about_count = sum(1 for char in all_detailed_data if char.get("about"))
            print(f"Characters with about text: {about_count}/{len(all_detailed_data)}")

        # Save final detailed data
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_detailed_data, f, ensure_ascii=False, indent=2)

        print(f"Final detailed data saved to {output_file}")

        # Clean up progress file
        if os.path.exists(progress_file):
            os.remove(progress_file)
            print(f"Cleaned up progress file: {progress_file}")


async def main() -> int:
    """
    Parse CLI arguments and run the JikanDetailedFetcher to fetch detailed Jikan data and write output.
    
    Returns:
        exit_code (int): 0 on success, 1 on error.
    """
    parser = argparse.ArgumentParser(description="Fetch detailed data from Jikan API")
    parser.add_argument(
        "data_type", choices=["episodes", "characters"], help="Type of data to fetch"
    )
    parser.add_argument("anime_id", help="Anime ID (MAL ID)")
    parser.add_argument("input_file", help="Input file path")
    parser.add_argument("output_file", help="Output file path")

    args = parser.parse_args()

    # Extract output directory to guard against empty dirname edge case
    output_dir = os.path.dirname(args.output_file)

    fetcher = JikanDetailedFetcher(args.anime_id, args.data_type)

    try:
        # Validate input file exists
        if not os.path.exists(args.input_file):
            print(
                f"Error: Input file {args.input_file} does not exist", file=sys.stderr
            )
            return 1

        # Create output directory if it doesn't exist
        # Guard: only call makedirs if output_dir is non-empty
        # (dirname('episodes.json') returns '', makedirs('') raises FileNotFoundError)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        await fetcher.fetch_detailed_data(args.input_file, args.output_file)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    finally:
        await fetcher.close()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(asyncio.run(main()))