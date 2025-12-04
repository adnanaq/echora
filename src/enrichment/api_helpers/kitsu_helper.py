#!/usr/bin/env python3
"""
Kitsu Helper for AI Enrichment Integration

Simple data fetcher for Kitsu API without modifying existing kitsu_client.py
"""

import asyncio
import json
import logging
import sys
from types import TracebackType
from typing import Any, Dict, List, Optional, Type, cast

import aiohttp

from src.cache_manager.instance import http_cache_manager as _cache_manager

logger = logging.getLogger(__name__)


class KitsuEnrichmentHelper:
    """Simple helper for Kitsu data fetching in AI enrichment pipeline."""

    def __init__(self) -> None:
        """Initialize Kitsu enrichment helper."""
        self.base_url = "https://kitsu.io/api/edge"

    async def _make_request(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform an HTTP GET request to a Kitsu API endpoint and return the parsed JSON response.
        
        Parameters:
            endpoint (str): API path appended to the helper's base_url (e.g. '/anime/1').
            params (Optional[Dict[str, Any]]): Query parameters to include in the request.
        
        Returns:
            Dict[str, Any]: The parsed JSON response body when the server returns HTTP 200; an empty dict on non-200 responses or on error.
        """
        headers = {
            "Accept": "application/vnd.api+json",
            "Content-Type": "application/vnd.api+json",
        }

        url = f"{self.base_url}{endpoint}"

        try:
            async with _cache_manager.get_aiohttp_session(
                "kitsu", timeout=aiohttp.ClientTimeout(total=30)
            ) as session:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        return cast(Dict[str, Any], await response.json())
                    else:
                        logger.warning(f"Kitsu API error: HTTP {response.status}")
                        return {}
        except Exception as e:
            logger.error(f"Kitsu API request failed: {e}")
            return {}

    async def get_anime_by_id(self, anime_id: int) -> Optional[Dict[str, Any]]:
        """Get anime by Kitsu ID."""
        try:
            response = await self._make_request(f"/anime/{anime_id}")
            return response.get("data") if response else None
        except Exception as e:
            logger.error(f"Kitsu get_anime_by_id failed for ID {anime_id}: {e}")
            return None

    async def get_anime_episodes(self, anime_id: int) -> List[Dict[str, Any]]:
        """Get ALL anime episodes by Kitsu ID with pagination."""
        all_episodes = []
        page = 0
        page_size = 20  # Kitsu default page size

        try:
            while True:
                params = {"page[limit]": page_size, "page[offset]": page * page_size}

                response = await self._make_request(
                    f"/anime/{anime_id}/episodes", params
                )

                if not response or "data" not in response:
                    break

                episodes = response["data"]
                if not episodes:  # No more episodes
                    break

                all_episodes.extend(episodes)
                logger.info(
                    f"Fetched page {page + 1}: {len(episodes)} episodes (total: {len(all_episodes)})"
                )

                # Check if there are more pages
                meta = response.get("meta", {})
                count = meta.get("count", 0)

                if len(all_episodes) >= count or len(episodes) < page_size:
                    # We have all episodes or got less than page size (last page)
                    break

                page += 1

                # Add small delay to respect rate limits
                await asyncio.sleep(0.1)

            logger.info(
                f"Total episodes fetched for anime {anime_id}: {len(all_episodes)}"
            )
            return all_episodes

        except Exception as e:
            logger.error(f"Kitsu get_anime_episodes failed for ID {anime_id}: {e}")
            return all_episodes  # Return what we got so far

    async def get_anime_categories(self, anime_id: int) -> List[Dict[str, Any]]:
        """Get anime categories by Kitsu ID."""
        try:
            response = await self._make_request(f"/anime/{anime_id}/categories")
            return response.get("data", []) if response else []
        except Exception as e:
            logger.error(f"Kitsu get_anime_categories failed for ID {anime_id}: {e}")
            return []

    async def fetch_all_data(self, anime_id: int) -> Dict[str, Any]:
        """
        Fetch the anime record, all episodes, and categories for the given Kitsu anime ID concurrently.
        
        Parameters:
            anime_id (int): The Kitsu anime identifier to fetch data for.
        
        Returns:
            dict: Mapping with keys:
                - "anime": The anime resource object from Kitsu, or `None` if it could not be retrieved.
                - "episodes": A list of episode resource objects, or an empty list if unavailable.
                - "categories": A list of category resource objects, or an empty list if unavailable.
        """
        try:
            # Fetch all data concurrently
            results = await asyncio.gather(
                self.get_anime_by_id(anime_id),
                self.get_anime_episodes(anime_id),
                self.get_anime_categories(anime_id),
                return_exceptions=True,
            )

            # Handle exceptions and assign results
            anime_data, episodes_data, categories_data = results

            return {
                "anime": anime_data if not isinstance(anime_data, Exception) else None,
                "episodes": (
                    episodes_data if not isinstance(episodes_data, Exception) else []
                ),
                "categories": (
                    categories_data
                    if not isinstance(categories_data, Exception)
                    else []
                ),
            }
        except Exception as e:
            logger.error(f"Kitsu fetch_all_data failed for ID {anime_id}: {e}")
            return {"anime": None, "episodes": [], "categories": []}

    async def close(self) -> None:
        """
        No-op close method retained for interface compatibility.
        
        This method performs no action because HTTP sessions are created per request; it exists so callers can await a close() when an async lifecycle API is required.
        """
        pass

    async def __aenter__(self) -> "KitsuEnrichmentHelper":
        """
        Enter the asynchronous context manager and return the helper instance.
        
        Returns:
            KitsuEnrichmentHelper: The KitsuEnrichmentHelper instance being managed by the context.
        """
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        """
        Exit the asynchronous context manager and ensure the helper is closed.
        
        Await the helper's close method to release any resources.
        
        Returns:
            bool: `False` to indicate any exception should be propagated.
        """
        await self.close()
        return False


async def main() -> int:
    """
    CLI entry point that fetches Kitsu anime data and writes it to a JSON file.
    
    Expects exactly two command-line arguments: an integer `anime_id` and an `output_file` path. Attempts to fetch anime details, episodes, and categories for the given ID and writes the resulting JSON to the specified file. Any errors or invalid usage are reported to stderr.
    
    Returns:
        int: 0 on success; 1 on invalid usage, fetch failure, or any error.
    """
    if len(sys.argv) != 3:
        print("Usage: python kitsu_helper.py <anime_id> <output_file>", file=sys.stderr)
        return 1

    try:
        anime_id = int(sys.argv[1])
        output_file = sys.argv[2]

        helper = KitsuEnrichmentHelper()
        data = await helper.fetch_all_data(anime_id)

        if data:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"Data for {anime_id} saved to {output_file}")
            return 0
        else:
            print(f"Could not fetch data for {anime_id}", file=sys.stderr)
            return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    sys.exit(asyncio.run(main()))