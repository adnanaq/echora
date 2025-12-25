#!/usr/bin/env python3
"""
AnimSchedule Data Fetcher
Fetches anime data from AnimSchedule API (async version)
Follows gemini_instructions.md Step 2.4

Usage:
    # Programmatic usage (no file output)
    from enrichment.api_helpers.animeschedule_fetcher import fetch_animeschedule_data
    data = await fetch_animeschedule_data("One Piece")

    # Programmatic usage (with file output)
    data = await fetch_animeschedule_data("One Piece", output_path="temp/as.json")

    # CLI usage (default output to CWD)
    python -m enrichment.api_helpers.animeschedule_fetcher "One Piece"
    # Output: animeschedule.json

    # CLI usage (custom output path)
    python -m enrichment.api_helpers.animeschedule_fetcher "One Piece" --output temp/as.json
    # Output: temp/as.json
"""

import argparse
import asyncio
import json
import logging
import sys
from typing import Any

import aiohttp
from http_cache.instance import http_cache_manager as _cache_manager

logger = logging.getLogger(__name__)


async def fetch_animeschedule_data(
    search_term: str, output_path: str | None = None
) -> dict[str, Any] | None:
    """
    Fetch AnimSchedule data for the first matching anime title.

    Parameters:
        search_term (str): Anime title to search for.
        output_path (Optional[str]): If provided, write the fetched anime data as pretty-printed JSON to this path.

    Returns:
        dict: The first matching anime's data if found.
        None: If no results are found or an HTTP/JSON error occurs.
    """

    print(f"🔄 Fetching AnimSchedule data for: {search_term}")

    # Get cached aiohttp session
    session = _cache_manager.get_aiohttp_session("animeschedule")

    try:
        # Search for anime on AnimSchedule
        search_url = f"https://animeschedule.net/api/v3/anime?q={search_term}"
        print(f"  📡 Searching: {search_url}")

        async with session.get(search_url) as response:
            response.raise_for_status()
            search_results = await response.json()

        if not search_results or not search_results.get("anime"):
            print("❌ No results found on AnimSchedule")
            return None

        # Take the first result (most relevant)
        anime_data: dict[str, Any] = search_results["anime"][0]

        # Conditionally write to file (matches crawler pattern)
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(anime_data, f, ensure_ascii=False, indent=2)
            print(f"Data written to {output_path}")

        print("✅ AnimSchedule data fetched successfully")
        return anime_data

    except aiohttp.ClientError as e:
        print(f"❌ AnimSchedule API error: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        return None
    finally:
        # Always close the session
        await session.close()


async def main() -> int:
    """
    Entry point for the CLI that fetches anime data from AnimSchedule using provided arguments.

    Parses command-line arguments `search_term` and optional `--output`, invokes `fetch_animeschedule_data`, and maps the outcome to an exit code.

    Returns:
        int: 0 if data was fetched successfully, 1 on failure.
    """
    parser = argparse.ArgumentParser(
        description="Fetch anime data from AnimSchedule API."
    )
    parser.add_argument("search_term", type=str, help="Anime title to search for")
    parser.add_argument(
        "--output",
        type=str,
        default="animeschedule.json",
        help="Output file path (default: animeschedule.json in current directory)",
    )
    args = parser.parse_args()

    try:
        result = await fetch_animeschedule_data(
            args.search_term,
            output_path=args.output,
        )
        return 0 if result else 1
    except Exception:
        logger.exception("Error fetching AnimSchedule data")
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(asyncio.run(main()))
