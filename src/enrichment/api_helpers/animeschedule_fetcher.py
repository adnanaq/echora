#!/usr/bin/env python3
"""
AnimSchedule Data Fetcher
Fetches anime data from AnimSchedule API (async version)
Follows gemini_instructions.md Step 2.4
"""

import asyncio
import json
import sys
from typing import Any, Dict, Optional

import aiohttp

from src.cache_manager.instance import http_cache_manager as _cache_manager


async def fetch_animeschedule_data(
    search_term: str, save_file: bool = False
) -> Optional[Dict[str, Any]]:
    """Fetch AnimSchedule data for an anime (async version)"""

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
        anime_data = search_results["anime"][0]

        # Save to temp file only if requested (for standalone usage)
        if save_file:
            with open("temp/as.json", "w", encoding="utf-8") as f:
                json.dump(anime_data, f, ensure_ascii=False, indent=2)

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


async def main() -> None:
    """Main function for command-line usage"""
    if len(sys.argv) != 2:
        print("Usage: python animeschedule_fetcher.py <search_term>")
        sys.exit(1)

    search_term = sys.argv[1]
    await fetch_animeschedule_data(search_term, save_file=True)


if __name__ == "__main__":
    asyncio.run(main())
