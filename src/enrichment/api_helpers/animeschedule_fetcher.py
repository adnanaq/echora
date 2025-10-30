#!/usr/bin/env python3
"""
AnimSchedule Data Fetcher
Fetches anime data from AnimSchedule API
Follows gemini_instructions.md Step 2.4
"""

import json
import sys
from typing import Any, Dict, Optional

import requests

from ..cache.config import get_cache_config
from ..cache.manager import HTTPCacheManager

# Initialize cache manager (singleton)
_cache_config = get_cache_config()
_cache_manager = HTTPCacheManager(_cache_config)


def fetch_animeschedule_data(
    search_term: str, save_file: bool = False
) -> Optional[Dict[str, Any]]:
    """Fetch AnimSchedule data for an anime"""

    print(f"🔄 Fetching AnimSchedule data for: {search_term}")

    try:
        # Get cached session
        session = _cache_manager.get_requests_session("animeschedule")

        # Search for anime on AnimSchedule
        search_url = f"https://animeschedule.net/api/v3/anime?q={search_term}"
        print(f"  📡 Searching: {search_url}")

        response = session.get(search_url)
        response.raise_for_status()
        search_results = response.json()

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

    except requests.exceptions.RequestException as e:
        print(f"❌ AnimSchedule API error: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        return None


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python animeschedule_fetcher.py <search_term>")
        sys.exit(1)

    search_term = sys.argv[1]
    fetch_animeschedule_data(search_term, save_file=True)
