#!/usr/bin/env python3
"""
Lightweight AniDB URL helper for Stage 3 relationship processing.

This helper extracts AniDB IDs from URLs and fetches anime data using the
existing AniDBEnrichmentHelper with proper API authentication.

Usage:
    from src.batch_enrichment.anidb_url_helper import fetch_anidb_url_data

    result = fetch_anidb_url_data("https://anidb.net/anime/23")

    # Result format:
    # {
    #     "title": "Anime Title",
    #     "relationship": None,  # Let AI infer from context
    #     "confidence": "high|medium|low",
    #     "context": {...},      # Rich context for AI analysis
    #     "url": "original_url"
    # }
"""

import asyncio
import logging
import re
from typing import Any

from .anidb_helper import AniDBEnrichmentHelper

logger = logging.getLogger(__name__)


def fetch_anidb_url_data(url: str) -> dict[str, Any] | None:
    """
    Fetch title and relationship data from an AniDB URL.

    Uses the existing AniDBEnrichmentHelper with proper API authentication
    and rate limiting.

    Args:
        url: Full AniDB URL (e.g., "https://anidb.net/anime/23")

    Returns:
        Dict with title, relationship, confidence, context, and original URL
        None if fetch fails
    """
    try:
        # Extract AniDB ID from URL
        anidb_id = _extract_anidb_id_from_url(url)
        if not anidb_id:
            logger.warning(f"Could not extract AniDB ID from URL: {url}")
            return _create_fallback_result(url, "unknown")

        logger.debug(f"Extracted AniDB ID {anidb_id} from URL: {url}")

        # Use asyncio to run the async API call
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            anime_data = loop.run_until_complete(_fetch_anidb_data_async(anidb_id))
        finally:
            loop.close()

        if not anime_data:
            logger.warning(f"No data returned from AniDB API for ID: {anidb_id}")
            return _create_fallback_result(url, anidb_id)

        return _parse_anidb_response(anime_data, url, anidb_id)

    except Exception as e:
        logger.error(f"Error fetching AniDB URL {url}: {e}")
        return _create_fallback_result(url, "error")


async def _fetch_anidb_data_async(anidb_id: int) -> dict[str, Any] | None:
    """Fetch data from AniDB API using the existing helper."""
    helper = AniDBEnrichmentHelper()
    try:
        # Use the existing comprehensive data fetching method
        return await helper.fetch_all_data(anidb_id)
    finally:
        await helper.close()


def _extract_anidb_id_from_url(url: str) -> int | None:
    """Extract AniDB anime ID from various URL formats."""

    # Common AniDB URL patterns
    patterns = [
        r"anidb\.net/anime/(\d+)",  # https://anidb.net/anime/23
        r"anidb\.info/a(\d+)",  # https://anidb.info/a23
        r"anidb\.net/a(\d+)",  # https://anidb.net/a23
    ]

    for pattern in patterns:
        match = re.search(pattern, url, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                continue

    return None


def _parse_anidb_response(
    anime_data: dict[str, Any], url: str, anidb_id: int
) -> dict[str, Any]:
    """Parse AniDB API response to extract title and context info."""
    try:
        # Extract title from AniDB titles hierarchy
        title = _extract_best_title(anime_data)
        if not title:
            title = f"AniDB Anime {anidb_id}"
            confidence = "low"
        else:
            confidence = "high"

        # Extract rich context for AI relationship inference
        context = _extract_context_data(anime_data)

        return {
            "title": title,
            "relationship": None,  # Let AI infer from context
            "confidence": confidence,
            "context": context,  # Provide rich context for AI inference
            "url": url,
            "source": "anidb",
        }

    except Exception as e:
        logger.error(f"Error parsing AniDB response for URL {url}: {e}")
        return _create_fallback_result(url, anidb_id)


def _extract_best_title(anime_data: dict[str, Any]) -> str | None:
    """Extract the best title from AniDB data using title hierarchy."""
    titles = anime_data.get("titles", {})

    # Priority order: main → english → japanese → first synonym
    if titles.get("main"):
        return titles["main"]
    elif titles.get("english"):
        return titles["english"]
    elif titles.get("japanese"):
        return titles["japanese"]
    elif titles.get("synonyms") and len(titles["synonyms"]) > 0:
        return titles["synonyms"][0]

    return None


def _extract_context_data(anime_data: dict[str, Any]) -> dict[str, Any]:
    """
    Extract contextual data for AI-powered relationship inference.

    Extract rich context from AniDB data for AI analysis instead of
    hardcoded relationship patterns.
    """
    context = {}

    # Basic metadata
    if anime_data.get("description"):
        context["description"] = anime_data["description"]

    if anime_data.get("type"):
        context["type"] = anime_data["type"]

    if anime_data.get("episodecount"):
        context["episodes"] = anime_data["episodecount"]

    # Date information for sequels/prequels inference
    if anime_data.get("startdate"):
        context["start_date"] = anime_data["startdate"]
    if anime_data.get("enddate"):
        context["end_date"] = anime_data["enddate"]

    # Tags/categories for genre context
    tags = anime_data.get("tags", [])
    if tags:
        # Extract top-weighted tag names for context
        tag_names = [tag["name"] for tag in tags[:10] if tag.get("name")]
        if tag_names:
            context["tags"] = tag_names

    categories = anime_data.get("categories", [])
    if categories:
        # Extract category names for genre context
        category_names = [cat["name"] for cat in categories if cat.get("name")]
        if category_names:
            context["categories"] = category_names

    # Creator information for franchise context
    creators = anime_data.get("creators", [])
    if creators:
        creator_names = [creator["name"] for creator in creators if creator.get("name")]
        if creator_names:
            context["creators"] = creator_names[:5]  # Limit to 5 most relevant

    # Character information for relationship context
    characters = anime_data.get("characters", [])
    if characters:
        # Extract main character names for franchise/character-based relationships
        main_chars = [char["name"] for char in characters[:5] if char.get("name")]
        if main_chars:
            context["main_characters"] = main_chars

    # Rating information for quality context
    ratings = anime_data.get("ratings", {})
    if ratings:
        permanent_rating = ratings.get("permanent", {})
        if permanent_rating.get("value"):
            context["rating"] = {
                "value": permanent_rating["value"],
                "count": permanent_rating.get("count", 0),
            }

    return context


def _create_fallback_result(url: str, anidb_id) -> dict[str, Any]:
    """Create fallback result when extraction fails."""
    return {
        "title": f"AniDB Anime {anidb_id}",
        "relationship": None,  # Cannot determine relationship when blocked
        "confidence": "none",  # No confidence when we can't access data
        "url": url,
        "source": "anidb",
    }


# Batch processing function for multiple URLs
def fetch_multiple_anidb_urls(urls: list[str]) -> dict[str, dict[str, Any]]:
    """
    Fetch data for multiple AniDB URLs with proper rate limiting.

    Uses a single AniDBEnrichmentHelper instance to ensure proper rate limiting
    across all requests.

    Args:
        urls: List of AniDB URLs

    Returns:
        Dict mapping URL to result data
    """
    results = {}

    # Extract all AniDB IDs first
    url_to_id = {}
    for url in urls:
        anidb_id = _extract_anidb_id_from_url(url)
        if anidb_id:
            url_to_id[url] = anidb_id
        else:
            logger.warning(f"Could not extract AniDB ID from URL: {url}")
            results[url] = _create_fallback_result(url, "unknown")

    if not url_to_id:
        logger.warning("No valid AniDB URLs found")
        return results

    # Use single helper instance for proper rate limiting
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        results.update(
            loop.run_until_complete(_fetch_multiple_anidb_data_async(url_to_id))
        )
    finally:
        loop.close()

    logger.info(f"Completed processing {len(results)}/{len(urls)} AniDB URLs")
    return results


async def _fetch_multiple_anidb_data_async(
    url_to_id: dict[str, int],
) -> dict[str, dict[str, Any]]:
    """Fetch multiple AniDB URLs using single helper instance for proper rate limiting."""
    helper = AniDBEnrichmentHelper()
    results = {}

    try:
        for i, (url, anidb_id) in enumerate(url_to_id.items()):
            logger.info(
                f"Processing AniDB URL {i + 1}/{len(url_to_id)}: {url} (ID: {anidb_id})"
            )

            anime_data = await helper.fetch_all_data(anidb_id)
            if anime_data:
                results[url] = _parse_anidb_response(anime_data, url, anidb_id)
            else:
                results[url] = _create_fallback_result(url, anidb_id)

            # The helper already handles rate limiting, but log progress
            logger.debug(f"Completed AniDB request {i + 1}/{len(url_to_id)}")
    finally:
        await helper.close()

    return results


if __name__ == "__main__":
    # Test the helper
    import sys

    if len(sys.argv) != 2:
        print("Usage: python anidb_url_helper.py <anidb_url>")
        sys.exit(1)

    url = sys.argv[1]
    result = fetch_anidb_url_data(url)

    if result:
        print(f"Title: {result['title']}")
        print(f"Relationship: {result['relationship']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Context keys: {list(result.get('context', {}).keys())}")
        print(f"URL: {result['url']}")
    else:
        print(f"Failed to fetch data for URL: {url}")
