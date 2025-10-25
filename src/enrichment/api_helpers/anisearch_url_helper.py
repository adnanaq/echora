#!/usr/bin/env python3
"""
Lightweight AniSearch URL helper for Stage 3 relationship processing.

This is a focused helper specifically for extracting title and relationship data
from AniSearch URLs in the offline database, using MODB's proven header strategy.

Usage:
    from src.batch_enrichment.anisearch_url_helper import fetch_anisearch_url_data

    result = fetch_anisearch_url_data("https://anisearch.com/anime/302")

    # Result format:
    # {
    #     "title": "Anime Title",
    #     "relationship": "Inferred relationship type",
    #     "confidence": "high|medium|low",
    #     "url": "original_url"
    # }
"""

import logging
import re
import time
from typing import Any

import requests
from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)


def fetch_anisearch_url_data(url: str) -> dict[str, Any] | None:
    """
    Fetch title and relationship data from an AniSearch URL.

    Uses MODB's proven strategy of custom headers to bypass 403 blocking.

    Args:
        url: Full AniSearch URL (e.g., "https://anisearch.com/anime/302")

    Returns:
        Dict with title, relationship, confidence, and original URL
        None if fetch fails
    """
    try:
        # Extract anime ID from URL for context
        anime_id_match = re.search(r"/anime/(\d+)", url)
        anime_id = anime_id_match.group(1) if anime_id_match else "unknown"

        # Create session with MODB's exact header strategy
        session = requests.Session()
        session.headers.update(
            {
                "host": "www.anisearch.com",  # MODB's key strategy
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            }
        )

        # Rate limiting (conservative for Stage 3 batch processing)
        time.sleep(1.0)

        logger.debug(f"Fetching AniSearch URL: {url}")

        response = session.get(url, timeout=10)

        if response.status_code == 200:
            return _parse_anisearch_response(response.text, url, anime_id)
        else:
            logger.warning(
                f"AniSearch returned status {response.status_code} for URL {url}"
            )
            return _create_fallback_result(url, anime_id)

    except Exception as e:
        logger.error(f"Error fetching AniSearch URL {url}: {e}")
        anime_id_match = re.search(r"/anime/(\d+)", url)
        anime_id = anime_id_match.group(1) if anime_id_match else "unknown"
        return _create_fallback_result(url, anime_id)


def _parse_anisearch_response(
    html_content: str, url: str, anime_id: str
) -> dict[str, Any]:
    """Parse AniSearch HTML response to extract title and relationship info."""
    try:
        soup = BeautifulSoup(html_content, "html.parser")

        # Extract title using multiple strategies
        title = _extract_title_from_page(soup)
        if not title:
            title = f"AniSearch Anime {anime_id}"
            confidence = "low"
        else:
            confidence = "high"

        # Extract additional context for AI analysis instead of hardcoded inference
        context = _extract_context_data(soup)

        return {
            "title": title,
            "relationship": None,  # Let AI infer from context
            "confidence": confidence,
            "context": context,  # Provide rich context for AI inference
            "url": url,
            "source": "anisearch",
        }

    except Exception as e:
        logger.error(f"Error parsing AniSearch response for URL {url}: {e}")
        return _create_fallback_result(url, anime_id)


def _extract_title_from_page(soup: BeautifulSoup) -> str | None:
    """Extract anime title from AniSearch page using multiple strategies."""

    # Strategy 1: Try h1 heading (most reliable for anime pages)
    h1_tag = soup.find("h1")
    if h1_tag:
        title = h1_tag.get_text().strip()
        if title and not title.lower().startswith("anisearch"):
            return title

    # Strategy 2: Try OpenGraph title
    og_title = soup.find("meta", property="og:title")
    if og_title and isinstance(og_title, Tag):
        content = og_title.get("content")
        if isinstance(content, str):
            title = content.strip()
            # Clean AniSearch-specific parts
            title = re.sub(r" - AniSearch.*$", "", title)
            if title and not title.lower().startswith("anisearch"):
                return title

    # Strategy 3: Try page title
    title_tag = soup.find("title")
    if title_tag:
        title_text = title_tag.get_text().strip()
        # Clean AniSearch-specific parts
        title = re.sub(r" - AniSearch.*$", "", title_text)
        if title and title != title_text and not title.lower().startswith("anisearch"):
            return title

    # Strategy 4: Try JSON-LD structured data
    json_ld_script = soup.find("script", type="application/ld+json")
    if json_ld_script and hasattr(json_ld_script, "string") and json_ld_script.string:
        try:
            import json

            script_text = (
                json_ld_script.string
                if isinstance(json_ld_script.string, str)
                else str(json_ld_script.string)
            )
            data = json.loads(script_text)
            if isinstance(data, dict) and "name" in data:
                name = data["name"].strip()
                if name and not name.lower().startswith("anisearch"):
                    return name
        except:
            pass

    return None


def _extract_context_data(soup: BeautifulSoup) -> dict[str, Any]:
    """
    Extract contextual data for AI-powered relationship inference.

    Instead of hardcoded regex patterns, extract rich context that AI can analyze.
    """
    context: dict[str, Any] = {}

    # Extract description from OpenGraph or meta description
    og_desc = soup.find("meta", property="og:description")
    if og_desc and isinstance(og_desc, Tag):
        content = og_desc.get("content")
        if isinstance(content, str):
            context["description"] = content.strip()
    elif soup.find("meta", attrs={"name": "description"}):
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc and isinstance(meta_desc, Tag):
            content = meta_desc.get("content")
            if isinstance(content, str):
                context["description"] = content.strip()

    # Extract JSON-LD structured data for rich context
    json_ld_script = soup.find("script", type="application/ld+json")
    if json_ld_script and hasattr(json_ld_script, "string") and json_ld_script.string:
        try:
            import json

            script_text = (
                json_ld_script.string
                if isinstance(json_ld_script.string, str)
                else str(json_ld_script.string)
            )
            json_data = json.loads(script_text)
            if isinstance(json_data, dict):
                # Extract relevant structured data
                if "startDate" in json_data:
                    context["start_date"] = json_data["startDate"]
                if "endDate" in json_data:
                    context["end_date"] = json_data["endDate"]
                if "numberOfEpisodes" in json_data:
                    context["episodes"] = json_data["numberOfEpisodes"]
                if "@type" in json_data:
                    context["type"] = json_data["@type"]
        except:
            pass

    # Extract genre information
    genre_links = soup.find_all("a", href=re.compile(r"/genre/"))
    genres = []
    for link in genre_links:
        genre_text = link.get_text().strip()
        if genre_text and len(genre_text) < 30:
            genres.append(genre_text)
    if genres:
        context["genres"] = genres[:5]  # Limit to 5 most relevant

    # Extract any related anime information
    related_sections = soup.find_all(
        ["div", "section"], class_=re.compile(r"related|similar|franchise")
    )
    if related_sections:
        context["has_related_section"] = True
        # Try to extract some related titles for context
        related_titles = []
        for section in related_sections:
            if isinstance(section, Tag):
                anime_links = section.find_all("a", href=re.compile(r"/anime/"))
                for link in anime_links:
                    title = link.get_text().strip()
                    if title and len(title) < 100:
                        related_titles.append(title)
        if related_titles:
            context["related_titles"] = related_titles[:3]  # Limit to 3 for context

    return context


def _create_fallback_result(url: str, anime_id: str) -> dict[str, Any]:
    """Create fallback result when extraction fails."""
    return {
        "title": f"AniSearch Anime {anime_id}",
        "relationship": "Other",
        "confidence": "low",
        "url": url,
        "source": "anisearch",
    }


# Batch processing function for multiple URLs
def fetch_multiple_anisearch_urls(urls: list[str]) -> dict[str, dict[str, Any]]:
    """
    Fetch data for multiple AniSearch URLs with proper rate limiting.

    Args:
        urls: List of AniSearch URLs

    Returns:
        Dict mapping URL to result data
    """
    results = {}

    for i, url in enumerate(urls):
        logger.info(f"Processing AniSearch URL {i+1}/{len(urls)}: {url}")

        result = fetch_anisearch_url_data(url)
        if result:
            results[url] = result

        # Rate limiting between requests
        if i < len(urls) - 1:  # Don't sleep after the last request
            time.sleep(1.5)  # Conservative rate limiting

    logger.info(f"Completed processing {len(results)}/{len(urls)} AniSearch URLs")
    return results


if __name__ == "__main__":
    # Test the helper
    import sys

    if len(sys.argv) != 2:
        print("Usage: python anisearch_url_helper.py <anisearch_url>")
        sys.exit(1)

    url = sys.argv[1]
    result = fetch_anisearch_url_data(url)

    if result:
        print(f"Title: {result['title']}")
        print(f"Relationship: {result['relationship']}")
        print(f"Confidence: {result['confidence']}")
        print(f"URL: {result['url']}")
    else:
        print(f"Failed to fetch data for URL: {url}")
