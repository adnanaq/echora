"""
Crawls character information from anime-planet.com anime character pages with Redis caching.

Extracts comprehensive character data including detailed enrichment from individual
character pages using concurrent batch processing. Results are cached in Redis for
24 hours to avoid repeated crawling.

Usage:
    python -m src.enrichment.crawlers.anime_planet_character_crawler <identifier> [--output PATH]

    <identifier>    anime-planet.com anime identifier (slug, path, or full URL)
    --output PATH   optional output file path (default: animeplanet_characters.json)
"""

import argparse
import asyncio
import json
import sys
import logging

from typing import Any, Dict, List, Optional

from crawl4ai import (
    AsyncWebCrawler,
    CrawlerRunConfig,
    CrawlResult,
    JsonCssExtractionStrategy,
)
from crawl4ai.types import RunManyReturn

from http_cache.config import get_cache_config
from http_cache.result_cache import cached_result

from .utils import sanitize_output_path

logger = logging.getLogger(__name__)

# Get TTL from config to keep cache control centralized
_CACHE_CONFIG = get_cache_config()
TTL_ANIME_PLANET = _CACHE_CONFIG.ttl_anime_planet

BASE_URL = "https://www.anime-planet.com"


def _normalize_characters_url(anime_identifier: str) -> str:
    """
    Normalize various input formats to full anime-planet characters URL.

    Accepts:
        - Full URL: "https://www.anime-planet.com/anime/dandadan/characters"
        - Slug: "dandadan"
        - Path: "/anime/dandadan" or "/anime/dandadan/characters"

    Returns:
        Full URL: "https://www.anime-planet.com/anime/dandadan/characters"
    """
    if not anime_identifier.startswith("http"):
        # Remove leading slashes and "anime/" prefix if present
        clean_id = anime_identifier.lstrip("/")
        if clean_id.startswith("anime/"):
            clean_id = clean_id[6:]  # Remove "anime/" prefix
        # Remove "/characters" suffix if present (we'll add it back)
        clean_id = clean_id.rstrip("/").replace("/characters", "")
        url = f"{BASE_URL}/anime/{clean_id}/characters"
    else:
        url = anime_identifier
        # Ensure URL ends with /characters
        if not url.endswith("/characters"):
            url = url.rstrip("/") + "/characters"

    if not url.startswith(f"{BASE_URL}/anime/"):
        raise ValueError(
            f"Invalid URL: Must be an anime-planet.com anime URL. "
            f"Expected format: '{BASE_URL}/anime/<slug>/characters' or just '<slug>'"
        )

    return url


def _extract_slug_from_characters_url(url: str) -> str:
    """
    Extract the anime slug from an anime-planet characters URL.
    
    Returns:
        The slug segment following `/anime/` in the URL (e.g., `"dandadan"`).
    
    Raises:
        ValueError: If a slug cannot be found in the provided URL.
    """
    # Extract slug from: https://www.anime-planet.com/anime/dandadan/characters
    import re

    match = re.search(r"/anime/([^/?#]+)", url)
    if not match:
        raise ValueError(f"Could not extract slug from URL: {url}")
    return match.group(1)


@cached_result(ttl=TTL_ANIME_PLANET, key_prefix="animeplanet_characters")
async def _fetch_animeplanet_characters_data(canonical_slug: str) -> Optional[Dict[str, Any]]:
    """
    Fetch and return enriched character data for a canonical anime slug from anime-planet.com.
    
    Performs a multi-phase crawl: it fetches the characters list page, normalizes the list into basic character entries,
    concurrently fetches individual character detail pages to enrich those entries, and returns the combined result.
    Results are cached (TTL configured by the module) based solely on the provided canonical slug. This function has no
    side effects beyond network requests and returns None if fetching or extraction fails.
    
    Parameters:
        canonical_slug (str): Canonical anime slug (e.g., "dandadan"); must already be normalized by the caller.
    
    Returns:
        dict: A dictionary with keys:
            - "characters": List[dict] — enriched character dictionaries.
            - "total_count": int — number of characters in the list.
        Returns `None` if the crawl or extraction fails.
    """
    # Build URL from canonical slug (caller already normalized)
    characters_url = f"{BASE_URL}/anime/{canonical_slug}/characters"

    # Helper function to get reusable character field schema
    def _get_character_fields_schema() -> List[Dict[str, Any]]:
        """
        Provide the CSS extraction schema for fields present in an anime-planet character list row.
        
        Returns:
            list[dict]: A list of field-schema dictionaries describing how to extract values from a character row.
                Each dictionary contains keys like `name`, `selector`, and `type`, and may include `attribute`
                for attribute extraction or `fields` for nested list items. The schema includes entries for:
                `name`, `url`, `image_src`, `image_data_src`, `tags_raw`, and voice actor lists for JP/US/ES/FR.
        """
        return [
            {"name": "name", "selector": "a.name", "type": "text"},
            {
                "name": "url",
                "selector": "a.name",
                "type": "attribute",
                "attribute": "href",
            },
            {
                "name": "image_src",
                "selector": "img",
                "type": "attribute",
                "attribute": "src",
            },
            {
                "name": "image_data_src",
                "selector": "img",
                "type": "attribute",
                "attribute": "data-src",
            },
            {
                "name": "tags_raw",
                "selector": ".tags a",
                "type": "list",
                "fields": [{"name": "tag", "type": "text"}],
            },
            # Voice actors by language
            {
                "name": "voice_actors_jp",
                "selector": ".tableActors .flagJP a",
                "type": "list",
                "fields": [
                    {"name": "name", "type": "text"},
                    {"name": "url", "type": "attribute", "attribute": "href"},
                ],
            },
            {
                "name": "voice_actors_us",
                "selector": ".tableActors .flagUS a",
                "type": "list",
                "fields": [
                    {"name": "name", "type": "text"},
                    {"name": "url", "type": "attribute", "attribute": "href"},
                ],
            },
            {
                "name": "voice_actors_es",
                "selector": ".tableActors .flagES a",
                "type": "list",
                "fields": [
                    {"name": "name", "type": "text"},
                    {"name": "url", "type": "attribute", "attribute": "href"},
                ],
            },
            {
                "name": "voice_actors_fr",
                "selector": ".tableActors .flagFR a",
                "type": "list",
                "fields": [
                    {"name": "name", "type": "text"},
                    {"name": "url", "type": "attribute", "attribute": "href"},
                ],
            },
        ]

    # CSS schema uses helper function to avoid field duplication
    list_schema = {
        "baseSelector": "body",
        "fields": [
            # Extract Main characters
            {
                "name": "main_characters",
                "selector": "h3.sub:first-of-type + table tr",
                "type": "nested_list",
                "fields": _get_character_fields_schema(),
            },
            # Extract Secondary characters
            {
                "name": "secondary_characters",
                "selector": "h3.sub:nth-of-type(2) + table tr",
                "type": "nested_list",
                "fields": _get_character_fields_schema(),
            },
            # Extract Minor characters
            {
                "name": "minor_characters",
                "selector": "h3.sub:nth-of-type(3) + table tr",
                "type": "nested_list",
                "fields": _get_character_fields_schema(),
            },
        ],
    }

    async with AsyncWebCrawler() as crawler:
        logger.info(f"Fetching character list: {characters_url}")

        # Phase 1: Fetch character list page
        list_config = CrawlerRunConfig(
            extraction_strategy=JsonCssExtractionStrategy(list_schema)
        )
        results: RunManyReturn = await crawler.arun(
            url=characters_url, config=list_config
        )

        if not results:
            logger.warning(f"Failed to fetch character list: {characters_url}")
            return None

        for result in results:
            if not isinstance(result, CrawlResult):
                raise TypeError(
                    f"Unexpected result type: {type(result)}, expected CrawlResult."
                )

            if result.success and result.extracted_content:
                data = json.loads(result.extracted_content)

                if not data:
                    logger.warning("Extraction returned empty data.")
                    continue

                # Phase 2: Process character list and assign roles
                characters_basic = _process_character_list(
                    data[0],
                )

                if not characters_basic:
                    logger.warning("No characters found.")
                    return None

                logger.info(f"Found {len(characters_basic)} characters")

                # Phase 3: Prepare character detail URLs for concurrent enrichment
                character_detail_urls = []
                for char in characters_basic:
                    # Extract slug from URL using simple string operations
                    url = char.get("url", "")
                    if url and "/characters/" in url:
                        char_slug = (
                            url.split("/characters/")[1].split("?")[0].split("#")[0]
                        )
                        character_detail_urls.append(
                            f"{BASE_URL}/characters/{char_slug}"
                        )

                if not character_detail_urls:
                    logger.warning("No valid character URLs found.")
                    return None

                # Phase 4: CONCURRENT BATCH ENRICHMENT using arun_many()
                logger.info(f"Enriching {len(character_detail_urls)} characters concurrently..."
                )

                detail_schema = _get_character_detail_schema()
                detail_config = CrawlerRunConfig(
                    extraction_strategy=JsonCssExtractionStrategy(detail_schema),
                )

                # Concurrent batch fetch of all character detail pages
                list_results = await crawler.arun_many(
                    urls=character_detail_urls,
                    config=detail_config,
                )

                # Validate the results - arun_many returns List[CrawlResultContainer]
                if not list_results:
                    logger.warning("No results returned from batch character fetch")
                    return None

                if not isinstance(list_results, list):
                    logger.warning(
                        f"Unexpected return type from arun_many: {type(list_results)}"
                    )
                    return None

                # Unwrap CrawlResultContainer objects to get CrawlResult objects
                # arun_many() returns List[CrawlResultContainer], each wrapping a CrawlResult
                from crawl4ai.models import CrawlResultContainer

                unwrapped_results: List[CrawlResult] = []
                for container in list_results:
                    # CrawlResultContainer is iterable and yields CrawlResult objects
                    if isinstance(container, CrawlResultContainer):
                        for result in container:
                            if isinstance(result, CrawlResult):
                                unwrapped_results.append(result)
                    elif isinstance(container, CrawlResult):
                        # Handle direct CrawlResult (shouldn't happen but be safe)
                        unwrapped_results.append(container)

                if not unwrapped_results:
                    logger.warning("No valid CrawlResult objects found after unwrapping")
                    return None

                # Replace list_results with unwrapped results
                list_results = unwrapped_results

                # Phase 5: Merge enriched data
                # Create a mapping from character name to index for proper matching
                char_name_to_index = {
                    char["name"]: i for i, char in enumerate(characters_basic)
                }

                enriched_count = 0
                for detail_result in list_results:

                    if detail_result.success and detail_result.extracted_content:
                        try:
                            detail_data = json.loads(detail_result.extracted_content)
                            if detail_data:
                                # Extract name from detail page to match with basic list
                                detail_name = detail_data[0].get("name_h1", "").strip()

                                if detail_name in char_name_to_index:
                                    idx = char_name_to_index[detail_name]
                                    enriched_data = _process_character_details(
                                        detail_data[0]
                                    )
                                    characters_basic[idx].update(enriched_data)
                                    enriched_count += 1
                                else:
                                    logger.warning(
                                        f"Could not match character: {detail_name}"
                                    )
                        except (
                            json.JSONDecodeError,
                            KeyError,
                            IndexError,
                            TypeError,
                        ) as e:
                            logger.warning(f"Failed to process enrichment: {e}")
                    else:
                        logger.warning(
                            f"Failed to enrich detail page - {detail_result.error_message}"
                        )

                logger.info(
                    f"Successfully enriched {enriched_count}/{len(characters_basic)} characters"
                )

                # Prepare output data
                output_data = {
                    "characters": characters_basic,
                    "total_count": len(characters_basic),
                }

                # Always return data (no conditional return or file writing)
                return output_data
            else:
                logger.warning(f"Extraction failed: {result.error_message}")
                return None
        return None


async def fetch_animeplanet_characters(
    slug: str, output_path: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Orchestrates retrieval of Anime-Planet character data, performs side effects (file output), and returns the result.

    Normalizes the provided identifier to a canonical slug, invokes the cached internal fetcher keyed by that slug, and — if data is returned — writes a sanitized JSON file to `output_path` when provided. The file write executes regardless of whether the data came from cache or a fresh crawl.

    Parameters:
        slug: Anime identifier in any supported form (slug, path, or full characters page URL).
        output_path: Optional filesystem path where the JSON result will be written (path is sanitized before writing).

    Returns:
        The enriched character data dictionary if data was found, `None` otherwise.
    """
    # Normalize identifier once so cache keys depend on canonical slug
    # This ensures cache reuse across different identifier formats
    characters_url = _normalize_characters_url(slug)
    canonical_slug = _extract_slug_from_characters_url(characters_url)

    # Fetch data from cache or crawl (pure function keyed only on canonical slug)
    data = await _fetch_animeplanet_characters_data(canonical_slug)

    if data is None:
        return None

    # Side effect: Write to file (always executes, even on cache hit)
    if output_path:
        safe_path = sanitize_output_path(output_path)
        with open(safe_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Data written to {safe_path}")

    return data


def _get_character_detail_schema() -> Dict[str, Any]:
    """
    Return a CSS-based extraction schema for an anime-planet character detail page.
    
    The schema describes selectors and field shapes used by the crawler to extract name, image, entry bar items (gender, hair color, ranks), metadata items (eye color, age, birthday, etc.), description paragraphs, tags, alternative names, and raw anime/manga role table rows. Table content is selected here but full table parsing is performed separately after extraction.
    
    Returns:
        Dict[str, Any]: Extraction schema mapping compatible with the crawler's CSS/JSON extraction engine.
    """
    return {
        "baseSelector": "body",
        "fields": [
            # Basic info
            {"name": "name_h1", "selector": "h1[itemprop='name']", "type": "text"},
            {
                "name": "image_detail",
                "selector": "img[itemprop='image']",
                "type": "attribute",
                "attribute": "src",
            },
            # Entry bar data (contains gender, hair color, ranks)
            {
                "name": "entry_bar_items",
                "selector": ".entryBar .pure-1",
                "type": "list",
                "fields": [{"name": "text", "type": "text"}],
            },
            # Loved rank
            {
                "name": "loved_rank",
                "selector": ".entryBar .fa-heart + a",
                "type": "text",
            },
            # Hated rank
            {
                "name": "hated_rank",
                "selector": ".entryBar .heartOff + a",
                "type": "text",
            },
            # Metadata section (eye color, age, birthday, etc.)
            {
                "name": "metadata_items",
                "selector": ".EntryMetadata__item",
                "type": "nested_list",
                "fields": [
                    {
                        "name": "title",
                        "selector": ".EntryMetadata__title",
                        "type": "text",
                    },
                    {
                        "name": "value",
                        "selector": ".EntryMetadata__value",
                        "type": "text",
                    },
                ],
            },
            # Description
            {
                "name": "description_paragraphs",
                "selector": ".entrySynopsis p",
                "type": "list",
                "fields": [{"name": "text", "type": "text"}],
            },
            # Tags
            {
                "name": "tags_detail",
                "selector": ".tags a",
                "type": "list",
                "fields": [{"name": "tag", "type": "text"}],
            },
            # Alternative names
            {
                "name": "alt_names_raw",
                "selector": ".entryAltNames li",
                "type": "list",
                "fields": [{"name": "name", "type": "text"}],
            },
            # Anime roles - extract table rows following "Anime Roles" header
            {
                "name": "anime_roles_raw",
                "selector": "h3:contains('Anime Roles') + table tr",
                "type": "nested_list",
                "fields": [
                    {
                        "name": "anime_title",
                        "selector": "td:nth-child(1) a",
                        "type": "text",
                    },
                    {
                        "name": "anime_url",
                        "selector": "td:nth-child(1) a",
                        "type": "attribute",
                        "attribute": "href",
                    },
                    {"name": "role", "selector": "td:nth-child(2)", "type": "text"},
                ],
            },
            # Manga roles - extract table rows following "Manga Roles" header
            {
                "name": "manga_roles_raw",
                "selector": "h3:contains('Manga Roles') + table tr",
                "type": "nested_list",
                "fields": [
                    {
                        "name": "manga_title",
                        "selector": "td:nth-child(1) a",
                        "type": "text",
                    },
                    {
                        "name": "manga_url",
                        "selector": "td:nth-child(1) a",
                        "type": "attribute",
                        "attribute": "href",
                    },
                    {"name": "role", "selector": "td:nth-child(2)", "type": "text"},
                ],
            },
        ],
    }


def _process_character_list(list_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Process character list data and assign roles based on section headers.

    Args:
        list_data: Extracted data with main_characters, secondary_characters, minor_characters

    Returns:
        Flat list of character dictionaries with role assigned
    """
    characters = []

    # Process each role section separately
    role_sections = {
        "Main": list_data.get("main_characters", []),
        "Secondary": list_data.get("secondary_characters", []),
        "Minor": list_data.get("minor_characters", []),
    }

    for role, character_rows in role_sections.items():
        for row in character_rows:
            name = row.get("name", "").strip()
            url = row.get("url", "").strip()

            if not name or not url:
                continue

            char_data = {
                "name": name,
                "url": url,
                "role": role,  # Role is determined by which section it came from
            }

            # Handle image (src or data-src fallback)
            image = row.get("image_src") or row.get("image_data_src")
            if image:
                char_data["image"] = image

            # Parse tags
            tags_raw = row.get("tags_raw", [])
            if tags_raw:
                tags = [
                    tag.get("tag", "").strip() for tag in tags_raw if tag.get("tag")
                ]
                if tags:
                    char_data["tags"] = tags

            # Parse voice actors using helper function
            voice_actors = _extract_voice_actors(row)
            if voice_actors:
                char_data["voice_actors"] = voice_actors

            characters.append(char_data)

    return characters


def _extract_voice_actors(character: Dict[str, Any]) -> Dict[str, List[Dict[str, str]]]:
    """Extract voice actors by language from character data.

    Args:
        character: Character dictionary containing voice actor data

    Returns:
        Dictionary mapping language codes to voice actor lists
    """
    voice_actors: Dict[str, List[Dict[str, str]]] = {}

    for lang_code in ["jp", "us", "es", "fr"]:
        va_list = character.get(f"voice_actors_{lang_code}", [])
        if va_list:
            voice_actors[lang_code] = [
                {"name": va.get("name", "").strip(), "url": va.get("url", "")}
                for va in va_list
                if va.get("name")
            ]

    return voice_actors


def _normalize_value(value: str) -> Optional[str]:
    """Convert '?' to None, otherwise return stripped value."""
    stripped = value.strip()
    return None if stripped == "?" else stripped


def _process_character_details(detail_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert extracted character detail page data into a flattened dictionary of enriched character attributes.
    
    Parameters:
        detail_data (Dict[str, Any]): Mapping produced by the detail-page CSS extraction schema. Expected keys include:
            - entry_bar_items: list of {"text": ...} entries used to extract `gender` and `hair_color`.
            - loved_rank, hated_rank: rank strings (e.g. "#1", "1,234") parsed to integers.
            - metadata_items: list of {"title": ..., "value": ...} entries converted into snake_case fields (e.g. "Eye Color" -> `eye_color`).
            - description_paragraphs: list of {"text": ...} entries; the first substantive paragraph is used for `description`.
            - alt_names_raw: list of alternative-name objects used to build `alternative_names`.
            - anime_roles_raw, manga_roles_raw: lists of role objects used to build `anime_roles` and `manga_roles`.
    
    Returns:
        Dict[str, Any]: Enriched character fields, which may include:
            - gender, hair_color (str or None)
            - loved_rank, hated_rank (int)
            - metadata-derived snake_case fields (str or None)
            - description (str)
            - alternative_names (List[str])
            - anime_roles, manga_roles (List[Dict] with title, url and optional role)
    """
    enriched: Dict[str, Any] = {}

    # Extract gender and hair color from entry bar
    entry_bar_items = detail_data.get("entry_bar_items", [])

    for item in entry_bar_items:
        text = item.get("text", "")

        if "Gender:" in text:
            gender_value = text.replace("Gender:", "").strip()
            enriched["gender"] = _normalize_value(gender_value)

        if "Hair Color:" in text:
            hair_value = text.replace("Hair Color:", "").strip()
            enriched["hair_color"] = _normalize_value(hair_value)

    # Extract loved/hated ranks from CSS
    loved_rank_text = detail_data.get("loved_rank", "").strip()
    if loved_rank_text:
        # Remove #, commas and convert to int
        try:
            enriched["loved_rank"] = int(
                loved_rank_text.replace("#", "").replace(",", "")
            )
        except ValueError:
            pass

    hated_rank_text = detail_data.get("hated_rank", "").strip()
    if hated_rank_text:
        try:
            enriched["hated_rank"] = int(
                hated_rank_text.replace("#", "").replace(",", "")
            )
        except ValueError:
            pass

    # Extract metadata (eye_color, age, birthday, etc.)
    metadata_items = detail_data.get("metadata_items", [])
    for item in metadata_items:
        title = item.get("title", "").strip()
        value = item.get("value", "").strip()

        if title and value:
            # Convert "Eye Color" -> "eye_color"
            field_name = title.lower().replace(" ", "_")
            enriched[field_name] = _normalize_value(value)

    # Extract description
    desc_paragraphs = detail_data.get("description_paragraphs", [])
    for p in desc_paragraphs:
        text = p.get("text", "").strip()
        if text and len(text) > 50 and "Tags" not in text:
            enriched["description"] = text
            break

    # Extract alternative names
    alt_names_raw = detail_data.get("alt_names_raw", [])
    if alt_names_raw:
        alt_names = [
            item.get("name", "").strip() for item in alt_names_raw if item.get("name")
        ]
        if alt_names:
            enriched["alternative_names"] = alt_names

    # Parse anime roles from CSS-extracted data
    anime_roles_raw = detail_data.get("anime_roles_raw", [])
    if anime_roles_raw:
        anime_roles = []
        for role_data in anime_roles_raw:
            anime_title = role_data.get("anime_title", "").strip()
            anime_url = role_data.get("anime_url", "").strip()
            role = role_data.get("role", "").strip()

            if anime_title and anime_url:
                anime_role = {
                    "anime_title": anime_title,
                    "anime_url": anime_url,
                }
                # Add role if present (Main, Secondary, Minor, etc.)
                if role:
                    anime_role["role"] = role
                anime_roles.append(anime_role)

        if anime_roles:
            enriched["anime_roles"] = anime_roles

    # Parse manga roles from CSS-extracted data
    manga_roles_raw = detail_data.get("manga_roles_raw", [])
    if manga_roles_raw:
        manga_roles = []
        for role_data in manga_roles_raw:
            manga_title = role_data.get("manga_title", "").strip()
            manga_url = role_data.get("manga_url", "").strip()
            role = role_data.get("role", "").strip()

            if manga_title and manga_url:
                manga_role = {
                    "manga_title": manga_title,
                    "manga_url": manga_url,
                }
                # Add role if present (Main, Secondary, Minor, etc.)
                if role:
                    manga_role["role"] = role
                manga_roles.append(manga_role)

        if manga_roles:
            enriched["manga_roles"] = manga_roles

    return enriched


async def main() -> int:
    """
    Parse command-line arguments and run the anime-planet character crawler, writing results to a file.
    
    Runs the crawler with the provided anime identifier (slug, path, or full URL) and writes output to the specified file path. Logs errors and returns a non-zero exit code on failure.
    
    Returns:
        int: 0 on success, 1 on failure.
    """
    parser = argparse.ArgumentParser(
        description="Crawl character data from an anime-planet.com anime characters page."
    )
    parser.add_argument(
        "identifier",
        type=str,
        help="Anime identifier: slug (e.g., 'dandadan'), path (e.g., '/anime/dandadan/characters'), or full URL",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="animeplanet_characters.json",
        help="Output file path (default: animeplanet_characters.json in current directory)",
    )
    args = parser.parse_args()

    try:
        data = await fetch_animeplanet_characters(
            args.identifier,
            output_path=args.output,
        )
        if data is None:
            logger.error(
                "No character data was extracted; see logs above for details."
            )
            return 1
    except (ValueError, OSError):
        logger.exception("Failed to fetch anime-planet character data")
        return 1
    except Exception:
        logger.exception("Unexpected error during character fetch")
        return 1
    return 0

if __name__ == "__main__":  # pragma: no cover
    sys.exit(asyncio.run(main()))
