"""
Crawls anime information from anime-planet.com via the crawl4ai Docker REST API.

Extracts comprehensive anime data including related anime, rankings, studios,
and all metadata from JSON-LD.  Results are cached in Redis for 24 hours.
"""

import json
import logging
import re
from typing import Any

from enrichment.sources.anime_planet.anime_planet_models import (
    AnimePlanetAggregateRating,
    AnimePlanetAnime,
    AnimePlanetMangaEntry,
    AnimePlanetRelatedEntry,
)
from enrichment.sources.anime_planet.animeplanet_mapper import anime_from_animeplanet
from enrichment.sources.base.crawl4ai_docker import crawl_single_url
from enrichment.sources.base.crawler_config import (
    get_docker_browser_config,
    get_docker_crawler_config,
)
from enrichment.sources.base.framework import BaseCrawler, DockerTransport, NullRepository
from http_cache.config import get_cache_config
from http_cache.result_cache import cached_result

logger = logging.getLogger(__name__)

_CACHE_CONFIG = get_cache_config()
TTL_ANIME_PLANET = _CACHE_CONFIG.ttl_anime_planet

BASE_ANIME_URL = "https://www.anime-planet.com/anime/"

_SEASON_SLUG_RE = re.compile(r"/seasons/([^/?#]+)")
_RANK_RE = re.compile(r"#(\d+)")
_AKA_PREFIX = "alt title:"


def _parse_season(season_url: str | None) -> str | None:
    """Extract season name from AP season href e.g. '/anime/seasons/fall-1999' → 'fall'."""
    if not season_url:
        return None
    match = _SEASON_SLUG_RE.search(season_url)
    if not match:
        return None
    return match.group(1).split("-")[0].lower()


def _parse_rank(rank_text: str | None) -> int | None:
    """Parse rank integer from text like 'Rank #157' → 157."""
    if not rank_text:
        return None
    match = _RANK_RE.search(rank_text)
    return int(match.group(1)) if match else None


def _parse_alt_title(aka: str | None) -> str | None:
    """Strip 'Alt title: ' prefix from h2.aka text and return the bare title."""
    if not aka:
        return None
    text = aka.strip()
    if text.lower().startswith(_AKA_PREFIX):
        text = text[len(_AKA_PREFIX) :].strip()
    return text or None


def _normalize_anime_url(anime_identifier: str) -> str:
    """Normalize various input formats to a full anime-planet URL.

    Accepts:
        - Full URL (www):     "https://www.anime-planet.com/anime/dandadan"
        - Full URL (non-www): "https://anime-planet.com/anime/dandadan"
        - Slug:               "dandadan"
        - Path:               "/anime/dandadan"

    Returns:
        Full URL: "https://www.anime-planet.com/anime/dandadan"
    """
    if not anime_identifier.startswith("http"):
        clean_id = anime_identifier.lstrip("/")
        if clean_id.startswith("anime/"):
            clean_id = clean_id[6:]
        url = f"{BASE_ANIME_URL}{clean_id}"
    else:
        # Normalize non-www to www (offline DB stores URLs without www)
        url = anime_identifier.replace(
            "https://anime-planet.com/", "https://www.anime-planet.com/"
        )

    if not url.startswith(BASE_ANIME_URL):
        raise ValueError(f"Not an anime-planet anime URL: {url!r}")
    return url


def _extract_slug_from_url(url: str) -> str:
    """Extract the anime slug from a canonical Anime-Planet anime URL.

    Raises:
        ValueError: If a slug cannot be found in the provided URL.
    """
    match = re.search(r"/anime/([^/?#]+)", url)
    if not match:
        raise ValueError(f"No anime slug in URL: {url!r}")
    return match.group(1)


def _extract_json_ld(html: str) -> dict[str, Any] | None:
    """Extract JSON-LD structured data from an HTML document.

    Parses the first <script type="application/ld+json"> block and returns
    its content as a dict.  HTML entities in `description` are unescaped and
    known malformed image URLs are corrected.
    """
    try:
        import html as html_lib
        from typing import cast

        match = re.search(
            r'<script type="application/ld\+json">\s*(\{.*?\})\s*</script>',
            html,
            re.DOTALL,
        )
        if match:
            json_text = match.group(1).replace(r"\/", "/")
            json_ld = cast(dict[str, Any], json.loads(json_text))

            if json_ld.get("description"):
                json_ld["description"] = html_lib.unescape(
                    cast(str, json_ld["description"])
                )

            if json_ld.get("image") and "anime-planet.comhttps://" in cast(
                str, json_ld["image"]
            ):
                json_ld["image"] = cast(str, json_ld["image"]).replace(
                    "https://www.anime-planet.comhttps://", "https://"
                )

            return json_ld
    except (json.JSONDecodeError, AttributeError) as e:
        logger.warning(f"Failed to extract JSON-LD: {e}")
    return None


def _get_anime_schema() -> dict[str, Any]:
    """Return the XPath extraction schema for an anime-planet anime page."""
    return {
        "name": "AnimePlanetAnime",
        "baseSelector": "//body",
        "fields": [
            {
                "name": "type_raw",
                "selector": "//section[contains(@class,'entryBar')]//span[@class='type']",
                "type": "text",
            },
            {
                "name": "season_url",
                "selector": "//section[contains(@class,'entryBar')]//a[contains(@href,'/anime/seasons/')]",
                "type": "attribute",
                "attribute": "href",
            },
            {
                "name": "rank_text",
                "selector": "//section[contains(@class,'entryBar')]//div[contains(.,'Rank #')]",
                "type": "text",
            },
            {
                "name": "avg_rating_title",
                "selector": "//div[contains(@class,'avgRating')]",
                "type": "attribute",
                "attribute": "title",
            },
            {
                "name": "studios",
                "selector": "//section[contains(@class,'entryBar')]//a[contains(@href,'/studios/')]",
                "type": "list",
                "fields": [{"name": "name", "selector": ".", "type": "text"}],
            },
            {
                "name": "aka",
                "selector": "//h2[contains(@class,'aka')]",
                "type": "text",
            },
            {
                "name": "tags",
                "selector": "//div[contains(@class,'tags')]//a[contains(@href,'/anime/tags/')]",
                "type": "list",
                "fields": [{"name": "name", "selector": ".", "type": "text"}],
            },
            {
                "name": "cover",
                "selector": "//img[@itemprop='image']",
                "type": "attribute",
                "attribute": "src",
            },
            {
                "name": "related_anime_raw",
                "selector": "//div[@id='tabs--relations--anime--same_franchise']//a[contains(@class,'RelatedEntry')]",
                "type": "nested_list",
                "fields": [
                    {
                        "name": "url",
                        "selector": ".",
                        "type": "attribute",
                        "attribute": "href",
                    },
                    {
                        "name": "title",
                        "selector": ".//p[contains(@class,'RelatedEntry__name')]",
                        "type": "text",
                    },
                    {
                        "name": "relation_subtype",
                        "selector": ".//span[contains(@class,'RelatedEntry__subtitle')]",
                        "type": "text",
                    },
                    # fa-tv li only — avoids picking up date spans from the calendar li
                    {
                        "name": "type",
                        "selector": ".//li[.//i[contains(@class,'fa-tv')]]//span[contains(@class,'RelatedEntry__metadata_item')]",
                        "type": "text",
                    },
                    {
                        "name": "image",
                        "selector": ".//img[contains(@class,'RelatedEntry__image')]",
                        "type": "attribute",
                        "attribute": "src",
                    },
                ],
            },
            {
                "name": "related_anime_other_raw",
                "selector": "//div[@id='tabs--relations--anime--other_franchise']//a[contains(@class,'RelatedEntry')]",
                "type": "nested_list",
                "fields": [
                    {
                        "name": "url",
                        "selector": ".",
                        "type": "attribute",
                        "attribute": "href",
                    },
                    {
                        "name": "title",
                        "selector": ".//p[contains(@class,'RelatedEntry__name')]",
                        "type": "text",
                    },
                    {
                        "name": "relation_subtype",
                        "selector": ".//span[contains(@class,'RelatedEntry__subtitle')]",
                        "type": "text",
                    },
                    {
                        "name": "type",
                        "selector": ".//li[.//i[contains(@class,'fa-tv')]]//span[contains(@class,'RelatedEntry__metadata_item')]",
                        "type": "text",
                    },
                    {
                        "name": "image",
                        "selector": ".//img[contains(@class,'RelatedEntry__image')]",
                        "type": "attribute",
                        "attribute": "src",
                    },
                ],
            },
            {
                "name": "related_manga_raw",
                "selector": "//div[contains(@id,'tabs--relations--manga')]//a[contains(@class,'RelatedEntry')]",
                "type": "nested_list",
                "fields": [
                    {
                        "name": "url",
                        "selector": ".",
                        "type": "attribute",
                        "attribute": "href",
                    },
                    {
                        "name": "title",
                        "selector": ".//p[contains(@class,'RelatedEntry__name')]",
                        "type": "text",
                    },
                    {
                        "name": "relation_subtype",
                        "selector": ".//span[contains(@class,'RelatedEntry__subtitle')]",
                        "type": "text",
                    },
                    # fa-book-open li — "One Shot" or "Vol: X - Ch: Y" counts
                    {
                        "name": "vol_ch",
                        "selector": ".//li[.//i[contains(@class,'fa-book-open')]]//span[contains(@class,'RelatedEntry__metadata_item')]",
                        "type": "text",
                    },
                    {
                        "name": "image",
                        "selector": ".//img[contains(@class,'RelatedEntry__image')]",
                        "type": "attribute",
                        "attribute": "src",
                    },
                ],
            },
        ],
    }


def _build_related_anime_entries(
    raw: list[dict[str, Any]],
) -> list[AnimePlanetRelatedEntry]:
    """Build AnimePlanetRelatedEntry models from raw XPath nested_list output.

    Parses the fa-tv metadata span (e.g. "OVA: 1 ep", "Movie", "TV Special: 9 ep")
    into separate type and episode_count fields.
    """
    entries = []
    for item in raw:
        url_val = (item.get("url") or "").strip()
        title = (item.get("title") or "").strip()
        if not url_val:
            continue
        slug_match = re.search(r"/anime/([^/?#]+)", url_val)
        if not slug_match:
            continue

        raw_type = (item.get("type") or "").strip()
        if ":" in raw_type:
            type_part, ep_part = raw_type.split(":", 1)
            type_clean: str | None = type_part.strip() or None
            ep_match = re.search(r"(\d+)", ep_part)
            episode_count: int | None = int(ep_match.group(1)) if ep_match else None
        else:
            type_clean = raw_type or None
            episode_count = None

        entries.append(
            AnimePlanetRelatedEntry(
                url=url_val,
                slug=slug_match.group(1),
                title=title,
                relation_subtype=item.get("relation_subtype") or None,
                type=type_clean,
                episode_count=episode_count,
                image=item.get("image") or None,
            )
        )
    return entries


def _build_related_manga_entries(
    raw: list[dict[str, Any]],
) -> list[AnimePlanetMangaEntry]:
    """Build AnimePlanetMangaEntry models from raw XPath nested_list output.

    Parses the fa-book-open metadata span into type, volumes, and chapters:
      - "One Shot"           → type="One Shot", chapters=1
      - "Vol: 114 - Ch: 179" → volumes=114, chapters=179
      - "Vol: 1"             → volumes=1
      - "Ch: 19"             → chapters=19
      - ""  / "- ?"          → all None (date bleed-through guard)
    """
    entries = []
    for item in raw:
        url_val = (item.get("url") or "").strip()
        title = (item.get("title") or "").strip()
        if not url_val:
            continue
        slug_match = re.search(r"/manga/([^/?#]+)", url_val)
        if not slug_match:
            continue

        vol_ch = (item.get("vol_ch") or "").strip()
        manga_type: str | None = None
        volumes: int | None = None
        chapters: int | None = None

        if vol_ch.lower() == "one shot":
            manga_type = "One Shot"
            chapters = 1
        elif vol_ch:
            vol_match = re.search(r"Vol:\s*(\d+)", vol_ch, re.IGNORECASE)
            ch_match = re.search(r"Ch:\s*(\d+)", vol_ch, re.IGNORECASE)
            volumes = int(vol_match.group(1)) if vol_match else None
            chapters = int(ch_match.group(1)) if ch_match else None

        entries.append(
            AnimePlanetMangaEntry(
                url=url_val,
                slug=slug_match.group(1),
                title=title,
                relation_subtype=item.get("relation_subtype") or None,
                type=manga_type,
                volumes=volumes,
                chapters=chapters,
                image=item.get("image") or None,
            )
        )
    return entries


def _parse_aggregate_rating(
    ar: dict[str, Any] | None,
) -> AnimePlanetAggregateRating | None:
    """Parse a JSON-LD aggregateRating dict into an AnimePlanetAggregateRating model."""
    if not ar:
        return None
    rating_value: float | None = None
    rating_count: int | None = None
    if ar.get("ratingValue") is not None:
        try:
            rating_value = float(ar["ratingValue"])
        except (ValueError, TypeError):
            pass
    if ar.get("ratingCount") is not None:
        try:
            rating_count = int(ar["ratingCount"])
        except (ValueError, TypeError):
            pass
    if rating_value is None and rating_count is None:
        return None
    return AnimePlanetAggregateRating(
        rating_value=rating_value, rating_count=rating_count
    )


def _build_anime_from_raw(raw: dict[str, Any]) -> AnimePlanetAnime:
    """Construct an AnimePlanetAnime model from a cached raw data dict.

    This is the post-processing step that converts the JSON-serializable cached
    dict into typed Pydantic models, mirroring the ``_build_anime_from_raw``
    pattern used in the MAL crawler.

    Args:
        raw: Dict returned by ``_fetch_animeplanet_anime_data`` — contains
             merged JSON-LD scalars, XPath primitives, and raw relation lists.

    Returns:
        Validated AnimePlanetAnime source model.
    """
    return AnimePlanetAnime(
        name=raw["name"],
        schema_type=raw.get("schema_type"),
        description=raw.get("description"),
        url=raw.get("url"),
        start_date=raw.get("start_date"),
        end_date=raw.get("end_date"),
        number_of_episodes=raw.get("number_of_episodes"),
        genres=raw.get("genres", []),
        aggregate_rating=_parse_aggregate_rating(raw.get("aggregate_rating")),
        type_raw=raw.get("type_raw"),
        season=_parse_season(raw.get("season_url")),
        rank=_parse_rank(raw.get("rank_text")),
        alt_title=_parse_alt_title(raw.get("aka")),
        cover=raw.get("cover"),
        studios=raw.get("studios", []),
        tags=raw.get("tags", []),
        slug=raw["slug"],
        related_anime=_build_related_anime_entries(raw.get("related_anime_raw", [])),
        related_anime_other=_build_related_anime_entries(
            raw.get("related_anime_other_raw", [])
        ),
        related_manga=_build_related_manga_entries(raw.get("related_manga_raw", [])),
    )


@cached_result(
    ttl=TTL_ANIME_PLANET,
    key_prefix="animeplanet_anime",
    dependencies=[_get_anime_schema],
)
async def _fetch_animeplanet_anime_data(
    canonical_slug: str,
) -> dict[str, Any] | None:
    """Fetch and extract raw anime data for a given anime-planet slug.

    Uses the crawl4ai Docker REST API with XPath extraction. Cached by
    canonical slug; cache is automatically invalidated when the extraction
    schema changes.

    Returns a JSON-serializable dict of primitives ready for
    ``_build_anime_from_raw``.  Model construction is deliberately left to
    ``_build_anime_from_raw`` so that Pydantic models are never stored in the
    cache (they are not JSON-serializable by default).

    Args:
        canonical_slug: Canonical anime slug (e.g. "one-piece").

    Returns:
        Raw data dict, or None on failure.
    """
    url = f"{BASE_ANIME_URL}{canonical_slug}"

    logger.info(f"Fetching anime data: {url}")
    result = await crawl_single_url(
        url,
        browser_config=get_docker_browser_config(),
        crawler_config=get_docker_crawler_config(
            _get_anime_schema(), wait_until="load"
        ),
    )

    if result is None:
        logger.warning(f"Crawl returned no result for {url}")
        return None

    status = result.get("status_code")
    if status == 404:
        logger.warning(f"Anime not found (404): {url}")
        return None
    if status and status >= 400:
        logger.error(f"HTTP {status} for anime {url}")
        return None
    if status and 300 <= status < 400:
        logger.debug(f"HTTP {status} (redirect followed) for anime {url}")

    if not result.get("success"):
        logger.warning(f"Crawl failed for {url}: {result.get('error_message')}")
        return None

    raw_list = json.loads(result.get("extracted_content") or "[]")
    if not raw_list:
        logger.warning(f"No data extracted from {url}")
        return None

    xpath = raw_list[0]
    html = result.get("html") or ""

    # JSON-LD is the primary source for title, dates, episodes, and ratings.
    json_ld = _extract_json_ld(html) if html else None
    if not json_ld or not json_ld.get("name"):
        logger.warning(f"No JSON-LD found for {url}")
        return None

    return {
        # JSON-LD scalars (primitive values only — no model construction)
        "name": json_ld["name"],
        "schema_type": json_ld.get("@type"),
        "description": json_ld.get("description"),
        "url": json_ld.get("url"),
        "start_date": json_ld.get("startDate"),
        "end_date": json_ld.get("endDate"),
        "number_of_episodes": json_ld.get("numberOfEpisodes"),
        "genres": json_ld.get("genre") or [],
        "aggregate_rating": json_ld.get(
            "aggregateRating"
        ),  # raw dict, parsed in _build_anime_from_raw
        # XPath scalars
        "type_raw": xpath.get("type_raw") or None,
        "season_url": xpath.get("season_url") or None,
        "rank_text": xpath.get("rank_text") or None,
        "aka": xpath.get("aka") or None,
        "cover": xpath.get("cover") or None,
        "studios": [s["name"] for s in xpath.get("studios", []) if s.get("name")],
        "tags": [t["name"] for t in xpath.get("tags", []) if t.get("name")],
        "slug": canonical_slug,
        # Raw relation lists — model construction happens in _build_anime_from_raw
        "related_anime_raw": xpath.get("related_anime_raw", []),
        "related_anime_other_raw": xpath.get("related_anime_other_raw", []),
        "related_manga_raw": xpath.get("related_manga_raw", []),
    }


class AnimePlanetAnimeCrawler(BaseCrawler[AnimePlanetAnime, dict[str, Any]]):
    """Crawler for Anime-Planet anime detail pages."""

    def normalize_identifier(self, identifier: str) -> str:
        return _normalize_anime_url(identifier)

    async def fetch_raw_data(self, url: str) -> dict[str, Any] | None:
        slug = _extract_slug_from_url(url)
        return await _fetch_animeplanet_anime_data(slug)

    def build_source_model(
        self, processed_raw: dict[str, Any], url: str
    ) -> AnimePlanetAnime:
        return _build_anime_from_raw(processed_raw)

    def map_to_canonical(self, source_model: AnimePlanetAnime) -> dict[str, Any]:
        return anime_from_animeplanet(source_model)


async def fetch_animeplanet_anime(url: str) -> dict[str, Any] | None:
    """Fetch and return canonical anime dict for an Anime-Planet anime URL.

    Args:
        url: Full Anime-Planet anime URL, slug, or path
            (e.g. "https://www.anime-planet.com/anime/dandadan" or "dandadan").

    Returns:
        Canonical anime dict, or None if the fetch or validation fails.
    """
    return await AnimePlanetAnimeCrawler(DockerTransport(), NullRepository()).crawl(url)
