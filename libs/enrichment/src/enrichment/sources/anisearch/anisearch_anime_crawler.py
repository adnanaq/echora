"""Crawls anime information from anisearch.com via the crawl4ai Docker REST API.

Extracts metadata and relations using XPath extraction.
Results are cached in Redis. Two sequential page fetches per anime
(main + /relations?show=overall) — sequential rather than concurrent
to avoid triggering Cloudflare bot detection.
"""

import html
import json
import logging
import re
from typing import Any

from enrichment.sources.anisearch.anisearch_anime_models import (
    AniSearchAnime,
    AniSearchRelatedEntry,
    AniSearchStatistics,
)
from enrichment.sources.anisearch.anisearch_mapper import anime_from_anisearch
from enrichment.sources.base.crawl4ai_docker import crawl_single_url
from enrichment.sources.base.crawler_config import (
    get_docker_browser_config,
    get_docker_crawler_config,
)
from enrichment.sources.base.framework import (
    BaseCrawler,
    DockerTransport,
    FileRepository,
    NullRepository,
)
from enrichment.sources.base.utils import parse_broadcast_string, parse_iso_date
from http_cache.config import get_cache_config
from http_cache.result_cache import cached_result

logger = logging.getLogger(__name__)

_CACHE_CONFIG = get_cache_config()
TTL_ANISEARCH = _CACHE_CONFIG.ttl_anisearch

BASE_ANIME_URL = "https://www.anisearch.com/anime/"

_LABEL_RE = re.compile(r"^\s*[^:]+:\s*")
_DATE_RANGE_RE = re.compile(r"(\d{2}\.\d{2}\.\d{4})\s*[-–‑]\s*(\d{2}\.\d{2}\.\d{4})")
_SINGLE_DATE_RE = re.compile(r"(\d{2}\.\d{2}\.\d{4})")
_SCORE_RE = re.compile(r"(\d+\.\d+)")
_RANK_RE = re.compile(r"#(\d+)")
_IMG_SRC_RE = re.compile(r'<img src="([^"]+)"')


def _extract_path_from_url(url: str) -> str:
    """Extract the anime path from a canonical AniSearch anime URL.

    Raises:
        ValueError: If the URL doesn't start with BASE_ANIME_URL or has no path.
    """
    if not url.startswith(BASE_ANIME_URL):
        raise ValueError(f"URL must start with {BASE_ANIME_URL!r}: {url!r}")
    path = url[len(BASE_ANIME_URL) :].strip("/")
    if not path:
        raise ValueError(f"URL does not contain anime path: {url!r}")
    return path


def _process_relation_tooltips(relations: list[dict[str, Any]]) -> None:
    """Extract image URL from HTML-escaped data-tooltip attribute (mutates in-place)."""
    for rel in relations:
        image = rel.get("image")
        if image:
            m = _IMG_SRC_RE.search(html.unescape(image))
            if m:
                rel["image"] = m.group(1)


def _get_main_schema() -> dict[str, Any]:
    """XPath extraction schema for the main AniSearch anime page."""
    return {
        "name": "AniSearchAnime",
        "baseSelector": "//body",
        "fields": [
            {
                "name": "cover_image",
                "selector": "//section[@id='information']//img[@id='details-cover']",
                "type": "attribute",
                "attribute": "src",
            },
            {
                "name": "title_alt",
                "selector": "//section[@id='information']//div[contains(@class,'title') and @lang='ja']//div[contains(@class,'grey')]",
                "type": "text",
            },
            {
                "name": "title_ja",
                "selector": "//section[@id='information']//div[contains(@class,'title') and @lang='ja']//strong[contains(@class,'f16')]",
                "type": "text",
            },
            {
                "name": "type",
                "selector": "//section[@id='information']//div[contains(@class,'type')]",
                "type": "text",
            },
            {
                "name": "status",
                "selector": "//section[@id='information']//div[contains(@class,'status')]",
                "type": "text",
            },
            {
                "name": "published",
                "selector": "//section[@id='information']//div[contains(@class,'released')]",
                "type": "text",
            },
            {
                "name": "studio",
                "selector": "//section[@id='information']//div[contains(@class,'company')]//a[contains(@href,'company')]",
                "type": "text",
            },
            {
                "name": "studio_url",
                "selector": "//section[@id='information']//div[contains(@class,'company')]//a[contains(@href,'company')]",
                "type": "attribute",
                "attribute": "href",
            },
            {
                "name": "broadcast_raw",
                "selector": "//section[@id='information']//div[contains(@class,'broadcast')]",
                "type": "text",
            },
            {
                "name": "source_material",
                "selector": "//section[@id='information']//div[contains(@class,'adapted')]",
                "type": "text",
            },
            {
                "name": "synonyms",
                "selector": "//section[@id='information']//div[contains(@class,'synonyms')]",
                "type": "text",
            },
            {
                "name": "description",
                "selector": "//section[@id='description']//div[contains(@class,'textblock') and contains(@class,'details-text')]",
                "type": "text",
            },
            {
                "name": "genres",
                "selector": "//section[@id='genres-tags']//ul[contains(@class,'cloud')]//a[contains(@href,'/genre/main/') or contains(@href,'/genre/subsidiary/')]",
                "type": "list",
                "fields": [{"name": "name", "selector": ".", "type": "text"}],
            },
            {
                "name": "tags",
                "selector": "//section[@id='genres-tags']//ul[contains(@class,'cloud')]//a[contains(@href,'/genre/tag/')]",
                "type": "list",
                "fields": [{"name": "name", "selector": ".", "type": "text"}],
            },
            {
                "name": "rating_score",
                "selector": "//*[@id='ratingstats']//tr[2]//td[1]//b",
                "type": "text",
            },
            {
                "name": "rank_toplist",
                "selector": "//*[@id='ratingstats']//tr[2]//td[2]//b",
                "type": "text",
            },
            {
                "name": "rank_trending",
                "selector": "//*[@id='ratingstats']//tr[3]//td[2]//b",
                "type": "text",
            },
            {
                "name": "websites",
                "selector": "//section[@id='information']//div[contains(@class,'websites')]//a",
                "type": "nested_list",
                "fields": [
                    {"name": "name", "selector": ".", "type": "text"},
                    {
                        "name": "url",
                        "selector": ".",
                        "type": "attribute",
                        "attribute": "href",
                    },
                ],
            },
        ],
    }


_RELATION_FIELDS: list[dict[str, Any]] = [
    {"name": "relation_type", "selector": ".//th//span", "type": "text"},
    {"name": "title", "selector": ".//th//a", "type": "text"},
    {"name": "url", "selector": ".//th//a", "type": "attribute", "attribute": "href"},
    {
        "name": "details",
        "selector": ".//td[@data-title='Type / Episodes / Year']",
        "type": "text",
    },
    {
        "name": "rating",
        "selector": ".//td[contains(@class,'rating')]//div[contains(@class,'star0')]",
        "type": "attribute",
        "attribute": "title",
    },
    {
        "name": "image",
        "selector": ".//th[@scope='row']",
        "type": "attribute",
        "attribute": "data-tooltip",
    },
]


def _get_relations_schema() -> dict[str, Any]:
    """XPath extraction schema for the /relations?show=overall sub-page."""
    return {
        "name": "AniSearchRelations",
        "baseSelector": "//body",
        "fields": [
            {
                "name": "anime_relations",
                "selector": "//section[@id='relations_anime']//tbody//tr",
                "type": "nested_list",
                "fields": _RELATION_FIELDS,
            },
            {
                "name": "manga_relations",
                "selector": "//section[@id='relations_manga']//tbody//tr",
                "type": "nested_list",
                "fields": _RELATION_FIELDS,
            },
        ],
    }


def _unwrap_result(result: dict[str, Any] | None, url: str) -> dict[str, Any] | None:
    """Extract the first XPath result dict from a crawl_single_url response."""
    if result is None:
        return None
    status = result.get("status_code")
    if status == 404:
        logger.warning(f"404 for {url}")
        return None
    if status and status not in (200, 302):
        logger.error(f"HTTP {status} for {url}")
        return None
    if not result.get("success"):
        logger.warning(f"Crawl failed for {url}: {result.get('error_message')}")
        return None
    raw_list = json.loads(result.get("extracted_content") or "[]")
    return raw_list[0] if raw_list else None


def _post_process_main(raw: dict[str, Any]) -> dict[str, Any]:
    """Clean raw XPath extraction dict into model-ready field values."""
    data: dict[str, Any] = {}

    data["cover_image"] = raw.get("cover_image") or None
    data["title_alt"] = (raw.get("title_alt") or "").strip() or None
    data["title_ja"] = (raw.get("title_ja") or "").strip() or None

    # Type: strip label prefix, take text before first comma
    # Raw: "Type: TV-Series, 1200 (~24 min, Total: ~480 h)"
    type_raw = _LABEL_RE.sub("", raw.get("type") or "").strip()
    data["type"] = type_raw.split(",")[0].strip() or None

    # Status: strip label prefix
    data["status"] = _LABEL_RE.sub("", raw.get("status") or "").strip() or None

    # Published: parse start_date / end_date from DD.MM.YYYY format.
    # Falls back to year-only (e.g. "2026 ‑ ?") via parse_iso_date for upcoming anime.
    published = _LABEL_RE.sub("", raw.get("published") or "").strip()
    m_range = _DATE_RANGE_RE.search(published)
    if m_range:
        data["start_date"] = m_range.group(1)
        data["end_date"] = m_range.group(2)
    else:
        m_single = _SINGLE_DATE_RE.search(published)
        if m_single:
            data["start_date"] = m_single.group(1)
        else:
            # e.g. "2026 ‑ ?" — extract bare year and normalise to ISO
            year_match = re.search(r"\b(\d{4})\b", published)
            data["start_date"] = (
                parse_iso_date(year_match.group(1)) if year_match else None
            )
        data["end_date"] = None

    data["studio"] = (raw.get("studio") or "").strip() or None
    broadcast_raw = _LABEL_RE.sub("", raw.get("broadcast_raw") or "").strip()
    day, time, tz = parse_broadcast_string(broadcast_raw)
    data["broadcast_day"] = day
    data["broadcast_time"] = time
    data["broadcast_timezone"] = tz

    studio_url = (raw.get("studio_url") or "").strip()
    data["studio_url"] = (
        f"https://www.anisearch.com{studio_url}"
        if studio_url.startswith("/")
        else f"https://www.anisearch.com/{studio_url}"
        if studio_url
        else None
    )

    # Source material: strip label prefix
    data["source_material"] = (
        _LABEL_RE.sub("", raw.get("source_material") or "").strip() or None
    )

    # Synonyms: strip label prefix, split on comma
    syn_clean = _LABEL_RE.sub("", raw.get("synonyms") or "").strip()
    data["synonyms"] = [s.strip() for s in syn_clean.split(",") if s.strip()]

    data["description"] = (raw.get("description") or "").strip() or None

    # Genres / tags: flatten list[{name: str}] → list[str]
    data["genres"] = [
        item["name"] for item in raw.get("genres", []) if item.get("name")
    ]
    data["tags"] = [item["name"] for item in raw.get("tags", []) if item.get("name")]

    data["websites"] = [
        {"name": w.get("name", ""), "url": w.get("url", "")}
        for w in raw.get("websites", [])
        if w.get("url")
    ]

    # Statistics
    score: float | None = None
    m_score = _SCORE_RE.search(raw.get("rating_score") or "")
    if m_score:
        score = float(m_score.group(1))

    def _parse_rank(text: str | None) -> int | None:
        if not text:
            return None
        m = _RANK_RE.search(text.replace(".", ""))
        return int(m.group(1)) if m else None

    stats: dict[str, Any] = {}
    if score is not None:
        stats["score"] = score
    rank = _parse_rank(raw.get("rank_toplist"))
    if rank is not None:
        stats["rank"] = rank
    trending = _parse_rank(raw.get("rank_trending"))
    if trending is not None:
        stats["trending"] = trending
    data["statistics"] = stats or None

    return data


def _parse_relations(
    raw: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not raw:
        return [], []
    anime = list(raw.get("anime_relations") or [])
    manga = list(raw.get("manga_relations") or [])
    _process_relation_tooltips(anime)
    _process_relation_tooltips(manga)
    return anime, manga


@cached_result(
    ttl=TTL_ANISEARCH,
    key_prefix="anisearch_anime",
    dependencies=[_get_main_schema, _get_relations_schema, _post_process_main],
)
async def _fetch_anisearch_anime_data(canonical_path: str) -> dict[str, Any] | None:
    """Fetch and extract raw anime data for a given AniSearch anime path.

    Two sequential page fetches (main, relations) to avoid
    triggering Cloudflare bot detection. Cached by canonical path; cache is
    automatically invalidated when any extraction schema changes.

    Returns a JSON-serializable dict of primitives ready for _build_anime_from_raw.
    Model construction is left to _build_anime_from_raw so Pydantic models are
    never stored in the cache.
    """
    base_url = f"{BASE_ANIME_URL}{canonical_path}"
    browser = get_docker_browser_config()

    logger.info(f"Fetching AniSearch main page: {base_url}")
    main_result = await crawl_single_url(
        base_url,
        browser_config=browser,
        crawler_config=get_docker_crawler_config(_get_main_schema()),
    )
    main_raw = _unwrap_result(main_result, base_url)
    if main_raw is None:
        logger.warning(f"No data extracted from AniSearch main page: {base_url}")
        return None

    # When the server redirected (slug-less URL), switch base_url to the canonical
    # slug form so the relations fetch and all downstream callers use it.
    redirected = (main_result or {}).get("redirected_url")
    if redirected:
        base_url = redirected

    rels_raw = _unwrap_result(
        await crawl_single_url(
            f"{base_url}/relations?show=overall",
            browser_config=browser,
            crawler_config=get_docker_crawler_config(_get_relations_schema()),
        ),
        f"{base_url}/relations?show=overall",
    )

    data = _post_process_main(main_raw)
    data["anime_relations"], data["manga_relations"] = _parse_relations(rels_raw)
    if redirected:
        data["_canonical_url"] = redirected
    return data


def _build_anime_from_raw(raw: dict[str, Any], url: str) -> AniSearchAnime:
    """Construct AniSearchAnime source model from a cached raw data dict."""
    stats_data = raw.get("statistics")
    statistics = AniSearchStatistics(**stats_data) if stats_data else None

    anime_relations = [
        AniSearchRelatedEntry(
            relation_type=r.get("relation_type"),
            title=r.get("title"),
            url=r.get("url"),
            details=r.get("details"),
            rating=r.get("rating"),
            image=r.get("image"),
        )
        for r in raw.get("anime_relations", [])
    ]
    manga_relations = [
        AniSearchRelatedEntry(
            relation_type=r.get("relation_type"),
            title=r.get("title"),
            url=r.get("url"),
            details=r.get("details"),
            rating=r.get("rating"),
            image=r.get("image"),
        )
        for r in raw.get("manga_relations", [])
    ]

    return AniSearchAnime(
        title=raw.get("title_ja"),
        title_japanese=raw.get("title_alt"),
        synonyms=raw.get("synonyms", []),
        type=raw.get("type"),
        source_material=raw.get("source_material"),
        start_date=raw.get("start_date"),
        end_date=raw.get("end_date"),
        synopsis=raw.get("description"),
        genres=raw.get("genres", []),
        tags=raw.get("tags", []),
        broadcast_day=raw.get("broadcast_day"),
        broadcast_time=raw.get("broadcast_time"),
        broadcast_timezone=raw.get("broadcast_timezone"),
        studio=raw.get("studio"),
        studio_url=raw.get("studio_url"),
        websites=raw.get("websites", []),
        statistics=statistics,
        cover_image=raw.get("cover_image"),
        anime_relations=anime_relations,
        manga_relations=manga_relations,
        url=url,
    )


class AniSearchAnimeCrawler(BaseCrawler[AniSearchAnime, dict[str, Any]]):
    """Crawler for AniSearch anime detail pages via the crawl4ai Docker REST API."""

    def normalize_identifier(self, identifier: str) -> str:
        if not identifier.startswith(BASE_ANIME_URL):
            raise ValueError(f"Not an AniSearch anime URL: {identifier!r}")
        return identifier

    async def fetch_raw_data(self, url: str) -> dict[str, Any] | None:
        return await _fetch_anisearch_anime_data(_extract_path_from_url(url))

    def build_source_model(
        self, processed_raw: dict[str, Any], url: str
    ) -> AniSearchAnime:
        return _build_anime_from_raw(
            processed_raw, processed_raw.get("_canonical_url", url)
        )

    def map_to_canonical(self, source_model: AniSearchAnime) -> dict[str, Any]:
        return anime_from_anisearch(source_model)


async def fetch_anisearch_anime(
    url: str, output_path: str | None = None
) -> dict[str, Any] | None:
    """Fetch canonical anime dict for an AniSearch anime URL.

    Args:
        url: Full AniSearch anime URL (e.g. "https://www.anisearch.com/anime/18878,dan-da-dan").
        output_path: Optional path to write JSON result.

    Returns:
        Canonical anime dict, or None if fetch or mapping fails.
    """
    repo = FileRepository(output_path) if output_path else NullRepository()
    return await AniSearchAnimeCrawler(DockerTransport(), repo).crawl(url)
