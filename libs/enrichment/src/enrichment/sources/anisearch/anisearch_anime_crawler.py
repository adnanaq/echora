"""Crawls anime information from anisearch.com via zendriver + lxml XPath.

Extracts metadata and relations using lxml XPath on raw page HTML.
Results are cached in Redis. Two sequential page fetches per anime
(main + /relations?show=overall) — sequential to avoid Cloudflare bot detection.
"""

import asyncio
import html
import logging
import re
from typing import Any, cast

from enrichment.sources.anisearch.anisearch_anime_models import (
    AniSearchAnime,
    AniSearchRelatedEntry,
    AniSearchStatistics,
)
from enrichment.sources.anisearch.anisearch_mapper import anime_from_anisearch
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
_ANISEARCH_BASE_URL = "https://www.anisearch.com"
_INTER_REQUEST_DELAY = 3.0

_LABEL_RE = re.compile(r"^\s*[^:]+:\s*")
_DATE_RANGE_RE = re.compile(r"(\d{2}\.\d{2}\.\d{4})\s*[-–‑]\s*(\d{2}\.\d{2}\.\d{4})")
_SINGLE_DATE_RE = re.compile(r"(\d{2}\.\d{2}\.\d{4})")
_SCORE_RE = re.compile(r"(\d+\.\d+)")
_RANK_RE = re.compile(r"#(\d+)")
_IMG_SRC_RE = re.compile(r'<img src="([^"]+)"')

# ---------------------------------------------------------------------------
# XPath selectors
# ---------------------------------------------------------------------------

_XPATHS: dict[str, str] = {
    # Main page
    "cover_image":     "//section[@id='information']//img[@id='details-cover']/@src",
    "title_alt":       "//section[@id='information']//div[contains(@class,'title') and @lang='ja']//div[contains(@class,'grey')]",
    "title_ja":        "//section[@id='information']//div[contains(@class,'title') and @lang='ja']//strong[contains(@class,'f16')]",
    "type":            "//section[@id='information']//div[contains(@class,'type')]",
    "status":          "//section[@id='information']//div[contains(@class,'status')]",
    "published":       "//section[@id='information']//div[contains(@class,'released')]",
    "studio":          "//section[@id='information']//div[contains(@class,'company')]//a[contains(@href,'company')]",
    "studio_url":      "//section[@id='information']//div[contains(@class,'company')]//a[contains(@href,'company')]/@href",
    "broadcast_raw":   "//section[@id='information']//div[contains(@class,'broadcast')]",
    "source_material": "//section[@id='information']//div[contains(@class,'adapted')]",
    "synonyms":        "//section[@id='information']//div[contains(@class,'synonyms')]",
    "description":     "//section[@id='description']//div[contains(@class,'textblock') and contains(@class,'details-text')]",
    "genres":          "//section[@id='genres-tags']//ul[contains(@class,'cloud')]//a[contains(@href,'/genre/main/') or contains(@href,'/genre/subsidiary/')]",
    "tags":            "//section[@id='genres-tags']//ul[contains(@class,'cloud')]//a[contains(@href,'/genre/tag/')]",
    "rating_score":    "//*[@id='ratingstats']//tr[2]//td[1]//b",
    "rank_toplist":    "//*[@id='ratingstats']//tr[2]//td[2]//b",
    "rank_trending":   "//*[@id='ratingstats']//tr[3]//td[2]//b",
    "websites":        "//section[@id='information']//div[contains(@class,'websites')]//a",
    # Relations sub-page
    "anime_relation_rows": "//section[@id='relations_anime']//tbody//tr",
    "manga_relation_rows": "//section[@id='relations_manga']//tbody//tr",
}

# ---------------------------------------------------------------------------
# lxml extraction helpers
# ---------------------------------------------------------------------------


def _extract_anime_from_html(html_text: str) -> dict[str, Any] | None:
    """Parse an AniSearch anime main page into the raw field dict.

    Returns a dict with the same keys that _post_process_main expects.
    Returns None if the HTML cannot be parsed.
    """
    from lxml import etree

    try:
        parser = etree.HTMLParser()
        tree = etree.fromstring(html_text.encode(), parser)
        if tree is None:  # pragma: no cover
            return None  # pragma: no cover
    except Exception:  # pragma: no cover
        return None  # pragma: no cover

    def _text(key: str) -> str | None:
        els = cast(list[Any], tree.xpath(_XPATHS[key]))
        if not els:
            return None
        raw = "".join(els[0].itertext())
        return re.sub(r" {2,}", " ", raw).strip() or None

    def _attr(key: str) -> str | None:
        vals = cast(list[str], tree.xpath(_XPATHS[key]))
        return vals[0].strip() if vals else None

    genre_els = cast(list[Any], tree.xpath(_XPATHS["genres"]))
    genres = [
        {"name": "".join(el.itertext()).strip()}
        for el in genre_els
        if "".join(el.itertext()).strip()
    ]

    tag_els = cast(list[Any], tree.xpath(_XPATHS["tags"]))
    tags = [
        {"name": "".join(el.itertext()).strip()}
        for el in tag_els
        if "".join(el.itertext()).strip()
    ]

    website_els = cast(list[Any], tree.xpath(_XPATHS["websites"]))
    websites = [
        {
            "name": "".join(el.itertext()).strip(),
            "url": el.get("href") or "",
        }
        for el in website_els
        if el.get("href")
    ]

    return {
        "cover_image":     _attr("cover_image"),
        "title_alt":       _text("title_alt"),
        "title_ja":        _text("title_ja"),
        "type":            _text("type"),
        "status":          _text("status"),
        "published":       _text("published"),
        "studio":          _text("studio"),
        "studio_url":      _attr("studio_url"),
        "broadcast_raw":   _text("broadcast_raw"),
        "source_material": _text("source_material"),
        "synonyms":        _text("synonyms"),
        "description":     _text("description"),
        "genres":          genres,
        "tags":            tags,
        "rating_score":    _text("rating_score"),
        "rank_toplist":    _text("rank_toplist"),
        "rank_trending":   _text("rank_trending"),
        "websites":        websites,
    }


def _extract_relations_from_html(html_text: str) -> dict[str, Any] | None:
    """Parse an AniSearch /relations?show=overall page into anime/manga relation lists.

    Returns a dict with keys 'anime_relations' and 'manga_relations', each a list
    of {relation_type, title, url, details, rating, image} dicts.
    Returns None if the HTML cannot be parsed.
    """
    from lxml import etree

    try:
        parser = etree.HTMLParser()
        tree = etree.fromstring(html_text.encode(), parser)
        if tree is None:  # pragma: no cover
            return None  # pragma: no cover
    except Exception:  # pragma: no cover
        return None  # pragma: no cover

    def _parse_rows(xpath_key: str) -> list[dict[str, Any]]:
        rows = cast(list[Any], tree.xpath(_XPATHS[xpath_key]))
        result = []
        for row in rows:
            span = row.xpath(".//th//span")
            relation_type = "".join(span[0].itertext()).strip() if span else None

            a = row.xpath(".//th//a")
            title = "".join(a[0].itertext()).strip() if a else None
            url = a[0].get("href") or None if a else None

            details_els = row.xpath(".//td[@data-title='Type / Episodes / Year']")
            details = "".join(details_els[0].itertext()).strip() if details_els else None

            rating_els = row.xpath(".//td[contains(@class,'rating')]//div[contains(@class,'star0')]")
            rating = rating_els[0].get("title") if rating_els else None

            image_attr = row.xpath(".//th[@scope='row']/@data-tooltip")
            image = image_attr[0] if image_attr else None

            result.append({
                "relation_type": relation_type,
                "title": title,
                "url": url,
                "details": details,
                "rating": rating,
                "image": image,
            })
        return result

    return {
        "anime_relations": _parse_rows("anime_relation_rows"),
        "manga_relations": _parse_rows("manga_relation_rows"),
    }


# ---------------------------------------------------------------------------
# URL and tooltip helpers
# ---------------------------------------------------------------------------


def _extract_path_from_url(url: str) -> str:
    """Extract the anime path from a canonical AniSearch anime URL.

    Raises:
        ValueError: If the URL doesn't start with BASE_ANIME_URL or has no path.
    """
    if not url.startswith(BASE_ANIME_URL):
        raise ValueError(f"URL must start with {BASE_ANIME_URL!r}: {url!r}")
    path = url[len(BASE_ANIME_URL):].strip("/")
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


# ---------------------------------------------------------------------------
# Browser navigation helper
# ---------------------------------------------------------------------------


async def _fetch_page_html(browser: Any, url: str, wait_selector: str | None = None) -> str | None:
    try:
        page = await browser.get(url)
        if wait_selector:
            await page.wait_for(selector=wait_selector, timeout=10)
        else:
            await asyncio.sleep(2)
        return await page.get_content()
    except Exception as exc:
        logger.warning(f"navigation failed for {url}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Post-processing helpers (unchanged — pure dict transforms)
# ---------------------------------------------------------------------------


def _post_process_main(raw: dict[str, Any]) -> dict[str, Any]:
    """Clean raw XPath extraction dict into model-ready field values."""
    data: dict[str, Any] = {}

    data["cover_image"] = raw.get("cover_image") or None
    data["title_alt"] = (raw.get("title_alt") or "").strip() or None
    data["title_ja"] = (raw.get("title_ja") or "").strip() or None

    type_raw = _LABEL_RE.sub("", raw.get("type") or "").strip()
    data["type"] = type_raw.split(",")[0].strip() or None

    data["status"] = _LABEL_RE.sub("", raw.get("status") or "").strip() or None

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

    data["source_material"] = (
        _LABEL_RE.sub("", raw.get("source_material") or "").strip() or None
    )

    syn_clean = _LABEL_RE.sub("", raw.get("synonyms") or "").strip()
    data["synonyms"] = [s.strip() for s in syn_clean.split(",") if s.strip()]

    data["description"] = (raw.get("description") or "").strip() or None

    data["genres"] = [
        item["name"] for item in raw.get("genres", []) if item.get("name")
    ]
    data["tags"] = [item["name"] for item in raw.get("tags", []) if item.get("name")]

    data["websites"] = [
        {"name": w.get("name", ""), "url": w.get("url", "")}
        for w in raw.get("websites", [])
        if w.get("url")
    ]

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


# ---------------------------------------------------------------------------
# Cached fetch
# ---------------------------------------------------------------------------


@cached_result(
    ttl=TTL_ANISEARCH,
    key_prefix="anisearch_anime",
    dependencies=[_extract_anime_from_html, _extract_relations_from_html],
)
async def _fetch_anisearch_anime_data(canonical_path: str) -> dict[str, Any] | None:
    """Fetch and extract raw anime data for a given AniSearch anime path.

    Two sequential page fetches (main, relations) with a single persistent
    browser session to avoid Cloudflare bot detection. Cached by canonical path;
    cache is automatically invalidated when any extraction function changes.

    Returns a JSON-serializable dict of primitives ready for _build_anime_from_raw.
    """
    import zendriver as zd

    base_url = f"{BASE_ANIME_URL}{canonical_path}"
    browser = await zd.start(headless=False)
    try:
        try:
            main_page = await browser.get(base_url)
            await main_page.wait_for(selector="#htitle", timeout=10)
            await asyncio.sleep(2)  # genres/stats sections render after htitle
            final_url = main_page.url  # capture post-redirect slug URL
            main_html = await main_page.get_content()
        except Exception as exc:
            logger.warning(f"navigation failed for {base_url}: {exc}")
            return None

        if not main_html:
            logger.warning(f"No HTML from AniSearch main page: {base_url}")
            return None

        main_raw = _extract_anime_from_html(main_html)
        if main_raw is None:
            logger.warning(f"Failed to extract data from AniSearch main page: {base_url}")
            return None

        await asyncio.sleep(_INTER_REQUEST_DELAY)

        canonical_base = final_url.rstrip("/") if final_url else base_url
        rels_url = f"{canonical_base}/relations?show=overall"
        rels_html = await _fetch_page_html(browser, rels_url, wait_selector="#relations_anime")
        rels_raw = _extract_relations_from_html(rels_html) if rels_html else None

    finally:
        try:
            await browser.stop()
        except Exception:
            pass

    data = _post_process_main(main_raw)
    data["anime_relations"], data["manga_relations"] = _parse_relations(rels_raw)
    if final_url and final_url != base_url:
        data["_canonical_url"] = final_url
    return data


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Crawler class
# ---------------------------------------------------------------------------


class AniSearchAnimeCrawler(BaseCrawler[AniSearchAnime, dict[str, Any]]):
    """Crawler for AniSearch anime detail pages via zendriver + lxml XPath."""

    def get_extraction_schema(self) -> dict[str, Any]:
        return {"xpaths": _XPATHS}

    def normalize_identifier(self, identifier: str) -> str:
        normalized = identifier.replace(
            "https://anisearch.com/", "https://www.anisearch.com/", 1
        )
        if not normalized.startswith(BASE_ANIME_URL):
            raise ValueError(f"Not an AniSearch anime URL: {identifier!r}")
        return normalized

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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
