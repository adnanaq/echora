"""Anime-Planet anime crawler — zendriver + lxml XPath.

Extracts comprehensive anime data including related anime, rankings, studios,
and all metadata from JSON-LD.  Results are cached in Redis for 24 hours.

    fetch_animeplanet_anime(url)  — fetch a single anime → canonical dict

CLI usage::

    uv run python -m enrichment.sources.anime_planet.anime_planet_anime_crawler \\
        https://www.anime-planet.com/anime/dandadan

    uv run python -m enrichment.sources.anime_planet.anime_planet_anime_crawler \\
        https://www.anime-planet.com/anime/dandadan --output dandadan.json
"""

import json
import logging
import re
from typing import Any, cast
from urllib.parse import urlsplit

from enrichment.sources.anime_planet.anime_planet_models import (
    AnimePlanetAnime,
    AnimePlanetMangaEntry,
    AnimePlanetRelatedEntry,
)
from enrichment.sources.anime_planet.animeplanet_mapper import anime_from_animeplanet
from enrichment.sources.base.browser import browser_session
from enrichment.sources.base.exceptions import ServiceNetworkError, ServiceParseError
from enrichment.sources.base.framework import (
    BaseCrawler,
    FileRepository,
    NullRepository,
)
from enrichment.sources.base.page_readiness import wait_for_page
from http_cache.result_cache import cached_result
from lxml import etree

logger = logging.getLogger(__name__)


BASE_ANIME_URL = "https://www.anime-planet.com/anime/"
ANIME_PLANET_SITE = "Anime-Planet"
# Where Anime-Planet redirects an anime address that names no anime, and an
# element only that search page has, so the wait for the page ends there too.
_SEARCH_PAGE_PATH = "/search.php"
_SEARCH_FILTER_SELECTOR = "div.quickFilter"

_SEASON_SLUG_RE = re.compile(r"/seasons/([^/?#]+)")
_RANK_RE = re.compile(r"#(\d+)")
_AKA_PREFIX = "alt title:"

# XPaths for entryBar metadata and relations
_XPATHS: dict[str, str] = {
    "type_raw": "//section[contains(@class,'entryBar')]//span[@class='type']",
    "season_url": "//section[contains(@class,'entryBar')]//a[contains(@href,'/anime/seasons/')]/@href",
    "year_text": "//section[contains(@class,'entryBar')]//span[contains(@class,'iconYear')]",
    "rank_text": "//section[contains(@class,'entryBar')]//div[contains(.,'Rank #')]",
    "studios": "//section[contains(@class,'entryBar')]//a[contains(@href,'/studios/')]",
    "aka": "//h2[contains(@class,'aka')]",
    "tags": "//div[contains(@class,'tags')]//a[contains(@href,'/anime/tags/')]",
    "cover": "//img[@itemprop='image']/@src",
    "related_anime": "//div[@id='tabs--relations--anime--same_franchise']//a[contains(@class,'RelatedEntry')]",
    "related_anime_other": "//div[@id='tabs--relations--anime--other_franchise']//a[contains(@class,'RelatedEntry')]",
    "related_manga": "//div[contains(@id,'tabs--relations--manga')]//a[contains(@class,'RelatedEntry')]",
}

# Sub-element XPaths applied to each RelatedEntry anchor element
_REL_TITLE_XPATH = ".//p[contains(@class,'RelatedEntry__name')]"
_REL_SUBTYPE_XPATH = ".//span[contains(@class,'RelatedEntry__subtitle')]"
_REL_TYPE_XPATH = ".//li[.//i[contains(@class,'fa-tv')]]//span[contains(@class,'RelatedEntry__metadata_item')]"
_REL_IMAGE_XPATH = ".//img[contains(@class,'RelatedEntry__image')]/@src"
_REL_VOLCH_XPATH = ".//li[.//i[contains(@class,'fa-book-open')]]//span[contains(@class,'RelatedEntry__metadata_item')]"


def _tc(el: Any) -> str:
    """Return all text content of an lxml element, stripped."""
    return "".join(el.itertext()).strip()


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
    its content as a dict.  HTML entities in ``description`` are unescaped and
    known malformed image URLs are corrected.

    Args:
        html: Full HTML of an Anime-Planet anime page.

    Returns:
        Parsed JSON-LD dict, or None if not found or malformed.
    """
    try:
        import html as html_lib

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


def _parse_related_entry_element(el: Any, *, is_manga: bool) -> dict[str, Any]:
    """Convert a single RelatedEntry anchor lxml element into a raw dict.

    Produces the same dict shape expected by ``_build_related_anime_entries``
    and ``_build_related_manga_entries``.

    Args:
        el: lxml element for an ``<a class="RelatedEntry ...">`` anchor.
        is_manga: True when parsing manga entries (extracts vol_ch instead of type).

    Returns:
        Dict with ``url``, ``title``, ``relation_subtype``, ``type``, ``image``,
        and (when ``is_manga``) ``vol_ch``.
    """

    def _et(xpath: str) -> str | None:
        els = cast(list[Any], el.xpath(xpath))
        return _tc(els[0]) if els else None

    def _attr(xpath: str) -> str | None:
        vals = cast(list[Any], el.xpath(xpath))
        return vals[0] if vals else None

    entry: dict[str, Any] = {
        "url": el.get("href", ""),
        "title": _et(_REL_TITLE_XPATH) or "",
        "relation_subtype": _et(_REL_SUBTYPE_XPATH),
        "image": _attr(_REL_IMAGE_XPATH),
    }
    if is_manga:
        entry["vol_ch"] = _et(_REL_VOLCH_XPATH)
    else:
        entry["type"] = _et(_REL_TYPE_XPATH)
    return entry


def _build_related_anime_entries(
    raw: list[dict[str, Any]],
) -> list[AnimePlanetRelatedEntry]:
    """Build AnimePlanetRelatedEntry models from raw relation dicts.

    Entries without an anime address are skipped; the model reads type and
    episode count from the fa-tv span.

    Args:
        raw: List of dicts produced by ``_parse_related_entry_element``.

    Returns:
        List of validated ``AnimePlanetRelatedEntry`` models.
    """
    entries = []
    for item in raw:
        url_val = (item.get("url") or "").strip()
        if not url_val:
            continue
        slug_match = re.search(r"/anime/([^/?#]+)", url_val)
        if not slug_match:
            continue

        entries.append(
            AnimePlanetRelatedEntry.model_validate(
                {**item, "url": url_val, "slug": slug_match.group(1)}
            )
        )
    return entries


def _build_related_manga_entries(
    raw: list[dict[str, Any]],
) -> list[AnimePlanetMangaEntry]:
    """Build AnimePlanetMangaEntry models from raw relation dicts.

    Entries without a manga address are skipped; the model reads type, volumes
    and chapters from the fa-book-open span.

    Args:
        raw: List of dicts produced by ``_parse_related_entry_element``.

    Returns:
        List of validated ``AnimePlanetMangaEntry`` models.
    """
    entries = []
    for item in raw:
        url_val = (item.get("url") or "").strip()
        if not url_val:
            continue
        slug_match = re.search(r"/manga/([^/?#]+)", url_val)
        if not slug_match:
            continue

        entries.append(
            AnimePlanetMangaEntry.model_validate(
                {**item, "url": url_val, "slug": slug_match.group(1)}
            )
        )
    return entries


def _extract_anime_from_html(html: str) -> dict[str, Any] | None:
    """Extract raw anime data from a rendered Anime-Planet anime page.

    Combines JSON-LD structured data (title, dates, episodes, ratings, genres)
    with lxml XPath extraction (type, season, rank, alt title, cover, studios,
    tags, related entries).  The slug field is injected by the caller.

    Args:
        html: Full rendered HTML of an Anime-Planet anime page.

    Returns:
        JSON-serialisable raw dict, or None if JSON-LD is absent or has no name.
    """
    if not html:
        return None

    json_ld = _extract_json_ld(html)
    if not json_ld or not json_ld.get("name"):
        logger.warning("No JSON-LD name found in page HTML")
        return None

    try:
        tree = etree.fromstring(html.encode(), etree.HTMLParser(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to parse anime page HTML")
        return None

    def _t(key: str) -> str | None:
        els = cast(list[Any], tree.xpath(_XPATHS[key]))
        return _tc(els[0]) if els else None

    def _a(key: str) -> str | None:
        vals = cast(list[Any], tree.xpath(_XPATHS[key]))
        return vals[0] if vals else None

    def _texts(key: str) -> list[str]:
        return [_tc(el) for el in cast(list[Any], tree.xpath(_XPATHS[key])) if _tc(el)]

    def _studios() -> list[dict[str, str | None]]:
        """Keep each studio's own page alongside its name.

        The XPath already selects the anchor, so the href costs nothing extra
        and is the only cross-provider identifier anime-planet offers.
        """
        found = []
        for el in cast(list[Any], tree.xpath(_XPATHS["studios"])):
            name = _tc(el)
            if not name:
                continue
            href = el.get("href") or ""
            found.append(
                {
                    "name": name,
                    "url": f"https://www.anime-planet.com{href}" if href else None,
                }
            )
        return found

    def _related(key: str, *, is_manga: bool = False) -> list[dict[str, Any]]:
        return [
            _parse_related_entry_element(el, is_manga=is_manga)
            for el in cast(list[Any], tree.xpath(_XPATHS[key]))
        ]

    return {
        "name": json_ld["name"],
        "description": json_ld.get("description"),
        "url": json_ld.get("url"),
        "start_date": json_ld.get("startDate"),
        "end_date": json_ld.get("endDate"),
        "number_of_episodes": json_ld.get("numberOfEpisodes"),
        "genres": json_ld.get("genre") or [],
        "aggregate_rating": json_ld.get("aggregateRating"),
        "type_raw": _t("type_raw"),
        "season_url": _a("season_url"),
        "year_text": _t("year_text"),
        "rank_text": _t("rank_text"),
        "aka": _t("aka"),
        "cover": _a("cover"),
        "studios": _studios(),
        "tags": _texts("tags"),
        "related_anime_raw": _related("related_anime"),
        "related_anime_other_raw": _related("related_anime_other"),
        "related_manga_raw": _related("related_manga", is_manga=True),
    }


def _build_anime_from_raw(raw: dict[str, Any]) -> AnimePlanetAnime:
    """Construct an AnimePlanetAnime model from a cached raw data dict.

    Post-processing step that converts the JSON-serialisable cached dict into
    typed Pydantic models.  Model construction is intentionally kept separate
    from caching so Pydantic objects are never stored in Redis.

    Args:
        raw: Dict returned by ``_fetch_animeplanet_anime_data`` — contains merged
             JSON-LD scalars, XPath primitives, and raw relation lists.

    Returns:
        Validated AnimePlanetAnime source model.
    """
    return AnimePlanetAnime(
        name=raw["name"],
        description=raw.get("description"),
        url=raw.get("url"),
        start_date=raw.get("start_date"),
        end_date=raw.get("end_date"),
        number_of_episodes=raw.get("number_of_episodes"),
        genres=raw.get("genres", []),
        aggregate_rating=raw.get("aggregate_rating"),
        type=raw.get("type_raw"),
        season=raw.get("season_url"),
        start_year=raw.get("year_text"),
        rank=raw.get("rank_text"),
        alt_title=raw.get("aka"),
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


async def _fetch_anime_html(url: str) -> str | None:
    """Navigate to an Anime-Planet anime page and return its rendered HTML.

    Anime-Planet answers an address with no anime by redirecting to its search
    page. The wait ends on either the anime's entry bar or the search page's
    quick filter, which anime pages do not have, and the address the tab ends
    on tells a missing page apart.

    Args:
        url: Full Anime-Planet anime URL
            (e.g. ``https://www.anime-planet.com/anime/dandadan``).

    Returns:
        Rendered page HTML, or None when the page does not exist.

    Raises:
        ServiceNetworkError: If the page does not load.
    """
    async with browser_session(headless=True) as session:
        try:
            page = await session.browser.get(url)
            await wait_for_page(
                page,
                f"section.entryBar, {_SEARCH_FILTER_SELECTOR}",
                url,
                cloudflare_site=ANIME_PLANET_SITE,
            )
            if urlsplit(page.url or "").path == _SEARCH_PAGE_PATH:
                return None
            return await page.get_content()
        except Exception as exc:
            raise ServiceNetworkError(
                service="anime_planet", cause=f"navigation failed for {url}: {exc}"
            ) from exc


@cached_result(
    service="anime_planet",
    key_prefix="animeplanet_anime",
)
async def _fetch_animeplanet_anime_data(
    canonical_slug: str,
) -> dict[str, Any] | None:
    """Fetch and extract raw anime data for a given Anime-Planet slug.

    Uses zendriver (CDP) + lxml XPath.  Cached by canonical slug; the cache
    is automatically invalidated when the extraction logic changes.

    Returns a JSON-serialisable dict of primitives ready for
    ``_build_anime_from_raw``.  Model construction is left to that function so
    Pydantic models are never stored in the cache.

    Args:
        canonical_slug: Canonical anime slug (e.g. ``"one-piece"``).

    Returns:
        Raw data dict, or None when the page does not exist.

    Raises:
        ServiceNetworkError: If the page does not load.
        ServiceParseError: If the page loads but its data cannot be read.
    """
    url = f"{BASE_ANIME_URL}{canonical_slug}"
    logger.info(f"Fetching anime data: {url}")

    html = await _fetch_anime_html(url)
    if html is None:
        logger.warning(f"Anime-Planet anime page not found: {url}")
        return None

    raw = _extract_anime_from_html(html)
    if not raw:
        raise ServiceParseError(service="anime_planet", cause=f"no anime data in {url}")

    raw["slug"] = canonical_slug
    return raw


class AnimePlanetAnimeCrawler(BaseCrawler[AnimePlanetAnime, dict[str, Any]]):
    """Crawler for Anime-Planet anime detail pages."""

    def get_extraction_schema(self) -> dict[str, str]:
        return _XPATHS

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


async def fetch_animeplanet_anime(
    url: str, output_path: str | None = None
) -> dict[str, Any] | None:
    """Fetch and return canonical anime dict for an Anime-Planet anime URL.

    Args:
        url: Full Anime-Planet anime URL, slug, or path
            (e.g. ``"https://www.anime-planet.com/anime/dandadan"`` or ``"dandadan"``).
        output_path: If provided, write the canonical dict to this JSON file.

    Returns:
        Canonical anime dict, or None if the fetch or validation fails.
    """
    repo = FileRepository(output_path) if output_path else NullRepository()
    return await AnimePlanetAnimeCrawler(repo).crawl(url)


if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Fetch Anime-Planet anime data")
    parser.add_argument("url", help="Anime-Planet anime URL or slug")
    parser.add_argument("--output", help="Write canonical JSON to this file")
    args = parser.parse_args()

    result = asyncio.run(fetch_animeplanet_anime(args.url, args.output))
    if result:
        print(json.dumps(result, indent=2, default=str))
    else:
        print("Failed to fetch anime data")
