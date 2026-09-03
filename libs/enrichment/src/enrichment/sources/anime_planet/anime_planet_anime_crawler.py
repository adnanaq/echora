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

from enrichment.sources.anime_planet.anime_planet_models import (
    AnimePlanetAggregateRating,
    AnimePlanetAnime,
    AnimePlanetMangaEntry,
    AnimePlanetRelatedEntry,
)
from enrichment.sources.anime_planet.animeplanet_mapper import anime_from_animeplanet
from enrichment.sources.base.framework import (
    BaseCrawler,
    FileRepository,
    NullRepository,
)
from http_cache.config import get_cache_config
from http_cache.result_cache import cached_result
from lxml import etree

logger = logging.getLogger(__name__)

_CACHE_CONFIG = get_cache_config()
TTL_ANIME_PLANET = _CACHE_CONFIG.ttl_anime_planet

BASE_ANIME_URL = "https://www.anime-planet.com/anime/"

_SEASON_SLUG_RE = re.compile(r"/seasons/([^/?#]+)")
_RANK_RE = re.compile(r"#(\d+)")
_AKA_PREFIX = "alt title:"

# XPaths for entryBar metadata and relations
_XPATHS: dict[str, str] = {
    "type_raw": "//section[contains(@class,'entryBar')]//span[@class='type']",
    "season_url": "//section[contains(@class,'entryBar')]//a[contains(@href,'/anime/seasons/')]/@href",
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

    Parses the fa-tv metadata span (e.g. "OVA: 1 ep", "Movie", "TV Special: 9 ep")
    into separate type and episode_count fields.

    Args:
        raw: List of dicts produced by ``_parse_related_entry_element``.

    Returns:
        List of validated ``AnimePlanetRelatedEntry`` models.
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
    """Build AnimePlanetMangaEntry models from raw relation dicts.

    Parses the fa-book-open metadata span into type, volumes, and chapters:
      - "One Shot"           → type="One Shot", chapters=1
      - "Vol: 114 - Ch: 179" → volumes=114, chapters=179
      - "Vol: 1"             → volumes=1
      - "Ch: 19"             → chapters=19
      - ""  / "- ?"          → all None (date bleed-through guard)

    Args:
        raw: List of dicts produced by ``_parse_related_entry_element``.

    Returns:
        List of validated ``AnimePlanetMangaEntry`` models.
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
    """Parse a JSON-LD aggregateRating dict into an AnimePlanetAggregateRating model.

    Args:
        ar: Raw ``aggregateRating`` dict from JSON-LD, or None.

    Returns:
        Parsed model, or None if both rating fields are absent or invalid.
    """
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

    tree = etree.fromstring(html, etree.HTMLParser(encoding="utf-8"))

    def _t(key: str) -> str | None:
        els = cast(list[Any], tree.xpath(_XPATHS[key]))
        return _tc(els[0]) if els else None

    def _a(key: str) -> str | None:
        vals = cast(list[Any], tree.xpath(_XPATHS[key]))
        return vals[0] if vals else None

    def _texts(key: str) -> list[str]:
        return [_tc(el) for el in cast(list[Any], tree.xpath(_XPATHS[key])) if _tc(el)]

    def _related(key: str, *, is_manga: bool = False) -> list[dict[str, Any]]:
        return [
            _parse_related_entry_element(el, is_manga=is_manga)
            for el in cast(list[Any], tree.xpath(_XPATHS[key]))
        ]

    return {
        "name": json_ld["name"],
        "schema_type": json_ld.get("@type"),
        "description": json_ld.get("description"),
        "url": json_ld.get("url"),
        "start_date": json_ld.get("startDate"),
        "end_date": json_ld.get("endDate"),
        "number_of_episodes": json_ld.get("numberOfEpisodes"),
        "genres": json_ld.get("genre") or [],
        "aggregate_rating": json_ld.get("aggregateRating"),
        "type_raw": _t("type_raw"),
        "season_url": _a("season_url"),
        "rank_text": _t("rank_text"),
        "aka": _t("aka"),
        "cover": _a("cover"),
        "studios": _texts("studios"),
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


async def _fetch_anime_html(url: str) -> str | None:
    """Navigate to an Anime-Planet anime page and return its rendered HTML.

    Args:
        url: Full Anime-Planet anime URL
            (e.g. ``https://www.anime-planet.com/anime/dandadan``).

    Returns:
        Rendered page HTML, or None on navigation failure.
    """
    import zendriver as zd

    browser = await zd.start(headless=True)
    try:
        page = await browser.get(url)
        await page.wait_for(selector="section.entryBar", timeout=20)
        return await page.get_content()
    except Exception as exc:
        logger.warning(f"navigation failed for {url}: {exc}")
        return None
    finally:
        try:
            await browser.stop()
        except Exception as exc:
            logger.debug(f"browser stop failed: {exc}")


@cached_result(
    ttl=TTL_ANIME_PLANET,
    key_prefix="animeplanet_anime",
    dependencies=[_extract_anime_from_html],
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
        Raw data dict, or None on failure.
    """
    url = f"{BASE_ANIME_URL}{canonical_slug}"
    logger.info(f"Fetching anime data: {url}")

    html = await _fetch_anime_html(url)
    if not html:
        logger.warning(f"Navigation returned no HTML for {url}")
        return None

    raw = _extract_anime_from_html(html)
    if not raw:
        logger.warning(f"No data extracted from {url}")
        return None

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
