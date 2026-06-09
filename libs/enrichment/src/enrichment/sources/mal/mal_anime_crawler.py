"""MAL anime detail crawler — zendriver + lxml XPath.

CLI usage::

    uv run python -m enrichment.sources.mal.mal_anime_crawler <url> [--output path]

Example::

    uv run python -m enrichment.sources.mal.mal_anime_crawler \\
        https://myanimelist.net/anime/21/One_Piece --output one_piece.json
"""

import argparse
import asyncio
import logging
import re
import sys
from typing import Any, cast

from enrichment.sources.base.framework import (
    BaseCrawler,
    DockerTransport,
    FileRepository,
    NullRepository,
)
from enrichment.sources.mal.mal_base import (
    MAL_BASE_URL,
    parse_aired_string,
    parse_broadcast_string,
    parse_duration_seconds,
    parse_episode_ranges,
    parse_number,
    parse_premiered,
)
from enrichment.sources.mal.mal_mapper import anime_from_mal
from enrichment.sources.mal.mal_models import (
    MalAnime,
    MalCompanyRef,
    MalEpisodeRange,
    MalExternalLink,
    MalRelatedEntry,
    MalThemeSong,
    MalTrailer,
)
from http_cache.config import get_cache_config
from http_cache.result_cache import cached_result

logger = logging.getLogger(__name__)

_CACHE_CONFIG = get_cache_config()
TTL_MAL = _CACHE_CONFIG.ttl_jikan

_INTER_REQUEST_DELAY = 3.0

# ---------------------------------------------------------------------------
# XPath selectors
# ---------------------------------------------------------------------------

_XPATHS: dict[str, str] = {
    # Sidebar text divs (dark_text label span + value)
    "type":            "//div[span[contains(@class,'dark_text')][contains(.,'Type:')]]",
    "episodes":        "//div[span[contains(@class,'dark_text')][contains(.,'Episodes:')]]",
    "status":          "//div[span[contains(@class,'dark_text')][contains(.,'Status:')]]",
    "duration_raw":    "//div[span[contains(@class,'dark_text')][contains(.,'Duration:')]]",
    "source_material": "//div[span[contains(@class,'dark_text')][contains(.,'Source:')]]",
    "rating":          "//div[span[contains(@class,'dark_text')][contains(.,'Rating:')]]",
    "aired_raw":       "//div[span[contains(@class,'dark_text')][contains(.,'Aired:')]]",
    "premiered_raw":   "//div[span[contains(@class,'dark_text')][contains(.,'Premiered:')]]",
    "broadcast_raw":   "//div[span[contains(@class,'dark_text')][contains(.,'Broadcast:')]]",
    "title_english":   "//div[span[contains(@class,'dark_text')][contains(.,'English:')]]",
    "title_japanese":  "//div[span[contains(@class,'dark_text')][contains(.,'Japanese:')]]",
    "synonyms_raw":    "//div[span[contains(@class,'dark_text')][contains(.,'Synonyms:')]]",
    "rank_div":        "//div[span[contains(@class,'dark_text')][contains(.,'Ranked:')]]",
    "popularity":      "//div[span[contains(@class,'dark_text')][contains(.,'Popularity:')]]",
    "members":         "//div[span[contains(@class,'dark_text')][contains(.,'Members:')]]",
    "favorites":       "//div[span[contains(@class,'dark_text')][contains(.,'Favorites:')]]",
    # Array anchors
    "genres":          "//div[span[contains(@class,'dark_text')][contains(.,'Genre')]]/a",
    "themes":          "//div[span[contains(@class,'dark_text')][contains(.,'Theme')]]/a",
    "demographics":    "//div[span[contains(@class,'dark_text')][contains(.,'Demographic')]]/a",
    "producers":       "//div[span[contains(@class,'dark_text')][contains(.,'Producers')]]/a",
    "licensors":       "//div[span[contains(@class,'dark_text')][contains(.,'Licensors')]]/a",
    "studios":         "//div[span[contains(@class,'dark_text')][contains(.,'Studios')]]/a",
    # Title / meta
    "title":           "//h1[contains(@class,'title-name')]/strong",
    "title_og":        "//meta[@property='og:title']/@content",
    # Schema.org stats
    "score":           "//span[@itemprop='ratingValue']",
    "scored_by":       "//span[@itemprop='ratingCount']",
    "synopsis":        "//p[@itemprop='description']",
    "cover_image_src": "//img[@itemprop='image']/@data-src",
    # Background (outer HTML of the containing <td>; regex in _build extracts text)
    "background_raw":  "//h2[@id='background']/parent::div/parent::td",
    # Related entries
    "related_tile_entries": "//div[contains(@class,'entries-tile')]/div[contains(@class,'entry')]",
    "related_table_rows":   "//table[contains(@class,'entries-table')]//tr[td[2]]",
    # External / streaming links
    "external_source_anchors": (
        "//h2[normalize-space()='Available At' or normalize-space()='Resources']"
        "/following-sibling::div[1][contains(@class,'external_links')]"
        "//a[@href and not(@href='#')]"
    ),
    "streaming_anchors": (
        "//h2[normalize-space()='Streaming Platforms']"
        "/following-sibling::div[1][contains(@class,'broadcasts')]"
        "//a[@href and @title]"
    ),
    # Theme songs (note: MAL typo "opnening" is intentional)
    "opening_theme_rows": "//div[contains(@class,'theme-songs') and contains(@class,'opnening')]//tr[td[2]]",
    "ending_theme_rows":  "//div[contains(@class,'theme-songs') and contains(@class,'ending')]//tr[td[2]]",
    # Trailer
    "trailer_anchor": "//div[contains(@class,'video-promotion')]//a",
    "trailer_title":  "//div[contains(@class,'video-promotion')]//span[contains(@class,'title')]",
    # Gallery (pics page only)
    "pic_surrounds":  "//div[contains(@class,'picSurround')]/a[@href]",
}

# ---------------------------------------------------------------------------
# lxml extraction
# ---------------------------------------------------------------------------


def _extract_anime_from_html(html_text: str) -> dict[str, Any] | None:
    """Parse a MAL anime detail page into the raw field dict used by _build_anime_from_raw.

    Args:
        html_text: Full HTML of a MAL anime detail page (e.g. /anime/21/One_Piece).

    Returns:
        Dict of raw extracted fields, or None if parsing fails entirely.
    """
    from lxml import etree

    try:
        parser = etree.HTMLParser(encoding="utf-8")
        tree = etree.fromstring(html_text.encode(), parser)
        if tree is None:  # pragma: no cover
            return None  # pragma: no cover
    except Exception:  # pragma: no cover
        return None  # pragma: no cover

    def _text(key: str) -> str | None:
        els = cast(list[Any], tree.xpath(_XPATHS[key]))
        if not els:
            return None
        return "".join(els[0].itertext()).strip() or None

    def _attr(key: str) -> str | None:
        vals = cast(list[str], tree.xpath(_XPATHS[key]))
        return vals[0].strip() if vals else None

    def _html(key: str) -> str | None:
        els = cast(list[Any], tree.xpath(_XPATHS[key]))
        if not els:
            return None
        return etree.tostring(els[0], encoding="unicode", method="html")

    def _sidebar(key: str, pattern: str) -> str | None:
        text = _text(key)
        if not text:
            return None
        m = re.search(pattern, text, re.DOTALL)
        return m.group(1).strip() or None if m else None

    def _company_list(key: str) -> list[dict[str, str]]:
        els = cast(list[Any], tree.xpath(_XPATHS[key]))
        return [
            {"name": "".join(el.itertext()).strip(), "source": el.get("href") or ""}
            for el in els
            if "".join(el.itertext()).strip()
        ]

    def _name_list(key: str) -> list[dict[str, str]]:
        els = cast(list[Any], tree.xpath(_XPATHS[key]))
        return [
            {"name": "".join(el.itertext()).strip()}
            for el in els
            if "".join(el.itertext()).strip()
        ]

    def _theme_rows(key: str) -> list[dict[str, Any]]:
        rows = []
        for tr in cast(list[Any], tree.xpath(_XPATHS[key])):
            td2 = cast(list[Any], tr.xpath(".//td[2]"))
            title_text = "".join(td2[0].itertext()).strip() if td2 else ""
            artist_els = cast(list[Any], tr.xpath(".//span[contains(@class,'theme-song-artist')]"))
            artist = "".join(artist_els[0].itertext()).strip() if artist_els else None
            ep_els = cast(list[Any], tr.xpath(".//span[contains(@class,'theme-song-episode')]"))
            episodes = "".join(ep_els[0].itertext()).strip() if ep_els else None
            rows.append({"title_text": title_text, "artist": artist, "episodes": episodes})
        return rows

    # Related tile entries
    tile_entries = []
    for entry in cast(list[Any], tree.xpath(_XPATHS["related_tile_entries"])):
        rel_els = cast(list[Any], entry.xpath(".//div[contains(@class,'relation')]"))
        relation_raw = "".join(rel_els[0].itertext()).strip() if rel_els else ""

        title_anchors = cast(list[Any], entry.xpath(".//div[contains(@class,'title')]/a"))
        title_text = "".join(title_anchors[0].itertext()).strip() if title_anchors else ""
        m = re.search(r"^(.*?)(?:\s*\([^)]+\))?\s*$", title_text)
        entry_title = m.group(1).strip() if m else title_text

        title_divs = cast(list[Any], entry.xpath(".//div[contains(@class,'title')]"))
        div_text = "".join(title_divs[0].itertext()).strip() if title_divs else ""
        type_m = re.search(r"\(([^)]+)\)\s*$", div_text)
        entry_type = type_m.group(1) if type_m else None

        tile_entries.append({
            "relation_raw": relation_raw,
            "title": entry_title,
            "entry_type": entry_type,
            "source": title_anchors[0].get("href") if title_anchors else None,
        })

    # Related table entries
    table_entries = []
    for row in cast(list[Any], tree.xpath(_XPATHS["related_table_rows"])):
        td1 = cast(list[Any], row.xpath("./td[1]"))
        relation = "".join(td1[0].itertext()).strip() if td1 else ""
        td2 = cast(list[Any], row.xpath("./td[2]"))
        links_html = etree.tostring(td2[0], encoding="unicode", method="html") if td2 else ""
        table_entries.append({"relation": relation, "links_html": links_html})

    # External sources
    external_sources_raw = []
    for a in cast(list[Any], tree.xpath(_XPATHS["external_source_anchors"])):
        cap = cast(list[Any], a.xpath(".//div[contains(@class,'caption')]"))
        name = "".join(cap[0].itertext()).strip() if cap else ""
        external_sources_raw.append({"name": name, "source": a.get("href") or ""})

    # Streaming links
    streaming_links_raw = [
        {"name": a.get("title") or "", "source": a.get("href") or ""}
        for a in cast(list[Any], tree.xpath(_XPATHS["streaming_anchors"]))
    ]

    # Trailer
    trailer_anchors = cast(list[Any], tree.xpath(_XPATHS["trailer_anchor"]))
    trailer_embed_url = trailer_anchors[0].get("href") if trailer_anchors else None

    return {
        "type":               _sidebar("type", r"Type:\s*(.*)"),
        "episodes":           _sidebar("episodes", r"Episodes:\s*(.*)"),
        "status":             _sidebar("status", r"Status:\s*(.*)"),
        "duration_raw":       _sidebar("duration_raw", r"Duration:\s*(.*)"),
        "source_material":    _sidebar("source_material", r"Source:\s*(.*)"),
        "rating":             _sidebar("rating", r"Rating:\s*(.*)"),
        "aired_raw":          _sidebar("aired_raw", r"Aired:\s*(.*)"),
        "premiered_raw":      _sidebar("premiered_raw", r"Premiered:\s*(.*)"),
        "broadcast_raw":      _sidebar("broadcast_raw", r"Broadcast:\s*(.*)"),
        "title_english":      _sidebar("title_english", r"English:\s*(.*)"),
        "title_japanese":     _sidebar("title_japanese", r"Japanese:\s*(.*)"),
        "synonyms_raw":       _sidebar("synonyms_raw", r"Synonyms:\s*(.*)"),
        "rank_html":          _html("rank_div"),
        "popularity":         _sidebar("popularity", r"Popularity:\s*#?(\d+)"),
        "members":            _sidebar("members", r"Members:\s*([\d,]+)"),
        "favorites":          _sidebar("favorites", r"Favorites:\s*([\d,]+)"),
        "genres":             _name_list("genres"),
        "themes":             _name_list("themes"),
        "demographics":       _name_list("demographics"),
        "producers":          _company_list("producers"),
        "licensors":          _company_list("licensors"),
        "studios":            _company_list("studios"),
        "background_raw":     _html("background_raw"),
        "related_tile_entries":  tile_entries,
        "related_table_entries": table_entries,
        "external_sources_raw":  external_sources_raw,
        "streaming_links_raw":   streaming_links_raw,
        "title":              _text("title"),
        "title_og":           _attr("title_og"),
        "score":              _text("score"),
        "scored_by":          _text("scored_by"),
        "synopsis":           _text("synopsis"),
        "cover_image_src":    _attr("cover_image_src"),
        "opening_themes_raw": _theme_rows("opening_theme_rows"),
        "ending_themes_raw":  _theme_rows("ending_theme_rows"),
        "trailer_embed_url":  trailer_embed_url,
        "trailer_title":      _text("trailer_title"),
        "picture_urls_raw":   [],
    }


def _extract_pics_from_html(html_text: str) -> list[str]:
    """Extract gallery image URLs from a MAL /pics page.

    Args:
        html_text: Full HTML of a MAL /anime/<id>/pics page.

    Returns:
        List of full-size image URLs from the gallery. Empty list on parse failure
        or if the page was captured before the gallery JS rendered.
    """
    from lxml import etree

    try:
        parser = etree.HTMLParser(encoding="utf-8")
        tree = etree.fromstring(html_text.encode(), parser)
        if tree is None:  # pragma: no cover
            return []  # pragma: no cover
    except Exception:  # pragma: no cover
        return []  # pragma: no cover

    return [
        a.get("href")
        for a in cast(list[Any], tree.xpath(_XPATHS["pic_surrounds"]))
        if a.get("href") and "myanimelist" in a.get("href", "") and "images/anime" in a.get("href", "")
    ]


# ---------------------------------------------------------------------------
# Browser navigation helper
# ---------------------------------------------------------------------------


async def _fetch_page_html(browser: Any, url: str, wait_selector: str | None = None) -> str | None:
    """Navigate to a URL with zendriver and return the rendered HTML.

    Args:
        browser: Active zendriver browser instance.
        url: Page URL to fetch.
        wait_selector: CSS selector to wait for before capturing HTML. If None,
            waits 2 seconds instead.

    Returns:
        Rendered HTML string, or None if navigation fails.
    """
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


async def _fetch_pics_html(browser: Any, url: str) -> str | None:
    """Fetch a MAL /pics page, scrolling to trigger lazy-loaded gallery images.

    The gallery uses an intersection-observer; images only load when they enter
    the viewport. scroll_down(amount=1000) scrolls 10x the page height, ensuring
    all images are triggered before HTML is captured.

    Args:
        browser: Active zendriver browser instance.
        url: MAL /pics page URL (e.g. https://myanimelist.net/anime/21/One_Piece/pics).

    Returns:
        Rendered HTML with all gallery images loaded, or None if navigation fails.
    """
    try:
        page = await browser.get(url)
        await page.wait_for(selector="div.picSurround", timeout=10)
        # Gallery uses intersection-observer lazy loading — scroll to load all images
        await page.scroll_down(amount=1000, speed=3000)
        await asyncio.sleep(2)
        return await page.get_content()
    except Exception as exc:
        logger.warning(f"navigation failed for {url}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Post-processing helpers (pure transforms — unchanged from crawl4ai version)
# ---------------------------------------------------------------------------


def _parse_trailer(raw: dict[str, Any]) -> MalTrailer | None:
    embed_url = raw.get("trailer_embed_url") or ""
    m = re.search(r"embed/([^?&]+)", embed_url)
    if not m:
        return None
    vid = m.group(1)
    return MalTrailer(
        source=f"https://www.youtube.com/watch?v={vid}",
        title=raw.get("trailer_title"),
        thumbnail=f"https://img.youtube.com/vi/{vid}/maxresdefault.jpg",
    )


def _normalize_mal_url(path: str) -> str:
    """Ensure a MAL path is a full absolute URL.

    Args:
        path: Relative path (e.g. ``/anime/21``) or full URL.

    Returns:
        Full URL with MAL_BASE_URL prepended if needed. Empty string if path is empty.
    """
    if not path:
        return ""
    if path.startswith("http"):
        return path
    return f"{MAL_BASE_URL}{path if path.startswith('/') else '/' + path}"


def _parse_structured_themes(raw_themes: list[dict[str, Any]]) -> list[MalThemeSong]:
    """Parse lxml-extracted theme song rows into MalThemeSong models.

    Each row must have a quoted title (``"Song Name"``) to be included; rows
    without quotes (junk text, headers) are skipped.

    Args:
        raw_themes: List of dicts with ``title_text``, ``artist``, and ``episodes`` keys,
            as extracted by the ``opening_theme_rows`` / ``ending_theme_rows`` XPaths.

    Returns:
        List of parsed MalThemeSong objects.
    """
    results = []
    for raw in raw_themes:
        title_text = (raw.get("title_text") or "").strip()

        if '"' not in title_text:
            continue

        title_match = re.search(r'"([^"]+)"', title_text)
        title = title_match.group(1) if title_match else title_text

        artist = (raw.get("artist") or "").strip()
        artist = re.sub(r"^by\s+", "", artist, flags=re.IGNORECASE).strip()

        episodes_raw = raw.get("episodes")
        episode_ranges = [
            MalEpisodeRange(start=s, end=e)
            for s, e in parse_episode_ranges(episodes_raw)
        ]

        results.append(
            MalThemeSong(title=title, artist=artist or None, episodes=episode_ranges)
        )
    return results


def _parse_all_related_entries(raw: dict[str, Any]) -> list[MalRelatedEntry]:
    """Merge tile and table related-entry formats into a unified MalRelatedEntry list.

    MAL renders related entries in two different layouts depending on the number
    of entries: a tile grid (``related_tile_entries``) for short lists and an
    HTML table (``related_table_entries``) for longer ones. Both are normalised
    here into the same structure.

    Args:
        raw: Raw extraction dict from ``_extract_anime_from_html``, expected to
            contain ``related_tile_entries`` and/or ``related_table_entries`` keys.

    Returns:
        Deduplicated list of MalRelatedEntry objects with title, source URL,
        relation type, and entry type populated.
    """
    unified_items = []

    for entry in raw.get("related_tile_entries", []):
        raw_rel = (entry.get("relation_raw") or "").strip()
        rel_parts = [p.strip() for p in raw_rel.split("\n") if p.strip()]
        relation = rel_parts[0] if rel_parts else ""

        entry_type = entry.get("entry_type")
        if not entry_type and len(rel_parts) > 1:
            type_match = re.search(r"\(([^)]+)\)", rel_parts[1])
            if type_match:
                entry_type = type_match.group(1)

        unified_items.append({
            "relation": relation,
            "title": entry.get("title"),
            "entry_type": entry_type,
            "source": entry.get("source"),
        })

    for row in raw.get("related_table_entries", []):
        raw_rel = (row.get("relation") or "").strip()
        rel_parts = [p.strip() for p in raw_rel.split("\n") if p.strip()]
        relation = rel_parts[0] if rel_parts else ""

        links_html = row.get("links_html") or ""
        for match in re.finditer(
            r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>(.*?)(?:</li>|<li>|<a|</ul>|$)',
            links_html,
            re.DOTALL,
        ):
            source_url = match.group(1).strip()
            title = re.sub(r"<[^>]+>", "", match.group(2)).strip()
            format_text = match.group(3).strip()

            entry_type = None
            type_match = re.search(r"\(([^)]+)\)", format_text)
            if type_match:
                entry_type = type_match.group(1)
            elif len(rel_parts) > 1:
                type_match = re.search(r"\(([^)]+)\)", rel_parts[1])
                if type_match:
                    entry_type = type_match.group(1)

            unified_items.append({
                "relation": relation,
                "title": title,
                "entry_type": entry_type,
                "source": source_url,
            })

    related_entries = []
    for item in unified_items:
        relation = (item.get("relation") or "").strip().rstrip(":")
        source_url = _normalize_mal_url(item.get("source", ""))
        title = (item.get("title") or "").strip()

        if not title or not source_url:
            continue

        related_entries.append(
            MalRelatedEntry(
                relation=relation,
                title=title,
                source=source_url,
                entry_type=item.get("entry_type"),
                is_anime="/anime/" in source_url,
            )
        )

    return related_entries


def _build_anime_from_raw(
    raw: dict[str, Any],
    url: str,
    picture_urls: list[str],
) -> MalAnime:
    """Transform the raw XPath extraction dict into a typed MalAnime model.

    Args:
        raw: Dict produced by ``_extract_anime_from_html``, augmented with
            ``_url`` and ``_picture_urls`` keys set by ``_fetch_mal_anime_data``.
        url: Canonical MAL URL for this anime (used as the model's ``url`` field).
        picture_urls: Gallery image URLs from ``_extract_pics_from_html``.

    Returns:
        Populated MalAnime Pydantic model ready for mapping to canonical form.
    """
    title = (raw.get("title") or raw.get("title_og") or "").strip()
    title_english = raw.get("title_english")
    title_japanese = raw.get("title_japanese")
    synonyms_str = raw.get("synonyms_raw")
    synonyms = [s.strip() for s in synonyms_str.split(",")] if synonyms_str else []

    anime_type = raw.get("type")
    status = raw.get("status")
    source_material = raw.get("source_material")
    rating = raw.get("rating")

    ep_count_raw = raw.get("episodes")
    episode_count: int | None = None
    if ep_count_raw and ep_count_raw.lower() not in ("unknown", "?", "n/a"):
        try:
            episode_count = int(ep_count_raw.replace(",", ""))
        except ValueError:
            episode_count = None

    duration_raw = raw.get("duration_raw")
    duration = parse_duration_seconds(duration_raw) if duration_raw else None

    aired_raw = raw.get("aired_raw")
    aired_from, aired_to = parse_aired_string(aired_raw)

    premiered_raw = raw.get("premiered_raw")
    season, year = parse_premiered(premiered_raw)

    broadcast_raw = raw.get("broadcast_raw")
    broadcast_day, broadcast_time, broadcast_timezone = parse_broadcast_string(broadcast_raw)

    score_val = raw.get("score")
    score = float(score_val.strip()) if score_val and score_val.strip() else None

    scored_by = parse_number(raw.get("scored_by"))

    rank_html = raw.get("rank_html")
    rank: int | None = None
    if rank_html:
        rank_match = re.search(r"#(\d+)", rank_html)
        if rank_match:
            rank = int(rank_match.group(1))

    popularity = parse_number(raw.get("popularity"))
    members = parse_number(raw.get("members"))
    favorites = parse_number(raw.get("favorites"))

    genres = [g["name"] for g in raw.get("genres", [])]
    themes = [t["name"] for t in raw.get("themes", [])]
    demographics = [d["name"] for d in raw.get("demographics", [])]

    producers = [
        MalCompanyRef(
            name=item["name"].strip(), source=_normalize_mal_url(item["source"])
        )
        for item in raw.get("producers", [])
        if item.get("name")
        and item.get("source")
        and "dbchanges.php" not in item["source"]
    ]
    licensors = [
        MalCompanyRef(
            name=item["name"].strip(), source=_normalize_mal_url(item["source"])
        )
        for item in raw.get("licensors", [])
        if item.get("name")
        and item.get("source")
        and "dbchanges.php" not in item["source"]
    ]
    studios = [
        MalCompanyRef(
            name=item["name"].strip(), source=_normalize_mal_url(item["source"])
        )
        for item in raw.get("studios", [])
        if item.get("name")
        and item.get("source")
        and "dbchanges.php" not in item["source"]
    ]

    opening_themes = _parse_structured_themes(raw.get("opening_themes_raw", []))
    ending_themes = _parse_structured_themes(raw.get("ending_themes_raw", []))

    cover_url = raw.get("cover_image_src") or ""
    if cover_url and cover_url.endswith(".jpg") and not cover_url.endswith("l.jpg"):
        cover_url = cover_url[:-4] + "l.jpg"
    picture_urls = list(
        dict.fromkeys(([cover_url] if cover_url else []) + picture_urls)
    )
    images: dict[str, str] = {}

    external_sources = [
        MalExternalLink(name=item["name"].strip(), source=item["source"])
        for item in raw.get("external_sources_raw", [])
        if item.get("name") and item.get("source")
    ]

    streaming = [
        MalExternalLink(name=item["name"].strip(), source=item["source"])
        for item in raw.get("streaming_links_raw", [])
        if item.get("name") and item.get("source")
    ]

    bg_html = raw.get("background_raw") or ""
    background: str | None = None
    if bg_html:
        bg_match = re.search(
            r'id="background".*?</h2>(.*?)(?:<div|<!--|$)', bg_html, re.DOTALL
        )
        if bg_match:
            bg_text = re.sub(r"<[^>]+>", "", bg_match.group(1)).strip()
            if bg_text and "no background information" not in bg_text.lower():
                background = bg_text

    related_entries = _parse_all_related_entries(raw)

    return MalAnime(
        source=url,
        title=title,
        title_english=title_english,
        title_japanese=title_japanese,
        synonyms=synonyms,
        type=anime_type,
        status=status,
        source_material=source_material,
        rating=rating,
        year=year,
        season=season,
        aired_from=aired_from,
        aired_to=aired_to,
        broadcast_day=broadcast_day,
        broadcast_time=broadcast_time,
        broadcast_timezone=broadcast_timezone,
        episode_count=episode_count,
        duration=duration,
        score=score,
        scored_by=scored_by,
        rank=rank,
        popularity=popularity,
        members=members,
        favorites=favorites,
        synopsis=raw.get("synopsis"),
        background=background,
        genres=genres,
        themes=themes,
        demographics=demographics,
        producers=producers,
        licensors=licensors,
        studios=studios,
        related_entries=related_entries,
        images=images,
        picture_urls=picture_urls,
        trailer=_parse_trailer(raw),
        opening_themes=opening_themes,
        ending_themes=ending_themes,
        external_sources=external_sources,
        streaming=streaming,
    )


# ---------------------------------------------------------------------------
# Cached fetch
# ---------------------------------------------------------------------------


@cached_result(
    ttl=TTL_MAL,
    key_prefix="mal_anime_scraped",
    dependencies=[_extract_anime_from_html, _extract_pics_from_html],
)
async def _fetch_mal_anime_data(url: str) -> dict[str, Any] | None:
    """Fetch and extract a MAL anime detail page via zendriver. Result is cached by URL.

    Opens a single browser session, fetches the main detail page and the /pics
    gallery page sequentially, waits for Vue-rendered sections (theme songs,
    related anime), and scrolls the /pics page to trigger lazy-loaded images.

    Args:
        url: Full MAL anime URL (e.g. ``https://myanimelist.net/anime/21/One_Piece``).

    Returns:
        Raw extraction dict with ``_url`` and ``_picture_urls`` keys populated,
        or None if navigation or extraction fails.
    """
    import zendriver as zd

    browser = await zd.start(headless=False)
    try:
        try:
            main_page = await browser.get(url)
            await main_page.wait_for(selector="h1.title-name", timeout=10)
            # Theme songs and related entries are Vue-rendered; wait for them
            try:
                await main_page.wait_for(selector="div.theme-songs", timeout=15)
            except Exception:
                pass
            await asyncio.sleep(2)
            final_url = main_page.url
            main_html = await main_page.get_content()
        except Exception as exc:
            logger.warning(f"navigation failed for {url}: {exc}")
            return None

        if not main_html:
            logger.warning(f"No HTML from MAL anime page: {url}")
            return None

        raw = _extract_anime_from_html(main_html)
        if raw is None:
            logger.warning(f"Failed to extract data from MAL anime page: {url}")
            return None

        canonical_url = final_url or url
        await asyncio.sleep(_INTER_REQUEST_DELAY)

        pics_url = f"{canonical_url}/pics"
        pics_html = await _fetch_pics_html(browser, pics_url)
        picture_urls = _extract_pics_from_html(pics_html) if pics_html else []

    finally:
        try:
            await browser.stop()
        except Exception:
            pass

    raw["_picture_urls"] = picture_urls
    raw["_url"] = canonical_url
    return raw


# ---------------------------------------------------------------------------
# Crawler class
# ---------------------------------------------------------------------------


class MalAnimeCrawler(BaseCrawler[MalAnime, dict[str, Any]]):
    """Crawler for MyAnimeList anime detail pages.

    Uses zendriver for browser automation and lxml XPath for extraction.
    Implements the BaseCrawler template-method pattern.
    """

    def get_extraction_schema(self) -> dict[str, Any]:
        return {"xpaths": _XPATHS}

    def normalize_identifier(self, identifier: str) -> str:
        return _normalize_mal_url(identifier)

    async def fetch_raw_data(self, url: str) -> dict[str, Any] | None:
        return await _fetch_mal_anime_data(url)

    def build_source_model(self, processed_raw: dict[str, Any], url: str) -> MalAnime:
        picture_urls = processed_raw.pop("_picture_urls", [])
        saved_url = processed_raw.pop("_url", url)
        return _build_anime_from_raw(processed_raw, saved_url, picture_urls)

    def map_to_canonical(self, source_model: MalAnime) -> dict[str, Any]:
        return anime_from_mal(source_model)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def fetch_mal_anime(
    url: str, output_path: str | None = None
) -> dict[str, Any] | None:
    """Fetch a MAL anime detail page and return the canonical anime dict.

    Args:
        url: Full MAL anime URL (e.g. ``https://myanimelist.net/anime/21/One_Piece``).
        output_path: Optional path to append the result as a JSONL line. Pass None
            to discard output (e.g. when the caller handles persistence).

    Returns:
        Canonical anime dict from ``anime_from_mal``, or None if fetching fails.
    """
    repo = FileRepository(output_path) if output_path else NullRepository()
    return await MalAnimeCrawler(DockerTransport(), repo).crawl(url)


async def main() -> int:
    """CLI entry point for fetching a MAL anime page.

    Returns:
        0 on success, 1 if extraction fails.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(description="Fetch anime data from MAL")
    parser.add_argument("url", type=str, help="MAL anime URL")
    parser.add_argument("--output", type=str, default="mal_anime.json", help="Output file path")
    args = parser.parse_args()

    anime_dict = await fetch_mal_anime(args.url, output_path=args.output)
    if anime_dict is None:
        logger.error(f"No data extracted for anime URL {args.url}")
        return 1
    logger.info(f"Done: {anime_dict.get('title')} ({anime_dict.get('episode_count')} episodes)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
