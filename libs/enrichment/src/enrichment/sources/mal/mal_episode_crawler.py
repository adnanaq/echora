"""MAL episode detail crawler — zendriver + lxml XPath.

CLI usage::

    uv run python -m enrichment.sources.mal.mal_episode_crawler <url> [--output path]

Example::

    uv run python -m enrichment.sources.mal.mal_episode_crawler \\
        https://myanimelist.net/anime/21/One_Piece/episode/1 --output ep1.json
"""

import argparse
import asyncio
import logging
import re
import sys
from typing import Any, cast

from enrichment.sources.base.framework import (
    BaseCrawler,
    FileRepository,
    NullRepository,
)
from enrichment.sources.mal.mal_base import (
    parse_duration_seconds,
    parse_iso_date,
)
from enrichment.sources.mal.mal_mapper import episode_from_mal
from enrichment.sources.mal.mal_models import (
    EpisodeCharacterRef,
    EpisodeStaffRef,
    EpisodeVARef,
    MalEpisode,
)
from http_cache.config import get_cache_config
from http_cache.result_cache import cached_result
from lxml import etree

logger = logging.getLogger(__name__)

_CACHE_CONFIG = get_cache_config()
TTL_MAL = _CACHE_CONFIG.ttl_jikan

_INTER_REQUEST_DELAY = 1.5

_XPATHS: dict[str, str] = {
    "title_header": "//h2[starts-with(normalize-space(.), '#')]",
    "subtitle": "//h2[starts-with(normalize-space(.), '#')]/following-sibling::p[1]",
    "info_box": "//div[contains(@class,'di-tc') and contains(@class,'ar')]",
    "synopsis_container": "//h2[normalize-space()='Synopsis']/parent::div",
    "char_tables": (
        "//table[contains(@class,'fl-l')]"
        "[preceding::h2[1][contains(normalize-space(),'Characters')]]"
    ),
    "staff_tables": (
        "//table[contains(@class,'fl-l')][preceding::h2[1][normalize-space()='Staff']]"
    ),
}


def _tc(el: Any) -> str:
    return "".join(el.itertext()).strip()


def _extract_episode_from_html(html: str) -> dict[str, Any] | None:
    """Extract raw field dict from a MAL episode detail page.

    Args:
        html: Full HTML of a MAL episode detail page.

    Returns:
        Dict with ``title_header``, ``subtitle_raw``, ``duration_raw``,
        ``aired_raw``, ``synopsis_raw``, ``characters``, and ``staff`` keys,
        or None if the page title header cannot be found.
    """
    if not html:
        return None
    tree = etree.fromstring(html, etree.HTMLParser(encoding="utf-8"))

    title_els = cast(list[Any], tree.xpath(_XPATHS["title_header"]))
    if not title_els:
        return None
    title_header = _tc(title_els[0])

    sub_els = cast(list[Any], tree.xpath(_XPATHS["subtitle"]))
    subtitle_raw = _tc(sub_els[0]) if sub_els else None

    info_els = cast(list[Any], tree.xpath(_XPATHS["info_box"]))
    info_text = _tc(info_els[0]) if info_els else ""
    dur_m = re.search(r"Duration:\s*(\d{1,2}:\d{2}:\d{2})", info_text)
    air_m = re.search(r"Aired:\s*([A-Za-z]+ \d{1,2},\s*\d{4})", info_text)
    duration_raw = dur_m.group(1) if dur_m else None
    aired_raw = air_m.group(1) if air_m else None

    syn_els = cast(list[Any], tree.xpath(_XPATHS["synopsis_container"]))
    synopsis_raw: str | None = None
    if syn_els:
        syn_text = _tc(syn_els[0])
        syn_m = re.search(r"Synopsis\s*(.*)", syn_text, re.DOTALL)
        synopsis_raw = syn_m.group(1).strip() if syn_m else None

    char_tables = cast(list[Any], tree.xpath(_XPATHS["char_tables"]))
    characters: list[dict[str, Any]] = []
    for t in char_tables:
        name_els = cast(
            list[Any],
            t.xpath(".//a[contains(@class,'fw-b')][contains(@href,'/character/')]"),
        )
        if not name_els:
            continue
        role_m = re.search(r"\b(Main|Supporting)\b", _tc(t))
        va_els = cast(list[Any], t.xpath(".//p[contains(@class,'pb8')]"))
        characters.append(
            {
                "char_name": _tc(name_els[0]),
                "char_url": name_els[0].get("href", ""),
                "role": role_m.group(1) if role_m else None,
                "voice_actors_html": etree.tostring(va_els[0], encoding="unicode")
                if va_els
                else "",
            }
        )

    staff_tables = cast(list[Any], tree.xpath(_XPATHS["staff_tables"]))
    staff: list[dict[str, Any]] = []
    for t in staff_tables:
        name_els = cast(
            list[Any],
            t.xpath(".//a[contains(@class,'fw-b')][contains(@href,'/people/')]"),
        )
        if not name_els:
            continue
        role_els = cast(list[Any], t.xpath(".//p[contains(@class,'pr12')]"))
        staff.append(
            {
                "name": _tc(name_els[0]),
                "person_url": name_els[0].get("href", ""),
                "role": _tc(role_els[0]) if role_els else None,
            }
        )

    raw: dict[str, Any] = {
        "title_header": title_header,
        "subtitle_raw": subtitle_raw,
        "duration_raw": duration_raw,
        "aired_raw": aired_raw,
        "synopsis_raw": synopsis_raw,
        "characters": characters,
        "staff": staff,
    }
    return raw


def _parse_title_info(
    header_text: str | None,
    subtitle_text: str | None,
    episode_number: int,
) -> tuple[str, str | None, str | None, bool, bool]:
    """Parse MAL episode header into (title, title_japanese, title_romaji, is_filler, is_recap).

    header_text comes from ``//h2[starts-with(normalize-space(.), '#')]`` and
    subtitle_text from its first p sibling. lxml itertext() emits the filler/recap
    badge text as trailing whitespace — whitespace collapsing handles it.

    Args:
        header_text: Raw h2 text, e.g. "#1 - I'm Luffy! Filler"
        subtitle_text: Raw subtitle text, e.g. "Romaji Title (Japanese Title)"
        episode_number: Fallback episode number for title when header_text is None.

    Returns:
        Tuple of (title, title_japanese, title_romaji, is_filler, is_recap).
    """
    is_filler = False
    is_recap = False
    title = f"Episode {episode_number}"

    if header_text:
        header_text = " ".join(header_text.split())
        m = re.match(r"^#\d+\s*-\s*(.*)", header_text)
        raw_title = m.group(1) if m else header_text

        filler_m = re.search(r"\s+\bFiller\b\s*$", raw_title, re.IGNORECASE)
        recap_m = re.search(r"\s+\bRecap\b\s*$", raw_title, re.IGNORECASE)
        if filler_m:
            is_filler = True
            raw_title = raw_title[: filler_m.start()]
        elif recap_m:
            is_recap = True
            raw_title = raw_title[: recap_m.start()]

        title = raw_title.strip()

    title_japanese: str | None = None
    title_romaji: str | None = None

    if subtitle_text:
        jp_m = re.search(r"\(([^)]+)\)\s*$", subtitle_text)
        if jp_m:
            title_japanese = jp_m.group(1).strip()
            title_romaji = subtitle_text[: jp_m.start()].strip()
        else:
            title_romaji = subtitle_text.strip()

    return title, title_japanese, title_romaji, is_filler, is_recap


def _parse_episode_characters(
    raw_items: list[dict[str, Any]] | None,
) -> list[EpisodeCharacterRef]:
    """Parse raw character dicts from HTML extraction into EpisodeCharacterRef list.

    Args:
        raw_items: List of dicts with ``char_name``, ``char_url``, ``role``,
            and ``voice_actors_html`` keys, as produced by
            ``_extract_episode_from_html``.

    Returns:
        List of EpisodeCharacterRef, skipping entries with missing name/URL
        or non-character URLs.
    """
    if not raw_items:
        return []

    characters: list[EpisodeCharacterRef] = []
    for item in raw_items:
        name = (item.get("char_name") or "").strip()
        url = (item.get("char_url") or "").strip()
        if not name or not url:
            continue

        char_id_match = re.search(r"/character/(\d+)", url)
        if not char_id_match:
            continue
        char_id = int(char_id_match.group(1))

        role = item.get("role") or "Supporting"

        voice_actors_html = item.get("voice_actors_html") or ""
        voice_actors: list[EpisodeVARef] = []
        if voice_actors_html:
            va_matches = re.finditer(
                r'<a[^>]+href="[^"]*/people/(\d+)[^"]*"[^>]*>(.*?)</a>\s*\(([^)]+)\)',
                voice_actors_html,
                re.DOTALL,
            )
            for m in va_matches:
                person_id = int(m.group(1))
                va_name = re.sub(r"<[^>]+>", "", m.group(2)).strip()
                language = m.group(3).strip()
                voice_actors.append(
                    EpisodeVARef(
                        person_id=person_id,
                        name=va_name,
                        language=language,
                    )
                )

        characters.append(
            EpisodeCharacterRef(
                mal_id=char_id, name=name, role=role, voice_actors=voice_actors
            )
        )

    return characters


def _parse_episode_staff(
    raw_items: list[dict[str, Any]] | None,
) -> list[EpisodeStaffRef]:
    """Parse raw staff dicts from HTML extraction into EpisodeStaffRef list.

    Args:
        raw_items: List of dicts with ``name``, ``person_url``, and ``role``
            keys, as produced by ``_extract_episode_from_html``.

    Returns:
        List of EpisodeStaffRef. Entries with no role are included with
        ``role=None`` — MAL leaves the role field blank for some credits
        (e.g. English dub talent). Entries with missing name or non-people
        URLs are skipped.
    """
    if not raw_items:
        return []

    staff: list[EpisodeStaffRef] = []
    for item in raw_items:
        name = (item.get("name") or "").strip()
        url = (item.get("person_url") or "").strip()
        if not name or not url:
            continue

        person_match = re.search(r"/people/(\d+)", url)
        if not person_match:
            continue
        person_id = int(person_match.group(1))

        role = (item.get("role") or "").strip() or None
        staff.append(EpisodeStaffRef(person_id=person_id, name=name, role=role))

    return staff


async def _fetch_episode_html(browser: Any, url: str) -> tuple[str, str] | None:
    """Navigate to a MAL episode URL and return (html, canonical_url).

    Args:
        browser: Active zendriver browser instance.
        url: MAL episode URL.

    Returns:
        Tuple of (rendered HTML, canonical URL after redirect), or None on
        failure.
    """
    try:
        page = await browser.get(url)
        await page.wait_for(selector="h2.fs18", timeout=15)
        return await page.get_content(), page.url or url
    except Exception as exc:
        logger.warning(f"navigation failed for {url}: {exc}")
        return None


@cached_result(
    ttl=TTL_MAL,
    key_prefix="mal_episode_detail",
    dependencies=[_extract_episode_from_html],
)
async def _fetch_mal_episode_data(url: str) -> dict[str, Any] | None:
    """Fetch a MAL episode page by full URL and extract data. Cached by URL."""
    import zendriver as zd

    browser = await zd.start(headless=True)
    try:
        result = await _fetch_episode_html(browser, url)
    finally:
        try:
            await browser.stop()
        except Exception as exc:
            logger.debug(f"browser stop failed: {exc}")

    if result is None:
        return None

    html, canonical_url = result
    raw = _extract_episode_from_html(html)
    if raw is None:
        logger.warning(f"extraction failed for episode {url}")
        return None

    raw["_url"] = canonical_url
    return raw


def _build_episode_from_raw(
    raw: dict[str, Any], episode_number: int, url: str
) -> MalEpisode:
    """Construct a MalEpisode from a raw extraction dict."""
    saved_url = raw.pop("_url", url)

    title, title_japanese, title_romaji, is_filler, is_recap = _parse_title_info(
        raw.get("title_header"),
        raw.get("subtitle_raw"),
        episode_number,
    )
    aired = parse_iso_date(raw.get("aired_raw"))
    duration = parse_duration_seconds(raw.get("duration_raw"))
    syn_raw = raw.get("synopsis_raw")
    synopsis = " ".join(syn_raw.split()).strip() if syn_raw else None
    if synopsis:
        source_idx = synopsis.find("(Source:")
        if source_idx != -1:
            synopsis = synopsis[:source_idx].strip() or None
    if synopsis and "doesn't seem to have a synopsis" in synopsis:
        synopsis = None
    characters = _parse_episode_characters(raw.get("characters"))
    staff = _parse_episode_staff(raw.get("staff"))

    return MalEpisode(
        episode_number=episode_number,
        source=saved_url,
        title=title,
        title_japanese=title_japanese,
        title_romaji=title_romaji,
        synopsis=synopsis,
        aired=aired,
        duration=duration,
        filler=is_filler,
        recap=is_recap,
        characters=characters,
        staff=staff,
    )


class MalEpisodeCrawler(BaseCrawler[MalEpisode, dict[str, Any]]):
    """Crawler for MyAnimeList episode detail pages."""

    def get_extraction_schema(self) -> dict[str, Any]:
        return {"xpaths": _XPATHS}

    def normalize_identifier(self, identifier: str) -> str:
        return identifier

    async def fetch_raw_data(self, url: str) -> dict[str, Any] | None:
        return await _fetch_mal_episode_data(url)

    def build_source_model(self, processed_raw: dict[str, Any], url: str) -> MalEpisode:
        m = re.search(r"/episode/(\d+)", url)
        episode_number = int(m.group(1)) if m else 0
        return _build_episode_from_raw(processed_raw, episode_number, url)

    def map_to_canonical(self, source_model: MalEpisode) -> dict[str, Any]:
        return episode_from_mal(source_model)


async def fetch_mal_episode(
    url: str, output_path: str | None = None
) -> dict[str, Any] | None:
    """Fetch a MAL episode detail page and return canonical dict.

    Args:
        url: Full MAL episode URL
            (e.g. "https://myanimelist.net/anime/21/One_Piece/episode/1").
        output_path: If provided, append the canonical dict as a JSONL line.

    Returns:
        Canonical episode dict if successful, None otherwise.
    """
    repo = FileRepository(output_path) if output_path else NullRepository()
    return await MalEpisodeCrawler(repo).crawl(url)


async def fetch_mal_episodes(
    urls: list[str],
    *,
    output_path: str | None = None,
) -> list[dict[str, Any] | None]:
    """Fetch multiple MAL episode pages in a single shared browser session.

    Cache is checked upfront for all URLs. Hits are returned immediately.
    Misses are fetched sequentially with inter-request delays. Each result
    is cached immediately so progress is not lost on cancellation.

    Args:
        urls: List of full MAL episode URLs.
        output_path: If provided, each canonical episode dict is appended as a
            JSONL line to this file as it completes.

    Returns:
        List aligned to ``urls`` — None for any failed fetch.
    """
    if not urls:
        return []

    logger.info(f"Fetching {len(urls)} MAL episodes...")
    repo = FileRepository(output_path) if output_path else NullRepository()

    cached_values, missing_indices = await _fetch_mal_episode_data.cache_batch_get(  # type: ignore[attr-defined]
        urls
    )

    episodes: list[dict[str, Any] | None] = [None] * len(urls)

    def _parse_cached(value: Any, fallback_url: str) -> dict[str, Any] | None:
        if not value or not isinstance(value, dict):
            return None
        raw = dict(value)
        url = raw.get("_url") or fallback_url
        m = re.search(r"/episode/(\d+)", url)
        if not m:
            logger.error(f"Cannot parse episode_number from URL: {url}")
            return None
        return episode_from_mal(_build_episode_from_raw(raw, int(m.group(1)), url))

    for idx, cached in enumerate(cached_values):
        parsed = _parse_cached(cached, urls[idx])
        if parsed is not None:
            episodes[idx] = parsed
            repo.save(parsed)
        else:
            if idx not in missing_indices:
                missing_indices.append(idx)

    if not missing_indices:
        return episodes

    missing_indices = sorted(set(missing_indices))

    import zendriver as zd

    browser = await zd.start(headless=True)
    try:
        for i, idx in enumerate(missing_indices):
            url = urls[idx]
            result = await _fetch_episode_html(browser, url)
            if result is None:
                episodes[idx] = None
                continue

            html, canonical_url = result
            raw = _extract_episode_from_html(html)
            if raw is None:
                logger.warning(f"extraction failed for episode {url}")
                episodes[idx] = None
                continue

            raw["_url"] = canonical_url
            await _fetch_mal_episode_data.cache_batch_set(  # type: ignore[attr-defined]
                [url], [raw]
            )

            m = re.search(r"/episode/(\d+)", canonical_url)
            if not m:
                logger.error(f"Cannot parse episode_number from URL: {canonical_url}")
                episodes[idx] = None
                continue

            canonical = episode_from_mal(
                _build_episode_from_raw(raw, int(m.group(1)), canonical_url)
            )
            episodes[idx] = canonical
            repo.save(canonical)

            if i < len(missing_indices) - 1:
                await asyncio.sleep(_INTER_REQUEST_DELAY)
    finally:
        try:
            await browser.stop()
        except Exception as exc:
            logger.debug(f"browser stop failed: {exc}")

    return episodes


async def main() -> int:
    """CLI entry point for fetching a MAL episode page."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(description="Fetch episode data from MAL")
    parser.add_argument(
        "url",
        type=str,
        help="Full MAL episode URL (e.g. https://myanimelist.net/anime/21/One_Piece/episode/1)",
    )
    parser.add_argument("--output", type=str, default="mal_episode.json")
    args = parser.parse_args()

    ep = await fetch_mal_episode(args.url, output_path=args.output)
    if ep is None:
        logger.error(f"Failed to fetch or parse episode data for: {args.url}")
        return 1

    logger.info(f"Fetched episode: {ep.get('title')}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
