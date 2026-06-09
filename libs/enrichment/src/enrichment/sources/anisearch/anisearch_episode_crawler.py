"""AniSearch episode crawler using zendriver + lxml XPath.

Fetches the /episodes sub-page of an AniSearch anime URL and extracts per-episode
metadata (number, filler/recap flags, duration, air date, and titles in up to 5
languages: EN, JA, DE, FR, IT).

Usage:
    ./pants run libs/enrichment/src/enrichment/sources/anisearch/anisearch_episode_crawler.py -- <url> [--output PATH]

    <url>        full anisearch.com anime URL (e.g. https://www.anisearch.com/anime/2227,one-piece/episodes)
    --output     optional JSONL output file path
"""

import argparse
import asyncio
import logging
import re
import sys
from typing import Any, cast

from common.utils.jsonl_utils import append_jsonl
from enrichment.sources.anisearch.anisearch_anime_models import (
    AniSearchEpisode,
    AniSearchEpisodesPage,
)
from enrichment.sources.anisearch.anisearch_mapper import episode_from_anisearch
from enrichment.sources.base.framework import (
    BaseCrawler,
    DockerTransport,
    NullRepository,
)
from enrichment.sources.base.utils import parse_iso_date, sanitize_output_path
from http_cache.config import get_cache_config
from http_cache.result_cache import cached_result

logger = logging.getLogger(__name__)

_CACHE_CONFIG = get_cache_config()
TTL_ANISEARCH = _CACHE_CONFIG.ttl_anisearch

_ANISEARCH_BASE_URL = "https://www.anisearch.com/anime/"

# ---------------------------------------------------------------------------
# XPath selectors
# ---------------------------------------------------------------------------

_XPATHS: dict[str, str] = {
    "episode_rows":       "//table[contains(@class,'episodes')]//tr[@data-episode='true']",
    "episode_number_raw": ".//th[@itemprop='episodeNumber']",
    "runtime":            ".//td[@data-title='Runtime']/div[@lang='ja']",
    "release_date":       ".//td[@data-title='Date of Original Release']/div[@lang='ja']",
    "title_en":           ".//td[@data-title='Title']/div[@lang='en']//span[@itemprop='name']",
    "title_ja":           ".//td[@data-title='Title']/div[@lang='ja']//span[@itemprop='name']",
    "title_de":           ".//td[@data-title='Title']/div[@lang='de']//span[@itemprop='name']",
    "title_fr":           ".//td[@data-title='Title']/div[@lang='fr']//span[@itemprop='name']",
    "title_it":           ".//td[@data-title='Title']/div[@lang='it']//span[@itemprop='name']",
}

# ---------------------------------------------------------------------------
# HTML extraction
# ---------------------------------------------------------------------------


def _extract_episodes_from_html(html_text: str) -> dict[str, Any] | None:
    """Parse an AniSearch /episodes page into a raw episodes dict.

    Each row dict uses the same field names as the old crawl4ai schema so that
    _parse_episode_row (and all its unit tests) stay unchanged.

    episode_number_raw is extracted via itertext() over the full <th> — this
    captures the episode number AND any nested <span>Filler</span>/<span>Recap</span>
    labels in one string (e.g. "279FillerRecap"), which the regex detection
    in _parse_episode_row relies on.
    """
    from lxml import etree

    try:
        parser = etree.HTMLParser()
        tree = etree.fromstring(html_text.encode(), parser)
        if tree is None:  # pragma: no cover
            return None  # pragma: no cover
    except Exception:  # pragma: no cover
        return None  # pragma: no cover

    rows = cast(list[Any], tree.xpath(_XPATHS["episode_rows"]))

    episodes = []
    for row in rows:
        def _text(xpath: str) -> str | None:
            els = cast(list[Any], row.xpath(xpath))
            if not els:
                return None
            return "".join(els[0].itertext()).strip() or None

        episodes.append({
            "episode_number_raw": _text(_XPATHS["episode_number_raw"]),
            "runtime":            _text(_XPATHS["runtime"]),
            "release_date":       _text(_XPATHS["release_date"]),
            "title_en":           _text(_XPATHS["title_en"]),
            "title_ja":           _text(_XPATHS["title_ja"]),
            "title_de":           _text(_XPATHS["title_de"]),
            "title_fr":           _text(_XPATHS["title_fr"]),
            "title_it":           _text(_XPATHS["title_it"]),
        })

    return {"episodes": episodes}


# ---------------------------------------------------------------------------
# Row parsing helpers (crawler responsibility — not mapper)
# ---------------------------------------------------------------------------


def _clean(val: str | None) -> str | None:
    v = (val or "").strip()
    return None if not v or v in ("?", " ") else v


def _parse_runtime_seconds(raw: str | None) -> int | None:
    """Parse "24 min" → 1440 seconds."""
    if not raw:
        return None
    m = re.search(r"(\d+)\s*min", raw, re.IGNORECASE)
    return int(m.group(1)) * 60 if m else None


def _split_title_ja(title_ja: str | None) -> tuple[str | None, str | None]:
    """Split "Romaji (Kanji)" → (title_romaji, title_japanese)."""
    if not title_ja:
        return None, None
    m = re.search(r"\(([^)]+)\)\s*$", title_ja)
    if m:
        return title_ja[: m.start()].strip() or None, m.group(1).strip()
    return title_ja.strip() or None, None


def _parse_episode_row(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Parse one raw XPath extraction dict into a cleaned episode field dict.

    Args:
        raw: Dict of raw XPath-extracted strings keyed by field name.

    Returns:
        Cleaned episode dict with keys: episode_number, is_filler, is_recap,
        duration, aired, title, title_romaji, title_japanese, titles.
        None if no episode number can be parsed (malformed row).

    Notes:
        is_filler and is_recap are independently detected — an episode can be
        neither, either, or both (e.g. a clip show recap that is also filler).
    """
    ep_raw = (raw.get("episode_number_raw") or "").strip()
    m = re.search(r"\d+", ep_raw)
    if not m:
        return None

    episode_number = int(m.group())
    is_filler = bool(re.search(r"filler", ep_raw, re.IGNORECASE))
    is_recap = bool(re.search(r"recap", ep_raw, re.IGNORECASE))
    duration = _parse_runtime_seconds(_clean(raw.get("runtime")))
    aired = parse_iso_date(_clean(raw.get("release_date")))

    # English title — strip dubbed prefixes separated by " | "
    title_en_raw = _clean(raw.get("title_en"))
    title = title_en_raw.split(" | ")[-1].strip() if title_en_raw else None

    title_romaji, title_japanese = _split_title_ja(_clean(raw.get("title_ja")))

    titles = {
        lang: v
        for lang, key in (("de", "title_de"), ("fr", "title_fr"), ("it", "title_it"))
        if (v := _clean(raw.get(key)))
    }

    return {
        "episode_number": episode_number,
        "is_filler": is_filler,
        "is_recap": is_recap,
        "duration": duration,
        "aired": aired,
        "title": title,
        "title_romaji": title_romaji,
        "title_japanese": title_japanese,
        "titles": titles,
    }


# ---------------------------------------------------------------------------
# Cached raw fetch
# ---------------------------------------------------------------------------


@cached_result(
    ttl=TTL_ANISEARCH,
    key_prefix="anisearch_episodes",
    dependencies=[_extract_episodes_from_html],
)
async def _fetch_anisearch_episode_data(url: str) -> dict[str, Any] | None:
    """Fetch the /episodes page and return the raw body dict. Cached by URL."""
    import zendriver as zd

    browser = await zd.start(headless=False)
    try:
        try:
            page = await browser.get(url)
            await page.wait_for(selector="table.episodes", timeout=15)
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(2)
            html_text = await page.get_content()
        except Exception as exc:
            logger.warning(f"navigation failed for {url}: {exc}")
            return None
    finally:
        try:
            await browser.stop()
        except Exception:
            pass

    if not html_text:
        logger.warning(f"No HTML from episodes page: {url}")
        return None

    return _extract_episodes_from_html(html_text)


# ---------------------------------------------------------------------------
# Crawler class
# ---------------------------------------------------------------------------


class AniSearchEpisodeCrawler(BaseCrawler[AniSearchEpisodesPage, list[dict[str, Any]]]):
    """Crawler for an AniSearch /episodes page.

    Returns a list of canonical episode dicts rather than a single dict —
    T_Canonical = list[dict] because one page contains all episodes.
    Repository persistence is not used here; callers handle file output.
    """

    def get_extraction_schema(self) -> dict[str, Any]:
        return {"xpaths": _XPATHS}

    def normalize_identifier(self, identifier: str) -> str:
        normalized = identifier.replace(
            "https://anisearch.com/", "https://www.anisearch.com/", 1
        )
        if not normalized.startswith(_ANISEARCH_BASE_URL):
            raise ValueError(f"Not an AniSearch anime URL: {identifier!r}")
        normalized = normalized.rstrip("/")
        if not normalized.endswith("/episodes"):
            normalized = f"{normalized}/episodes"
        return normalized

    async def fetch_raw_data(self, url: str) -> dict[str, Any] | None:
        return await _fetch_anisearch_episode_data(url)

    def build_source_model(
        self, processed_raw: dict[str, Any], url: str
    ) -> AniSearchEpisodesPage:
        episodes = []
        for raw_row in processed_raw.get("episodes") or []:
            parsed = _parse_episode_row(raw_row)
            if parsed is not None:
                episodes.append(AniSearchEpisode(**parsed, source=url))
        return AniSearchEpisodesPage(episodes=episodes, source=url)

    def map_to_canonical(
        self, source_model: AniSearchEpisodesPage
    ) -> list[dict[str, Any]]:
        return [episode_from_anisearch(ep) for ep in source_model.episodes]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def fetch_anisearch_episodes(
    url: str,
    output_path: str | None = None,
) -> list[dict[str, Any]] | None:
    """Fetch episode data for an AniSearch anime and optionally write to JSONL.

    Args:
        url: Full AniSearch anime URL (e.g. ``"https://www.anisearch.com/anime/2227,one-piece"``).
        output_path: If provided, append each episode as a JSON line to this path.

    Returns:
        List of canonical episode dicts or None if the page could not be fetched.
    """
    result = await AniSearchEpisodeCrawler(DockerTransport(), NullRepository()).crawl(url)
    if not result:
        return None

    if output_path:
        safe_path = sanitize_output_path(output_path)
        for episode in result:
            append_jsonl(safe_path, episode)
        logger.info(f"Episodes written to {safe_path}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


async def main() -> int:  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="Crawl episode data from anisearch.com."
    )
    parser.add_argument(
        "url",
        type=str,
        help="Full AniSearch anime URL (e.g. 'https://www.anisearch.com/anime/2227,one-piece')",
    )
    parser.add_argument("--output", type=str, default="anisearch_episodes.jsonl")
    args = parser.parse_args()

    try:
        data = await fetch_anisearch_episodes(args.url, output_path=args.output)
    except (ValueError, OSError):
        logger.exception("Failed to fetch AniSearch episode data")
        return 1
    except Exception:
        logger.exception("Unexpected error during episode fetch")
        return 1

    if not data:
        logger.warning("No episodes found.")
        return 0

    logger.info(f"Fetched {len(data)} episodes.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(asyncio.run(main()))
