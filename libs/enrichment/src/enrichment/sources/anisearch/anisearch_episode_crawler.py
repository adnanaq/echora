"""AniSearch episode crawler using the crawl4ai Docker REST transport.

Fetches the /episodes sub-page of an AniSearch anime URL and extracts per-episode
metadata (number, filler/recap flags, duration, air date, and titles in up to 5
languages: EN, JA, DE, FR, IT) using an XPath schema consistent with the rest of
the AniSearch crawler suite.

Usage:
    ./pants run libs/enrichment/src/enrichment/sources/anisearch/anisearch_episode_crawler.py -- <url> [--output PATH]

    <url>        full anisearch.com anime URL (e.g. https://www.anisearch.com/anime/2227,one-piece/episodes)
    --output     optional JSONL output file path
"""

import argparse
import asyncio
import json
import logging
import re
import sys
from typing import Any

from common.utils.jsonl_utils import append_jsonl
from enrichment.sources.anisearch.anisearch_anime_models import (
    AniSearchEpisode,
    AniSearchEpisodesPage,
)
from enrichment.sources.anisearch.anisearch_mapper import episode_from_anisearch
from enrichment.sources.base.crawl4ai_docker import crawl_single_url
from enrichment.sources.base.crawler_config import (
    get_docker_browser_config,
    get_docker_crawler_config,
)
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
# XPath extraction schema
# ---------------------------------------------------------------------------

_EPISODE_ROW_FIELDS: list[dict[str, Any]] = [
    {
        "name": "episode_number_raw",
        "selector": ".//th[@itemprop='episodeNumber']",
        "type": "text",
    },
    {
        "name": "runtime",
        "selector": ".//td[@data-title='Runtime']/div[@lang='ja']",
        "type": "text",
    },
    {
        "name": "release_date",
        "selector": ".//td[@data-title='Date of Original Release']/div[@lang='ja']",
        "type": "text",
    },
    {
        "name": "title_en",
        "selector": ".//td[@data-title='Title']/div[@lang='en']//span[@itemprop='name']",
        "type": "text",
    },
    {
        "name": "title_ja",
        "selector": ".//td[@data-title='Title']/div[@lang='ja']//span[@itemprop='name']",
        "type": "text",
    },
    {
        "name": "title_de",
        "selector": ".//td[@data-title='Title']/div[@lang='de']//span[@itemprop='name']",
        "type": "text",
    },
    {
        "name": "title_fr",
        "selector": ".//td[@data-title='Title']/div[@lang='fr']//span[@itemprop='name']",
        "type": "text",
    },
    {
        "name": "title_it",
        "selector": ".//td[@data-title='Title']/div[@lang='it']//span[@itemprop='name']",
        "type": "text",
    },
]


def _get_episode_schema() -> dict[str, Any]:
    """XPath extraction schema for an AniSearch /episodes page.

    Structure verified against https://www.anisearch.com/anime/2227,one-piece/episodes
    (April 2026):
    - Table: <table class="responsive-table episodes">
    - Row:   <tr data-episode="true" itemprop="episode">
    - Episode number: <th itemprop="episodeNumber"><b>N[<br><span>Filler|Recap</span>]</b></th>
    - Runtime: <td data-title="Runtime"><div lang="ja">24 min</div>...
    - Date:    <td data-title="Date of Original Release"><div lang="ja">DD. Mon YYYY</div>...
    - Title EN: <td data-title="Title"><div lang="en"><span itemprop="name">...</span></div>
    - Title JA: <td data-title="Title"><div lang="ja"><span itemprop="name">Romaji (Kanji)</span></div>
    - Title DE: <td data-title="Title"><div lang="de"><span itemprop="name">...</span></div>
    - Title FR: <td data-title="Title"><div lang="fr"><span itemprop="name">...</span></div>
    - Title IT: <td data-title="Title"><div lang="it"><span itemprop="name">...</span></div>
    - Future episodes: runtime/date cells are plain "?" text with no div children → None
    """
    return {
        "name": "AniSearchEpisodes",
        "baseSelector": "//body",
        "fields": [
            {
                "name": "episodes",
                "selector": "//table[contains(@class,'episodes')]//tr[@data-episode='true']",
                "type": "nested_list",
                "fields": _EPISODE_ROW_FIELDS,
            }
        ],
    }


# ---------------------------------------------------------------------------
# Row parsing helpers (crawler responsibility — not mapper)
# ---------------------------------------------------------------------------


def _clean(val: str | None) -> str | None:
    v = (val or "").strip()
    return None if not v or v in ("?", " ") else v


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
    dependencies=[
        _get_episode_schema,
        get_docker_browser_config,
        get_docker_crawler_config,
    ],
)
async def _fetch_anisearch_episodes_raw(url: str) -> dict[str, Any] | None:
    """Fetch the /episodes page and return the raw body dict. Cached by URL."""
    result = await crawl_single_url(
        url=url,
        browser_config=get_docker_browser_config(),
        crawler_config=get_docker_crawler_config(_get_episode_schema()),
    )
    if not result:
        logger.warning(f"No crawl result for {url}")
        return None

    status = result.get("status_code")
    if status == 404:
        logger.warning(f"Episodes page not found: {url}")
        return None
    if status and status != 200:
        logger.error(f"HTTP {status} fetching {url}")
        return None

    raw_list = json.loads(result.get("extracted_content") or "[]")
    return raw_list[0] if raw_list else None


# ---------------------------------------------------------------------------
# Crawler class
# ---------------------------------------------------------------------------


class AniSearchEpisodeCrawler(BaseCrawler[AniSearchEpisodesPage, list[dict[str, Any]]]):
    """Crawler for an AniSearch /episodes page.

    Returns a list of canonical episode dicts rather than a single dict —
    ``T_Canonical = list[dict]`` because one page contains all episodes.
    Repository persistence is not used here; callers handle file output.
    """

    def normalize_identifier(self, identifier: str) -> str:
        if not identifier.startswith(_ANISEARCH_BASE_URL):
            raise ValueError(f"Not an AniSearch anime URL: {identifier!r}")
        return identifier

    async def fetch_raw_data(self, url: str) -> dict[str, Any] | None:
        return await _fetch_anisearch_episodes_raw(url)

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
    anime_id: str,
    output_path: str | None = None,
) -> list[dict[str, Any]] | None:
    """Fetch episode data for an AniSearch anime and optionally write to JSONL.

    Args:
        anime_id: Full AniSearch anime URL (e.g. ``"https://www.anisearch.com/anime/2227,one-piece"``).
        output_path: If provided, append each episode as a JSON line to this path.

    Returns:
        List of canonical episode dicts or None if the page could not be fetched.
    """
    result = await AniSearchEpisodeCrawler(DockerTransport(), NullRepository()).crawl(
        anime_id
    )
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
    """CLI entry point for fetching episode data from AniSearch."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="Crawl episode data from anisearch.com."
    )
    parser.add_argument(
        "anime_id",
        type=str,
        help="Full AniSearch anime URL (e.g. 'https://www.anisearch.com/anime/2227,one-piece')",
    )
    parser.add_argument("--output", type=str, default="anisearch_episodes.jsonl")
    args = parser.parse_args()

    try:
        data = await fetch_anisearch_episodes(args.anime_id, output_path=args.output)
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
