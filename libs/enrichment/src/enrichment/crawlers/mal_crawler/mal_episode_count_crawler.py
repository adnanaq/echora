"""MAL Episode Count Crawler.

Fetches the aired episode count for any anime:
    fetch_mal_episode_count(anime_url)  — reads episode list page → int

MAL renders a span "(12/12)" or "(1,155/Unknown)" next to the Episodes heading.
The first number (before the slash) is always the current aired count.
This works for all anime regardless of episode count or airing status.
"""

import json
import logging
import re
from typing import Any

from enrichment.crawlers.crawl4ai_docker import crawl_single_url
from enrichment.crawlers.mal_crawler.mal_base import (
    get_mal_docker_browser_config,
    get_mal_docker_crawler_config,
    get_mal_scraping_limiter,
)
from http_cache.config import get_cache_config
from http_cache.result_cache import cached_result

logger = logging.getLogger(__name__)

_CACHE_CONFIG = get_cache_config()
TTL_MAL = _CACHE_CONFIG.ttl_jikan

_limiter = get_mal_scraping_limiter()


def _get_episode_count_schema() -> dict[str, Any]:
    """XPath schema to extract the episode count span from the episode list page.

    MAL renders a span immediately after the Episodes heading:
      <h2 class="h2_overwrite">Episodes</h2>
      <span class="di-ib pl4 fw-n fs10">(12/12)</span>       ← finished anime
      <span class="di-ib pl4 fw-n fs10">(1,155/Unknown)</span> ← ongoing anime

    The first number (before the slash) is the current aired episode count.
    """
    return {
        "name": "EpisodeCountPage",
        "baseSelector": "//h2[@class='h2_overwrite'][text()='Episodes']",
        "fields": [
            {
                "name": "episode_counter",
                "selector": "./following-sibling::span[1]",
                "type": "text",
            }
        ],
    }


@cached_result(
    ttl=TTL_MAL,
    key_prefix="mal_episode_count",
    dependencies=[_get_episode_count_schema],
)
async def _fetch_episode_count_data(url: str) -> str | None:
    """Fetch the episode list page and return the episode counter span text.

    Returns the raw counter string (e.g. "(12/12)" or "(1,155/Unknown)") or None on failure.
    """
    await _limiter.acquire()
    result = await crawl_single_url(
        url=url,
        browser_config=get_mal_docker_browser_config(),
        crawler_config=get_mal_docker_crawler_config(
            _get_episode_count_schema(),
            wait_until="networkidle",
            delay=1.0,
            magic=False,
        ),
    )
    if not result:
        logger.error(f"No result for episode list page {url}")
        return None

    status = result.get("status_code")
    if status and status != 200:
        logger.error(f"HTTP {status} for episode list page {url}")
        return None

    data = json.loads(result.get("extracted_content") or "[]")
    return data[0].get("episode_counter") if data else None


async def fetch_mal_episode_count(anime_url: str) -> int:
    """Return the current aired episode count for an anime.

    Reads the episode counter span on the episode list page, e.g.:
      "(12/12)"        → 12   (finished anime)
      "(1,155/Unknown" → 1155 (ongoing anime)

    Returns 0 on failure or when no episodes are listed.

    Args:
        anime_url: Full MAL anime slug URL
            (e.g. https://myanimelist.net/anime/21/One_Piece).
    """
    counter = await _fetch_episode_count_data(f"{anime_url}/episode")
    if not counter:
        return 0
    m = re.search(r"\(([0-9,]+)/", counter)
    return int(m.group(1).replace(",", "")) if m else 0
