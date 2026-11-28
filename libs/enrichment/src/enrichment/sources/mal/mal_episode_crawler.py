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
from typing import Any

from enrichment.sources.base.browser import browser_session
from enrichment.sources.base.framework import (
    BaseCrawler,
    FileRepository,
    NullRepository,
)
from enrichment.sources.base.page_readiness import wait_for_page
from enrichment.sources.mal.mal_base import MAL_DOMAIN
from enrichment.sources.mal.mal_episode_parser import (
    MAL_EPISODE_XPATHS,
    build_episode_from_raw,
    extract_episode_from_html,
)
from enrichment.sources.mal.mal_mapper import episode_from_mal
from enrichment.sources.mal.mal_models import (
    MalEpisode,
)
from http_cache.result_cache import cached_result

logger = logging.getLogger(__name__)


_INTER_REQUEST_DELAY = 1.5


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
        await wait_for_page(page, "h2.fs18", url)
        return await page.get_content(), page.url or url
    except Exception as exc:
        logger.warning(f"navigation failed for {url}: {exc}")
        return None


@cached_result(
    service="mal",
    key_prefix="mal_episode_detail",
)
async def _fetch_mal_episode_data(url: str) -> dict[str, Any] | None:
    """Fetch a MAL episode page by full URL and extract data. Cached by URL."""
    async with browser_session(headless=True, allowed_site=MAL_DOMAIN) as session:
        browser = session.browser
        result = await _fetch_episode_html(browser, url)

    if result is None:
        return None

    html, canonical_url = result
    raw = extract_episode_from_html(html)
    if raw is None:
        logger.warning(f"extraction failed for episode {url}")
        return None

    raw["_url"] = canonical_url
    return raw


class MalEpisodeCrawler(BaseCrawler[MalEpisode, dict[str, Any]]):
    """Crawler for MyAnimeList episode detail pages."""

    def get_extraction_schema(self) -> dict[str, Any]:
        return {"xpaths": MAL_EPISODE_XPATHS}

    def normalize_identifier(self, identifier: str) -> str:
        return identifier

    async def fetch_raw_data(self, url: str) -> dict[str, Any] | None:
        return await _fetch_mal_episode_data(url)

    def build_source_model(self, processed_raw: dict[str, Any], url: str) -> MalEpisode:
        m = re.search(r"/episode/(\d+)", url)
        episode_number = int(m.group(1)) if m else 0
        return build_episode_from_raw(processed_raw, episode_number, url)

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
        return episode_from_mal(build_episode_from_raw(raw, int(m.group(1)), url))

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

    async with browser_session(headless=True, allowed_site=MAL_DOMAIN) as session:
        browser = session.browser
        for i, idx in enumerate(missing_indices):
            url = urls[idx]
            result = await _fetch_episode_html(browser, url)
            if result is None:
                episodes[idx] = None
                continue

            html, canonical_url = result
            raw = extract_episode_from_html(html)
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
                build_episode_from_raw(raw, int(m.group(1)), canonical_url)
            )
            episodes[idx] = canonical
            repo.save(canonical)

            if i < len(missing_indices) - 1:
                await asyncio.sleep(_INTER_REQUEST_DELAY)

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
