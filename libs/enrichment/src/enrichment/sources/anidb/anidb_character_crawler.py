"""AniDB Character Crawler — zendriver + lxml XPath.

Fetches AniDB character web pages using a persistent zendriver Chrome session.
CF Turnstile is solved at most once per batch — the session remains trusted
across subsequent requests with a 2.5 s inter-request delay.

Public functions:
    fetch_anidb_characters(char_ids) — async generator yielding (char_id, page)
    fetch_anidb_character(char_id)   -> AniDBCharacterPage | None

Usage:
    from enrichment.sources.anidb.anidb_character_crawler import (
        fetch_anidb_character,
        fetch_anidb_characters,
    )
    page = await fetch_anidb_character(474)
    async for char_id, page in fetch_anidb_characters([474, 475, 476]):
        ...
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from collections.abc import AsyncGenerator
from typing import Any, cast

from enrichment.sources.anidb.anidb_mapper import character_from_anidb
from enrichment.sources.anidb.anidb_models import AniDBCharacter, AniDBCharacterPage
from enrichment.sources.base.utils import sanitize_output_path
from http_cache.config import get_cache_config
from http_cache.result_cache import cached_result

logger = logging.getLogger(__name__)

_TTL_ANIDB = get_cache_config().ttl_anidb

_BASE_URL = "https://anidb.net/character"
_CF_MARKERS = (
    "Just a moment",
    "cf-browser-verification",
    "cf-challenge",
    "Attention Required",
    "Please Unban Me",  # AniDB antileech page
)
_INTER_REQUEST_DELAY = 2.5  # seconds; keeps session trusted and avoids CF re-trigger

# ---------------------------------------------------------------------------
# XPath selectors — anchored on structural attributes (itemprop, id, class)
# rather than CSS class names that change frequently.
# ---------------------------------------------------------------------------
_XPATHS: dict[str, str] = {
    # Description block
    "description": "//div[contains(@class,'desc')][@itemprop='description']",
    # Tab 1: primary names and attributes
    "name_main": "//tr[contains(@class,'mainname')]//span[@itemprop='name']/text()",
    "name_kanji": (
        "//*[@id='tab_1_pane']"
        "//tr[contains(@class,'official') and contains(@class,'verified') and contains(@class,'yes')]"
        "//label[@itemprop='alternateName']/text()"
    ),
    "gender": "//span[@itemprop='gender']/text()",
    # Tag categories
    "abilities": (
        "//*[@id='tab_1_pane']"
        "//tr[contains(@class,'abilities') and not(contains(@class,'supernatural'))]"
        "//span[contains(@class,'tagname')]/text()"
    ),
    "supernatural_abilities": (
        "//*[@id='tab_1_pane']//tr[contains(@class,'supernatural')]"
        "//span[contains(@class,'tagname')]/text()"
    ),
    "looks": (
        "//*[@id='tab_1_pane']//tr[contains(@class,'looks')]"
        "//span[contains(@class,'tagname')]/text()"
    ),
    "personality": (
        "//*[@id='tab_1_pane']//tr[contains(@class,'personality')]"
        "//span[contains(@class,'tagname')]/text()"
    ),
    "role": (
        "//*[@id='tab_1_pane']//tr[contains(@class,'role')]"
        "//span[contains(@class,'tagname')]/text()"
    ),
    # Anime appearances table
    "animeography_rows": "//table[.//th[contains(@class,'anime')]]//tr[td]",
    # Tab 2: alternate names
    "official_names": (
        "//*[@id='tab_2_pane']//tr[contains(@class,'official')]"
        "//label[@itemprop='alternateName']/text()"
    ),
    "nicknames": (
        "//*[@id='tab_2_pane']//tr[contains(@class,'nick')]"
        "//td[contains(@class,'value')]/text()"
    ),
}


# ---------------------------------------------------------------------------
# HTML extraction
# ---------------------------------------------------------------------------


def _extract_from_html(html: str) -> AniDBCharacterPage | None:
    """Extract character fields from AniDB character page HTML via XPath.

    Returns None if the HTML cannot be parsed or contains no character data.
    """
    from lxml import etree

    try:
        parser = etree.HTMLParser()
        tree = etree.fromstring(html.encode(), parser)
    except Exception:
        return None

    def _texts(key: str) -> list[str]:
        return [
            t.strip() for t in cast(list[str], tree.xpath(_XPATHS[key])) if t.strip()
        ]

    def _first(key: str) -> str | None:
        return next(iter(_texts(key)), None)

    def _animeography() -> list[dict[str, str]]:
        entries = []
        for row in cast(list[Any], tree.xpath(_XPATHS["animeography_rows"])):
            title_els = cast(
                list[Any],
                row.xpath(
                    './/td[contains(@class,"name") and contains(@class,"anime")]//a'
                ),
            )
            role_parts = cast(
                list[str], row.xpath('.//td[contains(@class,"type")]//text()')
            )
            if not title_els:
                continue
            title = " ".join(title_els[0].itertext()).strip()
            href = title_els[0].get("href", "")
            role = role_parts[0].strip() if role_parts else ""
            url = f"https://anidb.net{href}" if href.startswith("/") else href
            if title:
                entries.append({"title": title, "role": role, "url": url})
        return entries

    def _description() -> str | None:
        nodes = cast(list[Any], tree.xpath(_XPATHS["description"]))
        if not nodes:
            return None
        raw = " ".join(nodes[0].itertext()).split()
        return " ".join(raw) or None

    return AniDBCharacterPage(
        name_main=_first("name_main"),
        name_kanji=_first("name_kanji"),
        description=_description(),
        gender=_first("gender"),
        abilities=_texts("abilities"),
        supernatural_abilities=_texts("supernatural_abilities"),
        looks=_texts("looks"),
        personality=_texts("personality"),
        role=_texts("role"),
        official_names=_texts("official_names"),
        nicknames=_texts("nicknames"),
        animeography=_animeography(),
    )


# ---------------------------------------------------------------------------
# CF bypass helpers
# ---------------------------------------------------------------------------


def _is_cf_blocked(html: str) -> bool:
    return any(m in html for m in _CF_MARKERS)


def _has_character_data(html: str) -> bool:
    return "tab_1_pane" in html or 'itemprop="name"' in html


async def _solve_cf(page: Any) -> bool:
    """Solve CF Turnstile on current page.

    AniDB's block page embeds the CF Turnstile checkbox inside their own HTML
    form with a 'Please Unban Me' submit button. Two steps:
    1. verify_cf — clicks the Turnstile checkbox and waits for CF to validate
    2. Click 'Please Unban Me' — submits the form with the CF token to AniDB
    """
    from zendriver.core.cloudflare import verify_cf

    cf_solved = False
    try:
        await verify_cf(page, click_delay=3.0, timeout=15)
        cf_solved = True
    except Exception:  # noqa: S110
        pass

    if cf_solved:
        try:
            btn = await page.find("Please Unban Me", best_match=True)
            if btn:
                await btn.click()
        except Exception:  # noqa: S110
            pass

    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        await asyncio.sleep(1)
        try:
            html = await page.get_content()
        except Exception:  # noqa: S112
            continue
        if not _is_cf_blocked(html):
            return True
    return False


# ---------------------------------------------------------------------------
# Browser fetch helpers
# ---------------------------------------------------------------------------


async def _fetch_page_html(browser: Any, url: str) -> tuple[str | None, Any]:
    """Navigate to url, return (html, page). Both None on browser crash."""
    try:
        page = await browser.get(url)
        await asyncio.sleep(2)
        return await page.get_content(), page
    except (RuntimeError, StopIteration) as exc:
        logger.warning(f"browser crash on {url}: {exc}")
        return None, None


# ---------------------------------------------------------------------------
# Cache stub — keyed by char_id, schema-hashed from _extract_from_html so
# any change to XPath extraction logic invalidates existing cache entries.
# Never called directly; only used for cache_batch_get / cache_batch_set.
# ---------------------------------------------------------------------------


@cached_result(
    ttl=_TTL_ANIDB,
    key_prefix="anidb_character",
    dependencies=[_extract_from_html],
)
async def _anidb_character_cache(
    char_id: int,
) -> dict[str, Any] | None:  # pragma: no cover
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def fetch_anidb_characters(
    char_ids: list[int],
) -> AsyncGenerator[tuple[int, AniDBCharacterPage | None]]:
    """Fetch AniDB character pages via zendriver, yielding results as each page arrives.

    Cache is checked upfront for all IDs in one Redis round-trip. Hits are
    yielded immediately. Misses are fetched in a single shared Chrome session —
    CF Turnstile is solved at most once. Each result is cached immediately after
    fetch so cancellation doesn't lose progress.

    Args:
        char_ids: AniDB numeric character IDs to fetch.

    Yields:
        (char_id, AniDBCharacterPage) on success, (char_id, None) on failure.
    """
    if not char_ids:
        return

    import zendriver as zd

    cached_values, missing_indices = await _anidb_character_cache.cache_batch_get(
        char_ids
    )  # type: ignore[attr-defined]
    missing_set = set(missing_indices)

    browser: Any = None
    succeeded = 0

    try:
        for i, char_id in enumerate(char_ids):
            if i not in missing_set:
                val = cached_values[i]
                yield char_id, AniDBCharacterPage.model_validate(val) if val else None
                continue

            # Cache miss — launch browser lazily on first miss
            if browser is None:
                browser = await zd.start(headless=False)

            url = f"{_BASE_URL}/{char_id}"
            html, current_page = await _fetch_page_html(browser, url)

            if html is None:
                logger.warning(f"browser crashed — restarting for char {char_id}")
                try:
                    await browser.stop()
                except Exception:  # noqa: S110
                    pass
                browser = await zd.start(headless=False)
                await asyncio.sleep(2)
                html, current_page = await _fetch_page_html(browser, url)
                if html is None:
                    logger.error(f"browser crashed again on char {char_id} — skipping")
                    yield char_id, None
                    continue

            if _is_cf_blocked(html):
                logger.info(f"CF block on char {char_id} — solving")
                if not await _solve_cf(current_page):
                    logger.error(f"CF did not clear for char {char_id}")
                    yield char_id, None
                    continue
                await current_page
                try:
                    html = await current_page.get_content()
                except Exception:
                    yield char_id, None
                    continue

            if not _has_character_data(html):
                logger.warning(f"no character data for char {char_id} (deleted/invalid)")
                yield char_id, None
            else:
                page = _extract_from_html(html)
                if page is not None:
                    succeeded += 1
                page_dict = page.model_dump(mode="json") if page is not None else None
                # Cache immediately so cancellation doesn't lose progress
                await _anidb_character_cache.cache_batch_set([char_id], [page_dict])  # type: ignore[attr-defined]
                yield char_id, page

            # Delay only between browser requests
            if any(j in missing_set for j in range(i + 1, len(char_ids))):
                await asyncio.sleep(_INTER_REQUEST_DELAY)

    finally:
        if browser is not None:
            try:
                await browser.stop()
            except Exception:  # noqa: S110
                pass
        cache_hits = len(char_ids) - len(missing_set)
        logger.info(f"anidb character fetch: {succeeded}/{len(missing_set)} succeeded, {cache_hits} cache hits")


async def fetch_anidb_character(char_id: int) -> AniDBCharacterPage | None:
    """Fetch a single AniDB character page. Convenience wrapper."""
    async for _, page in fetch_anidb_characters([char_id]):
        return page
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


async def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(description="Fetch character page data from AniDB")
    parser.add_argument(
        "character_id", type=int, help="AniDB character ID (e.g. 491 for Brook)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="anidb_character_page.json",
        help="Output file path (default: anidb_character_page.json)",
    )
    args = parser.parse_args()

    page = await fetch_anidb_character(args.character_id)
    if page is None:
        logger.error(f"No data for character {args.character_id}")
        return 1

    char = AniDBCharacter(id=args.character_id, name=page.name_main)
    canonical = character_from_anidb(char, page_data=page)
    safe_path = sanitize_output_path(args.output)
    with open(safe_path, "w", encoding="utf-8") as f:
        json.dump(canonical, f, ensure_ascii=False, indent=2)
    logger.info(f"Written to {safe_path}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
