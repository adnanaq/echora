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
from typing import Any

from enrichment.sources.anidb.anidb_mapper import character_from_anidb
from enrichment.sources.anidb.anidb_models import AniDBCharacterPage, AniDBCharacter
from enrichment.sources.base.utils import sanitize_output_path

logger = logging.getLogger(__name__)

_BASE_URL = "https://anidb.net/character"
_CF_MARKERS = ("Just a moment", "cf-browser-verification", "cf-challenge", "Attention Required")
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
        logger.debug("lxml: HTML parse failed")
        return None

    def _texts(key: str) -> list[str]:
        return [t.strip() for t in tree.xpath(_XPATHS[key]) if t.strip()]

    def _first(key: str) -> str | None:
        values = _texts(key)
        return values[0] if values else None

    def _animeography() -> list[dict[str, str]]:
        entries = []
        for row in tree.xpath(_XPATHS["animeography_rows"]):
            title_els = row.xpath('.//td[contains(@class,"name") and contains(@class,"anime")]//a')
            role_parts = row.xpath('.//td[contains(@class,"type")]//text()')
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
        nodes = tree.xpath(_XPATHS["description"])
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
    from zendriver.core.cloudflare import cf_is_interactive_challenge_present, verify_cf

    if await cf_is_interactive_challenge_present(page, timeout=10):
        try:
            await verify_cf(page, click_delay=1.0, timeout=20)
        except Exception as exc:
            logger.debug("verify_cf raised (may be auto-resolved): %s", exc)

        await asyncio.sleep(2)

        try:
            btn = await page.find("Please Unban Me", best_match=True)
            if btn:
                await btn.click()
                await asyncio.sleep(3)
        except Exception:
            pass

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        await asyncio.sleep(1)
        try:
            html = await page.get_content()
        except Exception:
            continue
        if not _is_cf_blocked(html):
            return True
    return False


# ---------------------------------------------------------------------------
# Browser fetch helpers
# ---------------------------------------------------------------------------


async def _fetch_page_html(browser: Any, url: str) -> tuple[str | None, bool]:
    """Navigate to url, return (html, crashed).

    crashed=True means the browser tab/process died and the caller should restart.
    """
    try:
        page = await browser.get(url)
        await asyncio.sleep(2)
        return await page.get_content(), False
    except (RuntimeError, StopIteration) as exc:
        logger.warning("browser crash on %s: %s", url, exc)
        return None, True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def fetch_anidb_characters(
    char_ids: list[int],
) -> AsyncGenerator[tuple[int, AniDBCharacterPage | None], None]:
    """Fetch AniDB character pages via zendriver, yielding results as each page arrives.

    Uses a single Chrome session across all requests. CF Turnstile is solved at
    most once per batch — subsequent requests stay trusted at a 2.5 s delay.
    Yields (char_id, page) immediately after each page is fetched so callers
    can map and write to JSONL without waiting for the full batch to complete.

    Args:
        char_ids: AniDB numeric character IDs to fetch.

    Yields:
        (char_id, AniDBCharacterPage) on success, (char_id, None) on failure.
    """
    if not char_ids:
        return

    import zendriver as zd

    succeeded = 0
    browser = await zd.start(headless=False)

    try:
        for idx, char_id in enumerate(char_ids):
            url = f"{_BASE_URL}/{char_id}"
            logger.debug("fetching anidb char %d (%d/%d)", char_id, idx + 1, len(char_ids))

            html, crashed = await _fetch_page_html(browser, url)

            if crashed:
                logger.warning("browser crashed — restarting for char %d", char_id)
                try:
                    await browser.stop()
                except Exception:
                    pass
                browser = await zd.start(headless=False)
                await asyncio.sleep(2)
                html, crashed = await _fetch_page_html(browser, url)
                if crashed:
                    logger.error("browser crashed again on char %d — skipping", char_id)
                    yield char_id, None
                    continue

            assert html is not None

            if _is_cf_blocked(html):
                logger.info("CF block on char %d — solving", char_id)
                try:
                    current_page = await browser.get(url)
                except Exception:
                    yield char_id, None
                    continue
                if not await _solve_cf(current_page):
                    logger.error("CF did not clear for char %d", char_id)
                    yield char_id, None
                    continue
                try:
                    html = await current_page.get_content()
                except Exception:
                    yield char_id, None
                    continue

            if not _has_character_data(html):
                logger.warning("no character data for char %d (deleted/invalid)", char_id)
                yield char_id, None
            else:
                page = _extract_from_html(html)
                if page is not None:
                    succeeded += 1
                yield char_id, page

            if idx < len(char_ids) - 1:
                await asyncio.sleep(_INTER_REQUEST_DELAY)

    finally:
        try:
            await browser.stop()
        except Exception:
            pass
        logger.info("anidb character page fetch: %d/%d succeeded", succeeded, len(char_ids))


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
    parser.add_argument("character_id", type=int, help="AniDB character ID (e.g. 491 for Brook)")
    parser.add_argument(
        "--output",
        type=str,
        default="anidb_character_page.json",
        help="Output file path (default: anidb_character_page.json)",
    )
    args = parser.parse_args()

    page = await fetch_anidb_character(args.character_id)
    if page is None:
        logger.error("No data for character %d", args.character_id)
        return 1

    char = AniDBCharacter(id=args.character_id, name=page.name_main)
    canonical = character_from_anidb(char, page_data=page)
    safe_path = sanitize_output_path(args.output)
    with open(safe_path, "w", encoding="utf-8") as f:
        json.dump(canonical, f, ensure_ascii=False, indent=2)
    logger.info("Written to %s", safe_path)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
