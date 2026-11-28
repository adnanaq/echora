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
from contextlib import AsyncExitStack, aclosing
from typing import Any, cast
from urllib.parse import urljoin

from enrichment.sources.anidb.anidb_mapper import character_from_anidb
from enrichment.sources.anidb.anidb_models import AniDBCharacter, AniDBCharacterPage
from enrichment.sources.base.browser import BrowserSession, browser_session
from enrichment.sources.base.cloudflare_challenge import (
    CloudflareChallengeError,
    PageRead,
    is_cloudflare_challenge,
    wait_through_challenge,
)
from enrichment.sources.base.cloudflare_clearance import save_clearance
from enrichment.sources.base.page_fetcher import parse_html_document
from enrichment.sources.base.utils import sanitize_output_path
from http_cache.result_cache import cached_result

logger = logging.getLogger(__name__)


_BASE_URL = "https://anidb.net/character"
_SITE = "AniDB"
_CLEARANCE_SITE = "anidb.net"
_CHARACTER_SELECTOR = "#tab_1_pane"
_PAGE_TIMEOUT_SECONDS = 15.0
# AniDB's antileech block page is identified by its <title>, which is present
# from the initial HTML — unlike the embedded CF Turnstile and the "Please Unban
# Me" button, which are injected asynchronously and are absent from an early
# get_content() snapshot. Matching the title (not bare "AntiLeech", which also
# appears as a footer link on normal pages) avoids both false negatives and
# false positives.
_ANTILEECH_TITLE = "<title>AniDB AntiLeech"
_UNBAN_BUTTON_TEXT = "Please Unban Me"
_INTER_REQUEST_DELAY = 2.5  # seconds; keeps session trusted and avoids CF re-trigger


def _class_token_xpath(token: str) -> str:
    """Build a token-boundary XPath predicate matching one CSS class name.

    Args:
        token: Exact CSS class token to match.

    Returns:
        XPath predicate expression matching ``token`` as a whole class name.
    """
    return f"contains(concat(' ', normalize-space(@class), ' '), ' {token} ')"


# ---------------------------------------------------------------------------
# XPath selectors — anchored on structural attributes (itemprop, id, class)
# rather than CSS class names that change frequently.
# ---------------------------------------------------------------------------
_XPATHS: dict[str, str] = {
    # Description block
    "description": f"//div[{_class_token_xpath('desc')}][@itemprop='description']",
    # Tab 1: primary names and attributes
    "name_main": (
        f"//tr[{_class_token_xpath('mainname')}]//span[@itemprop='name']/text()"
    ),
    "name_kanji": (
        "//*[@id='tab_1_pane']"
        f"//tr[{_class_token_xpath('official')} and {_class_token_xpath('verified')} and {_class_token_xpath('yes')}]"
        "//label[@itemprop='alternateName']/text()"
    ),
    "gender": "//span[@itemprop='gender']/text()",
    "picture": (
        f"//div[{_class_token_xpath('info')}]"
        f"//div[{_class_token_xpath('image')}]//img/@src"
    ),
    "date_of_birth": f"//*[@id='tab_1_pane']//tr[{_class_token_xpath('birthdate')}]//td",
    "age": (
        "//*[@id='tab_1_pane']"
        f"//tr[{_class_token_xpath('age')} and not({_class_token_xpath('range')})]//td"
    ),
    "blood_type": f"//*[@id='tab_1_pane']//tr[{_class_token_xpath('bloodtype')}]//td",
    "height": f"//*[@id='tab_1_pane']//tr[{_class_token_xpath('height')}]//td",
    "seiyuu_rows": f"(//table[{_class_token_xpath('seiyuulist')}])[1]//tr[td]",
    # Tag categories
    "abilities": (
        "//*[@id='tab_1_pane']"
        f"//tr[{_class_token_xpath('abilities')} and not({_class_token_xpath('supernatural')})]"
        f"//span[{_class_token_xpath('tagname')}]/text()"
    ),
    "supernatural_abilities": (
        f"//*[@id='tab_1_pane']//tr[{_class_token_xpath('supernatural')}]"
        f"//span[{_class_token_xpath('tagname')}]/text()"
    ),
    "looks": (
        f"//*[@id='tab_1_pane']//tr[{_class_token_xpath('looks')}]"
        f"//span[{_class_token_xpath('tagname')}]/text()"
    ),
    "personality": (
        f"//*[@id='tab_1_pane']//tr[{_class_token_xpath('personality')}]"
        f"//span[{_class_token_xpath('tagname')}]/text()"
    ),
    "role": (
        f"//*[@id='tab_1_pane']//tr[{_class_token_xpath('role')}]"
        f"//span[{_class_token_xpath('tagname')}]/text()"
    ),
    # Anime appearances table: the first animelist table is the character's own;
    # later ones belong to related characters.
    "animeography_rows": (f"(//table[{_class_token_xpath('animelist')}])[1]//tr[td]"),
    # Tab 2: alternate names
    "official_names": (
        f"//*[@id='tab_2_pane']//tr[{_class_token_xpath('official')}]"
        "//label[@itemprop='alternateName']/text()"
    ),
    "nicknames": (
        f"//*[@id='tab_2_pane']//tr[{_class_token_xpath('nick')}]"
        f"//td[{_class_token_xpath('value')}]/text()"
    ),
}


# ---------------------------------------------------------------------------
# HTML extraction
# ---------------------------------------------------------------------------


def _resolve_anidb_url(href: str) -> str:
    """Resolve an AniDB anchor ``href`` against ``https://anidb.net/``.

    Args:
        href: Raw ``href`` attribute value from an HTML anchor element.

    Returns:
        Absolute AniDB URL string, or empty string when ``href`` is blank.
    """
    cleaned = href.strip()
    return urljoin("https://anidb.net/", cleaned) if cleaned else ""


def _extract_text_list(tree: Any, xpath_key: str) -> list[str]:
    """Extract non-empty stripped strings matching an XPath text selector.

    Args:
        tree: Parsed lxml HTML element tree.
        xpath_key: Key into ``_XPATHS``.

    Returns:
        List of non-empty stripped text values.
    """
    return [
        text.strip()
        for text in cast(list[str], tree.xpath(_XPATHS[xpath_key]))
        if text.strip()
    ]


def _extract_profile_field(tree: Any, xpath_key: str) -> str | None:
    """Extract the full text of the first profile table cell matching ``xpath_key``.

    Nested elements such as ``<span>`` and ``<time>`` inside the ``<td>`` are
    included by joining ``itertext()`` and collapsing whitespace.

    Args:
        tree: Parsed lxml HTML element tree.
        xpath_key: Key into ``_XPATHS`` selecting ``<td>`` elements.

    Returns:
        Collapsed cell text, or ``None`` when no matching non-empty cell exists.
    """
    for cell in cast(list[Any], tree.xpath(_XPATHS[xpath_key])):
        text = " ".join(" ".join(cell.itertext()).split())
        if text:
            return text
    return None


def _extract_character_description(tree: Any) -> str | None:
    """Extract normalized character description text from the parsed HTML tree.

    Args:
        tree: Parsed lxml HTML element tree.

    Returns:
        Whitespace-collapsed description string, or ``None`` when absent.
    """
    nodes = cast(list[Any], tree.xpath(_XPATHS["description"]))
    if not nodes:
        return None
    return " ".join(" ".join(nodes[0].itertext()).split()) or None


def _extract_animeography_entries(tree: Any) -> list[dict[str, str]]:
    """Extract anime appearance entries from the character's primary anime table.

    Args:
        tree: Parsed lxml HTML element tree.

    Returns:
        List of dicts carrying ``title``, ``role``, and absolute ``url``.
    """
    rows = cast(list[Any], tree.xpath(_XPATHS["animeography_rows"]))
    entries: list[dict[str, str]] = []
    for row in rows:
        title_elements = cast(
            list[Any],
            row.xpath(
                f".//td[{_class_token_xpath('name')} and {_class_token_xpath('anime')}]//a"
            ),
        )
        if not title_elements:
            continue
        title = " ".join(title_elements[0].itertext()).strip()
        if not title:
            continue
        role_cells = cast(
            list[Any],
            row.xpath(
                f".//td[{_class_token_xpath('type')} and {_class_token_xpath('relation')}]"
            ),
        ) or cast(list[Any], row.xpath(f".//td[{_class_token_xpath('type')}]"))
        role = (
            " ".join(" ".join(role_cells[0].itertext()).split()) if role_cells else ""
        )
        entries.append(
            {
                "title": title,
                "role": role,
                "url": _resolve_anidb_url(title_elements[0].get("href", "")),
            }
        )
    return entries


def _extract_seiyuu_entries(tree: Any) -> list[dict[str, str]]:
    """Extract voice actor entries from the character's primary seiyuu table.

    Args:
        tree: Parsed lxml HTML element tree.

    Returns:
        List of dicts carrying ``name``, absolute ``url``, and optional ``anime_url``.
    """
    rows = cast(list[Any], tree.xpath(_XPATHS["seiyuu_rows"]))
    entries: list[dict[str, str]] = []
    for row in rows:
        creator_elements = cast(
            list[Any],
            row.xpath(f".//td[{_class_token_xpath('creator')}]//a"),
        )
        if not creator_elements:
            continue
        name = " ".join(creator_elements[0].itertext()).strip()
        if not name:
            continue
        creator_url = _resolve_anidb_url(creator_elements[0].get("href", ""))
        anime_elements = cast(
            list[Any],
            row.xpath(f".//td[{_class_token_xpath('anime')}]//a"),
        )
        anime_url = (
            _resolve_anidb_url(anime_elements[0].get("href", ""))
            if anime_elements
            else ""
        )
        entry: dict[str, str] = {"name": name, "url": creator_url}
        if anime_url:
            entry["anime_url"] = anime_url
        entries.append(entry)
    return entries


def _extract_from_html(html: str) -> AniDBCharacterPage | None:
    """Extract character fields from AniDB character page HTML via XPath.

    Args:
        html: Raw HTML markup of an AniDB character page.

    Returns:
        Parsed ``AniDBCharacterPage``, or ``None`` if HTML parsing fails.
    """
    try:
        tree = parse_html_document(html)
    except Exception:
        return None

    return AniDBCharacterPage(
        name_main=next(iter(_extract_text_list(tree, "name_main")), None),
        name_kanji=next(iter(_extract_text_list(tree, "name_kanji")), None),
        description=_extract_character_description(tree),
        picture=next(iter(_extract_text_list(tree, "picture")), None),
        gender=next(iter(_extract_text_list(tree, "gender")), None),
        date_of_birth=_extract_profile_field(tree, "date_of_birth"),
        age=_extract_profile_field(tree, "age"),
        blood_type=_extract_profile_field(tree, "blood_type"),
        height=_extract_profile_field(tree, "height"),
        abilities=_extract_text_list(tree, "abilities"),
        supernatural_abilities=_extract_text_list(tree, "supernatural_abilities"),
        looks=_extract_text_list(tree, "looks"),
        personality=_extract_text_list(tree, "personality"),
        role=_extract_text_list(tree, "role"),
        official_names=_extract_text_list(tree, "official_names"),
        nicknames=_extract_text_list(tree, "nicknames"),
        animeography=_extract_animeography_entries(tree),
        seiyuu=_extract_seiyuu_entries(tree),
    )


# ---------------------------------------------------------------------------
# CF bypass helpers
# ---------------------------------------------------------------------------


def _is_antileech(html: str) -> bool:
    return _ANTILEECH_TITLE in html


def _is_blocked(html: str) -> bool:
    """True if the page is any kind of block: CF interstitial or AniDB antileech."""
    return is_cloudflare_challenge(html) or _is_antileech(html)


def _has_character_data(html: str) -> bool:
    # Real character pages render their content under #tab_1_pane. The antileech
    # page also carries itemprop="name" in its header, so that marker must NOT be
    # used here — it caused antileech pages to be mistaken for character pages.
    return "tab_1_pane" in html


async def _solve_antileech(page: Any) -> bool:
    """Clear AniDB's AntiLeech block on the current page.

    Cloudflare interstitials are waited out by ``wait_through_challenge``; this
    runs only for the AntiLeech page, which needs its button. AntiLeech was not
    seen while this flow was reviewed, so its steps are kept as they were:
    verify_cf (which raises when the Turnstile auto-passes and has no checkbox),
    a short wait for the token, the 'Please Unban Me' button if present, and a
    wait for the reload. Returns True once the page is no longer blocked.
    """
    from zendriver.core.cloudflare import verify_cf

    try:
        await verify_cf(page, click_delay=3.0, timeout=15)
    except Exception:  # noqa: S110  (antileech Turnstile auto-passes, no checkbox)
        pass

    # Let the Turnstile finish and populate its token before submitting the form.
    await asyncio.sleep(5)

    try:
        btn = await page.find(_UNBAN_BUTTON_TEXT, best_match=True)
        if btn:
            logger.info("antileech — clicking 'Please Unban Me'")
            await btn.click()
            await asyncio.sleep(5)  # wait for the unban submit + reload
    except Exception:  # noqa: S110
        pass

    # The unban submit reloads the page, so get_content can raise while the
    # navigation is in flight. One failed read is not a failed solve — poll
    # until the block clears or the deadline passes.
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            html = await page.get_content()
        except Exception:
            await asyncio.sleep(1)
            continue
        if not _is_blocked(html):
            return True
        await asyncio.sleep(1)
    return False


# ---------------------------------------------------------------------------
# Browser fetch helpers
# ---------------------------------------------------------------------------


async def _fetch_page_html(browser: Any, url: str) -> tuple[PageRead | None, Any]:
    """Navigate to url and return what the page held once it settled, and the tab.

    Polls until the page shows real character content (``#tab_1_pane``) or
    AniDB's AntiLeech page, waiting out a Cloudflare interstitial on the way;
    see ``wait_through_challenge``. A read whose ``html`` is ``None`` means every
    ``get_content`` raised, a transport failure rather than a page that happens
    to be empty; a read that is not ``ready`` holds the last snapshot, which the
    caller treats as a deleted or invalid character.

    Returns ``(None, None)`` when navigating crashed the browser: a crash needs
    a browser restart, an unreadable page does not.

    Raises:
        CloudflareChallengeError: If a Cloudflare challenge does not clear.
    """
    try:
        page = await browser.get(url)
    except (RuntimeError, StopIteration) as exc:
        logger.warning(f"browser crash on {url}: {exc}")
        return None, None
    read = await wait_through_challenge(
        page,
        url,
        site=_SITE,
        selector=_CHARACTER_SELECTOR,
        stop_on=_is_antileech,
        timeout=_PAGE_TIMEOUT_SECONDS,
    )
    return read, page


# ---------------------------------------------------------------------------
# Cache stub — keyed by char_id, schema-hashed from _extract_from_html so
# any change to XPath extraction logic invalidates existing cache entries.
# Never called directly; only used for cache_batch_get / cache_batch_set.
# ---------------------------------------------------------------------------


@cached_result(
    service="anidb",
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

    cached_values, missing_indices = await _anidb_character_cache.cache_batch_get(
        char_ids
    )  # type: ignore[attr-defined]
    missing_set = set(missing_indices)

    session: BrowserSession | None = None
    succeeded = 0

    try:
        async with AsyncExitStack() as browser_stack:
            for i, char_id in enumerate(char_ids):
                if i not in missing_set:
                    val = cached_values[i]
                    yield (
                        char_id,
                        AniDBCharacterPage.model_validate(val) if val else None,
                    )
                    continue

                # Cache miss — launch browser lazily on first miss
                if session is None:
                    session = await browser_stack.enter_async_context(
                        browser_session(headless=False, clearance_site=_CLEARANCE_SITE)
                    )

                url = f"{_BASE_URL}/{char_id}"
                try:
                    read, current_page = await _fetch_page_html(session.browser, url)
                    if read is None:
                        logger.warning(
                            f"browser crashed — restarting for char {char_id}"
                        )
                        await session.restart()
                        await asyncio.sleep(2)
                        read, current_page = await _fetch_page_html(
                            session.browser, url
                        )
                except CloudflareChallengeError:
                    yield char_id, None
                    continue
                if read is None:
                    logger.error(f"browser crashed again on char {char_id} — skipping")
                    yield char_id, None
                    continue

                html = read.html
                if html is None:
                    logger.warning(
                        f"could not read page for char {char_id} — skipping "
                        f"(transport failure, not a missing character)"
                    )
                    yield char_id, None
                    continue

                if _is_antileech(html):
                    logger.info(f"antileech on char {char_id} — solving")
                    if not await _solve_antileech(current_page):
                        logger.error(f"antileech did not clear for char {char_id}")
                        yield char_id, None
                        continue
                    await current_page
                    try:
                        html = await current_page.get_content()
                    except Exception:
                        yield char_id, None
                        continue

                if not _has_character_data(html):
                    logger.warning(
                        f"no character data for char {char_id} (deleted/invalid)"
                    )
                    yield char_id, None
                else:
                    if read.challenged:
                        await save_clearance(session.browser, _CLEARANCE_SITE)
                    page = _extract_from_html(html)
                    if page is not None:
                        succeeded += 1
                    page_dict = (
                        page.model_dump(mode="json") if page is not None else None
                    )
                    # Cache immediately so cancellation doesn't lose progress
                    await _anidb_character_cache.cache_batch_set([char_id], [page_dict])  # type: ignore[attr-defined]
                    yield char_id, page

                # Delay only between browser requests
                if any(j in missing_set for j in range(i + 1, len(char_ids))):
                    await asyncio.sleep(_INTER_REQUEST_DELAY)

    finally:
        cache_hits = len(char_ids) - len(missing_set)
        logger.info(
            f"anidb character fetch: {succeeded}/{len(missing_set)} succeeded, {cache_hits} cache hits"
        )


async def fetch_anidb_character(char_id: int) -> AniDBCharacterPage | None:
    """Fetch a single AniDB character page. Convenience wrapper.

    `aclosing` closes the generator on return, so its browser is stopped before
    this function returns rather than in a background task later.
    """
    async with aclosing(fetch_anidb_characters([char_id])) as pages:
        async for _, page in pages:
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
