"""Unit tests for anime_planet_character_refs_crawler.py.

Fixture tests use a real HTML page captured from:
    https://www.anime-planet.com/anime/dandadan/characters (2026-06-10)
containing 25 character refs.
"""

from unittest.mock import AsyncMock

import pytest
from enrichment.sources.anime_planet.anime_planet_character_refs_crawler import (
    _XPATHS,
    _extract_refs_from_html,
    _fetch_refs_data,
    _fetch_refs_html,
    fetch_animeplanet_character_refs,
)

pytestmark = pytest.mark.asyncio

_DANDADAN_URL = "https://www.anime-planet.com/anime/dandadan/characters"

_EXPECTED_FIRST = {"url": "/characters/ken-takakura", "role": ""}
_EXPECTED_COUNT = 25


# =============================================================================
# _XPATHS invariant
# =============================================================================


def test_xpaths_targets_character_links() -> None:
    assert "characters" in _XPATHS
    assert "/characters/" in _XPATHS["characters"]
    assert "name" in _XPATHS["characters"]


# =============================================================================
# _extract_refs_from_html
# =============================================================================


def test_extract_from_fixture(ap_char_refs_html: str) -> None:
    refs = _extract_refs_from_html(ap_char_refs_html)
    assert refs is not None
    assert len(refs) == _EXPECTED_COUNT
    assert refs[0] == _EXPECTED_FIRST
    assert all("url" in r and r["role"] == "" for r in refs)


@pytest.mark.parametrize(
    "html",
    [
        "",
        "<html><body><p>no character links here</p></body></html>",
    ],
)
def test_extract_returns_none_on_no_links(html: str) -> None:
    assert _extract_refs_from_html(html) is None


def test_extract_ignores_anchors_without_href() -> None:
    html = (
        "<html><body>"
        '<a class="name" href="/characters/luffy">Luffy</a>'
        '<a class="name">No href</a>'
        "</body></html>"
    )
    refs = _extract_refs_from_html(html)
    assert refs == [{"url": "/characters/luffy", "role": ""}]


# =============================================================================
# _fetch_refs_html
# =============================================================================


async def test_fetch_refs_html_success(ap_char_refs_html: str) -> None:
    page = AsyncMock()
    page.wait_for = AsyncMock()
    page.get_content = AsyncMock(return_value=ap_char_refs_html)
    browser = AsyncMock()
    browser.get = AsyncMock(return_value=page)
    browser.stop = AsyncMock()

    with pytest.MonkeyPatch.context() as mp:
        import zendriver as zd
        mp.setattr(zd, "start", AsyncMock(return_value=browser))
        result = await _fetch_refs_html(_DANDADAN_URL)

    assert result == ap_char_refs_html
    page.wait_for.assert_awaited_once()


async def test_fetch_refs_html_navigation_failure() -> None:
    page = AsyncMock()
    page.wait_for = AsyncMock(side_effect=Exception("timeout"))
    browser = AsyncMock()
    browser.get = AsyncMock(return_value=page)
    browser.stop = AsyncMock()

    with pytest.MonkeyPatch.context() as mp:
        import zendriver as zd
        mp.setattr(zd, "start", AsyncMock(return_value=browser))
        result = await _fetch_refs_html(_DANDADAN_URL)

    assert result is None


async def test_fetch_refs_html_browser_stop_swallowed(ap_char_refs_html: str) -> None:
    page = AsyncMock()
    page.wait_for = AsyncMock()
    page.get_content = AsyncMock(return_value=ap_char_refs_html)
    browser = AsyncMock()
    browser.get = AsyncMock(return_value=page)
    browser.stop = AsyncMock(side_effect=Exception("stop failed"))

    with pytest.MonkeyPatch.context() as mp:
        import zendriver as zd
        mp.setattr(zd, "start", AsyncMock(return_value=browser))
        result = await _fetch_refs_html(_DANDADAN_URL)

    assert result == ap_char_refs_html


# =============================================================================
# _fetch_refs_data (cached)
# =============================================================================


def _disable_cache(mocker):
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )


_PATCH_FETCH_HTML = (
    "enrichment.sources.anime_planet.anime_planet_character_refs_crawler._fetch_refs_html"
)


async def test_fetch_refs_data_success(mocker, ap_char_refs_html: str) -> None:
    _disable_cache(mocker)
    mocker.patch(_PATCH_FETCH_HTML, new=AsyncMock(return_value=ap_char_refs_html))

    result = await _fetch_refs_data(_DANDADAN_URL)

    assert result is not None
    assert len(result) == _EXPECTED_COUNT
    assert result[0] == _EXPECTED_FIRST


async def test_fetch_refs_data_no_html_returns_none(mocker) -> None:
    _disable_cache(mocker)
    mocker.patch(_PATCH_FETCH_HTML, new=AsyncMock(return_value=None))

    assert await _fetch_refs_data(_DANDADAN_URL) is None


async def test_fetch_refs_data_empty_page_returns_none(mocker) -> None:
    _disable_cache(mocker)
    mocker.patch(
        _PATCH_FETCH_HTML,
        new=AsyncMock(return_value="<html><body></body></html>"),
    )

    assert await _fetch_refs_data(_DANDADAN_URL) is None


# =============================================================================
# fetch_animeplanet_character_refs (public API)
# =============================================================================


_PATCH_FETCH_DATA = (
    "enrichment.sources.anime_planet.anime_planet_character_refs_crawler._fetch_refs_data"
)


async def test_fetch_character_refs_returns_empty_on_none(mocker) -> None:
    mocker.patch(_PATCH_FETCH_DATA, new=AsyncMock(return_value=None))
    assert await fetch_animeplanet_character_refs(_DANDADAN_URL) == []


async def test_fetch_character_refs_returns_list_on_success(mocker) -> None:
    expected = [{"url": "/characters/ken-takakura", "role": ""}]
    mocker.patch(_PATCH_FETCH_DATA, new=AsyncMock(return_value=expected))
    assert await fetch_animeplanet_character_refs(_DANDADAN_URL) == expected
