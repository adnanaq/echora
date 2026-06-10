"""Unit tests for mal_character_refs_crawler.py.

Fixture tests use a 5-row HTML subset captured from:
    https://myanimelist.net/anime/21/One_Piece/characters (2026-06-09)
containing character tables for: Brook, Luffy, Aisa, Bariete, Shivercalero.
"""

from unittest.mock import AsyncMock

import pytest
from enrichment.sources.mal.mal_character_refs_crawler import (
    _CHAR_URL_XPATH,
    _extract_character_urls,
    _fetch_characters_page_html,
    _fetch_mal_characters_data,
    fetch_mal_character_refs,
)

pytestmark = pytest.mark.asyncio

_ONE_PIECE_CHARS_URL = "https://myanimelist.net/anime/21/One_Piece/characters"

_EXPECTED_URLS = [
    "https://myanimelist.net/character/5627/Brook",
    "https://myanimelist.net/character/40/Luffy_Monkey_D",
    "https://myanimelist.net/character/23021/Aisa",
    "https://myanimelist.net/character/153828/Bariete",
    "https://myanimelist.net/character/63333/Shivercalero_Donquixote",
]


# =============================================================================
# _CHAR_URL_XPATH invariant
# =============================================================================


def test_xpath_targets_character_tables() -> None:
    assert "js-anime-character-table" in _CHAR_URL_XPATH
    assert "/character/" in _CHAR_URL_XPATH
    assert _CHAR_URL_XPATH.endswith("/@href")


# =============================================================================
# _extract_character_urls
# =============================================================================


def test_extract_from_fixture(mal_char_refs_html) -> None:
    urls = _extract_character_urls(mal_char_refs_html)
    assert urls == _EXPECTED_URLS


def test_extract_empty_html_returns_empty() -> None:
    assert _extract_character_urls("") == []


def test_extract_no_tables_returns_empty() -> None:
    assert _extract_character_urls("<html><body><p>nothing</p></body></html>") == []


def test_extract_deduplicates_urls() -> None:
    html = """<html><body>
    <table class="js-anime-character-table">
      <tr><td><a href="https://myanimelist.net/character/40/Luffy">Luffy</a></td></tr>
    </table>
    <table class="js-anime-character-table">
      <tr><td><a href="https://myanimelist.net/character/40/Luffy">Luffy</a></td></tr>
    </table>
    </body></html>"""
    urls = _extract_character_urls(html)
    assert urls == ["https://myanimelist.net/character/40/Luffy"]


def test_extract_preserves_order() -> None:
    urls = _extract_character_urls(_build_html(["character/1/A", "character/2/B", "character/3/C"]))
    assert urls == [
        "https://myanimelist.net/character/1/A",
        "https://myanimelist.net/character/2/B",
        "https://myanimelist.net/character/3/C",
    ]


def test_extract_ignores_non_character_links() -> None:
    # Both links are in td[1]; only the one with /character/ in href is matched.
    html = """<html><body>
    <table class="js-anime-character-table">
      <tr>
        <td>
          <a href="https://myanimelist.net/anime/21/One_Piece">Anime</a>
          <a href="https://myanimelist.net/character/40/Luffy">Luffy</a>
        </td>
      </tr>
    </table>
    </body></html>"""
    urls = _extract_character_urls(html)
    assert urls == ["https://myanimelist.net/character/40/Luffy"]


# =============================================================================
# _fetch_characters_page_html
# =============================================================================


async def test_fetch_page_html_success(mal_char_refs_html) -> None:
    page_mock = AsyncMock()
    page_mock.wait_for = AsyncMock()
    page_mock.get_content = AsyncMock(return_value=mal_char_refs_html)
    browser_mock = AsyncMock()
    browser_mock.get = AsyncMock(return_value=page_mock)
    browser_mock.stop = AsyncMock()

    with pytest.MonkeyPatch.context() as mp:
        import zendriver as zd
        mp.setattr(zd, "start", AsyncMock(return_value=browser_mock))
        result = await _fetch_characters_page_html(_ONE_PIECE_CHARS_URL)

    assert result == mal_char_refs_html
    page_mock.wait_for.assert_awaited_once()


async def test_fetch_page_html_navigation_failure() -> None:
    page_mock = AsyncMock()
    page_mock.wait_for = AsyncMock(side_effect=Exception("timeout"))
    browser_mock = AsyncMock()
    browser_mock.get = AsyncMock(return_value=page_mock)
    browser_mock.stop = AsyncMock()

    with pytest.MonkeyPatch.context() as mp:
        import zendriver as zd
        mp.setattr(zd, "start", AsyncMock(return_value=browser_mock))
        result = await _fetch_characters_page_html(_ONE_PIECE_CHARS_URL)

    assert result is None


async def test_fetch_page_html_browser_stop_exception(mal_char_refs_html) -> None:
    page_mock = AsyncMock()
    page_mock.wait_for = AsyncMock()
    page_mock.get_content = AsyncMock(return_value=mal_char_refs_html)
    browser_mock = AsyncMock()
    browser_mock.get = AsyncMock(return_value=page_mock)
    browser_mock.stop = AsyncMock(side_effect=Exception("stop failed"))

    with pytest.MonkeyPatch.context() as mp:
        import zendriver as zd
        mp.setattr(zd, "start", AsyncMock(return_value=browser_mock))
        result = await _fetch_characters_page_html(_ONE_PIECE_CHARS_URL)

    assert result == mal_char_refs_html


# =============================================================================
# _fetch_mal_characters_data (cached)
# =============================================================================


def _disable_cache(mocker):
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )


async def test_fetch_data_success(mocker, mal_char_refs_html) -> None:
    _disable_cache(mocker)
    mocker.patch(
        "enrichment.sources.mal.mal_character_refs_crawler._fetch_characters_page_html",
        new_callable=AsyncMock,
        return_value=mal_char_refs_html,
    )
    result = await _fetch_mal_characters_data(_ONE_PIECE_CHARS_URL)
    assert result == _EXPECTED_URLS


async def test_fetch_data_navigation_failure_returns_none(mocker) -> None:
    _disable_cache(mocker)
    mocker.patch(
        "enrichment.sources.mal.mal_character_refs_crawler._fetch_characters_page_html",
        new_callable=AsyncMock,
        return_value=None,
    )
    assert await _fetch_mal_characters_data(_ONE_PIECE_CHARS_URL) is None


async def test_fetch_data_empty_html_returns_none(mocker) -> None:
    _disable_cache(mocker)
    mocker.patch(
        "enrichment.sources.mal.mal_character_refs_crawler._fetch_characters_page_html",
        new_callable=AsyncMock,
        return_value="",
    )
    assert await _fetch_mal_characters_data(_ONE_PIECE_CHARS_URL) is None


async def test_fetch_data_no_tables_returns_none(mocker) -> None:
    _disable_cache(mocker)
    mocker.patch(
        "enrichment.sources.mal.mal_character_refs_crawler._fetch_characters_page_html",
        new_callable=AsyncMock,
        return_value="<html><body><p>nothing</p></body></html>",
    )
    assert await _fetch_mal_characters_data(_ONE_PIECE_CHARS_URL) is None


# =============================================================================
# fetch_mal_character_refs (public API)
# =============================================================================


async def test_returns_empty_on_none(mocker) -> None:
    mocker.patch(
        "enrichment.sources.mal.mal_character_refs_crawler._fetch_mal_characters_data",
        new_callable=AsyncMock,
        return_value=None,
    )
    assert await fetch_mal_character_refs(_ONE_PIECE_CHARS_URL) == []


async def test_returns_urls_on_success(mocker) -> None:
    mocker.patch(
        "enrichment.sources.mal.mal_character_refs_crawler._fetch_mal_characters_data",
        new_callable=AsyncMock,
        return_value=_EXPECTED_URLS,
    )
    result = await fetch_mal_character_refs(_ONE_PIECE_CHARS_URL)
    assert result == _EXPECTED_URLS


# =============================================================================
# helpers
# =============================================================================


def _build_html(hrefs: list[str]) -> str:
    """Build minimal character-table HTML; hrefs must include 'character/' segment."""
    tables = "\n".join(
        f'<table class="js-anime-character-table"><tr>'
        f'<td><a href="https://myanimelist.net/{href}">Name</a></td>'
        f'</tr></table>'
        for href in hrefs
    )
    return f"<html><body>{tables}</body></html>"
