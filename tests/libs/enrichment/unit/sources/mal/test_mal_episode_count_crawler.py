"""Unit tests for mal_episode_count_crawler.py — episode counter span parsing."""

from unittest.mock import AsyncMock, patch

import pytest
from enrichment.sources.mal.mal_episode_count_crawler import (
    _EPISODE_COUNT_XPATH,
    _extract_episode_count,
    _fetch_episode_count_data,
    _fetch_episode_count_html,
    fetch_mal_episode_count,
)

pytestmark = pytest.mark.asyncio

ANIME_URL = "https://myanimelist.net/anime/21/One_Piece"
EPISODE_LIST_URL = f"{ANIME_URL}/episode"

_PATCH = "enrichment.sources.mal.mal_episode_count_crawler._fetch_episode_count_data"

_HTML_ONGOING = """<html><body>
  <h2 class="h2_overwrite">Episodes</h2>
  <span class="di-ib pl4 fw-n fs10">(1,155/Unknown)</span>
</body></html>"""

_HTML_FINISHED = """<html><body>
  <h2 class="h2_overwrite">Episodes</h2>
  <span class="di-ib pl4 fw-n fs10">(26/26)</span>
</body></html>"""


# =============================================================================
# _EPISODE_COUNT_XPATH invariant
# =============================================================================


def test_xpath_targets_episodes_heading() -> None:
    assert "h2_overwrite" in _EPISODE_COUNT_XPATH
    assert "Episodes" in _EPISODE_COUNT_XPATH
    assert "following-sibling::span" in _EPISODE_COUNT_XPATH


# =============================================================================
# _extract_episode_count
# =============================================================================


@pytest.mark.parametrize(
    "html, expected",
    [
        (_HTML_ONGOING, "(1,155/Unknown)"),
        (_HTML_FINISHED, "(26/26)"),
    ],
)
def test_extract_parses_counter(html: str, expected: str) -> None:
    assert _extract_episode_count(html) == expected


def test_extract_empty_html_returns_none() -> None:
    assert _extract_episode_count("") is None


def test_extract_missing_span_returns_none() -> None:
    assert _extract_episode_count("<html><body><h2 class='h2_overwrite'>Episodes</h2></body></html>") is None


# =============================================================================
# _fetch_episode_count_html
# =============================================================================


async def test_fetch_html_success() -> None:
    page_mock = AsyncMock()
    page_mock.wait_for = AsyncMock()
    page_mock.get_content = AsyncMock(return_value=_HTML_ONGOING)
    browser_mock = AsyncMock()
    browser_mock.get = AsyncMock(return_value=page_mock)
    browser_mock.stop = AsyncMock()

    with pytest.MonkeyPatch.context() as mp:
        import zendriver as zd
        mp.setattr(zd, "start", AsyncMock(return_value=browser_mock))
        result = await _fetch_episode_count_html(EPISODE_LIST_URL)

    assert result == _HTML_ONGOING
    page_mock.wait_for.assert_awaited_once()


async def test_fetch_html_navigation_failure_returns_none() -> None:
    page_mock = AsyncMock()
    page_mock.wait_for = AsyncMock(side_effect=Exception("timeout"))
    browser_mock = AsyncMock()
    browser_mock.get = AsyncMock(return_value=page_mock)
    browser_mock.stop = AsyncMock()

    with pytest.MonkeyPatch.context() as mp:
        import zendriver as zd
        mp.setattr(zd, "start", AsyncMock(return_value=browser_mock))
        result = await _fetch_episode_count_html(EPISODE_LIST_URL)

    assert result is None


async def test_fetch_html_browser_stop_exception_swallowed() -> None:
    page_mock = AsyncMock()
    page_mock.wait_for = AsyncMock()
    page_mock.get_content = AsyncMock(return_value=_HTML_FINISHED)
    browser_mock = AsyncMock()
    browser_mock.get = AsyncMock(return_value=page_mock)
    browser_mock.stop = AsyncMock(side_effect=Exception("stop failed"))

    with pytest.MonkeyPatch.context() as mp:
        import zendriver as zd
        mp.setattr(zd, "start", AsyncMock(return_value=browser_mock))
        result = await _fetch_episode_count_html(EPISODE_LIST_URL)

    assert result == _HTML_FINISHED


# =============================================================================
# _fetch_episode_count_data
# =============================================================================


async def test_fetch_data_success(mocker) -> None:
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )
    mocker.patch(
        "enrichment.sources.mal.mal_episode_count_crawler._fetch_episode_count_html",
        new=AsyncMock(return_value=_HTML_ONGOING),
    )
    result = await _fetch_episode_count_data(EPISODE_LIST_URL)
    assert result == "(1,155/Unknown)"


@pytest.mark.parametrize(
    "html",
    [
        None,
        "<html><body><p>no heading</p></body></html>",
    ],
)
async def test_fetch_data_returns_none_on_failure(html: str | None, mocker) -> None:
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )
    mocker.patch(
        "enrichment.sources.mal.mal_episode_count_crawler._fetch_episode_count_html",
        new=AsyncMock(return_value=html),
    )
    assert await _fetch_episode_count_data(EPISODE_LIST_URL) is None


# =============================================================================
# fetch_mal_episode_count
# =============================================================================


@pytest.mark.parametrize(
    "counter, expected",
    [
        ("(12/12)", 12),
        ("(1,155/Unknown)", 1155),
    ],
)
async def test_parse_count_from_counter(counter: str, expected: int) -> None:
    with patch(_PATCH, new=AsyncMock(return_value=counter)):
        assert await fetch_mal_episode_count(ANIME_URL) == expected


@pytest.mark.parametrize("counter", [None, ""])
async def test_returns_zero_on_no_counter(counter: str | None) -> None:
    with patch(_PATCH, new=AsyncMock(return_value=counter)):
        assert await fetch_mal_episode_count(ANIME_URL) == 0


async def test_fetches_episode_list_url() -> None:
    mock_fetch = AsyncMock(return_value="(12/12)")
    with patch(_PATCH, new=mock_fetch):
        await fetch_mal_episode_count(ANIME_URL)
        mock_fetch.assert_awaited_once_with(EPISODE_LIST_URL)
