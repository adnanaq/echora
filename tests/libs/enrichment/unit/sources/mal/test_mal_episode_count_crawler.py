"""Unit tests for mal_episode_count_crawler.py — episode counter span parsing."""

from unittest.mock import AsyncMock, patch

import pytest

from enrichment.sources.mal.mal_episode_count_crawler import (
    fetch_mal_episode_count,
)

ANIME_URL = "https://myanimelist.net/anime/21/One_Piece"
EPISODE_LIST_URL = f"{ANIME_URL}/episode"

_PATCH = "enrichment.sources.mal.mal_episode_count_crawler._fetch_episode_count_data"

# =============================================================================
# Parsing the "(N/...)" counter span
# =============================================================================


@pytest.mark.asyncio
async def test_fetch_mal_episode_count_finished_anime() -> None:
    """Finished anime counter '(12/12)' → returns 12."""
    with patch(_PATCH, new=AsyncMock(return_value="(12/12)")):
        assert await fetch_mal_episode_count(ANIME_URL) == 12


@pytest.mark.asyncio
async def test_fetch_mal_episode_count_ongoing_anime() -> None:
    """Ongoing anime counter '(1,155/Unknown)' → returns 1155 (comma stripped)."""
    with patch(_PATCH, new=AsyncMock(return_value="(1,155/Unknown)")):
        assert await fetch_mal_episode_count(ANIME_URL) == 1155


@pytest.mark.asyncio
async def test_fetch_mal_episode_count_no_commas() -> None:
    """Counter without comma '(47/47)' → returns 47."""
    with patch(_PATCH, new=AsyncMock(return_value="(47/47)")):
        assert await fetch_mal_episode_count(ANIME_URL) == 47


# =============================================================================
# Failure cases
# =============================================================================


@pytest.mark.asyncio
async def test_fetch_mal_episode_count_returns_zero_on_no_data() -> None:
    """_fetch_episode_count_data returns None → returns 0."""
    with patch(_PATCH, new=AsyncMock(return_value=None)):
        assert await fetch_mal_episode_count(ANIME_URL) == 0


@pytest.mark.asyncio
async def test_fetch_mal_episode_count_returns_zero_on_empty_string() -> None:
    """Empty counter string → returns 0."""
    with patch(_PATCH, new=AsyncMock(return_value="")):
        assert await fetch_mal_episode_count(ANIME_URL) == 0


@pytest.mark.asyncio
async def test_fetch_mal_episode_count_fetches_episode_list_url() -> None:
    """_fetch_episode_count_data is called with {anime_url}/episode."""
    mock_fetch = AsyncMock(return_value="(12/12)")
    with patch(_PATCH, new=mock_fetch):
        await fetch_mal_episode_count(ANIME_URL)
        mock_fetch.assert_awaited_once_with(EPISODE_LIST_URL)
