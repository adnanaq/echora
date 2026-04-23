"""Tests for AniSearchEnrichmentHelper — validates current public API."""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from enrichment.api_helpers.anisearch_helper import AniSearchEnrichmentHelper


@pytest.fixture
def helper() -> AniSearchEnrichmentHelper:
    return AniSearchEnrichmentHelper()


@pytest.fixture
def sample_anime_data() -> dict[str, Any]:
    return {
        "title": "Dandadan",
        "type": "TV",
        "status": "COMPLETED",
        "year": 2024,
    }


@pytest.fixture
def sample_episodes() -> list[dict[str, Any]]:
    return [
        {"episodeNumber": 1, "title": "That's How Love Starts"},
        {"episodeNumber": 2, "title": "That's a Space Alien"},
    ]


@pytest.fixture
def sample_characters() -> list[dict[str, Any]]:
    return [
        {"name": "Momo Ayase", "sources": ["https://www.anisearch.com/character/123"]},
        {"name": "Ken Takakura", "sources": ["https://www.anisearch.com/character/124"]},
    ]


@pytest.fixture
def sample_refs() -> list[dict[str, str]]:
    return [
        {"url": "https://www.anisearch.com/character/123,momo-ayase", "role": "Main Character"},
        {"url": "https://www.anisearch.com/character/124,ken-takakura", "role": "Main Character"},
    ]


# ── fetch_anime ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@patch("enrichment.api_helpers.anisearch_helper.fetch_anisearch_anime")
async def test_fetch_anime_success(mock_fetch, helper, sample_anime_data):
    mock_fetch.return_value = sample_anime_data
    result = await helper.fetch_anime(18878)
    assert result == sample_anime_data
    mock_fetch.assert_called_once()


@pytest.mark.asyncio
@patch("enrichment.api_helpers.anisearch_helper.fetch_anisearch_anime")
async def test_fetch_anime_returns_none(mock_fetch, helper):
    mock_fetch.return_value = None
    result = await helper.fetch_anime(18878)
    assert result is None


@pytest.mark.asyncio
@patch("enrichment.api_helpers.anisearch_helper.fetch_anisearch_anime")
async def test_fetch_anime_exception(mock_fetch, helper):
    mock_fetch.side_effect = Exception("Crawler error")
    result = await helper.fetch_anime(18878)
    assert result is None


# ── fetch_episodes ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@patch("enrichment.api_helpers.anisearch_helper.fetch_anisearch_episodes")
async def test_fetch_episodes_success(mock_fetch, helper, sample_episodes):
    mock_fetch.return_value = sample_episodes
    result = await helper.fetch_episodes(18878)
    assert result == sample_episodes
    mock_fetch.assert_called_once_with(anime_id="18878", output_path=None)


@pytest.mark.asyncio
@patch("enrichment.api_helpers.anisearch_helper.fetch_anisearch_episodes")
async def test_fetch_episodes_returns_none(mock_fetch, helper):
    mock_fetch.return_value = None
    result = await helper.fetch_episodes(18878)
    assert result is None


@pytest.mark.asyncio
@patch("enrichment.api_helpers.anisearch_helper.fetch_anisearch_episodes")
async def test_fetch_episodes_empty_list(mock_fetch, helper):
    mock_fetch.return_value = []
    result = await helper.fetch_episodes(18878)
    assert result is None


@pytest.mark.asyncio
@patch("enrichment.api_helpers.anisearch_helper.fetch_anisearch_episodes")
async def test_fetch_episodes_exception(mock_fetch, helper):
    mock_fetch.side_effect = Exception("Episode error")
    result = await helper.fetch_episodes(18878)
    assert result is None


# ── fetch_character_refs ──────────────────────────────────────────────────────


@pytest.mark.asyncio
@patch("enrichment.api_helpers.anisearch_helper.fetch_anisearch_character_refs")
async def test_fetch_character_refs_success(mock_fetch, helper, sample_refs):
    mock_fetch.return_value = sample_refs
    result = await helper.fetch_character_refs(18878)
    assert result == sample_refs
    mock_fetch.assert_called_once_with("18878")


@pytest.mark.asyncio
@patch("enrichment.api_helpers.anisearch_helper.fetch_anisearch_character_refs")
async def test_fetch_character_refs_empty(mock_fetch, helper):
    mock_fetch.return_value = []
    result = await helper.fetch_character_refs(18878)
    assert result == []


# ── fetch_characters ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
@patch("enrichment.api_helpers.anisearch_helper.fetch_anisearch_character_refs")
@patch("enrichment.api_helpers.anisearch_helper.fetch_anisearch_characters")
async def test_fetch_characters_success(mock_chars, mock_refs, helper, sample_refs, sample_characters):
    mock_refs.return_value = sample_refs
    mock_chars.return_value = sample_characters
    result = await helper.fetch_characters(18878)
    assert result == sample_characters


@pytest.mark.asyncio
@patch("enrichment.api_helpers.anisearch_helper.fetch_anisearch_character_refs")
async def test_fetch_characters_no_refs(mock_refs, helper):
    mock_refs.return_value = []
    result = await helper.fetch_characters(18878)
    assert result is None


@pytest.mark.asyncio
@patch("enrichment.api_helpers.anisearch_helper.fetch_anisearch_character_refs")
@patch("enrichment.api_helpers.anisearch_helper.fetch_anisearch_characters")
async def test_fetch_characters_all_null(mock_chars, mock_refs, helper, sample_refs):
    mock_refs.return_value = sample_refs
    mock_chars.return_value = [None, None]
    result = await helper.fetch_characters(18878)
    assert result is None


@pytest.mark.asyncio
@patch("enrichment.api_helpers.anisearch_helper.fetch_anisearch_character_refs")
async def test_fetch_characters_exception(mock_refs, helper):
    mock_refs.side_effect = Exception("Refs error")
    result = await helper.fetch_characters(18878)
    assert result is None


# ── fetch_all ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@patch("enrichment.api_helpers.anisearch_helper.fetch_anisearch_anime")
@patch("enrichment.api_helpers.anisearch_helper.fetch_anisearch_episodes")
@patch("enrichment.api_helpers.anisearch_helper.fetch_anisearch_character_refs")
@patch("enrichment.api_helpers.anisearch_helper.fetch_anisearch_characters")
async def test_fetch_all_complete(mock_chars, mock_refs, mock_ep, mock_anime,
                                   helper, sample_anime_data, sample_episodes, sample_characters, sample_refs):
    mock_anime.return_value = sample_anime_data
    mock_ep.return_value = sample_episodes
    mock_refs.return_value = sample_refs
    mock_chars.return_value = sample_characters

    result = await helper.fetch_all({"anisearch_id": "18878"}, {})

    assert result is not None
    assert result["title"] == "Dandadan"
    assert result["episodes"] == sample_episodes
    assert result["characters"] == sample_characters


@pytest.mark.asyncio
@patch("enrichment.api_helpers.anisearch_helper.append_jsonl")
@patch("enrichment.api_helpers.anisearch_helper.fetch_anisearch_anime")
@patch("enrichment.api_helpers.anisearch_helper.fetch_anisearch_episodes")
@patch("enrichment.api_helpers.anisearch_helper.fetch_anisearch_character_refs")
async def test_fetch_all_writes_jsonl_when_temp_dir(mock_refs, mock_ep, mock_anime, mock_append,
                                                     helper, sample_anime_data):
    mock_anime.return_value = sample_anime_data
    mock_ep.return_value = None
    mock_refs.return_value = []
    result = await helper.fetch_all({"anisearch_id": "18878"}, {}, temp_dir="/tmp")
    assert result is not None
    mock_append.assert_called_once()


@pytest.mark.asyncio
@patch("enrichment.api_helpers.anisearch_helper.fetch_anisearch_anime")
async def test_fetch_all_no_anisearch_id(mock_anime, helper):
    result = await helper.fetch_all({}, {})
    assert result is None
    mock_anime.assert_not_called()


@pytest.mark.asyncio
@patch("enrichment.api_helpers.anisearch_helper.fetch_anisearch_anime")
async def test_fetch_all_anime_returns_none(mock_anime, helper):
    mock_anime.return_value = None
    result = await helper.fetch_all({"anisearch_id": "18878"}, {})
    assert result is None


@pytest.mark.asyncio
@patch("enrichment.api_helpers.anisearch_helper.fetch_anisearch_anime")
@patch("enrichment.api_helpers.anisearch_helper.fetch_anisearch_episodes")
async def test_fetch_all_episode_failure_graceful(mock_ep, mock_anime, helper, sample_anime_data):
    mock_anime.return_value = sample_anime_data
    mock_ep.side_effect = Exception("Episode fetch failed")
    result = await helper.fetch_all({"anisearch_id": "18878"}, {})
    assert result is not None
    assert "episodes" not in result


@pytest.mark.asyncio
@patch("enrichment.api_helpers.anisearch_helper.fetch_anisearch_anime")
async def test_fetch_all_episode_method_raises_warning(mock_anime, helper, sample_anime_data):
    mock_anime.return_value = sample_anime_data
    with patch.object(helper, "fetch_episodes", side_effect=RuntimeError("method error")):
        result = await helper.fetch_all({"anisearch_id": "18878"}, {})
    assert result is not None
    assert "episodes" not in result


@pytest.mark.asyncio
@patch("enrichment.api_helpers.anisearch_helper.fetch_anisearch_anime")
@patch("enrichment.api_helpers.anisearch_helper.fetch_anisearch_character_refs")
async def test_fetch_all_character_failure_graceful(mock_refs, mock_anime, helper, sample_anime_data):
    mock_anime.return_value = sample_anime_data
    mock_refs.side_effect = Exception("Character fetch failed")
    result = await helper.fetch_all({"anisearch_id": "18878"}, {})
    assert result is not None
    assert "characters" not in result


@pytest.mark.asyncio
@patch("enrichment.api_helpers.anisearch_helper.fetch_anisearch_anime")
async def test_fetch_all_character_method_raises_warning(mock_anime, helper, sample_anime_data):
    mock_anime.return_value = sample_anime_data
    with patch.object(helper, "fetch_characters", side_effect=RuntimeError("method error")):
        result = await helper.fetch_all({"anisearch_id": "18878"}, {})
    assert result is not None
    assert "characters" not in result


@pytest.mark.asyncio
@patch("enrichment.api_helpers.anisearch_helper.fetch_anisearch_anime")
@patch("enrichment.api_helpers.anisearch_helper.fetch_anisearch_episodes")
async def test_fetch_all_episode_returns_none(mock_ep, mock_anime, helper, sample_anime_data):
    mock_anime.return_value = sample_anime_data
    mock_ep.return_value = None

    result = await helper.fetch_all({"anisearch_id": "18878"}, {})

    assert result is not None
    assert "episodes" not in result


@pytest.mark.asyncio
@patch("enrichment.api_helpers.anisearch_helper.fetch_anisearch_anime")
@patch("enrichment.api_helpers.anisearch_helper.fetch_anisearch_character_refs")
async def test_fetch_all_character_refs_empty(mock_refs, mock_anime, helper, sample_anime_data):
    mock_anime.return_value = sample_anime_data
    mock_refs.return_value = []

    result = await helper.fetch_all({"anisearch_id": "18878"}, {})

    assert result is not None
    assert "characters" not in result


@pytest.mark.asyncio
@patch("enrichment.api_helpers.anisearch_helper.fetch_anisearch_anime")
async def test_fetch_all_anime_exception(mock_anime, helper):
    mock_anime.side_effect = Exception("General error")
    result = await helper.fetch_all({"anisearch_id": "18878"}, {})
    assert result is None
