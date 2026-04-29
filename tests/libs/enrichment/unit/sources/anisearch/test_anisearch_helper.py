"""Tests for AniSearchHelper — validates current public API."""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from enrichment.sources.anisearch.anisearch_helper import AniSearchHelper

_URL_WITH_SLUG = "https://www.anisearch.com/anime/18878,dan-da-dan"
_URL_NO_SLUG = "https://www.anisearch.com/anime/18878"


@pytest.fixture
def helper() -> AniSearchHelper:
    return AniSearchHelper()


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
        {
            "name": "Ken Takakura",
            "sources": ["https://www.anisearch.com/character/124"],
        },
    ]


@pytest.fixture
def sample_refs() -> list[dict[str, str]]:
    return [
        {
            "url": "https://www.anisearch.com/character/123,momo-ayase",
            "role": "Main Character",
        },
        {
            "url": "https://www.anisearch.com/character/124,ken-takakura",
            "role": "Main Character",
        },
    ]


# ── fetch_anime ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@patch("enrichment.sources.anisearch.anisearch_helper.fetch_anisearch_anime")
async def test_fetch_anime_success(mock_fetch, helper, sample_anime_data):
    mock_fetch.return_value = sample_anime_data
    result = await helper.fetch_anime(_URL_WITH_SLUG)
    assert result == sample_anime_data
    mock_fetch.assert_called_once_with(_URL_WITH_SLUG, output_path=None)


@pytest.mark.asyncio
@patch("enrichment.sources.anisearch.anisearch_helper.fetch_anisearch_anime")
async def test_fetch_anime_without_slug(mock_fetch, helper, sample_anime_data):
    """URL without slug is passed through unchanged — crawler handles redirect."""
    mock_fetch.return_value = sample_anime_data
    result = await helper.fetch_anime(_URL_NO_SLUG)
    assert result == sample_anime_data
    mock_fetch.assert_called_once_with(_URL_NO_SLUG, output_path=None)


@pytest.mark.asyncio
@patch("enrichment.sources.anisearch.anisearch_helper.fetch_anisearch_anime")
async def test_fetch_anime_returns_none(mock_fetch, helper):
    mock_fetch.return_value = None
    result = await helper.fetch_anime(_URL_WITH_SLUG)
    assert result is None


@pytest.mark.asyncio
@patch("enrichment.sources.anisearch.anisearch_helper.fetch_anisearch_anime")
async def test_fetch_anime_exception(mock_fetch, helper):
    mock_fetch.side_effect = Exception("Crawler error")
    result = await helper.fetch_anime(_URL_WITH_SLUG)
    assert result is None


# ── fetch_episodes ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@patch("enrichment.sources.anisearch.anisearch_helper.fetch_anisearch_episodes")
async def test_fetch_episodes_passes_url_directly(mock_fetch, helper, sample_episodes):
    """Full anime URL is passed to fetch_anisearch_episodes — it appends /episodes."""
    mock_fetch.return_value = sample_episodes
    result = await helper.fetch_episodes(_URL_WITH_SLUG)
    assert result == sample_episodes
    mock_fetch.assert_called_once_with(_URL_WITH_SLUG, output_path=None)


@pytest.mark.asyncio
@patch("enrichment.sources.anisearch.anisearch_helper.fetch_anisearch_episodes")
async def test_fetch_episodes_without_slug(mock_fetch, helper, sample_episodes):
    """URL without slug also accepted — normalizer in episode crawler handles it."""
    mock_fetch.return_value = sample_episodes
    await helper.fetch_episodes(_URL_NO_SLUG)
    mock_fetch.assert_called_once_with(_URL_NO_SLUG, output_path=None)


@pytest.mark.asyncio
@patch("enrichment.sources.anisearch.anisearch_helper.fetch_anisearch_episodes")
async def test_fetch_episodes_returns_none(mock_fetch, helper):
    mock_fetch.return_value = None
    result = await helper.fetch_episodes(_URL_WITH_SLUG)
    assert result is None


@pytest.mark.asyncio
@patch("enrichment.sources.anisearch.anisearch_helper.fetch_anisearch_episodes")
async def test_fetch_episodes_empty_list(mock_fetch, helper):
    mock_fetch.return_value = []
    result = await helper.fetch_episodes(_URL_WITH_SLUG)
    assert result is None


@pytest.mark.asyncio
@patch("enrichment.sources.anisearch.anisearch_helper.fetch_anisearch_episodes")
async def test_fetch_episodes_exception(mock_fetch, helper):
    mock_fetch.side_effect = Exception("Episode error")
    result = await helper.fetch_episodes(_URL_WITH_SLUG)
    assert result is None


# ── fetch_character_refs ──────────────────────────────────────────────────────


@pytest.mark.asyncio
@patch("enrichment.sources.anisearch.anisearch_helper.fetch_anisearch_character_refs")
async def test_fetch_character_refs_passes_url_directly(
    mock_fetch, helper, sample_refs
):
    """Full anime URL is passed to fetch_anisearch_character_refs — it appends /characters."""
    mock_fetch.return_value = sample_refs
    result = await helper.fetch_character_refs(_URL_WITH_SLUG)
    assert result == sample_refs
    mock_fetch.assert_called_once_with(_URL_WITH_SLUG)


@pytest.mark.asyncio
@patch("enrichment.sources.anisearch.anisearch_helper.fetch_anisearch_character_refs")
async def test_fetch_character_refs_without_slug(mock_fetch, helper, sample_refs):
    mock_fetch.return_value = sample_refs
    await helper.fetch_character_refs(_URL_NO_SLUG)
    mock_fetch.assert_called_once_with(_URL_NO_SLUG)


@pytest.mark.asyncio
@patch("enrichment.sources.anisearch.anisearch_helper.fetch_anisearch_character_refs")
async def test_fetch_character_refs_empty(mock_fetch, helper):
    mock_fetch.return_value = []
    result = await helper.fetch_character_refs(_URL_WITH_SLUG)
    assert result == []


# ── fetch_characters ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
@patch("enrichment.sources.anisearch.anisearch_helper.fetch_anisearch_character_refs")
@patch("enrichment.sources.anisearch.anisearch_helper.fetch_anisearch_characters")
async def test_fetch_characters_success(
    mock_chars, mock_refs, helper, sample_refs, sample_characters
):
    mock_refs.return_value = sample_refs
    mock_chars.return_value = sample_characters
    result = await helper.fetch_characters(_URL_WITH_SLUG)
    assert result == sample_characters


@pytest.mark.asyncio
@patch("enrichment.sources.anisearch.anisearch_helper.fetch_anisearch_character_refs")
async def test_fetch_characters_no_refs(mock_refs, helper):
    mock_refs.return_value = []
    result = await helper.fetch_characters(_URL_WITH_SLUG)
    assert result is None


@pytest.mark.asyncio
@patch("enrichment.sources.anisearch.anisearch_helper.fetch_anisearch_character_refs")
@patch("enrichment.sources.anisearch.anisearch_helper.fetch_anisearch_characters")
async def test_fetch_characters_all_null(mock_chars, mock_refs, helper, sample_refs):
    mock_refs.return_value = sample_refs
    mock_chars.return_value = [None, None]
    result = await helper.fetch_characters(_URL_WITH_SLUG)
    assert result is None


@pytest.mark.asyncio
@patch("enrichment.sources.anisearch.anisearch_helper.fetch_anisearch_character_refs")
async def test_fetch_characters_exception(mock_refs, helper):
    mock_refs.side_effect = Exception("Refs error")
    result = await helper.fetch_characters(_URL_WITH_SLUG)
    assert result is None


# ── fetch_all ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@patch("enrichment.sources.anisearch.anisearch_helper.fetch_anisearch_anime")
@patch("enrichment.sources.anisearch.anisearch_helper.fetch_anisearch_episodes")
@patch("enrichment.sources.anisearch.anisearch_helper.fetch_anisearch_character_refs")
@patch("enrichment.sources.anisearch.anisearch_helper.fetch_anisearch_characters")
async def test_fetch_all_complete(
    mock_chars,
    mock_refs,
    mock_ep,
    mock_anime,
    helper,
    sample_anime_data,
    sample_episodes,
    sample_characters,
    sample_refs,
):
    mock_anime.return_value = sample_anime_data
    mock_ep.return_value = sample_episodes
    mock_refs.return_value = sample_refs
    mock_chars.return_value = sample_characters

    result = await helper.fetch_all({"anisearch_url": _URL_WITH_SLUG}, {})

    assert result is not None
    assert result["anime"] == sample_anime_data
    assert result["episodes"] == sample_episodes
    assert result["characters"] == sample_characters
    assert result["extras"] == {}


@pytest.mark.asyncio
@patch("enrichment.sources.anisearch.anisearch_helper.fetch_anisearch_anime")
@patch("enrichment.sources.anisearch.anisearch_helper.fetch_anisearch_episodes")
@patch("enrichment.sources.anisearch.anisearch_helper.fetch_anisearch_character_refs")
@patch("enrichment.sources.anisearch.anisearch_helper.fetch_anisearch_characters")
async def test_fetch_all_without_slug(
    mock_chars,
    mock_refs,
    mock_ep,
    mock_anime,
    helper,
    sample_anime_data,
    sample_episodes,
    sample_characters,
    sample_refs,
):
    """URL without slug is accepted and passed through to each sub-fetcher."""
    mock_anime.return_value = sample_anime_data
    mock_ep.return_value = sample_episodes
    mock_refs.return_value = sample_refs
    mock_chars.return_value = sample_characters

    result = await helper.fetch_all({"anisearch_url": _URL_NO_SLUG}, {})
    assert result is not None
    mock_anime.assert_called_once_with(_URL_NO_SLUG, output_path=None)


@pytest.mark.asyncio
@patch("enrichment.sources.anisearch.anisearch_helper.fetch_anisearch_anime")
@patch("enrichment.sources.anisearch.anisearch_helper.fetch_anisearch_episodes")
@patch("enrichment.sources.anisearch.anisearch_helper.fetch_anisearch_character_refs")
@patch("enrichment.sources.anisearch.anisearch_helper.fetch_anisearch_characters")
async def test_fetch_all_returns_normalized_payload_shape(
    mock_chars,
    mock_refs,
    mock_ep,
    mock_anime,
    helper,
    sample_anime_data,
    sample_episodes,
    sample_characters,
    sample_refs,
):
    mock_anime.return_value = sample_anime_data
    mock_ep.return_value = sample_episodes
    mock_refs.return_value = sample_refs
    mock_chars.return_value = sample_characters

    result = await helper.fetch_all({"anisearch_url": _URL_WITH_SLUG}, {})

    assert result == {
        "anime": sample_anime_data,
        "episodes": sample_episodes,
        "characters": sample_characters,
        "extras": {},
    }


@pytest.mark.asyncio
@patch("enrichment.sources.anisearch.anisearch_helper.append_jsonl")
@patch("enrichment.sources.anisearch.anisearch_helper.fetch_anisearch_anime")
@patch("enrichment.sources.anisearch.anisearch_helper.fetch_anisearch_episodes")
@patch("enrichment.sources.anisearch.anisearch_helper.fetch_anisearch_character_refs")
async def test_fetch_all_writes_jsonl_when_temp_dir(
    mock_refs, mock_ep, mock_anime, mock_append, helper, sample_anime_data
):
    mock_anime.return_value = sample_anime_data
    mock_ep.return_value = None
    mock_refs.return_value = []
    result = await helper.fetch_all(
        {"anisearch_url": _URL_WITH_SLUG}, {}, temp_dir="/tmp"
    )
    assert result is not None
    mock_append.assert_called_once()


@pytest.mark.asyncio
@patch("enrichment.sources.anisearch.anisearch_helper.fetch_anisearch_anime")
async def test_fetch_all_no_anisearch_url(mock_anime, helper):
    result = await helper.fetch_all({}, {})
    assert result is None
    mock_anime.assert_not_called()


@pytest.mark.asyncio
@patch("enrichment.sources.anisearch.anisearch_helper.fetch_anisearch_anime")
async def test_fetch_all_anime_returns_none(mock_anime, helper):
    mock_anime.return_value = None
    result = await helper.fetch_all({"anisearch_url": _URL_WITH_SLUG}, {})
    assert result is None


@pytest.mark.asyncio
@patch("enrichment.sources.anisearch.anisearch_helper.fetch_anisearch_anime")
@patch("enrichment.sources.anisearch.anisearch_helper.fetch_anisearch_episodes")
async def test_fetch_all_episode_failure_graceful(
    mock_ep, mock_anime, helper, sample_anime_data
):
    mock_anime.return_value = sample_anime_data
    mock_ep.side_effect = Exception("Episode fetch failed")
    with patch.object(helper, "fetch_characters", new=AsyncMock(return_value=[])):
        result = await helper.fetch_all({"anisearch_url": _URL_WITH_SLUG}, {})
    assert result is not None
    assert result["episodes"] == []


@pytest.mark.asyncio
@patch("enrichment.sources.anisearch.anisearch_helper.fetch_anisearch_anime")
async def test_fetch_all_episode_method_raises_warning(
    mock_anime, helper, sample_anime_data
):
    mock_anime.return_value = sample_anime_data
    with patch.object(
        helper, "fetch_episodes", side_effect=RuntimeError("method error")
    ):
        with patch.object(helper, "fetch_characters", new=AsyncMock(return_value=[])):
            result = await helper.fetch_all({"anisearch_url": _URL_WITH_SLUG}, {})
    assert result is not None
    assert result["episodes"] == []


@pytest.mark.asyncio
@patch("enrichment.sources.anisearch.anisearch_helper.fetch_anisearch_anime")
@patch("enrichment.sources.anisearch.anisearch_helper.fetch_anisearch_character_refs")
async def test_fetch_all_character_failure_graceful(
    mock_refs, mock_anime, helper, sample_anime_data
):
    mock_anime.return_value = sample_anime_data
    mock_refs.side_effect = Exception("Character fetch failed")
    with patch.object(helper, "fetch_episodes", new=AsyncMock(return_value=[])):
        result = await helper.fetch_all({"anisearch_url": _URL_WITH_SLUG}, {})
    assert result is not None
    assert result["characters"] == []


@pytest.mark.asyncio
@patch("enrichment.sources.anisearch.anisearch_helper.fetch_anisearch_anime")
async def test_fetch_all_character_method_raises_warning(
    mock_anime, helper, sample_anime_data
):
    mock_anime.return_value = sample_anime_data
    with patch.object(
        helper, "fetch_characters", side_effect=RuntimeError("method error")
    ):
        with patch.object(helper, "fetch_episodes", new=AsyncMock(return_value=[])):
            result = await helper.fetch_all({"anisearch_url": _URL_WITH_SLUG}, {})
    assert result is not None
    assert result["characters"] == []


@pytest.mark.asyncio
@patch("enrichment.sources.anisearch.anisearch_helper.fetch_anisearch_anime")
@patch("enrichment.sources.anisearch.anisearch_helper.fetch_anisearch_episodes")
async def test_fetch_all_episode_returns_none(
    mock_ep, mock_anime, helper, sample_anime_data
):
    mock_anime.return_value = sample_anime_data
    mock_ep.return_value = None
    with patch.object(helper, "fetch_characters", new=AsyncMock(return_value=[])):
        result = await helper.fetch_all({"anisearch_url": _URL_WITH_SLUG}, {})
    assert result is not None
    assert result["episodes"] == []


@pytest.mark.asyncio
@patch("enrichment.sources.anisearch.anisearch_helper.fetch_anisearch_anime")
@patch("enrichment.sources.anisearch.anisearch_helper.fetch_anisearch_character_refs")
async def test_fetch_all_character_refs_empty(
    mock_refs, mock_anime, helper, sample_anime_data
):
    mock_anime.return_value = sample_anime_data
    mock_refs.return_value = []
    with patch.object(helper, "fetch_episodes", new=AsyncMock(return_value=[])):
        result = await helper.fetch_all({"anisearch_url": _URL_WITH_SLUG}, {})
    assert result is not None
    assert result["characters"] == []


@pytest.mark.asyncio
@patch("enrichment.sources.anisearch.anisearch_helper.fetch_anisearch_anime")
async def test_fetch_all_anime_exception(mock_anime, helper):
    mock_anime.side_effect = Exception("General error")
    result = await helper.fetch_all({"anisearch_url": _URL_WITH_SLUG}, {})
    assert result is None


@pytest.mark.asyncio
async def test_fetch_all_uses_canonical_url_for_episodes_and_characters(
    helper, sample_episodes, sample_characters
) -> None:
    """When fetch_anime returns sources with a slug URL, subsequent calls use it — not the slug-less input."""
    canonical_url = "https://www.anisearch.com/anime/18878,dan-da-dan"
    anime_data_with_canonical = {"title": "Dandadan", "sources": [canonical_url]}

    mock_fetch_anime = AsyncMock(return_value=anime_data_with_canonical)
    mock_fetch_episodes = AsyncMock(return_value=sample_episodes)
    mock_fetch_characters = AsyncMock(return_value=sample_characters)

    with patch.object(helper, "fetch_anime", mock_fetch_anime):
        with patch.object(helper, "fetch_episodes", mock_fetch_episodes):
            with patch.object(helper, "fetch_characters", mock_fetch_characters):
                result = await helper.fetch_all({"anisearch_url": _URL_NO_SLUG}, {})

    assert result is not None
    mock_fetch_anime.assert_called_once_with(_URL_NO_SLUG)
    mock_fetch_episodes.assert_called_once_with(canonical_url)
    mock_fetch_characters.assert_called_once_with(canonical_url)
