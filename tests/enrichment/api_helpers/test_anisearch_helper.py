"""
Comprehensive tests for AniSearchEnrichmentHelper with 100% coverage.

Tests all methods, edge cases, and error scenarios using mocks.
"""

from typing import Any, Dict, List
from unittest.mock import patch

import pytest

# Import the class to test
from src.enrichment.api_helpers.anisearch_helper import AniSearchEnrichmentHelper


class TestAniSearchEnrichmentHelper:
    """Test suite for AniSearchEnrichmentHelper class."""

    @pytest.fixture
    def helper(self):
        """Create AniSearchEnrichmentHelper instance for testing."""
        return AniSearchEnrichmentHelper()

    @pytest.fixture
    def sample_anime_data(self) -> Dict[str, Any]:
        """Sample anime data matching crawler output format."""
        return {
            "japanese_title": "ダンダダン",
            "japanese_title_alt": "Dandadan",
            "cover_image": "https://cdn.anisearch.com/images/anime/cover/18/18878_600.webp",
            "type": "TV-Series, 12",
            "status": "Completed",
            "start_date": "04.10.2024",
            "end_date": "20.12.2024",
            "studio": "Science SARU Inc.",
            "source_material": "Manga",
            "genres": ["Action", "Comedy", "Fantasy"],
            "tags": ["Alien", "Ghost", "High School"],
            "description": "At first glance, the powerful Momo Ayase...",
            "screenshots": ["https://example.com/screenshot1.jpg"],
            "anime_relations": [
                {
                    "type": "Sequel",
                    "title": "Dan Da Dan: Season 2",
                    "url": "https://www.anisearch.com/anime/19952",
                }
            ],
            "manga_relations": [],
        }

    @pytest.fixture
    def sample_episode_data(self) -> List[Dict[str, Any]]:
        """Sample episode data matching crawler output format."""
        return [
            {
                "episodeNumber": 1,
                "runtime": "24 min",
                "releaseDate": "04.10.2024",
                "title": "That's How Love Starts, Ya Know!",
            },
            {
                "episodeNumber": 2,
                "runtime": "24 min",
                "releaseDate": "11.10.2024",
                "title": "That's a Space Alien, Ain't It?!",
            },
        ]

    @pytest.fixture
    def sample_character_data(self) -> Dict[str, Any]:
        """Sample character data matching crawler output format."""
        return {
            "characters": [
                {
                    "name": "Momo Ayase",
                    "role": "Main",
                    "url": "https://www.anisearch.com/character/123",
                    "image": "https://cdn.anisearch.com/images/character/123.webp",
                    "favorites": 150,
                },
                {
                    "name": "Ken Takakura",
                    "role": "Main",
                    "url": "https://www.anisearch.com/character/124",
                    "image": "https://cdn.anisearch.com/images/character/124.webp",
                    "favorites": 120,
                },
            ],
            "total_count": 2,
        }

    @pytest.fixture
    def sample_offline_data(self) -> Dict[str, Any]:
        """Sample offline anime data with sources."""
        return {
            "title": "Dandadan",
            "sources": [
                "https://myanimelist.net/anime/55102",
                "https://www.anisearch.com/anime/18878,dan-da-dan",
                "https://anilist.co/anime/171018",
            ],
        }

    # ========================================
    # Test: __init__
    # ========================================

    def test_init(self, helper):
        """Test helper initialization."""
        assert helper is not None
        assert isinstance(helper, AniSearchEnrichmentHelper)

    # ========================================
    # Test: extract_anisearch_id_from_url
    # ========================================

    @pytest.mark.asyncio
    async def test_extract_anisearch_id_from_url_full_url(self, helper):
        """Test extracting ID from full URL."""
        url = "https://www.anisearch.com/anime/18878,dan-da-dan"
        result = await helper.extract_anisearch_id_from_url(url)
        assert result == 18878

    @pytest.mark.asyncio
    async def test_extract_anisearch_id_from_url_simple(self, helper):
        """Test extracting ID from simple URL without slug."""
        url = "https://www.anisearch.com/anime/18878"
        result = await helper.extract_anisearch_id_from_url(url)
        assert result == 18878

    @pytest.mark.asyncio
    async def test_extract_anisearch_id_from_url_invalid(self, helper):
        """Test extracting ID from invalid URL."""
        url = "https://www.anisearch.com/manga/123"
        result = await helper.extract_anisearch_id_from_url(url)
        assert result is None

    @pytest.mark.asyncio
    async def test_extract_anisearch_id_from_url_no_id(self, helper):
        """Test extracting ID from URL without ID."""
        url = "https://www.anisearch.com/anime/"
        result = await helper.extract_anisearch_id_from_url(url)
        assert result is None

    @pytest.mark.asyncio
    async def test_extract_anisearch_id_from_url_exception(self, helper):
        """Test exception handling in ID extraction."""
        url = None  # Will cause exception
        result = await helper.extract_anisearch_id_from_url(url)
        assert result is None

    # ========================================
    # Test: find_anisearch_url
    # ========================================

    @pytest.mark.asyncio
    async def test_find_anisearch_url_found(self, helper, sample_offline_data):
        """Test finding AniSearch URL in sources."""
        result = await helper.find_anisearch_url(sample_offline_data)
        assert result == "https://www.anisearch.com/anime/18878,dan-da-dan"

    @pytest.mark.asyncio
    async def test_find_anisearch_url_not_found(self, helper):
        """Test finding AniSearch URL when not in sources."""
        offline_data = {
            "title": "Test Anime",
            "sources": [
                "https://myanimelist.net/anime/123",
                "https://anilist.co/anime/456",
            ],
        }
        result = await helper.find_anisearch_url(offline_data)
        assert result is None

    @pytest.mark.asyncio
    async def test_find_anisearch_url_no_sources(self, helper):
        """Test finding AniSearch URL when sources list is empty."""
        offline_data = {"title": "Test Anime", "sources": []}
        result = await helper.find_anisearch_url(offline_data)
        assert result is None

    @pytest.mark.asyncio
    async def test_find_anisearch_url_non_string_sources(self, helper):
        """Test finding AniSearch URL with non-string sources."""
        offline_data = {
            "title": "Test Anime",
            "sources": [
                "https://myanimelist.net/anime/123",
                123,  # Non-string source
                None,  # None source
                "https://www.anisearch.com/anime/456",
            ],
        }
        result = await helper.find_anisearch_url(offline_data)
        assert result == "https://www.anisearch.com/anime/456"

    @pytest.mark.asyncio
    async def test_find_anisearch_url_exception(self, helper):
        """Test exception handling in find_anisearch_url."""
        offline_data = {"sources": None}  # Will cause exception
        result = await helper.find_anisearch_url(offline_data)
        assert result is None

    # ========================================
    # Test: fetch_anime_data
    # ========================================

    @pytest.mark.asyncio
    @patch("src.enrichment.api_helpers.anisearch_helper.fetch_anisearch_anime")
    async def test_fetch_anime_data_success(
        self, mock_fetch, helper, sample_anime_data
    ):
        """Test successful anime data fetching."""
        mock_fetch.return_value = sample_anime_data

        result = await helper.fetch_anime_data(18878)

        assert result == sample_anime_data
        mock_fetch.assert_called_once_with(
            anime_id="18878",
            return_data=True,
            output_path=None,
        )

    @pytest.mark.asyncio
    @patch("src.enrichment.api_helpers.anisearch_helper.fetch_anisearch_anime")
    async def test_fetch_anime_data_no_data(self, mock_fetch, helper):
        """Test anime data fetching when crawler returns None."""
        mock_fetch.return_value = None

        result = await helper.fetch_anime_data(18878)

        assert result is None

    @pytest.mark.asyncio
    @patch("src.enrichment.api_helpers.anisearch_helper.fetch_anisearch_anime")
    async def test_fetch_anime_data_exception(self, mock_fetch, helper):
        """Test exception handling in fetch_anime_data."""
        mock_fetch.side_effect = Exception("Crawler error")

        result = await helper.fetch_anime_data(18878)

        assert result is None

    # ========================================
    # Test: fetch_episode_data
    # ========================================

    @pytest.mark.asyncio
    @patch("src.enrichment.api_helpers.anisearch_helper.fetch_anisearch_episodes")
    async def test_fetch_episode_data_success(
        self, mock_fetch, helper, sample_episode_data
    ):
        """Test successful episode data fetching."""
        mock_fetch.return_value = sample_episode_data

        result = await helper.fetch_episode_data(18878)

        assert result == sample_episode_data
        assert len(result) == 2
        mock_fetch.assert_called_once_with(
            anime_id="18878",
            return_data=True,
            output_path=None,
        )

    @pytest.mark.asyncio
    @patch("src.enrichment.api_helpers.anisearch_helper.fetch_anisearch_episodes")
    async def test_fetch_episode_data_no_data(self, mock_fetch, helper):
        """Test episode data fetching when crawler returns None."""
        mock_fetch.return_value = None

        result = await helper.fetch_episode_data(18878)

        assert result is None

    @pytest.mark.asyncio
    @patch("src.enrichment.api_helpers.anisearch_helper.fetch_anisearch_episodes")
    async def test_fetch_episode_data_empty_list(self, mock_fetch, helper):
        """Test episode data fetching when crawler returns empty list (treated as no data)."""
        mock_fetch.return_value = []

        result = await helper.fetch_episode_data(18878)

        # Empty list is treated as "no data" and returns None
        assert result is None

    @pytest.mark.asyncio
    @patch("src.enrichment.api_helpers.anisearch_helper.fetch_anisearch_episodes")
    async def test_fetch_episode_data_exception(self, mock_fetch, helper):
        """Test exception handling in fetch_episode_data."""
        mock_fetch.side_effect = Exception("Episode crawler error")

        result = await helper.fetch_episode_data(18878)

        assert result is None

    # ========================================
    # Test: fetch_character_data
    # ========================================

    @pytest.mark.asyncio
    @patch("src.enrichment.api_helpers.anisearch_helper.fetch_anisearch_characters")
    async def test_fetch_character_data_success(
        self, mock_fetch, helper, sample_character_data
    ):
        """Test successful character data fetching."""
        mock_fetch.return_value = sample_character_data

        result = await helper.fetch_character_data(18878)

        assert result == sample_character_data
        assert result["total_count"] == 2
        assert len(result["characters"]) == 2
        mock_fetch.assert_called_once_with(
            anime_id="18878",
            return_data=True,
            output_path=None,
        )

    @pytest.mark.asyncio
    @patch("src.enrichment.api_helpers.anisearch_helper.fetch_anisearch_characters")
    async def test_fetch_character_data_no_data(self, mock_fetch, helper):
        """Test character data fetching when crawler returns None."""
        mock_fetch.return_value = None

        result = await helper.fetch_character_data(18878)

        assert result is None

    @pytest.mark.asyncio
    @patch("src.enrichment.api_helpers.anisearch_helper.fetch_anisearch_characters")
    async def test_fetch_character_data_zero_count(self, mock_fetch, helper):
        """Test character data fetching with zero characters."""
        mock_fetch.return_value = {"characters": [], "total_count": 0}

        result = await helper.fetch_character_data(18878)

        assert result["total_count"] == 0
        assert result["characters"] == []

    @pytest.mark.asyncio
    @patch("src.enrichment.api_helpers.anisearch_helper.fetch_anisearch_characters")
    async def test_fetch_character_data_exception(self, mock_fetch, helper):
        """Test exception handling in fetch_character_data."""
        mock_fetch.side_effect = Exception("Character crawler error")

        result = await helper.fetch_character_data(18878)

        assert result is None

    # ========================================
    # Test: fetch_all_data
    # ========================================

    @pytest.mark.asyncio
    @patch("src.enrichment.api_helpers.anisearch_helper.fetch_anisearch_anime")
    @patch("src.enrichment.api_helpers.anisearch_helper.fetch_anisearch_episodes")
    @patch("src.enrichment.api_helpers.anisearch_helper.fetch_anisearch_characters")
    async def test_fetch_all_data_complete(
        self,
        mock_char,
        mock_ep,
        mock_anime,
        helper,
        sample_anime_data,
        sample_episode_data,
        sample_character_data,
    ):
        """Test fetching all data successfully (anime + episodes + characters)."""
        mock_anime.return_value = sample_anime_data
        mock_ep.return_value = sample_episode_data
        mock_char.return_value = sample_character_data

        result = await helper.fetch_all_data(18878)

        assert result is not None
        assert "japanese_title" in result
        assert "episodes" in result
        assert len(result["episodes"]) == 2
        assert "characters" in result
        assert len(result["characters"]) == 2

    @pytest.mark.asyncio
    @patch("src.enrichment.api_helpers.anisearch_helper.fetch_anisearch_anime")
    async def test_fetch_all_data_anime_only(
        self, mock_anime, helper, sample_anime_data
    ):
        """Test fetching all data with episodes and characters disabled."""
        mock_anime.return_value = sample_anime_data

        result = await helper.fetch_all_data(
            18878, include_episodes=False, include_characters=False
        )

        assert result is not None
        assert "japanese_title" in result
        assert "episodes" not in result
        assert "characters" not in result

    @pytest.mark.asyncio
    @patch("src.enrichment.api_helpers.anisearch_helper.fetch_anisearch_anime")
    @patch("src.enrichment.api_helpers.anisearch_helper.fetch_anisearch_episodes")
    async def test_fetch_all_data_with_episodes_only(
        self, mock_ep, mock_anime, helper, sample_anime_data, sample_episode_data
    ):
        """Test fetching all data with episodes but no characters."""
        mock_anime.return_value = sample_anime_data
        mock_ep.return_value = sample_episode_data

        result = await helper.fetch_all_data(
            18878, include_episodes=True, include_characters=False
        )

        assert result is not None
        assert "episodes" in result
        assert len(result["episodes"]) == 2
        assert "characters" not in result

    @pytest.mark.asyncio
    @patch("src.enrichment.api_helpers.anisearch_helper.fetch_anisearch_anime")
    async def test_fetch_all_data_anime_failure(self, mock_anime, helper):
        """Test fetch_all_data when anime data fetch fails."""
        mock_anime.return_value = None

        result = await helper.fetch_all_data(18878)

        assert result is None

    @pytest.mark.asyncio
    @patch("src.enrichment.api_helpers.anisearch_helper.fetch_anisearch_anime")
    @patch("src.enrichment.api_helpers.anisearch_helper.fetch_anisearch_episodes")
    async def test_fetch_all_data_episode_failure_graceful(
        self, mock_ep, mock_anime, helper, sample_anime_data
    ):
        """Test fetch_all_data continues when episode fetch fails (graceful degradation)."""
        mock_anime.return_value = sample_anime_data
        mock_ep.side_effect = Exception("Episode fetch failed")

        result = await helper.fetch_all_data(18878, include_episodes=True)

        # Should still return anime data even though episodes failed
        assert result is not None
        assert "japanese_title" in result
        assert "episodes" not in result  # Episodes not added due to failure

    @pytest.mark.asyncio
    @patch("src.enrichment.api_helpers.anisearch_helper.fetch_anisearch_anime")
    @patch("src.enrichment.api_helpers.anisearch_helper.fetch_anisearch_characters")
    async def test_fetch_all_data_character_failure_graceful(
        self, mock_char, mock_anime, helper, sample_anime_data
    ):
        """Test fetch_all_data continues when character fetch fails (graceful degradation)."""
        mock_anime.return_value = sample_anime_data
        mock_char.side_effect = Exception("Character fetch failed")

        result = await helper.fetch_all_data(18878, include_characters=True)

        # Should still return anime data even though characters failed
        assert result is not None
        assert "japanese_title" in result
        assert "characters" not in result  # Characters not added due to failure

    @pytest.mark.asyncio
    @patch("src.enrichment.api_helpers.anisearch_helper.fetch_anisearch_anime")
    @patch("src.enrichment.api_helpers.anisearch_helper.fetch_anisearch_episodes")
    async def test_fetch_all_data_episode_returns_none(
        self, mock_ep, mock_anime, helper, sample_anime_data
    ):
        """Test fetch_all_data when episode fetch returns None."""
        mock_anime.return_value = sample_anime_data
        mock_ep.return_value = None

        result = await helper.fetch_all_data(18878, include_episodes=True)

        assert result is not None
        assert "episodes" not in result  # Episodes not added when None

    @pytest.mark.asyncio
    @patch("src.enrichment.api_helpers.anisearch_helper.fetch_anisearch_anime")
    @patch("src.enrichment.api_helpers.anisearch_helper.fetch_anisearch_characters")
    async def test_fetch_all_data_character_returns_none(
        self, mock_char, mock_anime, helper, sample_anime_data
    ):
        """Test fetch_all_data when character fetch returns None."""
        mock_anime.return_value = sample_anime_data
        mock_char.return_value = None

        result = await helper.fetch_all_data(18878, include_characters=True)

        assert result is not None
        assert "characters" not in result  # Characters not added when None

    @pytest.mark.asyncio
    @patch("src.enrichment.api_helpers.anisearch_helper.fetch_anisearch_anime")
    async def test_fetch_all_data_exception(self, mock_anime, helper):
        """Test exception handling in fetch_all_data outer try-except (lines 219-221)."""
        mock_anime.side_effect = Exception("General error")

        result = await helper.fetch_all_data(18878)

        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_all_data_episode_method_raises_exception(
        self, helper, sample_anime_data
    ):
        """Test exception handling when fetch_episode_data itself raises (lines 198-199)."""
        with patch(
            "src.enrichment.api_helpers.anisearch_helper.fetch_anisearch_anime"
        ) as mock_anime:
            mock_anime.return_value = sample_anime_data

            # Make the helper's fetch_episode_data method raise an exception
            with patch.object(
                helper,
                "fetch_episode_data",
                side_effect=Exception("Episode method error"),
            ):
                result = await helper.fetch_all_data(18878, include_episodes=True)

                # Should still return anime data despite episode method exception
                assert result is not None
                assert "japanese_title" in result
                assert "episodes" not in result

    @pytest.mark.asyncio
    async def test_fetch_all_data_character_method_raises_exception(
        self, helper, sample_anime_data
    ):
        """Test exception handling when fetch_character_data itself raises (lines 212-213)."""
        with patch(
            "src.enrichment.api_helpers.anisearch_helper.fetch_anisearch_anime"
        ) as mock_anime:
            mock_anime.return_value = sample_anime_data

            # Make the helper's fetch_character_data method raise an exception
            with patch.object(
                helper,
                "fetch_character_data",
                side_effect=Exception("Character method error"),
            ):
                result = await helper.fetch_all_data(18878, include_characters=True)

                # Should still return anime data despite character method exception
                assert result is not None
                assert "japanese_title" in result
                assert "characters" not in result

    @pytest.mark.asyncio
    async def test_fetch_all_data_outer_exception_handler(
        self, helper, sample_anime_data
    ):
        """Test outer exception handler when exception occurs outside inner try-except blocks (lines 219-221)."""
        with patch(
            "src.enrichment.api_helpers.anisearch_helper.fetch_anisearch_anime"
        ) as mock_anime:
            mock_anime.return_value = sample_anime_data

            with patch(
                "src.enrichment.api_helpers.anisearch_helper.logger"
            ) as mock_logger:
                mock_logger.info.side_effect = [
                    None,
                    None,
                    None,
                    RuntimeError("Logger error"),
                ]

                result = await helper.fetch_all_data(
                    18878, include_episodes=False, include_characters=False
                )

                assert result is None

    # ========================================
    # Test: close
    # ========================================

    @pytest.mark.asyncio
    async def test_close(self, helper):
        """Test close method (should be no-op for stateless crawlers)."""
        # Should not raise any exceptions
        await helper.close()
        assert True  # If we get here, close() worked

    # ========================================
    # Edge Cases and Integration Tests
    # ========================================

    @pytest.mark.asyncio
    async def test_multiple_sequential_fetches(self, helper):
        """Test multiple sequential fetch operations."""
        with patch(
            "src.enrichment.api_helpers.anisearch_helper.fetch_anisearch_anime"
        ) as mock:
            mock.return_value = {"japanese_title": "Test 1"}
            result1 = await helper.fetch_anime_data(1)

            mock.return_value = {"japanese_title": "Test 2"}
            result2 = await helper.fetch_anime_data(2)

            assert result1["japanese_title"] == "Test 1"
            assert result2["japanese_title"] == "Test 2"

    @pytest.mark.asyncio
    async def test_large_anime_id(self, helper):
        """Test with very large anime ID."""
        with patch(
            "src.enrichment.api_helpers.anisearch_helper.fetch_anisearch_anime"
        ) as mock:
            mock.return_value = {"japanese_title": "Test"}
            result = await helper.fetch_anime_data(999999999)

            assert result is not None
            mock.assert_called_with(
                anime_id="999999999",
                return_data=True,
                output_path=None,
            )

    @pytest.mark.asyncio
    async def test_special_characters_in_url(self, helper):
        """Test URL extraction with special characters."""
        url = "https://www.anisearch.com/anime/18878,dan-da-dan-%E3%83%80%E3%83%B3%E3%83%80%E3%83%80%E3%83%B3"
        result = await helper.extract_anisearch_id_from_url(url)
        assert result == 18878


# --- Tests for context manager protocol ---


@pytest.mark.asyncio
async def test_context_manager_protocol():
    """Test AniSearchEnrichmentHelper implements async context manager protocol."""
    from src.enrichment.api_helpers.anisearch_helper import AniSearchEnrichmentHelper

    async with AniSearchEnrichmentHelper() as helper:
        assert helper is not None
        assert isinstance(helper, AniSearchEnrichmentHelper)
    # Should exit cleanly


@pytest.mark.asyncio
async def test_context_manager_close_method_exists():
    """Test that close() method exists (no-op for AniSearch crawlers)."""
    from src.enrichment.api_helpers.anisearch_helper import AniSearchEnrichmentHelper

    helper = AniSearchEnrichmentHelper()
    # Should have close() method
    assert hasattr(helper, "close")
    assert callable(helper.close)
    # Should be safe to call
    await helper.close()


@pytest.mark.asyncio
async def test_context_manager_cleanup_on_exception():
    """Test that context manager cleans up even when exception occurs."""
    from src.enrichment.api_helpers.anisearch_helper import AniSearchEnrichmentHelper

    with pytest.raises(ValueError, match="Test error"):
        async with AniSearchEnrichmentHelper():
            raise ValueError("Test error")
    # If we get here, cleanup happened correctly
