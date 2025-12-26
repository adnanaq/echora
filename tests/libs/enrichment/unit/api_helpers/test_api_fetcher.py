"""
Tests for ParallelAPIFetcher context manager protocol.
"""

import json
import os
import tempfile
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from enrichment.programmatic.api_fetcher import ParallelAPIFetcher


class TestParallelAPIFetcherContextManager:
    """Test async context manager protocol for ParallelAPIFetcher."""

    @pytest.mark.asyncio
    async def test_context_manager_initializes_helpers(self):
        """Test that __aenter__ initializes all helpers."""
        async with ParallelAPIFetcher() as fetcher:
            assert fetcher is not None
            assert isinstance(fetcher, ParallelAPIFetcher)
            # Helpers should be initialized
            assert fetcher.anilist_helper is not None
            assert fetcher.kitsu_helper is not None
            assert fetcher.anidb_helper is not None
            assert fetcher.anime_planet_helper is not None
            assert fetcher.anisearch_helper is not None
            assert fetcher.jikan_session is not None

    @pytest.mark.asyncio
    async def test_context_manager_closes_all_helpers(self):
        """Test that __aexit__ closes all helper resources and resets them to None for safe reusability."""
        fetcher = ParallelAPIFetcher()

        # Mock all helpers
        mock_anilist = AsyncMock()
        mock_kitsu = AsyncMock()
        mock_anidb = AsyncMock()
        mock_anime_planet = AsyncMock()
        mock_anisearch = AsyncMock()
        mock_jikan_session = AsyncMock()

        fetcher.anilist_helper = mock_anilist
        fetcher.kitsu_helper = mock_kitsu
        fetcher.anidb_helper = mock_anidb
        fetcher.anime_planet_helper = mock_anime_planet
        fetcher.anisearch_helper = mock_anisearch
        fetcher.jikan_session = mock_jikan_session

        async with fetcher:
            pass

        # All helpers should be closed
        mock_anilist.close.assert_awaited_once()
        mock_kitsu.close.assert_awaited_once()
        mock_anidb.close.assert_awaited_once()
        mock_anime_planet.close.assert_awaited_once()
        mock_anisearch.close.assert_awaited_once()
        mock_jikan_session.close.assert_awaited_once()

        # All helpers should be reset to None for safe reusability
        assert fetcher.anilist_helper is None
        assert fetcher.kitsu_helper is None
        assert fetcher.anidb_helper is None
        assert fetcher.anime_planet_helper is None
        assert fetcher.anisearch_helper is None
        assert fetcher.jikan_session is None

    @pytest.mark.asyncio
    async def test_context_manager_cleanup_on_exception(self):
        """Test that context manager cleans up even when exception occurs."""
        fetcher = ParallelAPIFetcher()

        # Mock helpers
        mock_anilist = AsyncMock()
        mock_jikan = AsyncMock()
        fetcher.anilist_helper = mock_anilist
        fetcher.jikan_session = mock_jikan

        with pytest.raises(ValueError, match="Test error"):
            async with fetcher:
                raise ValueError("Test error")

        # Helpers should still be closed despite exception
        mock_anilist.close.assert_awaited_once()
        mock_jikan.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_context_manager_handles_none_helpers(self):
        """Test that __aexit__ handles None helpers gracefully."""
        fetcher = ParallelAPIFetcher()

        # Set all helpers to None
        fetcher.anilist_helper = None
        fetcher.kitsu_helper = None
        fetcher.anidb_helper = None
        fetcher.anime_planet_helper = None
        fetcher.anisearch_helper = None
        fetcher.jikan_session = None

        # Should not raise any errors
        async with fetcher:
            pass

    @pytest.mark.asyncio
    async def test_no_cleanup_method_exists(self):
        """Test that cleanup() method was removed (should not exist)."""
        fetcher = ParallelAPIFetcher()

        # cleanup() method should NOT exist
        assert not hasattr(fetcher, "cleanup")


class TestFetchJikanComplete:
    """Test _fetch_jikan_complete method with various scenarios."""

    @pytest.fixture
    def mock_anime_response(self) -> dict[str, Any]:
        """Mock Jikan anime full response."""
        return {
            "data": {
                "mal_id": 1,
                "title": "Test Anime",
                "episodes": 12,
                "synopsis": "Test synopsis",
            }
        }

    @pytest.fixture
    def mock_characters_response(self) -> dict[str, Any]:
        """Mock Jikan characters response."""
        return {
            "data": [
                {
                    "character": {
                        "mal_id": 1,
                        "name": "Character 1",
                    },
                    "role": "Main",
                }
            ]
        }

    @pytest.fixture
    def offline_data(self) -> dict[str, Any]:
        """Mock offline data for fallback."""
        return {
            "title": "Test Anime",
            "episodes": 12,
        }

    @pytest.mark.asyncio
    async def test_fetch_jikan_complete_without_temp_dir(
        self,
        mock_anime_response: dict[str, Any],
        mock_characters_response: dict[str, Any],
        offline_data: dict[str, Any],
    ):
        """Test _fetch_jikan_complete with temp_dir=None (bug scenario)."""
        fetcher = ParallelAPIFetcher()
        fetcher.jikan_session = AsyncMock()

        # Mock the _fetch_jikan_async method to return our test data
        async def mock_fetch(url: str):
            if "full" in url:
                return mock_anime_response
            elif "characters" in url:
                return mock_characters_response
            return None

        with patch.object(fetcher, "_fetch_jikan_async", side_effect=mock_fetch):
            # This should NOT raise UnboundLocalError anymore
            result = await fetcher._fetch_jikan_complete(
                "1", offline_data, temp_dir=None
            )

        # Verify result structure
        assert result is not None
        assert "anime" in result
        assert "episodes" in result
        assert "characters" in result
        assert result["anime"]["mal_id"] == 1
        assert result["anime"]["title"] == "Test Anime"
        # Episodes should be empty list when temp_dir is None (no file I/O)
        assert result["episodes"] == []
        # Characters should use basic data from API
        assert len(result["characters"]) == 1
        assert result["characters"][0]["character"]["mal_id"] == 1

    @pytest.mark.asyncio
    async def test_fetch_jikan_complete_with_temp_dir(
        self,
        mock_anime_response: dict[str, Any],
        mock_characters_response: dict[str, Any],
        offline_data: dict[str, Any],
    ):
        """Test _fetch_jikan_complete with temp_dir provided (normal scenario)."""
        fetcher = ParallelAPIFetcher()
        fetcher.jikan_session = AsyncMock()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock the _fetch_jikan_async method
            async def mock_fetch(url: str):
                if "full" in url:
                    return mock_anime_response
                elif "characters" in url:
                    return mock_characters_response
                return None

            # Mock JikanDetailedFetcher to avoid actual API calls
            with patch.object(fetcher, "_fetch_jikan_async", side_effect=mock_fetch):
                with patch(
                    "enrichment.programmatic.api_fetcher.JikanDetailedFetcher"
                ) as mock_fetcher_class:
                    # Mock the fetch_detailed_data method to create output files
                    async def mock_fetch_detailed(input_file: str, output_file: str):
                        # Create mock output files
                        if "episodes" in output_file:
                            with open(output_file, "w") as f:
                                json.dump([{"mal_id": 1, "title": "Episode 1"}], f)
                        elif "characters" in output_file:
                            with open(output_file, "w") as f:
                                json.dump(
                                    [{"mal_id": 1, "name": "Detailed Character"}], f
                                )

                    mock_instance = AsyncMock()
                    mock_instance.fetch_detailed_data = AsyncMock(
                        side_effect=mock_fetch_detailed
                    )
                    mock_fetcher_class.return_value = mock_instance

                    result = await fetcher._fetch_jikan_complete(
                        "1", offline_data, temp_dir=temp_dir
                    )

            # Verify result structure
            assert result is not None
            assert "anime" in result
            assert "episodes" in result
            assert "characters" in result
            assert result["anime"]["mal_id"] == 1
            # Episodes should be loaded from file
            assert len(result["episodes"]) == 1
            assert result["episodes"][0]["mal_id"] == 1
            # Characters should be loaded from file
            assert len(result["characters"]) == 1
            assert result["characters"][0]["mal_id"] == 1

            # Verify jikan.json was saved
            jikan_file = os.path.join(temp_dir, "jikan.json")
            assert os.path.exists(jikan_file)
            with open(jikan_file) as f:
                saved_data = json.load(f)
                assert saved_data["data"]["mal_id"] == 1

    @pytest.mark.asyncio
    async def test_fetch_jikan_complete_with_no_episodes(
        self, mock_characters_response: dict[str, Any], offline_data: dict[str, Any]
    ):
        """Test _fetch_jikan_complete with anime that has no episodes."""
        fetcher = ParallelAPIFetcher()
        fetcher.jikan_session = AsyncMock()

        # Mock anime response with no episodes
        anime_response_no_episodes = {
            "data": {
                "mal_id": 2,
                "title": "Movie Anime",
                "episodes": None,  # Ongoing or unknown
            }
        }

        async def mock_fetch(url: str):
            if "full" in url:
                return anime_response_no_episodes
            elif "characters" in url:
                return mock_characters_response
            return None

        with patch.object(fetcher, "_fetch_jikan_async", side_effect=mock_fetch):
            result = await fetcher._fetch_jikan_complete(
                "2", offline_data, temp_dir=None
            )

        # Should use offline_data for episode count
        assert result is not None
        assert result["anime"]["episodes"] is None
        # No episodes should be fetched
        assert result["episodes"] == []

    @pytest.mark.asyncio
    async def test_fetch_jikan_complete_api_failure(self, offline_data: dict[str, Any]):
        """Test _fetch_jikan_complete when API request fails (returns None)."""
        fetcher = ParallelAPIFetcher()
        fetcher.jikan_session = AsyncMock()

        # Mock API failure (returns None)
        with patch.object(fetcher, "_fetch_jikan_async", return_value=None):
            result = await fetcher._fetch_jikan_complete(
                "999", offline_data, temp_dir=None
            )

        # Should return None on failure
        assert result is None
        # api_errors should be empty (no exception, just failed fetch)
        assert "jikan" not in fetcher.api_errors

    @pytest.mark.asyncio
    async def test_fetch_jikan_complete_exception_handling(
        self, offline_data: dict[str, Any]
    ):
        """Test _fetch_jikan_complete exception handling."""
        fetcher = ParallelAPIFetcher()
        fetcher.jikan_session = AsyncMock()

        # Mock exception during fetch
        with patch.object(
            fetcher, "_fetch_jikan_async", side_effect=Exception("Network error")
        ):
            result = await fetcher._fetch_jikan_complete(
                "1", offline_data, temp_dir=None
            )

        # Should return None on exception
        assert result is None
        # Error should be logged
        assert "jikan" in fetcher.api_errors
        assert "Network error" in fetcher.api_errors["jikan"]
