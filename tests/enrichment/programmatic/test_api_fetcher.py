"""
Tests for ParallelAPIFetcher context manager protocol.
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.enrichment.programmatic.api_fetcher import ParallelAPIFetcher


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
        """Test that __aexit__ closes all helper resources."""
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
