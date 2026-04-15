"""
Tests for ApiFetcher context manager protocol.
"""

import os
import tempfile
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from enrichment.exceptions import (
    AniListGraphQLError,
    ServiceBlockedError,
    ServiceNetworkError,
    ServiceRateLimitedError,
)
from enrichment.programmatic.api_fetcher import ApiFetcher


class TestApiFetcherContextManager:
    """Test async context manager protocol for ApiFetcher."""

    @pytest.mark.asyncio
    async def test_context_manager_returns_self(self):
        """Test that __aenter__ returns the fetcher without initializing helpers."""
        async with ApiFetcher() as fetcher:
            assert fetcher is not None
            assert isinstance(fetcher, ApiFetcher)
            # Helpers are NOT initialized until fetch_all_data is called
            assert fetcher._helpers == {}

    @pytest.mark.asyncio
    async def test_context_manager_closes_all_helpers(self):
        """Test that __aexit__ closes all helper resources in _helpers."""
        fetcher = ApiFetcher()

        mock_anilist = AsyncMock()
        mock_kitsu = AsyncMock()

        fetcher._helpers["anilist"] = mock_anilist
        fetcher._helpers["kitsu"] = mock_kitsu

        async with fetcher:
            pass

        mock_anilist.close.assert_awaited_once()
        mock_kitsu.close.assert_awaited_once()
        assert fetcher._helpers == {}

    @pytest.mark.asyncio
    async def test_context_manager_cleanup_on_exception(self):
        """Test that context manager cleans up even when exception occurs."""
        fetcher = ApiFetcher()

        mock_anilist = AsyncMock()
        fetcher._helpers["anilist"] = mock_anilist

        with pytest.raises(ValueError, match="Test error"):
            async with fetcher:
                raise ValueError("Test error")

        mock_anilist.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_context_manager_handles_empty_helpers(self):
        """Test that __aexit__ handles empty _helpers dict gracefully."""
        fetcher = ApiFetcher()
        # No helpers populated — should not raise
        async with fetcher:
            pass

    @pytest.mark.asyncio
    async def test_no_cleanup_method_exists(self):
        """Test that cleanup() method was removed (should not exist)."""
        fetcher = ApiFetcher()

        # cleanup() method should NOT exist
        assert not hasattr(fetcher, "cleanup")


class TestFetchMALViaService:
    """Test MAL fetching through the generic _fetch_service dispatcher."""

    @pytest.mark.asyncio
    async def test_mal_service_success(self):
        """MAL helper.fetch_all called with ids/offline_data/temp_dir."""
        fetcher = ApiFetcher()
        mock_helper = AsyncMock()
        expected = {"anime": {"title": "One Piece"}, "episodes": [], "characters": []}
        mock_helper.fetch_all = AsyncMock(return_value=expected)
        fetcher._helpers["mal"] = mock_helper

        ids = {"mal_url": "https://myanimelist.net/anime/21"}
        offline = {"title": "One Piece"}

        with tempfile.TemporaryDirectory() as temp_dir:
            result = await fetcher._fetch_service("mal", mock_helper, ids, offline, temp_dir)

        assert result == expected
        assert "mal" in fetcher.api_timings
        mock_helper.fetch_all.assert_awaited_once_with(ids, offline, temp_dir)

    @pytest.mark.asyncio
    async def test_mal_service_returns_none_on_failure(self):
        """helper.fetch_all returning None propagates as None (no error recorded)."""
        fetcher = ApiFetcher()
        mock_helper = AsyncMock()
        mock_helper.fetch_all = AsyncMock(return_value=None)

        result = await fetcher._fetch_service("mal", mock_helper, {}, {}, None)

        assert result is None
        assert "mal" in fetcher.api_timings  # timing still recorded
        assert "mal" not in fetcher.api_errors

    @pytest.mark.asyncio
    async def test_mal_service_exception_recorded(self):
        """Exception from helper.fetch_all is caught and recorded in api_errors."""
        fetcher = ApiFetcher()
        mock_helper = AsyncMock()
        mock_helper.fetch_all = AsyncMock(side_effect=Exception("Network error"))

        result = await fetcher._fetch_service("mal", mock_helper, {}, {}, None)

        assert result is None
        assert "mal" in fetcher.api_errors
        assert "Network error" in fetcher.api_errors["mal"]


class TestShouldInclude:
    """Tests for _should_include static method."""

    def test_only_includes_matching(self):
        assert ApiFetcher._should_include("mal", ["mal", "kitsu"], None) is True

    def test_only_excludes_non_matching(self):
        assert ApiFetcher._should_include("anidb", ["mal", "kitsu"], None) is False

    def test_skip_excludes_matching(self):
        assert ApiFetcher._should_include("mal", None, ["mal"]) is False

    def test_skip_includes_non_matching(self):
        assert ApiFetcher._should_include("kitsu", None, ["mal"]) is True

    def test_no_filter_includes_all(self):
        assert ApiFetcher._should_include("anything", None, None) is True


class TestInitializeHelpers:
    """Tests for initialize_helpers."""

    @pytest.mark.asyncio
    async def test_initializes_all_registry_services_when_no_filter(self):
        fetcher = ApiFetcher()
        with patch("http_cache.instance.http_cache_manager") as mock_cache:
            mock_cache.get_aiohttp_session.return_value = MagicMock()
            await fetcher.initialize_helpers()
        assert set(fetcher._helpers.keys()) == set(ApiFetcher._REGISTRY.keys())

    @pytest.mark.asyncio
    async def test_only_filter_initializes_subset(self):
        fetcher = ApiFetcher()
        with patch("http_cache.instance.http_cache_manager") as mock_cache:
            mock_cache.get_aiohttp_session.return_value = MagicMock()
            await fetcher.initialize_helpers(only_services=["kitsu", "anidb"])
        assert set(fetcher._helpers.keys()) == {"kitsu", "anidb"}

    @pytest.mark.asyncio
    async def test_skip_filter_excludes_service(self):
        fetcher = ApiFetcher()
        with patch("http_cache.instance.http_cache_manager") as mock_cache:
            mock_cache.get_aiohttp_session.return_value = MagicMock()
            await fetcher.initialize_helpers(skip_services=["anidb", "anisearch"])
        assert "anidb" not in fetcher._helpers
        assert "anisearch" not in fetcher._helpers
        assert "kitsu" in fetcher._helpers

    @pytest.mark.asyncio
    async def test_idempotent_does_not_reinitialize(self):
        fetcher = ApiFetcher()
        mock_helper = MagicMock()
        fetcher._helpers["kitsu"] = mock_helper
        with patch("http_cache.instance.http_cache_manager") as mock_cache:
            mock_cache.get_aiohttp_session.return_value = MagicMock()
            await fetcher.initialize_helpers(only_services=["kitsu"])
        assert fetcher._helpers["kitsu"] is mock_helper  # not replaced


class TestFetchAllData:
    """Tests for fetch_all_data orchestrator."""

    @pytest.mark.asyncio
    async def test_only_services_logs_and_builds_tasks(self):
        fetcher = ApiFetcher()
        fetcher._helpers["kitsu"] = AsyncMock()
        fetcher._helpers["kitsu"].fetch_all = AsyncMock(return_value={"title": "One Piece"})

        ids = {"kitsu_id": "12"}
        offline = {"title": "One Piece", "sources": []}

        with patch.object(fetcher, "initialize_helpers", new=AsyncMock()):
            with patch.object(fetcher, "_gather", new=AsyncMock(return_value={"kitsu": {"title": "One Piece"}})):
                with patch.object(fetcher, "_log_performance_metrics"):
                    result = await fetcher.fetch_all_data(
                        ids, offline, only_services=["kitsu"]
                    )

        assert result == {"kitsu": {"title": "One Piece"}}

    @pytest.mark.asyncio
    async def test_skip_services_logs_and_excludes(self):
        fetcher = ApiFetcher()
        ids = {"kitsu_id": "12", "anidb_id": "1"}
        offline = {"title": "Test", "sources": []}

        with patch.object(fetcher, "initialize_helpers", new=AsyncMock()):
            with patch.object(fetcher, "_gather", new=AsyncMock(return_value={})) as mock_gather:
                with patch.object(fetcher, "_log_performance_metrics"):
                    await fetcher.fetch_all_data(ids, offline, skip_services=["kitsu"])

        # kitsu should not appear in tasks
        tasks_passed = mock_gather.call_args[0][0]
        task_names = [name for name, _ in tasks_passed]
        assert "kitsu" not in task_names

    @pytest.mark.asyncio
    async def test_all_registered_services_build_tasks(self):
        """All registered services appear in gather tasks when helpers are pre-loaded."""
        fetcher = ApiFetcher()
        # Inject mock helpers for all registered services
        for name in fetcher._REGISTRY:
            fetcher._helpers[name] = AsyncMock()
            fetcher._helpers[name].fetch_all = AsyncMock(return_value={})

        ids = {"mal_url": "https://myanimelist.net/anime/21"}
        offline = {"title": "One Piece", "sources": []}

        with patch.object(fetcher, "initialize_helpers", new=AsyncMock()):
            with patch.object(fetcher, "_gather", new=AsyncMock(return_value={})) as mock_gather:
                with patch.object(fetcher, "_log_performance_metrics"):
                    await fetcher.fetch_all_data(ids, offline)

        task_names = [name for name, _ in mock_gather.call_args[0][0]]
        for name in fetcher._REGISTRY:
            assert name in task_names

    @pytest.mark.asyncio
    async def test_temp_dir_forwarded_to_fetch_service(self):
        """temp_dir is passed through to _fetch_service for each helper."""
        fetcher = ApiFetcher()
        mock_helper = AsyncMock()
        mock_helper.fetch_all = AsyncMock(return_value={"id": 1})
        fetcher._helpers["anidb"] = mock_helper

        ids = {"anidb_url": "https://anidb.net/anime/1"}
        offline = {"title": "Test", "sources": []}

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(fetcher, "initialize_helpers", new=AsyncMock()):
                with patch.object(fetcher, "_log_performance_metrics"):
                    await fetcher.fetch_all_data(ids, offline, temp_dir=temp_dir)

        mock_helper.fetch_all.assert_awaited_once_with(ids, offline, temp_dir)


class TestAnilistSyncWrapper:
    """Tests for _anilist_sync_wrapper (runs helper.fetch_all in a fresh event loop)."""

    def test_success_returns_result(self):
        """Wrapper creates new event loop, runs helper.fetch_all, returns result."""
        fetcher = ApiFetcher()
        mock_helper = MagicMock()
        expected = {"title": "One Piece"}

        with patch("asyncio.new_event_loop") as mock_new_loop:
            with patch("asyncio.set_event_loop"):
                mock_loop = MagicMock()
                mock_loop.run_until_complete.return_value = expected
                mock_new_loop.return_value = mock_loop

                result = fetcher._anilist_sync_wrapper(mock_helper, {}, {}, None)

        assert result == expected
        mock_loop.close.assert_called_once()

    def test_exception_propagates_and_loop_closes(self):
        """Exceptions from run_until_complete propagate; loop is always closed."""
        fetcher = ApiFetcher()
        mock_helper = MagicMock()

        with patch("asyncio.new_event_loop") as mock_new_loop:
            with patch("asyncio.set_event_loop"):
                mock_loop = MagicMock()
                mock_loop.run_until_complete.side_effect = RuntimeError("network fail")
                mock_new_loop.return_value = mock_loop

                with pytest.raises(RuntimeError, match="network fail"):
                    fetcher._anilist_sync_wrapper(mock_helper, {}, {}, None)

        mock_loop.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_anilist_via_executor_calls_wrapper(self):
        """_fetch_anilist_via_executor runs _anilist_sync_wrapper in a thread."""
        fetcher = ApiFetcher()
        mock_helper = AsyncMock()

        with patch.object(fetcher, "_anilist_sync_wrapper", return_value={"title": "Test"}) as mock_wrapper:
            result = await fetcher._fetch_anilist_via_executor(mock_helper, {}, {}, None)

        assert result == {"title": "Test"}
        mock_wrapper.assert_called_once_with(mock_helper, {}, {}, None)


class TestFetchService:
    """Tests for _fetch_service — the generic per-service dispatch."""

    @pytest.mark.asyncio
    async def test_success_records_timing(self):
        """Successful fetch records timing and returns result."""
        fetcher = ApiFetcher()
        mock_helper = AsyncMock()
        mock_helper.fetch_all = AsyncMock(return_value={"title": "One Piece"})

        ids = {"kitsu_url": "https://kitsu.io/anime/one-piece"}
        result = await fetcher._fetch_service("kitsu", mock_helper, ids, {}, None)

        assert result == {"title": "One Piece"}
        assert "kitsu" in fetcher.api_timings
        mock_helper.fetch_all.assert_awaited_once_with(ids, {}, None)

    @pytest.mark.asyncio
    async def test_exception_records_error_and_returns_none(self):
        """Exception from helper is caught, recorded, and None is returned."""
        fetcher = ApiFetcher()
        mock_helper = AsyncMock()
        mock_helper.fetch_all = AsyncMock(side_effect=RuntimeError("boom"))

        result = await fetcher._fetch_service("kitsu", mock_helper, {}, {}, None)

        assert result is None
        assert "kitsu" in fetcher.api_errors
        assert "boom" in fetcher.api_errors["kitsu"]

    @pytest.mark.asyncio
    async def test_passes_temp_dir_to_helper(self):
        """temp_dir forwarded to helper.fetch_all."""
        fetcher = ApiFetcher()
        mock_helper = AsyncMock()
        mock_helper.fetch_all = AsyncMock(return_value={"id": 1})
        ids = {"anidb_url": "https://anidb.net/anime/1"}

        with tempfile.TemporaryDirectory() as tmp:
            await fetcher._fetch_service("anidb", mock_helper, ids, {}, tmp)
            mock_helper.fetch_all.assert_awaited_once_with(ids, {}, tmp)

    @pytest.mark.asyncio
    async def test_anilist_uses_executor(self):
        """AniList service routes through executor (separate event loop)."""
        fetcher = ApiFetcher()
        mock_helper = AsyncMock()
        mock_helper.fetch_all = AsyncMock(return_value={"title": "One Piece"})

        ids = {"anilist_url": "https://anilist.co/anime/21"}
        result = await fetcher._fetch_service("anilist", mock_helper, ids, {}, None)

        # Result comes back (executor path still calls helper.fetch_all)
        assert result == {"title": "One Piece"}
        assert "anilist" in fetcher.api_timings


class TestGather:
    """Tests for _gather."""

    @pytest.mark.asyncio
    async def test_all_success(self):
        fetcher = ApiFetcher()

        async def coro_a():
            return {"a": 1}

        async def coro_b():
            return {"b": 2}

        result = await fetcher._gather([("a", coro_a()), ("b", coro_b())])
        assert result == {"a": {"a": 1}, "b": {"b": 2}}

    @pytest.mark.asyncio
    async def test_exception_recorded_in_errors(self):
        fetcher = ApiFetcher()

        async def coro_good():
            return {"ok": True}

        async def coro_bad():
            raise RuntimeError("api down")

        result = await fetcher._gather([("good", coro_good()), ("bad", coro_bad())])
        assert result["good"] == {"ok": True}
        assert result["bad"] is None
        assert "bad" in fetcher.api_errors
        assert "api down" in fetcher.api_errors["bad"]

    @pytest.mark.asyncio
    async def test_none_result_logged_as_warning(self):
        fetcher = ApiFetcher()

        async def coro_none():
            return None

        result = await fetcher._gather([("svc", coro_none())])
        assert result["svc"] is None
        assert "svc" not in fetcher.api_errors


class TestLogPerformanceMetrics:
    """Tests for _log_performance_metrics."""

    def test_logs_timings_and_errors(self):
        fetcher = ApiFetcher()
        fetcher.api_timings = {"mal": 1.5, "kitsu": 2.0}
        fetcher.api_errors = {"anidb": "timeout"}
        # Should not raise
        fetcher._log_performance_metrics(3.5)

    def test_zero_apis_does_not_divide_by_zero(self):
        fetcher = ApiFetcher()
        fetcher.api_timings = {}
        fetcher.api_errors = {}
        fetcher._log_performance_metrics(0.0)

    def test_success_rate_100_when_no_errors(self):
        fetcher = ApiFetcher()
        fetcher.api_timings = {"mal": 1.0, "kitsu": 0.5}
        fetcher.api_errors = {}
        # Should not raise; success rate = 100%
        fetcher._log_performance_metrics(1.5)
