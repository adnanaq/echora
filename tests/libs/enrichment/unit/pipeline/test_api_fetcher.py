"""
Tests for ApiFetcher context manager protocol.
"""

import os
import tempfile
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from enrichment.sources.base.exceptions import (
    AniListGraphQLError,
    ServiceBlockedError,
    ServiceNetworkError,
    ServiceRateLimitedError,
)
from enrichment.pipeline.api_fetcher import ApiFetcher


class TestApiFetcherContextManager:
    """Test async context manager protocol for ApiFetcher."""

    @pytest.mark.asyncio
    async def test_context_manager_returns_self(self):
        async with ApiFetcher() as fetcher:
            assert isinstance(fetcher, ApiFetcher)

    @pytest.mark.asyncio
    async def test_context_manager_does_not_raise(self):
        async with ApiFetcher():
            pass

    @pytest.mark.asyncio
    async def test_context_manager_propagates_exception(self):
        with pytest.raises(ValueError, match="Test error"):
            async with ApiFetcher():
                raise ValueError("Test error")

    @pytest.mark.asyncio
    async def test_fetch_all_data_closes_helpers_after_run(self):
        """Helpers created by _build_helpers are closed after fetch_all_data completes."""
        fetcher = ApiFetcher()
        mock_helper = AsyncMock()
        mock_helper.fetch_all = AsyncMock(return_value={"title": "One Piece"})

        with patch.object(fetcher, "_build_helpers", return_value={"mal": mock_helper}):
            with patch.object(fetcher, "_log_performance_metrics"):
                await fetcher.fetch_all_data({}, {})

        mock_helper.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fetch_all_data_closes_helpers_on_gather_exception(self):
        """Helpers are closed even when _gather raises."""
        fetcher = ApiFetcher()
        mock_helper = AsyncMock()

        with patch.object(fetcher, "_build_helpers", return_value={"mal": mock_helper}):
            with patch.object(fetcher, "_gather", side_effect=RuntimeError("gather boom")):
                with pytest.raises(RuntimeError, match="gather boom"):
                    await fetcher.fetch_all_data({}, {})

        mock_helper.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_cleanup_method_exists(self):
        assert not hasattr(ApiFetcher(), "cleanup")


class TestFetchMALViaService:
    """Test MAL fetching through the generic _fetch_service dispatcher."""

    @pytest.mark.asyncio
    async def test_mal_service_success(self):
        """MAL helper.fetch_all called with ids/offline_data/temp_dir."""
        fetcher = ApiFetcher()
        mock_helper = AsyncMock()
        expected = {"anime": {"title": "One Piece"}, "episodes": [], "characters": []}
        mock_helper.fetch_all = AsyncMock(return_value=expected)

        ids = {"mal_url": "https://myanimelist.net/anime/21"}
        offline = {"title": "One Piece"}

        with tempfile.TemporaryDirectory() as temp_dir:
            result = await fetcher._fetch_service(
                "mal", mock_helper, ids, offline, temp_dir
            )

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


class TestBuildHelpers:
    """Tests for _build_helpers — per-run helper factory."""

    def test_returns_all_registry_services_when_no_filter(self):
        fetcher = ApiFetcher()
        with patch("http_cache.instance.http_cache_manager"):
            helpers = fetcher._build_helpers()
        assert set(helpers.keys()) == set(ApiFetcher._REGISTRY.keys())

    def test_only_filter_returns_subset(self):
        fetcher = ApiFetcher()
        with patch("http_cache.instance.http_cache_manager"):
            helpers = fetcher._build_helpers(only_services=["kitsu", "anidb"])
        assert set(helpers.keys()) == {"kitsu", "anidb"}

    def test_skip_filter_excludes_service(self):
        fetcher = ApiFetcher()
        with patch("http_cache.instance.http_cache_manager"):
            helpers = fetcher._build_helpers(skip_services=["anidb", "anisearch"])
        assert "anidb" not in helpers
        assert "anisearch" not in helpers
        assert "kitsu" in helpers

    def test_each_call_returns_new_instances(self):
        """Two calls to _build_helpers must not return the same objects."""
        fetcher = ApiFetcher()
        with patch("http_cache.instance.http_cache_manager"):
            h1 = fetcher._build_helpers(only_services=["kitsu"])
            h2 = fetcher._build_helpers(only_services=["kitsu"])
        assert h1["kitsu"] is not h2["kitsu"]


class TestFetchAllData:
    """Tests for fetch_all_data orchestrator."""

    @pytest.mark.asyncio
    async def test_only_services_logs_and_builds_tasks(self):
        fetcher = ApiFetcher()
        mock_kitsu = AsyncMock()
        mock_kitsu.fetch_all = AsyncMock(return_value={"title": "One Piece"})

        ids = {"kitsu_id": "12"}
        offline = {"title": "One Piece", "sources": []}

        with patch.object(fetcher, "_build_helpers", return_value={"kitsu": mock_kitsu}):
            with patch.object(
                fetcher,
                "_gather",
                new=AsyncMock(return_value={"kitsu": {"title": "One Piece"}}),
            ):
                with patch.object(fetcher, "_log_performance_metrics"):
                    result = await fetcher.fetch_all_data(
                        ids, offline, only_services=["kitsu"]
                    )

        assert result == {
            "kitsu": {
                "anime": {"title": "One Piece"},
                "episodes": [],
                "characters": [],
                "extras": {},
            }
        }

    @pytest.mark.asyncio
    async def test_skip_services_logs_and_excludes(self):
        fetcher = ApiFetcher()
        ids = {"kitsu_id": "12", "anidb_id": "1"}
        offline = {"title": "Test", "sources": []}

        mock_anidb = AsyncMock()
        with patch.object(fetcher, "_build_helpers", return_value={"anidb": mock_anidb}):
            with patch.object(
                fetcher, "_gather", new=AsyncMock(return_value={})
            ) as mock_gather:
                with patch.object(fetcher, "_log_performance_metrics"):
                    await fetcher.fetch_all_data(ids, offline, skip_services=["kitsu"])

        task_names = [name for name, _ in mock_gather.call_args[0][0]]
        assert "kitsu" not in task_names

    @pytest.mark.asyncio
    async def test_all_registered_services_build_tasks(self):
        """All registered services appear in gather tasks when _build_helpers returns all."""
        fetcher = ApiFetcher()
        mock_helpers = {name: AsyncMock() for name in fetcher._REGISTRY}
        for h in mock_helpers.values():
            h.fetch_all = AsyncMock(return_value={})

        ids = {"mal_url": "https://myanimelist.net/anime/21"}
        offline = {"title": "One Piece", "sources": []}

        with patch.object(fetcher, "_build_helpers", return_value=mock_helpers):
            with patch.object(
                fetcher, "_gather", new=AsyncMock(return_value={})
            ) as mock_gather:
                with patch.object(fetcher, "_log_performance_metrics"):
                    await fetcher.fetch_all_data(ids, offline)

        task_names = [name for name, _ in mock_gather.call_args[0][0]]
        for name in fetcher._REGISTRY:
            assert name in task_names

    @pytest.mark.asyncio
    async def test_fetch_all_data_normalizes_mixed_helper_payload_shapes(self):
        fetcher = ApiFetcher()
        ids = {"mal_url": "https://myanimelist.net/anime/21", "anisearch_id": "12"}
        offline = {"title": "One Piece", "sources": []}

        mock_mal = AsyncMock()
        mock_mal.fetch_all = AsyncMock(
            return_value={
                "anime": {"title": "One Piece"},
                "episodes": [],
                "characters": [],
                "extras": {},
            }
        )
        mock_anisearch = AsyncMock()
        mock_anisearch.fetch_all = AsyncMock(
            return_value={"title": "One Piece", "episodes": [], "characters": []}
        )

        with patch.object(
            fetcher, "_build_helpers", return_value={"mal": mock_mal, "anisearch": mock_anisearch}
        ):
            with patch.object(fetcher, "_log_performance_metrics"):
                result = await fetcher.fetch_all_data(ids, offline)

        assert result == {
            "mal": {
                "anime": {"title": "One Piece"},
                "episodes": [],
                "characters": [],
                "extras": {},
            },
            "anisearch": {
                "anime": {"title": "One Piece"},
                "episodes": [],
                "characters": [],
                "extras": {},
            },
        }

    @pytest.mark.asyncio
    async def test_fetch_all_data_resets_stale_metrics_each_run(self):
        fetcher = ApiFetcher()
        fetcher.api_timings = {"stale": 9.9}
        fetcher.api_errors = {"broken": "old error"}

        mock_mal = AsyncMock()
        mock_mal.fetch_all = AsyncMock(
            return_value={
                "anime": {"title": "Fresh"},
                "episodes": [],
                "characters": [],
                "extras": {},
            }
        )

        with patch.object(fetcher, "_build_helpers", return_value={"mal": mock_mal}):
            with patch.object(fetcher, "_log_performance_metrics"):
                await fetcher.fetch_all_data(
                    {"mal_url": "https://myanimelist.net/anime/1"},
                    {"title": "Fresh", "sources": []},
                )

        assert "stale" not in fetcher.api_timings
        assert "broken" not in fetcher.api_errors
        assert "mal" in fetcher.api_timings

    @pytest.mark.asyncio
    async def test_temp_dir_forwarded_to_fetch_service(self):
        """temp_dir is passed through to _fetch_service for each helper."""
        fetcher = ApiFetcher()
        mock_helper = AsyncMock()
        mock_helper.fetch_all = AsyncMock(return_value={"id": 1})

        ids = {"anidb_url": "https://anidb.net/anime/1"}
        offline = {"title": "Test", "sources": []}

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(fetcher, "_build_helpers", return_value={"anidb": mock_helper}):
                with patch.object(fetcher, "_log_performance_metrics"):
                    await fetcher.fetch_all_data(ids, offline, temp_dir=temp_dir)

        mock_helper.fetch_all.assert_awaited_once_with(ids, offline, temp_dir)

    @pytest.mark.asyncio
    async def test_fetch_all_data_does_not_reuse_helper_instances_between_runs(self):
        """Each fetch_all_data call gets a fresh set of helper instances."""
        fetcher = ApiFetcher()
        seen_ids: list[int] = []

        def capture_helpers(**_kwargs: object) -> dict[str, AsyncMock]:
            h = AsyncMock()
            h.fetch_all = AsyncMock(return_value={})
            seen_ids.append(id(h))
            return {"mal": h}

        with patch.object(fetcher, "_build_helpers", side_effect=capture_helpers):
            with patch.object(fetcher, "_log_performance_metrics"):
                await fetcher.fetch_all_data({}, {})
                await fetcher.fetch_all_data({}, {})

        assert len(seen_ids) == 2
        assert seen_ids[0] != seen_ids[1]


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
    async def test_anilist_uses_same_path_as_other_services(self):
        """AniList routes through the same fetch_all path as every other service."""
        fetcher = ApiFetcher()
        mock_helper = AsyncMock()
        mock_helper.fetch_all = AsyncMock(return_value={"title": "One Piece"})

        ids = {"anilist_url": "https://anilist.co/anime/21"}
        result = await fetcher._fetch_service("anilist", mock_helper, ids, {}, None)

        assert result == {"title": "One Piece"}
        assert "anilist" in fetcher.api_timings
        mock_helper.fetch_all.assert_awaited_once_with(ids, {}, None)


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
