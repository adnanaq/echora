"""
Tests for ParallelAPIFetcher context manager protocol.
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
from enrichment.programmatic.api_fetcher import ParallelAPIFetcher


class TestParallelAPIFetcherContextManager:
    """Test async context manager protocol for ParallelAPIFetcher."""

    @pytest.mark.asyncio
    async def test_context_manager_returns_self(self):
        """Test that __aenter__ returns the fetcher without initializing helpers."""
        async with ParallelAPIFetcher() as fetcher:
            assert fetcher is not None
            assert isinstance(fetcher, ParallelAPIFetcher)
            # Helpers are NOT initialized until fetch_all_data is called with filters
            assert fetcher._helpers == {}
            assert fetcher.mal_session is None

    @pytest.mark.asyncio
    async def test_context_manager_closes_all_helpers(self):
        """Test that __aexit__ closes all helper resources in _helpers and mal_session."""
        fetcher = ParallelAPIFetcher()

        mock_anilist = AsyncMock()
        mock_kitsu = AsyncMock()
        mock_mal_session = AsyncMock()

        fetcher._helpers["anilist"] = mock_anilist
        fetcher._helpers["kitsu"] = mock_kitsu
        fetcher.mal_session = mock_mal_session

        async with fetcher:
            pass

        mock_anilist.close.assert_awaited_once()
        mock_kitsu.close.assert_awaited_once()
        mock_mal_session.close.assert_awaited_once()
        assert fetcher._helpers == {}
        assert fetcher.mal_session is None

    @pytest.mark.asyncio
    async def test_context_manager_cleanup_on_exception(self):
        """Test that context manager cleans up even when exception occurs."""
        fetcher = ParallelAPIFetcher()

        mock_anilist = AsyncMock()
        mock_mal_session = AsyncMock()
        fetcher._helpers["anilist"] = mock_anilist
        fetcher.mal_session = mock_mal_session

        with pytest.raises(ValueError, match="Test error"):
            async with fetcher:
                raise ValueError("Test error")

        mock_anilist.close.assert_awaited_once()
        mock_mal_session.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_context_manager_handles_empty_helpers(self):
        """Test that __aexit__ handles empty _helpers dict gracefully."""
        fetcher = ParallelAPIFetcher()
        # No helpers populated — should not raise
        async with fetcher:
            pass

    @pytest.mark.asyncio
    async def test_no_cleanup_method_exists(self):
        """Test that cleanup() method was removed (should not exist)."""
        fetcher = ParallelAPIFetcher()

        # cleanup() method should NOT exist
        assert not hasattr(fetcher, "cleanup")


class TestFetchMALComplete:
    """Test _fetch_mal method with various scenarios."""

    @pytest.fixture
    def mock_anime_response(self) -> dict[str, Any]:
        """Mock MAL anime full response."""
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
        """Mock MAL characters response."""
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
    async def test_fetch_mal_without_temp_dir(
        self,
        mock_anime_response: dict[str, Any],
        mock_characters_response: dict[str, Any],
        offline_data: dict[str, Any],
    ):
        """Test _fetch_mal writes streaming JSONL output paths."""
        fetcher = ParallelAPIFetcher()
        fetcher.mal_session = AsyncMock()

        helper = AsyncMock()
        helper.fetch_all = AsyncMock(
            return_value={
                "anime": mock_anime_response["data"],
                "episodes": [],
                "characters": mock_characters_response["data"],
            }
        )

        helper_cm = AsyncMock()
        helper_cm.__aenter__ = AsyncMock(return_value=helper)
        helper_cm.__aexit__ = AsyncMock(return_value=False)

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "enrichment.programmatic.api_fetcher.MalEnrichmentHelper",
                return_value=helper_cm,
            ):
                result = await fetcher._fetch_mal("1", temp_dir=temp_dir)

            # Verify result structure
            assert result is not None
            assert "anime" in result
            assert "episodes" in result
            assert "characters" in result
            assert result["anime"]["mal_id"] == 1
            assert result["anime"]["title"] == "Test Anime"
            assert result["episodes"] == []
            assert len(result["characters"]) == 1
            assert result["characters"][0]["character"]["mal_id"] == 1
            helper.fetch_all.assert_awaited_once_with(
                anime_output_path=os.path.join(temp_dir, "mal_anime.jsonl"),
                episodes_output_path=os.path.join(temp_dir, "mal_episodes.jsonl"),
                characters_output_path=os.path.join(temp_dir, "mal_characters.jsonl"),
            )

    @pytest.mark.asyncio
    async def test_fetch_mal_with_temp_dir(
        self,
        mock_anime_response: dict[str, Any],
        mock_characters_response: dict[str, Any],
        offline_data: dict[str, Any],
    ):
        """Test _fetch_mal with temp_dir provided (normal scenario)."""
        fetcher = ParallelAPIFetcher()
        fetcher.mal_session = AsyncMock()

        with tempfile.TemporaryDirectory() as temp_dir:
            helper = AsyncMock()
            helper.fetch_all = AsyncMock(
                return_value={
                    "anime": mock_anime_response["data"],
                    "episodes": [{"mal_id": 1, "title": "Episode 1"}],
                    "characters": [{"mal_id": 1, "name": "Detailed Character"}],
                }
            )

            helper_cm = AsyncMock()
            helper_cm.__aenter__ = AsyncMock(return_value=helper)
            helper_cm.__aexit__ = AsyncMock(return_value=False)

            with patch(
                "enrichment.programmatic.api_fetcher.MalEnrichmentHelper",
                return_value=helper_cm,
            ):
                result = await fetcher._fetch_mal("1", temp_dir=temp_dir)
            helper.fetch_all.assert_awaited_once_with(
                anime_output_path=os.path.join(temp_dir, "mal_anime.jsonl"),
                episodes_output_path=os.path.join(temp_dir, "mal_episodes.jsonl"),
                characters_output_path=os.path.join(temp_dir, "mal_characters.jsonl"),
            )

            # Verify result structure
            assert result is not None
            assert "anime" in result
            assert "episodes" in result
            assert "characters" in result
            assert result["anime"]["mal_id"] == 1
            # Episodes should be returned from helper (and persisted for debugging)
            assert len(result["episodes"]) == 1
            assert result["episodes"][0]["mal_id"] == 1
            # Characters should be returned from helper (and persisted for debugging)
            assert len(result["characters"]) == 1
            assert result["characters"][0]["mal_id"] == 1

    @pytest.mark.asyncio
    async def test_fetch_mal_with_no_episodes(
        self, mock_characters_response: dict[str, Any], offline_data: dict[str, Any]
    ):
        """Test _fetch_mal with anime that has no episodes."""
        fetcher = ParallelAPIFetcher()
        fetcher.mal_session = AsyncMock()

        # Mock anime response with no episodes
        anime_response_no_episodes = {
            "data": {
                "mal_id": 2,
                "title": "Movie Anime",
                "episodes": None,  # Ongoing or unknown
            }
        }

        helper = AsyncMock()
        helper.fetch_all = AsyncMock(
            return_value={
                "anime": anime_response_no_episodes["data"],
                "episodes": [],
                "characters": mock_characters_response["data"],
            }
        )

        helper_cm = AsyncMock()
        helper_cm.__aenter__ = AsyncMock(return_value=helper)
        helper_cm.__aexit__ = AsyncMock(return_value=False)

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "enrichment.programmatic.api_fetcher.MalEnrichmentHelper",
                return_value=helper_cm,
            ):
                result = await fetcher._fetch_mal("2", temp_dir=temp_dir)

            assert result is not None
            assert result["anime"]["episodes"] is None
            # No episodes should be fetched
            assert result["episodes"] == []
            helper.fetch_all.assert_awaited_once_with(
                anime_output_path=os.path.join(temp_dir, "mal_anime.jsonl"),
                episodes_output_path=os.path.join(temp_dir, "mal_episodes.jsonl"),
                characters_output_path=os.path.join(temp_dir, "mal_characters.jsonl"),
            )

    @pytest.mark.asyncio
    async def test_fetch_mal_api_failure(self, offline_data: dict[str, Any]):
        """Test _fetch_mal when API request fails (returns None)."""
        fetcher = ParallelAPIFetcher()
        fetcher.mal_session = AsyncMock()

        helper = AsyncMock()
        helper.fetch_all = AsyncMock(return_value=None)

        helper_cm = AsyncMock()
        helper_cm.__aenter__ = AsyncMock(return_value=helper)
        helper_cm.__aexit__ = AsyncMock(return_value=False)

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "enrichment.programmatic.api_fetcher.MalEnrichmentHelper",
                return_value=helper_cm,
            ):
                result = await fetcher._fetch_mal("999", temp_dir=temp_dir)

        # Should return None on failure
        assert result is None
        # api_errors should be empty (no exception, just failed fetch)
        assert "mal" not in fetcher.api_errors

    @pytest.mark.asyncio
    async def test_fetch_mal_exception_handling(
        self, offline_data: dict[str, Any]
    ):
        """Test _fetch_mal exception handling."""
        fetcher = ParallelAPIFetcher()
        fetcher.mal_session = AsyncMock()

        helper = AsyncMock()
        helper.fetch_all = AsyncMock(side_effect=Exception("Network error"))

        helper_cm = AsyncMock()
        helper_cm.__aenter__ = AsyncMock(return_value=helper)
        helper_cm.__aexit__ = AsyncMock(return_value=False)

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "enrichment.programmatic.api_fetcher.MalEnrichmentHelper",
                return_value=helper_cm,
            ):
                result = await fetcher._fetch_mal("1", temp_dir=temp_dir)

        # Should return None on exception
        assert result is None
        # Error should be logged
        assert "mal" in fetcher.api_errors
        assert "Network error" in fetcher.api_errors["mal"]


class TestShouldInclude:
    """Tests for _should_include static method."""

    def test_only_includes_matching(self):
        assert ParallelAPIFetcher._should_include("mal", ["mal", "kitsu"], None) is True

    def test_only_excludes_non_matching(self):
        assert ParallelAPIFetcher._should_include("anidb", ["mal", "kitsu"], None) is False

    def test_skip_excludes_matching(self):
        assert ParallelAPIFetcher._should_include("mal", None, ["mal"]) is False

    def test_skip_includes_non_matching(self):
        assert ParallelAPIFetcher._should_include("kitsu", None, ["mal"]) is True

    def test_no_filter_includes_all(self):
        assert ParallelAPIFetcher._should_include("anything", None, None) is True


class TestInitializeHelpers:
    """Tests for initialize_helpers."""

    @pytest.mark.asyncio
    async def test_initializes_all_registry_services_when_no_filter(self):
        fetcher = ParallelAPIFetcher()
        with patch("http_cache.instance.http_cache_manager") as mock_cache:
            mock_cache.get_aiohttp_session.return_value = MagicMock()
            await fetcher.initialize_helpers()
        assert set(fetcher._helpers.keys()) == set(ParallelAPIFetcher._REGISTRY.keys())
        assert fetcher.mal_session is not None

    @pytest.mark.asyncio
    async def test_only_filter_initializes_subset(self):
        fetcher = ParallelAPIFetcher()
        with patch("http_cache.instance.http_cache_manager") as mock_cache:
            mock_cache.get_aiohttp_session.return_value = MagicMock()
            await fetcher.initialize_helpers(only_services=["kitsu", "anidb"])
        assert set(fetcher._helpers.keys()) == {"kitsu", "anidb"}
        assert fetcher.mal_session is None  # mal not in only_services

    @pytest.mark.asyncio
    async def test_skip_filter_excludes_service(self):
        fetcher = ParallelAPIFetcher()
        with patch("http_cache.instance.http_cache_manager") as mock_cache:
            mock_cache.get_aiohttp_session.return_value = MagicMock()
            await fetcher.initialize_helpers(skip_services=["anidb", "anisearch"])
        assert "anidb" not in fetcher._helpers
        assert "anisearch" not in fetcher._helpers
        assert "kitsu" in fetcher._helpers

    @pytest.mark.asyncio
    async def test_idempotent_does_not_reinitialize(self):
        fetcher = ParallelAPIFetcher()
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
        fetcher = ParallelAPIFetcher()
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
        fetcher = ParallelAPIFetcher()
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
    async def test_anidb_task_receives_temp_dir(self):
        """Each helper is responsible for writing its own JSONL; api_fetcher constructs the path."""
        fetcher = ParallelAPIFetcher()
        ids = {"anidb_id": "1"}
        offline = {"title": "Test", "sources": []}

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(fetcher, "initialize_helpers", new=AsyncMock()):
                with patch.object(fetcher, "_fetch_anidb", new=AsyncMock(return_value={})) as mock_anidb:
                    with patch.object(fetcher, "_fetch_animeschedule", new=AsyncMock(return_value={})):
                        with patch.object(fetcher, "_gather", new=AsyncMock(return_value={})):
                            with patch.object(fetcher, "_log_performance_metrics"):
                                await fetcher.fetch_all_data(ids, offline, temp_dir=temp_dir)
        mock_anidb.assert_called_once_with("1", temp_dir)

    @pytest.mark.asyncio
    async def test_all_service_ids_build_tasks(self):
        """All ID-gated task branches (mal, anilist, anime_planet, anisearch) are exercised."""
        fetcher = ParallelAPIFetcher()
        ids = {
            "mal_id": "https://myanimelist.net/anime/21",
            "anilist_id": "21",
            "kitsu_id": "12",
            "anidb_id": "1",
            "anime_planet_slug": "one-piece",
            "anisearch_id": "42",
        }
        offline = {"title": "One Piece", "sources": []}

        with patch.object(fetcher, "initialize_helpers", new=AsyncMock()):
            with patch.object(fetcher, "_fetch_mal", new=AsyncMock(return_value={})):
                with patch.object(fetcher, "_fetch_anilist", new=AsyncMock(return_value={})):
                    with patch.object(fetcher, "_fetch_kitsu", new=AsyncMock(return_value={})):
                        with patch.object(fetcher, "_fetch_anidb", new=AsyncMock(return_value={})):
                            with patch.object(fetcher, "_fetch_anime_planet", new=AsyncMock(return_value={})):
                                with patch.object(fetcher, "_fetch_anisearch", new=AsyncMock(return_value={})):
                                    with patch.object(fetcher, "_fetch_animeschedule", new=AsyncMock(return_value={})):
                                        with patch.object(fetcher, "_gather", new=AsyncMock(return_value={})) as mock_gather:
                                            with patch.object(fetcher, "_log_performance_metrics"):
                                                await fetcher.fetch_all_data(ids, offline)

        task_names = [name for name, _ in mock_gather.call_args[0][0]]
        assert "mal" in task_names
        assert "anilist" in task_names
        assert "kitsu" in task_names
        assert "anidb" in task_names
        assert "anime_planet" in task_names
        assert "anisearch" in task_names
        assert "animeschedule" in task_names

    @pytest.mark.asyncio
    async def test_animeschedule_always_included_unless_filtered(self):
        fetcher = ParallelAPIFetcher()
        offline = {"title": "Test", "sources": []}

        with patch.object(fetcher, "initialize_helpers", new=AsyncMock()):
            with patch.object(fetcher, "_gather", new=AsyncMock(return_value={})) as mock_gather:
                with patch.object(fetcher, "_log_performance_metrics"):
                    await fetcher.fetch_all_data({}, offline)

        tasks_passed = mock_gather.call_args[0][0]
        task_names = [name for name, _ in tasks_passed]
        assert "animeschedule" in task_names


class TestFetchAnilistSync:
    """Tests for _fetch_anilist_sync (synchronous executor wrapper)."""

    def test_success_returns_result(self):
        fetcher = ParallelAPIFetcher()
        mock_result = {"title": "One Piece"}

        with patch(
            "enrichment.api_helpers.anilist_helper.AniListEnrichmentHelper"
        ) as mock_cls:
            mock_helper = MagicMock()
            mock_helper.fetch_all.return_value = (mock_result, [{"name": "Luffy"}])
            mock_helper.close.return_value = None
            mock_cls.return_value = mock_helper

            with patch("asyncio.new_event_loop") as mock_new_loop:
                with patch("asyncio.set_event_loop"):
                    mock_loop = MagicMock()
                    mock_loop.run_until_complete.side_effect = [
                        (mock_result, [{"name": "Luffy"}]),
                        None,
                    ]
                    mock_new_loop.return_value = mock_loop

                    result = fetcher._fetch_anilist_sync("21")

        assert result == mock_result
        assert "anilist" in fetcher.api_timings

    def test_success_with_none_result_logs_warning(self):
        fetcher = ParallelAPIFetcher()

        with patch("asyncio.new_event_loop") as mock_new_loop:
            with patch("asyncio.set_event_loop"):
                mock_loop = MagicMock()
                mock_loop.run_until_complete.side_effect = [(None, []), None]
                mock_new_loop.return_value = mock_loop

                with patch("enrichment.api_helpers.anilist_helper.AniListEnrichmentHelper"):
                    result = fetcher._fetch_anilist_sync("21")

        assert result is None

    def test_rate_limited_error_returns_none(self):
        fetcher = ParallelAPIFetcher()
        with patch("asyncio.new_event_loop") as mock_new_loop:
            with patch("asyncio.set_event_loop"):
                mock_loop = MagicMock()
                mock_loop.run_until_complete.side_effect = ServiceRateLimitedError(service="anilist", attempts=3)
                mock_new_loop.return_value = mock_loop
                with patch("enrichment.api_helpers.anilist_helper.AniListEnrichmentHelper"):
                    result = fetcher._fetch_anilist_sync("21")
        assert result is None
        assert "anilist" in fetcher.api_errors

    def test_blocked_error_returns_none(self):
        fetcher = ParallelAPIFetcher()
        with patch("asyncio.new_event_loop") as mock_new_loop:
            with patch("asyncio.set_event_loop"):
                mock_loop = MagicMock()
                mock_loop.run_until_complete.side_effect = ServiceBlockedError("blocked", service="anilist")
                mock_new_loop.return_value = mock_loop
                with patch("enrichment.api_helpers.anilist_helper.AniListEnrichmentHelper"):
                    result = fetcher._fetch_anilist_sync("21")
        assert result is None
        assert "anilist" in fetcher.api_errors

    def test_graphql_error_returns_none(self):
        fetcher = ParallelAPIFetcher()
        with patch("asyncio.new_event_loop") as mock_new_loop:
            with patch("asyncio.set_event_loop"):
                mock_loop = MagicMock()
                mock_loop.run_until_complete.side_effect = AniListGraphQLError([{"message": "some error"}])
                mock_new_loop.return_value = mock_loop
                with patch("enrichment.api_helpers.anilist_helper.AniListEnrichmentHelper"):
                    result = fetcher._fetch_anilist_sync("21")
        assert result is None
        assert "anilist" in fetcher.api_errors

    def test_network_error_returns_none(self):
        fetcher = ParallelAPIFetcher()
        with patch("asyncio.new_event_loop") as mock_new_loop:
            with patch("asyncio.set_event_loop"):
                mock_loop = MagicMock()
                mock_loop.run_until_complete.side_effect = ServiceNetworkError(service="anilist", cause="timeout")
                mock_new_loop.return_value = mock_loop
                with patch("enrichment.api_helpers.anilist_helper.AniListEnrichmentHelper"):
                    result = fetcher._fetch_anilist_sync("21")
        assert result is None
        assert "anilist" in fetcher.api_errors

    def test_generic_exception_returns_none(self):
        fetcher = ParallelAPIFetcher()
        with patch("asyncio.new_event_loop") as mock_new_loop:
            with patch("asyncio.set_event_loop"):
                mock_loop = MagicMock()
                mock_loop.run_until_complete.side_effect = RuntimeError("unexpected")
                mock_new_loop.return_value = mock_loop
                with patch("enrichment.api_helpers.anilist_helper.AniListEnrichmentHelper"):
                    result = fetcher._fetch_anilist_sync("21")
        assert result is None
        assert "anilist" in fetcher.api_errors

    @pytest.mark.asyncio
    async def test_fetch_anilist_runs_in_executor(self):
        fetcher = ParallelAPIFetcher()
        with patch.object(fetcher, "_fetch_anilist_sync", return_value={"title": "Test"}) as mock_sync:
            result = await fetcher._fetch_anilist("21")
        assert result == {"title": "Test"}
        mock_sync.assert_called_once_with("21", None)


class TestFetchKitsu:
    """Tests for _fetch_kitsu."""

    @pytest.mark.asyncio
    async def test_numeric_id_calls_fetch_all(self):
        fetcher = ParallelAPIFetcher()
        mock_helper = AsyncMock()
        mock_helper.fetch_all = AsyncMock(return_value={"title": "One Piece"})
        fetcher._helpers["kitsu"] = mock_helper

        result = await fetcher._fetch_kitsu("12")

        assert result == {"title": "One Piece"}
        mock_helper.fetch_all.assert_awaited_once_with(12, output_dir=None)
        assert "kitsu" in fetcher.api_timings

    @pytest.mark.asyncio
    async def test_slug_resolves_to_numeric_id(self):
        fetcher = ParallelAPIFetcher()
        mock_helper = AsyncMock()
        mock_helper.fetch_all = AsyncMock(return_value={"title": "One Piece"})
        fetcher._helpers["kitsu"] = mock_helper

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": [{"id": "12"}]})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_aiohttp = MagicMock()
        mock_aiohttp.ClientSession.return_value = mock_session
        mock_aiohttp.ClientTimeout.return_value = MagicMock()

        with patch.dict("sys.modules", {"aiohttp": mock_aiohttp}):
            result = await fetcher._fetch_kitsu("one-piece")

        assert result == {"title": "One Piece"}
        mock_helper.fetch_all.assert_awaited_once_with(12, output_dir=None)

    @pytest.mark.asyncio
    async def test_slug_resolution_non_200_returns_none(self):
        fetcher = ParallelAPIFetcher()
        fetcher._helpers["kitsu"] = AsyncMock()

        mock_response = AsyncMock()
        mock_response.status = 404
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_aiohttp = MagicMock()
        mock_aiohttp.ClientSession.return_value = mock_session
        mock_aiohttp.ClientTimeout.return_value = MagicMock()

        with patch.dict("sys.modules", {"aiohttp": mock_aiohttp}):
            result = await fetcher._fetch_kitsu("bad-slug")

        assert result is None

    @pytest.mark.asyncio
    async def test_slug_resolution_empty_data_returns_none(self):
        fetcher = ParallelAPIFetcher()
        fetcher._helpers["kitsu"] = AsyncMock()

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": []})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        mock_aiohttp = MagicMock()
        mock_aiohttp.ClientSession.return_value = mock_session
        mock_aiohttp.ClientTimeout.return_value = MagicMock()

        with patch.dict("sys.modules", {"aiohttp": mock_aiohttp}):
            result = await fetcher._fetch_kitsu("no-results-slug")

        assert result is None

    @pytest.mark.asyncio
    async def test_helper_not_initialized_raises(self):
        fetcher = ParallelAPIFetcher()
        result = await fetcher._fetch_kitsu("12")
        assert result is None
        assert "kitsu" in fetcher.api_errors

    @pytest.mark.asyncio
    async def test_exception_returns_none(self):
        fetcher = ParallelAPIFetcher()
        mock_helper = AsyncMock()
        mock_helper.fetch_all = AsyncMock(side_effect=RuntimeError("boom"))
        fetcher._helpers["kitsu"] = mock_helper

        result = await fetcher._fetch_kitsu("12")
        assert result is None
        assert "kitsu" in fetcher.api_errors


class TestFetchAnidbAnimePlanetAnisearch:
    """Tests for _fetch_anidb, _fetch_anime_planet, _fetch_anisearch."""

    @pytest.mark.asyncio
    async def test_fetch_anidb_success(self):
        fetcher = ParallelAPIFetcher()
        mock_helper = AsyncMock()
        mock_helper.fetch_all = AsyncMock(return_value={"id": 1})
        fetcher._helpers["anidb"] = mock_helper

        result = await fetcher._fetch_anidb("1")
        assert result == {"id": 1}
        assert "anidb" in fetcher.api_timings

    @pytest.mark.asyncio
    async def test_fetch_anidb_not_initialized(self):
        fetcher = ParallelAPIFetcher()
        result = await fetcher._fetch_anidb("1")
        assert result is None
        assert "anidb" in fetcher.api_errors

    @pytest.mark.asyncio
    async def test_fetch_anidb_exception(self):
        fetcher = ParallelAPIFetcher()
        mock_helper = AsyncMock()
        mock_helper.fetch_all = AsyncMock(side_effect=RuntimeError("fail"))
        fetcher._helpers["anidb"] = mock_helper
        result = await fetcher._fetch_anidb("1")
        assert result is None
        assert "anidb" in fetcher.api_errors

    @pytest.mark.asyncio
    async def test_fetch_anime_planet_success(self):
        fetcher = ParallelAPIFetcher()
        mock_helper = AsyncMock()
        mock_helper.fetch_all = AsyncMock(return_value={"slug": "one-piece"})
        fetcher._helpers["anime_planet"] = mock_helper

        result = await fetcher._fetch_anime_planet({"title": "One Piece"})
        assert result == {"slug": "one-piece"}
        assert "anime_planet" in fetcher.api_timings

    @pytest.mark.asyncio
    async def test_fetch_anime_planet_not_initialized(self):
        fetcher = ParallelAPIFetcher()
        result = await fetcher._fetch_anime_planet({"title": "Test"})
        assert result is None
        assert "anime_planet" in fetcher.api_errors

    @pytest.mark.asyncio
    async def test_fetch_anisearch_success(self):
        fetcher = ParallelAPIFetcher()
        mock_helper = AsyncMock()
        mock_helper.fetch_all = AsyncMock(return_value={"id": 42})
        fetcher._helpers["anisearch"] = mock_helper

        result = await fetcher._fetch_anisearch("42")
        assert result == {"id": 42}
        assert "anisearch" in fetcher.api_timings

    @pytest.mark.asyncio
    async def test_fetch_anisearch_not_initialized(self):
        fetcher = ParallelAPIFetcher()
        result = await fetcher._fetch_anisearch("42")
        assert result is None
        assert "anisearch" in fetcher.api_errors


class TestFetchAnimeschedule:
    """Tests for _fetch_animeschedule."""

    @pytest.mark.asyncio
    async def test_success(self):
        fetcher = ParallelAPIFetcher()
        mock_helper = AsyncMock()
        mock_helper.fetch_all = AsyncMock(return_value={"title": "Test"})
        fetcher._helpers["animeschedule"] = mock_helper

        result = await fetcher._fetch_animeschedule({"title": "Test", "sources": ["https://mal/1"]})
        assert result == {"title": "Test"}
        assert "animeschedule" in fetcher.api_timings

    @pytest.mark.asyncio
    async def test_empty_title_returns_none(self):
        fetcher = ParallelAPIFetcher()
        mock_helper = AsyncMock()
        fetcher._helpers["animeschedule"] = mock_helper

        result = await fetcher._fetch_animeschedule({"title": "", "sources": []})
        assert result is None
        mock_helper.fetch_all.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_not_initialized_returns_none(self):
        fetcher = ParallelAPIFetcher()
        result = await fetcher._fetch_animeschedule({"title": "Test"})
        assert result is None
        assert "animeschedule" in fetcher.api_errors

    @pytest.mark.asyncio
    async def test_exception_returns_none(self):
        fetcher = ParallelAPIFetcher()
        mock_helper = AsyncMock()
        mock_helper.fetch_all = AsyncMock(side_effect=RuntimeError("fail"))
        fetcher._helpers["animeschedule"] = mock_helper

        result = await fetcher._fetch_animeschedule({"title": "Test"})
        assert result is None
        assert "animeschedule" in fetcher.api_errors

    @pytest.mark.asyncio
    async def test_writes_jsonl_when_temp_dir_provided(self):
        fetcher = ParallelAPIFetcher()
        mock_helper = AsyncMock()
        mock_helper.fetch_all = AsyncMock(return_value={"title": "Test"})
        fetcher._helpers["animeschedule"] = mock_helper

        with tempfile.TemporaryDirectory() as temp_dir:
            await fetcher._fetch_animeschedule({"title": "Test", "sources": []}, temp_dir=temp_dir)
            mock_helper.fetch_all.assert_awaited_once_with(
                "Test",
                sources=None,
                output_path=os.path.join(temp_dir, "animeschedule.jsonl"),
            )


class TestGather:
    """Tests for _gather."""

    @pytest.mark.asyncio
    async def test_all_success(self):
        fetcher = ParallelAPIFetcher()

        async def coro_a():
            return {"a": 1}

        async def coro_b():
            return {"b": 2}

        result = await fetcher._gather([("a", coro_a()), ("b", coro_b())])
        assert result == {"a": {"a": 1}, "b": {"b": 2}}

    @pytest.mark.asyncio
    async def test_exception_recorded_in_errors(self):
        fetcher = ParallelAPIFetcher()

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
        fetcher = ParallelAPIFetcher()

        async def coro_none():
            return None

        result = await fetcher._gather([("svc", coro_none())])
        assert result["svc"] is None
        assert "svc" not in fetcher.api_errors


class TestLogPerformanceMetrics:
    """Tests for _log_performance_metrics."""

    def test_logs_timings_and_errors(self):
        fetcher = ParallelAPIFetcher()
        fetcher.api_timings = {"mal": 1.5, "kitsu": 2.0}
        fetcher.api_errors = {"anidb": "timeout"}
        # Should not raise
        fetcher._log_performance_metrics(3.5)

    def test_zero_apis_does_not_divide_by_zero(self):
        fetcher = ParallelAPIFetcher()
        fetcher.api_timings = {}
        fetcher.api_errors = {}
        fetcher._log_performance_metrics(0.0)

    def test_success_rate_100_when_no_errors(self):
        fetcher = ParallelAPIFetcher()
        fetcher.api_timings = {"mal": 1.0, "kitsu": 0.5}
        fetcher.api_errors = {}
        # Should not raise; success rate = 100%
        fetcher._log_performance_metrics(1.5)
