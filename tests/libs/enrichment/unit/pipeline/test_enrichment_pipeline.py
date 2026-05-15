"""
Tests for EnrichmentPipeline.
"""

import os
import json
import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

import pytest

from enrichment.pipeline.config import EnrichmentConfig
from enrichment.pipeline.enrichment_pipeline import EnrichmentPipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return EnrichmentConfig()


@pytest.fixture
def pipeline(config):
    return EnrichmentPipeline(config)


@pytest.fixture
def sample_anime():
    return {
        "title": "One Piece",
        "sources": ["https://myanimelist.net/anime/21"],
        "type": "TV",
    }


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestInit:
    def test_uses_default_config_when_none_given(self):
        p = EnrichmentPipeline()
        assert isinstance(p.config, EnrichmentConfig)

    def test_uses_provided_config(self, config):
        p = EnrichmentPipeline(config)
        assert p.config is config

    def test_timing_breakdown_starts_empty(self, pipeline):
        assert pipeline.timing_breakdown == {}

    def test_verbose_logging_calls_log_configuration(self):
        config = EnrichmentConfig(verbose_logging=True)
        with patch.object(EnrichmentConfig, "log_configuration") as mock_log:
            EnrichmentPipeline(config)
        mock_log.assert_called_once()

    def test_no_verbose_logging_skips_log_configuration(self):
        config = EnrichmentConfig(verbose_logging=False)
        with patch.object(EnrichmentConfig, "log_configuration") as mock_log:
            EnrichmentPipeline(config)
        mock_log.assert_not_called()


# ---------------------------------------------------------------------------
# _find_next_agent_id
# ---------------------------------------------------------------------------

class TestFindNextAgentId:
    def test_returns_1_when_temp_dir_missing(self, pipeline):
        with patch("os.listdir", side_effect=FileNotFoundError):
            assert pipeline._find_next_agent_id() == 1

    def test_returns_1_on_unexpected_error(self, pipeline):
        with patch("os.listdir", side_effect=RuntimeError("disk error")):
            assert pipeline._find_next_agent_id() == 1

    def test_returns_1_when_no_agent_dirs(self, pipeline):
        with patch("os.listdir", return_value=["unrelated_folder", "file.txt"]):
            assert pipeline._find_next_agent_id() == 1

    def test_fills_gap_in_ids(self, pipeline):
        # agent1 and agent3 exist → gap at 2
        with patch("os.listdir", return_value=["One_agent1", "Three_agent3"]):
            assert pipeline._find_next_agent_id() == 2

    def test_returns_next_sequential_when_no_gap(self, pipeline):
        with patch("os.listdir", return_value=["One_agent1", "Two_agent2"]):
            assert pipeline._find_next_agent_id() == 3

    def test_single_existing_id_returns_next(self, pipeline):
        with patch("os.listdir", return_value=["One_agent1"]):
            assert pipeline._find_next_agent_id() == 2


# ---------------------------------------------------------------------------
# _create_temp_dir
# ---------------------------------------------------------------------------

class TestCreateTempDir:
    def test_creates_dir_with_correct_name(self, pipeline):
        with patch("os.listdir", return_value=[]):
            with patch("os.makedirs") as mock_makedirs:
                path = pipeline._create_temp_dir("One Piece")

        assert "One_agent" in path
        assert mock_makedirs.call_count == 2
        mock_makedirs.assert_any_call(pipeline.config.temp_dir, exist_ok=True)

    def test_sanitizes_special_characters(self, pipeline):
        with patch("os.listdir", return_value=[]):
            with patch("os.makedirs"):
                path = pipeline._create_temp_dir("Sword Art!!! Online")

        assert "SwordArt" in path or "Sword" in path

    def test_empty_title_uses_unknown(self, pipeline):
        with patch("os.listdir", return_value=[]):
            with patch("os.makedirs"):
                path = pipeline._create_temp_dir("")

        assert "unknown_agent" in path

    def test_returns_full_path_under_temp_dir(self, pipeline):
        with patch("os.listdir", return_value=[]):
            with patch("os.makedirs"):
                path = pipeline._create_temp_dir("Naruto")

        assert path.startswith(pipeline.config.temp_dir)

    def test_repeated_calls_do_not_reuse_same_path_when_scan_state_is_stale(self, pipeline):
        # First scan returns empty → agent1; second scan returns agent1 dir → agent2
        with patch("os.listdir", side_effect=[[], ["Naruto_agent1"]]):
            with patch("os.makedirs"):
                first = pipeline._create_temp_dir("Naruto")
                second = pipeline._create_temp_dir("Naruto")

        assert first != second


# ---------------------------------------------------------------------------
# enrich_anime
# ---------------------------------------------------------------------------

class TestEnrichAnime:
    @pytest.mark.asyncio
    async def test_success_returns_full_result(self, pipeline, sample_anime, tmp_path):
        pipeline.config = EnrichmentConfig(temp_dir=str(tmp_path))

        mock_ids = {"mal_url": "https://myanimelist.net/anime/21"}
        mock_api_data = {"mal": {"title": "One Piece"}}

        pipeline.id_extractor.extract_all_ids = MagicMock(return_value=mock_ids)
        pipeline.id_extractor.validate_ids = MagicMock(return_value=mock_ids)
        pipeline.api_fetcher.fetch_all_data = AsyncMock(return_value=mock_api_data)

        result = await pipeline.enrich_anime(
            sample_anime, agent_dir="One_agent1"
        )

        assert result["offline_data"] is sample_anime
        assert result["extracted_ids"] == mock_ids
        assert result["api_data"] == mock_api_data
        assert result["enrichment_metadata"]["method"] == "programmatic"
        assert "total_time" in result["enrichment_metadata"]
        assert "temp_directory" in result["enrichment_metadata"]

    @pytest.mark.asyncio
    async def test_saves_current_anime_json(self, pipeline, sample_anime, tmp_path):
        pipeline.config = EnrichmentConfig(temp_dir=str(tmp_path))

        pipeline.id_extractor.extract_all_ids = MagicMock(return_value={})
        pipeline.id_extractor.validate_ids = MagicMock(return_value={})
        pipeline.api_fetcher.fetch_all_data = AsyncMock(return_value={})

        await pipeline.enrich_anime(sample_anime, agent_dir="One_agent1")

        saved_path = tmp_path / "One_agent1" / "current_anime.json"
        assert saved_path.exists()
        with open(saved_path) as f:
            saved = json.load(f)
        assert saved["title"] == "One Piece"

    @pytest.mark.asyncio
    async def test_auto_generates_agent_dir_when_none(self, pipeline, sample_anime, tmp_path):
        pipeline.config = EnrichmentConfig(temp_dir=str(tmp_path))

        pipeline.id_extractor.extract_all_ids = MagicMock(return_value={})
        pipeline.id_extractor.validate_ids = MagicMock(return_value={})
        pipeline.api_fetcher.fetch_all_data = AsyncMock(return_value={})

        result = await pipeline.enrich_anime(sample_anime)

        temp_dir = result["enrichment_metadata"]["temp_directory"]
        assert os.path.isdir(temp_dir)
        assert "_agent" in temp_dir

    @pytest.mark.asyncio
    async def test_timing_breakdown_recorded(self, pipeline, sample_anime, tmp_path):
        pipeline.config = EnrichmentConfig(temp_dir=str(tmp_path))

        pipeline.id_extractor.extract_all_ids = MagicMock(return_value={})
        pipeline.id_extractor.validate_ids = MagicMock(return_value={})
        pipeline.api_fetcher.fetch_all_data = AsyncMock(return_value={})

        await pipeline.enrich_anime(sample_anime, agent_dir="One_agent1")

        assert "id_extraction" in pipeline.timing_breakdown
        assert "api_fetching" in pipeline.timing_breakdown

    @pytest.mark.asyncio
    async def test_exception_returns_partial_when_skip_enabled(self, pipeline, sample_anime, tmp_path):
        pipeline.config = EnrichmentConfig(skip_failed_apis=True, temp_dir=str(tmp_path))

        pipeline.id_extractor.extract_all_ids = MagicMock(
            side_effect=RuntimeError("ID extraction failed")
        )

        result = await pipeline.enrich_anime(sample_anime, agent_dir="One_agent1")

        assert result["partial_data"] is True
        assert "ID extraction failed" in result["error"]
        assert result["offline_data"] is sample_anime

    @pytest.mark.asyncio
    async def test_exception_raises_when_skip_disabled(self, pipeline, sample_anime, tmp_path):
        pipeline.config = EnrichmentConfig(skip_failed_apis=False, temp_dir=str(tmp_path))

        pipeline.id_extractor.extract_all_ids = MagicMock(
            side_effect=RuntimeError("hard failure")
        )

        with pytest.raises(RuntimeError, match="hard failure"):
            await pipeline.enrich_anime(sample_anime, agent_dir="One_agent1")

    @pytest.mark.asyncio
    async def test_only_services_forwarded_to_fetcher(self, pipeline, sample_anime, tmp_path):
        pipeline.config = EnrichmentConfig(temp_dir=str(tmp_path))

        pipeline.id_extractor.extract_all_ids = MagicMock(return_value={})
        pipeline.id_extractor.validate_ids = MagicMock(return_value={})
        pipeline.api_fetcher.fetch_all_data = AsyncMock(return_value={})

        await pipeline.enrich_anime(
            sample_anime, agent_dir="One_agent1", only_services=["kitsu"]
        )

        pipeline.api_fetcher.fetch_all_data.assert_awaited_once()
        call_kwargs = pipeline.api_fetcher.fetch_all_data.call_args
        assert call_kwargs[0][3] is None  # skip_services
        assert call_kwargs[0][4] == ["kitsu"]  # only_services

    @pytest.mark.asyncio
    async def test_skip_services_forwarded_to_fetcher(self, pipeline, sample_anime, tmp_path):
        pipeline.config = EnrichmentConfig(temp_dir=str(tmp_path))

        pipeline.id_extractor.extract_all_ids = MagicMock(return_value={})
        pipeline.id_extractor.validate_ids = MagicMock(return_value={})
        pipeline.api_fetcher.fetch_all_data = AsyncMock(return_value={})

        await pipeline.enrich_anime(
            sample_anime, agent_dir="One_agent1", skip_services=["anidb"]
        )

        call_kwargs = pipeline.api_fetcher.fetch_all_data.call_args
        assert call_kwargs[0][3] == ["anidb"]  # skip_services
        assert call_kwargs[0][4] is None  # only_services


# ---------------------------------------------------------------------------
# enrich_batch
# ---------------------------------------------------------------------------

class TestEnrichBatch:
    @pytest.mark.asyncio
    async def test_returns_successful_results(self, pipeline, tmp_path):
        pipeline.config = EnrichmentConfig(temp_dir=str(tmp_path))
        anime_list = [{"title": "A"}, {"title": "B"}]

        async def fake_enrich(anime, **kwargs):
            return {"offline_data": anime, "api_data": {}}

        with patch.object(pipeline, "enrich_anime", side_effect=fake_enrich):
            results = await pipeline.enrich_batch(anime_list)

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_drops_failed_entries(self, pipeline, tmp_path):
        pipeline.config = EnrichmentConfig(temp_dir=str(tmp_path))
        anime_list = [{"title": "Good"}, {"title": "Bad"}]

        call_count = 0

        async def fake_enrich(anime, **kwargs):
            nonlocal call_count
            call_count += 1
            if anime["title"] == "Bad":
                raise RuntimeError("failed")
            return {"offline_data": anime}

        with patch.object(pipeline, "enrich_anime", side_effect=fake_enrich):
            results = await pipeline.enrich_batch(anime_list)

        assert len(results) == 1
        assert results[0]["offline_data"]["title"] == "Good"

    @pytest.mark.asyncio
    async def test_respects_batch_size_semaphore(self, pipeline, tmp_path):
        """Semaphore is created with config.batch_size — doesn't deadlock on small batch."""
        pipeline.config = EnrichmentConfig(batch_size=2, temp_dir=str(tmp_path))
        anime_list = [{"title": f"Anime{i}"} for i in range(5)]

        async def fake_enrich(anime, **kwargs):
            return {"offline_data": anime}

        with patch.object(pipeline, "enrich_anime", side_effect=fake_enrich):
            results = await pipeline.enrich_batch(anime_list)

        assert len(results) == 5


# ---------------------------------------------------------------------------
# get_performance_report
# ---------------------------------------------------------------------------

class TestGetPerformanceReport:
    def test_report_contains_config_values(self, pipeline):
        report = pipeline.get_performance_report()
        assert "Max Concurrent APIs" in report or "Total APIs configured" in report
        assert str(pipeline.config.max_concurrent_apis) in report

    def test_report_includes_timing_when_present(self, pipeline):
        pipeline.timing_breakdown = {"id_extraction": 0.05, "api_fetching": 2.3}
        report = pipeline.get_performance_report()
        assert "id_extraction" in report
        assert "api_fetching" in report

    def test_report_includes_api_timings_when_present(self, pipeline):
        pipeline.api_fetcher.api_timings = {"kitsu": 1.2, "mal": 0.8}
        report = pipeline.get_performance_report()
        assert "kitsu" in report
        assert "mal" in report

    def test_report_without_timing_does_not_raise(self, pipeline):
        pipeline.timing_breakdown = {}
        pipeline.api_fetcher.api_timings = {}
        report = pipeline.get_performance_report()
        assert isinstance(report, str)


# ---------------------------------------------------------------------------
# Context manager protocol
# ---------------------------------------------------------------------------

class TestContextManager:
    @pytest.mark.asyncio
    async def test_aenter_returns_self(self, pipeline):
        result = await pipeline.__aenter__()
        assert result is pipeline

    @pytest.mark.asyncio
    async def test_aexit_delegates_to_api_fetcher(self, pipeline):
        pipeline.api_fetcher.__aexit__ = AsyncMock(return_value=False)
        result = await pipeline.__aexit__(None, None, None)
        pipeline.api_fetcher.__aexit__.assert_awaited_once_with(None, None, None)
        assert result is False

    @pytest.mark.asyncio
    async def test_async_with_block(self, pipeline):
        async with pipeline as p:
            assert p is pipeline
