import asyncio
import json
import logging
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
from enrichment.pipeline.config import EnrichmentConfig
from enrichment.pipeline.enrichment_pipeline import EnrichmentPipeline
from enrichment.sources.base import browser as browser_module
from enrichment.sources.base.browser import OWNER_FILE_NAME

IDS = {"mal_url": "https://myanimelist.net/anime/21"}
API_DATA = {"mal": {"title": "One Piece"}}


@pytest.fixture
def pipeline(tmp_path: Path) -> EnrichmentPipeline:
    return EnrichmentPipeline(EnrichmentConfig(temp_dir=str(tmp_path / "temp")))


@pytest.fixture
def sample_anime() -> dict[str, object]:
    return {
        "title": "One Piece",
        "sources": ["https://myanimelist.net/anime/21"],
        "type": "TV",
    }


@pytest.fixture
def stubbed_sources(pipeline: EnrichmentPipeline) -> Iterator[None]:
    with (
        patch.object(
            pipeline.id_extractor, "extract_all_ids", autospec=True, return_value=IDS
        ),
        patch.object(
            pipeline.id_extractor, "validate_ids", autospec=True, return_value=IDS
        ),
        patch.object(
            pipeline.api_fetcher, "fetch_all_data", autospec=True, return_value=API_DATA
        ),
    ):
        yield


@pytest.fixture
def unlimited_browser_pool() -> Iterator[None]:
    blocking = browser_module._block_unused_resources
    yield
    browser_module._max_browsers = None
    browser_module._block_unused_resources = blocking
    browser_module._slots_by_loop.clear()


def _agent_dirs(pipeline: EnrichmentPipeline, *names: str) -> None:
    for name in names:
        (Path(pipeline.config.temp_dir) / name).mkdir(parents=True)


def test_enrichment_pipeline_no_config_uses_default_config() -> None:
    assert isinstance(EnrichmentPipeline().config, EnrichmentConfig)


def test_enrichment_pipeline_given_config_keeps_it() -> None:
    config = EnrichmentConfig()

    assert EnrichmentPipeline(config).config is config


def test_enrichment_pipeline_starts_with_empty_timing_breakdown(
    pipeline: EnrichmentPipeline,
) -> None:
    assert pipeline.timing_breakdown == {}


@pytest.mark.parametrize("verbose", [True, False])
def test_enrichment_pipeline_verbose_logging_logs_configuration_only_when_on(
    caplog: pytest.LogCaptureFixture, verbose: bool
) -> None:
    with caplog.at_level(logging.INFO, logger="enrichment.pipeline.config"):
        EnrichmentPipeline(EnrichmentConfig(verbose_logging=verbose))

    assert ("Enrichment Pipeline Configuration" in caplog.text) is verbose


def test_find_next_agent_id_missing_temp_dir_returns_one(
    pipeline: EnrichmentPipeline,
) -> None:
    assert pipeline._find_next_agent_id() == 1


def test_find_next_agent_id_unexpected_error_returns_one(
    pipeline: EnrichmentPipeline,
) -> None:
    with patch("os.listdir", autospec=True, side_effect=RuntimeError("disk error")):
        assert pipeline._find_next_agent_id() == 1


def test_find_next_agent_id_no_agent_dirs_returns_one(
    pipeline: EnrichmentPipeline,
) -> None:
    _agent_dirs(pipeline, "unrelated_folder")

    assert pipeline._find_next_agent_id() == 1


@pytest.mark.parametrize(
    ("existing", "expected"),
    [
        (("One_agent1", "Three_agent3"), 2),
        (("One_agent1", "Two_agent2"), 3),
        (("One_agent1",), 2),
    ],
    ids=["gap", "no_gap", "single"],
)
def test_find_next_agent_id_existing_dirs_returns_lowest_free_id(
    pipeline: EnrichmentPipeline, existing: tuple[str, ...], expected: int
) -> None:
    _agent_dirs(pipeline, *existing)

    assert pipeline._find_next_agent_id() == expected


def test_create_temp_dir_creates_first_word_agent_dir_under_temp_dir(
    pipeline: EnrichmentPipeline,
) -> None:
    path = pipeline._create_temp_dir("One Piece")

    assert path == os.path.join(pipeline.config.temp_dir, "One_agent1")
    assert os.path.isdir(path)


def test_create_temp_dir_special_characters_are_removed(
    pipeline: EnrichmentPipeline,
) -> None:
    path = pipeline._create_temp_dir("Sword!!! Art Online")

    assert os.path.basename(path) == "Sword_agent1"


def test_create_temp_dir_empty_title_uses_unknown(pipeline: EnrichmentPipeline) -> None:
    assert os.path.basename(pipeline._create_temp_dir("")) == "unknown_agent1"


def test_create_temp_dir_repeated_calls_return_different_paths(
    pipeline: EnrichmentPipeline,
) -> None:
    first = pipeline._create_temp_dir("Naruto")
    second = pipeline._create_temp_dir("Naruto")

    assert (os.path.basename(first), os.path.basename(second)) == (
        "Naruto_agent1",
        "Naruto_agent2",
    )


@pytest.mark.usefixtures("stubbed_sources")
async def test_enrich_anime_returns_offline_ids_and_api_data(
    pipeline: EnrichmentPipeline, sample_anime: dict[str, object]
) -> None:
    result = await pipeline.enrich_anime(sample_anime, agent_dir="One_agent1")

    assert result["offline_data"] is sample_anime
    assert result["extracted_ids"] == IDS
    assert result["api_data"] == API_DATA
    assert result["enrichment_metadata"]["method"] == "programmatic"
    assert "total_time" in result["enrichment_metadata"]
    assert "temp_directory" in result["enrichment_metadata"]


@pytest.mark.usefixtures("stubbed_sources")
async def test_enrich_anime_saves_current_anime_json(
    pipeline: EnrichmentPipeline, sample_anime: dict[str, object]
) -> None:
    await pipeline.enrich_anime(sample_anime, agent_dir="One_agent1")

    saved = Path(pipeline.config.temp_dir) / "One_agent1" / "current_anime.json"
    assert json.loads(saved.read_text())["title"] == "One Piece"


@pytest.mark.usefixtures("stubbed_sources")
async def test_enrich_anime_no_agent_dir_creates_agent_dir(
    pipeline: EnrichmentPipeline, sample_anime: dict[str, object]
) -> None:
    result = await pipeline.enrich_anime(sample_anime)

    temp_dir = result["enrichment_metadata"]["temp_directory"]
    assert os.path.isdir(temp_dir)
    assert os.path.basename(temp_dir) == "One_agent1"


@pytest.mark.usefixtures("stubbed_sources")
async def test_enrich_anime_records_timing_breakdown(
    pipeline: EnrichmentPipeline, sample_anime: dict[str, object]
) -> None:
    await pipeline.enrich_anime(sample_anime, agent_dir="One_agent1")

    assert {"id_extraction", "api_fetching"} <= pipeline.timing_breakdown.keys()


async def test_enrich_anime_failure_with_skip_enabled_returns_partial_result(
    pipeline: EnrichmentPipeline, sample_anime: dict[str, object]
) -> None:
    with patch.object(
        pipeline.id_extractor,
        "extract_all_ids",
        autospec=True,
        side_effect=RuntimeError("ID extraction failed"),
    ):
        result = await pipeline.enrich_anime(sample_anime, agent_dir="One_agent1")

    assert result["partial_data"] is True
    assert "ID extraction failed" in result["error"]
    assert result["offline_data"] is sample_anime


async def test_enrich_anime_failure_with_skip_disabled_raises(
    tmp_path: Path, sample_anime: dict[str, object]
) -> None:
    pipeline = EnrichmentPipeline(
        EnrichmentConfig(skip_failed_apis=False, temp_dir=str(tmp_path))
    )

    with (
        patch.object(
            pipeline.id_extractor,
            "extract_all_ids",
            autospec=True,
            side_effect=RuntimeError("hard failure"),
        ),
        pytest.raises(RuntimeError, match="hard failure"),
    ):
        await pipeline.enrich_anime(sample_anime, agent_dir="One_agent1")


@pytest.mark.usefixtures("stubbed_sources")
@pytest.mark.parametrize(
    ("options", "skip", "only"),
    [
        ({"only_services": ["kitsu"]}, None, ["kitsu"]),
        ({"skip_services": ["anidb"]}, ["anidb"], None),
    ],
    ids=["only", "skip"],
)
async def test_enrich_anime_service_selection_forwarded_to_fetcher(
    pipeline: EnrichmentPipeline,
    sample_anime: dict[str, object],
    options: dict[str, list[str]],
    skip: list[str] | None,
    only: list[str] | None,
) -> None:
    await pipeline.enrich_anime(sample_anime, agent_dir="One_agent1", **options)

    arguments = pipeline.api_fetcher.fetch_all_data.call_args.args
    assert (arguments[3], arguments[4]) == (skip, only)


@pytest.mark.usefixtures("stubbed_sources")
@pytest.mark.parametrize(
    ("options", "characters", "episodes"),
    [
        ({"fetch_characters": False}, False, True),
        ({"fetch_episodes": False}, True, False),
    ],
    ids=["no_characters", "no_episodes"],
)
async def test_enrich_anime_entity_flags_forwarded_to_fetcher(
    pipeline: EnrichmentPipeline,
    sample_anime: dict[str, object],
    options: dict[str, bool],
    characters: bool,
    episodes: bool,
) -> None:
    await pipeline.enrich_anime(sample_anime, agent_dir="One_agent1", **options)

    keywords = pipeline.api_fetcher.fetch_all_data.call_args.kwargs
    assert (keywords["fetch_characters"], keywords["fetch_episodes"]) == (
        characters,
        episodes,
    )


async def test_enrich_batch_returns_result_per_anime(
    pipeline: EnrichmentPipeline,
) -> None:
    async def enrich(anime: dict, **_: object) -> dict:
        return {"offline_data": anime}

    with patch.object(pipeline, "enrich_anime", autospec=True, side_effect=enrich):
        results = await pipeline.enrich_batch([{"title": "A"}, {"title": "B"}])

    assert [result["offline_data"]["title"] for result in results] == ["A", "B"]


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("failed"), asyncio.CancelledError()],
    ids=["error", "cancelled"],
)
async def test_enrich_batch_failed_entry_is_dropped(
    pipeline: EnrichmentPipeline, failure: BaseException
) -> None:
    async def enrich(anime: dict, **_: object) -> dict:
        if anime["title"] == "Bad":
            raise failure
        return {"offline_data": anime}

    with patch.object(pipeline, "enrich_anime", autospec=True, side_effect=enrich):
        results = await pipeline.enrich_batch([{"title": "Good"}, {"title": "Bad"}])

    assert [result["offline_data"]["title"] for result in results] == ["Good"]


async def test_enrich_batch_entity_flags_forwarded_to_each_anime(
    pipeline: EnrichmentPipeline,
) -> None:
    received: list[dict] = []

    async def enrich(anime: dict, **options: object) -> dict:
        received.append(options)
        return {"offline_data": anime}

    with patch.object(pipeline, "enrich_anime", autospec=True, side_effect=enrich):
        await pipeline.enrich_batch(
            [{"title": "A"}], fetch_characters=False, fetch_episodes=False
        )

    assert (received[0]["fetch_characters"], received[0]["fetch_episodes"]) == (
        False,
        False,
    )


async def test_enrich_batch_more_anime_than_batch_size_returns_every_result(
    tmp_path: Path,
) -> None:
    pipeline = EnrichmentPipeline(
        EnrichmentConfig(batch_size=2, temp_dir=str(tmp_path))
    )

    async def enrich(anime: dict, **_: object) -> dict:
        return {"offline_data": anime}

    with patch.object(pipeline, "enrich_anime", autospec=True, side_effect=enrich):
        results = await pipeline.enrich_batch(
            [{"title": f"Anime{i}"} for i in range(5)]
        )

    assert len(results) == 5


def test_get_performance_report_includes_browser_limit(
    pipeline: EnrichmentPipeline,
) -> None:
    assert "Max concurrent browsers: 4" in pipeline.get_performance_report()


def test_get_performance_report_includes_timings_when_present(
    pipeline: EnrichmentPipeline,
) -> None:
    pipeline.timing_breakdown = {"id_extraction": 0.05, "api_fetching": 2.3}
    pipeline.api_fetcher.api_timings = {"kitsu": 1.2, "mal": 0.8}

    report = pipeline.get_performance_report()

    assert all(
        name in report for name in ("id_extraction", "api_fetching", "kitsu", "mal")
    )


def test_get_performance_report_without_timings_returns_header_only(
    pipeline: EnrichmentPipeline,
) -> None:
    pipeline.timing_breakdown = {}
    pipeline.api_fetcher.api_timings = {}

    report = pipeline.get_performance_report()

    assert "Timing Breakdown" not in report
    assert "API Response Times" not in report


@pytest.mark.usefixtures("unlimited_browser_pool")
async def test_enrichment_pipeline_enter_applies_browser_settings_and_removes_abandoned_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    abandoned = tmp_path / "uc_abandoned"
    abandoned.mkdir()
    (abandoned / OWNER_FILE_NAME).write_text("999999999 1")
    pipeline = EnrichmentPipeline(
        EnrichmentConfig(
            max_concurrent_browsers=3,
            block_unused_resources=False,
            temp_dir=str(tmp_path / "temp"),
        )
    )

    async with pipeline as entered:
        assert entered is pipeline

    assert browser_module._max_browsers == 3
    assert browser_module._block_unused_resources is False
    assert not abandoned.exists()


async def test_enrichment_pipeline_exit_closes_api_fetcher_and_propagates_errors(
    pipeline: EnrichmentPipeline,
) -> None:
    with patch.object(
        pipeline.api_fetcher, "__aexit__", autospec=True, return_value=False
    ) as fetcher_exit:
        result = await pipeline.__aexit__(None, None, None)

    assert result is False
    fetcher_exit.assert_awaited_once_with(None, None, None)
