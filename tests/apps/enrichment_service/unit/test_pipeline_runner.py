import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from common.config import EnrichmentConfig
from enrichment.pipeline.enrichment_pipeline import EnrichmentPipeline
from enrichment_service import pipeline_runner
from http_cache import HTTPCacheManager, get_cache


class EchoPipeline(EnrichmentPipeline):
    built: list[EchoPipeline] = []

    def __init__(self, config: EnrichmentConfig, cache: HTTPCacheManager) -> None:
        self.config = config
        self.cache = cache
        EchoPipeline.built.append(self)

    async def __aenter__(self) -> EchoPipeline:
        return self

    async def __aexit__(self, *exc_info: object) -> bool:
        return False

    async def enrich_anime(
        self, offline_data: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        return {"anime": offline_data["title"], "kwargs": kwargs}


@pytest.fixture
def echo_pipeline() -> Iterator[list[EchoPipeline]]:
    EchoPipeline.built = []
    with patch.object(pipeline_runner, "EnrichmentPipeline", EchoPipeline):
        yield EchoPipeline.built


@pytest.fixture
def database(tmp_path: Path) -> Path:
    path = tmp_path / "anime.json"
    path.write_text(json.dumps({"data": [{"title": "Cowboy Bebop"}]}), "utf-8")
    return path


async def _run(database: Path, output_dir: Path, **selectors: Any) -> tuple:
    return await pipeline_runner.run_pipeline_and_write_artifact(
        config=selectors.pop("config", EnrichmentConfig()),
        cache=selectors.pop("cache", get_cache()),
        file_path=database,
        agent_dir=None,
        skip_services=None,
        only_services=None,
        output_dir=output_dir,
        **{"index": None, "title": None, **selectors},
    )


def test_get_anime_by_title_english_title_only_raises_value_error() -> None:
    database = {
        "data": [
            {
                "title": "Naruto: Shippuden",
                "title_english": "Naruto Hurricane Chronicles",
                "title_japanese": "ナルト 疾風伝",
                "synonyms": ["Naruto Shippuuden"],
            }
        ]
    }

    with pytest.raises(
        ValueError, match="No anime found matching title 'Naruto Hurricane Chronicles'"
    ):
        pipeline_runner.get_anime_by_title(database, "Naruto Hurricane Chronicles")


def test_get_anime_by_index_missing_data_key_raises_value_error() -> None:
    with pytest.raises(
        ValueError, match="Database payload missing required 'data' key"
    ):
        pipeline_runner.get_anime_by_index({}, 0)


def test_get_anime_by_index_data_not_list_raises_type_error() -> None:
    with pytest.raises(TypeError, match="Database payload 'data' must be a list"):
        pipeline_runner.get_anime_by_index({"data": "not_a_list"}, 0)


def test_get_anime_by_index_out_of_range_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Index 5 out of range"):
        pipeline_runner.get_anime_by_index({"data": [{"title": "A"}]}, 5)


def test_get_anime_by_title_single_exact_match_returns_it() -> None:
    database = {"data": [{"title": "Cowboy Bebop"}, {"title": "Trigun"}]}

    assert pipeline_runner.get_anime_by_title(database, "Cowboy Bebop") == {
        "title": "Cowboy Bebop"
    }


def test_get_anime_by_title_multiple_exact_matches_raises_value_error() -> None:
    database = {"data": [{"title": "Cowboy Bebop"}, {"title": "Cowboy Bebop"}]}

    with pytest.raises(ValueError, match="Multiple exact matches"):
        pipeline_runner.get_anime_by_title(database, "Cowboy Bebop")


def test_get_anime_by_title_single_partial_match_returns_it() -> None:
    database = {"data": [{"title": "Cowboy Bebop"}, {"title": "Trigun"}]}

    assert pipeline_runner.get_anime_by_title(database, "cowboy") == {
        "title": "Cowboy Bebop"
    }


def test_get_anime_by_title_multiple_partial_matches_raises_value_error() -> None:
    database = {"data": [{"title": "Cowboy Bebop"}, {"title": "Cowboy Funk"}]}

    with pytest.raises(ValueError, match="Multiple matches"):
        pipeline_runner.get_anime_by_title(database, "cowboy")


def test_get_anime_by_title_missing_data_key_raises_value_error() -> None:
    with pytest.raises(
        ValueError, match="Database payload missing required 'data' key"
    ):
        pipeline_runner.get_anime_by_title({}, "Naruto")


def test_artifact_name_title_without_letters_uses_unknown_slug() -> None:
    filename = pipeline_runner._artifact_name("!!!")

    assert filename.startswith("unknown_")
    assert filename.endswith(".json")


async def test_run_pipeline_and_write_artifact_index_selector_writes_full_payload(
    tmp_path: Path, database: Path, echo_pipeline: list[EchoPipeline]
) -> None:
    output_path, result, payload = await _run(database, tmp_path / "out", index=0)

    assert result["anime"] == "Cowboy Bebop"
    assert payload["schema_version"] == "1.0"
    assert payload["data"] == result
    assert Path(output_path).parent == tmp_path / "out"
    assert json.loads(Path(output_path).read_text("utf-8")) == payload


async def test_run_pipeline_and_write_artifact_builds_pipeline_with_given_config_and_cache(
    tmp_path: Path, database: Path, echo_pipeline: list[EchoPipeline]
) -> None:
    config = EnrichmentConfig()
    cache = get_cache()

    await _run(database, tmp_path / "out", index=0, config=config, cache=cache)

    assert len(echo_pipeline) == 1
    assert echo_pipeline[0].config is config
    assert echo_pipeline[0].cache is cache


async def test_run_pipeline_and_write_artifact_title_selector_enriches_match(
    tmp_path: Path, database: Path, echo_pipeline: list[EchoPipeline]
) -> None:
    _, result, _ = await _run(database, tmp_path / "out", title="Cowboy Bebop")

    assert result["anime"] == "Cowboy Bebop"


async def test_run_pipeline_and_write_artifact_fetch_flags_forwarded_and_recorded(
    tmp_path: Path, database: Path, echo_pipeline: list[EchoPipeline]
) -> None:
    _, result, payload = await _run(
        database,
        tmp_path / "out",
        index=0,
        fetch_characters=False,
        fetch_episodes=False,
    )

    assert result["kwargs"]["fetch_characters"] is False
    assert result["kwargs"]["fetch_episodes"] is False
    assert payload["source_inputs"]["fetch_characters"] is False
    assert payload["source_inputs"]["fetch_episodes"] is False


async def test_run_pipeline_and_write_artifact_both_selectors_raises_value_error(
    tmp_path: Path, database: Path
) -> None:
    with pytest.raises(ValueError, match="Provide either index or title, not both"):
        await _run(database, tmp_path / "out", index=0, title="Cowboy Bebop")


async def test_run_pipeline_and_write_artifact_no_selector_raises_value_error(
    tmp_path: Path, database: Path
) -> None:
    with pytest.raises(ValueError, match="Either index or title must be provided"):
        await _run(database, tmp_path / "out")
