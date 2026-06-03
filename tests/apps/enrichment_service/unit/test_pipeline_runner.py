from __future__ import annotations

import json
from pathlib import Path

import pytest
from enrichment_service import pipeline_runner


def test_get_anime_by_title_uses_only_primary_title() -> None:
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


def test_get_anime_by_index_requires_data_key() -> None:
    with pytest.raises(
        ValueError, match="Database payload missing required 'data' key"
    ):
        pipeline_runner.get_anime_by_index({}, 0)
    with pytest.raises(TypeError, match="Database payload 'data' must be a list"):
        pipeline_runner.get_anime_by_index({"data": "not_a_list"}, 0)


def test_get_anime_by_index_raises_when_out_of_range() -> None:
    with pytest.raises(ValueError, match="Index 5 out of range"):
        pipeline_runner.get_anime_by_index({"data": [{"title": "A"}]}, 5)


def test_get_anime_by_title_returns_single_exact_match() -> None:
    db = {"data": [{"title": "Cowboy Bebop"}, {"title": "Trigun"}]}
    result = pipeline_runner.get_anime_by_title(db, "Cowboy Bebop")
    assert result["title"] == "Cowboy Bebop"


def test_get_anime_by_title_raises_on_multiple_exact_matches() -> None:
    db = {"data": [{"title": "Cowboy Bebop"}, {"title": "Cowboy Bebop"}]}
    with pytest.raises(ValueError, match="Multiple exact matches"):
        pipeline_runner.get_anime_by_title(db, "Cowboy Bebop")


def test_get_anime_by_title_returns_single_partial_match() -> None:
    db = {"data": [{"title": "Cowboy Bebop"}, {"title": "Trigun"}]}
    result = pipeline_runner.get_anime_by_title(db, "cowboy")
    assert result["title"] == "Cowboy Bebop"


def test_get_anime_by_title_raises_on_multiple_partial_matches() -> None:
    db = {"data": [{"title": "Cowboy Bebop"}, {"title": "Cowboy Funk"}]}
    with pytest.raises(ValueError, match="Multiple matches"):
        pipeline_runner.get_anime_by_title(db, "cowboy")


def test_get_anime_by_title_requires_data_key() -> None:
    with pytest.raises(
        ValueError, match="Database payload missing required 'data' key"
    ):
        pipeline_runner.get_anime_by_title({}, "Naruto")


def test_artifact_name_falls_back_to_unknown_slug() -> None:
    filename = pipeline_runner._artifact_name("!!!")
    assert filename.startswith("unknown_")
    assert filename.endswith(".json")


@pytest.mark.asyncio
async def test_run_pipeline_and_write_artifact_returns_full_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "anime.json"
    input_path.write_text(
        json.dumps({"data": [{"title": "Cowboy Bebop"}]}),
        encoding="utf-8",
    )

    class _FakePipeline:
        async def __aenter__(self) -> _FakePipeline:
            return self

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            return False

        async def enrich_anime(self, anime_data, **kwargs):
            return {"anime": anime_data["title"], "kwargs": kwargs}

    monkeypatch.setattr(pipeline_runner, "EnrichmentPipeline", _FakePipeline)

    (
        output_path,
        result,
        payload,
    ) = await pipeline_runner.run_pipeline_and_write_artifact(
        file_path=input_path,
        index=0,
        title=None,
        agent_dir=None,
        skip_services=None,
        only_services=None,
        output_dir=tmp_path / "out",
    )

    assert result["anime"] == "Cowboy Bebop"
    assert payload["schema_version"] == "1.0"
    assert payload["data"] == result
    assert Path(output_path).exists()
    assert Path(output_path).parent == tmp_path / "out"


@pytest.mark.asyncio
async def test_run_pipeline_and_write_artifact_with_title_selector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "anime.json"
    input_path.write_text(
        json.dumps({"data": [{"title": "Cowboy Bebop"}]}), encoding="utf-8"
    )

    class _FakePipeline:
        async def __aenter__(self) -> "_FakePipeline":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            return False

        async def enrich_anime(self, anime_data, **kwargs):
            return {"anime": anime_data["title"]}

    monkeypatch.setattr(pipeline_runner, "EnrichmentPipeline", _FakePipeline)

    _, result, _ = await pipeline_runner.run_pipeline_and_write_artifact(
        file_path=input_path,
        index=None,
        title="Cowboy Bebop",
        agent_dir=None,
        skip_services=None,
        only_services=None,
        output_dir=tmp_path / "out",
    )

    assert result["anime"] == "Cowboy Bebop"


@pytest.mark.asyncio
async def test_run_pipeline_forwards_fetch_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "anime.json"
    input_path.write_text(
        json.dumps({"data": [{"title": "Cowboy Bebop"}]}), encoding="utf-8"
    )

    class _FakePipeline:
        async def __aenter__(self) -> "_FakePipeline":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            return False

        async def enrich_anime(self, anime_data, **kwargs):
            return {"anime": anime_data["title"], "kwargs": kwargs}

    monkeypatch.setattr(pipeline_runner, "EnrichmentPipeline", _FakePipeline)

    _, result, payload = await pipeline_runner.run_pipeline_and_write_artifact(
        file_path=input_path,
        index=0,
        title=None,
        agent_dir=None,
        skip_services=None,
        only_services=None,
        fetch_characters=False,
        fetch_episodes=False,
        output_dir=tmp_path / "out",
    )

    assert result["kwargs"]["fetch_characters"] is False
    assert result["kwargs"]["fetch_episodes"] is False
    assert payload["source_inputs"]["fetch_characters"] is False
    assert payload["source_inputs"]["fetch_episodes"] is False


@pytest.mark.asyncio
async def test_artifact_rejects_conflicting_selectors(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "anime.json"
    input_path.write_text(
        json.dumps({"data": [{"title": "Cowboy Bebop"}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Provide either index or title, not both"):
        await pipeline_runner.run_pipeline_and_write_artifact(
            file_path=input_path,
            index=0,
            title="Cowboy Bebop",
            agent_dir=None,
            skip_services=None,
            only_services=None,
            output_dir=tmp_path / "out",
        )


@pytest.mark.asyncio
async def test_artifact_rejects_missing_selectors(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "anime.json"
    input_path.write_text(
        json.dumps({"data": [{"title": "Cowboy Bebop"}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Either index or title must be provided"):
        await pipeline_runner.run_pipeline_and_write_artifact(
            file_path=input_path,
            index=None,
            title=None,
            agent_dir=None,
            skip_services=None,
            only_services=None,
            output_dir=tmp_path / "out",
        )
