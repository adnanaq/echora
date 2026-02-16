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

        async def enrich_anime(self, anime_data, **kwargs):  # noqa: ANN001, ANN003
            return {"anime": anime_data["title"], "kwargs": kwargs}

    monkeypatch.setattr(
        pipeline_runner, "ProgrammaticEnrichmentPipeline", _FakePipeline
    )

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
