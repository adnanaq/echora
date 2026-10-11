from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
from common.config import EnrichmentConfig
from enrichment_proto.v1 import enrichment_service_pb2
from enrichment_service.routes import pipeline as pipeline_route
from enrichment_service.routes.pipeline import run_pipeline
from enrichment_service.runtime import EnrichmentRuntime
from http_cache import get_cache

Request = enrichment_service_pb2.RunPipelineRequest


@pytest.fixture
def runtime(tmp_path: Path) -> EnrichmentRuntime:
    default_db = tmp_path / "db" / "anime.json"
    default_db.parent.mkdir(parents=True)
    default_db.write_text("{}", encoding="utf-8")
    return EnrichmentRuntime(
        default_file_path=str(default_db),
        output_dir=str(tmp_path / "out"),
        enrichment=EnrichmentConfig(),
        cache=get_cache(),
    )


@pytest.fixture
def runner(tmp_path: Path) -> Iterator[object]:
    with patch.object(
        pipeline_route,
        "run_pipeline_and_write_artifact",
        autospec=True,
        return_value=(str(tmp_path / "out" / "result.json"), {}, {}),
    ) as run:
        yield run


@pytest.mark.parametrize("file_path", ["../../../etc/passwd", "/etc/passwd"])
async def test_run_pipeline_file_outside_allowed_dir_returns_invalid_file_path(
    runtime: EnrichmentRuntime, file_path: str
) -> None:
    response = await run_pipeline(
        runtime, Request(file_path=file_path, title="Cowboy Bebop"), context=None
    )

    assert response.success is False
    assert response.error.code == "INVALID_FILE_PATH"
    assert response.error.retryable is False


async def test_run_pipeline_sibling_file_in_allowed_dir_succeeds(
    runtime: EnrichmentRuntime, runner: object
) -> None:
    sibling = Path(runtime.default_file_path).parent / "other.json"
    sibling.write_text("{}", encoding="utf-8")

    response = await run_pipeline(
        runtime, Request(file_path=str(sibling), title="Cowboy Bebop"), context=None
    )

    assert response.success is True
    assert not response.HasField("error")


@pytest.mark.parametrize("agent_dir", ["../../../etc", "/etc/evil"])
async def test_run_pipeline_unsafe_agent_dir_returns_invalid_agent_dir(
    runtime: EnrichmentRuntime, agent_dir: str
) -> None:
    response = await run_pipeline(
        runtime, Request(title="Cowboy Bebop", agent_dir=agent_dir), context=None
    )

    assert response.success is False
    assert response.error.code == "INVALID_AGENT_DIR"
    assert response.error.retryable is False


async def test_run_pipeline_safe_agent_dir_succeeds(
    runtime: EnrichmentRuntime, runner: object
) -> None:
    response = await run_pipeline(
        runtime, Request(title="Cowboy Bebop", agent_dir="Cowboy_agent1"), context=None
    )

    assert response.success is True
    assert not response.HasField("error")


async def test_run_pipeline_passes_runtime_config_and_cache_to_runner(
    runtime: EnrichmentRuntime, runner
) -> None:
    await run_pipeline(runtime, Request(title="Cowboy Bebop"), context=None)

    assert runner.call_args.kwargs["config"] is runtime.enrichment
    assert runner.call_args.kwargs["cache"] is runtime.cache


async def test_run_pipeline_negative_index_passed_as_none(
    runtime: EnrichmentRuntime, runner
) -> None:
    response = await run_pipeline(
        runtime, Request(title="Cowboy Bebop", index=-1), context=None
    )

    assert response.success is True
    assert runner.call_args.kwargs["index"] is None


async def test_run_pipeline_runner_error_returns_run_pipeline_failed(
    runtime: EnrichmentRuntime, runner
) -> None:
    runner.side_effect = RuntimeError("pipeline exploded")

    response = await run_pipeline(runtime, Request(title="Cowboy Bebop"), context=None)

    assert response.success is False
    assert response.error.code == "RUN_PIPELINE_FAILED"
    assert "pipeline exploded" in response.error.message


@pytest.mark.parametrize(
    ("skips", "fetch_characters", "fetch_episodes"),
    [
        ({}, True, True),
        ({"skip_characters": True}, False, True),
        ({"skip_episodes": True}, True, False),
    ],
)
async def test_run_pipeline_skip_flags_set_fetch_flags(
    runtime: EnrichmentRuntime,
    runner,
    skips: dict[str, bool],
    fetch_characters: bool,
    fetch_episodes: bool,
) -> None:
    await run_pipeline(runtime, Request(title="Cowboy Bebop", **skips), context=None)

    assert runner.call_args.kwargs["fetch_characters"] is fetch_characters
    assert runner.call_args.kwargs["fetch_episodes"] is fetch_episodes
