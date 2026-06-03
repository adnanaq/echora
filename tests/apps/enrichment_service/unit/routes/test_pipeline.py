"""Unit tests for the pipeline route path-containment and agent_dir guards."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from enrichment_proto.v1 import enrichment_service_pb2
from enrichment_service.routes.pipeline import run_pipeline
from enrichment_service.runtime import EnrichmentRuntime


def _make_runtime(tmp_path: Path) -> EnrichmentRuntime:
    default_db = tmp_path / "db" / "anime.json"
    default_db.parent.mkdir(parents=True, exist_ok=True)
    default_db.write_text("{}", encoding="utf-8")
    return EnrichmentRuntime(
        default_file_path=str(default_db),
        output_dir=str(tmp_path / "out"),
    )


@pytest.mark.asyncio
async def test_run_pipeline_rejects_path_traversal(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    request = enrichment_service_pb2.RunPipelineRequest(
        file_path="../../../etc/passwd",
        title="Cowboy Bebop",
    )

    response = await run_pipeline(runtime, request, context=None)

    assert response.success is False
    assert response.error.code == "INVALID_FILE_PATH"
    assert response.error.retryable is False


@pytest.mark.asyncio
async def test_run_pipeline_rejects_absolute_escape_path(tmp_path: Path) -> None:
    runtime = _make_runtime(tmp_path)
    request = enrichment_service_pb2.RunPipelineRequest(
        file_path="/etc/passwd",
        title="Cowboy Bebop",
    )

    response = await run_pipeline(runtime, request, context=None)

    assert response.success is False
    assert response.error.code == "INVALID_FILE_PATH"
    assert response.error.retryable is False


@pytest.mark.asyncio
async def test_run_pipeline_allows_sibling_file_in_allowed_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _make_runtime(tmp_path)
    sibling = Path(runtime.default_file_path).parent / "other.json"
    sibling.write_text("{}", encoding="utf-8")

    request = enrichment_service_pb2.RunPipelineRequest(
        file_path=str(sibling),
        title="Cowboy Bebop",
    )

    with patch(
        "enrichment_service.routes.pipeline.run_pipeline_and_write_artifact",
        new_callable=AsyncMock,
        return_value=(str(tmp_path / "out" / "result.json"), {}, {}),
    ):
        response = await run_pipeline(runtime, request, context=None)

    assert response.success is True
    assert not response.HasField("error")


@pytest.mark.asyncio
async def test_run_pipeline_rejects_agent_dir_with_path_traversal(
    tmp_path: Path,
) -> None:
    runtime = _make_runtime(tmp_path)
    request = enrichment_service_pb2.RunPipelineRequest(
        title="Cowboy Bebop",
        agent_dir="../../../etc",
    )

    response = await run_pipeline(runtime, request, context=None)

    assert response.success is False
    assert response.error.code == "INVALID_AGENT_DIR"
    assert response.error.retryable is False


@pytest.mark.asyncio
async def test_run_pipeline_rejects_agent_dir_with_absolute_path(
    tmp_path: Path,
) -> None:
    runtime = _make_runtime(tmp_path)
    request = enrichment_service_pb2.RunPipelineRequest(
        title="Cowboy Bebop",
        agent_dir="/tmp/evil",
    )

    response = await run_pipeline(runtime, request, context=None)

    assert response.success is False
    assert response.error.code == "INVALID_AGENT_DIR"
    assert response.error.retryable is False


@pytest.mark.asyncio
async def test_run_pipeline_accepts_safe_agent_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _make_runtime(tmp_path)
    request = enrichment_service_pb2.RunPipelineRequest(
        title="Cowboy Bebop",
        agent_dir="Cowboy_agent1",
    )

    with patch(
        "enrichment_service.routes.pipeline.run_pipeline_and_write_artifact",
        new_callable=AsyncMock,
        return_value=(str(tmp_path / "out" / "result.json"), {}, {}),
    ):
        response = await run_pipeline(runtime, request, context=None)

    assert response.success is True
    assert not response.HasField("error")


@pytest.mark.asyncio
async def test_run_pipeline_treats_negative_index_as_unset(
    tmp_path: Path,
) -> None:
    """index=-1 is treated as unset (None) rather than forwarded to the pipeline."""
    runtime = _make_runtime(tmp_path)
    request = enrichment_service_pb2.RunPipelineRequest(
        title="Cowboy Bebop",
        index=-1,
    )

    with patch(
        "enrichment_service.routes.pipeline.run_pipeline_and_write_artifact",
        new_callable=AsyncMock,
        return_value=(str(tmp_path / "out" / "result.json"), {}, {}),
    ) as mock_runner:
        response = await run_pipeline(runtime, request, context=None)

    assert response.success is True
    assert mock_runner.call_args.kwargs["index"] is None


@pytest.mark.asyncio
async def test_run_pipeline_returns_error_on_pipeline_exception(
    tmp_path: Path,
) -> None:
    """Pipeline failure is caught and returned as a structured error response."""
    runtime = _make_runtime(tmp_path)
    request = enrichment_service_pb2.RunPipelineRequest(title="Cowboy Bebop")

    with patch(
        "enrichment_service.routes.pipeline.run_pipeline_and_write_artifact",
        new_callable=AsyncMock,
        side_effect=RuntimeError("pipeline exploded"),
    ):
        response = await run_pipeline(runtime, request, context=None)

    assert response.success is False
    assert response.error.code == "RUN_PIPELINE_FAILED"
    assert "pipeline exploded" in response.error.message


@pytest.mark.asyncio
async def test_skip_characters_true_maps_to_fetch_characters_false(
    tmp_path: Path,
) -> None:
    runtime = _make_runtime(tmp_path)
    request = enrichment_service_pb2.RunPipelineRequest(
        title="Cowboy Bebop",
        skip_characters=True,
    )

    with patch(
        "enrichment_service.routes.pipeline.run_pipeline_and_write_artifact",
        new_callable=AsyncMock,
        return_value=(str(tmp_path / "out" / "result.json"), {}, {}),
    ) as mock_runner:
        await run_pipeline(runtime, request, context=None)

    call_kwargs = mock_runner.call_args.kwargs
    assert call_kwargs["fetch_characters"] is False
    assert call_kwargs["fetch_episodes"] is True


@pytest.mark.asyncio
async def test_skip_episodes_true_maps_to_fetch_episodes_false(
    tmp_path: Path,
) -> None:
    runtime = _make_runtime(tmp_path)
    request = enrichment_service_pb2.RunPipelineRequest(
        title="Cowboy Bebop",
        skip_episodes=True,
    )

    with patch(
        "enrichment_service.routes.pipeline.run_pipeline_and_write_artifact",
        new_callable=AsyncMock,
        return_value=(str(tmp_path / "out" / "result.json"), {}, {}),
    ) as mock_runner:
        await run_pipeline(runtime, request, context=None)

    call_kwargs = mock_runner.call_args.kwargs
    assert call_kwargs["fetch_characters"] is True
    assert call_kwargs["fetch_episodes"] is False


@pytest.mark.asyncio
async def test_default_flags_fetch_all(
    tmp_path: Path,
) -> None:
    """When skip_characters and skip_episodes are not set, both fetch flags are True."""
    runtime = _make_runtime(tmp_path)
    request = enrichment_service_pb2.RunPipelineRequest(title="Cowboy Bebop")

    with patch(
        "enrichment_service.routes.pipeline.run_pipeline_and_write_artifact",
        new_callable=AsyncMock,
        return_value=(str(tmp_path / "out" / "result.json"), {}, {}),
    ) as mock_runner:
        await run_pipeline(runtime, request, context=None)

    call_kwargs = mock_runner.call_args.kwargs
    assert call_kwargs["fetch_characters"] is True
    assert call_kwargs["fetch_episodes"] is True
