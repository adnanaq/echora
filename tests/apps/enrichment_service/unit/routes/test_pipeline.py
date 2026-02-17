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
