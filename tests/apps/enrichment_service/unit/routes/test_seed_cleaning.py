"""Unit tests for EnrichmentService seed-cleaning gRPC route handlers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import grpc
import pytest
from common.config import EnrichmentConfig
from enrichment_proto.v1 import enrichment_service_pb2
from enrichment_service.routes.seed_cleaning import (
    get_seed_cleaning_status,
    start_seed_cleaning,
)
from enrichment_service.runtime import EnrichmentRuntime
from http_cache import HTTPCacheManager
from shared_proto.v1 import review_cases_pb2


def _build_runtime(seed_cleaning_runner: MagicMock) -> EnrichmentRuntime:
    return EnrichmentRuntime(
        default_file_path="assets/seed_data/anime-offline-database.json",
        output_dir="assets/seed_data",
        enrichment=EnrichmentConfig(),
        cache=MagicMock(spec=HTTPCacheManager),
        seed_cleaning_runner=seed_cleaning_runner,
    )


@pytest.mark.asyncio
async def test_start_seed_cleaning_route_delegates_to_runner(tmp_path: Path) -> None:
    """Verify StartSeedCleaning validates inputs and calls seed_cleaning_runner."""
    runner = MagicMock()
    runner.start_run.return_value = review_cases_pb2.RunState(is_running=True)
    runtime = _build_runtime(runner)
    context = MagicMock(spec=grpc.aio.ServicerContext)

    seed_file = tmp_path / "seed.json"
    data_dir = tmp_path / "data"
    request = enrichment_service_pb2.StartSeedCleaningRequest(
        seed_file=str(seed_file),
        data_dir=str(data_dir),
        no_fetch=True,
    )

    response = await start_seed_cleaning(runtime, request, context)

    assert response.is_running is True
    runner.start_run.assert_called_once_with(
        seed_file=seed_file,
        data_dir=data_dir,
        no_fetch=True,
        refetch_stored=False,
    )


@pytest.mark.asyncio
async def test_start_seed_cleaning_route_rejects_missing_seed_file(
    tmp_path: Path,
) -> None:
    """Verify StartSeedCleaning aborts with INVALID_ARGUMENT when seed_file is empty."""
    runner = MagicMock()
    runtime = _build_runtime(runner)
    context = MagicMock(spec=grpc.aio.ServicerContext)
    context.abort = AsyncMock(side_effect=RuntimeError("aborted"))

    request = enrichment_service_pb2.StartSeedCleaningRequest(
        seed_file="",
        data_dir=str(tmp_path),
        no_fetch=True,
    )

    with pytest.raises(RuntimeError, match="aborted"):
        await start_seed_cleaning(runtime, request, context)

    assert context.abort.await_args.args[0] == grpc.StatusCode.INVALID_ARGUMENT


@pytest.mark.asyncio
async def test_get_seed_cleaning_status_route_reads_runner_state(
    tmp_path: Path,
) -> None:
    """Verify GetSeedCleaningStatus returns runner state for the requested folder."""
    runner = MagicMock()
    runner.read_run_state.return_value = review_cases_pb2.RunState(
        is_running=False, last_exit_code=0
    )
    runtime = _build_runtime(runner)
    context = MagicMock(spec=grpc.aio.ServicerContext)

    request = enrichment_service_pb2.GetSeedCleaningStatusRequest(
        data_dir=str(tmp_path)
    )
    response = await get_seed_cleaning_status(runtime, request, context)

    assert response.is_running is False
    assert response.last_exit_code == 0
    runner.read_run_state.assert_called_once_with(tmp_path)
