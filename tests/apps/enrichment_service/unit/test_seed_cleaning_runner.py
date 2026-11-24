"""Unit tests for in-service seed-cleaning runner and Jev gRPC handoff."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from common.config import EnrichmentConfig
from enrichment_service.seed_cleaning_runner import SeedCleaningRunner
from http_cache import HTTPCacheManager


@pytest.mark.asyncio
async def test_seed_cleaning_runner_starts_once_when_idle_and_records_exit_code(
    tmp_path: Path,
) -> None:
    """Verify start_run launches a background task once and records exit code 0."""
    data_dir = tmp_path / "seed_cleaning"
    data_dir.mkdir()
    seed_file = tmp_path / "seed.json"
    seed_file.write_text('{"data": []}', encoding="utf-8")

    runner = SeedCleaningRunner(
        enrichment=EnrichmentConfig(),
        cache=MagicMock(spec=HTTPCacheManager),
        agent_service_address="127.0.0.1:8005",
    )
    executed = asyncio.Event()

    async def _fake_execute(
        target_seed_file: Path,
        target_data_dir: Path,
        *,
        no_fetch: bool,
        refetch_stored: bool,
    ) -> None:
        assert target_seed_file == seed_file
        assert target_data_dir == data_dir
        assert no_fetch is True
        assert refetch_stored is False
        executed.set()

    with patch.object(runner, "_execute_run", side_effect=_fake_execute):
        first_state = runner.start_run(
            seed_file=seed_file,
            data_dir=data_dir,
            no_fetch=True,
        )
        second_state = runner.start_run(
            seed_file=seed_file,
            data_dir=data_dir,
            no_fetch=True,
        )
        assert first_state.is_running is True
        assert second_state.is_running is True
        await asyncio.wait_for(executed.wait(), timeout=2.0)
        await asyncio.sleep(0)

    final_state = runner.read_run_state(data_dir)
    assert final_state.is_running is False
    assert final_state.last_exit_code == 0


@pytest.mark.asyncio
async def test_seed_cleaning_runner_records_nonzero_exit_code_on_failure(
    tmp_path: Path,
) -> None:
    """Verify failed runs persist exit code 1 and clear running state."""
    data_dir = tmp_path / "seed_cleaning"
    data_dir.mkdir()
    seed_file = tmp_path / "seed.json"
    seed_file.write_text('{"data": []}', encoding="utf-8")

    runner = SeedCleaningRunner(
        enrichment=EnrichmentConfig(),
        cache=MagicMock(spec=HTTPCacheManager),
        agent_service_address="127.0.0.1:8005",
    )

    with patch.object(
        runner, "_execute_run", AsyncMock(side_effect=RuntimeError("boom"))
    ):
        runner.start_run(seed_file=seed_file, data_dir=data_dir, no_fetch=True)
        await asyncio.sleep(0.05)

    state = runner.read_run_state(data_dir)
    assert state.is_running is False
    assert state.last_exit_code == 1


def test_seed_cleaning_runner_detects_external_review_file_lock(
    tmp_path: Path,
) -> None:
    """Verify read_run_state detects an active external review.jsonl lock."""
    data_dir = tmp_path / "seed_cleaning"
    data_dir.mkdir()
    runner = SeedCleaningRunner(
        enrichment=EnrichmentConfig(),
        cache=MagicMock(spec=HTTPCacheManager),
        agent_service_address="127.0.0.1:8005",
    )

    with patch(
        "enrichment_service.seed_cleaning_runner.is_review_file_locked",
        return_value=True,
    ):
        state = runner.read_run_state(data_dir)

    assert state.is_running is True
    assert not state.HasField("last_exit_code")
