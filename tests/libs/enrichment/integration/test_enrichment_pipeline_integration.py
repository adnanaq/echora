import asyncio
from pathlib import Path

import pytest
from common.config import EnrichmentConfig
from enrichment.pipeline.enrichment_pipeline import EnrichmentPipeline
from http_cache import get_cache

pytestmark = pytest.mark.integration


@pytest.fixture
def config(tmp_path: Path) -> EnrichmentConfig:
    return EnrichmentConfig(enrichment_temp_dir=str(tmp_path))


def _agent_ids(directories: list[str]) -> list[int]:
    return sorted(int(Path(path).name.split("_agent")[1]) for path in directories)


async def test_create_temp_dir_concurrent_agents_get_unique_sequential_ids(
    config: EnrichmentConfig,
) -> None:
    async def create(_: int) -> str:
        return EnrichmentPipeline(config, get_cache())._create_temp_dir("TestAnime")

    directories = await asyncio.gather(*(create(agent) for agent in range(5)))

    assert _agent_ids(directories) == [1, 2, 3, 4, 5]
    assert all(Path(path).is_dir() for path in directories)


async def test_create_temp_dir_concurrent_agents_fill_gap_then_continue(
    config: EnrichmentConfig,
) -> None:
    for agent_id in (1, 2, 4):
        (Path(config.enrichment_temp_dir) / f"GapTest_agent{agent_id}").mkdir()

    async def create(_: int) -> str:
        return EnrichmentPipeline(config, get_cache())._create_temp_dir("GapTest")

    directories = await asyncio.gather(*(create(agent) for agent in range(3)))

    assert _agent_ids(directories) == [3, 5, 6]
