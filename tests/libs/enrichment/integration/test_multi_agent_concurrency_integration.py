import asyncio
import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from common.config import CacheConfig, EnrichmentConfig, RedisConfig
from enrichment.pipeline.enrichment_pipeline import EnrichmentPipeline
from enrichment.sources.anilist.anilist_helper import AniListHelper
from http_cache import HTTPCacheManager, close_cache, configure_cache, get_cache
from http_cache.aiohttp_adapter import CachedAiohttpSession

pytestmark = pytest.mark.integration

skip_live_tests = pytest.mark.skipif(
    not os.getenv("ENABLE_LIVE_CONCURRENCY_TESTS"),
    reason="Needs the AniList API and Redis; set ENABLE_LIVE_CONCURRENCY_TESTS=1",
)


@pytest.fixture
def config(tmp_path: Path) -> EnrichmentConfig:
    return EnrichmentConfig(enrichment_temp_dir=str(tmp_path))


@pytest_asyncio.fixture
async def redis_cache() -> AsyncIterator[HTTPCacheManager]:
    cache = configure_cache(
        CacheConfig(cache_enabled=True),
        RedisConfig(redis_url=os.environ["REDIS_URL"]),
    )
    yield cache
    await close_cache()


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


async def test_get_aiohttp_session_concurrent_calls_share_one_redis_client(
    redis_cache: HTTPCacheManager,
) -> None:
    async def open_session(task_id: int) -> CachedAiohttpSession:
        session = redis_cache.get_aiohttp_session(f"service_{task_id}")
        assert isinstance(session, CachedAiohttpSession)
        return session

    sessions = await asyncio.gather(*(open_session(task) for task in range(10)))

    assert len({id(session.storage.client) for session in sessions}) == 1
    for session in sessions:
        await session.close()


@skip_live_tests
async def test_make_request_concurrent_agents_receive_same_anime(
    redis_cache: HTTPCacheManager,
) -> None:
    async def fetch(_: int) -> dict:
        helper = AniListHelper(redis_cache)
        try:
            return await helper._make_request(
                helper._build_query_by_anilist_id(), {"id": 1535}
            )
        finally:
            await helper.close()

    results = await asyncio.gather(*(fetch(agent) for agent in range(3)))

    assert {result["Media"]["id"] for result in results} == {1535}


@skip_live_tests
async def test_fetch_anime_concurrent_anime_all_succeed(
    redis_cache: HTTPCacheManager,
) -> None:
    async def fetch(anilist_id: int) -> dict | None:
        helper = AniListHelper(redis_cache)
        try:
            return await helper.fetch_anime(anilist_id)
        finally:
            await helper.close()

    results = await asyncio.gather(*(fetch(anime) for anime in (1535, 5114, 16498)))

    assert all(results)
