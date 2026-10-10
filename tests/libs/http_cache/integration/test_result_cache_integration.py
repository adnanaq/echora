import asyncio
import os
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from common.config import CacheConfig, RedisConfig
from http_cache import close_cache, configure_cache, get_cache
from http_cache.result_cache import cached_result

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def cache() -> AsyncIterator[None]:
    manager = configure_cache(
        CacheConfig(cache_enabled=True),
        RedisConfig(redis_url=os.environ["REDIS_URL"]),
    )
    result_client = manager.get_result_client()
    assert result_client is not None
    await result_client.flushdb()
    yield
    await result_client.flushdb()
    await close_cache()


@pytest.mark.usefixtures("cache")
async def test_cached_result_concurrent_calls_share_one_client_and_serve_hits() -> None:
    calls: list[int] = []

    @cached_result(service="mal", key_prefix="integration")
    async def fetch(item_id: int) -> dict[str, int]:
        calls.append(item_id)
        return {"id": item_id}

    client_before = get_cache().get_result_client()
    first = await asyncio.gather(*(fetch(item_id) for item_id in range(10)))
    second = await asyncio.gather(*(fetch(item_id) for item_id in range(10)))

    assert first == second == [{"id": item_id} for item_id in range(10)]
    assert sorted(calls) == list(range(10))
    assert get_cache().get_result_client() is client_before


@pytest.mark.usefixtures("cache")
async def test_cached_result_stored_entry_expires_after_service_ttl() -> None:
    @cached_result(service="animeschedule", key_prefix="integration_ttl")
    async def fetch(item_id: int) -> dict[str, int]:
        return {"id": item_id}

    await fetch(1)

    redis_client = get_cache().get_result_client()
    assert redis_client is not None
    keys = await redis_client.keys("result_cache:integration_ttl:*")
    assert len(keys) == 1
    assert 86000 < await redis_client.ttl(keys[0]) <= 86400


@pytest.mark.usefixtures("cache")
async def test_cache_batch_set_then_get_round_trips_through_redis() -> None:
    @cached_result(service="anisearch", key_prefix="integration_batch")
    async def fetch(item_id: str) -> dict[str, str]:
        return {"id": item_id}

    await fetch.cache_batch_set(["a", "b"], [{"id": "a"}, None])

    assert await fetch.cache_batch_get(["a", "b"]) == ([{"id": "a"}, None], [1])
