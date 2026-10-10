from collections.abc import AsyncIterator
from unittest.mock import NonCallableMagicMock, create_autospec

import aiohttp
import pytest
import pytest_asyncio
import yarl
from common.config import CacheConfig, RedisConfig
from http_cache import HTTPCacheManager, close_cache, configure_cache
from http_cache.aiohttp_adapter import CachedAiohttpSession
from redis import exceptions
from redis.asyncio import Redis

from tests.conftest import TEST_REDIS_URL


@pytest_asyncio.fixture
async def redis_cache() -> AsyncIterator[tuple[HTTPCacheManager, Redis]]:
    redis_client = Redis.from_url(TEST_REDIS_URL, decode_responses=True)
    try:
        await redis_client.ping()
    except exceptions.ConnectionError:
        pytest.skip(f"Redis is not available on {TEST_REDIS_URL}")
    await redis_client.flushdb()
    cache = configure_cache(
        CacheConfig(cache_enabled=True), RedisConfig(redis_url=TEST_REDIS_URL)
    )
    yield cache, redis_client
    await redis_client.flushdb()
    await close_cache()
    await redis_client.aclose()


@pytest.fixture
def not_found_cache() -> NonCallableMagicMock:
    response = create_autospec(aiohttp.ClientResponse, instance=True)
    response.status = 404
    response.url = yarl.URL("https://example.com")
    response.from_cache = False
    response.json.return_value = {}
    session = create_autospec(CachedAiohttpSession, instance=True)
    session.get.return_value.__aenter__.return_value = response
    session.post.return_value.__aenter__.return_value = response
    session.__aenter__.return_value = session
    cache = create_autospec(HTTPCacheManager, instance=True)
    cache.get_aiohttp_session.return_value = session
    return cache
