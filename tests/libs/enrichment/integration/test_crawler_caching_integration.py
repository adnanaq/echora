from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from common.config import CacheConfig, RedisConfig
from enrichment.sources.anisearch.anisearch_anime_crawler import fetch_anisearch_anime
from enrichment.sources.anisearch.anisearch_episode_crawler import (
    fetch_anisearch_episodes,
)
from http_cache import close_cache, configure_cache
from redis import exceptions
from redis.asyncio import Redis

from tests.conftest import TEST_REDIS_URL

pytestmark = pytest.mark.integration

ANIME_URL = "https://www.anisearch.com/anime/18878,dan-da-dan"
ANISEARCH_TTL = 604800


@pytest_asyncio.fixture
async def redis_client() -> AsyncIterator[Redis]:
    client = Redis.from_url(TEST_REDIS_URL, decode_responses=True)
    try:
        await client.ping()
    except exceptions.ConnectionError:
        pytest.skip(f"Redis is not available on {TEST_REDIS_URL}")
    await client.flushdb()
    configure_cache(
        CacheConfig(cache_enabled=True), RedisConfig(redis_url=TEST_REDIS_URL)
    )
    yield client
    await client.flushdb()
    await close_cache()
    await client.aclose()


async def _stored_entry(redis_client: Redis, prefix: str) -> tuple[str, int]:
    keys = await redis_client.keys(f"result_cache:{prefix}:*")
    assert len(keys) == 1
    return keys[0], await redis_client.ttl(keys[0])


async def test_fetch_anisearch_anime_second_call_served_from_redis(
    redis_client: Redis,
) -> None:
    first = await fetch_anisearch_anime(ANIME_URL)
    key, ttl = await _stored_entry(redis_client, "anisearch_anime")
    second = await fetch_anisearch_anime(ANIME_URL)

    assert first is not None
    assert first["title"]
    assert second == first
    assert ANISEARCH_TTL - 60 < ttl <= ANISEARCH_TTL
    assert await redis_client.keys("result_cache:anisearch_anime:*") == [key]


async def test_fetch_anisearch_episodes_second_call_served_from_redis(
    redis_client: Redis,
) -> None:
    first = await fetch_anisearch_episodes(ANIME_URL)
    key, ttl = await _stored_entry(redis_client, "anisearch_episodes")
    second = await fetch_anisearch_episodes(ANIME_URL)

    assert first
    assert second == first
    assert ANISEARCH_TTL - 60 < ttl <= ANISEARCH_TTL
    assert await redis_client.keys("result_cache:anisearch_episodes:*") == [key]
