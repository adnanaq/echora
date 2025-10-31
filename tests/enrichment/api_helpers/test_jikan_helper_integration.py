"""
Integration test for JikanEnrichmentHelper to verify caching and singleton client usage.
"""

import time
import pytest
import redis
from src.enrichment.api_helpers.jikan_helper import JikanDetailedFetcher


@pytest.fixture(scope="module")
def redis_client():
    """Provides a Redis client for the test module, skipping tests if Redis is unavailable."""
    try:
        r = redis.Redis.from_url("redis://localhost:6379/0", decode_responses=True)
        r.ping()
        return r
    except redis.exceptions.ConnectionError:
        pytest.skip("Redis is not available on redis://localhost:6379/0")


@pytest.fixture(autouse=True)
def flush_redis(redis_client):
    """Ensures the Redis database is flushed before each test."""
    redis_client.flushall()


@pytest.mark.asyncio
async def test_jikan_episode_caching(redis_client):
    """
    Integration test to verify that Jikan API calls for episodes are cached.
    """
    fetcher = JikanDetailedFetcher(anime_id="21", data_type="episodes")

    # First call (should be a cache miss)
    start_time_1 = time.monotonic()
    result_1 = await fetcher.fetch_episode_detail(1)
    end_time_1 = time.monotonic()
    duration_1 = end_time_1 - start_time_1

    assert result_1 is not None
    assert result_1["episode_number"] == 1

    # Second call (should be a cache hit)
    start_time_2 = time.monotonic()
    result_2 = await fetcher.fetch_episode_detail(1)
    end_time_2 = time.monotonic()
    duration_2 = end_time_2 - start_time_2

    assert result_2 is not None
    assert result_2 == result_1

    print(f"First call (cache miss): {duration_1:.4f}s")
    print(f"Second call (cache hit): {duration_2:.4f}s")

    if duration_1 > 0.1:
        assert duration_2 < duration_1 / 5

    await fetcher.session.close()


@pytest.mark.asyncio
async def test_jikan_single_redis_instance():
    """
    Verifies that a single Redis client instance is used across multiple
    JikanDetailedFetcher instantiations.
    """
    fetcher1 = JikanDetailedFetcher(anime_id="21", data_type="episodes")
    fetcher2 = JikanDetailedFetcher(anime_id="20", data_type="episodes")

    client1 = fetcher1.session.storage.client
    client2 = fetcher2.session.storage.client

    assert client1 is client2

    # The sessions create storage that don't own the client, so closing them is safe.
    await fetcher1.session.close()
    await fetcher2.session.close()