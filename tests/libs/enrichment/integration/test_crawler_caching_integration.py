"""Integration: AniSearch crawlers cache their results in Redis.

Each crawler is called twice for the same URL. The second call must return the
same payload and skip the browser entirely, which is what the result cache
exists to do.
"""

import time
from collections.abc import AsyncGenerator
from unittest.mock import patch

import pytest
import pytest_asyncio
from enrichment.sources.anisearch.anisearch_anime_crawler import fetch_anisearch_anime
from enrichment.sources.anisearch.anisearch_episode_crawler import (
    fetch_anisearch_episodes,
)
from http_cache import result_cache
from redis import exceptions
from redis.asyncio import Redis

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration

ANIME_SLUG = "18878,dan-da-dan"
ANIME_URL = f"https://www.anisearch.com/anime/{ANIME_SLUG}"

# A cached call must be dramatically faster than one that drives a browser.
_CACHE_SPEEDUP = 5
_MEANINGFUL_DURATION_S = 0.1

RedisType = Redis


@pytest_asyncio.fixture(scope="module")
async def redis_client() -> AsyncGenerator[RedisType]:
    """Async Redis fixture for tests."""
    client = Redis.from_url("redis://localhost:6379/0", decode_responses=True)

    try:
        await client.ping()
    except exceptions.ConnectionError:
        pytest.skip("Redis is not available on redis://localhost:6379/0")

    await client.flushall()

    try:
        yield client
    finally:
        await client.flushall()
        try:
            await client.close()
        except RuntimeError:
            pass


@pytest_asyncio.fixture(scope="module")
async def browser_available() -> None:
    """Skip unless a browser can actually start here.

    The AniSearch crawlers drive a headful Chrome, which needs a display. The
    Pants sandbox does not pass one through, so without this the whole module
    fails as if the crawlers were broken.
    """
    import zendriver as zd

    try:
        browser = await zd.start(headless=False)
    except Exception as exc:
        pytest.skip(f"no usable browser in this environment: {exc}")
    else:
        await browser.stop()


@pytest_asyncio.fixture
async def shared_redis(redis_client, browser_available):
    """Point the result cache at the same client the assertions inspect."""
    from redis.asyncio import Redis as AsyncRedis

    real_client = AsyncRedis.from_url("redis://localhost:6379/0", decode_responses=True)
    with patch("http_cache.result_cache.Redis.from_url", return_value=real_client):
        result_cache._redis_client = real_client
        yield real_client


async def _timed(coro_factory):
    """Await a freshly built coroutine and return (result, elapsed_seconds)."""
    start = time.monotonic()
    result = await coro_factory()
    return result, time.monotonic() - start


def _assert_second_call_was_cached(first_s: float, second_s: float) -> None:
    if first_s > _MEANINGFUL_DURATION_S:
        assert second_s < first_s / _CACHE_SPEEDUP


@pytest.mark.asyncio
async def test_anime_crawler_caches_result(shared_redis):
    first, first_s = await _timed(lambda: fetch_anisearch_anime(ANIME_URL))
    assert first is not None
    assert first["title"]

    second, second_s = await _timed(lambda: fetch_anisearch_anime(ANIME_URL))
    assert second == first
    _assert_second_call_was_cached(first_s, second_s)


@pytest.mark.asyncio
async def test_episode_crawler_caches_result(shared_redis):
    first, first_s = await _timed(lambda: fetch_anisearch_episodes(ANIME_URL))
    assert first is not None
    assert len(first) > 0

    second, second_s = await _timed(lambda: fetch_anisearch_episodes(ANIME_URL))
    assert second == first
    _assert_second_call_was_cached(first_s, second_s)
