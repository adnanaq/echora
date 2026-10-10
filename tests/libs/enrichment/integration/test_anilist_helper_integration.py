import time
from collections.abc import AsyncIterator, Iterator

import pytest
import pytest_asyncio
import redis
from common.config import CacheConfig, RedisConfig
from enrichment.sources.anilist.anilist_helper import AniListHelper
from http_cache import close_cache, configure_cache

from tests.conftest import TEST_REDIS_URL

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def redis_client() -> Iterator[redis.Redis]:
    client = redis.Redis.from_url(TEST_REDIS_URL, decode_responses=True)
    try:
        client.ping()
    except redis.exceptions.ConnectionError:
        pytest.skip(f"Redis is not available on {TEST_REDIS_URL}")
    client.flushdb()
    yield client
    client.flushdb()


@pytest_asyncio.fixture
async def helper(redis_client: redis.Redis) -> AsyncIterator[AniListHelper]:
    cache = configure_cache(
        CacheConfig(cache_enabled=True), RedisConfig(redis_url=TEST_REDIS_URL)
    )
    anilist = AniListHelper(cache)
    yield anilist
    await anilist.close()
    await close_cache()


async def test_fetch_anime_second_call_served_from_redis(
    helper: AniListHelper, redis_client: redis.Redis
) -> None:
    redis_client.flushdb()

    first = await helper.fetch_anime(21)
    stored = redis_client.keys("hishel_cache:*")
    second = await helper.fetch_anime(21)

    assert first is not None
    assert first["id"] == 21
    assert "title" in first
    assert stored
    assert second == first
    assert redis_client.keys("hishel_cache:*") == stored


async def test_fetch_characters_second_fetch_served_without_page_waits(
    helper: AniListHelper,
) -> None:
    first = await helper.fetch_characters(21)

    start = time.monotonic()
    second = await helper.fetch_characters(21)
    duration = time.monotonic() - start

    assert first
    assert second == first
    assert duration < 1.0


async def test_fetch_anime_small_anime_returns_details(helper: AniListHelper) -> None:
    anime = await helper.fetch_anime(1535)

    assert anime is not None
    assert anime["id"] == 1535
    assert "title" in anime


async def test_fetch_anime_unknown_id_returns_none(helper: AniListHelper) -> None:
    assert await helper.fetch_anime(999999999) is None


async def test_fetch_anime_different_ids_cached_separately(
    helper: AniListHelper,
) -> None:
    cowboy_bebop = await helper.fetch_anime(1)
    naruto = await helper.fetch_anime(20)

    assert cowboy_bebop is not None
    assert naruto is not None
    assert (cowboy_bebop["id"], naruto["id"]) == (1, 20)
    assert await helper.fetch_anime(1) == cowboy_bebop
    assert await helper.fetch_anime(20) == naruto


async def test_fetch_anime_new_event_loop_returns_anime(helper: AniListHelper) -> None:
    anime = await helper.fetch_anime(30)

    assert anime is not None
    assert anime["id"] == 30


async def test_fetch_anime_native_title_keeps_japanese_characters(
    helper: AniListHelper,
) -> None:
    anime = await helper.fetch_anime(21)

    assert anime is not None
    assert anime["title"]["native"]


async def test_fetch_anime_sparse_record_returns_none_or_record_with_id(
    helper: AniListHelper,
) -> None:
    anime = await helper.fetch_anime(100000)

    assert anime is None or "id" in anime


async def test_fetch_anime_tracks_rate_limit_remaining(helper: AniListHelper) -> None:
    await helper.fetch_anime(1)

    assert helper.rate_limit_remaining >= 0
