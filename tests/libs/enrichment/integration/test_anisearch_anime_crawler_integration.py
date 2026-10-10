import pytest
from enrichment.sources.anisearch.anisearch_anime_crawler import fetch_anisearch_anime
from http_cache import HTTPCacheManager
from redis.asyncio import Redis

pytestmark = pytest.mark.integration

ANIME_URL = "https://www.anisearch.com/anime/18878,dan-da-dan"
ANISEARCH_TTL = 604800


async def test_fetch_anisearch_anime_second_call_served_from_redis(
    redis_cache: tuple[HTTPCacheManager, Redis],
) -> None:
    _, redis_client = redis_cache

    first = await fetch_anisearch_anime(ANIME_URL)
    keys = await redis_client.keys("result_cache:anisearch_anime:*")
    second = await fetch_anisearch_anime(ANIME_URL)

    assert first is not None
    assert first["title"]
    assert second == first
    assert len(keys) == 1
    assert ANISEARCH_TTL - 60 < await redis_client.ttl(keys[0]) <= ANISEARCH_TTL
    assert await redis_client.keys("result_cache:anisearch_anime:*") == keys
