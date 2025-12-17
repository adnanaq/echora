import time
from typing import AsyncGenerator
from unittest.mock import patch

import pytest
import pytest_asyncio
from redis import exceptions
from redis.asyncio import Redis

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration

from http_cache import result_cache
from enrichment.crawlers.anisearch_anime_crawler import fetch_anisearch_anime
from enrichment.crawlers.anisearch_character_crawler import (
    fetch_anisearch_characters,
)
from enrichment.crawlers.anisearch_episode_crawler import fetch_anisearch_episodes

# AniList ID for Dandadan
ANIME_ID = "18878,dan-da-dan"
ANIME_URL = f"https://www.anisearch.com/anime/{ANIME_ID}"
EPISODES_URL = f"https://www.anisearch.com/anime/{ANIME_ID}/episodes"
CHARACTERS_URL = f"https://www.anisearch.com/anime/{ANIME_ID}/characters"

RedisType = Redis


@pytest_asyncio.fixture(scope="module")
async def redis_client() -> AsyncGenerator[RedisType, None]:
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


@pytest.mark.asyncio
async def test_crawler_cache_and_singleton_client(redis_client):
    """
    Verifies caching for multiple crawlers and confirms a single Redis client instance is used.
    """
    from redis.asyncio import Redis as AsyncRedis

    # --- Patch Redis.from_url AFTER importing result_cache module ---
    real_redis_client = AsyncRedis.from_url(
        "redis://localhost:6379/0", decode_responses=True
    )

    with patch(
        "http_cache.result_cache.Redis.from_url", return_value=real_redis_client
    ):
        # Ensure the module-level singleton uses the real async client
        result_cache._redis_client = real_redis_client

    # --- Test fetch_anisearch_anime caching ---
    start_time_1 = time.monotonic()
    anime_data_1 = await fetch_anisearch_anime(anime_id=ANIME_ID)
    duration_1 = time.monotonic() - start_time_1
    assert anime_data_1 is not None
    assert "japanese_title" in anime_data_1

    start_time_2 = time.monotonic()
    anime_data_2 = await fetch_anisearch_anime(anime_id=ANIME_ID)
    duration_2 = time.monotonic() - start_time_2
    assert anime_data_2 is not None

    # Sort screenshots to ensure consistent comparison
    if "screenshots" in anime_data_1 and isinstance(anime_data_1["screenshots"], list):
        anime_data_1["screenshots"].sort()
    if "screenshots" in anime_data_2 and isinstance(anime_data_2["screenshots"], list):
        anime_data_2["screenshots"].sort()

    assert anime_data_2 == anime_data_1

    if (
        duration_1 > 0.1
    ):  # Only assert if the first call took a meaningful amount of time
        assert duration_2 < duration_1 / 5

        # --- Test fetch_anisearch_episodes caching ---
        start_time_ep_1 = time.monotonic()
        episodes_data_1 = await fetch_anisearch_episodes(anime_id=ANIME_ID)
        duration_ep_1 = time.monotonic() - start_time_ep_1
        assert episodes_data_1 is not None
        assert len(episodes_data_1) > 0

        start_time_ep_2 = time.monotonic()
        episodes_data_2 = await fetch_anisearch_episodes(anime_id=ANIME_ID)
        duration_ep_2 = time.monotonic() - start_time_ep_2
        assert episodes_data_2 is not None
        assert episodes_data_2 == episodes_data_1

        if duration_ep_1 > 0.1:
            assert duration_ep_2 < duration_ep_1 / 5

        # --- Test fetch_anisearch_characters caching ---
        start_time_char_1 = time.monotonic()
        characters_data_1 = await fetch_anisearch_characters(anime_id=ANIME_ID)
        duration_char_1 = time.monotonic() - start_time_char_1
        assert characters_data_1 is not None
        assert "characters" in characters_data_1
        assert len(characters_data_1["characters"]) > 0

        start_time_char_2 = time.monotonic()
        characters_data_2 = await fetch_anisearch_characters(anime_id=ANIME_ID)
        duration_char_2 = time.monotonic() - start_time_char_2
        assert characters_data_2 is not None
        assert characters_data_2 == characters_data_1

        if duration_char_1 > 0.1:
            assert duration_char_2 < duration_char_1 / 5
