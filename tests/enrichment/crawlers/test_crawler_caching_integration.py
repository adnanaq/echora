"""
Integration test for crawler caching and singleton Redis client verification.
"""

import time
import pytest
import redis
from unittest.mock import patch, AsyncMock

from src.enrichment.crawlers.anisearch_anime_crawler import fetch_anisearch_anime
from src.enrichment.crawlers.anisearch_character_crawler import fetch_anisearch_characters
from src.enrichment.crawlers.anisearch_episode_crawler import fetch_anisearch_episodes
from src.cache_manager.result_cache import close_result_cache_redis_client

# AniList ID for Dandadan
ANIME_ID = "18878,dan-da-dan"
ANIME_URL = f"https://www.anisearch.com/anime/{ANIME_ID}"
EPISODES_URL = f"https://www.anisearch.com/anime/{ANIME_ID}/episodes"
CHARACTERS_URL = f"https://www.anisearch.com/anime/{ANIME_ID}/characters"


@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown_redis_client():
    """Ensures Redis is available and cleans up the singleton client after tests."""
    try:
        r = redis.Redis.from_url("redis://localhost:6379/0", decode_responses=True)
        r.ping()
    except redis.exceptions.ConnectionError:
        pytest.skip("Redis is not available on redis://localhost:6379/0")

    yield
    # Teardown: Close the singleton client after all tests in this module
    pytest.mark.asyncio(close_result_cache_redis_client)()


@pytest.fixture(autouse=True)
async def flush_redis_before_each_test(redis_client):
    """Flushes Redis before each test to ensure a clean cache state."""
    await redis_client.flushall()


@pytest.mark.asyncio
@patch('src.cache_manager.result_cache.Redis.from_url')
async def test_crawler_cache_and_singleton_client(mock_redis_from_url):
    """
    Verifies caching for multiple crawlers and confirms a single Redis client instance is used.
    """
    # Configure the mock to return a real Redis client instance
    # This allows the underlying caching logic to interact with a real Redis server
    real_redis_client = redis.asyncio.Redis.from_url("redis://localhost:6379/0", decode_responses=True)
    mock_redis_from_url.return_value = real_redis_client

    # --- Test fetch_anisearch_anime caching ---
    print("\n--- Testing fetch_anisearch_anime caching ---")
    start_time_1 = time.monotonic()
    anime_data_1 = await fetch_anisearch_anime(url=ANIME_URL)
    duration_1 = time.monotonic() - start_time_1
    assert anime_data_1 is not None
    assert "japanese_title" in anime_data_1
    print(f"fetch_anisearch_anime (1st call - miss): {duration_1:.4f}s")

    start_time_2 = time.monotonic()
    anime_data_2 = await fetch_anisearch_anime(url=ANIME_URL)
    duration_2 = time.monotonic() - start_time_2
    assert anime_data_2 is not None
    assert anime_data_2 == anime_data_1
    print(f"fetch_anisearch_anime (2nd call - hit): {duration_2:.4f}s")

    if duration_1 > 0.1: # Only assert if the first call took a meaningful amount of time
        assert duration_2 < duration_1 / 5

    # --- Test fetch_anisearch_episodes caching ---
    print("\n--- Testing fetch_anisearch_episodes caching ---")
    start_time_ep_1 = time.monotonic()
    episodes_data_1 = await fetch_anisearch_episodes(url=EPISODES_URL)
    duration_ep_1 = time.monotonic() - start_time_ep_1
    assert episodes_data_1 is not None
    assert len(episodes_data_1) > 0
    print(f"fetch_anisearch_episodes (1st call - miss): {duration_ep_1:.4f}s")

    start_time_ep_2 = time.monotonic()
    episodes_data_2 = await fetch_anisearch_episodes(url=EPISODES_URL)
    duration_ep_2 = time.monotonic() - start_time_ep_2
    assert episodes_data_2 is not None
    assert episodes_data_2 == episodes_data_1
    print(f"fetch_anisearch_episodes (2nd call - hit): {duration_ep_2:.4f}s")

    if duration_ep_1 > 0.1:
        assert duration_ep_2 < duration_ep_1 / 5

    # --- Test fetch_anisearch_characters caching ---
    print("\n--- Testing fetch_anisearch_characters caching ---")
    start_time_char_1 = time.monotonic()
    characters_data_1 = await fetch_anisearch_characters(url=CHARACTERS_URL)
    duration_char_1 = time.monotonic() - start_time_char_1
    assert characters_data_1 is not None
    assert "characters" in characters_data_1
    assert len(characters_data_1["characters"]) > 0
    print(f"fetch_anisearch_characters (1st call - miss): {duration_char_1:.4f}s")

    start_time_char_2 = time.monotonic()
    characters_data_2 = await fetch_anisearch_characters(url=CHARACTERS_URL)
    duration_char_2 = time.monotonic() - start_time_char_2
    assert characters_data_2 is not None
    assert characters_data_2 == characters_data_1
    print(f"fetch_anisearch_characters (2nd call - hit): {duration_char_2:.4f}s")

    if duration_char_1 > 0.1:
        assert duration_char_2 < duration_char_1 / 5

    # --- Verify single Redis client instance ---
    # The mock should have been called only once across all crawler calls
    mock_redis_from_url.assert_called_once()
    print("\nConfirmed: Redis.from_url was called exactly once.")

    # Ensure the real client is closed after the test
    await real_redis_client.close()
