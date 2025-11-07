"""
Integration tests for AniListEnrichmentHelper to verify caching and real API calls.

These tests make real API calls to AniList GraphQL API to validate:
- Caching functionality with Redis
- Event loop management
- Rate limiting with cache optimization
- Pagination
"""

import logging

import pytest
import pytest_asyncio
import redis

from src.enrichment.api_helpers.anilist_helper import AniListEnrichmentHelper

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def redis_client():
    """Provides a Redis client for the test module, skipping tests if Redis is unavailable."""
    try:
        r = redis.Redis.from_url("redis://localhost:6379/0", decode_responses=True)
        r.ping()
        # Flush once at module start
        r.flushall()
        yield r
        # Optionally flush at module end
        r.flushall()
    except redis.exceptions.ConnectionError:
        pytest.skip("Redis is not available on redis://localhost:6379/0")


@pytest_asyncio.fixture
async def clean_helper():
    """Provide a fresh helper instance with event-loop-safe session management."""
    import asyncio
    import logging

    logger = logging.getLogger(__name__)
    logger.info(f"[clean_helper] SETUP - Event loop: {id(asyncio.get_running_loop())}")

    helper = AniListEnrichmentHelper()

    yield helper

    logger.info("[clean_helper] TEARDOWN - Closing helper")
    # Close helper session
    await helper.close()


@pytest.mark.asyncio
async def test_anilist_fetch_anime_caching(redis_client, clean_helper):
    """
    Integration test to verify that AniList API calls are cached.
    Tests fetching anime data with cache hit detection.
    """
    import time

    helper = clean_helper

    # First call (should be a cache miss)
    start_time_1 = time.monotonic()
    result_1 = await helper.fetch_anime_by_anilist_id(21)  # One Piece
    end_time_1 = time.monotonic()
    duration_1 = end_time_1 - start_time_1

    assert result_1 is not None
    assert result_1["id"] == 21
    assert "title" in result_1

    # Second call (should be a cache hit)
    start_time_2 = time.monotonic()
    result_2 = await helper.fetch_anime_by_anilist_id(21)
    end_time_2 = time.monotonic()
    duration_2 = end_time_2 - start_time_2

    assert result_2 is not None
    assert result_2 == result_1

    print(f"First call (cache miss): {duration_1:.4f}s")
    print(f"Second call (cache hit): {duration_2:.4f}s")

    # Cache hit should be significantly faster
    if duration_1 > 0.1:
        assert duration_2 < duration_1 / 3


@pytest.mark.asyncio
async def test_anilist_pagination_caching(redis_client, clean_helper):
    """
    Test that paginated requests are cached properly.
    Verifies cache hit detection optimizes rate limiting.
    """
    helper = clean_helper

    # Fetch characters (paginated)
    characters_1 = await helper.fetch_all_characters(
        21
    )  # One Piece has many characters

    assert len(characters_1) > 0
    print(f"First fetch: {len(characters_1)} characters")

    # Fetch again (should hit cache for all pages)
    import time

    start_time = time.monotonic()
    characters_2 = await helper.fetch_all_characters(21)
    end_time = time.monotonic()
    duration = end_time - start_time

    assert characters_2 == characters_1
    print(f"Second fetch (cached): {duration:.4f}s")

    # Cache hit should be very fast (no 0.5s sleep between pages)
    assert duration < 1.0


@pytest.mark.asyncio
async def test_anilist_full_data_fetch(redis_client, clean_helper):
    """
    Test fetching full anime data including characters, staff, and episodes.
    """
    helper = clean_helper

    # Fetch complete data
    anime_data = await helper.fetch_all_data_by_anilist_id(
        1535
    )  # Death Note (smaller dataset)

    assert anime_data is not None
    assert anime_data["id"] == 1535
    assert "title" in anime_data

    # Should have populated details
    if "characters" in anime_data:
        assert "edges" in anime_data["characters"]
        print(f"Characters fetched: {len(anime_data['characters']['edges'])}")

    if "staff" in anime_data:
        assert "edges" in anime_data["staff"]
        print(f"Staff fetched: {len(anime_data['staff']['edges'])}")


@pytest.mark.asyncio
async def test_anilist_not_found(redis_client, clean_helper):
    """
    Test handling of non-existent anime ID.
    """
    helper = clean_helper

    result = await helper.fetch_anime_by_anilist_id(999999999)

    # Should return None for non-existent ID
    assert result is None


@pytest.mark.asyncio
async def test_anilist_multiple_anime(redis_client, clean_helper):
    """
    Test fetching multiple different anime to verify caching per ID.
    """
    helper = clean_helper

    # Fetch different anime
    anime1 = await helper.fetch_anime_by_anilist_id(1)  # Cowboy Bebop
    anime2 = await helper.fetch_anime_by_anilist_id(20)  # Naruto

    assert anime1 is not None
    assert anime2 is not None
    assert anime1["id"] == 1
    assert anime2["id"] == 20
    assert anime1 != anime2

    # Fetch again (should hit cache)
    anime1_cached = await helper.fetch_anime_by_anilist_id(1)
    anime2_cached = await helper.fetch_anime_by_anilist_id(20)

    assert anime1_cached == anime1
    assert anime2_cached == anime2


@pytest.mark.asyncio
async def test_anilist_event_loop_safety(redis_client):
    """
    Test that helper works correctly across multiple test functions with different event loops.
    This test creates its own helper to verify event loop recreation.
    """
    import asyncio

    logger = logging.getLogger(__name__)
    logger.info(f"Event loop safety test - loop: {id(asyncio.get_running_loop())}")

    helper = AniListEnrichmentHelper()

    try:
        result = await helper.fetch_anime_by_anilist_id(30)  # Neon Genesis Evangelion

        assert result is not None
        assert result["id"] == 30
    finally:
        await helper.close()


@pytest.mark.asyncio
async def test_anilist_unicode_handling(redis_client, clean_helper):
    """
    Test handling of Unicode in anime titles (Japanese characters).
    """
    helper = clean_helper

    # Fetch anime with Japanese title
    result = await helper.fetch_anime_by_anilist_id(21)  # One Piece

    assert result is not None
    assert "title" in result

    # Should have native title with Japanese characters
    if "native" in result["title"]:
        assert len(result["title"]["native"]) > 0
        print(f"Native title: {result['title']['native']}")


@pytest.mark.asyncio
async def test_anilist_empty_results(redis_client, clean_helper):
    """
    Test handling of anime with no characters/staff/episodes data.
    """
    helper = clean_helper

    # Some anime might have minimal data
    # Test graceful handling
    try:
        result = await helper.fetch_all_data_by_anilist_id(
            100000
        )  # Unlikely to have full data

        # Should handle gracefully (either None or partial data)
        if result:
            assert "id" in result
    except Exception as e:
        # Should not raise unhandled exceptions
        pytest.fail(f"Should handle missing data gracefully: {e}")


@pytest.mark.asyncio
async def test_anilist_rate_limit_headers(redis_client, clean_helper):
    """
    Test that rate limit headers are tracked correctly.
    """
    helper = clean_helper

    initial_rate_limit = helper.rate_limit_remaining

    # Make a request
    await helper.fetch_anime_by_anilist_id(1)

    # Rate limit should be tracked (either updated from headers or unchanged)
    assert helper.rate_limit_remaining >= 0
    print(f"Initial rate limit: {initial_rate_limit}")
    print(f"After request: {helper.rate_limit_remaining}")
