"""
Tests for anime_planet_character_crawler.py main() function.
"""

from unittest.mock import patch

import pytest

# --- Tests for main() function ---


@pytest.mark.asyncio
@patch(
    "src.enrichment.crawlers.anime_planet_character_crawler.fetch_animeplanet_characters"
)
async def test_main_function_success(mock_fetch):
    """Test main() function handles successful execution."""
    from src.enrichment.crawlers.anime_planet_character_crawler import main

    mock_fetch.return_value = {"characters": [{"name": "Test Character"}], "total": 1}

    with patch("sys.argv", ["script.py", "dandadan", "--output", "/tmp/output.json"]):
        exit_code = await main()

    assert exit_code == 0
    # Verify the function was called (args vs kwargs may vary by implementation)
    mock_fetch.assert_awaited_once()
    call_args = mock_fetch.call_args
    # Check the identifier was passed correctly
    if call_args[0]:
        assert call_args[0][0] == "dandadan"
    else:
        assert call_args[1]["identifier"] == "dandadan"


@pytest.mark.asyncio
@patch(
    "src.enrichment.crawlers.anime_planet_character_crawler.fetch_animeplanet_characters"
)
async def test_main_function_with_default_output(mock_fetch):
    """Test main() function with default output path."""
    from src.enrichment.crawlers.anime_planet_character_crawler import main

    mock_fetch.return_value = {"characters": [], "total": 0}

    with patch("sys.argv", ["script.py", "test-slug"]):
        exit_code = await main()

    assert exit_code == 0
    # Verify default output path used
    call_args = mock_fetch.call_args
    assert call_args[1]["output_path"] == "animeplanet_characters.json"


@pytest.mark.asyncio
@patch(
    "src.enrichment.crawlers.anime_planet_character_crawler.fetch_animeplanet_characters"
)
async def test_main_function_error_handling(mock_fetch):
    """Test main() function handles errors and returns non-zero exit code."""
    from src.enrichment.crawlers.anime_planet_character_crawler import main

    mock_fetch.side_effect = Exception("Crawler error")

    with patch("sys.argv", ["script.py", "test-slug"]):
        exit_code = await main()

    assert exit_code == 1


@pytest.mark.asyncio
@patch(
    "src.enrichment.crawlers.anime_planet_character_crawler.fetch_animeplanet_characters"
)
async def test_main_function_with_full_url(mock_fetch):
    """Test main() function with full URL as identifier."""
    from src.enrichment.crawlers.anime_planet_character_crawler import main

    mock_fetch.return_value = {"characters": [], "total": 0}

    with patch(
        "sys.argv",
        ["script.py", "https://www.anime-planet.com/anime/planet/characters"],
    ):
        exit_code = await main()

    assert exit_code == 0
    mock_fetch.assert_awaited_once()


# --- Tests for cache key generation ---


@pytest.mark.asyncio
async def test_cache_key_only_depends_on_slug():
    """
    Test that cache key only depends on slug parameter, not return_data or output_path.

    This ensures that calling fetch_animeplanet_characters with the same slug but different
    return_data or output_path values reuses the same cache entry instead of creating
    duplicate cache entries.

    This test demonstrates the bug where cache keys include all parameters.
    """
    from unittest.mock import AsyncMock, MagicMock
    from src.enrichment.crawlers.anime_planet_character_crawler import (
        fetch_animeplanet_characters,
    )

    # Track cache keys used
    cache_keys_used = []

    # Mock cache config to enable caching
    mock_cache_config = MagicMock()
    mock_cache_config.enabled = True
    mock_cache_config.storage_type = "redis"

    # Mock the Redis client and AsyncWebCrawler
    with patch(
        "src.cache_manager.result_cache.get_cache_config",
        return_value=mock_cache_config,
    ), patch(
        "src.cache_manager.result_cache.get_result_cache_redis_client"
    ) as mock_redis, patch(
        "src.enrichment.crawlers.anime_planet_character_crawler.AsyncWebCrawler"
    ):
        # Setup mock Redis client
        redis_client = AsyncMock()

        # Capture cache keys when get() is called
        async def track_get(key: str):
            cache_keys_used.append(key)
            return None  # Simulate cache miss

        redis_client.get = track_get
        redis_client.setex = AsyncMock()
        mock_redis.return_value = redis_client

        # Mock the crawler to avoid actual HTTP requests
        mock_crawler_instance = AsyncMock()
        mock_crawler_instance.__aenter__ = AsyncMock(return_value=mock_crawler_instance)
        mock_crawler_instance.__aexit__ = AsyncMock(return_value=None)

        # Mock successful crawl results
        from crawl4ai import CrawlResult
        mock_result = MagicMock(spec=CrawlResult)
        mock_result.success = True
        mock_result.extracted_content = '[{"main_characters": [], "secondary_characters": [], "minor_characters": []}]'
        mock_crawler_instance.arun = AsyncMock(return_value=[mock_result])
        mock_crawler_instance.arun_many = AsyncMock(return_value=[])

        with patch(
            "src.enrichment.crawlers.anime_planet_character_crawler.AsyncWebCrawler",
            return_value=mock_crawler_instance,
        ):
            # Call 1: with return_data=True, no output_path
            await fetch_animeplanet_characters("test-slug", return_data=True, output_path=None)

            # Call 2: with return_data=False, with output_path
            await fetch_animeplanet_characters("test-slug", return_data=False, output_path="/tmp/test.json")

            # Call 3: with different return_data value
            await fetch_animeplanet_characters("test-slug", return_data=True, output_path="/tmp/other.json")

    # Verify that all three calls used THE SAME cache key
    assert len(cache_keys_used) == 3, f"Expected 3 cache lookups, got {len(cache_keys_used)}"

    # All cache keys should be identical (only slug matters)
    assert cache_keys_used[0] == cache_keys_used[1], (
        f"Cache keys differ based on parameters:\n"
        f"  Call 1: {cache_keys_used[0]}\n"
        f"  Call 2: {cache_keys_used[1]}"
    )
    assert cache_keys_used[1] == cache_keys_used[2], (
        f"Cache keys differ based on parameters:\n"
        f"  Call 2: {cache_keys_used[1]}\n"
        f"  Call 3: {cache_keys_used[2]}"
    )
