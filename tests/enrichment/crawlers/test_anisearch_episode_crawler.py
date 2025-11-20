"""
Tests for anisearch_episode_crawler.py main() function.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from crawl4ai import CrawlResult

from src.enrichment.crawlers.anisearch_episode_crawler import fetch_anisearch_episodes

# --- Tests for fetch_anisearch_episodes() function ---


@pytest.mark.asyncio
async def test_cache_key_only_depends_on_url(tmp_path: Path) -> None:
    """Test that cache key only depends on URL, not output_path or return_data."""
    from src.enrichment.crawlers.anisearch_episode_crawler import (
        fetch_anisearch_episodes,
    )

    output1 = tmp_path / "output1.json"
    output2 = tmp_path / "output2.json"

    # Expected cached data
    cached_data = [
        {"episodeNumber": 1, "runtime": "24 min", "releaseDate": "01.01.2024", "title": "Episode 1"},
        {"episodeNumber": 2, "runtime": "24 min", "releaseDate": "08.01.2024", "title": "Episode 2"},
    ]

    # Mock Redis client to track cache key generation
    mock_redis = AsyncMock()
    # First get() returns None (cache miss), second get() returns cached data (cache hit)
    mock_redis.get = AsyncMock(side_effect=[None, json.dumps(cached_data)])
    mock_redis.setex = AsyncMock()

    with patch(
        "src.cache_manager.result_cache.get_result_cache_redis_client",
        return_value=mock_redis
    ):
        with patch(
            "src.enrichment.crawlers.anisearch_episode_crawler.AsyncWebCrawler"
        ) as MockCrawler:
            mock_result = MagicMock(spec=CrawlResult)
            mock_result.success = True
            mock_result.url = "https://www.anisearch.com/anime/test/episodes"
            mock_result.extracted_content = json.dumps([
                {"episodeNumber": "01", "runtime": "24 min", "releaseDate": "01.01.2024", "title": "Episode 1"},
                {"episodeNumber": "02", "runtime": "24 min", "releaseDate": "08.01.2024", "title": "Episode 2"},
            ])

            mock_crawler = AsyncMock()
            mock_crawler.arun = AsyncMock(return_value=[mock_result])
            MockCrawler.return_value.__aenter__ = AsyncMock(return_value=mock_crawler)
            MockCrawler.return_value.__aexit__ = AsyncMock(return_value=None)

            # First call with output_file1, return_data=True
            result1 = await fetch_anisearch_episodes(
                "https://www.anisearch.com/anime/test/episodes",
                return_data=True,
                output_path=str(output1),
            )

            # Second call with output_file2, return_data=False
            result2 = await fetch_anisearch_episodes(
                "https://www.anisearch.com/anime/test/episodes",
                return_data=False,
                output_path=str(output2),
            )

    # Verify both files were written (even on cache hit, file writing happens in wrapper)
    assert output1.exists(), "First output file should be written"
    assert output2.exists(), "Second output file should be written on cache hit"

    # Verify return_data parameter works
    assert result1 is not None, "First call should return data"
    assert result2 is None, "Second call with return_data=False should return None"

    # Verify only ONE cache entry was created (same URL = same cache key)
    assert mock_redis.setex.call_count == 1, (
        "Should only create one cache entry for same URL, regardless of output_path/return_data"
    )

    # Verify cache key doesn't contain output_path or return_data
    cache_key = mock_redis.setex.call_args[0][0]
    assert "output_path" not in cache_key, "Cache key must not contain output_path"
    assert "return_data" not in cache_key, "Cache key must not contain return_data"


# --- Tests for main() function ---


@pytest.mark.asyncio
@patch("src.enrichment.crawlers.anisearch_episode_crawler.fetch_anisearch_episodes")
async def test_main_function_success(mock_fetch):
    """Test main() function handles successful execution."""
    from src.enrichment.crawlers.anisearch_episode_crawler import main

    mock_fetch.return_value = [
        {"episodeNumber": 1, "title": "Episode 1"},
        {"episodeNumber": 2, "title": "Episode 2"},
    ]

    with patch(
        "sys.argv",
        [
            "script.py",
            "https://www.anisearch.com/anime/18878/episodes",
            "--output",
            "/tmp/output.json",
        ],
    ):
        exit_code = await main()

    assert exit_code == 0
    # Verify the function was called (args vs kwargs may vary by implementation)
    mock_fetch.assert_awaited_once()
    call_args = mock_fetch.call_args
    # Check the URL was passed correctly (could be positional or keyword)
    if call_args[0]:
        assert call_args[0][0] == "https://www.anisearch.com/anime/18878/episodes"
    else:
        assert call_args[1]["url"] == "https://www.anisearch.com/anime/18878/episodes"


@pytest.mark.asyncio
@patch("src.enrichment.crawlers.anisearch_episode_crawler.fetch_anisearch_episodes")
async def test_main_function_with_default_output(mock_fetch):
    """Test main() function with default output path."""
    from src.enrichment.crawlers.anisearch_episode_crawler import main

    mock_fetch.return_value = []

    with patch(
        "sys.argv", ["script.py", "https://www.anisearch.com/anime/12345/episodes"]
    ):
        exit_code = await main()

    assert exit_code == 0
    # Verify default output path used
    call_args = mock_fetch.call_args
    assert call_args[1]["output_path"] == "anisearch_episodes.json"


@pytest.mark.asyncio
@patch("src.enrichment.crawlers.anisearch_episode_crawler.fetch_anisearch_episodes")
async def test_main_function_error_handling(mock_fetch):
    """Test main() function handles errors and returns non-zero exit code."""
    from src.enrichment.crawlers.anisearch_episode_crawler import main

    mock_fetch.side_effect = Exception("Crawler error")

    with patch(
        "sys.argv", ["script.py", "https://www.anisearch.com/anime/18878/episodes"]
    ):
        exit_code = await main()

    assert exit_code == 1


@pytest.mark.asyncio
@patch("src.enrichment.crawlers.anisearch_episode_crawler.fetch_anisearch_episodes")
async def test_main_function_handles_value_error(mock_fetch):
    """Test main() handles ValueError with specific error message."""
    from src.enrichment.crawlers.anisearch_episode_crawler import main

    mock_fetch.side_effect = ValueError("Invalid anime URL format")

    with patch("sys.argv", ["script.py", "invalid-url"]):
        exit_code = await main()

    assert exit_code == 1


@pytest.mark.asyncio
@patch("src.enrichment.crawlers.anisearch_episode_crawler.fetch_anisearch_episodes")
async def test_main_function_handles_os_error(mock_fetch):
    """Test main() handles OSError (file write failures)."""
    from src.enrichment.crawlers.anisearch_episode_crawler import main

    mock_fetch.side_effect = OSError("Permission denied")

    with patch(
        "sys.argv",
        [
            "script.py",
            "https://www.anisearch.com/anime/18878/episodes",
            "--output",
            "/invalid/path.json",
        ],
    ):
        exit_code = await main()

    assert exit_code == 1


@pytest.mark.asyncio
@patch("src.enrichment.crawlers.anisearch_episode_crawler.fetch_anisearch_episodes")
async def test_main_function_handles_unexpected_exception(mock_fetch):
    """Test main() handles unexpected exceptions and logs full traceback."""
    from src.enrichment.crawlers.anisearch_episode_crawler import main

    # Simulate truly unexpected exception
    mock_fetch.side_effect = RuntimeError("Unexpected internal error")

    with patch("sys.argv", ["script.py", "https://www.anisearch.com/anime/18878/episodes"]):
        exit_code = await main()

    assert exit_code == 1


@pytest.mark.asyncio
@patch("src.enrichment.crawlers.anisearch_episode_crawler.fetch_anisearch_episodes")
async def test_main_function_no_episodes_found(mock_fetch):
    """Test main() function when no episodes found."""
    from src.enrichment.crawlers.anisearch_episode_crawler import main

    mock_fetch.return_value = []

    with patch(
        "sys.argv", ["script.py", "https://www.anisearch.com/anime/99999/episodes"]
    ):
        exit_code = await main()

    # Should still return 0 even with empty list
    assert exit_code == 0



def test_normalize_episode_url_appends_suffix():
    from src.enrichment.crawlers.anisearch_episode_crawler import _normalize_episode_url

    url = "https://www.anisearch.com/anime/18878,dan-da-dan"
    result = _normalize_episode_url(url)
    assert result == "https://www.anisearch.com/anime/18878,dan-da-dan/episodes"


def test_normalize_episode_url_from_path():
    from src.enrichment.crawlers.anisearch_episode_crawler import _normalize_episode_url

    result = _normalize_episode_url("/18878,dan-da-dan")
    assert result == "https://www.anisearch.com/anime/18878,dan-da-dan/episodes"


def test_normalize_episode_url_from_id():
    from src.enrichment.crawlers.anisearch_episode_crawler import _normalize_episode_url

    result = _normalize_episode_url("18878,dan-da-dan")
    assert result == "https://www.anisearch.com/anime/18878,dan-da-dan/episodes"


def test_extract_anime_id_raises_on_invalid_url():
    from src.enrichment.crawlers.anisearch_episode_crawler import _extract_anime_id_from_episode_url

    with pytest.raises(ValueError, match="Invalid episode URL"):
        _extract_anime_id_from_episode_url("https://example.com/anime/123")


@pytest.mark.asyncio
async def test_fetch_episodes_returns_none_when_no_results():
    with patch("src.enrichment.crawlers.anisearch_episode_crawler.AsyncWebCrawler") as MockCrawler:
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.setex = AsyncMock()

        with patch("src.cache_manager.result_cache.get_result_cache_redis_client", return_value=mock_redis):
            mock_crawler = AsyncMock()
            mock_crawler.arun = AsyncMock(return_value=[])
            MockCrawler.return_value.__aenter__ = AsyncMock(return_value=mock_crawler)
            MockCrawler.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await fetch_anisearch_episodes("18878,dan-da-dan")
            assert result is None


@pytest.mark.asyncio
async def test_fetch_episodes_raises_on_wrong_result_type():
    with patch("src.enrichment.crawlers.anisearch_episode_crawler.AsyncWebCrawler") as MockCrawler:
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.setex = AsyncMock()

        with patch("src.cache_manager.result_cache.get_result_cache_redis_client", return_value=mock_redis):
            mock_crawler = AsyncMock()
            mock_crawler.arun = AsyncMock(return_value=["not a CrawlResult"])
            MockCrawler.return_value.__aenter__ = AsyncMock(return_value=mock_crawler)
            MockCrawler.return_value.__aexit__ = AsyncMock(return_value=None)

            with pytest.raises(TypeError, match="Unexpected result type"):
                await fetch_anisearch_episodes("18878,dan-da-dan")


@pytest.mark.asyncio
async def test_fetch_episodes_returns_none_on_extraction_failure():
    from crawl4ai import CrawlResult
    from unittest.mock import MagicMock

    with patch("src.enrichment.crawlers.anisearch_episode_crawler.AsyncWebCrawler") as MockCrawler:
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.setex = AsyncMock()

        with patch("src.cache_manager.result_cache.get_result_cache_redis_client", return_value=mock_redis):
            mock_result = MagicMock(spec=CrawlResult)
            mock_result.success = False
            mock_result.error_message = "Failed to extract"

            mock_crawler = AsyncMock()
            mock_crawler.arun = AsyncMock(return_value=[mock_result])
            MockCrawler.return_value.__aenter__ = AsyncMock(return_value=mock_crawler)
            MockCrawler.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await fetch_anisearch_episodes("18878,dan-da-dan")
            assert result is None


@pytest.mark.asyncio
async def test_fetch_episodes_wrapper_returns_none_when_data_is_none():
    with patch("src.enrichment.crawlers.anisearch_episode_crawler._fetch_anisearch_episodes_data") as mock_fetch:
        mock_fetch.return_value = None
        result = await fetch_anisearch_episodes("18878,dan-da-dan")
        assert result is None
