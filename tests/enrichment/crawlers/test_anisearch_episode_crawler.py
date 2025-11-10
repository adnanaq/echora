"""
Tests for anisearch_episode_crawler.py main() function.
"""

from unittest.mock import patch

import pytest

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
