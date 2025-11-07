"""
Tests for anime_planet_character_crawler.py main() function.
"""

import pytest
from unittest.mock import AsyncMock, patch


# --- Tests for main() function ---


@pytest.mark.asyncio
@patch("src.enrichment.crawlers.anime_planet_character_crawler.fetch_animeplanet_characters")
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
@patch("src.enrichment.crawlers.anime_planet_character_crawler.fetch_animeplanet_characters")
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
@patch("src.enrichment.crawlers.anime_planet_character_crawler.fetch_animeplanet_characters")
async def test_main_function_error_handling(mock_fetch):
    """Test main() function handles errors and returns non-zero exit code."""
    from src.enrichment.crawlers.anime_planet_character_crawler import main

    mock_fetch.side_effect = Exception("Crawler error")

    with patch("sys.argv", ["script.py", "test-slug"]):
        exit_code = await main()

    assert exit_code == 1


@pytest.mark.asyncio
@patch("src.enrichment.crawlers.anime_planet_character_crawler.fetch_animeplanet_characters")
async def test_main_function_with_full_url(mock_fetch):
    """Test main() function with full URL as identifier."""
    from src.enrichment.crawlers.anime_planet_character_crawler import main

    mock_fetch.return_value = {"characters": [], "total": 0}

    with patch("sys.argv", ["script.py", "https://www.anime-planet.com/anime/dandadan/characters"]):
        exit_code = await main()

    assert exit_code == 0
    mock_fetch.assert_awaited_once()
