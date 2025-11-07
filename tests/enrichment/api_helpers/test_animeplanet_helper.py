"""
Tests for animeplanet_helper.py main() function.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# --- Tests for main() function ---


@pytest.mark.asyncio
@patch("src.enrichment.api_helpers.animeplanet_helper.AnimePlanetEnrichmentHelper")
async def test_main_function_success(mock_helper_class):
    """Test main() function handles successful execution."""
    from src.enrichment.api_helpers.animeplanet_helper import main

    mock_helper = AsyncMock()
    mock_helper.fetch_anime_data = AsyncMock(return_value={"title": "Test", "slug": "test"})
    mock_helper.close = AsyncMock()
    mock_helper_class.return_value = mock_helper

    with patch("sys.argv", ["script.py", "test-anime", "/tmp/output.json"]):
        with patch("builtins.open", MagicMock()):
            exit_code = await main()

    assert exit_code == 0
    mock_helper_class.assert_called_once()
    mock_helper.fetch_anime_data.assert_awaited_once_with("test-anime")


@pytest.mark.asyncio
@patch("src.enrichment.api_helpers.animeplanet_helper.AnimePlanetEnrichmentHelper")
async def test_main_function_no_data_found(mock_helper_class):
    """Test main() function handles no data found."""
    from src.enrichment.api_helpers.animeplanet_helper import main

    mock_helper = AsyncMock()
    mock_helper.fetch_anime_data = AsyncMock(return_value=None)
    mock_helper.close = AsyncMock()
    mock_helper_class.return_value = mock_helper

    with patch("sys.argv", ["script.py", "nonexistent", "/tmp/output.json"]):
        exit_code = await main()

    assert exit_code == 1


@pytest.mark.asyncio
@patch("src.enrichment.api_helpers.animeplanet_helper.AnimePlanetEnrichmentHelper")
async def test_main_function_error_handling(mock_helper_class):
    """Test main() function handles errors and returns non-zero exit code."""
    from src.enrichment.api_helpers.animeplanet_helper import main

    mock_helper = AsyncMock()
    mock_helper.fetch_anime_data = AsyncMock(side_effect=Exception("Fetch error"))
    mock_helper.close = AsyncMock()
    mock_helper_class.return_value = mock_helper

    with patch("sys.argv", ["script.py", "test-anime", "/tmp/output.json"]):
        exit_code = await main()

    assert exit_code == 1


@pytest.mark.asyncio
async def test_main_function_invalid_arguments():
    """Test main() function with invalid number of arguments."""
    from src.enrichment.api_helpers.animeplanet_helper import main

    with patch("sys.argv", ["script.py"]):  # Missing arguments
        exit_code = await main()

    assert exit_code == 1
