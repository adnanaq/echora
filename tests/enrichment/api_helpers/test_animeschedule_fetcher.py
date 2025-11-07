"""
Tests for animeschedule_fetcher.py main() function.
"""

import pytest
from unittest.mock import AsyncMock, patch


# --- Tests for main() function ---


@pytest.mark.asyncio
@patch("src.enrichment.api_helpers.animeschedule_fetcher.fetch_animeschedule_data")
async def test_main_function_success(mock_fetch):
    """Test main() function handles successful execution."""
    from src.enrichment.api_helpers.animeschedule_fetcher import main

    mock_fetch.return_value = {"title": "Test Anime", "data": []}

    with patch("sys.argv", ["script.py", "test-anime"]):
        exit_code = await main()

    assert exit_code == 0
    mock_fetch.assert_awaited_once_with("test-anime", save_file=True)


@pytest.mark.asyncio
@patch("src.enrichment.api_helpers.animeschedule_fetcher.fetch_animeschedule_data")
async def test_main_function_no_data_found(mock_fetch):
    """Test main() function handles no data found."""
    from src.enrichment.api_helpers.animeschedule_fetcher import main

    mock_fetch.return_value = None

    with patch("sys.argv", ["script.py", "nonexistent"]):
        exit_code = await main()

    assert exit_code == 1


@pytest.mark.asyncio
@patch("src.enrichment.api_helpers.animeschedule_fetcher.fetch_animeschedule_data")
async def test_main_function_error_handling(mock_fetch):
    """Test main() function handles errors and returns non-zero exit code."""
    from src.enrichment.api_helpers.animeschedule_fetcher import main

    mock_fetch.side_effect = Exception("Fetch error")

    with patch("sys.argv", ["script.py", "test-anime"]):
        exit_code = await main()

    assert exit_code == 1


@pytest.mark.asyncio
async def test_main_function_invalid_arguments():
    """Test main() function with invalid number of arguments."""
    from src.enrichment.api_helpers.animeschedule_fetcher import main

    with patch("sys.argv", ["script.py"]):  # Missing search term
        exit_code = await main()

    assert exit_code == 1
