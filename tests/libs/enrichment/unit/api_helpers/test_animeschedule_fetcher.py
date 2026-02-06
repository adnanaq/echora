"""
Tests for animeschedule_fetcher.py - comprehensive coverage including edge cases.
"""

import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# --- Helper function for mocking ---


def create_mock_session_with_response(mock_response):
    """
    Create an aiohttp-style async session mock whose `.get()` returns an async context manager that yields the provided response.

    Parameters:
        mock_response: The value to be returned from the async context manager's `__aenter__` (typically a mocked response object).

    Returns:
        AsyncMock: A mock session whose `get()` method returns an async context manager that yields `mock_response`.
    """
    # Mock async context manager for session.get()
    mock_cm = MagicMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
    mock_cm.__aexit__ = AsyncMock(return_value=None)

    # Mock session
    mock_session = AsyncMock()
    mock_session.get = MagicMock(return_value=mock_cm)
    return mock_session


# --- Tests for fetch_animeschedule_data() function ---


@pytest.mark.asyncio
@patch("enrichment.api_helpers.animeschedule_fetcher._cache_manager")
async def test_fetch_animeschedule_data_success_no_output(mock_cache_manager):
    """Test fetch_animeschedule_data returns data without writing file."""
    from enrichment.api_helpers.animeschedule_fetcher import (
        fetch_animeschedule_data,
    )

    # Mock response
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json = AsyncMock(
        return_value={"anime": [{"id": 1, "title": "Test Anime"}]}
    )

    # Create mocked session
    mock_session = create_mock_session_with_response(mock_response)
    mock_cache_manager.get_aiohttp_session.return_value = mock_session

    # Call without output_path
    result = await fetch_animeschedule_data("Test Anime")

    assert result == {"id": 1, "title": "Test Anime"}
    mock_session.close.assert_awaited_once()


@pytest.mark.asyncio
@patch("enrichment.api_helpers.animeschedule_fetcher._cache_manager")
async def test_fetch_animeschedule_data_with_output_path(mock_cache_manager):
    """Test fetch_animeschedule_data writes file when output_path provided."""
    from enrichment.api_helpers.animeschedule_fetcher import (
        fetch_animeschedule_data,
    )

    # Mock response
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json = AsyncMock(
        return_value={"anime": [{"id": 1, "title": "Test Anime"}]}
    )

    # Create mocked session
    mock_session = create_mock_session_with_response(mock_response)
    mock_cache_manager.get_aiohttp_session.return_value = mock_session

    # Use temp file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
        tmp_path = tmp.name

    try:
        result = await fetch_animeschedule_data("Test Anime", output_path=tmp_path)

        # Verify data returned
        assert result == {"id": 1, "title": "Test Anime"}

        # Verify file written
        assert os.path.exists(tmp_path)
        with open(tmp_path, encoding="utf-8") as f:
            file_data = json.load(f)
        assert file_data == {"id": 1, "title": "Test Anime"}

        mock_session.close.assert_awaited_once()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@pytest.mark.asyncio
@patch("enrichment.api_helpers.animeschedule_fetcher._cache_manager")
async def test_fetch_animeschedule_data_no_results(mock_cache_manager):
    """Test fetch_animeschedule_data returns None when no anime found."""
    from enrichment.api_helpers.animeschedule_fetcher import (
        fetch_animeschedule_data,
    )

    # Mock response with empty results
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json = AsyncMock(return_value={"anime": []})

    # Create mocked session
    mock_session = create_mock_session_with_response(mock_response)
    mock_cache_manager.get_aiohttp_session.return_value = mock_session

    result = await fetch_animeschedule_data("Nonexistent Anime")

    assert result is None
    mock_session.close.assert_awaited_once()


@pytest.mark.asyncio
@patch("enrichment.api_helpers.animeschedule_fetcher._cache_manager")
async def test_fetch_animeschedule_data_client_error(mock_cache_manager):
    """Test fetch_animeschedule_data handles aiohttp.ClientError."""
    import aiohttp
    from enrichment.api_helpers.animeschedule_fetcher import (
        fetch_animeschedule_data,
    )

    # Mock async context manager that raises on __aenter__
    mock_cm = MagicMock()
    mock_cm.__aenter__ = AsyncMock(side_effect=aiohttp.ClientError("Connection error"))
    mock_cm.__aexit__ = AsyncMock(return_value=None)

    # Mock session
    mock_session = AsyncMock()
    mock_session.get = MagicMock(return_value=mock_cm)
    mock_cache_manager.get_aiohttp_session.return_value = mock_session

    result = await fetch_animeschedule_data("Test Anime")

    assert result is None
    mock_session.close.assert_awaited_once()


@pytest.mark.asyncio
@patch("enrichment.api_helpers.animeschedule_fetcher._cache_manager")
async def test_fetch_animeschedule_data_json_decode_error(mock_cache_manager):
    """Test fetch_animeschedule_data handles JSONDecodeError."""
    from enrichment.api_helpers.animeschedule_fetcher import (
        fetch_animeschedule_data,
    )

    # Mock response with invalid JSON
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json = AsyncMock(side_effect=json.JSONDecodeError("Invalid", "", 0))

    # Create mocked session
    mock_session = create_mock_session_with_response(mock_response)
    mock_cache_manager.get_aiohttp_session.return_value = mock_session

    result = await fetch_animeschedule_data("Test Anime")

    assert result is None
    mock_session.close.assert_awaited_once()


@pytest.mark.asyncio
@patch("enrichment.api_helpers.animeschedule_fetcher._cache_manager")
async def test_fetch_animeschedule_data_raise_for_status(mock_cache_manager):
    """Test fetch_animeschedule_data handles HTTP errors via raise_for_status."""
    import aiohttp
    from enrichment.api_helpers.animeschedule_fetcher import (
        fetch_animeschedule_data,
    )

    # Mock response with HTTP error - raise_for_status is sync method
    mock_response = AsyncMock()

    # Make raise_for_status a regular Mock that raises (not async)
    mock_response.raise_for_status = MagicMock(
        side_effect=aiohttp.ClientResponseError(
            request_info=MagicMock(), history=(), status=404, message="Not Found"
        )
    )

    # Create mocked session
    mock_session = create_mock_session_with_response(mock_response)
    mock_cache_manager.get_aiohttp_session.return_value = mock_session

    result = await fetch_animeschedule_data("Test Anime")

    assert result is None
    mock_session.close.assert_awaited_once()


# --- Tests for main() function ---


@pytest.mark.asyncio
@patch("enrichment.api_helpers.animeschedule_fetcher.fetch_animeschedule_data")
async def test_main_function_success_default_output(mock_fetch):
    """Test main() function handles successful execution with default output."""
    from enrichment.api_helpers.animeschedule_fetcher import main

    mock_fetch.return_value = {"title": "Test Anime", "data": []}

    with patch("sys.argv", ["script.py", "test-anime"]):
        exit_code = await main()

    assert exit_code == 0
    mock_fetch.assert_awaited_once_with("test-anime", output_path="animeschedule.json")


@pytest.mark.asyncio
@patch("enrichment.api_helpers.animeschedule_fetcher.fetch_animeschedule_data")
async def test_main_function_success_custom_output(mock_fetch):
    """Test main() function handles custom output path."""
    from enrichment.api_helpers.animeschedule_fetcher import main

    mock_fetch.return_value = {"title": "Test Anime", "data": []}

    with patch("sys.argv", ["script.py", "test-anime", "--output", "custom/path.json"]):
        exit_code = await main()

    assert exit_code == 0
    mock_fetch.assert_awaited_once_with("test-anime", output_path="custom/path.json")


@pytest.mark.asyncio
@patch("enrichment.api_helpers.animeschedule_fetcher.fetch_animeschedule_data")
async def test_main_function_no_data_found(mock_fetch):
    """Test main() function handles no data found."""
    from enrichment.api_helpers.animeschedule_fetcher import main

    mock_fetch.return_value = None

    with patch("sys.argv", ["script.py", "nonexistent"]):
        exit_code = await main()

    assert exit_code == 1


@pytest.mark.asyncio
@patch("enrichment.api_helpers.animeschedule_fetcher.fetch_animeschedule_data")
async def test_main_function_error_handling(mock_fetch):
    """Test main() function handles errors and returns non-zero exit code."""
    from enrichment.api_helpers.animeschedule_fetcher import main

    mock_fetch.side_effect = Exception("Fetch error")

    with patch("sys.argv", ["script.py", "test-anime"]):
        exit_code = await main()

    assert exit_code == 1


@pytest.mark.asyncio
async def test_main_function_missing_required_argument():
    """Test main() function with missing required search_term argument."""
    from enrichment.api_helpers.animeschedule_fetcher import main

    with patch("sys.argv", ["script.py"]):  # Missing search term
        with pytest.raises(SystemExit) as exc_info:
            await main()
        assert exc_info.value.code == 2  # argparse exits with 2 for invalid args
