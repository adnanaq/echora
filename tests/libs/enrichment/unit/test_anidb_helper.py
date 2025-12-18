"""
Tests for anidb_helper.py main() function.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# --- Tests for main() function ---


@pytest.mark.asyncio
@patch("enrichment.api_helpers.anidb_helper.AniDBEnrichmentHelper")
async def test_main_function_success_with_anidb_id(mock_helper_class):
    """Test main() function handles successful execution with AniDB ID."""
    from enrichment.api_helpers.anidb_helper import main

    mock_helper = AsyncMock()
    mock_helper.fetch_all_data = AsyncMock(
        return_value={"id": 123, "title": "Test Anime"}
    )
    mock_helper.close = AsyncMock()
    mock_helper_class.return_value = mock_helper

    with patch(
        "sys.argv", ["script.py", "--anidb-id", "123", "--output", "/tmp/output.json"]
    ):
        with patch("builtins.open", MagicMock()):
            exit_code = await main()

    assert exit_code == 0
    mock_helper.fetch_all_data.assert_awaited_once_with(123)


@pytest.mark.asyncio
@patch("enrichment.api_helpers.anidb_helper.AniDBEnrichmentHelper")
async def test_main_function_success_with_search_name(mock_helper_class):
    """Test main() function handles successful execution with search name."""
    import os
    import tempfile

    from enrichment.api_helpers.anidb_helper import main

    mock_helper = AsyncMock()
    # search_anime_by_name returns list, main() takes first result
    mock_helper.search_anime_by_name = AsyncMock(
        return_value=[{"id": 456, "title": "Test Anime"}]
    )
    mock_helper.close = AsyncMock()
    mock_helper_class.return_value = mock_helper

    # Use a real temp file to avoid JSON serialization issues
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        temp_path = f.name

    try:
        with patch(
            "sys.argv",
            ["script.py", "--search-name", "Test Anime", "--output", temp_path],
        ):
            exit_code = await main()

        assert exit_code == 0
        mock_helper.search_anime_by_name.assert_awaited_once_with("Test Anime")
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@pytest.mark.asyncio
@patch("enrichment.api_helpers.anidb_helper.AniDBEnrichmentHelper")
async def test_main_function_no_data_found(mock_helper_class):
    """Test main() function handles no data found."""
    from enrichment.api_helpers.anidb_helper import main

    mock_helper = AsyncMock()
    mock_helper.fetch_all_data = AsyncMock(return_value=None)
    mock_helper.close = AsyncMock()
    mock_helper_class.return_value = mock_helper

    with patch("sys.argv", ["script.py", "--anidb-id", "99999"]):
        exit_code = await main()

    assert exit_code == 1


@pytest.mark.asyncio
@patch("enrichment.api_helpers.anidb_helper.AniDBEnrichmentHelper")
async def test_main_function_error_handling(mock_helper_class):
    """Test main() function handles errors and returns non-zero exit code."""
    from enrichment.api_helpers.anidb_helper import main

    mock_helper = AsyncMock()
    mock_helper.fetch_all_data = AsyncMock(side_effect=Exception("API error"))
    mock_helper.close = AsyncMock()
    mock_helper_class.return_value = mock_helper

    with patch("sys.argv", ["script.py", "--anidb-id", "123"]):
        exit_code = await main()

    assert exit_code == 1


# --- Tests for context manager protocol ---


@pytest.mark.asyncio
async def test_context_manager_protocol():
    """Test AniDBEnrichmentHelper implements async context manager protocol."""
    from enrichment.api_helpers.anidb_helper import AniDBEnrichmentHelper

    async with AniDBEnrichmentHelper() as helper:
        assert helper is not None
        assert isinstance(helper, AniDBEnrichmentHelper)
        assert helper.session is None  # Lazy init - not created yet
    # Should exit cleanly, closing session if it was created


@pytest.mark.asyncio
async def test_context_manager_closes_session():
    """Test that context manager closes session on exit."""
    from enrichment.api_helpers.anidb_helper import AniDBEnrichmentHelper

    helper = AniDBEnrichmentHelper()

    # Create a mock session
    mock_session = AsyncMock()
    helper.session = mock_session

    async with helper:
        assert helper.session is mock_session

    # Session should be closed after context exit
    mock_session.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_context_manager_cleanup_on_exception():
    """Test that context manager cleans up even when exception occurs."""
    from enrichment.api_helpers.anidb_helper import AniDBEnrichmentHelper

    helper = AniDBEnrichmentHelper()
    mock_session = AsyncMock()
    helper.session = mock_session

    with pytest.raises(ValueError, match="Test error"):
        async with helper:
            raise ValueError("Test error")

    # Session should still be closed despite exception
    mock_session.close.assert_awaited_once()
