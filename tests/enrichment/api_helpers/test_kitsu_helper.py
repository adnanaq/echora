"""
Tests for kitsu_helper.py main() function.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# --- Tests for main() function ---


@pytest.mark.asyncio
@patch("src.enrichment.api_helpers.kitsu_helper.KitsuEnrichmentHelper")
async def test_main_function_success(mock_helper_class):
    """Test main() function handles successful execution."""
    from src.enrichment.api_helpers.kitsu_helper import main

    mock_helper = AsyncMock()
    mock_helper.fetch_all_data = AsyncMock(return_value={"id": 1, "title": "Test"})
    mock_helper.close = AsyncMock()
    mock_helper_class.return_value = mock_helper

    with patch("sys.argv", ["script.py", "123", "/tmp/output.json"]):
        with patch("builtins.open", MagicMock()):
            exit_code = await main()

    assert exit_code == 0
    mock_helper_class.assert_called_once()
    mock_helper.fetch_all_data.assert_awaited_once_with(123)


@pytest.mark.asyncio
@patch("src.enrichment.api_helpers.kitsu_helper.KitsuEnrichmentHelper")
async def test_main_function_no_data_found(mock_helper_class):
    """Test main() function handles no data found."""
    from src.enrichment.api_helpers.kitsu_helper import main

    mock_helper = AsyncMock()
    mock_helper.fetch_all_data = AsyncMock(return_value=None)
    mock_helper.close = AsyncMock()
    mock_helper_class.return_value = mock_helper

    with patch("sys.argv", ["script.py", "99999", "/tmp/output.json"]):
        exit_code = await main()

    assert exit_code == 1


@pytest.mark.asyncio
@patch("src.enrichment.api_helpers.kitsu_helper.KitsuEnrichmentHelper")
async def test_main_function_error_handling(mock_helper_class):
    """Test main() function handles errors and returns non-zero exit code."""
    from src.enrichment.api_helpers.kitsu_helper import main

    mock_helper = AsyncMock()
    mock_helper.fetch_all_data = AsyncMock(side_effect=Exception("API error"))
    mock_helper.close = AsyncMock()
    mock_helper_class.return_value = mock_helper

    with patch("sys.argv", ["script.py", "123", "/tmp/output.json"]):
        exit_code = await main()

    assert exit_code == 1


@pytest.mark.asyncio
async def test_main_function_invalid_arguments():
    """Test main() function with invalid number of arguments."""
    from src.enrichment.api_helpers.kitsu_helper import main

    with patch("sys.argv", ["script.py"]):  # Missing arguments
        exit_code = await main()

    assert exit_code == 1


# --- Tests for context manager protocol ---


@pytest.mark.asyncio
async def test_context_manager_protocol():
    """Test KitsuEnrichmentHelper implements async context manager protocol."""
    from src.enrichment.api_helpers.kitsu_helper import KitsuEnrichmentHelper

    async with KitsuEnrichmentHelper() as helper:
        assert helper is not None
        assert isinstance(helper, KitsuEnrichmentHelper)
    # Should exit cleanly (close() is no-op for Kitsu)


@pytest.mark.asyncio
async def test_context_manager_close_method_exists():
    """Test that close() method exists even if it's a no-op."""
    from src.enrichment.api_helpers.kitsu_helper import KitsuEnrichmentHelper

    helper = KitsuEnrichmentHelper()
    # Should have close() method
    assert hasattr(helper, "close")
    assert callable(helper.close)
    # Should be safe to call
    await helper.close()


@pytest.mark.asyncio
async def test_context_manager_cleanup_on_exception():
    """Test that context manager cleans up even when exception occurs."""
    from src.enrichment.api_helpers.kitsu_helper import KitsuEnrichmentHelper

    with pytest.raises(ValueError, match="Test error"):
        async with KitsuEnrichmentHelper() as helper:
            # close() should be called even when exception raised
            raise ValueError("Test error")
    # If we get here, cleanup happened correctly
