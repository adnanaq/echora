"""
Tests for kitsu_helper.py main() function.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# --- Tests for main() function ---


def _cm(response):
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=response)
    cm.__aexit__ = AsyncMock(return_value=None)
    return cm


@pytest.mark.asyncio
@patch("enrichment.api_helpers.kitsu_helper.KitsuEnrichmentHelper")
async def test_main_function_success(mock_helper_class):
    """Test main() function handles successful execution."""
    from enrichment.api_helpers.kitsu_helper import main

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
@patch("enrichment.api_helpers.kitsu_helper.KitsuEnrichmentHelper")
async def test_main_function_no_data_found(mock_helper_class):
    """Test main() function handles no data found."""
    from enrichment.api_helpers.kitsu_helper import main

    mock_helper = AsyncMock()
    mock_helper.fetch_all_data = AsyncMock(return_value=None)
    mock_helper.close = AsyncMock()
    mock_helper_class.return_value = mock_helper

    with patch("sys.argv", ["script.py", "99999", "/tmp/output.json"]):
        exit_code = await main()

    assert exit_code == 1


@pytest.mark.asyncio
@patch("enrichment.api_helpers.kitsu_helper.KitsuEnrichmentHelper")
async def test_main_function_error_handling(mock_helper_class):
    """Test main() function handles errors and returns non-zero exit code."""
    from enrichment.api_helpers.kitsu_helper import main

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
    from enrichment.api_helpers.kitsu_helper import main

    with patch("sys.argv", ["script.py"]):  # Missing arguments
        exit_code = await main()

    assert exit_code == 1


# --- Tests for context manager protocol ---


@pytest.mark.asyncio
async def test_context_manager_protocol():
    """Test KitsuEnrichmentHelper implements async context manager protocol."""
    from enrichment.api_helpers.kitsu_helper import KitsuEnrichmentHelper

    async with KitsuEnrichmentHelper() as helper:
        assert helper is not None
        assert isinstance(helper, KitsuEnrichmentHelper)
    # Should exit cleanly (close() is no-op for Kitsu)


@pytest.mark.asyncio
async def test_context_manager_close_method_exists():
    """Test that close() method exists even if it's a no-op."""
    from enrichment.api_helpers.kitsu_helper import KitsuEnrichmentHelper

    helper = KitsuEnrichmentHelper()
    # Should have close() method
    assert hasattr(helper, "close")
    assert callable(helper.close)
    # Should be safe to call
    await helper.close()


@pytest.mark.asyncio
async def test_context_manager_cleanup_on_exception():
    """Test that context manager cleans up even when exception occurs."""
    from enrichment.api_helpers.kitsu_helper import KitsuEnrichmentHelper

    with pytest.raises(ValueError, match="Test error"):
        async with KitsuEnrichmentHelper() as helper:
            # close() should be called even when exception raised
            raise ValueError("Test error")
    # If we get here, cleanup happened correctly


# --- Tests for cache-aware pagination throttling ---


@pytest.mark.asyncio
async def test_get_anime_episodes_skips_sleep_on_cache_hit():
    """Cached pages should not trigger artificial throttle sleep."""
    from enrichment.api_helpers.kitsu_helper import KitsuEnrichmentHelper

    helper = KitsuEnrichmentHelper()

    page1 = [{"id": str(i)} for i in range(1, 21)]
    page2 = [{"id": str(i)} for i in range(21, 41)]
    helper._make_request = AsyncMock(
        side_effect=[
            {"data": page1, "meta": {"count": 40}, "_from_cache": True},
            {"data": page2, "meta": {"count": 40}, "_from_cache": True},
        ]
    )

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        episodes = await helper.get_anime_episodes(12)

    assert len(episodes) == 40
    mock_sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_anime_categories_skips_sleep_on_cache_hit():
    """Cached category pages should not trigger artificial throttle sleep."""
    from enrichment.api_helpers.kitsu_helper import KitsuEnrichmentHelper

    helper = KitsuEnrichmentHelper()

    page1 = [
        {"attributes": {"title": f"Theme {i}", "description": f"Desc {i}"}}
        for i in range(1, 21)
    ]
    page2 = [{"attributes": {"title": "Theme 21", "description": "Desc 21"}}]
    helper._make_request = AsyncMock(
        side_effect=[
            {"data": page1, "meta": {"count": 21}, "_from_cache": True},
            {"data": page2, "meta": {"count": 21}, "_from_cache": True},
        ]
    )

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        categories = await helper.get_anime_categories(12)

    assert len(categories) == 21
    mock_sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_fetch_all_data_reuses_one_session_per_anime():
    """fetch_all_data should open a single cached session for the entire anime fetch."""
    from enrichment.api_helpers.kitsu_helper import KitsuEnrichmentHelper

    helper = KitsuEnrichmentHelper()

    def get_side_effect(url, **kwargs):  # type: ignore[no-untyped-def]
        resp = AsyncMock()
        resp.status = 200
        if url.endswith("/anime/12"):
            payload = {"data": {"id": "12"}}
        elif "/episodes" in url:
            payload = {"data": [{"id": "e1"}], "meta": {"count": 1}}
        elif "/categories" in url:
            payload = {
                "data": [
                    {"attributes": {"title": "Action", "description": "Desc"}}
                ],
                "meta": {"count": 1},
            }
        else:
            payload = {}
        resp.json = AsyncMock(return_value=payload)
        return _cm(resp)

    mock_session = MagicMock()
    mock_session.get = MagicMock(side_effect=get_side_effect)

    def make_session_cm(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        return _cm(mock_session)

    with patch(
        "enrichment.api_helpers.kitsu_helper._cache_manager.get_aiohttp_session",
        side_effect=make_session_cm,
    ) as mock_get_session:
        result = await helper.fetch_all_data(12)

    assert result["anime"] is not None
    assert len(result["episodes"]) == 1
    assert len(result["categories"]) == 1
    assert mock_get_session.call_count == 1
