"""
Tests for animeplanet_helper.py.
"""

import os
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

# --- Tests for main() function ---


@pytest.mark.asyncio
@patch("enrichment.api_helpers.animeplanet_helper.AnimePlanetEnrichmentHelper")
async def test_main_function_success(mock_helper_class):
    """Test main() function handles successful execution."""
    from enrichment.api_helpers.animeplanet_helper import main

    mock_helper = AsyncMock()
    mock_helper.fetch_anime_data = AsyncMock(
        return_value={"title": "Test", "slug": "test"}
    )
    mock_helper.close = AsyncMock()
    mock_helper_class.return_value = mock_helper
    mock_helper.__aenter__.return_value = mock_helper

    with patch("sys.argv", ["script.py", "test-anime", "output.json"]):
        with patch("builtins.open", MagicMock()):
            exit_code = await main()

    assert exit_code == 0
    mock_helper_class.assert_called_once()
    mock_helper.fetch_anime_data.assert_awaited_once_with("test-anime")


@pytest.mark.asyncio
@patch("enrichment.api_helpers.animeplanet_helper.AnimePlanetEnrichmentHelper")
async def test_main_function_no_data_found(mock_helper_class):
    """Test main() function handles no data found."""
    from enrichment.api_helpers.animeplanet_helper import main

    mock_helper = AsyncMock()
    mock_helper.fetch_anime_data = AsyncMock(return_value=None)
    mock_helper.close = AsyncMock()
    mock_helper_class.return_value = mock_helper
    mock_helper.__aenter__.return_value = mock_helper

    with patch("sys.argv", ["script.py", "nonexistent", "output.json"]):
        exit_code = await main()

    assert exit_code == 1


@pytest.mark.asyncio
@patch("enrichment.api_helpers.animeplanet_helper.AnimePlanetEnrichmentHelper")
async def test_main_function_error_handling(mock_helper_class):
    """Test main() function handles errors and returns non-zero exit code."""
    from enrichment.api_helpers.animeplanet_helper import main

    mock_helper = AsyncMock()
    mock_helper.fetch_anime_data = AsyncMock(side_effect=OSError("Fetch error"))
    mock_helper.close = AsyncMock()
    mock_helper_class.return_value = mock_helper
    mock_helper.__aenter__.return_value = mock_helper

    with patch("sys.argv", ["script.py", "test-anime", "output.json"]):
        exit_code = await main()

    assert exit_code == 1


@pytest.mark.asyncio
async def test_main_function_invalid_arguments():
    """Test main() function with invalid number of arguments."""
    from enrichment.api_helpers.animeplanet_helper import main

    with patch("sys.argv", ["script.py"]):  # Missing arguments
        exit_code = await main()

    assert exit_code == 1


# --- Tests for fetch_all ---


@pytest.mark.asyncio
async def test_fetch_all_returns_split_anime_and_characters():
    """fetch_all returns {"anime": ..., "characters": [...]} — not a merged flat dict."""
    from enrichment.api_helpers.animeplanet_helper import AnimePlanetEnrichmentHelper
    from enrichment.crawlers.anime_planet.anime_planet_models import AnimePlanetAnime

    anime_obj = AnimePlanetAnime(name="Dandadan", slug="dandadan")
    anime_canonical = {"title": "Dandadan", "slug": "dandadan"}
    characters = [{"name": "Okarun"}]

    helper = AnimePlanetEnrichmentHelper()

    with patch(
        "enrichment.api_helpers.animeplanet_helper.fetch_animeplanet_anime",
        new=AsyncMock(return_value=anime_obj),
    ):
        with patch(
            "enrichment.api_helpers.animeplanet_helper.anime_from_animeplanet",
            return_value=anime_canonical,
        ):
            with patch.object(
                helper,
                "fetch_character_data",
                new=AsyncMock(return_value={"characters": characters, "total_count": 1}),
            ):
                result = await helper.fetch_all("https://www.anime-planet.com/anime/dandadan")

    assert result == {"anime": anime_canonical, "characters": characters}


@pytest.mark.asyncio
async def test_fetch_all_writes_anime_before_characters(tmp_path):
    """Anime output file is written before character fetch begins."""
    from enrichment.api_helpers.animeplanet_helper import AnimePlanetEnrichmentHelper
    from enrichment.crawlers.anime_planet.anime_planet_models import AnimePlanetAnime

    anime_obj = AnimePlanetAnime(name="Dandadan", slug="dandadan")
    anime_canonical = {"title": "Dandadan", "slug": "dandadan"}
    anime_path = str(tmp_path / "ap_anime.jsonl")
    chars_path = str(tmp_path / "ap_chars.jsonl")

    write_calls: list[str] = []

    async def slow_char_fetch(slug: str):  # type: ignore[override]
        write_calls.append("char_fetch_started")
        return {"characters": [], "total_count": 0}

    helper = AnimePlanetEnrichmentHelper()

    original_append = None

    def tracking_append(path: str, data: object) -> None:
        write_calls.append(f"write:{os.path.basename(path)}")
        if original_append:
            original_append(path, data)

    with patch(
        "enrichment.api_helpers.animeplanet_helper.fetch_animeplanet_anime",
        new=AsyncMock(return_value=anime_obj),
    ):
        with patch(
            "enrichment.api_helpers.animeplanet_helper.anime_from_animeplanet",
            return_value=anime_canonical,
        ):
            with patch(
                "enrichment.api_helpers.animeplanet_helper.append_jsonl",
                side_effect=tracking_append,
            ) as mock_append:
                with patch.object(helper, "fetch_character_data", side_effect=slow_char_fetch):
                    await helper.fetch_all(
                        "https://www.anime-planet.com/anime/dandadan",
                        anime_output_path=anime_path,
                        characters_output_path=chars_path,
                    )

    # Anime write must happen before character fetch starts
    assert write_calls.index(f"write:{os.path.basename(anime_path)}") < write_calls.index(
        "char_fetch_started"
    )


@pytest.mark.asyncio
async def test_fetch_all_returns_none_when_anime_missing():
    """fetch_all returns None when the crawler finds no anime."""
    from enrichment.api_helpers.animeplanet_helper import AnimePlanetEnrichmentHelper

    helper = AnimePlanetEnrichmentHelper()
    with patch(
        "enrichment.api_helpers.animeplanet_helper.fetch_animeplanet_anime",
        new=AsyncMock(return_value=None),
    ):
        result = await helper.fetch_all("https://www.anime-planet.com/anime/unknown")

    assert result is None


@pytest.mark.asyncio
async def test_fetch_all_survives_character_fetch_failure():
    """fetch_all returns anime data even when character fetch raises."""
    from enrichment.api_helpers.animeplanet_helper import AnimePlanetEnrichmentHelper
    from enrichment.crawlers.anime_planet.anime_planet_models import AnimePlanetAnime

    anime_obj = AnimePlanetAnime(name="Dandadan", slug="dandadan")
    anime_canonical = {"title": "Dandadan"}

    helper = AnimePlanetEnrichmentHelper()
    with patch(
        "enrichment.api_helpers.animeplanet_helper.fetch_animeplanet_anime",
        new=AsyncMock(return_value=anime_obj),
    ):
        with patch(
            "enrichment.api_helpers.animeplanet_helper.anime_from_animeplanet",
            return_value=anime_canonical,
        ):
            with patch.object(
                helper,
                "fetch_character_data",
                new=AsyncMock(side_effect=RuntimeError("network error")),
            ):
                result = await helper.fetch_all("https://www.anime-planet.com/anime/dandadan")

    assert result is not None
    assert result["anime"] == anime_canonical
    assert result["characters"] == []


# --- Tests for context manager protocol ---


@pytest.mark.asyncio
async def test_context_manager_protocol():
    """Test AnimePlanetEnrichmentHelper implements async context manager protocol."""
    from enrichment.api_helpers.animeplanet_helper import (
        AnimePlanetEnrichmentHelper,
    )

    async with AnimePlanetEnrichmentHelper() as helper:
        assert helper is not None
        assert isinstance(helper, AnimePlanetEnrichmentHelper)
    # Should exit cleanly (close() is no-op for AnimePlanet)


@pytest.mark.asyncio
async def test_context_manager_close_method_exists():
    """Test that close() method exists even if it's a no-op."""
    from enrichment.api_helpers.animeplanet_helper import (
        AnimePlanetEnrichmentHelper,
    )

    helper = AnimePlanetEnrichmentHelper()
    # Should have close() method
    assert hasattr(helper, "close")
    assert callable(helper.close)
    # Should be safe to call
    await helper.close()


@pytest.mark.asyncio
async def test_context_manager_cleanup_on_exception():
    """Test that context manager cleans up even when exception occurs."""
    from enrichment.api_helpers.animeplanet_helper import (
        AnimePlanetEnrichmentHelper,
    )

    with pytest.raises(ValueError, match="Test error"):
        async with AnimePlanetEnrichmentHelper():
            # close() should be called even when exception raised
            raise ValueError("Test error")
    # If we get here, cleanup happened correctly
