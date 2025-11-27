"""
Tests for anime_planet_character_crawler.py
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

import pytest
from crawl4ai import CrawlResult
from crawl4ai.models import CrawlResultContainer

from src.enrichment.crawlers.anime_planet_character_crawler import (
    _normalize_characters_url,
    _extract_slug_from_characters_url,
    _get_character_detail_schema,
    _process_character_list,
    _extract_voice_actors,
    _normalize_value,
    _process_character_details,
    fetch_animeplanet_characters,
    main,
)


# --- Tests for _normalize_characters_url ---


def test_normalize_characters_url_with_slug():
    """Test URL normalization with just a slug."""
    url = _normalize_characters_url("dandadan")
    assert url == "https://www.anime-planet.com/anime/dandadan/characters"


def test_normalize_characters_url_with_path():
    """Test URL normalization with anime path."""
    url = _normalize_characters_url("/anime/dandadan")
    assert url == "https://www.anime-planet.com/anime/dandadan/characters"


def test_normalize_characters_url_with_anime_prefix():
    """Test URL normalization with anime/ prefix."""
    url = _normalize_characters_url("anime/dandadan")
    assert url == "https://www.anime-planet.com/anime/dandadan/characters"


def test_normalize_characters_url_with_characters_suffix():
    """Test URL normalization removes and re-adds characters suffix."""
    url = _normalize_characters_url("/anime/dandadan/characters")
    assert url == "https://www.anime-planet.com/anime/dandadan/characters"


def test_normalize_characters_url_with_full_url():
    """Test URL normalization with full URL."""
    url = _normalize_characters_url("https://www.anime-planet.com/anime/dandadan")
    assert url == "https://www.anime-planet.com/anime/dandadan/characters"


def test_normalize_characters_url_with_full_url_with_characters():
    """Test URL normalization with full URL already having /characters."""
    url = _normalize_characters_url("https://www.anime-planet.com/anime/dandadan/characters")
    assert url == "https://www.anime-planet.com/anime/dandadan/characters"


def test_normalize_characters_url_invalid_url():
    """Test URL normalization raises ValueError for invalid URLs."""
    with pytest.raises(ValueError, match="Invalid URL: Must be an anime-planet.com"):
        _normalize_characters_url("https://example.com/anime/test")


# --- Tests for _extract_slug_from_characters_url ---


def test_extract_slug_from_characters_url_success():
    """Test slug extraction from valid URL."""
    slug = _extract_slug_from_characters_url(
        "https://www.anime-planet.com/anime/dandadan/characters"
    )
    assert slug == "dandadan"


def test_extract_slug_from_characters_url_invalid():
    """Test slug extraction raises ValueError for invalid URL."""
    with pytest.raises(ValueError, match="Could not extract slug from URL"):
        _extract_slug_from_characters_url("https://www.anime-planet.com/invalid")


# --- Tests for _get_character_detail_schema ---


def test_get_character_detail_schema():
    """Test character detail schema structure."""
    schema = _get_character_detail_schema()
    assert "baseSelector" in schema
    assert schema["baseSelector"] == "body"
    assert "fields" in schema
    assert len(schema["fields"]) > 0
    # Check for expected fields
    field_names = [f["name"] for f in schema["fields"]]
    assert "name_h1" in field_names
    assert "image_detail" in field_names
    assert "entry_bar_items" in field_names
    assert "metadata_items" in field_names


# --- Tests for _process_character_list ---


def test_process_character_list_with_main_characters():
    """Test processing character list with main characters."""
    list_data = {
        "main_characters": [
            {
                "name": "Test Character",
                "url": "/characters/test-character",
                "image_src": "https://example.com/image.jpg",
                "tags_raw": [{"tag": "protagonist"}, {"tag": "hero"}],
                "voice_actors_jp": [{"name": "JP Voice", "url": "/people/jp-voice"}],
            }
        ],
        "secondary_characters": [],
        "minor_characters": [],
    }

    characters = _process_character_list(list_data)

    assert len(characters) == 1
    assert characters[0]["name"] == "Test Character"
    assert characters[0]["role"] == "Main"
    assert characters[0]["image"] == "https://example.com/image.jpg"
    assert characters[0]["tags"] == ["protagonist", "hero"]
    assert "voice_actors" in characters[0]


def test_process_character_list_with_all_roles():
    """Test processing character list with all role types."""
    list_data = {
        "main_characters": [
            {"name": "Main Char", "url": "/characters/main"}
        ],
        "secondary_characters": [
            {"name": "Secondary Char", "url": "/characters/secondary"}
        ],
        "minor_characters": [
            {"name": "Minor Char", "url": "/characters/minor"}
        ],
    }

    characters = _process_character_list(list_data)

    assert len(characters) == 3
    assert characters[0]["role"] == "Main"
    assert characters[1]["role"] == "Secondary"
    assert characters[2]["role"] == "Minor"


def test_process_character_list_with_data_src_fallback():
    """Test image fallback to data-src."""
    list_data = {
        "main_characters": [
            {
                "name": "Test",
                "url": "/characters/test",
                "image_data_src": "https://example.com/lazy.jpg",
            }
        ],
        "secondary_characters": [],
        "minor_characters": [],
    }

    characters = _process_character_list(list_data)

    assert characters[0]["image"] == "https://example.com/lazy.jpg"


def test_process_character_list_skips_empty_characters():
    """Test that characters without name or URL are skipped."""
    list_data = {
        "main_characters": [
            {"name": "", "url": "/characters/test"},
            {"name": "Valid", "url": ""},
            {"name": "Valid Char", "url": "/characters/valid"},
        ],
        "secondary_characters": [],
        "minor_characters": [],
    }

    characters = _process_character_list(list_data)

    assert len(characters) == 1
    assert characters[0]["name"] == "Valid Char"


# --- Tests for _extract_voice_actors ---


def test_extract_voice_actors_all_languages():
    """Test extracting voice actors for all languages."""
    character = {
        "voice_actors_jp": [{"name": "JP Voice", "url": "/people/jp"}],
        "voice_actors_us": [{"name": "US Voice", "url": "/people/us"}],
        "voice_actors_es": [{"name": "ES Voice", "url": "/people/es"}],
        "voice_actors_fr": [{"name": "FR Voice", "url": "/people/fr"}],
    }

    voice_actors = _extract_voice_actors(character)

    assert len(voice_actors) == 4
    assert "jp" in voice_actors
    assert "us" in voice_actors
    assert "es" in voice_actors
    assert "fr" in voice_actors
    assert voice_actors["jp"][0]["name"] == "JP Voice"


def test_extract_voice_actors_empty():
    """Test extracting voice actors from empty data."""
    character = {}
    voice_actors = _extract_voice_actors(character)
    assert voice_actors == {}


def test_extract_voice_actors_skips_empty_names():
    """Test that voice actors without names are skipped."""
    character = {
        "voice_actors_jp": [
            {"name": "", "url": "/people/empty"},
            {"name": "Valid Voice", "url": "/people/valid"},
        ]
    }

    voice_actors = _extract_voice_actors(character)

    assert len(voice_actors["jp"]) == 1
    assert voice_actors["jp"][0]["name"] == "Valid Voice"


# --- Tests for _normalize_value ---


def test_normalize_value_with_question_mark():
    """Test that '?' is normalized to None."""
    assert _normalize_value("?") is None
    assert _normalize_value(" ? ") is None


def test_normalize_value_with_valid_string():
    """Test that valid strings are stripped and returned."""
    assert _normalize_value("  Valid Value  ") == "Valid Value"
    assert _normalize_value("Test") == "Test"


# --- Tests for _process_character_details ---


def test_process_character_details_gender_and_hair():
    """Test extracting gender and hair color from entry bar."""
    detail_data = {
        "entry_bar_items": [
            {"text": "Gender: Male"},
            {"text": "Hair Color: Black"},
        ]
    }

    enriched = _process_character_details(detail_data)

    assert enriched["gender"] == "Male"
    assert enriched["hair_color"] == "Black"


def test_process_character_details_normalizes_question_marks():
    """Test that '?' values are normalized to None."""
    detail_data = {
        "entry_bar_items": [
            {"text": "Gender: ?"},
            {"text": "Hair Color: ?"},
        ]
    }

    enriched = _process_character_details(detail_data)

    assert enriched["gender"] is None
    assert enriched["hair_color"] is None


def test_process_character_details_loved_and_hated_ranks():
    """Test extracting loved and hated ranks."""
    detail_data = {
        "loved_rank": "#1,234",
        "hated_rank": "#5,678",
    }

    enriched = _process_character_details(detail_data)

    assert enriched["loved_rank"] == 1234
    assert enriched["hated_rank"] == 5678


def test_process_character_details_invalid_ranks():
    """Test that invalid rank values are skipped."""
    detail_data = {
        "loved_rank": "invalid",
        "hated_rank": "not-a-number",
    }

    enriched = _process_character_details(detail_data)

    assert "loved_rank" not in enriched
    assert "hated_rank" not in enriched


def test_process_character_details_metadata():
    """Test extracting metadata fields."""
    detail_data = {
        "metadata_items": [
            {"title": "Eye Color", "value": "Blue"},
            {"title": "Age", "value": "16"},
            {"title": "Birthday", "value": "Jan 1"},
        ]
    }

    enriched = _process_character_details(detail_data)

    assert enriched["eye_color"] == "Blue"
    assert enriched["age"] == "16"
    assert enriched["birthday"] == "Jan 1"


def test_process_character_details_description():
    """Test extracting description from paragraphs."""
    detail_data = {
        "description_paragraphs": [
            {"text": "Short intro"},
            {"text": "This is a long enough description that should be extracted and used as the character description."},
            {"text": "Another paragraph"},
        ]
    }

    enriched = _process_character_details(detail_data)

    assert "description" in enriched
    assert "long enough description" in enriched["description"]


def test_process_character_details_skips_short_descriptions():
    """Test that short descriptions are skipped."""
    detail_data = {
        "description_paragraphs": [
            {"text": "Too short"},
            {"text": "Also short"},
        ]
    }

    enriched = _process_character_details(detail_data)

    assert "description" not in enriched


def test_process_character_details_alternative_names():
    """Test extracting alternative names."""
    detail_data = {
        "alt_names_raw": [
            {"name": "Alt Name 1"},
            {"name": "Alt Name 2"},
            {"name": ""},
        ]
    }

    enriched = _process_character_details(detail_data)

    assert "alternative_names" in enriched
    assert len(enriched["alternative_names"]) == 2
    assert "Alt Name 1" in enriched["alternative_names"]


def test_process_character_details_anime_roles():
    """Test extracting anime roles."""
    detail_data = {
        "anime_roles_raw": [
            {
                "anime_title": "Test Anime",
                "anime_url": "/anime/test",
                "role": "Main",
            },
            {
                "anime_title": "Another Anime",
                "anime_url": "/anime/another",
                "role": "",
            },
        ]
    }

    enriched = _process_character_details(detail_data)

    assert "anime_roles" in enriched
    assert len(enriched["anime_roles"]) == 2
    assert enriched["anime_roles"][0]["role"] == "Main"
    assert "role" not in enriched["anime_roles"][1]


def test_process_character_details_manga_roles():
    """Test extracting manga roles."""
    detail_data = {
        "manga_roles_raw": [
            {
                "manga_title": "Test Manga",
                "manga_url": "/manga/test",
                "role": "Secondary",
            }
        ]
    }

    enriched = _process_character_details(detail_data)

    assert "manga_roles" in enriched
    assert len(enriched["manga_roles"]) == 1
    assert enriched["manga_roles"][0]["manga_title"] == "Test Manga"


def test_process_character_details_empty_data():
    """Test processing empty detail data."""
    enriched = _process_character_details({})
    assert enriched == {}


# --- Tests for fetch_animeplanet_characters ---


@pytest.mark.asyncio
async def test_fetch_animeplanet_characters_success():
    """Test successful character fetch."""
    mock_data = {
        "characters": [{"name": "Test Character"}],
        "total_count": 1,
    }

    with patch(
        "src.enrichment.crawlers.anime_planet_character_crawler._fetch_animeplanet_characters_data",
        new_callable=AsyncMock,
        return_value=mock_data,
    ):
        result = await fetch_animeplanet_characters("test-slug")

        assert result is not None
        assert result["total_count"] == 1


@pytest.mark.asyncio
async def test_fetch_animeplanet_characters_with_output_path():
    """Test character fetch writes to output file."""
    mock_data = {"characters": [], "total_count": 0}

    with patch(
        "src.enrichment.crawlers.anime_planet_character_crawler._fetch_animeplanet_characters_data",
        new_callable=AsyncMock,
        return_value=mock_data,
    ), patch("builtins.open", mock_open()) as mock_file, patch(
        "src.enrichment.crawlers.anime_planet_character_crawler.sanitize_output_path",
        return_value="/safe/path.json"
    ):
        await fetch_animeplanet_characters("test", output_path="/tmp/test.json")

        mock_file.assert_called_once_with("/safe/path.json", "w", encoding="utf-8")


@pytest.mark.asyncio
async def test_fetch_animeplanet_characters_returns_none_on_failure():
    """Test character fetch returns None when data fetch fails."""
    with patch(
        "src.enrichment.crawlers.anime_planet_character_crawler._fetch_animeplanet_characters_data",
        new_callable=AsyncMock,
        return_value=None,
    ):
        result = await fetch_animeplanet_characters("test")

        assert result is None


# --- Tests for _fetch_animeplanet_characters_data ---


@pytest.mark.asyncio
async def test_fetch_animeplanet_characters_data_success():
    """Test successful character data fetch with enrichment."""
    # Create mock crawler
    mock_crawler = AsyncMock()

    # Mock character list response
    list_result = MagicMock(spec=CrawlResult)
    list_result.success = True
    list_result.extracted_content = json.dumps([{
        "main_characters": [
            {
                "name": "Test Character",
                "url": "/characters/test-character",
                "image_src": "image.jpg",
                "tags_raw": [],
            }
        ],
        "secondary_characters": [],
        "minor_characters": [],
    }])

    # Mock character detail response
    detail_result = MagicMock(spec=CrawlResult)
    detail_result.success = True
    detail_result.extracted_content = json.dumps([{
        "name_h1": "Test Character",
        "entry_bar_items": [{"text": "Gender: Male"}],
    }])

    # Wrap in CrawlResultContainer
    detail_container = MagicMock(spec=CrawlResultContainer)
    detail_container.__iter__ = lambda self: iter([detail_result])

    mock_crawler.arun = AsyncMock(return_value=[list_result])
    mock_crawler.arun_many = AsyncMock(return_value=[detail_container])
    mock_crawler.__aenter__ = AsyncMock(return_value=mock_crawler)
    mock_crawler.__aexit__ = AsyncMock(return_value=None)

    with patch(
        "src.enrichment.crawlers.anime_planet_character_crawler.AsyncWebCrawler",
        return_value=mock_crawler,
    ):
        from src.enrichment.crawlers.anime_planet_character_crawler import (
            _fetch_animeplanet_characters_data,
        )

        result = await _fetch_animeplanet_characters_data("test-slug")

        assert result is not None
        assert "characters" in result
        assert len(result["characters"]) == 1
        assert result["characters"][0]["gender"] == "Male"


@pytest.mark.asyncio
async def test_fetch_animeplanet_characters_data_character_name_mismatch():
    """Test when detail character name doesn't match list character name."""
    with patch("src.enrichment.crawlers.anime_planet_character_crawler.AsyncWebCrawler") as MockCrawler:
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.setex = AsyncMock()

        with patch("src.cache_manager.result_cache.get_result_cache_redis_client", return_value=mock_redis):
            mock_crawler = AsyncMock()

            # Character list response
            list_result = MagicMock(spec=CrawlResult)
            list_result.success = True
            list_result.extracted_content = json.dumps([{
                "main_characters": [
                    {
                        "name": "Test Character",
                        "url": "/characters/test-character",
                        "image_src": "image.jpg",
                        "tags_raw": [],
                    }
                ],
                "secondary_characters": [],
                "minor_characters": [],
            }])

            # Detail response with different name
            detail_result = MagicMock(spec=CrawlResult)
            detail_result.success = True
            detail_result.extracted_content = json.dumps([{
                "name_h1": "Different Character Name",
                "entry_bar_items": [{"text": "Gender: Male"}],
            }])

            # Wrap in CrawlResultContainer
            detail_container = MagicMock(spec=CrawlResultContainer)
            detail_container.__iter__ = lambda self: iter([detail_result])

            mock_crawler.arun = AsyncMock(return_value=[list_result])
            mock_crawler.arun_many = AsyncMock(return_value=[detail_container])
            mock_crawler.__aenter__ = AsyncMock(return_value=mock_crawler)
            mock_crawler.__aexit__ = AsyncMock(return_value=None)
            MockCrawler.return_value = mock_crawler

            from src.enrichment.crawlers.anime_planet_character_crawler import (
                _fetch_animeplanet_characters_data,
            )

            result = await _fetch_animeplanet_characters_data("test-slug")

            # Should still return data, but without enrichment for mismatched character
            assert result is not None
            assert "characters" in result
            assert len(result["characters"]) == 1


@pytest.mark.asyncio
async def test_fetch_animeplanet_characters_data_list_fetch_fails():
    """Test when character list fetch fails."""
    with patch("src.enrichment.crawlers.anime_planet_character_crawler.AsyncWebCrawler") as MockCrawler:
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.setex = AsyncMock()

        with patch("src.cache_manager.result_cache.get_result_cache_redis_client", return_value=mock_redis):
            mock_crawler = AsyncMock()
            mock_crawler.arun = AsyncMock(return_value=None)
            mock_crawler.__aenter__ = AsyncMock(return_value=mock_crawler)
            mock_crawler.__aexit__ = AsyncMock(return_value=None)
            MockCrawler.return_value = mock_crawler

            from src.enrichment.crawlers.anime_planet_character_crawler import (
                _fetch_animeplanet_characters_data,
            )

            result = await _fetch_animeplanet_characters_data("test-slug")

            assert result is None


@pytest.mark.asyncio
async def test_fetch_animeplanet_characters_data_invalid_result_type():
    """Test handling of unexpected result type."""
    with patch("src.enrichment.crawlers.anime_planet_character_crawler.AsyncWebCrawler") as MockCrawler:
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.setex = AsyncMock()

        with patch("src.cache_manager.result_cache.get_result_cache_redis_client", return_value=mock_redis):
            mock_crawler = AsyncMock()
            mock_crawler.arun = AsyncMock(return_value=["not a CrawlResult"])
            mock_crawler.__aenter__ = AsyncMock(return_value=mock_crawler)
            mock_crawler.__aexit__ = AsyncMock(return_value=None)
            MockCrawler.return_value = mock_crawler

            with pytest.raises(TypeError, match="Unexpected result type"):
                from src.enrichment.crawlers.anime_planet_character_crawler import (
                    _fetch_animeplanet_characters_data,
                )

                await _fetch_animeplanet_characters_data("test-slug")


@pytest.mark.asyncio
async def test_fetch_animeplanet_characters_data_empty_extraction():
    """Test when extraction returns empty data."""
    with patch("src.enrichment.crawlers.anime_planet_character_crawler.AsyncWebCrawler") as MockCrawler:
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.setex = AsyncMock()

        with patch("src.cache_manager.result_cache.get_result_cache_redis_client", return_value=mock_redis):
            mock_crawler = AsyncMock()

            list_result = MagicMock(spec=CrawlResult)
            list_result.success = True
            list_result.extracted_content = "[]"

            mock_crawler.arun = AsyncMock(return_value=[list_result])
            mock_crawler.__aenter__ = AsyncMock(return_value=mock_crawler)
            mock_crawler.__aexit__ = AsyncMock(return_value=None)
            MockCrawler.return_value = mock_crawler

            from src.enrichment.crawlers.anime_planet_character_crawler import (
                _fetch_animeplanet_characters_data,
            )

            result = await _fetch_animeplanet_characters_data("test-slug")

            assert result is None


@pytest.mark.asyncio
async def test_fetch_animeplanet_characters_data_no_characters_found():
    """Test when no characters are found after processing."""
    with patch("src.enrichment.crawlers.anime_planet_character_crawler.AsyncWebCrawler") as MockCrawler:
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.setex = AsyncMock()

        with patch("src.cache_manager.result_cache.get_result_cache_redis_client", return_value=mock_redis):
            mock_crawler = AsyncMock()

            list_result = MagicMock(spec=CrawlResult)
            list_result.success = True
            list_result.extracted_content = json.dumps([{
                "main_characters": [],
                "secondary_characters": [],
                "minor_characters": [],
            }])

            mock_crawler.arun = AsyncMock(return_value=[list_result])
            mock_crawler.__aenter__ = AsyncMock(return_value=mock_crawler)
            mock_crawler.__aexit__ = AsyncMock(return_value=None)
            MockCrawler.return_value = mock_crawler

            from src.enrichment.crawlers.anime_planet_character_crawler import (
                _fetch_animeplanet_characters_data,
            )

            result = await _fetch_animeplanet_characters_data("test-slug")

            assert result is None


@pytest.mark.asyncio
async def test_fetch_animeplanet_characters_data_no_valid_urls():
    """Test when no valid character URLs are found."""
    with patch("src.enrichment.crawlers.anime_planet_character_crawler.AsyncWebCrawler") as MockCrawler:
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.setex = AsyncMock()

        with patch("src.cache_manager.result_cache.get_result_cache_redis_client", return_value=mock_redis):
            mock_crawler = AsyncMock()

            list_result = MagicMock(spec=CrawlResult)
            list_result.success = True
            list_result.extracted_content = json.dumps([{
                "main_characters": [
                    {"name": "Test", "url": "/invalid/url"},
                ],
                "secondary_characters": [],
                "minor_characters": [],
            }])

            mock_crawler.arun = AsyncMock(return_value=[list_result])
            mock_crawler.__aenter__ = AsyncMock(return_value=mock_crawler)
            mock_crawler.__aexit__ = AsyncMock(return_value=None)
            MockCrawler.return_value = mock_crawler

            from src.enrichment.crawlers.anime_planet_character_crawler import (
                _fetch_animeplanet_characters_data,
            )

            result = await _fetch_animeplanet_characters_data("test-slug")

            assert result is None


@pytest.mark.asyncio
async def test_fetch_animeplanet_characters_data_batch_enrichment_none():
    """Test when batch enrichment returns None."""
    with patch("src.enrichment.crawlers.anime_planet_character_crawler.AsyncWebCrawler") as MockCrawler:
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.setex = AsyncMock()

        with patch("src.cache_manager.result_cache.get_result_cache_redis_client", return_value=mock_redis):
            mock_crawler = AsyncMock()

            list_result = MagicMock(spec=CrawlResult)
            list_result.success = True
            list_result.extracted_content = json.dumps([{
                "main_characters": [
                    {
                        "name": "Test Character",
                        "url": "/characters/test-character",
                    }
                ],
                "secondary_characters": [],
                "minor_characters": [],
            }])

            mock_crawler.arun = AsyncMock(return_value=[list_result])
            mock_crawler.arun_many = AsyncMock(return_value=None)
            mock_crawler.__aenter__ = AsyncMock(return_value=mock_crawler)
            mock_crawler.__aexit__ = AsyncMock(return_value=None)
            MockCrawler.return_value = mock_crawler

            from src.enrichment.crawlers.anime_planet_character_crawler import (
                _fetch_animeplanet_characters_data,
            )

            result = await _fetch_animeplanet_characters_data("test-slug")

            assert result is None


@pytest.mark.asyncio
async def test_fetch_animeplanet_characters_data_batch_enrichment_invalid_type():
    """Test when batch enrichment returns invalid type."""
    with patch("src.enrichment.crawlers.anime_planet_character_crawler.AsyncWebCrawler") as MockCrawler:
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.setex = AsyncMock()

        with patch("src.cache_manager.result_cache.get_result_cache_redis_client", return_value=mock_redis):
            mock_crawler = AsyncMock()

            list_result = MagicMock(spec=CrawlResult)
            list_result.success = True
            list_result.extracted_content = json.dumps([{
                "main_characters": [
                    {
                        "name": "Test Character",
                        "url": "/characters/test-character",
                    }
                ],
                "secondary_characters": [],
                "minor_characters": [],
            }])

            mock_crawler.arun = AsyncMock(return_value=[list_result])
            mock_crawler.arun_many = AsyncMock(return_value="not a list")
            mock_crawler.__aenter__ = AsyncMock(return_value=mock_crawler)
            mock_crawler.__aexit__ = AsyncMock(return_value=None)
            MockCrawler.return_value = mock_crawler

            from src.enrichment.crawlers.anime_planet_character_crawler import (
                _fetch_animeplanet_characters_data,
            )

            result = await _fetch_animeplanet_characters_data("test-slug")

            assert result is None


@pytest.mark.asyncio
async def test_fetch_animeplanet_characters_data_no_unwrapped_results():
    """Test when no valid CrawlResult objects are found after unwrapping."""
    with patch("src.enrichment.crawlers.anime_planet_character_crawler.AsyncWebCrawler") as MockCrawler:
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.setex = AsyncMock()

        with patch("src.cache_manager.result_cache.get_result_cache_redis_client", return_value=mock_redis):
            mock_crawler = AsyncMock()

            list_result = MagicMock(spec=CrawlResult)
            list_result.success = True
            list_result.extracted_content = json.dumps([{
                "main_characters": [
                    {
                        "name": "Test Character",
                        "url": "/characters/test-character",
                    }
                ],
                "secondary_characters": [],
                "minor_characters": [],
            }])

            # Create a container that yields no results
            detail_container = MagicMock(spec=CrawlResultContainer)
            detail_container.__iter__ = lambda self: iter([])

            mock_crawler.arun = AsyncMock(return_value=[list_result])
            mock_crawler.arun_many = AsyncMock(return_value=[detail_container])
            mock_crawler.__aenter__ = AsyncMock(return_value=mock_crawler)
            mock_crawler.__aexit__ = AsyncMock(return_value=None)
            MockCrawler.return_value = mock_crawler

            from src.enrichment.crawlers.anime_planet_character_crawler import (
                _fetch_animeplanet_characters_data,
            )

            result = await _fetch_animeplanet_characters_data("test-slug")

            assert result is None


@pytest.mark.asyncio
async def test_fetch_animeplanet_characters_data_enrichment_json_error():
    """Test handling of JSON decode errors during enrichment."""
    with patch("src.enrichment.crawlers.anime_planet_character_crawler.AsyncWebCrawler") as MockCrawler:
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.setex = AsyncMock()

        with patch("src.cache_manager.result_cache.get_result_cache_redis_client", return_value=mock_redis):
            mock_crawler = AsyncMock()

            list_result = MagicMock(spec=CrawlResult)
            list_result.success = True
            list_result.extracted_content = json.dumps([{
                "main_characters": [
                    {
                        "name": "Test Character",
                        "url": "/characters/test-character",
                    }
                ],
                "secondary_characters": [],
                "minor_characters": [],
            }])

            detail_result = MagicMock(spec=CrawlResult)
            detail_result.success = True
            detail_result.extracted_content = "invalid json"

            detail_container = MagicMock(spec=CrawlResultContainer)
            detail_container.__iter__ = lambda self: iter([detail_result])

            mock_crawler.arun = AsyncMock(return_value=[list_result])
            mock_crawler.arun_many = AsyncMock(return_value=[detail_container])
            mock_crawler.__aenter__ = AsyncMock(return_value=mock_crawler)
            mock_crawler.__aexit__ = AsyncMock(return_value=None)
            MockCrawler.return_value = mock_crawler

            from src.enrichment.crawlers.anime_planet_character_crawler import (
                _fetch_animeplanet_characters_data,
            )

            result = await _fetch_animeplanet_characters_data("test-slug")

            # Should still return data, just not enriched
            assert result is not None
            assert len(result["characters"]) == 1


@pytest.mark.asyncio
async def test_fetch_animeplanet_characters_data_enrichment_failed():
    """Test when detail page fetch fails."""
    with patch("src.enrichment.crawlers.anime_planet_character_crawler.AsyncWebCrawler") as MockCrawler:
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.setex = AsyncMock()

        with patch("src.cache_manager.result_cache.get_result_cache_redis_client", return_value=mock_redis):
            mock_crawler = AsyncMock()

            list_result = MagicMock(spec=CrawlResult)
            list_result.success = True
            list_result.extracted_content = json.dumps([{
                "main_characters": [
                    {
                        "name": "Test Character",
                        "url": "/characters/test-character",
                    }
                ],
                "secondary_characters": [],
                "minor_characters": [],
            }])

            detail_result = MagicMock(spec=CrawlResult)
            detail_result.success = False
            detail_result.error_message = "Failed to fetch"

            detail_container = MagicMock(spec=CrawlResultContainer)
            detail_container.__iter__ = lambda self: iter([detail_result])

            mock_crawler.arun = AsyncMock(return_value=[list_result])
            mock_crawler.arun_many = AsyncMock(return_value=[detail_container])
            mock_crawler.__aenter__ = AsyncMock(return_value=mock_crawler)
            mock_crawler.__aexit__ = AsyncMock(return_value=None)
            MockCrawler.return_value = mock_crawler

            from src.enrichment.crawlers.anime_planet_character_crawler import (
                _fetch_animeplanet_characters_data,
            )

            result = await _fetch_animeplanet_characters_data("test-slug")

            # Should still return data without enrichment
            assert result is not None


@pytest.mark.asyncio
async def test_fetch_animeplanet_characters_data_list_extraction_failed():
    """Test when list extraction fails."""
    with patch("src.enrichment.crawlers.anime_planet_character_crawler.AsyncWebCrawler") as MockCrawler:
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.setex = AsyncMock()

        with patch("src.cache_manager.result_cache.get_result_cache_redis_client", return_value=mock_redis):
            mock_crawler = AsyncMock()

            list_result = MagicMock(spec=CrawlResult)
            list_result.success = False
            list_result.error_message = "Extraction failed"

            mock_crawler.arun = AsyncMock(return_value=[list_result])
            mock_crawler.__aenter__ = AsyncMock(return_value=mock_crawler)
            mock_crawler.__aexit__ = AsyncMock(return_value=None)
            MockCrawler.return_value = mock_crawler

            from src.enrichment.crawlers.anime_planet_character_crawler import (
                _fetch_animeplanet_characters_data,
            )

            result = await _fetch_animeplanet_characters_data("test-slug")

            assert result is None


@pytest.mark.asyncio
async def test_fetch_animeplanet_characters_data_direct_crawl_result():
    """Test handling direct CrawlResult in batch results."""
    with patch("src.enrichment.crawlers.anime_planet_character_crawler.AsyncWebCrawler") as MockCrawler:
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.setex = AsyncMock()

        with patch("src.cache_manager.result_cache.get_result_cache_redis_client", return_value=mock_redis):
            mock_crawler = AsyncMock()

            list_result = MagicMock(spec=CrawlResult)
            list_result.success = True
            list_result.extracted_content = json.dumps([{
                "main_characters": [
                    {
                        "name": "Test Character",
                        "url": "/characters/test-character",
                    }
                ],
                "secondary_characters": [],
                "minor_characters": [],
            }])

            detail_result = MagicMock(spec=CrawlResult)
            detail_result.success = True
            detail_result.extracted_content = json.dumps([{
                "name_h1": "Test Character",
            }])

            # Return direct CrawlResult instead of container
            mock_crawler.arun = AsyncMock(return_value=[list_result])
            mock_crawler.arun_many = AsyncMock(return_value=[detail_result])
            mock_crawler.__aenter__ = AsyncMock(return_value=mock_crawler)
            mock_crawler.__aexit__ = AsyncMock(return_value=None)
            MockCrawler.return_value = mock_crawler

            from src.enrichment.crawlers.anime_planet_character_crawler import (
                _fetch_animeplanet_characters_data,
            )

            result = await _fetch_animeplanet_characters_data("test-slug")

            assert result is not None


# --- Tests for main() function ---


@pytest.mark.asyncio
@patch(
    "src.enrichment.crawlers.anime_planet_character_crawler.fetch_animeplanet_characters"
)
async def test_main_function_success(mock_fetch):
    """Test main() function handles successful execution."""
    mock_fetch.return_value = {"characters": [{"name": "Test Character"}], "total": 1}

    with patch("sys.argv", ["script.py", "dandadan", "--output", "/tmp/output.json"]):
        exit_code = await main()

    assert exit_code == 0
    mock_fetch.assert_awaited_once()


@pytest.mark.asyncio
@patch(
    "src.enrichment.crawlers.anime_planet_character_crawler.fetch_animeplanet_characters"
)
async def test_main_function_with_default_output(mock_fetch):
    """Test main() function with default output path."""
    mock_fetch.return_value = {"characters": [], "total": 0}

    with patch("sys.argv", ["script.py", "test-slug"]):
        exit_code = await main()

    assert exit_code == 0
    call_args = mock_fetch.call_args
    assert call_args[1]["output_path"] == "animeplanet_characters.json"


@pytest.mark.asyncio
@patch(
    "src.enrichment.crawlers.anime_planet_character_crawler.fetch_animeplanet_characters"
)
async def test_main_function_error_handling(mock_fetch):
    """Test main() function handles errors and returns non-zero exit code."""
    mock_fetch.side_effect = Exception("Crawler error")

    with patch("sys.argv", ["script.py", "test-slug"]):
        exit_code = await main()

    assert exit_code == 1


@pytest.mark.asyncio
@patch(
    "src.enrichment.crawlers.anime_planet_character_crawler.fetch_animeplanet_characters"
)
async def test_main_function_handles_value_error(mock_fetch):
    """Test main() handles ValueError with specific error message."""
    mock_fetch.side_effect = ValueError("Invalid character slug format")

    with patch("sys.argv", ["script.py", "invalid-slug"]):
        exit_code = await main()

    assert exit_code == 1


@pytest.mark.asyncio
@patch(
    "src.enrichment.crawlers.anime_planet_character_crawler.fetch_animeplanet_characters"
)
async def test_main_function_handles_os_error(mock_fetch):
    """Test main() handles OSError (file write failures)."""
    mock_fetch.side_effect = OSError("Permission denied")

    with patch(
        "sys.argv",
        ["script.py", "dandadan", "--output", "/invalid/path.json"],
    ):
        exit_code = await main()

    assert exit_code == 1


@pytest.mark.asyncio
@patch(
    "src.enrichment.crawlers.anime_planet_character_crawler.fetch_animeplanet_characters"
)
async def test_main_function_handles_unexpected_exception(mock_fetch):
    """Test main() handles unexpected exceptions and logs full traceback."""
    mock_fetch.side_effect = RuntimeError("Unexpected internal error")

    with patch("sys.argv", ["script.py", "dandadan"]):
        exit_code = await main()

    assert exit_code == 1


@pytest.mark.asyncio
@patch(
    "src.enrichment.crawlers.anime_planet_character_crawler.fetch_animeplanet_characters"
)
async def test_main_function_with_full_url(mock_fetch):
    """Test main() function with full URL as identifier."""
    mock_fetch.return_value = {"characters": [], "total": 0}

    with patch(
        "sys.argv",
        ["script.py", "https://www.anime-planet.com/anime/planet/characters"],
    ):
        exit_code = await main()

    assert exit_code == 0
    mock_fetch.assert_awaited_once()


# --- Tests for cache key generation ---


@pytest.mark.asyncio
async def test_cache_key_only_depends_on_slug():
    """
    Test that cache key only depends on slug parameter, not output_path.
    """
    cache_keys_used = []

    mock_cache_config = MagicMock()
    mock_cache_config.enabled = True
    mock_cache_config.storage_type = "redis"
    mock_cache_config.max_cache_key_length = 200

    with patch(
        "src.cache_manager.result_cache.get_cache_config",
        return_value=mock_cache_config,
    ), patch(
        "src.cache_manager.result_cache.get_result_cache_redis_client"
    ) as mock_redis, patch(
        "src.enrichment.crawlers.anime_planet_character_crawler.AsyncWebCrawler"
    ):
        redis_client = AsyncMock()

        async def track_get(key: str):
            cache_keys_used.append(key)
            return None

        redis_client.get = track_get
        redis_client.setex = AsyncMock()
        mock_redis.return_value = redis_client

        mock_crawler_instance = AsyncMock()
        mock_crawler_instance.__aenter__ = AsyncMock(return_value=mock_crawler_instance)
        mock_crawler_instance.__aexit__ = AsyncMock(return_value=None)

        mock_result = MagicMock(spec=CrawlResult)
        mock_result.success = True
        mock_result.extracted_content = '[{"main_characters": [], "secondary_characters": [], "minor_characters": []}]'
        mock_crawler_instance.arun = AsyncMock(return_value=[mock_result])
        mock_crawler_instance.arun_many = AsyncMock(return_value=[])

        with patch(
            "src.enrichment.crawlers.anime_planet_character_crawler.AsyncWebCrawler",
            return_value=mock_crawler_instance,
        ):
            await fetch_animeplanet_characters("test-slug", output_path=None)
            await fetch_animeplanet_characters("test-slug", output_path="/tmp/test.json")
            await fetch_animeplanet_characters("test-slug", output_path="/tmp/other.json")

    assert len(cache_keys_used) == 3
    assert cache_keys_used[0] == cache_keys_used[1]
    assert cache_keys_used[1] == cache_keys_used[2]
