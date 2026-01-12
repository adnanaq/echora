import copy
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from crawl4ai import CrawlResult
from enrichment.crawlers.anidb_character_crawler import (
    _flatten_character_data,
    _get_character_schema,
    fetch_anidb_character,
)


def test_get_character_schema():
    """Test that the schema is a valid dictionary."""
    schema = _get_character_schema()
    assert isinstance(schema, dict)
    assert "description" in schema
    assert "baseSelector" in schema
    assert "fields" in schema
    assert len(schema["fields"]) > 0


@pytest.mark.parametrize(
    "input_data, expected_data",
    [
        (
            {
                "nicknames": [{"text": "Soul King"}, {"text": "Humming Swordsman"}],
                "abilities": [{"text": "Swordsmanship"}],
                "looks": [],
                "other_field": "value",
            },
            {
                "nicknames": ["Soul King", "Humming Swordsman"],
                "abilities": ["Swordsmanship"],
                "looks": [],
                "other_field": "value",
            },
        ),
        (
            {
                "nicknames": [{"text": "A"}, {}],
                "official_names": [],
            },
            {
                "nicknames": ["A"],
                "official_names": [],
            },
        ),
        ({}, {}),
        (
            {"nicknames": "not a list"},
            {"nicknames": "not a list"},
        ),
    ],
)
def test_flatten_character_data(input_data, expected_data):
    """Test flattening of character data."""
    # Create a copy to verify no mutation after fix

    input_copy = copy.deepcopy(input_data)

    flattened = _flatten_character_data(input_data)
    assert flattened == expected_data

    # Verify input was not mutated (function should return new dict)
    assert input_data == input_copy, "Input should not be mutated"


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_redis_cache_miss")
@patch("enrichment.crawlers.anidb_character_crawler.AsyncWebCrawler")
async def test_fetch_anidb_character_success(MockAsyncWebCrawler):
    """Test successful character fetch."""
    mock_crawler_instance = MockAsyncWebCrawler.return_value.__aenter__.return_value
    mock_result = CrawlResult(
        url="http://example.com/491",
        success=True,
        extracted_content=json.dumps(
            [
                {
                    "name_kanji": "ブルック",
                    "gender": "Male",
                    "nicknames": [{"text": "Soul King"}],
                }
            ]
        ),
        html="<html></html>",
    )
    mock_crawler_instance.arun.return_value = [mock_result]

    data = await fetch_anidb_character(491)

    assert data is not None
    assert data["name_kanji"] == "ブルック"
    assert data["gender"] == "Male"
    assert data["nicknames"] == ["Soul King"]
    mock_crawler_instance.arun.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_redis_cache_miss")
@patch("enrichment.crawlers.anidb_character_crawler.AsyncWebCrawler")
async def test_fetch_anidb_character_save_to_file(MockAsyncWebCrawler, tmp_path):
    """Test successful fetch and save to file."""
    mock_crawler_instance = MockAsyncWebCrawler.return_value.__aenter__.return_value
    character_data = {
        "name_kanji": "ブルック",
        "gender": "Male",
        "nicknames": [{"text": "Soul King"}],
    }
    mock_result = CrawlResult(
        url="http://example.com/491",
        success=True,
        extracted_content=json.dumps([character_data]),
        html="<html></html>",
    )
    mock_crawler_instance.arun.return_value = [mock_result]

    output_file = tmp_path / "character.json"
    result_data = await fetch_anidb_character(491, output_path=str(output_file))

    # Function always returns data and writes to file if output_path provided
    assert output_file.exists()
    with open(output_file, encoding="utf-8") as f:
        saved_data = json.load(f)

    assert saved_data["name_kanji"] == "ブルック"
    assert result_data is not None
    assert result_data["name_kanji"] == "ブルック"


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_redis_cache_miss")
@patch("enrichment.crawlers.anidb_character_crawler.AsyncWebCrawler")
async def test_fetch_anidb_character_without_output_path(MockAsyncWebCrawler):
    """Test fetch without output_path - only returns data, no file written."""
    mock_crawler_instance = MockAsyncWebCrawler.return_value.__aenter__.return_value
    character_data = {"name_kanji": "ブルック"}
    mock_result = CrawlResult(
        url="http://example.com/491",
        success=True,
        extracted_content=json.dumps([character_data]),
        html="<html></html>",
    )
    mock_crawler_instance.arun.return_value = [mock_result]

    result_data = await fetch_anidb_character(491)

    # Function always returns data when successful
    assert result_data is not None
    assert result_data["name_kanji"] == "ブルック"


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_redis_cache_miss")
@patch("enrichment.crawlers.anidb_character_crawler.AsyncWebCrawler")
async def test_fetch_anidb_character_crawl_failure(MockAsyncWebCrawler):
    """Test when the crawl itself fails."""
    mock_crawler_instance = MockAsyncWebCrawler.return_value.__aenter__.return_value
    mock_result = CrawlResult(
        url="http://example.com/491",
        success=False,
        error_message="Crawl failed",
        html="",
    )
    mock_crawler_instance.arun.return_value = [mock_result]

    data = await fetch_anidb_character(491)
    assert data is None


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_redis_cache_miss")
@patch("enrichment.crawlers.anidb_character_crawler.AsyncWebCrawler")
async def test_fetch_anidb_character_anti_leech(MockAsyncWebCrawler):
    """Test detection of AntiLeech protection."""
    mock_crawler_instance = MockAsyncWebCrawler.return_value.__aenter__.return_value
    mock_result = CrawlResult(
        url="http://example.com/491",
        success=False,
        error_message="Blocked",
        html="<html><body>AntiLeech</body></html>",
    )
    mock_crawler_instance.arun.return_value = [mock_result]

    data = await fetch_anidb_character(491)
    assert data is None


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_redis_cache_miss")
@patch("enrichment.crawlers.anidb_character_crawler.AsyncWebCrawler")
async def test_fetch_anidb_character_no_extracted_content(MockAsyncWebCrawler):
    """Test when crawl succeeds but no content is extracted."""
    mock_crawler_instance = MockAsyncWebCrawler.return_value.__aenter__.return_value
    mock_result = CrawlResult(
        url="http://example.com/491",
        success=True,
        extracted_content=None,
        html="<html></html>",
    )
    mock_crawler_instance.arun.return_value = [mock_result]

    data = await fetch_anidb_character(491)
    assert data is None


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_redis_cache_miss")
@patch("enrichment.crawlers.anidb_character_crawler.AsyncWebCrawler")
async def test_fetch_anidb_character_empty_json_list(MockAsyncWebCrawler):
    """Test when extracted content is an empty JSON list."""
    mock_crawler_instance = MockAsyncWebCrawler.return_value.__aenter__.return_value
    mock_result = CrawlResult(
        url="http://example.com/491",
        success=True,
        extracted_content="[]",
        html="<html></html>",
    )
    mock_crawler_instance.arun.return_value = [mock_result]

    data = await fetch_anidb_character(491)
    assert data is None


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_redis_cache_miss")
@patch("enrichment.crawlers.anidb_character_crawler.AsyncWebCrawler")
async def test_fetch_anidb_character_unexpected_result_type(MockAsyncWebCrawler):
    """Test when crawler returns an unexpected result type."""
    mock_crawler_instance = MockAsyncWebCrawler.return_value.__aenter__.return_value
    mock_crawler_instance.arun.return_value = ["not a CrawlResult"]

    with pytest.raises(TypeError):
        await fetch_anidb_character(491)


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_redis_cache_miss")
@patch("enrichment.crawlers.anidb_character_crawler.AsyncWebCrawler")
async def test_fetch_anidb_character_no_results_returned(MockAsyncWebCrawler):
    """Test when the crawler returns an empty list of results."""
    mock_crawler_instance = MockAsyncWebCrawler.return_value.__aenter__.return_value
    mock_crawler_instance.arun.return_value = []  # Empty list

    data = await fetch_anidb_character(491)
    assert data is None


@pytest.mark.asyncio
@patch(
    "enrichment.crawlers.anidb_character_crawler.fetch_anidb_character",
    new_callable=AsyncMock,
)
@patch("argparse.ArgumentParser.parse_args")
async def test_main_success(mock_parse_args, mock_fetch):
    """Test the main function successfully fetching data."""
    # Simulate CLI arguments
    mock_parse_args.return_value = MagicMock(character_id=491, output="output.json")
    mock_fetch.return_value = {"name_kanji": "ブルック"}

    from enrichment.crawlers import anidb_character_crawler

    exit_code = await anidb_character_crawler.main()
    assert exit_code == 0
    mock_fetch.assert_called_once_with(491, output_path="output.json")


@pytest.mark.asyncio
@patch(
    "enrichment.crawlers.anidb_character_crawler.fetch_anidb_character",
    new_callable=AsyncMock,
)
@patch("argparse.ArgumentParser.parse_args")
async def test_main_failure(mock_parse_args, mock_fetch):
    """Test the main function when fetching data fails."""
    mock_parse_args.return_value = MagicMock(
        character_id=999, output="anidb_character.json"
    )
    mock_fetch.return_value = None  # Simulate failure

    from enrichment.crawlers import anidb_character_crawler

    exit_code = await anidb_character_crawler.main()
    assert exit_code == 1


@pytest.mark.asyncio
async def test_http_status_code_403_detection():
    """Test detection of HTTP 403 status code even with success=True."""
    with patch("enrichment.crawlers.anidb_character_crawler.AsyncWebCrawler"):
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)
        mock_redis.setex = AsyncMock()

        with patch(
            "http_cache.result_cache.get_result_cache_redis_client",
            return_value=mock_redis,
        ):
            with patch(
                "enrichment.crawlers.anidb_character_crawler.AsyncWebCrawler"
            ) as MockCrawler:
                mock_crawler = AsyncMock()

                # Create result with success=True but status_code=403
                mock_result = MagicMock(spec=CrawlResult)
                mock_result.success = True
                mock_result.status_code = 403
                mock_result.html = "<html>AntiLeech detected</html>"
                mock_result.extracted_content = None

                mock_crawler.arun = AsyncMock(return_value=[mock_result])
                mock_crawler.__aenter__ = AsyncMock(return_value=mock_crawler)
                mock_crawler.__aexit__ = AsyncMock(return_value=None)
                MockCrawler.return_value = mock_crawler

                data = await fetch_anidb_character(491)
                assert data is None


@pytest.mark.asyncio
@patch(
    "enrichment.crawlers.anidb_character_crawler.fetch_anidb_character",
    new_callable=AsyncMock,
)
@patch("argparse.ArgumentParser.parse_args")
async def test_main_exception_handling(mock_parse_args, mock_fetch):
    """Test the main function handles exceptions properly."""
    mock_parse_args.return_value = MagicMock(
        character_id=999, output="anidb_character.json"
    )
    mock_fetch.side_effect = ValueError("Invalid character ID")

    from enrichment.crawlers import anidb_character_crawler

    exit_code = await anidb_character_crawler.main()
    assert exit_code == 1
