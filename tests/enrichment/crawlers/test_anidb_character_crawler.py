import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from crawl4ai import CrawlResult

from src.enrichment.crawlers.anidb_character_crawler import (
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
    flattened = _flatten_character_data(input_data)
    assert flattened == expected_data


@pytest.mark.asyncio
@patch("src.enrichment.crawlers.anidb_character_crawler.AsyncWebCrawler")
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
@patch("src.enrichment.crawlers.anidb_character_crawler.AsyncWebCrawler")
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
    result_data = await fetch_anidb_character(
        491, output_path=str(output_file), return_data=True
    )

    assert output_file.exists()
    with open(output_file, "r", encoding="utf-8") as f:
        saved_data = json.load(f)

    assert saved_data["name_kanji"] == "ブルック"
    assert result_data is not None
    assert result_data["name_kanji"] == "ブルック"


@pytest.mark.asyncio
@patch("src.enrichment.crawlers.anidb_character_crawler.AsyncWebCrawler")
async def test_fetch_anidb_character_no_return_data(MockAsyncWebCrawler, tmp_path):
    """Test fetch with return_data=False."""
    mock_crawler_instance = MockAsyncWebCrawler.return_value.__aenter__.return_value
    character_data = {"name_kanji": "ブルック"}
    mock_result = CrawlResult(
        url="http://example.com/491",
        success=True,
        extracted_content=json.dumps([character_data]),
        html="<html></html>",
    )
    mock_crawler_instance.arun.return_value = [mock_result]

    output_file = tmp_path / "character.json"
    result_data = await fetch_anidb_character(
        491, output_path=str(output_file), return_data=False
    )

    assert result_data is None
    assert output_file.exists()


@pytest.mark.asyncio
@patch("src.enrichment.crawlers.anidb_character_crawler.AsyncWebCrawler")
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
@patch("src.enrichment.crawlers.anidb_character_crawler.AsyncWebCrawler")
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
@patch("src.enrichment.crawlers.anidb_character_crawler.AsyncWebCrawler")
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
@patch("src.enrichment.crawlers.anidb_character_crawler.AsyncWebCrawler")
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
@patch("src.enrichment.crawlers.anidb_character_crawler.AsyncWebCrawler")
async def test_fetch_anidb_character_unexpected_result_type(MockAsyncWebCrawler):
    """Test when crawler returns an unexpected result type."""
    mock_crawler_instance = MockAsyncWebCrawler.return_value.__aenter__.return_value
    mock_crawler_instance.arun.return_value = ["not a CrawlResult"]

    with pytest.raises(TypeError):
        await fetch_anidb_character(491)


@pytest.mark.asyncio
@patch("src.enrichment.crawlers.anidb_character_crawler.AsyncWebCrawler")
async def test_fetch_anidb_character_no_results_returned(MockAsyncWebCrawler):
    """Test when the crawler returns an empty list of results."""
    mock_crawler_instance = MockAsyncWebCrawler.return_value.__aenter__.return_value
    mock_crawler_instance.arun.return_value = []  # Empty list

    data = await fetch_anidb_character(491)
    assert data is None


@pytest.mark.asyncio
@patch("src.enrichment.crawlers.anidb_character_crawler.fetch_anidb_character", new_callable=AsyncMock)
@patch("argparse.ArgumentParser.parse_args")
async def test_main_success(mock_parse_args, mock_fetch):
    """Test the main function successfully fetching data."""
    # Simulate CLI arguments
    mock_parse_args.return_value = MagicMock(character_id=491, output=None)
    mock_fetch.return_value = {"name_kanji": "ブルック"}

    from src.enrichment.crawlers import anidb_character_crawler

    with patch("builtins.print") as mock_print:
        await anidb_character_crawler.main()
        mock_fetch.assert_called_once_with(character_id=491, output_path=None)
        mock_print.assert_called()


@pytest.mark.asyncio
@patch("src.enrichment.crawlers.anidb_character_crawler.fetch_anidb_character", new_callable=AsyncMock)
@patch("argparse.ArgumentParser.parse_args")
async def test_main_failure(mock_parse_args, mock_fetch):
    """Test the main function when fetching data fails."""
    mock_parse_args.return_value = MagicMock(character_id=999, output=None)
    mock_fetch.return_value = None # Simulate failure

    from src.enrichment.crawlers import anidb_character_crawler

    with patch("builtins.print") as mock_print, pytest.raises(SystemExit) as e:
        await anidb_character_crawler.main()
        assert e.type == SystemExit
        assert e.value.code == 1


@patch("asyncio.run")
def test_dunder_main(mock_run):
    """Test the `if __name__ == '__main__'` block."""
    import runpy
    import asyncio
    runpy.run_path("src/enrichment/crawlers/anidb_character_crawler.py", run_name="__main__")
    mock_run.assert_called_once()
    call_arg = mock_run.call_args[0][0]
    assert asyncio.iscoroutine(call_arg)
    assert call_arg.__name__ == 'main'


@pytest.mark.asyncio
@patch("src.enrichment.crawlers.anidb_character_crawler.fetch_anidb_character", new_callable=AsyncMock)
@patch("argparse.ArgumentParser.parse_args")
async def test_main_failure_exit_code(mock_parse_args, mock_fetch, monkeypatch):
    """Test the main function's exit code on failure using monkeypatch."""
    mock_parse_args.return_value = MagicMock(character_id=999, output=None)
    mock_fetch.return_value = None  # Simulate failure

    exited = False
    exit_code = 0
    def mock_exit(code):
        nonlocal exited, exit_code
        exited = True
        exit_code = code

    monkeypatch.setattr("builtins.exit", mock_exit)

    from src.enrichment.crawlers import anidb_character_crawler
    await anidb_character_crawler.main()

    assert exited is True
    assert exit_code == 1
