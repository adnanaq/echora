import pytest
import json
import asyncio
import runpy
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import date, datetime

from src.enrichment.crawlers.anime_planet_anime_crawler import (
    _normalize_anime_url,
    _extract_slug_from_url,
    _extract_json_ld,
    _extract_rank,
    _extract_studios,
    _determine_season_from_date,
    _process_related_anime,
    fetch_animeplanet_anime,
    main as anime_planet_main,
)
from crawl4ai import CrawlResult

# region Helper Function Tests

@pytest.mark.parametrize("input_str, expected", [
    ("dandadan", "https://www.anime-planet.com/anime/dandadan"),
    ("/anime/dandadan", "https://www.anime-planet.com/anime/dandadan"),
    ("anime/dandadan", "https://www.anime-planet.com/anime/dandadan"),
    ("https://www.anime-planet.com/anime/dandadan", "https://www.anime-planet.com/anime/dandadan"),
])
def test_normalize_anime_url_success(input_str, expected):
    assert _normalize_anime_url(input_str) == expected

@pytest.mark.parametrize("invalid_url", [
    "https://www.google.com",
    "ftp://anime-planet.com/anime/dandadan",
])
def test_normalize_anime_url_failure(invalid_url):
    with pytest.raises(ValueError, match="Invalid URL"):
        _normalize_anime_url(invalid_url)

@pytest.mark.parametrize("url, expected_slug", [
    ("https://www.anime-planet.com/anime/dandadan", "dandadan"),
    ("https://www.anime-planet.com/anime/dandadan-2nd-season", "dandadan-2nd-season"),
    ("https://www.anime-planet.com/anime/dandadan?ref=123", "dandadan"),
    ("https://www.anime-planet.com/anime/dandadan#info", "dandadan"),
])
def test_extract_slug_from_url_success(url, expected_slug):
    assert _extract_slug_from_url(url) == expected_slug

@pytest.mark.parametrize("invalid_url", [
    "https://www.anime-planet.com/manga/dandadan",
    "https://www.anime-planet.com/anime/",
    "https://www.google.com",
])
def test_extract_slug_from_url_failure(invalid_url):
    with pytest.raises(ValueError, match="Could not extract slug"):
        _extract_slug_from_url(invalid_url)

def test_extract_json_ld():
    html = '''<script type="application/ld+json">
    {"name": "Dandadan", "description": "&quot;test&quot;", "image": "https://www.anime-planet.comhttps://a.com/img.jpg"}
    </script>'''
    data = _extract_json_ld(html)
    assert data["name"] == "Dandadan"
    assert data["description"] == '"test"'
    assert data["image"] == "https://a.com/img.jpg"
    assert _extract_json_ld("") is None
    assert _extract_json_ld("<script>{}</script>") is None
    assert _extract_json_ld("<script type='application/ld+json'>invalid json</script>") is None
    assert _extract_json_ld('<script type="application/ld+json">{"key": "val"}</script>') == {"key": "val"}

def test_extract_json_ld_decode_error():
    html = '<script type="application/ld+json">{"name": "Dandadan",</script>'
    assert _extract_json_ld(html) is None

@pytest.mark.parametrize("texts, expected", [
    ([{"text": "Rank #123"}], 123),
    ([{"text": "Overall Rank #456"}], 456),
    ([{"text": "Popularity #789"}], 789),
    ([{"text": "#invalid"}], None),
    ([{"text": "Rank #not-a-number"}], None),
    ([{"text": "No rank here"}], None),
    ([], None),
    ([{"foo": "bar"}], None),
])
def test_extract_rank(texts, expected):
    assert _extract_rank(texts) == expected

def test_extract_studios():
    assert _extract_studios([{"studio": "A"}, {"studio": "B"}, {"studio": "A"}]) == ["A", "B"]
    assert len(_extract_studios([{"studio": f"Studio {i}"} for i in range(10)])) == 5
    assert _extract_studios([]) == []
    assert _extract_studios([{"foo": "bar"}]) == []

@pytest.mark.parametrize("date_str, expected_season", [
    ("2024-01-15", "WINTER"), ("2024-02-20", "WINTER"), ("2024-12-25", "WINTER"),
    ("2024-03-01", "SPRING"), ("2024-04-05", "SPRING"), ("2024-05-31", "SPRING"),
    ("2024-06-10", "SUMMER"), ("2024-07-15", "SUMMER"), ("2024-08-20", "SUMMER"),
    ("2024-09-01", "FALL"), ("2024-10-15", "FALL"), ("2024-11-30", "FALL"),
    ("invalid-date", None), ("", None), ("2024--15", None),
])
def test_determine_season_from_date(date_str, expected_season):
    assert _determine_season_from_date(date_str) == expected_season

def test_process_related_anime():
    raw = [
        {"title": "Sequel", "url": "/anime/s", "start_date_attr": "2023-01-01", "metadata_text": "TV: 12 ep", "relation_subtype": "Sequel"},
        {"title": "Prequel", "url": "/anime/s2", "end_date_attr": "2022-01-01", "metadata_text": "Movie"},
        {"title": "Side Story", "url": "/anime/s3", "metadata_text": "Special: 1 ep"},
        {"title": "", "url": "/anime/s4"},
        {"title": "No URL", "url": ""},
        {"title": "Invalid URL", "url": "/manga/s5"},
        {"title": "No Start Date", "url": "/anime/s6", "end_date_attr": "2025-12-31"},
    ]
    processed = _process_related_anime(raw)
    assert len(processed) == 4
    assert processed[0]["slug"] == "s"
    assert processed[0]["relation_subtype"] == "Sequel"
    assert processed[0]["year"] == 2023
    assert processed[0]["episodes"] == 12
    assert processed[1]["slug"] == "s2"
    assert processed[1]["year"] == 2022
    assert "episodes" not in processed[1]
    assert processed[2]["type"] == "Special"
    assert processed[3]["slug"] == "s6"
    assert processed[3]["year"] == 2025

# endregion

# region Main Fetch Function Tests

@pytest.mark.asyncio
@patch("src.enrichment.crawlers.anime_planet_anime_crawler.AsyncWebCrawler")
async def test_fetch_animeplanet_anime_full_success(MockAsyncWebCrawler, tmp_path):
    mock_crawler_instance = MockAsyncWebCrawler.return_value.__aenter__.return_value
    output_path = tmp_path / "output.json"
    html_content = '''
    <script type="application/ld+json">
    {"@type":"TVSeries", "name":"D", "startDate":"2024-10-04", "endDate":"2025-03-28", "aggregateRating":{}, "genre":[], "url":"http://a.com/url", "image":"http://a.com/image.jpg"}
    </script>
    <meta property="og:image" content="http://a.com/poster.jpg">
    '''
    mock_result = CrawlResult(
        url="http://a.com", success=True, html=html_content,
        extracted_content=json.dumps([{
            "rank_text": [{"text": "#123"}],
            "studios_raw": [{"studio": "S1"}],
            "title_japanese": "Alt title: J",
            "related_anime_raw": []
        }])
    )
    mock_crawler_instance.arun.return_value = [mock_result]
    data = await fetch_animeplanet_anime("d", return_data=True, output_path=str(output_path))

    assert data["slug"] == "d"
    assert data["rank"] == 123
    assert data["status"] == "COMPLETED"
    assert data["studios"] == ["S1"]
    assert data["title_japanese"] == "J"
    assert data["type"] == "TVSeries"
    assert "rank_text" not in data
    assert "studios_raw" not in data
    assert output_path.exists()
    with open(output_path) as f:
        assert json.load(f)["slug"] == "d"

@pytest.mark.asyncio
@patch("src.enrichment.crawlers.anime_planet_anime_crawler.AsyncWebCrawler")
async def test_fetch_anime_status_logic(MockAsyncWebCrawler):
    mock_crawler_instance = MockAsyncWebCrawler.return_value.__aenter__.return_value
    base_result = {"url": "http://a.com", "success": True, "extracted_content": json.dumps([{}])}

    # UPCOMING (future date)
    future_date = f"{date.today().year + 1}-01-01"
    html_upcoming = f'<script type="application/ld+json">{{"startDate":"{future_date}"}}</script>'
    mock_crawler_instance.arun.return_value = [CrawlResult(**base_result, html=html_upcoming)]
    data = await fetch_animeplanet_anime("d")
    assert data["status"] == "UPCOMING"

    # AIRING (past start, no end)
    past_date = f"{date.today().year - 1}-01-01"
    html_airing = f'<script type="application/ld+json">{{"startDate":"{past_date}", "endDate":null}}</script>'
    mock_crawler_instance.arun.return_value = [CrawlResult(**base_result, html=html_airing)]
    data = await fetch_animeplanet_anime("d")
    assert data["status"] == "AIRING"

    # AIRING (with full datetime)
    past_datetime = datetime(date.today().year - 1, 1, 1).isoformat() + "Z"
    html_airing_dt = f'<script type="application/ld+json">{{"startDate":"{past_datetime}", "endDate":null}}</script>'
    mock_crawler_instance.arun.return_value = [CrawlResult(**base_result, html=html_airing_dt)]
    data = await fetch_animeplanet_anime("d")
    assert data["status"] == "AIRING"

    # UNKNOWN (no dates)
    html_unknown = '<script type="application/ld+json">{"startDate":null, "endDate":null}</script>'
    mock_crawler_instance.arun.return_value = [CrawlResult(**base_result, html=html_unknown)]
    data = await fetch_animeplanet_anime("d")
    assert data["status"] == "UNKNOWN"

    # UNKNOWN (malformed date)
    html_malformed = '<script type="application/ld+json">{{"startDate":"not-a-date"}}</script>'
    mock_crawler_instance.arun.return_value = [CrawlResult(**base_result, html=html_malformed)]
    data = await fetch_animeplanet_anime("d")
    assert data["status"] == "UNKNOWN"

@pytest.mark.asyncio
@patch("src.enrichment.crawlers.anime_planet_anime_crawler.AsyncWebCrawler")
async def test_fetch_failures_and_empty_cases(MockAsyncWebCrawler):
    mock_crawler_instance = MockAsyncWebCrawler.return_value.__aenter__.return_value
    
    # Case: Crawler returns None
    mock_crawler_instance.arun.return_value = None
    assert await fetch_animeplanet_anime("d") is None

    # Case: Crawler returns empty list
    mock_crawler_instance.arun.return_value = []
    assert await fetch_animeplanet_anime("d") is None

    # Case: Crawl failed
    mock_crawler_instance.arun.return_value = [CrawlResult(url="", success=False, error_message="e", html="")]
    assert await fetch_animeplanet_anime("d") is None

    # Case: Success but empty extracted content
    mock_crawler_instance.arun.return_value = [CrawlResult(url="", success=True, extracted_content="[]", html="")]
    assert await fetch_animeplanet_anime("d") is None

    # Case: Success but invalid JSON in extracted content
    mock_crawler_instance.arun.return_value = [CrawlResult(url="", success=True, extracted_content="not-json", html="")]
    with pytest.raises(json.JSONDecodeError):
        await fetch_animeplanet_anime("d")

    # Case: Unexpected result type from crawler
    mock_crawler_instance.arun.return_value = ["not a crawl result"]
    with pytest.raises(TypeError):
        await fetch_animeplanet_anime("d")

@pytest.mark.asyncio
@patch("src.enrichment.crawlers.anime_planet_anime_crawler.AsyncWebCrawler")
async def test_fetch_animeplanet_anime_edge_cases(MockAsyncWebCrawler):
    mock_crawler_instance = MockAsyncWebCrawler.return_value.__aenter__.return_value

    # Case: No HTML in result
    mock_result_no_html = CrawlResult(url="http://a.com", success=True, html="", extracted_content=json.dumps([{}]))
    mock_crawler_instance.arun.return_value = [mock_result_no_html]
    data = await fetch_animeplanet_anime("d")
    assert "type" not in data

    # Case: Empty JSON-LD
    html_empty_json = '<script type="application/ld+json">{}</script>'
    mock_result_empty_json = CrawlResult(url="http://a.com", success=True, html=html_empty_json, extracted_content=json.dumps([{}]))
    mock_crawler_instance.arun.return_value = [mock_result_empty_json]
    data = await fetch_animeplanet_anime("d")
    assert "title" not in data

    # Case: Fallback poster extraction
    html_og_poster = '<meta property="og:image" content="http://a.com/fallback.jpg">'.encode('utf-8')
    mock_result_og_poster = CrawlResult(url="http://a.com", success=True, html=html_og_poster, extracted_content=json.dumps([{"poster": None}]))
    mock_crawler_instance.arun.return_value = [mock_result_og_poster]
    data = await fetch_animeplanet_anime("d")
    assert data["poster"] == "http://a.com/fallback.jpg"

    # Case: json.loads returns empty list
    mock_result_empty_data = CrawlResult(url="http://a.com", success=True, html="", extracted_content="[]")
    mock_crawler_instance.arun.return_value = [mock_result_empty_data]
    assert await fetch_animeplanet_anime("d") is None

@pytest.mark.asyncio
@patch("src.enrichment.crawlers.anime_planet_anime_crawler.AsyncWebCrawler")
async def test_fetch_no_return_data(MockAsyncWebCrawler, tmp_path):
    mock_crawler_instance = MockAsyncWebCrawler.return_value.__aenter__.return_value
    output_path = tmp_path / "output.json"
    html_content = '<script type="application/ld+json">{"name":"D"}</script>'
    mock_result = CrawlResult(url="http://a.com", success=True, html=html_content, extracted_content=json.dumps([{}]))
    mock_crawler_instance.arun.return_value = [mock_result]
    
    # Should write to file but return None
    result = await fetch_animeplanet_anime("d", return_data=False, output_path=str(output_path))
    assert result is None
    assert output_path.exists()

# endregion

# region CLI and Main Tests

@patch("src.enrichment.crawlers.anime_planet_anime_crawler.fetch_animeplanet_anime")
def test_main(mock_fetch, capsys):
    with patch("sys.argv", ["script.py", "dandadan", "--output", "test.json"]):
        anime_planet_main()
    
    # Check that our main async function was called with correct args
    mock_fetch.assert_called_once()
    call_args = mock_fetch.call_args[0]
    call_kwargs = mock_fetch.call_args[1]
    
    assert call_args[0] == "dandadan"
    assert call_kwargs["return_data"] is False
    assert call_kwargs["output_path"] == "test.json"

@patch("src.enrichment.crawlers.anime_planet_anime_crawler.main")
def test_main_entrypoint(mock_main):
    runpy.run_path("src/enrichment/crawlers/anime_planet_anime_crawler.py", run_name="__main__")
    mock_main.assert_called_once()

# endregion