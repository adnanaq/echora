import asyncio
import gzip
import time
import xml.etree.ElementTree as ET
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import ClientError

from src.enrichment.api_helpers.anidb_helper import (
    AniDBEnrichmentHelper,
    AniDBRequestMetrics,
    CircuitBreakerState,
)


@pytest.fixture
def helper():
    """Fixture for AniDBEnrichmentHelper."""
    with patch("src.enrichment.api_helpers.anidb_helper.load_dotenv"):
        helper = AniDBEnrichmentHelper()
        # Prevent actual sleeping
        helper._adaptive_rate_limit = AsyncMock()
        yield helper


@pytest.fixture
def mock_session():
    """Fixture for mocking aiohttp.ClientSession."""
    session = MagicMock()
    response = session.get.return_value.__aenter__.return_value
    response.status = 200
    response.read = AsyncMock(return_value=b"<anime id='1'></anime>")
    response.text = AsyncMock(return_value="<anime id='1'></anime>")
    return session


@pytest.mark.asyncio
@patch("src.enrichment.api_helpers.anidb_helper.os.getenv")
def test_helper_initialization(mock_getenv):
    """Test that the helper initializes correctly."""
    # Mock os.getenv to return the default value provided in the call
    def getenv_side_effect(key, default=None):
        return default
    mock_getenv.side_effect = getenv_side_effect

    with patch("src.enrichment.api_helpers.anidb_helper.load_dotenv"):
        h = AniDBEnrichmentHelper()

    assert h.client_name == "animeenrichment"
    assert h.client_version == "1.0"
    assert h.circuit_breaker_state == CircuitBreakerState.CLOSED


def test_request_metrics():
    """Test the properties of AniDBRequestMetrics."""
    metrics = AniDBRequestMetrics()
    assert metrics.success_rate == 100.0
    assert metrics.error_rate == 0.0

    metrics.total_requests = 10
    metrics.successful_requests = 7
    assert metrics.success_rate == 70.0
    assert metrics.error_rate == 30.0


def test_generate_request_fingerprint(helper):
    """Test request fingerprint generation."""
    params1 = {"request": "anime", "aid": 123}
    params2 = {"aid": 123, "request": "anime"}
    params3 = {"request": "anime", "aid": 456}

    fp1 = helper._generate_request_fingerprint(params1)
    fp2 = helper._generate_request_fingerprint(params2)
    fp3 = helper._generate_request_fingerprint(params3)

    assert fp1 == fp2
    assert fp1 != fp3


@pytest.mark.asyncio
async def test_circuit_breaker_logic(helper):
    """Test the circuit breaker state transitions."""
    # Test opening the circuit
    helper.metrics.consecutive_failures = helper.circuit_breaker_threshold - 1
    helper._update_circuit_breaker(success=False)
    assert helper.circuit_breaker_state == CircuitBreakerState.OPEN
    assert helper.circuit_breaker_opened_at > 0

    # Test that requests are blocked when open
    assert not await helper._check_circuit_breaker()

    # Test half-open state after timeout
    helper.circuit_breaker_opened_at = time.time() - helper.circuit_breaker_timeout - 1
    assert await helper._check_circuit_breaker()
    assert helper.circuit_breaker_state == CircuitBreakerState.HALF_OPEN

    # Test closing from half-open
    helper._update_circuit_breaker(success=True)
    assert helper.circuit_breaker_state == CircuitBreakerState.CLOSED
    assert helper.metrics.consecutive_failures == 0

    # Test re-opening from half-open
    helper.circuit_breaker_state = CircuitBreakerState.HALF_OPEN
    helper.metrics.consecutive_failures = helper.circuit_breaker_threshold
    helper._update_circuit_breaker(success=False)
    assert helper.circuit_breaker_state == CircuitBreakerState.OPEN


@pytest.mark.asyncio
async def test_make_single_request_success(helper, mock_session):
    """Test a single successful request."""
    helper.session = mock_session
    params = {"request": "anime", "aid": 1}
    result = await helper._make_single_request(params, attempt=0)
    assert result == "<anime id='1'></anime>"
    mock_session.get.assert_called_once()


@pytest.mark.asyncio
async def test_make_single_request_gzip(helper, mock_session):
    """Test handling of gzipped responses."""
    gzipped_content = gzip.compress(b"<anime id='2'></anime>")
    mock_session.get.return_value.__aenter__.return_value.read = AsyncMock(
        return_value=gzipped_content
    )
    helper.session = mock_session
    params = {"request": "anime", "aid": 2}
    result = await helper._make_single_request(params, attempt=0)
    assert result == "<anime id='2'></anime>"


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [503, 555, 404])
async def test_make_single_request_http_errors(helper, mock_session, status_code):
    """Test handling of various HTTP error statuses."""
    mock_session.get.return_value.__aenter__.return_value.status = status_code
    helper.session = mock_session
    params = {"request": "anime", "aid": 1}
    result = await helper._make_single_request(params, attempt=0)
    assert result is None
    if status_code == 555:
        assert helper.circuit_breaker_state == CircuitBreakerState.OPEN


@pytest.mark.asyncio
async def test_make_single_request_api_error_xml(helper, mock_session):
    """Test handling of AniDB's <error> response."""
    error_xml = b"<error>Banned</error>"
    mock_session.get.return_value.__aenter__.return_value.read = AsyncMock(
        return_value=error_xml
    )
    helper.session = mock_session
    params = {"request": "anime", "aid": 1}
    result = await helper._make_single_request(params, attempt=0)
    assert result is None


@pytest.mark.asyncio
@patch("src.enrichment.api_helpers.anidb_helper.asyncio.sleep", new_callable=AsyncMock)
async def test_make_request_with_retry(mock_sleep, helper):
    """Test the retry logic."""
    helper.max_retries = 2
    helper._ensure_session_health = AsyncMock()
    helper.session = MagicMock()

    # Fail twice, then succeed
    side_effects = [None, None, "<anime id='1'></anime>"]
    helper._make_single_request = AsyncMock(side_effect=side_effects)

    result = await helper._make_request_with_retry({"request": "anime", "aid": 1})

    assert result == "<anime id='1'></anime>"
    assert helper._make_single_request.call_count == 3
    assert mock_sleep.call_count == 2  # Sleeps between retries


@pytest.mark.asyncio
@patch("src.enrichment.api_helpers.anidb_helper.asyncio.sleep", new_callable=AsyncMock)
async def test_make_request_with_retry_permanent_failure(mock_sleep, helper):
    """Test when retry logic is exhausted."""
    helper.max_retries = 1
    helper._ensure_session_health = AsyncMock()
    helper.session = MagicMock()

    # Always fail
    helper._make_single_request = AsyncMock(return_value=None)

    result = await helper._make_request_with_retry({"request": "anime", "aid": 1})

    assert result is None
    assert helper._make_single_request.call_count == 2
    assert mock_sleep.call_count == 1


def test_validate_anime_xml(helper):
    """Test XML validation logic."""
    assert helper._validate_anime_xml(ET.fromstring("<anime id='1'></anime>"))
    assert not helper._validate_anime_xml(ET.fromstring("<foo id='1'></foo>"))
    assert not helper._validate_anime_xml(ET.fromstring("<anime></anime>"))


@pytest.mark.asyncio
@patch(
    "src.enrichment.api_helpers.anidb_helper.fetch_anidb_character",
    new_callable=AsyncMock,
)
async def test_parse_anime_xml(mock_fetch_char, helper):
    """Test parsing of a complete anime XML."""
    mock_fetch_char.return_value = {"name": "Detailed Name"}
    xml_content = """
    <anime id="1" restricted="false">
        <type>TV Series</type>
        <episodecount>26</episodecount>
        <startdate>2000-01-01</startdate>
        <enddate>2000-06-30</enddate>
        <titles>
            <title type="main" xml:lang="x-jat">Cowboy Bebop</title>
            <title type="official" xml:lang="en">Cowboy Bebop</title>
        </titles>
        <description>Test description</description>
        <characters>
            <character id="101" type="main character in">
                <name>Spike Spiegel</name>
            </character>
        </characters>
        <episodes>
            <episode id="201">
                <epno type="1">1</epno>
                <length>24</length>
                <title xml:lang="en">Asteroid Blues</title>
            </episode>
        </episodes>
    </anime>
    """
    data = await helper._parse_anime_xml(xml_content)

    assert data["anidb_id"] == "1"
    assert data["title"] == "Cowboy Bebop"
    assert data["episode_count"] == "26"
    assert len(data["character_details"]) == 1
    assert data["character_details"][0]["name"] == "Detailed Name"
    assert len(data["episode_details"]) == 1
    assert data["episode_details"][0]["episode_number"] == 1
    mock_fetch_char.assert_called_once_with(101, return_data=True)


@pytest.mark.asyncio
async def test_get_anime_by_id_success(helper):
    """Test fetching anime by ID successfully."""
    xml_response = "<anime id='1'><titles><title type='main'>Test</title></titles></anime>"
    helper._make_request = AsyncMock(return_value=xml_response)
    helper._parse_anime_xml = AsyncMock(
        return_value={"anidb_id": "1", "title": "Test"}
    )

    result = await helper.get_anime_by_id(1)

    helper._make_request.assert_called_once_with({"request": "anime", "aid": 1})
    helper._parse_anime_xml.assert_called_once_with(xml_response)
    assert result["title"] == "Test"


@pytest.mark.asyncio
async def test_get_anime_by_id_not_found(helper):
    """Test fetching anime by ID when not found."""
    helper._make_request = AsyncMock(return_value=None)
    result = await helper.get_anime_by_id(99999)
    assert result is None


@pytest.mark.asyncio
async def test_search_anime_by_name(helper):
    """Test searching for an anime by name."""
    xml_response = "<anime id='1'><titles><title type='main'>Test</title></titles></anime>"
    helper._make_request = AsyncMock(return_value=xml_response)
    helper._parse_anime_xml = AsyncMock(
        return_value={"anidb_id": "1", "title": "Test"}
    )

    result = await helper.search_anime_by_name("Test")

    helper._make_request.assert_called_once_with(
        {"request": "anime", "aname": "Test"}
    )
    assert result[0]["title"] == "Test"


def test_decode_content(helper):
    """Test content decoding with fallbacks."""
    utf8_bytes = "你好".encode("utf-8")
    latin1_bytes = "é".encode("latin-1")
    invalid_bytes = b"\xff\xfe"

    assert helper._decode_content(utf8_bytes) == "你好"
    assert helper._decode_content(latin1_bytes) == "é"
    assert helper._decode_content(invalid_bytes) == 'ÿþ'


@pytest.mark.asyncio
async def test_close_session(helper, mock_session):
    """Test the close method."""
    helper.session = mock_session
    mock_session.close = AsyncMock()

    await helper.close()

    mock_session.close.assert_called_once()
    assert helper.session is None


@pytest.mark.asyncio
@patch("src.enrichment.api_helpers.anidb_helper.asyncio.sleep", new_callable=AsyncMock)
async def test_adaptive_rate_limit_logic(mock_sleep, helper):
    """Test the logic of the adaptive rate limiter."""
    helper._adaptive_rate_limit = AniDBEnrichmentHelper._adaptive_rate_limit.__get__(helper) # Un-mock the fixture's mock

    # Case 1: Normal operation
    helper.metrics.last_request_time = time.time() - 5
    helper.metrics.current_interval = 2.0
    await helper._adaptive_rate_limit()
    mock_sleep.assert_not_called()

    # Case 2: Needs to wait
    helper.metrics.last_request_time = time.time()
    await helper._adaptive_rate_limit()
    mock_sleep.assert_called_once()
    assert mock_sleep.call_args[0][0] > 0
    mock_sleep.reset_mock()

    # Case 3: Exponential backoff on error
    helper.metrics.consecutive_failures = 3
    helper.metrics.last_request_time = time.time()
    await helper._adaptive_rate_limit()
    # 2^3 = 8. Cooldown base is 5. 5*8=40, but max is 10. So interval is 10.
    assert mock_sleep.call_args[0][0] == pytest.approx(10, abs=0.1)
    mock_sleep.reset_mock()

    # Case 4: Extra delay for retry
    helper.metrics.consecutive_failures = 0
    helper.metrics.current_interval = 2.0
    helper.metrics.last_request_time = time.time()
    await helper._adaptive_rate_limit(is_retry=True)
    # Interval becomes 2.0 * 1.5 = 3.0
    assert mock_sleep.call_args[0][0] == pytest.approx(3.0, abs=0.1)


@pytest.mark.asyncio
@patch("src.enrichment.api_helpers.anidb_helper.aiohttp.ClientSession")
async def test_ensure_session_health(MockClientSession, helper):
    """Test session creation and recreation."""
    helper._ensure_session_health = AniDBEnrichmentHelper._ensure_session_health.__get__(helper)

    # 1. No session exists, should create one
    helper.session = None
    await helper._ensure_session_health()
    assert helper.session is not None
    MockClientSession.assert_called_once()

    # 2. Session is healthy, should do nothing
    MockClientSession.reset_mock()
    helper.session.close = AsyncMock()
    await helper._ensure_session_health()
    helper.session.close.assert_not_called()
    MockClientSession.assert_not_called()

    # 3. Session is old, should recreate
    helper._session_created_at = time.time() - (helper._session_max_age + 1)
    await helper._ensure_session_health()
    helper.session.close.assert_called_once()
    MockClientSession.assert_called_once()


@pytest.mark.asyncio
async def test_request_deduplication(helper):
    """Test that duplicate requests are skipped."""
    params = {"request": "anime", "aid": 999}
    fingerprint = helper._generate_request_fingerprint(params)
    helper.recent_requests.add(fingerprint)
    
    helper._make_request_with_retry = AsyncMock()
    
    # This should now return None because the fingerprint is in recent_requests
    result = await helper._make_request_with_retry(params)
    # The original test setup mocks _make_request_with_retry, so we need to call the real one
    # with a mock for the actual request part
    
    with patch.object(helper, "_make_single_request", new_callable=AsyncMock) as mock_single_request:
        # Add fingerprint to the set
        params = {"request": "anime", "aid": 999}
        fingerprint = helper._generate_request_fingerprint(params)
        helper.recent_requests.add(fingerprint)

        # Call the method that contains the deduplication logic
        result = await AniDBEnrichmentHelper._make_request_with_retry(helper, params)

        # Assert that no request was made and result is None
        assert result is None
        mock_single_request.assert_not_called()


@pytest.mark.asyncio
async def test_deduplication_cache_cleanup(helper):
    """Test that the recent_requests cache is cleaned up."""
    # Fill the cache
    for i in range(1001):
        helper.recent_requests.add(str(i))
    
    assert len(helper.recent_requests) > 1000

    with patch.object(helper, "_make_single_request", new_callable=AsyncMock, return_value="<anime/>"):
        params = {"request": "anime", "aid": "new"}
        await AniDBEnrichmentHelper._make_request_with_retry(helper, params)

    # Assert that the cache has been trimmed
    assert len(helper.recent_requests) < 1000


@pytest.mark.asyncio
async def test_request_when_circuit_breaker_open(helper):
    """Test that requests are blocked when the circuit breaker is open."""
    # Manually open the circuit breaker
    helper.circuit_breaker_state = CircuitBreakerState.OPEN
    helper.circuit_breaker_opened_at = time.time()

    with patch.object(helper, "_adaptive_rate_limit", new_callable=AsyncMock) as mock_rate_limit:
        params = {"request": "anime", "aid": 123}
        result = await AniDBEnrichmentHelper._make_request_with_retry(helper, params)
        
        assert result is None
        mock_rate_limit.assert_not_called() # Should not even get to rate limiting


@pytest.mark.asyncio
@patch("src.enrichment.api_helpers.anidb_helper.gzip.decompress")
async def test_gzip_decompression_failure(mock_decompress, helper, mock_session):
    """Test failure during gzip decompression."""
    mock_decompress.side_effect = gzip.BadGzipFile("Test error")
    
    gzipped_content = b"\x1f\x8b...invalid" # Starts with gzip magic number
    mock_session.get.return_value.__aenter__.return_value.read = AsyncMock(
        return_value=gzipped_content
    )
    helper.session = mock_session

    with pytest.raises(gzip.BadGzipFile):
        await helper._make_single_request({"request": "anime"}, attempt=0)


def test_validate_anime_xml_missing_elements(helper):
    """Test XML validation with missing critical elements."""
    # Missing <titles>
    xml_str_no_titles = "<anime id='1'><episodecount>1</episodecount></anime>"
    # Missing main title in <titles>
    xml_str_no_main_title = "<anime id='1'><titles><title type='official'>Title</title></titles></anime>"
    
    # These should log warnings but still return True as they are not fatal validation errors
    assert helper._validate_anime_xml(ET.fromstring(xml_str_no_titles))
    assert helper._validate_anime_xml(ET.fromstring(xml_str_no_main_title))


@pytest.mark.asyncio
@patch("src.enrichment.api_helpers.anidb_helper.fetch_anidb_character", new_callable=AsyncMock)
async def test_parse_character_xml_fetch_failure(mock_fetch_char, helper):
    """Test _parse_character_xml when fetch_anidb_character fails."""
    mock_fetch_char.side_effect = Exception("Network Error")
    
    char_xml_str = """
    <character id="101" type="main character in">
        <name>Spike Spiegel</name>
    </character>
    """
    char_element = ET.fromstring(char_xml_str)
    
    # The function should handle the exception gracefully and return the basic data
    char_data = await helper._parse_character_xml(char_element)
    
    assert char_data is not None
    assert char_data["name"] == "Spike Spiegel"
    mock_fetch_char.assert_called_once_with(101, return_data=True)

@pytest.fixture
def maximal_anime_xml():
    """Provides a comprehensive XML string with all possible fields for parsing tests."""
    return """
    <anime id="1" restricted="false">
        <type>TV Series</type>
        <episodecount>26</episodecount>
        <startdate>2000-01-01</startdate>
        <enddate>2000-06-30</enddate>
        <titles>
            <title type="main" xml:lang="x-jat">Cowboy Bebop</title>
            <title type="official" xml:lang="en">Cowboy Bebop EN</title>
            <title type="official" xml:lang="ja">カウボーイビバップ</title>
            <title type="synonym" xml:lang="en">CB</title>
            <title type="short" xml:lang="en">Bebop</title>
        </titles>
        <relatedanime>
            <anime id="2" type="Sequel">Cowboy Bebop: The Movie</anime>
        </relatedanime>
        <creators>
            <name id="10" type="Director">Watanabe Shinichirou</name>
        </creators>
        <description>Test description</description>
        <ratings>
            <permanent count="100">8.5</permanent>
            <temporary count="10">7.5</temporary>
            <review count="5">9.0</review>
        </ratings>
        <categories>
            <category id="1" parentid="0" hentai="false" weight="100">
                <name>Sci-Fi</name>
                <description>Science Fiction</description>
            </category>
        </categories>
        <characters>
            <character id="101" type="secondary character in">
                <rating votes="100">9.5</rating>
                <name>Spike Spiegel</name>
                <gender>Male</gender>
                <charactertype id="1">Human</charactertype>
                <description>A cool guy.</description>
                <picture>spike.jpg</picture>
                <seiyuu id="201" picture="seiyuu.jpg">Yamadera Kouichi</seiyuu>
            </character>
        </characters>
        <episodes>
            <episode id="201">
                <epno type="1">1</epno>
                <length>24</length>
                <airdate>2000-01-01</airdate>
                <rating votes="10">8.0</rating>
                <summary>Episode summary</summary>
                <title xml:lang="en">Asteroid Blues</title>
                <title xml:lang="x-jat">Asteroid Blues Romaji</title>
                <resources>
                    <resource type="28">
                        <externalentity>
                            <identifier>G6NQ5DWZ6</identifier>
                        </externalentity>
                    </resource>
                </resources>
            </episode>
            <episode id="202">
                <epno type="1">S1</epno> <!-- Test non-int episode number -->
            </episode>
        </episodes>
        <tags>
            <tag id="30" count="50" weight="200">
                <name>space</name>
                <description>Outer space</description>
            </tag>
        </tags>
        <url>http://anidb.net/a1</url>
        <picture>anime.jpg</picture>
    </anime>
    """

@pytest.mark.asyncio
@patch("src.enrichment.api_helpers.anidb_helper.fetch_anidb_character", new_callable=AsyncMock)
async def test_parse_anime_xml_maximal(mock_fetch_char, helper, maximal_anime_xml):
    """Test parsing of a comprehensive anime XML including all optional fields."""
    mock_fetch_char.return_value = {"name": "Detailed Spike"}
    data = await helper._parse_anime_xml(maximal_anime_xml)

    assert data["title_english"] == "Cowboy Bebop EN"
    assert data["title_japanese"] == "カウボーイビバップ"
    assert "CB" in data["synonyms"]
    assert "Bebop" in data["synonyms"]
    assert len(data["related_anime"]) == 1
    assert data["related_anime"][0]["title"] == "Cowboy Bebop: The Movie"
    assert len(data["creators"]) == 1
    assert data["creators"][0]["name"] == "Watanabe Shinichirou"
    assert data["ratings"]["permanent"]["value"] == 8.5
    assert data["ratings"]["temporary"]["count"] == 10
    assert data["ratings"]["review"]["value"] == 9.0
    assert len(data["categories"]) == 1
    assert data["categories"][0]["name"] == "Sci-Fi"
    assert len(data["tags"]) == 1
    assert data["tags"][0]["description"] == "Outer space"
    assert data["url"] == "http://anidb.net/a1"
    assert data["picture"] == "anime.jpg"
    assert data["episode_details"][0]["streaming"]["crunchyroll"] is not None
    assert data["episode_details"][1]["episode_number"] == "S1"
    assert data["character_details"][0]["type"] == "Secondary"
    assert data["character_details"][0]["rating"] == 9.5
    assert data["character_details"][0]["voice_actor"]["id"] == "201"


@pytest.mark.asyncio
async def test_get_anime_by_id_api_error(helper):
    """Test get_anime_by_id when the API returns a known error."""
    helper._make_request = AsyncMock(return_value="<error>Anime not found</error>")
    result = await helper.get_anime_by_id(99999)
    assert result is None

@pytest.mark.asyncio
async def test_get_anime_by_id_exception(helper):
    """Test get_anime_by_id when the request process raises an exception."""
    helper._make_request = AsyncMock(side_effect=Exception("Network timeout"))
    result = await helper.get_anime_by_id(99999)
    assert result is None

@pytest.mark.asyncio
async def test_search_anime_by_name_not_found(helper):
    """Test search_anime_by_name when the API returns an error."""
    helper._make_request = AsyncMock(return_value="<error>Anime not found</error>")
    result = await helper.search_anime_by_name("Nonexistent Anime")
    assert result is None

@pytest.mark.asyncio
async def test_search_anime_by_name_exception(helper):
    """Test search_anime_by_name when the request process raises an exception."""
    helper._make_request = AsyncMock(side_effect=Exception("Network timeout"))
    result = await helper.search_anime_by_name("Nonexistent Anime")
    assert result is None

@pytest.mark.asyncio
async def test_fetch_all_data_not_found(helper):
    """Test fetch_all_data for an ID that returns no data."""
    helper.get_anime_by_id = AsyncMock(return_value=None)
    result = await helper.fetch_all_data(99999)
    assert result is None

@pytest.mark.asyncio
async def test_fetch_all_data_exception(helper):
    """Test fetch_all_data when an exception occurs."""
    helper.get_anime_by_id = AsyncMock(side_effect=Exception("Unexpected error"))
    result = await helper.fetch_all_data(99999)
    assert result is None

@pytest.mark.asyncio
async def test_reset_circuit_breaker(helper):
    """Test manually resetting the circuit breaker."""
    helper.circuit_breaker_state = CircuitBreakerState.OPEN
    reset_happened = await helper.reset_circuit_breaker()
    assert reset_happened is True
    assert helper.circuit_breaker_state == CircuitBreakerState.CLOSED
    # Test resetting when already closed
    reset_happened_again = await helper.reset_circuit_breaker()
    assert reset_happened_again is False


@pytest.mark.asyncio
async def test_get_health_status(helper):
    """Test the health status report structure."""
    helper.circuit_breaker_state = CircuitBreakerState.OPEN
    helper.circuit_breaker_opened_at = time.time()
    status = helper.get_health_status()
    assert status["circuit_breaker"]["state"] == "open"
    assert status["request_metrics"]["total_requests"] == 0


@pytest.mark.asyncio
async def test_parse_anime_with_missing_episode_fields(helper):
    """Test parsing an episode that is missing non-critical fields like epno."""
    xml_content = """
    <anime id="1">
        <episodes>
            <episode id="201">
                <length>24</length>
            </episode>
        </episodes>
    </anime>
    """
    # This should parse without error, logging a warning
    data = await helper._parse_anime_xml(xml_content)
    assert data is not None
    assert len(data["episode_details"]) == 1
    assert data["episode_details"][0]["id"] == 201


@pytest.mark.asyncio
@patch("src.enrichment.api_helpers.anidb_helper.AniDBEnrichmentHelper.fetch_all_data", new_callable=AsyncMock)
@patch("argparse.ArgumentParser.parse_args")
async def test_main_by_id(mock_parse_args, mock_fetch, tmp_path):
    """Test the main function when fetching by anidb-id."""
    output_path = tmp_path / "output.json"
    mock_parse_args.return_value = MagicMock(anidb_id=1, search_name=None, output=str(output_path))
    mock_fetch.return_value = {"anidb_id": "1"}
    from src.enrichment.api_helpers import anidb_helper
    await anidb_helper.main()
    mock_fetch.assert_called_once_with(1)
    assert output_path.exists()


@pytest.mark.asyncio
@patch("src.enrichment.api_helpers.anidb_helper.AniDBEnrichmentHelper.search_anime_by_name", new_callable=AsyncMock)
@patch("argparse.ArgumentParser.parse_args")
async def test_main_by_search(mock_parse_args, mock_search, tmp_path):
    """Test the main function when searching by name."""
    output_path = tmp_path / "output.json"
    mock_parse_args.return_value = MagicMock(anidb_id=None, search_name="Bebop", output=str(output_path))
    mock_search.return_value = [{"title": "Bebop"}]
    from src.enrichment.api_helpers import anidb_helper
    await anidb_helper.main()
    mock_search.assert_called_once_with("Bebop")
    assert output_path.exists()


@pytest.mark.asyncio
@patch("argparse.ArgumentParser.parse_args")
async def test_main_no_args(mock_parse_args):
    """Test the main function with no arguments."""
    mock_parse_args.return_value = MagicMock(anidb_id=None, search_name=None)
    from src.enrichment.api_helpers import anidb_helper
    # Should just log an error and return
    await anidb_helper.main()


@pytest.mark.asyncio
@patch("src.enrichment.api_helpers.anidb_helper.AniDBEnrichmentHelper.fetch_all_data", new_callable=AsyncMock)
@patch("argparse.ArgumentParser.parse_args")
async def test_main_no_data_found(mock_parse_args, mock_fetch):
    """Test the main function when no data is found."""
    mock_parse_args.return_value = MagicMock(anidb_id=1, search_name=None, output="out.json")
    mock_fetch.return_value = None
    from src.enrichment.api_helpers import anidb_helper
    await anidb_helper.main()
    mock_fetch.assert_called_once_with(1)


@patch("asyncio.run")
def test_helper_dunder_main(mock_run):
    """Test the `if __name__ == '__main__'` block for the helper."""
    import runpy
    import asyncio
    runpy.run_path("src/enrichment/api_helpers/anidb_helper.py", run_name="__main__")
    mock_run.assert_called_once()
    call_arg = mock_run.call_args[0][0]
    assert asyncio.iscoroutine(call_arg)
    assert call_arg.__name__ == 'main'


@pytest.mark.asyncio
@patch("src.enrichment.api_helpers.anidb_helper.AniDBEnrichmentHelper.fetch_all_data", new_callable=AsyncMock)
@patch("argparse.ArgumentParser.parse_args")
async def test_main_keyboard_interrupt(mock_parse_args, mock_fetch):
    """Test the main function handles KeyboardInterrupt."""
    mock_parse_args.return_value = MagicMock(anidb_id=1, search_name=None, output="out.json")
    mock_fetch.side_effect = KeyboardInterrupt

    from src.enrichment.api_helpers import anidb_helper
    # Should catch the exception and exit gracefully
    await anidb_helper.main()


@pytest.mark.asyncio
async def test_make_request_client_error(helper):
    """Test the request retry logic when a ClientError is raised."""
    helper.max_retries = 1
    helper._ensure_session_health = AsyncMock()
    helper.session = MagicMock()
    helper.session.get.side_effect = ClientError("Connection failed")
    
    # Un-mock the retry logic for this test
    helper._make_request_with_retry = AniDBEnrichmentHelper._make_request_with_retry.__get__(helper)

    with patch("asyncio.sleep", new_callable=AsyncMock):
        result = await helper._make_request_with_retry({"request": "anime"})
        assert result is None
        assert helper.session.get.call_count == 2 # 1 attempt + 1 retry


def test_validate_anime_xml_missing_epno(helper):
    """Test _validate_anime_xml with an episode missing an epno tag."""
    xml_str = """<anime id='1'>
        <episodes><episode id='2'></episode></episodes>
    </anime>"""
    # Should log a warning but return True
    assert helper._validate_anime_xml(ET.fromstring(xml_str)) is True


def test_parse_episode_with_empty_title(helper):
    """Test parsing an episode with an empty title tag."""
    xml_str = """<episode id='1'><title xml:lang='en'></title></episode>"""
    result = helper._parse_episode_xml(ET.fromstring(xml_str))
    assert "en" not in result["titles"] # Empty text should not be added


def test_get_health_status_no_session(helper):
    """Test health status when session is None."""
    helper.session = None
    status = helper.get_health_status()
    assert status["session_health"]["session_active"] is False
    assert status["session_health"]["session_age"] == 0


@pytest.mark.asyncio
@patch("src.enrichment.api_helpers.anidb_helper.AniDBEnrichmentHelper.fetch_all_data", new_callable=AsyncMock)
@patch("argparse.ArgumentParser.parse_args")
async def test_main_generic_exception(mock_parse_args, mock_fetch):
    """Test the main function handles a generic Exception."""
    mock_parse_args.return_value = MagicMock(anidb_id=1, search_name=None, output="out.json")
    mock_fetch.side_effect = Exception("Generic error")

    from src.enrichment.api_helpers import anidb_helper
    # Should catch the exception and log it
    await anidb_helper.main()


