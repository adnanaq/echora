import asyncio
import gzip
import time
import xml.etree.ElementTree as ET
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from enrichment.api_helpers.anidb_helper import (
    AniDBEnrichmentHelper,
    AniDBRequestMetrics,
    CircuitBreakerState,
)


@pytest.fixture
def helper():
    """Fixture for AniDBEnrichmentHelper."""
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
@patch("enrichment.api_helpers.anidb_helper.os.getenv")
def test_helper_initialization(mock_getenv):
    """Test that the helper initializes correctly."""

    # Mock os.getenv to return the default value provided in the call
    def getenv_side_effect(key, default=None):
        return default

    mock_getenv.side_effect = getenv_side_effect

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
@patch("enrichment.api_helpers.anidb_helper.asyncio.sleep", new_callable=AsyncMock)
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
@patch("enrichment.api_helpers.anidb_helper.asyncio.sleep", new_callable=AsyncMock)
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


@pytest.mark.parametrize(
    "xml_str, expected",
    [
        ("<anime id='1'></anime>", True),
        ("<foo id='1'></foo>", False),
        ("<anime></anime>", False),
        ("<anime id='1'><episodecount>1</episodecount></anime>", True),
        (
            "<anime id='1'><titles><title type='official'>Title</title></titles></anime>",
            True,
        ),
        ("<anime id='1'><episodes><episode id='2'></episode></episodes></anime>", True),
    ],
)
def test_validate_anime_xml_consolidated(helper, xml_str, expected):
    """Consolidated test for XML validation logic with various edge cases."""
    assert helper._validate_anime_xml(ET.fromstring(xml_str)) == expected


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
@patch(
    "enrichment.api_helpers.anidb_helper.fetch_anidb_character", new_callable=AsyncMock
)
async def test_parse_anime_xml_comprehensive(
    mock_fetch_char, helper, maximal_anime_xml
):
    """Test parsing of a comprehensive anime XML including all optional fields."""
    mock_fetch_char.return_value = {"name": "Detailed Spike"}
    data = await helper._parse_anime_xml(maximal_anime_xml)

    assert data["anidb_id"] == "1"
    assert data["title"] == "Cowboy Bebop"
    assert data["title_english"] == "Cowboy Bebop EN"
    assert data["title_japanese"] == "カウボーイビバップ"
    assert "CB" in data["synonyms"]
    assert "Bebop" in data["synonyms"]
    assert len(data["related_anime"]) == 1
    assert data["related_anime"][0]["title"] == "Cowboy Bebop: The Movie"
    assert len(data["creators"]) == 1
    assert data["creators"][0]["name"] == "Watanabe Shinichirou"
    assert data["statistics"]["score"] == 8.5
    assert data["statistics"]["scored_by"] == 100
    assert len(data["categories"]) == 1
    assert data["categories"][0]["name"] == "Sci-Fi"
    assert len(data["tags"]) == 1
    assert data["tags"][0] == "space"
    assert data["url"] == "http://anidb.net/a1"
    assert data["cover"] == "https://cdn-eu.anidb.net/images/main/anime.jpg"
    assert data["episode_details"][0]["episode_number"] == 1
    assert data["episode_details"][0]["streaming"]["crunchyroll"] is not None
    assert data["episode_details"][1]["episode_number"] == "S1"
    assert len(data["character_details"]) == 1
    assert data["character_details"][0]["name_main"] == "Spike Spiegel"
    assert data["character_details"][0]["type"] == "Secondary"
    assert data["character_details"][0]["rating"] == 9.5
    assert data["character_details"][0]["voice_actor"]["id"] == "201"
    mock_fetch_char.assert_called_once_with(101)


@pytest.mark.asyncio
async def test_get_anime_by_id_workflow(helper):
    """Test the complete workflow of fetching anime by ID, including error paths."""
    # Success case
    xml_response = (
        "<anime id='1'><titles><title type='main'>Test</title></titles></anime>"
    )
    helper._make_request = AsyncMock(return_value=xml_response)
    helper._parse_anime_xml = AsyncMock(return_value={"anidb_id": "1", "title": "Test"})
    result = await helper.get_anime_by_id(1)
    assert result["title"] == "Test"

    # Not found (empty response)
    helper._make_request = AsyncMock(return_value=None)
    assert await helper.get_anime_by_id(999) is None

    # API Error case
    helper._make_request = AsyncMock(return_value="<error>Anime not found</error>")
    assert await helper.get_anime_by_id(999) is None

    # Exception case
    helper._make_request = AsyncMock(side_effect=Exception("Network timeout"))
    assert await helper.get_anime_by_id(999) is None


def test_decode_content(helper):
    """Test content decoding with fallbacks."""
    utf8_bytes = "你好".encode()
    latin1_bytes = "é".encode("latin-1")
    invalid_bytes = b"\xff\xfe"

    assert helper._decode_content(utf8_bytes) == "你好"
    assert helper._decode_content(latin1_bytes) == "é"
    assert helper._decode_content(invalid_bytes) == "ÿþ"


@pytest.mark.asyncio
async def test_session_management(helper, mock_session):
    """Test closing session and handling health checks."""
    helper.session = mock_session
    mock_session.close = AsyncMock()
    await helper.close()
    mock_session.close.assert_called_once()
    assert helper.session is None


@pytest.mark.asyncio
@patch("enrichment.api_helpers.anidb_helper.asyncio.sleep", new_callable=AsyncMock)
async def test_adaptive_rate_limit_logic(mock_sleep, helper):
    """Test the logic of the adaptive rate limiter across different scenarios."""
    helper._adaptive_rate_limit = AniDBEnrichmentHelper._adaptive_rate_limit.__get__(
        helper
    )

    # Case 1: Normal operation (no wait)
    helper.metrics.last_request_time = time.time() - 5
    helper.metrics.current_interval = 2.0
    await helper._adaptive_rate_limit()
    mock_sleep.assert_not_called()

    # Case 2: Needs to wait
    helper.metrics.last_request_time = time.time()
    await helper._adaptive_rate_limit()
    mock_sleep.assert_called_once()
    mock_sleep.reset_mock()

    # Case 3: Exponential backoff on error
    helper.metrics.consecutive_failures = 3
    helper.metrics.last_request_time = time.time()
    await helper._adaptive_rate_limit()
    assert mock_sleep.call_args[0][0] == pytest.approx(10, abs=0.1)


@pytest.mark.asyncio
@patch("enrichment.api_helpers.anidb_helper.aiohttp.ClientSession")
async def test_ensure_session_health(MockClientSession, helper):
    """Test session creation and expiration logic."""
    helper._ensure_session_health = (
        AniDBEnrichmentHelper._ensure_session_health.__get__(helper)
    )

    # 1. Create session
    helper.session = None
    await helper._ensure_session_health()
    assert helper.session is not None

    # 2. Recreate expired session
    helper._session_created_at = time.time() - (helper._session_max_age + 1)
    old_session_close = helper.session.close = AsyncMock()
    await helper._ensure_session_health()
    old_session_close.assert_called_once()


@pytest.mark.asyncio
async def test_circuit_breaker_blocking(helper):
    """Test that requests are blocked when the circuit breaker is open."""
    helper.circuit_breaker_state = CircuitBreakerState.OPEN
    helper.circuit_breaker_opened_at = time.time()

    with patch.object(
        helper, "_adaptive_rate_limit", new_callable=AsyncMock
    ) as mock_rate:
        result = await AniDBEnrichmentHelper._make_request_with_retry(
            helper, {"aid": 123}
        )
        assert result is None
        mock_rate.assert_not_called()


@pytest.mark.asyncio
@patch(
    "enrichment.api_helpers.anidb_helper.fetch_anidb_character", new_callable=AsyncMock
)
async def test_parse_character_xml_error_handling(mock_fetch_char, helper):
    """Test _parse_character_xml handles fetch failures and missing fields."""
    # Case 1: Fetch failure
    mock_fetch_char.side_effect = Exception("Network Error")
    char_xml = ET.fromstring("<character id='101'><name>Spike</name></character>")
    char_data = await helper._parse_character_xml(char_xml)
    assert char_data["name_main"] == "Spike"

    # Case 2: Missing non-critical episode fields in parsing
    xml_missing_ep = "<anime id='1'><episodes><episode id='201'><length>24</length></episode></episodes></anime>"
    data = await helper._parse_anime_xml(xml_missing_ep)
    assert data["episode_details"][0]["id"] == 201


def test_internal_parsers_granular(helper):
    """Granular tests for existing internal parsing methods and inlined logic."""
    # Test _parse_episode_xml
    ep_xml = ET.fromstring(
        "<episode id='10'><epno type='1'>5</epno><length>24</length></episode>"
    )
    ep_data = helper._parse_episode_xml(ep_xml)
    assert ep_data["episode_number"] == 5
    assert ep_data["episode_type"] == 1

    # Test inlined related anime parsing via main parser
    rel_xml = "<anime id='1'><relatedanime><anime id='2' type='Sequel'>Movie</anime></relatedanime></anime>"
    rel_data = asyncio.run(helper._parse_anime_xml(rel_xml))
    assert rel_data["related_anime"][0]["url"] == "https://anidb.net/anime/2"
    assert rel_data["related_anime"][0]["relation"] == "Sequel"

    # Test inlined creator parsing via main parser
    creator_xml = "<anime id='1'><creators><name id='1' type='Director'>Watanabe</name></creators></anime>"
    creator_data = asyncio.run(helper._parse_anime_xml(creator_xml))
    assert creator_data["creators"][0]["id"] == "1"
    assert creator_data["creators"][0]["name"] == "Watanabe"

    # Test inlined category parsing via main parser
    cat_xml = "<anime id='1'><categories><category id='1' weight='100'><name>Sci-Fi</name></category></categories></anime>"
    cat_data = asyncio.run(helper._parse_anime_xml(cat_xml))
    assert cat_data["categories"][0]["id"] == "1"
    assert cat_data["categories"][0]["name"] == "Sci-Fi"

    # Test inlined ratings parsing via main parser
    rat_xml = (
        "<anime id='1'><ratings><permanent count='10'>8.5</permanent></ratings></anime>"
    )
    rat_data = asyncio.run(helper._parse_anime_xml(rat_xml))
    assert rat_data["statistics"]["score"] == 8.5


@pytest.mark.asyncio
@patch(
    "enrichment.api_helpers.anidb_helper.AniDBEnrichmentHelper.fetch_all_data",
    new_callable=AsyncMock,
)
@patch("argparse.ArgumentParser.parse_args")
async def test_main_cli_scenarios(mock_parse_args, mock_fetch, tmp_path):
    """Consolidated test for various CLI entry point scenarios."""
    output_path = tmp_path / "output.json"
    from enrichment.api_helpers import anidb_helper

    # Case 1: Fetch by ID (Success)
    mock_parse_args.return_value = MagicMock(
        anidb_id=1, search_name=None, output=str(output_path), save_xml=None
    )
    mock_fetch.return_value = {"anidb_id": "1"}
    await anidb_helper.main()
    assert output_path.exists()

    # Case 2: KeyboardInterrupt
    mock_fetch.side_effect = KeyboardInterrupt
    assert await anidb_helper.main() == 1

    # Case 3: Generic Exception
    mock_fetch.side_effect = Exception("Generic error")
    assert await anidb_helper.main() == 1


@pytest.mark.asyncio
async def test_context_manager_protocol(helper):
    """Test AniDBEnrichmentHelper implements async context manager protocol."""
    from enrichment.api_helpers.anidb_helper import AniDBEnrichmentHelper

    mock_session = AsyncMock()
    async with AniDBEnrichmentHelper() as helper:
        helper.session = mock_session
        assert isinstance(helper, AniDBEnrichmentHelper)

    mock_session.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_tags_extraction_format():
    """Test tags extraction produces correct format and filters empty values."""
    from enrichment.api_helpers.anidb_helper import AniDBEnrichmentHelper

    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
    <anime id="1">
        <tags>
            <tag id="1" count="10" weight="100">
                <name>action</name>
            </tag>
            <tag id="2" count="5" weight="50">
                <name>adventure</name>
            </tag>
            <tag id="3" count="0" weight="0">
                <name></name>
            </tag>
        </tags>
    </anime>
    """

    helper = AniDBEnrichmentHelper()
    result = await helper._parse_anime_xml(xml_content)

    assert result["tags"] == ["action", "adventure"]
    assert "" not in result["tags"], "Empty tags should be filtered"
    assert None not in result["tags"], "None tags should be filtered"


@pytest.mark.asyncio
async def test_categories_extraction_format():
    """Test categories extraction produces correct format with all fields."""
    from enrichment.api_helpers.anidb_helper import AniDBEnrichmentHelper

    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
    <anime id="1">
        <categories>
            <category id="1" parentid="0" weight="600" hentai="false">
                <name>Action</name>
                <description>Action category</description>
            </category>
            <category id="2" parentid="1" weight="400" hentai="true">
                <name>Ecchi</name>
                <description>Ecchi category</description>
            </category>
        </categories>
    </anime>
    """

    helper = AniDBEnrichmentHelper()
    result = await helper._parse_anime_xml(xml_content)

    assert len(result["categories"]) == 2

    # Verify first category structure
    assert result["categories"][0] == {
        "id": "1",
        "name": "Action",
        "weight": 600,
        "hentai": False,
    }

    # Verify second category structure
    assert result["categories"][1]["id"] == "2"
    assert result["categories"][1]["hentai"] is True
    assert result["categories"][1]["weight"] == 400


@pytest.mark.asyncio
async def test_creators_extraction_format():
    """Test creators extraction produces correct format with all fields."""
    from enrichment.api_helpers.anidb_helper import AniDBEnrichmentHelper

    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
    <anime id="1">
        <creators>
            <name id="123" type="Director">John Doe</name>
            <name id="456" type="Music">Jane Smith</name>
            <name id="789" type="Animation Work">Studio A</name>
        </creators>
    </anime>
    """

    helper = AniDBEnrichmentHelper()
    result = await helper._parse_anime_xml(xml_content)

    assert len(result["creators"]) == 3

    # Verify structure of each creator
    assert result["creators"][0] == {"id": "123", "name": "John Doe", "type": "Director"}
    assert result["creators"][1] == {"id": "456", "name": "Jane Smith", "type": "Music"}
    assert result["creators"][2] == {
        "id": "789",
        "name": "Studio A",
        "type": "Animation Work",
    }
