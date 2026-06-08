"""Unit tests for anidb_helper.py — AniDBHelper orchestrator, HTTP transport, circuit breaker."""

import gzip
import json
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from enrichment.sources.anidb.anidb_helper import (
    AniDBHelper,
    AniDBRequestMetrics,
    CircuitBreakerState,
)

_ANIDB_URL = "https://anidb.net/anime/69"


async def _async_gen(items: list[Any]) -> AsyncGenerator[Any]:
    for item in items:
        yield item


# Minimal valid XML for unit tests that go through parse_anime_xml
_MINIMAL_XML = (
    '<anime id="69">'
    "<type>TV Series</type>"
    "<episodecount>0</episodecount>"
    "<titles>"
    '<title type="main" xml:lang="x-jat">One Piece</title>'
    "</titles>"
    "</anime>"
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def helper():
    """AniDBHelper with deterministic env config and rate-limiting disabled."""
    with patch("enrichment.sources.anidb.anidb_helper.os.getenv") as mock_getenv:
        mock_getenv.side_effect = lambda key, default=None: default
        h = AniDBHelper()
        with patch.object(h, "_adaptive_rate_limit", new_callable=AsyncMock):
            yield h


@pytest.fixture
def mock_session():
    """Mocked aiohttp session with a successful 200 XML response."""
    session = MagicMock()
    session.close = AsyncMock()
    cm = AsyncMock()
    response = AsyncMock()
    cm.__aenter__.return_value = response
    cm.__aexit__.return_value = False
    session.get.return_value = cm
    response.status = 200
    response.read = AsyncMock(return_value=b"<anime id='1'></anime>")
    return session


# =============================================================================
# INITIALIZATION
# =============================================================================


@patch("enrichment.sources.anidb.anidb_helper.os.getenv")
def test_helper_initialization(mock_getenv) -> None:
    mock_getenv.side_effect = lambda key, default=None: default
    h = AniDBHelper()
    assert h.client_name == "animeenrichment"
    assert h.client_version == "1.0"
    assert h.circuit_breaker_state == CircuitBreakerState.CLOSED


def test_request_metrics() -> None:
    metrics = AniDBRequestMetrics()
    assert metrics.success_rate == 100.0
    assert metrics.error_rate == 0.0

    metrics.total_requests = 10
    metrics.successful_requests = 7
    assert metrics.success_rate == 70.0
    assert metrics.error_rate == 30.0


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================


@pytest.mark.asyncio
async def test_circuit_breaker_logic(helper) -> None:
    with patch("enrichment.sources.anidb.anidb_helper.time.time") as mock_time:
        mock_time.return_value = 1000.0

        helper.metrics.consecutive_failures = helper.circuit_breaker_threshold - 1
        helper._update_circuit_breaker(success=False)
        assert helper.circuit_breaker_state == CircuitBreakerState.OPEN
        assert helper.circuit_breaker_opened_at == 1000.0

        assert not await helper._check_circuit_breaker()

        mock_time.return_value = 1000.0 + helper.circuit_breaker_timeout + 1
        assert await helper._check_circuit_breaker()
        assert helper.circuit_breaker_state == CircuitBreakerState.HALF_OPEN

        helper._update_circuit_breaker(success=True)
        assert helper.circuit_breaker_state == CircuitBreakerState.CLOSED
        assert helper.metrics.consecutive_failures == 0

        helper.circuit_breaker_state = CircuitBreakerState.HALF_OPEN
        helper.metrics.consecutive_failures = helper.circuit_breaker_threshold
        helper._update_circuit_breaker(success=False)
        assert helper.circuit_breaker_state == CircuitBreakerState.OPEN


@pytest.mark.asyncio
async def test_circuit_breaker_blocking(helper) -> None:
    helper.circuit_breaker_state = CircuitBreakerState.OPEN
    helper.circuit_breaker_opened_at = time.time()

    with patch.object(
        helper, "_adaptive_rate_limit", new_callable=AsyncMock
    ) as mock_rate:
        result = await AniDBHelper._make_request_with_retry(helper, {"aid": 123})
        assert result is None
        mock_rate.assert_not_called()


@pytest.mark.asyncio
async def test_reset_circuit_breaker_complete(helper) -> None:
    helper.circuit_breaker_state = CircuitBreakerState.OPEN
    helper.circuit_breaker_opened_at = time.time()
    helper.metrics.consecutive_failures = 5

    assert await helper.reset_circuit_breaker() is True
    assert helper.circuit_breaker_state == CircuitBreakerState.CLOSED
    assert helper.circuit_breaker_opened_at == 0.0
    assert helper.metrics.consecutive_failures == 0

    assert await helper.reset_circuit_breaker() is False


# =============================================================================
# HTTP SINGLE REQUEST
# =============================================================================


@pytest.mark.asyncio
async def test_make_single_request_success(helper, mock_session) -> None:
    helper.session = mock_session
    result = await helper._make_single_request(
        {"request": "anime", "aid": 1}, attempt=0
    )
    assert result == "<anime id='1'></anime>"
    mock_session.get.assert_called_once()


@pytest.mark.asyncio
async def test_make_single_request_gzip(helper, mock_session) -> None:
    gzipped = gzip.compress(b"<anime id='2'></anime>")
    mock_session.get.return_value.__aenter__.return_value.read = AsyncMock(
        return_value=gzipped
    )
    helper.session = mock_session
    result = await helper._make_single_request(
        {"request": "anime", "aid": 2}, attempt=0
    )
    assert result == "<anime id='2'></anime>"


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [503, 404])
async def test_make_single_request_http_errors(
    helper, mock_session, status_code
) -> None:
    mock_session.get.return_value.__aenter__.return_value.status = status_code
    helper.session = mock_session
    result = await helper._make_single_request(
        {"request": "anime", "aid": 1}, attempt=0
    )
    assert result is None


@pytest.mark.asyncio
async def test_make_single_request_555_raises_blocked(helper, mock_session) -> None:
    from enrichment.sources.base.exceptions import ServiceBlockedError

    mock_session.get.return_value.__aenter__.return_value.status = 555
    helper.session = mock_session
    with pytest.raises(ServiceBlockedError):
        await helper._make_single_request({"request": "anime", "aid": 1}, attempt=0)
    assert helper.circuit_breaker_state == CircuitBreakerState.OPEN


@pytest.mark.asyncio
async def test_make_single_request_api_error_xml(helper, mock_session) -> None:
    mock_session.get.return_value.__aenter__.return_value.read = AsyncMock(
        return_value=b"<error>Banned</error>"
    )
    helper.session = mock_session
    result = await helper._make_single_request(
        {"request": "anime", "aid": 1}, attempt=0
    )
    assert result is None


# =============================================================================
# RETRY LOGIC
# =============================================================================


@pytest.mark.asyncio
@patch("enrichment.sources.anidb.anidb_helper.asyncio.sleep", new_callable=AsyncMock)
async def test_make_request_with_retry(mock_sleep, helper) -> None:
    helper.max_retries = 2
    helper._ensure_session_health = AsyncMock()
    helper.session = MagicMock()
    helper._make_single_request = AsyncMock(
        side_effect=[None, None, "<anime id='1'></anime>"]
    )

    result = await helper._make_request_with_retry({"request": "anime", "aid": 1})

    assert result == "<anime id='1'></anime>"
    assert helper._make_single_request.call_count == 3
    assert mock_sleep.call_count == 2
    assert helper.metrics.total_requests == 3
    assert helper.metrics.successful_requests == 1
    assert helper.metrics.failed_requests == 2


@pytest.mark.asyncio
@patch("enrichment.sources.anidb.anidb_helper.asyncio.sleep", new_callable=AsyncMock)
async def test_make_request_with_retry_permanent_failure(mock_sleep, helper) -> None:
    helper.max_retries = 1
    helper._ensure_session_health = AsyncMock()
    helper.session = MagicMock()
    helper._make_single_request = AsyncMock(return_value=None)

    result = await helper._make_request_with_retry({"request": "anime", "aid": 1})

    assert result is None
    assert helper._make_single_request.call_count == 2
    assert mock_sleep.call_count == 1
    assert helper.metrics.total_requests == 2
    assert helper.metrics.successful_requests == 0
    assert helper.metrics.failed_requests == 2


# =============================================================================
# DECODE CONTENT
# =============================================================================


def test_decode_content(helper) -> None:
    assert helper._decode_content("你好".encode()) == "你好"
    assert helper._decode_content("é".encode("latin-1")) == "é"
    assert helper._decode_content(b"\xff\xfe") == "ÿþ"


# =============================================================================
# SESSION MANAGEMENT
# =============================================================================


@pytest.mark.asyncio
async def test_session_management(helper, mock_session) -> None:
    helper.session = mock_session
    mock_session.close = AsyncMock()
    await helper.close()
    mock_session.close.assert_called_once()
    assert helper.session is None


@pytest.mark.asyncio
@patch("enrichment.sources.anidb.anidb_helper._cache_manager.get_aiohttp_session")
@patch("enrichment.sources.anidb.anidb_helper.time.time")
async def test_ensure_session_health(mock_time, mock_get_session, helper) -> None:
    helper._ensure_session_health = AniDBHelper._ensure_session_health.__get__(helper)
    mock_session = AsyncMock()
    mock_get_session.return_value = mock_session

    mock_time.return_value = 1000.0
    helper.session = None
    await helper._ensure_session_health()
    assert helper.session is mock_session
    mock_get_session.assert_called_once()

    mock_get_session.reset_mock()
    mock_time.return_value = 2000.0
    helper._session_created_at = 1000.0
    old_close = helper.session.close = AsyncMock()
    await helper._ensure_session_health()
    old_close.assert_called_once()
    mock_get_session.assert_called_once()


# =============================================================================
# ADAPTIVE RATE LIMITING
# =============================================================================


@pytest.mark.asyncio
@patch("enrichment.sources.anidb.anidb_helper.asyncio.sleep", new_callable=AsyncMock)
@patch("enrichment.sources.anidb.anidb_helper.time.time")
async def test_adaptive_rate_limit_logic(mock_time, mock_sleep, helper) -> None:
    helper._adaptive_rate_limit = AniDBHelper._adaptive_rate_limit.__get__(helper)

    mock_time.return_value = 1000.0
    helper.metrics.last_request_time = 995.0
    await helper._adaptive_rate_limit()
    mock_sleep.assert_not_called()

    mock_time.return_value = 1010.0
    helper.metrics.last_request_time = 1010.0
    await helper._adaptive_rate_limit()
    mock_sleep.assert_called_once()
    mock_sleep.reset_mock()

    mock_time.return_value = 1020.0
    helper.metrics.consecutive_failures = 3
    helper.metrics.last_request_time = 1020.0
    await helper._adaptive_rate_limit()
    expected = min(
        helper.error_cooldown_base * (2**helper.metrics.consecutive_failures),
        helper.max_request_interval,
    )
    assert mock_sleep.call_args[0][0] == pytest.approx(expected, abs=0.1)


# =============================================================================
# CONTEXT MANAGER
# =============================================================================


@pytest.mark.asyncio
@patch("enrichment.sources.anidb.anidb_helper.os.getenv")
async def test_context_manager_protocol(mock_getenv) -> None:
    mock_getenv.side_effect = lambda key, default=None: default
    mock_session = AsyncMock()
    mock_session.close = AsyncMock()
    async with AniDBHelper() as ctx_helper:
        ctx_helper.session = mock_session
        assert isinstance(ctx_helper, AniDBHelper)
    mock_session.close.assert_awaited_once()


# =============================================================================
# _fetch_anime
# =============================================================================


@pytest.mark.asyncio
async def test_fetch_anime_returns_tuple_on_success(helper) -> None:
    helper._fetch_xml = AsyncMock(return_value=_MINIMAL_XML)
    anime_dict, anime_model = await helper._fetch_anime(_ANIDB_URL)
    assert anime_dict is not None
    assert anime_model is not None
    assert anime_dict["title"] == "One Piece"
    assert anime_model.id == 69


@pytest.mark.asyncio
async def test_fetch_anime_returns_none_on_invalid_url(helper) -> None:
    anime_dict, anime_model = await helper._fetch_anime(
        "https://anidb.net/character/40"
    )
    assert anime_dict is None
    assert anime_model is None


@pytest.mark.asyncio
async def test_fetch_anime_returns_none_when_xml_unavailable(helper) -> None:
    helper._fetch_xml = AsyncMock(return_value=None)
    anime_dict, anime_model = await helper._fetch_anime(_ANIDB_URL)
    assert anime_dict is None
    assert anime_model is None


@pytest.mark.asyncio
async def test_fetch_anime_returns_none_when_xml_parse_fails(helper) -> None:
    helper._fetch_xml = AsyncMock(return_value="<not-valid-anime-xml>")
    anime_dict, anime_model = await helper._fetch_anime(_ANIDB_URL)
    assert anime_dict is None
    assert anime_model is None


@pytest.mark.asyncio
async def test_fetch_anime_writes_jsonl_to_output_path(helper, tmp_path: Path) -> None:
    helper._fetch_xml = AsyncMock(return_value=_MINIMAL_XML)
    out = tmp_path / "anidb_anime.jsonl"
    await helper._fetch_anime(_ANIDB_URL, output_path=str(out))
    assert out.exists()
    line = json.loads(out.read_text())
    assert line["title"] == "One Piece"


# =============================================================================
# _fetch_episodes
# =============================================================================


@pytest.mark.asyncio
async def test_fetch_episodes_returns_regular_only(helper) -> None:
    from enrichment.sources.anidb.anidb_models import AniDBAnime, AniDBEpisode

    model = AniDBAnime(
        id=69,
        episodes=[
            AniDBEpisode(id=1, episode_type=1, episode_number=1),  # regular → included
            AniDBEpisode(id=2, episode_type=2, episode_number=1),  # special → excluded
            AniDBEpisode(
                id=3, episode_type=1, episode_number="S1"
            ),  # string ep → excluded
        ],
    )
    episodes = await helper._fetch_episodes(model)
    assert len(episodes) == 1
    assert episodes[0]["episode_number"] == 1


@pytest.mark.asyncio
async def test_fetch_episodes_empty_for_empty_model(helper) -> None:
    from enrichment.sources.anidb.anidb_models import AniDBAnime

    model = AniDBAnime(id=69, episodes=[])
    episodes = await helper._fetch_episodes(model)
    assert episodes == []


@pytest.mark.asyncio
async def test_fetch_episodes_writes_jsonl(helper, tmp_path: Path) -> None:
    from enrichment.sources.anidb.anidb_models import AniDBAnime, AniDBEpisode

    model = AniDBAnime(
        id=69,
        episodes=[
            AniDBEpisode(id=1, episode_type=1, episode_number=1),
            AniDBEpisode(id=2, episode_type=1, episode_number=2),
        ],
    )
    out = tmp_path / "anidb_episodes.jsonl"
    episodes = await helper._fetch_episodes(model, output_path=str(out))
    assert len(episodes) == 2
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["episode_number"] == 1


@pytest.mark.asyncio
async def test_fetch_episodes_from_onepiece(helper, onepiece_anime) -> None:
    """All regular One Piece episodes from real fixture are mapped without error."""
    episodes = await helper._fetch_episodes(onepiece_anime)
    assert len(episodes) > 1000
    assert all(isinstance(ep["episode_number"], int) for ep in episodes)


# =============================================================================
# fetch_all
# =============================================================================


@pytest.mark.asyncio
async def test_fetch_all_returns_none_when_no_anidb_url(helper) -> None:
    result = await helper.fetch_all({}, {})
    assert result is None


@pytest.mark.asyncio
async def test_fetch_all_returns_none_when_anime_fetch_fails(helper) -> None:
    helper._fetch_anime = AsyncMock(return_value=(None, None))
    result = await helper.fetch_all({"anidb_url": _ANIDB_URL}, {})
    assert result is None


@pytest.mark.asyncio
async def test_fetch_all_orchestrates_anime_and_episodes(
    helper, tmp_path: Path
) -> None:
    anime_dict = {"title": "One Piece", "sources": [_ANIDB_URL]}
    anime_model = MagicMock()
    anime_model.episodes = []

    helper._fetch_anime = AsyncMock(return_value=(anime_dict, anime_model))
    helper._fetch_episodes = AsyncMock(return_value=[{"episode_number": 1}])
    helper._fetch_characters = AsyncMock(return_value=[])

    result = await helper.fetch_all(
        {"anidb_url": _ANIDB_URL}, {}, temp_dir=str(tmp_path)
    )

    assert result is not None
    assert result["anime"]["title"] == "One Piece"
    assert result["episodes"] == [{"episode_number": 1}]
    assert result["characters"] == []
    assert "extras" in result
    helper._fetch_anime.assert_awaited_once_with(
        _ANIDB_URL, output_path=str(tmp_path / "anidb_anime.jsonl")
    )
    helper._fetch_episodes.assert_awaited_once_with(
        anime_model, output_path=str(tmp_path / "anidb_episodes.jsonl")
    )
    helper._fetch_characters.assert_awaited_once_with(
        anime_model, output_path=str(tmp_path / "anidb_characters.jsonl")
    )


@pytest.mark.asyncio
async def test_fetch_all_continues_when_episodes_fail(helper) -> None:
    anime_dict = {"title": "One Piece"}
    anime_model = MagicMock()

    helper._fetch_anime = AsyncMock(return_value=(anime_dict, anime_model))
    helper._fetch_episodes = AsyncMock(side_effect=Exception("network error"))

    result = await helper.fetch_all({"anidb_url": _ANIDB_URL}, {})

    assert result is not None
    assert result["episodes"] == []


@pytest.mark.asyncio
async def test_fetch_all_skips_episodes_when_false(helper) -> None:
    anime_dict = {"title": "One Piece"}
    anime_model = MagicMock()

    helper._fetch_anime = AsyncMock(return_value=(anime_dict, anime_model))
    helper._fetch_episodes = AsyncMock(return_value=[{"episode_number": 1}])

    result = await helper.fetch_all({"anidb_url": _ANIDB_URL}, {}, fetch_episodes=False)

    assert result is not None
    assert result["episodes"] == []
    helper._fetch_episodes.assert_not_awaited()


@pytest.mark.asyncio
async def test_fetch_all_includes_characters(helper) -> None:
    anime_dict = {"title": "One Piece"}
    anime_model = MagicMock()

    helper._fetch_anime = AsyncMock(return_value=(anime_dict, anime_model))
    helper._fetch_episodes = AsyncMock(return_value=[])
    helper._fetch_characters = AsyncMock(return_value=[{"name": "Luffy"}])

    result = await helper.fetch_all(
        {"anidb_url": _ANIDB_URL}, {}, fetch_characters=True
    )

    assert result is not None
    assert result["characters"] == [{"name": "Luffy"}]
    helper._fetch_characters.assert_awaited_once()


@pytest.mark.asyncio
async def test_fetch_all_skips_characters_when_false(helper) -> None:
    anime_dict = {"title": "One Piece"}
    anime_model = MagicMock()

    helper._fetch_anime = AsyncMock(return_value=(anime_dict, anime_model))
    helper._fetch_episodes = AsyncMock(return_value=[])
    helper._fetch_characters = AsyncMock(return_value=[{"name": "Luffy"}])

    result = await helper.fetch_all(
        {"anidb_url": _ANIDB_URL}, {}, fetch_characters=False
    )

    assert result is not None
    assert result["characters"] == []
    helper._fetch_characters.assert_not_awaited()


# =============================================================================
# CLI — main()
# =============================================================================


@pytest.mark.asyncio
async def test_main_cli_anime_subcommand(tmp_path: Path) -> None:
    out = tmp_path / "output.json"
    with (
        patch("sys.argv", ["anidb_helper", "anime", _ANIDB_URL, str(out)]),
        patch(
            "enrichment.sources.anidb.anidb_helper.AniDBHelper._fetch_xml",
            new=AsyncMock(return_value=_MINIMAL_XML),
        ),
    ):
        from enrichment.sources.anidb.anidb_helper import main

        rc = await main()

    assert rc == 0
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["title"] == "One Piece"


@pytest.mark.asyncio
async def test_main_cli_anime_returns_1_when_no_xml(tmp_path: Path) -> None:
    out = tmp_path / "output.json"
    with (
        patch("sys.argv", ["anidb_helper", "anime", _ANIDB_URL, str(out)]),
        patch(
            "enrichment.sources.anidb.anidb_helper.AniDBHelper._fetch_xml",
            new=AsyncMock(return_value=None),
        ),
    ):
        from enrichment.sources.anidb.anidb_helper import main

        rc = await main()

    assert rc == 1
    assert not out.exists()


@pytest.mark.asyncio
async def test_main_cli_episodes_subcommand(tmp_path: Path) -> None:
    out = tmp_path / "episodes.json"
    with (
        patch("sys.argv", ["anidb_helper", "episodes", _ANIDB_URL, str(out)]),
        patch(
            "enrichment.sources.anidb.anidb_helper.AniDBHelper._fetch_xml",
            new=AsyncMock(return_value=_MINIMAL_XML),
        ),
    ):
        from enrichment.sources.anidb.anidb_helper import main

        rc = await main()

    assert rc == 0
    assert out.exists()
    assert json.loads(out.read_text()) == []


@pytest.mark.asyncio
async def test_main_cli_characters_subcommand(tmp_path: Path) -> None:
    out = tmp_path / "characters.jsonl"
    with (
        patch("sys.argv", ["anidb_helper", "characters", _ANIDB_URL, str(out)]),
        patch(
            "enrichment.sources.anidb.anidb_helper.AniDBHelper._fetch_xml",
            new=AsyncMock(return_value=_MINIMAL_XML),
        ),
    ):
        from enrichment.sources.anidb.anidb_helper import main

        rc = await main()

    assert rc == 0
    # JSONL file may not exist if no characters were fetched (empty XML)
    if out.exists():
        lines = [
            json.loads(line) for line in out.read_text().splitlines() if line.strip()
        ]
        assert isinstance(lines, list)


# =============================================================================
# MISSING COVERAGE — fetch_all character exception
# =============================================================================


@pytest.mark.asyncio
async def test_fetch_all_continues_when_characters_fail(helper) -> None:
    anime_dict = {"title": "One Piece"}
    anime_model = MagicMock()

    helper._fetch_anime = AsyncMock(return_value=(anime_dict, anime_model))
    helper._fetch_episodes = AsyncMock(return_value=[])
    helper._fetch_characters = AsyncMock(side_effect=Exception("browser died"))

    result = await helper.fetch_all({"anidb_url": _ANIDB_URL}, {})

    assert result is not None
    assert result["characters"] == []


# =============================================================================
# MISSING COVERAGE — _fetch_characters with output_path + id-less chars
# =============================================================================


@pytest.mark.asyncio
async def test_fetch_characters_with_output_path_and_id_less_chars(
    helper, tmp_path: Path
) -> None:
    from enrichment.sources.anidb.anidb_models import AniDBAnime, AniDBCharacter

    model = AniDBAnime(
        id=69,
        characters=[
            AniDBCharacter(id=474, name="Luffy"),
            AniDBCharacter(id=None, name="Unknown"),
        ],
    )
    out = tmp_path / "chars.jsonl"

    with patch(
        "enrichment.sources.anidb.anidb_helper.fetch_anidb_characters",
        return_value=_async_gen([(474, None)]),
    ):
        chars = await helper._fetch_characters(model, output_path=str(out))

    assert len(chars) == 2
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2


# =============================================================================
# MISSING COVERAGE — _fetch_xml exception
# =============================================================================


@pytest.mark.asyncio
async def test_fetch_xml_exception_returns_none(helper) -> None:
    with patch.object(
        helper, "_make_request", new_callable=AsyncMock, side_effect=Exception("boom")
    ):
        result = await helper._fetch_xml(69)
    assert result is None


# =============================================================================
# MISSING COVERAGE — _adaptive_rate_limit is_retry multiplier
# =============================================================================


@pytest.mark.asyncio
@patch("enrichment.sources.anidb.anidb_helper.asyncio.sleep", new_callable=AsyncMock)
@patch("enrichment.sources.anidb.anidb_helper.time.time")
async def test_adaptive_rate_limit_is_retry_increases_interval(
    mock_time, mock_sleep, helper
) -> None:
    helper._adaptive_rate_limit = AniDBHelper._adaptive_rate_limit.__get__(helper)
    mock_time.return_value = 1010.0
    helper.metrics.last_request_time = 1010.0

    await helper._adaptive_rate_limit(is_retry=True)

    assert mock_sleep.called
    wait = mock_sleep.call_args[0][0]
    assert wait == pytest.approx(helper.min_request_interval * 1.5, abs=0.1)


# =============================================================================
# MISSING COVERAGE — _make_request via lock
# =============================================================================


@pytest.mark.asyncio
async def test_make_request_delegates_via_lock(helper) -> None:
    helper._make_request_with_retry = AsyncMock(return_value="<anime/>")
    result = await helper._make_request({"aid": 69})
    assert result == "<anime/>"
    helper._make_request_with_retry.assert_awaited_once_with({"aid": 69})


# =============================================================================
# MISSING COVERAGE — _make_request_with_retry exception paths
# =============================================================================


@pytest.mark.asyncio
@patch("enrichment.sources.anidb.anidb_helper.asyncio.sleep", new_callable=AsyncMock)
async def test_make_request_with_retry_exception_retries_then_raises(
    mock_sleep, helper
) -> None:
    from enrichment.sources.base.exceptions import ServiceNetworkError

    helper.max_retries = 1
    helper._ensure_session_health = AsyncMock()
    helper._make_single_request = AsyncMock(
        side_effect=[Exception("first"), Exception("second")]
    )

    with pytest.raises(ServiceNetworkError):
        await helper._make_request_with_retry({"aid": 69})

    assert helper.metrics.total_requests == 2
    assert helper.metrics.failed_requests == 2
    assert mock_sleep.call_count == 1


# =============================================================================
# MISSING COVERAGE — _make_single_request no session
# =============================================================================


@pytest.mark.asyncio
async def test_make_single_request_no_session_raises(helper) -> None:
    helper.session = None
    with pytest.raises(RuntimeError, match="Session not initialized"):
        await helper._make_single_request({"aid": 69}, attempt=0)


# =============================================================================
# MISSING COVERAGE — close() session.close raises
# =============================================================================


@pytest.mark.asyncio
async def test_close_exception_swallowed(helper) -> None:
    mock_session = MagicMock()
    mock_session.close = AsyncMock(side_effect=Exception("close failed"))
    helper.session = mock_session

    await helper.close()

    assert helper.session is None


@pytest.mark.asyncio
async def test_main_cli_all_subcommand(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    with (
        patch("sys.argv", ["anidb_helper", "all", _ANIDB_URL, str(out_dir)]),
        patch(
            "enrichment.sources.anidb.anidb_helper.AniDBHelper._fetch_xml",
            new=AsyncMock(return_value=_MINIMAL_XML),
        ),
    ):
        from enrichment.sources.anidb.anidb_helper import main

        rc = await main()

    assert rc == 0
    assert (out_dir / "anidb_anime.json").exists()
    assert (out_dir / "anidb_episodes.json").exists()
    assert (out_dir / "anidb_characters.json").exists()
    anime = json.loads((out_dir / "anidb_anime.json").read_text())
    assert anime["title"] == "One Piece"
