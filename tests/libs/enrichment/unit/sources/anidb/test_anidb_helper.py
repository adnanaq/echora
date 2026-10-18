import asyncio
import gzip
import json
import sys
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from unittest.mock import create_autospec, patch

import aiohttp
import pytest
from common.config import EnrichmentConfig
from enrichment.sources.anidb import anidb_helper
from enrichment.sources.anidb.anidb_helper import (
    AniDBHelper,
    _state,
    main,
    reset_client_state,
)
from enrichment.sources.anidb.anidb_models import (
    AniDBAnime,
    AniDBCharacter,
    AniDBEpisode,
)
from enrichment.sources.base.exceptions import (
    ServiceBlockedError,
    ServiceNetworkError,
    ServiceParseError,
)
from http_cache import HTTPCacheManager
from http_cache.aiohttp_adapter import CachedAiohttpSession

pytestmark = pytest.mark.asyncio

_ANIDB_URL = "https://anidb.net/anime/69"
_MINIMAL_XML = (
    '<anime id="69">'
    "<type>TV Series</type>"
    "<episodecount>0</episodecount>"
    "<titles>"
    '<title type="main" xml:lang="x-jat">One Piece</title>'
    "</titles>"
    "</anime>"
)
_SHORTEST_GAP = 0.001


@pytest.fixture(autouse=True)
def fresh_client_state() -> Iterator[None]:
    reset_client_state()
    yield
    reset_client_state()


@pytest.fixture
def response() -> aiohttp.ClientResponse:
    response = create_autospec(aiohttp.ClientResponse, instance=True)
    response.status = 200
    response.read.return_value = _MINIMAL_XML.encode()
    return response


@pytest.fixture
def session(response) -> CachedAiohttpSession:
    session = create_autospec(CachedAiohttpSession, instance=True)
    session.get.return_value.__aenter__.return_value = response
    return session


@pytest.fixture
def cache(session) -> HTTPCacheManager:
    cache = create_autospec(HTTPCacheManager, instance=True)
    cache.get_aiohttp_session.return_value = session
    return cache


@pytest.fixture
def helper(cache) -> AniDBHelper:
    return _helper(cache)


@pytest.fixture
def waits() -> Iterator[list[float]]:
    with patch.object(anidb_helper.asyncio, "sleep", autospec=True) as sleep:
        yield sleep


@pytest.fixture
def character_pages() -> Iterator:
    async def no_pages(char_ids: list[int]) -> AsyncIterator[tuple[int, None]]:
        for char_id in char_ids:
            yield char_id, None

    with patch.object(
        anidb_helper, "fetch_anidb_characters", autospec=True, side_effect=no_pages
    ) as fetch_characters:
        yield fetch_characters


def _helper(cache: HTTPCacheManager, **settings: object) -> AniDBHelper:
    config = EnrichmentConfig(
        anidb_min_request_interval=_SHORTEST_GAP,
        anidb_error_cooldown_base=0,
        **settings,
    )
    return AniDBHelper(config, cache)


def _helper_from(helper: AniDBHelper) -> AniDBHelper:
    return AniDBHelper(
        EnrichmentConfig(anidb_min_request_interval=_SHORTEST_GAP), helper._cache
    )


def _waited_seconds(waits) -> list[float]:
    return [call.args[0] for call in waits.await_args_list]


def _sent_params(session) -> list[dict]:
    return [call.kwargs["params"] for call in session.get.call_args_list]


def test_anidb_helper_settings_come_from_config(cache) -> None:
    helper = AniDBHelper(
        EnrichmentConfig(anidb_client="animeenrichment", anidb_clientver="1.0"), cache
    )

    assert (helper.client_name, helper.client_version) == ("animeenrichment", "1.0")
    assert helper._ban_remaining() == 0


async def test_make_single_request_ok_response_returns_body(helper, session) -> None:
    await helper._ensure_session_health()

    assert await helper._make_single_request({"aid": 69}, 0) == _MINIMAL_XML


async def test_make_single_request_sends_client_identity(helper, session) -> None:
    await helper._ensure_session_health()

    await helper._make_single_request({"request": "anime", "aid": 69}, 0)

    assert _sent_params(session) == [
        {
            "request": "anime",
            "aid": 69,
            "client": helper.client_name,
            "clientver": helper.client_version,
            "protover": helper.protocol_version,
        }
    ]


async def test_make_single_request_gzipped_body_returns_text(helper, response) -> None:
    response.read.return_value = gzip.compress(b"<anime id='2'></anime>")
    await helper._ensure_session_health()

    assert await helper._make_single_request({"aid": 2}, 0) == "<anime id='2'></anime>"


@pytest.mark.parametrize("status", [503, 404])
async def test_make_single_request_error_status_raises_network_error(
    helper, response, status: int
) -> None:
    response.status = status
    await helper._ensure_session_health()

    with pytest.raises(ServiceNetworkError, match=f"HTTP {status}"):
        await helper._make_single_request({"aid": 69}, 0)


@pytest.mark.parametrize(
    ("status", "body"), [(555, b""), (200, b'<error code="500">banned</error>')]
)
async def test_make_single_request_ban_raises_blocked_and_latches_ban(
    helper, response, status: int, body: bytes
) -> None:
    response.status = status
    response.read.return_value = body
    await helper._ensure_session_health()

    with pytest.raises(ServiceBlockedError):
        await helper._make_single_request({"aid": 69}, 0)

    assert helper._ban_remaining() > 0


async def test_make_single_request_anime_not_found_returns_none_without_ban(
    helper, response
) -> None:
    response.read.return_value = b"<error>Anime not found</error>"
    await helper._ensure_session_health()

    assert await helper._make_single_request({"aid": 69}, 0) is None
    assert helper._ban_remaining() == 0


async def test_make_single_request_other_error_body_raises_network_error_without_ban(
    helper, response
) -> None:
    response.read.return_value = b"<error>busy</error>"
    await helper._ensure_session_health()

    with pytest.raises(ServiceNetworkError, match="busy"):
        await helper._make_single_request({"aid": 69}, 0)
    assert helper._ban_remaining() == 0


async def test_make_single_request_without_session_raises_runtime_error(
    helper,
) -> None:
    with pytest.raises(RuntimeError, match="Session not initialized"):
        await helper._make_single_request({"aid": 69}, 0)


async def test_make_single_request_cache_only_miss_raises_blocked_error(
    helper, session, response
) -> None:
    response.status = 504
    await helper._ensure_session_health()

    with pytest.raises(ServiceBlockedError, match="not cached"):
        await helper._make_single_request({"aid": 69}, 0, cache_only=True)
    assert session.get.call_args.kwargs["headers"] == {
        "Cache-Control": "only-if-cached"
    }


@pytest.mark.parametrize(
    ("content", "expected"),
    [("你好".encode(), "你好"), ("é".encode("latin-1"), "é"), (b"\xff\xfe", "ÿþ")],
)
def test_decode_content_bytes_returns_text(
    helper, content: bytes, expected: str
) -> None:
    assert helper._decode_content(content) == expected


async def test_make_request_with_retry_ban_raises_without_retrying(
    helper, session, response, waits
) -> None:
    response.status = 555

    with pytest.raises(ServiceBlockedError):
        await helper._make_request_with_retry({"aid": 123})

    assert session.get.call_count == 1


async def test_make_request_with_retry_during_ban_reads_cache_only(
    helper, session
) -> None:
    helper._record_ban()

    assert await _helper_from(helper)._make_request_with_retry({"aid": 456}) == (
        _MINIMAL_XML
    )
    assert session.get.call_args.kwargs["headers"] == {
        "Cache-Control": "only-if-cached"
    }


async def test_make_request_with_retry_expired_ban_contacts_anidb(
    helper, session
) -> None:
    _state.ban_lifts_at = time.time() - 1

    assert await helper._make_request_with_retry({"aid": 69}) == _MINIMAL_XML
    assert "headers" not in session.get.call_args.kwargs


async def test_make_request_with_retry_anime_not_found_returns_none_without_retry(
    helper, session, response, waits
) -> None:
    response.read.return_value = b"<error>Anime not found</error>"

    assert await helper._make_request_with_retry({"aid": 123}) is None

    assert session.get.call_count == 1
    assert helper._ban_remaining() == 0


async def test_make_request_with_retry_later_success_returns_body(
    cache, response, waits
) -> None:
    helper = _helper(cache, anidb_max_retries=2)
    response.read.side_effect = [
        b"<error>busy</error>",
        b"<error>busy</error>",
        b"<anime/>",
    ]

    assert await helper._make_request_with_retry({"aid": 1}) == "<anime/>"
    assert len(_waited_seconds(waits)) >= 2


async def test_make_request_with_retry_repeated_failure_raises_network_error(
    cache, session, waits
) -> None:
    helper = _helper(cache, anidb_max_retries=1)
    session.get.side_effect = aiohttp.ClientError("connection refused")

    with pytest.raises(ServiceNetworkError):
        await helper._make_request_with_retry({"aid": 69})

    assert session.get.call_count == 2


async def test_make_request_with_retry_failures_carry_to_next_helper(
    helper, response, waits
) -> None:
    response.read.return_value = b"<error>busy</error>"

    with pytest.raises(ServiceNetworkError):
        await helper._make_request({"aid": 1})

    assert _state.consecutive_failures == helper.max_retries + 1


async def test_make_request_concurrent_requests_queue_one_at_a_time(
    cache, response
) -> None:
    in_flight = 0
    overlapped = False

    async def read() -> bytes:
        nonlocal in_flight, overlapped
        in_flight += 1
        overlapped = overlapped or in_flight > 1
        await asyncio.sleep(0)
        in_flight -= 1
        return b"<anime/>"

    response.read.side_effect = read
    await asyncio.gather(
        *(_helper(cache)._make_request({"aid": aid}) for aid in range(5))
    )

    assert not overlapped


async def test_make_request_successive_helpers_keep_shortest_gap(cache, waits) -> None:
    config = EnrichmentConfig()

    for aid in range(3):
        await AniDBHelper(config, cache)._make_request({"aid": aid})

    gaps = _waited_seconds(waits)
    assert len(gaps) == 2
    assert all(
        gap == pytest.approx(config.anidb_min_request_interval, abs=0.5) for gap in gaps
    )


async def test_adaptive_rate_limit_recent_request_waits_shortest_gap(
    cache, waits
) -> None:
    helper = AniDBHelper(EnrichmentConfig(), cache)
    _state.last_request_at = time.time()

    await helper._adaptive_rate_limit()

    assert _waited_seconds(waits) == [
        pytest.approx(helper.min_request_interval, abs=0.1)
    ]


async def test_adaptive_rate_limit_old_request_does_not_wait(cache, waits) -> None:
    helper = AniDBHelper(EnrichmentConfig(), cache)
    _state.last_request_at = time.time() - helper.max_request_interval

    await helper._adaptive_rate_limit()

    assert _waited_seconds(waits) == []


async def test_adaptive_rate_limit_retry_waits_half_again(cache, waits) -> None:
    helper = AniDBHelper(EnrichmentConfig(), cache)
    _state.last_request_at = time.time()

    await helper._adaptive_rate_limit(is_retry=True)

    assert _waited_seconds(waits) == [
        pytest.approx(helper.min_request_interval * 1.5, abs=0.1)
    ]


async def test_adaptive_rate_limit_after_failures_waits_error_cooldown(
    cache, waits
) -> None:
    helper = AniDBHelper(EnrichmentConfig(), cache)
    _state.consecutive_failures = 3
    _state.last_request_at = time.time()

    await helper._adaptive_rate_limit()

    expected = min(helper.error_cooldown_base * 2**3, helper.max_request_interval)
    assert _waited_seconds(waits) == [pytest.approx(expected, abs=0.1)]


async def test_ensure_session_health_without_session_opens_one(
    helper, cache, session
) -> None:
    await helper._ensure_session_health()

    assert helper.session is session
    assert cache.get_aiohttp_session.call_args.args == ("anidb",)


async def test_ensure_session_health_expired_session_replaces_it(
    helper, cache, session
) -> None:
    await helper._ensure_session_health()
    old_session = helper.session
    replacement = create_autospec(CachedAiohttpSession, instance=True)
    cache.get_aiohttp_session.return_value = replacement
    helper._session_created_at = time.time() - helper._session_max_age - 1

    await helper._ensure_session_health()

    assert helper.session is replacement
    assert old_session.close.await_count == 1


async def test_ensure_session_health_fresh_session_keeps_it(helper, session) -> None:
    await helper._ensure_session_health()

    await helper._ensure_session_health()

    assert helper.session is session
    assert session.close.await_count == 0


async def test_close_open_session_closes_and_forgets_it(helper, session) -> None:
    await helper._ensure_session_health()

    await helper.close()

    assert session.close.await_count == 1
    assert helper.session is None


async def test_close_failing_close_still_forgets_session(helper, session) -> None:
    session.close.side_effect = aiohttp.ClientError("close failed")
    await helper._ensure_session_health()

    await helper.close()

    assert helper.session is None


async def test_anidb_helper_context_manager_closes_session(cache, session) -> None:
    async with _helper(cache) as helper:
        await helper._ensure_session_health()

    assert session.close.await_count == 1


async def test_fetch_xml_anime_request_returns_body(helper, session) -> None:
    assert await helper._fetch_xml(69) == _MINIMAL_XML
    assert _sent_params(session)[0]["aid"] == 69


async def test_fetch_xml_network_failure_raises_network_error(
    cache, session, waits
) -> None:
    session.get.side_effect = aiohttp.ClientError("connection refused")

    with pytest.raises(ServiceNetworkError):
        await _helper(cache, anidb_max_retries=0)._fetch_xml(69)


async def test_fetch_anime_anime_page_returns_canonical_and_model(helper) -> None:
    anime, model = await helper._fetch_anime(_ANIDB_URL)

    assert anime is not None
    assert model is not None
    assert (anime["title"], model.id) == ("One Piece", 69)


async def test_fetch_anime_address_without_anime_id_returns_nothing(
    helper, session
) -> None:
    assert await helper._fetch_anime("https://anidb.net/character/40") == (None, None)
    assert session.get.call_count == 0


async def test_fetch_anime_anime_not_found_returns_nothing(cache, response) -> None:
    response.read.return_value = b"<error>Anime not found</error>"

    assert await _helper(cache)._fetch_anime(_ANIDB_URL) == (None, None)


async def test_fetch_anime_error_body_raises_network_error(
    cache, response, waits
) -> None:
    response.read.return_value = b"<error>busy</error>"

    with pytest.raises(ServiceNetworkError):
        await _helper(cache, anidb_max_retries=0)._fetch_anime(_ANIDB_URL)


async def test_fetch_anime_unreadable_xml_raises_parse_error(cache, response) -> None:
    response.read.return_value = b"<not-valid-anime-xml>"

    with pytest.raises(ServiceParseError):
        await _helper(cache)._fetch_anime(_ANIDB_URL)


async def test_fetch_anime_output_path_appends_canonical_anime(
    helper, tmp_path: Path
) -> None:
    output = tmp_path / "anidb_anime.jsonl"

    await helper._fetch_anime(_ANIDB_URL, output_path=str(output))

    assert json.loads(output.read_text())["title"] == "One Piece"


async def test_fetch_episodes_mixed_episode_types_returns_regular_only(
    helper,
) -> None:
    model = AniDBAnime(
        id=69,
        episodes=[
            AniDBEpisode(id=1, episode_type=1, episode_number=1),
            AniDBEpisode(id=2, episode_type=2, episode_number=1),
            AniDBEpisode(id=3, episode_type=1, episode_number="S1"),
        ],
    )

    episodes = await helper._fetch_episodes(model)

    assert [episode["episode_number"] for episode in episodes] == [1]


async def test_fetch_episodes_without_episodes_returns_empty_list(helper) -> None:
    assert await helper._fetch_episodes(AniDBAnime(id=69, episodes=[])) == []


async def test_fetch_episodes_output_path_appends_each_episode(
    helper, tmp_path: Path
) -> None:
    model = AniDBAnime(
        id=69,
        episodes=[
            AniDBEpisode(id=1, episode_type=1, episode_number=1),
            AniDBEpisode(id=2, episode_type=1, episode_number=2),
        ],
    )
    output = tmp_path / "anidb_episodes.jsonl"

    await helper._fetch_episodes(model, output_path=str(output))

    assert [
        json.loads(line)["episode_number"] for line in output.read_text().splitlines()
    ] == [1, 2]


async def test_fetch_episodes_one_piece_maps_every_regular_episode(
    helper, onepiece_anime
) -> None:
    episodes = await helper._fetch_episodes(onepiece_anime)

    assert len(episodes) > 1000
    assert all(isinstance(episode["episode_number"], int) for episode in episodes)


async def test_fetch_characters_output_path_appends_listed_and_unlisted_characters(
    helper, character_pages, tmp_path: Path
) -> None:
    model = AniDBAnime(
        id=69,
        characters=[
            AniDBCharacter(id=474, name="Luffy"),
            AniDBCharacter(id=None, name="Unknown"),
        ],
    )
    output = tmp_path / "chars.jsonl"

    characters = await helper._fetch_characters(model, output_path=str(output))

    assert len(characters) == 2
    assert len(output.read_text().splitlines()) == 2


async def test_fetch_all_without_anidb_url_returns_none(helper, session) -> None:
    assert await helper.fetch_all({}, {}) is None
    assert session.get.call_count == 0


async def test_fetch_all_anime_not_found_returns_none(cache, response) -> None:
    response.read.return_value = b"<error>Anime not found</error>"

    assert await _helper(cache).fetch_all({"anidb_url": _ANIDB_URL}, {}) is None


async def test_fetch_all_anime_request_failure_raises_network_error(
    cache, response, waits
) -> None:
    response.read.return_value = b"<error>busy</error>"

    with pytest.raises(ServiceNetworkError):
        await _helper(cache, anidb_max_retries=0).fetch_all(
            {"anidb_url": _ANIDB_URL}, {}
        )


async def test_fetch_all_one_piece_returns_anime_episodes_and_characters(
    helper, response, character_pages, onepiece_xml, tmp_path: Path
) -> None:
    response.read.return_value = onepiece_xml.encode()

    result = await helper.fetch_all(
        {"anidb_url": _ANIDB_URL}, {}, temp_dir=str(tmp_path)
    )

    assert result is not None
    assert result["anime"]["title"] == "One Piece"
    assert len(result["episodes"]) > 1000
    assert len(result["characters"]) > 0
    assert {path.name for path in tmp_path.iterdir()} == {
        "anidb_anime.jsonl",
        "anidb_episodes.jsonl",
        "anidb_characters.jsonl",
    }


@pytest.mark.parametrize(
    ("skipped", "kept"), [("episodes", "characters"), ("characters", "episodes")]
)
async def test_fetch_all_skipped_part_returns_empty_list(
    helper, response, character_pages, onepiece_xml, skipped: str, kept: str
) -> None:
    response.read.return_value = onepiece_xml.encode()

    result = await helper.fetch_all(
        {"anidb_url": _ANIDB_URL}, {}, **{f"fetch_{skipped}": False}
    )

    assert result is not None
    assert result[skipped] == []
    assert result[kept]


async def test_fetch_all_failing_episodes_returns_anime_without_episodes(
    helper, response, character_pages, onepiece_xml
) -> None:
    response.read.return_value = onepiece_xml.encode()

    with patch.object(
        anidb_helper, "episode_from_anidb", autospec=True, side_effect=ValueError("bad")
    ):
        result = await helper.fetch_all({"anidb_url": _ANIDB_URL}, {})

    assert result is not None
    assert result["episodes"] == []
    assert result["characters"]


async def test_fetch_all_failing_characters_returns_anime_without_characters(
    helper, response, character_pages, onepiece_xml
) -> None:
    response.read.return_value = onepiece_xml.encode()
    character_pages.side_effect = RuntimeError("browser died")

    result = await helper.fetch_all({"anidb_url": _ANIDB_URL}, {})

    assert result is not None
    assert result["characters"] == []
    assert result["episodes"]


@pytest.fixture
def fetched_xml() -> Iterator:
    with patch.object(AniDBHelper, "_fetch_xml", autospec=True) as fetch_xml:
        yield fetch_xml


async def test_main_anime_command_writes_anime(fetched_xml, tmp_path: Path) -> None:
    fetched_xml.return_value = _MINIMAL_XML
    output = tmp_path / "output.json"

    with patch.object(sys, "argv", ["anidb_helper", "anime", _ANIDB_URL, str(output)]):
        assert await main() == 0

    assert json.loads(output.read_text())["title"] == "One Piece"


async def test_main_anime_command_without_response_returns_one(
    fetched_xml, tmp_path: Path
) -> None:
    fetched_xml.return_value = None
    output = tmp_path / "output.json"

    with patch.object(sys, "argv", ["anidb_helper", "anime", _ANIDB_URL, str(output)]):
        assert await main() == 1

    assert not output.exists()


async def test_main_episodes_command_writes_episodes(
    fetched_xml, tmp_path: Path
) -> None:
    fetched_xml.return_value = _MINIMAL_XML
    output = tmp_path / "episodes.json"

    with patch.object(
        sys, "argv", ["anidb_helper", "episodes", _ANIDB_URL, str(output)]
    ):
        assert await main() == 0

    assert json.loads(output.read_text()) == []


async def test_main_characters_command_returns_zero(
    fetched_xml, character_pages, tmp_path: Path
) -> None:
    fetched_xml.return_value = _MINIMAL_XML
    output = tmp_path / "characters.jsonl"

    with patch.object(
        sys, "argv", ["anidb_helper", "characters", _ANIDB_URL, str(output)]
    ):
        assert await main() == 0


async def test_main_all_command_writes_every_part(
    fetched_xml, character_pages, tmp_path: Path
) -> None:
    fetched_xml.return_value = _MINIMAL_XML
    output_dir = tmp_path / "out"

    with patch.object(
        sys, "argv", ["anidb_helper", "all", _ANIDB_URL, str(output_dir)]
    ):
        assert await main() == 0

    assert {path.name for path in output_dir.iterdir()} >= {
        "anidb_anime.json",
        "anidb_episodes.json",
        "anidb_characters.json",
    }
    assert json.loads((output_dir / "anidb_anime.json").read_text())["title"] == (
        "One Piece"
    )
