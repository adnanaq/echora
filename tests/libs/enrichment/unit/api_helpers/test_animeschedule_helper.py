"""Unit tests for animeschedule_helper.py — 100% coverage including edge cases."""

import json
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────────

MINIMAL_RAW = {"id": "abc1", "title": "Test Anime", "route": "test-anime"}

MAPPED_RESULT = {"title": "Test Anime", "sources": ["https://animeschedule.net/anime/test-anime"]}


def _make_session(response: MagicMock) -> AsyncMock:
    """Build an aiohttp-style async session mock."""
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=response)
    cm.__aexit__ = AsyncMock(return_value=None)
    session = AsyncMock()
    session.get = MagicMock(return_value=cm)
    return session


def _ok_response(payload: dict) -> AsyncMock:
    r = AsyncMock()
    r.raise_for_status = MagicMock()
    r.json = AsyncMock(return_value=payload)
    return r


# ── _match_by_sources ─────────────────────────────────────────────────────────


def test_match_by_sources_empty_candidates():
    from enrichment.api_helpers.animeschedule_helper import _match_by_sources

    assert _match_by_sources([], ["https://myanimelist.net/anime/21"]) is None


def test_match_by_sources_no_match():
    from enrichment.api_helpers.animeschedule_helper import _match_by_sources

    candidates = [{"websites": {"mal": "myanimelist.net/anime/99"}}]
    assert _match_by_sources(candidates, ["https://myanimelist.net/anime/21"]) is None


def test_match_by_sources_match_mal():
    from enrichment.api_helpers.animeschedule_helper import _match_by_sources

    candidate = {"id": "1", "websites": {"mal": "myanimelist.net/anime/21"}}
    result = _match_by_sources([candidate], ["https://myanimelist.net/anime/21"])
    assert result is candidate


def test_match_by_sources_match_anilist():
    from enrichment.api_helpers.animeschedule_helper import _match_by_sources

    candidate = {"id": "1", "websites": {"aniList": "anilist.co/anime/21"}}
    result = _match_by_sources([candidate], ["https://anilist.co/anime/21"])
    assert result is candidate


def test_match_by_sources_match_kitsu():
    from enrichment.api_helpers.animeschedule_helper import _match_by_sources

    candidate = {"id": "1", "websites": {"kitsu": "kitsu.io/anime/one-piece"}}
    result = _match_by_sources([candidate], ["https://kitsu.io/anime/one-piece"])
    assert result is candidate


def test_match_by_sources_match_animeplanet():
    from enrichment.api_helpers.animeschedule_helper import _match_by_sources

    candidate = {"id": "1", "websites": {"animePlanet": "anime-planet.com/anime/one-piece"}}
    result = _match_by_sources([candidate], ["https://anime-planet.com/anime/one-piece"])
    assert result is candidate


def test_match_by_sources_match_anidb():
    from enrichment.api_helpers.animeschedule_helper import _match_by_sources

    candidate = {"id": "1", "websites": {"anidb": "anidb.net/anime/69"}}
    result = _match_by_sources([candidate], ["https://anidb.net/anime/69"])
    assert result is candidate


def test_match_by_sources_http_scheme_stripped():
    from enrichment.api_helpers.animeschedule_helper import _match_by_sources

    candidate = {"id": "1", "websites": {"mal": "myanimelist.net/anime/21"}}
    result = _match_by_sources([candidate], ["http://myanimelist.net/anime/21"])
    assert result is candidate


def test_match_by_sources_our_source_is_prefix_of_partial():
    """Our source is ID-only; AS partial has trailing slug."""
    from enrichment.api_helpers.animeschedule_helper import _match_by_sources

    candidate = {"id": "1", "websites": {"mal": "myanimelist.net/anime/21/One_Piece"}}
    result = _match_by_sources([candidate], ["https://myanimelist.net/anime/21"])
    assert result is candidate


def test_match_by_sources_partial_is_prefix_of_our_source():
    """AS partial is shorter; our source has the full slug."""
    from enrichment.api_helpers.animeschedule_helper import _match_by_sources

    candidate = {"id": "1", "websites": {"mal": "myanimelist.net/anime/21"}}
    result = _match_by_sources([candidate], ["https://myanimelist.net/anime/21/One_Piece"])
    assert result is candidate


def test_match_by_sources_picks_first_matching_candidate():
    from enrichment.api_helpers.animeschedule_helper import _match_by_sources

    c1 = {"id": "1", "websites": {"mal": "myanimelist.net/anime/99"}}
    c2 = {"id": "2", "websites": {"mal": "myanimelist.net/anime/21"}}
    result = _match_by_sources([c1, c2], ["https://myanimelist.net/anime/21"])
    assert result is c2


def test_match_by_sources_ignores_official_and_streams_keys():
    """official and streams are not cross-source keys and must not be checked."""
    from enrichment.api_helpers.animeschedule_helper import _match_by_sources

    candidate = {
        "id": "1",
        "websites": {
            "official": "myanimelist.net/anime/21",
            "streams": [{"platform": "mal", "url": "myanimelist.net/anime/21"}],
        },
    }
    assert _match_by_sources([candidate], ["https://myanimelist.net/anime/21"]) is None


def test_match_by_sources_skips_non_string_website_values():
    from enrichment.api_helpers.animeschedule_helper import _match_by_sources

    candidate = {"id": "1", "websites": {"mal": None, "aniList": 12345}}
    assert _match_by_sources([candidate], ["https://myanimelist.net/anime/21"]) is None


def test_match_by_sources_skips_empty_string_website_values():
    from enrichment.api_helpers.animeschedule_helper import _match_by_sources

    candidate = {"id": "1", "websites": {"mal": ""}}
    assert _match_by_sources([candidate], ["https://myanimelist.net/anime/21"]) is None


def test_match_by_sources_empty_sources_list():
    """Empty sources set means nothing can match — all candidates skipped."""
    from enrichment.api_helpers.animeschedule_helper import _match_by_sources

    candidate = {"id": "1", "websites": {"mal": "myanimelist.net/anime/21"}}
    assert _match_by_sources([candidate], []) is None


def test_match_by_sources_filters_empty_source_strings():
    """Empty strings in sources list are ignored during normalization."""
    from enrichment.api_helpers.animeschedule_helper import _match_by_sources

    candidate = {"id": "1", "websites": {"mal": "myanimelist.net/anime/21"}}
    # only empty strings — normalized set is empty, nothing matches
    assert _match_by_sources([candidate], ["", ""]) is None


# ── fetch_all ──────────────────────────────────────────────────


@pytest.mark.asyncio
@patch("enrichment.api_helpers.animeschedule_helper.anime_from_animeschedule", return_value=MAPPED_RESULT)
@patch("enrichment.api_helpers.animeschedule_helper._cache_manager")
async def test_fetch_success_no_sources(mock_cache, mock_mapper):
    from enrichment.api_helpers.animeschedule_helper import fetch_all

    session = _make_session(_ok_response({"anime": [MINIMAL_RAW]}))
    mock_cache.get_aiohttp_session.return_value = session

    result = await fetch_all("Test Anime")

    assert result == MAPPED_RESULT
    session.close.assert_awaited_once()


@pytest.mark.asyncio
@patch("enrichment.api_helpers.animeschedule_helper.anime_from_animeschedule", return_value=MAPPED_RESULT)
@patch("enrichment.api_helpers.animeschedule_helper._cache_manager")
async def test_fetch_success_with_matching_sources(mock_cache, mock_mapper):
    from enrichment.api_helpers.animeschedule_helper import fetch_all

    raw = {**MINIMAL_RAW, "websites": {"mal": "myanimelist.net/anime/21"}}
    session = _make_session(_ok_response({"anime": [raw]}))
    mock_cache.get_aiohttp_session.return_value = session

    result = await fetch_all(
        "Test Anime", sources=["https://myanimelist.net/anime/21"]
    )

    assert result == MAPPED_RESULT
    session.close.assert_awaited_once()


@pytest.mark.asyncio
@patch("enrichment.api_helpers.animeschedule_helper._cache_manager")
async def test_fetch_sources_provided_no_match_returns_none(mock_cache):
    from enrichment.api_helpers.animeschedule_helper import fetch_all

    raw = {**MINIMAL_RAW, "websites": {"mal": "myanimelist.net/anime/99"}}
    session = _make_session(_ok_response({"anime": [raw]}))
    mock_cache.get_aiohttp_session.return_value = session

    result = await fetch_all(
        "Test Anime", sources=["https://myanimelist.net/anime/21"]
    )

    assert result is None
    session.close.assert_awaited_once()


@pytest.mark.asyncio
@patch("enrichment.api_helpers.animeschedule_helper._cache_manager")
async def test_fetch_empty_anime_list_returns_none(mock_cache):
    from enrichment.api_helpers.animeschedule_helper import fetch_all

    session = _make_session(_ok_response({"anime": []}))
    mock_cache.get_aiohttp_session.return_value = session

    assert await fetch_all("Test Anime") is None
    session.close.assert_awaited_once()


@pytest.mark.asyncio
@patch("enrichment.api_helpers.animeschedule_helper._cache_manager")
async def test_fetch_null_response_returns_none(mock_cache):
    from enrichment.api_helpers.animeschedule_helper import fetch_all

    session = _make_session(_ok_response(None))
    mock_cache.get_aiohttp_session.return_value = session

    assert await fetch_all("Test Anime") is None
    session.close.assert_awaited_once()


@pytest.mark.asyncio
@patch("enrichment.api_helpers.animeschedule_helper.anime_from_animeschedule", return_value=MAPPED_RESULT)
@patch("enrichment.api_helpers.animeschedule_helper._cache_manager")
async def test_fetch_writes_jsonl_output(mock_cache, mock_mapper):
    from enrichment.api_helpers.animeschedule_helper import fetch_all

    session = _make_session(_ok_response({"anime": [MINIMAL_RAW]}))
    mock_cache.get_aiohttp_session.return_value = session

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        tmp_path = tmp.name

    await fetch_all("Test Anime", output_path=tmp_path)

    with open(tmp_path, encoding="utf-8") as f:
        lines = f.readlines()

    assert len(lines) == 1
    assert json.loads(lines[0]) == MAPPED_RESULT


@pytest.mark.asyncio
@patch("enrichment.api_helpers.animeschedule_helper._cache_manager")
async def test_fetch_client_error_returns_none(mock_cache):
    import aiohttp
    from enrichment.api_helpers.animeschedule_helper import fetch_all

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(side_effect=aiohttp.ClientError("conn error"))
    cm.__aexit__ = AsyncMock(return_value=None)
    session = AsyncMock()
    session.get = MagicMock(return_value=cm)
    mock_cache.get_aiohttp_session.return_value = session

    assert await fetch_all("Test Anime") is None
    session.close.assert_awaited_once()


@pytest.mark.asyncio
@patch("enrichment.api_helpers.animeschedule_helper._cache_manager")
async def test_fetch_json_decode_error_returns_none(mock_cache):
    import json as _json
    from enrichment.api_helpers.animeschedule_helper import fetch_all

    r = AsyncMock()
    r.raise_for_status = MagicMock()
    r.json = AsyncMock(side_effect=_json.JSONDecodeError("bad", "", 0))
    session = _make_session(r)
    mock_cache.get_aiohttp_session.return_value = session

    assert await fetch_all("Test Anime") is None
    session.close.assert_awaited_once()


@pytest.mark.asyncio
@patch("enrichment.api_helpers.animeschedule_helper._cache_manager")
async def test_fetch_http_error_returns_none(mock_cache):
    import aiohttp
    from enrichment.api_helpers.animeschedule_helper import fetch_all

    r = AsyncMock()
    r.raise_for_status = MagicMock(
        side_effect=aiohttp.ClientResponseError(
            request_info=MagicMock(), history=(), status=404, message="Not Found"
        )
    )
    session = _make_session(r)
    mock_cache.get_aiohttp_session.return_value = session

    assert await fetch_all("Test Anime") is None
    session.close.assert_awaited_once()


# ── main() ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@patch("enrichment.api_helpers.animeschedule_helper.fetch_all", new_callable=AsyncMock)
async def test_main_success_default_output(mock_fetch):
    from enrichment.api_helpers.animeschedule_helper import main

    mock_fetch.return_value = MAPPED_RESULT
    with patch("sys.argv", ["script.py", "Test Anime"]):
        assert await main() == 0

    mock_fetch.assert_awaited_once_with("Test Anime", output_path="animeschedule.jsonl")


@pytest.mark.asyncio
@patch("enrichment.api_helpers.animeschedule_helper.fetch_all", new_callable=AsyncMock)
async def test_main_success_custom_output(mock_fetch):
    from enrichment.api_helpers.animeschedule_helper import main

    mock_fetch.return_value = MAPPED_RESULT
    with patch("sys.argv", ["script.py", "Test Anime", "--output", "custom/out.jsonl"]):
        assert await main() == 0

    mock_fetch.assert_awaited_once_with("Test Anime", output_path="custom/out.jsonl")


@pytest.mark.asyncio
@patch("enrichment.api_helpers.animeschedule_helper.fetch_all", new_callable=AsyncMock)
async def test_main_no_result_returns_1(mock_fetch):
    from enrichment.api_helpers.animeschedule_helper import main

    mock_fetch.return_value = None
    with patch("sys.argv", ["script.py", "nonexistent"]):
        assert await main() == 1


@pytest.mark.asyncio
@patch("enrichment.api_helpers.animeschedule_helper.fetch_all", new_callable=AsyncMock)
async def test_main_exception_returns_1(mock_fetch):
    from enrichment.api_helpers.animeschedule_helper import main

    mock_fetch.side_effect = RuntimeError("boom")
    with patch("sys.argv", ["script.py", "Test Anime"]):
        assert await main() == 1


@pytest.mark.asyncio
async def test_main_missing_argument_exits_2():
    from enrichment.api_helpers.animeschedule_helper import main

    with patch("sys.argv", ["script.py"]):
        with pytest.raises(SystemExit) as exc:
            await main()
        assert exc.value.code == 2
