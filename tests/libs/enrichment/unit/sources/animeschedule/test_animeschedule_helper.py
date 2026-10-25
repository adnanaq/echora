import json
import sys
from pathlib import Path
from unittest.mock import create_autospec, patch

import aiohttp
import pytest
import yarl
from enrichment.sources.animeschedule import animeschedule_helper
from enrichment.sources.animeschedule.animeschedule_helper import (
    AnimescheduleHelper,
    _match_by_sources,
    main,
)
from enrichment.sources.base.exceptions import ServiceNetworkError, ServiceParseError
from http_cache import get_cache
from http_cache.aiohttp_adapter import CachedAiohttpSession

SEARCH_URL = "https://animeschedule.net/api/v3/anime"
MAL_URL = "https://myanimelist.net/anime/21"
ONE_PIECE_RESULT = {
    "id": "abc1",
    "title": "One Piece",
    "route": "one-piece",
    "websites": {"mal": "myanimelist.net/anime/21"},
}
OTHER_RESULT = {
    "id": "abc2",
    "title": "One Piece Film",
    "route": "one-piece-film",
    "websites": {"mal": "myanimelist.net/anime/99"},
}


@pytest.fixture
def search_session():
    session = create_autospec(CachedAiohttpSession, instance=True)
    response = create_autospec(aiohttp.ClientResponse, instance=True)
    session.get.return_value.__aenter__.return_value = response
    with (
        patch.object(get_cache(), "get_aiohttp_session", autospec=True) as get_session,
        patch.object(
            animeschedule_helper,
            "configure_cache",
            autospec=True,
            return_value=get_cache(),
        ),
    ):
        get_session.return_value.__aenter__.return_value = session
        yield session


def _answer(session: CachedAiohttpSession, payload: dict | None) -> None:
    session.get.return_value.__aenter__.return_value.json.return_value = payload


def _response(session: CachedAiohttpSession) -> aiohttp.ClientResponse:
    return session.get.return_value.__aenter__.return_value


def _raise_for(status: int) -> None:
    if status >= 400:
        raise _error_status(status)


def _error_status(status: int) -> aiohttp.ClientResponseError:
    url = yarl.URL(SEARCH_URL)
    request = aiohttp.RequestInfo(url=url, method="GET", headers={}, real_url=url)
    return aiohttp.ClientResponseError(
        request_info=request, history=(), status=status, message="Error"
    )


def test_match_by_sources_without_candidates_returns_none() -> None:
    assert _match_by_sources([], [MAL_URL]) is None


def test_match_by_sources_candidate_on_other_page_returns_none() -> None:
    assert _match_by_sources([OTHER_RESULT], [MAL_URL]) is None


@pytest.mark.parametrize(
    ("websites", "source"),
    [
        ({"mal": "myanimelist.net/anime/21"}, "https://myanimelist.net/anime/21"),
        ({"aniList": "anilist.co/anime/21"}, "https://anilist.co/anime/21"),
        ({"kitsu": "kitsu.io/anime/one-piece"}, "https://kitsu.io/anime/one-piece"),
        (
            {"animePlanet": "anime-planet.com/anime/one-piece"},
            "https://anime-planet.com/anime/one-piece",
        ),
        ({"anidb": "anidb.net/anime/69"}, "https://anidb.net/anime/69"),
        ({"mal": "myanimelist.net/anime/21"}, "http://myanimelist.net/anime/21"),
    ],
)
def test_match_by_sources_provider_link_matches_returns_candidate(
    websites: dict, source: str
) -> None:
    candidate = {"id": "1", "websites": websites}
    assert _match_by_sources([candidate], [source]) is candidate


def test_match_by_sources_candidate_link_with_title_matches_bare_source() -> None:
    candidate = {"id": "1", "websites": {"mal": "myanimelist.net/anime/21/One_Piece"}}
    assert _match_by_sources([candidate], [MAL_URL]) is candidate


def test_match_by_sources_bare_candidate_link_matches_source_with_title() -> None:
    candidate = {"id": "1", "websites": {"mal": "myanimelist.net/anime/21"}}
    assert _match_by_sources([candidate], [f"{MAL_URL}/One_Piece"]) is candidate


@pytest.mark.parametrize(
    ("link", "source"),
    [
        ("myanimelist.net/anime/2", MAL_URL),
        ("myanimelist.net/anime/21", "https://myanimelist.net/anime/2"),
        ("myanimelist.net/anime/210/Other", MAL_URL),
    ],
)
def test_match_by_sources_other_id_sharing_digits_returns_none(
    link: str, source: str
) -> None:
    assert _match_by_sources([{"id": "1", "websites": {"mal": link}}], [source]) is None


def test_match_by_sources_link_with_trailing_slash_matches_source() -> None:
    candidate = {
        "id": "1",
        "websites": {"aniList": "anilist.co/anime/154587/Sousou-no-Frieren/"},
    }
    assert _match_by_sources([candidate], ["https://anilist.co/anime/154587"]) is (
        candidate
    )


def test_match_by_sources_several_candidates_returns_first_match() -> None:
    assert _match_by_sources([OTHER_RESULT, ONE_PIECE_RESULT], [MAL_URL]) is (
        ONE_PIECE_RESULT
    )


def test_match_by_sources_official_and_stream_links_not_compared() -> None:
    candidate = {
        "id": "1",
        "websites": {
            "official": "myanimelist.net/anime/21",
            "streams": [{"platform": "mal", "url": "myanimelist.net/anime/21"}],
        },
    }
    assert _match_by_sources([candidate], [MAL_URL]) is None


@pytest.mark.parametrize("websites", [{"mal": None, "aniList": 12345}, {"mal": ""}])
def test_match_by_sources_empty_or_non_text_links_skipped(websites: dict) -> None:
    assert _match_by_sources([{"id": "1", "websites": websites}], [MAL_URL]) is None


@pytest.mark.parametrize("sources", [[], ["", ""]])
def test_match_by_sources_without_usable_sources_returns_none(
    sources: list[str],
) -> None:
    assert _match_by_sources([ONE_PIECE_RESULT], sources) is None


def _sent_params(session: CachedAiohttpSession) -> list[dict[str, str]]:
    return [call.kwargs["params"] for call in session.get.call_args_list]


async def test_fetch_all_mal_id_finds_anime_without_title_search(
    search_session,
) -> None:
    _answer(search_session, {"anime": [ONE_PIECE_RESULT]})

    result = await AnimescheduleHelper(get_cache()).fetch_all(
        {"mal_url": "https://myanimelist.net/anime/21/One_Piece"},
        {"title": "One Piece", "sources": [MAL_URL]},
    )

    assert result is not None
    assert result["anime"]["title"] == "One Piece"
    assert _sent_params(search_session) == [{"mal-ids": "21"}]


async def test_fetch_all_unknown_mal_id_tries_anilist_then_anidb(
    search_session,
) -> None:
    _response(search_session).status = 404

    await AnimescheduleHelper(get_cache()).fetch_all(
        {
            "mal_url": "https://myanimelist.net/anime/21",
            "anilist_url": "https://anilist.co/anime/21",
            "anidb_url": "https://anidb.net/anime/69",
        },
        {},
    )

    assert _sent_params(search_session) == [
        {"mal-ids": "21"},
        {"anilist-ids": "21"},
        {"anidb-ids": "69"},
    ]


def _answer_by_lookup(
    session: CachedAiohttpSession, id_status: int, title_status: int
) -> None:
    response = _response(session)

    def answer(*args: object, **kwargs: dict[str, str]) -> object:
        response.status = title_status if "q" in kwargs["params"] else id_status
        return session.get.return_value

    session.get.side_effect = answer


async def test_fetch_all_ids_not_found_searches_title_and_matches_sources(
    search_session,
) -> None:
    _answer_by_lookup(search_session, id_status=404, title_status=200)
    _answer(search_session, {"anime": [OTHER_RESULT, ONE_PIECE_RESULT]})

    result = await AnimescheduleHelper(get_cache()).fetch_all(
        {"mal_url": MAL_URL}, {"title": "One Piece", "sources": [MAL_URL]}
    )

    assert result is not None
    assert result["anime"]["title"] == "One Piece"
    assert _sent_params(search_session) == [{"mal-ids": "21"}, {"q": "One Piece"}]


async def test_fetch_all_id_result_linking_elsewhere_searches_title(
    search_session,
) -> None:
    _answer(search_session, {"anime": [OTHER_RESULT]})

    result = await AnimescheduleHelper(get_cache()).fetch_all(
        {"mal_url": MAL_URL}, {"title": "One Piece", "sources": [MAL_URL]}
    )

    assert result is None
    assert _sent_params(search_session) == [{"mal-ids": "21"}, {"q": "One Piece"}]


async def test_fetch_all_unknown_id_and_unmatched_title_returns_none(
    search_session,
) -> None:
    _answer_by_lookup(search_session, id_status=404, title_status=404)

    result = await AnimescheduleHelper(get_cache()).fetch_all(
        {"mal_url": MAL_URL}, {"title": "One Piece", "sources": [MAL_URL]}
    )

    assert result is None


@pytest.mark.parametrize(
    ("id_status", "title_status"), [(500, 500), (404, 500), (404, 503)]
)
async def test_fetch_all_status_outside_lookup_not_found_raises_service_network_error(
    search_session, id_status: int, title_status: int
) -> None:
    _answer_by_lookup(search_session, id_status=id_status, title_status=title_status)
    _response(search_session).raise_for_status.side_effect = lambda: _raise_for(
        _response(search_session).status
    )

    with pytest.raises(ServiceNetworkError):
        await AnimescheduleHelper(get_cache()).fetch_all(
            {"mal_url": MAL_URL}, {"title": "One Piece", "sources": [MAL_URL]}
        )


async def test_fetch_all_without_ids_or_title_sends_nothing(search_session) -> None:
    assert await AnimescheduleHelper(get_cache()).fetch_all({}, {}) is None
    search_session.get.assert_not_called()


async def test_fetch_all_output_directory_appends_mapped_result(
    search_session, tmp_path: Path
) -> None:
    _answer(search_session, {"anime": [ONE_PIECE_RESULT]})

    result = await AnimescheduleHelper(get_cache()).fetch_all(
        {"mal_url": MAL_URL}, {}, temp_dir=str(tmp_path)
    )

    assert result is not None
    assert (result["episodes"], result["characters"], result["extras"]) == ([], [], {})
    saved = (tmp_path / "animeschedule.jsonl").read_text().splitlines()
    assert [json.loads(line) for line in saved] == [result["anime"]]


async def test_find_by_title_without_sources_returns_first_result(
    search_session,
) -> None:
    _answer(search_session, {"anime": [ONE_PIECE_RESULT, OTHER_RESULT]})

    assert (
        await AnimescheduleHelper(get_cache())._find_by_title("One Piece", [])
        == ONE_PIECE_RESULT
    )


async def test_find_by_title_with_sources_and_no_matching_result_returns_none(
    search_session,
) -> None:
    _answer(search_session, {"anime": [OTHER_RESULT]})

    assert (
        await AnimescheduleHelper(get_cache())._find_by_title("One Piece", [MAL_URL])
        is None
    )


@pytest.mark.parametrize("payload", [{"anime": []}, None])
async def test_find_by_title_without_results_returns_none(
    search_session, payload: dict | None
) -> None:
    _answer(search_session, payload)

    assert (
        await AnimescheduleHelper(get_cache())._find_by_title("One Piece", []) is None
    )


async def test_find_by_title_title_with_query_characters_sends_whole_title(
    search_session,
) -> None:
    _answer(search_session, {"anime": []})

    await AnimescheduleHelper(get_cache())._find_by_title("009-1: R&B", [])

    request = search_session.get.call_args
    sent_url = yarl.URL(request.args[0]).update_query(request.kwargs.get("params", {}))
    assert (str(sent_url.with_query(None)), dict(sent_url.query)) == (
        SEARCH_URL,
        {"q": "009-1: R&B"},
    )


async def test_find_by_title_connection_error_raises_service_network_error(
    search_session,
) -> None:
    search_session.get.return_value.__aenter__.side_effect = aiohttp.ClientError(
        "connection refused"
    )

    with pytest.raises(ServiceNetworkError):
        await AnimescheduleHelper(get_cache())._find_by_title("One Piece", [])


async def test_find_by_title_error_status_raises_service_network_error(
    search_session,
) -> None:
    _response(search_session).status = 503
    _response(search_session).raise_for_status.side_effect = _error_status(503)

    with pytest.raises(ServiceNetworkError):
        await AnimescheduleHelper(get_cache())._find_by_title("One Piece", [])


async def test_find_by_title_invalid_json_raises_service_parse_error(
    search_session,
) -> None:
    _response(search_session).json.side_effect = json.JSONDecodeError("bad", "", 0)

    with pytest.raises(ServiceParseError):
        await AnimescheduleHelper(get_cache())._find_by_title("One Piece", [])


async def test_main_found_title_writes_default_output_and_returns_zero(
    search_session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _answer(search_session, {"anime": [ONE_PIECE_RESULT]})
    monkeypatch.chdir(tmp_path)

    with patch.object(sys, "argv", ["script.py", "One Piece"]):
        assert await main() == 0

    assert (tmp_path / "animeschedule.jsonl").exists()


async def test_main_found_title_writes_chosen_output_and_returns_zero(
    search_session, tmp_path: Path
) -> None:
    _answer(search_session, {"anime": [ONE_PIECE_RESULT]})
    output = tmp_path / "out.jsonl"

    with patch.object(sys, "argv", ["script.py", "One Piece", "--output", str(output)]):
        assert await main() == 0

    assert json.loads(output.read_text())["title"] == "One Piece"


async def test_main_without_result_returns_one(search_session, tmp_path: Path) -> None:
    _answer(search_session, {"anime": []})

    with patch.object(
        sys, "argv", ["script.py", "Nothing", "--output", str(tmp_path / "out.jsonl")]
    ):
        assert await main() == 1


async def test_main_search_error_returns_one(search_session, tmp_path: Path) -> None:
    search_session.get.return_value.__aenter__.side_effect = aiohttp.ClientError(
        "connection refused"
    )

    with patch.object(
        sys, "argv", ["script.py", "One Piece", "--output", str(tmp_path / "out.jsonl")]
    ):
        assert await main() == 1


async def test_main_without_title_argument_exits_with_usage_error() -> None:
    with (
        patch.object(sys, "argv", ["script.py"]),
        pytest.raises(SystemExit) as exit_info,
    ):
        await main()

    assert exit_info.value.code == 2
