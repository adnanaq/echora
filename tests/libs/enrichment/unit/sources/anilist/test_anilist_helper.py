import json
import logging
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import create_autospec, patch

import aiohttp
import pytest
import yarl
from enrichment.sources.anilist import anilist_helper
from enrichment.sources.anilist.anilist_helper import (
    AniListHelper,
    _extract_anilist_id,
    main,
)
from enrichment.sources.base.exceptions import (
    AniListGraphQLError,
    ServiceBlockedError,
    ServiceNetworkError,
    ServiceRateLimitedError,
)
from http_cache.aiohttp_adapter import CachedAiohttpSession

GRAPHQL_URL = "https://graphql.anilist.co"
ONE_PIECE_URL = "https://anilist.co/anime/21"
ONE_PIECE = {"id": 21, "idMal": 21, "title": {"romaji": "ONE PIECE"}}
LUFFY_EDGE = {"node": {"id": 40, "name": {"full": "Monkey D. Luffy"}}, "role": "MAIN"}
ZORO_EDGE = {"node": {"id": 62, "name": {"full": "Roronoa Zoro"}}, "role": "MAIN"}


@dataclass
class Reply:
    body: dict | None = None
    status: int = 200
    headers: dict[str, str] = field(default_factory=dict)
    cached: bool = True
    error: BaseException | None = None


def _media(media: dict | None) -> Reply:
    return Reply({"data": {"Media": media}})


def _characters_page(
    edges: list[dict], has_next_page: bool, cached: bool = True
) -> Reply:
    return Reply(
        {
            "data": {
                "Media": {
                    "characters": {
                        "edges": edges,
                        "pageInfo": {"hasNextPage": has_next_page},
                    }
                }
            }
        },
        cached=cached,
    )


@asynccontextmanager
async def _respond(reply: Reply):
    response = create_autospec(aiohttp.ClientResponse, instance=True)
    response.status = reply.status
    response.headers = reply.headers
    response.from_cache = reply.cached
    if isinstance(reply.error, json.JSONDecodeError):
        response.json.side_effect = reply.error
    else:
        response.json.return_value = reply.body
    if reply.status >= 400:
        url = yarl.URL(GRAPHQL_URL)
        response.raise_for_status.side_effect = aiohttp.ClientResponseError(
            request_info=aiohttp.RequestInfo(url, "POST", {}, url),
            history=(),
            status=reply.status,
        )
    yield response


@pytest.fixture
def anilist_api():
    api = SimpleNamespace(
        media={}, mal_media={}, character_pages={}, queue=[], waits=[], requests=[]
    )

    def reply_for(variables: dict) -> Reply:
        if api.queue:
            return api.queue.pop(0)
        if "idMal" in variables:
            return _media(api.mal_media.get(variables["idMal"]))
        if "page" in variables:
            pages = api.character_pages.get(variables["id"], [])
            page = variables["page"]
            if page > len(pages):
                return Reply({"data": {"Media": None}})
            return _characters_page(pages[page - 1], has_next_page=page < len(pages))
        media = api.media.get(variables["id"])
        return (
            _media(media) if media else Reply({"errors": [{"status": 404}]}, status=404)
        )

    def post(url, *, json, headers):
        api.requests.append({"url": url, "json": json, "headers": headers})
        reply = reply_for(json["variables"])
        if reply.error and not isinstance(
            reply.error, aiohttp.ClientResponseError | ValueError
        ):
            raise reply.error
        return _respond(reply)

    session = create_autospec(CachedAiohttpSession, instance=True)
    session.post.side_effect = post
    with (
        patch.object(
            anilist_helper.http_cache_manager,
            "get_aiohttp_session",
            autospec=True,
            return_value=session,
        ) as get_session,
        patch.object(anilist_helper.asyncio, "sleep", autospec=True) as sleep,
    ):
        sleep.side_effect = lambda seconds: api.waits.append(seconds)
        api.session = session
        api.get_session = get_session
        yield api


def test_extract_anilist_id_anime_address_returns_number() -> None:
    assert _extract_anilist_id("https://anilist.co/anime/21/") == 21


def test_extract_anilist_id_address_without_number_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Cannot extract AniList ID"):
        _extract_anilist_id("https://anilist.co/anime/one-piece")


def test_init_starts_without_session_and_with_full_rate_budget() -> None:
    helper = AniListHelper()
    assert (helper.base_url, helper.session, helper.rate_limit_remaining) == (
        GRAPHQL_URL,
        None,
        90,
    )


async def test_ensure_session_opens_one_cached_session_for_several_requests(
    anilist_api,
) -> None:
    anilist_api.media[21] = ONE_PIECE
    helper = AniListHelper()
    await helper.fetch_anime(21)
    await helper.fetch_anime(21)
    assert helper.session is anilist_api.session
    assert anilist_api.get_session.call_count == 1
    assert anilist_api.get_session.call_args.args == ("anilist",)


async def test_ensure_session_without_session_from_cache_manager_raises_runtime_error(
    anilist_api,
) -> None:
    anilist_api.get_session.return_value = None
    with pytest.raises(RuntimeError, match="Failed to initialize AniList session"):
        await AniListHelper()._ensure_session()


async def test_execute_request_returns_data_marked_with_cache_state(
    anilist_api,
) -> None:
    anilist_api.queue.append(Reply({"data": {"Media": ONE_PIECE}}, cached=False))
    result = await AniListHelper()._execute_request("query", {"id": 21})
    assert result == {"Media": ONE_PIECE, "_from_cache": False}


async def test_execute_request_posts_query_and_variables_as_json(anilist_api) -> None:
    anilist_api.media[21] = ONE_PIECE
    await AniListHelper()._execute_request(
        "query ($id: Int)", {"id": 21, "q": "ワンピース"}
    )
    request = anilist_api.requests[0]
    assert request["url"] == GRAPHQL_URL
    assert request["json"] == {
        "query": "query ($id: Int)",
        "variables": {"id": 21, "q": "ワンピース"},
    }
    assert request["headers"]["Content-Type"] == "application/json"
    assert request["headers"]["X-Hishel-Body-Key"] == "true"


async def test_execute_request_without_variables_sends_empty_variables(
    anilist_api,
) -> None:
    anilist_api.queue.append(Reply({"data": {}}))
    await AniListHelper()._execute_request("query")
    assert anilist_api.requests[0]["json"]["variables"] == {}


async def test_execute_request_rate_limit_header_updates_remaining_budget(
    anilist_api,
) -> None:
    anilist_api.queue.append(
        Reply({"data": {}}, headers={"X-RateLimit-Remaining": "42"})
    )
    helper = AniListHelper()
    await helper._execute_request("query")
    assert helper.rate_limit_remaining == 42


async def test_execute_request_without_rate_limit_header_keeps_budget(
    anilist_api,
) -> None:
    anilist_api.queue.append(Reply({"data": {}}))
    helper = AniListHelper()
    helper.rate_limit_remaining = 50
    await helper._execute_request("query")
    assert helper.rate_limit_remaining == 50


async def test_execute_request_low_budget_on_live_reply_waits_and_resets_budget(
    anilist_api,
) -> None:
    anilist_api.queue.append(
        Reply({"data": {}}, headers={"X-RateLimit-Remaining": "4"}, cached=False)
    )
    helper = AniListHelper()
    await helper._execute_request("query")
    assert anilist_api.waits == [60]
    assert helper.rate_limit_remaining == 90


@pytest.mark.parametrize(
    ("remaining", "cached"), [("5", False), ("4", True)], ids=["at_threshold", "cached"]
)
async def test_execute_request_budget_at_threshold_or_cached_reply_does_not_wait(
    anilist_api, remaining: str, cached: bool
) -> None:
    anilist_api.queue.append(
        Reply({"data": {}}, headers={"X-RateLimit-Remaining": remaining}, cached=cached)
    )
    await AniListHelper()._execute_request("query")
    assert anilist_api.waits == []


async def test_execute_request_rate_limited_reply_waits_retry_after_then_returns(
    anilist_api,
) -> None:
    anilist_api.queue.extend(
        [Reply(status=429, headers={"Retry-After": "120"}), Reply({"data": {"ok": 1}})]
    )
    result = await AniListHelper()._execute_request("query")
    assert result["ok"] == 1
    assert anilist_api.waits == [120]


async def test_execute_request_rate_limited_reply_without_retry_after_waits_default(
    anilist_api,
) -> None:
    anilist_api.queue.extend([Reply(status=429), Reply({"data": {}})])
    await AniListHelper()._execute_request("query")
    assert anilist_api.waits == [60]


async def test_execute_request_rate_limited_three_times_raises_rate_limited_error(
    anilist_api,
) -> None:
    anilist_api.queue.extend(
        [Reply(status=429, headers={"Retry-After": "1"}) for _ in range(3)]
    )
    with pytest.raises(ServiceRateLimitedError):
        await AniListHelper()._execute_request("query")
    assert anilist_api.waits == [1, 1]


async def test_execute_request_forbidden_reply_raises_service_blocked_error(
    anilist_api,
) -> None:
    anilist_api.queue.append(Reply(status=403))
    with pytest.raises(ServiceBlockedError):
        await AniListHelper()._execute_request("query")


async def test_execute_request_not_found_reply_returns_result_without_media(
    anilist_api,
) -> None:
    result = await AniListHelper()._execute_request("query", {"id": 99999})
    assert result == {"_from_cache": True}


async def test_execute_request_graphql_errors_raise_graphql_error(anilist_api) -> None:
    anilist_api.queue.append(
        Reply({"data": {"Media": None}, "errors": [{"message": "Invalid query"}]})
    )
    with pytest.raises(AniListGraphQLError):
        await AniListHelper()._execute_request("query")


async def test_execute_request_client_error_status_raises_service_network_error(
    anilist_api,
) -> None:
    anilist_api.queue.append(Reply(status=400))
    with pytest.raises(ServiceNetworkError):
        await AniListHelper()._execute_request("query")


async def test_execute_request_server_error_status_raises_client_response_error(
    anilist_api,
) -> None:
    anilist_api.queue.append(Reply(status=502))
    with pytest.raises(aiohttp.ClientResponseError):
        await AniListHelper()._execute_request("query")


@pytest.mark.parametrize(
    "error",
    [
        aiohttp.ClientConnectionError("connection refused"),
        TimeoutError("timed out"),
        json.JSONDecodeError("bad", "", 0),
    ],
    ids=["connection", "timeout", "invalid_json"],
)
async def test_execute_request_transport_or_decoding_failure_raises_service_network_error(
    anilist_api, error: BaseException
) -> None:
    anilist_api.queue.append(Reply(error=error))
    with pytest.raises(ServiceNetworkError):
        await AniListHelper()._execute_request("query")


def test_get_media_query_fields_selects_titles_dates_relations_and_links() -> None:
    selections = " ".join(AniListHelper()._get_media_query_fields().split())
    for selection in (
        "idMal",
        "title { romaji english native userPreferred }",
        "startDate { year month day }",
        "endDate { year month day }",
        "relations {",
        "studios {",
        "externalLinks {",
        "rankings {",
    ):
        assert selection in selections


def test_build_query_by_anilist_id_queries_media_by_id() -> None:
    query = AniListHelper()._build_query_by_anilist_id()
    assert "query ($id: Int)" in query
    assert "Media(id: $id, type: ANIME)" in query


def test_build_query_by_mal_id_queries_media_by_mal_id() -> None:
    query = AniListHelper()._build_query_by_mal_id()
    assert "query ($idMal: Int)" in query
    assert "Media(idMal: $idMal, type: ANIME)" in query


async def test_fetch_anime_existing_anime_returns_media(anilist_api) -> None:
    anilist_api.media[21] = ONE_PIECE
    assert await AniListHelper().fetch_anime(21) == ONE_PIECE
    assert anilist_api.requests[0]["json"]["variables"] == {"id": 21}


@pytest.mark.parametrize("anilist_id", [99999, 0, -1])
async def test_fetch_anime_unknown_anime_returns_none(
    anilist_api, anilist_id: int
) -> None:
    assert await AniListHelper().fetch_anime(anilist_id) is None


async def test_fetch_anime_empty_data_returns_none(anilist_api) -> None:
    anilist_api.queue.append(Reply({"data": {}}))
    assert await AniListHelper().fetch_anime(21) is None


async def test_fetch_anime_by_mal_id_existing_anime_returns_media(anilist_api) -> None:
    anilist_api.mal_media[21] = ONE_PIECE
    assert await AniListHelper().fetch_anime_by_mal_id(21) == ONE_PIECE
    assert anilist_api.requests[0]["json"]["variables"] == {"idMal": 21}


async def test_fetch_anime_by_mal_id_unknown_anime_returns_none(anilist_api) -> None:
    assert await AniListHelper().fetch_anime_by_mal_id(99999) is None


async def test_fetch_paginated_data_several_pages_returns_every_edge(
    anilist_api,
) -> None:
    pages = [
        [{"node": {"id": number}} for number in range(start, start + 50)]
        for start in (0, 50, 100)
    ]
    anilist_api.character_pages[21] = pages
    edges = await AniListHelper()._fetch_paginated_data(21, "query", "characters")
    assert edges == pages[0] + pages[1] + pages[2]
    assert [
        request["json"]["variables"]["page"] for request in anilist_api.requests
    ] == [
        1,
        2,
        3,
    ]


async def test_fetch_paginated_data_live_pages_wait_only_between_pages(
    anilist_api,
) -> None:
    anilist_api.queue.extend(
        [
            _characters_page([LUFFY_EDGE], has_next_page=True, cached=False),
            _characters_page([ZORO_EDGE], has_next_page=False, cached=False),
        ]
    )
    edges = await AniListHelper()._fetch_paginated_data(21, "query", "characters")
    assert edges == [LUFFY_EDGE, ZORO_EDGE]
    assert anilist_api.waits == [0.5]


async def test_fetch_paginated_data_cached_pages_do_not_wait(anilist_api) -> None:
    anilist_api.character_pages[21] = [[LUFFY_EDGE], [ZORO_EDGE]]
    await AniListHelper()._fetch_paginated_data(21, "query", "characters")
    assert anilist_api.waits == []


@pytest.mark.parametrize(
    "reply",
    [
        Reply(status=404),
        Reply({"data": {"Media": None}}),
        Reply({"data": {"Media": {}}}),
        Reply({"data": {"Media": {"characters": None}}}),
    ],
    ids=["not_found", "no_media", "no_connection", "null_connection"],
)
async def test_fetch_paginated_data_without_connection_returns_empty_list(
    anilist_api, reply: Reply
) -> None:
    anilist_api.queue.append(reply)
    assert await AniListHelper()._fetch_paginated_data(21, "query", "characters") == []


@pytest.mark.parametrize(
    "connection",
    [
        {"edges": [], "pageInfo": {"hasNextPage": False}},
        {"pageInfo": {"hasNextPage": False}},
    ],
    ids=["empty_edges", "missing_edges"],
)
async def test_fetch_paginated_data_page_without_edges_returns_empty_list(
    anilist_api, connection: dict
) -> None:
    anilist_api.queue.append(Reply({"data": {"Media": {"characters": connection}}}))
    assert await AniListHelper()._fetch_paginated_data(21, "query", "characters") == []


async def test_fetch_paginated_data_without_page_info_stops_after_first_page(
    anilist_api,
) -> None:
    anilist_api.queue.append(
        Reply({"data": {"Media": {"characters": {"edges": [LUFFY_EDGE]}}}})
    )
    edges = await AniListHelper()._fetch_paginated_data(21, "query", "characters")
    assert edges == [LUFFY_EDGE]
    assert len(anilist_api.requests) == 1


async def test_fetch_characters_returns_edges_from_characters_query(
    anilist_api,
) -> None:
    anilist_api.character_pages[21] = [[LUFFY_EDGE, ZORO_EDGE]]
    assert await AniListHelper().fetch_characters(21) == [LUFFY_EDGE, ZORO_EDGE]
    query = anilist_api.requests[0]["json"]["query"]
    assert "characters(page: $page, perPage: 50" in query
    assert "voiceActorRoles" in query


async def test_fetch_anime_canonical_maps_anime_from_address(anilist_api) -> None:
    anilist_api.media[21] = ONE_PIECE
    anime = await AniListHelper().fetch_anime_canonical(ONE_PIECE_URL)
    assert anime["title"] == "ONE PIECE"
    assert anime["sources"] == [ONE_PIECE_URL, "https://myanimelist.net/anime/21"]


async def test_fetch_anime_canonical_temp_dir_saves_anime(
    anilist_api, tmp_path: Path
) -> None:
    anilist_api.media[21] = ONE_PIECE
    anime = await AniListHelper().fetch_anime_canonical(ONE_PIECE_URL, str(tmp_path))
    saved = (tmp_path / "anilist.jsonl").read_text().splitlines()
    assert [json.loads(line) for line in saved] == [anime]


async def test_fetch_anime_canonical_without_temp_dir_saves_nothing(
    anilist_api, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    anilist_api.media[21] = ONE_PIECE
    monkeypatch.chdir(tmp_path)
    await AniListHelper().fetch_anime_canonical(ONE_PIECE_URL)
    assert list(tmp_path.iterdir()) == []


async def test_fetch_anime_canonical_unknown_anime_logs_and_returns_none(
    anilist_api, caplog
) -> None:
    with caplog.at_level(logging.WARNING):
        assert await AniListHelper().fetch_anime_canonical(ONE_PIECE_URL) is None
    assert f"No AniList data found for: {ONE_PIECE_URL}" in caplog.messages


async def test_fetch_characters_canonical_maps_every_valid_edge(anilist_api) -> None:
    anilist_api.character_pages[21] = [[LUFFY_EDGE, {"node": "broken"}, ZORO_EDGE]]
    characters = await AniListHelper().fetch_characters_canonical(ONE_PIECE_URL)
    assert [character["name"] for character in characters] == [
        "Monkey D. Luffy",
        "Roronoa Zoro",
    ]


async def test_fetch_characters_canonical_temp_dir_saves_each_character(
    anilist_api, tmp_path: Path
) -> None:
    anilist_api.character_pages[21] = [[LUFFY_EDGE, ZORO_EDGE]]
    characters = await AniListHelper().fetch_characters_canonical(
        ONE_PIECE_URL, str(tmp_path)
    )
    saved = (tmp_path / "anilist_characters.jsonl").read_text().splitlines()
    assert [json.loads(line) for line in saved] == characters


async def test_fetch_characters_canonical_without_characters_saves_nothing(
    anilist_api, tmp_path: Path
) -> None:
    assert (
        await AniListHelper().fetch_characters_canonical(ONE_PIECE_URL, str(tmp_path))
        == []
    )
    assert not (tmp_path / "anilist_characters.jsonl").exists()


async def test_fetch_all_anime_and_characters_return_payload(
    anilist_api, tmp_path: Path
) -> None:
    anilist_api.media[21] = ONE_PIECE
    anilist_api.character_pages[21] = [[LUFFY_EDGE]]

    result = await AniListHelper().fetch_all(
        {"anilist_url": ONE_PIECE_URL}, {}, str(tmp_path), fetch_episodes=False
    )

    assert result["anime"]["title"] == "ONE PIECE"
    assert [character["name"] for character in result["characters"]] == [
        "Monkey D. Luffy"
    ]
    assert {path.name for path in tmp_path.iterdir()} == {
        "anilist.jsonl",
        "anilist_characters.jsonl",
    }


async def test_fetch_all_anime_without_characters_returns_payload(anilist_api) -> None:
    anilist_api.media[21] = ONE_PIECE
    result = await AniListHelper().fetch_all({"anilist_url": ONE_PIECE_URL}, {})
    assert result["anime"]["title"] == "ONE PIECE"
    assert result["characters"] == []


async def test_fetch_all_characters_turned_off_skips_character_requests(
    anilist_api,
) -> None:
    anilist_api.media[21] = ONE_PIECE
    anilist_api.character_pages[21] = [[LUFFY_EDGE]]
    result = await AniListHelper().fetch_all(
        {"anilist_url": ONE_PIECE_URL}, {}, fetch_characters=False
    )
    assert result["characters"] == []
    assert [request["json"]["variables"] for request in anilist_api.requests] == [
        {"id": 21}
    ]


async def test_fetch_all_without_anime_or_characters_returns_none(anilist_api) -> None:
    assert await AniListHelper().fetch_all({"anilist_url": ONE_PIECE_URL}, {}) is None


async def test_fetch_all_without_anilist_link_returns_none(anilist_api) -> None:
    assert await AniListHelper().fetch_all({}, {}) is None
    assert anilist_api.requests == []


async def test_fetch_all_data_by_mal_id_found_anime_includes_character_edges(
    anilist_api,
) -> None:
    anilist_api.mal_media[21] = dict(ONE_PIECE)
    anilist_api.character_pages[21] = [[LUFFY_EDGE]]
    data = await AniListHelper()._fetch_all_data_by_mal_id(21)
    assert data["characters"] == {"edges": [LUFFY_EDGE]}


async def test_fetch_all_data_by_mal_id_anime_without_characters_omits_them(
    anilist_api,
) -> None:
    anilist_api.mal_media[21] = dict(ONE_PIECE)
    data = await AniListHelper()._fetch_all_data_by_mal_id(21)
    assert data == ONE_PIECE


async def test_fetch_all_data_by_mal_id_unknown_anime_returns_none(anilist_api) -> None:
    assert await AniListHelper()._fetch_all_data_by_mal_id(99999) is None


async def test_close_after_request_closes_and_forgets_session(anilist_api) -> None:
    anilist_api.media[21] = ONE_PIECE
    helper = AniListHelper()
    await helper.fetch_anime(21)
    await helper.close()
    anilist_api.session.close.assert_awaited_once()
    assert helper.session is None


async def test_close_without_session_does_nothing(anilist_api) -> None:
    helper = AniListHelper()
    await helper.close()
    assert helper.session is None
    anilist_api.session.close.assert_not_called()


async def test_close_runs_when_context_exits_after_error(anilist_api) -> None:
    anilist_api.media[21] = ONE_PIECE
    with pytest.raises(ValueError, match="Test error"):
        async with AniListHelper() as helper:
            await helper.fetch_anime(21)
            raise ValueError("Test error")
    anilist_api.session.close.assert_awaited_once()


async def test_main_anilist_address_writes_files_and_returns_zero(
    anilist_api, tmp_path: Path
) -> None:
    anilist_api.media[21] = ONE_PIECE
    anilist_api.character_pages[21] = [[LUFFY_EDGE]]
    with patch.object(
        sys, "argv", ["prog", "--url", ONE_PIECE_URL, "--output", str(tmp_path)]
    ):
        assert await main() == 0
    assert {path.name for path in tmp_path.iterdir()} == {
        "anilist.jsonl",
        "anilist_characters.jsonl",
    }
    anilist_api.session.close.assert_awaited_once()


async def test_main_unknown_anilist_address_returns_one(
    anilist_api, tmp_path: Path
) -> None:
    with patch.object(
        sys, "argv", ["prog", "--url", ONE_PIECE_URL, "--output", str(tmp_path)]
    ):
        assert await main() == 1


async def test_main_request_error_returns_one_and_closes_session(
    anilist_api, tmp_path: Path
) -> None:
    anilist_api.queue.append(Reply(status=403))
    with patch.object(
        sys, "argv", ["prog", "--url", ONE_PIECE_URL, "--output", str(tmp_path)]
    ):
        assert await main() == 1
    anilist_api.session.close.assert_awaited_once()


async def test_main_mal_id_resolves_anilist_address_and_returns_zero(
    anilist_api, tmp_path: Path
) -> None:
    anilist_api.mal_media[21] = ONE_PIECE
    anilist_api.media[21] = ONE_PIECE
    with patch.object(
        sys, "argv", ["prog", "--mal-id", "21", "--output", str(tmp_path)]
    ):
        assert await main() == 0
    saved = json.loads((tmp_path / "anilist.jsonl").read_text())
    assert saved["sources"][0] == ONE_PIECE_URL


async def test_main_unknown_mal_id_returns_one(anilist_api, tmp_path: Path) -> None:
    with patch.object(
        sys, "argv", ["prog", "--mal-id", "99999", "--output", str(tmp_path)]
    ):
        assert await main() == 1


async def test_main_mal_reply_without_anilist_id_returns_one(
    anilist_api, tmp_path: Path
) -> None:
    anilist_api.mal_media[21] = {"idMal": 21, "title": {"romaji": "ONE PIECE"}}
    with patch.object(
        sys, "argv", ["prog", "--mal-id", "21", "--output", str(tmp_path)]
    ):
        assert await main() == 1
