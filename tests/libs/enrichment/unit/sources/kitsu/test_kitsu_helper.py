import asyncio
import json
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, create_autospec, patch

import aiohttp
import pytest
import yarl
from enrichment.sources.base.exceptions import ServiceNotFoundError
from enrichment.sources.kitsu import kitsu_helper
from enrichment.sources.kitsu.kitsu_helper import KitsuHelper, main
from http_cache.aiohttp_adapter import CachedAiohttpSession

API = "https://kitsu.io/api/edge"
ONE_PIECE = {
    "id": "12",
    "attributes": {
        "canonicalTitle": "One Piece",
        "subtype": "TV",
        "status": "current",
        "slug": "one-piece",
    },
}
LUFFY_LINK = {"id": "mc1", "attributes": {"role": "main"}}
LUFFY_LINK_ROW = {
    **LUFFY_LINK,
    "type": "mediaCharacters",
    "relationships": {"character": {"data": {"id": "c1", "type": "characters"}}},
}
LUFFY = {
    "id": "c1",
    "type": "characters",
    "attributes": {"canonicalName": "Luffy", "slug": "luffy"},
}


@dataclass
class Collection:
    items: list[dict]
    included: list[dict] = field(default_factory=list)
    cached: bool = True


@asynccontextmanager
async def _respond(response: aiohttp.ClientResponse):
    yield response


def _response(
    url: str, status: int, payload: dict, cached: bool
) -> aiohttp.ClientResponse:
    response = create_autospec(aiohttp.ClientResponse, instance=True)
    response.status = status
    response.url = yarl.URL(url)
    response.from_cache = cached
    response.json.return_value = payload
    if status not in (200, 404):
        request = aiohttp.RequestInfo(yarl.URL(url), "GET", {}, yarl.URL(url))
        response.raise_for_status.side_effect = aiohttp.ClientResponseError(
            request_info=request, history=(), status=status
        )
    return response


def _route_key(url: str, params: dict | None) -> str:
    path = url.removeprefix(API)
    slug = (params or {}).get("filter[slug]")
    return f"{path}?filter[slug]={slug}" if slug else path


def _serve(routes: dict[str, Any], url: str, params: dict | None):
    route = routes.get(_route_key(url, params), 404)
    if isinstance(route, BaseException):
        raise route
    if isinstance(route, int):
        return _respond(_response(url, route, {}, cached=True))
    if isinstance(route, Collection):
        offset = (params or {}).get("page[offset]", 0)
        limit = (params or {}).get("page[limit]", 20)
        payload = {
            "data": route.items[offset : offset + limit],
            "included": route.included if offset == 0 else [],
            "meta": {"count": len(route.items)},
        }
        return _respond(_response(url, 200, payload, cached=route.cached))
    return _respond(_response(url, 200, {"data": route}, cached=True))


@pytest.fixture
def kitsu_api():
    routes: dict[str, Any] = {}
    session = create_autospec(CachedAiohttpSession, instance=True)
    session.get.side_effect = lambda url, **request: _serve(
        routes, url, request.get("params")
    )
    with patch.object(
        kitsu_helper._cache_manager, "get_aiohttp_session", autospec=True
    ) as get_session:
        get_session.return_value.__aenter__.return_value = session
        yield SimpleNamespace(routes=routes, session=session, get_session=get_session)


def _requested_offsets(session: CachedAiohttpSession, path: str) -> list[int]:
    return [
        call.kwargs["params"]["page[offset]"]
        for call in session.get.call_args_list
        if call.args[0] == f"{API}{path}"
    ]


def _one_piece_routes(routes: dict[str, Any]) -> None:
    routes["/anime/12"] = ONE_PIECE
    routes["/anime/12/genres"] = Collection(
        [{"id": "g1", "attributes": {"name": "Action"}}]
    )
    routes["/anime/12/categories"] = Collection(
        [{"id": "t1", "attributes": {"title": "Pirates", "description": "At sea"}}]
    )
    routes["/anime/12/anime-productions"] = Collection([])
    routes["/anime/12/episodes"] = Collection(
        [{"id": "e1", "attributes": {"number": 1, "canonicalTitle": "Ep 1"}}]
    )
    routes["/anime/12/characters"] = Collection([LUFFY_LINK_ROW], included=[LUFFY])
    routes["/media-characters/mc1/voices"] = Collection([])
    routes["/characters/c1/media-characters"] = Collection([])


async def test_make_request_ok_response_returns_payload_marked_cached(
    kitsu_api,
) -> None:
    kitsu_api.routes["/anime/12"] = ONE_PIECE
    payload = await KitsuHelper()._make_request("/anime/12")
    assert payload == {"data": ONE_PIECE, "_from_cache": True}


async def test_make_request_sends_json_api_headers_and_parameters(kitsu_api) -> None:
    kitsu_api.routes["/anime/12"] = ONE_PIECE
    await KitsuHelper()._make_request("/anime/12", {"include": "producer"})
    request = kitsu_api.session.get.call_args
    assert request.args[0] == f"{API}/anime/12"
    assert request.kwargs["headers"]["Accept"] == "application/vnd.api+json"
    assert request.kwargs["params"] == {"include": "producer"}


async def test_make_request_given_session_uses_it_without_opening_another(
    kitsu_api,
) -> None:
    kitsu_api.routes["/anime/12"] = ONE_PIECE
    await KitsuHelper()._make_request("/anime/12", session=kitsu_api.session)
    kitsu_api.get_session.assert_not_called()


async def test_make_request_missing_resource_raises_service_not_found_error(
    kitsu_api,
) -> None:
    with pytest.raises(ServiceNotFoundError):
        await KitsuHelper()._make_request("/anime/1")


async def test_make_request_server_error_raises_client_response_error(
    kitsu_api,
) -> None:
    kitsu_api.routes["/anime/1"] = 500
    with pytest.raises(aiohttp.ClientResponseError):
        await KitsuHelper()._make_request("/anime/1")


async def test_paginate_collection_over_several_pages_returns_every_item(
    kitsu_api,
) -> None:
    episodes = [{"id": str(number)} for number in range(45)]
    kitsu_api.routes["/anime/12/episodes"] = Collection(
        episodes, included=[{"id": "x"}]
    )
    items, included = await KitsuHelper()._paginate("/anime/12/episodes")
    assert items == episodes
    assert included == [{"id": "x"}]
    assert _requested_offsets(kitsu_api.session, "/anime/12/episodes") == [0, 20, 40]


async def test_paginate_cached_pages_continue_without_waiting(kitsu_api) -> None:
    kitsu_api.routes["/anime/12/episodes"] = Collection(
        [{"id": str(number)} for number in range(40)], cached=True
    )
    sleep = Mock(wraps=asyncio.sleep)
    with patch.object(kitsu_helper.asyncio, "sleep", sleep):
        items, _ = await KitsuHelper()._paginate("/anime/12/episodes")
    assert len(items) == 40
    assert sleep.call_args_list == []


async def test_paginate_live_pages_wait_between_requests(kitsu_api) -> None:
    kitsu_api.routes["/anime/12/episodes"] = Collection(
        [{"id": str(number)} for number in range(21)], cached=False
    )
    sleep = Mock(wraps=asyncio.sleep)
    with patch.object(kitsu_helper.asyncio, "sleep", sleep):
        items, _ = await KitsuHelper()._paginate("/anime/12/episodes")
    assert len(items) == 21
    assert [call.args for call in sleep.call_args_list] == [(0.1,)]


async def test_paginate_empty_collection_stops_after_first_request(kitsu_api) -> None:
    kitsu_api.routes["/anime/12/episodes"] = Collection([])
    assert await KitsuHelper()._paginate("/anime/12/episodes") == ([], [])
    assert _requested_offsets(kitsu_api.session, "/anime/12/episodes") == [0]


async def test_paginate_response_without_data_stops_with_nothing(kitsu_api) -> None:
    kitsu_api.session.get.side_effect = lambda url, **request: _respond(
        _response(url, 200, {}, cached=True)
    )
    assert await KitsuHelper()._paginate("/anime/12/episodes") == ([], [])


async def test_paginate_failed_later_page_returns_items_collected_so_far(
    kitsu_api,
) -> None:
    first_page = [{"id": str(number)} for number in range(20)]
    kitsu_api.routes["/anime/12/episodes"] = Collection(first_page + [{"id": "20"}])

    def serve_first_page_only(url, **request):
        if request["params"]["page[offset]"] > 0:
            return _respond(_response(url, 500, {}, cached=True))
        return _serve(kitsu_api.routes, url, request["params"])

    kitsu_api.session.get.side_effect = serve_first_page_only
    items, _ = await KitsuHelper()._paginate("/anime/12/episodes")
    assert items == first_page


async def test_get_anime_by_id_existing_anime_returns_resource(kitsu_api) -> None:
    kitsu_api.routes["/anime/12"] = ONE_PIECE
    assert await KitsuHelper().get_anime_by_id(12) == ONE_PIECE


async def test_get_anime_by_id_missing_anime_returns_none(kitsu_api) -> None:
    assert await KitsuHelper().get_anime_by_id(99999) is None


async def test_get_anime_by_id_server_error_returns_none(kitsu_api) -> None:
    kitsu_api.routes["/anime/99999"] = 503
    assert await KitsuHelper().get_anime_by_id(99999) is None


async def test_get_anime_episodes_returns_every_item_across_pages(kitsu_api) -> None:
    items = [{"id": str(number)} for number in range(25)]
    kitsu_api.routes["/anime/12/episodes"] = Collection(items)
    assert await KitsuHelper().get_anime_episodes(12) == items


async def test_get_anime_categories_returns_every_item_across_pages(kitsu_api) -> None:
    items = [{"id": str(number)} for number in range(25)]
    kitsu_api.routes["/anime/12/categories"] = Collection(items)
    assert await KitsuHelper().get_anime_categories(12) == items


async def test_get_anime_genres_returns_every_item_across_pages(kitsu_api) -> None:
    items = [{"id": str(number)} for number in range(25)]
    kitsu_api.routes["/anime/12/genres"] = Collection(items)
    assert await KitsuHelper().get_anime_genres(12) == items


async def test_get_anime_productions_names_companies_once_per_role(kitsu_api) -> None:
    def join_row(row_id: str, company_id: str, role: str) -> dict:
        return {
            "id": row_id,
            "attributes": {"role": role},
            "relationships": {"producer": {"data": {"id": company_id}}},
        }

    kitsu_api.routes["/anime/1376/anime-productions"] = Collection(
        [
            join_row("r1", "10", "producer"),
            join_row("r2", "10", "producer"),
            join_row("r3", "10", "licensor"),
            join_row("r4", "99", "studio"),
            {"id": "r5", "attributes": {"role": "studio"}},
        ],
        included=[
            {"id": "10", "type": "producers", "attributes": {"name": "VAP"}},
            {"id": "11", "type": "people", "attributes": {"name": "Not a company"}},
        ],
    )
    productions = await KitsuHelper().get_anime_productions(1376)
    assert [(p.name, p.role, p.company_id) for p in productions] == [
        ("VAP", "producer", "10"),
        ("VAP", "licensor", "10"),
    ]


async def test_get_anime_characters_attaches_sideloaded_characters(kitsu_api) -> None:
    missing = {
        "id": "mc2",
        "attributes": {"role": "supporting"},
        "relationships": {"character": {"data": {"id": "c999"}}},
    }
    kitsu_api.routes["/anime/12/characters"] = Collection(
        [LUFFY_LINK_ROW, missing], included=[LUFFY, {"id": "p1", "type": "people"}]
    )
    characters = await KitsuHelper().get_anime_characters(12)
    assert [link.id for link in characters] == ["mc1", "mc2"]
    assert characters[0].character.attributes.canonicalName == "Luffy"
    assert characters[1].character is None


async def test_get_character_voices_attaches_sideloaded_people(kitsu_api) -> None:
    kitsu_api.routes["/media-characters/mc1/voices"] = Collection(
        [
            {
                "id": "v1",
                "attributes": {"locale": "ja_jp"},
                "relationships": {"person": {"data": {"id": "p1"}}},
            },
            {
                "id": "v2",
                "attributes": {"locale": "en"},
                "relationships": {"person": {"data": {"id": "p404"}}},
            },
        ],
        included=[
            {"id": "p1", "type": "people", "attributes": {"name": "Mayumi Tanaka"}}
        ],
    )
    voices = await KitsuHelper().get_character_voices("mc1")
    assert voices[0].attributes.locale == "ja_jp"
    assert voices[0].person.attributes.name == "Mayumi Tanaka"
    assert voices[1].person is None


async def test_get_character_animeography_sets_media_and_media_type(kitsu_api) -> None:
    def appearance(entry_id: str, media_id: str, media_type: str) -> dict:
        return {
            "id": entry_id,
            "attributes": {"role": "main"},
            "relationships": {"media": {"data": {"id": media_id, "type": media_type}}},
        }

    kitsu_api.routes["/characters/c1/media-characters"] = Collection(
        [
            appearance("entry1", "m1", "anime"),
            appearance("entry2", "m2", "manga"),
            appearance("entry3", "m404", "anime"),
        ],
        included=[
            {
                "id": "m1",
                "type": "anime",
                "attributes": {"canonicalTitle": "One Piece"},
            },
            {
                "id": "m2",
                "type": "manga",
                "attributes": {"canonicalTitle": "One Piece Manga"},
            },
        ],
    )
    entries = await KitsuHelper().get_character_animeography("c1")
    assert [
        (entry.media_type, entry.media and entry.media.attributes.canonicalTitle)
        for entry in entries
    ] == [
        ("anime", "One Piece"),
        ("manga", "One Piece Manga"),
        ("anime", None),
    ]


async def test_fetch_anime_maps_anime_with_genres_themes_and_companies(
    kitsu_api, tmp_path: Path
) -> None:
    _one_piece_routes(kitsu_api.routes)
    kitsu_api.routes["/anime/12/categories"] = Collection(
        [
            {"id": "t1", "attributes": {"title": "Pirates", "description": "At sea"}},
            {"id": "t2", "attributes": {"title": None}},
        ]
    )
    kitsu_api.routes["/anime/12/anime-productions"] = Collection(
        [
            {
                "id": "r1",
                "attributes": {"role": "studio"},
                "relationships": {"producer": {"data": {"id": "18"}}},
            }
        ],
        included=[
            {"id": "18", "type": "producers", "attributes": {"name": "Toei Animation"}}
        ],
    )
    output = tmp_path / "kitsu_anime.jsonl"

    anime = await KitsuHelper().fetch_anime(12, output_path=str(output))

    assert (anime["title"], anime["genres"]) == ("One Piece", ["Action"])
    assert [(theme["name"], theme.get("description")) for theme in anime["themes"]] == [
        ("Pirates", "At sea")
    ]
    assert [company["name"] for company in anime["companies"]] == ["Toei Animation"]
    assert [json.loads(line) for line in output.read_text().splitlines()] == [anime]


async def test_fetch_anime_missing_anime_returns_none(kitsu_api) -> None:
    _one_piece_routes(kitsu_api.routes)
    del kitsu_api.routes["/anime/12"]
    assert await KitsuHelper().fetch_anime(12) is None


async def test_fetch_anime_cancelled_anime_request_returns_none(kitsu_api) -> None:
    _one_piece_routes(kitsu_api.routes)
    kitsu_api.routes["/anime/12"] = asyncio.CancelledError()
    assert await KitsuHelper().fetch_anime(12) is None


async def test_fetch_anime_cancelled_detail_requests_give_anime_without_those_details(
    kitsu_api,
) -> None:
    _one_piece_routes(kitsu_api.routes)
    for path in (
        "/anime/12/genres",
        "/anime/12/categories",
        "/anime/12/anime-productions",
    ):
        kitsu_api.routes[path] = asyncio.CancelledError()
    anime = await KitsuHelper().fetch_anime(12)
    assert anime["title"] == "One Piece"
    assert (anime["genres"], anime["themes"], anime["companies"]) == ([], [], [])


async def test_fetch_mappings_returns_every_mapping_as_stated(kitsu_api) -> None:
    kitsu_api.routes["/anime/186/mappings"] = Collection(
        [
            {
                "id": "1",
                "attributes": {
                    "externalSite": "myanimelist/anime",
                    "externalId": "210",
                },
            },
            {"id": "2", "attributes": {"externalSite": "anidb", "externalId": "193"}},
            {
                "id": "3",
                "attributes": {"externalSite": "aozora", "externalId": "5a3e0b"},
            },
            {
                "id": "4",
                "attributes": {"externalSite": "thetvdb/series", "externalId": "78463"},
            },
        ]
    )
    mappings = await KitsuHelper().fetch_mappings(186)
    assert [
        (mapping.attributes.external_site, mapping.attributes.external_id)
        for mapping in mappings
    ] == [
        ("myanimelist/anime", "210"),
        ("anidb", "193"),
        ("aozora", "5a3e0b"),
        ("thetvdb/series", "78463"),
    ]


async def test_fetch_mappings_without_mappings_returns_empty_list(kitsu_api) -> None:
    kitsu_api.routes["/anime/51111/mappings"] = Collection([])
    assert await KitsuHelper().fetch_mappings(51111) == []


async def test_fetch_episodes_maps_every_episode_and_saves_each(
    kitsu_api, tmp_path: Path
) -> None:
    kitsu_api.routes["/anime/12/episodes"] = Collection(
        [
            {"id": "e1", "attributes": {"number": 1, "canonicalTitle": "Episode 1"}},
            {"id": "e2", "attributes": {"number": 2, "canonicalTitle": "Episode 2"}},
        ]
    )
    output = tmp_path / "kitsu_episodes.jsonl"

    episodes = await KitsuHelper().fetch_episodes(
        12, anime_slug="one-piece", output_path=str(output)
    )

    assert [(episode["episode_number"], episode["title"]) for episode in episodes] == [
        (1, "Episode 1"),
        (2, "Episode 2"),
    ]
    assert episodes[0]["sources"] == ["https://kitsu.app/anime/one-piece/episodes/1"]
    assert [json.loads(line) for line in output.read_text().splitlines()] == episodes


async def test_fetch_characters_maps_characters_with_voices_and_saves_each(
    kitsu_api, tmp_path: Path
) -> None:
    _one_piece_routes(kitsu_api.routes)
    kitsu_api.routes["/media-characters/mc1/voices"] = Collection(
        [
            {
                "id": "v1",
                "attributes": {"locale": "ja_jp"},
                "relationships": {"person": {"data": {"id": "p1"}}},
            }
        ],
        included=[
            {"id": "p1", "type": "people", "attributes": {"name": "Mayumi Tanaka"}}
        ],
    )
    output = tmp_path / "characters.jsonl"

    characters = await KitsuHelper().fetch_characters(12, output_path=str(output))

    assert [(c["name"], c["roles"]) for c in characters] == [("Luffy", ["MAIN"])]
    assert [actor["name"] for actor in characters[0]["voice_actors"]] == [
        "Mayumi Tanaka"
    ]
    assert [json.loads(line) for line in output.read_text().splitlines()] == characters


async def test_fetch_characters_without_characters_returns_empty_list(
    kitsu_api,
) -> None:
    kitsu_api.routes["/anime/12/characters"] = Collection([])
    assert await KitsuHelper().fetch_characters(12) == []


async def test_fetch_characters_link_without_sideloaded_character_skipped(
    kitsu_api,
) -> None:
    kitsu_api.routes["/anime/12/characters"] = Collection([LUFFY_LINK_ROW])
    assert await KitsuHelper().fetch_characters(12) == []


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("fail"), asyncio.CancelledError()],
    ids=["error", "cancelled"],
)
async def test_fetch_characters_failed_voices_request_keeps_character(
    kitsu_api, failure: BaseException
) -> None:
    _one_piece_routes(kitsu_api.routes)
    kitsu_api.routes["/media-characters/mc1/voices"] = failure
    characters = await KitsuHelper().fetch_characters(12)
    assert [(c["name"], c["voice_actors"]) for c in characters] == [("Luffy", [])]


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("fail"), asyncio.CancelledError()],
    ids=["error", "cancelled"],
)
async def test_fetch_characters_failed_animeography_request_keeps_character(
    kitsu_api, failure: BaseException
) -> None:
    _one_piece_routes(kitsu_api.routes)
    kitsu_api.routes["/characters/c1/media-characters"] = failure
    characters = await KitsuHelper().fetch_characters(12)
    assert [(c["name"], c["animeography"]) for c in characters] == [("Luffy", [])]


async def test_fetch_characters_mapping_failure_drops_character(kitsu_api) -> None:
    _one_piece_routes(kitsu_api.routes)
    with patch.object(
        kitsu_helper,
        "character_from_kitsu",
        autospec=True,
        side_effect=ValueError("bad"),
    ):
        assert await KitsuHelper().fetch_characters(12) == []


async def test_fetch_all_numeric_link_returns_anime_episodes_and_characters(
    kitsu_api, tmp_path: Path
) -> None:
    _one_piece_routes(kitsu_api.routes)

    result = await KitsuHelper().fetch_all(
        {"kitsu_url": "https://kitsu.app/anime/12"}, {}, temp_dir=str(tmp_path)
    )

    assert result["anime"]["title"] == "One Piece"
    assert [episode["title"] for episode in result["episodes"]] == ["Ep 1"]
    assert result["episodes"][0]["sources"] == [
        "https://kitsu.app/anime/one-piece/episodes/1"
    ]
    assert [character["name"] for character in result["characters"]] == ["Luffy"]
    assert {path.name for path in tmp_path.iterdir()} == {
        "kitsu_anime.jsonl",
        "kitsu_episodes.jsonl",
        "kitsu_characters.jsonl",
    }
    assert kitsu_api.get_session.call_count == 1


async def test_fetch_all_slug_link_resolves_numeric_id_then_fetches(kitsu_api) -> None:
    _one_piece_routes(kitsu_api.routes)
    kitsu_api.routes["/anime?filter[slug]=one-piece"] = Collection([{"id": "12"}])

    result = await KitsuHelper().fetch_all(
        {"kitsu_url": "https://kitsu.app/anime/one-piece"}, {}
    )

    assert result["anime"]["title"] == "One Piece"


async def test_fetch_all_unknown_slug_returns_none(kitsu_api) -> None:
    kitsu_api.routes["/anime?filter[slug]=one-piece"] = Collection([])
    assert (
        await KitsuHelper().fetch_all(
            {"kitsu_url": "https://kitsu.app/anime/one-piece"}, {}
        )
        is None
    )


async def test_fetch_all_slug_lookup_failure_returns_none(kitsu_api) -> None:
    kitsu_api.routes["/anime?filter[slug]=one-piece"] = 503
    assert (
        await KitsuHelper().fetch_all(
            {"kitsu_url": "https://kitsu.app/anime/one-piece"}, {}
        )
        is None
    )


async def test_fetch_all_without_kitsu_link_returns_none(kitsu_api) -> None:
    assert await KitsuHelper().fetch_all({}, {}) is None
    kitsu_api.session.get.assert_not_called()


async def test_fetch_all_missing_anime_returns_none(kitsu_api) -> None:
    assert (
        await KitsuHelper().fetch_all(
            {"kitsu_url": "https://kitsu.app/anime/99999"}, {}
        )
        is None
    )


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("fail"), asyncio.CancelledError()],
    ids=["error", "cancelled"],
)
async def test_fetch_all_failed_episode_fetch_gives_empty_episodes(
    kitsu_api, failure: BaseException
) -> None:
    _one_piece_routes(kitsu_api.routes)
    with patch.object(
        KitsuHelper, "fetch_episodes", autospec=True, side_effect=failure
    ):
        result = await KitsuHelper().fetch_all(
            {"kitsu_url": "https://kitsu.app/anime/12"}, {}
        )
    assert result["episodes"] == []
    assert [character["name"] for character in result["characters"]] == ["Luffy"]


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("fail"), asyncio.CancelledError()],
    ids=["error", "cancelled"],
)
async def test_fetch_all_failed_character_fetch_gives_empty_characters(
    kitsu_api, failure: BaseException
) -> None:
    _one_piece_routes(kitsu_api.routes)
    with patch.object(
        KitsuHelper, "fetch_characters", autospec=True, side_effect=failure
    ):
        result = await KitsuHelper().fetch_all(
            {"kitsu_url": "https://kitsu.app/anime/12"}, {}
        )
    assert result["characters"] == []
    assert [episode["title"] for episode in result["episodes"]] == ["Ep 1"]


async def test_fetch_all_session_failure_returns_none(kitsu_api) -> None:
    kitsu_api.get_session.side_effect = RuntimeError("session failed")
    assert (
        await KitsuHelper().fetch_all({"kitsu_url": "https://kitsu.app/anime/12"}, {})
        is None
    )


async def test_fetch_all_episodes_and_characters_turned_off_fetches_neither(
    kitsu_api,
) -> None:
    _one_piece_routes(kitsu_api.routes)

    result = await KitsuHelper().fetch_all(
        {"kitsu_url": "https://kitsu.app/anime/12"},
        {},
        fetch_episodes=False,
        fetch_characters=False,
    )

    assert (result["episodes"], result["characters"]) == ([], [])
    requested = {call.args[0] for call in kitsu_api.session.get.call_args_list}
    assert not {f"{API}/anime/12/episodes", f"{API}/anime/12/characters"} & requested


async def test_close_completes_without_error() -> None:
    assert await KitsuHelper().close() is None


async def test_aenter_returns_helper_and_aexit_lets_errors_through() -> None:
    with pytest.raises(ValueError, match="Test error"):
        async with KitsuHelper() as helper:
            assert isinstance(helper, KitsuHelper)
            raise ValueError("Test error")


async def test_main_found_anime_writes_output_and_returns_zero(
    kitsu_api, tmp_path: Path
) -> None:
    _one_piece_routes(kitsu_api.routes)
    output = tmp_path / "output.json"
    with patch.object(
        sys,
        "argv",
        ["script.py", "https://kitsu.app/anime/12", "--output", str(output)],
    ):
        assert await main() == 0
    assert json.loads(output.read_text())["anime"]["title"] == "One Piece"


async def test_main_missing_anime_returns_one(kitsu_api, tmp_path: Path) -> None:
    output = tmp_path / "output.json"
    with patch.object(
        sys,
        "argv",
        ["script.py", "https://kitsu.app/anime/99999", "--output", str(output)],
    ):
        assert await main() == 1
    assert not output.exists()


async def test_main_fetch_error_returns_one(tmp_path: Path) -> None:
    output = tmp_path / "output.json"
    with (
        patch.object(
            sys,
            "argv",
            ["script.py", "https://kitsu.app/anime/12", "--output", str(output)],
        ),
        patch.object(
            KitsuHelper,
            "fetch_all",
            autospec=True,
            side_effect=RuntimeError("API error"),
        ),
    ):
        assert await main() == 1


async def test_main_without_link_argument_returns_one() -> None:
    with patch.object(sys, "argv", ["script.py"]):
        assert await main() == 1
