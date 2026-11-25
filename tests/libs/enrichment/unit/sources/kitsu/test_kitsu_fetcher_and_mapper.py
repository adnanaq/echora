"""Unit tests for KitsuFetcher and response-level Kitsu mappers."""

import json
from datetime import timedelta
from typing import Any
from unittest.mock import MagicMock

import pytest
from enrichment.sources.kitsu.kitsu_fetcher import KitsuFetcher
from enrichment.sources.kitsu.kitsu_mapper import (
    map_kitsu_anime_episode_responses,
    map_kitsu_anime_responses,
)
from shared_proto.v1.anime_pb2 import EntityType
from shared_proto.v1.merge_pb2 import PageKey


def _mock_get_response(
    status: int,
    payload: dict[str, Any],
    *,
    from_cache: bool = True,
) -> MagicMock:
    response = MagicMock()
    response.status = status
    response.headers = {"Content-Type": "application/vnd.api+json"}
    response.from_cache = from_cache
    response.url = "https://kitsu.io/api/edge/mock"
    raw_bytes = json.dumps(payload).encode("utf-8")

    async def _read() -> bytes:
        return raw_bytes

    response.read = _read
    response.raise_for_status = MagicMock()
    cm = MagicMock()

    async def _aenter(_self: object) -> MagicMock:
        return response

    async def _aexit(_self: object, *_args: object) -> bool:
        return False

    cm.__aenter__ = _aenter
    cm.__aexit__ = _aexit
    return cm


def _mock_kitsu_cache(
    routes: dict[str, dict[str, Any]],
    *,
    not_found_paths: set[str] | None = None,
) -> tuple[MagicMock, list[dict[str, Any]]]:
    recorded: list[dict[str, Any]] = []
    not_found = not_found_paths or set()

    def _make_session() -> MagicMock:
        session = MagicMock()

        def _get(
            url: str, *, headers: dict[str, str], params: dict[str, Any] | None = None
        ) -> MagicMock:
            recorded.append({"url": url, "headers": headers, "params": params})
            path = url.removeprefix("https://kitsu.io/api/edge")
            if path in not_found:
                return _mock_get_response(404, {"errors": [{"status": "404"}]})
            return _mock_get_response(200, routes[path])

        session.get = _get

        async def _aenter(_self: object) -> MagicMock:
            return session

        async def _aexit(_self: object, *_args: object) -> bool:
            return False

        ctx = MagicMock()
        ctx.__aenter__ = _aenter
        ctx.__aexit__ = _aexit
        return ctx

    cache = MagicMock()
    cache.get_aiohttp_session.side_effect = lambda *_a, **_kw: _make_session()
    return cache, recorded


@pytest.mark.asyncio
async def test_kitsu_fetcher_fetches_raw_anime_pages_and_maps_canonically() -> None:
    routes = {
        "/anime/12": {
            "data": {
                "id": "12",
                "type": "anime",
                "attributes": {
                    "slug": "one-piece",
                    "canonicalTitle": "One Piece",
                    "subtype": "TV",
                    "status": "current",
                },
            }
        },
        "/anime/12/genres": {
            "data": [{"id": "1", "type": "genres", "attributes": {"name": "Action"}}],
            "meta": {"count": 1},
        },
        "/anime/12/categories": {
            "data": [
                {
                    "id": "2",
                    "type": "categories",
                    "attributes": {"title": "Pirates", "description": "Sea"},
                }
            ],
            "meta": {"count": 1},
        },
        "/anime/12/anime-productions": {
            "data": [],
            "included": [],
            "meta": {"count": 0},
        },
        "/anime/12/mappings": {
            "data": [
                {
                    "id": "3",
                    "type": "mappings",
                    "attributes": {
                        "externalSite": "myanimelist/anime",
                        "externalId": "21",
                    },
                }
            ],
            "meta": {"count": 1},
        },
        "/anime/12/episodes": {
            "data": [
                {
                    "id": "101",
                    "type": "episodes",
                    "attributes": {"number": 1, "canonicalTitle": "I'm Luffy!"},
                }
            ],
            "meta": {"count": 1},
        },
        "/anime/12/characters": {
            "data": [
                {
                    "id": "501",
                    "type": "mediaCharacters",
                    "attributes": {"role": "main"},
                    "relationships": {
                        "character": {"data": {"id": "40", "type": "characters"}}
                    },
                }
            ],
            "included": [
                {
                    "id": "40",
                    "type": "characters",
                    "attributes": {"name": "Luffy", "slug": "luffy"},
                }
            ],
            "meta": {"count": 1},
        },
    }
    cache, calls = _mock_kitsu_cache(routes)
    fetcher = KitsuFetcher(cache)
    key = PageKey(
        provider="kitsu", entity_type=EntityType.ENTITY_TYPE_ANIME, address="12"
    )

    fetched = await fetcher.fetch_page(key, max_age=timedelta(seconds=120))

    assert fetched.is_not_found is False
    names = {r.request_name for r in fetched.responses}
    assert {
        "main",
        "genres_page_1",
        "categories_page_1",
        "productions_page_1",
        "mappings_page_1",
        "episodes_page_1",
        "characters_page_1",
    }.issubset(names)
    assert calls[0]["headers"]["Cache-Control"] == "max-age=120"

    canonical_anime = map_kitsu_anime_responses(fetched.responses)
    assert canonical_anime["title"] == "One Piece"
    assert canonical_anime["genres"] == ["Action"]
    assert canonical_anime["themes"][0]["name"] == "Pirates"
    canonical_episodes = map_kitsu_anime_episode_responses(fetched.responses)
    assert len(canonical_episodes) == 1
    assert canonical_episodes[0]["title"] == "I'm Luffy!"


@pytest.mark.asyncio
async def test_kitsu_fetcher_reports_not_found_on_404() -> None:
    cache, _ = _mock_kitsu_cache({}, not_found_paths={"/anime/999999"})
    fetcher = KitsuFetcher(cache)
    key = PageKey(
        provider="kitsu", entity_type=EntityType.ENTITY_TYPE_ANIME, address="999999"
    )

    fetched = await fetcher.fetch_page(key)

    assert fetched.is_not_found is True
    assert fetched.responses == ()


@pytest.mark.asyncio
async def test_kitsu_fetcher_fetches_character_page() -> None:
    routes = {
        "/characters/40": {
            "data": {
                "id": "40",
                "type": "characters",
                "attributes": {"name": "Luffy", "slug": "luffy"},
            }
        },
        "/characters/40/media-characters": {
            "data": [
                {
                    "id": "501",
                    "type": "mediaCharacters",
                    "attributes": {"role": "main"},
                    "relationships": {"media": {"data": {"id": "12", "type": "anime"}}},
                }
            ],
            "included": [
                {
                    "id": "12",
                    "type": "anime",
                    "attributes": {"canonicalTitle": "One Piece"},
                }
            ],
            "meta": {"count": 1},
        },
        "/media-characters/501/voices": {
            "data": [],
            "included": [],
            "meta": {"count": 0},
        },
    }
    cache, _ = _mock_kitsu_cache(routes)
    fetcher = KitsuFetcher(cache)
    key = PageKey(
        provider="kitsu", entity_type=EntityType.ENTITY_TYPE_CHARACTER, address="40"
    )

    fetched = await fetcher.fetch_page(key)

    assert fetched.is_not_found is False
    names = [r.request_name for r in fetched.responses]
    assert "main" in names
    assert "media_characters_page_1" in names
    assert "voices_501_page_1" in names


@pytest.mark.asyncio
async def test_kitsu_fetcher_resolves_slug_and_handles_missing_slug() -> None:
    routes = {
        "/anime": {"data": [{"id": "12", "type": "anime"}]},
        "/anime/12": {
            "data": {
                "id": "12",
                "type": "anime",
                "attributes": {"slug": "one-piece", "canonicalTitle": "One Piece"},
            }
        },
        "/anime/12/genres": {"data": [], "meta": {"count": 0}},
        "/anime/12/categories": {"data": [], "meta": {"count": 0}},
        "/anime/12/anime-productions": {
            "data": [],
            "included": [],
            "meta": {"count": 0},
        },
        "/anime/12/mappings": {"data": [], "meta": {"count": 0}},
        "/anime/12/episodes": {"data": [], "meta": {"count": 0}},
        "/anime/12/characters": {"data": [], "included": [], "meta": {"count": 0}},
    }
    cache, _ = _mock_kitsu_cache(routes)
    fetched = await KitsuFetcher(cache).fetch_page(
        PageKey(
            provider="kitsu",
            entity_type=EntityType.ENTITY_TYPE_ANIME,
            address="one-piece",
        )
    )
    assert fetched.is_not_found is False
    assert map_kitsu_anime_responses(fetched.responses)["title"] == "One Piece"

    missing_cache, _ = _mock_kitsu_cache({"/anime": {"data": []}})
    missing = await KitsuFetcher(missing_cache).fetch_page(
        PageKey(
            provider="kitsu",
            entity_type=EntityType.ENTITY_TYPE_ANIME,
            address="no-such-anime",
        )
    )
    assert missing.is_not_found is True
    assert missing.responses == ()


@pytest.mark.asyncio
async def test_kitsu_fetcher_reports_not_found_for_missing_character() -> None:
    cache, _ = _mock_kitsu_cache({}, not_found_paths={"/characters/999999"})
    fetched = await KitsuFetcher(cache).fetch_page(
        PageKey(
            provider="kitsu",
            entity_type=EntityType.ENTITY_TYPE_CHARACTER,
            address="999999",
        )
    )
    assert fetched.is_not_found is True
    assert fetched.responses == ()
