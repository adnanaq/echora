"""Unit tests for AniListFetcher and response-level AniList mappers."""

import json
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from enrichment.sources.anilist.anilist_fetcher import AniListFetcher
from enrichment.sources.anilist.anilist_mapper import (
    map_anilist_anime_character_responses,
    map_anilist_anime_responses,
)
from enrichment.sources.base.exceptions import (
    AniListGraphQLError,
    ServiceBlockedError,
    ServiceRateLimitedError,
)
from shared_proto.v1.anime_pb2 import EntityType
from shared_proto.v1.merge_pb2 import PageKey, PageResponse


def _mock_response(
    status: int,
    payload: dict[str, object],
    *,
    headers: dict[str, str] | None = None,
    from_cache: bool = False,
) -> MagicMock:
    response = MagicMock()
    response.status = status
    response.headers = headers or {"Content-Type": "application/json; charset=utf-8"}
    response.from_cache = from_cache
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


def _mock_cache(
    post_responses: list[MagicMock],
) -> tuple[MagicMock, list[dict[str, object]]]:
    recorded_calls: list[dict[str, object]] = []
    session = MagicMock()

    def _post(
        url: str, *, json: dict[str, object], headers: dict[str, str]
    ) -> MagicMock:
        recorded_calls.append({"url": url, "json": json, "headers": headers})
        return post_responses.pop(0)

    session.post = _post
    cache = MagicMock()
    cache.get_aiohttp_session.return_value = session
    return cache, recorded_calls


@pytest.mark.asyncio
async def test_anilist_fetcher_fetches_raw_anime_and_character_pages() -> None:
    main_payload = {
        "data": {
            "Media": {
                "id": 21,
                "title": {"romaji": "ONE PIECE", "english": "One Piece"},
                "format": "TV",
                "status": "RELEASING",
            }
        }
    }
    chars_payload_1 = {
        "data": {
            "Media": {
                "characters": {
                    "pageInfo": {"hasNextPage": True},
                    "edges": [
                        {
                            "node": {"id": 40, "name": {"full": "Monkey D. Luffy"}},
                            "role": "MAIN",
                            "voiceActorRoles": [],
                        }
                    ],
                }
            }
        }
    }
    chars_payload_2 = {
        "data": {
            "Media": {
                "characters": {
                    "pageInfo": {"hasNextPage": False},
                    "edges": [
                        {
                            "node": {"id": 62, "name": {"full": "Roronoa Zoro"}},
                            "role": "MAIN",
                            "voiceActorRoles": [],
                        }
                    ],
                }
            }
        }
    }
    cache, calls = _mock_cache(
        [
            _mock_response(200, main_payload, from_cache=True),
            _mock_response(200, chars_payload_1, from_cache=True),
            _mock_response(200, chars_payload_2, from_cache=True),
        ]
    )
    fetcher = AniListFetcher(cache)
    key = PageKey(
        provider="anilist", entity_type=EntityType.ENTITY_TYPE_ANIME, address="21"
    )

    fetched = await fetcher.fetch_page(key, max_age=timedelta(seconds=0))

    assert fetched.is_not_found is False
    assert [r.request_name for r in fetched.responses] == [
        "main",
        "characters_page_1",
        "characters_page_2",
    ]
    assert fetched.responses[0].content_type == "application/json"
    assert json.loads(fetched.responses[0].body) == main_payload
    assert calls[0]["headers"]["Cache-Control"] == "no-cache"  # type: ignore[index]

    canonical_anime = map_anilist_anime_responses(fetched.responses)
    assert canonical_anime["title"] == "ONE PIECE"
    canonical_chars = map_anilist_anime_character_responses(fetched.responses)
    assert [c["name"] for c in canonical_chars] == [
        "Monkey D. Luffy",
        "Roronoa Zoro",
    ]


@pytest.mark.asyncio
async def test_anilist_fetcher_reports_not_found_on_404() -> None:
    cache, _ = _mock_cache(
        [_mock_response(404, {"errors": [{"message": "Not Found"}]})]
    )
    fetcher = AniListFetcher(cache)
    key = PageKey(
        provider="anilist", entity_type=EntityType.ENTITY_TYPE_ANIME, address="99999999"
    )

    fetched = await fetcher.fetch_page(key)

    assert fetched.is_not_found is True
    assert fetched.responses == ()


@pytest.mark.asyncio
async def test_anilist_fetcher_fetches_character_page() -> None:
    char_payload = {
        "data": {
            "Character": {
                "id": 40,
                "name": {"full": "Monkey D. Luffy"},
                "media": {"pageInfo": {"hasNextPage": False}, "edges": []},
            }
        }
    }
    cache, _ = _mock_cache([_mock_response(200, char_payload, from_cache=True)])
    fetcher = AniListFetcher(cache)
    key = PageKey(
        provider="anilist", entity_type=EntityType.ENTITY_TYPE_CHARACTER, address="40"
    )

    fetched = await fetcher.fetch_page(key)

    assert fetched.is_not_found is False
    assert [r.request_name for r in fetched.responses] == ["main"]
    assert json.loads(fetched.responses[0].body) == char_payload


def test_map_anilist_anime_responses_requires_main_response() -> None:
    with pytest.raises(ValueError, match="main"):
        map_anilist_anime_responses(
            [
                PageResponse(
                    request_name="other", content_type="application/json", body=b"{}"
                )
            ]
        )


@pytest.mark.asyncio
async def test_anilist_fetcher_handles_null_media_and_error_statuses() -> None:
    cache_null, _ = _mock_cache([_mock_response(200, {"data": {"Media": None}})])
    key = PageKey(
        provider="anilist", entity_type=EntityType.ENTITY_TYPE_ANIME, address="21"
    )
    fetched_null = await AniListFetcher(cache_null).fetch_page(key)
    assert fetched_null.is_not_found is True
    assert fetched_null.responses == ()

    cache_403, _ = _mock_cache([_mock_response(403, {})])
    with pytest.raises(ServiceBlockedError):
        await AniListFetcher(cache_403).fetch_page(key)

    cache_gql, _ = _mock_cache([_mock_response(200, {"errors": [{"message": "bad"}]})])
    with pytest.raises(AniListGraphQLError):
        await AniListFetcher(cache_gql).fetch_page(key)


@pytest.mark.asyncio
async def test_anilist_fetcher_retries_429_and_closes_session() -> None:
    cache_429, _ = _mock_cache(
        [
            _mock_response(429, {}, headers={"Retry-After": "1"}),
            _mock_response(429, {}, headers={"Retry-After": "1"}),
            _mock_response(429, {}, headers={"Retry-After": "1"}),
        ]
    )
    key = PageKey(
        provider="anilist", entity_type=EntityType.ENTITY_TYPE_ANIME, address="21"
    )
    fetcher = AniListFetcher(cache_429)
    with (
        patch(
            "enrichment.sources.anilist.anilist_fetcher.asyncio.sleep",
            new=AsyncMock(),
        ),
        pytest.raises(ServiceRateLimitedError),
    ):
        await fetcher.fetch_page(key)

    cache_429.get_aiohttp_session.return_value.close = AsyncMock()
    await fetcher.close()
    assert fetcher.session is None
