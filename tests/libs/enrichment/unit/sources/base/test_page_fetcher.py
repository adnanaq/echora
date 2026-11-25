"""Unit tests for shared page fetcher abstractions and helpers."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from enrichment.sources.base.page_fetcher import (
    FetchedPage,
    build_cache_headers,
    collect_paginated_responses,
    index_responses_by_name,
    parse_json_response,
    read_response_bytes,
    validate_page_key,
)
from shared_proto.v1.anime_pb2 import EntityType
from shared_proto.v1.merge_pb2 import PageKey, PageResponse


def test_fetched_page_to_stored_page_preserves_fields() -> None:
    key = PageKey(
        provider="anilist", entity_type=EntityType.ENTITY_TYPE_ANIME, address="21"
    )
    response = PageResponse(
        request_name="main",
        content_type="application/json",
        body=b'{"data": {"Media": {"id": 21}}}',
    )
    fetched = FetchedPage(page_key=key, responses=(response,), is_not_found=False)
    fetched_at = datetime(2026, 11, 24, 12, 0, tzinfo=UTC)

    stored = fetched.to_stored_page(fetched_at=fetched_at)

    assert stored.page_key == key
    assert list(stored.responses) == [response]
    assert stored.is_not_found is False
    assert stored.is_dead is False
    assert stored.fetched_at.seconds == int(fetched_at.timestamp())


def test_build_cache_headers_maps_max_age() -> None:
    assert build_cache_headers(None) == {}
    assert build_cache_headers(timedelta(0)) == {"Cache-Control": "no-cache"}
    assert build_cache_headers(timedelta(seconds=-5)) == {"Cache-Control": "no-cache"}
    assert build_cache_headers(timedelta(minutes=5)) == {"Cache-Control": "max-age=300"}


def test_validate_page_key_enforces_provider_type_and_address() -> None:
    valid = PageKey(
        provider="anilist", entity_type=EntityType.ENTITY_TYPE_ANIME, address="21"
    )
    validate_page_key(
        valid,
        "anilist",
        frozenset({EntityType.ENTITY_TYPE_ANIME}),
        numeric_address=True,
    )

    with pytest.raises(ValueError, match="provider"):
        validate_page_key(
            PageKey(
                provider="kitsu", entity_type=EntityType.ENTITY_TYPE_ANIME, address="21"
            ),
            "anilist",
            frozenset({EntityType.ENTITY_TYPE_ANIME}),
        )

    with pytest.raises(ValueError, match="entity_type"):
        validate_page_key(
            PageKey(
                provider="anilist",
                entity_type=EntityType.ENTITY_TYPE_EPISODE,
                address="21",
            ),
            "anilist",
            frozenset({EntityType.ENTITY_TYPE_ANIME}),
        )

    with pytest.raises(ValueError, match="numeric"):
        validate_page_key(
            PageKey(
                provider="anilist",
                entity_type=EntityType.ENTITY_TYPE_ANIME,
                address="one-piece",
            ),
            "anilist",
            frozenset({EntityType.ENTITY_TYPE_ANIME}),
            numeric_address=True,
        )

    with pytest.raises(ValueError, match="address"):
        validate_page_key(
            PageKey(
                provider="kitsu",
                entity_type=EntityType.ENTITY_TYPE_ANIME,
                address="bad/slug",
            ),
            "kitsu",
            frozenset({EntityType.ENTITY_TYPE_ANIME}),
        )


def test_index_and_parse_json_response() -> None:
    main = PageResponse(
        request_name="main",
        content_type="application/json",
        body=b'{"id": 1}',
    )
    indexed = index_responses_by_name([main])
    assert indexed["main"] == main
    assert parse_json_response(main) == {"id": 1}


@pytest.mark.asyncio
async def test_read_response_bytes_preserves_raw_body_and_media_type() -> None:
    response = MagicMock()
    response.read = AsyncMock(return_value=b'{"raw": true}')
    response.headers = {"Content-Type": "application/vnd.api+json; charset=utf-8"}

    body, content_type = await read_response_bytes(response)

    assert body == b'{"raw": true}'
    assert content_type == "application/vnd.api+json"


def test_collect_paginated_responses_orders_numerically() -> None:
    pages = [
        PageResponse(
            request_name="characters_page_10",
            content_type="application/json",
            body=b'{"p": 10}',
        ),
        PageResponse(
            request_name="characters_page_2",
            content_type="application/json",
            body=b'{"p": 2}',
        ),
        PageResponse(
            request_name="characters_page_1",
            content_type="application/json",
            body=b'{"p": 1}',
        ),
    ]
    assert collect_paginated_responses(pages, "characters_page_") == [
        {"p": 1},
        {"p": 2},
        {"p": 10},
    ]
