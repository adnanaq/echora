import json
from datetime import timedelta
from unittest.mock import create_autospec

import aiohttp
import pytest
from enrichment.sources.animeschedule.animeschedule_fetcher import (
    AnimeScheduleFetcher,
)
from enrichment.sources.base.exceptions import (
    ServiceNetworkError,
    ServiceNotConfiguredError,
)
from http_cache import HTTPCacheManager
from pydantic import SecretStr
from shared_proto.v1.anime_pb2 import EntityType
from shared_proto.v1.merge_pb2 import PageKey


def _build_animeschedule_cache(
    status: int,
    payload: dict[str, object],
    *,
    get_error: Exception | None = None,
) -> tuple[HTTPCacheManager, list[dict[str, object]]]:
    recorded_calls: list[dict[str, object]] = []
    response = create_autospec(aiohttp.ClientResponse, instance=True)
    response.status = status
    response.headers = {"Content-Type": "application/json; charset=utf-8"}
    response.read.return_value = json.dumps(payload).encode("utf-8")
    response.__aenter__.return_value = response
    response.__aexit__.return_value = False

    session = create_autospec(aiohttp.ClientSession, instance=True)
    session.__aenter__.return_value = session
    session.__aexit__.return_value = False

    def _record_get(url: str, *, headers: dict[str, str]) -> aiohttp.ClientResponse:
        recorded_calls.append({"url": url, "headers": dict(headers)})
        if get_error is not None:
            raise get_error
        return response

    session.get.side_effect = _record_get

    cache = create_autospec(HTTPCacheManager, instance=True)
    cache.get_aiohttp_session.return_value = session
    return cache, recorded_calls


@pytest.mark.asyncio
async def test_animeschedule_fetcher_anime_slug_returns_raw_json_with_headers() -> None:
    payload = {
        "id": "as-1",
        "title": "Cowboy Bebop",
        "route": "cowboy-bebop",
        "episodes": 26,
        "websites": {"mal": "myanimelist.net/anime/1"},
    }
    cache, calls = _build_animeschedule_cache(200, payload)
    fetcher = AnimeScheduleFetcher(cache, SecretStr("test-token"))
    page_key = PageKey(
        provider="animeschedule",
        entity_type=EntityType.ENTITY_TYPE_ANIME,
        address="cowboy-bebop",
    )

    fetched = await fetcher.fetch_page(page_key, max_age=timedelta(seconds=60))

    assert fetched.is_not_found is False
    assert [response.request_name for response in fetched.responses] == ["main"]
    assert fetched.responses[0].content_type == "application/json"
    assert calls[0]["url"] == "https://animeschedule.net/api/v3/anime/cowboy-bebop"
    assert calls[0]["headers"] == {
        "Authorization": "Bearer test-token",
        "Cache-Control": "max-age=60",
    }


@pytest.mark.asyncio
async def test_animeschedule_fetcher_http_404_returns_not_found() -> None:
    cache, _ = _build_animeschedule_cache(404, {"message": "Not Found"})
    fetcher = AnimeScheduleFetcher(cache, SecretStr("test-token"))

    fetched = await fetcher.fetch_page(
        PageKey(
            provider="animeschedule",
            entity_type=EntityType.ENTITY_TYPE_ANIME,
            address="missing-anime",
        )
    )

    assert fetched.is_not_found is True
    assert fetched.responses == ()


@pytest.mark.asyncio
async def test_animeschedule_fetcher_missing_token_raises_not_configured() -> None:
    cache, _ = _build_animeschedule_cache(200, {})
    fetcher = AnimeScheduleFetcher(cache, None)

    with pytest.raises(ServiceNotConfiguredError):
        await fetcher.fetch_page(
            PageKey(
                provider="animeschedule",
                entity_type=EntityType.ENTITY_TYPE_ANIME,
                address="cowboy-bebop",
            )
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "transport_error",
    [TimeoutError("request timed out"), aiohttp.ClientError("connection reset")],
)
async def test_animeschedule_fetcher_transport_error_raises_network_error(
    transport_error: Exception,
) -> None:
    cache, _ = _build_animeschedule_cache(200, {}, get_error=transport_error)
    fetcher = AnimeScheduleFetcher(cache, SecretStr("test-token"))

    with pytest.raises(ServiceNetworkError):
        await fetcher.fetch_page(
            PageKey(
                provider="animeschedule",
                entity_type=EntityType.ENTITY_TYPE_ANIME,
                address="cowboy-bebop",
            )
        )
