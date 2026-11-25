"""Unit tests for AnimeScheduleFetcher and response-level AnimeSchedule mapper."""

import json
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from enrichment.sources.animeschedule.animeschedule_fetcher import (
    AnimeScheduleFetcher,
)
from enrichment.sources.animeschedule.animeschedule_mapper import (
    map_animeschedule_anime_responses,
)
from enrichment.sources.base.exceptions import ServiceNotConfiguredError
from pydantic import SecretStr
from shared_proto.v1.anime_pb2 import EntityType
from shared_proto.v1.merge_pb2 import PageKey, PageResponse


def _mock_as_cache(
    status: int, payload: dict[str, object]
) -> tuple[MagicMock, list[dict[str, object]]]:
    recorded: list[dict[str, object]] = []
    response = MagicMock()
    response.status = status
    response.headers = {"Content-Type": "application/json; charset=utf-8"}
    raw = json.dumps(payload).encode("utf-8")
    response.read = AsyncMock(return_value=raw)
    response.raise_for_status = MagicMock()

    resp_cm = MagicMock()

    async def _resp_aenter(_self: object) -> MagicMock:
        return response

    async def _resp_aexit(_self: object, *_args: object) -> bool:
        return False

    resp_cm.__aenter__ = _resp_aenter
    resp_cm.__aexit__ = _resp_aexit

    session = MagicMock()

    def _get(url: str, *, headers: dict[str, str]) -> MagicMock:
        recorded.append({"url": url, "headers": headers})
        return resp_cm

    session.get = _get
    sess_cm = MagicMock()

    async def _sess_aenter(_self: object) -> MagicMock:
        return session

    async def _sess_aexit(_self: object, *_args: object) -> bool:
        return False

    sess_cm.__aenter__ = _sess_aenter
    sess_cm.__aexit__ = _sess_aexit

    cache = MagicMock()
    cache.get_aiohttp_session.return_value = sess_cm
    return cache, recorded


@pytest.mark.asyncio
async def test_animeschedule_fetcher_fetches_raw_json_and_maps() -> None:
    payload = {
        "id": "as-1",
        "title": "Cowboy Bebop",
        "route": "cowboy-bebop",
        "episodes": 26,
        "websites": {"mal": "myanimelist.net/anime/1"},
    }
    cache, calls = _mock_as_cache(200, payload)
    fetcher = AnimeScheduleFetcher(cache, SecretStr("test-token"))
    key = PageKey(
        provider="animeschedule",
        entity_type=EntityType.ENTITY_TYPE_ANIME,
        address="cowboy-bebop",
    )

    fetched = await fetcher.fetch_page(key, max_age=timedelta(seconds=60))

    assert fetched.is_not_found is False
    assert [r.request_name for r in fetched.responses] == ["main"]
    assert calls[0]["url"] == "https://animeschedule.net/api/v3/anime/cowboy-bebop"
    assert calls[0]["headers"]["Authorization"] == "Bearer test-token"  # type: ignore[index]
    assert calls[0]["headers"]["Cache-Control"] == "max-age=60"  # type: ignore[index]

    canonical = map_animeschedule_anime_responses(fetched.responses)
    assert canonical["title"] == "Cowboy Bebop"
    assert canonical["episode_count"] == 26


@pytest.mark.asyncio
async def test_animeschedule_fetcher_reports_not_found_and_requires_token() -> None:
    cache_404, _ = _mock_as_cache(404, {"message": "Not Found"})
    fetcher = AnimeScheduleFetcher(cache_404, SecretStr("test-token"))
    fetched = await fetcher.fetch_page(
        PageKey(
            provider="animeschedule",
            entity_type=EntityType.ENTITY_TYPE_ANIME,
            address="missing-anime",
        )
    )
    assert fetched.is_not_found is True
    assert fetched.responses == ()

    unconfigured = AnimeScheduleFetcher(cache_404, None)
    with pytest.raises(ServiceNotConfiguredError):
        await unconfigured.fetch_page(
            PageKey(
                provider="animeschedule",
                entity_type=EntityType.ENTITY_TYPE_ANIME,
                address="cowboy-bebop",
            )
        )


def test_map_animeschedule_anime_responses_requires_main() -> None:
    with pytest.raises(ValueError, match="main"):
        map_animeschedule_anime_responses(
            [
                PageResponse(
                    request_name="other", content_type="application/json", body=b"{}"
                )
            ]
        )
