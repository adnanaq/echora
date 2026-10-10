from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any
from unittest.mock import create_autospec, patch

import aiohttp
import pytest
import yarl
from common.config import EnrichmentConfig
from enrichment.sources.anidb.anidb_helper import AniDBHelper
from enrichment.sources.anilist.anilist_helper import AniListHelper
from enrichment.sources.animeschedule.animeschedule_helper import AnimescheduleHelper
from enrichment.sources.kitsu.kitsu_helper import KitsuHelper
from http_cache import HTTPCacheManager
from http_cache.aiohttp_adapter import CachedAiohttpSession
from redis.asyncio import Redis

pytestmark = pytest.mark.integration


def _not_found_session() -> CachedAiohttpSession:
    response = create_autospec(aiohttp.ClientResponse, instance=True)
    response.status = 404
    response.url = yarl.URL("https://example.com")
    response.from_cache = False
    response.json.return_value = {}
    session = create_autospec(CachedAiohttpSession, instance=True)
    session.get.return_value.__aenter__.return_value = response
    session.post.return_value.__aenter__.return_value = response
    session.__aenter__.return_value = session
    return session


async def _open_anilist(cache: HTTPCacheManager) -> None:
    await AniListHelper(cache)._ensure_session()


async def _open_anidb(cache: HTTPCacheManager) -> None:
    await AniDBHelper(EnrichmentConfig(), cache)._ensure_session_health()


async def _open_kitsu(cache: HTTPCacheManager) -> None:
    await KitsuHelper(cache).get_anime_by_id(1)


async def _open_animeschedule(cache: HTTPCacheManager) -> None:
    await AnimescheduleHelper(cache)._search("Test Anime")


@pytest.mark.parametrize(
    ("open_session", "service"),
    [
        (_open_anilist, "anilist"),
        (_open_anidb, "anidb"),
        (_open_kitsu, "kitsu"),
        (_open_animeschedule, "animeschedule"),
    ],
)
async def test_helper_session_comes_from_given_cache_under_its_service(
    open_session: Callable[[HTTPCacheManager], Awaitable[Any]], service: str
) -> None:
    cache = create_autospec(HTTPCacheManager, instance=True)
    cache.get_aiohttp_session.return_value = _not_found_session()

    with patch.object(Redis, "from_url", autospec=True) as from_url:
        with suppress(Exception):
            await open_session(cache)

    assert {call.args[0] for call in cache.get_aiohttp_session.call_args_list} == {
        service
    }
    from_url.assert_not_called()
