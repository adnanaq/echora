from contextlib import suppress
from unittest.mock import NonCallableMagicMock, patch

import pytest
from enrichment.sources.kitsu.kitsu_helper import KitsuHelper
from redis.asyncio import Redis

pytestmark = pytest.mark.integration


async def test_get_anime_by_id_session_comes_from_given_cache_under_kitsu(
    not_found_cache: NonCallableMagicMock,
) -> None:
    with patch.object(Redis, "from_url", autospec=True) as from_url:
        with suppress(Exception):
            await KitsuHelper(not_found_cache).get_anime_by_id(1)

    requested = {
        call.args[0] for call in not_found_cache.get_aiohttp_session.call_args_list
    }
    assert requested == {"kitsu"}
    from_url.assert_not_called()
