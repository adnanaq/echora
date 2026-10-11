from contextlib import suppress
from unittest.mock import NonCallableMagicMock, patch

import pytest
from enrichment.sources.animeschedule.animeschedule_helper import (
    AnimescheduleHelper,
)
from redis.asyncio import Redis

pytestmark = pytest.mark.integration


async def test_search_session_comes_from_given_cache_under_animeschedule(
    not_found_cache: NonCallableMagicMock,
) -> None:
    with patch.object(Redis, "from_url", autospec=True) as from_url:
        with suppress(Exception):
            await AnimescheduleHelper(not_found_cache)._search("Test Anime")

    requested = {
        call.args[0] for call in not_found_cache.get_aiohttp_session.call_args_list
    }
    assert requested == {"animeschedule"}
    from_url.assert_not_called()
