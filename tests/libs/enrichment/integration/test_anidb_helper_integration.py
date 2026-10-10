from contextlib import suppress
from unittest.mock import NonCallableMagicMock, patch

import pytest
from common.config import EnrichmentConfig
from enrichment.sources.anidb.anidb_helper import AniDBHelper
from redis.asyncio import Redis

pytestmark = pytest.mark.integration


async def test_ensure_session_health_session_comes_from_given_cache_under_anidb(
    not_found_cache: NonCallableMagicMock,
) -> None:
    with patch.object(Redis, "from_url", autospec=True) as from_url:
        with suppress(Exception):
            await AniDBHelper(
                EnrichmentConfig(), not_found_cache
            )._ensure_session_health()

    requested = {
        call.args[0] for call in not_found_cache.get_aiohttp_session.call_args_list
    }
    assert requested == {"anidb"}
    from_url.assert_not_called()
