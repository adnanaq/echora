from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from enrichment.api_helpers.mal_client import MalClient


def _cm(response):
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=response)
    cm.__aexit__ = AsyncMock(return_value=None)
    return cm


@pytest.mark.asyncio
async def test_get_json_200_returns_dict_and_calls_limiter():
    resp = AsyncMock()
    resp.status = 200
    resp.json = AsyncMock(return_value={"data": {"ok": True}})

    session = MagicMock()
    session.get = MagicMock(return_value=_cm(resp))

    limiter = AsyncMock()
    client = MalClient(session=session, limiter=limiter, timeout_seconds=10.0)

    data = await client.get_json("https://example.test")
    assert data == {"data": {"ok": True}}
    limiter.acquire.assert_awaited()


@pytest.mark.asyncio
async def test_get_json_retries_429_then_succeeds():
    r429 = AsyncMock()
    r429.status = 429

    r200 = AsyncMock()
    r200.status = 200
    r200.json = AsyncMock(return_value={"data": {"ok": True}})

    session = MagicMock()
    session.get = MagicMock(side_effect=[_cm(r429), _cm(r200)])

    limiter = AsyncMock()
    client = MalClient(session=session, limiter=limiter, timeout_seconds=10.0)

    with patch("asyncio.sleep", new_callable=AsyncMock):
        data = await client.get_json("https://example.test", max_retries=1)

    assert data == {"data": {"ok": True}}


@pytest.mark.asyncio
async def test_client_without_explicit_limiter_uses_shared_limiter():
    resp = AsyncMock()
    resp.status = 200
    resp.json = AsyncMock(return_value={"data": {"ok": True}})

    session = MagicMock()
    session.get = MagicMock(return_value=_cm(resp))

    shared = AsyncMock()
    with patch("enrichment.api_helpers.mal_client.get_shared_mal_rate_limiter", return_value=shared):
        client = MalClient(session=session, timeout_seconds=10.0)
        await client.get_json("https://example.test")

    shared.acquire.assert_awaited()
