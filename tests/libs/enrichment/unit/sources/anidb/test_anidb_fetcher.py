import gzip
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta
from unittest.mock import create_autospec, patch

import aiohttp
import pytest
from common.config import EnrichmentConfig
from enrichment.sources.anidb.anidb_fetcher import AniDBFetcher
from enrichment.sources.anidb.anidb_helper import _state, reset_client_state
from enrichment.sources.base.browser import BrowserSession
from enrichment.sources.base.cloudflare_challenge import PageRead
from enrichment.sources.base.exceptions import (
    ServiceBlockedError,
    ServiceNetworkError,
    ServiceParseError,
)
from http_cache import HTTPCacheManager
from http_cache.network_request_pacer import NetworkRequestPacer
from shared_proto.v1.anime_pb2 import EntityType
from shared_proto.v1.merge_pb2 import PageKey


def _build_anidb_cache(
    xml_body: str = "",
    *,
    status: int = 200,
    compressed: bool = False,
    cache_enabled: bool = True,
    raise_on_status: Exception | None = None,
) -> tuple[HTTPCacheManager, aiohttp.ClientSession, list[dict[str, object]]]:
    recorded_calls: list[dict[str, object]] = []
    raw_bytes = (
        gzip.compress(xml_body.encode("utf-8"))
        if compressed
        else xml_body.encode("utf-8")
    )

    response = create_autospec(aiohttp.ClientResponse, instance=True)
    response.status = status
    response.headers = {"Content-Type": "application/xml"}
    response.read.return_value = raw_bytes
    if raise_on_status is not None:
        response.raise_for_status.side_effect = raise_on_status
    response.__aenter__.return_value = response
    response.__aexit__.return_value = False

    session = create_autospec(aiohttp.ClientSession, instance=True)
    session.__aenter__.return_value = session
    session.__aexit__.return_value = False

    def _record_get(
        url: str, *, params: dict[str, object], headers: dict[str, str]
    ) -> aiohttp.ClientResponse:
        recorded_calls.append({"url": url, "params": params, "headers": dict(headers)})
        return response

    session.get.side_effect = _record_get

    cache = create_autospec(HTTPCacheManager, instance=True)
    cache.enabled = cache_enabled
    cache.network_request_pacer = NetworkRequestPacer()
    cache.get_aiohttp_session.return_value = session
    return cache, session, recorded_calls


@pytest.mark.asyncio
async def test_anidb_fetcher_anime_xml_decompresses_and_returns_raw_response() -> None:
    reset_client_state()
    xml_body = (
        '<?xml version="1.0" encoding="UTF-8"?><anime id="23"><type>TV</type></anime>'
    )
    cache, session, calls = _build_anidb_cache(xml_body, compressed=True)
    fetcher = AniDBFetcher(EnrichmentConfig(), cache)
    page_key = PageKey(
        provider="anidb", entity_type=EntityType.ENTITY_TYPE_ANIME, address="23"
    )

    fetched = await fetcher.fetch_page(page_key, max_age=timedelta(seconds=0))

    assert fetched.is_not_found is False
    assert [response.request_name for response in fetched.responses] == ["main"]
    assert fetched.responses[0].content_type == "application/xml"
    assert b'<anime id="23"' in fetched.responses[0].body
    assert calls[0]["headers"] == {"Cache-Control": "no-cache"}
    session.__aexit__.assert_awaited_once()


@pytest.mark.asyncio
async def test_anidb_fetcher_anime_not_found_error_returns_not_found() -> None:
    reset_client_state()
    cache, _, _ = _build_anidb_cache("<error>Anime not found</error>")
    fetcher = AniDBFetcher(EnrichmentConfig(), cache)
    page_key = PageKey(
        provider="anidb", entity_type=EntityType.ENTITY_TYPE_ANIME, address="999999"
    )

    fetched = await fetcher.fetch_page(page_key)

    assert fetched.is_not_found is True
    assert fetched.responses == ()


@pytest.mark.asyncio
async def test_anidb_fetcher_anime_description_with_not_found_text_keeps_xml() -> None:
    reset_client_state()
    xml_body = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<anime id="23"><description>Plot mentions Anime not found here.</description></anime>'
    )
    cache, _, _ = _build_anidb_cache(xml_body)
    fetcher = AniDBFetcher(EnrichmentConfig(), cache)
    page_key = PageKey(
        provider="anidb", entity_type=EntityType.ENTITY_TYPE_ANIME, address="23"
    )

    fetched = await fetcher.fetch_page(page_key)

    assert fetched.is_not_found is False
    assert [response.request_name for response in fetched.responses] == ["main"]
    assert b"Plot mentions Anime not found here." in fetched.responses[0].body


@pytest.mark.asyncio
async def test_anidb_fetcher_non_ban_error_xml_raises_network_error() -> None:
    reset_client_state()
    cache, _, _ = _build_anidb_cache('<error code="302">client version missing</error>')
    fetcher = AniDBFetcher(EnrichmentConfig(), cache)
    page_key = PageKey(
        provider="anidb", entity_type=EntityType.ENTITY_TYPE_ANIME, address="23"
    )

    with pytest.raises(ServiceNetworkError, match="client version missing"):
        await fetcher.fetch_page(page_key)


@pytest.mark.asyncio
async def test_anidb_fetcher_non_xml_body_raises_parse_error() -> None:
    reset_client_state()
    cache, _, _ = _build_anidb_cache("Service temporarily unavailable")
    fetcher = AniDBFetcher(EnrichmentConfig(), cache)
    page_key = PageKey(
        provider="anidb", entity_type=EntityType.ENTITY_TYPE_ANIME, address="23"
    )

    with pytest.raises(ServiceParseError):
        await fetcher.fetch_page(page_key)


@pytest.mark.asyncio
async def test_anidb_fetcher_ban_xml_raises_blocked_and_sets_cooldown() -> None:
    reset_client_state()
    cache, _, _ = _build_anidb_cache('<error code="555">Banned</error>')
    fetcher = AniDBFetcher(EnrichmentConfig(), cache)
    page_key = PageKey(
        provider="anidb", entity_type=EntityType.ENTITY_TYPE_ANIME, address="23"
    )

    with pytest.raises(ServiceBlockedError):
        await fetcher.fetch_page(page_key)

    assert _state.ban_lifts_at > time.time()
    reset_client_state()


@pytest.mark.asyncio
async def test_anidb_fetcher_active_ban_with_cache_disabled_raises_blocked_without_request() -> (
    None
):
    reset_client_state()
    _state.ban_lifts_at = time.time() + 3600
    cache, _, calls = _build_anidb_cache('<anime id="23"/>', cache_enabled=False)
    fetcher = AniDBFetcher(EnrichmentConfig(), cache)
    page_key = PageKey(
        provider="anidb", entity_type=EntityType.ENTITY_TYPE_ANIME, address="23"
    )

    with pytest.raises(ServiceBlockedError):
        await fetcher.fetch_page(page_key)

    assert calls == []
    reset_client_state()


@pytest.mark.asyncio
async def test_anidb_fetcher_active_ban_with_cache_enabled_uses_only_if_cached() -> (
    None
):
    reset_client_state()
    _state.ban_lifts_at = time.time() + 3600
    hit_cache, _, hit_calls = _build_anidb_cache('<anime id="23"/>', cache_enabled=True)
    page_key = PageKey(
        provider="anidb", entity_type=EntityType.ENTITY_TYPE_ANIME, address="23"
    )

    fetched = await AniDBFetcher(EnrichmentConfig(), hit_cache).fetch_page(
        page_key, max_age=timedelta(seconds=60)
    )
    assert fetched.is_not_found is False
    assert hit_calls[0]["headers"] == {"Cache-Control": "max-age=60, only-if-cached"}

    miss_cache, _, miss_calls = _build_anidb_cache(status=504, cache_enabled=True)
    with pytest.raises(ServiceBlockedError, match="not cached"):
        await AniDBFetcher(EnrichmentConfig(), miss_cache).fetch_page(page_key)
    assert miss_calls[0]["headers"] == {"Cache-Control": "only-if-cached"}
    reset_client_state()


@pytest.mark.asyncio
async def test_anidb_fetcher_closes_session_when_http_error_occurs() -> None:
    reset_client_state()
    cache, session, _ = _build_anidb_cache(
        status=500, raise_on_status=aiohttp.ClientError("server error")
    )
    fetcher = AniDBFetcher(EnrichmentConfig(), cache)
    page_key = PageKey(
        provider="anidb", entity_type=EntityType.ENTITY_TYPE_ANIME, address="23"
    )

    with pytest.raises(ServiceNetworkError):
        await fetcher.fetch_page(page_key)

    session.__aexit__.assert_awaited_once()


@pytest.mark.asyncio
async def test_anidb_fetcher_character_page_returns_raw_html() -> None:
    char_html = (
        '<html><body><div id="tab_1_pane">'
        '<tr class="mainname"><span itemprop="name">Spike Spiegel</span></tr>'
        "</div></body></html>"
    )
    browser_session_instance = create_autospec(BrowserSession, instance=True)
    cache, _, _ = _build_anidb_cache()

    @asynccontextmanager
    async def _fake_browser_session(
        **_kwargs: object,
    ) -> AsyncIterator[BrowserSession]:
        yield browser_session_instance

    async def _fake_fetch_page_html(
        _browser: object, _url: str
    ) -> tuple[PageRead, object]:
        return PageRead(html=char_html, ready=True, challenged=False), object()

    with (
        patch(
            "enrichment.sources.anidb.anidb_fetcher.browser_session",
            side_effect=_fake_browser_session,
        ),
        patch(
            "enrichment.sources.anidb.anidb_fetcher._fetch_page_html",
            side_effect=_fake_fetch_page_html,
        ),
    ):
        fetched = await AniDBFetcher(EnrichmentConfig(), cache).fetch_page(
            PageKey(
                provider="anidb",
                entity_type=EntityType.ENTITY_TYPE_CHARACTER,
                address="1",
            )
        )

    assert fetched.is_not_found is False
    assert [response.request_name for response in fetched.responses] == ["main"]
    assert b"Spike Spiegel" in fetched.responses[0].body
