from datetime import timedelta
from unittest.mock import create_autospec

import aiohttp
import pytest
from enrichment.sources.base.exceptions import ServiceNetworkError, ServiceParseError
from enrichment.sources.syoboi.syoboi_fetcher import SyoboiFetcher
from http_cache import HTTPCacheManager
from shared_proto.v1.anime_pb2 import EntityType
from shared_proto.v1.merge_pb2 import PageKey


def _build_syoboi_cache(
    responses_by_command: dict[str, tuple[int, str]],
    *,
    get_error: Exception | None = None,
) -> tuple[HTTPCacheManager, list[dict[str, object]]]:
    recorded_calls: list[dict[str, object]] = []

    def _make_session(_service: str, **_kwargs: object) -> aiohttp.ClientSession:
        session = create_autospec(aiohttp.ClientSession, instance=True)
        session.__aenter__.return_value = session
        session.__aexit__.return_value = False

        def _record_get(
            url: str, *, params: dict[str, str], headers: dict[str, str]
        ) -> aiohttp.ClientResponse:
            recorded_calls.append(
                {"url": url, "params": dict(params), "headers": dict(headers)}
            )
            if get_error is not None:
                raise get_error
            status, body_text = responses_by_command[params["Command"]]
            response = create_autospec(aiohttp.ClientResponse, instance=True)
            response.status = status
            response.headers = {"Content-Type": "application/xml; charset=UTF-8"}
            response.read.return_value = body_text.encode("utf-8")
            response.__aenter__.return_value = response
            response.__aexit__.return_value = False
            return response

        session.get.side_effect = _record_get
        return session

    cache = create_autospec(HTTPCacheManager, instance=True)
    cache.get_aiohttp_session.side_effect = _make_session
    return cache, recorded_calls


@pytest.mark.asyncio
async def test_syoboi_fetcher_valid_tid_returns_title_and_programs_xml() -> None:
    title_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<TitleLookupResponse><Result><Code>200</Code></Result>"
        '<TitleItems><TitleItem id="1"><TID>1</TID><Title>Mahou Tsukai</Title></TitleItem></TitleItems>'
        "</TitleLookupResponse>"
    )
    prog_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<ProgLookupResponse><ProgItems>"
        '<ProgItem id="26"><TID>1</TID><StTime>2003-03-28 02:07:00</StTime></ProgItem>'
        "</ProgItems><Result><Code>200</Code></Result></ProgLookupResponse>"
    )
    cache, calls = _build_syoboi_cache(
        {"TitleLookup": (200, title_xml), "ProgLookup": (200, prog_xml)}
    )
    fetcher = SyoboiFetcher(cache)
    page_key = PageKey(
        provider="syoboi", entity_type=EntityType.ENTITY_TYPE_ANIME, address="1"
    )

    fetched = await fetcher.fetch_page(page_key, max_age=timedelta(0))

    assert fetched.is_not_found is False
    assert [response.request_name for response in fetched.responses] == [
        "main",
        "programs",
    ]
    assert all(
        response.content_type == "application/xml" for response in fetched.responses
    )
    assert calls[0]["headers"] == {"Cache-Control": "no-cache"}


@pytest.mark.asyncio
async def test_syoboi_fetcher_empty_title_items_returns_not_found() -> None:
    empty_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<TitleLookupResponse><Result><Code>200</Code></Result>"
        "<TitleItems></TitleItems></TitleLookupResponse>"
    )
    cache, _ = _build_syoboi_cache({"TitleLookup": (200, empty_xml)})
    fetcher = SyoboiFetcher(cache)

    fetched = await fetcher.fetch_page(
        PageKey(
            provider="syoboi",
            entity_type=EntityType.ENTITY_TYPE_ANIME,
            address="9999999",
        )
    )

    assert fetched.is_not_found is True
    assert fetched.responses == ()


@pytest.mark.asyncio
async def test_syoboi_fetcher_malformed_xml_raises_parse_error() -> None:
    truncated_xml = "<TitleLookupResponse><TitleItems><TitleItem"
    cache, _ = _build_syoboi_cache({"TitleLookup": (200, truncated_xml)})
    fetcher = SyoboiFetcher(cache)

    with pytest.raises(ServiceParseError):
        await fetcher.fetch_page(
            PageKey(
                provider="syoboi",
                entity_type=EntityType.ENTITY_TYPE_ANIME,
                address="1",
            )
        )


@pytest.mark.asyncio
async def test_syoboi_fetcher_timeout_raises_network_error() -> None:
    cache, _ = _build_syoboi_cache({}, get_error=TimeoutError("timed out"))
    fetcher = SyoboiFetcher(cache)

    with pytest.raises(ServiceNetworkError):
        await fetcher.fetch_page(
            PageKey(
                provider="syoboi",
                entity_type=EntityType.ENTITY_TYPE_ANIME,
                address="1",
            )
        )
