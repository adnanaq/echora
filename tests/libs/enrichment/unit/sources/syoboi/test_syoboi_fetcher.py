"""Unit tests for SyoboiFetcher."""

from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from enrichment.sources.syoboi.syoboi_fetcher import SyoboiFetcher
from shared_proto.v1.anime_pb2 import EntityType
from shared_proto.v1.merge_pb2 import PageKey


def _mock_syoboi_cache(
    responses_by_command: dict[str, tuple[int, str]],
) -> tuple[MagicMock, list[dict[str, Any]]]:
    recorded: list[dict[str, Any]] = []

    def _make_session() -> MagicMock:
        session = MagicMock()

        def _get(
            url: str, *, params: dict[str, str], headers: dict[str, str]
        ) -> MagicMock:
            recorded.append({"url": url, "params": params, "headers": headers})
            status, body_text = responses_by_command[params["Command"]]
            response = MagicMock()
            response.status = status
            response.headers = {"Content-Type": "application/xml; charset=UTF-8"}
            response.read = AsyncMock(return_value=body_text.encode("utf-8"))
            response.raise_for_status = MagicMock()

            resp_cm = MagicMock()

            async def _resp_aenter(_self: object) -> MagicMock:
                return response

            async def _resp_aexit(_self: object, *_args: object) -> bool:
                return False

            resp_cm.__aenter__ = _resp_aenter
            resp_cm.__aexit__ = _resp_aexit
            return resp_cm

        session.get = _get
        sess_cm = MagicMock()

        async def _sess_aenter(_self: object) -> MagicMock:
            return session

        async def _sess_aexit(_self: object, *_args: object) -> bool:
            return False

        sess_cm.__aenter__ = _sess_aenter
        sess_cm.__aexit__ = _sess_aexit
        return sess_cm

    cache = MagicMock()
    cache.get_aiohttp_session.side_effect = lambda *_a, **_kw: _make_session()
    return cache, recorded


@pytest.mark.asyncio
async def test_syoboi_fetcher_fetches_title_and_programs_xml() -> None:
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
    cache, calls = _mock_syoboi_cache(
        {"TitleLookup": (200, title_xml), "ProgLookup": (200, prog_xml)}
    )
    fetcher = SyoboiFetcher(cache)
    key = PageKey(
        provider="syoboi", entity_type=EntityType.ENTITY_TYPE_ANIME, address="1"
    )

    fetched = await fetcher.fetch_page(key, max_age=timedelta(0))

    assert fetched.is_not_found is False
    assert [r.request_name for r in fetched.responses] == ["main", "programs"]
    assert all(r.content_type == "application/xml" for r in fetched.responses)
    assert calls[0]["headers"]["Cache-Control"] == "no-cache"


@pytest.mark.asyncio
async def test_syoboi_fetcher_reports_not_found_when_no_title_items() -> None:
    empty_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<TitleLookupResponse><Result><Code>200</Code></Result>"
        "<TitleItems></TitleItems></TitleLookupResponse>"
    )
    cache, _ = _mock_syoboi_cache({"TitleLookup": (200, empty_xml)})
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
