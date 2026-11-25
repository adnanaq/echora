"""Unit tests for AniDBFetcher and response-level AniDB mappers."""

import gzip
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from common.config import EnrichmentConfig
from enrichment.sources.anidb.anidb_fetcher import AniDBFetcher
from enrichment.sources.anidb.anidb_helper import reset_client_state
from enrichment.sources.anidb.anidb_mapper import (
    map_anidb_anime_episode_responses,
    map_anidb_anime_responses,
    map_anidb_character_page_responses,
)
from enrichment.sources.base.exceptions import ServiceBlockedError
from shared_proto.v1.anime_pb2 import EntityType
from shared_proto.v1.merge_pb2 import PageKey, PageResponse


def _mock_anidb_cache(xml_body: str, *, compressed: bool = False) -> MagicMock:
    raw = (
        gzip.compress(xml_body.encode("utf-8"))
        if compressed
        else xml_body.encode("utf-8")
    )
    response = MagicMock()
    response.status = 200
    response.headers = {"Content-Type": "application/xml"}
    response.read = AsyncMock(return_value=raw)
    response.raise_for_status = MagicMock()

    cm = MagicMock()

    async def _aenter(_self: object) -> MagicMock:
        return response

    async def _aexit(_self: object, *_args: object) -> bool:
        return False

    cm.__aenter__ = _aenter
    cm.__aexit__ = _aexit

    session = MagicMock()
    session.get.return_value = cm
    session.closed = False
    cache = MagicMock()
    cache.get_aiohttp_session.return_value = session
    return cache


@pytest.mark.asyncio
async def test_anidb_fetcher_fetches_raw_anime_xml_and_maps_canonically() -> None:
    reset_client_state()
    xml_body = """<?xml version="1.0" encoding="UTF-8"?>
    <anime id="23" restricted="false">
      <type>TV Series</type>
      <episodecount>26</episodecount>
      <startdate>1998-04-03</startdate>
      <enddate>1999-04-24</enddate>
      <titles>
        <title xml:lang="x-jat" type="main">Cowboy Bebop</title>
      </titles>
      <episodes>
        <episode id="101">
          <epno type="1">1</epno>
          <length>25</length>
          <airdate>1998-04-03</airdate>
          <title xml:lang="en">Asteroid Blues</title>
        </episode>
      </episodes>
    </anime>
    """
    cache = _mock_anidb_cache(xml_body, compressed=True)
    fetcher = AniDBFetcher(EnrichmentConfig(), cache)
    key = PageKey(
        provider="anidb", entity_type=EntityType.ENTITY_TYPE_ANIME, address="23"
    )

    with patch("enrichment.sources.anidb.anidb_fetcher.asyncio.sleep", new=AsyncMock()):
        fetched = await fetcher.fetch_page(key, max_age=timedelta(seconds=0))

    assert fetched.is_not_found is False
    assert [r.request_name for r in fetched.responses] == ["main"]
    assert fetched.responses[0].content_type == "application/xml"
    assert b'<anime id="23"' in fetched.responses[0].body

    anime = map_anidb_anime_responses(
        fetched.responses, anidb_url="https://anidb.net/anime/23"
    )
    assert anime["title"] == "Cowboy Bebop"
    assert anime["episode_count"] == 26
    episodes = map_anidb_anime_episode_responses(fetched.responses)
    assert len(episodes) == 1
    assert episodes[0]["title"] == "Asteroid Blues"


@pytest.mark.asyncio
async def test_anidb_fetcher_reports_not_found_and_ban_errors() -> None:
    reset_client_state()
    not_found_cache = _mock_anidb_cache("<error>Anime not found</error>")
    fetcher = AniDBFetcher(EnrichmentConfig(), not_found_cache)
    key = PageKey(
        provider="anidb", entity_type=EntityType.ENTITY_TYPE_ANIME, address="999999"
    )
    fetched = await fetcher.fetch_page(key)
    assert fetched.is_not_found is True
    assert fetched.responses == ()

    reset_client_state()
    banned_cache = _mock_anidb_cache('<error code="555">Banned</error>')
    banned_fetcher = AniDBFetcher(EnrichmentConfig(), banned_cache)
    with pytest.raises(ServiceBlockedError):
        await banned_fetcher.fetch_page(key)
    reset_client_state()


@pytest.mark.asyncio
async def test_anidb_fetcher_fetches_character_page_and_maps() -> None:
    char_html = """
    <html><body>
      <div id="layout-main">
        <h1 class="character">Character: Spike Spiegel</h1>
        <div class="info"><table class="g_definition">
          <tr><th>Name</th><td>Spike Spiegel</td></tr>
        </table></div>
      </div>
    </body></html>
    """
    session_obj = MagicMock()
    session_obj.browser = MagicMock()

    async def _aenter(_self: object) -> MagicMock:
        return session_obj

    async def _aexit(_self: object, *_args: object) -> bool:
        return False

    cm = MagicMock()
    cm.__aenter__ = _aenter
    cm.__aexit__ = _aexit

    with (
        patch(
            "enrichment.sources.anidb.anidb_fetcher.browser_session", return_value=cm
        ),
        patch(
            "enrichment.sources.anidb.anidb_fetcher._fetch_page_html",
            new=AsyncMock(
                return_value=(MagicMock(html=char_html, challenged=False), MagicMock())
            ),
        ),
        patch(
            "enrichment.sources.anidb.anidb_fetcher._has_character_data",
            return_value=True,
        ),
        patch(
            "enrichment.sources.anidb.anidb_fetcher._is_antileech", return_value=False
        ),
    ):
        fetched = await AniDBFetcher(EnrichmentConfig(), MagicMock()).fetch_page(
            PageKey(
                provider="anidb",
                entity_type=EntityType.ENTITY_TYPE_CHARACTER,
                address="1",
            )
        )

    assert fetched.is_not_found is False
    assert [r.request_name for r in fetched.responses] == ["main"]
    page_model = map_anidb_character_page_responses(fetched.responses)
    assert page_model is not None


def test_map_anidb_anime_responses_requires_main() -> None:
    with pytest.raises(ValueError, match="main"):
        map_anidb_anime_responses(
            [
                PageResponse(
                    request_name="other", content_type="application/xml", body=b"<x/>"
                )
            ]
        )
