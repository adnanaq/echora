"""Unit tests for MalFetcher and response-level MAL mappers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from enrichment.sources.base.exceptions import ServiceNetworkError
from enrichment.sources.mal.mal_fetcher import MalFetcher
from enrichment.sources.mal.mal_mapper import (
    map_mal_anime_responses,
    map_mal_character_responses,
)
from shared_proto.v1.anime_pb2 import EntityType
from shared_proto.v1.merge_pb2 import PageKey, PageResponse


def _mock_browser_session(pages_by_url: dict[str, tuple[str, str]]) -> MagicMock:
    browser = MagicMock()

    async def _get(url: str) -> MagicMock:
        html, final_url = pages_by_url[url]
        page = MagicMock()
        page.url = final_url
        page.get_content = AsyncMock(return_value=html)
        return page

    browser.get = _get
    session_obj = MagicMock()
    session_obj.browser = browser

    async def _aenter(_self: object) -> MagicMock:
        return session_obj

    async def _aexit(_self: object, *_args: object) -> bool:
        return False

    cm = MagicMock()
    cm.__aenter__ = _aenter
    cm.__aexit__ = _aexit
    return cm


@pytest.mark.asyncio
async def test_mal_fetcher_fetches_raw_anime_pages_and_maps_canonically() -> None:
    anime_html = """
    <html><head><link rel="canonical" href="https://myanimelist.net/anime/1/Cowboy_Bebop" /></head>
    <body>
      <h1 class="title-name"><strong>Cowboy Bebop</strong></h1>
      <div class="spaceit_pad"><span class="dark_text">Type:</span> <a href="/topanime.php?type=tv">TV</a></div>
      <div class="spaceit_pad"><span class="dark_text">Episodes:</span> 26</div>
      <div class="spaceit_pad"><span class="dark_text">Status:</span> Finished Airing</div>
    </body></html>
    """
    pics_html = "<html><body><div class='border_solid'>Pictures</div></body></html>"
    chars_html = (
        "<html><body><table class='js-anime-character-table'></table></body></html>"
    )
    pages = {
        "https://myanimelist.net/anime/1": (
            anime_html,
            "https://myanimelist.net/anime/1/Cowboy_Bebop",
        ),
        "https://myanimelist.net/anime/1/Cowboy_Bebop/pics": (
            pics_html,
            "https://myanimelist.net/anime/1/Cowboy_Bebop/pics",
        ),
        "https://myanimelist.net/anime/1/Cowboy_Bebop/characters": (
            chars_html,
            "https://myanimelist.net/anime/1/Cowboy_Bebop/characters",
        ),
    }

    with (
        patch(
            "enrichment.sources.mal.mal_fetcher.browser_session",
            return_value=_mock_browser_session(pages),
        ),
        patch("enrichment.sources.mal.mal_fetcher.wait_for_page", new=AsyncMock()),
        patch("enrichment.sources.mal.mal_fetcher.asyncio.sleep", new=AsyncMock()),
    ):
        fetched = await MalFetcher().fetch_page(
            PageKey(
                provider="mal", entity_type=EntityType.ENTITY_TYPE_ANIME, address="1"
            )
        )

    assert fetched.is_not_found is False
    assert [r.request_name for r in fetched.responses] == ["main", "pics", "characters"]
    assert all(r.content_type == "text/html" for r in fetched.responses)

    canonical = map_mal_anime_responses(
        fetched.responses, source_url="https://myanimelist.net/anime/1/Cowboy_Bebop"
    )
    assert canonical["title"] == "Cowboy Bebop"
    assert canonical["episode_count"] == 26


@pytest.mark.asyncio
async def test_mal_fetcher_reports_not_found_on_error404_marker() -> None:
    pages = {
        "https://myanimelist.net/anime/9999999": (
            '<html><body><div class="error404">Not Found</div></body></html>',
            "https://myanimelist.net/anime/9999999",
        )
    }
    with (
        patch(
            "enrichment.sources.mal.mal_fetcher.browser_session",
            return_value=_mock_browser_session(pages),
        ),
        patch("enrichment.sources.mal.mal_fetcher.wait_for_page", new=AsyncMock()),
    ):
        fetched = await MalFetcher().fetch_page(
            PageKey(
                provider="mal",
                entity_type=EntityType.ENTITY_TYPE_ANIME,
                address="9999999",
            )
        )

    assert fetched.is_not_found is True
    assert fetched.responses == ()


@pytest.mark.asyncio
async def test_mal_fetcher_fetches_character_page_and_maps() -> None:
    char_html = """
    <html><body>
      <h2 class="normal_header">Jet Black <small>(ジェット・ブラック)</small></h2>
      <div id="content"><table border="0"><tr><td class="borderClass">
        <div style="text-align: center;"><img class="portrait" src="https://cdn.myanimelist.net/images/characters/1.jpg" /></div>
      </td><td>
        <h2 class="normal_header">Jet Black</h2>
        Member Favorites: 1,234
      </td></tr></table></div>
    </body></html>
    """
    pages = {
        "https://myanimelist.net/character/3": (
            char_html,
            "https://myanimelist.net/character/3/Jet_Black",
        )
    }
    with (
        patch(
            "enrichment.sources.mal.mal_fetcher.browser_session",
            return_value=_mock_browser_session(pages),
        ),
        patch("enrichment.sources.mal.mal_fetcher.wait_for_page", new=AsyncMock()),
    ):
        fetched = await MalFetcher().fetch_page(
            PageKey(
                provider="mal",
                entity_type=EntityType.ENTITY_TYPE_CHARACTER,
                address="3",
            )
        )

    assert fetched.is_not_found is False
    assert [r.request_name for r in fetched.responses] == ["main"]
    canonical = map_mal_character_responses(
        fetched.responses, source_url="https://myanimelist.net/character/3/Jet_Black"
    )
    assert canonical["name"] == "Jet Black"


@pytest.mark.asyncio
async def test_mal_fetcher_uses_canonical_link_when_browser_url_lacks_slug() -> None:
    anime_html = """
    <html><head><link rel="canonical" href="https://myanimelist.net/anime/1/Cowboy_Bebop" /></head>
    <body><h1 class="title-name"><strong>Cowboy Bebop</strong></h1></body></html>
    """
    pics_html = "<html><body><div class='border_solid'>Pictures</div></body></html>"
    chars_html = (
        "<html><body><table class='js-anime-character-table'></table></body></html>"
    )
    pages = {
        "https://myanimelist.net/anime/1": (
            anime_html,
            "https://myanimelist.net/anime/1",
        ),
        "https://myanimelist.net/anime/1/Cowboy_Bebop/pics": (
            pics_html,
            "https://myanimelist.net/anime/1/Cowboy_Bebop/pics",
        ),
        "https://myanimelist.net/anime/1/Cowboy_Bebop/characters": (
            chars_html,
            "https://myanimelist.net/anime/1/Cowboy_Bebop/characters",
        ),
    }

    with (
        patch(
            "enrichment.sources.mal.mal_fetcher.browser_session",
            return_value=_mock_browser_session(pages),
        ),
        patch("enrichment.sources.mal.mal_fetcher.wait_for_page", new=AsyncMock()),
        patch("enrichment.sources.mal.mal_fetcher.asyncio.sleep", new=AsyncMock()),
    ):
        fetched = await MalFetcher().fetch_page(
            PageKey(
                provider="mal", entity_type=EntityType.ENTITY_TYPE_ANIME, address="1"
            )
        )

    assert [r.request_name for r in fetched.responses] == ["main", "pics", "characters"]


def test_map_mal_character_responses_extracts_canonical_url_when_omitted() -> None:
    char_html = b"""
    <html>
      <head><link rel="canonical" href="https://myanimelist.net/character/3/Jet_Black" /></head>
      <body><h2 class="normal_header">Jet Black</h2></body>
    </html>
    """
    canonical = map_mal_character_responses(
        [PageResponse(request_name="main", content_type="text/html", body=char_html)]
    )
    assert canonical["sources"] == ["https://myanimelist.net/character/3/Jet_Black"]


@pytest.mark.asyncio
async def test_mal_fetcher_raises_network_error_on_empty_html() -> None:
    pages = {"https://myanimelist.net/anime/1": ("", "https://myanimelist.net/anime/1")}
    with (
        patch(
            "enrichment.sources.mal.mal_fetcher.browser_session",
            return_value=_mock_browser_session(pages),
        ),
        patch("enrichment.sources.mal.mal_fetcher.wait_for_page", new=AsyncMock()),
        pytest.raises(ServiceNetworkError),
    ):
        await MalFetcher().fetch_page(
            PageKey(
                provider="mal", entity_type=EntityType.ENTITY_TYPE_ANIME, address="1"
            )
        )


def test_map_mal_anime_responses_requires_main() -> None:
    with pytest.raises(ValueError, match="main"):
        map_mal_anime_responses(
            [
                PageResponse(
                    request_name="pics", content_type="text/html", body=b"<html/>"
                )
            ]
        )
