from typing import Any, Self
from unittest.mock import AsyncMock, patch

import pytest
from enrichment.sources.base.exceptions import ServiceNetworkError
from enrichment.sources.mal.mal_fetcher import MalFetcher
from shared_proto.v1.anime_pb2 import EntityType
from shared_proto.v1.merge_pb2 import PageKey


class _FakePage:
    def __init__(self, html: str, final_url: str) -> None:
        self.url = final_url
        self._html = html

    async def get_content(self) -> str:
        return self._html


class _FakeBrowser:
    def __init__(self, pages_by_url: dict[str, tuple[str, str]]) -> None:
        self._pages_by_url = pages_by_url

    async def get(self, url: str) -> _FakePage:
        html, final_url = self._pages_by_url[url]
        return _FakePage(html, final_url)


class _FakeBrowserSession:
    def __init__(self, pages_by_url: dict[str, tuple[str, str]]) -> None:
        self.browser = _FakeBrowser(pages_by_url)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_args: Any) -> bool:
        return False


@pytest.mark.asyncio
async def test_mal_fetcher_anime_page_returns_raw_html_responses() -> None:
    anime_html = """
    <html><head><link rel="canonical" href="https://myanimelist.net/anime/1/Cowboy_Bebop" /></head>
    <body>
      <h1 class="title-name"><strong>Cowboy Bebop</strong></h1>
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
            return_value=_FakeBrowserSession(pages),
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
    assert [response.request_name for response in fetched.responses] == [
        "main",
        "pics",
        "characters",
    ]
    assert all(response.content_type == "text/html" for response in fetched.responses)


@pytest.mark.asyncio
async def test_mal_fetcher_error404_marker_returns_not_found() -> None:
    pages = {
        "https://myanimelist.net/anime/9999999": (
            '<html><body><div class="error404">Not Found</div></body></html>',
            "https://myanimelist.net/anime/9999999",
        )
    }
    with (
        patch(
            "enrichment.sources.mal.mal_fetcher.browser_session",
            return_value=_FakeBrowserSession(pages),
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
async def test_mal_fetcher_character_page_returns_main_response() -> None:
    char_html = """
    <html><body>
      <h2 class="normal_header">Jet Black <small>(ジェット・ブラック)</small></h2>
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
            return_value=_FakeBrowserSession(pages),
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
    assert [response.request_name for response in fetched.responses] == ["main"]


@pytest.mark.asyncio
async def test_mal_fetcher_browser_url_without_slug_uses_canonical_link() -> None:
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
            return_value=_FakeBrowserSession(pages),
        ),
        patch("enrichment.sources.mal.mal_fetcher.wait_for_page", new=AsyncMock()),
        patch("enrichment.sources.mal.mal_fetcher.asyncio.sleep", new=AsyncMock()),
    ):
        fetched = await MalFetcher().fetch_page(
            PageKey(
                provider="mal", entity_type=EntityType.ENTITY_TYPE_ANIME, address="1"
            )
        )

    assert [response.request_name for response in fetched.responses] == [
        "main",
        "pics",
        "characters",
    ]


@pytest.mark.asyncio
async def test_mal_fetcher_empty_main_html_raises_network_error() -> None:
    pages = {"https://myanimelist.net/anime/1": ("", "https://myanimelist.net/anime/1")}
    with (
        patch(
            "enrichment.sources.mal.mal_fetcher.browser_session",
            return_value=_FakeBrowserSession(pages),
        ),
        patch("enrichment.sources.mal.mal_fetcher.wait_for_page", new=AsyncMock()),
        pytest.raises(ServiceNetworkError),
    ):
        await MalFetcher().fetch_page(
            PageKey(
                provider="mal", entity_type=EntityType.ENTITY_TYPE_ANIME, address="1"
            )
        )
