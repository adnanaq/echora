from typing import Any, Self
from unittest.mock import patch

import pytest
from enrichment.sources.base.exceptions import ServiceNetworkError
from enrichment.sources.mal import mal_fetcher
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
    def __init__(
        self,
        pages_by_url: dict[str, tuple[str, str]],
        failing_urls: set[str] | None = None,
    ) -> None:
        self._pages_by_url = pages_by_url
        self._failing_urls = failing_urls or set()

    async def get(self, url: str) -> _FakePage:
        if url in self._failing_urls:
            raise RuntimeError(f"boom for {url}")
        html, final_url = self._pages_by_url[url]
        return _FakePage(html, final_url)


class _FakeBrowserSession:
    def __init__(
        self,
        pages_by_url: dict[str, tuple[str, str]],
        failing_urls: set[str] | None = None,
    ) -> None:
        self.browser = _FakeBrowser(pages_by_url, failing_urls=failing_urls)

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
        patch.object(mal_fetcher, "wait_for_page", autospec=True),
        patch.object(mal_fetcher.asyncio, "sleep", autospec=True),
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
        patch.object(mal_fetcher, "wait_for_page", autospec=True),
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
        patch.object(mal_fetcher, "wait_for_page", autospec=True),
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
        patch.object(mal_fetcher, "wait_for_page", autospec=True),
        patch.object(mal_fetcher.asyncio, "sleep", autospec=True),
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
        patch.object(mal_fetcher, "wait_for_page", autospec=True),
        pytest.raises(ServiceNetworkError),
    ):
        await MalFetcher().fetch_page(
            PageKey(
                provider="mal", entity_type=EntityType.ENTITY_TYPE_ANIME, address="1"
            )
        )


@pytest.mark.asyncio
async def test_mal_fetcher_episode_page_uses_placeholder_slug_and_returns_html() -> (
    None
):
    episode_html = """
    <html><head>
      <meta property="og:url" content="https://myanimelist.net/anime/21/One_Piece/episode/1" />
    </head><body>
      <h2 class="fs18">#1 - I'm Luffy!</h2>
    </body></html>
    """
    pages = {
        "https://myanimelist.net/anime/21/_/episode/1": (
            episode_html,
            "https://myanimelist.net/anime/21/One_Piece/episode/1",
        )
    }
    with (
        patch(
            "enrichment.sources.mal.mal_fetcher.browser_session",
            return_value=_FakeBrowserSession(pages),
        ),
        patch.object(mal_fetcher, "wait_for_page", autospec=True),
    ):
        fetched = await MalFetcher().fetch_page(
            PageKey(
                provider="mal",
                entity_type=EntityType.ENTITY_TYPE_EPISODE,
                address="21:1",
            )
        )

    assert fetched.is_not_found is False
    assert [response.request_name for response in fetched.responses] == ["main"]


@pytest.mark.asyncio
async def test_mal_fetcher_episode_page_error404_returns_not_found() -> None:
    pages = {
        "https://myanimelist.net/anime/21/_/episode/99999": (
            '<html><body><div class="error404">Not Found</div></body></html>',
            "https://myanimelist.net/anime/21/_/episode/99999",
        )
    }
    with (
        patch(
            "enrichment.sources.mal.mal_fetcher.browser_session",
            return_value=_FakeBrowserSession(pages),
        ),
        patch.object(mal_fetcher, "wait_for_page", autospec=True),
    ):
        fetched = await MalFetcher().fetch_page(
            PageKey(
                provider="mal",
                entity_type=EntityType.ENTITY_TYPE_EPISODE,
                address="21:99999",
            )
        )

    assert fetched.is_not_found is True
    assert fetched.responses == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_address", ["21", "21:", ":1", "21:²", "²:1", "21:abc"])
async def test_mal_fetcher_episode_page_invalid_address_raises_value_error(
    bad_address: str,
) -> None:
    with pytest.raises(ValueError, match="MAL episode address"):
        await MalFetcher().fetch_page(
            PageKey(
                provider="mal",
                entity_type=EntityType.ENTITY_TYPE_EPISODE,
                address=bad_address,
            )
        )


@pytest.mark.asyncio
async def test_mal_fetcher_navigation_failure_raises_service_network_error() -> None:
    with (
        patch(
            "enrichment.sources.mal.mal_fetcher.browser_session",
            return_value=_FakeBrowserSession(
                {}, failing_urls={"https://myanimelist.net/anime/1"}
            ),
        ),
        pytest.raises(ServiceNetworkError, match="navigation failed"),
    ):
        await MalFetcher().fetch_page(
            PageKey(
                provider="mal", entity_type=EntityType.ENTITY_TYPE_ANIME, address="1"
            )
        )


@pytest.mark.asyncio
async def test_mal_fetcher_optional_subpage_failure_keeps_main_response() -> None:
    anime_html = """
    <html><head><link rel="canonical" href="https://myanimelist.net/anime/1/Cowboy_Bebop" /></head>
    <body><h1 class="title-name"><strong>Cowboy Bebop</strong></h1></body></html>
    """
    pages = {
        "https://myanimelist.net/anime/1": (
            anime_html,
            "https://myanimelist.net/anime/1/Cowboy_Bebop",
        )
    }
    failing = {
        "https://myanimelist.net/anime/1/Cowboy_Bebop/pics",
        "https://myanimelist.net/anime/1/Cowboy_Bebop/characters",
    }
    with (
        patch(
            "enrichment.sources.mal.mal_fetcher.browser_session",
            return_value=_FakeBrowserSession(pages, failing_urls=failing),
        ),
        patch.object(mal_fetcher, "wait_for_page", autospec=True),
        patch.object(mal_fetcher.asyncio, "sleep", autospec=True),
    ):
        fetched = await MalFetcher().fetch_page(
            PageKey(
                provider="mal", entity_type=EntityType.ENTITY_TYPE_ANIME, address="1"
            )
        )

    assert fetched.is_not_found is False
    assert [response.request_name for response in fetched.responses] == ["main"]


@pytest.mark.asyncio
async def test_mal_fetcher_character_page_error404_returns_not_found() -> None:
    pages = {
        "https://myanimelist.net/character/9999999": (
            '<html><body><div class="error404">Not Found</div></body></html>',
            "https://myanimelist.net/character/9999999",
        )
    }
    with (
        patch(
            "enrichment.sources.mal.mal_fetcher.browser_session",
            return_value=_FakeBrowserSession(pages),
        ),
        patch.object(mal_fetcher, "wait_for_page", autospec=True),
    ):
        fetched = await MalFetcher().fetch_page(
            PageKey(
                provider="mal",
                entity_type=EntityType.ENTITY_TYPE_CHARACTER,
                address="9999999",
            )
        )

    assert fetched.is_not_found is True
    assert fetched.responses == ()
