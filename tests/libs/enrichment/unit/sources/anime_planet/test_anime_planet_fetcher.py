from typing import Any, Self
from unittest.mock import AsyncMock, patch

import pytest
from enrichment.sources.anime_planet.anime_planet_fetcher import AnimePlanetFetcher
from enrichment.sources.base.exceptions import ServiceNetworkError
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
async def test_anime_planet_fetcher_anime_page_returns_main_and_characters() -> None:
    anime_html = """
    <html>
      <head>
        <script type="application/ld+json">
          {"@type": "TVSeries", "name": "Cowboy Bebop", "numberOfEpisodes": 26}
        </script>
      </head>
      <body>
        <section class="entryBar"><span class="type">TV (26 eps)</span></section>
      </body>
    </html>
    """
    chars_html = "<html><body><a class='name' href='/characters/spike-spiegel'>Spike</a></body></html>"
    pages = {
        "https://www.anime-planet.com/anime/cowboy-bebop": (
            anime_html,
            "https://www.anime-planet.com/anime/cowboy-bebop",
        ),
        "https://www.anime-planet.com/anime/cowboy-bebop/characters": (
            chars_html,
            "https://www.anime-planet.com/anime/cowboy-bebop/characters",
        ),
    }

    with (
        patch(
            "enrichment.sources.anime_planet.anime_planet_fetcher.browser_session",
            return_value=_FakeBrowserSession(pages),
        ),
        patch(
            "enrichment.sources.anime_planet.anime_planet_fetcher.wait_for_page",
            new=AsyncMock(),
        ),
    ):
        fetched = await AnimePlanetFetcher().fetch_page(
            PageKey(
                provider="anime_planet",
                entity_type=EntityType.ENTITY_TYPE_ANIME,
                address="cowboy-bebop",
            )
        )

    assert fetched.is_not_found is False
    assert [response.request_name for response in fetched.responses] == [
        "main",
        "characters",
    ]


@pytest.mark.asyncio
async def test_anime_planet_fetcher_search_redirect_returns_not_found() -> None:
    pages = {
        "https://www.anime-planet.com/anime/no-such-anime": (
            "<html><body><div class='quickFilter'>Search</div></body></html>",
            "https://www.anime-planet.com/search.php?name=no-such-anime",
        )
    }
    with (
        patch(
            "enrichment.sources.anime_planet.anime_planet_fetcher.browser_session",
            return_value=_FakeBrowserSession(pages),
        ),
        patch(
            "enrichment.sources.anime_planet.anime_planet_fetcher.wait_for_page",
            new=AsyncMock(),
        ),
    ):
        fetched = await AnimePlanetFetcher().fetch_page(
            PageKey(
                provider="anime_planet",
                entity_type=EntityType.ENTITY_TYPE_ANIME,
                address="no-such-anime",
            )
        )

    assert fetched.is_not_found is True
    assert fetched.responses == ()


@pytest.mark.asyncio
async def test_anime_planet_fetcher_character_search_redirect_returns_not_found() -> (
    None
):
    pages = {
        "https://www.anime-planet.com/characters/no-such-character": (
            "<html><body><div class='quickFilter'>Search</div></body></html>",
            "https://www.anime-planet.com/search.php?name=no-such-character",
        )
    }
    with (
        patch(
            "enrichment.sources.anime_planet.anime_planet_fetcher.browser_session",
            return_value=_FakeBrowserSession(pages),
        ),
        patch(
            "enrichment.sources.anime_planet.anime_planet_fetcher.wait_for_page",
            new=AsyncMock(),
        ),
    ):
        fetched = await AnimePlanetFetcher().fetch_page(
            PageKey(
                provider="anime_planet",
                entity_type=EntityType.ENTITY_TYPE_CHARACTER,
                address="no-such-character",
            )
        )

    assert fetched.is_not_found is True
    assert fetched.responses == ()


@pytest.mark.asyncio
async def test_anime_planet_fetcher_character_page_returns_main_response() -> None:
    char_html = """
    <html><body>
      <h1 itemprop="name">Spike Spiegel</h1>
      <div class="pure-g entryBar"><div class="pure-1 md-1-5">Gender: Male</div></div>
    </body></html>
    """
    pages = {
        "https://www.anime-planet.com/characters/spike-spiegel": (
            char_html,
            "https://www.anime-planet.com/characters/spike-spiegel",
        )
    }
    with (
        patch(
            "enrichment.sources.anime_planet.anime_planet_fetcher.browser_session",
            return_value=_FakeBrowserSession(pages),
        ),
        patch(
            "enrichment.sources.anime_planet.anime_planet_fetcher.wait_for_page",
            new=AsyncMock(),
        ),
    ):
        fetched = await AnimePlanetFetcher().fetch_page(
            PageKey(
                provider="anime_planet",
                entity_type=EntityType.ENTITY_TYPE_CHARACTER,
                address="spike-spiegel",
            )
        )

    assert fetched.is_not_found is False
    assert [response.request_name for response in fetched.responses] == ["main"]


@pytest.mark.asyncio
async def test_anime_planet_fetcher_empty_anime_html_raises_network_error() -> None:
    pages = {
        "https://www.anime-planet.com/anime/cowboy-bebop": (
            "",
            "https://www.anime-planet.com/anime/cowboy-bebop",
        )
    }
    with (
        patch(
            "enrichment.sources.anime_planet.anime_planet_fetcher.browser_session",
            return_value=_FakeBrowserSession(pages),
        ),
        patch(
            "enrichment.sources.anime_planet.anime_planet_fetcher.wait_for_page",
            new=AsyncMock(),
        ),
        pytest.raises(ServiceNetworkError, match="no HTML"),
    ):
        await AnimePlanetFetcher().fetch_page(
            PageKey(
                provider="anime_planet",
                entity_type=EntityType.ENTITY_TYPE_ANIME,
                address="cowboy-bebop",
            )
        )


@pytest.mark.asyncio
async def test_anime_planet_fetcher_empty_character_html_raises_network_error() -> None:
    pages = {
        "https://www.anime-planet.com/characters/spike-spiegel": (
            "",
            "https://www.anime-planet.com/characters/spike-spiegel",
        )
    }
    with (
        patch(
            "enrichment.sources.anime_planet.anime_planet_fetcher.browser_session",
            return_value=_FakeBrowserSession(pages),
        ),
        patch(
            "enrichment.sources.anime_planet.anime_planet_fetcher.wait_for_page",
            new=AsyncMock(),
        ),
        pytest.raises(ServiceNetworkError, match="no HTML"),
    ):
        await AnimePlanetFetcher().fetch_page(
            PageKey(
                provider="anime_planet",
                entity_type=EntityType.ENTITY_TYPE_CHARACTER,
                address="spike-spiegel",
            )
        )


@pytest.mark.asyncio
async def test_anime_planet_fetcher_empty_optional_characters_html_keeps_main_only() -> (
    None
):
    anime_html = "<html><body><section class='entryBar'></section></body></html>"
    pages = {
        "https://www.anime-planet.com/anime/cowboy-bebop": (
            anime_html,
            "https://www.anime-planet.com/anime/cowboy-bebop",
        ),
        "https://www.anime-planet.com/anime/cowboy-bebop/characters": (
            "",
            "https://www.anime-planet.com/anime/cowboy-bebop/characters",
        ),
    }
    with (
        patch(
            "enrichment.sources.anime_planet.anime_planet_fetcher.browser_session",
            return_value=_FakeBrowserSession(pages),
        ),
        patch(
            "enrichment.sources.anime_planet.anime_planet_fetcher.wait_for_page",
            new=AsyncMock(),
        ),
    ):
        fetched = await AnimePlanetFetcher().fetch_page(
            PageKey(
                provider="anime_planet",
                entity_type=EntityType.ENTITY_TYPE_ANIME,
                address="cowboy-bebop",
            )
        )

    assert fetched.is_not_found is False
    assert [response.request_name for response in fetched.responses] == ["main"]
