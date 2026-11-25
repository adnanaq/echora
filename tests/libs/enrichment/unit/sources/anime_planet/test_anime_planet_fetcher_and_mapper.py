"""Unit tests for AnimePlanetFetcher and response-level Anime-Planet mappers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from enrichment.sources.anime_planet.anime_planet_fetcher import AnimePlanetFetcher
from enrichment.sources.anime_planet.animeplanet_mapper import (
    map_animeplanet_anime_responses,
    map_animeplanet_character_responses,
)
from shared_proto.v1.anime_pb2 import EntityType
from shared_proto.v1.merge_pb2 import PageKey, PageResponse


def _mock_ap_session(pages_by_url: dict[str, tuple[str, str]]) -> MagicMock:
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
async def test_anime_planet_fetcher_fetches_raw_anime_pages_and_maps() -> None:
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
            return_value=_mock_ap_session(pages),
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
    assert [r.request_name for r in fetched.responses] == ["main", "characters"]
    canonical = map_animeplanet_anime_responses(fetched.responses, slug="cowboy-bebop")
    assert canonical["title"] == "Cowboy Bebop"


@pytest.mark.asyncio
async def test_anime_planet_fetcher_detects_search_redirect_as_not_found() -> None:
    pages = {
        "https://www.anime-planet.com/anime/no-such-anime": (
            "<html><body><div class='quickFilter'>Search</div></body></html>",
            "https://www.anime-planet.com/search.php?name=no-such-anime",
        )
    }
    with (
        patch(
            "enrichment.sources.anime_planet.anime_planet_fetcher.browser_session",
            return_value=_mock_ap_session(pages),
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
async def test_anime_planet_fetcher_fetches_character_page_and_maps() -> None:
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
            return_value=_mock_ap_session(pages),
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
    assert [r.request_name for r in fetched.responses] == ["main"]
    canonical = map_animeplanet_character_responses(
        fetched.responses,
        source_url="https://www.anime-planet.com/characters/spike-spiegel",
    )
    assert canonical["name"] == "Spike Spiegel"


def test_map_animeplanet_anime_responses_requires_main() -> None:
    with pytest.raises(ValueError, match="main"):
        map_animeplanet_anime_responses(
            [
                PageResponse(
                    request_name="characters", content_type="text/html", body=b"<html/>"
                )
            ]
        )


def test_map_animeplanet_responses_extract_canonical_url_when_omitted() -> None:
    anime_html = b"""
    <html>
      <head>
        <link rel="canonical" href="https://www.anime-planet.com/anime/cowboy-bebop" />
        <script type="application/ld+json">
          {"@type": "TVSeries", "name": "Cowboy Bebop", "numberOfEpisodes": 26}
        </script>
      </head>
      <body><section class="entryBar"><span class="type">TV (26 eps)</span></section></body>
    </html>
    """
    anime = map_animeplanet_anime_responses(
        [PageResponse(request_name="main", content_type="text/html", body=anime_html)]
    )
    assert anime["sources"] == ["https://www.anime-planet.com/anime/cowboy-bebop"]

    char_html = b"""
    <html>
      <head><link rel="canonical" href="https://www.anime-planet.com/characters/spike-spiegel" /></head>
      <body><h1 itemprop="name">Spike Spiegel</h1></body>
    </html>
    """
    char = map_animeplanet_character_responses(
        [PageResponse(request_name="main", content_type="text/html", body=char_html)]
    )
    assert char["sources"] == ["https://www.anime-planet.com/characters/spike-spiegel"]
