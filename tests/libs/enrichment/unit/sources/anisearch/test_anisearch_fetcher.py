from unittest.mock import patch

import pytest
from enrichment.sources.anisearch.anisearch_fetcher import AniSearchFetcher
from enrichment.sources.base.exceptions import ServiceNetworkError
from enrichment.sources.base.polite_http import FetchedPage as PoliteFetchedPage
from shared_proto.v1.anime_pb2 import EntityType
from shared_proto.v1.merge_pb2 import PageKey


@pytest.mark.asyncio
async def test_anisearch_fetcher_anime_page_returns_main_and_sub_pages() -> None:
    pages_by_url = {
        "https://www.anisearch.com/anime/475": PoliteFetchedPage(
            url="https://www.anisearch.com/anime/475,tanjou-debut",
            html="<html><body><section id='information'></section></body></html>",
        ),
        "https://www.anisearch.com/anime/475,tanjou-debut/relations?show=overall": PoliteFetchedPage(
            url="https://www.anisearch.com/anime/475,tanjou-debut/relations?show=overall",
            html="<html><body><section id='relations_anime'></section></body></html>",
        ),
        "https://www.anisearch.com/anime/475,tanjou-debut/episodes": PoliteFetchedPage(
            url="https://www.anisearch.com/anime/475,tanjou-debut/episodes",
            html="<html><body><table class='episodes'></table></body></html>",
        ),
        "https://www.anisearch.com/anime/475,tanjou-debut/characters": PoliteFetchedPage(
            url="https://www.anisearch.com/anime/475,tanjou-debut/characters",
            html="<html><body><section id='characters'></section></body></html>",
        ),
    }

    async def _fake_fetch(url: str) -> PoliteFetchedPage | None:
        return pages_by_url.get(url)

    with patch(
        "enrichment.sources.anisearch.anisearch_fetcher.fetch_anisearch_page",
        side_effect=_fake_fetch,
    ):
        fetched = await AniSearchFetcher().fetch_page(
            PageKey(
                provider="anisearch",
                entity_type=EntityType.ENTITY_TYPE_ANIME,
                address="475",
            )
        )

    assert fetched.is_not_found is False
    assert [response.request_name for response in fetched.responses] == [
        "main",
        "relations",
        "episodes",
        "characters",
    ]


@pytest.mark.asyncio
async def test_anisearch_fetcher_missing_anime_page_returns_not_found() -> None:
    async def _fake_missing(_url: str) -> None:
        return None

    with patch(
        "enrichment.sources.anisearch.anisearch_fetcher.fetch_anisearch_page",
        side_effect=_fake_missing,
    ):
        fetched = await AniSearchFetcher().fetch_page(
            PageKey(
                provider="anisearch",
                entity_type=EntityType.ENTITY_TYPE_ANIME,
                address="999999",
            )
        )

    assert fetched.is_not_found is True
    assert fetched.responses == ()


@pytest.mark.asyncio
async def test_anisearch_fetcher_empty_anime_html_raises_network_error() -> None:
    async def _fake_empty(url: str) -> PoliteFetchedPage:
        return PoliteFetchedPage(url=url, html="")

    with (
        patch(
            "enrichment.sources.anisearch.anisearch_fetcher.fetch_anisearch_page",
            side_effect=_fake_empty,
        ),
        pytest.raises(ServiceNetworkError, match="no HTML"),
    ):
        await AniSearchFetcher().fetch_page(
            PageKey(
                provider="anisearch",
                entity_type=EntityType.ENTITY_TYPE_ANIME,
                address="475",
            )
        )


@pytest.mark.asyncio
async def test_anisearch_fetcher_character_page_returns_main_and_ography_pages() -> (
    None
):
    char_pages = {
        "https://www.anisearch.com/character/10": PoliteFetchedPage(
            url="https://www.anisearch.com/character/10,spike-spiegel",
            html="<html><body><h1 id='htitle'>Spike Spiegel</h1></body></html>",
        ),
        "https://www.anisearch.com/character/10,spike-spiegel/anime": PoliteFetchedPage(
            url="https://www.anisearch.com/character/10,spike-spiegel/anime",
            html="<html><body></body></html>",
        ),
        "https://www.anisearch.com/character/10,spike-spiegel/manga": PoliteFetchedPage(
            url="https://www.anisearch.com/character/10,spike-spiegel/manga",
            html="<html><body></body></html>",
        ),
    }

    async def _fake_fetch(url: str) -> PoliteFetchedPage | None:
        return char_pages.get(url)

    with patch(
        "enrichment.sources.anisearch.anisearch_fetcher.fetch_anisearch_page",
        side_effect=_fake_fetch,
    ):
        fetched = await AniSearchFetcher().fetch_page(
            PageKey(
                provider="anisearch",
                entity_type=EntityType.ENTITY_TYPE_CHARACTER,
                address="10",
            )
        )

    assert fetched.is_not_found is False
    assert [response.request_name for response in fetched.responses] == [
        "main",
        "anime",
        "manga",
    ]


@pytest.mark.asyncio
async def test_anisearch_fetcher_missing_character_page_returns_not_found() -> None:
    async def _fake_missing(_url: str) -> None:
        return None

    with patch(
        "enrichment.sources.anisearch.anisearch_fetcher.fetch_anisearch_page",
        side_effect=_fake_missing,
    ):
        fetched = await AniSearchFetcher().fetch_page(
            PageKey(
                provider="anisearch",
                entity_type=EntityType.ENTITY_TYPE_CHARACTER,
                address="999999",
            )
        )

    assert fetched.is_not_found is True
    assert fetched.responses == ()


@pytest.mark.asyncio
async def test_anisearch_fetcher_empty_character_html_raises_network_error() -> None:
    async def _fake_empty(url: str) -> PoliteFetchedPage:
        return PoliteFetchedPage(url=url, html="")

    with (
        patch(
            "enrichment.sources.anisearch.anisearch_fetcher.fetch_anisearch_page",
            side_effect=_fake_empty,
        ),
        pytest.raises(ServiceNetworkError, match="no HTML"),
    ):
        await AniSearchFetcher().fetch_page(
            PageKey(
                provider="anisearch",
                entity_type=EntityType.ENTITY_TYPE_CHARACTER,
                address="10",
            )
        )
