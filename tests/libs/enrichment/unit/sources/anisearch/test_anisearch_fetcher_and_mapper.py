"""Unit tests for AniSearchFetcher and response-level AniSearch mappers."""

from unittest.mock import patch

import pytest
from enrichment.sources.anisearch.anisearch_fetcher import AniSearchFetcher
from enrichment.sources.anisearch.anisearch_mapper import (
    map_anisearch_anime_episode_responses,
    map_anisearch_anime_responses,
    map_anisearch_character_responses,
)
from enrichment.sources.base.polite_http import FetchedPage as PoliteFetchedPage
from shared_proto.v1.anime_pb2 import EntityType
from shared_proto.v1.merge_pb2 import PageKey, PageResponse


@pytest.mark.asyncio
async def test_anisearch_fetcher_fetches_raw_anime_pages_and_maps() -> None:
    main_html = """
    <html><body>
      <section id="information">
        <div class="title" lang="en"><strong class="f16">Tanjou Debut</strong></div>
        <div class="type">OVA, 2 (1994)</div>
        <div class="status">Completed</div>
      </section>
    </body></html>
    """
    relations_html = (
        "<html><body><section id='relations_anime'></section></body></html>"
    )
    episodes_html = """
    <html><body>
      <table class="episodes"><tbody>
        <tr data-episode="true">
          <th itemprop="episodeNumber">1</th>
          <td data-title="Title"><div lang="en"><span itemprop="name">A new love</span></div></td>
          <td data-title="Runtime"><div lang="ja">25 min.</div></td>
          <td data-title="Date of Original Release"><div lang="ja">21.12.1994</div></td>
        </tr>
      </tbody></table>
    </body></html>
    """
    characters_html = "<html><body><section id='characters'></section></body></html>"

    pages_by_url = {
        "https://www.anisearch.com/anime/475": PoliteFetchedPage(
            url="https://www.anisearch.com/anime/475,tanjou-debut",
            html=main_html,
        ),
        "https://www.anisearch.com/anime/475,tanjou-debut/relations?show=overall": PoliteFetchedPage(
            url="https://www.anisearch.com/anime/475,tanjou-debut/relations?show=overall",
            html=relations_html,
        ),
        "https://www.anisearch.com/anime/475,tanjou-debut/episodes": PoliteFetchedPage(
            url="https://www.anisearch.com/anime/475,tanjou-debut/episodes",
            html=episodes_html,
        ),
        "https://www.anisearch.com/anime/475,tanjou-debut/characters": PoliteFetchedPage(
            url="https://www.anisearch.com/anime/475,tanjou-debut/characters",
            html=characters_html,
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
    assert [r.request_name for r in fetched.responses] == [
        "main",
        "relations",
        "episodes",
        "characters",
    ]
    anime = map_anisearch_anime_responses(
        fetched.responses, source_url="https://www.anisearch.com/anime/475,tanjou-debut"
    )
    assert anime["title"] == "Tanjou Debut"
    episodes = map_anisearch_anime_episode_responses(
        fetched.responses,
        source_url="https://www.anisearch.com/anime/475,tanjou-debut/episodes",
    )
    assert len(episodes) == 1
    assert episodes[0]["title"] == "A new love"


@pytest.mark.asyncio
async def test_anisearch_fetcher_reports_not_found_and_fetches_character() -> None:
    async def _fake_missing(_url: str) -> None:
        return None

    with patch(
        "enrichment.sources.anisearch.anisearch_fetcher.fetch_anisearch_page",
        side_effect=_fake_missing,
    ):
        missing = await AniSearchFetcher().fetch_page(
            PageKey(
                provider="anisearch",
                entity_type=EntityType.ENTITY_TYPE_ANIME,
                address="999999",
            )
        )
    assert missing.is_not_found is True
    assert missing.responses == ()

    char_html = """
    <html><body>
      <h1 id="htitle">Spike Spiegel</h1>
      <section id="description"><div lang="en" class="textblock">Space cowboy.</div></section>
    </body></html>
    """
    char_pages = {
        "https://www.anisearch.com/character/10": PoliteFetchedPage(
            url="https://www.anisearch.com/character/10,spike-spiegel",
            html=char_html,
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

    with patch(
        "enrichment.sources.anisearch.anisearch_fetcher.fetch_anisearch_page",
        side_effect=lambda url: char_pages.get(url),
    ):
        char_fetched = await AniSearchFetcher().fetch_page(
            PageKey(
                provider="anisearch",
                entity_type=EntityType.ENTITY_TYPE_CHARACTER,
                address="10",
            )
        )
    assert [r.request_name for r in char_fetched.responses] == [
        "main",
        "anime",
        "manga",
    ]
    mapped_char = map_anisearch_character_responses(
        char_fetched.responses,
        source_url="https://www.anisearch.com/character/10,spike-spiegel",
    )
    assert mapped_char["name"] == "Spike Spiegel"


def test_map_anisearch_anime_responses_requires_main() -> None:
    with pytest.raises(ValueError, match="main"):
        map_anisearch_anime_responses(
            [
                PageResponse(
                    request_name="episodes", content_type="text/html", body=b"<html/>"
                )
            ]
        )


def test_map_anisearch_responses_extract_canonical_url_when_omitted() -> None:
    anime_html = b"""
    <html>
      <head><link rel="canonical" href="https://www.anisearch.com/anime/475,tanjou-debut" /></head>
      <body>
        <section id="information">
          <div class="title" lang="en"><strong class="f16">Tanjou Debut</strong></div>
          <div class="type">OVA, 2 (1994)</div>
          <div class="status">Completed</div>
        </section>
      </body>
    </html>
    """
    anime = map_anisearch_anime_responses(
        [PageResponse(request_name="main", content_type="text/html", body=anime_html)]
    )
    assert anime["sources"] == ["https://www.anisearch.com/anime/475,tanjou-debut"]

    char_html = b"""
    <html>
      <head><link rel="canonical" href="https://www.anisearch.com/character/10,spike-spiegel" /></head>
      <body><h1 id="htitle">Spike Spiegel</h1></body>
    </html>
    """
    char = map_anisearch_character_responses(
        [PageResponse(request_name="main", content_type="text/html", body=char_html)]
    )
    assert char["sources"] == ["https://www.anisearch.com/character/10,spike-spiegel"]
