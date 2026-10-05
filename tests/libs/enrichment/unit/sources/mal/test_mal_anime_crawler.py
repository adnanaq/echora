import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import create_autospec, patch

import pytest
import zendriver
from enrichment.sources.base.browser import BrowserSession
from enrichment.sources.base.framework import NullRepository
from enrichment.sources.mal import mal_anime_crawler
from enrichment.sources.mal.mal_anime_crawler import (
    _XPATHS,
    MalAnimeCrawler,
    _build_anime_from_raw,
    _extract_anime_from_html,
    _extract_pics_from_html,
    _fetch_mal_anime_data,
    _fetch_pics_html,
    _normalize_mal_url,
    _parse_all_related_entries,
    _parse_structured_themes,
    _parse_trailer,
    fetch_mal_anime,
    main,
)
from http_cache import result_cache
from http_cache.config import CacheConfig
from lxml import html as lxml_html

ONE_PIECE_URL = "https://myanimelist.net/anime/21"
ONE_PIECE_CANONICAL_URL = "https://myanimelist.net/anime/21/One_Piece"
DELETED_URL = "https://myanimelist.net/anime/60661"


def _page_has(page_html: str, selector: str) -> bool:
    if not page_html:
        return False
    document = lxml_html.fromstring(page_html)
    for part in selector.split(","):
        tag, css_class = part.strip().split(".")
        class_test = (
            f"contains(concat(' ', normalize-space(@class), ' '), ' {css_class} ')"
        )
        if document.xpath(f"//{tag}[{class_test}]"):
            return True
    return False


def _tab_showing(page_html: str, url: str) -> zendriver.Tab:
    tab = create_autospec(zendriver.Tab, instance=True)

    def wait_for(selector=None, text=None, timeout=10):
        if not _page_has(page_html, selector):
            raise TimeoutError(f"{selector} not on page")

    tab.wait_for.side_effect = wait_for
    tab.evaluate.return_value = "complete"
    tab.get_content.return_value = page_html
    tab.url = url
    return tab


def _browser_serving(pages: dict[str, str]) -> zendriver.Browser:
    browser = create_autospec(zendriver.Browser, instance=True)

    def get(url="about:blank", new_tab=False, new_window=False):
        return _tab_showing(pages.get(url, ""), url)

    browser.get.side_effect = get
    return browser


def _one_piece_site(main_html: str, gallery_html: str) -> dict[str, str]:
    return {
        ONE_PIECE_URL: main_html,
        f"{ONE_PIECE_URL}/pics": main_html,
        f"{ONE_PIECE_CANONICAL_URL}/pics": gallery_html,
    }


@pytest.fixture
def open_browser():
    def install(browser: zendriver.Browser):
        @asynccontextmanager
        async def browser_session(**settings) -> AsyncIterator[BrowserSession]:
            yield BrowserSession(
                headless=settings["headless"],
                allowed_site=settings.get("allowed_site"),
                clearance_site=None,
                block_unused_resources=True,
                browser=browser,
            )

        return patch.object(mal_anime_crawler, "browser_session", browser_session)

    with (
        patch.object(
            result_cache,
            "get_cache_config",
            autospec=True,
            return_value=CacheConfig(cache_enabled=False),
        ),
        patch.object(mal_anime_crawler, "_INTER_REQUEST_DELAY", 0),
    ):
        yield install


def _build(raw: dict, picture_urls: list[str] | None = None):
    return _build_anime_from_raw(
        raw, url=ONE_PIECE_CANONICAL_URL, picture_urls=picture_urls or []
    )


def test_xpaths_opening_theme_rows_matches_mal_misspelled_class() -> None:
    assert "opnening" in _XPATHS["opening_theme_rows"]


def test_xpaths_link_and_background_queries_target_their_sections() -> None:
    assert "ending" in _XPATHS["ending_theme_rows"]
    assert "Available At" in _XPATHS["external_source_anchors"]
    assert "Resources" in _XPATHS["external_source_anchors"]
    assert "external_links" in _XPATHS["external_source_anchors"]
    assert "Streaming Platforms" in _XPATHS["streaming_anchors"]
    assert "@title" in _XPATHS["streaming_anchors"]
    assert "background" in _XPATHS["background_raw"]
    assert "parent::td" in _XPATHS["background_raw"]


def test_extract_anime_from_html_empty_string_returns_none() -> None:
    assert _extract_anime_from_html("") is None


def test_extract_anime_from_html_empty_page_returns_empty_fields() -> None:
    raw = _extract_anime_from_html("<html><body></body></html>")
    assert raw is not None
    assert raw["title"] is None
    assert raw["genres"] == []
    assert raw["related_tile_entries"] == []
    assert raw["related_table_entries"] == []


def test_extract_anime_from_html_one_piece_page_reads_titles(mal_anime_html) -> None:
    raw = _extract_anime_from_html(mal_anime_html)
    assert (
        raw["title"],
        raw["title_og"],
        raw["title_english"],
        raw["title_japanese"],
    ) == ("One Piece", "One Piece", "One Piece", "ONE PIECE")


def test_extract_anime_from_html_one_piece_page_reads_sidebar(mal_anime_html) -> None:
    raw = _extract_anime_from_html(mal_anime_html)
    assert raw["type"] == "TV"
    assert raw["status"] == "Currently Airing"
    assert raw["source_material"] == "Manga"
    assert raw["episodes"] == "Unknown"
    assert "PG-13" in (raw["rating"] or "")
    assert raw["aired_raw"] is not None and "1999" in raw["aired_raw"]
    assert raw["premiered_raw"] == "Fall 1999"
    assert raw["broadcast_raw"] is not None and "JST" in raw["broadcast_raw"]


def test_extract_anime_from_html_one_piece_page_reads_statistics(
    mal_anime_html,
) -> None:
    raw = _extract_anime_from_html(mal_anime_html)
    assert raw["score"] == "8.73"
    assert raw["rank_html"] is not None and "#" in raw["rank_html"]
    assert raw["popularity"] is not None and raw["popularity"].isdigit()
    assert raw["members"] is not None and "," in raw["members"]
    assert raw["synopsis"] is not None and "Luffy" in raw["synopsis"]
    assert (
        raw["cover_image_src"] is not None and "myanimelist" in raw["cover_image_src"]
    )


def test_extract_anime_from_html_one_piece_page_reads_genres_demographics_and_studio(
    mal_anime_html,
) -> None:
    raw = _extract_anime_from_html(mal_anime_html)
    assert "Action" in [genre["name"] for genre in raw["genres"]]
    assert "Shounen" in [demographic["name"] for demographic in raw["demographics"]]
    assert any("Toei" in studio["name"] for studio in raw["studios"])


def test_extract_anime_from_html_one_piece_page_reads_links(mal_anime_html) -> None:
    raw = _extract_anime_from_html(mal_anime_html)
    assert "Official Site" in [link["name"] for link in raw["external_sources_raw"]]
    assert "Crunchyroll" in [link["name"] for link in raw["streaming_links_raw"]]


def test_extract_anime_from_html_one_piece_page_reads_trailer_background_and_related(
    mal_anime_html,
) -> None:
    raw = _extract_anime_from_html(mal_anime_html)
    assert "youtube" in (raw["trailer_embed_url"] or "").lower()
    assert 'id="background"' in (raw["background_raw"] or "")
    assert "One Piece" in [entry["title"] for entry in raw["related_tile_entries"]]
    assert len(raw["related_table_entries"]) >= 1


def test_extract_anime_from_html_one_piece_page_reads_every_theme_song(
    mal_anime_html,
) -> None:
    raw = _extract_anime_from_html(mal_anime_html)
    openings = [
        row for row in raw["opening_themes_raw"] if '"' in (row.get("title_text") or "")
    ]
    endings = [
        row for row in raw["ending_themes_raw"] if '"' in (row.get("title_text") or "")
    ]
    assert (len(openings), len(endings)) == (30, 27)


def test_extract_anime_from_html_one_piece_page_reads_canonical_url(
    mal_anime_html,
) -> None:
    raw = _extract_anime_from_html(mal_anime_html)
    assert raw["canonical_url"] == ONE_PIECE_CANONICAL_URL


def test_extract_pics_from_html_empty_string_returns_empty_list() -> None:
    assert _extract_pics_from_html("") == []


def test_extract_pics_from_html_keeps_only_anime_images() -> None:
    page = """<html><body>
      <div class="picSurround"><a href="https://cdn.myanimelist.net/images/anime/1/123l.jpg">x</a></div>
      <div class="picSurround"><a href="https://cdn.myanimelist.net/images/characters/1/456.jpg">x</a></div>
      <div class="picSurround"><a href="https://otherdomain.com/image.jpg">x</a></div>
    </body></html>"""
    assert _extract_pics_from_html(page) == [
        "https://cdn.myanimelist.net/images/anime/1/123l.jpg"
    ]


def test_extract_pics_from_html_one_piece_gallery_reads_every_image(
    mal_anime_pics_html,
) -> None:
    urls = _extract_pics_from_html(mal_anime_pics_html)
    assert len(urls) == 20
    assert all(
        url.startswith("https://cdn.myanimelist.net/images/anime/") for url in urls
    )


async def test_fetch_pics_html_returns_page_content(mal_anime_pics_html) -> None:
    browser = _browser_serving({f"{ONE_PIECE_CANONICAL_URL}/pics": mal_anime_pics_html})

    result = await _fetch_pics_html(browser, f"{ONE_PIECE_CANONICAL_URL}/pics")

    assert result == mal_anime_pics_html


async def test_fetch_pics_html_navigation_error_returns_none() -> None:
    browser = create_autospec(zendriver.Browser, instance=True)
    browser.get.side_effect = ConnectionError("navigation failed")

    assert await _fetch_pics_html(browser, f"{ONE_PIECE_CANONICAL_URL}/pics") is None


def test_build_anime_from_raw_prefers_canonical_url_over_requested_url() -> None:
    anime = _build_anime_from_raw(
        {"canonical_url": ONE_PIECE_CANONICAL_URL}, url=ONE_PIECE_URL, picture_urls=[]
    )
    assert anime.source == ONE_PIECE_CANONICAL_URL


def test_build_anime_from_raw_without_canonical_url_keeps_requested_url() -> None:
    anime = _build_anime_from_raw({}, url=ONE_PIECE_URL, picture_urls=[])
    assert anime.source == ONE_PIECE_URL


def test_build_anime_from_raw_without_premiered_takes_year_from_aired() -> None:
    anime = _build({"aired_raw": "Mar 5, 2027"})
    assert (anime.year, anime.aired_from, anime.month) == (2027, "2027-03-05", None)


def test_build_anime_from_raw_closed_range_sets_both_dates() -> None:
    anime = _build({"aired_raw": "Oct 20, 1999 to Nov 5, 2000"})
    assert (anime.aired_from, anime.aired_to) == ("1999-10-20", "2000-11-05")


def test_build_anime_from_raw_with_premiered_keeps_premiered_year() -> None:
    anime = _build({"aired_raw": "Dec 27, 2026 to ?", "premiered_raw": "Winter 2027"})
    assert (anime.year, anime.season) == (2027, "winter")


def test_build_anime_from_raw_month_and_year_aired_sets_year_and_month_without_date() -> (
    None
):
    anime = _build({"aired_raw": "Oct 1977"})
    assert (anime.year, anime.month, anime.aired_from) == (1977, "October", None)


def test_build_anime_from_raw_year_only_aired_sets_year_without_date_or_month() -> None:
    anime = _build({"aired_raw": "1988"})
    assert (anime.year, anime.month, anime.aired_from) == (1988, None, None)


def test_build_anime_from_raw_one_piece_page_reads_core_fields(
    mal_anime_extracted,
) -> None:
    anime = _build(mal_anime_extracted)
    assert anime.title == "One Piece"
    assert anime.score == pytest.approx(8.73)
    assert (anime.aired_from, anime.aired_to) == ("1999-10-20", None)
    assert (anime.season, anime.year) == ("fall", 1999)
    assert anime.broadcast_timezone == "JST"
    assert anime.broadcast_day is not None and anime.broadcast_time is not None
    assert isinstance(anime.rank, int) and anime.rank > 0


def test_build_anime_from_raw_one_piece_page_reads_genres_demographics_and_studio(
    mal_anime_extracted,
) -> None:
    anime = _build(mal_anime_extracted)
    assert {"Action", "Adventure"} <= set(anime.genres)
    assert "Shounen" in anime.demographics
    assert len(anime.studios) == 1 and "Toei" in anime.studios[0].name


def test_build_anime_from_raw_one_piece_page_separates_external_and_streaming_links(
    mal_anime_extracted,
) -> None:
    anime = _build(mal_anime_extracted)
    external_names = {link.name for link in anime.external_sources}
    streaming_names = {link.name for link in anime.streaming}
    assert (len(anime.external_sources), len(anime.streaming)) == (11, 3)
    assert "Official Site" in external_names
    assert "Crunchyroll" in streaming_names
    assert external_names.isdisjoint(streaming_names)


def test_build_anime_from_raw_one_piece_page_reads_synopsis_background_and_pictures(
    mal_anime_extracted,
) -> None:
    anime = _build(mal_anime_extracted)
    assert anime.synopsis is not None and "Luffy" in anime.synopsis
    assert anime.background is not None and len(anime.background) > 10
    assert len(anime.picture_urls) >= 1


def test_build_anime_from_raw_one_piece_page_reads_theme_songs_without_platform_links(
    mal_anime_extracted,
) -> None:
    anime = _build(mal_anime_extracted)
    opening_titles = [theme.title for theme in anime.opening_themes]
    assert {"We Are! (ウィーアー!)", "Believe"} <= set(opening_titles)
    assert not {"Apple Music", "Youtube Music"} & set(opening_titles)
    assert "memories" in [theme.title for theme in anime.ending_themes]


def test_build_anime_from_raw_one_piece_page_reads_related_entries_and_trailer(
    mal_anime_extracted,
) -> None:
    anime = _build(mal_anime_extracted)
    related_titles = [entry.title for entry in anime.related_entries]
    assert "One Piece" in related_titles
    assert any("Ganzack" in title for title in related_titles)
    assert anime.trailer is not None


def test_build_anime_from_raw_unknown_episode_count_gives_none(
    mal_anime_extracted,
) -> None:
    assert _build({**mal_anime_extracted, "episodes": "Unknown"}).episode_count is None


def test_build_anime_from_raw_numeric_episode_count_gives_integer(
    mal_anime_extracted,
) -> None:
    assert _build({**mal_anime_extracted, "episodes": "1080"}).episode_count == 1080


def test_build_anime_from_raw_non_numeric_episode_count_gives_none(
    mal_anime_extracted,
) -> None:
    assert _build({**mal_anime_extracted, "episodes": "TBD"}).episode_count is None


def test_build_anime_from_raw_rank_html_gives_rank_number(mal_anime_extracted) -> None:
    assert _build({**mal_anime_extracted, "rank_html": "#17<sup>2</sup>"}).rank == 17


def test_build_anime_from_raw_missing_rank_gives_none(mal_anime_extracted) -> None:
    assert _build({**mal_anime_extracted, "rank_html": None}).rank is None


def test_build_anime_from_raw_add_some_placeholder_companies_dropped(
    mal_anime_extracted,
) -> None:
    raw = {
        **mal_anime_extracted,
        "licensors": [
            {
                "name": "add some",
                "source": "https://myanimelist.net/dbchanges.php?aid=1&t=producers",
            }
        ],
        "studios": [
            {
                "name": "add some",
                "source": "https://myanimelist.net/dbchanges.php?aid=2&t=producers",
            }
        ],
        "producers": [
            {
                "name": "Arch",
                "source": "https://myanimelist.net/anime/producer/1966/Arch",
            }
        ],
    }
    anime = _build(raw)
    assert (anime.licensors, anime.studios) == ([], [])
    assert [producer.name for producer in anime.producers] == ["Arch"]


def test_build_anime_from_raw_external_link_without_name_dropped(
    mal_anime_extracted,
) -> None:
    raw = {
        **mal_anime_extracted,
        "external_sources_raw": [
            {"name": "", "source": "https://example.com"},
            {"name": "Valid", "source": "https://valid.com"},
        ],
    }
    assert [link.name for link in _build(raw).external_sources] == ["Valid"]


def test_build_anime_from_raw_streaming_link_without_address_dropped(
    mal_anime_extracted,
) -> None:
    raw = {
        **mal_anime_extracted,
        "streaming_links_raw": [
            {"name": "Crunchyroll", "source": ""},
            {"name": "Netflix", "source": "https://netflix.com"},
        ],
    }
    assert [link.name for link in _build(raw).streaming] == ["Netflix"]


def test_build_anime_from_raw_without_link_sections_gives_empty_link_lists(
    mal_anime_extracted,
) -> None:
    raw = {
        key: value
        for key, value in mal_anime_extracted.items()
        if key not in ("external_sources_raw", "streaming_links_raw")
    }
    anime = _build(raw)
    assert (anime.external_sources, anime.streaming) == ([], [])


def test_build_anime_from_raw_background_section_gives_background_text() -> None:
    background = '<td><div><h2 id="background">Background</h2></div>The story begins in the Grand Line.</td>'
    anime = _build({"title": "Test", "background_raw": background})
    assert anime.background is not None and "Grand Line" in anime.background


def test_build_anime_from_raw_placeholder_background_gives_none(
    mal_anime_extracted,
) -> None:
    placeholder = '<td><div><h2 id="background">Background</h2></div>No background information has been added to this title.</td>'
    assert (
        _build({**mal_anime_extracted, "background_raw": placeholder}).background
        is None
    )


def test_build_anime_from_raw_cover_image_adds_large_version_to_pictures() -> None:
    anime = _build(
        {"cover_image_src": "https://cdn.myanimelist.net/images/anime/1/123.jpg"}
    )
    assert anime.picture_urls == ["https://cdn.myanimelist.net/images/anime/1/123l.jpg"]


def test_build_anime_from_raw_gallery_picture_matching_cover_kept_once() -> None:
    cover_large = "https://cdn.myanimelist.net/images/anime/1/123l.jpg"
    gallery = [cover_large, "https://cdn.myanimelist.net/images/anime/1/456l.jpg"]
    anime = _build(
        {"cover_image_src": "https://cdn.myanimelist.net/images/anime/1/123.jpg"},
        picture_urls=gallery,
    )
    assert anime.picture_urls == gallery


def test_build_anime_from_raw_synonyms_split_on_commas(mal_anime_extracted) -> None:
    anime = _build({**mal_anime_extracted, "synonyms_raw": "OP, One Piece TV"})
    assert {"OP", "One Piece TV"} <= set(anime.synonyms)


def test_build_anime_from_raw_duration_per_episode_gives_seconds(
    mal_anime_extracted,
) -> None:
    anime = _build({**mal_anime_extracted, "duration_raw": "24 min. per ep."})
    assert anime.duration == 1440


def test_build_anime_from_raw_missing_title_takes_page_title(
    mal_anime_extracted,
) -> None:
    anime = _build({**mal_anime_extracted, "title": None, "title_og": "Fallback"})
    assert anime.title == "Fallback"


def test_parse_trailer_youtube_embed_gives_video_and_thumbnail() -> None:
    trailer = _parse_trailer(
        {
            "trailer_embed_url": "https://www.youtube.com/embed/abc123?autoplay=1",
            "trailer_title": "T",
        }
    )
    assert trailer is not None
    assert "abc123" in trailer.source and "abc123" in trailer.thumbnail


def test_parse_trailer_missing_embed_returns_none() -> None:
    assert _parse_trailer({}) is None


def test_parse_trailer_non_youtube_embed_returns_none() -> None:
    assert _parse_trailer({"trailer_embed_url": "https://vimeo.com/12345"}) is None


def test_normalize_mal_url_empty_string_returns_empty_string() -> None:
    assert _normalize_mal_url("") == ""


def test_normalize_mal_url_full_url_unchanged() -> None:
    assert _normalize_mal_url(ONE_PIECE_URL) == ONE_PIECE_URL


def test_normalize_mal_url_path_with_leading_slash_gives_full_url() -> None:
    assert _normalize_mal_url("/anime/21") == ONE_PIECE_URL


def test_normalize_mal_url_path_without_leading_slash_gives_full_url() -> None:
    assert _normalize_mal_url("anime/21") == ONE_PIECE_URL


def test_parse_structured_themes_quoted_title_gives_title_artist_and_episodes() -> None:
    themes = _parse_structured_themes(
        [
            {
                "title_text": '"We Are!" by Hiroshi Kitadani',
                "artist": "Hiroshi Kitadani",
                "episodes": "1-130",
            }
        ]
    )
    assert len(themes) == 1
    assert (themes[0].title, themes[0].artist) == ("We Are!", "Hiroshi Kitadani")
    assert len(themes[0].episodes) == 1


def test_parse_structured_themes_unquoted_row_skipped() -> None:
    rows = [{"title_text": "Listen on Spotify", "artist": "", "episodes": None}]
    assert _parse_structured_themes(rows) == []


def test_parse_structured_themes_by_prefix_removed_from_artist() -> None:
    rows = [{"title_text": '"Kokoro e"', "artist": "by Rhythm", "episodes": None}]
    assert _parse_structured_themes(rows)[0].artist == "Rhythm"


def test_parse_structured_themes_empty_artist_gives_none() -> None:
    rows = [{"title_text": '"Kokoro e"', "artist": "", "episodes": None}]
    assert _parse_structured_themes(rows)[0].artist is None


def test_parse_all_related_entries_tile_gives_relation_title_and_type() -> None:
    raw = {
        "related_tile_entries": [
            {
                "relation_raw": "Adaptation\n(Manga)",
                "title": "One Piece Manga",
                "source": "/manga/103",
            },
        ],
        "related_table_entries": [],
    }
    entries = _parse_all_related_entries(raw)
    assert [(entry.relation, entry.title, entry.entry_type) for entry in entries] == [
        ("Adaptation", "One Piece Manga", "Manga")
    ]


def test_parse_all_related_entries_tile_with_stated_type_keeps_it() -> None:
    raw = {
        "related_tile_entries": [
            {
                "relation_raw": "Sequel\n(OVA)",
                "title": "Test",
                "entry_type": "Movie",
                "source": "/anime/999",
            },
        ],
        "related_table_entries": [],
    }
    assert _parse_all_related_entries(raw)[0].entry_type == "Movie"


def test_parse_all_related_entries_table_link_type_gives_entry_type() -> None:
    links_html = (
        '<ul><li><a href="https://myanimelist.net/anime/22/S">Sequel</a> (TV)</li></ul>'
    )
    raw = {
        "related_tile_entries": [],
        "related_table_entries": [{"relation": "Sequel", "links_html": links_html}],
    }
    assert _parse_all_related_entries(raw)[0].entry_type == "TV"


def test_parse_all_related_entries_table_relation_type_gives_entry_type() -> None:
    links_html = (
        '<ul><li><a href="https://myanimelist.net/anime/22/S">Sequel</a></li></ul>'
    )
    raw = {
        "related_tile_entries": [],
        "related_table_entries": [
            {"relation": "Sequel\n(TV)", "links_html": links_html}
        ],
    }
    assert _parse_all_related_entries(raw)[0].entry_type == "TV"


def test_parse_all_related_entries_missing_title_or_source_skipped() -> None:
    raw = {
        "related_tile_entries": [
            {"relation_raw": "Adaptation", "title": "", "source": "/manga/103"},
            {"relation_raw": "Adaptation", "title": "Valid", "source": ""},
        ],
        "related_table_entries": [],
    }
    assert _parse_all_related_entries(raw) == []


def test_parse_all_related_entries_one_piece_page_reads_related_works(
    mal_anime_extracted,
) -> None:
    titles = [entry.title for entry in _parse_all_related_entries(mal_anime_extracted)]
    assert "One Piece" in titles
    assert any("Ganzack" in title for title in titles)


async def test_fetch_mal_anime_data_bare_url_returns_page_data_and_gallery_pictures(
    open_browser, mal_anime_html, mal_anime_pics_html
) -> None:
    browser = _browser_serving(_one_piece_site(mal_anime_html, mal_anime_pics_html))
    with open_browser(browser):
        result = await _fetch_mal_anime_data(ONE_PIECE_URL)

    assert result is not None
    assert result["title"] == "One Piece"
    assert result["_url"] == ONE_PIECE_CANONICAL_URL
    assert result["_picture_urls"] == _extract_pics_from_html(mal_anime_pics_html)


async def test_fetch_mal_anime_data_deleted_page_logs_not_found_and_returns_none(
    open_browser, mal_anime_not_found_html, caplog
) -> None:
    browser = _browser_serving({DELETED_URL: mal_anime_not_found_html})
    with open_browser(browser), caplog.at_level(logging.WARNING):
        result = await _fetch_mal_anime_data(DELETED_URL)

    assert result is None
    assert f"MAL anime page not found: {DELETED_URL}" in caplog.messages


async def test_fetch_mal_anime_data_page_without_title_logs_navigation_failure(
    open_browser, caplog
) -> None:
    browser = _browser_serving({ONE_PIECE_URL: "<html><body></body></html>"})
    with open_browser(browser), caplog.at_level(logging.WARNING):
        result = await _fetch_mal_anime_data(ONE_PIECE_URL)

    assert result is None
    assert any(
        message.startswith(f"navigation failed for {ONE_PIECE_URL}")
        for message in caplog.messages
    )


async def test_fetch_mal_anime_data_empty_content_returns_none(
    open_browser, mal_anime_html
) -> None:
    browser = create_autospec(zendriver.Browser, instance=True)
    browser.get.return_value = _tab_showing(mal_anime_html, ONE_PIECE_URL)
    browser.get.return_value.get_content.return_value = ""
    with open_browser(browser):
        assert await _fetch_mal_anime_data(ONE_PIECE_URL) is None


async def test_fetch_mal_anime_data_gallery_failure_returns_data_without_pictures(
    open_browser, mal_anime_html
) -> None:
    browser = _browser_serving(
        _one_piece_site(mal_anime_html, "<html><body></body></html>")
    )
    with open_browser(browser):
        result = await _fetch_mal_anime_data(ONE_PIECE_URL)

    assert result is not None and result["title"] == "One Piece"
    assert result["_picture_urls"] == []


def test_get_extraction_schema_returns_xpaths() -> None:
    assert MalAnimeCrawler(NullRepository()).get_extraction_schema() == {
        "xpaths": _XPATHS
    }


def test_normalize_identifier_path_gives_full_url() -> None:
    assert (
        MalAnimeCrawler(NullRepository()).normalize_identifier("/anime/21")
        == ONE_PIECE_URL
    )


async def test_fetch_raw_data_one_piece_page_returns_page_data(
    open_browser, mal_anime_html, mal_anime_pics_html
) -> None:
    browser = _browser_serving(_one_piece_site(mal_anime_html, mal_anime_pics_html))
    with open_browser(browser):
        raw = await MalAnimeCrawler(NullRepository()).fetch_raw_data(ONE_PIECE_URL)

    assert raw is not None and raw["title"] == "One Piece"


def test_build_source_model_one_piece_page_gives_title_and_drops_internal_keys(
    mal_anime_extracted,
) -> None:
    raw = dict(mal_anime_extracted)
    anime = MalAnimeCrawler(NullRepository()).build_source_model(raw, ONE_PIECE_URL)
    assert anime.title == "One Piece"
    assert not {"_picture_urls", "_url"} & raw.keys()


def test_map_to_canonical_one_piece_page_gives_canonical_title(
    mal_anime_extracted,
) -> None:
    crawler = MalAnimeCrawler(NullRepository())
    anime = crawler.build_source_model(dict(mal_anime_extracted), ONE_PIECE_URL)
    assert crawler.map_to_canonical(anime)["title"] == "One Piece"


async def test_fetch_mal_anime_one_piece_page_returns_canonical_anime(
    open_browser, mal_anime_html, mal_anime_pics_html
) -> None:
    browser = _browser_serving(_one_piece_site(mal_anime_html, mal_anime_pics_html))
    with open_browser(browser):
        anime = await fetch_mal_anime(ONE_PIECE_URL)

    assert anime is not None and anime["title"] == "One Piece"


async def test_fetch_mal_anime_deleted_page_returns_none(
    open_browser, mal_anime_not_found_html
) -> None:
    browser = _browser_serving({DELETED_URL: mal_anime_not_found_html})
    with open_browser(browser):
        assert await fetch_mal_anime(DELETED_URL) is None


async def test_main_without_data_returns_one(tmp_path) -> None:
    arguments = ["prog", ONE_PIECE_URL, "--output", str(tmp_path / "out.json")]
    with (
        patch.object(sys, "argv", arguments),
        patch.object(
            mal_anime_crawler, "fetch_mal_anime", autospec=True, return_value=None
        ),
    ):
        assert await main() == 1


async def test_main_with_data_returns_zero(tmp_path) -> None:
    arguments = ["prog", ONE_PIECE_URL, "--output", str(tmp_path / "out.json")]
    with (
        patch.object(sys, "argv", arguments),
        patch.object(
            mal_anime_crawler,
            "fetch_mal_anime",
            autospec=True,
            return_value={"title": "One Piece", "episode_count": 1000},
        ),
    ):
        assert await main() == 0
