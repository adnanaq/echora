"""Unit tests for mal_anime_crawler.py — lxml XPath extraction and post-processing.

HTML fixtures are real pages captured from:
- https://myanimelist.net/anime/21 (2026-06-09) — main anime page
- https://myanimelist.net/anime/21/One_Piece/pics (2026-06-09) — gallery page

mal_anime_extracted is the XPath-extracted dict derived from the HTML fixture.
"""

from unittest.mock import AsyncMock

import pytest
from enrichment.sources.mal.mal_anime_crawler import (
    _XPATHS,
    _build_anime_from_raw,
    _extract_anime_from_html,
    _extract_pics_from_html,
    _fetch_mal_anime_data,
    _fetch_page_html,
    _fetch_pics_html,
    _normalize_mal_url,
    _parse_all_related_entries,
    _parse_structured_themes,
    _parse_trailer,
    fetch_mal_anime,
)

# =============================================================================
# _XPATHS — structural invariants
# =============================================================================


def test_xpaths_key_selectors() -> None:
    assert "opnening" in _XPATHS["opening_theme_rows"]  # MAL typo — intentional
    assert "ending" in _XPATHS["ending_theme_rows"]
    assert "Available At" in _XPATHS["external_source_anchors"]
    assert "Resources" in _XPATHS["external_source_anchors"]
    assert "external_links" in _XPATHS["external_source_anchors"]
    assert "Streaming Platforms" in _XPATHS["streaming_anchors"]
    assert "@title" in _XPATHS["streaming_anchors"]
    assert "background" in _XPATHS["background_raw"]
    assert "parent::td" in _XPATHS["background_raw"]


# =============================================================================
# _extract_anime_from_html
# =============================================================================


def test_extract_anime_empty_html_returns_none() -> None:
    assert _extract_anime_from_html("") is None


def test_extract_anime_minimal_html_returns_partial_dict() -> None:
    raw = _extract_anime_from_html("<html><body></body></html>")
    assert raw is not None
    assert raw["title"] is None
    assert raw["genres"] == []
    assert raw["related_tile_entries"] == []
    assert raw["related_table_entries"] == []


def test_extract_anime_from_fixture(mal_anime_html) -> None:
    raw = _extract_anime_from_html(mal_anime_html)
    assert raw is not None

    # Titles
    assert raw["title"] == "One Piece"
    assert raw["title_og"] == "One Piece"
    assert raw["title_english"] == "One Piece"
    assert raw["title_japanese"] == "ONE PIECE"

    # Sidebar
    assert raw["type"] == "TV"
    assert raw["status"] == "Currently Airing"
    assert raw["source_material"] == "Manga"
    assert raw["episodes"] == "Unknown"
    assert "PG-13" in (raw["rating"] or "")
    assert raw["aired_raw"] is not None and "1999" in raw["aired_raw"]
    assert raw["premiered_raw"] == "Fall 1999"
    assert raw["broadcast_raw"] is not None and "JST" in raw["broadcast_raw"]

    # Stats
    assert raw["score"] == "8.73"
    assert raw["rank_html"] is not None and "#" in raw["rank_html"]
    assert raw["popularity"] is not None and raw["popularity"].isdigit()
    assert raw["members"] is not None and "," in raw["members"]
    assert raw["synopsis"] is not None and "Luffy" in raw["synopsis"]
    assert raw["cover_image_src"] is not None and "myanimelist" in raw["cover_image_src"]

    # Taxonomy
    assert "Action" in [g["name"] for g in raw["genres"]]
    assert "Shounen" in [d["name"] for d in raw["demographics"]]
    assert any("Toei" in s["name"] for s in raw["studios"])

    # Links
    assert "Official Site" in [e["name"] for e in raw["external_sources_raw"]]
    assert "Crunchyroll" in [s["name"] for s in raw["streaming_links_raw"]]

    # Content sections
    assert raw["trailer_embed_url"] is not None and "youtube" in raw["trailer_embed_url"].lower()
    assert raw["background_raw"] is not None and 'id="background"' in raw["background_raw"]
    assert "One Piece" in [e["title"] for e in raw["related_tile_entries"]]
    assert len(raw["related_table_entries"]) >= 1

    # Theme song counts match benchmark
    valid_opens = [r for r in raw["opening_themes_raw"] if '"' in (r.get("title_text") or "")]
    valid_ends = [r for r in raw["ending_themes_raw"] if '"' in (r.get("title_text") or "")]
    assert len(valid_opens) == 30
    assert len(valid_ends) == 27


# =============================================================================
# _extract_pics_from_html
# =============================================================================


def test_extract_pics_empty_html_returns_empty() -> None:
    assert _extract_pics_from_html("") == []


def test_extract_pics_filters_non_anime_urls() -> None:
    html = """<html><body>
      <div class="picSurround"><a href="https://cdn.myanimelist.net/images/anime/1/123l.jpg">x</a></div>
      <div class="picSurround"><a href="https://cdn.myanimelist.net/images/characters/1/456.jpg">x</a></div>
      <div class="picSurround"><a href="https://otherdomain.com/image.jpg">x</a></div>
    </body></html>"""
    urls = _extract_pics_from_html(html)
    assert len(urls) == 1 and "123l.jpg" in urls[0]


# =============================================================================
# _fetch_page_html
# =============================================================================


@pytest.mark.asyncio
async def test_fetch_page_html_with_wait_selector(mocker) -> None:
    page = mocker.AsyncMock()
    page.wait_for = AsyncMock()
    page.get_content = AsyncMock(return_value="<html>ok</html>")
    browser = mocker.AsyncMock()
    browser.get = AsyncMock(return_value=page)

    result = await _fetch_page_html(browser, "https://example.com", wait_selector="div.main")
    page.wait_for.assert_called_once()
    assert result == "<html>ok</html>"


@pytest.mark.asyncio
async def test_fetch_page_html_without_wait_selector_uses_sleep(mocker) -> None:
    page = mocker.AsyncMock()
    page.get_content = AsyncMock(return_value="<html>ok</html>")
    browser = mocker.AsyncMock()
    browser.get = AsyncMock(return_value=page)
    mocker.patch("asyncio.sleep", new_callable=AsyncMock)

    result = await _fetch_page_html(browser, "https://example.com")
    assert result == "<html>ok</html>"


@pytest.mark.asyncio
async def test_fetch_page_html_exception_returns_none(mocker) -> None:
    browser = mocker.AsyncMock()
    browser.get = AsyncMock(side_effect=Exception("timeout"))
    assert await _fetch_page_html(browser, "https://example.com") is None


# =============================================================================
# _fetch_pics_html
# =============================================================================


@pytest.mark.asyncio
async def test_fetch_pics_html_scrolls_and_returns_content(mocker) -> None:
    page = mocker.AsyncMock()
    page.wait_for = AsyncMock()
    page.scroll_down = AsyncMock()
    page.get_content = AsyncMock(return_value="<html>pics</html>")
    browser = mocker.AsyncMock()
    browser.get = AsyncMock(return_value=page)
    mocker.patch("asyncio.sleep", new_callable=AsyncMock)

    result = await _fetch_pics_html(browser, "https://myanimelist.net/anime/21/One_Piece/pics")
    page.scroll_down.assert_called_once()
    assert result == "<html>pics</html>"


@pytest.mark.asyncio
async def test_fetch_pics_html_exception_returns_none(mocker) -> None:
    browser = mocker.AsyncMock()
    browser.get = AsyncMock(side_effect=Exception("nav failed"))
    assert await _fetch_pics_html(browser, "https://myanimelist.net/anime/21/One_Piece/pics") is None


# =============================================================================
# _build_anime_from_raw
# =============================================================================


def _build(raw: dict, picture_urls: list[str] | None = None):
    return _build_anime_from_raw(
        raw,
        url="https://myanimelist.net/anime/21/One_Piece",
        picture_urls=picture_urls or [],
    )


def test_build_from_fixture(mal_anime_extracted) -> None:
    anime = _build(mal_anime_extracted)

    # Core fields
    assert anime.title == "One Piece"
    assert anime.score == pytest.approx(8.73)
    assert anime.aired_from == "1999-10-20" and anime.aired_to is None
    assert anime.season == "fall" and anime.year == 1999
    assert anime.broadcast_timezone == "JST"
    assert anime.broadcast_day is not None and anime.broadcast_time is not None
    assert isinstance(anime.rank, int) and anime.rank > 0
    assert anime.episode_count is None  # "Unknown" → None

    # Taxonomy + companies
    assert "Action" in anime.genres and "Adventure" in anime.genres
    assert "Shounen" in anime.demographics
    assert len(anime.studios) == 1 and "Toei" in anime.studios[0].name

    # Links
    assert len(anime.external_sources) == 11
    assert "Official Site" in {e.name for e in anime.external_sources}
    assert len(anime.streaming) == 3
    assert "Crunchyroll" in {s.name for s in anime.streaming}
    assert {e.name for e in anime.external_sources}.isdisjoint({s.name for s in anime.streaming})

    # Content
    assert anime.synopsis is not None and "Luffy" in anime.synopsis
    assert anime.background is not None and len(anime.background) > 10
    assert len(anime.picture_urls) >= 1

    # Themes — counts match benchmark
    open_titles = [t.title for t in anime.opening_themes]
    assert "We Are! (ウィーアー!)" in open_titles and "Believe" in open_titles
    assert not any(t.title in ("Apple Music", "Youtube Music") for t in anime.opening_themes)
    assert "memories" in [t.title for t in anime.ending_themes]

    # Related entries
    assert len(anime.related_entries) > 0
    assert "One Piece" in [e.title for e in anime.related_entries]
    assert any("Ganzack" in e.title for e in anime.related_entries)

    # Trailer
    assert anime.trailer is not None


def test_build_episode_count_integer_parses(mal_anime_extracted) -> None:
    assert _build({**mal_anime_extracted, "episodes": "1080"}).episode_count == 1080


def test_build_episode_count_invalid_string_returns_none(mal_anime_extracted) -> None:
    assert _build({**mal_anime_extracted, "episodes": "TBD"}).episode_count is None


def test_build_rank_from_html_string(mal_anime_extracted) -> None:
    assert _build({**mal_anime_extracted, "rank_html": "#17<sup>2</sup>"}).rank == 17
    assert _build({**mal_anime_extracted, "rank_html": None}).rank is None


def test_build_dbchanges_placeholder_filtered(mal_anime_extracted) -> None:
    raw = {
        **mal_anime_extracted,
        "licensors": [{"name": "add some", "source": "https://myanimelist.net/dbchanges.php?aid=1&t=producers"}],
        "studios": [{"name": "add some", "source": "https://myanimelist.net/dbchanges.php?aid=2&t=producers"}],
        "producers": [{"name": "Arch", "source": "https://myanimelist.net/anime/producer/1966/Arch"}],
    }
    anime = _build(raw)
    assert anime.licensors == [] and anime.studios == []
    assert len(anime.producers) == 1 and anime.producers[0].name == "Arch"


def test_build_link_field_edge_cases(mal_anime_extracted) -> None:
    # Empty name skipped
    raw = {**mal_anime_extracted, "external_sources_raw": [
        {"name": "", "source": "https://example.com"},
        {"name": "Valid", "source": "https://valid.com"},
    ]}
    assert len(_build(raw).external_sources) == 1

    # Empty source skipped
    raw = {**mal_anime_extracted, "streaming_links_raw": [
        {"name": "Crunchyroll", "source": ""},
        {"name": "Netflix", "source": "https://netflix.com"},
    ]}
    assert len(_build(raw).streaming) == 1

    # Missing keys → empty lists
    raw = {k: v for k, v in mal_anime_extracted.items() if k not in ("external_sources_raw", "streaming_links_raw")}
    anime = _build(raw)
    assert anime.external_sources == [] and anime.streaming == []


def test_build_background_edge_cases(mal_anime_extracted) -> None:
    # Extracted from minimal HTML
    bg = '<td><div><h2 id="background">Background</h2></div>The story begins in the Grand Line.</td>'
    anime = _build({"title": "Test", "background_raw": bg})
    assert anime.background is not None and "Grand Line" in anime.background

    # Placeholder ignored
    ph = '<td><div><h2 id="background">Background</h2></div>No background information has been added to this title.</td>'
    assert _build({**mal_anime_extracted, "background_raw": ph}).background is None


def test_build_cover_url_l_suffix_conversion(mal_anime_extracted) -> None:
    raw = {**mal_anime_extracted, "cover_image_src": "https://cdn.myanimelist.net/images/anime/1/123.jpg"}
    assert any("123l.jpg" in u for u in _build(raw).picture_urls)


def test_build_picture_urls_deduped_and_merged(mal_anime_extracted) -> None:
    extra = ["https://myanimelist.net/images/anime/1/123l.jpg"]
    assert any("123l.jpg" in u for u in _build(mal_anime_extracted, picture_urls=extra).picture_urls)


def test_build_misc_field_parsing(mal_anime_extracted) -> None:
    # Synonyms
    anime = _build({**mal_anime_extracted, "synonyms_raw": "OP, One Piece TV"})
    assert "OP" in anime.synonyms and "One Piece TV" in anime.synonyms

    # Duration
    anime = _build({**mal_anime_extracted, "duration_raw": "24 min. per ep."})
    assert anime.duration == 1440

    # Title fallback to og
    anime = _build({**mal_anime_extracted, "title": None, "title_og": "Fallback"})
    assert anime.title == "Fallback"


# =============================================================================
# _parse_trailer
# =============================================================================


def test_parse_trailer_extracts_youtube_id() -> None:
    trailer = _parse_trailer({"trailer_embed_url": "https://www.youtube.com/embed/abc123?autoplay=1", "trailer_title": "T"})
    assert trailer is not None
    assert "abc123" in trailer.source and "abc123" in trailer.thumbnail


def test_parse_trailer_missing_or_non_youtube_returns_none() -> None:
    assert _parse_trailer({}) is None
    assert _parse_trailer({"trailer_embed_url": "https://vimeo.com/12345"}) is None


# =============================================================================
# _normalize_mal_url
# =============================================================================


def test_normalize_mal_url_variants() -> None:
    assert _normalize_mal_url("") == ""
    full = "https://myanimelist.net/anime/21"
    assert _normalize_mal_url(full) == full
    assert _normalize_mal_url("/anime/21").startswith("https://myanimelist.net")
    assert _normalize_mal_url("anime/21").startswith("https://myanimelist.net/")


# =============================================================================
# _parse_structured_themes
# =============================================================================


def test_parse_structured_themes_valid_with_episodes() -> None:
    themes = _parse_structured_themes(
        [{"title_text": '"We Are!" by Hiroshi Kitadani', "artist": "Hiroshi Kitadani", "episodes": "1-130"}]
    )
    assert len(themes) == 1
    assert themes[0].title == "We Are!" and themes[0].artist == "Hiroshi Kitadani"
    assert len(themes[0].episodes) == 1


def test_parse_structured_themes_no_quotes_skipped() -> None:
    assert _parse_structured_themes([{"title_text": "Listen on Spotify", "artist": "", "episodes": None}]) == []


def test_parse_structured_themes_artist_normalization() -> None:
    # by-prefix stripped
    themes = _parse_structured_themes([{"title_text": '"Kokoro e"', "artist": "by Rhythm", "episodes": None}])
    assert themes[0].artist == "Rhythm"

    # empty artist → None
    themes = _parse_structured_themes([{"title_text": '"Kokoro e"', "artist": "", "episodes": None}])
    assert themes[0].artist is None


# =============================================================================
# _parse_all_related_entries
# =============================================================================


def test_parse_related_tile_entries() -> None:
    raw = {
        "related_tile_entries": [
            {"relation_raw": "Adaptation\n(Manga)", "title": "One Piece Manga", "source": "/manga/103"},
        ],
        "related_table_entries": [],
    }
    entries = _parse_all_related_entries(raw)
    assert len(entries) == 1
    assert entries[0].relation == "Adaptation"
    assert entries[0].title == "One Piece Manga"
    assert entries[0].entry_type == "Manga"


def test_parse_related_tile_entry_type_already_set_not_overridden() -> None:
    raw = {
        "related_tile_entries": [
            {"relation_raw": "Sequel\n(OVA)", "title": "Test", "entry_type": "Movie", "source": "/anime/999"},
        ],
        "related_table_entries": [],
    }
    assert _parse_all_related_entries(raw)[0].entry_type == "Movie"


def test_parse_related_table_entries() -> None:
    # Type from format text in link
    links_html = '<ul><li><a href="https://myanimelist.net/anime/22/S">Sequel</a> (TV)</li></ul>'
    raw = {"related_tile_entries": [], "related_table_entries": [{"relation": "Sequel", "links_html": links_html}]}
    assert _parse_all_related_entries(raw)[0].entry_type == "TV"

    # Type fallback from relation parts
    links_html2 = '<ul><li><a href="https://myanimelist.net/anime/22/S">Sequel</a></li></ul>'
    raw2 = {"related_tile_entries": [], "related_table_entries": [{"relation": "Sequel\n(TV)", "links_html": links_html2}]}
    assert _parse_all_related_entries(raw2)[0].entry_type == "TV"


def test_parse_related_skips_empty_title_or_source() -> None:
    raw = {
        "related_tile_entries": [
            {"relation_raw": "Adaptation", "title": "", "source": "/manga/103"},
            {"relation_raw": "Adaptation", "title": "Valid", "source": ""},
        ],
        "related_table_entries": [],
    }
    assert _parse_all_related_entries(raw) == []


def test_parse_related_entries_from_fixture(mal_anime_extracted) -> None:
    entries = _parse_all_related_entries(mal_anime_extracted)
    titles = [e.title for e in entries]
    assert "One Piece" in titles
    assert any("Ganzack" in t for t in titles)


# =============================================================================
# _fetch_mal_anime_data — async, zendriver mocked
# =============================================================================


def _make_browser_mock(mocker, html: str | None, pics_html: str | None = ""):
    page_mock = mocker.AsyncMock()
    page_mock.wait_for = AsyncMock()
    if html is None:
        page_mock.wait_for.side_effect = Exception("timeout")
    else:
        page_mock.get_content = AsyncMock(return_value=html)
        page_mock.url = "https://myanimelist.net/anime/21/One_Piece"

    pics_mock = mocker.AsyncMock()
    pics_mock.wait_for = AsyncMock()
    pics_mock.scroll_down = AsyncMock()
    pics_mock.get_content = AsyncMock(return_value=pics_html or "")

    browser_mock = mocker.AsyncMock()
    browser_mock.get = AsyncMock(side_effect=[page_mock, pics_mock])
    browser_mock.stop = AsyncMock()
    return browser_mock


@pytest.mark.asyncio
async def test_failure_cases(mocker) -> None:
    mocker.patch("http_cache.result_cache.get_cache_config", return_value=mocker.MagicMock(cache_enabled=False))

    # Navigation failure
    bm = _make_browser_mock(mocker, html=None)
    bm.stop.side_effect = Exception("stop failed")
    mocker.patch("zendriver.start", new_callable=AsyncMock, return_value=bm)
    assert await _fetch_mal_anime_data("https://myanimelist.net/anime/99999") is None

    # Empty HTML
    mocker.patch("zendriver.start", new_callable=AsyncMock, return_value=_make_browser_mock(mocker, html=""))
    assert await _fetch_mal_anime_data("https://myanimelist.net/anime/99998") is None

    # Extraction returns None
    mocker.patch("enrichment.sources.mal.mal_anime_crawler._extract_anime_from_html", return_value=None)
    mocker.patch("zendriver.start", new_callable=AsyncMock, return_value=_make_browser_mock(mocker, html="<html></html>"))
    assert await _fetch_mal_anime_data("https://myanimelist.net/anime/99997") is None


@pytest.mark.asyncio
async def test_theme_songs_wait_timeout_still_succeeds(mocker, mal_anime_html) -> None:
    mocker.patch("http_cache.result_cache.get_cache_config", return_value=mocker.MagicMock(cache_enabled=False))

    page_main = mocker.AsyncMock()
    # First wait_for (h1.title-name) succeeds; second (div.theme-songs) times out
    page_main.wait_for = AsyncMock(side_effect=[None, Exception("timeout waiting for theme-songs")])
    page_main.get_content = AsyncMock(return_value=mal_anime_html)
    page_main.url = "https://myanimelist.net/anime/21/One_Piece"

    page_pics = mocker.AsyncMock()
    page_pics.wait_for = AsyncMock()
    page_pics.get_content = AsyncMock(return_value="")

    bm = mocker.AsyncMock()
    bm.get = AsyncMock(side_effect=[page_main, page_pics])
    bm.stop = AsyncMock()
    mocker.patch("zendriver.start", new_callable=AsyncMock, return_value=bm)

    result = await _fetch_mal_anime_data("https://myanimelist.net/anime/21")
    assert result is not None and result["title"] == "One Piece"


@pytest.mark.asyncio
async def test_success(mocker, mal_anime_html, mal_anime_pics_html) -> None:
    mocker.patch("http_cache.result_cache.get_cache_config", return_value=mocker.MagicMock(cache_enabled=False))
    mocker.patch("zendriver.start", new_callable=AsyncMock,
                 return_value=_make_browser_mock(mocker, html=mal_anime_html, pics_html=mal_anime_pics_html))

    result = await _fetch_mal_anime_data("https://myanimelist.net/anime/21")
    assert result is not None
    assert result["title"] == "One Piece"
    assert result["_url"] == "https://myanimelist.net/anime/21/One_Piece"
    assert isinstance(result["_picture_urls"], list)


@pytest.mark.asyncio
async def test_pics_failure_still_returns_data(mocker, mal_anime_html) -> None:
    mocker.patch("http_cache.result_cache.get_cache_config", return_value=mocker.MagicMock(cache_enabled=False))

    page_main = mocker.AsyncMock()
    page_main.wait_for = AsyncMock()
    page_main.get_content = AsyncMock(return_value=mal_anime_html)
    page_main.url = "https://myanimelist.net/anime/21/One_Piece"

    page_pics = mocker.AsyncMock()
    page_pics.get_content = AsyncMock(side_effect=Exception("pics failed"))

    bm = mocker.AsyncMock()
    bm.get = AsyncMock(side_effect=[page_main, page_pics])
    bm.stop = AsyncMock()
    mocker.patch("zendriver.start", new_callable=AsyncMock, return_value=bm)

    result = await _fetch_mal_anime_data("https://myanimelist.net/anime/21")
    assert result is not None and result["title"] == "One Piece"
    assert result["_picture_urls"] == []


# =============================================================================
# MalAnimeCrawler class
# =============================================================================


def test_crawler_schema_and_normalize() -> None:
    from enrichment.sources.base.framework import NullRepository
    from enrichment.sources.mal.mal_anime_crawler import MalAnimeCrawler

    crawler = MalAnimeCrawler(NullRepository())
    assert crawler.get_extraction_schema() == {"xpaths": _XPATHS}
    assert crawler.normalize_identifier("/anime/21").startswith("https://myanimelist.net")
    full = "https://myanimelist.net/anime/21"
    assert crawler.normalize_identifier(full) == full


@pytest.mark.asyncio
async def test_crawler_fetch_raw_data_delegates(mocker) -> None:
    from enrichment.sources.base.framework import NullRepository
    from enrichment.sources.mal.mal_anime_crawler import MalAnimeCrawler

    mock_result = {"title": "Test"}
    mocker.patch("enrichment.sources.mal.mal_anime_crawler._fetch_mal_anime_data",
                 new_callable=AsyncMock, return_value=mock_result)

    crawler = MalAnimeCrawler(NullRepository())
    assert await crawler.fetch_raw_data("https://myanimelist.net/anime/21") == mock_result


def test_crawler_build_source_model_and_map(mal_anime_extracted) -> None:
    from enrichment.sources.base.framework import NullRepository
    from enrichment.sources.mal.mal_anime_crawler import MalAnimeCrawler

    crawler = MalAnimeCrawler(NullRepository())
    raw = dict(mal_anime_extracted)
    anime = crawler.build_source_model(raw, "https://myanimelist.net/anime/21")
    assert anime.title == "One Piece"
    assert "_picture_urls" not in raw and "_url" not in raw  # popped in-place

    canonical = crawler.map_to_canonical(anime)
    assert isinstance(canonical, dict) and canonical.get("title") == "One Piece"


# =============================================================================
# fetch_mal_anime + main()
# =============================================================================


@pytest.mark.asyncio
async def test_none_and_success(mocker, mal_anime_extracted) -> None:
    mocker.patch("enrichment.sources.mal.mal_anime_crawler._fetch_mal_anime_data",
                 new_callable=AsyncMock, return_value=None)
    assert await fetch_mal_anime("https://myanimelist.net/anime/21") is None

    mocker.patch("enrichment.sources.mal.mal_anime_crawler._fetch_mal_anime_data",
                 new_callable=AsyncMock, return_value=mal_anime_extracted)
    result = await fetch_mal_anime("https://myanimelist.net/anime/21")
    assert result is not None and result["title"] == "One Piece"


@pytest.mark.asyncio
async def test_main_exit_codes(mocker, tmp_path) -> None:
    from enrichment.sources.mal.mal_anime_crawler import main

    out = str(tmp_path / "out.json")
    mocker.patch("sys.argv", ["prog", "https://myanimelist.net/anime/21", "--output", out])
    mocker.patch("enrichment.sources.mal.mal_anime_crawler.fetch_mal_anime", return_value=None)
    assert await main() == 1

    mocker.patch("sys.argv", ["prog", "https://myanimelist.net/anime/21", "--output", out])
    mocker.patch("enrichment.sources.mal.mal_anime_crawler.fetch_mal_anime",
                 return_value={"title": "One Piece", "episode_count": 1000})
    assert await main() == 0
