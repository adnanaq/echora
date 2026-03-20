"""Unit tests for mal_anime_crawler.py — schema structure and post-processing helpers.

These tests validate the text-anchor + regex extraction pattern using in-memory raw
dicts that mirror the shape of real extraction output. No network calls are made.
"""

import json

import pytest

from enrichment.crawlers.mal_crawler.mal_anime_crawler import (
    _build_anime_from_raw,
    _fetch_mal_anime_data,
    _get_anime_schema,
    _normalize_mal_url,
    _parse_all_related_entries,
    _parse_picture_urls,
    _parse_structured_themes,
    _parse_trailer,
    fetch_mal_anime,
)


# =============================================================================
# Schema structure — verifies the benchmark extraction pattern is in place
# =============================================================================


def test_schema_has_fields_list() -> None:
    """Anime schema must have a 'fields' list (XPath strategy requirement)."""
    schema = _get_anime_schema()
    assert "fields" in schema
    assert isinstance(schema["fields"], list)


def test_schema_external_sources_uses_text_anchor() -> None:
    """external_sources_raw must anchor on h2 label text, not a fragile CSS class."""
    schema = _get_anime_schema()
    ext_field = next(f for f in schema["fields"] if f["name"] == "external_sources_raw")
    selector = ext_field["selector"]
    assert "Available At" in selector or "Resources" in selector
    # Old broken class name must not be present
    assert "streaming-platform_links" not in selector


def test_schema_streaming_links_uses_text_anchor() -> None:
    """streaming_links_raw must anchor on 'Streaming Platforms' h2 text."""
    schema = _get_anime_schema()
    stream_field = next(f for f in schema["fields"] if f["name"] == "streaming_links_raw")
    selector = stream_field["selector"]
    assert "Streaming Platforms" in selector
    assert "broadcasts" in selector


def test_schema_external_sources_name_uses_caption_div() -> None:
    """external_sources_raw name sub-field must target the .caption div."""
    schema = _get_anime_schema()
    ext_field = next(f for f in schema["fields"] if f["name"] == "external_sources_raw")
    name_field = next(f for f in ext_field["fields"] if f["name"] == "name")
    assert "caption" in name_field["selector"]


def test_schema_streaming_links_name_uses_title_attribute() -> None:
    """streaming_links_raw name sub-field must read the anchor's title attribute."""
    schema = _get_anime_schema()
    stream_field = next(f for f in schema["fields"] if f["name"] == "streaming_links_raw")
    name_field = next(f for f in stream_field["fields"] if f["name"] == "name")
    assert name_field.get("attribute") == "title"


# =============================================================================
# _build_anime_from_raw — external and streaming link post-processing
# =============================================================================


def _build(extra: dict | None = None):
    """Helper: call _build_anime_from_raw with a minimal raw dict."""
    raw: dict = {"title": "One Piece"}
    if extra:
        raw.update(extra)
    return _build_anime_from_raw(
        raw,
        url="https://myanimelist.net/anime/21/One_Piece",
        picture_urls=[],
    )


def test_build_external_sources_parsed() -> None:
    """external_sources_raw list is converted to MalExternalLink objects."""
    anime = _build({
        "external_sources_raw": [
            {"name": "Official Site", "source": "https://example.com"},
            {"name": "Twitter", "source": "https://twitter.com/onepieceanime"},
        ]
    })
    assert len(anime.external_sources) == 2
    assert anime.external_sources[0].name == "Official Site"
    assert anime.external_sources[1].name == "Twitter"


def test_build_streaming_links_parsed() -> None:
    """streaming_links_raw list is converted to MalExternalLink objects."""
    anime = _build({
        "streaming_links_raw": [
            {"name": "Crunchyroll", "source": "https://crunchyroll.com/one-piece"},
            {"name": "Netflix", "source": "https://netflix.com/title/80107103"},
        ]
    })
    assert len(anime.streaming) == 2
    assert anime.streaming[0].name == "Crunchyroll"
    assert anime.streaming[1].name == "Netflix"


def test_build_empty_link_name_skipped() -> None:
    """Link entries with empty name are skipped."""
    anime = _build({
        "external_sources_raw": [
            {"name": "", "source": "https://example.com"},
            {"name": "Valid Site", "source": "https://valid.com"},
        ]
    })
    assert len(anime.external_sources) == 1
    assert anime.external_sources[0].name == "Valid Site"


def test_build_empty_link_source_skipped() -> None:
    """Link entries with empty source are skipped."""
    anime = _build({
        "streaming_links_raw": [
            {"name": "Crunchyroll", "source": ""},
            {"name": "Netflix", "source": "https://netflix.com"},
        ]
    })
    assert len(anime.streaming) == 1
    assert anime.streaming[0].name == "Netflix"


def test_build_missing_link_fields_returns_empty_lists() -> None:
    """None or missing external/streaming link lists result in empty lists."""
    anime = _build()
    assert anime.external_sources == []
    assert anime.streaming == []


def test_build_dbchanges_placeholder_filtered_from_companies() -> None:
    """Licensors/studios/producers with dbchanges.php source ('add some') are dropped."""
    anime = _build({
        "licensors": [{"name": "add some", "source": "https://myanimelist.net/dbchanges.php?aid=62312&t=producers"}],
        "studios": [{"name": "add some", "source": "https://myanimelist.net/dbchanges.php?aid=63764&t=producers"}],
        "producers": [{"name": "Arch", "source": "https://myanimelist.net/anime/producer/1966/Arch"}],
    })
    assert anime.licensors == []
    assert anime.studios == []
    assert len(anime.producers) == 1
    assert anime.producers[0].name == "Arch"


def test_build_both_link_types_populated_independently() -> None:
    """External and streaming links are populated independently."""
    anime = _build({
        "external_sources_raw": [
            {"name": "Official Site", "source": "https://example.com"},
        ],
        "streaming_links_raw": [
            {"name": "Crunchyroll", "source": "https://crunchyroll.com"},
        ],
    })
    assert len(anime.external_sources) == 1
    assert len(anime.streaming) == 1
    assert anime.external_sources[0].name != anime.streaming[0].name


# =============================================================================
# _parse_trailer (lines 442-443)
# =============================================================================


def test_parse_trailer_returns_mal_trailer() -> None:
    raw = {"trailer_embed_url": "https://www.youtube.com/embed/abc123?autoplay=1", "trailer_title": "Trailer"}
    trailer = _parse_trailer(raw)
    assert trailer is not None
    assert "abc123" in trailer.source
    assert "abc123" in trailer.thumbnail


def test_parse_trailer_no_embed_url_returns_none() -> None:
    assert _parse_trailer({}) is None


def test_parse_trailer_non_matching_url_returns_none() -> None:
    assert _parse_trailer({"trailer_embed_url": "https://vimeo.com/12345"}) is None


# =============================================================================
# _parse_picture_urls (lines 452-457)
# =============================================================================


def test_parse_picture_urls_returns_matching_urls() -> None:
    pics = [
        {"url": "https://cdn.myanimelist.net/images/anime/1/123.jpg"},
        {"url": "https://cdn.myanimelist.net/images/anime/2/456.jpg"},
        {"url": "https://otherdomain.com/image.jpg"},
        {"url": ""},
    ]
    result = _parse_picture_urls(pics)
    assert len(result) == 2
    assert all("myanimelist" in u for u in result)


# =============================================================================
# _normalize_mal_url (lines 462-466)
# =============================================================================


def test_normalize_mal_url_empty_returns_empty() -> None:
    assert _normalize_mal_url("") == ""


def test_normalize_mal_url_full_url_passthrough() -> None:
    url = "https://myanimelist.net/anime/21"
    assert _normalize_mal_url(url) == url


def test_normalize_mal_url_slash_path_prepends_base() -> None:
    result = _normalize_mal_url("/anime/21")
    assert result.startswith("https://myanimelist.net")
    assert "/anime/21" in result


def test_normalize_mal_url_no_slash_prepends_base_with_slash() -> None:
    result = _normalize_mal_url("anime/21")
    assert result.startswith("https://myanimelist.net/")


# =============================================================================
# _parse_structured_themes (lines 473-497)
# =============================================================================


def test_parse_structured_themes_valid_entry() -> None:
    themes = _parse_structured_themes([
        {"title_text": '"We Are!" by Hiroshi Kitadani', "artist": "Hiroshi Kitadani", "episodes": "1-130"},
    ])
    assert len(themes) == 1
    assert themes[0].title == "We Are!"
    assert themes[0].artist == "Hiroshi Kitadani"


def test_parse_structured_themes_no_quotes_skipped() -> None:
    """Entries without quoted title (e.g. Spotify junk) are skipped."""
    themes = _parse_structured_themes([
        {"title_text": "Listen on Spotify", "artist": "", "episodes": None},
    ])
    assert themes == []


def test_parse_structured_themes_artist_by_prefix_stripped() -> None:
    themes = _parse_structured_themes([
        {"title_text": '"Kokoro e"', "artist": "by Rhythm", "episodes": None},
    ])
    assert themes[0].artist == "Rhythm"


def test_parse_structured_themes_empty_artist_becomes_none() -> None:
    themes = _parse_structured_themes([
        {"title_text": '"Kokoro e"', "artist": "", "episodes": None},
    ])
    assert themes[0].artist is None


# =============================================================================
# _parse_all_related_entries (lines 510-579)
# =============================================================================


def test_parse_all_related_entries_tile_entries() -> None:
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


def test_parse_all_related_entries_tile_type_from_rel_parts() -> None:
    """entry_type extracted from second part of relation when not explicitly set."""
    raw = {
        "related_tile_entries": [
            {"relation_raw": "Adaptation\n(Manga)", "title": "One Piece Manga", "entry_type": None, "source": "/manga/103"},
        ],
        "related_table_entries": [],
    }
    entries = _parse_all_related_entries(raw)
    assert entries[0].entry_type == "Manga"


def test_parse_all_related_entries_table_entries() -> None:
    """Type absent from format_text falls back to second part of relation cell."""
    links_html = '<ul><li><a href="https://myanimelist.net/anime/22/One_Piece_Sequel">Sequel Title</a></li></ul>'
    raw = {
        "related_tile_entries": [],
        "related_table_entries": [
            {"relation": "Sequel\n(TV)", "links_html": links_html},
        ],
    }
    entries = _parse_all_related_entries(raw)
    assert len(entries) == 1
    assert entries[0].title == "Sequel Title"
    assert entries[0].entry_type == "TV"


def test_parse_all_related_entries_table_type_from_format_text() -> None:
    """entry_type extracted from format_text (TV) when present."""
    links_html = '<ul><li><a href="https://myanimelist.net/anime/22/One_Piece_Sequel">Sequel Title</a> (TV)</li></ul>'
    raw = {
        "related_tile_entries": [],
        "related_table_entries": [
            {"relation": "Sequel", "links_html": links_html},
        ],
    }
    entries = _parse_all_related_entries(raw)
    assert entries[0].entry_type == "TV"


def test_parse_all_related_entries_skips_empty_title_or_source() -> None:
    raw = {
        "related_tile_entries": [
            {"relation_raw": "Adaptation", "title": "", "source": "/manga/103"},
            {"relation_raw": "Adaptation", "title": "Valid", "source": ""},
        ],
        "related_table_entries": [],
    }
    entries = _parse_all_related_entries(raw)
    assert entries == []


# =============================================================================
# _build_anime_from_raw — additional branches
# =============================================================================


def test_build_episode_count_invalid_string_returns_none() -> None:
    """Non-numeric episode string triggers ValueError → episode_count is None."""
    anime = _build({"episodes": "TBD"})
    assert anime.episode_count is None


def test_build_rank_extracted_from_rank_html() -> None:
    anime = _build({"rank_html": "#17<sup>2</sup>"})
    assert anime.rank == 17


def test_build_cover_url_without_l_suffix_gets_converted() -> None:
    """Cover URL ending in .jpg (not l.jpg) is converted to l.jpg large variant."""
    anime = _build({"cover_image_src": "https://cdn.myanimelist.net/images/anime/1/123.jpg"})
    assert any("123l.jpg" in u for u in anime.picture_urls)


def test_build_background_extracted() -> None:
    bg_html = '<div id="background"><h2>Background</h2>The story begins in the Grand Line.</div>'
    anime = _build({"background_raw": bg_html})
    assert anime.background is not None
    assert "Grand Line" in anime.background


def test_build_background_placeholder_ignored() -> None:
    bg_html = '<div id="background"><h2>Background</h2>No background information has been added to this title.</div>'
    anime = _build({"background_raw": bg_html})
    assert anime.background is None


# =============================================================================
# _fetch_mal_anime_data — async branches (lines 807-856)
# =============================================================================


@pytest.mark.asyncio
async def test_fetch_mal_anime_data_no_result(mocker) -> None:
    mocker.patch("http_cache.result_cache.get_cache_config", return_value=mocker.MagicMock(cache_enabled=False))
    mocker.patch(
        "enrichment.crawlers.mal_crawler.mal_anime_crawler.crawl_batch_urls",
        return_value=[],
    )
    mocker.patch.object(
        _fetch_mal_anime_data.__wrapped__.__globals__["_limiter"],
        "acquire",
        return_value=None,
    )
    result = await _fetch_mal_anime_data("https://myanimelist.net/anime/99999")
    assert result is None


@pytest.mark.asyncio
async def test_fetch_mal_anime_data_404(mocker) -> None:
    mocker.patch("http_cache.result_cache.get_cache_config", return_value=mocker.MagicMock(cache_enabled=False))
    mocker.patch(
        "enrichment.crawlers.mal_crawler.mal_anime_crawler.crawl_batch_urls",
        return_value=[{"status_code": 404, "extracted_content": None}],
    )
    mocker.patch.object(
        _fetch_mal_anime_data.__wrapped__.__globals__["_limiter"],
        "acquire",
        return_value=None,
    )
    result = await _fetch_mal_anime_data("https://myanimelist.net/anime/99998")
    assert result is None


@pytest.mark.asyncio
async def test_fetch_mal_anime_data_http_error(mocker) -> None:
    mocker.patch("http_cache.result_cache.get_cache_config", return_value=mocker.MagicMock(cache_enabled=False))
    mocker.patch(
        "enrichment.crawlers.mal_crawler.mal_anime_crawler.crawl_batch_urls",
        return_value=[{"status_code": 500, "extracted_content": None}],
    )
    mocker.patch.object(
        _fetch_mal_anime_data.__wrapped__.__globals__["_limiter"],
        "acquire",
        return_value=None,
    )
    result = await _fetch_mal_anime_data("https://myanimelist.net/anime/99997")
    assert result is None


@pytest.mark.asyncio
async def test_fetch_mal_anime_data_empty_content(mocker) -> None:
    mocker.patch("http_cache.result_cache.get_cache_config", return_value=mocker.MagicMock(cache_enabled=False))
    mocker.patch(
        "enrichment.crawlers.mal_crawler.mal_anime_crawler.crawl_batch_urls",
        return_value=[{"status_code": 200, "extracted_content": "[]"}],
    )
    mocker.patch.object(
        _fetch_mal_anime_data.__wrapped__.__globals__["_limiter"],
        "acquire",
        return_value=None,
    )
    result = await _fetch_mal_anime_data("https://myanimelist.net/anime/99996")
    assert result is None


@pytest.mark.asyncio
async def test_fetch_mal_anime_data_success(mocker) -> None:
    mocker.patch("http_cache.result_cache.get_cache_config", return_value=mocker.MagicMock(cache_enabled=False))
    raw = {"title": "One Piece"}
    pics_raw = [{"picture_urls_raw": [{"url": "https://cdn.myanimelist.net/images/anime/1/123.jpg"}]}]
    mocker.patch(
        "enrichment.crawlers.mal_crawler.mal_anime_crawler.crawl_batch_urls",
        side_effect=[
            [{"status_code": 200, "metadata": {"og:url": "https://myanimelist.net/anime/21/One_Piece"}, "extracted_content": json.dumps([raw])}],
            [{"status_code": 200, "extracted_content": json.dumps(pics_raw)}],
        ],
    )
    mocker.patch.object(
        _fetch_mal_anime_data.__wrapped__.__globals__["_limiter"],
        "acquire",
        return_value=None,
    )
    result = await _fetch_mal_anime_data("https://myanimelist.net/anime/21")
    assert result is not None
    assert result["title"] == "One Piece"
    assert "123.jpg" in result["_picture_urls"][0]
    assert result["_url"] == "https://myanimelist.net/anime/21/One_Piece"


@pytest.mark.asyncio
async def test_fetch_mal_anime_data_slug_falls_back_to_crawler_url(mocker) -> None:
    """When og:url is absent from metadata, _url falls back to result['url']."""
    mocker.patch("http_cache.result_cache.get_cache_config", return_value=mocker.MagicMock(cache_enabled=False))
    raw = {"title": "One Piece"}
    mocker.patch(
        "enrichment.crawlers.mal_crawler.mal_anime_crawler.crawl_batch_urls",
        side_effect=[
            [{"status_code": 200, "url": "https://myanimelist.net/anime/21/One_Piece", "metadata": {}, "extracted_content": json.dumps([raw])}],
            [{"status_code": 200, "extracted_content": "[]"}],
        ],
    )
    mocker.patch.object(
        _fetch_mal_anime_data.__wrapped__.__globals__["_limiter"],
        "acquire",
        return_value=None,
    )
    result = await _fetch_mal_anime_data("https://myanimelist.net/anime/21")
    assert result is not None
    assert result["_url"] == "https://myanimelist.net/anime/21/One_Piece"


# =============================================================================
# fetch_mal_anime — async (lines 872-881)
# =============================================================================


@pytest.mark.asyncio
async def test_fetch_mal_anime_returns_none_when_no_data(mocker) -> None:
    mocker.patch(
        "enrichment.crawlers.mal_crawler.mal_anime_crawler._fetch_mal_anime_data",
        return_value=None,
    )
    result = await fetch_mal_anime("https://myanimelist.net/anime/21")
    assert result is None


@pytest.mark.asyncio
async def test_fetch_mal_anime_returns_parsed_anime(mocker) -> None:
    raw = {
        "title": "One Piece",
        "_picture_urls": [],
        "_url": "https://myanimelist.net/anime/21/One_Piece",
    }
    mocker.patch(
        "enrichment.crawlers.mal_crawler.mal_anime_crawler._fetch_mal_anime_data",
        return_value=raw,
    )
    result = await fetch_mal_anime("https://myanimelist.net/anime/21")
    assert result is not None
    assert result.title == "One Piece"


# =============================================================================
# main() CLI (lines 886-908)
# =============================================================================


@pytest.mark.asyncio
async def test_main_returns_1_when_no_anime(mocker, tmp_path) -> None:
    mocker.patch("sys.argv", ["prog", "https://myanimelist.net/anime/21", "--output", str(tmp_path / "out.json")])
    mocker.patch(
        "enrichment.crawlers.mal_crawler.mal_anime_crawler.fetch_mal_anime",
        return_value=None,
    )
    from enrichment.crawlers.mal_crawler.mal_anime_crawler import main
    assert await main() == 1


@pytest.mark.asyncio
async def test_main_returns_0_on_success(mocker, tmp_path) -> None:
    from enrichment.crawlers.mal_crawler.mal_models import MalScrapedAnime
    anime = MalScrapedAnime(source="https://myanimelist.net/anime/21", title="One Piece")
    mocker.patch("sys.argv", ["prog", "https://myanimelist.net/anime/21", "--output", str(tmp_path / "out.json")])
    mocker.patch(
        "enrichment.crawlers.mal_crawler.mal_anime_crawler.fetch_mal_anime",
        return_value=anime,
    )
    mocker.patch("enrichment.mappers.mal_mapper.anime_from_mal", return_value={"title": "One Piece"})
    from enrichment.crawlers.mal_crawler.mal_anime_crawler import main
    assert await main() == 0
    assert (tmp_path / "out.json").exists()
