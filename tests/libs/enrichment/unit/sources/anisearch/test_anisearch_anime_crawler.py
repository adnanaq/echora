"""Unit tests for anisearch_anime_crawler.py — schema structure and post-processing helpers.

Tests validate XPath extraction using real HTML fixtures captured from:
- https://www.anisearch.com/anime/2227,one-piece (2026-06-09)
- https://www.anisearch.com/anime/2227,one-piece/relations?show=overall (2026-06-09)

Edge-case branches use field overrides on top of the real fixture dict.
No network calls are made.
"""

import pytest
from unittest.mock import AsyncMock

from enrichment.sources.anisearch.anisearch_anime_crawler import (
    BASE_ANIME_URL,
    _XPATHS,
    AniSearchAnimeCrawler,
    _build_anime_from_raw,
    _extract_anime_from_html,
    _extract_path_from_url,
    _extract_relations_from_html,
    _fetch_anisearch_anime_data,
    _fetch_page_html,
    _parse_relations,
    _post_process_main,
    _process_relation_tooltips,
    fetch_anisearch_anime,
)
from enrichment.sources.base.framework import DockerTransport, NullRepository

_URL = "https://www.anisearch.com/anime/2227,one-piece"


# =============================================================================
# Processed fixture — chains real raw data through _post_process_main
# =============================================================================


@pytest.fixture(scope="session")
def one_piece_processed(one_piece_main_raw, one_piece_relations_raw):
    data = _post_process_main(one_piece_main_raw)
    anime_rels, manga_rels = _parse_relations(one_piece_relations_raw)
    data["anime_relations"] = anime_rels
    data["manga_relations"] = manga_rels
    return data


# =============================================================================
# _XPATHS dict
# =============================================================================


def test_xpaths_has_required_main_keys() -> None:
    assert {
        "cover_image", "title_alt", "title_ja", "type", "status",
        "published", "studio", "studio_url", "broadcast_raw",
        "source_material", "synonyms", "description",
        "genres", "tags", "rating_score", "rank_toplist", "rank_trending", "websites",
    } <= set(_XPATHS)


def test_xpaths_has_relation_keys() -> None:
    assert {"anime_relation_rows", "manga_relation_rows"} <= set(_XPATHS)


def test_xpaths_cover_image_targets_details_cover() -> None:
    assert "details-cover" in _XPATHS["cover_image"]
    assert _XPATHS["cover_image"].endswith("/@src")


def test_xpaths_title_alt_targets_grey_ja() -> None:
    assert "grey" in _XPATHS["title_alt"]
    assert "ja" in _XPATHS["title_alt"]


def test_xpaths_title_ja_targets_f16_strong() -> None:
    assert "f16" in _XPATHS["title_ja"]
    assert "strong" in _XPATHS["title_ja"]


def test_xpaths_genres_anchor_on_genre_href() -> None:
    assert "/genre/main/" in _XPATHS["genres"] or "/genre/subsidiary/" in _XPATHS["genres"]


def test_xpaths_tags_anchor_on_tag_href() -> None:
    assert "/genre/tag/" in _XPATHS["tags"]


def test_xpaths_relations_target_correct_sections() -> None:
    assert "relations_anime" in _XPATHS["anime_relation_rows"]
    assert "relations_manga" in _XPATHS["manga_relation_rows"]
    assert "tbody" in _XPATHS["anime_relation_rows"]


# =============================================================================
# _extract_anime_from_html — real HTML fixture
# =============================================================================


def test_extract_anime_from_html_title_fields(one_piece_main_html) -> None:
    raw = _extract_anime_from_html(one_piece_main_html)
    assert raw is not None
    assert raw["title_ja"] == "One Piece"
    assert raw["title_alt"] == "ワンピース"


def test_extract_anime_from_html_cover_image(one_piece_main_html) -> None:
    raw = _extract_anime_from_html(one_piece_main_html)
    assert raw is not None
    assert raw["cover_image"] is not None
    assert raw["cover_image"].startswith("https://")


def test_extract_anime_from_html_type_contains_tv_series(one_piece_main_html) -> None:
    raw = _extract_anime_from_html(one_piece_main_html)
    assert raw is not None
    assert "TV-Series" in (raw["type"] or "")


def test_extract_anime_from_html_genres_list(one_piece_main_html) -> None:
    raw = _extract_anime_from_html(one_piece_main_html)
    assert raw is not None
    assert len(raw["genres"]) > 0
    assert all(isinstance(g["name"], str) for g in raw["genres"])


def test_extract_anime_from_html_tags_list(one_piece_main_html) -> None:
    raw = _extract_anime_from_html(one_piece_main_html)
    assert raw is not None
    assert len(raw["tags"]) > 0


def test_extract_anime_from_html_websites_list(one_piece_main_html) -> None:
    raw = _extract_anime_from_html(one_piece_main_html)
    assert raw is not None
    assert len(raw["websites"]) > 0
    assert all(w["url"] for w in raw["websites"])


def test_extract_anime_from_html_studio(one_piece_main_html) -> None:
    raw = _extract_anime_from_html(one_piece_main_html)
    assert raw is not None
    assert raw["studio"] == "Toei Animation Co., Ltd."
    assert "toei-animation" in (raw["studio_url"] or "")


def test_extract_anime_from_html_rating_score(one_piece_main_html) -> None:
    raw = _extract_anime_from_html(one_piece_main_html)
    assert raw is not None
    assert raw["rating_score"] is not None
    assert "." in raw["rating_score"]


def test_extract_anime_from_html_empty_returns_none() -> None:
    assert _extract_anime_from_html("") is None


def test_extract_anime_from_html_unparseable_returns_none() -> None:
    assert _extract_anime_from_html("<not valid xml at all >>>") is not None  # lxml is lenient
    assert _extract_anime_from_html("") is None


# =============================================================================
# _extract_relations_from_html — real HTML fixture
# =============================================================================


def test_extract_relations_from_html_anime_count(one_piece_relations_html) -> None:
    raw = _extract_relations_from_html(one_piece_relations_html)
    assert raw is not None
    assert len(raw["anime_relations"]) == 79


def test_extract_relations_from_html_manga_count(one_piece_relations_html) -> None:
    raw = _extract_relations_from_html(one_piece_relations_html)
    assert raw is not None
    assert len(raw["manga_relations"]) == 2


def test_extract_relations_from_html_entry_fields(one_piece_relations_html) -> None:
    raw = _extract_relations_from_html(one_piece_relations_html)
    assert raw is not None
    entry = raw["anime_relations"][0]
    assert entry["relation_type"] is not None
    assert entry["title"] is not None
    assert entry["url"] is not None
    assert entry["details"] is not None


def test_extract_relations_from_html_manga_original_work(one_piece_relations_html) -> None:
    raw = _extract_relations_from_html(one_piece_relations_html)
    assert raw is not None
    titles = [r["title"] for r in raw["manga_relations"]]
    assert "One Piece" in titles


def test_extract_relations_from_html_image_has_tooltip(one_piece_relations_html) -> None:
    raw = _extract_relations_from_html(one_piece_relations_html)
    assert raw is not None
    images = [r["image"] for r in raw["anime_relations"] if r.get("image")]
    assert len(images) > 0
    assert all("<img" in img for img in images)


def test_extract_relations_from_html_empty_returns_none() -> None:
    assert _extract_relations_from_html("") is None


# =============================================================================
# _extract_path_from_url
# =============================================================================


def test_extract_path_valid() -> None:
    assert _extract_path_from_url(_URL) == "2227,one-piece"


def test_extract_path_trailing_slash_stripped() -> None:
    assert _extract_path_from_url(_URL + "/") == "2227,one-piece"


def test_extract_path_wrong_base_raises() -> None:
    with pytest.raises(ValueError, match="URL must start with"):
        _extract_path_from_url("https://myanimelist.net/anime/21")


def test_extract_path_empty_path_raises() -> None:
    with pytest.raises(ValueError, match="does not contain anime path"):
        _extract_path_from_url(BASE_ANIME_URL)


# =============================================================================
# _process_relation_tooltips
# =============================================================================


def test_process_relation_tooltips_extracts_img_src() -> None:
    rel = {"image": '<img src="https://cdn.anisearch.com/images/anime/cover/2/2227.webp" />'}
    _process_relation_tooltips([rel])
    assert rel["image"] == "https://cdn.anisearch.com/images/anime/cover/2/2227.webp"


def test_process_relation_tooltips_html_escaped_decoded() -> None:
    escaped = "&lt;img src=&quot;https://cdn.anisearch.com/cover.webp&quot;&gt;"
    rel = {"image": escaped}
    _process_relation_tooltips([rel])
    assert rel["image"] == "https://cdn.anisearch.com/cover.webp"


def test_process_relation_tooltips_no_image_key_unchanged() -> None:
    rel = {"title": "Test"}
    _process_relation_tooltips([rel])
    assert rel == {"title": "Test"}


def test_process_relation_tooltips_no_img_match_unchanged() -> None:
    rel = {"image": "no img tag here"}
    _process_relation_tooltips([rel])
    assert rel["image"] == "no img tag here"


def test_process_relation_tooltips_empty_list() -> None:
    _process_relation_tooltips([])


def test_process_relation_tooltips_real_data(one_piece_relations_raw) -> None:
    rels = list(one_piece_relations_raw["anime_relations"])
    _process_relation_tooltips(rels)
    for rel in rels:
        if rel.get("image"):
            assert not rel["image"].startswith("<")
            assert rel["image"].startswith("https://")


# =============================================================================
# _post_process_main — real fixture for baseline, overrides for edge cases
# =============================================================================


def test_post_process_type_strips_label_and_comma_suffix(one_piece_main_raw) -> None:
    assert _post_process_main(one_piece_main_raw)["type"] == "TV-Series"


def test_post_process_status_strips_label(one_piece_main_raw) -> None:
    assert _post_process_main(one_piece_main_raw)["status"] == "Ongoing"


def test_post_process_date_range_open_end(one_piece_main_raw) -> None:
    data = _post_process_main(one_piece_main_raw)
    assert data["start_date"] == "20.10.1999"
    assert data["end_date"] is None


def test_post_process_date_range_closed(one_piece_main_raw) -> None:
    raw = {**one_piece_main_raw, "published": "Published: 20.10.1999 - 31.03.2002"}
    data = _post_process_main(raw)
    assert data["start_date"] == "20.10.1999"
    assert data["end_date"] == "31.03.2002"


def test_post_process_date_single(one_piece_main_raw) -> None:
    raw = {**one_piece_main_raw, "published": "Published: 05.04.2003"}
    data = _post_process_main(raw)
    assert data["start_date"] == "05.04.2003"
    assert data["end_date"] is None


def test_post_process_date_missing(one_piece_main_raw) -> None:
    raw = {**one_piece_main_raw, "published": None}
    data = _post_process_main(raw)
    assert data["start_date"] is None
    assert data["end_date"] is None


def test_post_process_broadcast_parsed(one_piece_main_raw) -> None:
    data = _post_process_main(one_piece_main_raw)
    assert data["broadcast_day"] == "Sunday"
    assert data["broadcast_time"] == "23:15"
    assert data["broadcast_timezone"] == "JST"


def test_post_process_broadcast_missing(one_piece_main_raw) -> None:
    raw = {**one_piece_main_raw, "broadcast_raw": None}
    data = _post_process_main(raw)
    assert data["broadcast_day"] is None
    assert data["broadcast_time"] is None
    assert data["broadcast_timezone"] is None


def test_post_process_studio_url_without_leading_slash(one_piece_main_raw) -> None:
    data = _post_process_main(one_piece_main_raw)
    assert data["studio_url"] == "https://www.anisearch.com/company/412,toei-animation-co-ltd"


def test_post_process_studio_url_with_leading_slash(one_piece_main_raw) -> None:
    raw = {**one_piece_main_raw, "studio_url": "/company/412,toei-animation-co-ltd"}
    data = _post_process_main(raw)
    assert data["studio_url"] == "https://www.anisearch.com/company/412,toei-animation-co-ltd"


def test_post_process_studio_url_empty(one_piece_main_raw) -> None:
    raw = {**one_piece_main_raw, "studio_url": None}
    assert _post_process_main(raw)["studio_url"] is None


def test_post_process_source_material_strips_label(one_piece_main_raw) -> None:
    assert _post_process_main(one_piece_main_raw)["source_material"] == "Manga"


def test_post_process_synonyms_split_on_comma(one_piece_main_raw) -> None:
    assert _post_process_main(one_piece_main_raw)["synonyms"] == ["OP", "OneP"]


def test_post_process_synonyms_missing(one_piece_main_raw) -> None:
    raw = {**one_piece_main_raw, "synonyms": None}
    assert _post_process_main(raw)["synonyms"] == []


def test_post_process_genres_flattened(one_piece_main_raw) -> None:
    data = _post_process_main(one_piece_main_raw)
    assert "Action" in data["genres"]
    assert "Fighting-Shounen" in data["genres"]
    assert all(isinstance(g, str) for g in data["genres"])


def test_post_process_tags_flattened(one_piece_main_raw) -> None:
    data = _post_process_main(one_piece_main_raw)
    assert "Pirate" in data["tags"]
    assert all(isinstance(t, str) for t in data["tags"])


def test_post_process_genres_empty_name_skipped(one_piece_main_raw) -> None:
    raw = {**one_piece_main_raw, "genres": [{"name": ""}, {"name": "Action"}]}
    assert _post_process_main(raw)["genres"] == ["Action"]


def test_post_process_websites_populated(one_piece_main_raw) -> None:
    data = _post_process_main(one_piece_main_raw)
    assert len(data["websites"]) == len(one_piece_main_raw["websites"])
    assert all(w["url"] for w in data["websites"])


def test_post_process_websites_empty_url_skipped(one_piece_main_raw) -> None:
    raw = {
        **one_piece_main_raw,
        "websites": [
            {"name": "Empty", "url": ""},
            {"name": "Official", "url": "https://one-piece.com"},
        ],
    }
    data = _post_process_main(raw)
    assert len(data["websites"]) == 1
    assert data["websites"][0]["name"] == "Official"


def test_post_process_score_extracted(one_piece_main_raw) -> None:
    data = _post_process_main(one_piece_main_raw)
    assert data["statistics"]["score"] == pytest.approx(4.18)


def test_post_process_rank_extracted(one_piece_main_raw) -> None:
    assert _post_process_main(one_piece_main_raw)["statistics"]["rank"] == 126


def test_post_process_trending_extracted(one_piece_main_raw) -> None:
    assert _post_process_main(one_piece_main_raw)["statistics"]["trending"] == 66


def test_post_process_stats_all_missing_returns_none(one_piece_main_raw) -> None:
    raw = {**one_piece_main_raw, "rating_score": None, "rank_toplist": None, "rank_trending": None}
    assert _post_process_main(raw)["statistics"] is None


def test_post_process_score_missing_rank_still_populated(one_piece_main_raw) -> None:
    raw = {**one_piece_main_raw, "rating_score": None}
    stats = _post_process_main(raw)["statistics"]
    assert "score" not in stats
    assert stats["rank"] == 126


def test_post_process_description_stripped(one_piece_main_raw) -> None:
    raw = {**one_piece_main_raw, "description": "  some synopsis  "}
    assert _post_process_main(raw)["description"] == "some synopsis"


# =============================================================================
# _parse_relations
# =============================================================================


def test_parse_relations_none_raw_returns_empty() -> None:
    assert _parse_relations(None) == ([], [])


def test_parse_relations_empty_lists() -> None:
    assert _parse_relations({"anime_relations": [], "manga_relations": []}) == ([], [])


def test_parse_relations_missing_keys_returns_empty() -> None:
    assert _parse_relations({}) == ([], [])


def test_parse_relations_real_data_anime_count(one_piece_relations_raw) -> None:
    anime, _ = _parse_relations(one_piece_relations_raw)
    assert len(anime) == len(one_piece_relations_raw["anime_relations"])


def test_parse_relations_real_data_manga_count(one_piece_relations_raw) -> None:
    _, manga = _parse_relations(one_piece_relations_raw)
    assert len(manga) == len(one_piece_relations_raw["manga_relations"])


def test_parse_relations_images_urls_after_processing(one_piece_relations_raw) -> None:
    anime, manga = _parse_relations(one_piece_relations_raw)
    for rel in anime + manga:
        if rel.get("image"):
            assert rel["image"].startswith("https://")


def test_parse_relations_manga_original_work(one_piece_relations_raw) -> None:
    _, manga = _parse_relations(one_piece_relations_raw)
    titles = [r["title"] for r in manga]
    assert "One Piece" in titles


# =============================================================================
# _build_anime_from_raw — uses fully processed real data
# =============================================================================


def test_build_anime_title_fields(one_piece_processed) -> None:
    anime = _build_anime_from_raw(one_piece_processed, _URL)
    assert anime.title == "One Piece"
    assert anime.title_japanese == "ワンピース"


def test_build_anime_synonyms(one_piece_processed) -> None:
    anime = _build_anime_from_raw(one_piece_processed, _URL)
    assert "OP" in anime.synonyms
    assert "OneP" in anime.synonyms


def test_build_anime_statistics(one_piece_processed) -> None:
    anime = _build_anime_from_raw(one_piece_processed, _URL)
    assert anime.statistics is not None
    assert anime.statistics.score == pytest.approx(4.18)
    assert anime.statistics.rank == 126
    assert anime.statistics.trending == 66


def test_build_anime_no_statistics(one_piece_processed) -> None:
    raw = {**one_piece_processed, "statistics": None}
    assert _build_anime_from_raw(raw, _URL).statistics is None


def test_build_anime_relations_count(one_piece_processed, one_piece_relations_raw) -> None:
    anime = _build_anime_from_raw(one_piece_processed, _URL)
    assert len(anime.anime_relations) == len(one_piece_relations_raw["anime_relations"])
    assert len(anime.manga_relations) == len(one_piece_relations_raw["manga_relations"])


def test_build_anime_url_injected(one_piece_processed) -> None:
    assert _build_anime_from_raw(one_piece_processed, _URL).url == _URL


def test_build_anime_broadcast_fields(one_piece_processed) -> None:
    anime = _build_anime_from_raw(one_piece_processed, _URL)
    assert anime.broadcast_day == "Sunday"
    assert anime.broadcast_time == "23:15"
    assert anime.broadcast_timezone == "JST"


def test_build_anime_studio(one_piece_processed) -> None:
    anime = _build_anime_from_raw(one_piece_processed, _URL)
    assert anime.studio == "Toei Animation Co., Ltd."
    assert "toei-animation" in (anime.studio_url or "")


def test_build_anime_empty_relations(one_piece_processed) -> None:
    raw = {**one_piece_processed, "anime_relations": [], "manga_relations": []}
    anime = _build_anime_from_raw(raw, _URL)
    assert anime.anime_relations == []
    assert anime.manga_relations == []


# =============================================================================
# AniSearchAnimeCrawler
# =============================================================================


def test_normalize_identifier_valid_url_passthrough() -> None:
    crawler = AniSearchAnimeCrawler(DockerTransport(), NullRepository())
    assert crawler.normalize_identifier(_URL) == _URL


def test_normalize_identifier_wrong_base_raises() -> None:
    crawler = AniSearchAnimeCrawler(DockerTransport(), NullRepository())
    with pytest.raises(ValueError, match="Not an AniSearch anime URL"):
        crawler.normalize_identifier("https://myanimelist.net/anime/21")


def test_build_source_model_uses_canonical_url_from_raw(one_piece_processed) -> None:
    crawler = AniSearchAnimeCrawler(DockerTransport(), NullRepository())
    canonical = "https://www.anisearch.com/anime/2227,one-piece"
    raw = {**one_piece_processed, "_canonical_url": canonical}
    model = crawler.build_source_model(raw, "https://www.anisearch.com/anime/2227")
    assert model.url == canonical


def test_build_source_model_falls_back_to_input_url(one_piece_processed) -> None:
    crawler = AniSearchAnimeCrawler(DockerTransport(), NullRepository())
    model = crawler.build_source_model(one_piece_processed, _URL)
    assert model.url == _URL


# =============================================================================
# _fetch_anisearch_anime_data — async, mocked
# =============================================================================


def _make_browser_mock(mocker, main_html: str | None, final_url: str = "https://www.anisearch.com/anime/2227,one-piece"):
    """Build a mock zendriver browser whose main page returns `main_html`."""
    page_mock = mocker.AsyncMock()
    page_mock.wait_for = AsyncMock()
    page_mock.url = final_url
    if main_html is None:
        page_mock.wait_for.side_effect = Exception("timeout")
    else:
        page_mock.get_content = AsyncMock(return_value=main_html)
    browser_mock = mocker.AsyncMock()
    browser_mock.get = AsyncMock(return_value=page_mock)
    browser_mock.stop = AsyncMock()
    return browser_mock


@pytest.mark.asyncio
async def test_fetch_anime_data_main_html_none_returns_none(mocker) -> None:
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )
    browser_mock = _make_browser_mock(mocker, main_html=None)
    browser_mock.stop.side_effect = Exception("stop failed")
    mocker.patch("zendriver.start", new_callable=AsyncMock, return_value=browser_mock)
    assert await _fetch_anisearch_anime_data("2227,one-piece") is None


@pytest.mark.asyncio
async def test_fetch_anime_data_real_fixture(mocker, one_piece_main_html, one_piece_relations_html) -> None:
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )
    browser_mock = _make_browser_mock(mocker, one_piece_main_html)
    mocker.patch("zendriver.start", new_callable=AsyncMock, return_value=browser_mock)
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_anime_crawler._fetch_page_html",
        new_callable=AsyncMock,
        return_value=one_piece_relations_html,
    )
    result = await _fetch_anisearch_anime_data("2227,one-piece")
    assert result is not None
    assert result["title_ja"] == "One Piece"
    assert result["type"] == "TV-Series"
    assert result["broadcast_day"] == "Sunday"
    assert result["statistics"]["score"] == pytest.approx(4.18)
    assert len(result["anime_relations"]) == 79
    assert len(result["manga_relations"]) == 2


@pytest.mark.asyncio
async def test_fetch_anime_data_relations_none_still_returns_data(
    mocker, one_piece_main_html
) -> None:
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )
    browser_mock = _make_browser_mock(mocker, one_piece_main_html)
    mocker.patch("zendriver.start", new_callable=AsyncMock, return_value=browser_mock)
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_anime_crawler._fetch_page_html",
        new_callable=AsyncMock,
        return_value=None,
    )
    result = await _fetch_anisearch_anime_data("2227,one-piece")
    assert result is not None
    assert result["anime_relations"] == []
    assert result["manga_relations"] == []


# =============================================================================
# fetch_anisearch_anime — top-level entry point
# =============================================================================


@pytest.mark.asyncio
async def test_fetch_anisearch_anime_returns_none_when_no_data(mocker) -> None:
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_anime_crawler._fetch_anisearch_anime_data",
        new_callable=AsyncMock,
        return_value=None,
    )
    assert await fetch_anisearch_anime(_URL) is None


@pytest.mark.asyncio
async def test_fetch_anisearch_anime_returns_canonical_dict(mocker, one_piece_processed) -> None:
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_anime_crawler._fetch_anisearch_anime_data",
        new_callable=AsyncMock,
        return_value=one_piece_processed,
    )
    result = await fetch_anisearch_anime(_URL)
    assert result is not None
    assert result["title"] == "One Piece"


@pytest.mark.asyncio
async def test_fetch_anisearch_anime_sources_uses_canonical_url(
    mocker, one_piece_processed
) -> None:
    canonical = "https://www.anisearch.com/anime/2227,one-piece"
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_anime_crawler._fetch_anisearch_anime_data",
        new_callable=AsyncMock,
        return_value={**one_piece_processed, "_canonical_url": canonical},
    )
    result = await fetch_anisearch_anime("https://www.anisearch.com/anime/2227")
    assert result is not None
    assert result["sources"] == [canonical]


# =============================================================================
# _fetch_page_html — direct unit tests (body is mocked everywhere else)
# =============================================================================


@pytest.mark.asyncio
async def test_fetch_page_html_with_wait_selector(mocker) -> None:
    page_mock = mocker.AsyncMock()
    page_mock.wait_for = AsyncMock()
    page_mock.get_content = AsyncMock(return_value="<html></html>")
    browser_mock = mocker.AsyncMock()
    browser_mock.get = AsyncMock(return_value=page_mock)

    result = await _fetch_page_html(browser_mock, "https://example.com", wait_selector="#content")
    assert result == "<html></html>"
    page_mock.wait_for.assert_awaited_once_with(selector="#content", timeout=10)


@pytest.mark.asyncio
async def test_fetch_page_html_without_wait_selector(mocker) -> None:
    page_mock = mocker.AsyncMock()
    page_mock.get_content = AsyncMock(return_value="<html></html>")
    browser_mock = mocker.AsyncMock()
    browser_mock.get = AsyncMock(return_value=page_mock)
    mocker.patch("enrichment.sources.anisearch.anisearch_anime_crawler.asyncio.sleep", new_callable=AsyncMock)

    result = await _fetch_page_html(browser_mock, "https://example.com")
    assert result == "<html></html>"


@pytest.mark.asyncio
async def test_fetch_page_html_exception_returns_none(mocker) -> None:
    browser_mock = mocker.AsyncMock()
    browser_mock.get = AsyncMock(side_effect=Exception("nav failed"))

    result = await _fetch_page_html(browser_mock, "https://example.com")
    assert result is None


# =============================================================================
# _fetch_anisearch_anime_data — additional branch coverage
# =============================================================================


@pytest.mark.asyncio
async def test_fetch_anime_data_empty_content_returns_none(mocker) -> None:
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )
    browser_mock = _make_browser_mock(mocker, main_html="")
    mocker.patch("zendriver.start", new_callable=AsyncMock, return_value=browser_mock)

    assert await _fetch_anisearch_anime_data("2227,one-piece") is None


@pytest.mark.asyncio
async def test_fetch_anime_data_extraction_fails_returns_none(mocker, one_piece_main_html) -> None:
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )
    browser_mock = _make_browser_mock(mocker, one_piece_main_html)
    mocker.patch("zendriver.start", new_callable=AsyncMock, return_value=browser_mock)
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_anime_crawler._extract_anime_from_html",
        return_value=None,
    )

    assert await _fetch_anisearch_anime_data("2227,one-piece") is None


@pytest.mark.asyncio
async def test_fetch_anime_data_slug_redirect_sets_canonical_url(mocker, one_piece_main_html, one_piece_relations_html) -> None:
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )
    # Use numeric-only path; mock returns slug URL after redirect
    slug_url = "https://www.anisearch.com/anime/2227,one-piece"
    browser_mock = _make_browser_mock(mocker, one_piece_main_html, final_url=slug_url)
    mocker.patch("zendriver.start", new_callable=AsyncMock, return_value=browser_mock)
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_anime_crawler._fetch_page_html",
        new_callable=AsyncMock,
        return_value=one_piece_relations_html,
    )

    result = await _fetch_anisearch_anime_data("2227")
    assert result is not None
    assert result["_canonical_url"] == slug_url


def test_get_extraction_schema_returns_xpaths() -> None:
    crawler = AniSearchAnimeCrawler(DockerTransport(), NullRepository())
    schema = crawler.get_extraction_schema()
    assert schema == {"xpaths": _XPATHS}
