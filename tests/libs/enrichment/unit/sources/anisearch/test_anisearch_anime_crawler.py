"""Unit tests for anisearch_anime_crawler.py — schema structure and post-processing helpers.

Tests validate the XPath extraction pattern using real crawl4ai output saved as fixtures
(tests/fixtures/one_piece_*.json, captured from anisearch.com/anime/2227,one-piece).
Edge-case branches use field overrides on top of the real fixture dict.
No network calls are made.
"""

import json
from unittest.mock import AsyncMock

import pytest

from enrichment.sources.anisearch.anisearch_anime_crawler import (
    BASE_ANIME_URL,
    AniSearchAnimeCrawler,
    _build_anime_from_raw,
    _extract_path_from_url,
    _fetch_anisearch_anime_data,
    _get_main_schema,
    _get_relations_schema,
    _parse_relations,
    _post_process_main,
    _process_relation_tooltips,
    _unwrap_result,
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
    escaped = '&lt;img src=&quot;https://cdn.anisearch.com/cover.webp&quot;&gt;'
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
    _process_relation_tooltips([])  # must not raise


def test_process_relation_tooltips_real_data(one_piece_relations_raw) -> None:
    """Relations fixture contains HTML in image field — verify tooltip extraction."""
    rels = list(one_piece_relations_raw["anime_relations"])
    _process_relation_tooltips(rels)
    for rel in rels:
        if rel.get("image"):
            assert not rel["image"].startswith("<"), "HTML tag not stripped from image"
            assert rel["image"].startswith("https://"), "Image is not a URL after extraction"


# =============================================================================
# Schema structure — verifies XPath selectors are semantically correct
# =============================================================================


def test_main_schema_has_fields_list() -> None:
    schema = _get_main_schema()
    assert "fields" in schema
    assert isinstance(schema["fields"], list)


def test_main_schema_cover_image_targets_details_cover_id() -> None:
    schema = _get_main_schema()
    field = next(f for f in schema["fields"] if f["name"] == "cover_image")
    assert "details-cover" in field["selector"]
    assert field.get("attribute") == "src"


def test_main_schema_title_alt_targets_grey_div() -> None:
    schema = _get_main_schema()
    field = next(f for f in schema["fields"] if f["name"] == "title_alt")
    assert "grey" in field["selector"]
    assert "lang='ja'" in field["selector"]


def test_main_schema_title_ja_targets_f16_strong() -> None:
    schema = _get_main_schema()
    field = next(f for f in schema["fields"] if f["name"] == "title_ja")
    assert "f16" in field["selector"]
    assert "strong" in field["selector"]


def test_main_schema_genres_anchor_on_genre_href() -> None:
    schema = _get_main_schema()
    field = next(f for f in schema["fields"] if f["name"] == "genres")
    assert "/genre/main/" in field["selector"] or "/genre/subsidiary/" in field["selector"]


def test_main_schema_tags_anchor_on_tag_href() -> None:
    schema = _get_main_schema()
    field = next(f for f in schema["fields"] if f["name"] == "tags")
    assert "/genre/tag/" in field["selector"]


def test_main_schema_websites_is_nested_list() -> None:
    schema = _get_main_schema()
    field = next(f for f in schema["fields"] if f["name"] == "websites")
    assert field["type"] == "nested_list"
    names = [f["name"] for f in field["fields"]]
    assert "name" in names and "url" in names


def test_main_schema_studio_url_reads_href_attribute() -> None:
    schema = _get_main_schema()
    field = next(f for f in schema["fields"] if f["name"] == "studio_url")
    assert field.get("attribute") == "href"


def test_relations_schema_targets_relations_anime_section() -> None:
    schema = _get_relations_schema()
    field = next(f for f in schema["fields"] if f["name"] == "anime_relations")
    assert "relations_anime" in field["selector"]
    assert "tbody" in field["selector"]


def test_relations_schema_targets_relations_manga_section() -> None:
    schema = _get_relations_schema()
    field = next(f for f in schema["fields"] if f["name"] == "manga_relations")
    assert "relations_manga" in field["selector"]


# =============================================================================
# _unwrap_result
# =============================================================================


def test_unwrap_result_none_input() -> None:
    assert _unwrap_result(None, _URL) is None


def test_unwrap_result_404_returns_none() -> None:
    assert _unwrap_result({"status_code": 404}, _URL) is None


def test_unwrap_result_non_200_returns_none() -> None:
    assert _unwrap_result({"status_code": 500}, _URL) is None


def test_unwrap_result_success_false_returns_none() -> None:
    assert _unwrap_result({"success": False, "error_message": "blocked"}, _URL) is None


def test_unwrap_result_empty_json_returns_none() -> None:
    assert _unwrap_result({"success": True, "status_code": 200, "extracted_content": "[]"}, _URL) is None


def test_unwrap_result_valid_returns_first_item(one_piece_main_raw) -> None:
    result = {
        "success": True,
        "status_code": 200,
        "extracted_content": json.dumps([one_piece_main_raw]),
    }
    assert _unwrap_result(result, _URL) == one_piece_main_raw


def test_unwrap_result_missing_status_but_success_true(one_piece_main_raw) -> None:
    result = {"success": True, "extracted_content": json.dumps([one_piece_main_raw])}
    assert _unwrap_result(result, _URL) == one_piece_main_raw


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


def test_parse_relations_images_are_urls_after_processing(one_piece_relations_raw) -> None:
    anime, manga = _parse_relations(one_piece_relations_raw)
    for rel in anime + manga:
        if rel.get("image"):
            assert rel["image"].startswith("https://"), f"Image not a URL: {rel['image']}"


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
# AniSearchAnimeCrawler.normalize_identifier
# =============================================================================


def test_normalize_identifier_valid_url_passthrough() -> None:
    crawler = AniSearchAnimeCrawler(DockerTransport(), NullRepository())
    assert crawler.normalize_identifier(_URL) == _URL


def test_normalize_identifier_wrong_base_raises() -> None:
    crawler = AniSearchAnimeCrawler(DockerTransport(), NullRepository())
    with pytest.raises(ValueError, match="Not an AniSearch anime URL"):
        crawler.normalize_identifier("https://myanimelist.net/anime/21")


# =============================================================================
# _fetch_anisearch_anime_data — async, with real fixture data in mocked responses
# =============================================================================


@pytest.mark.asyncio
async def test_fetch_main_page_none_returns_none(mocker) -> None:
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_anime_crawler.crawl_single_url",
        new_callable=AsyncMock,
        return_value=None,
    )
    assert await _fetch_anisearch_anime_data("2227,one-piece") is None


@pytest.mark.asyncio
async def test_fetch_success_with_real_fixture(mocker, one_piece_main_raw, one_piece_relations_raw) -> None:
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_anime_crawler.crawl_single_url",
        new_callable=AsyncMock,
        side_effect=[
            {"status_code": 200, "success": True, "extracted_content": json.dumps([one_piece_main_raw])},
            {"status_code": 200, "success": True, "extracted_content": json.dumps([one_piece_relations_raw])},
        ],
    )
    result = await _fetch_anisearch_anime_data("2227,one-piece")
    assert result is not None
    assert result["title_ja"] == "One Piece"
    assert result["type"] == "TV-Series"
    assert result["broadcast_day"] == "Sunday"
    assert result["statistics"]["score"] == pytest.approx(4.18)
    assert len(result["anime_relations"]) == len(one_piece_main_raw.get("anime_relations", result["anime_relations"]))


@pytest.mark.asyncio
async def test_fetch_relations_none_still_returns_data(mocker, one_piece_main_raw) -> None:
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_anime_crawler.crawl_single_url",
        new_callable=AsyncMock,
        side_effect=[
            {"status_code": 200, "success": True, "extracted_content": json.dumps([one_piece_main_raw])},
            None,
        ],
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
