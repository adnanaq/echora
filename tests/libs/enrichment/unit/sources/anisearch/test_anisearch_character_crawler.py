"""Unit tests for AniSearch character detail crawler and mapper.

Baseline tests use real XPath extraction output captured from:
- https://www.anisearch.com/character/4852,monkey-d-luffy (2026-04-23)

Edge-case branches use inline overrides on top of the real fixture dict.
No network calls are made.
"""

import json
from unittest.mock import AsyncMock

import pytest
from enrichment.sources.anisearch.anisearch_character_crawler import (
    _ANISEARCH_BASE_URL,
    AniSearchCharacterCrawler,
    _absolutize_anime_url,
    _batch_fetch_ography,
    _build_character,
    _extract_attributes,
    _extract_voice_actors,
    _fetch_anisearch_character_data,
    _fetch_character_ography_data,
    _get_character_schema,
    _get_ography_schema,
    _ography_to_roles,
    _parse_favorites,
    _parse_ography_result,
    _post_process_character,
    fetch_anisearch_character,
    fetch_anisearch_characters,
)
from enrichment.sources.anisearch.anisearch_mapper import character_from_anisearch
from enrichment.sources.base.framework import DockerTransport, NullRepository

pytestmark = pytest.mark.asyncio

_LUFFY_URL = "https://www.anisearch.com/character/4852,monkey-d-luffy"


# =============================================================================
# Processed fixture — real raw data through _post_process_character
# =============================================================================


@pytest.fixture(scope="session")
def luffy_char_processed(luffy_char_raw):
    return _post_process_character(luffy_char_raw)


# =============================================================================
# _get_character_schema
# =============================================================================


def test_character_schema_has_required_fields() -> None:
    schema = _get_character_schema()
    names = {f["name"] for f in schema["fields"]}
    assert {
        "name",
        "name_native",
        "image",
        "favorites",
        "description",
        "anime_roles",
    } <= names


def test_character_schema_image_reads_src_attribute() -> None:
    schema = _get_character_schema()
    field = next(f for f in schema["fields"] if f["name"] == "image")
    assert field.get("attribute") == "src"
    assert "details-cover" in field["selector"]


def test_character_schema_name_targets_htitle() -> None:
    schema = _get_character_schema()
    field = next(f for f in schema["fields"] if f["name"] == "name")
    assert "htitle" in field["selector"]


def test_character_schema_anime_roles_is_list_with_url_and_title() -> None:
    schema = _get_character_schema()
    field = next(f for f in schema["fields"] if f["name"] == "anime_roles")
    assert field["type"] == "list"
    subfield_names = {f["name"] for f in field["fields"]}
    assert {"url", "title"} <= subfield_names


def test_character_schema_name_native_targets_infoblock_ja() -> None:
    schema = _get_character_schema()
    field = next(f for f in schema["fields"] if f["name"] == "name_native")
    assert "infoblock" in field["selector"]
    assert "lang='ja'" in field["selector"]


# =============================================================================
# _get_ography_schema
# =============================================================================


def test_ography_schema_entries_is_list_with_url_and_title() -> None:
    schema = _get_ography_schema()
    entries = next(f for f in schema["fields"] if f["name"] == "entries")
    assert entries["type"] == "list"
    subfield_names = {f["name"] for f in entries["fields"]}
    assert {"url", "title"} == subfield_names
    url_field = next(f for f in entries["fields"] if f["name"] == "url")
    assert url_field.get("attribute") == "href"


# =============================================================================
# _parse_favorites
# =============================================================================


def test_parse_favorites_string_integer() -> None:
    assert _parse_favorites("678") == 678


def test_parse_favorites_with_comma() -> None:
    assert _parse_favorites("1,234") == 1234


def test_parse_favorites_none_returns_none() -> None:
    assert _parse_favorites(None) is None


def test_parse_favorites_empty_string_returns_none() -> None:
    assert _parse_favorites("") is None


def test_parse_favorites_no_digits_returns_none() -> None:
    assert _parse_favorites("n/a") is None


# =============================================================================
# _absolutize_anime_url
# =============================================================================


def test_absolutize_anime_url_relative() -> None:
    assert _absolutize_anime_url("anime/2227,one-piece") == (
        f"{_ANISEARCH_BASE_URL}/anime/2227,one-piece"
    )


def test_absolutize_anime_url_already_absolute() -> None:
    url = f"{_ANISEARCH_BASE_URL}/anime/2227,one-piece"
    assert _absolutize_anime_url(url) == url


# =============================================================================
# _post_process_character — real fixture
# =============================================================================


def test_post_process_favorites_parsed_to_int(luffy_char_raw) -> None:
    data = _post_process_character(luffy_char_raw)
    assert data["favorites"] == 678


def test_post_process_anime_roles_urls_absolutized(luffy_char_raw) -> None:
    data = _post_process_character(luffy_char_raw)
    for role in data["anime_roles"]:
        assert role["url"].startswith("https://"), f"Not absolute: {role['url']}"


def test_post_process_favorites_none_stays_none(luffy_char_raw) -> None:
    raw = {**luffy_char_raw, "favorites": None}
    assert _post_process_character(raw)["favorites"] is None


def test_post_process_favorites_empty_string_is_none(luffy_char_raw) -> None:
    raw = {**luffy_char_raw, "favorites": ""}
    assert _post_process_character(raw)["favorites"] is None


# =============================================================================
# _extract_voice_actors — real _html from fixture
# =============================================================================


def test_extract_voice_actors_returns_list(luffy_char_raw) -> None:
    vas = _extract_voice_actors(luffy_char_raw["_html"])
    assert len(vas) > 0


def test_extract_voice_actors_mayumi_tanaka_japanese(luffy_char_raw) -> None:
    vas = _extract_voice_actors(luffy_char_raw["_html"])
    mayumi = next((v for v in vas if "TANAKA" in v.name or "Tanaka" in v.name), None)
    assert mayumi is not None
    assert mayumi.language == "Japanese"
    assert mayumi.url is not None and "anisearch.com" in mayumi.url


def test_extract_voice_actors_colleen_clinkenbeard_english(luffy_char_raw) -> None:
    vas = _extract_voice_actors(luffy_char_raw["_html"])
    colleen = next(
        (v for v in vas if "CLINKENBEARD" in v.name or "Clinkenbeard" in v.name), None
    )
    assert colleen is not None
    assert colleen.language == "English"


def test_extract_voice_actors_empty_html_returns_empty() -> None:
    assert _extract_voice_actors("") == []


def test_extract_voice_actors_no_infoblock_returns_empty() -> None:
    assert (
        _extract_voice_actors("<html><body><p>No infoblock here</p></body></html>")
        == []
    )


def test_extract_voice_actors_skips_whitespace_only_name() -> None:
    # regex matches href but name is only whitespace → skipped
    html = (
        '<ul class="infoblock"><li>'
        '<div class="title" lang="ja">.</div>'
        '<a href="person/999,noname">   </a>'
        "</li></ul>"
    )
    assert _extract_voice_actors(html) == []


# =============================================================================
# _extract_attributes — real _html from fixture
# =============================================================================


def test_extract_attributes_returns_dict(luffy_char_raw) -> None:
    attrs = _extract_attributes(luffy_char_raw["_html"])
    assert isinstance(attrs, dict)
    assert len(attrs) > 0
    assert attrs.get("gender") == "Male"
    assert "age" in attrs


def test_extract_attributes_empty_html_returns_empty() -> None:
    assert _extract_attributes("") == {}


# =============================================================================
# _build_character — uses processed fixture
# =============================================================================


def test_build_character_name(luffy_char_processed) -> None:
    char = _build_character(
        luffy_char_processed, luffy_char_processed.get("_html", ""), _LUFFY_URL
    )
    assert char.name == "Monkey D. Luffy"


def test_build_character_name_native(luffy_char_processed) -> None:
    char = _build_character(
        luffy_char_processed, luffy_char_processed.get("_html", ""), _LUFFY_URL
    )
    assert char.name_native == "モンキー・D・ルフィ"


def test_build_character_image(luffy_char_processed) -> None:
    char = _build_character(
        luffy_char_processed, luffy_char_processed.get("_html", ""), _LUFFY_URL
    )
    assert char.image is not None and char.image.startswith("https://")


def test_build_character_favorites(luffy_char_processed) -> None:
    char = _build_character(
        luffy_char_processed, luffy_char_processed.get("_html", ""), _LUFFY_URL
    )
    assert char.favorites == 678


def test_build_character_description_placeholder_is_none(luffy_char_processed) -> None:
    # Luffy's page has AniSearch placeholder text — must be nulled out
    char = _build_character(
        luffy_char_processed, luffy_char_processed.get("_html", ""), _LUFFY_URL
    )
    assert char.description is None


def test_build_character_source_url_injected(luffy_char_processed) -> None:
    char = _build_character(
        luffy_char_processed, luffy_char_processed.get("_html", ""), _LUFFY_URL
    )
    assert char.source == _LUFFY_URL


def test_build_character_role_injected(luffy_char_processed) -> None:
    char = _build_character(
        luffy_char_processed,
        luffy_char_processed.get("_html", ""),
        _LUFFY_URL,
        role="Main Character",
    )
    assert char.role == "Main Character"


def test_build_character_tags_populated(luffy_char_processed) -> None:
    char = _build_character(
        luffy_char_processed, luffy_char_processed.get("_html", ""), _LUFFY_URL
    )
    assert len(char.tags) > 0 and all(isinstance(t, str) for t in char.tags)


def test_build_character_voice_actors_populated(luffy_char_processed) -> None:
    char = _build_character(
        luffy_char_processed, luffy_char_processed.get("_html", ""), _LUFFY_URL
    )
    assert len(char.voice_actors) > 0


def test_build_character_anime_roles_urls_absolute(luffy_char_processed) -> None:
    char = _build_character(
        luffy_char_processed, luffy_char_processed.get("_html", ""), _LUFFY_URL
    )
    for role in char.anime_roles:
        assert role.url is None or role.url.startswith("https://")


def test_build_character_anime_ography_injected(luffy_char_processed) -> None:
    ography = [
        {"url": "https://www.anisearch.com/anime/2227,one-piece", "title": "One Piece"}
    ]
    char = _build_character(
        luffy_char_processed,
        luffy_char_processed.get("_html", ""),
        _LUFFY_URL,
        anime_ography=ography,
    )
    assert len(char.anime_ography) == 1
    assert char.anime_ography[0].title == "One Piece"


def test_build_character_screenshot_images(luffy_char_processed) -> None:
    char = _build_character(
        luffy_char_processed, luffy_char_processed.get("_html", ""), _LUFFY_URL
    )
    assert len(char.screenshot_images) > 0


def test_build_character_attributes_populated(luffy_char_processed) -> None:
    char = _build_character(
        luffy_char_processed, luffy_char_processed.get("_html", ""), _LUFFY_URL
    )
    assert len(char.attributes) > 0


def test_build_character_empty_name_is_none(luffy_char_processed) -> None:
    raw = {**luffy_char_processed, "name": ""}
    char = _build_character(raw, "", _LUFFY_URL)
    assert char.name is None


def test_build_character_real_description_passes_through(luffy_char_processed) -> None:
    raw = {
        **luffy_char_processed,
        "description": "A fearless pirate who wants to be King.",
    }
    char = _build_character(raw, "", _LUFFY_URL)
    assert char.description == "A fearless pirate who wants to be King."


# =============================================================================
# _fetch_anisearch_character_data (async, mocked)
# =============================================================================


async def test_fetch_character_data_none_result_returns_none(mocker) -> None:
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler.crawl_batch_urls",
        new_callable=AsyncMock,
        return_value=[None],
    )
    assert await _fetch_anisearch_character_data(_LUFFY_URL) is None


async def test_fetch_character_data_404_returns_none(mocker) -> None:
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler.crawl_batch_urls",
        new_callable=AsyncMock,
        return_value=[{"status_code": 404, "extracted_content": "[]", "html": ""}],
    )
    assert await _fetch_anisearch_character_data(_LUFFY_URL) is None


async def test_fetch_character_data_real_fixture(mocker, luffy_char_raw) -> None:
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )
    html = luffy_char_raw.get("_html", "")
    raw_without_html = {k: v for k, v in luffy_char_raw.items() if k != "_html"}
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler.crawl_batch_urls",
        new_callable=AsyncMock,
        return_value=[
            {
                "status_code": 200,
                "extracted_content": json.dumps([raw_without_html]),
                "html": html,
            }
        ],
    )
    result = await _fetch_anisearch_character_data(_LUFFY_URL)
    assert result is not None
    assert result["name"] == "Monkey D. Luffy"
    assert result["favorites"] == 678  # post-processed to int
    assert result["_html"] == html


async def test_fetch_character_data_redirect_logs_debug_continues(
    mocker, luffy_char_raw
) -> None:
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )
    raw_without_html = {k: v for k, v in luffy_char_raw.items() if k != "_html"}
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler.crawl_batch_urls",
        new_callable=AsyncMock,
        return_value=[
            {
                "status_code": 301,
                "extracted_content": json.dumps([raw_without_html]),
                "html": "",
            }
        ],
    )
    result = await _fetch_anisearch_character_data(_LUFFY_URL)
    assert result is not None
    assert result["name"] == "Monkey D. Luffy"


async def test_fetch_character_data_empty_items_returns_none(mocker) -> None:
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler.crawl_batch_urls",
        new_callable=AsyncMock,
        return_value=[{"status_code": 200, "extracted_content": "[]", "html": ""}],
    )
    assert await _fetch_anisearch_character_data(_LUFFY_URL) is None


# =============================================================================
# _fetch_character_ography_data (async, mocked)
# =============================================================================


async def test_fetch_ography_data_none_result_returns_none(mocker) -> None:
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler.crawl_batch_urls",
        new_callable=AsyncMock,
        return_value=[None],
    )
    assert await _fetch_character_ography_data(f"{_LUFFY_URL}/anime") is None


async def test_fetch_ography_data_404_returns_none(mocker) -> None:
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler.crawl_batch_urls",
        new_callable=AsyncMock,
        return_value=[{"status_code": 404, "extracted_content": "[]"}],
    )
    assert await _fetch_character_ography_data(f"{_LUFFY_URL}/anime") is None


async def test_fetch_ography_data_empty_items_returns_none(mocker) -> None:
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler.crawl_batch_urls",
        new_callable=AsyncMock,
        return_value=[{"status_code": 200, "extracted_content": "[]"}],
    )
    assert await _fetch_character_ography_data(f"{_LUFFY_URL}/anime") is None


async def test_fetch_ography_data_valid_returns_list(mocker) -> None:
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler.crawl_batch_urls",
        new_callable=AsyncMock,
        return_value=[
            {
                "status_code": 200,
                "extracted_content": json.dumps(
                    [
                        {
                            "entries": [
                                {"url": "anime/2227,one-piece", "title": "One Piece"}
                            ]
                        }
                    ]
                ),
            }
        ],
    )
    result = await _fetch_character_ography_data(f"{_LUFFY_URL}/anime")
    assert result is not None
    assert result[0]["title"] == "One Piece"
    assert result[0]["url"].startswith("https://")


# =============================================================================
# _parse_ography_result
# =============================================================================


def test_parse_ography_result_none_returns_none() -> None:
    assert _parse_ography_result(None, "https://example.com") is None


def test_parse_ography_result_404_returns_none() -> None:
    assert (
        _parse_ography_result(
            {"status_code": 404, "extracted_content": "[]"}, "https://example.com"
        )
        is None
    )


def test_parse_ography_result_empty_items_returns_none() -> None:
    assert (
        _parse_ography_result(
            {"status_code": 200, "extracted_content": "[]"}, "https://example.com"
        )
        is None
    )


def test_parse_ography_result_valid_returns_absolute_list() -> None:
    result = {
        "status_code": 200,
        "extracted_content": json.dumps(
            [{"entries": [{"url": "anime/2227,one-piece", "title": "One Piece"}]}]
        ),
    }
    entries = _parse_ography_result(result, "https://example.com")
    assert entries is not None
    assert len(entries) == 1
    assert entries[0]["title"] == "One Piece"
    assert entries[0]["url"].startswith("https://")


# =============================================================================
# AniSearchCharacterCrawler.post_process_raw_data
# =============================================================================


async def test_crawler_post_process_fetches_both_ography_pages(mocker) -> None:
    ography_entry = [
        {"url": "https://www.anisearch.com/anime/2227,one-piece", "title": "One Piece"}
    ]
    mock_ography = AsyncMock(return_value=ography_entry)
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler._fetch_character_ography_data",
        mock_ography,
    )
    crawler = AniSearchCharacterCrawler(DockerTransport(), NullRepository())
    result = await crawler.post_process_raw_data({"_html": ""}, _LUFFY_URL)
    assert "_anime_ography" in result
    assert "_manga_ography" in result
    assert result["_anime_ography"] == ography_entry
    assert result["_manga_ography"] == ography_entry
    assert mock_ography.call_count == 2


async def test_crawler_post_process_ography_none_on_failure(mocker) -> None:
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler._fetch_character_ography_data",
        new_callable=AsyncMock,
        return_value=None,
    )
    crawler = AniSearchCharacterCrawler(DockerTransport(), NullRepository())
    result = await crawler.post_process_raw_data(
        {"_html": "", "name": "Luffy"}, _LUFFY_URL
    )
    assert result["_anime_ography"] is None
    assert result["_manga_ography"] is None
    assert result["name"] == "Luffy"


# =============================================================================
# fetch_anisearch_character (top-level, mocked)
# =============================================================================


async def test_fetch_anisearch_character_none_data_returns_none(mocker) -> None:
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler._fetch_anisearch_character_data",
        new_callable=AsyncMock,
        return_value=None,
    )
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler._fetch_character_ography_data",
        new_callable=AsyncMock,
        return_value=None,
    )
    assert await fetch_anisearch_character(_LUFFY_URL) is None


async def test_fetch_anisearch_character_returns_canonical_dict(
    mocker, luffy_char_processed
) -> None:
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler._fetch_anisearch_character_data",
        new_callable=AsyncMock,
        return_value=luffy_char_processed,
    )
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler._fetch_character_ography_data",
        new_callable=AsyncMock,
        return_value=None,
    )
    result = await fetch_anisearch_character(_LUFFY_URL)
    assert result is not None
    assert result["name"] == "Monkey D. Luffy"
    assert result["sources"] == [_LUFFY_URL]


# =============================================================================
# fetch_anisearch_characters (batch, mocked)
# =============================================================================


async def test_fetch_anisearch_characters_empty_refs_returns_empty() -> None:
    assert await fetch_anisearch_characters([]) == []


async def test_fetch_anisearch_characters_all_cached_no_crawl(
    mocker, luffy_char_processed
) -> None:
    refs = [{"url": _LUFFY_URL, "role": "Main Character"}]
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler._fetch_anisearch_character_data.cache_batch_get",
        new_callable=AsyncMock,
        return_value=([luffy_char_processed], []),
    )
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler._batch_fetch_ography",
        new_callable=AsyncMock,
        side_effect=[[None], [None]],
    )
    results = await fetch_anisearch_characters(refs)
    assert len(results) == 1
    assert results[0] is not None
    assert results[0]["name"] == "Monkey D. Luffy"


async def test_fetch_anisearch_characters_on_result_callback_invoked(
    mocker, luffy_char_processed
) -> None:
    refs = [{"url": _LUFFY_URL, "role": "Main Character"}]
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler._fetch_anisearch_character_data.cache_batch_get",
        new_callable=AsyncMock,
        return_value=([luffy_char_processed], []),
    )
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler._batch_fetch_ography",
        new_callable=AsyncMock,
        side_effect=[[None], [None]],
    )
    collected: list = []
    await fetch_anisearch_characters(refs, on_result=collected.append)
    assert len(collected) == 1
    assert collected[0]["name"] == "Monkey D. Luffy"


async def test_fetch_anisearch_characters_uncached_crawl_succeeds(
    mocker, luffy_char_raw
) -> None:
    refs = [{"url": _LUFFY_URL, "role": "Main Character"}]
    html = luffy_char_raw.get("_html", "")
    raw_without_html = {k: v for k, v in luffy_char_raw.items() if k != "_html"}
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler._fetch_anisearch_character_data.cache_batch_get",
        new_callable=AsyncMock,
        return_value=([None], [0]),
    )
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler._fetch_anisearch_character_data.cache_batch_set",
        new_callable=AsyncMock,
    )
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler.crawl_batch_urls",
        new_callable=AsyncMock,
        return_value=[
            {
                "status_code": 200,
                "extracted_content": json.dumps([raw_without_html]),
                "html": html,
            }
        ],
    )
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler._batch_fetch_ography",
        new_callable=AsyncMock,
        side_effect=[[None], [None]],
    )
    results = await fetch_anisearch_characters(refs)
    assert len(results) == 1
    assert results[0] is not None
    assert results[0]["name"] == "Monkey D. Luffy"


async def test_fetch_anisearch_characters_uncached_crawl_empty_items_raw_stays_none(
    mocker,
) -> None:
    # empty items → raw_data[0] stays None → step 3 skips (covers `if raw is None: continue`)
    refs = [{"url": _LUFFY_URL, "role": "Main Character"}]
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler._fetch_anisearch_character_data.cache_batch_get",
        new_callable=AsyncMock,
        return_value=([None], [0]),
    )
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler._fetch_anisearch_character_data.cache_batch_set",
        new_callable=AsyncMock,
    )
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler.crawl_batch_urls",
        new_callable=AsyncMock,
        return_value=[{"status_code": 200, "extracted_content": "[]", "html": ""}],
    )
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler._batch_fetch_ography",
        new_callable=AsyncMock,
        side_effect=[[None], [None]],
    )
    results = await fetch_anisearch_characters(refs)
    assert results == [None]


async def test_fetch_anisearch_characters_uncached_crawl_none_result_skipped(
    mocker,
) -> None:
    # crawl returns None for the result item → `if not result: continue`
    refs = [{"url": _LUFFY_URL, "role": "Main Character"}]
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler._fetch_anisearch_character_data.cache_batch_get",
        new_callable=AsyncMock,
        return_value=([None], [0]),
    )
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler._fetch_anisearch_character_data.cache_batch_set",
        new_callable=AsyncMock,
    )
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler.crawl_batch_urls",
        new_callable=AsyncMock,
        return_value=[None],
    )
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler._batch_fetch_ography",
        new_callable=AsyncMock,
        side_effect=[[None], [None]],
    )
    results = await fetch_anisearch_characters(refs)
    assert results == [None]


async def test_fetch_anisearch_characters_uncached_crawl_4xx_skipped(mocker) -> None:
    # 4xx status in batch crawl loop → logged and skipped
    refs = [{"url": _LUFFY_URL, "role": "Main Character"}]
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler._fetch_anisearch_character_data.cache_batch_get",
        new_callable=AsyncMock,
        return_value=([None], [0]),
    )
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler._fetch_anisearch_character_data.cache_batch_set",
        new_callable=AsyncMock,
    )
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler.crawl_batch_urls",
        new_callable=AsyncMock,
        return_value=[{"status_code": 404, "extracted_content": "[]", "html": ""}],
    )
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler._batch_fetch_ography",
        new_callable=AsyncMock,
        side_effect=[[None], [None]],
    )
    results = await fetch_anisearch_characters(refs)
    assert results == [None]


async def test_fetch_anisearch_characters_uncached_crawl_3xx_logs_and_continues(
    mocker, luffy_char_raw
) -> None:
    # 3xx redirect in batch loop → debug log, still parses content
    refs = [{"url": _LUFFY_URL, "role": "Main Character"}]
    raw_without_html = {k: v for k, v in luffy_char_raw.items() if k != "_html"}
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler._fetch_anisearch_character_data.cache_batch_get",
        new_callable=AsyncMock,
        return_value=([None], [0]),
    )
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler._fetch_anisearch_character_data.cache_batch_set",
        new_callable=AsyncMock,
    )
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler.crawl_batch_urls",
        new_callable=AsyncMock,
        return_value=[
            {
                "status_code": 301,
                "extracted_content": json.dumps([raw_without_html]),
                "html": luffy_char_raw.get("_html", ""),
            }
        ],
    )
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler._batch_fetch_ography",
        new_callable=AsyncMock,
        side_effect=[[None], [None]],
    )
    results = await fetch_anisearch_characters(refs)
    assert len(results) == 1
    assert results[0] is not None
    assert results[0]["name"] == "Monkey D. Luffy"


# =============================================================================
# _batch_fetch_ography
# =============================================================================


async def test_batch_fetch_ography_missing_indices_fetched(mocker) -> None:
    sub_urls = [f"{_LUFFY_URL}/anime"]
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler._fetch_character_ography_data.cache_batch_get",
        new_callable=AsyncMock,
        return_value=([None], [0]),
    )
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler._fetch_character_ography_data.cache_batch_set",
        new_callable=AsyncMock,
    )
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler.crawl_batch_urls",
        new_callable=AsyncMock,
        return_value=[
            {
                "status_code": 200,
                "extracted_content": json.dumps(
                    [
                        {
                            "entries": [
                                {"url": "anime/2227,one-piece", "title": "One Piece"}
                            ]
                        }
                    ]
                ),
            }
        ],
    )
    results = await _batch_fetch_ography(sub_urls)
    assert len(results) == 1
    assert results[0] is not None
    assert results[0][0]["title"] == "One Piece"


# =============================================================================
# AniSearchCharacterCrawler
# =============================================================================


def test_crawler_normalize_identifier_passthrough() -> None:
    crawler = AniSearchCharacterCrawler(DockerTransport(), NullRepository())
    assert crawler.normalize_identifier(_LUFFY_URL) == _LUFFY_URL


async def test_crawler_fetch_raw_data_delegates(mocker, luffy_char_processed) -> None:
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler._fetch_anisearch_character_data",
        new_callable=AsyncMock,
        return_value=luffy_char_processed,
    )
    crawler = AniSearchCharacterCrawler(DockerTransport(), NullRepository())
    result = await crawler.fetch_raw_data(_LUFFY_URL)
    assert result is luffy_char_processed


def test_crawler_build_source_model_with_role(luffy_char_processed) -> None:
    crawler = AniSearchCharacterCrawler(
        DockerTransport(), NullRepository(), role="Main Character"
    )
    char = crawler.build_source_model(luffy_char_processed, _LUFFY_URL)
    assert char.name == "Monkey D. Luffy"
    assert char.role == "Main Character"


def test_crawler_map_to_canonical(luffy_char_processed) -> None:
    crawler = AniSearchCharacterCrawler(DockerTransport(), NullRepository())
    char = _build_character(
        luffy_char_processed, luffy_char_processed.get("_html", ""), _LUFFY_URL
    )
    result = crawler.map_to_canonical(char)
    assert result["name"] == "Monkey D. Luffy"
    assert result["sources"] == [_LUFFY_URL]


# =============================================================================
# character_from_anisearch (mapper)
# =============================================================================


def test_character_from_anisearch_happy_path(luffy_char_processed) -> None:
    char = _build_character(
        luffy_char_processed, luffy_char_processed.get("_html", ""), _LUFFY_URL
    )
    result = character_from_anisearch(char)
    assert result["name"] == "Monkey D. Luffy"
    assert result["sources"] == [_LUFFY_URL]
    assert result.get("name_native") == "モンキー・D・ルフィ"
    assert result.get("images") and char.image in result["images"]
    assert result.get("traits") and all(isinstance(t, str) for t in result["traits"])
    names = [v["name"] for v in result.get("voice_actors", [])]
    assert any("TANAKA" in n or "Tanaka" in n for n in names)
    assert "description" not in result
    assert result.get("attributes", {}).get("gender") == "Male"


def test_character_from_anisearch_role_in_roles(luffy_char_processed) -> None:
    char = _build_character(
        luffy_char_processed,
        luffy_char_processed.get("_html", ""),
        _LUFFY_URL,
        role="Main Character",
    )
    result = character_from_anisearch(char)
    assert "MAIN" in result.get("roles", [])
