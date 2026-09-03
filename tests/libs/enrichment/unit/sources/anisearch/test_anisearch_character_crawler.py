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
    _XPATHS,
    AniSearchCharacterCrawler,
    _absolutize_anime_url,
    _build_character_from_raw,
    _extract_attributes,
    _extract_character_from_html,
    _extract_ography_from_html,
    _extract_voice_actors,
    _fetch_anisearch_character_data,
    _fetch_character_ography_data,
    _fetch_ography,
    _fetch_page_html,
    _parse_favorites,
    _post_process_character,
    fetch_anisearch_character,
    fetch_anisearch_characters,
)
from enrichment.sources.anisearch.anisearch_mapper import character_from_anisearch
from enrichment.sources.base.framework import NullRepository

pytestmark = pytest.mark.asyncio

_LUFFY_URL = "https://www.anisearch.com/character/4852,monkey-d-luffy"


# =============================================================================
# Processed fixture — real raw data through _post_process_character
# =============================================================================


@pytest.fixture(scope="session")
def luffy_char_processed(luffy_char_raw):
    return _post_process_character(luffy_char_raw)


# =============================================================================
# _XPATHS dict
# =============================================================================


def test_xpaths_has_required_character_keys() -> None:
    assert {
        "name",
        "name_native",
        "image",
        "favorites",
        "description",
        "anime_roles",
        "tags",
        "screenshot_images",
        "picture_images",
    } <= set(_XPATHS)


def test_xpaths_image_targets_details_cover() -> None:
    assert "details-cover" in _XPATHS["image"]


def test_xpaths_name_targets_htitle() -> None:
    assert "htitle" in _XPATHS["name"]


def test_xpaths_name_native_targets_infoblock_ja() -> None:
    assert "infoblock" in _XPATHS["name_native"]
    assert "ja" in _XPATHS["name_native"]


def test_xpaths_anime_roles_targets_anime_links() -> None:
    assert "anime/" in _XPATHS["anime_roles"]


def test_xpaths_ography_entries_targets_covers_list() -> None:
    assert "covers" in _XPATHS["ography_entries"]
    assert "anime/" in _XPATHS["ography_entries"]


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


def test_post_process_favorites_empty_string_none(luffy_char_raw) -> None:
    raw = {**luffy_char_raw, "favorites": ""}
    assert _post_process_character(raw)["favorites"] is None


# =============================================================================
# _extract_voice_actors — real _html from fixture
# =============================================================================


def test_extract_voice_actors_returns_list(luffy_char_html) -> None:
    vas = _extract_voice_actors(luffy_char_html)
    assert len(vas) > 0


def test_extract_voice_actors_mayumi_tanaka_japanese(luffy_char_html) -> None:
    vas = _extract_voice_actors(luffy_char_html)
    mayumi = next((v for v in vas if "TANAKA" in v.name or "Tanaka" in v.name), None)
    assert mayumi is not None
    assert mayumi.language == "Japanese"
    assert mayumi.url is not None and "anisearch.com" in mayumi.url


def test_extract_voice_actors_colleen_clinkenbeard_english(luffy_char_html) -> None:
    vas = _extract_voice_actors(luffy_char_html)
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


def test_extract_attributes_returns_dict(luffy_char_html) -> None:
    attrs = _extract_attributes(luffy_char_html)
    assert isinstance(attrs, dict)
    assert len(attrs) > 0
    assert attrs.get("gender") == "Male"
    assert "age" in attrs


def test_extract_attributes_empty_html_returns_empty() -> None:
    assert _extract_attributes("") == {}


# =============================================================================
# _build_character_from_raw — uses processed fixture
# =============================================================================


def test_build_character_name(luffy_char_processed) -> None:
    char = _build_character_from_raw(
        luffy_char_processed, luffy_char_processed.get("_html", ""), _LUFFY_URL
    )
    assert char.name == "Monkey D. Luffy"


def test_build_character_name_native(luffy_char_processed) -> None:
    char = _build_character_from_raw(
        luffy_char_processed, luffy_char_processed.get("_html", ""), _LUFFY_URL
    )
    assert char.name_native == "モンキー・D・ルフィ"


def test_build_character_image(luffy_char_processed) -> None:
    char = _build_character_from_raw(
        luffy_char_processed, luffy_char_processed.get("_html", ""), _LUFFY_URL
    )
    assert char.image is not None and char.image.startswith("https://")


def test_build_character_favorites(luffy_char_processed) -> None:
    char = _build_character_from_raw(
        luffy_char_processed, luffy_char_processed.get("_html", ""), _LUFFY_URL
    )
    assert char.favorites == 678


def test_character_description_placeholder_none(luffy_char_processed) -> None:
    # Luffy's page has AniSearch placeholder text — must be nulled out
    char = _build_character_from_raw(
        luffy_char_processed, luffy_char_processed.get("_html", ""), _LUFFY_URL
    )
    assert char.description is None


def test_build_character_source_url_injected(luffy_char_processed) -> None:
    char = _build_character_from_raw(
        luffy_char_processed, luffy_char_processed.get("_html", ""), _LUFFY_URL
    )
    assert char.source == _LUFFY_URL


def test_build_character_role_injected(luffy_char_processed) -> None:
    char = _build_character_from_raw(
        luffy_char_processed,
        luffy_char_processed.get("_html", ""),
        _LUFFY_URL,
        role="Main Character",
    )
    assert char.role == "Main Character"


def test_build_character_tags_populated(luffy_char_processed) -> None:
    char = _build_character_from_raw(
        luffy_char_processed, luffy_char_processed.get("_html", ""), _LUFFY_URL
    )
    assert len(char.tags) > 0 and all(isinstance(t, str) for t in char.tags)


def test_build_character_voice_actors_populated(luffy_char_processed) -> None:
    char = _build_character_from_raw(
        luffy_char_processed, luffy_char_processed.get("_html", ""), _LUFFY_URL
    )
    assert len(char.voice_actors) > 0


def test_build_character_anime_roles_urls_absolute(luffy_char_processed) -> None:
    char = _build_character_from_raw(
        luffy_char_processed, luffy_char_processed.get("_html", ""), _LUFFY_URL
    )
    for role in char.anime_roles:
        assert role.url is None or role.url.startswith("https://")


def test_build_character_anime_ography_injected(luffy_char_processed) -> None:
    ography = [
        {"url": "https://www.anisearch.com/anime/2227,one-piece", "title": "One Piece"}
    ]
    char = _build_character_from_raw(
        luffy_char_processed,
        luffy_char_processed.get("_html", ""),
        _LUFFY_URL,
        anime_ography=ography,
    )
    assert len(char.anime_ography) == 1
    assert char.anime_ography[0].title == "One Piece"


def test_build_character_screenshot_images(luffy_char_processed) -> None:
    char = _build_character_from_raw(
        luffy_char_processed, luffy_char_processed.get("_html", ""), _LUFFY_URL
    )
    assert len(char.screenshot_images) > 0


def test_build_character_attributes_populated(luffy_char_processed) -> None:
    char = _build_character_from_raw(
        luffy_char_processed, luffy_char_processed.get("_html", ""), _LUFFY_URL
    )
    assert len(char.attributes) > 0


def test_build_character_empty_name_none(luffy_char_processed) -> None:
    raw = {**luffy_char_processed, "name": ""}
    char = _build_character_from_raw(raw, "", _LUFFY_URL)
    assert char.name is None


def test_build_character_real_description_passes_through(luffy_char_processed) -> None:
    raw = {
        **luffy_char_processed,
        "description": "A fearless pirate who wants to be King.",
    }
    char = _build_character_from_raw(raw, "", _LUFFY_URL)
    assert char.description == "A fearless pirate who wants to be King."


# =============================================================================
# _extract_character_from_html
# =============================================================================


def test_extract_character_from_html_empty_body_returns_partial() -> None:
    result = _extract_character_from_html("<html><body></body></html>")
    assert result is not None
    assert result["name"] is None


# =============================================================================
# _fetch_page_html
# =============================================================================


@pytest.mark.asyncio
async def test_fetch_page_html_with_wait_selector(mocker) -> None:
    page_mock = mocker.AsyncMock()
    page_mock.wait_for = AsyncMock()
    page_mock.get_content = AsyncMock(return_value="<html></html>")
    browser_mock = mocker.AsyncMock()
    browser_mock.get = AsyncMock(return_value=page_mock)

    result = await _fetch_page_html(browser_mock, "https://example.com", wait_selector="#htitle")
    assert result == "<html></html>"
    page_mock.wait_for.assert_awaited_once_with(selector="#htitle", timeout=10)


@pytest.mark.asyncio
async def test_fetch_page_html_without_wait_selector(mocker) -> None:
    page_mock = mocker.AsyncMock()
    page_mock.get_content = AsyncMock(return_value="<html></html>")
    browser_mock = mocker.AsyncMock()
    browser_mock.get = AsyncMock(return_value=page_mock)
    mocker.patch("enrichment.sources.anisearch.anisearch_character_crawler.asyncio.sleep", new_callable=AsyncMock)

    result = await _fetch_page_html(browser_mock, "https://example.com")
    assert result == "<html></html>"


@pytest.mark.asyncio
async def test_fetch_page_html_exception_returns_none(mocker) -> None:
    browser_mock = mocker.AsyncMock()
    browser_mock.get = AsyncMock(side_effect=Exception("nav failed"))

    assert await _fetch_page_html(browser_mock, "https://example.com") is None


# =============================================================================
# _fetch_anisearch_character_data (async, mocked)
# =============================================================================


async def test_fetch_character_data_fetch_error_returns_none(mocker) -> None:
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )
    browser_mock = mocker.AsyncMock()
    browser_mock.stop.side_effect = Exception("stop failed")
    mocker.patch("zendriver.start", new_callable=AsyncMock, return_value=browser_mock)
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler._fetch_page_html",
        new_callable=AsyncMock,
        return_value=None,
    )
    assert await _fetch_anisearch_character_data(_LUFFY_URL) is None


async def test_fetch_character_data_extraction_fails_returns_none(mocker) -> None:
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )
    mocker.patch("zendriver.start", new_callable=AsyncMock, return_value=mocker.AsyncMock())
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler._fetch_page_html",
        new_callable=AsyncMock,
        return_value="<html></html>",
    )
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler._extract_character_from_html",
        return_value=None,
    )
    assert await _fetch_anisearch_character_data(_LUFFY_URL) is None


async def test_fetch_character_data_real_fixture(mocker, luffy_char_html) -> None:
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )
    mocker.patch(
        "zendriver.start",
        new_callable=AsyncMock,
        return_value=mocker.AsyncMock(),
    )
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler._fetch_page_html",
        new_callable=AsyncMock,
        return_value=luffy_char_html,
    )
    result = await _fetch_anisearch_character_data(_LUFFY_URL)
    assert result is not None
    assert result["name"] == "Monkey D. Luffy"
    assert result["favorites"] == 682  # post-processed to int
    assert result["_html"] == luffy_char_html


# =============================================================================
# _fetch_character_ography_data (async, mocked)
# =============================================================================


async def test_fetch_ography_data_fetch_error_returns_none(mocker) -> None:
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )
    browser_mock = mocker.AsyncMock()
    browser_mock.stop.side_effect = Exception("stop failed")
    mocker.patch("zendriver.start", new_callable=AsyncMock, return_value=browser_mock)
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler._fetch_page_html",
        new_callable=AsyncMock,
        return_value=None,
    )
    assert await _fetch_character_ography_data(f"{_LUFFY_URL}/anime") is None


async def test_fetch_ography_data_empty_html_returns_empty_list(mocker) -> None:
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )
    mocker.patch(
        "zendriver.start",
        new_callable=AsyncMock,
        return_value=mocker.AsyncMock(),
    )
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler._fetch_page_html",
        new_callable=AsyncMock,
        return_value="<html><body></body></html>",
    )
    assert await _fetch_character_ography_data(f"{_LUFFY_URL}/anime") == []


async def test_fetch_ography_data_valid_returns_list(mocker, luffy_anime_ography_html) -> None:
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )
    mocker.patch(
        "zendriver.start",
        new_callable=AsyncMock,
        return_value=mocker.AsyncMock(),
    )
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler._fetch_page_html",
        new_callable=AsyncMock,
        return_value=luffy_anime_ography_html,
    )
    result = await _fetch_character_ography_data(f"{_LUFFY_URL}/anime")
    assert result is not None
    assert len(result) == 49
    assert all(e["url"].startswith("https://") for e in result)


# =============================================================================
# _extract_ography_from_html
# =============================================================================


def test_extract_ography_from_html_empty_returns_empty_list() -> None:
    assert _extract_ography_from_html("<html><body></body></html>") == []


def test_extract_ography_from_html_no_covers_list_returns_empty_list() -> None:
    assert _extract_ography_from_html("<html><body><p>nothing here</p></body></html>") == []


def test_extract_ography_from_html_valid_returns_absolute_list(luffy_anime_ography_html) -> None:
    entries = _extract_ography_from_html(luffy_anime_ography_html)
    assert entries is not None
    assert len(entries) == 49
    assert all(e["url"].startswith("https://") for e in entries)
    assert all(e["title"] for e in entries)


def test_extract_ography_from_html_multiple_entries(luffy_manga_ography_html) -> None:
    entries = _extract_ography_from_html(luffy_manga_ography_html)
    assert entries is not None
    assert len(entries) == 10
    assert all(e["url"].startswith("https://") for e in entries)


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
    crawler = AniSearchCharacterCrawler(NullRepository())
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
    crawler = AniSearchCharacterCrawler(NullRepository())
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
        "enrichment.sources.anisearch.anisearch_character_crawler._fetch_character_ography_data.cache_batch_get",
        new_callable=AsyncMock,
        return_value=([None], []),  # all cached, no misses → no browser init
    )
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler._fetch_ography",
        new_callable=AsyncMock,
        side_effect=[None, None],
    )
    results = await fetch_anisearch_characters(refs)
    assert len(results) == 1
    assert results[0] is not None
    assert results[0]["name"] == "Monkey D. Luffy"


async def test_fetch_characters_cached_detail_ography_miss_starts_browser(
    mocker, luffy_char_processed
) -> None:
    refs = [{"url": _LUFFY_URL, "role": "Main Character"}]
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler._fetch_anisearch_character_data.cache_batch_get",
        new_callable=AsyncMock,
        return_value=([luffy_char_processed], []),
    )
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler._fetch_character_ography_data.cache_batch_get",
        new_callable=AsyncMock,
        return_value=([None], [0]),
    )
    browser_mock = mocker.AsyncMock()
    mocker.patch("zendriver.start", new_callable=AsyncMock, return_value=browser_mock)
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler._fetch_ography",
        new_callable=AsyncMock,
        side_effect=[None, None],
    )
    results = await fetch_anisearch_characters(refs)
    assert len(results) == 1
    assert results[0] is not None
    assert results[0]["name"] == "Monkey D. Luffy"


async def test_fetch_anisearch_characters_writes_output_path(
    mocker, luffy_char_processed, tmp_path
) -> None:
    import json

    refs = [{"url": _LUFFY_URL, "role": "Main Character"}]
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler._fetch_anisearch_character_data.cache_batch_get",
        new_callable=AsyncMock,
        return_value=([luffy_char_processed], []),
    )
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler._fetch_character_ography_data.cache_batch_get",
        new_callable=AsyncMock,
        return_value=([None], []),  # all cached, no misses → no browser init
    )
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler._fetch_ography",
        new_callable=AsyncMock,
        side_effect=[None, None],
    )
    out = str(tmp_path / "chars.jsonl")
    await fetch_anisearch_characters(refs, output_path=out)
    lines = (tmp_path / "chars.jsonl").read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["name"] == "Monkey D. Luffy"


async def test_fetch_anisearch_characters_uncached_crawl_succeeds(
    mocker, luffy_char_raw
) -> None:
    refs = [{"url": _LUFFY_URL, "role": "Main Character"}]
    html = luffy_char_raw.get("_html", "")
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler._fetch_anisearch_character_data.cache_batch_get",
        new_callable=AsyncMock,
        return_value=([None], [0]),
    )
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler._fetch_anisearch_character_data.cache_batch_set",
        new_callable=AsyncMock,
    )
    browser_mock = mocker.AsyncMock()
    browser_mock.stop.side_effect = Exception("stop failed")
    mocker.patch("zendriver.start", new_callable=AsyncMock, return_value=browser_mock)
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler._fetch_page_html",
        new_callable=AsyncMock,
        return_value=html,
    )
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler._fetch_ography",
        new_callable=AsyncMock,
        side_effect=[None, None],
    )
    results = await fetch_anisearch_characters(refs)
    assert len(results) == 1
    assert results[0] is not None
    assert results[0]["name"] == "Monkey D. Luffy"


async def test_uncached_crawl_fetch_error_stays_none(mocker) -> None:
    # _fetch_page_html returns None → cache written as None, character stays None
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
        "zendriver.start",
        new_callable=AsyncMock,
        return_value=mocker.AsyncMock(),
    )
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler._fetch_page_html",
        new_callable=AsyncMock,
        return_value=None,
    )
    results = await fetch_anisearch_characters(refs)
    assert results == [None]


async def test_uncached_crawl_unparseable_html_stays_none(mocker) -> None:
    # empty HTML → _extract_character_from_html returns None → character stays None
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
        "zendriver.start",
        new_callable=AsyncMock,
        return_value=mocker.AsyncMock(),
    )
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler._fetch_page_html",
        new_callable=AsyncMock,
        return_value="",
    )
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler._fetch_ography",
        new_callable=AsyncMock,
        side_effect=[None, None],
    )
    results = await fetch_anisearch_characters(refs)
    assert results == [None]


# =============================================================================
# _fetch_ography
# =============================================================================


async def test_fetch_ography_no_browser_delegates_to_cached_fn(mocker) -> None:
    # When browser=None, misses are fetched via _fetch_character_ography_data.
    # Use a single mock object so cache_batch_get and __call__ share the same reference.
    url = f"{_LUFFY_URL}/anime"
    ography_entry = [{"url": "https://www.anisearch.com/anime/2227,one-piece", "title": "One Piece"}]
    mock_fn = AsyncMock(return_value=ography_entry)
    mock_fn.cache_batch_get = AsyncMock(return_value=([None], [0]))
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler._fetch_character_ography_data",
        mock_fn,
    )
    result = await _fetch_ography(url, browser=None)
    assert result is not None
    assert result[0]["title"] == "One Piece"


async def test_fetch_ography_with_browser_uses_fetch_page_html(mocker, luffy_anime_ography_html) -> None:
    # When browser is provided, misses are navigated via _fetch_page_html
    url = f"{_LUFFY_URL}/anime"
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
        "enrichment.sources.anisearch.anisearch_character_crawler._fetch_page_html",
        new_callable=AsyncMock,
        return_value=luffy_anime_ography_html,
    )
    mock_browser = mocker.AsyncMock()
    result = await _fetch_ography(url, browser=mock_browser)
    assert result is not None
    assert len(result) == 49
    assert all(e["url"].startswith("https://") for e in result)


async def test_fetch_ography_cached_returns_directly(mocker) -> None:
    url = f"{_LUFFY_URL}/anime"
    cached = [{"url": "https://www.anisearch.com/anime/2227,one-piece", "title": "One Piece"}]
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler._fetch_character_ography_data.cache_batch_get",
        new_callable=AsyncMock,
        return_value=([cached], []),
    )
    result = await _fetch_ography(url, browser=None)
    assert result == cached


# =============================================================================
# AniSearchCharacterCrawler
# =============================================================================


def test_crawler_normalize_identifier_passthrough() -> None:
    crawler = AniSearchCharacterCrawler(NullRepository())
    assert crawler.normalize_identifier(_LUFFY_URL) == _LUFFY_URL
    assert crawler.get_extraction_schema() == {"xpaths": _XPATHS}


async def test_crawler_fetch_raw_data_delegates(mocker, luffy_char_processed) -> None:
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_crawler._fetch_anisearch_character_data",
        new_callable=AsyncMock,
        return_value=luffy_char_processed,
    )
    crawler = AniSearchCharacterCrawler(NullRepository())
    result = await crawler.fetch_raw_data(_LUFFY_URL)
    assert result is luffy_char_processed


def test_crawler_build_source_model_with_role(luffy_char_processed) -> None:
    crawler = AniSearchCharacterCrawler(NullRepository(), role="Main Character")
    char = crawler.build_source_model(luffy_char_processed, _LUFFY_URL)
    assert char.name == "Monkey D. Luffy"
    assert char.role == "Main Character"


def test_crawler_map_to_canonical(luffy_char_processed) -> None:
    crawler = AniSearchCharacterCrawler(NullRepository())
    char = _build_character_from_raw(
        luffy_char_processed, luffy_char_processed.get("_html", ""), _LUFFY_URL
    )
    result = crawler.map_to_canonical(char)
    assert result["name"] == "Monkey D. Luffy"
    assert result["sources"] == [_LUFFY_URL]


# =============================================================================
# character_from_anisearch (mapper)
# =============================================================================


def test_character_from_anisearch_happy_path(luffy_char_processed) -> None:
    char = _build_character_from_raw(
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
    char = _build_character_from_raw(
        luffy_char_processed,
        luffy_char_processed.get("_html", ""),
        _LUFFY_URL,
        role="Main Character",
    )
    result = character_from_anisearch(char)
    assert "MAIN" in result.get("roles", [])
