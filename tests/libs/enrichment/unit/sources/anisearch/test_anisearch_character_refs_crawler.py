"""Unit tests for AniSearch character refs crawler.

Baseline tests use real XPath extraction output captured from:
- https://www.anisearch.com/anime/2227,one-piece/characters (2026-04-23)

Edge-case branches use inline overrides on top of the real fixture dict.
No network calls are made.
"""

import json
from unittest.mock import AsyncMock

import pytest
from enrichment.sources.anisearch.anisearch_character_refs_crawler import (
    _ANISEARCH_BASE_URL,
    _absolutize,
    _fetch_anisearch_character_refs_data,
    _get_character_refs_schema,
    _normalize_characters_page_url,
    _post_process_refs,
    fetch_anisearch_character_refs,
)

pytestmark = pytest.mark.asyncio

_ONE_PIECE_CHARS_URL = "https://www.anisearch.com/anime/2227,one-piece/characters"


# =============================================================================
# _get_character_refs_schema
# =============================================================================


def test_refs_schema_structure() -> None:
    schema = _get_character_refs_schema()
    assert {f["name"] for f in schema["fields"]} == {
        "chara1",
        "chara2",
        "chara3",
        "chara4",
        "chara5",
        "chara50",
    }
    for field in schema["fields"]:
        assert field["type"] == "list"
        assert "character/" in field["selector"]
    url_subfield = schema["fields"][0]["fields"][0]
    assert url_subfield["name"] == "url"
    assert url_subfield.get("attribute") == "href"


# =============================================================================
# _normalize_characters_page_url
# =============================================================================


def test_normalize_id_slug_builds_full_url() -> None:
    assert _normalize_characters_page_url("2227,one-piece") == _ONE_PIECE_CHARS_URL


def test_normalize_bare_id_builds_full_url() -> None:
    assert _normalize_characters_page_url("2227") == (
        "https://www.anisearch.com/anime/2227/characters"
    )


def test_normalize_already_full_url_passthrough() -> None:
    assert _normalize_characters_page_url(_ONE_PIECE_CHARS_URL) == _ONE_PIECE_CHARS_URL


def test_normalize_base_url_without_characters_appends_it() -> None:
    url = "https://www.anisearch.com/anime/2227,one-piece"
    assert _normalize_characters_page_url(url) == _ONE_PIECE_CHARS_URL


def test_normalize_trailing_slash_stripped() -> None:
    assert _normalize_characters_page_url("2227,one-piece/") == _ONE_PIECE_CHARS_URL


# =============================================================================
# _absolutize
# =============================================================================


def test_absolutize_non_absolute_inputs() -> None:
    assert _absolutize("character/4852,monkey-d-luffy") == (
        f"{_ANISEARCH_BASE_URL}/character/4852,monkey-d-luffy"
    )
    assert _absolutize("/character/4852,monkey-d-luffy") == (
        f"{_ANISEARCH_BASE_URL}/character/4852,monkey-d-luffy"
    )


def test_absolutize_already_absolute_passthrough() -> None:
    url = f"{_ANISEARCH_BASE_URL}/character/4852,monkey-d-luffy"
    assert _absolutize(url) == url


# =============================================================================
# _post_process_refs — real fixture
# =============================================================================


def test_post_process_refs_returns_url_role_dicts(one_piece_refs_raw) -> None:
    refs = _post_process_refs(one_piece_refs_raw)
    assert refs
    for ref in refs:
        assert "url" in ref and "role" in ref
        assert ref["url"].startswith("https://"), f"Relative URL: {ref['url']}"


def test_post_process_refs_role_mapping(one_piece_refs_raw) -> None:
    refs = _post_process_refs(one_piece_refs_raw)
    luffy = next(r for r in refs if "monkey-d-luffy" in r["url"])
    assert luffy["role"] == "Main Character"
    chara2_count = sum(1 for r in refs if r["role"] == "Secondary Character")
    assert chara2_count == len(one_piece_refs_raw["chara2"])


def test_post_process_refs_count_invariants(one_piece_refs_raw) -> None:
    refs = _post_process_refs(one_piece_refs_raw)
    urls = [r["url"] for r in refs]
    assert len(urls) == len(set(urls))
    assert len(refs) <= sum(len(v) for v in one_piece_refs_raw.values())


def test_post_process_refs_empty_sections_skipped() -> None:
    raw = {
        "chara1": [{"url": "character/1,test"}],
        "chara2": [],
        "chara3": [],
        "chara4": [],
        "chara5": [],
        "chara50": [],
    }
    refs = _post_process_refs(raw)
    assert len(refs) == 1


def test_post_process_refs_missing_url_skipped() -> None:
    raw = {
        "chara1": [{"url": ""}, {"url": "character/1,test"}],
        "chara2": [],
        "chara3": [],
        "chara4": [],
        "chara5": [],
        "chara50": [],
    }
    refs = _post_process_refs(raw)
    assert len(refs) == 1


# =============================================================================
# _fetch_anisearch_character_refs_data (async, mocked)
# =============================================================================


async def test_fetch_refs_none_result_returns_none(mocker) -> None:
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_refs_crawler.crawl_single_url",
        new_callable=AsyncMock,
        return_value=None,
    )
    assert await _fetch_anisearch_character_refs_data(_ONE_PIECE_CHARS_URL) is None


async def test_fetch_refs_404_returns_none(mocker) -> None:
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_refs_crawler.crawl_single_url",
        new_callable=AsyncMock,
        return_value={"status_code": 404, "extracted_content": "[]"},
    )
    assert await _fetch_anisearch_character_refs_data(_ONE_PIECE_CHARS_URL) is None


async def test_fetch_refs_empty_extracted_content_returns_none(mocker) -> None:
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_refs_crawler.crawl_single_url",
        new_callable=AsyncMock,
        return_value={"status_code": 200, "extracted_content": "[]"},
    )
    assert await _fetch_anisearch_character_refs_data(_ONE_PIECE_CHARS_URL) is None


async def test_fetch_refs_redirect_logs_debug_continues(
    mocker, one_piece_refs_raw
) -> None:
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_refs_crawler.crawl_single_url",
        new_callable=AsyncMock,
        return_value={
            "status_code": 301,
            "extracted_content": json.dumps([one_piece_refs_raw]),
        },
    )
    refs = await _fetch_anisearch_character_refs_data(_ONE_PIECE_CHARS_URL)
    assert refs is not None and len(refs) > 0


async def test_fetch_refs_real_fixture_returns_refs(mocker, one_piece_refs_raw) -> None:
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_refs_crawler.crawl_single_url",
        new_callable=AsyncMock,
        return_value={
            "status_code": 200,
            "extracted_content": json.dumps([one_piece_refs_raw]),
        },
    )
    refs = await _fetch_anisearch_character_refs_data(_ONE_PIECE_CHARS_URL)
    assert refs is not None
    assert len(refs) > 0
    assert refs[0]["url"].startswith("https://")


# =============================================================================
# fetch_anisearch_character_refs (public API, mocked)
# =============================================================================


async def test_fetch_character_refs_returns_empty_on_failure(mocker) -> None:
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_refs_crawler._fetch_anisearch_character_refs_data",
        new_callable=AsyncMock,
        return_value=None,
    )
    assert await fetch_anisearch_character_refs("2227,one-piece") == []


async def test_fetch_character_refs_returns_list_on_success(
    mocker, one_piece_refs_raw
) -> None:
    expected = _post_process_refs(one_piece_refs_raw)
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_refs_crawler._fetch_anisearch_character_refs_data",
        new_callable=AsyncMock,
        return_value=expected,
    )
    result = await fetch_anisearch_character_refs("2227,one-piece")
    assert result == expected
