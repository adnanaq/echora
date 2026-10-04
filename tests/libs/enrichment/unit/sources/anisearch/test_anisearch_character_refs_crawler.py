"""Unit tests for AniSearch character refs crawler.

Baseline tests use real XPath extraction output captured from:
- https://www.anisearch.com/anime/2227,one-piece/characters (2026-04-23)

Edge-case branches use inline overrides on top of the real fixture dict.
No network calls are made.
"""

from unittest.mock import AsyncMock, patch

import pytest
from enrichment.sources.anisearch import (
    anisearch_character_refs_crawler as crawler_module,
)
from enrichment.sources.anisearch.anisearch_character_refs_crawler import (
    _ANISEARCH_BASE_URL,
    _XPATHS,
    _absolutize,
    _extract_refs_from_html,
    _fetch_anisearch_character_refs_data,
    _normalize_characters_page_url,
    _post_process_refs,
    fetch_anisearch_character_refs,
)
from enrichment.sources.base.polite_http import FetchedPage

pytestmark = pytest.mark.asyncio

_ONE_PIECE_CHARS_URL = "https://www.anisearch.com/anime/2227,one-piece/characters"


# =============================================================================
# _XPATHS
# =============================================================================


def test_xpaths_covers_all_sections() -> None:
    assert set(_XPATHS) == {"chara1", "chara2", "chara3", "chara4", "chara5", "chara50"}


def test_xpaths_each_targets_character_href() -> None:
    for section_id, xpath in _XPATHS.items():
        assert f"@id='{section_id}'" in xpath
        assert "character/" in xpath
        assert xpath.endswith("/@href")


# =============================================================================
# _extract_refs_from_html
# =============================================================================


def test_extract_refs_returns_none_on_empty() -> None:
    assert _extract_refs_from_html("") is None


def test_extract_refs_returns_section_dict_on_valid_html() -> None:
    html = """
    <html><body>
      <section id="chara1">
        <a href="character/4852,monkey-d-luffy">Luffy</a>
        <a href="character/1234,zoro">Zoro</a>
      </section>
      <section id="chara2">
        <a href="character/5000,nami">Nami</a>
      </section>
    </body></html>
    """
    raw = _extract_refs_from_html(html)
    assert raw is not None
    assert raw["chara1"] == ["character/4852,monkey-d-luffy", "character/1234,zoro"]
    assert raw["chara2"] == ["character/5000,nami"]
    assert raw["chara3"] == []


def test_extract_refs_empty_sections_return_empty_list() -> None:
    raw = _extract_refs_from_html("<html><body></body></html>")
    assert raw is not None
    assert all(v == [] for v in raw.values())


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
    refs = _post_process_refs(one_piece_refs_raw, _ONE_PIECE_CHARS_URL)
    assert refs
    for ref in refs:
        assert "url" in ref and "role" in ref
        assert ref["url"].startswith("https://"), f"Relative URL: {ref['url']}"
        # Carried so the mapper can attach `role` to the right ography entry.
        assert ref["anime_url"] == _ONE_PIECE_CHARS_URL


def test_post_process_refs_role_mapping(one_piece_refs_raw) -> None:
    refs = _post_process_refs(one_piece_refs_raw, _ONE_PIECE_CHARS_URL)
    luffy = next(r for r in refs if "monkey-d-luffy" in r["url"])
    assert luffy["role"] == "Main Character"
    chara2_count = sum(1 for r in refs if r["role"] == "Secondary Character")
    assert chara2_count == len(one_piece_refs_raw["chara2"])


def test_post_process_refs_count_invariants(one_piece_refs_raw) -> None:
    refs = _post_process_refs(one_piece_refs_raw, _ONE_PIECE_CHARS_URL)
    urls = [r["url"] for r in refs]
    assert len(urls) == len(set(urls))
    assert len(refs) <= sum(len(v) for v in one_piece_refs_raw.values())


def test_post_process_refs_empty_sections_skipped() -> None:
    raw = {
        "chara1": ["character/1,test"],
        "chara2": [],
        "chara3": [],
        "chara4": [],
        "chara5": [],
        "chara50": [],
    }
    refs = _post_process_refs(raw, _ONE_PIECE_CHARS_URL)
    assert len(refs) == 1


def test_post_process_refs_missing_url_skipped() -> None:
    raw = {
        "chara1": ["", "character/1,test"],
        "chara2": [],
        "chara3": [],
        "chara4": [],
        "chara5": [],
        "chara50": [],
    }
    refs = _post_process_refs(raw, _ONE_PIECE_CHARS_URL)
    assert len(refs) == 1


# =============================================================================
# _fetch_anisearch_character_refs_data (async, mocked)
# =============================================================================


def _cast_page(html: str | None) -> FetchedPage | None:
    return None if html is None else FetchedPage(url=_ONE_PIECE_CHARS_URL, html=html)


@pytest.mark.usefixtures("cache_off")
async def test_fetch_anisearch_character_refs_data_page_unreadable_returns_none() -> (
    None
):
    with patch.object(
        crawler_module, "fetch_anisearch_page", autospec=True, return_value=None
    ):
        assert await _fetch_anisearch_character_refs_data(_ONE_PIECE_CHARS_URL) is None


@pytest.mark.usefixtures("cache_off")
async def test_fetch_anisearch_character_refs_data_real_refs_returns_absolute_urls(
    one_piece_refs_raw: dict,
) -> None:
    with (
        patch.object(
            crawler_module,
            "fetch_anisearch_page",
            autospec=True,
            return_value=_cast_page("<html></html>"),
        ),
        patch.object(
            crawler_module,
            "_extract_refs_from_html",
            autospec=True,
            return_value=one_piece_refs_raw,
        ),
    ):
        refs = await _fetch_anisearch_character_refs_data(_ONE_PIECE_CHARS_URL)

    assert refs == _post_process_refs(one_piece_refs_raw, _ONE_PIECE_CHARS_URL)
    assert refs[0]["url"].startswith("https://")


@pytest.mark.usefixtures("cache_off")
async def test_fetch_anisearch_character_refs_data_extraction_fails_returns_none() -> (
    None
):
    with (
        patch.object(
            crawler_module,
            "fetch_anisearch_page",
            autospec=True,
            return_value=_cast_page("<html></html>"),
        ),
        patch.object(
            crawler_module, "_extract_refs_from_html", autospec=True, return_value=None
        ),
    ):
        assert await _fetch_anisearch_character_refs_data(_ONE_PIECE_CHARS_URL) is None


@pytest.mark.usefixtures("cache_off")
async def test_fetch_anisearch_character_refs_data_empty_page_returns_none() -> None:
    with patch.object(
        crawler_module,
        "fetch_anisearch_page",
        autospec=True,
        return_value=_cast_page(""),
    ):
        assert await _fetch_anisearch_character_refs_data(_ONE_PIECE_CHARS_URL) is None


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
    expected = _post_process_refs(one_piece_refs_raw, _ONE_PIECE_CHARS_URL)
    mocker.patch(
        "enrichment.sources.anisearch.anisearch_character_refs_crawler._fetch_anisearch_character_refs_data",
        new_callable=AsyncMock,
        return_value=expected,
    )
    result = await fetch_anisearch_character_refs("2227,one-piece")
    assert result == expected
