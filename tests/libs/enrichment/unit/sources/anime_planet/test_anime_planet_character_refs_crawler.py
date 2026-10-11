import pytest
from enrichment.sources.anime_planet import anime_planet_character_refs_crawler
from enrichment.sources.anime_planet.anime_planet_character_refs_crawler import (
    _XPATHS,
    _extract_refs_from_html,
    _fetch_refs_data,
    _fetch_refs_html,
    fetch_animeplanet_character_refs,
)

pytestmark = pytest.mark.asyncio

_DANDADAN_URL = "https://www.anime-planet.com/anime/dandadan/characters"
_EXPECTED_FIRST = {"url": "/characters/ken-takakura", "role": ""}
_EXPECTED_COUNT = 25
_PAGE_WITHOUT_CHARACTERS = "<html><body></body></html>"


@pytest.fixture
def characters_page(open_browser, browser_showing):
    def serve(html: str | None):
        return open_browser(anime_planet_character_refs_crawler, browser_showing(html))

    return serve


def test_xpaths_characters_query_targets_character_name_links() -> None:
    assert "/characters/" in _XPATHS["characters"]
    assert "name" in _XPATHS["characters"]


def test_extract_refs_from_html_dandadan_page_returns_every_character(
    ap_char_refs_html: str,
) -> None:
    refs = _extract_refs_from_html(ap_char_refs_html)

    assert refs is not None
    assert len(refs) == _EXPECTED_COUNT
    assert refs[0] == _EXPECTED_FIRST
    assert all(ref["role"] == "" for ref in refs)


@pytest.mark.parametrize(
    "html",
    [
        "",
        "<html><body><p>no character links here</p></body></html>",
        '<?xml version="1.0" encoding="utf-8"?><html><body><p>x</p></body></html>',
    ],
)
def test_extract_refs_from_html_page_without_character_links_returns_none(
    html: str,
) -> None:
    assert _extract_refs_from_html(html) is None


def test_extract_refs_from_html_link_without_address_returns_other_links() -> None:
    html = (
        "<html><body>"
        '<a class="name" href="/characters/luffy">Luffy</a>'
        '<a class="name">No href</a>'
        "</body></html>"
    )

    assert _extract_refs_from_html(html) == [{"url": "/characters/luffy", "role": ""}]


async def test_fetch_refs_html_characters_page_returns_content(
    characters_page, ap_char_refs_html: str
) -> None:
    with characters_page(ap_char_refs_html):
        assert await _fetch_refs_html(_DANDADAN_URL) == ap_char_refs_html


async def test_fetch_refs_html_navigation_failure_returns_none(characters_page) -> None:
    with characters_page(None):
        assert await _fetch_refs_html(_DANDADAN_URL) is None


async def test_fetch_refs_data_characters_page_returns_refs(
    characters_page, ap_char_refs_html: str
) -> None:
    with characters_page(ap_char_refs_html):
        refs = await _fetch_refs_data(_DANDADAN_URL)

    assert refs is not None
    assert (len(refs), refs[0]) == (_EXPECTED_COUNT, _EXPECTED_FIRST)


@pytest.mark.parametrize("html", [None, _PAGE_WITHOUT_CHARACTERS])
async def test_fetch_refs_data_unreadable_page_returns_none(
    characters_page, html: str | None
) -> None:
    with characters_page(html):
        assert await _fetch_refs_data(_DANDADAN_URL) is None


async def test_fetch_animeplanet_character_refs_characters_page_returns_refs(
    characters_page, ap_char_refs_html: str
) -> None:
    with characters_page(ap_char_refs_html):
        refs = await fetch_animeplanet_character_refs(_DANDADAN_URL)

    assert refs[0] == _EXPECTED_FIRST


@pytest.mark.parametrize("html", [None, _PAGE_WITHOUT_CHARACTERS])
async def test_fetch_animeplanet_character_refs_unreadable_page_returns_empty_list(
    characters_page, html: str | None
) -> None:
    with characters_page(html):
        assert await fetch_animeplanet_character_refs(_DANDADAN_URL) == []
