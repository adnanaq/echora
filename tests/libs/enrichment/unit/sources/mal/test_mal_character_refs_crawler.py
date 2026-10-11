import re
import time

import pytest
from enrichment.sources.mal import mal_character_refs_crawler
from enrichment.sources.mal.mal_character_refs_crawler import (
    _CHARACTER_LINK_IN_TABLE_XPATH,
    _CHARACTER_TABLE_XPATH,
    _extract_character_urls,
    _fetch_characters_page_html,
    _fetch_mal_characters_data,
    fetch_mal_character_refs,
)
from lxml import etree

pytestmark = pytest.mark.asyncio

_ONE_PIECE_CHARS_URL = "https://myanimelist.net/anime/21/One_Piece/characters"
_EXPECTED_URLS = [
    "https://myanimelist.net/character/5627/Brook",
    "https://myanimelist.net/character/40/Luffy_Monkey_D",
    "https://myanimelist.net/character/23021/Aisa",
    "https://myanimelist.net/character/153828/Bariete",
    "https://myanimelist.net/character/63333/Shivercalero_Donquixote",
]
_PAGE_WITHOUT_TABLES = "<html><body><p>nothing</p></body></html>"
_TABLE_WITHOUT_CHARACTERS = (
    '<html><body><table class="js-anime-character-table"><tr><td>x</td></tr>'
    "</table></body></html>"
)


@pytest.fixture
def characters_browser(open_browser, browser_serving):
    def serve(html: str):
        return open_browser(
            mal_character_refs_crawler,
            browser_serving({_ONE_PIECE_CHARS_URL: html}),
        )

    return serve


def _cast_list(hrefs: list[str]) -> str:
    tables = "\n".join(
        f'<table class="js-anime-character-table"><tr>'
        f'<td><a href="https://myanimelist.net/{href}">Name</a></td>'
        f"</tr></table>"
        for href in hrefs
    )
    return f"<html><body>{tables}</body></html>"


def _long_cast_list(fixture_html: str, characters: int) -> str:
    tree = etree.fromstring(fixture_html.encode(), etree.HTMLParser(encoding="utf-8"))
    tables = [
        etree.tostring(table, encoding="unicode", method="html")
        for table in tree.xpath(_CHARACTER_TABLE_XPATH)
    ]
    body = "".join(
        re.sub(
            r"/character/\d+/", f"/character/{number}/", tables[number % len(tables)]
        )
        for number in range(characters)
    )
    return f"<html><body>{body}</body></html>"


def test_character_xpaths_target_character_links_in_cast_tables() -> None:
    assert "js-anime-character-table" in _CHARACTER_TABLE_XPATH
    assert _CHARACTER_LINK_IN_TABLE_XPATH.startswith(".//")
    assert "/character/" in _CHARACTER_LINK_IN_TABLE_XPATH
    assert _CHARACTER_LINK_IN_TABLE_XPATH.endswith("/@href")


def test_extract_character_urls_one_piece_page_returns_cast_in_order(
    mal_char_refs_html,
) -> None:
    assert _extract_character_urls(mal_char_refs_html) == _EXPECTED_URLS


@pytest.mark.parametrize(
    "html",
    [
        "",
        _PAGE_WITHOUT_TABLES,
        '<?xml version="1.0" encoding="utf-8"?><html><body><p>x</p></body></html>',
    ],
)
def test_extract_character_urls_page_without_cast_returns_empty_list(
    html: str,
) -> None:
    assert _extract_character_urls(html) == []


def test_extract_character_urls_repeated_character_returns_it_once() -> None:
    html = _cast_list(["character/40/Luffy", "character/40/Luffy"])
    assert _extract_character_urls(html) == [
        "https://myanimelist.net/character/40/Luffy"
    ]


def test_extract_character_urls_several_characters_returns_page_order() -> None:
    html = _cast_list(["character/1/A", "character/2/B", "character/3/C"])

    assert _extract_character_urls(html) == [
        "https://myanimelist.net/character/1/A",
        "https://myanimelist.net/character/2/B",
        "https://myanimelist.net/character/3/C",
    ]


def test_extract_character_urls_long_cast_returns_every_character_in_order(
    mal_char_refs_html,
) -> None:
    urls = _extract_character_urls(_long_cast_list(mal_char_refs_html, 1500))

    ids = [int(url.split("/character/")[1].split("/")[0]) for url in urls]
    assert ids == list(range(1500))


def test_extract_character_urls_long_cast_finishes_within_five_seconds(
    mal_char_refs_html,
) -> None:
    page = _long_cast_list(mal_char_refs_html, 1500)

    start = time.perf_counter()
    _extract_character_urls(page)

    assert time.perf_counter() - start < 5


def test_extract_character_urls_anime_link_beside_character_returns_character() -> None:
    html = """<html><body>
    <table class="js-anime-character-table">
      <tr>
        <td>
          <a href="https://myanimelist.net/anime/21/One_Piece">Anime</a>
          <a href="https://myanimelist.net/character/40/Luffy">Luffy</a>
        </td>
      </tr>
    </table>
    </body></html>"""

    assert _extract_character_urls(html) == [
        "https://myanimelist.net/character/40/Luffy"
    ]


async def test_fetch_characters_page_html_characters_page_returns_content(
    characters_browser, mal_char_refs_html
) -> None:
    with characters_browser(mal_char_refs_html):
        assert await _fetch_characters_page_html(_ONE_PIECE_CHARS_URL) == (
            mal_char_refs_html
        )


async def test_fetch_characters_page_html_page_without_cast_table_returns_none(
    characters_browser,
) -> None:
    with characters_browser(_PAGE_WITHOUT_TABLES):
        assert await _fetch_characters_page_html(_ONE_PIECE_CHARS_URL) is None


async def test_fetch_mal_characters_data_characters_page_returns_character_urls(
    characters_browser, mal_char_refs_html
) -> None:
    with characters_browser(mal_char_refs_html):
        assert await _fetch_mal_characters_data(_ONE_PIECE_CHARS_URL) == _EXPECTED_URLS


@pytest.mark.parametrize("html", ["", _TABLE_WITHOUT_CHARACTERS])
async def test_fetch_mal_characters_data_page_without_characters_returns_none(
    characters_browser, html: str
) -> None:
    with characters_browser(html):
        assert await _fetch_mal_characters_data(_ONE_PIECE_CHARS_URL) is None


async def test_fetch_mal_character_refs_characters_page_returns_character_urls(
    characters_browser, mal_char_refs_html
) -> None:
    with characters_browser(mal_char_refs_html):
        assert await fetch_mal_character_refs(_ONE_PIECE_CHARS_URL) == _EXPECTED_URLS


async def test_fetch_mal_character_refs_page_without_characters_returns_empty_list(
    characters_browser,
) -> None:
    with characters_browser(_TABLE_WITHOUT_CHARACTERS):
        assert await fetch_mal_character_refs(_ONE_PIECE_CHARS_URL) == []
