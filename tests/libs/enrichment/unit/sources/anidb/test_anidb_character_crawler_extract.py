from unittest.mock import patch

import pytest
from enrichment.sources.anidb.anidb_character_crawler import (
    _extract_from_html,
    _has_character_data,
    _is_blocked,
)

_FULL_HTML = """<html><head><meta charset="utf-8"></head><body>
<table><tr class="mainname"><td><span itemprop="name">Monkey D. Luffy</span></td></tr></table>
<div id="tab_1_pane">
  <table>
    <tr class="official verified yes">
      <td><label itemprop="alternateName">モンキー・D・ルフィ</label></td>
    </tr>
    <tr class="abilities"><td><span class="tagname">Combat</span></td></tr>
    <tr class="abilities supernatural"><td><span class="tagname">Elasticity</span></td></tr>
    <tr class="looks"><td><span class="tagname">Scar</span></td></tr>
    <tr class="personality"><td><span class="tagname">Carefree</span></td></tr>
    <tr class="role"><td><span class="tagname">Pirate Captain</span></td></tr>
  </table>
</div>
<span itemprop="gender">Male</span>
<div class="desc" itemprop="description">The future King of the Pirates.</div>
<table id="animelist_474" class="animelist">
  <thead><tr><th class="anime">Anime</th><th class="type">Type</th></tr></thead>
  <tbody>
    <tr>
      <td class="name anime"><a href="/anime/69">One Piece</a></td>
      <td class="type">main character in</td>
    </tr>
    <tr>
      <td class="name anime"><a href="https://anidb.net/anime/100">Movie</a></td>
      <td class="type">appears in</td>
    </tr>
    <tr><td class="other">no-link-row</td><td class="type">something</td></tr>
  </tbody>
</table>
<div id="tab_2_pane">
  <table>
    <tr class="official"><td><label itemprop="alternateName">Rufi</label></td></tr>
    <tr class="nick"><td class="value">Straw Hat</td></tr>
  </table>
</div>
</body></html>"""

_EMPTY_HTML = "<html><body><p>Nothing here</p></body></html>"


def test_extract_full_html_all_fields() -> None:
    page = _extract_from_html(_FULL_HTML)
    assert page is not None
    assert page.name_main == "Monkey D. Luffy"
    assert page.name_kanji == "モンキー・D・ルフィ"
    assert page.gender == "Male"
    assert page.description == "The future King of the Pirates."
    assert page.abilities == ["Combat"]
    assert page.supernatural_abilities == ["Elasticity"]
    assert page.looks == ["Scar"]
    assert page.personality == ["Carefree"]
    assert page.role == ["Pirate Captain"]
    assert page.official_names == ["Rufi"]
    assert page.nicknames == ["Straw Hat"]
    assert len(page.animeography) == 2


def test_extract_animeography_relative_url() -> None:
    page = _extract_from_html(_FULL_HTML)
    assert page is not None
    entry = next(e for e in page.animeography if "One Piece" in e["title"])
    assert entry["url"] == "https://anidb.net/anime/69"
    assert entry["role"] == "main character in"


def test_extract_animeography_absolute_url() -> None:
    page = _extract_from_html(_FULL_HTML)
    assert page is not None
    entry = next(e for e in page.animeography if "Movie" in e["title"])
    assert entry["url"] == "https://anidb.net/anime/100"
    assert entry["role"] == "appears in"


def test_extract_animeography_row_without_link_skipped() -> None:
    page = _extract_from_html(_FULL_HTML)
    assert page is not None
    assert not any("no-link" in e["title"] for e in page.animeography)


def test_extract_animeography_empty_title_skipped() -> None:
    html = """<html><body>
    <div id="tab_1_pane"><span itemprop="name">X</span></div>
    <table>
      <thead><tr><th class="anime">A</th></tr></thead>
      <tbody>
        <tr>
          <td class="name anime"><a href="/anime/1"></a></td>
          <td class="type">main character in</td>
        </tr>
      </tbody>
    </table>
    </body></html>"""
    page = _extract_from_html(html)
    assert page is not None
    assert page.animeography == []


def test_extract_animeography_no_role_cell() -> None:
    html = """<html><body>
    <div id="tab_1_pane"><span itemprop="name">X</span></div>
    <table class="animelist">
      <thead><tr><th class="anime">A</th></tr></thead>
      <tbody>
        <tr><td class="name anime"><a href="/anime/1">Show</a></td></tr>
      </tbody>
    </table>
    </body></html>"""
    page = _extract_from_html(html)
    assert page is not None
    assert page.animeography[0]["role"] == ""


def test_extract_empty_html_returns_empty_page() -> None:
    page = _extract_from_html(_EMPTY_HTML)
    assert page is not None
    assert page.name_main is None
    assert page.abilities == []
    assert page.animeography == []


def test_extract_description_whitespace_normalised() -> None:
    html = '<html><body><div class="desc" itemprop="description">  hello   world  </div></body></html>'
    page = _extract_from_html(html)
    assert page is not None
    assert page.description == "hello world"


def test_extract_empty_description_div_returns_none() -> None:
    html = (
        '<html><body><div class="desc" itemprop="description">   </div></body></html>'
    )
    page = _extract_from_html(html)
    assert page is not None
    assert page.description is None


def test_extract_lxml_parse_error_returns_none() -> None:
    with patch("lxml.etree.fromstring", side_effect=Exception("bad parse")):
        assert _extract_from_html("<html/>") is None


def test_extract_from_html_class_token_boundary_avoids_substring_collisions() -> None:
    html = """<html><head><meta charset="utf-8"></head><body>
    <div id="tab_1_pane">
      <table>
        <tr class="image"><td>NOT_AN_AGE</td></tr>
        <tr class="age range"><td>adolescent</td></tr>
        <tr class="line_height"><td>NOT_HEIGHT</td></tr>
        <tr class="g_odd birthdate"><td><time>05.05.1999</time></td></tr>
        <tr class="age"><td><span>19</span></td></tr>
        <tr class="bloodtype"><td><span>F</span></td></tr>
        <tr class="height"><td><span itemprop="height">174 cm</span></td></tr>
      </table>
    </div>
    </body></html>"""
    page = _extract_from_html(html)
    assert page is not None
    assert page.date_of_birth == "05.05.1999"
    assert page.age == "19"
    assert page.blood_type == "F"
    assert page.height == "174 cm"


def test_extract_from_html_resolves_protocol_and_bare_relative_urls() -> None:
    html = """<html><head><meta charset="utf-8"></head><body>
    <div id="tab_1_pane">
      <tr class="official verified yes">
        <td><label itemprop="alternateName">モンキー・Ｄ・ルフィ</label></td>
      </tr>
    </div>
    <table class="seiyuulist">
      <tbody>
        <tr>
          <td class="name creator"><a href="//anidb.net/creator/456">Protocol Relative</a></td>
          <td class="name anime"><a href="anime/77">Bare Anime</a></td>
        </tr>
        <tr>
          <td class="name creator"><a href="creator/123">Bare Creator</a></td>
          <td class="name anime"><a href="//anidb.net/anime/88">Protocol Anime</a></td>
        </tr>
        <tr>
          <td class="name creator"><a href="/creator/999">   </a></td>
          <td class="name anime"><a href="/anime/99">Ignored Anime</a></td>
        </tr>
      </tbody>
    </table>
    <table class="animelist">
      <tbody>
        <tr>
          <td class="type relation">main character in</td>
          <td class="name anime"><a href="anime/77">Bare Anime</a></td>
          <td class="type anime">TV Series</td>
        </tr>
        <tr>
          <td class="type relation">appears in</td>
          <td class="name anime"><a href="//anidb.net/anime/88">Protocol Anime</a></td>
          <td class="type anime">Movie</td>
        </tr>
      </tbody>
    </table>
    </body></html>"""
    page = _extract_from_html(html)
    assert page is not None
    assert page.name_kanji == "モンキー・Ｄ・ルフィ"
    assert page.seiyuu == [
        {
            "name": "Protocol Relative",
            "url": "https://anidb.net/creator/456",
            "anime_url": "https://anidb.net/anime/77",
        },
        {
            "name": "Bare Creator",
            "url": "https://anidb.net/creator/123",
            "anime_url": "https://anidb.net/anime/88",
        },
    ]
    assert page.animeography == [
        {
            "title": "Bare Anime",
            "role": "main character in",
            "url": "https://anidb.net/anime/77",
        },
        {
            "title": "Protocol Anime",
            "role": "appears in",
            "url": "https://anidb.net/anime/88",
        },
    ]


@pytest.mark.parametrize(
    "marker",
    [
        "Just a moment",
        "cf-browser-verification",
        "cf-challenge",
        "Attention Required",
        "<title>AniDB AntiLeech",
    ],
)
def test_is_blocked_each_block_marker_returns_true(marker: str) -> None:
    assert _is_blocked(f"<html><body>{marker}</body></html>") is True


def test_is_blocked_clean_html_returns_false() -> None:
    assert _is_blocked("<html><body>Normal page</body></html>") is False


def test_has_character_data_tab_pane() -> None:
    assert _has_character_data('<div id="tab_1_pane">content</div>') is True


def test_has_character_data_itemprop_name_alone_is_not_enough() -> None:
    assert _has_character_data('<span itemprop="name">Luffy</span>') is False


def test_has_character_data_neither() -> None:
    assert _has_character_data("<html><body>empty</body></html>") is False
