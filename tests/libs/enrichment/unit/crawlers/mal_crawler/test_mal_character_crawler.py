"""Unit tests for mal_character_crawler.py — HTML extraction helpers.

All tests use HTML fixtures that mirror the real MAL character page structure.
No network calls are made.
"""

from enrichment.crawlers.mal_crawler.mal_character_crawler import (
    _extract_bio_data,
    _extract_description,
    _extract_name_and_native,
    _extract_ography,
    _extract_voice_actors,
    _parse_character_raw,
)


# =============================================================================
# _extract_name_and_native
# =============================================================================


def test_extract_name_and_native_with_kanji() -> None:
    name, native = _extract_name_and_native("Monkey D., Luffy (モンキー・D・ルフィ)")
    assert name == "Monkey D., Luffy"
    assert native == "モンキー・D・ルフィ"


def test_extract_name_and_native_no_kanji() -> None:
    name, native = _extract_name_and_native("Roronoa, Zoro")
    assert name == "Roronoa, Zoro"
    assert native is None


# =============================================================================
# _extract_bio_data
# =============================================================================

_BIO_HTML = """
<h2 class="normal_header">Character Information</h2>
Age: 17; 19<br>
Height: 172 cm<br>
Blood type: F<br>
Devil fruit: Gomu Gomu no Mi<br>
<h2>Next Section</h2>
"""


def test_extract_bio_data_key_value_pairs() -> None:
    result = _extract_bio_data(_BIO_HTML)
    assert result["Age"] == "17; 19"
    assert result["Height"] == "172 cm"
    assert result["Devil fruit"] == "Gomu Gomu no Mi"


def test_extract_bio_data_stops_at_next_section() -> None:
    """Bio data parser must not bleed into the next h2 section."""
    result = _extract_bio_data(_BIO_HTML)
    # "Next Section" is h2 text, not bio data
    assert "Next Section" not in result


def test_extract_bio_data_empty_html_returns_empty() -> None:
    result = _extract_bio_data("")
    assert result == {}


def test_extract_bio_data_no_header_returns_empty() -> None:
    result = _extract_bio_data("<p>Some text without a normal_header</p>")
    assert result == {}


def test_extract_bio_data_spoiler_content_stripped() -> None:
    html = """
    <h2 class="normal_header">Info</h2>
    Age: 19<br>
    <div class="spoiler_content">Secret: hidden value</div>
    Blood type: F<br>
    """
    result = _extract_bio_data(html)
    assert "Age" in result
    assert "Secret" not in result


# =============================================================================
# _extract_description
# =============================================================================

_DESC_HTML = """
<h2 class="normal_header">About</h2>
The main character of One Piece.<br>
He ate the Gomu Gomu no Mi.<br>
<div class="normal_header">Voice Actors</div>
"""


def test_extract_description_returns_text() -> None:
    result = _extract_description(_DESC_HTML)
    assert result is not None
    assert "One Piece" in result


def test_extract_description_stops_before_next_section() -> None:
    result = _extract_description(_DESC_HTML)
    assert "Voice Actors" not in (result or "")


def test_extract_description_empty_returns_none() -> None:
    result = _extract_description("")
    assert result is None


def test_extract_description_key_value_lines_excluded() -> None:
    """Lines that look like 'Key: Value' (with short key) are excluded."""
    html = """
    <h2 class="normal_header">About</h2>
    Age: 17<br>
    He is the captain of the Straw Hat Pirates.<br>
    """
    result = _extract_description(html)
    assert result is not None
    assert "captain" in result
    assert "Age: 17" not in result


# =============================================================================
# _extract_voice_actors
# =============================================================================

_VA_HTML = """
<div class="normal_header">Voice Actors</div>
<table>
  <tr>
    <td><img src="https://cdn.myanimelist.net/va.jpg"></td>
    <td>
      <a href="https://myanimelist.net/people/70/Tanaka_Mayumi">Tanaka, Mayumi</a>
      <small>Japanese</small>
    </td>
  </tr>
  <tr>
    <td><img src="https://cdn.myanimelist.net/va2.jpg"></td>
    <td>
      <a href="https://myanimelist.net/people/81/Colleen_Clinkenbeard">Clinkenbeard, Colleen</a>
      <small>English</small>
    </td>
  </tr>
</table>
"""


def test_extract_voice_actors_returns_all() -> None:
    result = _extract_voice_actors(_VA_HTML)
    assert len(result) == 2


def test_extract_voice_actors_name_and_language() -> None:
    result = _extract_voice_actors(_VA_HTML)
    assert result[0].name == "Tanaka, Mayumi"
    assert result[0].language == "Japanese"
    assert result[0].person_id == 70
    assert result[0].sources == ["https://myanimelist.net/people/70/Tanaka_Mayumi"]


def test_extract_voice_actors_english_va() -> None:
    result = _extract_voice_actors(_VA_HTML)
    assert result[1].name == "Clinkenbeard, Colleen"
    assert result[1].language == "English"
    assert result[1].person_id == 81


def test_extract_voice_actors_no_section_returns_empty() -> None:
    result = _extract_voice_actors("<div>No VA section here</div>")
    assert result == []


# =============================================================================
# _extract_ography
# =============================================================================

_OGRAPHY_HTML = """
<div class="normal_header">Animeography</div>
<table>
  <tr>
    <td><a href="https://myanimelist.net/anime/21/One_Piece">One Piece</a></td>
    <td><small>Main</small></td>
  </tr>
  <tr>
    <td><a href="https://myanimelist.net/anime/28933/One_Piece_Film_Gold">One Piece Film: Gold</a></td>
    <td><small>Main</small></td>
  </tr>
</table>
<div class="normal_header">Mangaography</div>
<table>
  <tr>
    <td><a href="https://myanimelist.net/manga/103/One_Piece">One Piece</a></td>
    <td><small>Main</small></td>
  </tr>
</table>
"""


def test_extract_ography_anime_entries() -> None:
    result = _extract_ography(_OGRAPHY_HTML, "Animeography")
    assert len(result) == 2
    titles = [e.title for e in result]
    assert "One Piece" in titles


def test_extract_ography_manga_entries() -> None:
    result = _extract_ography(_OGRAPHY_HTML, "Mangaography")
    assert len(result) == 1
    assert result[0].title == "One Piece"


def test_extract_ography_role_extracted() -> None:
    result = _extract_ography(_OGRAPHY_HTML, "Animeography")
    assert result[0].role == "Main"


def test_extract_ography_sources_populated() -> None:
    result = _extract_ography(_OGRAPHY_HTML, "Animeography")
    assert len(result[0].sources) == 1
    assert "myanimelist.net/anime/21" in result[0].sources[0]


def test_extract_ography_missing_section_returns_empty() -> None:
    result = _extract_ography(_OGRAPHY_HTML, "NonExistentSection")
    assert result == []


# =============================================================================
# _parse_character_raw
# =============================================================================


def test_parse_character_raw_minimal() -> None:
    """Minimal raw dict produces a valid MalScrapedCharacter."""
    raw = {
        "name_header": "Monkey D., Luffy (モンキー・D・ルフィ)",
        "content_html": "",
        "image_src": None,
    }
    char = _parse_character_raw(raw, char_id=40)
    assert char.mal_id == 40
    assert char.name == "Monkey D., Luffy"
    assert char.name_native == "モンキー・D・ルフィ"


def test_parse_character_raw_url_from_raw() -> None:
    raw = {
        "name_header": "Luffy",
        "_url": "https://myanimelist.net/character/40/Monkey_D_Luffy",
        "content_html": "",
        "image_src": None,
    }
    char = _parse_character_raw(raw, char_id=40)
    assert char.url == "https://myanimelist.net/character/40/Monkey_D_Luffy"


def test_parse_character_raw_favorites_extracted() -> None:
    raw = {
        "name_header": "Luffy",
        "favorites": "123,456",
        "content_html": "",
        "image_src": None,
    }
    char = _parse_character_raw(raw, char_id=40)
    assert char.favorites == 123456
