"""Unit tests for mal_character_crawler.py.

Fixture tests use real HTML captured from:
    https://myanimelist.net/character/40/Luffy_Monkey_D (2026-06-09, with scroll)
via the mal_character_extracted session fixture.

Edge-case tests use synthetic HTML to isolate specific parsing branches.
"""

from unittest.mock import AsyncMock, patch

import pytest
from enrichment.sources.mal.mal_character_crawler import (
    _XPATHS,
    _build_character_from_raw,
    _extract_bio_data,
    _extract_character_from_html,
    _extract_description,
    _extract_name_and_native,
    _extract_ography,
    _extract_voice_actors,
    _fetch_character_html,
    _fetch_mal_character_data,
    MalCharacterCrawler,
    fetch_mal_character,
    fetch_mal_characters,
)

_LUFFY_URL = "https://myanimelist.net/character/40/Luffy_Monkey_D"


# =============================================================================
# _XPATHS invariants
# =============================================================================


def test_xpaths_has_required_keys() -> None:
    for key in ("name_header", "image_src", "favorites_td", "content"):
        assert key in _XPATHS, f"_XPATHS missing key: {key}"
    assert all(isinstance(v, str) and v for v in _XPATHS.values())


# =============================================================================
# _extract_character_from_html
# =============================================================================


def test_extract_character_from_html_from_fixture(mal_character_html) -> None:
    raw = _extract_character_from_html(mal_character_html)
    assert raw is not None
    assert raw["name_header"] == "Luffy Monkey D. (モンキー・D・ルフィ)"
    assert raw["image_src"] == "https://cdn.myanimelist.net/images/characters/9/310307.jpg"
    assert raw["favorites"] == "148,888"
    assert len(raw["content_html"]) > 10_000
    assert "Animeography" in raw["content_html"]
    assert "Voice Actors" in raw["content_html"]


def test_extract_character_from_html_empty_returns_none() -> None:
    assert _extract_character_from_html("") is None


def test_extract_character_from_html_no_character_data_returns_none() -> None:
    assert _extract_character_from_html("<html><body><p>nothing</p></body></html>") is None


def test_extract_character_from_html_missing_image_and_favorites() -> None:
    html = """<html><body>
    <h2 class="normal_header">Test Char</h2>
    <div id="content"><p>Some content</p></div>
    </body></html>"""
    raw = _extract_character_from_html(html)
    assert raw is not None
    assert raw["name_header"] == "Test Char"
    assert raw["image_src"] is None
    assert raw["favorites"] is None


# =============================================================================
# _extract_name_and_native
# =============================================================================


def test_extract_name_and_native_from_fixture(mal_character_extracted) -> None:
    name, native = _extract_name_and_native(mal_character_extracted["name_header"])
    assert name == "Luffy Monkey D."
    assert native == "モンキー・D・ルフィ"


def test_extract_name_and_native_with_comma_inversion() -> None:
    name, native = _extract_name_and_native("Monkey D., Luffy (モンキー・D・ルフィ)")
    assert name == "Monkey D., Luffy"
    assert native == "モンキー・D・ルフィ"


def test_extract_name_and_native_no_native() -> None:
    name, native = _extract_name_and_native("Roronoa, Zoro")
    assert name == "Roronoa, Zoro"
    assert native is None


def test_extract_name_and_native_empty() -> None:
    _, native = _extract_name_and_native(None)
    assert native is None


# =============================================================================
# _extract_bio_data
# =============================================================================


def test_extract_bio_data_from_fixture(mal_character_extracted) -> None:
    attrs, spoilers = _extract_bio_data(mal_character_extracted["content_html"])
    assert attrs.get("age") == "17; 19"
    assert attrs.get("height") == "172 cm"
    assert attrs.get("blood_type") == "F"
    assert "devil_fruit" in attrs
    assert "bounty" in spoilers


def test_extract_bio_data_key_value_pairs() -> None:
    html = """
<h2 class="normal_header">Character Information</h2>
Age: 17; 19<br>
Height: 172 cm<br>
Blood type: F<br>
Devil fruit: Gomu Gomu no Mi<br>
<h2>Next Section</h2>
"""
    attrs, _ = _extract_bio_data(html)
    assert attrs["age"] == "17; 19"
    assert attrs["height"] == "172 cm"
    assert attrs["devil_fruit"] == "Gomu Gomu no Mi"
    assert "Next Section" not in attrs


def test_extract_bio_data_empty_html_returns_empty() -> None:
    attrs, spoilers = _extract_bio_data("")
    assert attrs == {}
    assert spoilers == {}


def test_extract_bio_data_no_header_returns_empty() -> None:
    attrs, spoilers = _extract_bio_data("<p>Some text without a normal_header</p>")
    assert attrs == {}
    assert spoilers == {}


_SPOILER_ONLY_HTML = """
<h2 class="normal_header">Info</h2>
Bounty:
<div class="spoiler" id="spoiler123">
  <input type="button" class="button-secondary" value="Show">
  <span class="spoiler_content" style="display: none;">
    <input type="button" class="button-secondary" value="Hide"><br>
    3,000,000,000
  </span>
</div><br>
Age: 19<br>
"""

_SPOILER_SUFFIX_HTML = """
<h2 class="normal_header">Info</h2>
Devil fruit:
Gomu Gomu no Mi,
<div class="spoiler" id="spoiler456">
  <input type="button" class="button-secondary" value="Show">
  <span class="spoiler_content" style="display: none;">
    <input type="button" class="button-secondary" value="Hide"><br>
    Hito Hito no Mi
  </span>
</div><br>
Age: 17<br>
"""


def test_bio_data_spoiler_only_value_not_in_attrs() -> None:
    attrs, spoilers = _extract_bio_data(_SPOILER_ONLY_HTML)
    assert "bounty" not in attrs
    assert "bounty" in spoilers
    assert "3,000,000,000" in spoilers["bounty"]


def test_extract_bio_data_spoiler_suffix_split() -> None:
    attrs, spoilers = _extract_bio_data(_SPOILER_SUFFIX_HTML)
    assert attrs.get("devil_fruit") == "Gomu Gomu no Mi"
    assert spoilers.get("devil_fruit") == "Hito Hito no Mi"


def test_extract_bio_data_spoiler_no_content_span_ignored() -> None:
    html = """
<h2 class="normal_header">Info</h2>
Key:
<div class="spoiler">No proper spoiler_content span here</div><br>
"""
    _, spoilers = _extract_bio_data(html)
    assert "key" not in spoilers


# =============================================================================
# _extract_description
# =============================================================================


def test_extract_description_from_fixture(mal_character_extracted) -> None:
    desc, _ = _extract_description(mal_character_extracted["content_html"])
    assert desc is not None
    assert "Straw Hat" in desc
    assert "Blood type" not in desc
    assert "Height" not in desc


def test_extract_description_synthetic_html() -> None:
    html = """
<h2 class="normal_header">About</h2>
The main character of One Piece.<br>
He ate the Gomu Gomu no Mi.<br>
<div class="normal_header">Voice Actors</div>
"""
    desc, _ = _extract_description(html)
    assert desc is not None
    assert "One Piece" in desc
    assert "Voice Actors" not in desc


def test_extract_description_empty_returns_none() -> None:
    desc, desc_spoiler = _extract_description("")
    assert desc is None
    assert desc_spoiler is None


def test_extract_description_key_value_lines_excluded() -> None:
    html = """
    <h2 class="normal_header">About</h2>
    Age: 17<br>
    He is the captain of the Straw Hat Pirates.<br>
    """
    desc, _ = _extract_description(html)
    assert desc is not None
    assert "captain" in desc
    assert "Age: 17" not in desc


def test_extract_description_captures_prose_spoiler() -> None:
    html = """
<h2 class="normal_header">About</h2>
He is the captain of the Straw Hat Pirates.<br>
<div class="spoiler" id="spoiler789">
  <input type="button" class="button-secondary" value="Show">
  <span class="spoiler_content" style="display: none;">
    <input type="button" class="button-secondary" value="Hide"><br>
    He is also the son of Dragon, the most wanted criminal.
  </span>
</div><br>
"""
    desc, desc_spoiler = _extract_description(html)
    assert desc is not None
    assert "captain" in desc
    assert desc_spoiler is not None
    assert "Dragon" in desc_spoiler


# =============================================================================
# _extract_voice_actors
# =============================================================================


def test_extract_voice_actors_from_fixture(mal_character_extracted) -> None:
    vas = _extract_voice_actors(mal_character_extracted["content_html"])
    assert len(vas) == 28
    assert vas[0].name == "Tanaka, Mayumi"
    assert vas[0].language == "Japanese"
    assert vas[0].person_id == 75
    assert "myanimelist.net/people/75" in vas[0].sources[0]
    assert all(len(v.sources) == 1 for v in vas)
    english = next(v for v in vas if v.language == "English")
    assert english.name == "Clinkenbeard, Colleen"
    assert english.person_id == 472


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


def test_extract_voice_actors_synthetic_html() -> None:
    result = _extract_voice_actors(_VA_HTML)
    assert len(result) == 2
    assert result[0].name == "Tanaka, Mayumi"
    assert result[0].language == "Japanese"
    assert result[0].person_id == 70
    assert result[0].sources == ["https://myanimelist.net/people/70/Tanaka_Mayumi"]
    assert result[1].name == "Clinkenbeard, Colleen"
    assert result[1].language == "English"
    assert result[1].person_id == 81


def test_extract_voice_actors_no_section_returns_empty() -> None:
    assert _extract_voice_actors("<div>No VA section here</div>") == []


def test_extract_voice_actors_row_without_person_link_skipped() -> None:
    html = """
<div class="normal_header">Voice Actors</div>
<table>
  <tr><td>No link in this row at all</td></tr>
  <tr>
    <td>
      <a href="https://myanimelist.net/people/70/Tanaka_Mayumi">Tanaka, Mayumi</a>
      <small>Japanese</small>
    </td>
  </tr>
</table>
"""
    result = _extract_voice_actors(html)
    assert len(result) == 1
    assert result[0].name == "Tanaka, Mayumi"


# =============================================================================
# _extract_ography
# =============================================================================


def test_extract_ography_from_fixture(mal_character_extracted) -> None:
    anime = _extract_ography(mal_character_extracted["content_html"], "Animeography")
    assert len(anime) == 60
    assert anime[0].title == "One Piece"
    assert anime[0].role == "Main"
    assert "myanimelist.net/anime/21" in anime[0].sources[0]

    manga = _extract_ography(mal_character_extracted["content_html"], "Mangaography")
    assert len(manga) == 16
    assert manga[0].title == "One Piece"
    assert manga[0].role == "Main"


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


def test_extract_ography_synthetic_html() -> None:
    anime = _extract_ography(_OGRAPHY_HTML, "Animeography")
    assert len(anime) == 2
    assert anime[0].title == "One Piece"
    assert anime[0].role == "Main"
    assert "myanimelist.net/anime/21" in anime[0].sources[0]

    manga = _extract_ography(_OGRAPHY_HTML, "Mangaography")
    assert len(manga) == 1
    assert manga[0].title == "One Piece"


def test_extract_ography_missing_section_returns_empty() -> None:
    assert _extract_ography(_OGRAPHY_HTML, "NonExistentSection") == []


def test_extract_ography_row_without_title_skipped() -> None:
    html = """
<div class="normal_header">Animeography</div>
<table>
  <tr>
    <td><a href="https://myanimelist.net/anime/21/One_Piece"><img src="x.jpg"></a></td>
  </tr>
  <tr>
    <td><a href="https://myanimelist.net/anime/21/One_Piece">One Piece</a></td>
    <td><small>Main</small></td>
  </tr>
</table>
"""
    result = _extract_ography(html, "Animeography")
    assert len(result) == 1
    assert result[0].title == "One Piece"


# =============================================================================
# _build_character_from_raw
# =============================================================================


def test_build_character_from_raw_from_fixture(mal_character_extracted) -> None:
    char = _build_character_from_raw(mal_character_extracted, _LUFFY_URL)
    assert char.name == "Luffy Monkey D."
    assert char.name_native == "モンキー・D・ルフィ"
    assert char.favorites == 148888
    assert char.images == ["https://cdn.myanimelist.net/images/characters/9/310307.jpg"]
    assert char.source == _LUFFY_URL
    assert char.character_info.get("age") == "17; 19"
    assert len(char.animeography) == 60
    assert len(char.mangaography) == 16
    assert len(char.voice_actors) == 28


def test_build_character_from_raw_favorites_no_comma(mal_character_extracted) -> None:
    char = _build_character_from_raw({**mal_character_extracted, "favorites": "12345"}, _LUFFY_URL)
    assert char.favorites == 12345


def test_build_character_from_raw_missing_favorites_defaults_to_zero(mal_character_extracted) -> None:
    raw = {k: v for k, v in mal_character_extracted.items() if k != "favorites"}
    char = _build_character_from_raw(raw, _LUFFY_URL)
    assert char.favorites == 0


def test_build_character_from_raw_missing_image_empty_list(mal_character_extracted) -> None:
    char = _build_character_from_raw({**mal_character_extracted, "image_src": None}, _LUFFY_URL)
    assert char.images == []


def test_build_character_from_raw_url_from_explicit_arg(mal_character_extracted) -> None:
    custom_url = "https://myanimelist.net/character/40/SomeSlug"
    char = _build_character_from_raw(mal_character_extracted, custom_url)
    assert char.source == custom_url


# =============================================================================
# MalCharacterCrawler
# =============================================================================


def test_mal_character_crawler_get_extraction_schema() -> None:
    from enrichment.sources.base.framework import DockerTransport, NullRepository
    crawler = MalCharacterCrawler(DockerTransport(), NullRepository())
    schema = crawler.get_extraction_schema()
    assert schema == {"xpaths": _XPATHS}


def test_mal_character_crawler_normalize_identifier() -> None:
    from enrichment.sources.base.framework import DockerTransport, NullRepository
    crawler = MalCharacterCrawler(DockerTransport(), NullRepository())
    assert crawler.normalize_identifier(_LUFFY_URL) == _LUFFY_URL


# =============================================================================
# _fetch_character_html
# =============================================================================


@pytest.mark.asyncio
async def test_fetch_character_html_success(mal_character_html) -> None:
    page_mock = AsyncMock()
    page_mock.get_content = AsyncMock(return_value=mal_character_html)
    page_mock.wait_for = AsyncMock()
    page_mock.scroll_down = AsyncMock()
    page_mock.url = _LUFFY_URL
    browser_mock = AsyncMock()
    browser_mock.get = AsyncMock(return_value=page_mock)

    result = await _fetch_character_html(browser_mock, _LUFFY_URL)
    assert result is not None
    html, url = result
    assert url == _LUFFY_URL
    assert len(html) > 1000
    page_mock.scroll_down.assert_awaited_once()


@pytest.mark.asyncio
async def test_fetch_character_html_navigation_failure() -> None:
    browser_mock = AsyncMock()
    browser_mock.get = AsyncMock(side_effect=Exception("nav failed"))
    result = await _fetch_character_html(browser_mock, _LUFFY_URL)
    assert result is None


# =============================================================================
# _fetch_mal_character_data
# =============================================================================


def _disable_cache(mocker):
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )


def _mock_browser(mocker):
    browser_mock = AsyncMock()
    browser_mock.stop = AsyncMock()
    mocker.patch("zendriver.start", new_callable=AsyncMock, return_value=browser_mock)
    return browser_mock


@pytest.mark.asyncio
async def test_fetch_mal_character_data_navigation_failure(mocker) -> None:
    _disable_cache(mocker)
    _mock_browser(mocker)
    mocker.patch(
        "enrichment.sources.mal.mal_character_crawler._fetch_character_html",
        new_callable=AsyncMock,
        return_value=None,
    )
    assert await _fetch_mal_character_data(_LUFFY_URL) is None


@pytest.mark.asyncio
async def test_fetch_mal_character_data_extraction_failure(mocker) -> None:
    _disable_cache(mocker)
    _mock_browser(mocker)
    mocker.patch(
        "enrichment.sources.mal.mal_character_crawler._fetch_character_html",
        new_callable=AsyncMock,
        return_value=("<html><body></body></html>", _LUFFY_URL),
    )
    assert await _fetch_mal_character_data(_LUFFY_URL) is None


@pytest.mark.asyncio
async def test_fetch_mal_character_data_success(mocker, mal_character_html) -> None:
    _disable_cache(mocker)
    _mock_browser(mocker)
    mocker.patch(
        "enrichment.sources.mal.mal_character_crawler._fetch_character_html",
        new_callable=AsyncMock,
        return_value=(mal_character_html, _LUFFY_URL),
    )
    result = await _fetch_mal_character_data(_LUFFY_URL)
    assert result is not None
    raw, url = result
    assert raw["name_header"] == "Luffy Monkey D. (モンキー・D・ルフィ)"
    assert url == _LUFFY_URL


# =============================================================================
# fetch_mal_character
# =============================================================================


@pytest.mark.asyncio
async def test_fetch_mal_character_returns_none_when_no_data(mocker) -> None:
    mocker.patch(
        "enrichment.sources.mal.mal_character_crawler._fetch_mal_character_data",
        new_callable=AsyncMock,
        return_value=None,
    )
    assert await fetch_mal_character(_LUFFY_URL) is None


@pytest.mark.asyncio
async def test_fetch_mal_character_returns_parsed_character(mocker, mal_character_extracted) -> None:
    mocker.patch(
        "enrichment.sources.mal.mal_character_crawler._fetch_mal_character_data",
        new_callable=AsyncMock,
        return_value=(mal_character_extracted, _LUFFY_URL),
    )
    result = await fetch_mal_character(_LUFFY_URL)
    assert result is not None
    assert result["name"] == "Luffy Monkey D."
    assert result["sources"] == [_LUFFY_URL]
    assert result["favorites"] == 148888


# =============================================================================
# fetch_mal_characters
# =============================================================================


@pytest.mark.asyncio
async def test_empty_list() -> None:
    assert await fetch_mal_characters([]) == []


@pytest.mark.asyncio
async def test_all_cached(mocker, mal_character_extracted) -> None:
    mocker.patch.object(
        _fetch_mal_character_data,
        "cache_batch_get",
        new=AsyncMock(return_value=([[mal_character_extracted, _LUFFY_URL]], [])),
    )
    result = await fetch_mal_characters([_LUFFY_URL])
    assert len(result) == 1
    assert result[0] is not None
    assert result[0]["name"] == "Luffy Monkey D."


@pytest.mark.asyncio
async def test_malformed_cache_falls_through(mocker, mal_character_html) -> None:
    mocker.patch.object(
        _fetch_mal_character_data,
        "cache_batch_get",
        new=AsyncMock(return_value=(["not_a_tuple"], [])),
    )
    cache_set = AsyncMock()
    mocker.patch.object(_fetch_mal_character_data, "cache_batch_set", new=cache_set)
    _mock_browser(mocker)
    mocker.patch(
        "enrichment.sources.mal.mal_character_crawler._fetch_character_html",
        new_callable=AsyncMock,
        return_value=(mal_character_html, _LUFFY_URL),
    )
    result = await fetch_mal_characters([_LUFFY_URL])
    assert result[0] is not None
    assert result[0]["name"] == "Luffy Monkey D."


@pytest.mark.asyncio
async def test_navigation_failure(mocker) -> None:
    mocker.patch.object(
        _fetch_mal_character_data,
        "cache_batch_get",
        new=AsyncMock(return_value=([None], [0])),
    )
    mocker.patch.object(_fetch_mal_character_data, "cache_batch_set", new=AsyncMock())
    _mock_browser(mocker)
    mocker.patch(
        "enrichment.sources.mal.mal_character_crawler._fetch_character_html",
        new_callable=AsyncMock,
        return_value=None,
    )
    assert await fetch_mal_characters([_LUFFY_URL]) == [None]


@pytest.mark.asyncio
async def test_extraction_failure(mocker) -> None:
    mocker.patch.object(
        _fetch_mal_character_data,
        "cache_batch_get",
        new=AsyncMock(return_value=([None], [0])),
    )
    mocker.patch.object(_fetch_mal_character_data, "cache_batch_set", new=AsyncMock())
    _mock_browser(mocker)
    mocker.patch(
        "enrichment.sources.mal.mal_character_crawler._fetch_character_html",
        new_callable=AsyncMock,
        return_value=("<html><body></body></html>", _LUFFY_URL),
    )
    assert await fetch_mal_characters([_LUFFY_URL]) == [None]


@pytest.mark.asyncio
async def test_success(mocker, mal_character_html) -> None:
    mocker.patch.object(
        _fetch_mal_character_data,
        "cache_batch_get",
        new=AsyncMock(return_value=([None], [0])),
    )
    cache_set = AsyncMock()
    mocker.patch.object(_fetch_mal_character_data, "cache_batch_set", new=cache_set)
    _mock_browser(mocker)
    mocker.patch(
        "enrichment.sources.mal.mal_character_crawler._fetch_character_html",
        new_callable=AsyncMock,
        return_value=(mal_character_html, _LUFFY_URL),
    )
    result = await fetch_mal_characters([_LUFFY_URL])
    assert len(result) == 1
    assert result[0] is not None
    assert result[0]["name"] == "Luffy Monkey D."
    cache_set.assert_awaited_once()


@pytest.mark.asyncio
async def test_merges_cached_and_fetched(mocker, mal_character_extracted, mal_character_html) -> None:
    url2 = "https://myanimelist.net/character/41/Roronoa_Zoro"
    zoro_raw = {**mal_character_extracted, "name_header": "Zoro Roronoa", "favorites": "10,000"}
    mocker.patch.object(
        _fetch_mal_character_data,
        "cache_batch_get",
        new=AsyncMock(return_value=([[mal_character_extracted, _LUFFY_URL], None], [1])),
    )
    cache_set = AsyncMock()
    mocker.patch.object(_fetch_mal_character_data, "cache_batch_set", new=cache_set)
    _mock_browser(mocker)
    mocker.patch(
        "enrichment.sources.mal.mal_character_crawler._fetch_character_html",
        new_callable=AsyncMock,
        return_value=(mal_character_html, url2),
    )
    mocker.patch(
        "enrichment.sources.mal.mal_character_crawler._extract_character_from_html",
        return_value=zoro_raw,
    )
    result = await fetch_mal_characters([_LUFFY_URL, url2])
    assert len(result) == 2
    assert result[0]["name"] == "Luffy Monkey D."
    assert result[1]["name"] == "Zoro Roronoa"
    cache_set.assert_awaited_once()


@pytest.mark.asyncio
async def test_inter_request_delay(mocker, mal_character_html) -> None:
    url2 = "https://myanimelist.net/character/41/Roronoa_Zoro"
    mocker.patch.object(
        _fetch_mal_character_data,
        "cache_batch_get",
        new=AsyncMock(return_value=([None, None], [0, 1])),
    )
    mocker.patch.object(_fetch_mal_character_data, "cache_batch_set", new=AsyncMock())
    _mock_browser(mocker)
    mocker.patch(
        "enrichment.sources.mal.mal_character_crawler._fetch_character_html",
        new_callable=AsyncMock,
        return_value=(mal_character_html, _LUFFY_URL),
    )
    sleep_mock = mocker.patch("asyncio.sleep", new_callable=AsyncMock)
    result = await fetch_mal_characters([_LUFFY_URL, url2])
    assert all(r is not None for r in result)
    # sleep called once between 2 misses (not after the last one)
    assert sleep_mock.await_count >= 1


# =============================================================================
# main()
# =============================================================================


@pytest.mark.asyncio
async def test_main_returns_1_when_no_character(mocker, tmp_path) -> None:
    mocker.patch("sys.argv", ["prog", _LUFFY_URL, "--output", str(tmp_path / "out.json")])
    mocker.patch(
        "enrichment.sources.mal.mal_character_crawler.fetch_mal_character",
        return_value=None,
    )
    from enrichment.sources.mal.mal_character_crawler import main
    assert await main() == 1


@pytest.mark.asyncio
async def test_main_returns_0_on_success(mocker, tmp_path) -> None:
    mocker.patch("sys.argv", ["prog", _LUFFY_URL, "--output", str(tmp_path / "out.json")])
    mocker.patch(
        "enrichment.sources.mal.mal_character_crawler.fetch_mal_character",
        return_value={"name": "Luffy Monkey D."},
    )
    from enrichment.sources.mal.mal_character_crawler import main
    assert await main() == 0


# =============================================================================
# Branch coverage gap tests
# =============================================================================


def test_extract_bio_data_key_too_long_skipped() -> None:
    long_key = "a" * 50
    html = f"""
<h2 class="normal_header">Info</h2>
{long_key}: should be ignored<br>
Age: 17<br>
"""
    attrs, _ = _extract_bio_data(html)
    assert long_key.lower() not in attrs
    assert attrs.get("age") == "17"


@pytest.mark.asyncio
async def test_fetch_mal_character_data_browser_stop_exception(mocker, mal_character_html) -> None:
    _disable_cache(mocker)
    browser_mock = AsyncMock()
    browser_mock.stop = AsyncMock(side_effect=Exception("stop failed"))
    mocker.patch("zendriver.start", new_callable=AsyncMock, return_value=browser_mock)
    mocker.patch(
        "enrichment.sources.mal.mal_character_crawler._fetch_character_html",
        new_callable=AsyncMock,
        return_value=(mal_character_html, _LUFFY_URL),
    )
    result = await _fetch_mal_character_data(_LUFFY_URL)
    assert result is not None


@pytest.mark.asyncio
async def test_parse_cached_non_dict_raw(mocker, mal_character_html) -> None:
    mocker.patch.object(
        _fetch_mal_character_data,
        "cache_batch_get",
        new=AsyncMock(return_value=([[42, _LUFFY_URL]], [])),
    )
    mocker.patch.object(_fetch_mal_character_data, "cache_batch_set", new=AsyncMock())
    _mock_browser(mocker)
    mocker.patch(
        "enrichment.sources.mal.mal_character_crawler._fetch_character_html",
        new_callable=AsyncMock,
        return_value=(mal_character_html, _LUFFY_URL),
    )
    result = await fetch_mal_characters([_LUFFY_URL])
    assert result[0] is not None
    assert result[0]["name"] == "Luffy Monkey D."


@pytest.mark.asyncio
async def test_browser_stop_exception(mocker, mal_character_html) -> None:
    mocker.patch.object(
        _fetch_mal_character_data,
        "cache_batch_get",
        new=AsyncMock(return_value=([None], [0])),
    )
    mocker.patch.object(_fetch_mal_character_data, "cache_batch_set", new=AsyncMock())
    browser_mock = AsyncMock()
    browser_mock.stop = AsyncMock(side_effect=Exception("stop failed"))
    mocker.patch("zendriver.start", new_callable=AsyncMock, return_value=browser_mock)
    mocker.patch(
        "enrichment.sources.mal.mal_character_crawler._fetch_character_html",
        new_callable=AsyncMock,
        return_value=(mal_character_html, _LUFFY_URL),
    )
    result = await fetch_mal_characters([_LUFFY_URL])
    assert result[0] is not None
    assert result[0]["name"] == "Luffy Monkey D."
