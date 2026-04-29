"""Unit tests for mal_character_crawler.py — HTML extraction helpers.

Baseline tests use real XPath extraction output from
https://myanimelist.net/character/40/Luffy_Monkey_D (2026-04-17)
via the mal_character_raw session fixture.

Edge-case tests use {**mal_character_raw, "field": override} to isolate
specific parsing branches without duplicating the full fixture.
"""

import json
from unittest.mock import AsyncMock

import pytest

from enrichment.sources.mal.mal_character_crawler import (
    _extract_bio_data,
    _extract_description,
    _extract_name_and_native,
    _extract_ography,
    _extract_voice_actors,
    _fetch_mal_character_data,
    _get_character_schema,
    _build_character_from_raw,
    fetch_mal_character,
    fetch_mal_characters,
)

_LUFFY_URL = "https://myanimelist.net/character/40/Luffy_Monkey_D"


def _parse(raw: dict, url: str = _LUFFY_URL):
    return _build_character_from_raw(raw, url)


# =============================================================================
# _extract_name_and_native
# =============================================================================


def test_extract_name_and_native_from_fixture(mal_character_raw) -> None:
    name, native = _extract_name_and_native(mal_character_raw["name_header"])
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
# _extract_bio_data — real content_html
# =============================================================================


def test_extract_bio_data_from_fixture(mal_character_raw) -> None:
    attrs, spoilers = _extract_bio_data(mal_character_raw["content_html"])
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


def test_extract_bio_data_spoiler_value_only_in_spoilers_not_attrs() -> None:
    attrs, spoilers = _extract_bio_data(_SPOILER_ONLY_HTML)
    assert "bounty" not in attrs
    assert "bounty" in spoilers
    assert "3,000,000,000" in spoilers["bounty"]


def test_extract_bio_data_spoiler_suffix_split() -> None:
    attrs, spoilers = _extract_bio_data(_SPOILER_SUFFIX_HTML)
    assert attrs.get("devil_fruit") == "Gomu Gomu no Mi"
    assert spoilers.get("devil_fruit") == "Hito Hito no Mi"
    assert ",," not in attrs.get("devil_fruit", "")
    assert ", ," not in attrs.get("devil_fruit", "")


# =============================================================================
# _extract_description — real content_html
# =============================================================================


def test_extract_description_from_fixture(mal_character_raw) -> None:
    desc, _ = _extract_description(mal_character_raw["content_html"])
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
# _extract_voice_actors — real content_html
# =============================================================================


def test_extract_voice_actors_from_fixture(mal_character_raw) -> None:
    vas = _extract_voice_actors(mal_character_raw["content_html"])
    assert len(vas) == 26
    assert vas[0].name == "Tanaka, Mayumi"
    assert vas[0].language == "Japanese"
    assert vas[0].person_id == 75
    assert "myanimelist.net/people/75" in vas[0].sources[0]
    english = next(v for v in vas if v.language == "English")
    assert english.name == "Clinkenbeard, Colleen"
    assert english.person_id == 472


def test_extract_voice_actors_sources_populated_from_fixture(mal_character_raw) -> None:
    vas = _extract_voice_actors(mal_character_raw["content_html"])
    assert all(len(v.sources) == 1 for v in vas)


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
# _extract_ography — real content_html
# =============================================================================


def test_extract_ography_anime_from_fixture(mal_character_raw) -> None:
    result = _extract_ography(mal_character_raw["content_html"], "Animeography")
    assert len(result) == 60
    assert result[0].title == "One Piece"
    assert result[0].role == "Main"
    assert "myanimelist.net/anime/21" in result[0].sources[0]


def test_extract_ography_manga_from_fixture(mal_character_raw) -> None:
    result = _extract_ography(mal_character_raw["content_html"], "Mangaography")
    assert len(result) == 14
    assert result[0].title == "One Piece"
    assert result[0].role == "Main"


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
    assert "One Piece" in [e.title for e in anime]
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
# _build_character_from_raw — fixture-grounded
# =============================================================================


def test_build_character_from_raw_from_fixture(mal_character_raw) -> None:
    char = _parse(mal_character_raw)
    assert char.name == "Luffy Monkey D."
    assert char.name_native == "モンキー・D・ルフィ"
    assert char.favorites == 148210
    assert char.images == ["https://myanimelist.net/images/characters/9/310307.jpg"]
    assert char.source == _LUFFY_URL
    assert char.character_info.get("age") == "17; 19"
    assert len(char.animeography) == 60
    assert len(char.mangaography) == 14
    assert len(char.voice_actors) == 26


def test_build_character_from_raw_favorites_no_comma(mal_character_raw) -> None:
    char = _parse({**mal_character_raw, "favorites": "12345"})
    assert char.favorites == 12345


def test_build_character_from_raw_missing_favorites_defaults_zero(mal_character_raw) -> None:
    char = _parse({k: v for k, v in mal_character_raw.items() if k != "favorites"})
    assert char.favorites == 0


def test_build_character_from_raw_missing_image_empty_list(mal_character_raw) -> None:
    char = _parse({**mal_character_raw, "image_src": None})
    assert char.images == []


def test_build_character_from_raw_url_from_explicit_arg(mal_character_raw) -> None:
    custom_url = "https://myanimelist.net/character/40/SomeSlug"
    char = _parse(mal_character_raw, url=custom_url)
    assert char.source == custom_url


# =============================================================================
# _get_character_schema
# =============================================================================


def test_get_character_schema_has_fields() -> None:
    schema = _get_character_schema()
    assert "fields" in schema
    assert isinstance(schema["fields"], list)


# =============================================================================
# _fetch_mal_character_data — async branches
# =============================================================================


@pytest.mark.asyncio
async def test_fetch_mal_character_data_no_results(mocker) -> None:
    mocker.patch("http_cache.result_cache.get_cache_config", return_value=mocker.MagicMock(cache_enabled=False))
    mocker.patch("enrichment.sources.mal.mal_character_crawler.crawl_batch_urls", return_value=[])
    mocker.patch.object(_fetch_mal_character_data.__wrapped__.__globals__["_limiter"], "acquire", return_value=None)
    assert await _fetch_mal_character_data("https://myanimelist.net/character/999") is None


@pytest.mark.asyncio
async def test_fetch_mal_character_data_http_error(mocker) -> None:
    mocker.patch("http_cache.result_cache.get_cache_config", return_value=mocker.MagicMock(cache_enabled=False))
    mocker.patch("enrichment.sources.mal.mal_character_crawler.crawl_batch_urls", return_value=[{"status_code": 403, "extracted_content": None}])
    mocker.patch.object(_fetch_mal_character_data.__wrapped__.__globals__["_limiter"], "acquire", return_value=None)
    assert await _fetch_mal_character_data("https://myanimelist.net/character/998") is None


@pytest.mark.asyncio
async def test_fetch_mal_character_data_empty_content(mocker) -> None:
    mocker.patch("http_cache.result_cache.get_cache_config", return_value=mocker.MagicMock(cache_enabled=False))
    mocker.patch("enrichment.sources.mal.mal_character_crawler.crawl_batch_urls", return_value=[{"status_code": 200, "extracted_content": "[]"}])
    mocker.patch.object(_fetch_mal_character_data.__wrapped__.__globals__["_limiter"], "acquire", return_value=None)
    assert await _fetch_mal_character_data("https://myanimelist.net/character/997") is None


@pytest.mark.asyncio
async def test_fetch_mal_character_data_success(mocker, mal_character_raw) -> None:
    mocker.patch("http_cache.result_cache.get_cache_config", return_value=mocker.MagicMock(cache_enabled=False))
    raw = {k: v for k, v in mal_character_raw.items() if not k.startswith("_")}
    mocker.patch(
        "enrichment.sources.mal.mal_character_crawler.crawl_batch_urls",
        return_value=[{"status_code": 200, "metadata": {"og:url": _LUFFY_URL}, "extracted_content": json.dumps([raw])}],
    )
    mocker.patch.object(_fetch_mal_character_data.__wrapped__.__globals__["_limiter"], "acquire", return_value=None)
    result = await _fetch_mal_character_data(_LUFFY_URL)
    assert result is not None
    result_raw, canonical_url = result
    assert result_raw["name_header"] == mal_character_raw["name_header"]
    assert canonical_url == _LUFFY_URL


# =============================================================================
# fetch_mal_character — async
# =============================================================================


@pytest.mark.asyncio
async def test_fetch_mal_character_returns_none_when_no_data(mocker) -> None:
    mocker.patch("enrichment.sources.mal.mal_character_crawler._fetch_mal_character_data", new_callable=AsyncMock, return_value=None)
    assert await fetch_mal_character(_LUFFY_URL) is None


@pytest.mark.asyncio
async def test_fetch_mal_character_returns_parsed_character(mocker, mal_character_raw) -> None:
    raw = {k: v for k, v in mal_character_raw.items() if not k.startswith("_")}
    mocker.patch(
        "enrichment.sources.mal.mal_character_crawler._fetch_mal_character_data",
        new_callable=AsyncMock,
        return_value=(raw, _LUFFY_URL),
    )
    result = await fetch_mal_character(_LUFFY_URL)
    assert result is not None
    assert result["name"] == "Luffy Monkey D."
    assert result["sources"] == [_LUFFY_URL]
    assert result["favorites"] == 148210


# =============================================================================
# fetch_mal_characters — async
# =============================================================================


@pytest.mark.asyncio
async def test_fetch_mal_characters_empty_list() -> None:
    assert await fetch_mal_characters([]) == []


@pytest.mark.asyncio
async def test_fetch_mal_characters_none_result(mocker) -> None:
    mocker.patch.object(_fetch_mal_character_data, "cache_batch_get", new=AsyncMock(return_value=([None], [0])))
    mocker.patch.object(_fetch_mal_character_data, "cache_batch_set", new=AsyncMock())
    mocker.patch("enrichment.sources.mal.mal_character_crawler.crawl_batch_urls", return_value=[None])
    assert await fetch_mal_characters([_LUFFY_URL]) == [None]


@pytest.mark.asyncio
async def test_fetch_mal_characters_http_error(mocker) -> None:
    mocker.patch.object(_fetch_mal_character_data, "cache_batch_get", new=AsyncMock(return_value=([None], [0])))
    mocker.patch.object(_fetch_mal_character_data, "cache_batch_set", new=AsyncMock())
    mocker.patch(
        "enrichment.sources.mal.mal_character_crawler.crawl_batch_urls",
        return_value=[{"url": _LUFFY_URL, "status_code": 403, "extracted_content": None}],
    )
    assert await fetch_mal_characters([_LUFFY_URL]) == [None]


@pytest.mark.asyncio
async def test_fetch_mal_characters_empty_content(mocker) -> None:
    mocker.patch.object(_fetch_mal_character_data, "cache_batch_get", new=AsyncMock(return_value=([None], [0])))
    mocker.patch.object(_fetch_mal_character_data, "cache_batch_set", new=AsyncMock())
    mocker.patch(
        "enrichment.sources.mal.mal_character_crawler.crawl_batch_urls",
        return_value=[{"url": _LUFFY_URL, "status_code": 200, "extracted_content": "[]"}],
    )
    assert await fetch_mal_characters([_LUFFY_URL]) == [None]


@pytest.mark.asyncio
async def test_fetch_mal_characters_success(mocker, mal_character_raw) -> None:
    mocker.patch.object(_fetch_mal_character_data, "cache_batch_get", new=AsyncMock(return_value=([None], [0])))
    cache_set = AsyncMock()
    mocker.patch.object(_fetch_mal_character_data, "cache_batch_set", new=cache_set)
    raw = {k: v for k, v in mal_character_raw.items() if not k.startswith("_")}
    mocker.patch(
        "enrichment.sources.mal.mal_character_crawler.crawl_batch_urls",
        return_value=[{"url": _LUFFY_URL, "status_code": 200, "extracted_content": json.dumps([raw])}],
    )
    result = await fetch_mal_characters([_LUFFY_URL])
    assert len(result) == 1
    assert result[0] is not None
    assert result[0]["name"] == "Luffy Monkey D."
    cache_set.assert_awaited_once()


@pytest.mark.asyncio
async def test_fetch_mal_characters_uses_cache_and_merges_results(mocker, mal_character_raw) -> None:
    url1 = _LUFFY_URL
    url2 = "https://myanimelist.net/character/41/Roronoa_Zoro"
    raw1 = {k: v for k, v in mal_character_raw.items() if not k.startswith("_")}
    raw2 = {**raw1, "name_header": "Zoro Roronoa"}

    mocker.patch.object(_fetch_mal_character_data, "cache_batch_get", new=AsyncMock(return_value=([[raw1, url1], None], [1])))
    cache_set = AsyncMock()
    mocker.patch.object(_fetch_mal_character_data, "cache_batch_set", new=cache_set)
    mocker.patch(
        "enrichment.sources.mal.mal_character_crawler.crawl_batch_urls",
        return_value=[{"url": url2, "status_code": 200, "extracted_content": json.dumps([raw2])}],
    )

    result = await fetch_mal_characters([url1, url2])
    assert len(result) == 2
    assert result[0]["name"] == "Luffy Monkey D."
    assert result[1]["name"] == "Zoro Roronoa"
    cache_set.assert_awaited_once()


@pytest.mark.asyncio
async def test_fetch_mal_characters_chunks_requests(mocker, mal_character_raw) -> None:
    urls = [f"https://myanimelist.net/character/{i}" for i in range(1, 32)]
    raw = {k: v for k, v in mal_character_raw.items() if not k.startswith("_")}

    mocker.patch.object(_fetch_mal_character_data, "cache_batch_get", new=AsyncMock(return_value=([None] * len(urls), list(range(len(urls))))))
    cache_set = AsyncMock()
    mocker.patch.object(_fetch_mal_character_data, "cache_batch_set", new=cache_set)

    async def _batch_result(batch_urls, **kwargs):
        return [{"url": u, "status_code": 200, "extracted_content": json.dumps([raw])} for u in batch_urls]

    mocker.patch("enrichment.sources.mal.mal_character_crawler.crawl_batch_urls", side_effect=_batch_result)

    result = await fetch_mal_characters(urls)
    assert len(result) == len(urls)
    assert all(item is not None for item in result)
    assert cache_set.await_count == 2


# =============================================================================
# main() CLI
# =============================================================================


@pytest.mark.asyncio
async def test_main_returns_1_when_no_character(mocker, tmp_path) -> None:
    mocker.patch("sys.argv", ["prog", _LUFFY_URL, "--output", str(tmp_path / "out.json")])
    mocker.patch("enrichment.sources.mal.mal_character_crawler.fetch_mal_character", return_value=None)
    from enrichment.sources.mal.mal_character_crawler import main
    assert await main() == 1


@pytest.mark.asyncio
async def test_main_returns_0_on_success(mocker, tmp_path) -> None:
    mocker.patch("sys.argv", ["prog", _LUFFY_URL, "--output", str(tmp_path / "out.json")])
    mocker.patch("enrichment.sources.mal.mal_character_crawler.fetch_mal_character", return_value={"name": "Luffy Monkey D."})
    from enrichment.sources.mal.mal_character_crawler import main
    assert await main() == 0
