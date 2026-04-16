"""Unit tests for mal_character_crawler.py — HTML extraction helpers.

All tests use HTML fixtures that mirror the real MAL character page structure.
No network calls are made.
"""

import json
from unittest.mock import AsyncMock

import pytest

from enrichment.crawlers.mal_crawler.mal_character_crawler import (
    _extract_bio_data,
    _extract_description,
    _extract_name_and_native,
    _extract_ography,
    _extract_voice_actors,
    _fetch_mal_character_data,
    _get_character_schema,
    _parse_character_raw,
    fetch_mal_character,
    fetch_mal_characters,
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
    attrs, spoilers = _extract_bio_data(_BIO_HTML)
    assert attrs["age"] == "17; 19"
    assert attrs["height"] == "172 cm"
    assert attrs["devil_fruit"] == "Gomu Gomu no Mi"


def test_extract_bio_data_stops_at_next_section() -> None:
    """Bio data parser must not bleed into the next h2 section."""
    attrs, spoilers = _extract_bio_data(_BIO_HTML)
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
    """Spoiler-only field must appear in spoilers dict, not in attributes."""
    attrs, spoilers = _extract_bio_data(_SPOILER_ONLY_HTML)
    assert "bounty" not in attrs
    assert "bounty" in spoilers
    assert "3,000,000,000" in spoilers["bounty"]


def test_extract_bio_data_spoiler_suffix_split() -> None:
    """Non-spoiler value goes to attrs; spoiler value goes to spoilers — same key."""
    attrs, spoilers = _extract_bio_data(_SPOILER_SUFFIX_HTML)
    assert attrs.get("devil_fruit") == "Gomu Gomu no Mi"
    assert spoilers.get("devil_fruit") == "Hito Hito no Mi"


def test_extract_bio_data_attrs_no_double_comma() -> None:
    """Non-spoiler value in attrs must not have a trailing comma artifact."""
    attrs, _spoilers = _extract_bio_data(_SPOILER_SUFFIX_HTML)
    assert ",," not in attrs.get("devil_fruit", "")
    assert ", ," not in attrs.get("devil_fruit", "")


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
    desc, _spoiler = _extract_description(_DESC_HTML)
    assert desc is not None
    assert "One Piece" in desc


def test_extract_description_stops_before_next_section() -> None:
    desc, _spoiler = _extract_description(_DESC_HTML)
    assert "Voice Actors" not in (desc or "")


def test_extract_description_empty_returns_none() -> None:
    desc, desc_spoiler = _extract_description("")
    assert desc is None
    assert desc_spoiler is None


def test_extract_description_key_value_lines_excluded() -> None:
    """Lines that look like 'Key: Value' (with short key) are excluded."""
    html = """
    <h2 class="normal_header">About</h2>
    Age: 17<br>
    He is the captain of the Straw Hat Pirates.<br>
    """
    desc, _spoiler = _extract_description(html)
    assert desc is not None
    assert "captain" in desc
    assert "Age: 17" not in desc


_DESC_SPOILER_HTML = """
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


def test_extract_description_captures_prose_spoiler() -> None:
    """Prose spoiler div in description section is captured in second return value."""
    desc, desc_spoiler = _extract_description(_DESC_SPOILER_HTML)
    assert desc is not None
    assert "captain" in desc
    assert desc_spoiler is not None
    assert "Dragon" in desc_spoiler


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
    """Minimal raw dict produces a valid MalCharacter."""
    raw = {
        "name_header": "Monkey D., Luffy (モンキー・D・ルフィ)",
        "content_html": "",
        "image_src": None,
    }
    char = _parse_character_raw(
        raw, url="https://myanimelist.net/character/40/Monkey_D_Luffy"
    )
    assert char.source == "https://myanimelist.net/character/40/Monkey_D_Luffy"
    assert char.name == "Monkey D., Luffy"
    assert char.name_native == "モンキー・D・ルフィ"
    assert char.spoilers == {}


def test_parse_character_raw_url_from_raw() -> None:
    raw = {
        "name_header": "Luffy",
        "_url": "https://myanimelist.net/character/40/Monkey_D_Luffy",
        "content_html": "",
        "image_src": None,
    }
    char = _parse_character_raw(
        raw, url="https://myanimelist.net/character/40/Monkey_D_Luffy"
    )
    assert char.source == "https://myanimelist.net/character/40/Monkey_D_Luffy"


def test_parse_character_raw_favorites_extracted() -> None:
    raw = {
        "name_header": "Luffy",
        "favorites": "123,456",
        "content_html": "",
        "image_src": None,
    }
    char = _parse_character_raw(
        raw, url="https://myanimelist.net/character/40/Monkey_D_Luffy"
    )
    assert char.favorites == 123456


# =============================================================================
# _get_character_schema (line 55)
# =============================================================================


def test_get_character_schema_has_fields() -> None:
    schema = _get_character_schema()
    assert "fields" in schema
    assert isinstance(schema["fields"], list)


# =============================================================================
# _extract_voice_actors — row without person link skipped (line 246)
# =============================================================================

_VA_NO_LINK_HTML = """
<div class="normal_header">Voice Actors</div>
<table>
  <tr>
    <td>No link in this row at all</td>
  </tr>
  <tr>
    <td>
      <a href="https://myanimelist.net/people/70/Tanaka_Mayumi">Tanaka, Mayumi</a>
      <small>Japanese</small>
    </td>
  </tr>
</table>
"""


def test_extract_voice_actors_row_without_person_link_skipped() -> None:
    result = _extract_voice_actors(_VA_NO_LINK_HTML)
    assert len(result) == 1
    assert result[0].name == "Tanaka, Mayumi"


# =============================================================================
# _extract_ography — row without title skipped (line 292)
# =============================================================================

_OGRAPHY_IMG_ONLY_HTML = """
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


def test_extract_ography_row_without_title_skipped() -> None:
    result = _extract_ography(_OGRAPHY_IMG_ONLY_HTML, "Animeography")
    assert len(result) == 1
    assert result[0].title == "One Piece"


# =============================================================================
# _fetch_mal_character_data — async branches (lines 320-338)
# =============================================================================


@pytest.mark.asyncio
async def test_fetch_mal_character_data_no_results(mocker) -> None:
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )
    mocker.patch(
        "enrichment.crawlers.mal_crawler.mal_character_crawler.crawl_batch_urls",
        return_value=[],
    )
    mocker.patch.object(
        _fetch_mal_character_data.__wrapped__.__globals__["_limiter"],
        "acquire",
        return_value=None,
    )
    result = await _fetch_mal_character_data("https://myanimelist.net/character/999")
    assert result is None


@pytest.mark.asyncio
async def test_fetch_mal_character_data_http_error(mocker) -> None:
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )
    mocker.patch(
        "enrichment.crawlers.mal_crawler.mal_character_crawler.crawl_batch_urls",
        return_value=[{"status_code": 403, "extracted_content": None}],
    )
    mocker.patch.object(
        _fetch_mal_character_data.__wrapped__.__globals__["_limiter"],
        "acquire",
        return_value=None,
    )
    result = await _fetch_mal_character_data("https://myanimelist.net/character/998")
    assert result is None


@pytest.mark.asyncio
async def test_fetch_mal_character_data_empty_content(mocker) -> None:
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )
    mocker.patch(
        "enrichment.crawlers.mal_crawler.mal_character_crawler.crawl_batch_urls",
        return_value=[{"status_code": 200, "extracted_content": "[]"}],
    )
    mocker.patch.object(
        _fetch_mal_character_data.__wrapped__.__globals__["_limiter"],
        "acquire",
        return_value=None,
    )
    result = await _fetch_mal_character_data("https://myanimelist.net/character/997")
    assert result is None


@pytest.mark.asyncio
async def test_fetch_mal_character_data_success(mocker) -> None:
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )
    raw = {"name_header": "Luffy", "content_html": "", "image_src": None}
    mocker.patch(
        "enrichment.crawlers.mal_crawler.mal_character_crawler.crawl_batch_urls",
        return_value=[
            {
                "status_code": 200,
                "metadata": {"og:url": "https://myanimelist.net/character/996/Luffy"},
                "extracted_content": json.dumps([raw]),
            }
        ],
    )
    mocker.patch.object(
        _fetch_mal_character_data.__wrapped__.__globals__["_limiter"],
        "acquire",
        return_value=None,
    )
    result = await _fetch_mal_character_data("https://myanimelist.net/character/996")
    assert result is not None
    raw, canonical_url = result
    assert raw["name_header"] == "Luffy"
    assert canonical_url == "https://myanimelist.net/character/996/Luffy"


# =============================================================================
# fetch_mal_character — async (lines 379-385)
# =============================================================================


@pytest.mark.asyncio
async def test_fetch_mal_character_returns_none_when_no_data(mocker) -> None:
    mocker.patch(
        "enrichment.crawlers.mal_crawler.mal_character_crawler._fetch_mal_character_data",
        new_callable=AsyncMock,
        return_value=None,
    )
    result = await fetch_mal_character("https://myanimelist.net/character/40")
    assert result is None


@pytest.mark.asyncio
async def test_fetch_mal_character_returns_parsed_character(mocker) -> None:
    raw = {
        "name_header": "Luffy",
        "content_html": "",
        "image_src": None,
        "favorites": "0",
    }
    mocker.patch(
        "enrichment.crawlers.mal_crawler.mal_character_crawler._fetch_mal_character_data",
        new_callable=AsyncMock,
        return_value=(raw, "https://myanimelist.net/character/40/Luffy"),
    )
    result = await fetch_mal_character("https://myanimelist.net/character/40")
    assert result is not None
    assert result["name"] == "Luffy"
    assert result["sources"] == ["https://myanimelist.net/character/40/Luffy"]


# =============================================================================
# fetch_mal_characters — async (lines 402-430)
# =============================================================================


@pytest.mark.asyncio
async def test_fetch_mal_characters_empty_list() -> None:
    result = await fetch_mal_characters([])
    assert result == []


@pytest.mark.asyncio
async def test_fetch_mal_characters_none_result(mocker) -> None:
    mocker.patch.object(
        _fetch_mal_character_data,
        "cache_batch_get",
        new=AsyncMock(return_value=([None], [0])),
    )
    mocker.patch.object(
        _fetch_mal_character_data,
        "cache_batch_set",
        new=AsyncMock(),
    )
    mocker.patch(
        "enrichment.crawlers.mal_crawler.mal_character_crawler.crawl_batch_urls",
        return_value=[None],
    )
    result = await fetch_mal_characters(["https://myanimelist.net/character/40"])
    assert result == [None]


@pytest.mark.asyncio
async def test_fetch_mal_characters_http_error(mocker) -> None:
    mocker.patch.object(
        _fetch_mal_character_data,
        "cache_batch_get",
        new=AsyncMock(return_value=([None], [0])),
    )
    mocker.patch.object(
        _fetch_mal_character_data,
        "cache_batch_set",
        new=AsyncMock(),
    )
    mocker.patch(
        "enrichment.crawlers.mal_crawler.mal_character_crawler.crawl_batch_urls",
        return_value=[
            {
                "url": "https://myanimelist.net/character/40",
                "status_code": 403,
                "extracted_content": None,
            }
        ],
    )
    result = await fetch_mal_characters(["https://myanimelist.net/character/40"])
    assert result == [None]


@pytest.mark.asyncio
async def test_fetch_mal_characters_empty_content(mocker) -> None:
    mocker.patch.object(
        _fetch_mal_character_data,
        "cache_batch_get",
        new=AsyncMock(return_value=([None], [0])),
    )
    mocker.patch.object(
        _fetch_mal_character_data,
        "cache_batch_set",
        new=AsyncMock(),
    )
    mocker.patch(
        "enrichment.crawlers.mal_crawler.mal_character_crawler.crawl_batch_urls",
        return_value=[
            {
                "url": "https://myanimelist.net/character/40",
                "status_code": 200,
                "extracted_content": "[]",
            }
        ],
    )
    result = await fetch_mal_characters(["https://myanimelist.net/character/40"])
    assert result == [None]


@pytest.mark.asyncio
async def test_fetch_mal_characters_success(mocker) -> None:
    mocker.patch.object(
        _fetch_mal_character_data,
        "cache_batch_get",
        new=AsyncMock(return_value=([None], [0])),
    )
    cache_set = AsyncMock()
    mocker.patch.object(
        _fetch_mal_character_data,
        "cache_batch_set",
        new=cache_set,
    )
    raw = {"name_header": "Luffy", "content_html": "", "image_src": None}
    mocker.patch(
        "enrichment.crawlers.mal_crawler.mal_character_crawler.crawl_batch_urls",
        return_value=[
            {
                "url": "https://myanimelist.net/character/40",
                "status_code": 200,
                "extracted_content": json.dumps([raw]),
            }
        ],
    )
    result = await fetch_mal_characters(["https://myanimelist.net/character/40"])
    assert len(result) == 1
    assert result[0] is not None
    assert result[0]["name"] == "Luffy"
    cache_set.assert_awaited_once()


@pytest.mark.asyncio
async def test_fetch_mal_characters_uses_cache_and_merges_results(mocker) -> None:
    url1 = "https://myanimelist.net/character/40"
    url2 = "https://myanimelist.net/character/41"
    raw1 = {"name_header": "Luffy", "content_html": "", "image_src": None}
    raw2 = {"name_header": "Zoro", "content_html": "", "image_src": None}

    mocker.patch.object(
        _fetch_mal_character_data,
        "cache_batch_get",
        new=AsyncMock(return_value=([[raw1, url1], None], [1])),
    )
    cache_set = AsyncMock()
    mocker.patch.object(
        _fetch_mal_character_data,
        "cache_batch_set",
        new=cache_set,
    )
    mocker.patch(
        "enrichment.crawlers.mal_crawler.mal_character_crawler.crawl_batch_urls",
        return_value=[
            {
                "url": url2,
                "status_code": 200,
                "extracted_content": json.dumps([raw2]),
            }
        ],
    )
    result = await fetch_mal_characters([url1, url2])
    assert len(result) == 2
    assert result[0] is not None
    assert result[0]["name"] == "Luffy"
    assert result[1] is not None
    assert result[1]["name"] == "Zoro"
    cache_set.assert_awaited_once()


@pytest.mark.asyncio
async def test_fetch_mal_characters_chunks_requests(mocker) -> None:
    urls = [
        f"https://myanimelist.net/character/{i}" for i in range(1, 32)
    ]  # 31 URLs > _CHARACTER_BATCH_SIZE=30 → forces 2 batches
    raw = {"name_header": "Luffy", "content_html": "", "image_src": None}

    mocker.patch.object(
        _fetch_mal_character_data,
        "cache_batch_get",
        new=AsyncMock(return_value=([None] * len(urls), list(range(len(urls))))),
    )
    cache_set = AsyncMock()
    mocker.patch.object(
        _fetch_mal_character_data,
        "cache_batch_set",
        new=cache_set,
    )

    async def _batch_result(batch_urls: list[str], **kwargs) -> list[dict[str, str]]:
        return [
            {"url": u, "status_code": 200, "extracted_content": json.dumps([raw])}
            for u in batch_urls
        ]

    mocker.patch(
        "enrichment.crawlers.mal_crawler.mal_character_crawler.crawl_batch_urls",
        side_effect=_batch_result,
    )

    result = await fetch_mal_characters(urls)
    assert len(result) == len(urls)
    assert all(item is not None for item in result)
    assert cache_set.await_count == 2


# =============================================================================
# main() CLI (lines 435-453)
# =============================================================================


@pytest.mark.asyncio
async def test_main_returns_1_when_no_character(mocker, tmp_path) -> None:
    mocker.patch(
        "sys.argv",
        [
            "prog",
            "https://myanimelist.net/character/40",
            "--output",
            str(tmp_path / "out.json"),
        ],
    )
    mocker.patch(
        "enrichment.crawlers.mal_crawler.mal_character_crawler.fetch_mal_character",
        return_value=None,
    )
    from enrichment.crawlers.mal_crawler.mal_character_crawler import main

    assert await main() == 1


@pytest.mark.asyncio
async def test_main_returns_0_on_success(mocker, tmp_path) -> None:
    mocker.patch(
        "sys.argv",
        [
            "prog",
            "https://myanimelist.net/character/40",
            "--output",
            str(tmp_path / "out.json"),
        ],
    )
    mocker.patch(
        "enrichment.crawlers.mal_crawler.mal_character_crawler.fetch_mal_character",
        return_value={"name": "Luffy"},
    )
    from enrichment.crawlers.mal_crawler.mal_character_crawler import main

    assert await main() == 0
