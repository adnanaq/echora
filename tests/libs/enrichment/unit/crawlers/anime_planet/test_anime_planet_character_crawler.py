"""Unit tests for Anime-Planet character crawler, refs crawler, and mapper.

Coverage targets:
- All regex extraction helpers (pure functions, no I/O)
- _build_character assembly
- _fetch_character_data async branches (no result, HTTP error, empty content, success)
- fetch_animeplanet_character URL normalization + success/failure
- fetch_animeplanet_characters batch flows (all cached, all live, partial, chunked, on_result)
- _role_from_header / _parse_character_refs (refs crawler pure functions)
- _fetch_refs_data async branches
- fetch_animeplanet_character_refs public wrapper
- character_from_animeplanet mapper (roles, VAs, attributes, dedup)
- main() CLI
"""

import json
from unittest.mock import AsyncMock

import pytest

from enrichment.crawlers.anime_planet.anime_planet_character_crawler import (
    _CHARACTER_BATCH_SIZE,
    _build_character,
    _extract_alt_names,
    _extract_anime_roles,
    _extract_description,
    _extract_entry_bar,
    _extract_manga_roles,
    _extract_metadata,
    _extract_tags,
    _extract_vas_from_cell,
    _fetch_character_data,
    _get_character_schema,
    _parse_loved_count,
    _parse_rank,
    _strip_tags,
    fetch_animeplanet_character,
    fetch_animeplanet_characters,
)
from enrichment.crawlers.anime_planet.anime_planet_character_models import (
    AnimePlanetCharacter,
    AnimePlanetCharacterAnimeRole,
    AnimePlanetCharacterMangaRole,
    AnimePlanetVoiceActor,
)
from enrichment.crawlers.anime_planet.anime_planet_character_refs_crawler import (
    _fetch_refs_data,
    _parse_character_refs,
    _role_from_header,
    fetch_animeplanet_character_refs,
)
from enrichment.mappers.animeplanet_mapper import character_from_animeplanet

pytestmark = pytest.mark.asyncio


# =============================================================================
# _strip_tags
# =============================================================================


def test_strip_tags_removes_html() -> None:
    assert _strip_tags("<b>Hello</b>") == "Hello"


def test_strip_tags_unescapes_entities() -> None:
    assert _strip_tags("&amp;&#39;") == "&'"


def test_strip_tags_empty() -> None:
    assert _strip_tags("") == ""


# =============================================================================
# _parse_rank
# =============================================================================


def test_parse_rank_none() -> None:
    assert _parse_rank(None) is None


def test_parse_rank_empty() -> None:
    assert _parse_rank("") is None


def test_parse_rank_with_prefix() -> None:
    assert _parse_rank("Rank #42") == 42


def test_parse_rank_plain_number() -> None:
    assert _parse_rank("157") == 157


def test_parse_rank_no_digits() -> None:
    assert _parse_rank("no digits here") is None


# =============================================================================
# _parse_loved_count
# =============================================================================


def test_parse_loved_count_none() -> None:
    assert _parse_loved_count(None) is None


def test_parse_loved_count_empty() -> None:
    assert _parse_loved_count("") is None


def test_parse_loved_count_with_commas() -> None:
    assert _parse_loved_count("36,485 users") == 36485


def test_parse_loved_count_plain() -> None:
    assert _parse_loved_count("1000 users") == 1000


def test_parse_loved_count_no_digits() -> None:
    assert _parse_loved_count("users") is None


# =============================================================================
# _extract_entry_bar
# =============================================================================

_ENTRY_BAR_BOTH = '<section class="entryBar">Gender: Male<br>Hair Color: Black</section>'
_ENTRY_BAR_GENDER_ONLY = '<section class="entryBar someExtraClass">Gender: Female</section>'
_ENTRY_BAR_HAIR_ONLY = '<section class="someClass entryBar">Hair Color: Blonde</section>'
_ENTRY_BAR_NEITHER = '<section class="entryBar">Species: Human</section>'


def test_extract_entry_bar_both() -> None:
    result = _extract_entry_bar(_ENTRY_BAR_BOTH)
    assert result["gender"] == "Male"
    assert result["hair_color"] == "Black"


def test_extract_entry_bar_gender_only() -> None:
    result = _extract_entry_bar(_ENTRY_BAR_GENDER_ONLY)
    assert result["gender"] == "Female"
    assert result["hair_color"] is None


def test_extract_entry_bar_hair_only() -> None:
    result = _extract_entry_bar(_ENTRY_BAR_HAIR_ONLY)
    assert result["gender"] is None
    assert result["hair_color"] == "Blonde"


def test_extract_entry_bar_no_section() -> None:
    result = _extract_entry_bar("<div>No entryBar here</div>")
    assert result == {"gender": None, "hair_color": None}


def test_extract_entry_bar_no_gender_or_hair() -> None:
    result = _extract_entry_bar(_ENTRY_BAR_NEITHER)
    assert result["gender"] is None
    assert result["hair_color"] is None


# =============================================================================
# _extract_metadata
# =============================================================================

_METADATA_MULTI = """
<h3 class="EntryMetadata__title">Eye Color</h3>
<div class="EntryMetadata__value">Black</div>
<h3 class="EntryMetadata__title">Birthday</h3>
<div class="EntryMetadata__value">May 5</div>
"""

_METADATA_EMPTY_KEY = """
<h3 class="EntryMetadata__title">   </h3>
<div class="EntryMetadata__value">Orphan</div>
"""


def test_extract_metadata_multiple() -> None:
    result = _extract_metadata(_METADATA_MULTI)
    assert result == {"Eye Color": "Black", "Birthday": "May 5"}


def test_extract_metadata_empty_html() -> None:
    assert _extract_metadata("") == {}


def test_extract_metadata_skips_empty_key() -> None:
    assert _extract_metadata(_METADATA_EMPTY_KEY) == {}


# =============================================================================
# _extract_alt_names
# =============================================================================


def test_extract_alt_names_comma_separated() -> None:
    html = '<h2 class="aka">Aka: Straw Hat, Monkey Luffy</h2>'
    assert _extract_alt_names(html) == ["Straw Hat", "Monkey Luffy"]


def test_extract_alt_names_single() -> None:
    html = '<h2 class="aka">Aka: Straw Hat</h2>'
    assert _extract_alt_names(html) == ["Straw Hat"]


def test_extract_alt_names_none() -> None:
    assert _extract_alt_names("<p>No aka here</p>") == []


def test_extract_alt_names_strips_inner_html() -> None:
    html = '<h2 class="aka">Aka: <a href="#">Straw Hat</a>, Luffy</h2>'
    assert _extract_alt_names(html) == ["Straw Hat", "Luffy"]


# =============================================================================
# _extract_description
# =============================================================================


def test_extract_description_plain() -> None:
    html = '<div itemprop="description">A rubber man who dreams of becoming Pirate King.</div>'
    assert _extract_description(html) == "A rubber man who dreams of becoming Pirate King."


def test_extract_description_strips_tags() -> None:
    html = '<div itemprop="description">Main <b>character</b> of <a href="#">One Piece</a>.</div>'
    assert _extract_description(html) == "Main character of One Piece."


def test_extract_description_no_match() -> None:
    assert _extract_description("<p>No description here</p>") is None


def test_extract_description_whitespace_only() -> None:
    html = '<div itemprop="description">   <span>  </span>  </div>'
    assert _extract_description(html) is None


# =============================================================================
# _extract_tags
# =============================================================================


def test_extract_tags_multiple() -> None:
    html = (
        '<a href="/characters/tags/pirate">Pirate</a>'
        '<a href="/characters/tags/rubber-powers">Rubber Powers</a>'
    )
    assert _extract_tags(html) == ["Pirate", "Rubber Powers"]


def test_extract_tags_none() -> None:
    assert _extract_tags("<p>No tags here</p>") == []


def test_extract_tags_html_entities() -> None:
    html = '<a href="/characters/tags/fire-user">Fire &amp; Ice</a>'
    assert _extract_tags(html) == ["Fire & Ice"]


# =============================================================================
# _extract_vas_from_cell
# =============================================================================

_VA_CELL_JP_US = """
<div class="flag flagJP"><a href="/people/mayumi-tanaka">Mayumi Tanaka</a></div>
<div class="flag flagUS"><a href="/people/colleen-clinkenbeard">Colleen Clinkenbeard</a></div>
"""

_VA_CELL_ALL_LANGS = """
<div class="flag flagJP"><a href="/people/jp-va">JP VA</a></div>
<div class="flag flagUS"><a href="/people/us-va">US VA</a></div>
<div class="flag flagES"><a href="/people/es-va">ES VA</a></div>
<div class="flag flagFR"><a href="/people/fr-va">FR VA</a></div>
<div class="flag flagDE"><a href="/people/de-va">DE VA</a></div>
<div class="flag flagKO"><a href="/people/ko-va">KO VA</a></div>
"""

_VA_CELL_MULTI_SAME_LANG = """
<div class="flag flagUS"><a href="/people/va1">VA One</a></div>
<div class="flag flagUS"><a href="/people/va2">VA Two</a></div>
"""


def test_extract_vas_jp_us() -> None:
    result = _extract_vas_from_cell(_VA_CELL_JP_US)
    assert "jp" in result and "us" in result
    assert result["jp"][0].name == "Mayumi Tanaka"
    assert result["us"][0].name == "Colleen Clinkenbeard"


def test_extract_vas_all_six_languages() -> None:
    result = _extract_vas_from_cell(_VA_CELL_ALL_LANGS)
    assert set(result.keys()) == {"jp", "us", "es", "fr", "de", "ko"}


def test_extract_vas_multiple_same_language() -> None:
    result = _extract_vas_from_cell(_VA_CELL_MULTI_SAME_LANG)
    assert len(result["us"]) == 2
    names = [va.name for va in result["us"]]
    assert "VA One" in names and "VA Two" in names


def test_extract_vas_empty_cell() -> None:
    assert _extract_vas_from_cell("") == {}


def test_extract_vas_url_stored() -> None:
    result = _extract_vas_from_cell(_VA_CELL_JP_US)
    assert result["jp"][0].url == "/people/mayumi-tanaka"


# =============================================================================
# _extract_anime_roles
# =============================================================================

_ANIME_ROLES_HTML = """
<h3>Anime Roles</h3>
<table><tbody>
<tr>
<td><a href="/anime/one-piece">One Piece</a></td>
<td>Main</td>
<td><div class="flag flagJP"><a href="/people/mayumi-tanaka">Mayumi Tanaka</a></div></td>
</tr>
<tr>
<td><a href="/anime/one-piece-film">One Piece Film</a></td>
<td></td>
<td></td>
</tr>
</tbody></table>
"""

_ANIME_ROLES_TWO_CELLS = """
<h3>Anime Roles</h3>
<table><tbody>
<tr>
<td><a href="/anime/test-anime">Test Anime</a></td>
<td>Secondary</td>
</tr>
</tbody></table>
"""

_ANIME_ROLES_BAD_ROWS = """
<h3>Anime Roles</h3>
<table><tbody>
<tr><td>Only one cell</td></tr>
<tr><td>No href here at all</td><td>Main</td></tr>
<tr><td><img src="x.jpg"></td><td>Main</td></tr>
</tbody></table>
"""


def test_extract_anime_roles_title_url_role() -> None:
    roles = _extract_anime_roles(_ANIME_ROLES_HTML)
    assert roles[0].title == "One Piece"
    assert roles[0].url == "/anime/one-piece"
    assert roles[0].role == "Main"


def test_extract_anime_roles_voice_actors_parsed() -> None:
    roles = _extract_anime_roles(_ANIME_ROLES_HTML)
    assert "jp" in roles[0].voice_actors
    assert roles[0].voice_actors["jp"][0].name == "Mayumi Tanaka"


def test_extract_anime_roles_empty_role_is_none() -> None:
    roles = _extract_anime_roles(_ANIME_ROLES_HTML)
    assert roles[1].role is None


def test_extract_anime_roles_no_section() -> None:
    assert _extract_anime_roles("<p>Nothing here</p>") == []


def test_extract_anime_roles_skips_bad_rows() -> None:
    assert _extract_anime_roles(_ANIME_ROLES_BAD_ROWS) == []


def test_extract_anime_roles_two_cells_no_actors() -> None:
    roles = _extract_anime_roles(_ANIME_ROLES_TWO_CELLS)
    assert len(roles) == 1
    assert roles[0].voice_actors == {}


# =============================================================================
# _extract_manga_roles
# =============================================================================

_MANGA_ROLES_HTML = """
<h3>Manga Roles</h3>
<table><tbody>
<tr>
<td><a href="/manga/one-piece">One Piece</a></td>
<td>Main</td>
</tr>
</tbody></table>
"""


def test_extract_manga_roles_basic() -> None:
    roles = _extract_manga_roles(_MANGA_ROLES_HTML)
    assert len(roles) == 1
    assert roles[0].title == "One Piece"
    assert roles[0].url == "/manga/one-piece"
    assert roles[0].role == "Main"


def test_extract_manga_roles_no_section() -> None:
    assert _extract_manga_roles("<p>Nothing</p>") == []


def test_extract_manga_roles_skips_row_without_href() -> None:
    html = """
    <h3>Manga Roles</h3>
    <table><tbody>
    <tr><td>No link here</td><td>Main</td></tr>
    <tr><td><a href="/manga/real">Real Title</a></td><td>Supporting</td></tr>
    </tbody></table>
    """
    roles = _extract_manga_roles(html)
    assert len(roles) == 1
    assert roles[0].title == "Real Title"


def test_extract_manga_roles_skips_single_cell_row() -> None:
    html = """
    <h3>Manga Roles</h3>
    <table><tbody>
    <tr><td>Single cell</td></tr>
    </tbody></table>
    """
    assert _extract_manga_roles(html) == []


def test_extract_manga_roles_empty_role_is_none() -> None:
    html = """
    <h3>Manga Roles</h3>
    <table><tbody>
    <tr>
    <td><a href="/manga/test">Test</a></td>
    <td></td>
    </tr>
    </tbody></table>
    """
    roles = _extract_manga_roles(html)
    assert roles[0].role is None


# =============================================================================
# _build_character
# =============================================================================

_BUILD_HTML = """
<section class="entryBar">Gender: Male<br>Hair Color: Black</section>
<h2 class="aka">Aka: Straw Hat</h2>
<div itemprop="description">Pirate who dreams of being king.</div>
<a href="/characters/tags/pirate">Pirate</a>
<h3 class="EntryMetadata__title">Birthday</h3>
<div class="EntryMetadata__value">May 5</div>
<h3>Anime Roles</h3>
<table><tbody>
<tr>
<td><a href="/anime/one-piece">One Piece</a></td>
<td>Main</td>
<td><div class="flag flagJP"><a href="/people/mayumi-tanaka">Mayumi Tanaka</a></div></td>
</tr>
</tbody></table>
"""

_CHAR_URL = "https://www.anime-planet.com/characters/monkey-d-luffy"


def test_build_character_slug_extracted() -> None:
    char = _build_character({}, "", _CHAR_URL)
    assert char.slug == "monkey-d-luffy"
    assert char.url == _CHAR_URL


def test_build_character_full_assembly() -> None:
    raw = {
        "name": "Monkey D. Luffy",
        "image": "https://cdn.ap.com/luffy.jpg",
        "loved_rank": "42",
        "hated_rank": "7",
    }
    char = _build_character(raw, _BUILD_HTML, _CHAR_URL)
    assert char.name == "Monkey D. Luffy"
    assert char.image == "https://cdn.ap.com/luffy.jpg"
    assert char.loved_rank == 42
    assert char.hated_rank == 7
    assert char.gender == "Male"
    assert char.hair_color == "Black"
    assert char.alt_names == ["Straw Hat"]
    assert char.description == "Pirate who dreams of being king."
    assert "Pirate" in char.tags
    assert char.attributes.get("Birthday") == "May 5"
    assert len(char.anime_roles) == 1


def test_build_character_empty_raw() -> None:
    char = _build_character({}, "", _CHAR_URL)
    assert char.name == ""
    assert char.image is None
    assert char.loved_rank is None
    assert char.tags == []
    assert char.anime_roles == []


# =============================================================================
# _get_character_schema
# =============================================================================


def test_get_character_schema_structure() -> None:
    schema = _get_character_schema()
    assert schema["name"] == "AnimePlanetCharacter"
    assert schema["baseSelector"] == "//body"
    field_names = [f["name"] for f in schema["fields"]]
    assert "name" in field_names
    assert "image" in field_names
    assert "loved_rank" in field_names
    assert "hated_rank" in field_names
    assert "loved_count" in field_names


def test_get_character_schema_no_body_html_field() -> None:
    # body_html XPath extraction returns empty — must NOT be in schema
    schema = _get_character_schema()
    field_names = [f["name"] for f in schema["fields"]]
    assert "body_html" not in field_names


# =============================================================================
# _fetch_character_data — async branches
# =============================================================================


async def test_fetch_character_data_no_results(mocker) -> None:
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )
    mocker.patch(
        "enrichment.crawlers.anime_planet.anime_planet_character_crawler.crawl_batch_urls",
        return_value=[],
    )
    result = await _fetch_character_data("https://www.anime-planet.com/characters/test")
    assert result is None


async def test_fetch_character_data_http_error(mocker) -> None:
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )
    mocker.patch(
        "enrichment.crawlers.anime_planet.anime_planet_character_crawler.crawl_batch_urls",
        return_value=[{"status_code": 403, "extracted_content": None, "html": ""}],
    )
    result = await _fetch_character_data("https://www.anime-planet.com/characters/test")
    assert result is None


async def test_fetch_character_data_empty_content(mocker) -> None:
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )
    mocker.patch(
        "enrichment.crawlers.anime_planet.anime_planet_character_crawler.crawl_batch_urls",
        return_value=[{"status_code": 200, "extracted_content": "[]", "html": ""}],
    )
    result = await _fetch_character_data("https://www.anime-planet.com/characters/test")
    assert result is None


async def test_fetch_character_data_success(mocker) -> None:
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )
    raw_fields = {"name": "Luffy", "image": None}
    mocker.patch(
        "enrichment.crawlers.anime_planet.anime_planet_character_crawler.crawl_batch_urls",
        return_value=[{
            "status_code": 200,
            "extracted_content": json.dumps([raw_fields]),
            "html": "<body>page html</body>",
        }],
    )
    result = await _fetch_character_data("https://www.anime-planet.com/characters/test")
    assert result is not None
    assert result["name"] == "Luffy"
    assert result["_html"] == "<body>page html</body>"


# =============================================================================
# fetch_animeplanet_character
# =============================================================================


async def test_fetch_animeplanet_character_returns_none_on_failure(mocker) -> None:
    mocker.patch(
        "enrichment.crawlers.anime_planet.anime_planet_character_crawler._fetch_character_data",
        new_callable=AsyncMock,
        return_value=None,
    )
    assert await fetch_animeplanet_character("https://www.anime-planet.com/characters/test") is None


async def test_fetch_animeplanet_character_returns_character(mocker) -> None:
    mocker.patch(
        "enrichment.crawlers.anime_planet.anime_planet_character_crawler._fetch_character_data",
        new_callable=AsyncMock,
        return_value={"name": "Luffy", "_html": ""},
    )
    result = await fetch_animeplanet_character("https://www.anime-planet.com/characters/monkey-d-luffy")
    assert result is not None
    assert result.name == "Luffy"


@pytest.mark.parametrize("url, expected_fragment", [
    ("/characters/monkey-d-luffy", "https://www.anime-planet.com/characters/monkey-d-luffy"),
    ("monkey-d-luffy", "https://www.anime-planet.com/characters/monkey-d-luffy"),
    ("https://www.anime-planet.com/characters/monkey-d-luffy", "https://www.anime-planet.com/characters/monkey-d-luffy"),
])
async def test_fetch_animeplanet_character_url_normalization(url, expected_fragment, mocker) -> None:
    captured: list[str] = []

    async def _fake_fetch(u: str):
        captured.append(u)
        return {"name": "X", "_html": ""}

    mocker.patch(
        "enrichment.crawlers.anime_planet.anime_planet_character_crawler._fetch_character_data",
        side_effect=_fake_fetch,
    )
    await fetch_animeplanet_character(url)
    assert captured[0] == expected_fragment


# =============================================================================
# fetch_animeplanet_characters — batch flows
# =============================================================================


async def test_fetch_animeplanet_characters_empty_list() -> None:
    assert await fetch_animeplanet_characters([]) == []


async def test_fetch_animeplanet_characters_all_cached(mocker) -> None:
    url = "https://www.anime-planet.com/characters/luffy"
    mocker.patch.object(
        _fetch_character_data,
        "cache_batch_get",
        new=AsyncMock(return_value=([{"name": "Luffy", "_html": ""}], [])),
    )
    result = await fetch_animeplanet_characters([url])
    assert len(result) == 1
    assert result[0] is not None
    assert result[0].name == "Luffy"


async def test_fetch_animeplanet_characters_cached_fires_on_result(mocker) -> None:
    url = "https://www.anime-planet.com/characters/luffy"
    mocker.patch.object(
        _fetch_character_data,
        "cache_batch_get",
        new=AsyncMock(return_value=([{"name": "Luffy", "_html": ""}], [])),
    )
    fired: list[AnimePlanetCharacter] = []
    await fetch_animeplanet_characters([url], on_result=fired.append)
    assert len(fired) == 1
    assert fired[0].name == "Luffy"


async def test_fetch_animeplanet_characters_live_success(mocker) -> None:
    url = "https://www.anime-planet.com/characters/zoro"
    mocker.patch.object(
        _fetch_character_data,
        "cache_batch_get",
        new=AsyncMock(return_value=([None], [0])),
    )
    cache_set = AsyncMock()
    mocker.patch.object(_fetch_character_data, "cache_batch_set", new=cache_set)
    mocker.patch(
        "enrichment.crawlers.anime_planet.anime_planet_character_crawler.crawl_batch_urls",
        return_value=[{
            "url": url,
            "status_code": 200,
            "extracted_content": json.dumps([{"name": "Zoro"}]),
            "html": "",
        }],
    )
    result = await fetch_animeplanet_characters([url])
    assert result[0] is not None
    assert result[0].name == "Zoro"
    cache_set.assert_awaited_once()


async def test_fetch_animeplanet_characters_live_fires_on_result(mocker) -> None:
    url = "https://www.anime-planet.com/characters/nami"
    mocker.patch.object(
        _fetch_character_data,
        "cache_batch_get",
        new=AsyncMock(return_value=([None], [0])),
    )
    mocker.patch.object(_fetch_character_data, "cache_batch_set", new=AsyncMock())
    mocker.patch(
        "enrichment.crawlers.anime_planet.anime_planet_character_crawler.crawl_batch_urls",
        return_value=[{
            "url": url,
            "status_code": 200,
            "extracted_content": json.dumps([{"name": "Nami"}]),
            "html": "",
        }],
    )
    fired: list[AnimePlanetCharacter] = []
    await fetch_animeplanet_characters([url], on_result=fired.append)
    assert len(fired) == 1
    assert fired[0].name == "Nami"


async def test_fetch_animeplanet_characters_none_result(mocker) -> None:
    url = "https://www.anime-planet.com/characters/unknown"
    mocker.patch.object(
        _fetch_character_data,
        "cache_batch_get",
        new=AsyncMock(return_value=([None], [0])),
    )
    mocker.patch.object(_fetch_character_data, "cache_batch_set", new=AsyncMock())
    mocker.patch(
        "enrichment.crawlers.anime_planet.anime_planet_character_crawler.crawl_batch_urls",
        return_value=[None],
    )
    assert await fetch_animeplanet_characters([url]) == [None]


async def test_fetch_animeplanet_characters_http_error(mocker) -> None:
    url = "https://www.anime-planet.com/characters/blocked"
    mocker.patch.object(
        _fetch_character_data,
        "cache_batch_get",
        new=AsyncMock(return_value=([None], [0])),
    )
    mocker.patch.object(_fetch_character_data, "cache_batch_set", new=AsyncMock())
    mocker.patch(
        "enrichment.crawlers.anime_planet.anime_planet_character_crawler.crawl_batch_urls",
        return_value=[{"url": url, "status_code": 403, "extracted_content": None}],
    )
    assert await fetch_animeplanet_characters([url]) == [None]


async def test_fetch_animeplanet_characters_empty_extracted_content(mocker) -> None:
    url = "https://www.anime-planet.com/characters/empty"
    mocker.patch.object(
        _fetch_character_data,
        "cache_batch_get",
        new=AsyncMock(return_value=([None], [0])),
    )
    mocker.patch.object(_fetch_character_data, "cache_batch_set", new=AsyncMock())
    mocker.patch(
        "enrichment.crawlers.anime_planet.anime_planet_character_crawler.crawl_batch_urls",
        return_value=[{"url": url, "status_code": 200, "extracted_content": "[]"}],
    )
    assert await fetch_animeplanet_characters([url]) == [None]


async def test_fetch_animeplanet_characters_partial_cache(mocker) -> None:
    url1 = "https://www.anime-planet.com/characters/char1"
    url2 = "https://www.anime-planet.com/characters/char2"
    mocker.patch.object(
        _fetch_character_data,
        "cache_batch_get",
        new=AsyncMock(return_value=([{"name": "Char1", "_html": ""}, None], [1])),
    )
    mocker.patch.object(_fetch_character_data, "cache_batch_set", new=AsyncMock())
    mocker.patch(
        "enrichment.crawlers.anime_planet.anime_planet_character_crawler.crawl_batch_urls",
        return_value=[{
            "url": url2,
            "status_code": 200,
            "extracted_content": json.dumps([{"name": "Char2"}]),
            "html": "",
        }],
    )
    result = await fetch_animeplanet_characters([url1, url2])
    assert result[0].name == "Char1"  # type: ignore[union-attr]
    assert result[1].name == "Char2"  # type: ignore[union-attr]


async def test_fetch_animeplanet_characters_chunks_into_batches(mocker) -> None:
    # _CHARACTER_BATCH_SIZE + 1 URLs forces exactly 2 batch crawl calls
    urls = [
        f"https://www.anime-planet.com/characters/char{i}"
        for i in range(_CHARACTER_BATCH_SIZE + 1)
    ]
    mocker.patch.object(
        _fetch_character_data,
        "cache_batch_get",
        new=AsyncMock(return_value=([None] * len(urls), list(range(len(urls))))),
    )
    cache_set = AsyncMock()
    mocker.patch.object(_fetch_character_data, "cache_batch_set", new=cache_set)

    async def _batch(batch_urls: list[str], **kwargs) -> list[dict]:
        return [
            {
                "url": u,
                "status_code": 200,
                "extracted_content": json.dumps([{"name": f"C{i}"}]),
                "html": "",
            }
            for i, u in enumerate(batch_urls)
        ]

    mocker.patch(
        "enrichment.crawlers.anime_planet.anime_planet_character_crawler.crawl_batch_urls",
        side_effect=_batch,
    )
    result = await fetch_animeplanet_characters(urls)
    assert len(result) == len(urls)
    assert all(c is not None for c in result)
    assert cache_set.await_count == 2


async def test_fetch_animeplanet_characters_normalizes_relative_urls(mocker) -> None:
    mocker.patch.object(
        _fetch_character_data,
        "cache_batch_get",
        new=AsyncMock(return_value=([None], [0])),
    )
    mocker.patch.object(_fetch_character_data, "cache_batch_set", new=AsyncMock())
    captured: list[str] = []

    async def _batch(batch_urls: list[str], **kwargs) -> list[None]:
        captured.extend(batch_urls)
        return [None]

    mocker.patch(
        "enrichment.crawlers.anime_planet.anime_planet_character_crawler.crawl_batch_urls",
        side_effect=_batch,
    )
    await fetch_animeplanet_characters(["/characters/luffy"])
    assert captured[0] == "https://www.anime-planet.com/characters/luffy"


async def test_fetch_animeplanet_characters_normalizes_bare_slug(mocker) -> None:
    mocker.patch.object(
        _fetch_character_data,
        "cache_batch_get",
        new=AsyncMock(return_value=([None], [0])),
    )
    mocker.patch.object(_fetch_character_data, "cache_batch_set", new=AsyncMock())
    captured: list[str] = []

    async def _batch(batch_urls: list[str], **kwargs) -> list[None]:
        captured.extend(batch_urls)
        return [None]

    mocker.patch(
        "enrichment.crawlers.anime_planet.anime_planet_character_crawler.crawl_batch_urls",
        side_effect=_batch,
    )
    await fetch_animeplanet_characters(["luffy"])
    assert captured[0] == "https://www.anime-planet.com/characters/luffy"


# =============================================================================
# Refs crawler — _role_from_header
# =============================================================================


@pytest.mark.parametrize("header, expected", [
    ("Main Characters", "Main"),
    ("Secondary Characters", "Secondary"),
    ("Minor Characters", "Minor"),
    ("Unknown Role Type", "Unknown Role Type"),  # full text returned — first word not recognised
    ("", ""),
])
def test_role_from_header(header: str, expected: str) -> None:
    assert _role_from_header(header) == expected


def test_role_from_header_strips_html() -> None:
    assert _role_from_header("<span>Main</span> Characters") == "Main"


# =============================================================================
# Refs crawler — _parse_character_refs
# =============================================================================

_REFS_HREF_FIRST = """
<h3 class="sub">Main Characters</h3>
<a href="/characters/monkey-d-luffy" class="name">Monkey D. Luffy</a>
<a href="/characters/roronoa-zoro" class="name">Roronoa Zoro</a>
<h3 class="sub">Secondary Characters</h3>
<a href="/characters/nami" class="name">Nami</a>
"""

_REFS_CLASS_FIRST = """
<h3 class="sub">Main Characters</h3>
<a class="name" href="/characters/monkey-d-luffy">Monkey D. Luffy</a>
"""


def test_parse_character_refs_href_first_order() -> None:
    refs = _parse_character_refs(_REFS_HREF_FIRST)
    assert len(refs) == 3
    assert refs[0] == {"url": "/characters/monkey-d-luffy", "role": "Main"}
    assert refs[2] == {"url": "/characters/nami", "role": "Secondary"}


def test_parse_character_refs_class_first_order() -> None:
    refs = _parse_character_refs(_REFS_CLASS_FIRST)
    assert len(refs) == 1
    assert refs[0]["url"] == "/characters/monkey-d-luffy"
    assert refs[0]["role"] == "Main"


def test_parse_character_refs_no_sections() -> None:
    assert _parse_character_refs("<p>No sections here</p>") == []


def test_parse_character_refs_empty_section() -> None:
    html = '<h3 class="sub">Main Characters</h3><p>No character links</p>'
    assert _parse_character_refs(html) == []


def test_parse_character_refs_preserves_document_order() -> None:
    refs = _parse_character_refs(_REFS_HREF_FIRST)
    urls = [r["url"] for r in refs]
    assert urls == [
        "/characters/monkey-d-luffy",
        "/characters/roronoa-zoro",
        "/characters/nami",
    ]


# =============================================================================
# _fetch_refs_data — async branches
# =============================================================================


async def test_fetch_refs_data_no_result(mocker) -> None:
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )
    mocker.patch(
        "enrichment.crawlers.anime_planet.anime_planet_character_refs_crawler.crawl_single_url",
        return_value=None,
    )
    result = await _fetch_refs_data("https://www.anime-planet.com/anime/dandadan/characters")
    assert result is None


async def test_fetch_refs_data_http_error(mocker) -> None:
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )
    mocker.patch(
        "enrichment.crawlers.anime_planet.anime_planet_character_refs_crawler.crawl_single_url",
        return_value={"status_code": 404, "html": ""},
    )
    result = await _fetch_refs_data("https://www.anime-planet.com/anime/dandadan/characters")
    assert result is None


async def test_fetch_refs_data_empty_html(mocker) -> None:
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )
    mocker.patch(
        "enrichment.crawlers.anime_planet.anime_planet_character_refs_crawler.crawl_single_url",
        return_value={"status_code": 200, "html": ""},
    )
    result = await _fetch_refs_data("https://www.anime-planet.com/anime/dandadan/characters")
    assert result is None


async def test_fetch_refs_data_no_refs_returns_none(mocker) -> None:
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )
    mocker.patch(
        "enrichment.crawlers.anime_planet.anime_planet_character_refs_crawler.crawl_single_url",
        return_value={"status_code": 200, "html": "<body>no character sections</body>"},
    )
    result = await _fetch_refs_data("https://www.anime-planet.com/anime/dandadan/characters")
    assert result is None


async def test_fetch_refs_data_success(mocker) -> None:
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )
    mocker.patch(
        "enrichment.crawlers.anime_planet.anime_planet_character_refs_crawler.crawl_single_url",
        return_value={"status_code": 200, "html": _REFS_HREF_FIRST},
    )
    result = await _fetch_refs_data("https://www.anime-planet.com/anime/dandadan/characters")
    assert result is not None
    assert len(result) == 3


# =============================================================================
# fetch_animeplanet_character_refs — public wrapper
# =============================================================================


async def test_fetch_animeplanet_character_refs_empty_on_none(mocker) -> None:
    mocker.patch(
        "enrichment.crawlers.anime_planet.anime_planet_character_refs_crawler._fetch_refs_data",
        new_callable=AsyncMock,
        return_value=None,
    )
    result = await fetch_animeplanet_character_refs(
        "https://www.anime-planet.com/anime/test/characters"
    )
    assert result == []


async def test_fetch_animeplanet_character_refs_returns_refs(mocker) -> None:
    refs = [{"url": "/characters/luffy", "role": "Main"}]
    mocker.patch(
        "enrichment.crawlers.anime_planet.anime_planet_character_refs_crawler._fetch_refs_data",
        new_callable=AsyncMock,
        return_value=refs,
    )
    result = await fetch_animeplanet_character_refs(
        "https://www.anime-planet.com/anime/test/characters"
    )
    assert result == refs


# =============================================================================
# character_from_animeplanet mapper
# =============================================================================


def _make_char(**kwargs) -> AnimePlanetCharacter:
    defaults: dict = {
        "name": "Test Char",
        "slug": "test-char",
        "url": "https://www.anime-planet.com/characters/test-char",
    }
    return AnimePlanetCharacter(**{**defaults, **kwargs})


def test_mapper_basic_fields() -> None:
    char = _make_char(description="A test description", tags=["Hero"])
    result = character_from_animeplanet(char)
    assert result["name"] == "Test Char"
    assert result["sources"] == ["https://www.anime-planet.com/characters/test-char"]
    assert result["description"] == "A test description"
    assert result["traits"] == ["Hero"]


def test_mapper_optional_fields_absent_or_empty_when_not_provided() -> None:
    result = character_from_animeplanet(_make_char())
    # None/missing fields are excluded by exclude_none=True
    assert "description" not in result
    # List fields default to [] in the Character model
    assert result.get("traits", []) == []
    assert result.get("nicknames", []) == []
    assert result.get("images", []) == []
    assert result.get("voice_actors", []) == []
    assert result.get("roles", []) == []


def test_mapper_image_wrapped_in_list() -> None:
    char = _make_char(image="https://cdn.ap.com/img.webp")
    assert character_from_animeplanet(char)["images"] == ["https://cdn.ap.com/img.webp"]


def test_mapper_nicknames_from_alt_names() -> None:
    char = _make_char(alt_names=["Straw Hat", "Luffy"])
    assert character_from_animeplanet(char)["nicknames"] == ["Straw Hat", "Luffy"]


def test_mapper_attribute_key_normalization() -> None:
    char = _make_char(attributes={"Eye Color": "Black", "Hair Color": "Red"})
    attrs = character_from_animeplanet(char)["attributes"]
    assert "eye_color" in attrs
    assert "hair_color" in attrs
    assert "Eye Color" not in attrs
    assert "Hair Color" not in attrs


def test_mapper_gender_hair_in_attributes() -> None:
    char = _make_char(gender="Male", hair_color="Black")
    attrs = character_from_animeplanet(char)["attributes"]
    assert attrs["gender"] == "Male"
    assert attrs["hair_color"] == "Black"


def test_mapper_ranks_in_attributes_as_strings() -> None:
    char = _make_char(loved_rank=10, hated_rank=5)
    attrs = character_from_animeplanet(char)["attributes"]
    assert attrs["loved_rank"] == "10"
    assert attrs["hated_rank"] == "5"


def test_mapper_loved_count_maps_to_favorites() -> None:
    char = _make_char(loved_count=36485)
    assert character_from_animeplanet(char)["favorites"] == 36485


def test_mapper_loved_count_none_excluded() -> None:
    char = _make_char()
    assert "favorites" not in character_from_animeplanet(char)


def test_mapper_roles_aggregated_from_anime_and_manga() -> None:
    char = _make_char(
        anime_roles=[AnimePlanetCharacterAnimeRole(title="A", url="/anime/a", role="Main")],
        manga_roles=[AnimePlanetCharacterMangaRole(title="B", url="/manga/b", role="Minor")],
    )
    roles = character_from_animeplanet(char)["roles"]
    assert "MAIN" in roles
    assert "BACKGROUND" in roles  # Minor → BACKGROUND via _missing_


def test_mapper_role_dedup() -> None:
    char = _make_char(
        anime_roles=[
            AnimePlanetCharacterAnimeRole(title="A", url="/anime/a", role="Main"),
            AnimePlanetCharacterAnimeRole(title="B", url="/anime/b", role="Main"),
        ]
    )
    roles = character_from_animeplanet(char)["roles"]
    assert roles.count("MAIN") == 1


def test_mapper_animeography_built() -> None:
    char = _make_char(
        anime_roles=[
            AnimePlanetCharacterAnimeRole(title="One Piece", url="/anime/one-piece", role="Main")
        ]
    )
    ography = character_from_animeplanet(char)["animeography"]
    assert ography[0]["title"] == "One Piece"
    assert ography[0]["sources"] == ["https://www.anime-planet.com/anime/one-piece"]
    assert ography[0]["role"] == "MAIN"


def test_mapper_mangaography_built() -> None:
    char = _make_char(
        manga_roles=[
            AnimePlanetCharacterMangaRole(title="One Piece", url="/manga/one-piece", role="Main")
        ]
    )
    ography = character_from_animeplanet(char)["mangaography"]
    assert ography[0]["title"] == "One Piece"
    assert ography[0]["sources"] == ["https://www.anime-planet.com/manga/one-piece"]


def test_mapper_voice_actors_dedup_across_anime_roles() -> None:
    va = AnimePlanetVoiceActor(name="Mayumi Tanaka", url="/people/mayumi-tanaka")
    char = _make_char(
        anime_roles=[
            AnimePlanetCharacterAnimeRole(
                title="One Piece", url="/anime/one-piece", role="Main",
                voice_actors={"jp": [va]},
            ),
            AnimePlanetCharacterAnimeRole(
                title="Film", url="/anime/film", role="Main",
                voice_actors={"jp": [va]},
            ),
        ]
    )
    jp_vas = [
        v for v in character_from_animeplanet(char)["voice_actors"]
        if v["language"] == "Japanese"
    ]
    assert len(jp_vas) == 1


def test_mapper_voice_actors_flag_to_language() -> None:
    char = _make_char(
        anime_roles=[
            AnimePlanetCharacterAnimeRole(
                title="A", url="/anime/a", role="Main",
                voice_actors={
                    "jp": [AnimePlanetVoiceActor(name="JP", url="/people/jp")],
                    "us": [AnimePlanetVoiceActor(name="US", url="/people/us")],
                    "es": [AnimePlanetVoiceActor(name="ES", url="/people/es")],
                    "fr": [AnimePlanetVoiceActor(name="FR", url="/people/fr")],
                    "de": [AnimePlanetVoiceActor(name="DE", url="/people/de")],
                    "ko": [AnimePlanetVoiceActor(name="KO", url="/people/ko")],
                },
            )
        ]
    )
    languages = {v["language"] for v in character_from_animeplanet(char)["voice_actors"]}
    assert languages == {"Japanese", "English", "Spanish", "French", "German", "Korean"}


def test_mapper_voice_actor_sources_include_base_url() -> None:
    char = _make_char(
        anime_roles=[
            AnimePlanetCharacterAnimeRole(
                title="A", url="/anime/a", role="Main",
                voice_actors={"jp": [AnimePlanetVoiceActor(name="VA", url="/people/va")]},
            )
        ]
    )
    sources = character_from_animeplanet(char)["voice_actors"][0]["sources"]
    assert sources == ["https://www.anime-planet.com/people/va"]


# =============================================================================
# main() CLI
# =============================================================================


async def test_main_returns_1_when_no_character(mocker, tmp_path) -> None:
    mocker.patch(
        "sys.argv",
        ["prog", "https://www.anime-planet.com/characters/test", "--output", str(tmp_path / "out.json")],
    )
    mocker.patch(
        "enrichment.crawlers.anime_planet.anime_planet_character_crawler.fetch_animeplanet_character",
        new_callable=AsyncMock,
        return_value=None,
    )
    from enrichment.crawlers.anime_planet.anime_planet_character_crawler import main
    assert await main() == 1


async def test_main_returns_0_and_writes_file(mocker, tmp_path) -> None:
    char = AnimePlanetCharacter(
        name="Luffy",
        slug="monkey-d-luffy",
        url="https://www.anime-planet.com/characters/monkey-d-luffy",
    )
    output = tmp_path / "out.json"
    mocker.patch(
        "sys.argv",
        ["prog", "https://www.anime-planet.com/characters/monkey-d-luffy", "--output", str(output)],
    )
    mocker.patch(
        "enrichment.crawlers.anime_planet.anime_planet_character_crawler.fetch_animeplanet_character",
        new_callable=AsyncMock,
        return_value=char,
    )
    from enrichment.crawlers.anime_planet.anime_planet_character_crawler import main
    assert await main() == 0
    assert output.exists()
    data = json.loads(output.read_text())
    assert data["name"] == "Luffy"
