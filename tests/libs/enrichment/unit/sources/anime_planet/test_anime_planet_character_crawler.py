"""Unit tests for anime_planet_character_crawler.py.

Fixture-grounded tests use the monkey-d-luffy HTML fixture (2026-06-10).
Edge-case tests use synthetic inline HTML snippets.
"""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from enrichment.sources.anime_planet.anime_planet_character_crawler import (
    _XPATHS,
    _build_character_from_raw,
    _extract_alt_names,
    _extract_anime_roles,
    _extract_character_from_html,
    _extract_description,
    _extract_entry_bar,
    _extract_manga_roles,
    _extract_metadata,
    _extract_tags,
    _extract_vas_from_cell,
    _fetch_character_data,
    _fetch_page_html,
    _parse_loved_count,
    _parse_rank,
    fetch_animeplanet_character,
    fetch_animeplanet_characters,
)
from enrichment.sources.anime_planet.anime_planet_character_models import (
    AnimePlanetVoiceActor,
)
from enrichment.sources.base.framework import NullRepository

pytestmark = pytest.mark.asyncio

_LUFFY_URL = "https://www.anime-planet.com/characters/monkey-d-luffy"
_PATCH_FETCH_DATA = (
    "enrichment.sources.anime_planet.anime_planet_character_crawler._fetch_character_data"
)
_PATCH_FETCH_PAGE = (
    "enrichment.sources.anime_planet.anime_planet_character_crawler._fetch_page_html"
)

# Minimal raw dict that passes _build_character_from_raw without a full fixture.
_MINIMAL_HTML = (
    "<html><body>"
    "<h1 itemprop='name'>Monkey D. Luffy</h1>"
    "<section class='entryBar'>Gender: Male</section>"
    "</body></html>"
)
_MINIMAL_RAW: dict[str, Any] = {
    "name": "Monkey D. Luffy",
    "image": "https://example.com/luffy.jpg",
    "loved_rank": "#12",
    "hated_rank": "#106",
    "loved_count": "37,007 users",
    "_html": _MINIMAL_HTML,
}


# =============================================================================
# _XPATHS invariant
# =============================================================================


def test_xpaths_cover_required_fields() -> None:
    for key in ("name", "image", "loved_rank", "hated_rank", "loved_count"):
        assert key in _XPATHS, f"_XPATHS missing key: {key!r}"
    assert "entryBar" in _XPATHS["loved_rank"]
    assert "entryBar" in _XPATHS["hated_rank"]


# =============================================================================
# _parse_rank
# =============================================================================


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, None),
        ("", None),
        ("no-number", None),
        ("#12", 12),
        ("Rank 42", 42),
    ],
)
def test_parse_rank(raw: str | None, expected: int | None) -> None:
    assert _parse_rank(raw) == expected


# =============================================================================
# _parse_loved_count
# =============================================================================


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, None),
        ("", None),
        ("37,007 users", 37007),
        ("1000", 1000),
    ],
)
def test_parse_loved_count(raw: str | None, expected: int | None) -> None:
    assert _parse_loved_count(raw) == expected


# =============================================================================
# _extract_entry_bar
# =============================================================================


@pytest.mark.parametrize(
    "html, expected_gender, expected_hair",
    [
        ("<html><body><p>no section</p></body></html>", None, None),
        (
            '<html><body><section class="entryBar">Gender: Female</section></body></html>',
            "Female",
            None,
        ),
        (
            '<html><body><section class="entryBar">'
            "Gender: Male Hair Color: Black</section></body></html>",
            "Male",
            "Black",
        ),
    ],
)
def test_extract_entry_bar(
    html: str, expected_gender: str | None, expected_hair: str | None
) -> None:
    result = _extract_entry_bar(html)
    assert result["gender"] == expected_gender
    assert result["hair_color"] == expected_hair


def test_extract_entry_bar_from_fixture(ap_character_html: str) -> None:
    result = _extract_entry_bar(ap_character_html)
    assert result["gender"] == "Male"
    assert result["hair_color"] == "Black"


# =============================================================================
# _extract_metadata
# =============================================================================


def test_extract_metadata_no_match() -> None:
    assert _extract_metadata("<html><body></body></html>") == {}


def test_extract_metadata_from_fixture(ap_character_html: str) -> None:
    meta = _extract_metadata(ap_character_html)
    assert "Birthday" in meta
    assert meta["Birthday"] == "May 5"


# =============================================================================
# _extract_alt_names
# =============================================================================


def test_extract_alt_names_no_match() -> None:
    assert _extract_alt_names("<html><body></body></html>") == []


def test_extract_alt_names_from_fixture(ap_character_html: str) -> None:
    assert "Straw Hat" in _extract_alt_names(ap_character_html)


def test_extract_alt_names_multiple() -> None:
    html = '<html><body><h2 class="aka">Aka: Name One, Name Two, Name Three</h2></body></html>'
    assert _extract_alt_names(html) == ["Name One", "Name Two", "Name Three"]


# =============================================================================
# _extract_description
# =============================================================================


def test_extract_description_from_fixture(ap_character_html: str) -> None:
    desc = _extract_description(ap_character_html)
    assert desc is not None
    assert "Shanks" in desc


@pytest.mark.parametrize(
    "html",
    [
        "<html><body><p>no desc div</p></body></html>",
        '<html><body><div itemprop="description">   </div></body></html>',
    ],
)
def test_extract_description_returns_none(html: str) -> None:
    assert _extract_description(html) is None


# =============================================================================
# _extract_tags
# =============================================================================


def test_extract_tags_no_tags() -> None:
    assert _extract_tags("<html><body></body></html>") == []


def test_extract_tags_with_tags() -> None:
    html = (
        "<html><body>"
        '<a href="/characters/tags/hero">Hero</a>'
        '<a href="/characters/tags/pirate">Pirate</a>'
        "</body></html>"
    )
    assert _extract_tags(html) == ["Hero", "Pirate"]


# =============================================================================
# _extract_vas_from_cell
# =============================================================================


def test_extract_vas_empty() -> None:
    assert _extract_vas_from_cell("") == {}
    assert _extract_vas_from_cell("<td>no flags here</td>") == {}


def test_extract_vas_known_flags() -> None:
    cell = (
        '<div class="flag flagJP"></div>'
        '<a href="/people/mayumi-tanaka">Mayumi TANAKA</a>'
        '<div class="flag flagUS"></div>'
        '<a href="/people/colleen-clinkenbeard">Colleen CLINKENBEARD</a>'
    )
    result = _extract_vas_from_cell(cell)
    assert result["jp"] == [
        AnimePlanetVoiceActor(name="Mayumi TANAKA", url="/people/mayumi-tanaka")
    ]
    assert result["us"][0].name == "Colleen CLINKENBEARD"


# =============================================================================
# _extract_anime_roles
# =============================================================================


def test_extract_anime_roles_no_section() -> None:
    assert _extract_anime_roles("<html><body></body></html>") == []


def test_extract_anime_roles_filters_bad_rows() -> None:
    html = (
        "<h3>Anime Roles</h3>"
        "<table><tbody>"
        "<tr><td>only one cell</td></tr>"
        '<tr><td><a href="/not-matching/slug">Title</a></td><td>Main</td></tr>'
        "</tbody></table>"
    )
    assert _extract_anime_roles(html) == []


def test_extract_anime_roles_empty_role_becomes_none() -> None:
    html = (
        "<h3>Anime Roles</h3>"
        "<table><tbody>"
        '<tr><td><a href="/anime/test-show">Test Show</a></td><td>   </td></tr>'
        "</tbody></table>"
    )
    roles = _extract_anime_roles(html)
    assert len(roles) == 1
    assert roles[0].role is None
    assert roles[0].title == "Test Show"
    assert roles[0].url == "/anime/test-show"


def test_extract_anime_roles_from_fixture(ap_character_html: str) -> None:
    roles = _extract_anime_roles(ap_character_html)
    assert len(roles) == 25
    one_piece = next(r for r in roles if "One Piece" in r.title)
    assert one_piece.role == "Main"
    assert "jp" in one_piece.voice_actors


# =============================================================================
# _extract_manga_roles
# =============================================================================


def test_extract_manga_roles_no_section() -> None:
    assert _extract_manga_roles("<html><body></body></html>") == []


def test_extract_manga_roles_filters_bad_rows() -> None:
    html = (
        "<h3>Manga Roles</h3>"
        "<table><tbody>"
        "<tr><td>only one cell</td></tr>"
        '<tr><td><a href="/not-matching/slug">Title</a></td><td>Lead</td></tr>'
        "</tbody></table>"
    )
    assert _extract_manga_roles(html) == []


def test_extract_manga_roles_from_fixture(ap_character_html: str) -> None:
    roles = _extract_manga_roles(ap_character_html)
    assert len(roles) == 16
    main = next(r for r in roles if r.title == "One Piece")
    assert main.role == "Main"
    assert main.url == "/manga/one-piece"


# =============================================================================
# _extract_character_from_html
# =============================================================================


@pytest.mark.parametrize(
    "html",
    [
        "",
        "<html><body><p>no character name element</p></body></html>",
    ],
)
def test_extract_returns_none_on_bad_html(html: str) -> None:
    assert _extract_character_from_html(html) is None


def test_extract_from_html_fixture(ap_character_html: str) -> None:
    raw = _extract_character_from_html(ap_character_html)
    assert raw is not None
    assert raw["name"] == "Monkey D. Luffy"
    assert raw["image"] is not None and "luffy" in raw["image"]
    assert raw["loved_rank"] == "#12"
    assert raw["hated_rank"] == "#106"
    assert raw["loved_count"] == "37,007 users"
    assert raw["_html"] is ap_character_html


# =============================================================================
# _build_character_from_raw
# =============================================================================


def test_build_character_from_raw_fixture(
    ap_character_html: str, ap_character_extracted: dict
) -> None:
    char = _build_character_from_raw(ap_character_extracted, ap_character_html, _LUFFY_URL)
    assert char.name == "Monkey D. Luffy"
    assert char.slug == "monkey-d-luffy"
    assert char.url == _LUFFY_URL
    assert char.loved_rank == 12
    assert char.hated_rank == 106
    assert char.loved_count == 37007
    assert char.gender == "Male"
    assert char.hair_color == "Black"
    assert char.description is not None and "Shanks" in char.description
    assert "Straw Hat" in char.alt_names
    assert len(char.anime_roles) == 25
    assert len(char.manga_roles) == 16


def test_build_character_no_image(
    ap_character_html: str, ap_character_extracted: dict
) -> None:
    raw = {**ap_character_extracted, "image": None}
    char = _build_character_from_raw(raw, ap_character_html, _LUFFY_URL)
    assert char.image is None


# =============================================================================
# _fetch_page_html
# =============================================================================


async def test_fetch_page_html_success() -> None:
    page = AsyncMock()
    page.wait_for = AsyncMock()
    page.get_content = AsyncMock(return_value="<html>luffy</html>")
    browser = AsyncMock()
    browser.get = AsyncMock(return_value=page)

    result = await _fetch_page_html(browser, _LUFFY_URL)

    assert result == "<html>luffy</html>"
    page.wait_for.assert_awaited_once()


async def test_fetch_page_html_navigation_failure() -> None:
    page = AsyncMock()
    page.wait_for = AsyncMock(side_effect=Exception("timeout"))
    browser = AsyncMock()
    browser.get = AsyncMock(return_value=page)

    result = await _fetch_page_html(browser, _LUFFY_URL)

    assert result is None


# =============================================================================
# _fetch_character_data
# =============================================================================


async def test_fetch_character_data_success(mocker: Any) -> None:
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )
    mocker.patch(_PATCH_FETCH_PAGE, new=AsyncMock(return_value=_MINIMAL_HTML))
    import zendriver as zd

    mocker.patch.object(zd, "start", new=AsyncMock(return_value=AsyncMock()))

    result = await _fetch_character_data(_LUFFY_URL)

    assert result is not None
    assert result["name"] == "Monkey D. Luffy"


async def test_fetch_character_data_no_html(mocker: Any) -> None:
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )
    mocker.patch(_PATCH_FETCH_PAGE, new=AsyncMock(return_value=None))
    import zendriver as zd

    mocker.patch.object(zd, "start", new=AsyncMock(return_value=AsyncMock()))

    assert await _fetch_character_data(_LUFFY_URL) is None


async def test_fetch_character_data_browser_stop_swallowed(mocker: Any) -> None:
    mocker.patch(
        "http_cache.result_cache.get_cache_config",
        return_value=mocker.MagicMock(cache_enabled=False),
    )
    mocker.patch(_PATCH_FETCH_PAGE, new=AsyncMock(return_value=_MINIMAL_HTML))
    browser = AsyncMock()
    browser.stop = AsyncMock(side_effect=Exception("stop failed"))
    import zendriver as zd

    mocker.patch.object(zd, "start", new=AsyncMock(return_value=browser))

    result = await _fetch_character_data(_LUFFY_URL)

    assert result is not None


# =============================================================================
# AnimePlanetCharacterCrawler
# =============================================================================


def test_crawler_get_extraction_schema() -> None:
    from enrichment.sources.anime_planet.anime_planet_character_crawler import (
        AnimePlanetCharacterCrawler,
    )

    crawler = AnimePlanetCharacterCrawler(NullRepository())
    assert crawler.get_extraction_schema() is _XPATHS


def test_crawler_normalize_identifier() -> None:
    from enrichment.sources.anime_planet.anime_planet_character_crawler import (
        AnimePlanetCharacterCrawler,
    )

    crawler = AnimePlanetCharacterCrawler(NullRepository())
    assert crawler.normalize_identifier(_LUFFY_URL) == _LUFFY_URL


def test_crawler_build_source_model(
    ap_character_html: str, ap_character_extracted: dict
) -> None:
    from enrichment.sources.anime_planet.anime_planet_character_crawler import (
        AnimePlanetCharacterCrawler,
    )

    crawler = AnimePlanetCharacterCrawler(NullRepository())
    raw = {**ap_character_extracted, "_html": ap_character_html}
    char = crawler.build_source_model(raw, _LUFFY_URL)
    assert char.name == "Monkey D. Luffy"
    assert char.slug == "monkey-d-luffy"


def test_crawler_map_to_canonical(
    ap_character_html: str, ap_character_extracted: dict
) -> None:
    from enrichment.sources.anime_planet.anime_planet_character_crawler import (
        AnimePlanetCharacterCrawler,
    )

    crawler = AnimePlanetCharacterCrawler(NullRepository())
    char = _build_character_from_raw(ap_character_extracted, ap_character_html, _LUFFY_URL)
    canonical = crawler.map_to_canonical(char)
    assert canonical["name"] == "Monkey D. Luffy"
    assert any("monkey-d-luffy" in str(s) for s in canonical.get("sources", []))


# =============================================================================
# fetch_animeplanet_character
# =============================================================================


@pytest.mark.usefixtures("mock_redis_cache_miss")
@patch(_PATCH_FETCH_PAGE)
async def test_fetch_character_success(
    mock_page: AsyncMock, ap_character_html: str
) -> None:
    mock_page.return_value = ap_character_html
    import zendriver as zd

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(zd, "start", AsyncMock(return_value=AsyncMock()))
        result = await fetch_animeplanet_character(_LUFFY_URL)

    assert result is not None
    assert result["name"] == "Monkey D. Luffy"


@pytest.mark.usefixtures("mock_redis_cache_miss")
@patch(_PATCH_FETCH_PAGE, new_callable=AsyncMock)
async def test_fetch_character_failure_returns_none(mock_page: AsyncMock) -> None:
    mock_page.return_value = None
    import zendriver as zd

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(zd, "start", AsyncMock(return_value=AsyncMock()))
        result = await fetch_animeplanet_character(_LUFFY_URL)

    assert result is None


# =============================================================================
# fetch_animeplanet_characters
# =============================================================================


async def test_fetch_characters_empty_list() -> None:
    assert await fetch_animeplanet_characters([]) == []


async def test_fetch_characters_all_cached(mocker: Any) -> None:
    mocker.patch.object(
        _fetch_character_data,
        "cache_batch_get",
        new=AsyncMock(return_value=([_MINIMAL_RAW], [])),
    )
    import zendriver as zd

    start_spy = mocker.patch.object(zd, "start", new=AsyncMock())

    result = await fetch_animeplanet_characters([_LUFFY_URL])

    assert result[0] is not None
    assert result[0]["name"] == "Monkey D. Luffy"
    start_spy.assert_not_called()


async def test_fetch_characters_all_missing_success(mocker: Any) -> None:
    mocker.patch.object(
        _fetch_character_data,
        "cache_batch_get",
        new=AsyncMock(return_value=([None], [0])),
    )
    cache_set = mocker.patch.object(
        _fetch_character_data, "cache_batch_set", new=AsyncMock()
    )
    mocker.patch(_PATCH_FETCH_PAGE, new=AsyncMock(return_value=_MINIMAL_HTML))
    import zendriver as zd

    mocker.patch.object(zd, "start", new=AsyncMock(return_value=AsyncMock()))

    result = await fetch_animeplanet_characters([_LUFFY_URL])

    assert result[0] is not None
    cache_set.assert_awaited_once()


async def test_fetch_characters_inter_request_delay(mocker: Any) -> None:
    zoro_url = "https://www.anime-planet.com/characters/roronoa-zoro"
    mocker.patch.object(
        _fetch_character_data,
        "cache_batch_get",
        new=AsyncMock(return_value=([None, None], [0, 1])),
    )
    mocker.patch.object(_fetch_character_data, "cache_batch_set", new=AsyncMock())
    mocker.patch(_PATCH_FETCH_PAGE, new=AsyncMock(return_value=_MINIMAL_HTML))
    sleep_mock = mocker.patch("asyncio.sleep", new=AsyncMock())
    import zendriver as zd

    mocker.patch.object(zd, "start", new=AsyncMock(return_value=AsyncMock()))

    result = await fetch_animeplanet_characters([_LUFFY_URL, zoro_url])

    assert len(result) == 2
    sleep_mock.assert_awaited_once()


async def test_fetch_characters_html_none_returns_none_at_index(mocker: Any) -> None:
    mocker.patch.object(
        _fetch_character_data,
        "cache_batch_get",
        new=AsyncMock(return_value=([None], [0])),
    )
    mocker.patch.object(_fetch_character_data, "cache_batch_set", new=AsyncMock())
    mocker.patch(_PATCH_FETCH_PAGE, new=AsyncMock(return_value=None))
    import zendriver as zd

    mocker.patch.object(zd, "start", new=AsyncMock(return_value=AsyncMock()))

    assert await fetch_animeplanet_characters([_LUFFY_URL]) == [None]


async def test_fetch_characters_extract_none_returns_none_at_index(mocker: Any) -> None:
    mocker.patch.object(
        _fetch_character_data,
        "cache_batch_get",
        new=AsyncMock(return_value=([None], [0])),
    )
    mocker.patch.object(_fetch_character_data, "cache_batch_set", new=AsyncMock())
    mocker.patch(
        _PATCH_FETCH_PAGE,
        new=AsyncMock(return_value="<html><body><p>no name</p></body></html>"),
    )
    import zendriver as zd

    mocker.patch.object(zd, "start", new=AsyncMock(return_value=AsyncMock()))

    assert await fetch_animeplanet_characters([_LUFFY_URL]) == [None]


async def test_fetch_characters_browser_stop_swallowed(mocker: Any) -> None:
    mocker.patch.object(
        _fetch_character_data,
        "cache_batch_get",
        new=AsyncMock(return_value=([None], [0])),
    )
    mocker.patch.object(_fetch_character_data, "cache_batch_set", new=AsyncMock())
    mocker.patch(_PATCH_FETCH_PAGE, new=AsyncMock(return_value=_MINIMAL_HTML))
    browser = AsyncMock()
    browser.stop = AsyncMock(side_effect=Exception("stop failed"))
    import zendriver as zd

    mocker.patch.object(zd, "start", new=AsyncMock(return_value=browser))

    result = await fetch_animeplanet_characters([_LUFFY_URL])

    assert result[0] is not None


async def test_fetch_characters_with_output_path(mocker: Any, tmp_path: Any) -> None:
    mocker.patch.object(
        _fetch_character_data,
        "cache_batch_get",
        new=AsyncMock(return_value=([_MINIMAL_RAW], [])),
    )
    out = tmp_path / "chars.jsonl"

    result = await fetch_animeplanet_characters([_LUFFY_URL], output_path=str(out))

    assert result[0] is not None
    assert out.exists()
    assert "Luffy" in out.read_text()


async def test_fetch_characters_defensive_missing_append(mocker: Any) -> None:
    # cache_batch_get returns None cached value but empty missing_indices list —
    # exercises the defensive `if idx not in missing_indices: append` branch.
    mocker.patch.object(
        _fetch_character_data,
        "cache_batch_get",
        new=AsyncMock(return_value=([None], [])),
    )
    mocker.patch.object(_fetch_character_data, "cache_batch_set", new=AsyncMock())
    mocker.patch(_PATCH_FETCH_PAGE, new=AsyncMock(return_value=_MINIMAL_HTML))
    import zendriver as zd

    mocker.patch.object(zd, "start", new=AsyncMock(return_value=AsyncMock()))

    result = await fetch_animeplanet_characters([_LUFFY_URL])

    assert result[0] is not None
