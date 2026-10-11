import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import ANY, call, patch

import pytest
from enrichment.sources.anime_planet import (
    anime_planet_character_crawler as crawler_module,
)
from enrichment.sources.anime_planet.anime_planet_character_crawler import (
    _XPATHS,
    AnimePlanetCharacterCrawler,
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
    fetch_animeplanet_character,
    fetch_animeplanet_characters,
)
from enrichment.sources.anime_planet.anime_planet_character_models import (
    AnimePlanetCharacter,
    AnimePlanetVoiceActor,
)
from enrichment.sources.base.cloudflare_challenge import CloudflareChallengeError
from enrichment.sources.base.framework import NullRepository

pytestmark = pytest.mark.asyncio

_LUFFY_URL = "https://www.anime-planet.com/characters/monkey-d-luffy"
_ZORO_URL = "https://www.anime-planet.com/characters/roronoa-zoro"
_NAMI_URL = "https://www.anime-planet.com/characters/nami"
_MINIMAL_HTML = (
    "<html><body>"
    "<h1 itemprop='name'>Monkey D. Luffy</h1>"
    "<section class='entryBar'>Gender: Male</section>"
    "</body></html>"
)
_PAGE_WITHOUT_NAME = "<html><body><p>no name</p></body></html>"
_MINIMAL_RAW: dict[str, Any] = {
    "name": "Monkey D. Luffy",
    "image": "https://example.com/luffy.jpg",
    "loved_rank": "#12",
    "hated_rank": "#106",
    "loved_count": "37,007 users",
}
_AD_ANNOTATED_ROLES_HEADING = (
    '<h3><a href="#" class="google-anno" data-google-vignette="false" '
    'data-google-interstitial="false" style="position: initial !important;">'
    '<svg viewBox="100 -1000 840 840" width="calc(22.4px - 2px)" height="22.4px">'
    '<path d="M168-144q-29.7 0-50.85-21.15Z"></path></svg>&nbsp;'
    '<span class="google-anno-t" style="text-decoration: underline dotted !important;">'
    "Anime</span></a> Roles</h3>"
)
_AD_SUGGESTION_CHIP = (
    '<div class="google-anno-skip google-anno-sc" tabindex="0" role="link" '
    'aria-label="Anime &amp; Manga" data-google-vignette="false" '
    'data-google-interstitial="false" style="display: inline-flex !important;">'
    '<span><span><svg viewBox="0 -960 960 960" width="16px" height="16px">'
    '<path d="M168-144q-29.7 0-50.85-21.15Z"></path></svg></span>'
    "<span>Anime</span></span><span>&amp; Manga</span></div>"
)
_DESCRIPTION_BLOCK = re.compile(
    r'(<div[^>]+itemprop="description">.*?)(</div>)', re.DOTALL
)


@pytest.fixture
def character_browser(open_browser):
    with patch.object(crawler_module, "_INTER_REQUEST_DELAY", 0):
        yield lambda browser: open_browser(crawler_module, browser)


@pytest.fixture
def cached():
    def install(values: list[dict | None], missing: list[int]):
        return patch.object(
            _fetch_character_data,
            "cache_batch_get",
            autospec=True,
            return_value=(values, missing),
        )

    return install


@pytest.fixture
def cache_writes() -> Iterator:
    with patch.object(
        _fetch_character_data, "cache_batch_set", autospec=True
    ) as cache_batch_set:
        yield cache_batch_set


def _character(**fields: str | None) -> AnimePlanetCharacter:
    return AnimePlanetCharacter.model_validate(
        {"name": "Luffy", "slug": "luffy", "url": _LUFFY_URL, **fields}
    )


def _with_ad_annotated_roles_heading(html: str) -> str:
    return html.replace("<h3>Anime Roles</h3>", _AD_ANNOTATED_ROLES_HEADING)


def _with_ad_suggestion_chip_in_description(html: str) -> str:
    return _DESCRIPTION_BLOCK.sub(
        lambda block: block.group(1) + _AD_SUGGESTION_CHIP + block.group(2),
        html,
        count=1,
    )


def _challenge_not_cleared(url: str):
    return patch.object(
        crawler_module,
        "wait_for_page",
        autospec=True,
        side_effect=CloudflareChallengeError("Anime-Planet", url),
    )


def test_xpaths_character_queries_are_defined() -> None:
    assert {"name", "image", "loved_rank", "hated_rank", "loved_count"} <= (
        _XPATHS.keys()
    )
    assert "entryBar" in _XPATHS["loved_rank"]
    assert "entryBar" in _XPATHS["hated_rank"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        ("", None),
        ("no-number", None),
        ("#12", 12),
        ("Rank 42", 42),
        ("#1,081", 1081),
        ("#1,234,567", 1234567),
    ],
)
def test_anime_planet_character_rank_text_gives_whole_number(
    raw: str | None, expected: int | None
) -> None:
    character = _character(loved_rank=raw, hated_rank=raw)
    assert (character.loved_rank, character.hated_rank) == (expected, expected)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        ("", None),
        ("37,007 users", 37007),
        ("1000", 1000),
        ("1,234,567 users", 1234567),
    ],
)
def test_anime_planet_character_loved_count_text_gives_whole_number(
    raw: str | None, expected: int | None
) -> None:
    assert _character(loved_count=raw).loved_count == expected


@pytest.mark.parametrize(
    ("html", "expected"),
    [
        ("<html><body><p>no section</p></body></html>", (None, None)),
        (
            '<html><body><section class="entryBar">Gender: Female</section></body></html>',
            ("Female", None),
        ),
        (
            '<html><body><section class="entryBar">'
            "Gender: Male Hair Color: Black</section></body></html>",
            ("Male", "Black"),
        ),
    ],
)
def test_extract_entry_bar_entry_bar_text_returns_gender_and_hair(
    html: str, expected: tuple[str | None, str | None]
) -> None:
    result = _extract_entry_bar(html)
    assert (result["gender"], result["hair_color"]) == expected


def test_extract_entry_bar_luffy_page_returns_gender_and_hair(
    ap_character_html: str,
) -> None:
    result = _extract_entry_bar(ap_character_html)
    assert (result["gender"], result["hair_color"]) == ("Male", "Black")


def test_extract_metadata_page_without_metadata_returns_empty_dict() -> None:
    assert _extract_metadata("<html><body></body></html>") == {}


def test_extract_metadata_luffy_page_returns_birthday(ap_character_html: str) -> None:
    assert _extract_metadata(ap_character_html)["Birthday"] == "May 5"


def test_extract_alt_names_page_without_alt_names_returns_empty_list() -> None:
    assert _extract_alt_names("<html><body></body></html>") == []


def test_extract_alt_names_luffy_page_returns_straw_hat(ap_character_html: str) -> None:
    assert "Straw Hat" in _extract_alt_names(ap_character_html)


def test_extract_alt_names_comma_list_returns_each_name() -> None:
    html = '<html><body><h2 class="aka">Aka: Name One, Name Two, Name Three</h2></body></html>'
    assert _extract_alt_names(html) == ["Name One", "Name Two", "Name Three"]


def test_extract_description_luffy_page_returns_description(
    ap_character_html: str,
) -> None:
    description = _extract_description(ap_character_html)
    assert description is not None
    assert "Shanks" in description


@pytest.mark.parametrize(
    "html",
    [
        "<html><body><p>no desc div</p></body></html>",
        '<html><body><div itemprop="description">   </div></body></html>',
    ],
)
def test_extract_description_without_text_returns_none(html: str) -> None:
    assert _extract_description(html) is None


def test_extract_tags_page_without_tags_returns_empty_list() -> None:
    assert _extract_tags("<html><body></body></html>") == []


def test_extract_tags_tag_links_return_tag_names() -> None:
    html = (
        "<html><body>"
        '<a href="/characters/tags/hero">Hero</a>'
        '<a href="/characters/tags/pirate">Pirate</a>'
        "</body></html>"
    )
    assert _extract_tags(html) == ["Hero", "Pirate"]


@pytest.mark.parametrize("cell", ["", "<td>no flags here</td>"])
def test_extract_vas_from_cell_without_flags_returns_empty_dict(cell: str) -> None:
    assert _extract_vas_from_cell(cell) == {}


def test_extract_vas_from_cell_flagged_links_return_actors_by_country() -> None:
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


def test_extract_anime_roles_page_without_section_returns_empty_list() -> None:
    assert _extract_anime_roles("<html><body></body></html>") == []


def test_extract_anime_roles_rows_without_anime_link_return_empty_list() -> None:
    html = (
        "<h3>Anime Roles</h3>"
        "<table><tbody>"
        "<tr><td>only one cell</td></tr>"
        '<tr><td><a href="/not-matching/slug">Title</a></td><td>Main</td></tr>'
        "</tbody></table>"
    )
    assert _extract_anime_roles(html) == []


def test_extract_anime_roles_blank_role_returns_none_role() -> None:
    html = (
        "<h3>Anime Roles</h3>"
        "<table><tbody>"
        '<tr><td><a href="/anime/test-show">Test Show</a></td><td>   </td></tr>'
        "</tbody></table>"
    )

    roles = _extract_anime_roles(html)

    assert [(role.title, role.url, role.role) for role in roles] == [
        ("Test Show", "/anime/test-show", None)
    ]


def test_extract_anime_roles_luffy_page_returns_main_role_with_voice_actors(
    ap_character_html: str,
) -> None:
    roles = _extract_anime_roles(ap_character_html)

    one_piece = next(role for role in roles if "One Piece" in role.title)
    assert len(roles) == 25
    assert one_piece.role == "Main"
    assert "jp" in one_piece.voice_actors


def test_extract_anime_roles_ad_annotated_heading_returns_empty_list(
    each_ap_character_html: str,
) -> None:
    annotated = _with_ad_annotated_roles_heading(each_ap_character_html)

    assert annotated != each_ap_character_html
    assert _extract_anime_roles(each_ap_character_html)
    assert _extract_anime_roles(annotated) == []


def test_extract_description_ad_suggestion_chip_changes_description(
    each_ap_character_html: str,
) -> None:
    chipped = _with_ad_suggestion_chip_in_description(each_ap_character_html)

    assert chipped != each_ap_character_html
    assert _extract_description(chipped) != _extract_description(each_ap_character_html)


def test_extract_character_from_html_ad_markup_returns_same_character(
    each_ap_character_html: str,
) -> None:
    changed = _with_ad_suggestion_chip_in_description(
        _with_ad_annotated_roles_heading(each_ap_character_html)
    )

    extracted = _extract_character_from_html(changed)

    assert extracted == _extract_character_from_html(each_ap_character_html)
    assert extracted is not None
    assert extracted["anime_roles"]
    assert extracted["description"]


def test_extract_manga_roles_page_without_section_returns_empty_list() -> None:
    assert _extract_manga_roles("<html><body></body></html>") == []


def test_extract_manga_roles_rows_without_manga_link_return_empty_list() -> None:
    html = (
        "<h3>Manga Roles</h3>"
        "<table><tbody>"
        "<tr><td>only one cell</td></tr>"
        '<tr><td><a href="/not-matching/slug">Title</a></td><td>Lead</td></tr>'
        "</tbody></table>"
    )
    assert _extract_manga_roles(html) == []


def test_extract_manga_roles_luffy_page_returns_main_role(
    ap_character_html: str,
) -> None:
    roles = _extract_manga_roles(ap_character_html)

    main = next(role for role in roles if role.title == "One Piece")
    assert len(roles) == 16
    assert (main.role, main.url) == ("Main", "/manga/one-piece")


@pytest.mark.parametrize("html", ["", _PAGE_WITHOUT_NAME])
def test_extract_character_from_html_page_without_name_returns_none(html: str) -> None:
    assert _extract_character_from_html(html) is None


def test_extract_character_from_html_luffy_page_returns_every_field(
    ap_character_html: str,
) -> None:
    raw = _extract_character_from_html(ap_character_html)

    assert raw is not None
    assert raw["name"] == "Monkey D. Luffy"
    assert "luffy" in raw["image"]
    assert (raw["loved_rank"], raw["hated_rank"], raw["loved_count"]) == (
        "#12",
        "#106",
        "37,007 users",
    )
    assert "_html" not in raw
    assert raw["gender"] == "Male"
    assert (len(raw["anime_roles"]), len(raw["manga_roles"])) == (25, 16)


def test_build_character_from_raw_luffy_page_returns_typed_character(
    ap_character_extracted: dict,
) -> None:
    character = _build_character_from_raw(ap_character_extracted, _LUFFY_URL)

    assert (character.name, character.slug, character.url) == (
        "Monkey D. Luffy",
        "monkey-d-luffy",
        _LUFFY_URL,
    )
    assert (character.loved_rank, character.hated_rank, character.loved_count) == (
        12,
        106,
        37007,
    )
    assert (character.gender, character.hair_color) == ("Male", "Black")
    assert character.description is not None
    assert "Shanks" in character.description
    assert "Straw Hat" in character.alt_names
    assert (len(character.anime_roles), len(character.manga_roles)) == (25, 16)


def test_build_character_from_raw_without_image_returns_no_image(
    ap_character_extracted: dict,
) -> None:
    raw = {**ap_character_extracted, "image": None}
    assert _build_character_from_raw(raw, _LUFFY_URL).image is None


async def test_fetch_page_html_character_page_returns_content(browser_showing) -> None:
    browser = browser_showing("<html>luffy</html>")
    assert await _fetch_page_html(browser, _LUFFY_URL) == "<html>luffy</html>"


async def test_fetch_page_html_navigation_failure_returns_none(browser_showing) -> None:
    assert await _fetch_page_html(browser_showing(None), _LUFFY_URL) is None


async def test_fetch_page_html_challenge_not_cleared_raises(browser_showing) -> None:
    with (
        _challenge_not_cleared(_LUFFY_URL),
        pytest.raises(CloudflareChallengeError),
    ):
        await _fetch_page_html(browser_showing(""), _LUFFY_URL)


async def test_fetch_character_data_character_page_returns_raw(
    browser_showing,
    character_browser,
) -> None:
    with character_browser(browser_showing(_MINIMAL_HTML)):
        raw = await _fetch_character_data(_LUFFY_URL)

    assert raw is not None
    assert raw["name"] == "Monkey D. Luffy"


async def test_fetch_character_data_navigation_failure_returns_none(
    browser_showing,
    character_browser,
) -> None:
    with character_browser(browser_showing(None)):
        assert await _fetch_character_data(_LUFFY_URL) is None


async def test_fetch_character_data_challenge_not_cleared_returns_none(
    browser_showing,
    character_browser,
) -> None:
    with (
        character_browser(browser_showing(_MINIMAL_HTML)),
        _challenge_not_cleared(_LUFFY_URL),
    ):
        assert await _fetch_character_data(_LUFFY_URL) is None


def test_anime_planet_character_crawler_get_extraction_schema_returns_xpaths() -> None:
    assert AnimePlanetCharacterCrawler(NullRepository()).get_extraction_schema() is (
        _XPATHS
    )


def test_anime_planet_character_crawler_normalize_identifier_returns_url() -> None:
    crawler = AnimePlanetCharacterCrawler(NullRepository())
    assert crawler.normalize_identifier(_LUFFY_URL) == _LUFFY_URL


def test_anime_planet_character_crawler_build_source_model_returns_character(
    ap_character_extracted: dict,
) -> None:
    crawler = AnimePlanetCharacterCrawler(NullRepository())

    character = crawler.build_source_model(ap_character_extracted, _LUFFY_URL)

    assert (character.name, character.slug) == ("Monkey D. Luffy", "monkey-d-luffy")


def test_anime_planet_character_crawler_map_to_canonical_returns_name_and_source(
    ap_character_extracted: dict,
) -> None:
    crawler = AnimePlanetCharacterCrawler(NullRepository())

    canonical = crawler.map_to_canonical(
        _build_character_from_raw(ap_character_extracted, _LUFFY_URL)
    )

    assert canonical["name"] == "Monkey D. Luffy"
    assert any("monkey-d-luffy" in str(source) for source in canonical["sources"])


async def test_fetch_animeplanet_character_luffy_page_returns_canonical_character(
    browser_showing, character_browser, ap_character_html: str
) -> None:
    with character_browser(browser_showing(ap_character_html)):
        character = await fetch_animeplanet_character(_LUFFY_URL)

    assert character is not None
    assert character["name"] == "Monkey D. Luffy"


async def test_fetch_animeplanet_character_navigation_failure_returns_none(
    browser_showing,
    character_browser,
) -> None:
    with character_browser(browser_showing(None)):
        assert await fetch_animeplanet_character(_LUFFY_URL) is None


async def test_fetch_animeplanet_characters_without_urls_returns_empty_list() -> None:
    assert await fetch_animeplanet_characters([]) == []


async def test_fetch_animeplanet_characters_all_cached_returns_without_fetching(
    browser_showing, character_browser, cached
) -> None:
    with cached([_MINIMAL_RAW], []), character_browser(browser_showing(None)):
        characters = await fetch_animeplanet_characters([_LUFFY_URL])

    assert characters[0]["name"] == "Monkey D. Luffy"


async def test_fetch_animeplanet_characters_cache_miss_fetches_and_caches_raw(
    browser_showing, character_browser, cached, cache_writes
) -> None:
    with cached([None], [0]), character_browser(browser_showing(_MINIMAL_HTML)):
        characters = await fetch_animeplanet_characters([_LUFFY_URL])

    assert characters[0]["name"] == "Monkey D. Luffy"
    [written] = cache_writes.await_args_list
    urls, raw_values = written.args
    assert urls == [_LUFFY_URL]
    assert raw_values[0]["name"] == "Monkey D. Luffy"


async def test_fetch_animeplanet_characters_cached_none_without_miss_fetches_page(
    browser_showing, character_browser, cached, cache_writes
) -> None:
    with cached([None], []), character_browser(browser_showing(_MINIMAL_HTML)):
        characters = await fetch_animeplanet_characters([_LUFFY_URL])

    assert characters[0] is not None


async def test_fetch_animeplanet_characters_several_misses_wait_between_requests(
    browser_showing, character_browser, cached, cache_writes
) -> None:
    with (
        cached([None, None], [0, 1]),
        character_browser(browser_showing(_MINIMAL_HTML)),
        patch.object(crawler_module.asyncio, "sleep", autospec=True) as sleep,
    ):
        characters = await fetch_animeplanet_characters([_LUFFY_URL, _ZORO_URL])

    assert len(characters) == 2
    assert sleep.await_args_list == [call(crawler_module._INTER_REQUEST_DELAY)]


@pytest.mark.parametrize("html", [None, _PAGE_WITHOUT_NAME])
async def test_fetch_animeplanet_characters_unreadable_page_returns_none(
    character_browser, browser_showing, cached, cache_writes, html: str | None
) -> None:
    with cached([None], [0]), character_browser(browser_showing(html)):
        assert await fetch_animeplanet_characters([_LUFFY_URL]) == [None]


async def test_fetch_animeplanet_characters_challenge_not_cleared_stops_batch(
    browser_showing,
    character_browser,
    cached,
    cache_writes,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with (
        cached([None, None, None], [0, 1, 2]),
        character_browser(browser_showing("")),
        patch.object(
            crawler_module,
            "_fetch_page_html",
            autospec=True,
            side_effect=[
                _MINIMAL_HTML,
                CloudflareChallengeError("Anime-Planet", _ZORO_URL),
            ],
        ),
    ):
        characters = await fetch_animeplanet_characters(
            [_LUFFY_URL, _ZORO_URL, _NAMI_URL]
        )

    assert characters[0] is not None
    assert characters[1:] == [None, None]
    assert cache_writes.await_args_list == [call([_LUFFY_URL], [ANY])]
    assert f"Anime-Planet: stopping the character batch at {_ZORO_URL}" in caplog.text


async def test_fetch_animeplanet_characters_output_path_appends_characters(
    cached, tmp_path: Path
) -> None:
    output = tmp_path / "chars.jsonl"

    with cached([_MINIMAL_RAW], []):
        characters = await fetch_animeplanet_characters(
            [_LUFFY_URL], output_path=str(output)
        )

    assert characters[0] is not None
    assert "Luffy" in output.read_text()
