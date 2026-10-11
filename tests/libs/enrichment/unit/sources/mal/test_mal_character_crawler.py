import sys
from collections.abc import Iterator
from unittest.mock import call, patch

import pytest
from enrichment.sources.base.framework import NullRepository
from enrichment.sources.mal import mal_character_crawler
from enrichment.sources.mal.mal_character_crawler import (
    _XPATHS,
    MalCharacterCrawler,
    _build_character_from_raw,
    _extract_bio_data,
    _extract_character_from_html,
    _extract_description,
    _extract_name_and_native,
    _extract_ography,
    _extract_voice_actors,
    _fetch_character_html,
    _fetch_mal_character_data,
    fetch_mal_character,
    fetch_mal_characters,
    main,
)

_LUFFY_URL = "https://myanimelist.net/character/40/Luffy_Monkey_D"
_ZORO_URL = "https://myanimelist.net/character/41/Roronoa_Zoro"
_LUFFY_NAME = "Luffy Monkey D."
_PAGE_WITHOUT_CHARACTER = '<html><body><h2 class="normal_header"></h2></body></html>'
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


@pytest.fixture
def character_browser(open_browser, browser_serving):
    def serve(pages: dict[str, str]):
        return open_browser(mal_character_crawler, browser_serving(pages))

    return serve


@pytest.fixture
def cached():
    def install(values: list, missing: list[int]):
        return patch.object(
            _fetch_mal_character_data,
            "cache_batch_get",
            autospec=True,
            return_value=(values, missing),
        )

    return install


@pytest.fixture
def cache_writes() -> Iterator:
    with patch.object(
        _fetch_mal_character_data, "cache_batch_set", autospec=True
    ) as cache_batch_set:
        yield cache_batch_set


def test_xpaths_character_queries_are_defined() -> None:
    assert {"name_header", "image_src", "favorites_td", "content"} <= _XPATHS.keys()
    assert all(isinstance(query, str) and query for query in _XPATHS.values())


def test_extract_character_from_html_luffy_page_returns_every_field(
    mal_character_html,
) -> None:
    raw = _extract_character_from_html(mal_character_html)

    assert raw is not None
    assert raw["name_header"] == "Luffy Monkey D. (モンキー・D・ルフィ)"
    assert raw["image_src"] == (
        "https://cdn.myanimelist.net/images/characters/9/310307.jpg"
    )
    assert raw["favorites"] == "148,888"
    assert "content_html" not in raw
    assert raw["animeography"]
    assert raw["voice_actors"]
    assert raw["character_info"].get("height") == "172 cm"


@pytest.mark.parametrize("html", ["", "<html><body><p>nothing</p></body></html>"])
def test_extract_character_from_html_page_without_character_returns_none(
    html: str,
) -> None:
    assert _extract_character_from_html(html) is None


def test_extract_character_from_html_without_image_or_favorites_returns_none_fields() -> (
    None
):
    html = """<html><body>
    <h2 class="normal_header">Test Char</h2>
    <div id="content"><p>Some content</p></div>
    </body></html>"""

    raw = _extract_character_from_html(html)

    assert raw is not None
    assert (raw["name_header"], raw["image_src"], raw["favorites"]) == (
        "Test Char",
        None,
        None,
    )


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("Luffy Monkey D. (モンキー・D・ルフィ)", (_LUFFY_NAME, "モンキー・D・ルフィ")),
        (
            "Monkey D., Luffy (モンキー・D・ルフィ)",
            ("Monkey D., Luffy", "モンキー・D・ルフィ"),
        ),
        ("Roronoa, Zoro", ("Roronoa, Zoro", None)),
    ],
)
def test_extract_name_and_native_header_returns_name_and_native(
    header: str, expected: tuple[str, str | None]
) -> None:
    assert _extract_name_and_native(header) == expected


def test_extract_name_and_native_without_header_returns_no_native() -> None:
    assert _extract_name_and_native(None)[1] is None


def test_extract_bio_data_luffy_page_returns_attributes_and_spoilers(
    mal_character_content_html,
) -> None:
    attributes, spoilers = _extract_bio_data(mal_character_content_html)

    assert (attributes.get("age"), attributes.get("height")) == ("17; 19", "172 cm")
    assert attributes.get("blood_type") == "F"
    assert "devil_fruit" in attributes
    assert "bounty" in spoilers


def test_extract_bio_data_key_value_lines_return_attributes_until_next_heading() -> (
    None
):
    html = """
<h2 class="normal_header">Character Information</h2>
Age: 17; 19<br>
Height: 172 cm<br>
Blood type: F<br>
Devil fruit: Gomu Gomu no Mi<br>
<h2>Next Section</h2>
"""

    attributes, _ = _extract_bio_data(html)

    assert attributes == {
        "age": "17; 19",
        "height": "172 cm",
        "blood_type": "F",
        "devil_fruit": "Gomu Gomu no Mi",
    }


@pytest.mark.parametrize("html", ["", "<p>Some text without a normal_header</p>"])
def test_extract_bio_data_without_information_heading_returns_empty(html: str) -> None:
    assert _extract_bio_data(html) == ({}, {})


def test_extract_bio_data_spoiler_only_value_returns_spoiler_without_attribute() -> (
    None
):
    attributes, spoilers = _extract_bio_data(_SPOILER_ONLY_HTML)

    assert "bounty" not in attributes
    assert "3,000,000,000" in spoilers["bounty"]


def test_extract_bio_data_value_with_spoiler_suffix_returns_both_parts() -> None:
    attributes, spoilers = _extract_bio_data(_SPOILER_SUFFIX_HTML)

    assert attributes.get("devil_fruit") == "Gomu Gomu no Mi"
    assert spoilers.get("devil_fruit") == "Hito Hito no Mi"


def test_extract_bio_data_spoiler_without_content_returns_no_spoiler() -> None:
    html = """
<h2 class="normal_header">Info</h2>
Key:
<div class="spoiler">No proper spoiler_content span here</div><br>
"""

    _, spoilers = _extract_bio_data(html)

    assert "key" not in spoilers


def test_extract_bio_data_overlong_key_returns_other_attributes() -> None:
    long_key = "a" * 50
    html = f"""
<h2 class="normal_header">Info</h2>
{long_key}: should be ignored<br>
Age: 17<br>
"""

    attributes, _ = _extract_bio_data(html)

    assert attributes == {"age": "17"}


def test_extract_description_luffy_page_returns_prose_without_attributes(
    mal_character_content_html,
) -> None:
    description, _ = _extract_description(mal_character_content_html)

    assert description is not None
    assert "Straw Hat" in description
    assert "Blood type" not in description
    assert "Height" not in description


def test_extract_description_prose_returns_text_before_next_heading() -> None:
    html = """
<h2 class="normal_header">About</h2>
The main character of One Piece.<br>
He ate the Gomu Gomu no Mi.<br>
<div class="normal_header">Voice Actors</div>
"""

    description, _ = _extract_description(html)

    assert description is not None
    assert "One Piece" in description
    assert "Voice Actors" not in description


def test_extract_description_empty_page_returns_nothing() -> None:
    assert _extract_description("") == (None, None)


def test_extract_description_key_value_line_returns_prose_only() -> None:
    html = """
<h2 class="normal_header">About</h2>
Age: 17<br>
He is the captain of the Straw Hat Pirates.<br>
"""

    description, _ = _extract_description(html)

    assert description is not None
    assert "captain" in description
    assert "Age: 17" not in description


def test_extract_description_prose_spoiler_returns_spoiler_separately() -> None:
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

    description, spoiler = _extract_description(html)

    assert description is not None
    assert "captain" in description
    assert spoiler is not None
    assert "Dragon" in spoiler


def test_extract_voice_actors_luffy_page_returns_every_actor(
    mal_character_content_html,
) -> None:
    actors = _extract_voice_actors(mal_character_content_html)

    english = next(actor for actor in actors if actor.language == "English")
    assert len(actors) == 28
    assert (actors[0].name, actors[0].language, actors[0].person_id) == (
        "Tanaka, Mayumi",
        "Japanese",
        75,
    )
    assert "myanimelist.net/people/75" in actors[0].sources[0]
    assert all(len(actor.sources) == 1 for actor in actors)
    assert (english.name, english.person_id) == ("Clinkenbeard, Colleen", 472)


def test_extract_voice_actors_actor_rows_return_name_language_and_link() -> None:
    actors = _extract_voice_actors(_VA_HTML)

    assert [
        (actor.name, actor.language, actor.person_id, actor.sources) for actor in actors
    ] == [
        (
            "Tanaka, Mayumi",
            "Japanese",
            70,
            ["https://myanimelist.net/people/70/Tanaka_Mayumi"],
        ),
        (
            "Clinkenbeard, Colleen",
            "English",
            81,
            ["https://myanimelist.net/people/81/Colleen_Clinkenbeard"],
        ),
    ]


def test_extract_voice_actors_page_without_section_returns_empty_list() -> None:
    assert _extract_voice_actors("<div>No VA section here</div>") == []


def test_extract_voice_actors_row_without_person_link_returns_other_rows() -> None:
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

    assert [actor.name for actor in _extract_voice_actors(html)] == ["Tanaka, Mayumi"]


@pytest.mark.parametrize(
    ("section", "expected_count"), [("Animeography", 60), ("Mangaography", 16)]
)
def test_extract_ography_luffy_page_returns_main_role_first(
    mal_character_content_html, section: str, expected_count: int
) -> None:
    entries = _extract_ography(mal_character_content_html, section)

    assert len(entries) == expected_count
    assert (entries[0].title, entries[0].role) == ("One Piece", "Main")


@pytest.mark.parametrize(
    ("section", "expected_titles"),
    [
        ("Animeography", ["One Piece", "One Piece Film: Gold"]),
        ("Mangaography", ["One Piece"]),
        ("NonExistentSection", []),
    ],
)
def test_extract_ography_section_returns_its_titles(
    section: str, expected_titles: list[str]
) -> None:
    entries = _extract_ography(_OGRAPHY_HTML, section)

    assert [entry.title for entry in entries] == expected_titles


def test_extract_ography_anime_section_returns_anime_link() -> None:
    first = _extract_ography(_OGRAPHY_HTML, "Animeography")[0]

    assert first.role == "Main"
    assert "myanimelist.net/anime/21" in first.sources[0]


def test_extract_ography_row_without_title_returns_other_rows() -> None:
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

    assert [entry.title for entry in _extract_ography(html, "Animeography")] == [
        "One Piece"
    ]


def test_build_character_from_raw_luffy_page_returns_full_character(
    mal_character_extracted,
) -> None:
    character = _build_character_from_raw(mal_character_extracted, _LUFFY_URL)

    assert (character.name, character.name_native) == (
        _LUFFY_NAME,
        "モンキー・D・ルフィ",
    )
    assert character.favorites == 148888
    assert character.images == [
        "https://cdn.myanimelist.net/images/characters/9/310307.jpg"
    ]
    assert character.source == _LUFFY_URL
    assert character.character_info.get("age") == "17; 19"
    assert (
        len(character.animeography),
        len(character.mangaography),
        len(character.voice_actors),
    ) == (60, 16, 28)


def test_build_character_from_raw_favorites_without_comma_returns_count(
    mal_character_extracted,
) -> None:
    raw = {**mal_character_extracted, "favorites": "12345"}
    assert _build_character_from_raw(raw, _LUFFY_URL).favorites == 12345


def test_build_character_from_raw_without_favorites_returns_zero(
    mal_character_extracted,
) -> None:
    raw = {
        key: value
        for key, value in mal_character_extracted.items()
        if key != "favorites"
    }
    assert _build_character_from_raw(raw, _LUFFY_URL).favorites == 0


def test_build_character_from_raw_without_image_returns_empty_images(
    mal_character_extracted,
) -> None:
    raw = {**mal_character_extracted, "image_src": None}
    assert _build_character_from_raw(raw, _LUFFY_URL).images == []


def test_build_character_from_raw_given_address_becomes_source(
    mal_character_extracted,
) -> None:
    address = "https://myanimelist.net/character/40/SomeSlug"
    assert _build_character_from_raw(mal_character_extracted, address).source == address


def test_mal_character_crawler_get_extraction_schema_returns_xpaths() -> None:
    assert MalCharacterCrawler(NullRepository()).get_extraction_schema() == {
        "xpaths": _XPATHS
    }


def test_mal_character_crawler_normalize_identifier_returns_url() -> None:
    crawler = MalCharacterCrawler(NullRepository())
    assert crawler.normalize_identifier(_LUFFY_URL) == _LUFFY_URL


@pytest.mark.asyncio
async def test_fetch_character_html_character_page_returns_content_and_address(
    browser_serving, mal_character_html
) -> None:
    browser = browser_serving({_LUFFY_URL: mal_character_html})

    assert await _fetch_character_html(browser, _LUFFY_URL) == (
        mal_character_html,
        _LUFFY_URL,
    )


@pytest.mark.asyncio
async def test_fetch_character_html_page_without_heading_returns_none(
    browser_serving,
) -> None:
    browser = browser_serving({_LUFFY_URL: "<html><body></body></html>"})
    assert await _fetch_character_html(browser, _LUFFY_URL) is None


@pytest.mark.asyncio
async def test_fetch_mal_character_data_character_page_returns_raw_and_address(
    character_browser, mal_character_html
) -> None:
    with character_browser({_LUFFY_URL: mal_character_html}):
        result = await _fetch_mal_character_data(_LUFFY_URL)

    assert result is not None
    raw, address = result
    assert raw["name_header"] == "Luffy Monkey D. (モンキー・D・ルフィ)"
    assert address == _LUFFY_URL


@pytest.mark.asyncio
@pytest.mark.parametrize("html", ["", _PAGE_WITHOUT_CHARACTER])
async def test_fetch_mal_character_data_unreadable_page_returns_none(
    character_browser, html: str
) -> None:
    with character_browser({_LUFFY_URL: html}):
        assert await _fetch_mal_character_data(_LUFFY_URL) is None


@pytest.mark.asyncio
async def test_fetch_mal_character_character_page_returns_canonical_character(
    character_browser, mal_character_html
) -> None:
    with character_browser({_LUFFY_URL: mal_character_html}):
        character = await fetch_mal_character(_LUFFY_URL)

    assert character is not None
    assert (character["name"], character["sources"], character["favorites"]) == (
        _LUFFY_NAME,
        [_LUFFY_URL],
        148888,
    )


@pytest.mark.asyncio
async def test_fetch_mal_character_unreadable_page_returns_none(
    character_browser,
) -> None:
    with character_browser({_LUFFY_URL: ""}):
        assert await fetch_mal_character(_LUFFY_URL) is None


@pytest.mark.asyncio
async def test_fetch_mal_characters_without_urls_returns_empty_list() -> None:
    assert await fetch_mal_characters([]) == []


@pytest.mark.asyncio
async def test_fetch_mal_characters_all_cached_returns_without_fetching(
    character_browser, cached, mal_character_extracted
) -> None:
    with (
        cached([[mal_character_extracted, _LUFFY_URL]], []),
        character_browser({}),
    ):
        characters = await fetch_mal_characters([_LUFFY_URL])

    assert [character["name"] for character in characters] == [_LUFFY_NAME]


@pytest.mark.asyncio
@pytest.mark.parametrize("cached_value", ["not_a_tuple", [42, _LUFFY_URL]])
async def test_fetch_mal_characters_unusable_cached_value_fetches_page(
    character_browser, cached, cache_writes, mal_character_html, cached_value
) -> None:
    with (
        cached([cached_value], []),
        character_browser({_LUFFY_URL: mal_character_html}),
    ):
        characters = await fetch_mal_characters([_LUFFY_URL])

    assert characters[0]["name"] == _LUFFY_NAME


@pytest.mark.asyncio
@pytest.mark.parametrize("html", ["", _PAGE_WITHOUT_CHARACTER])
async def test_fetch_mal_characters_unreadable_page_returns_none(
    character_browser, cached, cache_writes, html: str
) -> None:
    with cached([None], [0]), character_browser({_LUFFY_URL: html}):
        assert await fetch_mal_characters([_LUFFY_URL]) == [None]


@pytest.mark.asyncio
async def test_fetch_mal_characters_cache_miss_fetches_and_caches_raw(
    character_browser, cached, cache_writes, mal_character_html
) -> None:
    with cached([None], [0]), character_browser({_LUFFY_URL: mal_character_html}):
        characters = await fetch_mal_characters([_LUFFY_URL])

    assert characters[0]["name"] == _LUFFY_NAME
    [written] = cache_writes.await_args_list
    urls, values = written.args
    assert urls == [_LUFFY_URL]
    assert values[0][1] == _LUFFY_URL


@pytest.mark.asyncio
async def test_fetch_mal_characters_cached_and_missing_returns_both_in_order(
    character_browser, cached, cache_writes, mal_character_extracted, mal_character_html
) -> None:
    with (
        cached([[mal_character_extracted, _LUFFY_URL], None], [1]),
        character_browser({_ZORO_URL: mal_character_html}),
    ):
        characters = await fetch_mal_characters([_LUFFY_URL, _ZORO_URL])

    assert [character["sources"] for character in characters] == [
        [_LUFFY_URL],
        [_ZORO_URL],
    ]
    assert cache_writes.await_count == 1


@pytest.mark.asyncio
async def test_fetch_mal_characters_several_misses_wait_between_requests(
    character_browser, cached, cache_writes, mal_character_html
) -> None:
    pages = {_LUFFY_URL: mal_character_html, _ZORO_URL: mal_character_html}
    with (
        cached([None, None], [0, 1]),
        character_browser(pages),
        patch.object(mal_character_crawler.asyncio, "sleep", autospec=True) as sleep,
    ):
        characters = await fetch_mal_characters([_LUFFY_URL, _ZORO_URL])

    assert all(character is not None for character in characters)
    assert sleep.await_args_list == [call(mal_character_crawler._INTER_REQUEST_DELAY)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("character", "exit_code"), [({"name": _LUFFY_NAME}, 0), (None, 1)]
)
async def test_main_fetched_character_returns_exit_code(
    tmp_path, character: dict | None, exit_code: int
) -> None:
    arguments = ["prog", _LUFFY_URL, "--output", str(tmp_path / "out.json")]
    with (
        patch.object(sys, "argv", arguments),
        patch.object(
            mal_character_crawler,
            "fetch_mal_character",
            autospec=True,
            return_value=character,
        ),
    ):
        assert await main() == exit_code
