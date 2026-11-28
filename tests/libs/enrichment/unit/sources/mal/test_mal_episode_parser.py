import pytest
from enrichment.sources.mal.mal_episode_parser import (
    MAL_EPISODE_XPATHS,
    _parse_episode_characters,
    _parse_episode_staff,
    _parse_title_info,
    build_episode_from_raw,
    extract_episode_from_html,
    extract_episode_list_from_html,
)

_EP1_URL = "https://myanimelist.net/anime/21/One_Piece/episode/1"
_EP50_URL = "https://myanimelist.net/anime/21/One_Piece/episode/50"
_EP279_URL = "https://myanimelist.net/anime/21/One_Piece/episode/279"
_EP1152_URL = "https://myanimelist.net/anime/21/One_Piece/episode/1152"
_LUFFY_TITLE = "I'm Luffy! The Man Who's Gonna Be King of the Pirates!"


def test_xpaths_episode_queries_are_defined() -> None:
    assert {
        "title_header",
        "subtitle",
        "info_box",
        "char_tables",
        "staff_tables",
    } <= MAL_EPISODE_XPATHS.keys()


def test_extract_episode_from_html_regular_episode_reads_cast_and_staff(
    mal_episode_html,
) -> None:
    raw = extract_episode_from_html(mal_episode_html)

    assert raw is not None
    assert raw["title_header"].startswith("#1")
    assert len(raw["characters"]) == 10
    assert len(raw["staff"]) == 13
    assert len([member for member in raw["staff"] if not member.get("role")]) == 4


def test_extract_episode_from_html_filler_episode_reads_badge_without_cast(
    mal_episode_filler_html,
) -> None:
    raw = extract_episode_from_html(mal_episode_filler_html)

    assert raw is not None
    assert "Filler" in raw["title_header"]
    assert (raw["characters"], raw["staff"]) == ([], [])


def test_extract_episode_from_html_recap_episode_reads_badge(
    mal_episode_recap_html,
) -> None:
    raw = extract_episode_from_html(mal_episode_recap_html)

    assert raw is not None
    assert "Recap" in raw["title_header"]


def test_extract_episode_from_html_episode_without_duration_returns_no_duration(
    mal_episode_no_synopsis_html,
) -> None:
    raw = extract_episode_from_html(mal_episode_no_synopsis_html)

    assert raw is not None
    assert raw.get("duration_raw") is None


@pytest.mark.parametrize("html", ["", "<html><body><p>no h2 here</p></body></html>"])
def test_extract_episode_from_html_page_without_episode_title_returns_none(
    html: str,
) -> None:
    assert extract_episode_from_html(html) is None


def test_extract_episode_from_html_tables_without_person_links_return_no_cast() -> None:
    html = """<html><body>
    <h2 class="fs18">#1 - Title</h2>
    <h2>Characters</h2>
    <table class="fl-l"><tr><td>no link</td></tr></table>
    <h2>Staff</h2>
    <table class="fl-l"><tr><td>no link</td></tr></table>
    </body></html>"""

    raw = extract_episode_from_html(html)

    assert raw is not None
    assert (raw["characters"], raw["staff"]) == ([], [])


@pytest.mark.parametrize(
    ("page", "number", "expected"),
    [
        (
            "mal_episode_extracted",
            1,
            (
                _LUFFY_TITLE,
                "俺はルフィ！海賊王になる男だ！",
                "Ore wa Luffy! Kaizoku Ou ni Naru Otoko Da!",
                False,
                False,
            ),
        ),
        (
            "mal_episode_filler_extracted",
            50,
            (
                "Usopp vs. Daddy the Parent! Showdown at High!",
                "ウソップＶＳ子連れのダディ真昼の決闘",
                "Usopp vs Kozure no Dadi Mahiru no Kettou",
                True,
                False,
            ),
        ),
        (
            "mal_episode_recap_extracted",
            279,
            (
                "Jump Towards the Falls! Luffy's Feelings!",
                "滝に向かって飛べ！ルフィの想い！！",
                "Taki ni Mukatte Tobe! Luffy no Omoi!!",
                False,
                True,
            ),
        ),
        (
            "mal_episode_no_synopsis_extracted",
            1152,
            (
                "Her Father and Mother's Legacy! Bonney's Nika Punch",
                "父と母の想い! ボニーの解放の拳[ニカパンチ]",
                "Chichi to Haha no Omoi! Bonney no Nika Punch",
                False,
                False,
            ),
        ),
    ],
)
def test_parse_title_info_episode_page_returns_titles_and_badges(
    request, page: str, number: int, expected: tuple
) -> None:
    raw = request.getfixturevalue(page)

    assert _parse_title_info(raw["title_header"], raw["subtitle_raw"], number) == (
        expected
    )


@pytest.mark.parametrize(
    ("header", "subtitle", "number", "expected"),
    [
        ("#50 - Title\n  Filler", None, 50, ("Title", None, None, True, False)),
        ("#279 - Title\n  Recap", None, 279, ("Title", None, None, False, True)),
        ("#1 - Title FILLER", None, 1, ("Title", None, None, True, False)),
        (None, None, 42, ("Episode 42", None, None, False, False)),
        (
            "#1 - Test",
            "Romaji Title Only",
            1,
            ("Test", None, "Romaji Title Only", False, False),
        ),
    ],
)
def test_parse_title_info_header_variants_return_titles_and_badges(
    header: str | None, subtitle: str | None, number: int, expected: tuple
) -> None:
    assert _parse_title_info(header, subtitle, number) == expected


def test_parse_episode_characters_regular_episode_reads_cast_with_voice_actors(
    mal_episode_extracted,
) -> None:
    characters = _parse_episode_characters(mal_episode_extracted["characters"])

    luffy = characters[0]
    japanese_voice = next(
        actor for actor in luffy.voice_actors if actor.language == "Japanese"
    )
    assert len(characters) == 10
    assert (luffy.name, luffy.mal_id, luffy.role) == ("Monkey D., Luffy", 40, "Main")
    assert len(luffy.voice_actors) == 4
    assert (japanese_voice.person_id, japanese_voice.name) == (75, "Tanaka, Mayumi")


def test_parse_episode_characters_filler_episode_returns_empty_list(
    mal_episode_filler_extracted,
) -> None:
    assert _parse_episode_characters(mal_episode_filler_extracted["characters"]) == []


@pytest.mark.parametrize(
    "items",
    [
        None,
        [],
        [{"char_name": "Luffy", "char_url": ""}],
        [{"char_name": "Luffy", "char_url": "https://myanimelist.net/anime/21"}],
    ],
)
def test_parse_episode_characters_without_character_link_returns_empty_list(
    items: list[dict] | None,
) -> None:
    assert _parse_episode_characters(items) == []


def test_parse_episode_characters_missing_role_returns_supporting() -> None:
    items = [
        {
            "char_name": "X",
            "char_url": "https://myanimelist.net/character/999",
            "role": None,
            "voice_actors_html": "",
        }
    ]

    assert _parse_episode_characters(items)[0].role == "Supporting"


def test_parse_episode_characters_several_voice_actors_reads_each_language() -> None:
    voice_actors_html = (
        '<a class="fw-b" href="https://myanimelist.net/people/70/X">Tanaka, Mayumi</a>'
        " (Japanese)<br>"
        '<a class="fw-b" href="https://myanimelist.net/people/81/Y">Clinkenbeard, Colleen</a>'
        " (English)<br>"
    )
    items = [
        {
            "char_name": "Luffy",
            "char_url": "https://myanimelist.net/character/40",
            "role": "Main",
            "voice_actors_html": voice_actors_html,
        }
    ]

    voice_actors = _parse_episode_characters(items)[0].voice_actors

    assert [(actor.person_id, actor.language) for actor in voice_actors] == [
        (70, "Japanese"),
        (81, "English"),
    ]


def test_parse_episode_staff_regular_episode_reads_named_and_unnamed_roles(
    mal_episode_extracted,
) -> None:
    staff = _parse_episode_staff(mal_episode_extracted["staff"])

    first_named = next(member for member in staff if member.role is not None)
    assert len(staff) == 13
    assert (first_named.name, first_named.person_id, first_named.role) == (
        "Takegami, Junki",
        5163,
        "Script",
    )
    assert len([member for member in staff if member.role is None]) == 4


def test_parse_episode_staff_filler_episode_returns_empty_list(
    mal_episode_filler_extracted,
) -> None:
    assert _parse_episode_staff(mal_episode_filler_extracted["staff"]) == []


@pytest.mark.parametrize(
    "items",
    [
        None,
        [],
        [{"name": "", "person_url": "https://myanimelist.net/people/1", "role": "X"}],
        [
            {
                "name": "X",
                "person_url": "https://myanimelist.net/character/1",
                "role": "X",
            }
        ],
    ],
)
def test_parse_episode_staff_without_name_or_person_link_returns_empty_list(
    items: list[dict] | None,
) -> None:
    assert _parse_episode_staff(items) == []


@pytest.mark.parametrize(("role", "expected"), [("", None), ("Script", "Script")])
def test_parse_episode_staff_person_link_returns_member_with_role(
    role: str, expected: str | None
) -> None:
    items = [
        {
            "name": "Takegami, Junki",
            "person_url": "https://myanimelist.net/people/999/X",
            "role": role,
        }
    ]

    staff = _parse_episode_staff(items)

    assert [(member.person_id, member.role) for member in staff] == [(999, expected)]


def test_build_episode_from_raw_regular_episode_returns_full_episode(
    mal_episode_extracted,
) -> None:
    episode = build_episode_from_raw(dict(mal_episode_extracted), 1, _EP1_URL)

    assert episode.title == _LUFFY_TITLE
    assert episode.title_japanese == "俺はルフィ！海賊王になる男だ！"
    assert (episode.aired, episode.duration) == ("1999-10-20", 1477)
    assert episode.synopsis is not None
    assert "Alvida" in episode.synopsis
    assert (episode.filler, episode.recap) == (False, False)
    assert (len(episode.characters), len(episode.staff)) == (10, 13)


def test_build_episode_from_raw_filler_episode_returns_badge_without_cast(
    mal_episode_filler_extracted,
) -> None:
    episode = build_episode_from_raw(dict(mal_episode_filler_extracted), 50, _EP50_URL)

    assert episode.title == "Usopp vs. Daddy the Parent! Showdown at High!"
    assert (episode.filler, episode.recap) == (True, False)
    assert episode.aired == "2000-11-29"
    assert (episode.characters, episode.staff) == ([], [])


def test_build_episode_from_raw_recap_episode_returns_badge_and_duration(
    mal_episode_recap_extracted,
) -> None:
    episode = build_episode_from_raw(dict(mal_episode_recap_extracted), 279, _EP279_URL)

    assert episode.title == "Jump Towards the Falls! Luffy's Feelings!"
    assert (episode.recap, episode.filler) == (True, False)
    assert (episode.aired, episode.duration) == ("2006-10-01", 1440)


def test_build_episode_from_raw_episode_without_synopsis_returns_no_synopsis(
    mal_episode_no_synopsis_extracted,
) -> None:
    episode = build_episode_from_raw(
        dict(mal_episode_no_synopsis_extracted), 1152, _EP1152_URL
    )

    assert episode.title == "Her Father and Mother's Legacy! Bonney's Nika Punch"
    assert (episode.synopsis, episode.duration) == (None, None)
    assert episode.aired == "2025-12-07"


def test_build_episode_from_raw_without_title_returns_episode_number_title() -> None:
    episode = build_episode_from_raw(
        {}, 42, "https://myanimelist.net/anime/21/One_Piece/episode/42"
    )

    assert episode.title == "Episode 42"


def test_build_episode_from_raw_saved_address_wins_over_given_address(
    mal_episode_extracted,
) -> None:
    raw = {**mal_episode_extracted, "_url": _EP1_URL}

    episode = build_episode_from_raw(raw, 1, "https://other.url/episode/1")

    assert episode.source == _EP1_URL


@pytest.mark.parametrize(
    "page_name",
    [
        "one_piece_episodes",
        "one_piece_episodes_offset_100",
        "one_piece_episodes_offset_200",
        "cowboy_bebop_episodes",
    ],
)
def test_extract_episode_list_from_html_saved_page_matches_saved_values(
    read_page, data_regression, page_name: str
) -> None:
    data_regression.check(
        {
            "episodes": [
                episode.model_dump(mode="json")
                for episode in extract_episode_list_from_html(read_page(page_name))
            ]
        },
        basename=page_name,
    )


def test_extract_episode_list_from_html_reads_every_row_with_slug_url(
    read_page,
) -> None:
    first_page = extract_episode_list_from_html(read_page("one_piece_episodes"))
    second_page = extract_episode_list_from_html(
        read_page("one_piece_episodes_offset_100")
    )

    assert [episode.episode_number for episode in first_page] == list(range(1, 101))
    assert [episode.episode_number for episode in second_page] == list(range(101, 201))
    assert first_page[49].source == (
        "https://myanimelist.net/anime/21/One_Piece/episode/50"
    )
    assert first_page[49].filler is True
    assert first_page[0].filler is False


def test_extract_episode_list_from_html_empty_page_gives_no_episodes() -> None:
    assert extract_episode_list_from_html("") == []
    assert extract_episode_list_from_html("<html><body></body></html>") == []


def test_extract_episode_list_from_html_recap_mark_sets_recap(read_page) -> None:
    episodes_by_number = {
        episode.episode_number: episode
        for episode in extract_episode_list_from_html(
            read_page("one_piece_episodes_offset_200")
        )
    }

    assert episodes_by_number[279].recap is True
    assert episodes_by_number[279].filler is False
    assert sum(episode.recap for episode in episodes_by_number.values()) == 5
