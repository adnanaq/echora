import pytest
from enrichment.sources.anisearch.anisearch_anime_models import (
    AniSearchAnime,
    AniSearchCharacter,
    AniSearchCharacterAnimeRole,
    AniSearchEpisode,
    AniSearchRelatedEntry,
    AniSearchStatistics,
    AniSearchVoiceActorRef,
)
from enrichment.sources.anisearch.anisearch_mapper import (
    anime_from_anisearch,
    character_from_anisearch,
    episode_from_anisearch,
)

ONE_PIECE_URL = "https://www.anisearch.com/anime/2227,one-piece"
ONE_PIECE_EPISODES_URL = f"{ONE_PIECE_URL}/episodes"
LUFFY_URL = "https://www.anisearch.com/character/3006,monkey-d-luffy"


def _episode(**fields) -> AniSearchEpisode:
    return AniSearchEpisode(
        **{
            "episode_number": 1,
            "duration": 1440,
            "aired": "1999-10-20",
            "title": "I'm Luffy! The Man Who's Gonna Be King Of The Pirates!",
            "title_romaji": "Ore wa Luffy! Kaizoku Ou ni naru Otoko da!",
            "title_japanese": "俺はルフィ!海賊王になる男だ!",
            "source": ONE_PIECE_EPISODES_URL,
            **fields,
        }
    )


def _character(**fields) -> AniSearchCharacter:
    return AniSearchCharacter(
        **{"source": LUFFY_URL, "name": "Monkey D. Luffy", **fields}
    )


def test_anime_from_anisearch_titles_synopsis_genres_and_tags_mapped() -> None:
    mapped = anime_from_anisearch(
        AniSearchAnime(
            title="One Piece",
            title_japanese="ワンピース",
            synonyms=["OP"],
            synopsis="Pirates.",
            genres=["Adventure"],
            tags=["Pirates"],
            url=ONE_PIECE_URL,
        )
    )
    assert (mapped["title"], mapped["title_japanese"], mapped["synonyms"]) == (
        "One Piece",
        "ワンピース",
        ["OP"],
    )
    assert (mapped["synopsis"], mapped["genres"], mapped["tags"]) == (
        "Pirates.",
        ["Adventure"],
        ["Pirates"],
    )
    assert mapped["sources"] == [ONE_PIECE_URL]


def test_anime_from_anisearch_without_title_takes_japanese_title() -> None:
    assert anime_from_anisearch(AniSearchAnime(title_japanese="ワンピース"))[
        "title"
    ] == ("ワンピース")


def test_anime_from_anisearch_without_any_title_gives_empty_title() -> None:
    assert anime_from_anisearch(AniSearchAnime())["title"] == ""


def test_anime_from_anisearch_without_page_url_gives_no_sources() -> None:
    assert not anime_from_anisearch(AniSearchAnime()).get("sources")


@pytest.mark.parametrize(
    ("raw_type", "expected"),
    [("TV-Series", "TV"), ("Movie", "MOVIE"), (None, "UNKNOWN")],
)
def test_anime_from_anisearch_type_gives_anime_type(
    raw_type: str | None, expected: str
) -> None:
    assert anime_from_anisearch(AniSearchAnime(type=raw_type))["type"] == expected


def test_anime_from_anisearch_source_material_gives_source_material_type() -> None:
    mapped = anime_from_anisearch(AniSearchAnime(source_material="Light Novel"))
    assert mapped["source_material"] == "LIGHT NOVEL"


def test_anime_from_anisearch_without_source_material_omits_it() -> None:
    assert "source_material" not in anime_from_anisearch(AniSearchAnime())


def test_anime_from_anisearch_full_start_date_sets_year_season_and_date() -> None:
    mapped = anime_from_anisearch(
        AniSearchAnime(start_date="1999-10-20", start_year=1999)
    )
    assert (mapped["year"], mapped["season"], mapped["aired_dates"]["aired_from"]) == (
        1999,
        "FALL",
        "1999-10-19T15:00:00Z",
    )
    assert "month" not in mapped


def test_anime_from_anisearch_year_only_start_sets_year_without_season_month_or_date() -> (
    None
):
    mapped = anime_from_anisearch(AniSearchAnime(start_year=2027, status="Upcoming"))
    assert mapped["year"] == 2027
    assert not {"season", "month", "aired_dates"} & mapped.keys()


def test_anime_from_anisearch_month_and_year_start_sets_year_and_month() -> None:
    mapped = anime_from_anisearch(
        AniSearchAnime(start_year=2008, start_month="November")
    )
    assert (mapped["year"], mapped["month"]) == (2008, "November")
    assert "aired_dates" not in mapped


def test_anime_from_anisearch_end_date_alone_gives_aired_to_only() -> None:
    mapped = anime_from_anisearch(AniSearchAnime(end_date="2002-03-31"))
    assert mapped["aired_dates"] == {"aired_to": "2002-03-30T15:00:00Z"}


def test_anime_from_anisearch_undated_takes_anisearch_status() -> None:
    assert (
        anime_from_anisearch(AniSearchAnime(status="Upcoming"))["status"] == "UPCOMING"
    )


def test_anime_from_anisearch_stated_status_wins_over_dates() -> None:
    mapped = anime_from_anisearch(
        AniSearchAnime(start_date="1994-04-28", status="Completed")
    )
    assert mapped["status"] == "FINISHED"


def test_anime_from_anisearch_stated_status_wins_over_future_start_date() -> None:
    mapped = anime_from_anisearch(
        AniSearchAnime(start_date="2099-01-01", status="Completed")
    )
    assert mapped["status"] == "FINISHED"


def test_anime_from_anisearch_without_stated_status_derives_status_from_dates() -> None:
    mapped = anime_from_anisearch(AniSearchAnime(start_date="1999-10-20"))
    assert mapped["status"] == "ONGOING"


def test_anime_from_anisearch_unrecognised_status_derives_status_from_dates() -> None:
    mapped = anime_from_anisearch(
        AniSearchAnime(start_date="1999-10-20", status="Something New")
    )
    assert mapped["status"] == "ONGOING"


def test_anime_from_anisearch_statistics_score_rescaled_from_five_stars() -> None:
    statistics = anime_from_anisearch(
        AniSearchAnime(
            statistics=AniSearchStatistics(score=4.18, scored_by=7902, rank=124)
        )
    )["statistics"]["anisearch"]
    assert (statistics["score"], statistics["scored_by"], statistics["rank"]) == (
        8.36,
        7902,
        124,
    )


def test_anime_from_anisearch_trending_alone_gives_no_statistics() -> None:
    mapped = anime_from_anisearch(
        AniSearchAnime(statistics=AniSearchStatistics(trending=5))
    )
    assert not mapped.get("statistics")


def test_anime_from_anisearch_cover_image_becomes_cover() -> None:
    mapped = anime_from_anisearch(AniSearchAnime(cover_image="https://cdn/cover.webp"))
    assert mapped["images"]["covers"] == ["https://cdn/cover.webp"]


def test_anime_from_anisearch_broadcast_parts_give_broadcast() -> None:
    mapped = anime_from_anisearch(
        AniSearchAnime(
            broadcast_day="Sunday", broadcast_time="09:30", broadcast_timezone="JST"
        )
    )
    assert mapped["broadcast"] == {"day": "Sunday", "time": "09:30", "timezone": "JST"}


def test_anime_from_anisearch_broadcast_day_alone_gives_partial_broadcast() -> None:
    mapped = anime_from_anisearch(AniSearchAnime(broadcast_day="Sunday"))
    assert mapped["broadcast"] == {"day": "Sunday"}


def test_anime_from_anisearch_without_broadcast_omits_broadcast() -> None:
    assert "broadcast" not in anime_from_anisearch(AniSearchAnime())


def test_anime_from_anisearch_studio_becomes_studio_company() -> None:
    mapped = anime_from_anisearch(
        AniSearchAnime(
            studio="Toei Animation Co., Ltd.",
            studio_url="https://www.anisearch.com/company/412,toei-animation-co-ltd",
        )
    )
    assert mapped["companies"] == [
        {
            "name": "Toei Animation Co., Ltd.",
            "roles": ["STUDIO"],
            "sources": ["https://www.anisearch.com/company/412,toei-animation-co-ltd"],
        }
    ]
    assert "studios" not in mapped


def test_anime_from_anisearch_studio_without_page_gives_company_without_sources() -> (
    None
):
    mapped = anime_from_anisearch(AniSearchAnime(studio="Toei Animation"))
    assert mapped["companies"][0]["name"] == "Toei Animation"
    assert not mapped["companies"][0].get("sources")


def test_anime_from_anisearch_without_studio_gives_no_companies() -> None:
    assert not anime_from_anisearch(AniSearchAnime()).get("companies")


def test_anime_from_anisearch_websites_with_address_become_external_sources() -> None:
    mapped = anime_from_anisearch(
        AniSearchAnime(
            websites=[
                {"name": "Official Website", "url": "https://www.onepiece-anime.com/"},
                {"name": "Broken"},
            ]
        )
    )
    assert [
        (link["platform"], link["source"]) for link in mapped["external_sources"]
    ] == [("official_site", "https://www.onepiece-anime.com/")]


def test_anime_from_anisearch_anime_relations_grouped_by_relation() -> None:
    mapped = anime_from_anisearch(
        AniSearchAnime(
            anime_relations=[
                AniSearchRelatedEntry(
                    relation_type="Sequel",
                    title="One Piece Film: Red",
                    url="anime/16735,one-piece-film-red",
                    details="Movie, 1 (2022)",
                    image="https://cdn/red.webp",
                ),
                AniSearchRelatedEntry(
                    relation_type="Side Story",
                    title="Episode of Nami",
                    url="https://www.anisearch.com/anime/8000,episode-of-nami",
                    details="TV-Special, 1 (2012)",
                ),
                AniSearchRelatedEntry(title="Unlabelled"),
                AniSearchRelatedEntry(relation_type="Sequel"),
                AniSearchRelatedEntry(relation_type="Sequel", title="Stampede"),
            ]
        )
    )
    related = mapped["related_anime"]
    assert related["SEQUEL"][0] == {
        "title": "One Piece Film: Red",
        "type": "MOVIE",
        "sources": ["https://www.anisearch.com/anime/16735,one-piece-film-red"],
        "images": ["https://cdn/red.webp"],
    }
    assert [entry["title"] for entry in related["SEQUEL"]] == [
        "One Piece Film: Red",
        "Stampede",
    ]
    assert related["SIDE_STORY"][0]["sources"] == [
        "https://www.anisearch.com/anime/8000,episode-of-nami"
    ]
    assert related["OTHER"][0]["title"] == "Unlabelled"
    assert related["OTHER"][0]["type"] == "UNKNOWN"
    assert not related["OTHER"][0].get("sources")


def test_anime_from_anisearch_manga_relations_grouped_by_relation() -> None:
    mapped = anime_from_anisearch(
        AniSearchAnime(
            manga_relations=[
                AniSearchRelatedEntry(
                    relation_type="Adaptation",
                    title="One Piece",
                    url="/manga/1,one-piece",
                    details="Manga, 110 (1997)",
                ),
                AniSearchRelatedEntry(relation_type="Spin-off", title="Wanted!"),
                AniSearchRelatedEntry(relation_type="Adaptation"),
                AniSearchRelatedEntry(
                    relation_type="Adaptation",
                    title="One Piece Novel",
                    image="https://cdn/novel.webp",
                ),
            ]
        )
    )
    related = mapped["related_source_material"]
    assert related["ADAPTATION"] == [
        {
            "title": "One Piece",
            "type": "MANGA",
            "sources": ["https://www.anisearch.com/manga/1,one-piece"],
            "images": [],
        },
        {
            "title": "One Piece Novel",
            "type": "UNKNOWN",
            "sources": [],
            "images": ["https://cdn/novel.webp"],
        },
    ]
    assert [entry["title"] for entry in related["SPIN_OFF"]] == ["Wanted!"]


def test_character_from_anisearch_name_and_page_only_gives_name_and_source() -> None:
    mapped = character_from_anisearch(_character())
    assert (mapped["name"], mapped["sources"]) == ("Monkey D. Luffy", [LUFFY_URL])
    assert not {
        key
        for key in (
            "name_native",
            "description",
            "favorites",
            "images",
            "traits",
            "roles",
            "animeography",
            "mangaography",
            "voice_actors",
            "attributes",
        )
        if mapped.get(key)
    }


def test_character_from_anisearch_without_name_gives_empty_name() -> None:
    assert character_from_anisearch(_character(name=None))["name"] == ""


def test_character_from_anisearch_profile_fields_map_to_character_fields() -> None:
    mapped = character_from_anisearch(
        _character(
            name_native="モンキー・D・ルフィ",
            description="Captain.",
            favorites=1200,
            tags=["Rubber"],
            attributes={"gender": "Male"},
        )
    )
    assert (mapped["name_native"], mapped["description"], mapped["favorites"]) == (
        "モンキー・D・ルフィ",
        "Captain.",
        1200,
    )
    assert (mapped["traits"], mapped["attributes"]) == (["Rubber"], {"gender": "Male"})


def test_character_from_anisearch_images_join_portrait_screenshots_and_pictures() -> (
    None
):
    mapped = character_from_anisearch(
        _character(
            image="https://cdn/portrait.webp",
            screenshot_images=["https://cdn/shot.webp"],
            picture_images=["https://cdn/picture.webp"],
        )
    )
    assert mapped["images"] == [
        "https://cdn/portrait.webp",
        "https://cdn/shot.webp",
        "https://cdn/picture.webp",
    ]


def test_character_from_anisearch_screenshots_without_portrait_give_images() -> None:
    mapped = character_from_anisearch(
        _character(screenshot_images=["https://cdn/shot.webp"])
    )
    assert mapped["images"] == ["https://cdn/shot.webp"]


def test_character_from_anisearch_section_role_gives_roles() -> None:
    assert character_from_anisearch(_character(role="Main Character"))["roles"] == [
        "MAIN"
    ]


def test_character_from_anisearch_section_role_applies_only_to_its_own_anime() -> None:
    mapped = character_from_anisearch(
        _character(
            role="Main Character",
            anime_url="https://www.anisearch.com/anime/2227,one-piece/characters",
            anime_ography=[
                AniSearchCharacterAnimeRole(
                    title="One Piece",
                    url="https://www.anisearch.com/anime/2227,one-piece",
                ),
                AniSearchCharacterAnimeRole(
                    title="One Piece Film: Red",
                    url="https://www.anisearch.com/anime/22270,one-piece-film-red",
                ),
                AniSearchCharacterAnimeRole(title="Unlinked"),
            ],
        )
    )
    assert [(entry["title"], entry["role"]) for entry in mapped["animeography"]] == [
        ("One Piece", "MAIN"),
        ("One Piece Film: Red", "UNKNOWN"),
        ("Unlinked", "UNKNOWN"),
    ]


def test_character_from_anisearch_stated_entry_role_wins_over_section_role() -> None:
    mapped = character_from_anisearch(
        _character(
            role="Main Character",
            anime_url=ONE_PIECE_URL,
            anime_ography=[
                AniSearchCharacterAnimeRole(
                    title="One Piece", url=ONE_PIECE_URL, role="Secondary Character"
                )
            ],
        )
    )
    assert mapped["animeography"][0]["role"] == "SUPPORTING"


def test_character_from_anisearch_section_role_without_anime_page_leaves_entries_unknown() -> (
    None
):
    mapped = character_from_anisearch(
        _character(
            role="Main Character",
            anime_ography=[
                AniSearchCharacterAnimeRole(title="One Piece", url=ONE_PIECE_URL)
            ],
        )
    )
    assert mapped["animeography"][0]["role"] == "UNKNOWN"


def test_character_from_anisearch_unrecognised_anime_page_leaves_entries_unknown() -> (
    None
):
    mapped = character_from_anisearch(
        _character(
            role="Main Character",
            anime_url="https://example.com/anime/2227",
            anime_ography=[
                AniSearchCharacterAnimeRole(title="One Piece", url=ONE_PIECE_URL)
            ],
        )
    )
    assert mapped["animeography"][0]["role"] == "UNKNOWN"


def test_character_from_anisearch_without_full_list_uses_detail_page_appearances() -> (
    None
):
    mapped = character_from_anisearch(
        _character(
            anime_roles=[
                AniSearchCharacterAnimeRole(
                    title="One Piece", url=ONE_PIECE_URL, role="Main Character"
                ),
                AniSearchCharacterAnimeRole(title=""),
            ]
        )
    )
    assert mapped["animeography"] == [
        {"title": "One Piece", "role": "MAIN", "sources": [ONE_PIECE_URL]}
    ]


def test_character_from_anisearch_manga_appearances_give_mangaography() -> None:
    mapped = character_from_anisearch(
        _character(
            manga_ography=[
                AniSearchCharacterAnimeRole(
                    title="One Piece",
                    url="https://www.anisearch.com/manga/1,one-piece",
                    role="Main Character",
                ),
                AniSearchCharacterAnimeRole(title="Unlinked"),
                AniSearchCharacterAnimeRole(title=""),
            ]
        )
    )
    assert [
        (entry["title"], entry["role"], entry.get("sources") or [])
        for entry in mapped["mangaography"]
    ] == [
        ("One Piece", "MAIN", ["https://www.anisearch.com/manga/1,one-piece"]),
        ("Unlinked", "UNKNOWN", []),
    ]


def test_character_from_anisearch_named_voice_actors_mapped_with_language() -> None:
    mapped = character_from_anisearch(
        _character(
            voice_actors=[
                AniSearchVoiceActorRef(
                    name="Mayumi Tanaka",
                    language="Japanese",
                    url="https://www.anisearch.com/person/1,mayumi-tanaka",
                ),
                AniSearchVoiceActorRef(name="Unlinked Actor", language="German"),
                AniSearchVoiceActorRef(name="", language="English"),
            ]
        )
    )
    assert [
        (actor["name"], actor["language"], actor.get("sources") or [])
        for actor in mapped["voice_actors"]
    ] == [
        (
            "Mayumi Tanaka",
            "Japanese",
            ["https://www.anisearch.com/person/1,mayumi-tanaka"],
        ),
        ("Unlinked Actor", "German", []),
    ]


def test_episode_from_anisearch_number_flags_and_duration_mapped() -> None:
    mapped = episode_from_anisearch(_episode(is_filler=True, is_recap=True))
    assert (
        mapped["episode_number"],
        mapped["filler"],
        mapped["recap"],
        mapped["duration"],
    ) == (1, True, True, 1440)


def test_episode_from_anisearch_default_flags_give_regular_episode() -> None:
    mapped = episode_from_anisearch(_episode())
    assert (mapped["filler"], mapped["recap"]) == (False, False)


def test_episode_from_anisearch_without_duration_omits_duration() -> None:
    assert "duration" not in episode_from_anisearch(_episode(duration=None))


def test_episode_from_anisearch_aired_date_normalized_to_utc() -> None:
    assert episode_from_anisearch(_episode())["aired"] == "1999-10-19T15:00:00Z"


def test_episode_from_anisearch_without_aired_date_omits_aired() -> None:
    assert "aired" not in episode_from_anisearch(_episode(aired=None))


def test_episode_from_anisearch_titles_mapped() -> None:
    mapped = episode_from_anisearch(_episode(titles={"de": "Ich bin Ruffy!"}))
    assert mapped["title_romaji"] == "Ore wa Luffy! Kaizoku Ou ni naru Otoko da!"
    assert mapped["title_japanese"] == "俺はルフィ!海賊王になる男だ!"
    assert mapped["titles"] == {"de": "Ich bin Ruffy!"}


def test_episode_from_anisearch_without_japanese_title_omits_it() -> None:
    assert "title_japanese" not in episode_from_anisearch(_episode(title_japanese=None))


def test_episode_from_anisearch_without_title_gives_numbered_title() -> None:
    assert episode_from_anisearch(_episode(title=None))["title"] == "Episode 1"


def test_episode_from_anisearch_episodes_page_becomes_source() -> None:
    assert episode_from_anisearch(_episode())["sources"] == [ONE_PIECE_EPISODES_URL]


def test_episode_from_anisearch_without_episodes_page_gives_no_sources() -> None:
    assert not episode_from_anisearch(_episode(source=None)).get("sources")


def test_episode_from_anisearch_output_has_no_anime_id() -> None:
    assert "anime_id" not in episode_from_anisearch(_episode())
