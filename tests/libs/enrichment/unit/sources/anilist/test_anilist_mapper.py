import pytest
from enrichment.sources.anilist.anilist_anime_models import AniListAnime
from enrichment.sources.anilist.anilist_character_models import (
    AniListCharacterEdge,
    AniListFuzzyDate,
)
from enrichment.sources.anilist.anilist_mapper import (
    _fuzzy_date_str,
    anime_from_anilist,
    character_from_anilist,
)


def _anime(**overrides) -> AniListAnime:
    data = {
        "id": 21,
        "idMal": 21,
        "title": {"romaji": "ONE PIECE", "english": "ONE PIECE", "native": "ONE PIECE"},
        "format": "TV",
        "status": "RELEASING",
        "source": "MANGA",
        "episodes": 1000,
        "duration": 24,
        "isAdult": False,
        "seasonYear": 1999,
        "season": "FALL",
        "countryOfOrigin": "JP",
        "description": "Monkey D. Luffy sets sail.",
        "averageScore": 87,
        "popularity": 673293,
        "favourites": 98448,
        "stats": {
            "scoreDistribution": [
                {"score": 70, "amount": 20793},
                {"score": 80, "amount": 35929},
                {"score": 90, "amount": 74887},
                {"score": 100, "amount": 137750},
            ]
        },
        "genres": ["Action", "Adventure"],
        "synonyms": ["OP"],
        "tags": [],
        "studios": {"edges": []},
        "relations": {"edges": []},
        "externalLinks": [],
        "rankings": [],
    }
    return AniListAnime.model_validate({**data, **overrides})


def _relation(node: dict, relation_type: str) -> dict:
    return {
        "relations": {
            "edges": [
                {
                    "node": {
                        "seasonYear": None,
                        "averageScore": None,
                        "coverImage": None,
                        "episodes": None,
                        "chapters": None,
                        "volumes": None,
                        **node,
                    },
                    "relationType": relation_type,
                }
            ]
        }
    }


def _character_edge(**overrides) -> AniListCharacterEdge:
    data = {
        "node": {
            "id": 40,
            "name": {
                "full": "Monkey D. Luffy",
                "native": "モンキー・D・ルフィ",
                "alternative": ["Luffy"],
                "alternativeSpoiler": [],
            },
            "image": {"large": "https://anilist.co/img/luffy.jpg"},
            "description": None,
            "gender": "Male",
            "age": "19",
            "bloodType": "F",
            "favourites": 50000,
            "siteUrl": "https://anilist.co/character/40",
        },
        "role": "MAIN",
        "voiceActorRoles": [
            {
                "voiceActor": {
                    "id": 95,
                    "name": {"full": "Mayumi Tanaka", "native": "田中真弓"},
                    "languageV2": "Japanese",
                    "image": {"large": "https://anilist.co/img/tanaka.jpg"},
                    "siteUrl": "https://anilist.co/staff/95",
                },
            }
        ],
    }
    return AniListCharacterEdge.model_validate({**data, **overrides})


def _usopp_edge(description: str) -> AniListCharacterEdge:
    return _character_edge(
        node={
            "id": 40,
            "name": {
                "full": "Usopp",
                "native": None,
                "alternative": [],
                "alternativeSpoiler": [],
            },
            "description": description,
        }
    )


@pytest.mark.parametrize(
    ("date", "expected"),
    [
        (AniListFuzzyDate(year=1999, month=10, day=20), "1999-10-20"),
        (AniListFuzzyDate(year=1999, month=10), "1999-10"),
        (AniListFuzzyDate(year=1999), "1999"),
    ],
)
def test_fuzzy_date_str_known_parts_give_date_at_that_precision(
    date: AniListFuzzyDate, expected: str
) -> None:
    assert _fuzzy_date_str(date) == expected


@pytest.mark.parametrize("date", [None, AniListFuzzyDate(month=10, day=20)])
def test_fuzzy_date_str_without_year_returns_none(
    date: AniListFuzzyDate | None,
) -> None:
    assert _fuzzy_date_str(date) is None


def test_anime_from_anilist_full_start_date_sets_aired_dates_without_month() -> None:
    result = anime_from_anilist(
        _anime(
            startDate={"year": 1999, "month": 10, "day": 20},
            endDate={"year": 2000, "month": 3, "day": 26},
        )
    )
    assert result["aired_dates"] == {
        "aired_from": "1999-10-19T15:00:00Z",
        "aired_to": "2000-03-25T15:00:00Z",
    }
    assert "month" not in result


def test_anime_from_anilist_month_without_day_sets_month_without_aired_dates() -> None:
    result = anime_from_anilist(
        _anime(startDate={"year": 1977, "month": 10, "day": None})
    )
    assert result["month"] == "October"
    assert "aired_dates" not in result


def test_anime_from_anilist_without_season_year_takes_year_from_start_date() -> None:
    result = anime_from_anilist(
        _anime(
            seasonYear=None,
            season=None,
            startDate={"year": 2027, "month": None, "day": None},
        )
    )
    assert result["year"] == 2027
    assert not {"month", "aired_dates", "season"} & result.keys()


def test_anime_from_anilist_season_year_wins_over_start_date_year() -> None:
    result = anime_from_anilist(
        _anime(seasonYear=2027, startDate={"year": 2026, "month": 12, "day": 27})
    )
    assert result["year"] == 2027


def test_anime_from_anilist_titles_and_synopsis_mapped() -> None:
    result = anime_from_anilist(
        _anime(
            title={
                "romaji": "ONE PIECE",
                "english": "One Piece",
                "native": "ワンピース",
            }
        )
    )
    assert (result["title"], result["title_english"], result["title_japanese"]) == (
        "ONE PIECE",
        "One Piece",
        "ワンピース",
    )
    assert result["synopsis"] == "Monkey D. Luffy sets sail."


def test_anime_from_anilist_without_title_gives_empty_title() -> None:
    assert anime_from_anilist(_anime(title=None))["title"] == ""


@pytest.mark.parametrize(("raw_format", "expected"), [("TV", "TV"), (None, "UNKNOWN")])
def test_anime_from_anilist_format_gives_anime_type(
    raw_format: str | None, expected: str
) -> None:
    assert anime_from_anilist(_anime(format=raw_format))["type"] == expected


@pytest.mark.parametrize(
    ("raw_status", "expected"),
    [("RELEASING", "ONGOING"), ("FINISHED", "FINISHED")],
)
def test_anime_from_anilist_status_gives_anime_status(
    raw_status: str, expected: str
) -> None:
    assert anime_from_anilist(_anime(status=raw_status))["status"] == expected


@pytest.mark.parametrize(
    ("raw_source", "expected"),
    [("MANGA", "MANGA"), ("VIDEO_GAME", "GAME")],
)
def test_anime_from_anilist_source_gives_source_material(
    raw_source: str, expected: str
) -> None:
    assert anime_from_anilist(_anime(source=raw_source))["source_material"] == expected


def test_anime_from_anilist_without_source_omits_source_material() -> None:
    assert "source_material" not in anime_from_anilist(_anime(source=None))


def test_anime_from_anilist_episode_count_kept() -> None:
    assert anime_from_anilist(_anime())["episode_count"] == 1000


def test_anime_from_anilist_without_episode_count_gives_zero() -> None:
    assert anime_from_anilist(_anime(episodes=None))["episode_count"] == 0


def test_anime_from_anilist_duration_in_minutes_gives_seconds() -> None:
    assert anime_from_anilist(_anime(duration=24))["duration"] == 1440


def test_anime_from_anilist_without_duration_omits_duration() -> None:
    assert "duration" not in anime_from_anilist(_anime(duration=None))


@pytest.mark.parametrize("is_adult", [False, True])
def test_anime_from_anilist_adult_flag_gives_nsfw(is_adult: bool) -> None:
    assert anime_from_anilist(_anime(isAdult=is_adult))["nsfw"] is is_adult


def test_anime_from_anilist_season_year_and_season_mapped() -> None:
    result = anime_from_anilist(_anime(season="SPRING"))
    assert (result["year"], result["season"]) == (1999, "SPRING")


def test_anime_from_anilist_without_season_omits_season() -> None:
    assert "season" not in anime_from_anilist(_anime(season=None))


def test_anime_from_anilist_country_of_origin_kept() -> None:
    assert anime_from_anilist(_anime(countryOfOrigin="CN"))["country_of_origin"] == "CN"


def test_anime_from_anilist_without_country_omits_country_of_origin() -> None:
    assert "country_of_origin" not in anime_from_anilist(_anime(countryOfOrigin=None))


def test_anime_from_anilist_ids_give_anilist_and_mal_sources() -> None:
    assert anime_from_anilist(_anime(id=21, idMal=21))["sources"] == [
        "https://anilist.co/anime/21",
        "https://myanimelist.net/anime/21",
    ]


def test_anime_from_anilist_without_mal_id_gives_anilist_source_only() -> None:
    assert anime_from_anilist(_anime(idMal=None))["sources"] == [
        "https://anilist.co/anime/21"
    ]


def test_anime_from_anilist_genres_and_synonyms_kept() -> None:
    result = anime_from_anilist(_anime())
    assert (result["genres"], result["synonyms"]) == (["Action", "Adventure"], ["OP"])


def test_anime_from_anilist_demographic_tag_becomes_demographic() -> None:
    result = anime_from_anilist(
        _anime(tags=[{"name": "Shounen", "category": "Demographic", "isAdult": False}])
    )
    assert "Shounen" in result["demographics"]
    assert "Shounen" not in result.get("tags", [])


def test_anime_from_anilist_theme_tag_becomes_theme() -> None:
    result = anime_from_anilist(
        _anime(
            tags=[
                {
                    "name": "Travel",
                    "description": "Moving around",
                    "category": "Theme-Action",
                    "isAdult": False,
                }
            ]
        )
    )
    assert [theme["name"] for theme in result["themes"]] == ["Travel"]


def test_anime_from_anilist_other_tag_becomes_plain_tag() -> None:
    result = anime_from_anilist(
        _anime(
            tags=[{"name": "Pirates", "category": "Cast-Main Cast", "isAdult": False}]
        )
    )
    assert "Pirates" in result["tags"]


def test_anime_from_anilist_adult_tag_becomes_content_warning_only() -> None:
    result = anime_from_anilist(
        _anime(tags=[{"name": "Nudity", "category": "Theme-Ecchi", "isAdult": True}])
    )
    assert "Nudity" in result["content_warnings"]
    assert "Nudity" not in result.get("tags", [])
    assert not result.get("themes")


def test_anime_from_anilist_animation_studios_and_other_companies_get_roles() -> None:
    result = anime_from_anilist(
        _anime(
            studios={
                "edges": [
                    {
                        "node": {
                            "id": 18,
                            "name": "Toei Animation",
                            "isAnimationStudio": True,
                        }
                    },
                    {
                        "node": {
                            "id": 102,
                            "name": "Funimation",
                            "isAnimationStudio": False,
                        }
                    },
                ]
            }
        )
    )
    companies = {company["name"]: company for company in result["companies"]}
    assert companies["Toei Animation"]["roles"] == ["STUDIO"]
    assert companies["Toei Animation"]["sources"] == ["https://anilist.co/studio/18"]
    assert companies["Funimation"]["roles"] == ["PRODUCER"]
    assert not {"studios", "producers", "licensors"} & result.keys()


def test_anime_from_anilist_streaming_link_becomes_streaming_source() -> None:
    result = anime_from_anilist(
        _anime(
            externalLinks=[
                {
                    "id": 1,
                    "url": "https://crunchyroll.com/one-piece",
                    "site": "Crunchyroll",
                    "type": "STREAMING",
                },
            ]
        )
    )
    assert [
        (entry["platform"], entry["source"]) for entry in result["streaming_sources"]
    ] == [("Crunchyroll", "https://crunchyroll.com/one-piece")]


def test_anime_from_anilist_info_link_becomes_labelled_external_source() -> None:
    result = anime_from_anilist(
        _anime(
            externalLinks=[
                {
                    "id": 2,
                    "url": "https://one-piece.com",
                    "site": "Official Site",
                    "type": "INFO",
                },
            ]
        )
    )
    assert result["external_sources"] == [
        {
            "platform": "official_site",
            "source": "https://one-piece.com",
            "label": "Official Site",
        }
    ]


def test_anime_from_anilist_social_link_becomes_external_source() -> None:
    result = anime_from_anilist(
        _anime(
            externalLinks=[
                {
                    "id": 3,
                    "url": "https://twitter.com/onepiece",
                    "site": "Twitter",
                    "type": "SOCIAL",
                },
            ]
        )
    )
    assert [entry["platform"] for entry in result["external_sources"]] == ["twitter"]


def test_anime_from_anilist_unusable_links_dropped() -> None:
    result = anime_from_anilist(
        _anime(
            externalLinks=[
                {"id": 4, "url": None, "site": "Broken", "type": "INFO"},
                {"id": 5, "url": "https://example.com", "site": None, "type": "INFO"},
                {"id": 6, "url": "not a link", "site": "Odd", "type": "INFO"},
                {"id": 7, "url": "https://example.com/x", "site": "X", "type": "OTHER"},
            ]
        )
    )
    assert (result["external_sources"], result["streaming_sources"]) == ([], [])


def test_anime_from_anilist_youtube_trailer_gives_short_link_and_thumbnail() -> None:
    result = anime_from_anilist(
        _anime(
            trailer={
                "id": "abc123",
                "site": "youtube",
                "thumbnail": "https://img.youtube.com/abc123.jpg",
            }
        )
    )
    assert result["trailers"] == [
        {
            "source": "https://youtu.be/abc123",
            "thumbnail": "https://img.youtube.com/abc123.jpg",
        }
    ]


@pytest.mark.parametrize(
    "trailer", [None, {"id": "abc", "site": "dailymotion", "thumbnail": None}]
)
def test_anime_from_anilist_missing_or_non_youtube_trailer_gives_no_trailers(
    trailer: dict | None,
) -> None:
    assert anime_from_anilist(_anime(trailer=trailer)).get("trailers", []) == []


@pytest.mark.parametrize(
    ("cover", "expected"),
    [
        ({"extraLarge": "https://xl.jpg", "large": "https://l.jpg"}, "https://xl.jpg"),
        ({"extraLarge": None, "large": "https://l.jpg"}, "https://l.jpg"),
    ],
)
def test_anime_from_anilist_cover_takes_largest_size(
    cover: dict, expected: str
) -> None:
    assert anime_from_anilist(_anime(coverImage=cover))["images"]["covers"] == [
        expected
    ]


def test_anime_from_anilist_banner_becomes_banner_image() -> None:
    result = anime_from_anilist(_anime(bannerImage="https://banner.jpg"))
    assert result["images"]["banners"] == ["https://banner.jpg"]


def test_anime_from_anilist_statistics_score_votes_members_and_favourites() -> None:
    statistics = anime_from_anilist(_anime(averageScore=87))["statistics"]["anilist"]
    assert statistics["score"] == 8.7
    assert (
        statistics["members"],
        statistics["favorites"],
        statistics["scored_by"],
    ) == (
        673293,
        98448,
        269359,
    )


def test_anime_from_anilist_without_average_score_omits_score() -> None:
    statistics = anime_from_anilist(_anime(averageScore=None))["statistics"]["anilist"]
    assert "score" not in statistics


def test_anime_from_anilist_without_score_distribution_omits_votes() -> None:
    statistics = anime_from_anilist(_anime(stats=None))["statistics"]["anilist"]
    assert "scored_by" not in statistics


def test_anime_from_anilist_rankings_become_contextual_ranks() -> None:
    result = anime_from_anilist(
        _anime(
            rankings=[
                {
                    "rank": 22,
                    "context": "highest rated all time",
                    "format": "TV",
                    "allTime": True,
                },
            ]
        )
    )
    rank = result["statistics"]["anilist"]["contextual_ranks"][0]
    assert (rank["rank"], rank["context"], rank["all_time"]) == (
        22,
        "highest rated all time",
        True,
    )


def test_anime_from_anilist_without_rankings_omits_contextual_ranks() -> None:
    statistics = anime_from_anilist(_anime(rankings=[]))["statistics"]["anilist"]
    assert "contextual_ranks" not in statistics


def test_anime_from_anilist_next_airing_episode_gives_broadcast_time_in_utc() -> None:
    result = anime_from_anilist(
        _anime(
            nextAiringEpisode={
                "episode": 1110,
                "airingAt": 1743865200,
                "timeUntilAiring": 600,
            }
        )
    )
    assert result["broadcast"]["next_episode_at"] == "2025-04-05T15:00:00Z"


def test_anime_from_anilist_without_next_airing_episode_omits_broadcast() -> None:
    assert "broadcast" not in anime_from_anilist(_anime(nextAiringEpisode=None))


def test_anime_from_anilist_anime_format_relation_becomes_related_anime() -> None:
    result = anime_from_anilist(
        _anime(
            **_relation(
                {
                    "id": 100,
                    "format": "MOVIE",
                    "status": "FINISHED",
                    "title": {"romaji": "OP Film"},
                    "seasonYear": 2000,
                    "averageScore": 80,
                    "episodes": 1,
                    "coverImage": {"large": "https://film.jpg"},
                },
                "SIDE_STORY",
            )
        )
    )
    entry = result["related_anime"]["SIDE_STORY"][0]
    assert (entry["title"], entry["type"], entry["status"], entry["year"]) == (
        "OP Film",
        "MOVIE",
        "FINISHED",
        2000,
    )
    assert (entry["score"], entry["episode_count"], entry["images"]) == (
        8.0,
        1,
        ["https://film.jpg"],
    )
    assert entry["sources"] == ["https://anilist.co/anime/100"]


def test_anime_from_anilist_relation_without_title_score_or_status_keeps_entry() -> (
    None
):
    result = anime_from_anilist(
        _anime(
            **_relation(
                {"id": 300, "format": "TV", "status": None, "title": None}, "SEQUEL"
            )
        )
    )
    entry = result["related_anime"]["SEQUEL"][0]
    assert entry["title"] == ""
    assert not {"score", "status"} & entry.keys()


def test_anime_from_anilist_manga_format_relation_becomes_related_source_material() -> (
    None
):
    result = anime_from_anilist(
        _anime(
            **_relation(
                {
                    "id": 200,
                    "format": "MANGA",
                    "status": "RELEASING",
                    "title": {"romaji": "OP Manga"},
                    "averageScore": 90,
                    "chapters": 1100,
                    "volumes": 105,
                    "coverImage": {"extraLarge": "https://xl.jpg"},
                },
                "ADAPTATION",
            )
        )
    )
    entry = result["related_source_material"]["ADAPTATION"][0]
    assert (entry["title"], entry["type"], entry["status"], entry["score"]) == (
        "OP Manga",
        "MANGA",
        "ONGOING",
        9.0,
    )
    assert (entry["chapters"], entry["volumes"], entry["images"]) == (
        1100,
        105,
        ["https://xl.jpg"],
    )
    assert entry["sources"] == ["https://anilist.co/manga/200"]


def test_anime_from_anilist_source_material_without_score_or_status_omits_them() -> (
    None
):
    result = anime_from_anilist(
        _anime(
            **_relation(
                {"id": 201, "format": "NOVEL", "status": None, "title": None},
                "SOURCE",
            )
        )
    )
    entry = result["related_source_material"]["SOURCE"][0]
    assert entry["title"] == ""
    assert not {"score", "status"} & entry.keys()


def test_character_from_anilist_names_mapped() -> None:
    result = character_from_anilist(_character_edge())
    assert (result["name"], result["name_native"], result["name_variations"]) == (
        "Monkey D. Luffy",
        "モンキー・D・ルフィ",
        ["Luffy"],
    )


def test_character_from_anilist_without_name_gives_empty_name() -> None:
    edge = _character_edge()
    edge.node.name = None
    result = character_from_anilist(edge)
    assert result["name"] == ""
    assert "name_native" not in result


@pytest.mark.parametrize(
    ("role", "expected"), [("MAIN", "MAIN"), ("SUPPORTING", "SUPPORTING")]
)
def test_character_from_anilist_role_gives_character_role(
    role: str, expected: str
) -> None:
    assert character_from_anilist(_character_edge(role=role))["roles"] == [expected]


def test_character_from_anilist_favourites_and_image_mapped() -> None:
    result = character_from_anilist(_character_edge())
    assert result["favorites"] == 50000
    assert result["images"] == ["https://anilist.co/img/luffy.jpg"]


def test_character_from_anilist_without_image_gives_no_images() -> None:
    edge = _character_edge()
    edge.node.image = None
    assert character_from_anilist(edge).get("images", []) == []


def test_character_from_anilist_site_url_becomes_source() -> None:
    result = character_from_anilist(_character_edge())
    assert result["sources"] == ["https://anilist.co/character/40"]


def test_character_from_anilist_without_site_url_builds_source_from_id() -> None:
    edge = _character_edge()
    edge.node.site_url = None
    assert character_from_anilist(edge)["sources"] == [
        "https://anilist.co/character/40"
    ]


def test_character_from_anilist_profile_fields_become_attributes() -> None:
    edge = _character_edge()
    edge.node.date_of_birth = AniListFuzzyDate(year=1980, month=5, day=5)
    assert character_from_anilist(edge)["attributes"] == {
        "gender": "Male",
        "age": "19",
        "blood_type": "F",
        "date_of_birth": "1980-05-05",
    }


def test_character_from_anilist_without_birth_date_omits_date_of_birth() -> None:
    assert (
        "date_of_birth" not in character_from_anilist(_character_edge())["attributes"]
    )


def test_character_from_anilist_description_fields_become_attributes_and_prose() -> (
    None
):
    result = character_from_anilist(
        _usopp_edge("__Height:__ 174 cm\n\nUsopp is a liar.")
    )
    assert result["description"] == "Usopp is a liar."
    assert result["attributes"]["height"] == "174 cm"


def test_character_from_anilist_spoiler_description_field_becomes_spoiler() -> None:
    result = character_from_anilist(
        _usopp_edge("__Bounty:__ ~!500,000,000!~\n\nUsopp is brave.")
    )
    assert result["spoilers"]["bounty"] == "500,000,000"
    assert "bounty" not in result["attributes"]


def test_character_from_anilist_prose_over_several_lines_kept_whole() -> None:
    result = character_from_anilist(
        _usopp_edge("__Height:__ 174 cm\n\nFirst prose line.\nSecond prose line.")
    )
    assert "First prose line." in result["description"]
    assert "Second prose line." in result["description"]
    assert result["attributes"]["height"] == "174 cm"


def test_character_from_anilist_without_description_omits_description() -> None:
    assert "description" not in character_from_anilist(_character_edge())


def test_character_from_anilist_spoiler_alternative_names_become_nicknames() -> None:
    edge = _character_edge()
    edge.node.name.alternative_spoiler = ["God Usopp", "King of Snipers"]
    assert character_from_anilist(edge)["nicknames"] == ["God Usopp", "King of Snipers"]


def test_character_from_anilist_voice_actor_mapped_with_language_and_links() -> None:
    actor = character_from_anilist(_character_edge())["voice_actors"][0]
    assert (actor["name"], actor["native_name"], actor["language"]) == (
        "Mayumi Tanaka",
        "田中真弓",
        "Japanese",
    )
    assert actor["image"] == "https://anilist.co/img/tanaka.jpg"
    assert actor["sources"] == ["https://anilist.co/staff/95"]


def test_character_from_anilist_voice_actor_without_site_url_builds_source_from_id() -> (
    None
):
    edge = _character_edge()
    edge.voice_actor_roles[0].voice_actor.site_url = None
    actor = character_from_anilist(edge)["voice_actors"][0]
    assert actor["sources"] == ["https://anilist.co/staff/95"]


def test_character_from_anilist_voice_actor_without_image_or_native_name_omits_them() -> (
    None
):
    edge = _character_edge()
    edge.voice_actor_roles[0].voice_actor.image = None
    edge.voice_actor_roles[0].voice_actor.name.native = None
    actor = character_from_anilist(edge)["voice_actors"][0]
    assert not {"image", "native_name"} & actor.keys()


def test_character_from_anilist_voice_actor_without_name_dropped() -> None:
    edge = _character_edge()
    edge.voice_actor_roles[0].voice_actor.name = None
    assert character_from_anilist(edge).get("voice_actors", []) == []


def test_character_from_anilist_role_without_voice_actor_dropped() -> None:
    edge = _character_edge()
    edge.voice_actor_roles[0].voice_actor = None
    assert character_from_anilist(edge).get("voice_actors", []) == []


def test_character_from_anilist_voice_actors_in_several_languages_all_kept() -> None:
    edge = _character_edge(
        voiceActorRoles=[
            {
                "voiceActor": {
                    "id": 95,
                    "name": {"full": "Mayumi Tanaka"},
                    "languageV2": "Japanese",
                    "siteUrl": "https://anilist.co/staff/95",
                },
            },
            {
                "voiceActor": {
                    "id": 360,
                    "name": {"full": "Sonny Strait"},
                    "languageV2": "English",
                    "siteUrl": "https://anilist.co/staff/360",
                },
            },
        ]
    )
    assert [
        (actor["name"], actor["language"])
        for actor in character_from_anilist(edge)["voice_actors"]
    ] == [("Mayumi Tanaka", "Japanese"), ("Sonny Strait", "English")]
