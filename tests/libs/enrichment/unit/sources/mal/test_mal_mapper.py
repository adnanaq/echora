import pytest
from common.models.anime import Anime, Episode
from enrichment.sources.mal.mal_mapper import (
    anime_from_mal,
    character_from_mal,
    episode_from_mal,
)
from enrichment.sources.mal.mal_models import (
    EpisodeCharacterRef,
    EpisodeStaffRef,
    EpisodeVARef,
    MalAnime,
    MalCharacter,
    MalCompanyRef,
    MalEpisode,
    MalEpisodeRange,
    MalExternalLink,
    MalOgraphyEntry,
    MalRelatedEntry,
    MalThemeSong,
    MalTrailer,
    MalVoiceActorRef,
)

ONE_PIECE_URL = "https://myanimelist.net/anime/21"
LUFFY_URL = "https://myanimelist.net/character/40"
EPISODE_URL = "https://myanimelist.net/anime/21/One_Piece/episode/1"


def _anime(**fields) -> MalAnime:
    return MalAnime(**{"source": ONE_PIECE_URL, "title": "One Piece", **fields})


def _one_piece() -> MalAnime:
    return _anime(
        title_english="One Piece",
        title_japanese="ワンピース",
        type="TV",
        status="Currently Airing",
        source_material="Manga",
        score=8.7,
        scored_by=2644378,
        rank=54,
        popularity=17,
        members=2644378,
        episode_count=1122,
        season="fall",
        year=1999,
        synopsis="The story of Monkey D. Luffy...",
        genres=["Action", "Adventure"],
        broadcast_day="Sundays",
        broadcast_time="23:15",
        broadcast_timezone="JST",
        studios=[
            MalCompanyRef(
                name="Toei Animation",
                source="https://myanimelist.net/anime/producer/18",
            )
        ],
        producers=[
            MalCompanyRef(
                name="Fuji TV", source="https://myanimelist.net/anime/producer/29"
            )
        ],
        aired_from="1999-10-20",
        duration=1440,
        opening_themes=[MalThemeSong(title="We Are!", artist="Kitadani Hiroshi")],
        related_entries=[
            MalRelatedEntry(
                relation="Side Story",
                title="One Piece Film: Gold",
                source="https://myanimelist.net/anime/28933",
                entry_type="Movie",
                is_anime=True,
            ),
        ],
    )


def _character(**fields) -> MalCharacter:
    return MalCharacter(**{"source": LUFFY_URL, "name": "Monkey D., Luffy", **fields})


def _episode(**fields) -> MalEpisode:
    return MalEpisode(
        **{
            "episode_number": 1,
            "source": EPISODE_URL,
            "title": "I'm Luffy! The Man Who Will Become the Pirate King!",
            **fields,
        }
    )


def test_anime_from_mal_one_piece_page_maps_titles_and_scalars() -> None:
    result = anime_from_mal(_one_piece())
    assert (result["title"], result["title_english"], result["title_japanese"]) == (
        "One Piece",
        "One Piece",
        "ワンピース",
    )
    assert (result["type"], result["status"], result["source_material"]) == (
        "TV",
        "ONGOING",
        "MANGA",
    )
    assert (result["duration"], result["episode_count"], result["year"]) == (
        1440,
        1122,
        1999,
    )
    assert result["synopsis"] == "The story of Monkey D. Luffy..."


def test_anime_from_mal_lowercase_season_gives_season() -> None:
    assert anime_from_mal(_one_piece())["season"] == "FALL"


def test_anime_from_mal_without_season_or_source_material_omits_them() -> None:
    assert not {"season", "source_material"} & anime_from_mal(_anime()).keys()


def test_anime_from_mal_without_episode_count_gives_zero_episodes() -> None:
    assert anime_from_mal(_anime())["episode_count"] == 0


def test_anime_from_mal_rating_text_gives_rating() -> None:
    anime = _anime(rating="PG-13 - Teens 13 or older")
    assert anime_from_mal(anime)["rating"] == "PG-13 - Teens 13 or older"


def test_anime_from_mal_without_rating_gives_unknown_rating() -> None:
    assert anime_from_mal(_anime())["rating"] == "UNKNOWN"


def test_anime_from_mal_background_kept() -> None:
    assert anime_from_mal(_anime(background="Began in 1997."))["background"] == (
        "Began in 1997."
    )


def test_anime_from_mal_placeholder_background_omitted() -> None:
    anime = _anime(background="No background information has been added to this title.")
    assert "background" not in anime_from_mal(anime)


def test_anime_from_mal_statistics_kept_as_stated() -> None:
    statistics = anime_from_mal(_one_piece())["statistics"]["mal"]
    assert statistics == {
        "score": 8.7,
        "scored_by": 2644378,
        "rank": 54,
        "members": 2644378,
        "popularity": 17,
    }


def test_anime_from_mal_without_statistics_gives_no_mal_statistics() -> None:
    assert not anime_from_mal(_anime()).get("statistics")


def test_anime_from_mal_broadcast_parts_give_broadcast() -> None:
    assert anime_from_mal(_one_piece())["broadcast"] == {
        "day": "Sundays",
        "time": "23:15",
        "timezone": "JST",
    }


def test_anime_from_mal_without_broadcast_omits_broadcast() -> None:
    assert "broadcast" not in anime_from_mal(_anime())


def test_anime_from_mal_aired_dates_normalized_to_utc() -> None:
    result = anime_from_mal(_anime(aired_from="1999-10-20", aired_to="2000-03-26"))
    assert result["aired_dates"] == {
        "aired_from": "1999-10-19T15:00:00Z",
        "aired_to": "2000-03-25T15:00:00Z",
    }


def test_anime_from_mal_without_dates_omits_aired_dates() -> None:
    assert "aired_dates" not in anime_from_mal(_anime())


def test_anime_from_mal_month_kept_when_stated() -> None:
    assert anime_from_mal(_anime(month="October"))["month"] == "October"


def test_anime_from_mal_without_month_omits_month() -> None:
    assert "month" not in anime_from_mal(_one_piece())


def test_anime_from_mal_studios_producers_and_licensors_become_companies() -> None:
    anime = _one_piece().model_copy(
        update={
            "licensors": [
                MalCompanyRef(
                    name="4Kids", source="https://myanimelist.net/anime/producer/1"
                )
            ]
        }
    )
    assert anime_from_mal(anime)["companies"] == [
        {
            "name": "Toei Animation",
            "roles": ["STUDIO"],
            "sources": ["https://myanimelist.net/anime/producer/18"],
        },
        {
            "name": "Fuji TV",
            "roles": ["PRODUCER"],
            "sources": ["https://myanimelist.net/anime/producer/29"],
        },
        {
            "name": "4Kids",
            "roles": ["LICENSOR"],
            "sources": ["https://myanimelist.net/anime/producer/1"],
        },
    ]


def test_anime_from_mal_genres_themes_demographics_and_synonyms_mapped() -> None:
    result = anime_from_mal(
        _anime(
            genres=["Action"],
            themes=["Pirates"],
            demographics=["Shounen"],
            synonyms=["OP"],
        )
    )
    assert (result["genres"], result["demographics"], result["synonyms"]) == (
        ["Action"],
        ["Shounen"],
        ["OP"],
    )
    assert [theme["name"] for theme in result["themes"]] == ["Pirates"]


def test_anime_from_mal_page_url_becomes_only_source() -> None:
    assert anime_from_mal(_one_piece())["sources"] == [ONE_PIECE_URL]


def test_anime_from_mal_empty_page_url_gives_no_sources() -> None:
    assert not anime_from_mal(_anime(source="")).get("sources")


def test_anime_from_mal_gallery_pictures_become_covers() -> None:
    pictures = ["https://cdn.myanimelist.net/images/anime/1/1l.jpg"]
    assert anime_from_mal(_anime(picture_urls=pictures))["images"]["covers"] == pictures


def test_anime_from_mal_anime_relation_becomes_related_anime() -> None:
    entry = anime_from_mal(_one_piece())["related_anime"]["SIDE_STORY"][0]
    assert (entry["title"], entry["type"], entry["sources"]) == (
        "One Piece Film: Gold",
        "MOVIE",
        ["https://myanimelist.net/anime/28933"],
    )


def test_anime_from_mal_manga_relations_become_related_source_material() -> None:
    result = anime_from_mal(
        _anime(
            related_entries=[
                MalRelatedEntry(
                    relation="Adaptation",
                    title="One Piece",
                    source="https://myanimelist.net/manga/13",
                    entry_type="Manga",
                    is_anime=False,
                ),
                MalRelatedEntry(
                    relation="Adaptation",
                    title="One Piece: Novel",
                    source="https://myanimelist.net/manga/14",
                    is_anime=False,
                ),
                MalRelatedEntry(
                    relation="Sequel",
                    title="Sequel",
                    source="https://myanimelist.net/anime/22",
                    is_anime=True,
                ),
                MalRelatedEntry(
                    relation="Sequel",
                    title="Sequel 2",
                    source="https://myanimelist.net/anime/23",
                    is_anime=True,
                ),
            ]
        )
    )
    adaptations = result["related_source_material"]["ADAPTATION"]
    assert [(entry["title"], entry["type"]) for entry in adaptations] == [
        ("One Piece", "MANGA"),
        ("One Piece: Novel", "UNKNOWN"),
    ]
    assert [entry["title"] for entry in result["related_anime"]["SEQUEL"]] == [
        "Sequel",
        "Sequel 2",
    ]


def test_anime_from_mal_theme_songs_keep_artist_and_episode_ranges() -> None:
    result = anime_from_mal(
        _anime(
            opening_themes=[
                MalThemeSong(
                    title="We Are!",
                    artist="Kitadani Hiroshi",
                    episodes=[MalEpisodeRange(start=1, end=47)],
                )
            ],
            ending_themes=[MalThemeSong(title="memories")],
        )
    )
    assert result["opening_themes"][0]["title"] == "We Are!"
    assert result["opening_themes"][0]["artist"] == "Kitadani Hiroshi"
    assert result["opening_themes"][0]["episodes"] == [{"start": 1, "end": 47}]
    assert [song["title"] for song in result["ending_themes"]] == ["memories"]


def test_anime_from_mal_external_links_with_host_become_external_sources() -> None:
    result = anime_from_mal(
        _anime(
            external_sources=[
                MalExternalLink(name="Official Site", source="https://one-piece.com"),
                MalExternalLink(name="Broken", source="not a link"),
            ]
        )
    )
    assert [
        (link["platform"], link["source"], link["label"])
        for link in result["external_sources"]
    ] == [("official_site", "https://one-piece.com", "Official Site")]


def test_anime_from_mal_streaming_links_become_streaming_sources() -> None:
    result = anime_from_mal(
        _anime(
            streaming=[
                MalExternalLink(
                    name="Crunchyroll", source="https://crunchyroll.com/one-piece"
                )
            ]
        )
    )
    assert [
        (entry["platform"], entry["source"]) for entry in result["streaming_sources"]
    ] == [("Crunchyroll", "https://crunchyroll.com/one-piece")]


def test_anime_from_mal_trailer_kept_with_title_and_thumbnail() -> None:
    trailer = MalTrailer(
        source="https://www.youtube.com/watch?v=gAX3Zj-JGE0",
        title="PV 1",
        thumbnail="https://img.youtube.com/vi/gAX3Zj-JGE0/maxresdefault.jpg",
    )
    assert anime_from_mal(_anime(trailer=trailer))["trailers"] == [
        {
            "source": "https://www.youtube.com/watch?v=gAX3Zj-JGE0",
            "title": "PV 1",
            "thumbnail": "https://img.youtube.com/vi/gAX3Zj-JGE0/maxresdefault.jpg",
        }
    ]


def test_anime_from_mal_without_trailer_gives_no_trailers() -> None:
    assert anime_from_mal(_anime(trailer=None))["trailers"] == []


def test_anime_from_mal_output_keys_are_anime_fields() -> None:
    assert set(anime_from_mal(_one_piece())) <= set(Anime.model_fields)


def test_character_from_mal_profile_fields_mapped() -> None:
    result = character_from_mal(
        _character(
            name_native="モンキー・D・ルフィ",
            description="The main character of One Piece.",
            nicknames=["Straw Hat Luffy"],
            favorites=123456,
            images=["https://cdn.myanimelist.net/images/characters/9/310307.jpg"],
        )
    )
    assert (result["name"], result["name_native"], result["favorites"]) == (
        "Monkey D., Luffy",
        "モンキー・D・ルフィ",
        123456,
    )
    assert (result["description"], result["nicknames"]) == (
        "The main character of One Piece.",
        ["Straw Hat Luffy"],
    )
    assert result["images"] == [
        "https://cdn.myanimelist.net/images/characters/9/310307.jpg"
    ]
    assert result["sources"] == [LUFFY_URL]


def test_character_from_mal_empty_profile_fields_left_at_defaults() -> None:
    result = character_from_mal(_character(source="", description=""))
    assert result["name"] == "Monkey D., Luffy"
    assert "description" not in result
    assert (result["nicknames"], result["images"], result["sources"]) == ([], [], [])
    assert (result["attributes"], result["voice_actors"], result["roles"]) == (
        {},
        [],
        [],
    )


def test_character_from_mal_character_info_becomes_text_attributes() -> None:
    result = character_from_mal(
        _character(character_info={"age": "17; 19", "height_cm": 172})
    )
    assert result["attributes"] == {"age": "17; 19", "height_cm": "172"}


def test_character_from_mal_spoilers_kept_when_stated() -> None:
    spoilers = {"devil_fruit": "Hito Hito no Mi"}
    assert character_from_mal(_character(spoilers=spoilers))["spoilers"] == spoilers


def test_character_from_mal_without_spoilers_gives_empty_spoilers() -> None:
    assert character_from_mal(_character()).get("spoilers", {}) == {}


def test_character_from_mal_voice_actors_mapped_with_language_and_links() -> None:
    result = character_from_mal(
        _character(
            voice_actors=[
                MalVoiceActorRef(
                    person_id=70,
                    name="Tanaka, Mayumi",
                    language="Japanese",
                    sources=["https://myanimelist.net/people/70/Mayumi_Tanaka"],
                )
            ]
        )
    )
    actor = result["voice_actors"][0]
    assert (actor["name"], actor["language"], actor["sources"]) == (
        "Tanaka, Mayumi",
        "Japanese",
        ["https://myanimelist.net/people/70/Mayumi_Tanaka"],
    )


def test_character_from_mal_animeography_gives_appearances_and_unique_roles() -> None:
    result = character_from_mal(
        _character(
            animeography=[
                MalOgraphyEntry(
                    title="One Piece", role="Main", sources=[ONE_PIECE_URL]
                ),
                MalOgraphyEntry(title="One Piece Film: Red", role="Main"),
                MalOgraphyEntry(title="Cameo"),
            ]
        )
    )
    assert [(entry["title"], entry["role"]) for entry in result["animeography"]] == [
        ("One Piece", "MAIN"),
        ("One Piece Film: Red", "MAIN"),
        ("Cameo", "UNKNOWN"),
    ]
    assert result["animeography"][0]["sources"] == [ONE_PIECE_URL]
    assert result["roles"] == ["MAIN"]


def test_character_from_mal_animeography_without_roles_leaves_roles_empty() -> None:
    result = character_from_mal(
        _character(animeography=[MalOgraphyEntry(title="Cameo")])
    )
    assert result["roles"] == []


def test_character_from_mal_mangaography_gives_manga_appearances() -> None:
    result = character_from_mal(
        _character(
            mangaography=[
                MalOgraphyEntry(
                    title="One Piece",
                    role="Main",
                    sources=["https://myanimelist.net/manga/13"],
                )
            ]
        )
    )
    assert result["mangaography"] == [
        {
            "title": "One Piece",
            "role": "MAIN",
            "sources": ["https://myanimelist.net/manga/13"],
        }
    ]


def test_episode_from_mal_scalars_mapped() -> None:
    result = episode_from_mal(
        _episode(
            title_japanese="俺はルフィ！",
            title_romaji="Ore wa Luffy!",
            synopsis="Luffy meets Coby.",
            duration=1440,
            filler=True,
            recap=True,
        )
    )
    assert (result["episode_number"], result["duration"], result["synopsis"]) == (
        1,
        1440,
        "Luffy meets Coby.",
    )
    assert result["title"] == "I'm Luffy! The Man Who Will Become the Pirate King!"
    assert (result["title_japanese"], result["title_romaji"]) == (
        "俺はルフィ！",
        "Ore wa Luffy!",
    )
    assert (result["filler"], result["recap"]) == (True, True)
    assert result["sources"] == [EPISODE_URL]


def test_episode_from_mal_aired_date_normalized_to_utc() -> None:
    assert episode_from_mal(_episode(aired="1999-10-20"))["aired"] == (
        "1999-10-19T15:00:00Z"
    )


@pytest.mark.parametrize("aired", [None, "Not available"])
def test_episode_from_mal_missing_or_unreadable_air_date_omits_aired(
    aired: str | None,
) -> None:
    assert "aired" not in episode_from_mal(_episode(aired=aired))


def test_episode_from_mal_characters_mapped_with_voice_actors_and_links() -> None:
    result = episode_from_mal(
        _episode(
            characters=[
                EpisodeCharacterRef(
                    mal_id=40,
                    name="Monkey D., Luffy",
                    role="Main",
                    voice_actors=[
                        EpisodeVARef(
                            person_id=70, name="Tanaka, Mayumi", language="Japanese"
                        )
                    ],
                )
            ]
        )
    )
    character = result["characters"][0]
    assert (character["name"], character["role"], character["sources"]) == (
        "Monkey D., Luffy",
        "MAIN",
        ["https://myanimelist.net/character/40"],
    )
    actor = character["voice_actors"][0]
    assert (actor["name"], actor["language"], actor["sources"]) == (
        "Tanaka, Mayumi",
        "Japanese",
        ["https://myanimelist.net/people/70"],
    )


def test_episode_from_mal_staff_mapped_with_links() -> None:
    result = episode_from_mal(
        _episode(
            staff=[
                EpisodeStaffRef(person_id=999, name="Takegami, Junki", role="Script")
            ]
        )
    )
    assert result["staff"] == [
        {
            "name": "Takegami, Junki",
            "role": "Script",
            "sources": ["https://myanimelist.net/people/999"],
        }
    ]


def test_episode_from_mal_anime_id_added_when_given() -> None:
    assert episode_from_mal(_episode(), anime_id="uuid-123")["anime_id"] == "uuid-123"


def test_episode_from_mal_output_keys_are_episode_fields() -> None:
    assert set(episode_from_mal(_episode())) <= set(Episode.model_fields)
