"""Unit tests for mal_mapper.py — value normalization and field mapping."""

import pytest

from enrichment.crawlers.mal_crawler.mal_models import (
    MalCompanyRef,
    MalExternalLink,
    MalRelatedEntry,
    MalScrapedAnime,
    MalScrapedCharacter,
    MalScrapedEpisode,
    MalThemeSong,
    EpisodeCharacterRef,
    EpisodeStaffRef,
    EpisodeVARef,
    MalVoiceActorRef,
)
from enrichment.mappers.mal_mapper import anime_from_mal, character_from_mal, episode_from_mal
from enrichment.mappers.normalization import SOURCE_MATERIAL, parse_duration, parse_theme_song


# =============================================================================
# normalization lookup tables
# =============================================================================



def test_source_material_manga() -> None:
    assert SOURCE_MATERIAL["manga"] == "MANGA"


def test_source_material_light_novel() -> None:
    assert SOURCE_MATERIAL["light novel"] == "LIGHT NOVEL"


def test_parse_duration_minutes() -> None:
    assert parse_duration("24 min.") == 1440


def test_parse_duration_hours_and_minutes() -> None:
    assert parse_duration("1 hr. 30 min.") == 5400


def test_parse_duration_none() -> None:
    assert parse_duration(None) is None


# =============================================================================
# parse_theme_song
# =============================================================================


def test_parse_theme_song_full() -> None:
    result = parse_theme_song('1: "Hands Up!" by V6 (eps 1-26)')
    assert result is not None
    assert result["title"] == "Hands Up!"
    assert result["artist"] == "V6"
    assert result["episodes"] == "1-26"


def test_parse_theme_song_no_artist() -> None:
    result = parse_theme_song('"We Are!"')
    assert result is not None
    assert result["title"] == "We Are!"
    assert result["artist"] is None


def test_parse_theme_song_none() -> None:
    assert parse_theme_song(None) is None


# =============================================================================
# anime_from_mal
# =============================================================================


def _make_sample_anime() -> MalScrapedAnime:
    return MalScrapedAnime(
        anime_id=21,
        url="https://myanimelist.net/anime/21",
        title="One Piece",
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
        studios=[MalCompanyRef(name="Toei Animation", source="https://myanimelist.net/anime/producer/18")],
        producers=[MalCompanyRef(name="Fuji TV", source="https://myanimelist.net/anime/producer/29")],
        aired_from="1999-10-20",
        duration=1440,
        opening_themes=[MalThemeSong(title="We Are!", artist="Kitadani Hiroshi")],
    related_entries=[
        MalRelatedEntry(
            relation="Side Story",
            title="One Piece Film: Gold",
            mal_id=28933,
            source="https://myanimelist.net/anime/28933",
            entry_type="Movie",
            is_anime=True,
        ),
    ],
)


def test_anime_from_mal_title() -> None:
    result = anime_from_mal(_make_sample_anime())
    assert result["title"] == "One Piece"


def test_anime_from_mal_type_normalization() -> None:
    result = anime_from_mal(_make_sample_anime())
    assert result["type"] == "TV"


def test_anime_from_mal_status_normalization() -> None:
    result = anime_from_mal(_make_sample_anime())
    assert result["status"] == "ONGOING"


def test_anime_from_mal_source_material_normalization() -> None:
    result = anime_from_mal(_make_sample_anime())
    assert result["source_material"] == "MANGA"


def test_anime_from_mal_duration_passed_through() -> None:
    result = anime_from_mal(_make_sample_anime())
    assert result["duration"] == 1440


def test_anime_from_mal_statistics_built() -> None:
    result = anime_from_mal(_make_sample_anime())
    assert "statistics" in result
    assert "mal" in result["statistics"]
    mal_stats = result["statistics"]["mal"]
    assert mal_stats["score"] == 8.7
    assert mal_stats["popularity"] == 17


def test_anime_from_mal_season_uppercased() -> None:
    result = anime_from_mal(_make_sample_anime())
    assert result["season"] == "FALL"


def test_anime_from_mal_broadcast_built() -> None:
    result = anime_from_mal(_make_sample_anime())
    assert "broadcast" in result
    broadcast = result["broadcast"]
    assert broadcast["day"] == "Sundays"
    assert broadcast["time"] == "23:15"
    assert broadcast["timezone"] == "JST"


def test_anime_from_mal_studios_mapped() -> None:
    result = anime_from_mal(_make_sample_anime())
    assert "studios" in result
    assert result["studios"][0]["name"] == "Toei Animation"
    assert "https://myanimelist.net/anime/producer/18" in result["studios"][0]["sources"]


def test_anime_from_mal_related_anime_split() -> None:
    result = anime_from_mal(_make_sample_anime())
    assert "related_anime" in result
    assert result["related_anime"]["SIDE_STORY"][0]["title"] == "One Piece Film: Gold"


def test_anime_from_mal_opening_themes_parsed() -> None:
    result = anime_from_mal(_make_sample_anime())
    assert "opening_themes" in result
    assert result["opening_themes"][0]["title"] == "We Are!"


def test_anime_from_mal_field_names_valid() -> None:
    """All output keys should be valid Anime model field names."""
    from common.models.anime import Anime
    result = anime_from_mal(_make_sample_anime())
    valid_fields = set(Anime.model_fields.keys())
    for key in result:
        assert key in valid_fields, f"Mapper output key '{key}' not in Anime model"


def test_anime_from_mal_sources() -> None:
    sample = _make_sample_anime()
    result = anime_from_mal(sample)
    assert result["sources"] == [sample.url]


# =============================================================================
# character_from_mal
# =============================================================================


def _make_sample_character() -> MalScrapedCharacter:
    return MalScrapedCharacter(
        mal_id=40,
        url="https://myanimelist.net/character/40",
        name="Monkey D., Luffy",
        name_native="モンキー・D・ルフィ",
        description="The main character of One Piece.",
        nicknames=["Straw Hat Luffy"],
        favorites=123456,
        images=["https://cdn.myanimelist.net/images/characters/9/310307.jpg"],
        character_info={
            "Age": "17; 19",
            "Birthdate": "May 5, Taurus",
            "Height": "172 cm",
            "Blood type": "F",
            "Devil fruit": "Gomu Gomu no Mi",
        },
        voice_actors=[
            MalVoiceActorRef(person_id=70, name="Tanaka, Mayumi", language="Japanese"),
        ],
        animeography=[{"title": "One Piece", "role": "Main", "sources": ["https://myanimelist.net/anime/21"]}],
    )


def test_character_from_mal_name() -> None:
    result = character_from_mal(_make_sample_character())
    assert result["name"] == "Monkey D., Luffy"



# TODO: attributes mapping from character_info not yet implemented
# def test_character_from_mal_character_info() -> None:
#     result = character_from_mal(_make_sample_character())
#     assert "attributes" in result
#     assert result["attributes"]["age"] == "17; 19"
#     assert result["attributes"]["devil_fruit"] == "Gomu Gomu no Mi"


def test_character_from_mal_voice_actors() -> None:
    result = character_from_mal(_make_sample_character())
    assert "voice_actors" in result
    assert result["voice_actors"][0]["name"] == "Tanaka, Mayumi"
    assert result["voice_actors"][0]["language"] == "Japanese"


def test_character_from_mal_animeography() -> None:
    result = character_from_mal(_make_sample_character())
    assert "animeography" in result
    assert result["animeography"][0]["title"] == "One Piece"


def test_character_from_mal_roles_derived_from_animeography() -> None:
    result = character_from_mal(_make_sample_character())
    assert "roles" in result
    assert "MAIN" in result["roles"]


# =============================================================================
# episode_from_mal
# =============================================================================


def _make_sample_episode() -> MalScrapedEpisode:
    return MalScrapedEpisode(
        episode_number=1,
        url="https://myanimelist.net/anime/21/episode/1",
        title="I'm Luffy! The Man Who Will Become the Pirate King!",
        title_japanese="俺はルフィ！",
        title_romaji="Ore wa Luffy!",
        synopsis="Luffy meets Coby.",
        aired="1999-10-20",
        duration=1440,
        characters=[
            EpisodeCharacterRef(
                mal_id=40,
                name="Monkey D., Luffy",
                role="Main",
                voice_actors=[EpisodeVARef(person_id=70, name="Tanaka, Mayumi", language="Japanese")],
            )
        ],
        staff=[
            EpisodeStaffRef(person_id=999, name="Takegami, Junki", role="Script"),
        ],
    )


def test_episode_from_mal_title() -> None:
    result = episode_from_mal(_make_sample_episode())
    assert result["title"] == "I'm Luffy! The Man Who Will Become the Pirate King!"


def test_episode_from_mal_episode_number() -> None:
    result = episode_from_mal(_make_sample_episode())
    assert result["episode_number"] == 1


def test_episode_from_mal_duration() -> None:
    result = episode_from_mal(_make_sample_episode())
    assert result["duration"] == 1440


def test_episode_from_mal_source_url() -> None:
    result = episode_from_mal(_make_sample_episode())
    assert result["sources"] == ["https://myanimelist.net/anime/21/episode/1"]


def test_episode_from_mal_characters_mapped() -> None:
    result = episode_from_mal(_make_sample_episode())
    assert "characters" in result
    chars = result["characters"]
    assert len(chars) == 1
    assert chars[0]["name"] == "Monkey D., Luffy"
    assert chars[0]["voice_actors"][0]["language"] == "Japanese"
    assert chars[0]["sources"] == ["https://myanimelist.net/character/40"]
    assert chars[0]["role"] == "MAIN"


def test_episode_from_mal_staff_mapped() -> None:
    result = episode_from_mal(_make_sample_episode())
    assert "staff" in result
    assert result["staff"][0]["name"] == "Takegami, Junki"
    assert result["staff"][0]["role"] == "Script"
    assert result["staff"][0]["sources"] == ["https://myanimelist.net/people/999"]


def test_episode_from_mal_with_anime_id() -> None:
    result = episode_from_mal(_make_sample_episode(), anime_id="uuid-123")
    assert result["anime_id"] == "uuid-123"


def test_episode_from_mal_field_names_valid() -> None:
    """All output keys should be valid Episode model field names."""
    from common.models.anime import Episode
    result = episode_from_mal(_make_sample_episode())
    valid_fields = set(Episode.model_fields.keys())
    for key in result:
        assert key in valid_fields, f"Mapper output key '{key}' not in Episode model"
