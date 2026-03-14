"""Unit tests for mal_models.py — model validation and field defaults."""

import pytest
from pydantic import ValidationError

from enrichment.crawlers.mal_crawler.mal_models import (
    CharacterRef,
    EpisodeCharacterRef,
    EpisodeStaffRef,
    EpisodeVARef,
    MalCompanyRef,
    MalExternalLink,
    MalFetchResult,
    MalRelatedEntry,
    MalScrapedAnime,
    MalScrapedCharacter,
    MalScrapedEpisode,
)


# =============================================================================
# MalScrapedAnime
# =============================================================================


def test_mal_scraped_anime_minimal() -> None:
    """Minimal required fields only."""
    anime = MalScrapedAnime(
        anime_id=21,
        url="https://myanimelist.net/anime/21",
        title="One Piece",
    )
    assert anime.anime_id == 21
    assert anime.title == "One Piece"
    assert anime.synonyms == []
    assert anime.genres == []
    assert anime.related_entries == []
    assert anime.picture_urls == []


def test_mal_scraped_anime_extra_field_rejected() -> None:
    """extra='forbid' rejects unknown fields."""
    with pytest.raises(ValidationError):
        MalScrapedAnime(
            anime_id=21,
            url="...",
            title="One Piece",
            unknown_field="value",  # type: ignore[call-arg]
        )


def test_mal_scraped_anime_full() -> None:
    anime = MalScrapedAnime(
        anime_id=21,
        url="https://myanimelist.net/anime/21",
        title="One Piece",
        title_english="One Piece",
        title_japanese="ワンピース",
        type="TV",
        status="Currently Airing",
        score=8.7,
        scored_by=2644378,
        rank=54,
        popularity=17,
        episode_count=1122,
        year=1999,
        season="fall",
        genres=["Action", "Adventure"],
        studios=[MalCompanyRef(name="Toei Animation", source="...")],
    )
    assert anime.score == 8.7
    assert anime.rank == 54
    assert len(anime.studios) == 1
    assert anime.studios[0].name == "Toei Animation"


def test_mal_scraped_anime_related_entries() -> None:
    entry = MalRelatedEntry(
        relation="Side Story",
        title="One Piece Film: Gold",
        mal_id=28933,
        source="https://myanimelist.net/anime/28933",
        entry_type="Movie",
        is_anime=True,
    )
    anime = MalScrapedAnime(
        anime_id=21, url="...", title="One Piece", related_entries=[entry]
    )
    assert len(anime.related_entries) == 1
    assert anime.related_entries[0].relation == "Side Story"
    assert anime.related_entries[0].is_anime is True


# =============================================================================
# CharacterRef
# =============================================================================


def test_character_ref_defaults() -> None:
    ref = CharacterRef(char_id=40, name="Luffy", role="Main")
    assert ref.favorites == 0


# =============================================================================
# MalScrapedCharacter
# =============================================================================


def test_mal_scraped_character_minimal() -> None:
    char = MalScrapedCharacter(
        mal_id=40,
        url="https://myanimelist.net/character/40",
        name="Monkey D., Luffy",
    )
    assert char.name == "Monkey D., Luffy"
    assert char.character_info == {}
    assert char.animeography == []
    assert char.mangaography == []


def test_mal_scraped_character_bio_data() -> None:
    char = MalScrapedCharacter(
        mal_id=40,
        url="...",
        name="Luffy",
        character_info={
            "Age": "17; 19",
            "Height": "172 cm",
            "Devil fruit": "Gomu Gomu no Mi",
            "Bounty": "3,000,000,000",
        },
    )
    assert char.character_info["Age"] == "17; 19"
    assert char.character_info["Devil fruit"] == "Gomu Gomu no Mi"


def test_mal_scraped_character_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        MalScrapedCharacter(
            mal_id=40,
            url="...",
            name="Luffy",
            unexpected_field="x",  # type: ignore[call-arg]
        )


# =============================================================================
# MalScrapedEpisode
# =============================================================================


def test_mal_scraped_episode_minimal() -> None:
    ep = MalScrapedEpisode(
        episode_number=1,
        url="...",
        title="Episode 1",
    )
    assert ep.episode_number == 1
    assert ep.filler is False
    assert ep.recap is False
    assert ep.characters == []
    assert ep.staff == []


def test_mal_scraped_episode_filler_recap_flags() -> None:
    filler_ep = MalScrapedEpisode(episode_number=50, url="...", title="Showdown at High!", filler=True)
    recap_ep = MalScrapedEpisode(episode_number=279, url="...", title="Luffy's Feelings!", recap=True)
    assert filler_ep.filler is True
    assert filler_ep.recap is False
    assert recap_ep.recap is True
    assert recap_ep.filler is False


def test_mal_scraped_episode_with_community_data() -> None:
    char = EpisodeCharacterRef(
        mal_id=40,
        name="Luffy",
        role="Main",
        voice_actors=[EpisodeVARef(person_id=70, name="Tanaka, Mayumi", language="Japanese")],
    )
    staff = EpisodeStaffRef(person_id=999, name="Takegami, Junki", role="Script")
    ep = MalScrapedEpisode(
        episode_number=1,
        url="...",
        title="Episode 1",
        characters=[char],
        staff=[staff],
    )
    assert len(ep.characters) == 1
    assert ep.characters[0].role == "Main"
    assert len(ep.staff) == 1
    assert ep.staff[0].role == "Script"


# =============================================================================
# MalFetchResult
# =============================================================================


def test_mal_fetch_result_defaults() -> None:
    anime = MalScrapedAnime(anime_id=21, url="...", title="One Piece")
    result = MalFetchResult(anime=anime)
    assert result.characters == []
    assert result.episodes == []
