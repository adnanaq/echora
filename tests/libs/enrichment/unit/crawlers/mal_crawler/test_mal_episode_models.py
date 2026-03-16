"""Unit tests for MalScrapedEpisode and related episode models."""

import pytest
from pydantic import ValidationError

from enrichment.crawlers.mal_crawler.mal_models import (
    EpisodeCharacterRef,
    EpisodeStaffRef,
    EpisodeVARef,
    MalFetchResult,
    MalScrapedAnime,
    MalScrapedEpisode,
)


# =============================================================================
# EpisodeVARef
# =============================================================================


def test_episode_va_ref_fields() -> None:
    va = EpisodeVARef(person_id=70, name="Tanaka, Mayumi", language="Japanese")
    assert va.person_id == 70
    assert va.language == "Japanese"


# =============================================================================
# EpisodeStaffRef
# =============================================================================


def test_episode_staff_ref_fields() -> None:
    staff = EpisodeStaffRef(person_id=999, name="Takegami, Junki", role="Script")
    assert staff.person_id == 999
    assert staff.role == "Script"


# =============================================================================
# EpisodeCharacterRef
# =============================================================================


def test_episode_character_ref_minimal() -> None:
    char = EpisodeCharacterRef(mal_id=40, name="Luffy", role="Main")
    assert char.mal_id == 40
    assert char.voice_actors == []


def test_episode_character_ref_with_voice_actors() -> None:
    char = EpisodeCharacterRef(
        mal_id=40,
        name="Luffy",
        role="Main",
        voice_actors=[EpisodeVARef(person_id=70, name="Tanaka, Mayumi", language="Japanese")],
    )
    assert len(char.voice_actors) == 1
    assert char.voice_actors[0].language == "Japanese"


# =============================================================================
# MalScrapedEpisode
# =============================================================================


def test_mal_scraped_episode_minimal() -> None:
    ep = MalScrapedEpisode(
        episode_number=1,
        source="...",
        title="Episode 1",
    )
    assert ep.episode_number == 1
    assert ep.filler is False
    assert ep.recap is False
    assert ep.characters == []
    assert ep.staff == []


def test_mal_scraped_episode_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        MalScrapedEpisode(
            episode_number=1,
            source="...",
            title="Episode 1",
            unknown_field="x",  # type: ignore[call-arg]
        )


def test_mal_scraped_episode_filler_recap_flags() -> None:
    filler_ep = MalScrapedEpisode(episode_number=50, source="...", title="Showdown at High!", filler=True)
    recap_ep = MalScrapedEpisode(episode_number=279, source="...", title="Luffy's Feelings!", recap=True)
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
        source="...",
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
    anime = MalScrapedAnime(source="...", title="One Piece")
    result = MalFetchResult(anime=anime)
    assert result.characters == []
    assert result.episodes == []


def test_mal_fetch_result_with_data() -> None:
    anime = MalScrapedAnime(source="...", title="One Piece")
    ep = MalScrapedEpisode(episode_number=1, source="...", title="I'm Luffy!")
    result = MalFetchResult(anime=anime, episodes=[ep])
    assert len(result.episodes) == 1
    assert result.episodes[0].title == "I'm Luffy!"
