"""Unit tests for mal_mapper.py — episode_from_mal value normalization."""

from enrichment.sources.mal.mal_models import (
    EpisodeCharacterRef,
    EpisodeStaffRef,
    EpisodeVARef,
    MalEpisode,
)
from enrichment.sources.mal.mal_mapper import episode_from_mal


def _make_sample_episode() -> MalEpisode:
    return MalEpisode(
        episode_number=1,
        source="https://myanimelist.net/anime/21/One_Piece/episode/1",
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
                voice_actors=[
                    EpisodeVARef(
                        person_id=70, name="Tanaka, Mayumi", language="Japanese"
                    )
                ],
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
    assert result["sources"] == ["https://myanimelist.net/anime/21/One_Piece/episode/1"]


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
