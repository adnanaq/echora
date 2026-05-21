"""Unit tests for MalCharacter and related character models."""

import pytest
from enrichment.sources.mal.mal_models import (
    MalCharacter,
    MalVoiceActorRef,
)
from pydantic import ValidationError

# =============================================================================
# MalVoiceActorRef
# =============================================================================


def test_mal_voice_actor_ref_minimal() -> None:
    va = MalVoiceActorRef(person_id=70, name="Tanaka, Mayumi", language="Japanese")
    assert va.person_id == 70
    assert va.language == "Japanese"
    assert va.image_url is None
    assert va.sources == []


def test_mal_voice_actor_ref_with_image() -> None:
    va = MalVoiceActorRef(
        person_id=70,
        name="Tanaka, Mayumi",
        language="Japanese",
        image_url="https://cdn.myanimelist.net/images/voiceactors/2/40132.jpg",
    )
    assert va.image_url is not None


def test_mal_voice_actor_ref_with_sources() -> None:
    va = MalVoiceActorRef(
        person_id=70,
        name="Tanaka, Mayumi",
        language="Japanese",
        sources=["https://myanimelist.net/people/70/Mayumi_Tanaka"],
    )
    assert va.sources == ["https://myanimelist.net/people/70/Mayumi_Tanaka"]


# =============================================================================
# MalCharacter
# =============================================================================


def test_mal_scraped_character_minimal() -> None:
    char = MalCharacter(
        source="https://myanimelist.net/character/40",
        name="Monkey D., Luffy",
    )
    assert char.name == "Monkey D., Luffy"
    assert char.character_info == {}
    assert char.animeography == []
    assert char.mangaography == []


def test_mal_scraped_character_bio_data() -> None:
    char = MalCharacter(
        source="...",
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
        MalCharacter(
            source="...",
            name="Luffy",
            unexpected_field="x",  # type: ignore[call-arg]
        )


def test_mal_scraped_character_nicknames() -> None:
    char = MalCharacter(
        source="...",
        name="Monkey D., Luffy",
        nicknames=["Straw Hat Luffy", "Straw Hat"],
    )
    assert len(char.nicknames) == 2
    assert "Straw Hat Luffy" in char.nicknames


def test_mal_scraped_character_voice_actors() -> None:
    char = MalCharacter(
        source="...",
        name="Luffy",
        voice_actors=[
            MalVoiceActorRef(person_id=70, name="Tanaka, Mayumi", language="Japanese"),
        ],
    )
    assert len(char.voice_actors) == 1
    assert char.voice_actors[0].language == "Japanese"
