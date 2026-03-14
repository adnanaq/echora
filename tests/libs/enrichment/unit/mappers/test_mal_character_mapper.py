"""Unit tests for mal_mapper.py — character_from_mal value normalization."""

from enrichment.crawlers.mal_crawler.mal_models import (
    MalScrapedCharacter,
    MalVoiceActorRef,
)
from enrichment.mappers.mal_mapper import character_from_mal


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
            MalVoiceActorRef(
                person_id=70,
                name="Tanaka, Mayumi",
                language="Japanese",
                sources=["https://myanimelist.net/people/70/Mayumi_Tanaka"],
            ),
        ],
        animeography=[{"title": "One Piece", "role": "Main", "sources": ["https://myanimelist.net/anime/21"]}],
    )


def test_character_from_mal_name() -> None:
    result = character_from_mal(_make_sample_character())
    assert result["name"] == "Monkey D., Luffy"


def test_character_from_mal_voice_actors() -> None:
    result = character_from_mal(_make_sample_character())
    assert "voice_actors" in result
    va = result["voice_actors"][0]
    assert va["name"] == "Tanaka, Mayumi"
    assert va["language"] == "Japanese"
    assert va["sources"] == ["https://myanimelist.net/people/70/Mayumi_Tanaka"]


def test_character_from_mal_animeography() -> None:
    result = character_from_mal(_make_sample_character())
    assert "animeography" in result
    assert result["animeography"][0]["title"] == "One Piece"


def test_character_from_mal_roles_derived_from_animeography() -> None:
    result = character_from_mal(_make_sample_character())
    assert "roles" in result
    assert "MAIN" in result["roles"]


# TODO: attributes mapping from character_info not yet implemented
# def test_character_from_mal_character_info() -> None:
#     result = character_from_mal(_make_sample_character())
#     assert "attributes" in result
#     assert result["attributes"]["age"] == "17; 19"
#     assert result["attributes"]["devil_fruit"] == "Gomu Gomu no Mi"
