import pytest
from enrichment.sources.anime_planet.anime_planet_character_models import (
    AnimePlanetCharacter,
    AnimePlanetCharacterAnimeRole,
    AnimePlanetVoiceActor,
)
from pydantic import ValidationError

TANAKA = {"name": "Mayumi TANAKA", "url": "/people/mayumi-tanaka"}
SANDS = {"name": "Tara SANDS", "url": "/people/tara-sands"}
RODRIGUES = {"name": "Andre RODRIGUES", "url": "/people/andre-rodrigues"}
WARD = {"name": "Luca WARD", "url": "/people/luca-ward"}


def test_anime_planet_character_anime_role_known_flags_become_language_codes() -> None:
    role = AnimePlanetCharacterAnimeRole(
        title="One Piece",
        url="/anime/one-piece",
        voice_actors={
            "jp": [TANAKA],
            "us": [SANDS],
            "fr": [],
            "es": [],
            "de": [],
            "ko": [],
        },
    )

    assert role.voice_actors == {
        "ja": [AnimePlanetVoiceActor(**TANAKA)],
        "en": [AnimePlanetVoiceActor(**SANDS)],
        "fr": [],
        "es": [],
        "de": [],
        "ko": [],
    }


def test_anime_planet_character_anime_role_unknown_flags_become_undetermined_keeping_every_actor(
    caplog: pytest.LogCaptureFixture,
) -> None:
    role = AnimePlanetCharacterAnimeRole(
        title="One Piece",
        url="/anime/one-piece",
        voice_actors={"jp": [TANAKA], "br": [RODRIGUES], "it": [WARD]},
    )

    assert role.voice_actors == {
        "ja": [AnimePlanetVoiceActor(**TANAKA)],
        "und": [AnimePlanetVoiceActor(**RODRIGUES), AnimePlanetVoiceActor(**WARD)],
    }
    assert "Anime-Planet flag 'br' names no known language; using und" in caplog.text
    assert "Anime-Planet flag 'it' names no known language; using und" in caplog.text


def test_anime_planet_character_anime_role_without_voice_actors_returns_empty_dict() -> (
    None
):
    role = AnimePlanetCharacterAnimeRole(title="One Piece", url="/anime/one-piece")

    assert role.voice_actors == {}


def test_anime_planet_character_anime_role_voice_actors_list_raises_validation_error() -> (
    None
):
    with pytest.raises(ValidationError, match="voice_actors"):
        AnimePlanetCharacterAnimeRole(
            title="One Piece", url="/anime/one-piece", voice_actors=[TANAKA]
        )


def test_anime_planet_character_tags_and_alt_names_strip_and_drop_empty_texts() -> None:
    character = AnimePlanetCharacter(
        name="Monkey D. Luffy",
        slug="monkey-d-luffy",
        url="https://www.anime-planet.com/characters/monkey-d-luffy",
        tags=[" Pirates ", ""],
        alt_names=["Straw Hat ", " "],
    )
    assert (character.tags, character.alt_names) == (["Pirates"], ["Straw Hat"])
