"""Source-faithful Pydantic models for anime-planet.com character scraped data.

Field names mirror what anime-planet.com actually sends — no renaming to
canonical names here.  Each model converts the values it holds, such as a voice
actor's flag into a language code, and the mapper (animeplanet_mapper.py) copies
them into the canonical Character model.
"""

import logging

from common.models.language_codes import UNDETERMINED_LANGUAGE
from enrichment.sources.base.field_types import (
    NumberFromText,
    StrippedText,
    StrippedTextList,
)
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# Anime-Planet labels each voice actor with a flag named by a country code ("jp",
# "us"), which is no language code. Each known flag becomes the BCP 47 code of
# that country's dub.
_FLAG_LANGUAGES: dict[str, str] = {
    "jp": "ja",
    "us": "en",
    "es": "es",
    "fr": "fr",
    "de": "de",
    "ko": "ko",
}


def resolve_flag_language(flag: str) -> str:
    """Resolve the language code of a dub from its Anime-Planet flag.

    A flag missing from the known flags becomes ``und`` with a warning, so its
    voice actors are kept and the warning names the flag to add.

    Args:
        flag: The country code of the flag, in lower case, such as ``jp``.

    Returns:
        The BCP 47 language code of the dub, or ``und`` for an unknown flag.
    """
    if language := _FLAG_LANGUAGES.get(flag):
        return language
    logger.warning(
        f"Anime-Planet flag {flag!r} names no known language; using {UNDETERMINED_LANGUAGE}"
    )
    return UNDETERMINED_LANGUAGE


class AnimePlanetVoiceActor(BaseModel):
    """Voice actor entry linked to a specific anime role."""

    name: str
    url: str  # relative AP people path: /people/mayumi-tanaka


class AnimePlanetCharacterAnimeRole(BaseModel):
    """An anime title in which the character appears, with role and voice actors."""

    title: str
    url: str  # relative: /anime/one-piece
    role: str | None = None  # "Main" / "Secondary" / "Minor"
    # Keyed by language code ("ja"); the validator converts the flags the page shows ("jp")
    voice_actors: dict[str, list[AnimePlanetVoiceActor]] = Field(default_factory=dict)

    @field_validator("voice_actors", mode="before")
    @classmethod
    def group_voice_actors_by_language(cls, actors_by_flag: object) -> object:
        """Group the voice actors under the language code of their flag.

        Two flags may resolve to one language, such as two unknown flags both
        becoming ``und``, so their voice actors are joined rather than replaced.

        Args:
            actors_by_flag: The voice actors keyed by flag, as the crawler read them.

        Returns:
            The voice actors keyed by language code, or the value unchanged when it
            is not a mapping, for Pydantic to reject.
        """
        if not isinstance(actors_by_flag, dict):
            return actors_by_flag
        actors_by_language: dict[str, list[object]] = {}
        for flag, actors in actors_by_flag.items():
            language = resolve_flag_language(str(flag).lower())
            actors_by_language.setdefault(language, []).extend(actors)
        return actors_by_language


class AnimePlanetCharacterMangaRole(BaseModel):
    """A manga title in which the character appears."""

    title: str
    url: str  # relative: /manga/one-piece
    role: str | None = None  # "Main" / "Secondary" / "Minor"


class AnimePlanetCharacter(BaseModel):
    """Full character data scraped from the character detail page."""

    # Identity
    name: str
    slug: str
    url: str  # full: https://www.anime-planet.com/characters/{slug}

    # From detail page
    image: StrippedText = None
    gender: str | None = None
    hair_color: str | None = None
    loved_rank: NumberFromText = None  # "#1,081" → 1081
    hated_rank: NumberFromText = None
    loved_count: NumberFromText = None  # "36,485 users" → 36485
    description: str | None = None
    tags: StrippedTextList = Field(default_factory=list)
    alt_names: StrippedTextList = Field(default_factory=list)  # from h2.aka
    attributes: dict[str, str] = Field(default_factory=dict)
    # All EntryMetadata items not explicitly modelled (eye_color, birthday, age, etc.)

    anime_roles: list[AnimePlanetCharacterAnimeRole] = Field(default_factory=list)
    manga_roles: list[AnimePlanetCharacterMangaRole] = Field(default_factory=list)
