"""Source-faithful Pydantic models for anime-planet.com character scraped data.

Field names mirror what anime-planet.com actually sends — no renaming to
canonical names here.  The mapper (animeplanet_mapper.py) handles all
translation to the canonical Character model.
"""

from pydantic import BaseModel, Field


class AnimePlanetVoiceActor(BaseModel):
    """Voice actor entry linked to a specific anime role."""

    name: str
    url: str  # relative AP people path: /people/mayumi-tanaka


class AnimePlanetCharacterAnimeRole(BaseModel):
    """An anime title in which the character appears, with role and voice actors."""

    title: str
    url: str  # relative: /anime/one-piece
    role: str | None = None  # "Main" / "Secondary" / "Minor"
    voice_actors: dict[str, list[AnimePlanetVoiceActor]] = Field(default_factory=dict)
    # keys: "jp", "us", "es", "fr", "de", "ko"


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
    image: str | None = None
    gender: str | None = None
    hair_color: str | None = None
    loved_rank: int | None = None
    hated_rank: int | None = None
    loved_count: int | None = None  # absolute count, e.g. 36485
    description: str | None = None
    tags: list[str] = Field(default_factory=list)
    alt_names: list[str] = Field(default_factory=list)  # from h2.aka
    attributes: dict[str, str] = Field(default_factory=dict)
    # All EntryMetadata items not explicitly modelled (eye_color, birthday, age, etc.)

    anime_roles: list[AnimePlanetCharacterAnimeRole] = Field(default_factory=list)
    manga_roles: list[AnimePlanetCharacterMangaRole] = Field(default_factory=list)
