from __future__ import annotations

"""
Type models for MyAnimeList-derived data (e.g. from Jikan v4).

These are permissive: payloads are large and may evolve.
We model only the fields we use while allowing extra keys at runtime.
All fields are based on the Jikan v4 OpenAPI specification and verified against real samples.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, field_validator

from common.utils.datetime_utils import normalize_to_utc


# =============================================================================
# Common building blocks
# =============================================================================


class MalImageVariant(BaseModel):
    model_config = ConfigDict(extra="allow")

    image_url: str | None = None
    small_image_url: str | None = None
    large_image_url: str | None = None


class MalImages(BaseModel):
    model_config = ConfigDict(extra="allow")

    jpg: MalImageVariant | None = None
    webp: MalImageVariant | None = None


class MalUrl(BaseModel):
    model_config = ConfigDict(extra="allow")

    mal_id: int
    type: str | None = None
    name: str | None = None
    url: str | None = None


class MalTitle(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    title: str


class MalRelation(BaseModel):
    model_config = ConfigDict(extra="allow")

    relation: str
    entry: list[MalUrl] = []


class MalExternalLink(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    url: str


class MalAiredDates(BaseModel):
    model_config = ConfigDict(extra="allow")

    from_date: datetime | None = None
    to: datetime | None = None
    string: str | None = None

    @field_validator("from_date", "to", mode="before")
    @classmethod
    def normalize_dates(cls, v: str | datetime | None) -> datetime | None:
        return normalize_to_utc(v)


# =============================================================================
# /v4/anime/{id}/full  -> data
# =============================================================================


class MalAnimeFull(BaseModel):
    model_config = ConfigDict(extra="allow")

    mal_id: int
    url: str | None = None
    images: MalImages | None = None

    title: str
    title_english: str | None = None
    title_japanese: str | None = None
    title_synonyms: list[str] = []
    titles: list[MalTitle] = []

    type: str | None = None
    source: str | None = None
    status: str | None = None
    rating: str | None = None

    episodes: int | None = None
    synopsis: str | None = None
    background: str | None = None

    year: int | None = None
    season: str | None = None

    aired: MalAiredDates | None = None

    genres: list[MalUrl] = []
    explicit_genres: list[MalUrl] = []
    themes: list[MalUrl] = []
    demographics: list[MalUrl] = []

    producers: list[MalUrl] = []
    licensors: list[MalUrl] = []
    studios: list[MalUrl] = []

    relations: list[MalRelation] = []
    external: list[MalExternalLink] = []
    streaming: list[MalExternalLink] = []


# =============================================================================
# /v4/anime/{id}/episodes/{episode} -> data
# =============================================================================


class MalAnimeEpisode(BaseModel):
    model_config = ConfigDict(extra="allow")

    mal_id: int
    url: str | None = None
    title: str | None = None
    title_japanese: str | None = None
    title_romanji: str | None = None
    aired: datetime | None = None  # Normalized to UTC
    score: float | None = None
    filler: bool = False
    recap: bool = False
    forum_url: str | None = None
    synopsis: str | None = None
    duration: int | None = None

    @field_validator("aired", mode="before")
    @classmethod
    def normalize_aired(cls, v: str | datetime | None) -> datetime | None:
        return normalize_to_utc(v)


# =============================================================================
# /v4/anime/{id}/characters -> data[]
# =============================================================================


class MalPerson(BaseModel):
    model_config = ConfigDict(extra="allow")

    mal_id: int
    url: str | None = None
    images: MalImages | None = None
    name: str


class MalCharacterRef(BaseModel):
    model_config = ConfigDict(extra="allow")

    mal_id: int
    url: str | None = None
    images: MalImages | None = None
    name: str
    name_kanji: str | None = None


class MalVoiceActorEntry(BaseModel):
    model_config = ConfigDict(extra="allow")

    person: MalPerson
    language: str


class MalAnimeCharacterEntry(BaseModel):
    model_config = ConfigDict(extra="allow")

    character: MalCharacterRef
    role: str
    favorites: int | None = None
    voice_actors: list[MalVoiceActorEntry] = []


# =============================================================================
# /v4/characters/{id} -> data
# =============================================================================


class MalCharacter(BaseModel):
    model_config = ConfigDict(extra="allow")

    mal_id: int
    url: str | None = None
    images: MalImages | None = None
    name: str
    name_kanji: str | None = None
    nicknames: list[str] = []
    favorites: int | None = None
    about: str | None = None


# =============================================================================
# /v4/characters/{id}/full -> data
# =============================================================================


class MalAnimeRef(BaseModel):
    model_config = ConfigDict(extra="allow")

    mal_id: int
    url: str | None = None
    images: MalImages | None = None
    title: str


class MalMangaRef(BaseModel):
    model_config = ConfigDict(extra="allow")

    mal_id: int
    url: str | None = None
    images: MalImages | None = None
    title: str


class MalCharacterAnimeEntry(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str | None = None
    anime: MalAnimeRef


class MalCharacterMangaEntry(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str | None = None
    manga: MalMangaRef


class MalCharacterVoiceEntry(BaseModel):
    model_config = ConfigDict(extra="allow")

    person: MalPerson
    language: str


class MalCharacterFull(MalCharacter):
    anime: list[MalCharacterAnimeEntry] = []
    manga: list[MalCharacterMangaEntry] = []
    voices: list[MalCharacterVoiceEntry] = []
