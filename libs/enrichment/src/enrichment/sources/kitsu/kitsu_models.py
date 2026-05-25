"""Kitsu typed models for anime, character, episode, and related resources.

All models use extra="allow" so unknown API fields are preserved without crashing.
Fields use camelCase to match the Kitsu API attribute names directly.
"""

from common.models.anime import ThemeEntry
from pydantic import BaseModel, ConfigDict, Field


class KitsuImage(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    tiny: str | None = None
    small: str | None = None
    medium: str | None = None
    large: str | None = None
    original: str | None = None


class KitsuTitles(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    en: str | None = None  # English title
    en_jp: str | None = None  # Romanized title
    en_us: str | None = None  # US English — present on episodes, not anime
    ja_jp: str | None = None  # Japanese title


class KitsuAnimeAttributes(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    slug: str | None = None
    synopsis: str | None = None
    canonicalTitle: str | None = None
    titles: KitsuTitles = Field(default_factory=KitsuTitles)
    abbreviatedTitles: list[str] = Field(default_factory=list)
    averageRating: str | None = None  # "78.34" — string from API
    userCount: int | None = None
    favoritesCount: int | None = None
    startDate: str | None = None  # "1999-10-20"
    endDate: str | None = None
    popularityRank: int | None = None
    ratingRank: int | None = None
    ageRating: str | None = None  # "G", "PG", "R", "R17+", "R18+"
    subtype: str | None = None  # "TV", "movie", "OVA", "ONA", "special", "music"
    status: str | None = None  # "current", "finished", "tba", "unreleased", "upcoming"
    posterImage: KitsuImage | None = None
    coverImage: KitsuImage | None = None
    episodeCount: int | None = None
    episodeLength: int | None = None  # minutes
    youtubeVideoId: str | None = None
    nextRelease: str | None = None  # ISO datetime e.g. "2023-10-08T09:30:00.000+09:00"
    nsfw: bool = False


class KitsuAnime(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    id: str
    type: str | None = None
    attributes: KitsuAnimeAttributes = Field(default_factory=KitsuAnimeAttributes)
    # Populated by the helper after fetching /genres and /categories endpoints
    genres: list[str] = Field(default_factory=list)
    themes: list[ThemeEntry] = Field(default_factory=list)


class KitsuGenreAttributes(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    name: str | None = None
    slug: str | None = None


class KitsuGenre(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    id: str
    attributes: KitsuGenreAttributes = Field(default_factory=KitsuGenreAttributes)


class KitsuCategoryAttributes(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    title: str | None = None
    description: str | None = None
    slug: str | None = None


class KitsuCategory(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    id: str
    attributes: KitsuCategoryAttributes = Field(default_factory=KitsuCategoryAttributes)


class KitsuCharacterNames(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    en: str | None = None  # English name
    ja_jp: str | None = None  # Japanese name


class KitsuCharacterAttributes(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    slug: str | None = None
    name: str | None = None
    names: KitsuCharacterNames = Field(default_factory=KitsuCharacterNames)
    canonicalName: str | None = None
    otherNames: list[str] = Field(default_factory=list)
    malId: int | None = None
    description: str | None = None  # may contain HTML
    image: KitsuImage | None = None


class KitsuCharacter(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    id: str
    attributes: KitsuCharacterAttributes = Field(
        default_factory=KitsuCharacterAttributes
    )


class KitsuMediaCharacterAttributes(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    role: str | None = None  # "main", "supporting"


class KitsuMediaCharacter(BaseModel):
    """A mediaCharacters resource: character's appearance in one specific anime."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    id: str
    attributes: KitsuMediaCharacterAttributes = Field(
        default_factory=KitsuMediaCharacterAttributes
    )
    character: KitsuCharacter | None = None  # resolved from included[] at parse time
    # Populated by the helper after fetching /voices and /media-characters endpoints
    voices: list["KitsuCharacterVoice"] = Field(default_factory=list)
    animeography: list["KitsuAnimeographyEntry"] = Field(default_factory=list)


class KitsuPersonAttributes(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    name: str | None = None
    description: str | None = None
    image: KitsuImage | None = None


class KitsuPerson(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    id: str  # used to build source URL: https://kitsu.app/people/{id}
    attributes: KitsuPersonAttributes = Field(default_factory=KitsuPersonAttributes)


class KitsuCharacterVoiceAttributes(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    locale: str | None = None  # "ja_jp", "en", "pt_br", "es", "ko", "de", "it", etc.


class KitsuCharacterVoice(BaseModel):
    """A characterVoices resource: one voice actor for one locale."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    id: str
    attributes: KitsuCharacterVoiceAttributes = Field(
        default_factory=KitsuCharacterVoiceAttributes
    )
    person: KitsuPerson | None = None  # resolved from included[] at parse time


class KitsuAnimeographyAttributes(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    role: str | None = None  # "main", "supporting"


class KitsuAnimeographyEntry(BaseModel):
    """A mediaCharacters resource: one of a character's appearances across all media."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    id: str
    attributes: KitsuAnimeographyAttributes = Field(
        default_factory=KitsuAnimeographyAttributes
    )
    media_type: str | None = None  # "anime" or "manga" — from relationship data.type
    media: KitsuAnime | None = None  # resolved from included[] at parse time


class KitsuEpisodeAttributes(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    canonicalTitle: str | None = None
    titles: KitsuTitles = Field(default_factory=KitsuTitles)
    synopsis: str | None = None
    description: str | None = None
    number: int | None = None
    seasonNumber: int | None = None
    airdate: str | None = None  # "1999-10-20"
    length: int | None = None  # minutes
    thumbnail: KitsuImage | None = None


class KitsuEpisode(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    id: str
    attributes: KitsuEpisodeAttributes = Field(default_factory=KitsuEpisodeAttributes)
