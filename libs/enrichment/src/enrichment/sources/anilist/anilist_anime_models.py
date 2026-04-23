"""AniList typed models for anime and relations.

All models use ConfigDict(populate_by_name=True, extra="allow") so they can be
constructed either from the camelCase API response or from snake_case aliases.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AniListTitle(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    romaji: str | None = None
    english: str | None = None
    native: str | None = None
    user_preferred: str | None = Field(None, alias="userPreferred")


class AniListCoverImage(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    extra_large: str | None = Field(None, alias="extraLarge")
    large: str | None = None
    medium: str | None = None
    color: str | None = None


class AniListTrailer(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    id: str | None = None
    site: str | None = None
    thumbnail: str | None = None


class AniListTag(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    name: str
    description: str | None = None
    category: str | None = None
    is_adult: bool = Field(False, alias="isAdult")


class AniListStudioNode(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    id: int
    name: str
    is_animation_studio: bool = Field(False, alias="isAnimationStudio")


class AniListStudioEdge(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    node: AniListStudioNode
    is_main: bool = Field(False, alias="isMain")


class AniListRelationNode(BaseModel):
    """A single node inside a relation edge (may be anime or manga/novel)."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    id: int
    id_mal: int | None = Field(None, alias="idMal")
    title: AniListTitle | None = None
    format: str | None = None
    status: str | None = None
    season_year: int | None = Field(None, alias="seasonYear")
    average_score: int | None = Field(None, alias="averageScore")
    cover_image: AniListCoverImage | None = Field(None, alias="coverImage")
    episodes: int | None = None
    chapters: int | None = None
    volumes: int | None = None


class AniListRelationEdge(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    node: AniListRelationNode
    relation_type: str = Field(..., alias="relationType")


class AniListExternalLink(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    id: int
    url: str | None = None
    site: str | None = None
    type: str | None = None
    language: str | None = None


class AniListNextAiringEpisode(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    episode: int | None = None
    airing_at: int | None = Field(None, alias="airingAt")
    time_until_airing: int | None = Field(None, alias="timeUntilAiring")


class AniListRanking(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    rank: int
    context: str
    format: str | None = None
    year: int | None = None
    season: str | None = None
    all_time: bool = Field(False, alias="allTime")


class AniListAnime(BaseModel):
    """Root model for an AniList Media (ANIME) response."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    id: int
    id_mal: int | None = Field(None, alias="idMal")
    title: AniListTitle | None = None
    description: str | None = None
    source: str | None = None
    format: str | None = None
    episodes: int | None = None
    duration: int | None = None
    status: str | None = None
    season: str | None = None
    season_year: int | None = Field(None, alias="seasonYear")
    country_of_origin: str | None = Field(None, alias="countryOfOrigin")
    is_adult: bool = Field(False, alias="isAdult")
    cover_image: AniListCoverImage | None = Field(None, alias="coverImage")
    banner_image: str | None = Field(None, alias="bannerImage")
    trailer: AniListTrailer | None = None
    average_score: int | None = Field(None, alias="averageScore")
    popularity: int | None = None
    favourites: int | None = None
    genres: list[str] = Field(default_factory=list)
    synonyms: list[str] = Field(default_factory=list)
    tags: list[AniListTag] = Field(default_factory=list)
    studios: list[AniListStudioEdge] = Field(default_factory=list)
    relations: list[AniListRelationEdge] = Field(default_factory=list)
    external_links: list[AniListExternalLink] = Field(
        default_factory=list, alias="externalLinks"
    )
    next_airing_episode: AniListNextAiringEpisode | None = Field(
        None, alias="nextAiringEpisode"
    )
    rankings: list[AniListRanking] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _unwrap_edges(cls, data: Any) -> Any:
        """Unwrap {"edges": [...]} connection containers from AniList GraphQL response."""
        for key in ("studios", "relations"):
            if isinstance(data.get(key), dict):
                data[key] = data[key].get("edges", [])
        return data
