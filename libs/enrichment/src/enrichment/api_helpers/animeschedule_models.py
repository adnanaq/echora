"""Pydantic models for the AnimSchedule API v3 anime response.

These mirror the JSON structure with camelCase → snake_case aliases.
All normalization and mapping to the canonical Anime model lives in
mappers/animeschedule_mapper.py.
"""
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AnimScheduleStats(BaseModel):
    """Community statistics sub-object."""

    average_score: float | None = Field(None, alias="averageScore")
    rating_count: int | None = Field(None, alias="ratingCount")
    tracked_count: int | None = Field(None, alias="trackedCount")
    tracked_rating: int | None = Field(None, alias="trackedRating")

    model_config = ConfigDict(populate_by_name=True)


class AnimScheduleNames(BaseModel):
    """Title variants for an anime entry."""

    native: str | None = None       # Japanese title
    romaji: str | None = None
    english: str | None = None
    abbreviation: str | None = None
    synonyms: list[str] = Field(default_factory=list)

    model_config = ConfigDict(populate_by_name=True)


class AnimScheduleSeason(BaseModel):
    """Season metadata sub-object.

    ``season`` can be an empty string for year-only entries (e.g. "2026").
    """

    title: str | None = None
    year: str | None = None
    season: str | None = None   # "" for year-only entries
    route: str | None = None

    model_config = ConfigDict(populate_by_name=True)


class AnimScheduleStream(BaseModel):
    """Single streaming platform entry inside ``websites.streams``."""

    platform: str
    url: str    # partial URL (no scheme)
    name: str

    model_config = ConfigDict(populate_by_name=True)


class AnimScheduleAnime(BaseModel):
    """Typed representation of an AnimSchedule API v3 anime object.

    Zero-dates ("0001-01-01T00:00:00Z") are stored as-is; the mapper
    treats them as "not set" via ``_is_zero_date()``.
    ``websites`` values are partial URLs (no scheme) except for the
    ``streams`` key which holds a list of dicts.
    """

    # ── Identity ─────────────────────────────────────────────────────────
    id: str
    title: str
    route: str          # URL slug, e.g. "dandadan"

    # ── Metadata ─────────────────────────────────────────────────────────
    description: str | None = None      # may contain HTML tags
    status: str | None = None           # "Finished"|"Ongoing"|"Upcoming"|"Delayed"
    episodes: int | None = None
    length_min: int | None = Field(None, alias="lengthMin")

    # ── Dates (ISO; "0001-01-01T00:00:00Z" ≡ not set) ───────────────────
    premier: str | None = None          # JP original premiere
    sub_premier: str | None = Field(None, alias="subPremier")
    dub_premier: str | None = Field(None, alias="dubPremier")
    jpn_time: str | None = Field(None, alias="jpnTime")     # latest known JP air time
    sub_time: str | None = Field(None, alias="subTime")
    dub_time: str | None = Field(None, alias="dubTime")
    delayed_from: str | None = Field(None, alias="delayedFrom")
    delayed_until: str | None = Field(None, alias="delayedUntil")
    delayd_reason: str | None = Field(None, alias="delayedDesc")

    # ── Calendar ─────────────────────────────────────────────────────────
    month: str | None = None
    year: int | None = None
    season: AnimScheduleSeason | None = None

    # ── Taxonomy ({name, route} dicts) ───────────────────────────────────
    genres: list[dict[str, str]] = Field(default_factory=list)
    studios: list[dict[str, str]] = Field(default_factory=list)
    sources: list[dict[str, str]] = Field(default_factory=list)     # source material
    media_types: list[dict[str, str]] = Field(default_factory=list, alias="mediaTypes")

    # ── Media ────────────────────────────────────────────────────────────
    image_version_route: str | None = Field(None, alias="imageVersionRoute")

    # ── Community data ───────────────────────────────────────────────────
    stats: AnimScheduleStats | None = None

    # ── Names / titles ───────────────────────────────────────────────────
    names: AnimScheduleNames | None = None

    # ── Relations: key → list of slugs ───────────────────────────────────
    # Keys: "sequels"|"prequels"|"parents"|"sideStories"|"spinoffs"|"alternatives"|"other"
    relations: dict[str, list[str]] = Field(default_factory=dict)

    # ── Cross-source links ───────────────────────────────────────────────
    # String values are partial URLs (no scheme).
    # "streams" key holds list[dict] — handled separately by the mapper.
    websites: dict[str, Any] = Field(default_factory=dict)

    # ── DB timestamps ────────────────────────────────────────────────────
    created_at: str | None = Field(None, alias="createdAt")
    updated_at: str | None = Field(None, alias="updatedAt")

    model_config = ConfigDict(populate_by_name=True, extra="allow")
