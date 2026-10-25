"""Pydantic models for the AnimeSchedule API v3 anime response.

These mirror the JSON structure with camelCase → snake_case aliases. The
models turn AnimeSchedule's values into field values (HTML descriptions into
plain text, its "not set" date into None); mapping to the canonical Anime model
lives in animeschedule_mapper.py.
"""

from typing import Annotated, Any

from enrichment.sources.base.field_types import (
    PlainText,
    StrippedText,
    strip_text_or_none,
)
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

_UNSET_DATE_PREFIX = "0001-01-01"


def _unset_date_to_none(value: object) -> object:
    """Turn AnimeSchedule's "not set" date ("0001-01-01T00:00:00Z") into None.

    >>> _unset_date_to_none("0001-01-01T00:00:00Z") is None
    True
    >>> _unset_date_to_none("1999-10-20T00:00:00Z")
    '1999-10-20T00:00:00Z'
    """
    value = strip_text_or_none(value)
    if isinstance(value, str) and value.startswith(_UNSET_DATE_PREFIX):
        return None
    return value


AnimeScheduleDate = Annotated[str | None, BeforeValidator(_unset_date_to_none)]


class AnimeScheduleStats(BaseModel):
    """Community statistics sub-object."""

    average_score: float | None = Field(None, alias="averageScore")
    rating_count: int | None = Field(None, alias="ratingCount")
    tracked_count: int | None = Field(None, alias="trackedCount")
    tracked_rating: int | None = Field(None, alias="trackedRating")

    model_config = ConfigDict(populate_by_name=True)


class AnimeScheduleNames(BaseModel):
    """Title variants for an anime entry."""

    native: StrippedText = None  # Japanese title
    romaji: StrippedText = None
    english: StrippedText = None
    abbreviation: StrippedText = None
    synonyms: list[str] = Field(default_factory=list)

    model_config = ConfigDict(populate_by_name=True)


class AnimeScheduleSeason(BaseModel):
    """Season metadata sub-object.

    ``season`` is None for year-only entries (e.g. "2026"), which AnimeSchedule
    sends as an empty string.
    """

    title: str | None = None
    year: str | None = None
    season: StrippedText = None
    route: str | None = None

    model_config = ConfigDict(populate_by_name=True)


class AnimeScheduleStream(BaseModel):
    """Single streaming platform entry inside ``websites.streams``."""

    platform: str
    url: str  # partial URL (no scheme)
    name: str

    model_config = ConfigDict(populate_by_name=True)


class AnimeScheduleAnime(BaseModel):
    """Typed representation of an AnimeSchedule API v3 anime object.

    Dates AnimeSchedule sends as "not set" ("0001-01-01T00:00:00Z") are None.
    ``websites`` values are partial URLs (no scheme) except for the
    ``streams`` key which holds a list of dicts.
    """

    # ── Identity ─────────────────────────────────────────────────────────
    id: str
    title: str
    route: str  # URL slug, e.g. "dandadan"

    # ── Metadata ─────────────────────────────────────────────────────────
    description: PlainText = None
    status: str | None = None  # "Finished"|"Ongoing"|"Upcoming"|"Delayed"
    episodes: int | None = None
    length_min: int | None = Field(None, alias="lengthMin")

    # ── Dates (ISO) ──────────────────────────────────────────────────────
    premier: AnimeScheduleDate = None  # JP original premiere
    sub_premier: AnimeScheduleDate = Field(None, alias="subPremier")
    dub_premier: AnimeScheduleDate = Field(None, alias="dubPremier")
    jpn_time: AnimeScheduleDate = Field(
        None, alias="jpnTime"
    )  # latest known JP air time
    sub_time: AnimeScheduleDate = Field(None, alias="subTime")
    dub_time: AnimeScheduleDate = Field(None, alias="dubTime")
    delayed_from: AnimeScheduleDate = Field(None, alias="delayedFrom")
    delayed_until: AnimeScheduleDate = Field(None, alias="delayedUntil")
    delayed_reason: StrippedText = Field(None, alias="delayedDesc")

    # ── Calendar ─────────────────────────────────────────────────────────
    month: str | None = None
    year: int | None = None
    season: AnimeScheduleSeason | None = None

    # ── Taxonomy ({name, route} dicts) ───────────────────────────────────
    genres: list[dict[str, str]] = Field(default_factory=list)
    studios: list[dict[str, str]] = Field(default_factory=list)
    sources: list[dict[str, str]] = Field(default_factory=list)  # source material
    media_types: list[dict[str, str]] = Field(default_factory=list, alias="mediaTypes")

    # ── Media ────────────────────────────────────────────────────────────
    image_version_route: str | None = Field(None, alias="imageVersionRoute")

    # ── Community data ───────────────────────────────────────────────────
    stats: AnimeScheduleStats | None = None

    # ── Names / titles ───────────────────────────────────────────────────
    names: AnimeScheduleNames | None = None

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
