"""Source-faithful Pydantic models for anime-planet.com scraped data.

Field names mirror what anime-planet.com actually sends — no renaming to
canonical names here. The models turn the page text into field values (season,
year, rank, relation details); the mapper (animeplanet_mapper.py) translates them
to the canonical Anime model.
"""

import re
from typing import Annotated, Any, Self

from common.models.anime import SourceMaterialType
from enrichment.sources.base.field_types import (
    NumberFromText,
    StrippedText,
    strip_text_or_none,
)
from enrichment.sources.base.utils import parse_partial_date, split_date_range
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

_SEASON_SLUG = re.compile(r"/seasons/([^/?#]+)")
_ALT_TITLE_PREFIX = "alt title:"
_FIRST_NUMBER = re.compile(r"(\d+)")
_VOLUMES = re.compile(r"Vol:\s*(\d+)", re.IGNORECASE)
_CHAPTERS = re.compile(r"Ch:\s*(\d+)", re.IGNORECASE)


def _parse_season(value: object) -> object:
    """Read the season from AP's season link, e.g. '/anime/seasons/fall-1999' → 'fall'.

    >>> _parse_season("/anime/seasons/fall-1999")
    'fall'
    """
    value = strip_text_or_none(value)
    if not isinstance(value, str):
        return value
    if "/" not in value:
        return value.lower()
    match = _SEASON_SLUG.search(value)
    return match.group(1).split("-")[0].lower() if match else None


def _parse_start_year(value: object) -> object:
    """Read the first year from the entry bar's year text, e.g. ' 1999 - ? ' → 1999.

    >>> _parse_start_year(" 1999 - ? ")
    1999
    """
    if isinstance(value, str) or value is None:
        year, _ = parse_partial_date(split_date_range(value)[0])
        return year
    return value


def _parse_alt_title(value: object) -> object:
    """Strip the 'Alt title: ' prefix from h2.aka text and return the bare title.

    >>> _parse_alt_title("Alt title: ワンピース")
    'ワンピース'
    """
    value = strip_text_or_none(value)
    if isinstance(value, str) and value.lower().startswith(_ALT_TITLE_PREFIX):
        value = value[len(_ALT_TITLE_PREFIX) :].strip() or None
    return value


# Tags anime-planet uses for source material: "Based on a Manga", "Original Work".
_SOURCE_TAG_PREFIXES = ("based on a ", "based on an ")
_ORIGINAL_WORK_TAG = "original work"


def _entry_bar_type(value: object) -> object:
    """Read the type from the entry bar's type text, e.g. "Web\\n (1 ep x 2 min)" → "Web".

    >>> _entry_bar_type("Music Video\\n  \\n  (1 ep)")
    'Music Video'
    """
    value = strip_text_or_none(value)
    if isinstance(value, str):
        return value.split("\n")[0].split("(")[0].strip() or None
    return value


def _source_from_tag(tag: str) -> str | None:
    """Return the source a tag names ("Based on a Manga" → "Manga"), or None.

    >>> _source_from_tag("Based on an Otome Game"), _source_from_tag("Action")
    ('Otome Game', None)
    """
    lowered = tag.strip().lower()
    if lowered == _ORIGINAL_WORK_TAG:
        return tag.strip()
    for prefix in _SOURCE_TAG_PREFIXES:
        if lowered.startswith(prefix):
            return tag.strip()[len(prefix) :].strip() or None
    return None


def _decimal_or_none(value: object) -> object:
    """Read a rating AP's JSON-LD sends as text; text that is not one becomes None.

    >>> _decimal_or_none("4.315"), _decimal_or_none("n/a")
    (4.315, None)
    """
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return value


def _whole_number_or_none(value: object) -> object:
    """Read a count AP's JSON-LD sends as text; text that is not one becomes None.

    >>> _whole_number_or_none("64676"), _whole_number_or_none("1000.5")
    (64676, None)
    """
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return value


class AnimePlanetRelatedEntry(BaseModel):
    """Related anime entry from the relations tab."""

    title: Annotated[str, BeforeValidator(lambda value: (value or "").strip())]
    url: str  # relative href, e.g. "/anime/one-piece-film-red"
    slug: str  # extracted from url
    relation_subtype: StrippedText = (
        None  # "Same Franchise", "Sequel", "Spin Off", etc.
    )
    type: StrippedText = None  # "TV", "Movie", "OVA" — from the fa-tv span
    episode_count: int | None = None  # from "OVA: 1 ep", None when not shown
    image: StrippedText = None

    @model_validator(mode="before")
    @classmethod
    def _split_type_and_episodes(cls, data: Any) -> Any:
        """Split the fa-tv span ("OVA: 1 ep") into type and episode count."""
        if not isinstance(data, dict):
            return data
        raw_type = data.get("type")
        if isinstance(raw_type, str) and ":" in raw_type:
            type_part, episode_part = raw_type.split(":", 1)
            episodes = _FIRST_NUMBER.search(episode_part)
            data = {
                **data,
                "type": type_part,
                "episode_count": int(episodes.group(1)) if episodes else None,
            }
        return data


class AnimePlanetMangaEntry(BaseModel):
    """Related manga entry from the relations tab."""

    title: Annotated[str, BeforeValidator(lambda value: (value or "").strip())]
    url: str  # relative href, e.g. "/manga/one-piece"
    slug: str  # extracted from url
    relation_subtype: StrippedText = None
    type: str | None = None  # "One Shot" when AP shows it; None otherwise
    volumes: int | None = None  # from "Vol: X - Ch: Y" or "Vol: X"
    chapters: int | None = None  # from "Vol: X - Ch: Y" or "Ch: Y"; 1 for One Shot
    image: StrippedText = None

    @model_validator(mode="before")
    @classmethod
    def _read_volumes_and_chapters(cls, data: Any) -> Any:
        """Read type, volumes and chapters from the fa-book-open span.

        "One Shot" gives type "One Shot" and 1 chapter; "Vol: 114 - Ch: 179"
        gives 114 volumes and 179 chapters; "" and "- ?" (a date that bled into
        the span) give nothing.
        """
        if not isinstance(data, dict) or "vol_ch" not in data:
            return data
        data = dict(data)
        vol_ch = (data.pop("vol_ch") or "").strip()
        if vol_ch.lower() == "one shot":
            data.update(type="One Shot", chapters=1)
        elif vol_ch:
            volumes = _VOLUMES.search(vol_ch)
            chapters = _CHAPTERS.search(vol_ch)
            data.update(
                volumes=int(volumes.group(1)) if volumes else None,
                chapters=int(chapters.group(1)) if chapters else None,
            )
        return data


class AnimePlanetAggregateRating(BaseModel):
    """Aggregate user rating from JSON-LD aggregateRating block."""

    model_config = ConfigDict(populate_by_name=True)

    rating_value: Annotated[float | None, BeforeValidator(_decimal_or_none)] = Field(
        None, alias="ratingValue"
    )  # 4.315 (anime-planet uses 0–5 scale)
    rating_count: Annotated[int | None, BeforeValidator(_whole_number_or_none)] = Field(
        None, alias="ratingCount"
    )  # 64676

    @property
    def is_empty(self) -> bool:
        """Tell whether neither the rating nor its count is known."""
        return self.rating_value is None and self.rating_count is None


class AnimePlanetStudio(BaseModel):
    """A studio credited on an anime page, with its anime-planet page."""

    name: str
    url: str | None = None  # https://www.anime-planet.com/anime/studios/{slug}


class AnimePlanetAnime(BaseModel):
    """Scraped anime data from an anime-planet.com anime page.

    Populated from two sources:
    - JSON-LD <script type="application/ld+json"> block
    - XPath extraction via JsonXPathExtractionStrategy (entryBar, tags, relations)
    """

    # ── From JSON-LD ──────────────────────────────────────────────────────
    name: str
    description: str | None = None  # synopsis
    url: str | None = None  # canonical AP URL
    start_date: str | None = None  # "1999-10-20"
    end_date: str | None = None
    number_of_episodes: int | None = None
    genres: list[str] = []
    aggregate_rating: AnimePlanetAggregateRating | None = None

    # ── From XPath ───────────────────────────────────────────────────────
    # Entry-bar type, read from "TV\n  (1156+ eps)".
    type: Annotated[str | None, BeforeValidator(_entry_bar_type)] = None
    # Season name, e.g. "fall", read from the seasons link "/anime/seasons/fall-1999".
    season: Annotated[str | None, BeforeValidator(_parse_season)] = None
    # First year in the entry bar, e.g. "1999 - ?" → 1999.
    start_year: Annotated[int | None, BeforeValidator(_parse_start_year)] = None
    rank: NumberFromText = None  # "Rank #1,005" → 1005
    studios: list[AnimePlanetStudio] = []
    # h2.aka text without its "Alt title: " prefix, e.g. "ワンピース".
    alt_title: Annotated[str | None, BeforeValidator(_parse_alt_title)] = None
    tags: list[str] = []  # ["Action", "Adventure", ...]
    cover: str | None = None  # from img[itemprop="image"] — actual poster
    slug: str  # canonical slug, set by caller

    # ── Relations ─────────────────────────────────────────────────────────
    related_anime: list[AnimePlanetRelatedEntry] = []
    related_anime_other: list[AnimePlanetRelatedEntry] = []
    related_manga: list[AnimePlanetMangaEntry] = []

    @property
    def source_material(self) -> str | None:
        """The source the tags name, e.g. "Manga" from "Based on a Manga".

        anime-planet has no source field; it tags the source instead. When a
        page carries several source tags ("Based on a 4-Koma Manga" and "Based
        on a Manga"), the first whose source has a canonical type is used, so a
        tag with no mapping yet ("Based on an Otome Game") gives way to "Based
        on a Visual Novel". With none mapped, the first source tag is used.
        """
        sources = [source for tag in self.tags if (source := _source_from_tag(tag))]
        known = (
            source
            for source in sources
            if SourceMaterialType.from_provider_value(source) is not None
        )
        return next(known, sources[0] if sources else None)

    @model_validator(mode="after")
    def _drop_empty_rating(self) -> Self:
        """Store no rating when neither its value nor its count is known."""
        if self.aggregate_rating is not None and self.aggregate_rating.is_empty:
            self.aggregate_rating = None
        return self
