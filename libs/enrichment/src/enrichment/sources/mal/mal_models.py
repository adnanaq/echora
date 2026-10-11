"""Scraped data models for MAL direct scraping.

Field names in these models are intentionally chosen to match the canonical
models in libs/common/src/common/models/anime.py wherever possible. This means
the mapper (mal_mapper.py) only performs value normalization (e.g., "Currently
Airing" → "ONGOING"), never field renaming.
"""

import re
from typing import Annotated, Any

from enrichment.sources.base.field_types import (
    NumberFromText,
    StrippedText,
    strip_text_or_none,
)
from enrichment.sources.base.utils import (
    month_name,
    parse_broadcast_string,
    parse_iso_date,
    parse_partial_date,
    split_date_range,
)
from enrichment.sources.mal.mal_base import (
    parse_duration_seconds,
    parse_number,
    parse_premiered,
    parse_score,
)
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    model_validator,
)

# MAL prints the rank as "#4536" inside the sidebar HTML.
_RANK = re.compile(r"#([\d,]+)")


def _text_parsed_by(parse: Any) -> Any:
    """Run a MAL text parser on text values and leave other values alone."""

    def parse_text(value: object) -> object:
        return parse(value) if isinstance(value, str) else value

    return parse_text


def _rank_from_html(value: object) -> object:
    """Read the rank from MAL's sidebar HTML ("Ranked: #4536"); "N/A" gives None.

    >>> _rank_from_html("<span>Ranked:</span> #4536<sup>2</sup>")
    4536
    """
    if not isinstance(value, str):
        return value
    match = _RANK.search(value)
    return int(match.group(1).replace(",", "")) if match else None


def _synonym_list(value: object) -> object:
    """Split MAL's comma-separated synonyms into names."""
    if isinstance(value, str):
        return [name.strip() for name in value.split(",")]
    return [] if value is None else value


def _required_text(value: object) -> object:
    """Strip text that must be present."""
    return value.strip() if isinstance(value, str) else value


def _aired_and_premiered(aired: object, premiered: object) -> dict[str, Any]:
    """Read dates, year, month and season from MAL's "Aired" and "Premiered" values.

    "Premiered" ("Fall 1999") gives the season and year. Movies, specials and
    announcements have no "Premiered"; "Aired" still states the year. A month
    is kept only when "Aired" gives no day, since a full date carries it.

    Examples:
        A TV series gives both values. A full date leaves ``month`` empty; the
        merge reads it from ``aired_from``:

        >>> _aired_and_premiered("Apr 5, 2009 to Jul 4, 2010", "Spring 2009")
        {'aired_from': '2009-04-05', 'aired_to': '2010-07-04', 'season': 'spring', 'year': 2009, 'month': None}

        An announced title without a day keeps its year and month:

        >>> _aired_and_premiered("Jul 2025", None)
        {'aired_from': None, 'aired_to': None, 'season': None, 'year': 2025, 'month': 'July'}
        >>> _aired_and_premiered("Not available", None)
        {'aired_from': None, 'aired_to': None, 'season': None, 'year': None, 'month': None}
    """
    aired_text = aired if isinstance(aired, str) else None
    aired_start, aired_end = split_date_range(aired_text)
    aired_from, aired_to = parse_iso_date(aired_start), parse_iso_date(aired_end)
    aired_year, aired_month = parse_partial_date(aired_start)
    season, year = parse_premiered(premiered if isinstance(premiered, str) else None)
    return {
        "aired_from": aired_from,
        "aired_to": aired_to,
        "season": season,
        "year": aired_year if year is None else year,
        "month": month_name(aired_month) if aired_from is None else None,
    }


def _background_text(value: object) -> object:
    """Strip the background; MAL's placeholder for a missing one becomes None.

    >>> _background_text("No background information has been added to this title.") is None
    True
    """
    value = strip_text_or_none(value)
    if isinstance(value, str) and "no background information" in value.lower():
        return None
    return value


def _ranked_or_none(value: int | None) -> int | None:
    """Treat rank 0 as not ranked; MAL shows #0 for an anime it has not ranked."""
    return value or None


MalCount = Annotated[int | None, BeforeValidator(_text_parsed_by(parse_number))]

# =============================================================================
# ANIME SCRAPING MODELS
# =============================================================================


class MalTrailer(BaseModel):
    """Trailer extracted from the video-promotion block on the MAL anime page.

    Fields mirror TrailerEntry in the canonical model so the mapper is a
    straight pass-through with no transformation logic.
    """

    model_config = ConfigDict(extra="forbid")

    source: str
    title: str | None = None
    thumbnail: str | None = None


class MalCompanyRef(BaseModel):
    """Studio, producer, or licensor reference from MAL sidebar."""

    name: str
    source: str


class MalRelatedEntry(BaseModel):
    """Related anime/manga entry from the entries-table on an anime detail page."""

    relation: str  # Raw MAL string: "Side Story", "Sequel", "Adaptation", etc.
    title: str
    source: str
    entry_type: str | None = (
        None  # "(TV)", "(Manga)", "(Movie)" — from parens annotation
    )
    is_anime: bool  # True if /anime/ in URL, False if /manga/


class MalExternalLink(BaseModel):
    """External link (official site, streaming, resource) from MAL."""

    name: str
    source: str


class MalEpisodeRange(BaseModel):
    """Represents a range of episodes (e.g., 1-12) or a single episode (start=end)."""

    start: int
    end: int | None = None


class MalThemeSong(BaseModel):
    """Theme song with structured episode coverage from MAL detail page."""

    title: str
    artist: str | None = None
    episodes: list[MalEpisodeRange] = []


class MalAnime(BaseModel):
    """Scraped from /anime/{id} detail page.

    Field names match canonical Anime model fields wherever possible so the
    mapper only normalizes values, not field names.
    """

    model_config = ConfigDict(extra="forbid")

    # Identity
    source: str

    # Titles
    title: Annotated[str, BeforeValidator(_required_text)]
    title_english: StrippedText = None
    title_japanese: StrippedText = None
    synonyms: Annotated[list[str], BeforeValidator(_synonym_list)] = []

    # Classification
    type: str | None = None  # Raw: "TV", "Movie", "OVA", etc.
    status: str | None = None  # Raw: "Currently Airing", "Finished Airing"
    source_material: str | None = None  # Raw: "Manga", "Light novel", etc.
    rating: str | None = None  # Raw: "PG-13 - Teens 13 or older"

    # Temporal
    year: int | None = None
    season: str | None = None  # Raw: "fall", "spring", etc. (lowercased)
    aired_from: str | None = None  # ISO date string parsed from "Oct 20, 1999 to ?"
    aired_to: str | None = None  # ISO date string; None if ongoing
    month: str | None = None  # Premiere month name, set only when "Aired" gives no day
    broadcast_day: str | None = None  # "Sundays"
    broadcast_time: str | None = None  # "23:15"
    broadcast_timezone: str | None = None  # "JST"

    # Counts
    episode_count: NumberFromText = None  # "1,155"; "Unknown" gives None
    # Seconds, from "24 min." or "1 hr. 30 min."
    duration: Annotated[
        int | None, BeforeValidator(_text_parsed_by(parse_duration_seconds))
    ] = None

    # Statistics, read from MAL's display text ("2,644,378", "#17", "N/A")
    score: Annotated[float | None, BeforeValidator(_text_parsed_by(parse_score))] = None
    scored_by: MalCount = None
    rank: Annotated[
        int | None,
        BeforeValidator(_rank_from_html),
        AfterValidator(_ranked_or_none),
    ] = None
    # Rank number (17, not "#17")
    popularity: Annotated[MalCount, AfterValidator(_ranked_or_none)] = None
    members: MalCount = None
    favorites: MalCount = None

    # Text
    synopsis: str | None = None
    background: Annotated[str | None, BeforeValidator(_background_text)] = None

    # Arrays
    genres: list[str] = []
    themes: list[str] = []  # MAL "Themes" sidebar section
    demographics: list[str] = []
    producers: list[MalCompanyRef] = []
    licensors: list[MalCompanyRef] = []
    studios: list[MalCompanyRef] = []

    # Relations (from entries-table, not the truncated entries-tile)
    related_entries: list[MalRelatedEntry] = []

    # Media
    images: dict[str, str] = {}  # {"jpg": url, "webp": url, "large_jpg": url}
    picture_urls: list[str] = []  # Gallery images from /anime/{id}/pics
    trailer: MalTrailer | None = None
    opening_themes: list[MalThemeSong] = []
    ending_themes: list[MalThemeSong] = []

    # Links
    external_sources: list[MalExternalLink] = []
    streaming: list[MalExternalLink] = []

    # =============================================================================
    # CHARACTER SCRAPING MODELS
    # =============================================================================

    @model_validator(mode="before")
    @classmethod
    def _read_dates_and_broadcast(cls, data: Any) -> Any:
        """Turn MAL's "Aired", "Premiered" and "Broadcast" text into fields."""
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if "aired" in data or "premiered" in data:
            data.update(
                _aired_and_premiered(
                    data.pop("aired", None), data.pop("premiered", None)
                )
            )
        if "broadcast" in data:
            broadcast = strip_text_or_none(data.pop("broadcast"))
            day, time, timezone = parse_broadcast_string(
                broadcast if isinstance(broadcast, str) else None
            )
            data.update(
                broadcast_day=day, broadcast_time=time, broadcast_timezone=timezone
            )
        return data


class MalOgraphyEntry(BaseModel):
    """A single anime/manga ography entry from a MAL character page."""

    title: str
    role: str | None = None  # "Main" or "Supporting"
    sources: list[str] = []  # MAL anime or manga page URLs


class MalVoiceActorRef(BaseModel):
    """Voice actor reference from the characters list or character detail page."""

    person_id: int
    name: str
    language: str
    image_url: str | None = None
    sources: list[str] = []


class MalCharacter(BaseModel):
    """Scraped from /character/{id} detail page.

    Field names match canonical Character model fields wherever possible.
    """

    model_config = ConfigDict(extra="forbid")

    source: str
    name: str
    name_native: str | None = None  # Kanji/native script in parens from page title
    description: str | None = None  # "About" section on character page
    nicknames: list[str] = []
    favorites: int = 0
    images: list[str] = []  # Cover image URLs

    # Free-form biographical data — keys vary per character (not all characters have the same fields)
    # Common keys: Age, Birthdate, Height, Weight, Blood type, Affiliation, Position,
    # Devil fruit, Bounty, Type, etc. Stored as dict[str, str] — values are raw strings.
    character_info: dict[str, Any] = {}

    # Spoiler values keyed by the same field names as character_info, plus "description"
    # for the prose description spoiler. Empty if the character page has no spoilers.
    spoilers: dict[str, str] = {}

    voice_actors: list[MalVoiceActorRef] = []

    # From character detail page
    animeography: list[MalOgraphyEntry] = []
    mangaography: list[MalOgraphyEntry] = []


# =============================================================================
# EPISODE SCRAPING MODELS
# =============================================================================


class EpisodeVARef(BaseModel):
    """Voice actor credit for a specific episode."""

    person_id: int  # MAL person ID
    name: str
    language: str  # "Japanese", "English", etc.


class EpisodeCharacterRef(BaseModel):
    """Character appearance in a specific episode (community-contributed on MAL)."""

    mal_id: int  # Character MAL ID
    name: str
    role: str  # "Main" or "Supporting"
    voice_actors: list[EpisodeVARef] = []


class EpisodeStaffRef(BaseModel):
    """Staff credit for a specific episode (community-contributed on MAL)."""

    person_id: int  # MAL person ID
    name: str
    role: str | None


class MalEpisode(BaseModel):
    """Scraped from /anime/{id}/episode/{num} detail page.

    Field names match canonical Episode model fields wherever possible.
    """

    model_config = ConfigDict(extra="forbid")

    episode_number: int
    source: str

    title: str
    title_japanese: str | None = None
    title_romaji: str | None = None
    synopsis: str | None = None
    aired: str | None = None  # ISO date string or raw "Oct 20, 1999"
    duration: int | None = None  # Seconds (parsed from "00:24:37")

    # MAL appends "Filler" / "Recap" to the title h2 — stripped out and surfaced as flags.
    filler: bool = False
    recap: bool = False

    # Community-contributed — empty for many episodes, especially older shows past ~ep 700
    characters: list[EpisodeCharacterRef] = []
    staff: list[EpisodeStaffRef] = []


# =============================================================================
# RESULT CONTAINER
# =============================================================================


class MalFetchResult(BaseModel):
    """Container for a complete MAL fetch of anime + characters + episodes."""

    anime: MalAnime
    characters: list[MalCharacter] = []
    episodes: list[MalEpisode] = []
