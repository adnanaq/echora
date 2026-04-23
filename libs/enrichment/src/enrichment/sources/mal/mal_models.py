"""Scraped data models for MAL direct scraping.

Field names in these models are intentionally chosen to match the canonical
models in libs/common/src/common/models/anime.py wherever possible. This means
the mapper (mal_mapper.py) only performs value normalization (e.g., "Currently
Airing" → "ONGOING"), never field renaming.

These models replace the old Jikan-shaped models (MalAnimeFull, MalAnimeCharacterEntry,
etc.) from libs/common/src/common/models/mal_models.py.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict

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
    title: str
    title_english: str | None = None
    title_japanese: str | None = None
    synonyms: list[str] = []

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
    broadcast_day: str | None = None  # "Sundays"
    broadcast_time: str | None = None  # "23:15"
    broadcast_timezone: str | None = None  # "JST"

    # Counts
    episode_count: int | None = None
    duration: int | None = None  # Seconds (parsed from "24 min.")

    # Statistics (directly usable as int/float — already parsed from MAL HTML)
    score: float | None = None
    scored_by: int | None = None
    rank: int | None = None
    popularity: int | None = None  # Rank number (17, not "#17")
    members: int | None = None
    favorites: int | None = None

    # Text
    synopsis: str | None = None
    background: str | None = None

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
    role: str  # "Script", "Animation Director", etc.


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
